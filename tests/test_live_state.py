"""
test_live_state.py — Capa de estado transaccional del runner vivo (ARQ-04).

StateStore: lock exclusivo (fcntl.flock), escrituras en staging, commit con
os.replace + manifiesto sha256, recuperación de staging huérfano y migración
explícita del esquema de ledger.json. Todo en tmp_path, sin red.
"""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.live_state as ls
from src.live_state import (
    MANIFEST_NAME,
    STATE_SCHEMA_VERSION,
    StateLockedError,
    StateStore,
    migrate_ledger_payload,
)
from src.position_ledger import LEDGER_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict:
    """Bytes de cada archivo de estado visible (excluye lock y staging)."""
    return {
        p.name: p.read_bytes()
        for p in sorted(root.iterdir())
        if p.is_file() and not p.name.startswith(".")
    }


def _commit(root: Path, run_id: str, files: dict, as_of: str = "2024-01-02") -> dict:
    with StateStore(root) as store:
        store.begin(run_id, as_of=as_of)
        for name, text in files.items():
            store.write_text(name, text)
        return store.commit()


# ---------------------------------------------------------------------------
# begin / commit / manifest
# ---------------------------------------------------------------------------
def test_staged_writes_invisible_until_commit(tmp_path):
    with StateStore(tmp_path) as store:
        store.begin("r1", as_of="2024-01-02")
        store.write_text("last_rebalance.txt", "2024-01-02")
        assert not (tmp_path / "last_rebalance.txt").exists()
        store.commit()
    assert (tmp_path / "last_rebalance.txt").read_text() == "2024-01-02"
    assert not any((tmp_path / ".staging").iterdir())


def test_commit_writes_manifest_with_sha256(tmp_path):
    manifest = _commit(tmp_path, "r1", {"a.txt": "alpha", "b.txt": "beta"})
    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert on_disk == manifest
    assert on_disk["schema_version"] == STATE_SCHEMA_VERSION
    assert on_disk["run_id"] == "r1"
    assert on_disk["as_of"] == "2024-01-02"
    assert on_disk["files"] == ["a.txt", "b.txt"]
    assert on_disk["sha256"] == {
        "a.txt": _sha(tmp_path / "a.txt"),
        "b.txt": _sha(tmp_path / "b.txt"),
    }


def test_manifest_carries_forward_untouched_files(tmp_path):
    _commit(tmp_path, "r1", {"a.txt": "alpha", "b.txt": "beta"})
    m2 = _commit(tmp_path, "r2", {"a.txt": "alpha-2"}, as_of="2024-01-03")
    assert m2["run_id"] == "r2"
    assert m2["files"] == ["a.txt", "b.txt"]
    assert m2["sha256"]["a.txt"] == _sha(tmp_path / "a.txt")
    assert m2["sha256"]["b.txt"] == _sha(tmp_path / "b.txt")


def test_verify_reports_tampered_files(tmp_path):
    _commit(tmp_path, "r1", {"a.txt": "alpha", "b.txt": "beta"})
    with StateStore(tmp_path) as store:
        assert store.verify() == []
    (tmp_path / "b.txt").write_text("edited by hand")
    with StateStore(tmp_path) as store:
        assert store.verify() == ["b.txt"]


def test_append_csv_extends_committed_and_staged_rows(tmp_path):
    with StateStore(tmp_path) as store:
        store.begin("r1", as_of="2024-01-02")
        store.append_csv("h.csv", pd.DataFrame([{"run_id": "r1", "x": 1}]))
        store.commit()
        store.begin("r2", as_of="2024-01-03")
        store.append_csv("h.csv", pd.DataFrame([{"run_id": "r2", "x": 2}]))
        store.append_csv("h.csv", pd.DataFrame([{"run_id": "r2", "x": 3}]))
        store.commit()
    out = pd.read_csv(tmp_path / "h.csv")
    assert out["run_id"].tolist() == ["r1", "r2", "r2"]
    assert out["x"].tolist() == [1, 2, 3]


def test_append_csv_keeps_committed_rows_byte_for_byte(tmp_path):
    legacy = "date,nav,forced\n2026-09-25,1.50,True\n2026-09-25,,False\n"
    _commit(tmp_path, "r1", {"h.csv": legacy})
    with StateStore(tmp_path) as store:
        store.begin("r2", as_of="2026-09-29")
        store.append_csv("h.csv", pd.DataFrame([{"date": "2026-09-29", "nav": 2.0,
                                                 "forced": False, "run_id": "r2"}]))
        store.commit()
    text = (tmp_path / "h.csv").read_text()
    assert text.splitlines()[0] == "date,nav,forced,run_id"
    assert text.splitlines()[1:3] == ["2026-09-25,1.50,True,", "2026-09-25,,False,"]
    assert text.splitlines()[3] == "2026-09-29,2.0,False,r2"


# ---------------------------------------------------------------------------
# Fallos: abort, staging huérfano, commit interrumpido
# ---------------------------------------------------------------------------
def test_exception_inside_transaction_keeps_previous_state(tmp_path):
    _commit(tmp_path, "r1", {"ledger.json": '{"v": 1}', "history.csv": "a\n1\n"})
    before = _snapshot(tmp_path)
    with StateStore(tmp_path) as store:
        with pytest.raises(RuntimeError, match="boom"):
            with store.transaction("r2", as_of="2024-01-03"):
                store.write_text("ledger.json", '{"v": 2}')
                raise RuntimeError("boom")
    assert _snapshot(tmp_path) == before
    assert not any((tmp_path / ".staging").iterdir())


def test_orphan_staging_is_discarded_with_warning(tmp_path, caplog):
    _commit(tmp_path, "r1", {"ledger.json": '{"v": 1}'})
    before = _snapshot(tmp_path)
    orphan = tmp_path / ".staging" / "crashed-run"
    orphan.mkdir(parents=True)
    (orphan / "ledger.json").write_text('{"v": 999}')
    with caplog.at_level(logging.WARNING, logger="live_state"):
        with StateStore(tmp_path):
            pass
    assert not orphan.exists()
    assert _snapshot(tmp_path) == before
    assert any("crashed-run" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_interrupted_commit_is_rolled_forward_on_next_open(tmp_path, monkeypatch):
    _commit(tmp_path, "r1", {"ledger.json": "L1", "history.csv": "H1"})
    real_replace = ls.os.replace

    def crash_after_first(src, dst):
        if Path(dst) == tmp_path / "history.csv":
            raise OSError("simulated crash mid-commit")
        return real_replace(src, dst)

    store = StateStore(tmp_path)
    store.acquire()
    try:
        store.begin("r2", as_of="2024-01-03")
        store.write_text("ledger.json", "L2")
        store.write_text("history.csv", "H2")
        monkeypatch.setattr(ls.os, "replace", crash_after_first)
        with pytest.raises(OSError, match="mid-commit"):
            store.commit()
        monkeypatch.setattr(ls.os, "replace", real_replace)
        store.abort()  # no debe descartar un commit ya journalizado
    finally:
        store.release()
    # Estado a medias: ledger nuevo, history viejo
    assert (tmp_path / "ledger.json").read_text() == "L2"
    assert (tmp_path / "history.csv").read_text() == "H1"

    with StateStore(tmp_path) as store:
        manifest = store.read_manifest()
        assert store.verify() == []
    assert (tmp_path / "ledger.json").read_text() == "L2"
    assert (tmp_path / "history.csv").read_text() == "H2"
    assert manifest["run_id"] == "r2"
    assert not any((tmp_path / ".staging").iterdir())


# ---------------------------------------------------------------------------
# Lock exclusivo
# ---------------------------------------------------------------------------
_TRY_LOCK = (
    "import sys\n"
    "from pathlib import Path\n"
    "from src.live_state import StateStore, StateLockedError\n"
    "try:\n"
    "    StateStore(Path(sys.argv[1])).acquire()\n"
    "except StateLockedError as exc:\n"
    "    print(exc)\n"
    "    sys.exit(4)\n"
)


def _try_lock_in_subprocess(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _TRY_LOCK, str(root)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


def test_second_process_is_rejected_while_lock_held(tmp_path):
    with StateStore(tmp_path):
        proc = _try_lock_in_subprocess(tmp_path)
    assert proc.returncode == 4, proc.stderr
    assert "lock" in proc.stdout.lower()
    assert str(tmp_path) in proc.stdout


def test_lock_is_released_on_exit(tmp_path):
    with StateStore(tmp_path):
        pass
    assert _try_lock_in_subprocess(tmp_path).returncode == 0


def test_second_store_in_same_process_is_rejected(tmp_path):
    with StateStore(tmp_path):
        with pytest.raises(StateLockedError):
            StateStore(tmp_path).acquire()


# ---------------------------------------------------------------------------
# Guardas de uso
# ---------------------------------------------------------------------------
def test_begin_requires_lock(tmp_path):
    with pytest.raises(RuntimeError, match="lock"):
        StateStore(tmp_path).begin("r1")


def test_writes_require_begin(tmp_path):
    with StateStore(tmp_path) as store:
        with pytest.raises(RuntimeError, match="begin"):
            store.write_text("a.txt", "x")


@pytest.mark.parametrize("name", ["../evil.txt", "sub/a.txt", ".lock", MANIFEST_NAME, ""])
def test_rejects_invalid_or_reserved_names(tmp_path, name):
    with StateStore(tmp_path) as store:
        store.begin("r1")
        with pytest.raises(ValueError):
            store.write_text(name, "x")


# ---------------------------------------------------------------------------
# Migración explícita de ledger.json
# ---------------------------------------------------------------------------
LEGACY_LEDGER = {
    "cash": 100.0,
    "realized_pnl": 1.5,
    "lots": [{"ticker": "AAA", "shares": 2.0, "entry_price": 10.0,
              "entry_date": "2024-01-02T00:00:00"}],
}


def test_migrate_unversioned_ledger_to_current():
    original = json.loads(json.dumps(LEGACY_LEDGER))
    out = migrate_ledger_payload(original)
    assert out["schema_version"] == LEDGER_SCHEMA_VERSION
    assert out["txns"] == []
    assert out["lots"] == [{**lot, "currency": "USD"}
                           for lot in LEGACY_LEDGER["lots"]]
    assert out["cash"] == 100.0 and out["realized_pnl"] == 1.5
    assert original == LEGACY_LEDGER  # no muta la entrada


def test_migrate_v1_ledger_adds_base_and_lot_currency():
    v1 = {**LEGACY_LEDGER, "schema_version": 1, "txns": []}
    original = json.loads(json.dumps(v1))
    out = migrate_ledger_payload(original)
    assert out["schema_version"] == LEDGER_SCHEMA_VERSION == 2
    assert out["base_currency"] == "USD"
    assert [lot["currency"] for lot in out["lots"]] == ["USD"]
    assert original == v1  # no muta la entrada (ni sus lotes)


def test_migrate_current_ledger_is_identity():
    current = {**LEGACY_LEDGER, "schema_version": LEDGER_SCHEMA_VERSION, "txns": []}
    assert migrate_ledger_payload(current) == current


def test_migrate_rejects_newer_schema():
    with pytest.raises(ValueError, match="schema_version"):
        migrate_ledger_payload({**LEGACY_LEDGER, "schema_version": LEDGER_SCHEMA_VERSION + 1})
