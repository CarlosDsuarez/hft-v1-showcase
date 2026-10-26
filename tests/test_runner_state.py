"""
test_runner_state.py — run_once persiste estado de forma transaccional (ARQ-04).

Atomicidad ante fallo a mitad de escritura, lock entre procesos, idempotencia
por run_id/as_of y audit trail de transacciones. Sintético, sin red, sin
broker; estado en tmp_path vía HFT_HOME.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.position_ledger import CASH_TOLERANCE, PositionLedger
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIFT = pd.Series([1.1, 0.9, 1.0, 1.05, 0.95, 1.0], index=TICKERS)
EXIT_LOCKED = runner.EXIT_LOCKED


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    use_market_data(monkeypatch, _fake_prices())
    return home / "data" / "live_state"


def _files(state: Path) -> dict:
    """Bytes de cada archivo de estado visible (excluye lock y staging)."""
    return {
        p.name: p.read_bytes()
        for p in sorted(state.iterdir())
        if p.is_file() and not p.name.startswith(".")
    }


def _manifest(state: Path) -> dict:
    return json.loads((state / "state_manifest.json").read_text())


# ---------------------------------------------------------------------------
# Atomicidad
# ---------------------------------------------------------------------------
def test_failure_between_ledger_and_history_keeps_previous_state(state_dir, monkeypatch):
    assert runner.run_once(force=True, start="2024-01-01") == 0
    before = _files(state_dir)

    real_to_csv = pd.DataFrame.to_csv

    def failing_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and Path(str(path_or_buf)).name == "rebalance_history.csv":
            raise OSError("disk full while writing history")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    use_market_data(monkeypatch, _fake_prices() * DRIFT)
    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
    with pytest.raises(OSError, match="disk full"):
        runner.run_once(force=True, start="2024-01-01")
    monkeypatch.setattr(pd.DataFrame, "to_csv", real_to_csv)

    assert _files(state_dir) == before
    reloaded = runner.load_ledger(state_dir / "ledger.json")
    assert reloaded.to_dict() == json.loads(before["ledger.json"])
    staging = state_dir / ".staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_orphan_staging_discarded_with_warning_on_next_run(state_dir, caplog):
    orphan = state_dir / ".staging" / "killed-run"
    orphan.mkdir(parents=True)
    (orphan / "ledger.json").write_text('{"cash": -1}')
    with caplog.at_level(logging.WARNING):
        assert runner.run_once(force=True, start="2024-01-01") == 0
    assert not orphan.exists()
    assert any("killed-run" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)
    # El contenido huérfano (cash=-1) nunca se publica
    assert json.loads((state_dir / "ledger.json").read_text())["cash"] >= -CASH_TOLERANCE


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------
def test_check_only_does_not_block_on_writer_lock(state_dir):
    from src.live_state import StateStore

    # Un --check-only de monitoreo no debe competir con el rebalanceo real
    with StateStore(state_dir):
        assert runner.run_once(check_only=True, start="2024-01-01") == 0


def test_check_only_never_touches_writer_staging(state_dir):
    from src.live_state import StateStore

    with StateStore(state_dir) as writer:
        writer.begin("in-flight", as_of="2024-05-03")
        writer.write_text("ledger.json", "{}")
        assert runner.run_once(check_only=True, start="2024-01-01") == 0
        assert (state_dir / ".staging" / "in-flight" / "ledger.json").exists()
    orphan = state_dir / ".staging" / "crashed"
    orphan.mkdir(parents=True)
    # Sin lock no hay recuperación: descartar staging es tarea del escritor
    assert runner.run_once(check_only=True, start="2024-01-01") == 0
    assert orphan.exists()


def test_run_once_exits_by_lock_without_fetching_data(state_dir, monkeypatch, capsys):
    from src.live_state import StateStore

    def no_fetch(*_a, **_k):
        raise AssertionError("market data must not be read while locked")

    monkeypatch.setattr(runner, "make_market_data_provider", no_fetch)
    with StateStore(state_dir):
        rc = runner.run_once(force=True, start="2024-01-01")
    assert rc == EXIT_LOCKED
    assert "lock" in capsys.readouterr().err.lower()
    assert _files(state_dir) == {}


def test_lock_exit_code_is_distinct_from_other_exit_codes():
    # ARQ-06: 0 ok, 1 no toca, 2 sin historia, 3 broker (error o bloqueado),
    # 4 lock, 5 error inesperado de la CLI
    assert EXIT_LOCKED == 4
    assert EXIT_LOCKED not in (runner.EXIT_OK, runner.EXIT_NOT_DUE,
                               runner.EXIT_NO_DATA, runner.EXIT_BROKER,
                               runner.EXIT_UNEXPECTED)


def test_broker_step_error_exits_3_and_commits_nothing(state_dir, monkeypatch):
    from tests.test_runner_modes import FakeBroker

    class BrokerDown(FakeBroker):
        def submit_orders(self, *_a, **_k):
            raise RuntimeError("broker down")

    monkeypatch.setattr(runner, "make_broker_client", lambda _root: BrokerDown())
    assert runner.run_once(force=True, start="2024-01-01",
                           mode="broker", submit=True) == 3
    # ARQ-06: nada se persiste antes de un execute exitoso (ni el espejo)
    assert not (state_dir / "state_manifest.json").exists()
    assert not (state_dir / "ledger.json").exists()
    assert not (state_dir / "last_rebalance.txt").exists()
    assert not (state_dir / "rebalance_history.csv").exists()


def test_forced_submit_supersedes_last_registered_run_not_dry_run(state_dir, monkeypatch):
    from tests.test_runner_modes import FakeBroker

    monkeypatch.setattr(runner, "make_broker_client", lambda _root: FakeBroker())
    kw = dict(start="2024-01-01", mode="broker")
    assert runner.run_once(submit=True, **kw) == 0
    first = pd.read_csv(state_dir / "rebalance_history.csv")["run_id"].iloc[-1]
    assert runner.run_once(force=True, **kw) == 0           # dry-run: no escribe
    assert runner.run_once(force=True, submit=True, **kw) == 0
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert hist["mode"].tolist() == ["broker", "broker"]
    assert hist["supersedes"].iloc[-1] == first


def test_second_process_exits_by_lock(state_dir):
    from src.live_state import StateStore

    env = {
        **os.environ,
        "HFT_HOME": str(state_dir.parents[1]),
        # Si el lock fallara, la descarga debe fallar rápido (sin red real).
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "",
    }
    with StateStore(state_dir):
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.run_periodic_rebalance", "--force"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
    assert proc.returncode == EXIT_LOCKED, proc.stderr[-2000:]
    assert "lock" in proc.stderr.lower()
    assert _files(state_dir) == {}


# ---------------------------------------------------------------------------
# Audit trail: manifiesto, run_id, transacciones
# ---------------------------------------------------------------------------
def test_history_row_and_manifest_share_run_id(state_dir):
    assert runner.run_once(force=True, start="2024-01-01") == 0
    manifest = _manifest(state_dir)
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert hist["run_id"].tolist() == [manifest["run_id"]]
    assert manifest["as_of"] == hist["date"].iloc[0]
    for name in ("ledger.json", "last_rebalance.txt", "latest_snapshot.csv",
                 "rebalance_history.csv", "transactions.csv"):
        assert name in manifest["files"], name


def test_ledger_roundtrip_with_lots_and_transactions(state_dir, monkeypatch):
    assert runner.run_once(force=True, start="2024-01-01") == 0
    use_market_data(monkeypatch, _fake_prices() * DRIFT)
    assert runner.run_once(force=True, start="2024-01-01") == 0

    saved = json.loads((state_dir / "ledger.json").read_text())
    ledger = runner.load_ledger(state_dir / "ledger.json")
    assert ledger.to_dict() == saved
    assert saved["lots"] and saved["txns"]
    assert any(t["shares"] < 0 for t in saved["txns"])  # hubo ventas FIFO

    txns = pd.read_csv(state_dir / "transactions.csv")
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert list(dict.fromkeys(txns["run_id"])) == hist["run_id"].tolist()
    expected = pd.DataFrame(saved["txns"])
    got = txns.drop(columns="run_id")[expected.columns]
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    rebuilt = PositionLedger.from_dict(saved)
    assert rebuilt.to_dict() == saved


def test_exit_log_rows_carry_run_id(state_dir, monkeypatch):
    # 1a corrida 10 días antes (sigue >= required_history del stack)
    use_market_data(monkeypatch, _fake_prices().iloc[:-10])
    assert runner.run_once(force=True, start="2024-01-01") == 0
    use_market_data(monkeypatch, _fake_prices())
    assert runner.run_once(force=True, start="2024-01-01") == 0
    exits = pd.read_csv(state_dir / "exit_log.csv")
    assert len(exits) > 0
    assert set(exits["run_id"]) == {_manifest(state_dir)["run_id"]}


# ---------------------------------------------------------------------------
# Idempotencia por as_of
# ---------------------------------------------------------------------------
def test_rerun_same_as_of_without_force_writes_nothing(state_dir):
    assert runner.run_once(start="2024-01-01") == 0
    before = _files(state_dir)
    # every_n=0 haría "due" por conteo de días: la guarda es por as_of/run_id
    assert runner.run_once(start="2024-01-01", every_n=0) == 1
    assert _files(state_dir) == before


def test_forced_rerun_same_as_of_supersedes_previous(state_dir):
    assert runner.run_once(start="2024-01-01") == 0
    first = _manifest(state_dir)["run_id"]
    assert runner.run_once(force=True, start="2024-01-01") == 0
    second = _manifest(state_dir)["run_id"]
    assert second != first

    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert hist["run_id"].tolist() == [first, second]
    assert hist["forced"].tolist() == [False, True]
    assert pd.isna(hist["supersedes"].iloc[0])
    assert hist["supersedes"].iloc[1] == first
