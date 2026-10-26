"""
live_state.py — Capa de estado transaccional del runner vivo (ARQ-04).

`StateStore(root)` agrupa las escrituras de una corrida en una transacción:

  - Lock exclusivo por archivo: `fcntl.flock` sobre `root/.lock` (no
    bloqueante). Si otro proceso lo tiene → `StateLockedError` (el runner
    sale con un código propio). Solo POSIX (macOS/Linux/Docker).
  - `begin(run_id, as_of)`: toda escritura va a `root/.staging/<run_id>/`;
    el estado visible en `root/` no cambia hasta `commit()`.
  - `commit()`: escribe el manifiesto `state_manifest.json`
    {schema_version, run_id, as_of, committed_at_utc, files, sha256} en
    staging, marca el punto de commit con un journal (`.commit`, orden de
    archivos) y mueve cada archivo con `os.replace` (atómico por archivo),
    el manifiesto al final.
  - Al adquirir el lock se recupera lo que dejó un proceso muerto:
      staging SIN journal → corrida no comprometida: se descarta (WARNING);
      staging CON journal → commit interrumpido: se completa (roll-forward).
  - `verify()`: archivos cuyo sha256 ya no coincide con el manifiesto.

`migrate_ledger_payload()` migra explícitamente `ledger.json` hasta
`LEDGER_SCHEMA_VERSION` (v0 = JSON legado del runner, sin versión ni txns;
v1 = sin moneda: todo en BASE_CURRENCY).

Uso:
    with StateStore(settings.state_dir) as store:
        with store.transaction(new_run_id(), as_of="2026-09-29"):
            store.write_json("ledger.json", ledger.to_dict())
            store.append_csv("rebalance_history.csv", row_df)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

from src.position_ledger import LEDGER_SCHEMA_VERSION
from src.settings import BASE_CURRENCY

logger = logging.getLogger("live_state")

STATE_SCHEMA_VERSION = 1
MANIFEST_NAME = "state_manifest.json"
LOCK_NAME = ".lock"
STAGING_DIRNAME = ".staging"
_JOURNAL_NAME = ".commit"


class StateLockedError(RuntimeError):
    """Otro proceso tiene el lock exclusivo del directorio de estado."""


def new_run_id(now: Optional[datetime] = None) -> str:
    """Id único y ordenable por tiempo: ``YYYYMMDDTHHMMSSZ-<8 hex>``."""
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


class StateStore:
    """Directorio de estado con lock exclusivo y commits por corrida.

    Args:
        root: Directorio de estado (p. ej. ``settings.state_dir``).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock_fd: Optional[int] = None
        self._run_id: Optional[str] = None
        self._as_of: Optional[str] = None
        self._staged: List[str] = []

    # -----------------------------------------------------------------
    # Lock
    # -----------------------------------------------------------------
    def acquire(self) -> None:
        """Toma el lock exclusivo y recupera staging huérfano.

        Raises:
            StateLockedError: Otro proceso (o store) tiene el lock.
        """
        if self._lock_fd is not None:
            raise RuntimeError(f"Lock of {self.root} already held by this store.")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / LOCK_NAME
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 256, 0).decode(errors="replace").strip() or "unknown"
            os.close(fd)
            raise StateLockedError(
                f"State dir {self.root} is locked by another run ({holder}); "
                f"lock file {lock_path}. Nothing was read or written: retry "
                "when that run finishes."
            ) from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, (f"pid={os.getpid()} since="
                       f"{datetime.now(timezone.utc).isoformat()}\n").encode(), 0)
        self._lock_fd = fd
        try:
            self._recover()
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Libera el lock (el archivo .lock se conserva a propósito)."""
        if self._lock_fd is None:
            return
        fd, self._lock_fd = self._lock_fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def close(self) -> None:
        """Aborta la transacción abierta (si hay) y libera el lock."""
        try:
            self.abort()
        finally:
            self.release()

    def __enter__(self) -> "StateStore":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Lectura del estado comprometido
    # -----------------------------------------------------------------
    def path(self, name: str) -> Path:
        """Ruta del archivo comprometido ``name`` (puede no existir)."""
        return self.root / _check_name(name)

    def read_manifest(self) -> Optional[Dict[str, Any]]:
        """Manifiesto del último commit, o None si nunca hubo uno."""
        p = self.root / MANIFEST_NAME
        return json.loads(p.read_text()) if p.exists() else None

    def verify(self) -> List[str]:
        """Archivos del manifiesto ausentes o con sha256 distinto (ordenados)."""
        manifest = self.read_manifest() or {}
        return sorted(
            name for name, digest in manifest.get("sha256", {}).items()
            if not (self.root / name).exists()
            or _sha256(self.root / name) != digest
        )

    # -----------------------------------------------------------------
    # Transacción
    # -----------------------------------------------------------------
    def begin(self, run_id: str, as_of: Optional[str] = None) -> None:
        """Abre la transacción ``run_id``; las escrituras van a staging."""
        if self._lock_fd is None:
            raise RuntimeError(
                "StateStore.begin requires the lock: use `with StateStore(root)` "
                "or call acquire() first."
            )
        if self._run_id is not None:
            raise RuntimeError(f"Transaction {self._run_id} is still open.")
        run_dir = self._staging_dir(_check_name(run_id))
        run_dir.mkdir(parents=True)
        self._run_id, self._as_of, self._staged = run_id, as_of, []

    @contextmanager
    def transaction(self, run_id: str, as_of: Optional[str] = None) -> Iterator["StateStore"]:
        """begin → cuerpo → commit; ante excepción en el cuerpo, abort."""
        self.begin(run_id, as_of=as_of)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        self.commit()

    def write_text(self, name: str, text: str) -> None:
        _write_durable(self._stage_path(name), text.encode("utf-8"))

    def write_json(self, name: str, obj: Any) -> None:
        self.write_text(name, json.dumps(obj, indent=2))

    def write_csv(self, name: str, df: pd.DataFrame, index: bool = False) -> None:
        path = self._stage_path(name)
        df.to_csv(path, index=index)
        _fsync_file(path)

    def append_csv(self, name: str, rows: pd.DataFrame) -> None:
        """Filas ya comprometidas (o ya staged) + ``rows``; columnas por unión.

        Las filas previas se leen como texto crudo: se reescriben idénticas
        (sin reformatear floats); columnas nuevas quedan vacías en ellas.
        """
        if rows is None or rows.empty:
            return
        path = self._stage_path(name)
        base = path if path.exists() else self.root / name
        if base.exists():
            prev = pd.read_csv(base, dtype=str, keep_default_na=False)
            rows = pd.concat([prev, rows], ignore_index=True)
        rows.to_csv(path, index=False)
        _fsync_file(path)

    def commit(self) -> Dict[str, Any]:
        """Publica la transacción; devuelve el manifiesto escrito.

        El journal marca el punto de commit: si el proceso muere mientras se
        mueven archivos, el próximo ``acquire()`` completa el commit.
        """
        run_dir = self._require_txn()
        staged = [n for n in self._staged if (run_dir / n).exists()]
        prev = self.read_manifest() or {}
        hashes = {
            n: h for n, h in prev.get("sha256", {}).items()
            if (self.root / n).exists()
        }
        hashes.update({n: _sha256(run_dir / n) for n in staged})
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_id": self._run_id,
            "as_of": self._as_of,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": sorted(hashes),
            "sha256": {n: hashes[n] for n in sorted(hashes)},
        }
        _write_durable(run_dir / MANIFEST_NAME,
                       json.dumps(manifest, indent=2).encode("utf-8"))
        order = staged + [MANIFEST_NAME]
        journal_tmp = run_dir / (_JOURNAL_NAME + ".tmp")
        _write_durable(journal_tmp, json.dumps({"order": order}).encode("utf-8"))
        os.replace(journal_tmp, run_dir / _JOURNAL_NAME)
        _fsync_dir(run_dir)
        self._publish(run_dir, order)
        logger.info("Committed run %s (as_of=%s): %s", self._run_id, self._as_of, staged)
        self._clear_txn()
        return manifest

    def abort(self) -> None:
        """Descarta la transacción abierta; nunca un commit ya journalizado."""
        if self._run_id is None:
            return
        run_id, run_dir = self._run_id, self._staging_dir(self._run_id)
        self._clear_txn()
        if (run_dir / _JOURNAL_NAME).exists():
            logger.error(
                "Run %s failed mid-commit; staging kept at %s and completed on "
                "the next acquire().", run_id, run_dir,
            )
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.info("Run %s aborted: staged writes discarded.", run_id)

    # -----------------------------------------------------------------
    # Internos
    # -----------------------------------------------------------------
    def _staging_dir(self, run_id: str) -> Path:
        return self.root / STAGING_DIRNAME / run_id

    def _require_txn(self) -> Path:
        if self._run_id is None:
            raise RuntimeError("No open transaction: call begin(run_id) first.")
        return self._staging_dir(self._run_id)

    def _stage_path(self, name: str) -> Path:
        run_dir = self._require_txn()
        _check_name(name)
        if name not in self._staged:
            self._staged.append(name)
        return run_dir / name

    def _clear_txn(self) -> None:
        self._run_id, self._as_of, self._staged = None, None, []

    def _publish(self, run_dir: Path, order: List[str]) -> None:
        """Mueve a root los archivos aún en staging (idempotente) y limpia."""
        for name in order:
            src = run_dir / name
            if src.exists():
                os.replace(src, self.root / name)
        _fsync_dir(self.root)
        shutil.rmtree(run_dir, ignore_errors=True)

    def _recover(self) -> None:
        staging = self.root / STAGING_DIRNAME
        if not staging.is_dir():
            return
        for run_dir in sorted(staging.iterdir()):
            journal = run_dir / _JOURNAL_NAME
            if run_dir.is_dir() and journal.exists():
                logger.warning(
                    "Completing interrupted commit of run %s (roll-forward from %s).",
                    run_dir.name, run_dir,
                )
                self._publish(run_dir, json.loads(journal.read_text())["order"])
            else:
                logger.warning(
                    "Discarding uncommitted staging of run %s (%s): the previous "
                    "committed state is kept.", run_dir.name, run_dir,
                )
                if run_dir.is_dir():
                    shutil.rmtree(run_dir, ignore_errors=True)
                else:
                    run_dir.unlink()


# ===================================================================
# ledger.json: migración explícita de esquema
# ===================================================================
def _ledger_v0_to_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    """v0 (runner legado, sin versión): añade schema_version=1 y txns vacías."""
    out = dict(payload)
    out["schema_version"] = 1
    out.setdefault("txns", [])
    return out


def _ledger_v1_to_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """v1 (sin moneda): ledger y cada lote en BASE_CURRENCY (lo unico que existia)."""
    out = dict(payload)
    out["schema_version"] = 2
    out.setdefault("base_currency", BASE_CURRENCY)
    out["lots"] = [{"currency": out["base_currency"], **lot}
                   for lot in payload.get("lots", [])]
    return out


_LEDGER_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    0: _ledger_v0_to_v1,
    1: _ledger_v1_to_v2,
}


def migrate_ledger_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lleva un payload de ledger.json a ``LEDGER_SCHEMA_VERSION`` (sin mutarlo).

    Raises:
        ValueError: schema_version desconocido o más nuevo que este código.
    """
    version = payload.get("schema_version", 0)
    if not isinstance(version, int) or not 0 <= version <= LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"ledger.json schema_version={version!r} is not supported "
            f"(this code reads 0..{LEDGER_SCHEMA_VERSION})."
        )
    out = dict(payload)
    while version < LEDGER_SCHEMA_VERSION:
        out = _LEDGER_MIGRATIONS[version](out)
        logger.info("ledger.json migrated schema_version %d -> %d",
                    version, out["schema_version"])
        version = out["schema_version"]
    return out


# ===================================================================
# Helpers de archivo
# ===================================================================
def _check_name(name: str) -> str:
    """Nombre plano de archivo de estado (sin rutas, ocultos ni reservados)."""
    if (not isinstance(name, str) or not name or Path(name).name != name
            or name.startswith(".") or name == MANIFEST_NAME):
        raise ValueError(f"Invalid or reserved state file name: {name!r}")
    return name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_durable(path: Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # algunos FS no soportan fsync de directorios
        pass
    finally:
        os.close(fd)
