"""
settings.py — Configuracion centralizada y rutas ancladas.

Fuente unica de:
  - Constantes compartidas (SEED, TRADING_DAYS, INITIAL_CASH, LOOKBACK_DAYS,
    BASE_CURRENCY, BROKER_ENV). Los modulos las importan de aqui.
  - Rutas de runtime (data/, data/live_state/, results/), SIEMPRE absolutas
    y nunca relativas al cwd:
        repo_root   = raiz del repo (Path(__file__).parents[1]); codigo,
                      universos (universe*.csv) y `.env`.
        home        = $HFT_HOME si esta definido (ruta absoluta), si no repo_root.
        data_dir    = home / "data"
        state_dir   = home / "data" / "live_state"
        results_dir = home / "results"
        hft_dir     = data_dir / "hft"   (cache del intradía: calendario)
        bars_dir    = data_dir / "bars"  (barras intradía CSV.gz, sin parquet)
  - Lectura de `.env` pura: `read_dotenv()` devuelve un dict, NO muta
    os.environ y solo busca en repo_root.
  - `setup_logging(level)`: unico punto que configura el root logger. Solo lo
    llaman entrypoints (bloques __main__, scripts/, experiments/); importar
    la libreria nunca añade handlers.
  - Logs por corrida: `run_logging(run_id)` marca cada registro con
    ``run=<run_id>`` (`RunIdFilter`; fuera de una corrida, ``run=-``) y
    `log_to_file(dir)` añade, solo mientras dura el bloque, un archivo
    rotativo ``<dir>/runner.log`` (el runner: ``state_dir/logs/``).

Uso:
    from src.settings import load_settings
    s = load_settings()
    s.state_dir / "ledger.json"
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Union

# ---------------------------------------------------------------------------
# Constantes compartidas (no cambiar valores sin revisar estrategia/backtests)
# ---------------------------------------------------------------------------
SEED = 42
TRADING_DAYS = 252
INITIAL_CASH = 1_000_000.0
# Ventana default de descarga Yahoo: ~3.5 meses calendario
# (≈70-73 hábiles; cubre WARMUP 60+5 del runner)
LOOKBACK_DAYS = 105
BASE_CURRENCY = "USD"
BROKER_ENV = "paper"  # unico entorno soportado por src/brokers/alpaca_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_ENV_VAR = "HFT_HOME"


@dataclass(frozen=True)
class Settings:
    """Configuracion resuelta (rutas absolutas + valores base)."""

    repo_root: Path
    data_dir: Path
    state_dir: Path
    results_dir: Path
    seed: int = SEED
    base_currency: str = BASE_CURRENCY
    initial_cash: float = INITIAL_CASH
    broker_env: str = BROKER_ENV
    trading_days: int = TRADING_DAYS
    lookback_days: int = LOOKBACK_DAYS

    @property
    def price_backup_csv(self) -> Path:
        """Fallback de precios (CsvProvider) cuando Yahoo no responde."""
        return self.data_dir / "raw_prices_backup.csv"

    @property
    def volume_backup_csv(self) -> Path:
        """Volumenes del fallback CSV (opcional; sin archivo → volumen NaN)."""
        return self.data_dir / "raw_volumes_backup.csv"

    @property
    def bars_cache_dir(self) -> Path:
        """Cache parquet por simbolo de CachedProvider (opt-in)."""
        return self.data_dir / "cache" / "bars"

    @property
    def logs_dir(self) -> Path:
        """Logs rotativos y alertas del runner vivo (fuera del manifiesto)."""
        return self.state_dir / LOGS_DIRNAME

    def universe_path(self, name: str = "universe.csv") -> Path:
        """Archivo de universo versionado en la raiz del repo."""
        return self.repo_root / name

    @property
    def hft_dir(self) -> Path:
        """Cache del intradía (calendario y manifiestos de ajuste)."""
        return self.data_dir / "hft"

    @property
    def bars_dir(self) -> Path:
        """Almacén de barras intradía en CSV.gz. No usa parquet."""
        return self.data_dir / "bars"


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Resuelve Settings desde `env` (default: os.environ), nunca desde el cwd.

    Raises:
        ValueError: HFT_HOME definido pero no es ruta absoluta.
    """
    source = os.environ if env is None else env
    repo_root = REPO_ROOT
    raw_home = (source.get(HOME_ENV_VAR) or "").strip()
    if raw_home:
        home = Path(raw_home).expanduser()
        if not home.is_absolute():
            raise ValueError(
                f"{HOME_ENV_VAR} debe ser una ruta absoluta (recibido: {raw_home!r})."
            )
    else:
        home = repo_root
    data_dir = home / "data"
    return Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        state_dir=data_dir / "live_state",
        results_dir=home / "results",
    )


def read_dotenv(path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Parser minimo de `.env` (KEY=valor). Devuelve dict; no toca os.environ.

    Args:
        path: Archivo a leer. Default: `<repo_root>/.env` (nunca el del cwd).
    """
    p = Path(path) if path is not None else REPO_ROOT / ".env"
    if not p.exists():
        return {}
    values: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [run=%(run_id)s] — %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOGS_DIRNAME = "logs"
LOG_FILE_NAME = "runner.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 10
NO_RUN_ID = "-"

_RUN_ID: ContextVar[str] = ContextVar("hft_run_id", default=NO_RUN_ID)


def current_run_id() -> Optional[str]:
    """run_id de la corrida en curso (`run_logging`), o None fuera de una."""
    run_id = _RUN_ID.get()
    return None if run_id == NO_RUN_ID else run_id


class RunIdFilter(logging.Filter):
    """Pone ``record.run_id`` (corrida en curso o "-"); nunca descarta registros.

    Va en el handler, no en el logger: así cubre también los registros que
    propagan desde loggers hijos (market_data, live_state, ...).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = _RUN_ID.get()
        return True


def _add_run_id_filter(handler: logging.Handler) -> None:
    if not any(isinstance(f, RunIdFilter) for f in handler.filters):
        handler.addFilter(RunIdFilter())


def setup_logging(level: Union[int, str] = logging.INFO) -> None:
    """Configura el root logger (solo entrypoints). Idempotente: no duplica handlers."""
    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level.upper() if isinstance(level, str) else level)
    for handler in root.handlers:
        _add_run_id_filter(handler)


@contextmanager
def run_logging(run_id: str) -> Iterator[str]:
    """Ámbito de una corrida: todo registro emitido dentro lleva ``run_id``."""
    token = _RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _RUN_ID.reset(token)


@contextmanager
def log_to_file(
    log_dir: Path,
    level: int = logging.INFO,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> Iterator[Path]:
    """Añade al root logger, solo durante el bloque, ``log_dir/runner.log``
    rotativo (``max_bytes`` × ``backup_count``) con el formato de LOG_FORMAT.

    Si el root está por encima de ``level`` (sin `setup_logging`), lo baja a
    ``level`` mientras dura el bloque para que el archivo tenga la corrida
    completa; al salir restaura nivel y handlers.

    Rotación no segura entre procesos: el runner solo lo usa con el lock de
    ``state_dir`` tomado (una corrida escritora a la vez).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_FILE_NAME
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    _add_run_id_filter(handler)
    root = logging.getLogger()
    prev_level = root.level
    if root.getEffectiveLevel() > level:
        root.setLevel(level)
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_level)
