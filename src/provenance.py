"""
provenance.py — Procedencia de una corrida viva (ARQ-21).

Qué código, qué config y qué datos produjeron cada fila de
``rebalance_history.csv``:

  - `code_version()`: git SHA de ``repo_root`` (+ ``dirty`` si hay cambios
    sin commitear en archivos versionados). Sin git (imagen Docker sin
    ``.git``) se usa ``$HFT_GIT_SHA`` (``docker build --build-arg
    GIT_SHA=...``); sin ninguno, ``"unknown"``. Nunca lanza.
  - `run_provenance()`: columnas de procedencia de la fila de history
    (git, fuente de datos, versiones de esquema de config/estado/ledger).
    El hash de la config ya lo registra el runner (``config_hash``).

Uso:
    from src.provenance import code_version, run_provenance
    row.update(run_provenance(code_version(), data_source="yahoo"))
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.live_state import STATE_SCHEMA_VERSION
from src.position_ledger import LEDGER_SCHEMA_VERSION
from src.schemas import CONFIG_SCHEMA_VERSION
from src.settings import REPO_ROOT

logger = logging.getLogger("provenance")

GIT_SHA_ENV_VAR = "HFT_GIT_SHA"
UNKNOWN = "unknown"
_GIT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class CodeVersion:
    """Versión del código: ``sha`` (o "unknown") y ``dirty`` (None = no se sabe)."""

    sha: str = UNKNOWN
    dirty: Optional[bool] = None

    @property
    def short(self) -> str:
        tag = self.sha[:12] if self.sha != UNKNOWN else UNKNOWN
        return f"{tag}-dirty" if self.dirty else tag


def _git(repo_root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True,
        check=True, timeout=_GIT_TIMEOUT_S,
    )
    return out.stdout.strip()


def code_version(
    repo_root: Path = REPO_ROOT,
    env: Optional[Mapping[str, str]] = None,
) -> CodeVersion:
    """SHA del checkout en ``repo_root``; ``$HFT_GIT_SHA`` si no hay git.

    ``dirty`` solo mira archivos versionados (``data/``, ``results/`` y
    otros no rastreados no cuentan). Nunca lanza: ante cualquier fallo
    devuelve lo que sepa (o ``CodeVersion()``).
    """
    source = os.environ if env is None else env
    try:
        sha = _git(repo_root, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        sha = (source.get(GIT_SHA_ENV_VAR) or "").strip()
        logger.debug("git no disponible en %s (%s); %s=%r", repo_root, exc,
                     GIT_SHA_ENV_VAR, sha)
        return CodeVersion(sha=sha or UNKNOWN, dirty=None)
    try:
        dirty: Optional[bool] = bool(
            _git(repo_root, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return CodeVersion(sha=sha, dirty=dirty)


def run_provenance(code: CodeVersion, data_source: str = "") -> Dict[str, Any]:
    """Columnas de procedencia de una fila de rebalance_history.

    Args:
        code: Versión del código de la corrida (`code_version()`).
        data_source: Fuente que sirvió los precios (``Bars.meta.source``,
            p. ej. "yahoo", "csv", "cache+yahoo").
    """
    return {
        "git_sha": code.sha,
        "git_dirty": code.dirty,
        "data_source": data_source,
        "config_schema": CONFIG_SCHEMA_VERSION,
        "state_schema": STATE_SCHEMA_VERSION,
        "ledger_schema": LEDGER_SCHEMA_VERSION,
    }
