"""
history_requirements.py — Historia mínima que cada componente declara.

Convención: ``required_history(...)`` de cada módulo devuelve el número de
**filas hábiles con índice <= t** que necesita para producir su salida sin
NaN (t incluido). El runner toma el máximo, suma un margen y lo convierte a
días calendario para la descarga.

Modos:
  - ``STRICT``: si falta historia, ``InsufficientHistoryError``. Default en vivo.
  - ``PERMISSIVE``: el componente excluye lo que no puede calcular y lo
    registra en su meta (nunca rellena con 0 en silencio).
"""

from __future__ import annotations

import math
from typing import Mapping, Union

from src.settings import TRADING_DAYS

STRICT = "strict"
PERMISSIVE = "permissive"
HISTORY_MODES = (STRICT, PERMISSIVE)

# Filas extra sobre el máximo declarado (festivos, huecos de datos, NaN iniciales).
DEFAULT_MARGIN_ROWS = 21
# Holgura calendario adicional (festivos de mercado ≈ 10/año, fines de semana largos).
_CALENDAR_SLACK_DAYS = 10


class InsufficientHistoryError(ValueError):
    """Historia/datos insuficientes para algún componente.

    ``missing``: componente → filas requeridas (int) o motivo (str, p.ej.
    "no data" cuando hay filas suficientes pero la serie está vacía).
    """

    def __init__(self, available: int, missing: Mapping[str, Union[int, str]]):
        self.available = int(available)
        self.missing = dict(missing)
        detail = ", ".join(
            f"{k} needs {v}" if isinstance(v, int) else f"{k}: {v}"
            for k, v in self.missing.items()
        )
        super().__init__(
            f"Insufficient history: {self.available} rows available; {detail}"
        )


def validate_mode(mode: str) -> str:
    if mode not in HISTORY_MODES:
        raise ValueError(f"history mode must be one of {HISTORY_MODES}, got {mode!r}")
    return mode


def missing_requirements(available: int, requirements: Mapping[str, int]) -> dict:
    """Subconjunto de ``requirements`` que ``available`` filas no cubren."""
    return {k: int(v) for k, v in requirements.items() if available < v}


def check_history(available: int, requirements: Mapping[str, int]) -> None:
    """Lanza ``InsufficientHistoryError`` si alguna exigencia no se cumple."""
    missing = missing_requirements(available, requirements)
    if missing:
        raise InsufficientHistoryError(available, missing)


def business_rows_to_calendar_days(rows: int) -> int:
    """Días calendario a descargar para obtener al menos ``rows`` filas hábiles."""
    return int(math.ceil(rows * 365.25 / TRADING_DAYS)) + _CALENDAR_SLACK_DAYS
