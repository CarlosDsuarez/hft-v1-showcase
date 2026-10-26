"""Estrategias intradía. El registro vive en ``base``.

Importar este paquete registra las estrategias ya implementadas.
Hoy son el ORB (HFT-09A), el momentum intradía (HFT-09B), la
reversión a VWAP (HFT-09C), los pares OU (HFT-09D) y
Avellaneda–Stoikov (HFT-09E, ``research_only``).

El momentum horario (``hourly_momentum``) es exploratorio. Se registra
al importar ``src.hft.strategies.hourly_momentum``, no al importar este
paquete, y no entra en ningún set por defecto de backtest ni de sesión.
La sesión la suma solo con ``--with hourly_momentum``.
"""

from src.hft.strategies.avellaneda_stoikov import AvellanedaStoikov
from src.hft.strategies.base import (
    STRATEGIES,
    Strategy,
    StrategyContext,
    register,
    validate_strategy,
)
from src.hft.strategies.intraday_momentum import IntradayMomentum
from src.hft.strategies.orb import OpeningRangeBreakout
from src.hft.strategies.pairs_ou import PairsOu
from src.hft.strategies.vwap_reversion import VwapReversion

__all__ = [
    "STRATEGIES",
    "AvellanedaStoikov",
    "IntradayMomentum",
    "OpeningRangeBreakout",
    "PairsOu",
    "Strategy",
    "StrategyContext",
    "VwapReversion",
    "register",
    "validate_strategy",
]
