"""Hourly intraday momentum (exploratory) — SHOWCASE STUB.

Registered at import but outside the default backtest/session sets, as in the
private version. Signal logic omitted.
"""

from __future__ import annotations

from src.hft.strategies._showcase_stub import StubStrategy, stub_theory
from src.hft.strategies.base import register

STRATEGY_ID = "hourly_momentum"


class HourlyMomentum(StubStrategy):
    id = STRATEGY_ID
    theory = stub_theory(
        title="Exploratory hourly variant of intraday momentum",
        references=("Gao, Han, Li & Zhou (2018), Market Intraday Momentum, JFE",),
    )


def _register_default() -> HourlyMomentum:
    strategy = HourlyMomentum()
    register(strategy)
    return strategy


# Registered, but not part of the default backtest and session sets.
DEFAULT_HOURLY_MOMENTUM = _register_default()
