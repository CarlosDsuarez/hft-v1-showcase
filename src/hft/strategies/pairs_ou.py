"""Pairs trading on an Ornstein–Uhlenbeck spread — SHOWCASE STUB.

The private version fits each hand-picked pair intraday (hedge ratio,
stationarity test, mean-reversion speed), trades the spread's z-score with
leg-level hang/timeout handling, and submits both legs atomically. That logic
is omitted here; this module keeps the strategy's place in the registry,
session and backtest so the rest of the system runs unchanged.
"""

from __future__ import annotations

from src.hft.strategies._showcase_stub import StubStrategy, stub_theory
from src.hft.strategies.base import register

STRATEGY_ID = "pairs_ou"


class PairsOu(StubStrategy):
    id = STRATEGY_ID
    theory = stub_theory(
        title="Intraday pairs trading on a mean-reverting (OU) spread",
        references=("Gatev, Goetzmann & Rouwenhorst (2006), Pairs Trading, RFS",),
    )
    atomic_entries = True

    def hung_legs(self):
        return ()


def _register_default() -> PairsOu:
    strategy = PairsOu()
    register(strategy)
    return strategy


DEFAULT_PAIRS_OU = _register_default()
