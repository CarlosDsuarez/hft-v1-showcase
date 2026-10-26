"""Avellaneda–Stoikov market making (simulation only) — SHOWCASE STUB.

``research_only`` stays True: the engine refuses it in paper mode. Quoting
logic and parameter estimation omitted.
"""

from __future__ import annotations

from src.hft.strategies._showcase_stub import StubStrategy, stub_theory
from src.hft.strategies.base import register

STRATEGY_ID = "avellaneda_stoikov"


class AvellanedaStoikov(StubStrategy):
    id = STRATEGY_ID
    theory = stub_theory(
        title="Inventory-aware market making around a reservation price",
        references=("Avellaneda & Stoikov (2008), High-frequency trading in a limit order book, "
                    "Quantitative Finance",),
        research_only=True,
    )
    replaces_passive_limits = True


def _register_default() -> AvellanedaStoikov:
    strategy = AvellanedaStoikov()
    register(strategy)
    return strategy


DEFAULT_AVELLANEDA = _register_default()
