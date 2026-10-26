"""SHOWCASE: factory for interface-only strategy stubs.

Four of the six intraday strategies are published as stubs. Each stub keeps
the registry contract (``id``, ``TheoryCard``, frozen ``params`` dataclass,
``required_history``, ``on_bar``) and the constructor signatures the session
and backtest use, so the engine, risk layer, simulator and reports run
unchanged. ``on_bar`` never emits an order intent. The signal logic, its
parameters and its research notes live in the private repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from src.hft.strategies.base import StrategyContext
from src.hft.types import OrderIntent, TheoryCard


@dataclass(frozen=True)
class StubParams:
    """No published parameters."""


def stub_theory(*, title: str, references: Tuple[str, ...], research_only: bool = False) -> TheoryCard:
    omitted = "SHOWCASE: omitted from the public version"
    return TheoryCard(
        hypothesis=f"{title}. {omitted}.",
        mechanism=omitted,
        references=references,
        data_requirements=("1-minute bars from the engine's bar store",),
        params=(),
        holding_period="intraday, flat by the close",
        failure_regimes=(omitted,),
        kill_criteria=("go/no-go gates in src/hft/decision.py",),
        research_only=research_only,
    )


class StubStrategy:
    """Base for interface-only strategies: valid for the registry, never trades."""

    id: str = ""
    theory: TheoryCard

    def __init__(self, params: StubParams | None = None, **_ignored) -> None:
        self.params = params or StubParams()

    @classmethod
    def from_config(cls, config, sessions=(), **inputs):
        """Same signature as the private strategies; inputs are kept, not used."""
        strategy = cls()
        strategy.sessions = tuple(sessions)
        strategy.universe = tuple(getattr(config, "universe", ()) or ())
        for name, value in inputs.items():   # e.g. closes=, shortable=
            setattr(strategy, name, value)
        return strategy

    def required_history(self) -> Tuple[int, int]:
        return (0, 0)

    def on_bar(self, ctx: StrategyContext) -> Sequence[OrderIntent]:
        return ()

    def take_cancels(self) -> Tuple[str, ...]:
        return ()
