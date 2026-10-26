"""Market intraday momentum — SHOWCASE STUB.

The private version trades the sign of the first half-hour return in the last
half hour of the session. Signal logic omitted. ``ConsolidatedCloses`` (the
daily-close input the backtest and runner pass in) is kept because other
modules build it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.hft.strategies._showcase_stub import StubStrategy, stub_theory
from src.hft.strategies.base import register

STRATEGY_ID = "intraday_momentum"


@dataclass(frozen=True)
class ConsolidatedCloses:
    """Daily closes for the symbol, indexed by session date. ``feed`` must be ``sip``."""

    close: pd.Series
    feed: str = "sip"

    def __post_init__(self) -> None:
        if not isinstance(self.close, pd.Series):
            raise TypeError("close must be a Series")
        feed = str(self.feed).strip().lower()
        if feed != "sip":
            raise ValueError("the prior close must be the consolidated daily close (feed=sip)")
        object.__setattr__(self, "feed", feed)
        object.__setattr__(self, "close", _dated_closes(self.close))


def _dated_closes(closes: pd.Series) -> pd.Series:
    if closes.empty:
        return pd.Series(dtype=float)
    index = [_index_date(value) for value in closes.index]
    out = pd.Series(closes.to_numpy(dtype=float), index=pd.Index(index, name="date"))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _index_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class IntradayMomentum(StubStrategy):
    id = STRATEGY_ID
    theory = stub_theory(
        title="First-half-hour return predicts the last half hour",
        references=("Gao, Han, Li & Zhou (2018), Market Intraday Momentum, JFE",),
    )


def _register_default() -> IntradayMomentum:
    strategy = IntradayMomentum()
    register(strategy)
    return strategy


DEFAULT_INTRADAY_MOMENTUM = _register_default()
