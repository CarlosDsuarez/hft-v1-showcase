"""Showcase: stubbed strategies keep the registry/engine contract and never trade."""
import pytest

import src.hft.strategies as strategies
import src.hft.strategies.hourly_momentum  # noqa: F401  (registers on import)
from src.hft.config import load_hft_config
from src.hft.strategies.base import validate_strategy

STUBS = ("intraday_momentum", "pairs_ou", "hourly_momentum", "avellaneda_stoikov")


def test_all_six_strategies_are_registered():
    assert {"orb", "vwap_reversion", *STUBS} <= set(strategies.STRATEGIES)


@pytest.mark.parametrize("strategy_id", STUBS)
def test_stub_passes_registry_contract_and_emits_nothing(strategy_id):
    stub = strategies.STRATEGIES[strategy_id]
    assert validate_strategy(stub) == ()
    built = type(stub).from_config(load_hft_config(), ())
    assert built.on_bar(ctx=None) == ()
    assert built.required_history() == (0, 0)


def test_market_maker_stays_research_only():
    assert strategies.STRATEGIES["avellaneda_stoikov"].theory.research_only is True
