"""
test_runner_currency.py — Runner con universo mixto COP/USD (sin red).

Datos via InstrumentProvider (id ↔ yahoo) + FX: estrategia y ledger ven USD;
los lotes guardan la moneda local; el plan broker marca not_routable lo que
Alpaca no opera.
"""

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.brokers.alpaca_adapter import AlpacaClient, BrokerConfig
from src.fx import MissingFxError
from src.instruments import load_instruments
from src.market_data import CachedProvider, FakeProvider, FallbackChain, InstrumentProvider
from src.portfolio_state import PortfolioState
from src.position_ledger import PositionLedger
from src.settings import load_settings
from src.strategy import load_strategy_config
from tests.test_runner_paths import N_ROWS, TICKERS, _fake_prices, after_close

HEADER = ("ticker,yahoo_ticker,broker_symbol,market,currency,tradable_brokers,"
          "asset_type,sector")
START = "2024-01-01"


@pytest.fixture
def master(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("\n".join([
        HEADER,
        "USA1,USA1,USA1,us,USD,alpaca,stock,Tech",
        "USA2,USA2,USA2,us,USD,alpaca,stock,Tech",
        "COL1,COL1.CL,,co,COP,,stock,Energy",
    ]) + "\n", encoding="utf-8")
    return load_instruments(path)


def _yahoo_close(n=30):
    idx = pd.bdate_range(START, periods=n)
    return pd.DataFrame({"USA1": np.linspace(10.0, 12.0, n), "USA2": 20.0,
                         "COL1.CL": 8_000.0, "USDCOP=X": 4_000.0}, index=idx)


def _use_universe(monkeypatch, ids):
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": list(ids)}))


class TestProviderWiring:
    def test_default_provider_translates_ids_with_master(self, tmp_path):
        s = load_settings(env={"HFT_HOME": str(tmp_path)})
        provider = runner.make_market_data_provider(s)
        assert isinstance(provider, InstrumentProvider)
        assert isinstance(provider.inner, FallbackChain)
        assert provider.instruments["ICOLCAP"].yahoo_symbol == "ICOLCAP.CL"

    def test_cache_stores_yahoo_symbols_behind_the_adapter(self, tmp_path):
        s = load_settings(env={"HFT_HOME": str(tmp_path)})
        provider = runner.make_market_data_provider(s, sources=("csv",), cache=True)
        assert isinstance(provider.inner, CachedProvider)


class TestFetchDataInBase:
    def _fetch(self, master, close):
        provider = InstrumentProvider(FakeProvider(close), master)
        data = runner.fetch_data(START, {"sizing": 5}, provider=provider,
                                 now=after_close(close.index[-1]))
        return data, provider

    def test_local_prices_converted_before_strategy(self, master, monkeypatch):
        _use_universe(monkeypatch, ["USA1", "USA2", "COL1"])
        data, _ = self._fetch(master, _yahoo_close())
        assert data.prices["COL1"].iloc[-1] == pytest.approx(2.0)  # 8000 / 4000
        assert data.prices["USA2"].iloc[-1] == 20.0
        assert data.currencies == {"USA1": "USD", "USA2": "USD", "COL1": "COP"}

    def test_base_only_universe_downloads_no_fx(self, master, monkeypatch):
        _use_universe(monkeypatch, ["USA1", "USA2"])
        data, provider = self._fetch(master, _yahoo_close())
        assert len(provider.inner.calls) == 1
        assert "USDCOP=X" not in provider.inner.calls[0][0]

    def test_missing_fx_aborts_fetch(self, master, monkeypatch):
        _use_universe(monkeypatch, ["USA1", "COL1"])
        with pytest.raises(MissingFxError, match="COP"):
            self._fetch(master, _yahoo_close().drop(columns="USDCOP=X"))

    def test_run_once_missing_fx_is_data_error(self, master, tmp_path, monkeypatch):
        monkeypatch.setenv("HFT_HOME", str(tmp_path / "home"))
        _use_universe(monkeypatch, ["USA1", "COL1"])
        close = _yahoo_close().drop(columns="USDCOP=X")
        provider = InstrumentProvider(FakeProvider(close), master)
        monkeypatch.setattr(runner, "make_market_data_provider", lambda *a, **k: provider)
        monkeypatch.setattr(runner, "utcnow", lambda: after_close(close.index[-1]))
        assert runner.run_once(force=True, start=START) == runner.EXIT_DATA
        assert not (tmp_path / "home" / "data" / "live_state" / "ledger.json").exists()


class TestSimLedgerCurrency:
    def test_sim_plan_records_lot_currency(self):
        sim = runner.build_sim_plan(
            PositionLedger(1_000.0), pd.Series({"USA1": 0.5, "COL1": 0.5}),
            pd.Series({"USA1": 10.0, "COL1": 2.0}), pd.Timestamp("2024-02-09"),
            load_strategy_config(), currencies={"USA1": "USD", "COL1": "COP"})
        lots = {lot["ticker"]: lot["currency"]
                for lot in sim.ledger_after.to_dict()["lots"]}
        assert lots == {"USA1": "USD", "COL1": "COP"}

    def test_run_once_passes_instrument_currencies_to_sim(self, master, tmp_path,
                                                          monkeypatch):
        monkeypatch.setenv("HFT_HOME", str(tmp_path / "home"))
        _use_universe(monkeypatch, [*TICKERS, "COL1"])
        close = _fake_prices()
        close["COL1.CL"] = close["AAA"] * 4_000.0
        close["USDCOP=X"] = 4_000.0
        volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
        provider = InstrumentProvider(FakeProvider(close, volume), master)
        monkeypatch.setattr(runner, "make_market_data_provider", lambda *a, **k: provider)
        monkeypatch.setattr(runner, "utcnow", lambda: after_close(close.index[-1]))
        seen = {}
        real = runner.build_sim_plan

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(runner, "build_sim_plan", spy)
        assert len(close) == N_ROWS
        assert runner.run_once(force=True, start=START) == 0
        assert seen["currencies"]["COL1"] == "COP"
        assert seen["currencies"]["AAA"] == "USD"


class TestBrokerPlanRouting:
    def test_co_local_order_is_not_routable_in_broker_plan(self, tmp_path,
                                                           monkeypatch):
        def no_http(*a, **k):
            raise AssertionError("dry-run must not touch the network")

        monkeypatch.setattr("urllib.request.urlopen", no_http)
        client = AlpacaClient(BrokerConfig("PKTEST1234", "s",
                                           "https://paper-api.alpaca.markets"),
                              kill_dir=tmp_path)  # maestro del repo
        portfolio = PortfolioState(
            positions=pd.Series(dtype=float), avg_entry=pd.Series(dtype=float),
            cash=1_000.0, equity=1_000.0, as_of=pd.Timestamp("2024-02-09"),
            source="broker")
        bplan = runner.BrokerPlan(
            account=runner.BrokerAccount(summary={"market_open": True},
                                         portfolio=portfolio),
            orders=pd.DataFrame([
                {"ticker": "ICOLCAP", "side": "buy", "qty": 10.0, "notional": 30.0},
                {"ticker": "AAPL", "side": "buy", "qty": 1.0, "notional": 200.0},
            ]))
        execution = runner.execute_broker(client, bplan, pd.Timestamp("2024-02-09"),
                                          tmp_path, submit=False)
        plan = pd.read_csv(tmp_path / "broker_plan_2024-02-09.csv")
        assert dict(zip(plan["ticker"], plan["status"])) == {
            "ICOLCAP": "not_routable", "AAPL": "dry_run"}
        assert execution.n_not_routable == 1
