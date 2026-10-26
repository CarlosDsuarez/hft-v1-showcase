"""
test_multicurrency.py — Universo mixto COP/USD: maestro → datos locales + FX →
to_base → ledger en USD, con moneda por lote. Offline (FakeProvider).
"""

import pandas as pd
import pytest

from src.fx import to_base
from src.instruments import load_instruments
from src.market_data import FakeProvider, InstrumentProvider
from src.position_ledger import LEDGER_SCHEMA_VERSION, PositionLedger

DATES = pd.bdate_range("2024-01-01", periods=3)
START, END = "2024-01-01", "2024-01-06"
HEADER = ("ticker,yahoo_ticker,broker_symbol,market,currency,tradable_brokers,"
          "asset_type,sector")


@pytest.fixture
def master(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("\n".join([
        HEADER,
        "USA1,USA1,USA1,us,USD,alpaca,stock,Tech",
        "COL1,COL1.CL,,co,COP,,stock,Energy",
    ]) + "\n", encoding="utf-8")
    return load_instruments(path)


def _provider(master):
    # Precios en moneda local tal como los publica Yahoo + par USDCOP.
    close = pd.DataFrame({
        "USA1": [10.0, 11.0, 12.0],
        "COL1.CL": [8_000.0, 8_000.0, 12_500.0],   # COP
        "USDCOP=X": [4_000.0, 4_000.0, 5_000.0],   # COP por USD
    }, index=DATES)
    return InstrumentProvider(FakeProvider(close), master)


def _base_prices(master):
    provider = _provider(master)
    bars = provider.get_bars(["USA1", "COL1"], START, END)
    fx = provider.get_fx(["COP"], START, END)
    return to_base(bars.close, fx, master.currencies(bars.close.columns))


def _invested(master):
    prices = _base_prices(master)
    ledger = PositionLedger(initial_cash=1_000.0)
    trades = ledger.trades_from_target_weights(
        pd.Series({"USA1": 0.5, "COL1": 0.5}), prices.iloc[0])
    ledger.apply_trades(trades, prices.iloc[0], t=DATES[0],
                        currencies=master.currencies())
    return ledger, prices


class TestMixedUniverseNav:
    def test_prices_reach_ledger_in_usd(self, master):
        prices = _base_prices(master)
        # COL1: 8000 COP / 4000 = 2 USD; 12500 / 5000 = 2.5 USD.
        assert prices["COL1"].tolist() == pytest.approx([2.0, 2.0, 2.5])
        assert prices["USA1"].tolist() == [10.0, 11.0, 12.0]

    def test_nav_in_usd_with_known_fx(self, master):
        ledger, prices = _invested(master)
        pos = ledger.positions()
        assert pos["USA1"] == pytest.approx(50.0)     # 500 USD / 10
        assert pos["COL1"] == pytest.approx(250.0)    # 500 USD / 2
        assert ledger.nav(prices.iloc[0]) == pytest.approx(1_000.0)
        # t2: 50 × 12 + 250 × (12500 COP / 5000) = 600 + 625 USD.
        assert ledger.nav(prices.iloc[2]) == pytest.approx(1_225.0)

    def test_lots_record_instrument_currency(self, master):
        ledger, _ = _invested(master)
        lots = {lot["ticker"]: lot for lot in ledger.to_dict()["lots"]}
        assert lots["COL1"]["currency"] == "COP"
        assert lots["USA1"]["currency"] == "USD"
        # entry_price en moneda base del ledger
        assert lots["COL1"]["entry_price"] == pytest.approx(2.0)

    def test_currency_exposure_in_base(self, master):
        ledger, prices = _invested(master)
        exposure = ledger.currency_exposure(prices.iloc[2])
        assert exposure.to_dict() == pytest.approx({"COP": 625.0, "USD": 600.0})


class TestLotCurrency:
    def _ledger(self):
        lg = PositionLedger(initial_cash=1_000.0)
        lg.apply_trades(pd.DataFrame([{"ticker": "COL1", "shares": 10.0}]),
                        pd.Series({"COL1": 2.0}), currencies={"COL1": "COP"})
        return lg

    def test_roundtrip_keeps_currency_and_base(self):
        lg = self._ledger()
        payload = lg.to_dict()
        assert payload["schema_version"] == LEDGER_SCHEMA_VERSION == 2
        assert payload["base_currency"] == "USD"
        again = PositionLedger.from_dict(payload)
        assert again.to_dict() == payload

    def test_legacy_lots_default_to_base_currency(self):
        legacy = {"schema_version": 1, "cash": 0.0, "realized_pnl": 0.0,
                  "lots": [{"ticker": "AAA", "shares": 1.0, "entry_price": 5.0,
                            "entry_date": None}], "txns": []}
        lg = PositionLedger.from_dict(legacy)
        assert lg.base_currency == "USD"
        assert lg.to_dict()["lots"][0]["currency"] == "USD"

    def test_buy_without_mapping_inherits_held_currency(self):
        lg = self._ledger()
        lg.apply_trades(pd.DataFrame([{"ticker": "COL1", "shares": 1.0}]),
                        pd.Series({"COL1": 2.0}))
        assert {lot["currency"] for lot in lg.to_dict()["lots"]} == {"COP"}

    def test_new_ticker_without_mapping_is_base_currency(self):
        lg = PositionLedger(initial_cash=100.0)
        lg.apply_trades(pd.DataFrame([{"ticker": "AAA", "shares": 1.0}]),
                        pd.Series({"AAA": 5.0}))
        assert lg.to_dict()["lots"][0]["currency"] == "USD"

    def test_currency_change_for_held_ticker_rejected_atomically(self):
        lg = self._ledger()
        before = lg.to_dict()
        with pytest.raises(ValueError, match="COL1.*COP.*USD"):
            lg.apply_trades(pd.DataFrame([{"ticker": "COL1", "shares": 1.0}]),
                            pd.Series({"COL1": 2.0}), currencies={"COL1": "USD"})
        assert lg.to_dict() == before

    def test_invalid_currency_code_rejected(self):
        lg = PositionLedger(initial_cash=100.0)
        with pytest.raises(ValueError, match="currency"):
            lg.apply_trades(pd.DataFrame([{"ticker": "AAA", "shares": 1.0}]),
                            pd.Series({"AAA": 5.0}), currencies={"AAA": "cop"})
