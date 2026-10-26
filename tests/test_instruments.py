"""
test_instruments.py — Maestro de instrumentos (src/instruments.py).

Offline: lee los CSV de universo del repo y CSV sinteticos en tmp_path.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.instruments import (
    KNOWN_BROKERS, REQUIRED_COLUMNS, Instrument, load_instruments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MIXED_PATH = REPO_ROOT / "universe_mixed.csv"
CO_PATH = REPO_ROOT / "universe_co.csv"
US_PATH = REPO_ROOT / "universe.csv"

HEADER = ("ticker,yahoo_ticker,broker_symbol,market,currency,tradable_brokers,"
          "asset_type,sector")


def _write(tmp_path, *rows, header=HEADER, name="instruments.csv"):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


class TestRepoMaster:
    def test_mixed_universe_loads_all_instruments(self):
        master = load_instruments(MIXED_PATH)
        assert len(master) == 120
        assert isinstance(master["AAPL"], Instrument)

    def test_co_local_uses_yahoo_suffix_and_is_not_routable(self):
        icolcap = load_instruments(MIXED_PATH)["ICOLCAP"]
        assert icolcap.yahoo_symbol == "ICOLCAP.CL"
        assert (icolcap.market, icolcap.currency) == ("co", "COP")
        assert icolcap.broker_symbol is None
        assert icolcap.tradable_brokers == ()
        assert not icolcap.routable("alpaca")

    def test_co_adr_is_routable_in_usd(self):
        ec = load_instruments(MIXED_PATH)["EC"]
        assert (ec.market, ec.currency, ec.broker_symbol) == ("co", "USD", "EC")
        assert ec.routable("alpaca")

    def test_class_share_symbols_both_ways(self):
        master = load_instruments(MIXED_PATH)
        assert master["BRK-B"].yahoo_symbol == "BRK-B"
        assert master["BRK-B"].broker_symbol == "BRK.B"
        assert master.by_broker_symbol("BRK.B").id == "BRK-B"
        assert master.by_yahoo_symbol("ICOLCAP.CL").id == "ICOLCAP"
        assert master.by_broker_symbol("NOPE") is None

    def test_default_path_resolves_from_repo_root_not_cwd(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert len(load_instruments()) == 120

    def test_co_file_defines_same_instruments_as_mixed(self):
        mixed = load_instruments(MIXED_PATH)
        co = load_instruments(CO_PATH)
        assert len(co) == 20
        for inst in co.values():
            assert mixed[inst.id] == inst

    def test_every_us_universe_id_is_defined(self):
        master = load_instruments(MIXED_PATH)
        us = pd.read_csv(US_PATH)
        for row in us.itertuples():
            inst = master[row.ticker]
            assert (inst.sector, inst.asset_type) == (row.sector, row.asset_type)
            assert inst.currency == "USD" and inst.routable("alpaca")

    def test_currencies_by_id(self):
        master = load_instruments(MIXED_PATH)
        assert master.currencies(["AAPL", "ICOLCAP", "EC"]) == {
            "AAPL": "USD", "ICOLCAP": "COP", "EC": "USD"}


class TestValidation:
    def test_universe_without_master_columns_rejected(self):
        with pytest.raises(ValueError, match="missing columns"):
            load_instruments(US_PATH)

    def test_required_columns_documented(self):
        assert set(REQUIRED_COLUMNS) == set(HEADER.split(","))
        assert "alpaca" in KNOWN_BROKERS

    def test_duplicate_id_rejected(self, tmp_path):
        path = _write(tmp_path,
                      "AAA,AAA,AAA,us,USD,alpaca,stock,Tech",
                      "AAA,AAA2,AAA2,us,USD,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="duplicate ticker.*AAA"):
            load_instruments(path)

    def test_duplicate_yahoo_symbol_rejected(self, tmp_path):
        path = _write(tmp_path,
                      "AAA,X,AAA,us,USD,alpaca,stock,Tech",
                      "BBB,X,BBB,us,USD,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="duplicate yahoo_ticker.*X"):
            load_instruments(path)

    def test_duplicate_broker_symbol_rejected(self, tmp_path):
        path = _write(tmp_path,
                      "AAA,AAA,Z,us,USD,alpaca,stock,Tech",
                      "BBB,BBB,Z,us,USD,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="duplicate broker_symbol.*Z"):
            load_instruments(path)

    def test_unknown_broker_rejected(self, tmp_path):
        path = _write(tmp_path, "AAA,AAA,AAA,us,USD,ibkr,stock,Tech")
        with pytest.raises(ValueError, match="unknown broker.*ibkr"):
            load_instruments(path)

    def test_routable_without_broker_symbol_rejected(self, tmp_path):
        path = _write(tmp_path, "AAA,AAA,,us,USD,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="AAA.*broker_symbol"):
            load_instruments(path)

    def test_bad_currency_rejected(self, tmp_path):
        path = _write(tmp_path, "AAA,AAA,AAA,us,usd,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="currency"):
            load_instruments(path)

    def test_blank_required_field_rejected(self, tmp_path):
        path = _write(tmp_path, "AAA,,AAA,us,USD,alpaca,stock,Tech")
        with pytest.raises(ValueError, match="yahoo_ticker"):
            load_instruments(path)

    def test_blank_tradable_brokers_means_not_routable(self, tmp_path):
        path = _write(tmp_path, "LOC,LOC.CL,,co,COP,,stock,Energy")
        inst = load_instruments(path)["LOC"]
        assert inst.tradable_brokers == () and inst.broker_symbol is None


# ---------------------------------------------------------------------------
# Adaptador de datos: id ↔ yahoo_symbol, precios locales + serie FX
# ---------------------------------------------------------------------------
from src.fx import MissingFxError  # noqa: E402
from src.market_data import FakeProvider, InstrumentProvider  # noqa: E402

DATES = pd.bdate_range("2024-01-01", periods=3)
START, END = "2024-01-01", "2024-01-06"


def _mixed_master(tmp_path):
    return load_instruments(_write(
        tmp_path,
        "USA1,USA1,USA1,us,USD,alpaca,stock,Tech",
        "CL-B,CL-B,CL.B,us,USD,alpaca,stock,Tech",
        "COL1,COL1.CL,,co,COP,,stock,Energy",
    ))


def _yahoo_close():
    return pd.DataFrame({
        "USA1": [10.0, 11.0, 12.0],
        "CL-B": [5.0, 5.0, 5.0],
        "COL1.CL": [8_000.0, 8_000.0, 10_000.0],
        "USDCOP=X": [4_000.0, 4_000.0, 5_000.0],
    }, index=DATES)


class TestInstrumentProvider:
    def test_requests_yahoo_symbols_and_returns_ids(self, tmp_path):
        inner = FakeProvider(_yahoo_close())
        provider = InstrumentProvider(inner, _mixed_master(tmp_path))
        bars = provider.get_bars(["USA1", "COL1"], START, END)
        assert inner.calls[0][0] == ("USA1", "COL1.CL")
        assert list(bars.close.columns) == ["USA1", "COL1"]
        assert bars.meta.requested == ("USA1", "COL1")
        # Precios en moneda LOCAL: la conversion es de to_base.
        assert bars.close["COL1"].iloc[-1] == 10_000.0

    def test_unknown_ids_pass_through_unchanged(self, tmp_path):
        close = _yahoo_close().rename(columns={"USA1": "ZZZ"})
        inner = FakeProvider(close)
        provider = InstrumentProvider(inner, _mixed_master(tmp_path))
        bars = provider.get_bars(["ZZZ"], START, END)
        assert inner.calls[0][0] == ("ZZZ",)
        assert list(bars.close.columns) == ["ZZZ"]

    def test_currencies_assume_base_only_for_unknown_ids(self, tmp_path):
        provider = InstrumentProvider(FakeProvider(_yahoo_close()),
                                      _mixed_master(tmp_path))
        assert provider.currencies(["USA1", "COL1", "ZZZ"]) == {
            "USA1": "USD", "COL1": "COP", "ZZZ": "USD"}

    def test_fx_is_base_per_unit_of_local_currency(self, tmp_path):
        inner = FakeProvider(_yahoo_close())
        provider = InstrumentProvider(inner, _mixed_master(tmp_path))
        fx = provider.get_fx(["COP"], START, END)
        assert inner.calls[-1][0] == ("USDCOP=X",)
        assert list(fx.columns) == ["COP"]
        assert fx["COP"].tolist() == pytest.approx([1 / 4000, 1 / 4000, 1 / 5000])

    def test_fx_missing_pair_raises(self, tmp_path):
        inner = FakeProvider(_yahoo_close().drop(columns="USDCOP=X"))
        provider = InstrumentProvider(inner, _mixed_master(tmp_path))
        with pytest.raises(MissingFxError, match="COP"):
            provider.get_fx(["COP"], START, END)


# ---------------------------------------------------------------------------
# Ruta de investigacion (data_client): ids del maestro y precios en base
# ---------------------------------------------------------------------------
class TestDataClientUsesMaster:
    def _fake_chain(self, monkeypatch):
        import src.data_client as dc
        close = pd.DataFrame({"AAPL": [190.0, 191.0, 192.0],
                              "ICOLCAP.CL": [12_000.0, 12_000.0, 12_000.0],
                              "USDCOP=X": [4_000.0, 4_000.0, 4_800.0]}, index=DATES)
        fake = FakeProvider(close)
        monkeypatch.setattr(dc, "build_provider", lambda *a, **k: fake)
        return dc, fake

    def test_fetch_bars_downloads_yahoo_symbol_and_returns_base(self, monkeypatch):
        from src.market_data import RESEARCH_POLICY
        dc, fake = self._fake_chain(monkeypatch)
        bars = dc.fetch_bars(["AAPL", "ICOLCAP"], START, END, policy=RESEARCH_POLICY)
        assert fake.calls[0][0] == ("AAPL", "ICOLCAP.CL")
        assert list(bars.close.columns) == ["AAPL", "ICOLCAP"]
        assert bars.close["ICOLCAP"].tolist() == pytest.approx([3.0, 3.0, 2.5])
        assert bars.close["AAPL"].iloc[-1] == 192.0

    def test_fetch_prices_compat_wrapper_too(self, monkeypatch):
        dc, fake = self._fake_chain(monkeypatch)
        prices = dc.fetch_prices(["ICOLCAP"], START, END)
        assert fake.calls[0][0] == ("ICOLCAP.CL",)
        assert prices["ICOLCAP"].iloc[-1] == pytest.approx(2.5)
