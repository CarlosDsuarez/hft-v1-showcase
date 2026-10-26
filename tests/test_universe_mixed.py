"""
test_universe_mixed.py — Valida el universo mixto USA+COL (v2, hibrido ADR-first).

Offline: solo lee CSVs locales y datos sinteticos (sin red).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_client import load_universe
from src.universe_filter import filter_universe_at_date

REPO_ROOT = Path(__file__).resolve().parents[1]
CO_PATH = REPO_ROOT / "universe_co.csv"
MIXED_PATH = REPO_ROOT / "universe_mixed.csv"
PROBE_CSV = REPO_ROOT / "research" / "data_quality_co.csv"

REQUIRED = {"ticker", "asset_type", "sector", "name"}
EXTENDED = {"yahoo_ticker", "market", "currency", "adr_proxy", "inclusion_reason"}


class TestUniverseCO:
    def test_co_has_20_unique_tickers(self):
        df = pd.read_csv(CO_PATH)
        assert len(df) == 20
        assert df["ticker"].nunique() == 20

    def test_co_columns(self):
        df = pd.read_csv(CO_PATH)
        assert REQUIRED.issubset(df.columns)
        assert EXTENDED.issubset(df.columns)

    def test_co_markets_and_currencies(self):
        df = pd.read_csv(CO_PATH)
        assert (df["market"] == "co").all()
        assert set(df["currency"].unique()) <= {"USD", "COP"}
        adrs = df[df["currency"] == "USD"]["ticker"].tolist()
        assert set(adrs) == {"EC", "CIB", "AVAL", "TGLS"}
        assert (df["yahoo_ticker"].astype(str) != "").all()

    def test_co_sector_coverage(self):
        df = pd.read_csv(CO_PATH)
        assert df["sector"].nunique() >= 5

    def test_co_mix_adr_etf_local(self):
        df = pd.read_csv(CO_PATH)
        assert (df["asset_type"] == "etf").sum() == 2
        assert (df["asset_type"] == "stock").sum() == 18


class TestUniverseMixed:
    def test_mixed_is_120(self):
        df = pd.read_csv(MIXED_PATH)
        assert len(df) == 120
        assert df["ticker"].nunique() == 120

    def test_mixed_split(self):
        df = pd.read_csv(MIXED_PATH)
        assert (df["market"] == "us").sum() == 100
        assert (df["market"] == "co").sum() == 20

    def test_mixed_co_matches_co_file(self):
        mixed = pd.read_csv(MIXED_PATH)
        co = pd.read_csv(CO_PATH)
        assert set(mixed[mixed["market"] == "co"]["ticker"]) == set(co["ticker"])

    def test_load_universe_market_filter(self):
        co = load_universe(str(CO_PATH), market="co")
        assert len(co) == 20
        us = load_universe(str(MIXED_PATH), market="us")
        assert len(us) == 100
        full = load_universe(str(MIXED_PATH))
        assert len(full) == 120

    def test_load_universe_market_requires_column(self):
        with pytest.raises(ValueError, match="no 'market' column"):
            load_universe(str(REPO_ROOT / "universe.csv"), market="co")


def _synthetic_two_markets(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=60)
    us_tickers = [f"US{i}" for i in range(6)]
    co_tickers = ["CO_LIQ", "CO_ILI"]
    prices = pd.DataFrame(100.0, index=dates, columns=us_tickers + co_tickers)
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=us_tickers + co_tickers)
    # Random walk liquido para todos menos CO_ILI
    for c in us_tickers + ["CO_LIQ"]:
        rets = rng.normal(0.001, 0.01, len(dates))
        prices[c] = 100 * np.exp(np.cumsum(rets))
    # CO_ILI: retornos grandes + volumen enano -> Amihud extremo
    rets = rng.normal(0.0, 0.08, len(dates))
    prices["CO_ILI"] = 50 * np.exp(np.cumsum(rets))
    volumes["CO_ILI"] = 100.0
    markets = pd.Series(
        {c: ("us" if c.startswith("US") else "co") for c in prices.columns}
    )
    return dates, prices, volumes, markets


class TestByMarketFilter:
    def test_by_market_excludes_illiquid_co_only(self):
        dates, prices, volumes, markets = _synthetic_two_markets()
        t = dates[-1]
        active, log = filter_universe_at_date(
            t, prices, volumes, amihud_percentile_threshold=90.0,
            min_volume=10, window=30, by_market=True, markets=markets,
        )
        assert "CO_ILI" not in active
        assert "CO_LIQ" in active
        # Toda exclusion Amihud lleva tag de su mercado (no umbral global)
        amihud_log = log[log["reason"].str.startswith("amihud")]
        assert len(amihud_log) > 0
        assert amihud_log["reason"].str.contains(r"\[(?:us|co)\]").all()
        assert log[log["ticker"] == "CO_ILI"]["reason"].str.contains(r"\[co\]").any()

    def test_by_market_without_mapping_falls_back_to_global(self):
        dates, prices, volumes, _ = _synthetic_two_markets()
        t = dates[-1]
        active_fb, _ = filter_universe_at_date(
            t, prices, volumes, amihud_percentile_threshold=50.0,
            min_volume=10, window=30, by_market=True, markets=None,
        )
        active_gl, _ = filter_universe_at_date(
            t, prices, volumes, amihud_percentile_threshold=50.0,
            min_volume=10, window=30,
        )
        assert active_fb == active_gl


