"""
test_fx.py — Conversion a moneda base (src/fx.py), sin red.

Convencion: fx[ccy] = unidades de moneda base por 1 unidad de `ccy`
(USD base: fx["COP"] = 1 / USDCOP).
"""

import numpy as np
import pandas as pd
import pytest

from src.fx import MissingFxError, fx_pair_symbol, to_base

DATES = pd.bdate_range("2024-01-01", periods=3)
CCY = {"AAPL": "USD", "ICOLCAP": "COP"}


def _prices():
    return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0],
                         "ICOLCAP": [12_000.0, 12_000.0, 13_000.0]}, index=DATES)


def test_pair_symbol_quotes_local_per_base():
    assert fx_pair_symbol("COP", "USD") == "USDCOP=X"


def test_converts_local_columns_and_keeps_base_columns():
    fx = pd.DataFrame({"COP": [1 / 4000, 1 / 4000, 1 / 3250]}, index=DATES)
    out = to_base(_prices(), fx, CCY, base="USD")
    pd.testing.assert_series_equal(out["AAPL"], _prices()["AAPL"])
    np.testing.assert_allclose(out["ICOLCAP"], [3.0, 3.0, 4.0])


def test_uses_last_known_rate_without_look_ahead():
    # Sin tasa el dia 2: se usa la del dia 1 (nunca la del dia 3).
    fx = pd.DataFrame({"COP": [1 / 4000, 1 / 2000]}, index=DATES[[0, 2]])
    out = to_base(_prices(), fx, CCY, base="USD")
    np.testing.assert_allclose(out["ICOLCAP"], [3.0, 3.0, 6.5])


def test_dates_before_first_rate_are_nan():
    fx = pd.DataFrame({"COP": [1 / 4000]}, index=DATES[[1]])
    out = to_base(_prices(), fx, CCY, base="USD")
    assert np.isnan(out["ICOLCAP"].iloc[0])
    assert out["ICOLCAP"].iloc[1] == pytest.approx(3.0)


def test_invalid_rates_are_missing_not_zero_or_inf():
    fx = pd.DataFrame({"COP": [1 / 4000, 0.0, -1.0]}, index=DATES)
    out = to_base(_prices(), fx, CCY, base="USD")
    # 0 / negativo no son tasas: se arrastra la ultima valida.
    np.testing.assert_allclose(out["ICOLCAP"], [3.0, 3.0, 3.25])


def test_missing_currency_series_raises():
    with pytest.raises(MissingFxError, match="COP"):
        to_base(_prices(), pd.DataFrame(index=DATES), CCY, base="USD")


def test_column_without_currency_raises():
    with pytest.raises(ValueError, match="ICOLCAP"):
        to_base(_prices(), pd.DataFrame(index=DATES), {"AAPL": "USD"}, base="USD")


def test_base_only_needs_no_fx():
    prices = _prices()[["AAPL"]]
    out = to_base(prices, pd.DataFrame(), {"AAPL": "USD"}, base="USD")
    pd.testing.assert_frame_equal(out, prices)
