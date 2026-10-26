"""Modelo único de costos (HFT-07): spread, impacto, fees y ADV causal."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.hft.config import load_hft_config
from src.hft.costs import (
    CostModel,
    Quote,
    average_daily_value,
    daily_volatility,
    impact_return,
    implementation_shortfall,
    paper_valuation_fees,
)
from src.order_execution import compute_adv


def _model(**overrides) -> CostModel:
    raw = dict(
        spread_floor_bps=1.0,
        tick=0.01,
        impact_y=1.0,
        adv_window=20,
        commission_per_share=0.0,
        sec_section31_rate=0.0000206,
        sec_section31_verified_on="2026-04-04",
        finra_taf_per_share=0.000195,
        finra_taf_cap=9.79,
        finra_taf_verified_on="2026-10-04",
        crypto_maker_bps=15.0,
        crypto_taker_bps=25.0,
        crypto_fees_verified_on="2026-10-04",
    )
    raw.update(overrides)
    return CostModel(**raw)


def test_la_config_versionada_arma_el_unico_modelo():
    cfg = load_hft_config()
    model = CostModel.from_config(cfg)
    assert model.commission_per_share == 0
    assert model.sec_section31_rate == pytest.approx(cfg.costs.sec_section31_rate)
    assert model.finra_taf_cap == pytest.approx(9.79)
    assert model.impact_y == 1
    sell = model.fees(side="sell", qty=100, price=100, liquidity="taker")
    assert paper_valuation_fees(
        model, side="sell", qty=100, price=100, liquidity="taker",
    ) == pytest.approx(sell)
    assert model.paper_fees(side="sell", qty=100, price=100, liquidity="taker") == pytest.approx(sell)


def test_fees_de_acciones_solo_en_la_venta_y_con_tope():
    model = _model()
    assert model.fees(side="buy", qty=100, price=100, liquidity="taker") == 0
    # 100 * 100 * 0.0000206 + min(100 * 0.000195, 9.79)
    assert model.fees(side="sell", qty=100, price=100, liquidity="taker") == pytest.approx(0.206 + 0.0195)
    capped = model.fees(side="sell", qty=100_000, price=50, liquidity="maker")
    sec = 100_000 * 50 * 0.0000206
    assert capped == pytest.approx(sec + 9.79)


def test_finra_no_cobra_taf_si_el_precio_no_llega_a_la_tasa():
    model = _model()
    price = 0.0001
    fees = model.fees(side="sell", qty=10, price=price, liquidity="taker")
    assert fees == pytest.approx(10 * price * 0.0000206)


def test_cripto_es_maker_o_taker_en_bps_sin_sec():
    model = _model()
    assert model.fees(
        side="buy", qty=2, price=100, liquidity="taker", asset_class="crypto",
    ) == pytest.approx(200 * 25 / 10_000)
    assert model.fees(
        side="sell", qty=2, price=100, liquidity="maker", asset_class="crypto",
    ) == pytest.approx(200 * 15 / 10_000)


def test_medio_spread_usa_el_quote_y_si_no_el_maximo():
    model = _model(tick=0.05, spread_floor_bps=1.0)
    quote = Quote(bid=100, ask=100.04)
    assert model.half_spread(100, quote=quote) == pytest.approx(0.02)
    # tick 0.05, piso 1 bp de 100 = 0.01, Roll/2 = 0.10
    assert model.half_spread(100, roll=0.20, corwin=-0.01) == pytest.approx(0.10)
    assert model.half_spread(100, roll=float("nan"), corwin=0.0) == pytest.approx(0.05)
    off = _model(tick=0.05, spread_floor_bps=0, use_spread_estimators=False)
    assert off.half_spread(100, roll=1.0) == pytest.approx(0.05)


def test_impacto_de_raiz_cuadrada_con_numeros_conocidos():
    # q = 100 * 50 = 5000; ADV = 1e6; sqrt(q/ADV) = sqrt(0.005)
    fraction = impact_return(0.02, 100, 50, 1_000_000, y=1)
    assert fraction == pytest.approx(0.02 * math.sqrt(0.005))
    model = _model(impact_y=1)
    assert model.impact_per_share(50, 100, 0.02, 1_000_000) == pytest.approx(50 * fraction)
    assert impact_return(0.02, 100, 50, 1_000_000, y=0) == 0


def test_adv_excluye_el_dia_y_coincide_con_compute_adv():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = pd.DataFrame({"SPY": [10.0, 20.0, 30.0, 40.0]}, index=index)
    volumes = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 100.0]}, index=index)
    t = pd.Timestamp("2024-01-05")
    got = average_daily_value(prices, volumes, t, window=2)
    ref = compute_adv(volumes, prices, t, window=2)
    pd.testing.assert_series_equal(got, ref)
    # Días 3 y 4: (20*1 + 30*1) / 2 = 25. El día 5 no entra, ni el día 2.
    assert got["SPY"] == pytest.approx(25)
    assert pd.Timestamp("2024-01-05") not in volumes.loc[volumes.index < t].index


def test_la_volatilidad_diaria_no_usa_el_dia_t():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    # El último precio es un salto que no puede entrar: t es ese día.
    prices = pd.Series([100.0, 110.0, 121.0, 133.1, 1.0], index=index)
    asof = pd.Timestamp("2024-01-08")
    sigma = daily_volatility(prices, asof, window=10)
    history = prices.loc[prices.index < asof]
    rets = (history / history.shift(1)).dropna().map(math.log)
    assert sigma == pytest.approx(float(rets.std(ddof=1)))
    assert math.isfinite(sigma)
    assert math.isnan(daily_volatility(prices.iloc[:2], asof, window=10))


def test_el_shortfall_de_una_compra_cara_es_positivo():
    cost = implementation_shortfall(
        side="buy", qty=10, benchmark=100, fill_price=100.2, fees=0.05,
    )
    assert cost == pytest.approx(2.05)
    same = implementation_shortfall(
        side="sell", qty=10, benchmark=100, fill_price=100, fees=0,
    )
    assert same == 0


def test_from_config_deja_el_cruce_como_modo_por_defecto():
    from src.hft.execution.sim import SimExecution
    cfg = load_hft_config()
    sim = SimExecution.from_config(cfg, adv={"SPY": 1e9}, sigma={"SPY": 0.01}, seed=1)
    assert isinstance(sim.costs, CostModel)
    assert sim.fill_mode == "cross"
    assert sim.label == "pessimistic-cross"
    assert sim.optimistic is False
    assert sim.latency_bars == 0
    assert sim.partial_prob == pytest.approx(cfg.sim.partial_prob)
    assert sim.participation_cap == pytest.approx(cfg.sim.participation_cap)


def test_una_clave_de_costos_desconocida_es_error():
    raw = load_hft_config().to_dict()
    costs = dict(raw["costs"])
    costs["cat_fee"] = 0.01
    raw["costs"] = costs
    with pytest.raises(ValueError, match="desconocidas"):
        from src.hft.config import hft_config_from_dict
        hft_config_from_dict(raw)
