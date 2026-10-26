"""Sizing intradía: riesgo fijo, vol targeting y Kelly acotado (HFT-08)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.hft.config import SizingConfig
from src.hft.sizing import (
    apply_sizing_caps,
    equity_at_risk,
    fixed_risk_qty,
    kelly_fraction,
    kelly_notional,
    size_fixed_risk,
    size_kelly,
    size_vol_target,
    vol_target_notional,
    IntradaySizer,
)
from src.hft.types import Bracket, order_intent

T0 = datetime(2024, 7, 2, 14, 0, tzinfo=timezone.utc)


def _config(**overrides) -> SizingConfig:
    base = dict(
        risk_capital_usd=50_000.0,
        risk_frac=0.01,
        sigma_target=0.10,
        kelly_c=0.25,
        min_notional=100.0,
        max_leverage=1.0,
    )
    base.update(overrides)
    return SizingConfig(**base)


def test_equity_de_riesgo_es_el_minimo_de_cuenta_y_capital():
    assert equity_at_risk(100_000, 25_000) == 25_000
    assert equity_at_risk(10_000, 25_000) == 10_000


def test_riesgo_fijo_con_numeros_conocidos():
    # 1% de 50_000 / distancia de 1 USD = 500 acciones.
    assert fixed_risk_qty(50_000, 0.01, 100.0, 99.0) == pytest.approx(500.0)
    assert fixed_risk_qty(50_000, 0.01, 100.0, 101.0) == pytest.approx(500.0)
    with pytest.raises(ValueError, match="coincidir"):
        fixed_risk_qty(50_000, 0.01, 100.0, 100.0)


def test_vol_targeting_con_numeros_conocidos():
    # 0.10 * 50_000 / 0.20 = 25_000 USD.
    assert vol_target_notional(50_000, 0.10, 0.20) == pytest.approx(25_000.0)


def test_kelly_fraccional_acotado_y_fuera_de_muestra():
    # f = 0.25 * 0.02 / 0.10² = 0.5; nocional = 0.5 * 50_000 = 25_000.
    assert kelly_fraction(0.02, 0.10, 0.25, out_of_sample=True) == pytest.approx(0.5)
    assert kelly_notional(50_000, 0.02, 0.10, 0.25, out_of_sample=True) == pytest.approx(25_000)
    assert kelly_notional(50_000, -0.02, 0.10, 0.25, out_of_sample=True) == 0.0
    with pytest.raises(ValueError, match="0.25"):
        kelly_fraction(0.02, 0.10, 0.5, out_of_sample=True)
    with pytest.raises(ValueError, match="fuera de muestra"):
        kelly_fraction(0.02, 0.10, 0.25, out_of_sample=False)


def test_topes_de_apalancamiento_acciones_enteras_y_minimo():
    capped = apply_sizing_caps(
        600, 100.0,
        equity_riesgo=50_000, min_notional=100, max_leverage=1, whole_shares=False,
    )
    assert capped == pytest.approx(500.0)
    whole = apply_sizing_caps(
        10.9, 100.0,
        equity_riesgo=50_000, min_notional=100, max_leverage=4, whole_shares=True,
    )
    assert whole == 10
    dust = apply_sizing_caps(
        1, 100.0,
        equity_riesgo=50_000, min_notional=1_000, max_leverage=1, whole_shares=False,
    )
    assert dust == 0.0


def test_size_fijo_vol_y_kelly_ya_con_topes():
    cfg = _config()
    assert size_fixed_risk(cfg, 100_000, 100.0, 99.0) == pytest.approx(500.0)
    assert size_vol_target(cfg, 100_000, 100.0, 0.20) == pytest.approx(250.0)
    assert size_kelly(
        cfg, 100_000, 100.0, 0.02, 0.10, out_of_sample=True,
    ) == pytest.approx(250.0)
    short = size_fixed_risk(cfg, 100_000, 100.0, 99.0, whole_shares=True)
    assert short == 500
    assert size_fixed_risk(
        _config(min_notional=1_000_000), 100_000, 100.0, 99.0,
    ) == 0.0


def test_la_cuenta_chica_pisa_al_capital_configurado():
    cfg = _config(risk_capital_usd=50_000, risk_frac=0.01, max_leverage=4)
    # equity_riesgo = 10_000; qty = 0.01 * 10_000 / 1 = 100.
    assert size_fixed_risk(cfg, 10_000, 50.0, 49.0) == pytest.approx(100.0)


@dataclass
class _Estado:
    equity: float
    nav: float
    positions: tuple = ()
    marks: tuple = ()

    def position(self, symbol: str) -> float:
        for name, qty in self.positions:
            if name == symbol:
                return qty
        return 0.0


def test_el_sizer_del_motor_reescribe_con_el_stop_del_bracket():
    cfg = _config(max_leverage=2)
    sizer = IntradaySizer(cfg)
    intent = order_intent(
        "orb", "SPY", "buy", 1.0, T0,
        limit_price=100.0, bracket=Bracket(stop=99.0),
    )
    sized = sizer.size([intent], _Estado(equity=100_000, nav=100_000), T0)
    assert [item.intent_id for item in sized] == [intent.intent_id]
    assert sized[0].qty == pytest.approx(500.0)
    assert sized[0].bracket.stop == 99.0


def test_el_sizer_redondea_el_corto_y_no_emite_cero():
    cfg = _config(max_leverage=4, min_notional=100)
    sizer = IntradaySizer(cfg)
    short = order_intent("s", "SPY", "sell", 10.9, T0, limit_price=100.0)
    sized = sizer.size([short], _Estado(100_000, 100_000), T0)
    assert sized[0].qty == 10
    dust = order_intent("s", "SPY", "buy", 0.5, T0, limit_price=100.0)
    with pytest.raises(ValueError, match="quedó en 0"):
        sizer.size([dust], _Estado(100_000, 100_000), T0)


def _orb_config() -> SizingConfig:
    # Los valores de config/hft.toml: 1% de 25_000 y apalancamiento 1.
    return _config(risk_capital_usd=25_000.0, max_leverage=1.0)


def test_el_tope_por_simbolo_del_riesgo_recorta_la_orden():
    # Caso del 2026-10-06: stop a 0.08 pedía 3125 TLT, el apalancamiento
    # lo dejaba en 323 ($24,942) y el riesgo ($10k) lo rechazaba.
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    intent = order_intent(
        "orb", "TLT", "buy", 1.0, T0,
        limit_price=77.22, bracket=Bracket(stop=77.14),
    )
    sized = sizer.size([intent], _Estado(100_000, 100_000), T0)
    assert sized[0].qty == 129
    assert sized[0].qty * 77.22 <= 10_000.0


def test_sin_tope_por_simbolo_el_sizer_no_cambia():
    sizer = IntradaySizer(_orb_config())
    intent = order_intent(
        "orb", "TLT", "buy", 1.0, T0,
        limit_price=77.22, bracket=Bracket(stop=77.14),
    )
    sized = sizer.size([intent], _Estado(100_000, 100_000), T0)
    assert sized[0].qty == 323


def test_el_tope_del_corto_usa_la_marca_si_es_mayor_que_el_limite():
    # El riesgo valora con la marca: con 780.20 > 780.13, 12 y no 12.8.
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    short = order_intent("vwap", "SPY", "sell", 32.0, T0, limit_price=780.13)
    state = _Estado(100_000, 100_000, marks=(("SPY", 780.20),))
    sized = sizer.size([short], state, T0)
    assert sized[0].qty == 12
    assert sized[0].qty * 780.20 <= 10_000.0


def test_el_tope_descuenta_la_posicion_que_ya_hay():
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    intent = order_intent(
        "orb", "TLT", "buy", 1.0, T0,
        limit_price=77.22, bracket=Bracket(stop=77.14),
    )
    state = _Estado(100_000, 100_000, positions=(("TLT", 100.0),))
    sized = sizer.size([intent], state, T0)
    assert sized[0].qty == 29
    assert (100 + sized[0].qty) * 77.22 <= 10_000.0


def test_con_la_posicion_llena_la_intencion_pasa_al_riesgo_sin_error():
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    intent = order_intent(
        "orb", "TLT", "buy", 1.0, T0,
        limit_price=77.22, bracket=Bracket(stop=77.14),
    )
    state = _Estado(100_000, 100_000, positions=(("TLT", 130.0),))
    sized = sizer.size([intent], state, T0)
    assert sized[0].qty == 323  # sin recorte: el riesgo la rechaza con motivo


def test_el_tope_cuenta_las_ordenes_previas_del_mismo_lote():
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    first = order_intent("a", "SPY", "buy", 10.0, T0, limit_price=100.0)
    second = order_intent("b", "SPY", "buy", 95.0, T0, limit_price=100.0)
    sized = sizer.size([first, second], _Estado(100_000, 100_000), T0)
    assert sized[0].qty == pytest.approx(10.0)
    assert sized[1].qty == pytest.approx(90.0)


def test_el_tope_no_recorta_la_salida():
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    exit_ = order_intent("orb", "TLT", "sell", 120.0, T0, limit_price=77.0)
    state = _Estado(100_000, 100_000, positions=(("TLT", 120.0),))
    sized = sizer.size([exit_], state, T0)
    assert sized[0].qty == pytest.approx(120.0)


def test_sin_cupo_en_el_lote_la_entrada_se_omite():
    # Caso del 2026-10-07: el ORB abrió cinco ETF a las 09:34 y los
    # primeros se llevaron los 25_000 del apalancamiento. IWM quedaba en
    # 0 y el sizer lanzaba: la sesión de signals moría con exit 5.
    sizer = IntradaySizer(_orb_config())
    first = order_intent(
        "orb", "TLT", "buy", 1.0, T0,
        limit_price=77.22, bracket=Bracket(stop=77.14),
    )
    second = order_intent(
        "orb", "IWM", "buy", 1.0, T0,
        limit_price=245.0, bracket=Bracket(stop=244.65),
    )
    sized = sizer.size([first, second], _Estado(100_000, 100_000), T0)
    assert [item.intent_id for item in sized] == [first.intent_id]
    assert sized[0].qty == 323


def test_la_salida_no_gasta_cupo_ni_se_recorta():
    # Libro de 26_000 contra un cupo de 25_000: el flatten tiene que
    # cerrar todo, no dejar un resto abierto.
    sizer = IntradaySizer(_orb_config(), max_position_notional=15_000.0)
    spy = order_intent("orb", "SPY", "sell", 20.0, T0, limit_price=700.0)
    qqq = order_intent("orb", "QQQ", "sell", 20.0, T0, limit_price=600.0)
    state = _Estado(100_000, 100_000, positions=(("SPY", 20.0), ("QQQ", 20.0)))
    sized = sizer.size([spy, qqq], state, T0)
    assert [item.qty for item in sized] == [20.0, 20.0]


def test_la_salida_bajo_el_nocional_minimo_igual_sale():
    sizer = IntradaySizer(_config(min_notional=100))
    dust = order_intent("orb", "SPY", "sell", 0.5, T0, limit_price=100.0)
    state = _Estado(100_000, 100_000, positions=(("SPY", 0.5),))
    sized = sizer.size([dust], state, T0)
    assert sized[0].qty == pytest.approx(0.5)


def test_el_sizer_entrega_la_cantidad_canonica():
    # 2026-10-12: 10_000 / 65.76 = 152.06812652068126 en el intent y
    # Alpaca recibió 152.068126521. Se trunca a 9 decimales: el tope
    # por símbolo nunca se pasa por redondear.
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    intent = order_intent("vwap_reversion", "XLE", "buy", 1_000.0, T0, limit_price=65.76)
    sized = sizer.size([intent], _Estado(100_000, 100_000), T0)
    assert sized[0].qty == 152.06812652
    assert sized[0].qty * 65.76 <= 10_000.0


def test_la_salida_tambien_sale_canonica():
    # Una posición que viene de sumar fills trae ruido binario.
    held = 152.06812652099998
    sizer = IntradaySizer(_orb_config(), max_position_notional=10_000.0)
    exit_ = order_intent("vwap_reversion", "XLE", "sell", held, T0, limit_price=65.80)
    state = _Estado(100_000, 100_000, positions=(("XLE", held),))
    sized = sizer.size([exit_], state, T0)
    assert sized[0].qty == 152.068126521
