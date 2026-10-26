"""Dueño de cada posición del motor intradía (HFT-PAPER-08)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.hft.attribution import Attribution
from src.hft.book import HftBook
from src.hft.types import Fill, order_intent

UTC = timezone.utc
T0 = datetime(2025, 12, 1, 15, 17, tzinfo=UTC)


def _trade(owner: str, symbol: str, side: str, qty: float, minute: int, price: float = 100.0):
    ts = T0 + timedelta(minutes=minute)
    intent = order_intent(owner, symbol, side, qty, ts, reason="test")
    return intent, Fill(intent.intent_id, ts, qty, price, 0.0, "taker")


def _apply(attribution: Attribution, *trades) -> None:
    intents = {intent.intent_id: intent for intent, _fill in trades}
    attribution.apply([fill for _intent, fill in trades], intents)


def test_el_fill_de_una_estrategia_cuenta_para_ella():
    attribution = Attribution(["orb", "vwap_reversion"])
    _apply(
        attribution,
        _trade("vwap_reversion", "IWM", "sell", 40.0, 0),
        _trade("orb", "KO", "buy", 112.0, 1),
    )
    assert attribution.owned("vwap_reversion") == (("IWM", -40.0),)
    assert attribution.owned("orb") == (("KO", 112.0),)
    assert attribution.owners("IWM") == {"vwap_reversion": -40.0}
    assert attribution.position("orb", "IWM") == 0.0


def test_cerrar_lo_propio_borra_la_fila():
    attribution = Attribution(["vwap_reversion"])
    _apply(
        attribution,
        _trade("vwap_reversion", "IWM", "sell", 40.0, 0),
        _trade("vwap_reversion", "IWM", "buy", 40.0, 1),
    )
    assert attribution.owned("vwap_reversion") == ()
    assert attribution.owners("IWM") == {}


def test_la_casa_reduce_a_las_estrategias_sin_dar_vuelta_y_se_queda_el_resto():
    # El flatten de sesión compra 50 IWM: cubre los 30 cortos de a y los 10
    # de b (en orden de id) y los 10 que sobran quedan a nombre de flatten.
    attribution = Attribution(["a", "b"])
    _apply(
        attribution,
        _trade("a", "IWM", "sell", 30.0, 0),
        _trade("b", "IWM", "sell", 10.0, 1),
        _trade("flatten", "IWM", "buy", 50.0, 2),
    )
    assert attribution.owners("IWM") == {"flatten": pytest.approx(10.0)}
    assert attribution.owned("a") == ()
    assert attribution.owned("b") == ()


def test_la_casa_no_toca_a_quien_tiene_su_mismo_signo():
    attribution = Attribution(["a", "b"])
    _apply(
        attribution,
        _trade("a", "SPY", "buy", 5.0, 0),
        _trade("b", "SPY", "sell", 3.0, 1),
        _trade("flatten", "SPY", "sell", 5.0, 2),
    )
    # La venta reduce el largo de a; el corto de b sigue igual.
    assert attribution.owners("SPY") == {"b": -3.0}


def test_la_casa_reduce_primero_estrategias_y_despues_otra_casa():
    # Un relanzamiento sin cuadre deja SPY a nombre de reconcile. El flatten
    # que lo cierra no deja dos filas de la casa con signos opuestos.
    attribution = Attribution(["a"])
    _apply(
        attribution,
        _trade("reconcile", "SPY", "buy", 7.0, 0),
        _trade("flatten", "SPY", "sell", 7.0, 1),
    )
    assert attribution.owners("SPY") == {}


def test_un_dueno_que_no_corre_es_de_la_casa():
    # hourly_momentum sin --with: su fill de hoy cuenta con la regla de la casa.
    attribution = Attribution(["vwap_reversion"])
    _apply(
        attribution,
        _trade("vwap_reversion", "XLE", "sell", 10.0, 0),
        _trade("hourly_momentum", "XLE", "buy", 15.0, 1),
    )
    assert attribution.owners("XLE") == {"hourly_momentum": pytest.approx(5.0)}


def test_el_lote_es_todo_o_nada():
    attribution = Attribution(["a"])
    good_intent, good_fill = _trade("a", "SPY", "buy", 5.0, 0)
    _missing_intent, missing_fill = _trade("a", "QQQ", "buy", 5.0, 1)
    with pytest.raises(ValueError, match="intención"):
        attribution.apply([good_fill, missing_fill], {good_intent.intent_id: good_intent})
    assert attribution.owners("SPY") == {}


def test_la_suma_de_los_duenos_es_el_libro():
    # Misma secuencia de fills al libro y a la atribución, con un corto que
    # se da vuelta, la casa y un dueño que no corre.
    attribution = Attribution(["a", "b"])
    book = HftBook(1_000_000)
    trades = [
        _trade("a", "SPY", "buy", 10.0, 0, 100.0),
        _trade("b", "QQQ", "sell", 4.0, 1, 50.0),
        _trade("a", "SPY", "sell", 15.0, 2, 101.0),
        _trade("flatten", "SPY", "buy", 2.0, 3, 102.0),
        _trade("hourly_momentum", "QQQ", "buy", 6.0, 4, 51.0),
        _trade("b", "QQQ", "buy", 1.0, 5, 51.0),
        _trade("flatten", "QQQ", "sell", 3.0, 6, 52.0),
    ]
    for trade in trades:
        intents = {trade[0].intent_id: trade[0]}
        book.apply_fills([trade[1]], intents)
        attribution.apply([trade[1]], intents)
        for symbol in ("SPY", "QQQ"):
            total = sum(attribution.owners(symbol).values())
            assert total == pytest.approx(book.position(symbol), abs=1e-9)


def test_lo_ignorable_no_deja_fila():
    attribution = Attribution(["a"])
    _apply(
        attribution,
        _trade("a", "SPY", "buy", 1.0, 0),
        _trade("a", "SPY", "sell", 1.0 - 1e-12, 1),
    )
    assert attribution.owned("a") == ()
