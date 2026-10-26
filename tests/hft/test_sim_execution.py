"""Simulador pesimista: cada regla, y que el pesimismo no se invierta."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, bar_close_time, session_at
from src.hft.costs import CostModel, Quote, implementation_shortfall
from src.hft.data.bars import bars_from_records
from src.hft.data.synthetic import synthetic_bars
from src.hft.engine import Engine
from src.hft.execution.sim import SimExecution
from src.hft.features import log_return
from src.hft.types import Bracket, TheoryCard, TheoryParam, order_intent
from src.settings import SEED

UTC = timezone.utc
DAY = date(2024, 1, 3)


def _model(**overrides) -> CostModel:
    raw = dict(
        spread_floor_bps=0.0,
        tick=0.01,
        impact_y=0.0,
        adv_window=20,
        commission_per_share=0.0,
        sec_section31_rate=0.0,
        sec_section31_verified_on="2026-04-04",
        finra_taf_per_share=0.0,
        finra_taf_cap=0.0,
        finra_taf_verified_on="2026-10-04",
        crypto_maker_bps=0.0,
        crypto_taker_bps=0.0,
        crypto_fees_verified_on="2026-10-04",
        use_spread_estimators=False,
    )
    raw.update(overrides)
    return CostModel(**raw)


def _sim(costs=None, **kwargs) -> SimExecution:
    costs = _model() if costs is None else costs
    kwargs.setdefault("adv", {"SPY": 1e12})
    kwargs.setdefault("sigma", {"SPY": 0.0})
    kwargs.setdefault("partial_prob", 0.0)
    kwargs.setdefault("seed", SEED)
    return SimExecution(costs, **kwargs)


def _bars(rows):
    frame = pd.DataFrame(rows)
    symbols = list(dict.fromkeys(frame["symbol"]))
    return bars_from_records(frame, symbols, source="test", feed="iex", timeframe="1Min")


def _row(minute, open_, close, *, low=None, high=None, volume=10_000.0, symbol="SPY"):
    start = session_at(DAY, time(9, 30), time(16, 0)).open_utc
    ts = start + timedelta(minutes=minute)
    if low is None:
        low = min(open_, close) - 0.05
    if high is None:
        high = max(open_, close) + 0.05
    return {
        "symbol": symbol,
        "ts": ts,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
        "vw": (high + low + close) / 3,
        "n": 10,
    }


def _at(minute) -> datetime:
    start = session_at(DAY, time(9, 30), time(16, 0)).open_utc
    return start + timedelta(minutes=minute)


def _intent(minute, side="buy", qty=10, **kwargs):
    return order_intent("s", "SPY", side, qty, _at(minute), reason="test", **kwargs)


def _now(bars):
    ts = bars.close.index[-1].to_pydatetime()
    return bar_close_time(ts, "1Min") + timedelta(seconds=5)


def test_market_llena_en_la_apertura_siguiente_con_el_spread():
    bars = _bars([
        _row(0, 100, 100.5),
        _row(1, 102, 102.4),
        _row(2, 103, 103.2),
    ])
    sim = _sim()
    buy = sim.submit([_intent(0, "buy", qty=4)], now=_now(bars), bars=bars)
    assert len(buy) == 1
    assert buy[0].ts == _at(1)
    assert buy[0].price == pytest.approx(102.01)
    assert buy[0].liquidity == "taker"
    assert buy[0].fees == 0
    assert buy[0].qty == 4
    sell_sim = _sim()
    sell = sell_sim.submit([_intent(0, "sell", qty=4, seq=1)], now=_now(bars), bars=bars)
    assert sell[0].price == pytest.approx(101.99)
    assert sell[0].liquidity == "taker"


def test_latencia_extra_corre_el_fill_una_barra_mas():
    bars = _bars([_row(i, 100 + i, 100.2 + i) for i in range(4)])
    sim = _sim(latency_bars=1)
    fills = sim.submit([_intent(0, "buy", qty=1)], now=_now(bars), bars=bars)
    assert fills[0].ts == _at(2)
    assert fills[0].price == pytest.approx(102.01)


def test_marketable_que_alcanza_y_el_que_se_queda_en_limite():
    bars = _bars([
        _row(0, 100, 100.2, low=99.4, high=100.4),
        _row(1, 100, 100.2, low=100.2, high=100.6),
        _row(2, 100, 99.0, low=99.5, high=100.2),
    ])
    reached = _sim()
    fit = reached.submit(
        [_intent(0, "buy", qty=1, order_type="marketable_limit", limit_price=103)],
        now=_now(bars), bars=bars,
    )
    assert fit[0].price == pytest.approx(min(103, 100.01))
    assert fit[0].liquidity == "taker"
    assert fit[0].ts == _at(1)

    rested = _sim()
    intent = _intent(0, "buy", qty=1, order_type="marketable_limit", limit_price=100, seq=1)
    first = rested.submit([intent], now=_now(_head(bars, 2)), bars=_head(bars, 2))
    assert first == ()
    later = rested.submit([], now=_now(bars), bars=bars)
    assert later[0].price == pytest.approx(100)
    assert later[0].liquidity == "maker"
    assert later[0].ts == _at(2)


def test_limite_pasivo_solo_si_cruza_por_un_tick_y_respeta_el_volumen():
    bars = _bars([
        _row(0, 101, 101),
        _row(1, 100.5, 100.4, low=100.0, high=100.6, volume=1_000),
        _row(2, 100.2, 100.0, low=99.90, high=100.3, volume=1_000),
    ])
    sim = _sim(participation_cap=0.10)
    none = sim.submit(
        [_intent(0, "buy", qty=500, order_type="limit", limit_price=100)],
        now=_now(bars), bars=bars,
    )
    # En la barra 1 el low toca 100 pero no baja de 100 - 0.01.
    assert [fill.ts for fill in none] == [_at(2)]
    assert none[0].price == pytest.approx(100)
    assert none[0].qty == pytest.approx(100)
    assert none[0].liquidity == "maker"


def test_el_modo_touch_esta_etiquetado_y_contiene_al_cruce():
    bars = _bars([
        _row(0, 101, 101),
        _row(1, 100.4, 100.3, low=100.0, high=100.5, volume=50_000),
    ])
    # low=100 cruza 100.20 (umbral 100.19) y solo toca 100.005 (umbral 99.995).
    deep = _intent(0, "buy", qty=5, order_type="limit", limit_price=100.20, seq=0)
    touch_only = _intent(0, "buy", qty=5, order_type="limit", limit_price=100.005, seq=1)
    missed = _intent(0, "buy", qty=5, order_type="limit", limit_price=99.0, seq=2)
    cross = _sim(fill_mode="cross")
    touch = _sim(fill_mode="touch")
    assert cross.label == "pessimistic-cross" and touch.label == "optimistic-touch"
    assert touch.optimistic is True and cross.optimistic is False
    now = _now(bars)
    cross_ids = {fill.intent_id for fill in cross.submit([deep, touch_only, missed], now=now, bars=bars)}
    touch_ids = {fill.intent_id for fill in touch.submit([deep, touch_only, missed], now=now, bars=bars)}
    assert cross_ids <= touch_ids
    assert touch_only.intent_id in touch_ids
    assert touch_only.intent_id not in cross_ids
    assert missed.intent_id not in touch_ids
    assert deep.intent_id in cross_ids


def test_stop_cobra_el_gap_y_el_slippage_en_contra():
    bars = _bars([
        _row(0, 100, 100),
        _row(1, 98, 97.5, low=97.0, high=98.2),
    ])
    sim = _sim()
    fills = sim.submit(
        [_intent(0, "sell", qty=2, order_type="stop", stop_price=100)],
        now=_now(bars), bars=bars,
    )
    assert fills[0].price == pytest.approx(98 - 0.01)
    assert fills[0].liquidity == "taker"

    inside = _bars([
        _row(0, 101, 101),
        _row(1, 101, 100.2, low=99.0, high=101.2),
    ])
    sim = _sim()
    fills = sim.submit(
        [_intent(0, "sell", qty=2, order_type="stop", stop_price=100, seq=3)],
        now=_now(inside), bars=inside,
    )
    assert fills[0].price == pytest.approx(100 - 0.01)

    buy = _bars([
        _row(0, 99, 99),
        _row(1, 103, 104, low=102.5, high=104.5),
    ])
    sim = _sim()
    fills = sim.submit(
        [_intent(0, "buy", qty=2, order_type="stop", stop_price=100, seq=4)],
        now=_now(buy), bars=buy,
    )
    assert fills[0].price == pytest.approx(103 + 0.01)


def test_si_stop_y_take_caben_en_la_barra_gana_el_stop():
    bars = _bars([
        _row(0, 100, 100),
        _row(1, 100, 100, low=98.0, high=102.0, volume=100_000),
        _row(2, 100, 100, low=98.0, high=102.0, volume=100_000),
    ])
    sim = _sim()
    intent = _intent(
        0, "buy", qty=8, order_type="market",
        bracket=Bracket(take=101, stop=99),
    )
    fills = sim.submit([intent], now=_now(bars), bars=bars)
    spawned = sim.take_spawned()
    assert [fill.intent_id for fill in fills] == [intent.intent_id, f"{intent.intent_id}:stop"]
    assert fills[0].price == pytest.approx(100.01)
    assert fills[1].price == pytest.approx(99 - 0.01)
    assert fills[1].liquidity == "taker"
    take_fills = [fill for fill in fills if fill.intent_id.endswith(":take")]
    assert take_fills == []
    ids = {item.intent_id for item in spawned}
    assert f"{intent.intent_id}:stop" in ids and f"{intent.intent_id}:take" in ids


def test_solo_el_take_si_el_stop_no_entra_en_la_barra():
    bars = _bars([
        _row(0, 100, 100),
        _row(1, 100, 101.5, low=99.5, high=102.0, volume=100_000),
    ])
    sim = _sim()
    intent = _intent(0, "buy", qty=3, bracket=Bracket(take=101, stop=99), seq=2)
    fills = sim.submit([intent], now=_now(bars), bars=bars)
    assert [fill.intent_id for fill in fills] == [intent.intent_id, f"{intent.intent_id}:take"]
    assert fills[1].price == pytest.approx(101)
    assert fills[1].liquidity == "maker"


def test_flatten_es_un_market_en_la_barra_siguiente():
    bars = _bars([
        _row(0, 100, 100.2),
        _row(1, 105, 105.2),
    ])
    early = _bars([_row(0, 100, 100.2)])
    sim = _sim()
    now = bar_close_time(_at(0), "1Min") + timedelta(seconds=5)
    assert sim.flatten({"SPY": 5}, now=now, bars=early) == ()
    fills = sim.submit([], now=_now(bars), bars=bars)
    assert len(fills) == 1
    assert fills[0].ts == _at(1)
    assert fills[0].qty == 5
    assert fills[0].price == pytest.approx(105 - 0.01)


def test_el_impacto_encarece_la_compra():
    costs = _model(impact_y=1.0, tick=0.0001, spread_floor_bps=0.0)
    bars = _bars([_row(0, 100, 100), _row(1, 50, 50)])
    sim = _sim(costs, adv={"SPY": 1_000_000}, sigma={"SPY": 0.02})
    fills = sim.submit([_intent(0, "buy", qty=100)], now=_now(bars), bars=bars)
    fraction = 0.02 * np.sqrt((100 * 50) / 1_000_000)
    assert fills[0].price == pytest.approx(50 + 50 * fraction + 0.0001)


def test_las_fees_del_fill_salen_del_modelo_y_del_paper():
    costs = _model(sec_section31_rate=0.0000206, finra_taf_per_share=0.000195, finra_taf_cap=9.79)
    bars = _bars([_row(0, 100, 100), _row(1, 100, 100)])
    sim = _sim(costs)
    fills = sim.submit([_intent(0, "sell", qty=100)], now=_now(bars), bars=bars)
    expected = costs.fees(side="sell", qty=100, price=fills[0].price, liquidity="taker")
    assert fills[0].fees == pytest.approx(expected)
    assert fills[0].fees == pytest.approx(
        costs.paper_fees(side="sell", qty=100, price=fills[0].price, liquidity="taker"),
    )
    assert expected > 0


def test_un_quote_pisa_al_tick():
    bars = _bars([_row(0, 100, 100), _row(1, 100, 100)])

    def quotes(symbol, ts):
        return Quote(bid=100, ask=100.04)

    sim = _sim(quote_fn=quotes)
    fills = sim.submit([_intent(0, "buy", qty=1)], now=_now(bars), bars=bars)
    assert fills[0].price == pytest.approx(100 + 0.02)


def test_misma_semilla_mismos_parciales_y_el_rollback_no_duplica():
    bars = _bars([_row(i, 100, 100.1, volume=100_000) for i in range(6)])
    intents = [_intent(0, "buy", qty=10, seq=i) for i in range(5)]
    left = _sim(partial_prob=1.0, seed=7)
    right = _sim(partial_prob=1.0, seed=7)
    now = _now(bars)
    assert left.submit(intents, now=now, bars=bars) == right.submit(intents, now=now, bars=bars)
    again = _sim(partial_prob=1.0, seed=7)
    first = again.submit(intents, now=now, bars=bars)
    assert all(fill.qty < 10 - 1e-9 for fill in first)
    again.rollback_last()
    assert again.submit(intents, now=now, bars=bars) == first
    held = _sim(partial_prob=1.0, seed=7)
    assert held.submit(intents, now=now, bars=bars)
    assert held.submit(intents, now=now, bars=bars) == ()


def _upto(bars, n):
    return bars_from_records(
        pd.DataFrame([
            {"symbol": "SPY", "ts": ts, "o": bars.open["SPY"].iloc[i], "h": bars.high["SPY"].iloc[i],
             "l": bars.low["SPY"].iloc[i], "c": bars.close["SPY"].iloc[i],
             "v": bars.volume["SPY"].iloc[i], "vw": bars.vwap["SPY"].iloc[i], "n": 10}
            for i, ts in enumerate(bars.close.index[:n])
        ]),
        ["SPY"], source="test", feed="iex", timeframe="1Min",
    )


def test_el_rollback_deshace_parciales_hijos_de_bracket_y_spreads():
    # Una orden viva de antes del envío se llena en parte, su bracket crece y
    # el spread de la barra queda en cache. Deshacer deja todo como estaba:
    # repetir el envío da los mismos fills y las mismas órdenes vivas.
    rows = [_row(i, 100 + 0.03 * (i % 3), 100.05 - 0.02 * (i % 2), volume=100_000) for i in range(8)]
    bars = _bars(rows)
    sim = _sim(_model(use_spread_estimators=True), partial_prob=1.0, seed=11, roll_window=2)
    entry = _intent(0, "buy", qty=10, bracket=Bracket(take=200, stop=50))
    assert sim.submit([entry], now=_now(_upto(bars, 2)), bars=_upto(bars, 2))
    sim.take_spawned()
    sim.commit_last()
    window = _upto(bars, 4)
    first = sim.submit([], now=_now(window), bars=window)
    assert first and first[0].intent_id == entry.intent_id
    spawned = sim.take_spawned()
    live = sim.open_orders()
    sim.rollback_last()
    again = sim.submit([], now=_now(window), bars=window)
    assert again == first
    assert sim.take_spawned() == spawned
    assert sim.open_orders() == live


def test_el_envio_no_copia_las_ordenes_terminadas(monkeypatch):
    # El backtest envía en cada barra. Si deshacer copia todas las órdenes de
    # la corrida, el costo por barra crece con el año y la corrida es
    # cuadrática. Solo se copian las que el envío puede tocar.
    import copy

    from src.hft.execution import sim as sim_module

    bars = _bars([_row(i, 100, 100.1, volume=100_000) for i in range(4)])
    sim = _sim()
    done = [_intent(0, "buy", qty=1, seq=i) for i in range(30)]
    assert len(sim.submit(done, now=_now(_upto(bars, 2)), bars=_upto(bars, 2))) == 30
    live = _intent(1, "buy", qty=1, order_type="limit", limit_price=50.0, seq=99)
    sim.submit([live], now=_now(_upto(bars, 3)), bars=_upto(bars, 3))
    copied = []
    real = copy.deepcopy

    def counting(value, memo=None, *args):
        items = value.values() if isinstance(value, dict) else (value,)
        copied.extend(
            item.intent.intent_id for item in items if isinstance(item, sim_module._Working)
        )
        return real(value, memo, *args)

    monkeypatch.setattr(copy, "deepcopy", counting)
    sim.submit([], now=_now(bars), bars=bars)
    assert set(copied) <= {live.intent_id}


def test_una_barra_no_mira_las_ordenes_terminadas(monkeypatch):
    # Un año junta miles de órdenes terminadas. Si cada barra las vuelve a
    # recorrer (fills, stop y take en la misma barra, open_orders, punto de
    # deshacer), el costo por barra crece con la corrida. Solo se miran las
    # que todavía pueden moverse.
    from src.hft.execution import sim as sim_module

    bars = _bars([_row(i, 100, 100.1, volume=100_000) for i in range(6)])
    sim = _sim()
    filled = [_intent(0, "buy", qty=1, seq=i) for i in range(20)]
    # El take 100.10 se cruza en la barra 1 (high 100.15) y cancela el stop.
    closed = _intent(0, "buy", qty=2, bracket=Bracket(take=100.10, stop=50), seq=20)
    cancelled = _intent(0, "buy", qty=1, order_type="limit", limit_price=50.0, seq=21)
    resting = _intent(0, "buy", qty=1, order_type="limit", limit_price=50.0, seq=22)
    guarded = _intent(0, "buy", qty=1, bracket=Bracket(take=200, stop=50), seq=23)
    window = _upto(bars, 3)
    sim.submit([*filled, closed, cancelled, resting, guarded], now=_now(window), bars=window)
    sim.take_spawned()
    sim.commit_last()
    assert sim.cancel(cancelled.intent_id)
    live = {resting.intent_id, f"{guarded.intent_id}:stop", f"{guarded.intent_id}:take"}
    assert {intent.intent_id for intent in sim.open_orders()} == live
    finished = [order for key, order in sim._orders.items() if key not in live]
    assert len(finished) == 25
    reads = []

    class _Terminada(sim_module._Working):
        def __getattribute__(self, name):
            reads.append(name)
            return super().__getattribute__(name)

    for order in finished:
        order.__class__ = _Terminada
    matched = []
    real_match = sim._match

    def spy(order, ts, bars):
        matched.append(order)
        return real_match(order, ts, bars)

    monkeypatch.setattr(sim, "_match", spy)
    assert sim.submit([], now=_now(bars), bars=bars) == ()
    assert {intent.intent_id for intent in sim.open_orders()} == live
    assert not [order for order in matched if any(order is done for done in finished)]
    assert reads == []
    # Tres barras nuevas por tres órdenes vivas.
    assert len(matched) == 9


def _escalera():
    # Padre A: límite 20 @ 100 con tope 10 por barra. Padre B: market 5.
    # Barra 1: A llena 10, B llena 5, el take de A llena 10 y cancela su stop.
    # Barra 2: A llena los otros 10 y reactiva su stop (creado antes que los
    # hijos de B). Los dos stops caen en la barra 2.
    bars = _bars([
        _row(0, 100, 100.1, volume=100),
        _row(1, 100, 100.1, volume=100),
        _row(2, 100, 100, low=99.4, high=100.05, volume=100),
    ])
    parent_a = _intent(
        0, "buy", qty=20, order_type="limit", limit_price=100.0,
        bracket=Bracket(take=100.10, stop=99.5), seq=0,
    )
    parent_b = _intent(0, "buy", qty=5, bracket=Bracket(take=200, stop=99.5), seq=1)
    return bars, parent_a, parent_b


def test_el_hijo_reactivado_conserva_su_turno_en_la_barra():
    # El orden de los fills (y del rng de parciales) es el de inserción. El
    # stop de A se creó antes que el de B: aunque se reactive después, va
    # primero.
    bars, parent_a, parent_b = _escalera()
    a, b = parent_a.intent_id, parent_b.intent_id
    sim = _sim()
    fills = sim.submit([parent_a, parent_b], now=_now(bars), bars=bars)
    assert [fill.intent_id for fill in fills] == [a, b, f"{a}:take", a, f"{a}:stop", f"{b}:stop"]
    assert sim.open_orders() == ()


def test_el_rollback_revive_lo_que_el_envio_termino():
    # El envío llena el resto de A, reactiva y llena su stop, y cierra el
    # bracket de B. Deshacer deja vivas de nuevo A y los hijos de B, en su
    # orden, y repetir el envío da lo mismo.
    bars, parent_a, parent_b = _escalera()
    a, b = parent_a.intent_id, parent_b.intent_id
    sim = _sim()
    window = _upto(bars, 2)
    sim.submit([parent_a, parent_b], now=_now(window), bars=window)
    sim.take_spawned()
    sim.commit_last()
    before = [intent.intent_id for intent in sim.open_orders()]
    assert before == [a, f"{b}:stop", f"{b}:take"]
    first = sim.submit([], now=_now(bars), bars=bars)
    assert [fill.intent_id for fill in first] == [a, f"{a}:stop", f"{b}:stop"]
    assert sim.open_orders() == ()
    sim.rollback_last()
    assert [intent.intent_id for intent in sim.open_orders()] == before
    assert sim.submit([], now=_now(bars), bars=bars) == first
    assert sim.open_orders() == ()


def _neto(fills, parent):
    # Compras del padre menos ventas de sus hijos de bracket.
    out = 0.0
    for fill in fills:
        if fill.intent_id == parent.intent_id:
            out += fill.qty
        elif fill.intent_id.startswith(f"{parent.intent_id}:"):
            out -= fill.qty
    return out


def test_el_stop_reactivado_tras_el_take_protege_solo_el_fill_nuevo():
    # Barra 1: A llena 10 y su take vende 10 (posición 0); el take cancela el
    # stop con 10 sin usar. Barra 2: A llena los otros 10. El stop revive con
    # esos 10, no con 20: al caer deja la posición en 0, no en -10.
    bars, parent_a, _ = _escalera()
    a = parent_a.intent_id
    sim = _sim()
    fills = sim.submit([parent_a], now=_now(bars), bars=bars)
    assert [fill.qty for fill in fills if fill.intent_id == f"{a}:stop"] == [pytest.approx(10)]
    assert _neto(fills, parent_a) == pytest.approx(0)
    # La intención dice lo mismo: lo que el hijo ya llenó más lo que le queda.
    stop = [intent for intent in sim.take_spawned() if intent.intent_id == f"{a}:stop"][-1]
    assert stop.qty == pytest.approx(10)


def test_el_take_que_perdio_contra_el_stop_revive_solo_con_el_fill_nuevo():
    # Barra 1: A llena 10, el stop y el take caben y gana el stop (vende 10);
    # el take queda cancelado con 10 sin usar. Barra 2: A llena los otros 10.
    # Barra 3: sube al take, que vende esos 10 y no 20.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, low=98.9, high=101.2, volume=100),
        _row(2, 100, 100, low=99.5, high=100.5, volume=100),
        _row(3, 101, 101, low=100.5, high=101.5, volume=1_000),
    ])
    parent = _intent(
        0, "buy", qty=20, order_type="limit", limit_price=100.0,
        bracket=Bracket(take=101, stop=99),
    )
    p = parent.intent_id
    sim = _sim()
    fills = sim.submit([parent], now=_now(bars), bars=bars)
    assert [(fill.intent_id, fill.qty) for fill in fills] == [
        (p, pytest.approx(10)),
        (f"{p}:stop", pytest.approx(10)),
        (p, pytest.approx(10)),
        (f"{p}:take", pytest.approx(10)),
    ]
    assert sim.open_orders() == ()


def test_el_hijo_cancelado_revive_solo_con_el_fill_nuevo():
    # Cancelar el stop deja sin proteger los 10 que había. El fill siguiente
    # del padre lo revive con sus 10; el take, que siguió vivo, cubre 20.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, low=99.95, high=100.05, volume=100),
        _row(2, 100, 100, low=99.95, high=100.05, volume=100),
    ])
    parent = _intent(
        0, "buy", qty=20, order_type="limit", limit_price=100.0,
        bracket=Bracket(take=101, stop=99),
    )
    p = parent.intent_id
    sim = _sim()
    window = _upto(bars, 2)
    sim.submit([parent], now=_now(window), bars=window)
    assert sim.cancel(f"{p}:stop")
    sim.submit([], now=_now(bars), bars=bars)
    left = {intent.intent_id: intent.qty for intent in sim.open_orders()}
    assert left == {f"{p}:take": pytest.approx(20), f"{p}:stop": pytest.approx(10)}


def _qty(sim):
    return {intent.intent_id: intent.qty for intent in sim.open_orders()}


def test_el_take_parcial_baja_el_stop_hermano():
    # Padre market 20, lleno en la barra 1. Ahí mismo el take vende 10 (tope
    # 10% de 100). En la barra 2 cae el stop: vende los 10 que quedan, no 20.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, low=99.9, high=101.2, volume=100),
        _row(2, 100, 100, low=98.5, high=100.1, volume=100),
    ])
    parent = _intent(0, "buy", qty=20, bracket=Bracket(take=101, stop=99))
    p = parent.intent_id
    sim = _sim()
    fills = sim.submit([parent], now=_now(bars), bars=bars)
    assert [(fill.intent_id, fill.qty) for fill in fills] == [
        (p, pytest.approx(20)),
        (f"{p}:take", pytest.approx(10)),
        (f"{p}:stop", pytest.approx(10)),
    ]
    assert _neto(fills, parent) == pytest.approx(0)
    assert sim.open_orders() == ()


def test_si_caben_los_dos_tras_un_take_parcial_el_stop_vende_lo_que_queda():
    # Barra 1: el take vende 10 de 20. Barra 2: stop y take caben, gana el
    # stop y llena completo, pero completo son los 10 que quedan.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, low=99.9, high=101.2, volume=100),
        _row(2, 100, 100, low=98.5, high=101.5, volume=100),
    ])
    parent = _intent(0, "buy", qty=20, bracket=Bracket(take=101, stop=99))
    p = parent.intent_id
    sim = _sim()
    fills = sim.submit([parent], now=_now(bars), bars=bars)
    assert [(fill.intent_id, fill.qty) for fill in fills] == [
        (p, pytest.approx(20)),
        (f"{p}:take", pytest.approx(10)),
        (f"{p}:stop", pytest.approx(10)),
    ]
    assert sim.open_orders() == ()


def test_el_stop_parcial_baja_el_take_hermano():
    # Al revés: un parcial del stop (partial_prob) le quita al take lo que
    # vendió. Los dos hijos cubren siempre lo que sigue abierto.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, volume=100),
        _row(2, 100, 100, low=98.5, high=100.1, volume=100),
    ])
    parent = _intent(0, "buy", qty=20, bracket=Bracket(take=101, stop=99))
    p = parent.intent_id
    sim = _sim(partial_prob=1.0, seed=7)
    fills = sim.submit([parent], now=_now(bars), bars=bars)
    sold = {
        f"{p}:{kind}": sum(fill.qty for fill in fills if fill.intent_id == f"{p}:{kind}")
        for kind in ("stop", "take")
    }
    assert sold[f"{p}:stop"] > 0
    assert sold[f"{p}:take"] == 0
    left = {
        intent_id: qty - sold[intent_id]
        for intent_id, qty in _qty(sim).items()
        if intent_id in sold
    }
    assert left[f"{p}:take"] == pytest.approx(left[f"{p}:stop"])
    assert left[f"{p}:stop"] == pytest.approx(_neto(fills, parent))


def test_el_hijo_que_cubria_menos_no_cancela_entero_al_hermano():
    # Cancelar el stop y llenar el resto del padre deja stop 10 y take 20
    # sobre una posición de 20. Cae el stop y vende sus 10: quedan 10
    # abiertos y el take baja a 10, no se cancela entero.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, low=99.95, high=100.05, volume=100),
        _row(2, 100, 100, low=99.95, high=100.05, volume=100),
        _row(3, 100, 100, low=98.5, high=100.1, volume=100),
    ])
    parent = _intent(
        0, "buy", qty=20, order_type="limit", limit_price=100.0,
        bracket=Bracket(take=101, stop=99),
    )
    p = parent.intent_id
    sim = _sim()
    window = _upto(bars, 2)
    fills = list(sim.submit([parent], now=_now(window), bars=window))
    assert sim.cancel(f"{p}:stop")
    fills += sim.submit([], now=_now(bars), bars=bars)
    assert [(fill.intent_id, fill.qty) for fill in fills] == [
        (p, pytest.approx(10)),
        (p, pytest.approx(10)),
        (f"{p}:stop", pytest.approx(10)),
    ]
    assert _qty(sim) == {f"{p}:take": pytest.approx(10)}
    assert _neto(fills, parent) == pytest.approx(10)


def test_el_rollback_devuelve_al_stop_lo_que_le_quito_el_take():
    # El envío en que el take vende 10 deja el stop en 10 (su intención
    # también: no llenó nada). Deshacerlo le devuelve los 20; repetirlo da
    # lo mismo.
    bars = _bars([
        _row(0, 100, 100, volume=100),
        _row(1, 100, 100, volume=100),
        _row(2, 100, 100, low=99.9, high=101.2, volume=100),
    ])
    parent = _intent(0, "buy", qty=20, bracket=Bracket(take=101, stop=99))
    p = parent.intent_id
    sim = _sim()
    window = _upto(bars, 2)
    sim.submit([parent], now=_now(window), bars=window)
    sim.take_spawned()
    sim.commit_last()
    before = _qty(sim)
    assert before == {f"{p}:stop": pytest.approx(20), f"{p}:take": pytest.approx(20)}
    first = sim.submit([], now=_now(bars), bars=bars)
    assert [(fill.intent_id, fill.qty) for fill in first] == [(f"{p}:take", pytest.approx(10))]
    after = _qty(sim)
    assert after == {f"{p}:stop": pytest.approx(10), f"{p}:take": pytest.approx(20)}
    sim.rollback_last()
    assert _qty(sim) == before
    assert sim.submit([], now=_now(bars), bars=bars) == first
    assert _qty(sim) == after


def test_a_igual_secuencia_el_costo_pesimista_no_es_menor():
    bars = _bars([_row(i, 100 + 0.01 * i, 100.02 + 0.01 * i) for i in range(6)])
    orders = [
        _intent(0, "buy", qty=3, seq=0),
        _intent(1, "sell", qty=3, seq=1),
        _intent(2, "buy", qty=2, order_type="limit", limit_price=99.0, seq=2),
    ]
    # El límite 99 no se cruza: la secuencia de fills de mercado es la que se compara.
    pessimistic = _sim(_model(tick=0.05))
    optimistic = _sim(_model(tick=0.0001, spread_floor_bps=0))
    now = _now(bars)
    pes = pessimistic.submit(orders, now=now, bars=bars)
    opt = optimistic.submit(orders, now=now, bars=bars)
    assert [fill.intent_id for fill in pes] == [fill.intent_id for fill in opt]
    pes_cost = 0.0
    opt_cost = 0.0
    for fill, other in zip(pes, opt):
        side = "buy" if "-buy-" in fill.intent_id else "sell"
        bench = float(bars.open["SPY"].loc[pd.Timestamp(fill.ts)])
        pes_cost += implementation_shortfall(
            side=side, qty=fill.qty, benchmark=bench, fill_price=fill.price, fees=fill.fees,
        )
        opt_cost += implementation_shortfall(
            side=side, qty=other.qty, benchmark=bench, fill_price=other.price, fees=other.fees,
        )
        if side == "buy":
            assert fill.price >= other.price - 1e-12
        else:
            assert fill.price <= other.price + 1e-12
    assert pes_cost >= opt_cost - 1e-9


def test_con_latencia_el_momentum_no_gana_mas():
    rows = []
    sessions = []
    for offset in range(30):
        day = date(2024, 1, 2) + timedelta(days=offset)
        sessions.append(day)
        start = session_at(day, time(9, 30), time(9, 36)).open_utc
        path = [(100, 101), (101, 102), (102, 102), (102, 102), (102, 102), (102, 102)]
        for minute, (open_, close) in enumerate(path):
            rows.append({
                "symbol": "SPY",
                "ts": start + timedelta(minutes=minute),
                "o": open_,
                "h": max(open_, close) + 0.02,
                "l": min(open_, close) - 0.02,
                "c": close,
                "v": 100_000,
                "vw": (open_ + close) / 2,
                "n": 10,
            })
    bars = _bars(rows)
    orders = []
    for offset, day in enumerate(sessions):
        start = session_at(day, time(9, 30), time(9, 36)).open_utc
        orders.append(order_intent("mom", "SPY", "buy", 1, start, reason="impulso", seq=offset))
        orders.append(order_intent(
            "mom", "SPY", "sell", 1, start + timedelta(minutes=1), reason="salida", seq=offset,
        ))
    now = _now(bars)

    def pnl(latency):
        sim = _sim(latency_bars=latency)
        fills = {fill.intent_id: fill for fill in sim.submit(orders, now=now, bars=bars)}
        total = 0.0
        for buy_order, sell_order in zip(orders[0::2], orders[1::2]):
            buy = fills[buy_order.intent_id]
            sell = fills[sell_order.intent_id]
            total += (sell.price - buy.price) - buy.fees - sell.fees
        return total

    without = pnl(0)
    with_latency = pnl(1)
    assert with_latency <= without
    assert with_latency < without


def test_una_estrategia_aleatoria_pierde_aproximadamente_el_costo():
    sessions = [
        session_at(date(2024, 1, 2) + timedelta(days=i), time(9, 30), time(10, 0))
        for i in range(24)
    ]
    bars = synthetic_bars(
        sessions, ["SPY"], timeframe="1Min", seed=SEED, gap_prob=0.0,
        mu=0.0, sigma=0.0004, jump_prob=0.0,
    )
    tick = 0.10
    costs = _model(tick=tick)
    rng = np.random.default_rng(SEED)
    orders = []
    trips = []
    index = list(bars.close.index)
    by_day: dict = {}
    for ts in index:
        by_day.setdefault(pd.Timestamp(ts).date(), []).append(pd.Timestamp(ts))
    seq = 0
    for stamps in by_day.values():
        for start in range(0, len(stamps) - 3, 3):
            side = "buy" if float(rng.random()) < 0.5 else "sell"
            entry_ts = stamps[start].to_pydatetime()
            exit_ts = stamps[start + 1].to_pydatetime()
            exit_side = "sell" if side == "buy" else "buy"
            entry = order_intent("rw", "SPY", side, 1, entry_ts, reason="ruido", seq=seq)
            exit_ = order_intent("rw", "SPY", exit_side, 1, exit_ts, reason="salida", seq=seq)
            orders.append(entry)
            orders.append(exit_)
            trips.append((entry.intent_id, exit_.intent_id, side))
            seq += 1
    sim = _sim(costs, partial_prob=0.0)
    fills = {fill.intent_id: fill for fill in sim.submit(orders, now=_now(bars), bars=bars)}
    pnls = []
    gross = []
    for entry_id, exit_id, side in trips:
        entry = fills[entry_id]
        exit_ = fills[exit_id]
        if side == "buy":
            trade = (exit_.price - entry.price) - entry.fees - exit_.fees
        else:
            trade = (entry.price - exit_.price) - entry.fees - exit_.fees
        entry_open = float(bars.open["SPY"].loc[pd.Timestamp(entry.ts)])
        exit_open = float(bars.open["SPY"].loc[pd.Timestamp(exit_.ts)])
        mark = (exit_open - entry_open) if side == "buy" else (entry_open - exit_open)
        pnls.append(trade)
        gross.append(mark)
    pnls = np.asarray(pnls)
    gross = np.asarray(gross)
    n = len(pnls)
    assert n >= 150
    mean = float(pnls.mean())
    se = float(pnls.std(ddof=1) / np.sqrt(n))
    assert mean < 0
    assert mean / se < -2
    # El P&L más el costo explícito (shortfall) recupera el cambio entre aperturas.
    gap = float((pnls - gross).mean())
    assert gap < 0
    assert abs(float(gross.mean())) < 4 * float(gross.std(ddof=1) / np.sqrt(n))
    # gap es -costos: dos veces el tick, porque no hay fees en este modelo.
    assert gap == pytest.approx(-2 * tick, abs=1e-9)


def test_el_motor_ejecuta_el_market_en_t_mas_1_con_este_simulador():
    bars = _bars([
        _row(0, 100, 101),
        _row(1, 102, 103),
        _row(2, 104, 105),
    ])
    sim = _sim()
    engine = _engine_with(sim)
    for i, ts in enumerate(bars.close.index):
        moment = bar_close_time(ts.to_pydatetime(), "1Min") + timedelta(seconds=5)
        end = bars.close.index[: i + 1]
        engine.step(moment, _head(bars, i + 1))
        del end
    assert engine.book.position("SPY") == 1
    fill = [fill for event in engine.journal for fill in event.fills][0]
    assert fill.price == pytest.approx(102.01)
    assert engine.book.cash == pytest.approx(100_000 - 102.01)


def test_el_bracket_del_simulador_entra_al_libro_por_el_motor():
    bars = _bars([
        _row(0, 100, 100),
        _row(1, 100, 100, low=98.0, high=102.0, volume=100_000),
    ])

    class _ConBracket:
        id = "br"
        params = _Params()
        theory = _theory()

        def required_history(self):
            return (1, 0)

        def on_bar(self, ctx):
            if len(ctx.bars.close.index) != 1:
                return []
            return [order_intent(
                self.id, "SPY", "buy", 2, ctx.bar_ts,
                bracket=Bracket(take=101, stop=99), reason="bracket",
            )]

    engine = _engine_with(_sim(), _ConBracket())
    for i, ts in enumerate(bars.close.index):
        moment = bar_close_time(ts.to_pydatetime(), "1Min") + timedelta(seconds=5)
        engine.step(moment, _head(bars, i + 1))
    fills = [fill for event in engine.journal for fill in event.fills]
    assert len(fills) == 2
    assert fills[1].intent_id == fills[0].intent_id + ":stop"
    assert engine.book.position("SPY") == 0
    assert engine.book.realized_pnl < 0


def test_un_limite_vivo_se_llena_en_una_barra_posterior_del_motor():
    bars = _bars([
        _row(0, 101, 101, low=100.8, high=101.2),
        _row(1, 101, 100.8, low=100.4, high=101.1),
        _row(2, 100.2, 99.8, low=99.90, high=100.3, volume=100_000),
    ])
    sim = _sim()

    class _Limite:
        id = "lim"
        params = _Params()
        theory = _theory()

        def required_history(self):
            return (1, 0)

        def on_bar(self, ctx):
            if len(ctx.bars.close.index) != 1:
                return []
            return [order_intent(
                self.id, "SPY", "buy", 4, ctx.bar_ts,
                order_type="limit", limit_price=100, reason="pasivo",
            )]

    engine = _engine_with(sim, _Limite())
    for i, ts in enumerate(bars.close.index):
        moment = bar_close_time(ts.to_pydatetime(), "1Min") + timedelta(seconds=5)
        engine.step(moment, _head(bars, i + 1))
    fills = [fill for event in engine.journal for fill in event.fills]
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100)
    assert fills[0].ts == _at(2)
    assert engine.book.position("SPY") == 4


@dataclass(frozen=True)
class _Params:
    qty: float = 1.0


def _theory() -> TheoryCard:
    return TheoryCard(
        hypothesis="Regla fija para probar el simulador dentro del motor.",
        mechanism="Nadie paga: el test mira el fill, no un edge.",
        references=("HFT-07",),
        data_requirements=("barras 1Min",),
        params=(TheoryParam("qty", 1, "Cantidad del test.", "HFT-07"),),
        holding_period="una barra",
        failure_regimes=("no aplica",),
        kill_criteria=("no usarla fuera del test",),
        research_only=True,
    )


class _Compra:
    id = "una"
    params = _Params()
    theory = _theory()

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        if len(ctx.bars.close.index) != 1:
            return []
        return [order_intent(self.id, "SPY", "buy", 1, ctx.bar_ts, reason="primera")]


class _Identidad:
    def size(self, intents, state, now):
        return list(intents)

    def filter(self, intents, state, now):
        return list(intents), []


class _Reloj:
    def now(self):
        raise AssertionError("el motor leyó el reloj")


def _head(bars, n):
    frames = {
        name: getattr(bars, name).iloc[:n]
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    return type(bars)(meta=bars.meta, **frames)


def _engine_with(sim, strategy=None):
    return Engine(
        [strategy or _Compra()],
        lambda window: log_return(window.close),
        _Identidad(),
        _Identidad(),
        sim,
        FakeCalendar([session_at(DAY, time(9, 30), time(16, 0))]),
        _Reloj(),
        book=HftBook(100_000),
        timeframe="1Min",
        grace_s=5,
        latency_bars=1,
    )
