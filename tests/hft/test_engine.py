"""Motor único: t+1, paridad backtest/vivo y un solo features_fn (HFT-06)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, bar_close_time, session_at
from src.hft.data.bars import IntradayBars, bars_from_records, empty_bars
from src.hft.data.synthetic import synthetic_bars
from src.hft.engine import OTHER_OWNER, Engine
from src.hft.features import log_return
from src.hft.types import Fill, TheoryCard, TheoryParam, order_intent
from src.settings import SEED

UTC = timezone.utc
DAY = date(2024, 1, 3)
SESSION = session_at(DAY, time(9, 30), time(10, 0))
FEE = 0.25


class _RelojQueFalla:
    def now(self):
        raise AssertionError("el motor leyó el reloj")


class FakeClock:
    """Reloj del arnés. El motor no lo recibe: el arnés le pasa ``now`` a ``step``."""

    def __init__(self, moment: datetime):
        self._now = moment

    def now(self) -> datetime:
        return self._now

    def set(self, moment: datetime) -> None:
        self._now = moment


@dataclass(frozen=True)
class _Qty:
    qty: float = 1.0


def _theory(qty: float) -> TheoryCard:
    return TheoryCard(
        hypothesis="Regla de agenda para probar el motor, sin hipótesis de mercado.",
        mechanism="Nadie paga: el test mira la ruta, no un edge.",
        references=("HFT-06",),
        data_requirements=("barras 1Min ya finales",),
        params=(TheoryParam(
            "qty", qty, "Cantidad fija del test, no estimada.", "HFT-06",
        ),),
        holding_period="unas barras",
        failure_regimes=("no aplica",),
        kill_criteria=("no usarla fuera del test",),
        research_only=True,
    )


class Agenda:
    """Compra y vende en barras fijas. No lee el reloj ni el azar."""

    id = "agenda"

    def __init__(self, qty: float = 2.0):
        self.params = _Qty(qty)
        self.theory = _theory(qty)
        self.actions = {3: "buy", 8: "sell", 12: "sell", 18: "buy"}

    def required_history(self):
        return (1, 1)

    def on_bar(self, ctx):
        side = self.actions.get(len(ctx.bars.close.index))
        if side is None:
            return []
        return [order_intent(
            self.id, "SPY", side, self.params.qty, ctx.bar_ts, reason=f"agenda {side}",
        )]


class _CompraUnaVez:
    id = "una"
    params = _Qty(1.0)
    theory = _theory(1.0)

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        if len(ctx.bars.close.index) != 1:
            return []
        return [order_intent(self.id, "SPY", "buy", self.params.qty, ctx.bar_ts, reason="primera")]


class _Identidad:
    def size(self, intents, state, now):
        return list(intents)

    def filter(self, intents, state, now):
        return list(intents), []


class _RechazaCompras:
    def filter(self, intents, state, now):
        approved = []
        rejected = []
        for intent in intents:
            if intent.side == "buy":
                rejected.append((intent, "compra bloqueada"))
            else:
                approved.append(intent)
        return approved, rejected


class _Doble:
    def size(self, intents, state, now):
        return [replace(intent, qty=intent.qty * 2) for intent in intents]


class _Cien:
    def size(self, intents, state, now):
        return [replace(intent, qty=100) for intent in intents]


class _Apertura:
    """Fill de mercado al open de la última barra visible. No es el simulador de HFT-07."""

    def __init__(self, fee_per_share: float = FEE):
        self.fee_per_share = fee_per_share

    def submit(self, intents, *, now, bars):
        ts = bars.close.index[-1].to_pydatetime()
        fills = []
        for intent in intents:
            if intent.order_type != "market":
                continue
            fills.append(Fill(
                intent_id=intent.intent_id,
                ts=ts,
                qty=intent.qty,
                price=float(bars.open[intent.symbol].iloc[-1]),
                fees=self.fee_per_share * intent.qty,
                liquidity="taker",
            ))
        return fills


def _head(bars: IntradayBars, n: int) -> IntradayBars:
    frames = {
        name: getattr(bars, name).iloc[:n]
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    return IntradayBars(meta=bars.meta, **frames)


def _engine(strategy, bars_cash: float = 100_000, **kwargs) -> Engine:
    passthrough = _Identidad()
    return Engine(
        [strategy],
        kwargs.pop("features_fn", lambda window: log_return(window.close)),
        kwargs.pop("risk", passthrough),
        kwargs.pop("sizer", passthrough),
        kwargs.pop("execution", _Apertura()),
        FakeCalendar([SESSION]),
        _RelojQueFalla(),
        book=kwargs.pop("book", HftBook(bars_cash)),
        timeframe="1Min",
        grace_s=5,
        **kwargs,
    )


def _manual(prices: list) -> IntradayBars:
    """Barras de un minuto desde la apertura, con OHLC válido."""
    start = SESSION.open_utc
    rows = []
    for i, (open_, close) in enumerate(prices):
        high = max(open_, close) + 0.1
        low = min(open_, close) - 0.1
        rows.append({
            "symbol": "SPY",
            "ts": start + timedelta(minutes=i),
            "o": open_,
            "h": high,
            "l": low,
            "c": close,
            "v": 1_000,
            "vw": (high + low + close) / 3,
            "n": 10,
        })
    return bars_from_records(
        pd.DataFrame(rows), ["SPY"], source="test", feed="iex", timeframe="1Min",
    )


def _ready(ts: datetime, extra_s: float = 0) -> datetime:
    return bar_close_time(ts, "1Min") + timedelta(seconds=5 + extra_s)


def _drive(engine: Engine, bars: IntradayBars, *, late: bool, include_next: bool, clock: FakeClock):
    index = bars.close.index
    for i, ts in enumerate(index):
        extra = 20 if late and i % 5 == 0 else 0
        moment = _ready(ts.to_pydatetime(), extra)
        clock.set(moment)
        end = min(i + (2 if include_next else 1), len(index))
        engine.step(clock.now(), _head(bars, end))


def _economic(engine: Engine):
    intents = []
    fills = []
    for event in engine.journal:
        for intent in event.intents:
            intents.append((
                intent.intent_id, intent.side, intent.qty, intent.symbol,
                intent.signal_ts, intent.order_type,
            ))
        for fill in event.fills:
            fills.append((
                fill.intent_id, fill.qty, fill.price, fill.fees, fill.liquidity, fill.ts,
            ))
    book = engine.book
    return intents, fills, (book.cash, book.realized_pnl, book.fees, book.positions())


def test_la_intencion_de_t_se_ejecuta_en_t_mas_1():
    bars = _manual([(100, 101), (102, 103), (104, 105)])
    engine = _engine(_CompraUnaVez())
    first = bars.close.index[0].to_pydatetime()
    engine.step(_ready(first), _head(bars, 2))
    assert engine.book.position("SPY") == 0
    assert engine.journal[-1].fills == ()
    assert engine.journal[-1].orders == ()
    assert len(engine.journal[-1].held) == 1
    assert engine.journal[-1].intents[0].signal_ts == first

    second = bars.close.index[1].to_pydatetime()
    engine.step(_ready(second), _head(bars, 3))
    assert engine.book.position("SPY") == 1
    fill = engine.journal[-1].fills[0]
    assert fill.price == 102
    assert fill.fees == FEE
    assert fill.ts == second
    assert engine.journal[-1].held == ()


def test_una_barra_que_no_es_final_no_entra():
    bars = _manual([(100, 101), (102, 103)])
    seen = []

    class _Cuenta:
        id = "cuenta"
        params = _Qty()
        theory = _theory(1.0)

        def required_history(self):
            return (1, 0)

        def on_bar(self, ctx):
            seen.append(len(ctx.bars.close.index))
            return []

    engine = _engine(_Cuenta())
    first = bars.close.index[0].to_pydatetime()
    # La segunda barra ya está en el frame, pero su cierre + gracia no llegó.
    engine.step(_ready(first), bars)
    assert seen == [1]


def test_un_lote_de_barras_no_mira_el_futuro():
    bars = _manual([(100, 101), (102, 103), (104, 105)])
    seen = []

    class _Cuenta:
        id = "cuenta"
        params = _Qty()
        theory = _theory(1.0)

        def required_history(self):
            return (1, 0)

        def on_bar(self, ctx):
            seen.append((len(ctx.bars.close.index), float(ctx.bars.close["SPY"].iloc[-1])))
            return []

    engine = _engine(_Cuenta())
    last = bars.close.index[-1].to_pydatetime()
    engine.step(_ready(last), bars)
    assert seen == [(1, 101), (2, 103), (3, 105)]


def test_el_riesgo_rechaza_con_motivo_y_no_llega_al_libro():
    bars = _manual([(100, 101), (102, 103)])
    engine = _engine(_CompraUnaVez(), risk=_RechazaCompras())
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert engine.book.position("SPY") == 0
    assert engine.book.cash == 100_000
    reasons = [event.reason for event in engine.journal[-2].rejections]
    assert reasons == ["compra bloqueada"]
    assert all(not event.fills for event in engine.journal)


def test_el_sizer_fija_la_cantidad_que_se_ejecuta():
    bars = _manual([(100, 101), (102, 103)])
    engine = _engine(_CompraUnaVez(), sizer=_Doble())
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert engine.book.position("SPY") == 2
    assert engine.journal[-1].fills[0].qty == 2


def test_latencia_de_dos_barras_espera_a_t_mas_2():
    bars = _manual([(100, 101), (102, 103), (104, 105)])
    engine = _engine(_CompraUnaVez(), latency_bars=2)
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert engine.journal[1].fills == ()
    assert engine.journal[2].fills[0].price == 104
    assert engine.book.position("SPY") == 1


def test_un_fill_imposible_no_toca_el_libro_ni_suelta_la_intencion():
    bars = _manual([(100, 101), (110, 111)])
    book = HftBook(1_000)
    engine = _engine(_CompraUnaVez(), book=book, sizer=_Cien())
    engine.step(_ready(bars.close.index[0].to_pydatetime()), _head(bars, 1))
    with pytest.raises(ValueError, match="cash"):
        engine.step(_ready(bars.close.index[1].to_pydatetime()), _head(bars, 2))
    assert book.cash == 1_000
    assert book.position("SPY") == 0
    assert len(engine._pending) == 1


def test_paridad_backtest_y_vivo_con_barras_tardias():
    bars = synthetic_bars(
        [SESSION], ["SPY"], timeframe="1Min", seed=SEED, gap_prob=0.0, sigma=0.0008,
    )
    assert len(bars.close.index) == 30
    calls = []

    def features(window: IntradayBars):
        out = log_return(window.close)
        calls.append(tuple(out.iloc[-1].tolist()))
        return out

    strategy = Agenda()
    execution = _Apertura()
    risk = _Identidad()
    backtest = _engine(strategy, features_fn=features, risk=risk, sizer=risk, execution=execution)
    live = _engine(strategy, features_fn=features, risk=risk, sizer=risk, execution=execution)
    assert backtest.features_fn is live.features_fn is features
    assert backtest.strategies[0] is live.strategies[0]

    _drive(backtest, bars, late=False, include_next=False, clock=FakeClock(SESSION.open_utc))
    logged = list(calls)
    calls.clear()
    _drive(live, bars, late=True, include_next=True, clock=FakeClock(SESSION.open_utc))
    assert len(logged) == len(calls) == 30
    for left, right in zip(logged, calls):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert (pd.isna(x) and pd.isna(y)) or x == y

    assert _economic(backtest) == _economic(live)
    opens = bars.open["SPY"]
    qty = 2
    fee = FEE * qty
    long_pnl = (float(opens.iloc[8]) - float(opens.iloc[3])) * qty - 2 * fee
    short_pnl = (float(opens.iloc[12]) - float(opens.iloc[18])) * qty - 2 * fee
    expected = long_pnl + short_pnl
    assert backtest.book.position("SPY") == 0
    assert backtest.book.fees_open() == 0
    assert backtest.book.realized_pnl == pytest.approx(expected)
    assert backtest.book.cash == pytest.approx(100_000 + expected)
    assert backtest.book.fees == pytest.approx(4 * fee)
    assert backtest._pending == []


class _SinCupo:
    """Omite todo: el lote ya gastó el apalancamiento."""

    def size(self, intents, state, now):
        return []


class _Desordena:
    def size(self, intents, state, now):
        return list(reversed(intents)) + [replace(intents[0], intent_id="ajena")]


def test_la_intencion_que_el_sizer_omite_queda_como_rechazo():
    bars = _manual([(100, 101), (102, 103)])
    engine = _engine(_CompraUnaVez(), sizer=_SinCupo())
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert engine.book.position("SPY") == 0
    first = engine.journal[0]
    assert first.intents == ()
    assert [event.reason for event in first.rejections] == ["sizing_sin_cupo"]
    assert first.rejections[0].symbol == "SPY"
    assert first.rejections[0].intent_id
    assert all(not event.fills for event in engine.journal)


def test_el_sizer_no_puede_reordenar_ni_inventar_intenciones():
    bars = _manual([(100, 101)])
    engine = _engine(_CompraUnaVez(), sizer=_Desordena())
    with pytest.raises(ValueError, match="intent_id"):
        engine.step(_ready(bars.close.index[0].to_pydatetime()), _head(bars, 1))


def _con_posicion(symbol: str, qty: float, price: float) -> HftBook:
    """Libro que arranca con una posición, como el relanzamiento del paper."""
    book = HftBook(100_000)
    intent = order_intent("reconcile", symbol, "buy", qty, SESSION.open_utc, reason="reinicio")
    book.apply_fills(
        [Fill(intent.intent_id, SESSION.open_utc, qty, price, 0.0, "taker")],
        {intent.intent_id: intent},
    )
    return book


class _Anota:
    """Sizer y riesgo que solo anotan lo que reciben."""

    def __init__(self) -> None:
        self.calls = []

    def size(self, intents, state, now):
        self.calls.append(state)
        return list(intents)

    def filter(self, intents, state, now):
        self.calls.append(state)
        return list(intents), []


def test_sin_barras_la_posicion_abierta_queda_sin_marca_hasta_su_barra():
    # Relanzamiento: el libro trae SPY y todavía no hay frames. No hay marca:
    # ni la NAV ni el P&L no realizado se inventan (tampoco con el costo).
    engine = _engine(_CompraUnaVez(), book=_con_posicion("SPY", 10, 1.0))
    empty = empty_bars(["SPY"], source="test", feed="iex", timeframe="1Min")
    result = engine.step(_ready(SESSION.open_utc), empty)
    state = result.events[0].state
    assert state["unmarked"] == ["SPY"]
    assert state["nav"] is None
    assert state["unrealized_pnl"] is None
    bars = _manual([(100, 101)])
    engine.step(_ready(bars.close.index[0].to_pydatetime()), bars)
    state = engine.journal[-1].state
    assert state["unmarked"] == []
    assert state["nav"] == pytest.approx(100_000 - 10 * 1.0 + 10 * 101)
    assert state["unrealized_pnl"] == pytest.approx(10 * (101 - 1.0))


def test_con_una_posicion_sin_marca_no_se_dimensiona_ninguna_intencion():
    # TLT abierta y solo llegan barras de SPY: sin NAV no hay cupo que medir.
    # La intención queda rechazada con motivo y ni el sizer ni el riesgo la ven.
    gate = _Anota()
    engine = _engine(_CompraUnaVez(), book=_con_posicion("TLT", 10, 90.0), sizer=gate, risk=gate)
    bars = _manual([(100, 101), (102, 103)])
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    first = engine.journal[0]
    assert first.intents == ()
    assert [event.reason for event in first.rejections] == ["libro_sin_marca"]
    assert first.rejections[0].symbol == "SPY"
    assert first.state["unmarked"] == ["TLT"]
    assert first.state["nav"] is None
    assert gate.calls == []
    assert engine.book.position("SPY") == 0


class _MiraPendiente:
    """Compra en la primera barra y anota posición y pendiente que ve el contexto."""

    id = "mira"
    params = _Qty(10.0)
    theory = _theory(10.0)

    def __init__(self, order_type: str = "limit"):
        self.order_type = order_type
        self.seen = []

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        self.seen.append((ctx.position("SPY"), ctx.pending_qty("SPY")))
        if len(ctx.bars.close.index) != 1:
            return []
        limit = 200.0 if self.order_type == "limit" else None
        return [order_intent(
            self.id, "SPY", "buy", self.params.qty, ctx.bar_ts,
            order_type=self.order_type, limit_price=limit, reason="primera",
        )]


class _ParcialConStop:
    """Llena 4 de 10 al recibir la orden y el resto en la barra siguiente.

    Cuelga un stop de protección que sigue vivo: no es una entrada pendiente.
    """

    def __init__(self):
        self._open = {}
        self._spawned = []

    def submit(self, intents, *, now, bars):
        fills = []
        for intent_id, (intent, left) in list(self._open.items()):
            if intent.order_type == "stop":
                continue
            fills.append(Fill(intent_id, now, left, 100.0, 0.0, "taker"))
            del self._open[intent_id]
        for intent in intents:
            fills.append(Fill(intent.intent_id, now, 4.0, 100.0, 0.0, "taker"))
            self._open[intent.intent_id] = (intent, intent.qty - 4.0)
            stop = order_intent(
                intent.strategy_id, "SPY", "sell", intent.qty, intent.signal_ts,
                order_type="stop", stop_price=90.0, intent_id=f"{intent.intent_id}-stop",
            )
            self._open[stop.intent_id] = (stop, stop.qty)
            self._spawned.append(stop)
        return fills

    def open_orders(self):
        return tuple(intent for intent, _left in self._open.values())

    def take_spawned(self):
        out = tuple(self._spawned)
        self._spawned = []
        return out


def test_el_contexto_ve_lo_que_falta_llenar_de_las_ordenes_abiertas():
    # HFT-PAPER-15: el libro solo trae lo lleno. Una entrada abierta tiene
    # que verse en el contexto, o la estrategia la repite.
    bars = _manual([(100, 101), (102, 103), (104, 105), (106, 107)])
    strategy = _MiraPendiente()
    engine = _engine(strategy, execution=_ParcialConStop())
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert strategy.seen == [(0.0, 0.0), (4.0, 6.0), (10.0, 0.0), (10.0, 0.0)]


def test_el_contexto_ve_la_intencion_que_espera_su_barra_de_ejecucion():
    bars = _manual([(100, 101), (102, 103), (104, 105)])
    strategy = _MiraPendiente(order_type="market")
    engine = _engine(strategy, latency_bars=2)
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert strategy.seen == [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]


class _SaleConLoQueFalta:
    """Compra 10 en la primera barra y después sale con ``ctx.exit_qty``."""

    id = "sale"
    params = _Qty(10.0)
    theory = _theory(10.0)

    def __init__(self):
        self.seen = []

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        self.seen.append((ctx.position("SPY"), ctx.pending_qty("SPY"), ctx.exit_qty("SPY")))
        if len(ctx.bars.close.index) == 1:
            return [order_intent(
                self.id, "SPY", "buy", self.params.qty, ctx.bar_ts,
                order_type="limit", limit_price=200.0, reason="entrada",
            )]
        qty = ctx.exit_qty("SPY")
        if qty <= 1e-9:
            return []
        return [order_intent(self.id, "SPY", "sell", qty, ctx.bar_ts, reason="salida")]


class _RestoYSalidaVivos:
    """Llena 4 de la compra al recibirla. El resto y las ventas quedan vivos."""

    def __init__(self):
        self.sent = []
        self._open = {}

    def submit(self, intents, *, now, bars):
        fills = []
        for intent in intents:
            self.sent.append(intent)
            self._open[intent.intent_id] = intent
            if intent.side == "buy":
                fills.append(Fill(intent.intent_id, now, 4.0, 100.0, 0.0, "taker"))
        return fills

    def open_orders(self):
        return tuple(self._open.values())


def test_la_salida_no_se_repite_aunque_quede_vivo_el_resto_de_la_entrada():
    # HFT-PAPER-16: con un solo neto por símbolo, resto de compra (+6) y
    # venta viva (-4) dan +2 y la salida se mandaba otra vez.
    bars = _manual([(100, 101), (102, 103), (104, 105), (106, 107)])
    strategy = _SaleConLoQueFalta()
    execution = _RestoYSalidaVivos()
    engine = _engine(strategy, execution=execution)
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert strategy.seen == [
        (0.0, 0.0, 0.0),
        (4.0, 6.0, 4.0),
        (4.0, 2.0, 0.0),
        (4.0, 2.0, 0.0),
    ]
    sells = [intent for intent in execution.sent if intent.side == "sell"]
    assert [intent.qty for intent in sells] == [4.0]


class _CompraYCancela:
    """Compra con límite en la primera barra. Una barra después pide cancelar."""

    params = _Qty(10.0)
    theory = _theory(10.0)

    def __init__(self, strategy_id: str, cancel_at: int = 0, symbol: str = "SPY"):
        self.id = strategy_id
        self.cancel_at = cancel_at
        self.symbol = symbol
        self._cancels = []

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        n = len(ctx.bars.close.index)
        if n == self.cancel_at:
            self._cancels.append(self.symbol)
        if n != 1:
            return []
        return [order_intent(
            self.id, self.symbol, "buy", self.params.qty, ctx.bar_ts,
            order_type="limit", limit_price=50.0, reason="entrada",
        )]

    def take_cancels(self):
        out = tuple(self._cancels)
        self._cancels = []
        return out


class _LimitesConStop:
    """No llena límites. Cada compra cuelga un stop hijo que también queda vivo."""

    def __init__(self):
        self.sent = []
        self.cancelled = []
        self._open = {}
        self._spawned = []

    def submit(self, intents, *, now, bars):
        for intent in intents:
            self.sent.append(intent)
            self._open[intent.intent_id] = intent
            stop = order_intent(
                intent.strategy_id, intent.symbol, "sell", intent.qty, intent.signal_ts,
                order_type="stop", stop_price=40.0, intent_id=f"{intent.intent_id}-stop",
            )
            self._open[stop.intent_id] = stop
            self._spawned.append(stop)
        return ()

    def open_orders(self):
        return tuple(self._open.values())

    def take_spawned(self):
        out = tuple(self._spawned)
        self._spawned = []
        return out

    def cancel(self, intent_id):
        self.cancelled.append(intent_id)
        return self._open.pop(intent_id, None) is not None


def _engine_multi(strategies, execution, **kwargs) -> Engine:
    passthrough = _Identidad()
    return Engine(
        strategies,
        lambda window: None,
        passthrough,
        passthrough,
        execution,
        FakeCalendar([SESSION]),
        _RelojQueFalla(),
        book=HftBook(100_000),
        timeframe="1Min",
        grace_s=5,
        **kwargs,
    )


def test_el_motor_cancela_lo_vivo_que_la_estrategia_pide():
    # Pata colgada de pairs_ou: la estrategia no ve la ejecución. El motor
    # cancela por ella, igual en backtest, sim y paper (HFT-PAPER-16).
    # Cada una en su símbolo: dos dueños no comparten uno (HFT-PAPER-08).
    bars = _manual_multi([(100, 101), (102, 103), (104, 105)])
    pide = _CompraYCancela("pide", cancel_at=2)
    otra = _CompraYCancela("otra", symbol="QQQ")
    execution = _LimitesConStop()
    engine = _engine_multi([pide, otra], execution)
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    entry = next(intent for intent in execution.sent if intent.strategy_id == "pide")
    # Solo la entrada de quien lo pidió. Ni la de la otra estrategia ni el
    # stop hijo: ese protege una posición, no la abre.
    assert execution.cancelled == [entry.intent_id]
    assert engine.journal[1].cancelled == (entry.intent_id,)
    assert engine.journal[2].cancelled == ()
    still = sorted((intent.strategy_id, intent.order_type) for intent in execution.open_orders())
    assert still == [("otra", "limit"), ("otra", "stop"), ("pide", "stop")]


def test_lo_que_espera_su_barra_se_descarta_sin_enviarse():
    bars = _manual([(100, 101), (102, 103), (104, 105)])
    pide = _CompraYCancela("pide", cancel_at=2)
    execution = _LimitesConStop()
    engine = _engine_multi([pide], execution, latency_bars=2)
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))
    assert execution.sent == []
    assert execution.cancelled == []
    held = engine.journal[0].held
    assert len(held) == 1
    assert engine.journal[1].cancelled == held
    assert engine.journal[1].held == ()


# Ventana acotada. Sin recorte, cada barra de un backtest largo copia todo
# lo visto y la corrida es cuadrática. La ventana trae las sesiones previas
# que se calientan al arrancar y la sesión en curso hasta t.

NY = ZoneInfo("America/New_York")
DAYS = (
    date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
    date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9),
)
SHORT = tuple(session_at(day, time(9, 30), time(10, 0)) for day in DAYS)
_FRAMES = ("open", "high", "low", "close", "volume", "vwap", "trades")


class _MiraVentana:
    """Anota qué días trae la ventana en cada barra. No opera."""

    id = "mira"
    params = _Qty(1.0)
    theory = _theory(1.0)

    def __init__(self, history):
        self._history = history
        self.seen = []

    def required_history(self):
        return self._history

    def on_bar(self, ctx):
        stamps = pd.DatetimeIndex(ctx.bars.close.index)
        days = sorted(set(stamps.tz_convert(NY).date))
        self.seen.append((ctx.bar_ts, days, len(stamps), ctx.bars.meta.received))
        return []


def _engine_sesiones(strategy, **kwargs) -> Engine:
    passthrough = _Identidad()
    return Engine(
        [strategy],
        lambda window: None,
        passthrough,
        passthrough,
        _Apertura(),
        FakeCalendar(SHORT),
        _RelojQueFalla(),
        book=kwargs.pop("book", HftBook(100_000)),
        timeframe="1Min",
        grace_s=5,
        **kwargs,
    )


def _de_la_sesion(bars: IntradayBars, session) -> IntradayBars:
    index = bars.close.index
    mask = (index >= pd.Timestamp(session.open_utc)) & (index < pd.Timestamp(session.close_utc))
    return IntradayBars(meta=bars.meta, **{name: getattr(bars, name).loc[mask] for name in _FRAMES})


def _al_cierre(session) -> datetime:
    return session.close_utc + timedelta(seconds=5)


def test_la_ventana_trae_las_sesiones_declaradas_y_la_sesion_en_curso():
    strategy = _MiraVentana((5, 2))
    engine = _engine_sesiones(strategy)
    bars = synthetic_bars(SHORT, ["SPY"], seed=SEED)
    for session in SHORT:
        engine.step(_al_cierre(session), _de_la_sesion(bars, session))
    assert len(strategy.seen) == 30 * len(SHORT)
    for bar_ts, days, n, _received in strategy.seen:
        position = DAYS.index(bar_ts.astimezone(NY).date())
        assert days == list(DAYS[max(0, position - 2): position + 1])
        minute = int((bar_ts - SHORT[position].open_utc).total_seconds() // 60)
        assert n == 30 * min(position, 2) + minute + 1


def test_la_marca_de_un_simbolo_sin_barras_recientes_no_se_pierde():
    # TLT abierta y sin barras desde el primer día: su marca sigue siendo el
    # último cierre visto aunque ese día ya no entre en la ventana. Sin ella
    # el libro queda sin marca y todo se rechaza (libro_sin_marca).
    strategy = _MiraVentana((1, 1))
    engine = _engine_sesiones(strategy, book=_con_posicion("TLT", 10, 90.0))
    bars = synthetic_bars(SHORT, ["SPY", "TLT"], seed=SEED)
    later = bars.close.index >= pd.Timestamp(SHORT[1].open_utc)
    frames = {name: getattr(bars, name).copy() for name in _FRAMES}
    for frame in frames.values():
        frame.loc[later, "TLT"] = float("nan")
    bars = IntradayBars(meta=bars.meta, **frames)
    last_tlt = float(frames["close"]["TLT"].dropna().iloc[-1])
    for session in SHORT:
        engine.step(_al_cierre(session), _de_la_sesion(bars, session))
    state = engine.journal[-1].state
    assert state["unmarked"] == []
    assert state["unrealized_pnl"] == pytest.approx(10 * (last_tlt - 90.0))
    assert strategy.seen[-1][3] == ("SPY", "TLT")
    # Sin barras nuevas el estado sale del libro y de la última marca vista.
    idle = engine.step(_al_cierre(SHORT[-1]) + timedelta(minutes=1), _de_la_sesion(bars, SHORT[-1]))
    assert idle.events[0].state["unmarked"] == []


def test_el_calentamiento_del_paper_entra_entero_en_la_ventana():
    # El paper calienta max(sesiones, 1) sesiones previas y corre un día con
    # barras que llegan de a una: el recorte no le quita nada.
    strategy = _MiraVentana((5, 2))
    engine = _engine_sesiones(strategy)
    bars = synthetic_bars(SHORT, ["SPY"], seed=SEED)
    for session in SHORT[-3:-1]:
        engine._ingest(_de_la_sesion(bars, session), _al_cierre(session))
    today = _de_la_sesion(bars, SHORT[-1])
    for i, ts in enumerate(today.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(today, i + 1))
    assert len(strategy.seen) == 30
    for i, (_bar_ts, days, n, _received) in enumerate(strategy.seen):
        assert days == list(DAYS[-3:])
        assert n == 60 + i + 1


# Dueño de cada posición (HFT-PAPER-08). Cada estrategia ve solo lo suyo:
# su posición, lo que falta llenar de sus órdenes y nada del resto. El
# motor rechaza el símbolo que tiene otra.


def _manual_multi(prices: list, symbols=("SPY", "QQQ")) -> IntradayBars:
    """``_manual`` con los mismos precios para varios símbolos."""
    one = _manual(prices)
    frames = {}
    for name in _FRAMES:
        frame = getattr(one, name)
        frames[name] = pd.DataFrame(
            {symbol: frame["SPY"].to_numpy() for symbol in symbols}, index=frame.index,
        )
    return IntradayBars(meta=one.meta, **frames)


_SEIS = [(100, 101), (102, 103), (104, 105), (106, 107), (108, 109), (110, 111)]


class _Guion:
    """Entra según un guion por barra. Desde ``flatten_from`` cierra lo que ve.

    El cierre recorre ``ctx.positions`` con ``ctx.exit_qty``, como el
    ``_flatten`` de ORB o de VWAP. Anota lo que le muestra el contexto.
    """

    params = _Qty(1.0)
    theory = _theory(1.0)

    def __init__(self, strategy_id, entries=None, *, flatten_from=None, atomic=False,
                 order_type="market"):
        self.id = strategy_id
        self.entries = dict(entries or {})
        self.flatten_from = flatten_from
        self.order_type = order_type
        self.seen = []
        if atomic:
            self.atomic_entries = True

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        n = len(ctx.bars.close.index)
        self.seen.append({
            "n": n, "positions": ctx.positions, "pending": ctx.pending, "book": ctx.book_items(),
        })
        out = []
        for symbol, side, qty in self.entries.get(n, ()):
            limit = 500.0 if self.order_type == "limit" else None
            out.append(order_intent(
                self.id, symbol, side, qty, ctx.bar_ts,
                order_type=self.order_type, limit_price=limit, reason="entrada",
            ))
        if self.flatten_from is not None and n >= self.flatten_from:
            for symbol, _held in ctx.positions:
                qty = ctx.exit_qty(symbol)
                if abs(qty) <= 1e-9:
                    continue
                out.append(order_intent(
                    self.id, symbol, "sell" if qty > 0 else "buy", abs(qty), ctx.bar_ts,
                    reason="flatten",
                ))
        return out


def _run_bars(engine: Engine, bars: IntradayBars) -> None:
    for i, ts in enumerate(bars.close.index):
        engine.step(_ready(ts.to_pydatetime()), _head(bars, i + 1))


def _of(engine: Engine, strategy_id: str):
    return [
        (intent.intent_id, intent.symbol, intent.side, intent.qty)
        for event in engine.journal for intent in event.intents
        if intent.strategy_id == strategy_id
    ]


def _rejected_by(engine: Engine, reason: str):
    return [
        event_.intent_id
        for event in engine.journal for event_ in event.rejections
        if event_.reason == reason
    ]


def test_la_salida_de_una_estrategia_no_cierra_la_posicion_de_otra():
    # Backtest 2025-12-01..12: el flatten de ORB compró los 40 IWM del corto
    # de vwap_reversion. b sale de todo lo que ve en cada barra; a compra.
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]})
    b = _Guion("b", flatten_from=1)
    engine = _engine_multi([a, b], _Apertura())
    _run_bars(engine, _manual(_SEIS))
    assert _of(engine, "b") == []
    assert engine.book.position("SPY") == 10.0
    assert engine.attribution.owned("a") == (("SPY", 10.0),)
    assert all(row["positions"] == () for row in b.seen)
    # El equity de b sigue saliendo del libro entero.
    assert b.seen[-1]["book"] == (("SPY", 10.0),)
    assert a.seen[-1]["positions"] == (("SPY", 10.0),)


def test_una_estrategia_no_ve_las_ordenes_vivas_de_otra():
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]}, order_type="limit")
    b = _Guion("b")
    engine = _engine_multi([a, b], _LimitesConStop())
    _run_bars(engine, _manual(_SEIS[:3]))
    assert a.seen[1]["pending"] == (("SPY", 10.0),)
    assert all(row["pending"] == () for row in b.seen)


def test_el_motor_rechaza_el_simbolo_con_posicion_de_otra():
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]})
    b = _Guion("b", {3: [("SPY", "sell", 4.0), ("QQQ", "buy", 2.0)]})
    engine = _engine_multi([a, b], _Apertura())
    _run_bars(engine, _manual_multi(_SEIS))
    rejected = _rejected_by(engine, OTHER_OWNER)
    assert len(rejected) == 1 and "-SPY-sell-" in rejected[0]
    assert [row[1] for row in _of(engine, "b")] == ["QQQ"]
    assert engine.book.position("SPY") == 10.0
    assert engine.attribution.owned("b") == (("QQQ", 2.0),)


def test_el_motor_rechaza_el_simbolo_con_orden_viva_de_otra():
    # La entrada de a no se llena y queda viva con su stop hijo.
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]}, order_type="limit")
    b = _Guion("b", {3: [("SPY", "buy", 4.0)]})
    execution = _LimitesConStop()
    engine = _engine_multi([a, b], execution)
    _run_bars(engine, _manual(_SEIS))
    assert len(_rejected_by(engine, OTHER_OWNER)) == 1
    assert [intent.strategy_id for intent in execution.sent] == ["a"]


class _RechazaA:
    def size(self, intents, state, now):
        return list(intents)

    def filter(self, intents, state, now):
        approved = [intent for intent in intents if intent.strategy_id != "a"]
        rejected = [(intent, "riesgo") for intent in intents if intent.strategy_id == "a"]
        return approved, rejected


def test_dentro_de_la_barra_el_primer_id_reclama_el_simbolo():
    # Mismo símbolo en la misma barra: gana el primer id. Aunque después el
    # riesgo le rechace la suya, la otra no entra en esa barra.
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]})
    b = _Guion("b", {1: [("SPY", "buy", 5.0)]})
    gate = _RechazaA()
    engine = Engine(
        [b, a], lambda window: None, gate, gate, _Apertura(), FakeCalendar([SESSION]),
        _RelojQueFalla(), book=HftBook(100_000), timeframe="1Min", grace_s=5,
    )
    _run_bars(engine, _manual(_SEIS[:3]))
    first = engine.journal[0]
    assert [(item.reason, item.intent_id.split("-")[0]) for item in first.rejections] == [
        (OTHER_OWNER, "b"), ("riesgo", "a"),
    ]
    assert engine.book.position("SPY") == 0.0


@pytest.mark.parametrize("atomic, sent", [(True, []), (False, ["SPY"])])
def test_entradas_atomicas_rechazan_la_pareja_entera(atomic, sent):
    # pairs_ou no abre media pareja si otra estrategia tiene una pata.
    a = _Guion("a", {1: [("QQQ", "buy", 5.0)]})
    pair = _Guion("pair", {3: [("SPY", "buy", 2.0), ("QQQ", "sell", 2.0)]}, atomic=atomic)
    engine = _engine_multi([a, pair], _Apertura())
    _run_bars(engine, _manual_multi(_SEIS))
    assert [row[1] for row in _of(engine, "pair")] == sent
    assert len(_rejected_by(engine, OTHER_OWNER)) == 2 - len(sent)


def test_el_flatten_de_la_casa_deja_a_todas_en_cero():
    a = _Guion("a", {1: [("SPY", "buy", 10.0)]})
    b = _Guion("b", {1: [("QQQ", "sell", 5.0)]})
    engine = _engine_multi([a, b], _Apertura())
    _run_bars(engine, _manual_multi(_SEIS[:3]))
    stamp = engine.journal[-1].bar_ts
    intents = [
        order_intent("flatten", "SPY", "sell", 10.0, stamp, reason="flatten_time"),
        order_intent("flatten", "QQQ", "buy", 5.0, stamp, reason="flatten_time", seq=1),
    ]
    for intent in intents:
        engine._live[intent.intent_id] = intent
    engine._accept((), [
        Fill(intent.intent_id, stamp, intent.qty, 105.0, 0.0, "taker") for intent in intents
    ])
    assert engine.book.positions() == {}
    assert engine.attribution.owned("a") == ()
    assert engine.attribution.owned("b") == ()


def test_seed_siembra_libro_y_dueno_sin_pasar_por_la_ejecucion():
    # Relanzamiento: a recupera su SPY y lo gestiona. Lo de reconcile no es
    # de nadie: ninguna estrategia lo ve y bloquea entradas.
    a = _Guion("a", {1: [("QQQ", "buy", 1.0)]}, flatten_from=2)
    engine = _engine_multi([a], _Apertura())
    ts = SESSION.open_utc
    mine = order_intent("a", "SPY", "buy", 7.0, ts, reason="reinicio")
    house = order_intent("reconcile", "QQQ", "buy", 3.0, ts, reason="reinicio")
    engine.seed(
        [mine, house],
        [Fill(mine.intent_id, ts, 7.0, 99.0, 0.0, "taker"),
         Fill(house.intent_id, ts, 3.0, 98.0, 0.0, "taker")],
    )
    assert engine.book.positions() == {"QQQ": 3.0, "SPY": 7.0}
    _run_bars(engine, _manual_multi(_SEIS[:4]))
    assert a.seen[0]["positions"] == (("SPY", 7.0),)
    rejected = _rejected_by(engine, OTHER_OWNER)
    assert len(rejected) == 1 and "-QQQ-buy-" in rejected[0]
    assert engine.book.positions() == {"QQQ": 3.0}
    assert engine.attribution.owners("QQQ") == {"reconcile": 3.0}


def test_un_libro_con_posiciones_al_construir_queda_a_nombre_de_reconcile():
    engine = _engine(_CompraUnaVez(), book=_con_posicion("SPY", 10, 1.0))
    assert engine.attribution.owners("SPY") == {"reconcile": 10.0}


def test_agregar_una_estrategia_no_cambia_los_intents_de_las_otras():
    # Fase 6(d) del plan hourly: con la nueva, las demás emiten lo mismo que
    # sin ella. La nueva solo coincide con la otra por la salida de esta.
    def otras():
        return _Guion("otras", {1: [("QQQ", "buy", 5.0)]}, flatten_from=4)

    solo = _engine_multi([otras()], _Apertura())
    _run_bars(solo, _manual_multi(_SEIS))
    nueva = _Guion("nueva", {1: [("SPY", "sell", 8.0)]})
    juntas = _engine_multi([otras(), nueva], _Apertura())
    _run_bars(juntas, _manual_multi(_SEIS))
    assert _of(solo, "otras") and _of(juntas, "otras") == _of(solo, "otras")
    assert _rejected_by(juntas, OTHER_OWNER) == []
    assert juntas.attribution.owned("nueva") == (("SPY", -8.0),)
