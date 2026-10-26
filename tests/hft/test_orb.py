"""Opening Range Breakout (HFT-09A). Sintético, sin red."""

from __future__ import annotations

from datetime import date, time, timedelta

import pandas as pd
import pytest

from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, bar_close_time, session_at
from src.hft.config import load_hft_config
from src.hft.costs import CostModel
from src.hft.data.bars import bars_from_records
from src.hft.engine import Engine
from src.hft.execution.sim import SimExecution
from src.hft.strategies.base import StrategyContext, validate_strategy
from src.hft.strategies.orb import (
    OpeningRangeBreakout,
    OrbParams,
    adjust_daily_for_splits,
)
from src.hft.types import Fill, order_intent

# Tres sesiones de historia y la de hoy. El lookback del test es 3,
# no las 14 del paper: el motor no tiene que recorrer dos semanas.
_LOOK = 3
_PRIOR_BARS = 30
_TODAY_BARS = 26
_VOL = 200_000.0


def _params(**overrides) -> OrbParams:
    payload = dict(
        opening_minutes=5,
        rvol_sessions=_LOOK,
        min_rvol=1.0,
        top_k=20,
        min_price=5.0,
        min_adv_shares=1_000,
        adv_sessions=_LOOK,
        min_atr=0.5,
        atr_sessions=_LOOK,
        stop_atr_frac=0.1,
        risk_frac=0.01,
        doji_eps=0.0,
        amihud_percentile=100.0,
        limit_through_ticks=1,
    )
    payload.update(overrides)
    return OrbParams(**payload)


def _sessions(n: int = 4):
    cursor = date(2024, 1, 2)
    found = []
    while len(found) < n:
        if cursor.weekday() < 5:
            found.append(session_at(cursor, time(9, 30), time(10, 0)))
        cursor += timedelta(days=1)
    return tuple(found)


def _doji(_i: int):
    return (100.0, 101.0, 99.0, 100.0, _VOL)


def _momentum(i: int):
    open_ = 100.0 + 0.5 * i
    close = open_ + 0.5
    return (open_, close, open_ - 0.01, close, _VOL)


def _sin_continuacion(i: int):
    if i < 5:
        open_ = 100.0 + 0.25 * i
        close = open_ + 0.25
        return (open_, close, open_ - 0.01, close, _VOL)
    price = 101.25
    return (price, price, price - 0.01, price, _VOL)


def _bars(sessions, today, symbols, today_bars: int = _TODAY_BARS):
    """``today(i) -> {símbolo: (o, h, l, c, v)}`` solo en la última sesión."""
    rows = []
    last = sessions[-1]
    names = list(symbols)
    for session in sessions:
        n = today_bars if session is last else _PRIOR_BARS
        for i in range(n):
            ts = session.open_utc + timedelta(minutes=i)
            if session is last:
                per_symbol = today(i)
            else:
                per_symbol = {name: _doji(i) for name in names}
            for symbol in names:
                open_, high, low, close, volume = per_symbol[symbol]
                rows.append({
                    "symbol": symbol,
                    "ts": ts,
                    "o": open_,
                    "h": high,
                    "l": low,
                    "c": close,
                    "v": volume,
                    "vw": (high + low + close) / 3.0,
                    "n": 10,
                })
    return bars_from_records(
        pd.DataFrame(rows), names, source="test", feed="iex", timeframe="1Min",
    )


def _one(path):
    def today(i):
        return {"AAA": path(i)}
    return today


class _Pasa:
    def size(self, intents, state, now):
        return list(intents)

    def filter(self, intents, state, now):
        return tuple(intents), ()


class _Reloj:
    def now(self):
        raise AssertionError("el motor leyó el reloj")


def _costs() -> CostModel:
    return CostModel(
        spread_floor_bps=0.0,
        tick=0.01,
        impact_y=0.0,
        adv_window=14,
        commission_per_share=0.0,
        sec_section31_rate=0.0000206,
        sec_section31_verified_on="2026-04-04",
        finra_taf_per_share=0.000195,
        finra_taf_cap=9.79,
        finra_taf_verified_on="2026-01-01",
        crypto_maker_bps=15.0,
        crypto_taker_bps=25.0,
        crypto_fees_verified_on="2026-01-01",
        use_spread_estimators=False,
    )


def _engine(strategy, bars, sessions):
    symbols = list(bars.close.columns)
    execution = SimExecution(
        _costs(),
        partial_prob=0.0,
        adv={symbol: 1.0e12 for symbol in symbols},
        sigma={symbol: 0.0 for symbol in symbols},
        timeframe="1Min",
        seed=0,
    )
    return Engine(
        [strategy],
        lambda window: None,
        _Pasa(),
        _Pasa(),
        execution,
        FakeCalendar(sessions),
        _Reloj(),
        book=HftBook(100_000.0),
        timeframe="1Min",
        grace_s=5.0,
        latency_bars=1,
    )


def _head(bars, n: int):
    frames = {
        name: getattr(bars, name).iloc[:n]
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    return type(bars)(meta=bars.meta, **frames)


def _drive(engine, bars):
    for i, ts in enumerate(bars.close.index):
        moment = bar_close_time(ts.to_pydatetime(), "1Min") + timedelta(seconds=5)
        engine.step(moment, _head(bars, i + 1))


def _strategy(sessions, splits=None, **overrides):
    return OpeningRangeBreakout(
        _params(**overrides), sessions=sessions, flatten_minutes=5.0, splits=splits,
    )


def _context(bars, bar_ts):
    return StrategyContext(
        now=bar_close_time(bar_ts, "1Min") + timedelta(seconds=5),
        bar_ts=bar_ts,
        bars=bars,
        features=None,
        positions=(),
        cash=100_000.0,
        realized_pnl=0.0,
    )


def _bar_at(session, minute: int):
    return session.open_utc + timedelta(minutes=minute)


def test_los_defaults_son_los_del_paper_y_del_toml():
    """Secciones 2.1, 3 y 4. El TOML versionado no se despega del dataclass."""
    cfg = load_hft_config()
    assert cfg.schema_version == 1
    strategy = OpeningRangeBreakout.from_config(cfg)
    assert strategy.params == OrbParams()
    assert strategy.params.opening_minutes == 5
    assert strategy.params.rvol_sessions == 14
    assert strategy.params.min_rvol == 1.0
    assert strategy.params.top_k == 20
    assert strategy.params.min_price == 5.0
    assert strategy.params.min_adv_shares == 1_000_000
    assert strategy.params.min_atr == 0.5
    assert strategy.params.stop_atr_frac == 0.1
    assert strategy.params.risk_frac == 0.01
    assert strategy.params.doji_eps == 0.0
    assert strategy.sizing.max_leverage == 1.0
    assert strategy.sizing.risk_frac == strategy.params.risk_frac
    assert validate_strategy(strategy) == ()
    assert strategy.theory.research_only is False
    text = " ".join(
        (
            strategy.theory.hypothesis,
            strategy.theory.mechanism,
            *strategy.theory.references,
            *strategy.theory.failure_regimes,
            *strategy.theory.kill_criteria,
        )
    )
    assert "4729284" in text
    assert "marketable_limit" in strategy.theory.hypothesis
    assert "2.1" in strategy.theory.hypothesis
    assert "Crabel" in text
    joined = " ".join(strategy.theory.kill_criteria)
    assert "60 sesiones" in joined
    assert "2σ" in joined
    for param in strategy.theory.params:
        assert param.justification.strip()
        assert param.source.strip()
        assert strategy.theory.param_map()[param.name] == getattr(strategy.params, param.name)


def test_con_momentum_de_apertura_el_pnl_es_positivo():
    sessions = _sessions()
    bars = _bars(sessions, _one(_momentum), ["AAA"])
    engine = _engine(_strategy(sessions), bars, sessions)
    _drive(engine, bars)
    assert engine.book.position("AAA") == 0.0
    assert engine.book.realized_pnl > 0
    entries = [
        intent
        for event in engine.journal
        for intent in event.intents
        if intent.order_type == "marketable_limit"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.side == "buy"
    assert entry.qty == int(entry.qty) and entry.qty >= 1
    assert entry.bracket is not None and entry.bracket.take is None
    assert entry.bracket.stop < entry.limit_price
    assert entry.signal_ts == _bar_at(sessions[-1], 4)
    filled = [
        fill
        for event in engine.journal
        for fill in event.fills
        if fill.intent_id == entry.intent_id
    ]
    assert len(filled) == 1
    assert filled[0].ts == _bar_at(sessions[-1], 5)


def test_sin_continuacion_el_pnl_es_el_costo():
    sessions = _sessions()
    bars = _bars(sessions, _one(_sin_continuacion), ["AAA"])
    engine = _engine(_strategy(sessions), bars, sessions)
    _drive(engine, bars)
    assert engine.book.position("AAA") == 0.0
    realized = engine.book.realized_pnl
    assert realized < 0
    buys = [
        fill
        for event in engine.journal
        for fill in event.fills
        if fill.intent_id and not str(fill.intent_id).endswith(":stop")
    ]
    entry_fills = [
        fill for fill in buys
        if any(
            intent.intent_id == fill.intent_id and intent.order_type == "marketable_limit"
            for event in engine.journal
            for intent in event.intents
        )
    ]
    assert len(entry_fills) == 1
    qty = entry_fills[0].qty
    loss_per_share = -realized / qty
    # Compra al open + 1 tick y venta al open − 1 tick, con el precio quieto.
    # Un stop de 0.10×ATR (~0.20) sería otra magnitud.
    assert 0.015 < loss_per_share < 0.05


def test_no_hay_entrada_antes_de_cerrar_el_rango():
    sessions = _sessions()
    bars = _bars(sessions, _one(_momentum), ["AAA"])
    strategy = _strategy(sessions)
    today = sessions[-1]
    for minute in range(4):
        assert strategy.on_bar(_context(bars, _bar_at(today, minute))) == ()
    # La barra de las 09:35 ya está en el frame: no puede adelantar la decisión.
    entrada = strategy.on_bar(_context(bars, _bar_at(today, 4)))
    assert len(entrada) == 1
    assert entrada[0].signal_ts == _bar_at(today, 4)
    engine = _engine(strategy, bars, sessions)
    _drive(engine, bars)
    range_ts = _bar_at(today, 4)
    for event in engine.journal:
        if event.bar_ts is None or event.bar_ts < range_ts:
            assert event.intents == ()


def test_un_split_historico_no_mueve_el_rango_ni_el_stop_ajustado():
    """El split es de un día ya cerrado. El rango de hoy sale de raw.

    El paper midió con IQFeed sin ajustar. Aquí ATR, ADV y el volumen
    histórico del RVOL se ajustan; el open/close de hoy no.
    """
    sessions = _sessions()
    split_day = sessions[2].date
    clean = _bars(sessions, _one(_momentum), ["AAA"])
    raw_rows = []
    last = sessions[-1]
    for session in sessions:
        n = _TODAY_BARS if session is last else _PRIOR_BARS
        scale = 2.0 if session.date < split_day else 1.0
        for i in range(n):
            ts = session.open_utc + timedelta(minutes=i)
            if session is last:
                open_, high, low, close, volume = _momentum(i)
            else:
                open_, high, low, close, volume = _doji(i)
            raw_rows.append({
                "symbol": "AAA",
                "ts": ts,
                "o": open_ * scale,
                "h": high * scale,
                "l": low * scale,
                "c": close * scale,
                "v": volume / scale,
                "vw": (high + low + close) / 3.0 * scale,
                "n": 10,
            })
    raw = bars_from_records(
        pd.DataFrame(raw_rows), ["AAA"], source="test", feed="iex", timeframe="1Min",
    )
    bar_ts = _bar_at(last, 4)
    ajustada = _strategy(sessions, splits={split_day: 2.0})
    cruda = _strategy(sessions)
    limpia = _strategy(sessions)
    orden_ajustada = ajustada.on_bar(_context(raw, bar_ts))
    orden_limpia = limpia.on_bar(_context(clean, bar_ts))
    orden_cruda = cruda.on_bar(_context(raw, bar_ts))
    assert len(orden_ajustada) == len(orden_limpia) == len(orden_cruda) == 1
    aj, li, cr = orden_ajustada[0], orden_limpia[0], orden_cruda[0]
    assert aj.limit_price == pytest.approx(li.limit_price)
    assert aj.bracket.stop == pytest.approx(li.bracket.stop)
    assert aj.limit_price == pytest.approx(cr.limit_price)
    assert cr.bracket.stop != pytest.approx(aj.bracket.stop)
    # El día del split no se toca; los anteriores sí, y se componen.
    daily = pd.DataFrame(
        {"AAA": [200.0, 200.0, 100.0]},
        index=pd.to_datetime([sessions[0].date, sessions[1].date, sessions[2].date]),
    )
    volume = pd.DataFrame({"AAA": [100.0, 100.0, 200.0]}, index=daily.index)
    out_o, _, _, out_c, out_v = adjust_daily_for_splits(
        daily, daily, daily, daily, volume, splits={split_day: 2.0},
    )
    assert list(out_c["AAA"]) == pytest.approx([100.0, 100.0, 100.0])
    assert list(out_o["AAA"]) == pytest.approx([100.0, 100.0, 100.0])
    assert list(out_v["AAA"]) == pytest.approx([200.0, 200.0, 200.0])


def test_doji_no_opera():
    sessions = _sessions()
    bars = _bars(sessions, _one(_doji), ["AAA"])
    assert _strategy(sessions).on_bar(_context(bars, _bar_at(sessions[-1], 4))) == ()


def test_rvol_bajo_no_opera():
    sessions = _sessions()

    def quiet(i):
        open_, high, low, close, _volume = _momentum(i)
        return (open_, high, low, close, 1_000.0)

    bars = _bars(sessions, _one(quiet), ["AAA"])
    assert _strategy(sessions).on_bar(_context(bars, _bar_at(sessions[-1], 4))) == ()


def test_top_k_se_queda_con_el_mayor_rvol():
    sessions = _sessions()

    def today(i):
        base = _momentum(i)
        loud = (base[0], base[1], base[2], base[3], _VOL * 3)
        return {"AAA": loud, "BBB": base}

    bars = _bars(sessions, today, ["AAA", "BBB"])
    orders = _strategy(sessions, top_k=1).on_bar(_context(bars, _bar_at(sessions[-1], 4)))
    assert [item.symbol for item in orders] == ["AAA"]


def test_amihud_excluye_al_iliquido():
    sessions = _sessions()
    last = sessions[-1]
    rows = []
    closes = {"AAA": 100.0, "BBB": 100.0}
    for session in sessions:
        if session is not last:
            closes["BBB"] = {0: 100.0, 1: 40.0, 2: 100.0}[sessions.index(session)]
        n = _TODAY_BARS if session is last else _PRIOR_BARS
        for i in range(n):
            ts = session.open_utc + timedelta(minutes=i)
            for symbol, volume in (("AAA", _VOL), ("BBB", 1_000.0)):
                if session is last:
                    open_, high, low, close, _vol = _momentum(i)
                elif symbol == "AAA":
                    open_, high, low, close, _vol = _doji(i)
                else:
                    price = closes["BBB"]
                    open_ = close = price
                    high = price + 1.0
                    low = price - 1.0
                rows.append({
                    "symbol": symbol,
                    "ts": ts,
                    "o": open_,
                    "h": high,
                    "l": low,
                    "c": close,
                    "v": volume,
                    "vw": close,
                    "n": 10,
                })
    bars = bars_from_records(
        pd.DataFrame(rows), ["AAA", "BBB"], source="test", feed="iex", timeframe="1Min",
    )
    orders = _strategy(
        sessions, amihud_percentile=90.0, min_adv_shares=1, min_atr=0.01,
    ).on_bar(_context(bars, _bar_at(last, 4)))
    assert [item.symbol for item in orders] == ["AAA"]


class _SalidaLentaConStop:
    """Llena la entrada al recibirla y cuelga su stop de bracket, que no dispara.

    Cada salida (la orden que reduce lo ya llenado) queda abierta ``hold``
    barras. El stop hijo sigue vivo todo el tiempo: protege la posición,
    no la cierra, y no puede contar como salida en curso (HFT-PAPER-16).
    """

    def __init__(self, hold: int = 2) -> None:
        self.hold = hold
        self.sent: list = []
        self.open_during: list = []
        self._held: dict = {}
        self._children: dict = {}
        self._spawned: list = []
        self._filled: dict = {}

    def submit(self, intents, *, now, bars):
        fills = []
        for intent_id, slot in list(self._held.items()):
            slot[1] += 1
            if slot[1] >= self.hold:
                fills.append(self._fill(slot[0], now, bars))
                del self._held[intent_id]
        for intent in intents:
            self.sent.append(intent)
            held = self._filled.get(intent.symbol, 0.0)
            if (held > 0 and intent.side == "sell") or (held < 0 and intent.side == "buy"):
                self._held[intent.intent_id] = [intent, 0]
                continue
            fills.append(self._fill(intent, now, bars))
            if intent.bracket is not None:
                stop = order_intent(
                    intent.strategy_id, intent.symbol,
                    "sell" if intent.side == "buy" else "buy", intent.qty, intent.signal_ts,
                    order_type="stop", stop_price=intent.bracket.stop,
                    intent_id=f"{intent.intent_id}-stop",
                )
                self._children[stop.intent_id] = stop
                self._spawned.append(stop)
        if self._held:
            self.open_during.append(pd.Timestamp(bars.close.index[-1]))
        return tuple(fills)

    def open_orders(self):
        return tuple(slot[0] for slot in self._held.values()) + tuple(self._children.values())

    def take_spawned(self):
        out = tuple(self._spawned)
        self._spawned = []
        return out

    def _fill(self, intent, now, bars):
        price = intent.limit_price
        if price is None:
            price = float(bars.close[intent.symbol].dropna().iloc[-1])
        signed = intent.qty if intent.side == "buy" else -intent.qty
        self._filled[intent.symbol] = self._filled.get(intent.symbol, 0.0) + signed
        return Fill(intent.intent_id, now, intent.qty, float(price), 0.0, "taker")


def test_el_flatten_sale_una_vez_aunque_siga_abierto_y_el_stop_no_cuenta():
    sessions = _sessions()
    bars = _bars(sessions, _one(_momentum), ["AAA"], today_bars=30)
    execution = _SalidaLentaConStop(hold=2)
    engine = _engine(_strategy(sessions), bars, sessions)
    engine.execution = execution
    _drive(engine, bars)
    entry = next(intent for intent in execution.sent if intent.bracket is not None)
    flattens = [
        intent
        for event in engine.journal
        for intent in event.intents
        if intent.reason == "orb flatten_time"
    ]
    # Una sola, del tamaño entero: el stop vivo no la tapa y la venta
    # abierta no se repite en las barras que tarda en llenarse.
    assert [(intent.side, intent.qty) for intent in flattens] == [("sell", entry.qty)]
    assert len(execution.open_during) == 2
    held = [event.state["positions"].get("AAA", 0.0) for event in engine.journal]
    assert min(held) >= 0.0
    assert engine.book.position("AAA") == 0.0
