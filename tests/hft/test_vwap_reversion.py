"""Reversión a VWAP con filtro de régimen (HFT-09C). Sintético, sin red."""

from __future__ import annotations

import math
from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, bar_close_time, session_at
from src.hft.config import load_hft_config
from src.hft.costs import CostModel
from src.hft.data.bars import bars_from_records
from src.hft.engine import Engine
from src.hft.execution.sim import SimExecution
from src.hft.features import session_vwap, trend_score, vwap_zscore
from src.hft.strategies.base import StrategyContext, validate_strategy
from src.hft.strategies.vwap_reversion import (
    VwapReversion,
    VwapReversionParams,
    realized_vol_percentile,
)
from src.hft.types import Fill

_MINUTES = 120
_SYMBOL = "AAA"
_VOL = 200_000.0


def _params(**overrides) -> VwapReversionParams:
    payload = dict(vol_sessions=5)
    payload.update(overrides)
    return VwapReversionParams(**payload)


def _sessions(n: int, minutes: int = _MINUTES):
    cursor = date(2024, 1, 2)
    found = []
    while len(found) < n:
        if cursor.weekday() < 5:
            total = 30 + minutes
            found.append(session_at(
                cursor, time(9, 30), time(9 + total // 60, total % 60),
            ))
        cursor += timedelta(days=1)
    return tuple(found)


def _reversion_prices(n: int, *, target_z: float, warm: int = 80, hold: int = 8) -> np.ndarray:
    """Seno alrededor de 100 y un choque que deja z en la banda de entrada.

    El seno no llega a |z| = 2. El choque se sostiene ``hold`` barras y
    después el precio vuelve cerca del VWAP. La construcción es la misma
    cuenta que ``vwap_zscore`` cuando high = low = close.
    """
    prices = [100.0 + 0.8 * np.sin(i / 3.0) for i in range(warm)]

    def z_of(path: list) -> tuple:
        arr = np.asarray(path, dtype=float)
        vwap = np.cumsum(arr) / np.arange(1, len(arr) + 1)
        dev = arr - vwap
        sigma = pd.Series(dev).rolling(30, min_periods=30).std(ddof=1).to_numpy()
        return vwap, sigma

    for _ in range(hold):
        vwap, sigma = z_of(prices)
        prices.append(float(vwap[-1] + target_z * sigma[-1]))
    for _ in range(4):
        vwap, sigma = z_of(prices)
        prices.append(float(vwap[-1] + (-0.2 if target_z < 0 else 0.2) * sigma[-1]))
    while len(prices) < n:
        prices.append(prices[-1])
    return np.asarray(prices, dtype=float)


def _trend_prices(n: int, slope: float, amp: float = 0.0, phase: float = 0.0) -> np.ndarray:
    i = np.arange(n, dtype=float)
    return 100.0 + slope * i + amp * np.sin(i / 2.0 + phase)


def _bars(sessions, series):
    rows = []
    names = [_SYMBOL]
    for session, prices in zip(sessions, series):
        for i, price in enumerate(prices):
            ts = session.open_utc + timedelta(minutes=i)
            px = float(price)
            rows.append({
                "symbol": _SYMBOL,
                "ts": ts,
                "o": px,
                "h": px,
                "l": px,
                "c": px,
                "v": _VOL,
                "vw": px,
                "n": 10,
            })
    return bars_from_records(
        pd.DataFrame(rows), names, source="test", feed="iex", timeframe="1Min",
    )


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


class _Pasa:
    def size(self, intents, state, now):
        return list(intents)

    def filter(self, intents, state, now):
        return tuple(intents), ()


class _Reloj:
    def now(self):
        raise AssertionError("el motor leyó el reloj")


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


def _strategy(sessions, **overrides):
    shortable = overrides.pop("shortable", ())
    flatten = overrides.pop("flatten_minutes", 5.0)
    return VwapReversion(
        _params(**overrides),
        sessions=sessions,
        flatten_minutes=flatten,
        shortable=shortable,
    )


def _context(bars, bar_ts, positions=()):
    return StrategyContext(
        now=bar_close_time(bar_ts, "1Min") + timedelta(seconds=5),
        bar_ts=bar_ts,
        bars=bars,
        features=None,
        positions=positions,
        cash=100_000.0,
        realized_pnl=0.0,
    )


def _entries(engine):
    return [
        intent
        for event in engine.journal
        for intent in event.intents
        if intent.order_type in ("marketable_limit", "limit")
    ]


def test_los_defaults_son_los_del_prompt_y_del_toml():
    cfg = load_hft_config()
    strategy = VwapReversion.from_config(cfg)
    assert strategy.params == VwapReversionParams()
    assert strategy.params.z_n == 30
    assert strategy.params.entry_z == 2.0
    assert strategy.params.exit_z == 0.5
    assert strategy.params.stop_z == 3.5
    assert strategy.params.skip_open_minutes == 15
    assert strategy.params.entry_cutoff_minutes == 15.0
    assert strategy.params.time_stop_minutes == 60
    assert strategy.params.vol_sessions == 20
    assert strategy.params.max_vol_percentile == 70.0
    assert strategy.params.trend_threshold == 0.5
    assert strategy.params.entry_style == "marketable_limit"
    assert strategy.params.risk_frac == cfg.sizing.risk_frac
    assert strategy.sizing.risk_frac == strategy.params.risk_frac
    assert strategy.flatten_minutes == cfg.risk.flatten_minutes
    assert validate_strategy(strategy) == ()
    assert strategy.theory.research_only is False
    text = " ".join((
        strategy.theory.hypothesis,
        strategy.theory.mechanism,
        *strategy.theory.references,
        *strategy.theory.failure_regimes,
        *strategy.theory.kill_criteria,
    ))
    assert "Grossman" in text and "Miller" in text
    assert "Nagel" in text
    assert "quién" in strategy.theory.mechanism or "Paga quien" in strategy.theory.mechanism
    joined = " ".join(strategy.theory.kill_criteria)
    assert "60 sesiones" in joined
    for param in strategy.theory.params:
        assert param.justification.strip()
        assert param.source.strip()
        assert strategy.theory.param_map()[param.name] == getattr(strategy.params, param.name)


def test_ou_alrededor_del_vwap_deja_pnl_positivo():
    sessions = _sessions(6)
    priors = [
        _trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i)
        for i in range(5)
    ]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    engine = _engine(_strategy(sessions), bars, sessions)
    _drive(engine, bars)
    assert engine.book.position(_SYMBOL) == 0.0
    entries = _entries(engine)
    assert entries
    assert all(intent.side == "buy" for intent in entries)
    assert all(intent.order_type == "marketable_limit" for intent in entries)
    assert all(intent.signal_ts >= sessions[-1].open_utc for intent in entries)
    assert engine.book.realized_pnl > 0


def test_la_tendencia_bloquea_entradas_y_el_pnl_no_se_derrumba():
    """Hoy es tendencia, con vol por debajo de los días previos.

    z pasa de +2, así que sin el filtro de pendiente se abriría un corto.
    El percentil de vol no es el que bloquea.
    """
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=3.0, amp=6.0, phase=i) for i in range(5)]
    today = _trend_prices(_MINUTES, slope=2.0)
    bars = _bars(sessions, priors + [today])
    strategy = _strategy(sessions, shortable=(_SYMBOL,))
    stamp = sessions[-1].open_utc + timedelta(minutes=80)
    z = vwap_zscore(
        bars.close, session_vwap(
            bars.high, bars.low, bars.close, bars.volume, sessions=sessions,
        ),
        n=30, sessions=sessions,
    )
    slope = trend_score(
        bars.close, ma_window=50, look=21, by_session=True, sessions=sessions,
    )
    ts = pd.Timestamp(stamp)
    assert float(z.at[ts, _SYMBOL]) > 2.0
    assert abs(float(slope.at[ts, _SYMBOL])) >= 0.5
    percentile = realized_vol_percentile(
        bars.close, sessions, _SYMBOL, stamp, n_prior=5,
    )
    assert percentile < 70.0
    assert strategy.on_bar(_context(bars, stamp)) == ()
    engine = _engine(strategy, bars, sessions)
    _drive(engine, bars)
    assert _entries(engine) == []
    assert engine.book.position(_SYMBOL) == 0.0
    assert engine.book.realized_pnl == 0.0


def test_z_con_historia_insuficiente_es_nan_y_no_hay_entrada():
    """La pendiente y el percentil sí alcanzan. El z no: no se entra."""
    minutes = 20
    sessions = _sessions(2, minutes=60)
    rng = np.random.default_rng(0)
    prior = 100.0 + rng.normal(0.0, 1.0, minutes)
    today = np.full(minutes, 100.0)
    bars = _bars(sessions, [prior, today])
    # El builder usa la sesión de 60 min pero solo escribimos 20 barras:
    # recortar el índice al tramo que sí tiene filas.
    strategy = _strategy(
        sessions,
        z_n=30,
        trend_ma=5,
        trend_look=3,
        vol_sessions=1,
        skip_open_minutes=0,
        entry_cutoff_minutes=15.0,
    )
    stamp = sessions[-1].open_utc + timedelta(minutes=minutes - 1)
    ts = pd.Timestamp(stamp)
    vwap = session_vwap(bars.high, bars.low, bars.close, bars.volume, sessions=sessions)
    z = vwap_zscore(bars.close, vwap, n=30, sessions=sessions)
    slope = trend_score(bars.close, 5, 3, by_session=True, sessions=sessions)
    assert pd.isna(z.at[ts, _SYMBOL])
    assert abs(float(slope.at[ts, _SYMBOL])) < 0.5
    percentile = realized_vol_percentile(
        bars.close, sessions, _SYMBOL, stamp, n_prior=1,
    )
    assert percentile < 70.0
    assert strategy.on_bar(_context(bars, stamp)) == ()


def test_el_corto_solo_sale_si_el_simbolo_es_shortable():
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=2.4)
    bars = _bars(sessions, priors + [today])
    stamp = sessions[-1].open_utc + timedelta(minutes=80)
    blocked = _strategy(sessions)
    opened = _strategy(sessions, shortable=(_SYMBOL,))
    assert blocked.on_bar(_context(bars, stamp)) == ()
    order = opened.on_bar(_context(bars, stamp))
    assert len(order) == 1
    assert order[0].side == "sell"
    assert order[0].order_type == "marketable_limit"


def test_la_variante_pasiva_es_un_experimento():
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    stamp = sessions[-1].open_utc + timedelta(minutes=80)
    strategy = _strategy(sessions, entry_style="passive_limit")
    assert strategy.theory.research_only is True
    assert "experimento" in strategy.theory.hypothesis
    order = strategy.on_bar(_context(bars, stamp))
    assert len(order) == 1
    assert order[0].order_type == "limit"
    assert "experimento_pasivo" in order[0].reason


def test_el_percentil_de_vol_no_mira_el_futuro():
    sessions = _sessions(6)
    series = [_trend_prices(_MINUTES, slope=1.5, amp=2.0, phase=i) for i in range(6)]
    bars = _bars(sessions, series)
    stamp = sessions[-1].open_utc + timedelta(minutes=80)
    full = realized_vol_percentile(
        bars.close, sessions, _SYMBOL, stamp, n_prior=5,
    )
    cut = bars.close.loc[:pd.Timestamp(stamp)]
    trimmed = realized_vol_percentile(
        cut, sessions, _SYMBOL, stamp, n_prior=5,
    )
    assert full == pytest.approx(trimmed)
    assert math.isfinite(full)


def test_no_hay_entrada_antes_de_open_mas_15():
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    stamp = sessions[-1].open_utc + timedelta(minutes=80)
    strategy = _strategy(sessions, skip_open_minutes=100)
    assert strategy.on_bar(_context(bars, stamp)) == ()
    allowed = _strategy(sessions)
    assert allowed.on_bar(_context(bars, stamp))


class _EntradaLenta:
    """La primera orden queda abierta ``hold`` barras, como XLE el 2026-10-12.

    Paper 2026-10-12: la compra de XLE de las 14:39 UTC salió en el paso
    de las 14:41 y no se llenó hasta las 14:43:12. El resto se llena en
    el ``submit`` que la recibe, al límite o al último cierre.
    """

    def __init__(self, hold: int = 2) -> None:
        self.hold = hold
        self.sent: list = []
        self.open_during: list = []
        self._first = None
        self._age = 0

    def submit(self, intents, *, now, bars):
        fills = []
        if self._first is not None:
            self._age += 1
            if self._age >= self.hold:
                fills.append(self._fill(self._first, now, bars))
                self._first = None
        for intent in intents:
            self.sent.append(intent)
            if not self.sent[:-1]:
                self._first = intent
                continue
            fills.append(self._fill(intent, now, bars))
        if self._first is not None:
            self.open_during.append(pd.Timestamp(bars.close.index[-1]))
        return tuple(fills)

    def open_orders(self):
        return () if self._first is None else (self._first,)

    def _fill(self, intent, now, bars):
        price = intent.limit_price
        if price is None:
            price = float(bars.close[intent.symbol].dropna().iloc[-1])
        return Fill(intent.intent_id, now, intent.qty, float(price), 0.0, "taker")


def test_no_reentra_mientras_su_entrada_sigue_abierta():
    # HFT-PAPER-15: con la compra abierta el libro seguía plano y la
    # estrategia mandó otra. La cuenta quedó con el doble (304 XLE).
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    execution = _EntradaLenta(hold=2)
    engine = _engine(_strategy(sessions), bars, sessions)
    engine.execution = execution
    _drive(engine, bars)
    first = execution.sent[0]
    assert first.side == "buy" and first.order_type == "marketable_limit"
    assert len(execution.open_during) == 2
    while_open = [
        intent
        for intent in _entries(engine)
        if intent.intent_id != first.intent_id
        and pd.Timestamp(intent.signal_ts) in execution.open_during
    ]
    assert while_open == []
    assert engine.book.position(_SYMBOL) == 0.0


class _SalidaLenta:
    """Llena las entradas al recibirlas. Cada salida queda abierta ``hold`` barras.

    Es ``_EntradaLenta`` del otro lado: una salida es la orden que reduce
    lo que este adaptador ya llenó. La que llega mientras otra sigue
    abierta también espera, como en el broker (HFT-PAPER-16).
    """

    def __init__(self, hold: int = 2) -> None:
        self.hold = hold
        self.sent: list = []
        self.open_during: list = []
        self._held: dict = {}
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
        if self._held:
            self.open_during.append(pd.Timestamp(bars.close.index[-1]))
        return tuple(fills)

    def open_orders(self):
        return tuple(slot[0] for slot in self._held.values())

    def _fill(self, intent, now, bars):
        price = intent.limit_price
        if price is None:
            price = float(bars.close[intent.symbol].dropna().iloc[-1])
        signed = intent.qty if intent.side == "buy" else -intent.qty
        self._filled[intent.symbol] = self._filled.get(intent.symbol, 0.0) + signed
        return Fill(intent.intent_id, now, intent.qty, float(price), 0.0, "taker")


def _exits_sent(execution):
    return [intent for intent in execution.sent if intent.order_type == "market"]


def _assert_one_exit_per_entry(engine, execution):
    proposed = [
        intent
        for event in engine.journal
        for intent in event.intents
        if intent.order_type == "market"
    ]
    while_open = [
        intent
        for intent in proposed
        if pd.Timestamp(intent.signal_ts) in execution.open_during
    ]
    assert while_open == []
    bought = sum(intent.qty for intent in execution.sent if intent.side == "buy")
    sold = sum(intent.qty for intent in execution.sent if intent.side == "sell")
    assert sold == pytest.approx(bought)
    held = [event.state["positions"].get(_SYMBOL, 0.0) for event in engine.journal]
    assert min(held) >= 0.0
    assert engine.book.position(_SYMBOL) == 0.0


def test_no_repite_la_salida_mientras_sigue_abierta():
    # HFT-PAPER-16: con la venta abierta el libro seguía largo y cada barra
    # mandaba otra venta del tamaño entero. Al llenarse todas, quedaba corto.
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    execution = _SalidaLenta(hold=2)
    engine = _engine(_strategy(sessions), bars, sessions)
    engine.execution = execution
    _drive(engine, bars)
    exits = _exits_sent(execution)
    assert exits and all("exit_z" in intent.reason for intent in exits)
    assert len(execution.open_during) >= 2
    _assert_one_exit_per_entry(engine, execution)


def test_no_repite_el_flatten_mientras_sigue_abierto():
    sessions = _sessions(6)
    priors = [_trend_prices(_MINUTES, slope=2.0, amp=4.0, phase=i) for i in range(5)]
    today = _reversion_prices(_MINUTES, target_z=-2.4)
    bars = _bars(sessions, priors + [today])
    execution = _SalidaLenta(hold=2)
    # Sin exit_z, stop_z ni time_stop al alcance: la posición llega al flatten.
    strategy = _strategy(sessions, exit_z=0.001, stop_z=50.0, time_stop_minutes=10_000)
    engine = _engine(strategy, bars, sessions)
    engine.execution = execution
    _drive(engine, bars)
    exits = _exits_sent(execution)
    assert [intent.reason for intent in exits] == ["vwap flatten_time"]
    _assert_one_exit_per_entry(engine, execution)
