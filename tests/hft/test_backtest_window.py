"""Ventana acotada en el backtest: recortar no cambia lo que se decide.

Sin recorte, cada barra copia todo lo visto y un año de 1Min es
cuadrático. El motor entrega las sesiones que se calientan al arrancar
y la sesión en curso (``Engine.history_sessions``). Una estrategia que
lee solo lo que declara en ``required_history`` decide igual con las
dos ventanas: mismos intents, fills y P&L, bit a bit.

ORB no: arma su ATR de Wilder con todos los días previos de la ventana.
Con la corrida entera, el stop de un día dependía de cuándo arrancó el
backtest. Ahora ve los mismos días que el paper.

El impacto del simulador (ADV y σ diarios) sí lee toda la historia: sale
de un resumen por día que da los mismos números que recorrerla entera.
"""

from __future__ import annotations

import math
from datetime import time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.hft import backtest
from src.hft.calendar import FakeCalendar, session_at
from src.hft.config import hft_config_from_dict, load_hft_config
from src.hft.data.bars import IntradayBars
from src.hft.data.synthetic import FakeIntradayProvider, synthetic_bars
from src.hft.engine import Engine
from src.hft.strategies.hourly_momentum import HourlyMomentum
from src.hft.strategies.intraday_momentum import ConsolidatedCloses, IntradayMomentum
from src.hft.strategies.orb import OpeningRangeBreakout
from src.hft.strategies.pairs_ou import PairsOu
from src.hft.strategies.vwap_reversion import VwapReversion

WARMUP = 6
SAMPLE = 30
SYMBOLS = ("SPY", "QQQ", "IWM")
TIMEFRAME = "15Min"
_FRAMES = ("open", "high", "low", "close", "volume", "vwap", "trades")


class _VentanaCompleta(Engine):
    """La ventana de antes del recorte: toda la historia ingerida."""

    def history_sessions(self) -> int:
        return 10**6


@lru_cache(maxsize=1)
def _datos():
    raw = load_hft_config().to_dict()
    raw["universe"] = list(SYMBOLS)
    raw["timeframe"] = TIMEFRAME
    raw["pairs_ou"]["pairs"] = [{
        "leg_a": "SPY", "leg_b": "QQQ", "link": "par sintético del test de la ventana",
    }]
    # Historia corta y señales que caben en barras de 15 minutos: con 30
    # sesiones el recorte actúa desde la segunda y las cinco operan.
    raw["orb"].update(
        opening_minutes=15, rvol_sessions=3, adv_sessions=3, atr_sessions=3, min_atr=0.05,
    )
    raw["intraday_momentum"]["sigma_sessions"] = 3
    raw["vwap_reversion"].update(
        vol_sessions=3, z_n=6, trend_ma=6, trend_look=3, entry_z=1.5,
    )
    raw["pairs_ou"]["formation_sessions"] = 2
    raw["hourly_momentum"].update(sigma_sessions=3, atr_sessions=3)
    raw["costs"]["adv_window"] = 3
    config = hft_config_from_dict(raw)
    days = pd.bdate_range("2024-09-03", periods=WARMUP + SAMPLE)
    sessions = tuple(session_at(day.date(), time(9, 30), time(16, 0)) for day in days)
    bars = synthetic_bars(
        sessions, SYMBOLS, timeframe=TIMEFRAME, seed=7, sigma=0.0012,
        jump_prob=0.002, jump_scale=0.004, gap_prob=0.03, ou_kappa=0.002,
    )
    spy = bars.close["SPY"].dropna()
    closes = spy.groupby(pd.DatetimeIndex(spy.index).tz_convert("America/New_York").date).last()
    closes.index = pd.to_datetime(closes.index)
    return config, sessions, bars, ConsolidatedCloses(close=closes, feed="sip")


def _estrategias(config, sessions, closes, ids):
    built = {
        "orb": lambda: OpeningRangeBreakout.from_config(config, sessions),
        "intraday_momentum": lambda: IntradayMomentum.from_config(config, sessions, closes=closes),
        "vwap_reversion": lambda: VwapReversion.from_config(config, sessions, shortable=config.universe),
        "pairs_ou": lambda: PairsOu.from_config(config, sessions, shortable=config.universe),
        "hourly_momentum": lambda: HourlyMomentum.from_config(config, sessions),
    }
    return [built[name]() for name in ids]


def _corre(monkeypatch, tmp_path, engine_cls, ids, first: int):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    config, sessions, bars, closes = _datos()
    engines = []

    class _Anotado(engine_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            engines.append(self)

    monkeypatch.setattr(backtest, "Engine", _Anotado)
    report = backtest.run(
        config,
        sessions[first].date,
        sessions[-1].date,
        provider=FakeIntradayProvider(bars, timeframe=TIMEFRAME),
        calendar=FakeCalendar(sessions),
        strategies=_estrategias(config, sessions, closes, ids),
        record_trial=False,
        persist=False,
    )
    return report, engines[0]


def _fills(engine):
    return [
        (fill.intent_id, fill.ts, fill.qty, fill.price, fill.fees, fill.liquidity)
        for event in engine.journal
        for fill in event.fills
    ]


def _intents(report):
    return [
        (intent.intent_id, intent.qty, intent.limit_price, intent.stop_price, intent.bracket)
        for intent in report.intents
    ]




def test_el_stop_de_orb_no_depende_del_dia_de_arranque(tmp_path, monkeypatch):
    early, _ = _corre(monkeypatch, tmp_path, Engine, ("orb",), WARMUP)
    late, _ = _corre(monkeypatch, tmp_path, Engine, ("orb",), WARMUP + 8)
    _config, sessions, _bars, _closes = _datos()
    since = sessions[WARMUP + 8].open_utc
    later = {
        intent.intent_id: intent
        for intent in early.intents
        if intent.signal_ts >= since and intent.bracket is not None
    }
    common = [intent for intent in late.intents if intent.intent_id in later]
    assert len(common) >= 3
    for intent in common:
        twin = later[intent.intent_id]
        assert (intent.limit_price, intent.bracket) == (twin.limit_price, twin.bracket)


def _impacto_de_referencia(frames, session_date, config):
    """El cálculo de antes: recorre toda la historia en cada sesión."""
    symbols = [str(symbol) for symbol in config.universe]
    adv = {symbol: backtest._NO_HISTORY_ADV for symbol in symbols}
    sigma = {symbol: 0.0 for symbol in symbols}
    window = max(1, int(config.costs.adv_window))
    close = frames["close"]
    volume = frames["volume"]
    days = pd.DatetimeIndex(close.index).tz_convert("America/New_York").date
    prior = sorted({day for day in days if day < session_date})
    for symbol in close.columns:
        dollar = []
        lasts = []
        for day in prior:
            mask = days == day
            pair = pd.DataFrame({
                "px": close.loc[mask, symbol].to_numpy(dtype=float),
                "vol": volume.loc[mask, symbol].to_numpy(dtype=float),
            }).dropna()
            pair = pair[(pair["px"] > 0) & (pair["vol"] >= 0)]
            if pair.empty:
                continue
            dollar.append(float((pair["px"] * pair["vol"]).sum()))
            lasts.append(float(pair["px"].iloc[-1]))
        used = [value for value in dollar[-window:] if value > 0]
        if used:
            adv[str(symbol)] = float(np.mean(used))
        if len(lasts) >= 3:
            rets = np.diff(np.log(np.asarray(lasts, dtype=float)))
            if rets.size >= 2 and np.isfinite(rets).all():
                std = float(rets.std(ddof=1))
                if math.isfinite(std) and std >= 0.0:
                    sigma[str(symbol)] = std
    return adv, sigma


def _pedazo(bars, start, end, symbols):
    index = bars.close.index
    mask = (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
    return IntradayBars(
        meta=bars.meta,
        **{name: getattr(bars, name).loc[mask, list(symbols)] for name in _FRAMES},
    )


def test_el_impacto_por_dia_da_lo_mismo_que_recorrer_toda_la_historia():
    # ADV y σ diarios salen de un resumen por día, no de toda la historia en
    # cada sesión. Mismos números: IWM aparece a mitad de la corrida, un día
    # entra en dos pedazos y a otro le llega tarde su última barra.
    config, sessions, bars, _closes = _datos()
    engine = backtest._build_engine(
        config, [HourlyMomentum.from_config(config, sessions)], FakeCalendar(sessions),
        np.random.default_rng(0), cost_multiplier=1.0, extra_latency_bars=0,
        end=sessions[-1].date,
    )
    tape = backtest._ImpactTape()
    for k, session in enumerate(sessions):
        if k >= 3:
            backtest._refresh_impact(engine, tape, session.date, config)
            adv, sigma = _impacto_de_referencia(engine._frames, session.date, config)
            assert engine.execution._adv == adv
            assert engine.execution._sigma == sigma
        symbols = SYMBOLS if k >= 12 else SYMBOLS[:2]
        ready = session.close_utc + timedelta(seconds=5)
        if k == 8:
            middle = session.open_utc + timedelta(hours=3)
            engine._ingest(_pedazo(bars, session.open_utc, middle, symbols), ready)
            backtest._refresh_impact(engine, tape, session.date, config)
            engine._ingest(_pedazo(bars, middle, session.close_utc, symbols), ready)
        elif k == 5:
            late = session.close_utc - timedelta(minutes=15)
            engine._ingest(_pedazo(bars, session.open_utc, late, symbols), ready)
            held = (_pedazo(bars, late, session.close_utc, symbols), ready)
        else:
            engine._ingest(_pedazo(bars, session.open_utc, session.close_utc, symbols), ready)
    after = sessions[-1].date + timedelta(days=1)
    backtest._refresh_impact(engine, tape, after, config)
    before_late = dict(engine.execution._sigma)
    engine._ingest(*held)
    backtest._refresh_impact(engine, tape, after, config)
    assert engine.execution._sigma != before_late
    adv, sigma = _impacto_de_referencia(engine._frames, after, config)
    assert engine.execution._adv == adv
    assert engine.execution._sigma == sigma
    assert set(adv) == set(SYMBOLS)
    assert sigma["IWM"] > 0
