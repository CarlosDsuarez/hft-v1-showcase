"""Paridad del backtest con el paper (Fase 4, 2026-10-13).

El paper pide ``config_symbols`` (universo, momentum, A–S y patas de
los pares), le da cierres SIP diarios al momentum y usa el calendario
de Alpaca, con medias sesiones. El backtest tiene que ver lo mismo.
Todo es sintético y sin red.
"""

from __future__ import annotations

import math
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from src.hft.backtest import run
from src.hft.calendar import NY, FakeCalendar, session_at
from src.hft.config import hft_config_from_dict, load_hft_config
from src.hft.data.bars import bars_from_records
from src.hft.data.synthetic import FakeIntradayProvider
from src.hft.session import config_symbols
from src.hft.types import order_intent
from src.settings import load_settings
from tests.hft.test_backtest import _Qty, _theory

DAY = date(2025, 1, 6)
SESSION = session_at(DAY, time(9, 30), time(16, 0))
PRICE = 200.0


def _config(**sections):
    raw = load_hft_config().to_dict()
    raw["sim"]["partial_prob"] = 0.0
    for name, values in sections.items():
        raw[name].update(values)
    return hft_config_from_dict(raw)


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    return load_settings()


def _flat_bars(sessions, symbols, n=None, price=PRICE):
    rows = []
    for session in sessions:
        total = int((session.close_utc - session.open_utc).total_seconds() // 60)
        for i in range(total if n is None else n):
            ts = session.open_utc + timedelta(minutes=i)
            px = price + (0.01 if i % 2 else 0.0)
            for symbol in symbols:
                rows.append({
                    "symbol": symbol, "ts": ts, "o": px, "h": px + 0.02,
                    "l": px - 0.02, "c": px, "v": 50_000, "vw": px, "n": 20,
                })
    return bars_from_records(
        pd.DataFrame(rows), list(symbols), source="test", feed="iex", timeframe="1Min",
    )


class _IdaYVuelta:
    """Compra ``symbol`` en la segunda barra y vende en la sexta."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.id = f"idayvuelta_{symbol.lower()}"
        self.params = _Qty(1.0)
        self.theory = _theory(1.0)

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        n = len(ctx.bars.close.index)
        if n == 2:
            return [order_intent(
                self.id, self.symbol, "buy", 1.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE + 5.0,
                reason="entrada de prueba",
            )]
        if n == 6:
            return [order_intent(
                self.id, self.symbol, "sell", 1.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE - 5.0,
                reason="salida de prueba",
            )]
        return []


def test_el_backtest_pide_los_mismos_simbolos_que_el_paper(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    cfg = _config()
    provider = FakeIntradayProvider(_flat_bars([SESSION], ("SPY",), n=10))
    run(
        cfg, DAY, DAY,
        provider=provider,
        strategies=[_IdaYVuelta("SPY")],
        calendar=FakeCalendar([SESSION]),
        record_trial=False,
        persist=False,
        seed=1,
    )
    assert provider.calls
    expected = config_symbols(cfg)
    assert set(expected) - set(cfg.universe), "la config trae patas fuera del universo"
    assert all(wanted == expected for wanted, _start, _end, _step in provider.calls)


def test_una_pata_sin_historia_opera_con_impacto_cero(tmp_path, monkeypatch):
    # KO es pata de un par, no está en el universo. Sin sesiones previas
    # el simulador necesita el ADV por defecto también para ella.
    _home(tmp_path, monkeypatch)
    cfg = _config()
    assert "KO" in config_symbols(cfg) and "KO" not in cfg.universe
    report = run(
        cfg, DAY, DAY,
        provider=FakeIntradayProvider(_flat_bars([SESSION], ("SPY", "KO"), n=10)),
        strategies=[_IdaYVuelta("KO")],
        calendar=FakeCalendar([SESSION]),
        record_trial=False,
        persist=False,
        seed=1,
    )
    closed = report.blotter[report.blotter["closed"] == True]  # noqa: E712
    assert list(closed["symbol"]) == ["KO"]


def _pair_sessions():
    days = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6), date(2025, 1, 7))
    return [session_at(day, time(9, 30), time(16, 0)) for day in days]


def _ou_pair_bars(sessions, seed: int):
    """ln A = α + β ln B + s, con s un AR(1) de vida media 30 min. ln B es un paseo."""
    per = int((sessions[0].close_utc - sessions[0].open_utc).total_seconds() // 60)
    n = per * len(sessions)
    rng = np.random.default_rng(seed)
    kappa, mu, beta, alpha, sigma = math.log(2.0) / 30.0, 0.02, 1.4, 0.1, 1e-3
    b = math.exp(-kappa)
    residual = np.empty(n)
    residual[0] = mu
    shock = rng.normal(0.0, sigma, n - 1)
    for i in range(n - 1):
        residual[i + 1] = mu * (1.0 - b) + b * residual[i] + float(shock[i])
    log_b = math.log(50.0) + np.cumsum(rng.normal(0.0, 0.002, n))
    log_a = alpha + beta * log_b + residual
    rows = []
    cursor = 0
    for session in sessions:
        for i in range(per):
            ts = session.open_utc + timedelta(minutes=i)
            for symbol, series in (("AAA", log_a), ("BBB", log_b)):
                px = float(math.exp(series[cursor]))
                rows.append({
                    "symbol": symbol, "ts": ts, "o": px, "h": px, "l": px,
                    "c": px, "v": 200_000.0, "vw": px, "n": 10,
                })
            cursor += 1
    return bars_from_records(
        pd.DataFrame(rows), ["AAA", "BBB"], source="test", feed="iex", timeframe="1Min",
    )




def _store_with(root, bars, *, timeframe="1Min", feed="iex"):
    from src.hft.data.bar_store import BarStore

    store = BarStore(root)
    rows = []
    close = bars.close
    for symbol in close.columns:
        for ts in close.index:
            px = float(close.at[ts, symbol])
            rows.append({
                "symbol": symbol, "ts": ts.to_pydatetime(), "o": px, "h": px, "l": px,
                "c": px, "v": float(bars.volume.at[ts, symbol]), "vw": px, "n": 10,
            })
    records = pd.DataFrame(rows)
    store.upsert(
        records,
        timeframe=timeframe,
        feed=feed,
        fetched_start=close.index[0].to_pydatetime(),
        fetched_end=close.index[-1].to_pydatetime() + timedelta(days=1),
        fetched_at=close.index[-1].to_pydatetime() + timedelta(days=2),
    )
    return store


def test_la_procedencia_lista_los_archivos_de_las_patas(tmp_path, monkeypatch):
    from src.hft.data.bar_store import StoreProvider

    _home(tmp_path, monkeypatch)
    cfg = _config()
    store = _store_with(tmp_path / "bars", _flat_bars([SESSION], ("SPY", "KO"), n=10))
    report = run(
        cfg, DAY, DAY,
        provider=StoreProvider(store, feed="iex", timeframe="1Min"),
        strategies=[_IdaYVuelta("SPY")],
        calendar=FakeCalendar([SESSION]),
        record_trial=False,
        persist=False,
        seed=1,
    )
    paths = [item["path"] for item in report.provenance["data"]["files"]]
    assert any("/KO/" in path for path in paths), paths
    assert any("/SPY/" in path for path in paths), paths


MOMENTUM_DAY = date(2025, 2, 3)
MOMENTUM_SESSION = session_at(MOMENTUM_DAY, time(9, 30), time(16, 0))


def _prior_days(day: date, n: int):
    from src.hft.calendar import fallback_sessions

    sessions = fallback_sessions(day - timedelta(days=60), day - timedelta(days=1))
    return [session.date for session in sessions][-n:]


def _sip_daily(days, *, last_close: float, seed: int = 3):
    """Cierres diarios con ruido de 1 %; el último es ``last_close``."""
    from src.hft.calendar import NY
    from datetime import datetime

    rng = np.random.default_rng(seed)
    closes = last_close * np.exp(rng.normal(0.0, 0.01, len(days)))
    closes[-1] = last_close
    rows = []
    for day, px in zip(days, closes):
        ts = datetime(day.year, day.month, day.day, tzinfo=NY)
        rows.append({
            "symbol": "SPY", "ts": ts, "o": px, "h": px, "l": px, "c": float(px),
            "v": 1_000_000.0, "vw": px, "n": 1000,
        })
    return pd.DataFrame(rows)


def _put_sip_daily(store, frame):
    from datetime import datetime

    first = frame["ts"].min()
    last = frame["ts"].max()
    store.upsert(
        frame,
        timeframe="1Day",
        feed="sip",
        fetched_start=first,
        fetched_end=last + timedelta(days=1),
        fetched_at=datetime(2026, 10, 13, tzinfo=first.tzinfo),
    )


def test_los_cierres_del_store_son_sip_diarios_y_anteriores_al_fin(tmp_path, monkeypatch):
    from src.hft.backtest import momentum_closes_from_store
    from src.hft.data.bar_store import BarStore

    _home(tmp_path, monkeypatch)
    cfg = _config()
    store = BarStore(tmp_path / "bars")
    days = _prior_days(MOMENTUM_DAY, 25) + [MOMENTUM_DAY]
    _put_sip_daily(store, _sip_daily(days, last_close=98.0))
    closes = momentum_closes_from_store(store, cfg, MOMENTUM_DAY, MOMENTUM_DAY)
    assert closes is not None
    assert closes.feed == "sip"
    index = list(closes.close.index)
    # Fechas de sesión en NY, y nada del último día pedido ni posterior.
    assert index == days[:-1][-len(index):]
    assert max(index) < MOMENTUM_DAY
    assert len(index) >= cfg.intraday_momentum.sigma_sessions + 1


def _momentum_session_store():
    """Barras 1Min de SPY en el BarStore de ``HFT_HOME``, en 100."""
    bars = _flat_bars([MOMENTUM_SESSION], ("SPY",), price=100.0)
    return _store_with(load_settings().bars_dir, bars)


def _momentum_trades(output_dir):
    blotter = pd.read_csv(output_dir / "blotter.csv")
    return blotter[blotter["strategy_id"] == "intraday_momentum"]




def test_el_cli_acepta_los_cierres_en_un_csv(tmp_path, monkeypatch):
    # El recorrido completo ya lo cubre el test del BarStore; aquí solo
    # importa que main entregue a run los cierres del CSV.
    from types import SimpleNamespace

    import src.hft.backtest as backtest

    _home(tmp_path, monkeypatch)
    days = _prior_days(MOMENTUM_DAY, 25)
    frame = _sip_daily(days, last_close=98.0)
    path = tmp_path / "spy_sip_daily.csv"
    pd.DataFrame({
        "date": [day.isoformat() for day in days],
        "close": frame["c"],
    }).to_csv(path, index=False)
    seen = {}

    def _run(config, start, end, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path, daily=pd.DataFrame())

    monkeypatch.setattr(backtest, "run", _run)
    day = MOMENTUM_DAY.isoformat()
    assert backtest.main(["--start", day, "--end", day, "--momentum-closes", str(path)]) == 0
    closes = seen["momentum_closes"]
    assert closes.feed == "sip"
    assert list(closes.close.index) == days
    assert closes.close.iloc[-1] == 98.0


HALF_DAY = date(2025, 11, 28)


class _AlpacaCalendar:
    """``GET /v2/calendar`` falso: días NYSE y la media sesión de HALF_DAY."""

    def __init__(self) -> None:
        self.calls = []

    def get_calendar(self, start, end):
        from src.hft.calendar import fallback_sessions

        self.calls.append((start, end))
        rows = []
        for session in fallback_sessions(start, end):
            close = "13:00" if session.date == HALF_DAY else "16:00"
            rows.append({"date": session.date.isoformat(), "open": "09:30", "close": close})
        return rows


class _EntraTarde:
    """Compra SPY en la barra de las 12:50 ET."""

    id = "entratarde"

    def __init__(self) -> None:
        self.params = _Qty(1.0)
        self.theory = _theory(1.0)

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        if ctx.bar_ts.astimezone(NY).time() == time(12, 50):
            return [order_intent(
                self.id, "SPY", "buy", 1.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE + 5.0,
                reason="entrada de prueba",
            )]
        return []


def test_run_usa_el_cache_del_calendario_y_respeta_la_media_sesion(tmp_path, monkeypatch):
    from src.hft.calendar import HISTORY_LOOKBACK_DAYS, SessionCalendar

    settings = _home(tmp_path, monkeypatch)
    # El paper llena este cache con /v2/calendar; el backtest solo lo lee.
    SessionCalendar(client=_AlpacaCalendar(), use_cache=True).load(
        HALF_DAY - timedelta(days=HISTORY_LOOKBACK_DAYS), HALF_DAY,
    )
    assert (settings.data_dir / "hft" / "calendar.csv").is_file()
    half = session_at(HALF_DAY, time(9, 30), time(13, 0))
    report = run(
        _config(), HALF_DAY, HALF_DAY,
        provider=FakeIntradayProvider(_flat_bars([half], ("SPY",))),
        strategies=[_EntraTarde()],
        record_trial=False,
        persist=False,
        seed=1,
    )
    # Cierre 13:00: las 12:50 ya pasaron el corte de entradas (15 min).
    # Con el respaldo NYSE (16:00) la compra entraba.
    assert report.blotter.empty
    reasons = [item.reason for item in report.rejections]
    assert len(reasons) == 1 and reasons[0].startswith("entry_cutoff"), reasons


def test_el_cli_puede_pedir_el_calendario_a_alpaca(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import src.hft.backtest as backtest

    settings = _home(tmp_path, monkeypatch)
    client = _AlpacaCalendar()
    monkeypatch.setattr(backtest, "_calendar_client", lambda: client)
    seen = {}

    def _run(config, start, end, **kwargs):
        seen.update(kwargs)
        kwargs["calendar"].load(start, end)
        return SimpleNamespace(output_dir=tmp_path, daily=pd.DataFrame())

    monkeypatch.setattr(backtest, "run", _run)
    day = HALF_DAY.isoformat()
    assert backtest.main(["--start", day, "--end", day, "--calendar", "alpaca"]) == 0
    assert client.calls
    (session,) = seen["calendar"].sessions
    assert session.close_utc == session_at(HALF_DAY, time(9, 30), time(13, 0)).close_utc
    # Como en el paper, la respuesta queda en el cache para correr sin red.
    assert "2025-11-28" in (settings.data_dir / "hft" / "calendar.csv").read_text(encoding="utf-8")
