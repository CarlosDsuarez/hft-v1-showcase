"""Sesión intradía con fakes (HFT-12). Sin red."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from src.brokers.alpaca_adapter import AlpacaClient, BrokerConfig, make_intent_order_id
from src.hft.calendar import NY, Session, SessionCalendar, flatten_time, session_at
from src.hft.config import load_hft_config
from src.hft.costs import CostModel
from src.hft.data.bars import bars_from_records
from src.hft.execution.alpaca_exec import AlpacaExecution
from src.hft.execution.sim import SimExecution
from src.hft.risk import AssetFlag, RiskState
from src.hft.session import (
    EXIT_BROKER,
    EXIT_DATA,
    EXIT_GUARD,
    EXIT_LOCK,
    EXIT_NO_SESSION,
    EXIT_OK,
    EXIT_RISK,
    EXIT_UNEXPECTED,
    SessionHooks,
    SessionRequest,
    run_session,
)
from src.hft.types import Bracket, Fill, TheoryCard, TheoryParam, ensure_utc, order_intent
from src.instruments import load_instruments
from src.live_state import StateStore
from src.settings import load_settings

pytestmark = pytest.mark.integration

DAY = date(2024, 1, 3)
UTC = timezone.utc
CONFIG = load_hft_config()


@dataclass(frozen=True)
class _Params:
    qty: float


def _card(qty: float) -> TheoryCard:
    return TheoryCard(
        hypothesis="entrada de prueba",
        mechanism="el test fija la cantidad",
        references=("HFT-12",),
        data_requirements=("barras 1Min",),
        params=(TheoryParam("qty", float(qty), "cantidad fija", "test"),),
        holding_period="intradía",
        failure_regimes=("sin barra",),
        kill_criteria=("datos viejos",),
    )


class _Once:
    """Compra una vez si el libro está plano."""

    id = "once"
    params = _Params(10.0)
    theory = _card(10.0)

    def __init__(self) -> None:
        self._done = False

    def required_history(self):
        return (0, 0)

    def on_bar(self, ctx):
        if self._done or ctx.position("SPY") != 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "buy", 10.0, ctx.bar_ts,
            limit_price=100.0, reason="apertura",
        )]


class _Always:
    """Intenta comprar en cada barra. Sirve para ver el rechazo por dato viejo."""

    id = "always"
    params = _Params(1.0)
    theory = _card(1.0)

    def required_history(self):
        return (0, 0)

    def on_bar(self, ctx):
        return [order_intent(
            self.id, "SPY", "buy", 1.0, ctx.bar_ts,
            limit_price=100.0, reason="siempre",
        )]


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self._now = ensure_utc(now, "now")

    def now(self) -> datetime:
        return self._now

    def sleep_until(self, target: datetime) -> None:
        target = ensure_utc(target, "target")
        if target > self._now:
            self._now = target


class FakeIntradayProvider:
    def __init__(
        self,
        session: Session,
        *,
        clock: FakeClock | None = None,
        delay_s: float = 0.0,
        fail_after: int | None = None,
        always_fail: bool = False,
    ) -> None:
        self.session = session
        self.clock = clock
        self.delay_s = float(delay_s)
        self.fail_after = fail_after
        self.always_fail = always_fail
        self.calls = 0
        self._frame = _records(session)

    def get_bars(self, symbols, start, end, timeframe):
        del timeframe
        self.calls += 1
        if self.always_fail or (self.fail_after is not None and self.calls > self.fail_after):
            raise RuntimeError("feed caído")
        frame = self._frame
        start_ts = pd.Timestamp(ensure_utc(start, "start"))
        end_ts = pd.Timestamp(ensure_utc(end, "end"))
        mask = (frame["ts"] >= start_ts) & (frame["ts"] < end_ts)
        if self.delay_s and self.clock is not None:
            visible = pd.Timestamp(self.clock.now() - timedelta(seconds=self.delay_s + 60))
            mask = mask & (frame["ts"] <= visible)
        return bars_from_records(
            frame.loc[mask],
            list(symbols),
            source="fake",
            feed="iex",
            timeframe="1Min",
        )


class FakeBroker:
    def __init__(
        self,
        positions: dict | None = None,
        orders: list | None = None,
        entries: dict | None = None,
    ) -> None:
        self.positions = {key: float(value) for key, value in (positions or {}).items()}
        self.orders = list(orders or [])
        # avg_entry por símbolo; 100.0 si no se fija. Alpaca manda 0.0 cuando falta.
        self.entries = {key: float(value) for key, value in (entries or {}).items()}
        self.cancel_calls = 0
        self.account_number = "PA123"

    def check(self):
        return {
            "account_number": self.account_number,
            "equity": 100_000.0,
            "buying_power": 100_000.0,
            "account_blocked": False,
            "trading_blocked": False,
        }

    def get_account(self):
        return {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0}

    def get_positions(self):
        rows = []
        for ticker, qty in self.positions.items():
            if abs(qty) <= 1e-8:
                continue
            rows.append({
                "ticker": ticker,
                "qty": abs(qty),
                "side": "short" if qty < 0 else "long",
                "avg_entry": self.entries.get(ticker, 100.0),
                "market_value": abs(qty) * 100.0,
            })
        return pd.DataFrame(
            rows, columns=["ticker", "qty", "side", "avg_entry", "market_value"],
        )

    def list_orders(self, status="open", limit=500):
        del status, limit
        return [dict(order) for order in self.orders if order.get("status") == "open"]

    def cancel_all_orders(self):
        self.cancel_calls += 1
        for order in self.orders:
            if order.get("status") == "open":
                order["status"] = "canceled"


class FillingExecution:
    """Market al instante. No reenvía un client_order_id que ya existe."""

    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.posts: list[str] = []

    def submit(self, intents, *, now, bars):
        del bars
        fills = []
        known = {order["client_order_id"] for order in self.broker.orders}
        for intent in intents:
            client_id = make_intent_order_id(
                intent.strategy_id, intent.symbol, intent.signal_ts, intent.side,
            )
            if client_id in known:
                continue
            known.add(client_id)
            price = float(intent.limit_price or 100.0)
            self.broker.orders.append({
                "client_order_id": client_id,
                "symbol": intent.symbol,
                "side": intent.side,
                "qty": float(intent.qty),
                "status": "filled",
            })
            self.posts.append(client_id)
            signed = float(intent.qty) if intent.side == "buy" else -float(intent.qty)
            self.broker.positions[intent.symbol] = (
                self.broker.positions.get(intent.symbol, 0.0) + signed
            )
            if abs(self.broker.positions[intent.symbol]) <= 1e-8:
                self.broker.positions.pop(intent.symbol, None)
            fills.append(Fill(intent.intent_id, ensure_utc(now, "now"), float(intent.qty), price, 0.0, "taker"))
        return tuple(fills)

    def open_orders(self):
        return ()

    def cancel(self, intent_id: str) -> bool:
        del intent_id
        return False

    def poll(self):
        return ()

    def take_spawned(self):
        return ()

    def commit_last(self):
        return None

    def rollback_last(self):
        return None


class _Notes:
    def __init__(self) -> None:
        self.alerts = []

    def notify(self, alert) -> None:
        self.alerts.append(alert)


def _records(session: Session) -> pd.DataFrame:
    rows = []
    stamp = session.open_utc
    while stamp < session.close_utc:
        rows.append({
            "symbol": "SPY",
            "ts": stamp,
            "o": 100.0,
            "h": 100.0,
            "l": 100.0,
            "c": 100.0,
            "v": 10_000.0,
            "vw": 100.0,
            "n": 100.0,
        })
        stamp += timedelta(minutes=1)
    return pd.DataFrame(rows)


def _dirs(tmp_path: Path):
    state = tmp_path / "state"
    results = tmp_path / "results"
    kill = tmp_path / "kill"
    return state, results, kill


def _run(tmp_path: Path, session: Session, **kwargs) -> int:
    state, results, kill = _dirs(tmp_path)
    clock = kwargs.pop("clock", None) or FakeClock(session.open_utc)
    provider = kwargs.pop("provider", None) or FakeIntradayProvider(session, clock=clock)
    request = SessionRequest(
        config=CONFIG,
        mode=kwargs.pop("mode", "paper"),
        session_date=session.date,
        clock=clock,
        calendar=SessionCalendar(sessions=(session,), use_cache=False),
        provider=provider,
        state_dir=state,
        results_dir=results,
        kill_dir=kwargs.pop("kill_dir", kill),
        strategies=kwargs.pop("strategies", (_Once(),)),
        symbols=kwargs.pop("symbols", ("SPY",)),
        notifier=kwargs.pop("notifier", _Notes()),
        submit=kwargs.pop("submit", True),
        **kwargs,
    )
    return run_session(request)


def _journal(tmp_path: Path, session: Session) -> list:
    path = tmp_path / "state" / session.date.isoformat() / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _session(close: time = time(10, 0), open_at: time = time(9, 30)) -> Session:
    return session_at(DAY, open_at, close)


def test_sesion_normal_termina_plana(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6
    events = _journal(tmp_path, session)
    assert any(item["type"] == "heartbeat" for item in events)
    assert any(item["type"] == "session_start" for item in events)
    buys = [order for order in broker.orders if order["side"] == "buy"]
    sells = [order for order in broker.orders if order["side"] == "sell"]
    assert len(buys) == 1
    assert sells
    summary = json.loads((tmp_path / "state" / "hft_summary.json").read_text(encoding="utf-8"))
    assert summary["exit"] == 0
    assert summary["report"].startswith("HFT-13")


def test_reinicio_no_reenvia_la_orden(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    execution = FillingExecution(broker)
    clock = FakeClock(session.open_utc)
    provider = FakeIntradayProvider(session, clock=clock)

    def crash(payload):
        if payload["sent"]:
            raise RuntimeError("caída")

    first = _run(
        tmp_path, session,
        broker=broker, execution=execution, clock=clock, provider=provider,
        hooks=SessionHooks(after_step=crash),
    )
    assert first == EXIT_UNEXPECTED
    buys = [order["client_order_id"] for order in broker.orders if order["side"] == "buy"]
    assert len(buys) == 1
    clock2 = FakeClock(session.open_utc)
    second = _run(
        tmp_path, session,
        broker=broker, execution=execution, clock=clock2,
        provider=FakeIntradayProvider(session, clock=clock2),
    )
    assert second == EXIT_OK
    buys_after = [order["client_order_id"] for order in broker.orders if order["side"] == "buy"]
    assert buys_after == buys
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6


class _RampProvider(FakeIntradayProvider):
    """SPY cierra en 100.0 la primera barra y sube 0.5 por minuto."""

    def __init__(self, session: Session, **kwargs) -> None:
        super().__init__(session, **kwargs)
        frame = self._frame.copy()
        frame["c"] = 100.0 + 0.5 * np.arange(len(frame))
        self._frame = frame


def _crash_with_spy_open(tmp_path: Path, session: Session, broker: FakeBroker) -> None:
    """Primera corrida: compra 10 SPY y muere justo después de enviar."""

    def crash(payload):
        if payload["sent"]:
            raise RuntimeError("caída")

    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        hooks=SessionHooks(after_step=crash),
    )
    assert code == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": 10.0}


def _book(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "state" / "hft_book.json").read_text(encoding="utf-8"))


def _seeds(book: dict) -> list:
    return [
        (row["symbol"], row["side"], row["qty"], row["price"])
        for row in book["fills"] if row["intent_id"].startswith("reconcile-")
    ]


def test_relanzamiento_siembra_la_posicion_al_costo_del_broker(tmp_path, monkeypatch):
    # Paper 2026-10-12: tras el crash quedaron XLE 304.25 (avg 65.72) y SPY 5
    # (avg 777.97). El relanzamiento sembraba el libro a $1/acción: NAV inflada
    # ~$23.6k y el aplanado realizaba ese P&L falso en hft_summary.json.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(entries={"SPY": 98.0})
    _crash_with_spy_open(tmp_path, session, broker)
    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
    )
    assert code == EXIT_OK
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6
    book = _book(tmp_path)
    assert _seeds(book) == [("SPY", "buy", 10.0, 98.0)]
    # Costo 98 (broker), aplanado a 100 (barras): solo el movimiento real, 2 × 10.
    assert book["realized_pnl"] == pytest.approx(20.0)
    assert book["cash"] == pytest.approx(CONFIG.initial_cash + 20.0)
    events = _journal(tmp_path, session)
    start = max(i for i, item in enumerate(events) if item["type"] == "session_start")
    navs = [item["nav"] for item in events[start:] if item["type"] == "step"]
    assert navs and navs[0] == pytest.approx(CONFIG.initial_cash + 20.0)


def test_relanzamiento_sin_avg_entry_siembra_al_primer_cierre(tmp_path, monkeypatch):
    # Alpaca manda avg_entry 0.0 cuando falta: se usa el primer cierre final de la sesión.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(entries={"SPY": 0.0})
    _crash_with_spy_open(tmp_path, session, broker)
    clock = FakeClock(session.open_utc + timedelta(minutes=5))
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=_RampProvider(session, clock=clock),
    )
    assert code == EXIT_OK
    book = _book(tmp_path)
    assert _seeds(book) == [("SPY", "buy", 10.0, 100.0)]
    assert book["realized_pnl"] == pytest.approx(0.0)


def test_relanzamiento_sin_precio_alguno_sale_7_y_no_siembra_a_un_dolar(tmp_path, monkeypatch):
    # Sin avg_entry y sin barra final todavía: no hay costo honesto. Exit 7 (se relanza).
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(entries={"SPY": 0.0})
    _crash_with_spy_open(tmp_path, session, broker)
    notes = _Notes()
    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=FakeIntradayProvider(session, clock=clock), notifier=notes,
    )
    assert code == EXIT_DATA
    assert broker.positions == {"SPY": 10.0}
    assert not (tmp_path / "state" / "hft_book.json").exists()
    assert any("SPY" in alert.message for alert in notes.alerts if alert.exit_code == EXIT_DATA)


def test_kill_a_mitad_aplana_y_sale_8(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    kill = tmp_path / "kill"
    kill.mkdir()

    def arm(payload):
        if payload["step"] >= 2:
            (kill / "KILL").write_text("stop\n", encoding="utf-8")

    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker),
        kill_dir=kill, hooks=SessionHooks(after_step=arm),
    )
    assert code == EXIT_RISK
    assert broker.cancel_calls >= 1
    reasons = [item.get("reason", "") for item in _journal(tmp_path, session) if item["type"] == "risk_stop"]
    assert reasons and "KILL" in reasons[0]


def test_perdida_diaria_sale_8(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    state = RiskState(
        equity=100_000.0,
        nav=100_000.0,
        buying_power=100_000.0,
        cash=100_000.0,
        session_start_pnl=1_000.0,
        session_date=session.date,
        assets=(("SPY", AssetFlag(True, True)),),
    )
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), risk_state=state,
    )
    assert code == EXIT_RISK
    assert broker.orders == []


def test_dato_viejo_no_abre_entrada(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    provider = FakeIntradayProvider(session, clock=clock, delay_s=120.0)
    code = _run(
        tmp_path, session,
        mode="sim", submit=False, clock=clock, provider=provider,
        strategies=(_Always(),),
    )
    assert code == EXIT_OK
    reasons = []
    sent = []
    for item in _journal(tmp_path, session):
        if item["type"] != "step":
            continue
        reasons.extend(row["reason"] for row in item["rejections"])
        sent.extend(item["sent"])
    assert any(reason.startswith("stale_data") for reason in reasons)
    assert sent == []


def test_arranque_tarde_no_abre_entradas_del_atraso(tmp_path, monkeypatch):
    """Arrancar a mitad de sesión: las barras del atraso son viejas aunque
    la última no lo sea. Antes el box tenía el cierre de la última barra y
    el catch-up aprobaba la entrada de las 09:30 con datos de hace 20 min."""
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc + timedelta(minutes=20))
    code = _run(
        tmp_path, session,
        mode="sim", submit=False, clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
    )
    assert code == EXIT_OK
    reasons = []
    sent = []
    for item in _journal(tmp_path, session):
        if item["type"] != "step":
            continue
        reasons.extend(row["reason"] for row in item["rejections"])
        sent.extend(item["sent"])
    assert any(reason.startswith("stale_data") for reason in reasons)
    assert sent == []


def test_media_sesion_aplana_con_el_cierre_real(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session(open_at=time(12, 30), close=time(13, 0))
    broker = FakeBroker()
    code = _run(tmp_path, session, broker=broker, execution=FillingExecution(broker))
    assert code == EXIT_OK
    flats = [
        item for item in _journal(tmp_path, session)
        if item["type"] == "flatten" and item["why"] == "flatten_time"
    ]
    assert flats
    seen = datetime.fromisoformat(flats[0]["ts"]).astimezone(NY)
    expected = flatten_time(session, CONFIG.risk.flatten_minutes).astimezone(NY)
    assert (seen.hour, seen.minute) == (expected.hour, expected.minute)
    assert expected.hour == 12 and expected.minute == 55


def test_posicion_ajena_sin_adopt_sale_2(tmp_path, monkeypatch):
    # HFT-PAPER-05: era exit 3 y el supervisor lo relanzaba 20 veces sin poder arreglarlo.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(positions={"AAPL": 5.0})
    code = _run(tmp_path, session, broker=broker, execution=FillingExecution(broker))
    assert code == EXIT_GUARD
    assert broker.orders == []
    assert broker.positions["AAPL"] == 5.0


def test_orden_ajena_sin_adopt_sale_2(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(orders=[{
        "client_order_id": "hft-20240103-AAPL-buy", "symbol": "AAPL",
        "side": "buy", "qty": 5.0, "status": "open",
    }])
    code = _run(tmp_path, session, broker=broker, execution=FillingExecution(broker))
    assert code == EXIT_GUARD
    assert broker.orders[0]["status"] == "open"


def test_adopt_deja_la_posicion_ajena_fuera(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker(positions={"AAPL": 5.0})
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), adopt=True,
    )
    assert code == EXIT_OK
    assert broker.positions["AAPL"] == 5.0
    assert any(item["type"] == "adopt" for item in _journal(tmp_path, session))


def test_sin_sesion_sale_1(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    code = run_session(SessionRequest(
        config=CONFIG,
        mode="signals",
        session_date=date(2024, 1, 6),
        clock=FakeClock(session.open_utc),
        calendar=SessionCalendar(sessions=(), use_cache=False),
        provider=FakeIntradayProvider(session),
        state_dir=tmp_path / "state",
        results_dir=tmp_path / "results",
        kill_dir=tmp_path / "kill",
        symbols=("SPY",),
        notifier=_Notes(),
    ))
    assert code == EXIT_NO_SESSION
    assert not (tmp_path / "state").exists()


def test_lock_tomado_sale_4(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    state = tmp_path / "state"
    holder = StateStore(state)
    holder.acquire()
    try:
        code = _run(tmp_path, session, mode="sim", submit=False)
    finally:
        holder.release()
    assert code == EXIT_LOCK


def test_feed_roto_sale_7(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    provider = FakeIntradayProvider(session, always_fail=True)
    code = _run(tmp_path, session, mode="sim", submit=False, provider=provider)
    assert code == EXIT_DATA


def test_credenciales_rechazadas_sale_7_con_mensaje_claro(tmp_path, monkeypatch):
    from src.hft.data.alpaca_data import DataAuthError

    class _Rejecting(FakeIntradayProvider):
        def get_bars(self, symbols, start, end, timeframe):
            raise DataAuthError("HTTP 401 /v2/stocks/bars: credenciales rechazadas", status=401)

    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    notes = _Notes()
    code = _run(tmp_path, session, mode="sim", submit=False, provider=_Rejecting(session), notifier=notes)
    assert code == EXIT_DATA
    text = " ".join(str(vars(alert)) for alert in notes.alerts)
    assert "credenciales rechazadas" in text
    assert "feed no alcanzable" not in text


def test_fallos_de_api_aplanan_y_salen_3(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    provider = FakeIntradayProvider(session, clock=clock, fail_after=1)
    code = _run(tmp_path, session, mode="sim", submit=False, clock=clock, provider=provider)
    assert code == EXIT_BROKER
    errors = [item for item in _journal(tmp_path, session) if item["type"] == "api_error"]
    assert len(errors) >= 3


def test_signals_escribe_csv_y_no_postea(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    code = _run(tmp_path, session, mode="signals", submit=False)
    assert code == EXIT_OK
    path = tmp_path / "results" / "hft" / "signals" / f"{session.date.isoformat()}.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "symbol,side,qty,limit,stop,reason"
    assert any(line.startswith("SPY,buy,") for line in text[1:])


def test_signals_imprime_cada_senal_una_vez(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    with caplog.at_level("INFO", logger="hft.session"):
        code = _run(tmp_path, session, mode="signals", submit=False)
    assert code == EXIT_OK
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("señal ")]
    assert any(line.startswith("señal SPY buy ") for line in lines)
    path = tmp_path / "results" / "hft" / "signals" / f"{session.date.isoformat()}.csv"
    assert len(lines) == len(path.read_text(encoding="utf-8").splitlines()) - 1


def test_paper_sin_submit_no_postea(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    code = _run(tmp_path, session, mode="paper", submit=False, broker=broker)
    assert code == EXIT_OK
    assert broker.orders == []
    assert any(item["type"] == "step" and item["sent"] for item in _journal(tmp_path, session))


def test_paper_sin_submit_no_cancela_en_el_broker(tmp_path, monkeypatch):
    # HFT-PAPER-02: el dry run hacía DELETE /v2/orders en la cuenta real al aplanar.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    code = _run(tmp_path, session, mode="paper", submit=False, broker=broker)
    assert code == EXIT_OK
    assert any(item["type"] == "flatten" for item in _journal(tmp_path, session))
    assert broker.cancel_calls == 0


class _Burst:
    """Cuatro compras de $10k en la misma barra: más que el apalancamiento ($25k)."""

    id = "burst"
    params = _Params(100.0)
    theory = _card(100.0)

    def __init__(self) -> None:
        self._done = False

    def required_history(self):
        return (0, 0)

    def on_bar(self, ctx):
        if self._done:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "buy", 100.0, ctx.bar_ts,
            limit_price=100.0, reason="rafaga", seq=seq,
        ) for seq in range(4)]


def test_sizing_en_cero_es_rechazo_y_no_tumba_la_sesion(tmp_path, monkeypatch):
    # HFT-PAPER-01: dry run de paper 2026-10-07, ORB con varias entradas a las 09:34.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    code = _run(
        tmp_path, session, strategies=(_Burst(),),
        broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    reasons = [
        rejection["reason"]
        for item in _journal(tmp_path, session) if item["type"] == "step"
        for rejection in item["rejections"]
    ]
    assert any(reason == "sizing_sin_cupo" for reason in reasons)
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6


class _ShortOnce(_Once):
    """Abre un corto una vez."""

    id = "short_once"

    def on_bar(self, ctx):
        if self._done or ctx.position("SPY") != 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "sell", 10.0, ctx.bar_ts,
            limit_price=100.0, reason="corto",
        )]


class _BrokerConFichas(FakeBroker):
    def __init__(self, shortable: bool) -> None:
        super().__init__()
        self.shortable = shortable
        self.asset_calls = []

    def get_asset(self, symbol, as_of=None):
        self.asset_calls.append((symbol, as_of))
        return {"symbol": symbol, "tradable": True,
                "shortable": self.shortable, "easy_to_borrow": self.shortable}


def _short_reasons(tmp_path, session):
    reasons, sent = [], []
    for item in _journal(tmp_path, session):
        if item["type"] == "step":
            reasons.extend(row["reason"] for row in item["rejections"])
            sent.extend(item["sent"])
    return reasons, sent


def test_paper_lee_las_fichas_de_short_del_broker(tmp_path, monkeypatch):
    """Antes el paper dejaba assets vacío y todo corto salía short_no_habilitado."""
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = _BrokerConFichas(shortable=True)
    code = _run(
        tmp_path, session, mode="paper", submit=False, broker=broker,
        strategies=(_ShortOnce(),),
    )
    assert code == EXIT_OK
    assert ("SPY", session.date) in broker.asset_calls
    reasons, sent = _short_reasons(tmp_path, session)
    assert not any(reason.startswith("short_no_habilitado") for reason in reasons)
    assert sent


def test_paper_sin_ficha_shortable_sigue_cerrado(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    code = _run(
        tmp_path, session, mode="paper", submit=False,
        broker=_BrokerConFichas(shortable=False), strategies=(_ShortOnce(),),
    )
    assert code == EXIT_OK
    reasons, sent = _short_reasons(tmp_path, session)
    assert any(reason.startswith("short_no_habilitado") for reason in reasons)
    assert sent == []


def test_allow_research_en_paper_sale_2(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    code = _run(tmp_path, session, mode="paper", allow_research=True, submit=False)
    assert code == EXIT_GUARD


def _spy_default_set(monkeypatch) -> list:
    """Ids del set que arma la sesión cuando no recibe ``strategies``."""
    import src.hft.session as session_mod

    real = session_mod._default_strategies
    seen = []

    def _spy(*args, **kwargs):
        built = real(*args, **kwargs)
        seen.append(tuple(strategy.id for strategy in built))
        return built

    monkeypatch.setattr(session_mod, "_default_strategies", _spy)
    return seen


_SET_DE_HOY = ("orb", "intraday_momentum", "vwap_reversion", "pairs_ou")


def test_paper_sin_with_corre_el_set_de_hoy(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    seen = _spy_default_set(monkeypatch)
    code = _run(
        tmp_path, _session(), mode="paper", submit=False, strategies=None,
        broker=FakeBroker(),
    )
    assert code == EXIT_OK
    assert seen == [_SET_DE_HOY]


def test_paper_with_hourly_momentum_la_suma_y_no_es_research(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    seen = _spy_default_set(monkeypatch)
    code = _run(
        tmp_path, _session(), mode="paper", submit=False, strategies=None,
        broker=FakeBroker(),
        with_strategies=("hourly_momentum",),
    )
    assert code == EXIT_OK
    assert seen == [(*_SET_DE_HOY, "hourly_momentum")]


def test_with_fuera_de_la_lista_sale_2_sin_tomar_el_lock(tmp_path, monkeypatch):
    # A–S por --with saltaría la puerta de --allow-research.
    monkeypatch.delenv("HFT_KILL", raising=False)
    code = _run(
        tmp_path, _session(), mode="paper", submit=False, strategies=None,
        broker=FakeBroker(),
        with_strategies=("avellaneda_stoikov",),
    )
    assert code == EXIT_GUARD
    assert not (tmp_path / "state").exists()


def test_sim_con_parcial_cero_tambien_queda_plano(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    sim = SimExecution.from_config(CONFIG, rng=np.random.default_rng(1))
    sim.partial_prob = 0.0
    sim._adv = {"SPY": 1.0e12}
    sim._sigma = {"SPY": 0.0}
    code = _run(tmp_path, session, mode="sim", submit=False, execution=sim)
    assert code == EXIT_OK
    summary = json.loads((tmp_path / "state" / "hft_summary.json").read_text(encoding="utf-8"))
    assert summary["positions"] == {}


class _OnceTlt(_Once):
    """Compra TLT una vez."""

    id = "once_tlt"

    def on_bar(self, ctx):
        if self._done or ctx.position("TLT") != 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "TLT", "buy", 10.0, ctx.bar_ts,
            limit_price=100.0, reason="apertura",
        )]


class _SparseProvider(FakeIntradayProvider):
    """SPY y TLT cada minuto, salvo las barras de TLT en ``missing``."""

    def __init__(self, session: Session, missing, **kwargs) -> None:
        super().__init__(session, **kwargs)
        spy = _records(session)
        tlt = spy.assign(symbol="TLT")
        tlt = tlt.loc[~tlt["ts"].isin([pd.Timestamp(stamp) for stamp in missing])]
        self._frame = pd.concat([spy, tlt], ignore_index=True)


def test_barra_faltante_con_posicion_abierta_no_tumba_la_sesion(tmp_path, monkeypatch):
    # HFT-PAPER-03: paper 2026-10-09. IEX no trajo la barra de TLT de las 13:42 UTC
    # y la sesión murió con "sin marca para la posición abierta TLT".
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    hole = session.open_utc + timedelta(minutes=5)
    provider = _SparseProvider(session, missing=(hole,), clock=clock)
    broker = FakeBroker()
    code = _run(
        tmp_path, session, symbols=("SPY", "TLT"), strategies=(_OnceTlt(),),
        clock=clock, provider=provider, broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    assert abs(broker.positions.get("TLT", 0.0)) < 1e-6
    steps = [item for item in _journal(tmp_path, session) if item["type"] == "step"]
    assert any(item["ts"] > hole.isoformat() for item in steps)


class _BracketBroker(FakeBroker):
    """Las patas de un bracket traen un client_order_id de Alpaca (UUID), sin ``h-``.

    Con ``nested=True`` cuelgan del padre en ``legs``, como en ``GET /v2/orders``.
    """

    def list_orders(self, status="open", after=None, limit=500, *, nested=False):
        del after, limit
        wanted = (lambda order: True) if status == "all" else (
            lambda order: order.get("status") == "open"
        )
        if not nested:
            return [_public(order) for order in self.orders if wanted(order)]
        rows = []
        for order in self.orders:
            if order.get("_parent") or not wanted(order):
                continue
            row = _public(order)
            row["legs"] = [
                _public(leg) for leg in self.orders
                if leg.get("_parent") == order["client_order_id"]
            ] or None
            rows.append(row)
        return rows


def _public(order: dict) -> dict:
    return {key: value for key, value in order.items() if not key.startswith("_")}


class _BracketFilling(FillingExecution):
    """Llena la entrada y deja viva la pata stop, como Alpaca con un OTO."""

    def submit(self, intents, *, now, bars):
        fills = super().submit(intents, now=now, bars=bars)
        for intent in intents:
            if intent.bracket is None or intent.bracket.stop is None:
                continue
            parent = make_intent_order_id(
                intent.strategy_id, intent.symbol, intent.signal_ts, intent.side,
            )
            for order in self.broker.orders:
                if order["client_order_id"] == parent:
                    order["order_class"] = "oto"
            self.broker.orders.append({
                "client_order_id": "9f6361f2-6a37-475f-93d2-303f6136ae84",
                "symbol": intent.symbol,
                "side": "sell" if intent.side == "buy" else "buy",
                "qty": float(intent.qty),
                "type": "stop",
                "order_class": "oto",
                "status": "open",
                "_parent": parent,
            })
        return fills


class _OnceBracket(_Once):
    """Compra SPY una vez con stop de bracket."""

    id = "once_bracket"

    def on_bar(self, ctx):
        if self._done or ctx.position("SPY") != 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "buy", 10.0, ctx.bar_ts,
            limit_price=100.0, bracket=Bracket(stop=95.0), reason="apertura",
        )]


def test_relanzamiento_reconoce_la_pata_de_su_bracket(tmp_path, monkeypatch):
    # HFT-PAPER-04: paper 2026-10-09. Tras el crash, la pata stop del bracket propio
    # (UUID de Alpaca) salía como orden ajena y los 19 relanzamientos murieron en exit 3.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = _BracketBroker()
    execution = _BracketFilling(broker)

    def crash(payload):
        if payload["sent"]:
            raise RuntimeError("caída")

    clock = FakeClock(session.open_utc)
    first = _run(
        tmp_path, session, strategies=(_OnceBracket(),),
        broker=broker, execution=execution, clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        hooks=SessionHooks(after_step=crash),
    )
    assert first == EXIT_UNEXPECTED
    assert [order["status"] for order in broker.orders if order.get("_parent")] == ["open"]
    clock2 = FakeClock(session.open_utc)
    second = _run(
        tmp_path, session, strategies=(_OnceBracket(),),
        broker=broker, execution=execution, clock=clock2,
        provider=FakeIntradayProvider(session, clock=clock2),
    )
    assert second == EXIT_OK
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6
    assert not any(item["type"] == "adopt" for item in _journal(tmp_path, session))


class _OnceThenCancel(_Once):
    """Compra una vez y dos barras después pide cancelar lo vivo en SPY."""

    id = "once_cancel"

    def __init__(self) -> None:
        super().__init__()
        self._seen = 0
        self._cancels: list = []

    def on_bar(self, ctx):
        self._seen += 1
        if self._seen == 3:
            self._cancels.append("SPY")
        return super().on_bar(ctx)

    def take_cancels(self):
        out = tuple(self._cancels)
        self._cancels = []
        return out


def test_el_journal_anota_lo_que_el_motor_cancelo_a_pedido(tmp_path, monkeypatch):
    # HFT-PAPER-16: la pata viva de un par se cancela por pedido de la
    # estrategia. El paso del journal dice qué orden fue.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    code = _run(
        tmp_path, session, mode="paper", submit=False, broker=broker,
        strategies=(_OnceThenCancel(),),
    )
    assert code == EXIT_OK
    steps = [item for item in _journal(tmp_path, session) if item["type"] == "step"]
    sent = [row["intent_id"] for item in steps for row in item["sent"]]
    cancelled = [intent_id for item in steps for intent_id in item.get("cancelled", [])]
    assert len(sent) == 1
    assert cancelled == sent
    assert broker.cancel_calls == 0


# --- 2026-10-12: fill fraccional repetido y orden propia vista como ajena ---


class _FraccionUnaVez(_Once):
    """Compra fraccional, como el XLE de vwap_reversion del 2026-10-12.

    A 9 decimales la cantidad redondea hacia arriba (…652|6 → …653).
    """

    id = "fraccion"
    QTY = 15.2068126526

    def on_bar(self, ctx):
        if self._done or ctx.position("SPY") != 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "buy", self.QTY, ctx.bar_ts,
            limit_price=100.0, reason="apertura",
        )]


class _Resp:
    def __init__(self, payload) -> None:
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeAlpaca(FakeBroker):
    """Broker con estado detrás de un ``AlpacaClient`` real (HTTP falso).

    Todo POST se llena al instante, al límite o a 100. Como en Alpaca,
    un fraccional llega en dos actividades: la parte entera (``fill``)
    y la fracción (``partial_fill``). ``activity_lag`` es cuántas
    lecturas de ``/v2/account/activities`` pasan sin verlas, mientras
    ``GET /v2/orders`` ya muestra el ``filled_qty`` completo.
    """

    def __init__(self, *, activity_lag: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.activity_lag = int(activity_lag)
        self.activity_reads = 0
        self.activities: list = []
        self.posts: list = []

    def list_orders(self, status="open", after=None, limit=500, *, nested=False):
        # Como ``GET /v2/orders``: con status="all" también las llenas.
        del after, limit, nested
        if status == "all":
            return [dict(order) for order in self.orders]
        return super().list_orders(status=status)

    def install(self, monkeypatch) -> None:
        def _urlopen(req, timeout=None):
            del timeout
            body = json.loads(req.data.decode()) if req.data else None
            parsed = urlparse(req.full_url)
            return _Resp(self.handle(req.get_method(), parsed.path, body))

        monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    def handle(self, method, path, body):
        if method == "GET" and path == "/v2/account":
            return {
                "account_number": "", "equity": "100000", "buying_power": "100000",
                "account_blocked": False, "trading_blocked": False,
            }
        if method == "GET" and path.startswith("/v2/assets/"):
            symbol = path.rsplit("/", 1)[-1]
            return {
                "id": f"asset-{symbol}", "class": "us_equity", "exchange": "ARCA",
                "symbol": symbol, "name": symbol, "status": "active",
                "tradable": True, "marginable": True, "shortable": True,
                "easy_to_borrow": True, "fractionable": True,
            }
        if method == "POST" and path == "/v2/orders":
            return self._fill(body)
        if method == "GET" and path == "/v2/orders":
            return [dict(order) for order in self.orders]
        if method == "GET" and path == "/v2/account/activities":
            self.activity_reads += 1
            return [dict(row) for since, row in self.activities if since <= self.activity_reads]
        if method == "DELETE" and path.startswith("/v2/orders"):
            return {}
        raise AssertionError(f"{method} {path}")

    def _fill(self, body):
        self.posts.append(dict(body))
        order_id = f"o-{len(self.orders) + 1}"
        qty = Decimal(body["qty"])
        price = body.get("limit_price") or "100"
        self.orders.append({
            "id": order_id,
            "client_order_id": body["client_order_id"],
            "symbol": body["symbol"],
            "side": body["side"],
            "qty": body["qty"],
            "filled_qty": body["qty"],
            "filled_avg_price": price,
            "status": "filled",
        })
        whole = Decimal(int(qty))
        parts = [("fill", whole), ("partial_fill", qty - whole)] if 0 < whole < qty else [("fill", qty)]
        visible = self.activity_reads + 1 + self.activity_lag
        for n, (kind, part) in enumerate(parts):
            self.activities.append((visible, {
                "id": f"{order_id}::{n}", "type": kind, "order_id": order_id,
                "qty": str(part), "price": price, "side": body["side"],
                "symbol": body["symbol"],
            }))
        signed = float(qty) if body["side"] == "buy" else -float(qty)
        symbol = body["symbol"]
        self.positions[symbol] = self.positions.get(symbol, 0.0) + signed
        if abs(self.positions[symbol]) <= 1e-8:
            self.positions.pop(symbol, None)
        return {"id": order_id, "status": "filled", "filled_qty": body["qty"]}


def _alpaca_execution(tmp_path: Path, broker: FakeAlpaca, clock: FakeClock, monkeypatch):
    broker.install(monkeypatch)
    master = load_instruments(load_settings().universe_path("universe_hft.csv"))
    client = AlpacaClient(
        BrokerConfig("PKTEST1234", "secreto-falso"),
        kill_dir=tmp_path / "kill",
        instruments=master,
    )
    return AlpacaExecution(client, CostModel.from_config(CONFIG), clock=clock.now)


def test_fill_fraccional_con_actividades_tardias_no_tumba_la_sesion(tmp_path, monkeypatch):
    """La orden ya dice filled y las actividades llegan una lectura después."""
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    broker = FakeAlpaca(activity_lag=1)
    code = _run(
        tmp_path, session,
        broker=broker,
        execution=_alpaca_execution(tmp_path, broker, clock, monkeypatch),
        clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        strategies=(_FraccionUnaVez(),),
    )
    assert code == EXIT_OK
    buys = [Decimal(post["qty"]) for post in broker.posts if post["side"] == "buy"]
    sells = [Decimal(post["qty"]) for post in broker.posts if post["side"] == "sell"]
    assert len(buys) == 1
    # El flatten vende lo que el libro tiene: si el fill se contó dos
    # veces, vende el doble y el broker queda corto.
    assert sum(sells) == buys[0]
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-9


def test_la_cantidad_del_intent_es_la_que_recibe_el_broker(tmp_path, monkeypatch):
    """Intent, journal y POST llevan la misma cantidad canónica (9 decimales)."""
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    broker = FakeAlpaca()
    code = _run(
        tmp_path, session,
        broker=broker,
        execution=_alpaca_execution(tmp_path, broker, clock, monkeypatch),
        clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        strategies=(_FraccionUnaVez(),),
    )
    assert code == EXIT_OK
    posted = [post["qty"] for post in broker.posts if post["side"] == "buy"]
    sent = [
        row["qty"]
        for item in _journal(tmp_path, session) if item["type"] == "step"
        for row in item["sent"] if row["side"] == "buy"
    ]
    assert posted == ["15.206812652"]
    assert sent == [float(posted[0])]


class _CaeTrasEnviar:
    """El proceso muere justo después del POST, dentro de ``Engine.step``.

    Es lo que pasó el 2026-10-12 con el ValueError: el evento ``step``
    con ``sent`` no llegó al journal.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.armed = True

    def submit(self, intents, *, now, bars):
        fills = self.inner.submit(intents, now=now, bars=bars)
        if self.armed and intents:
            self.armed = False
            raise RuntimeError("caída tras el POST")
        return fills

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _crash_after_post(tmp_path: Path, session: Session, broker: FakeBroker) -> None:
    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=_CaeTrasEnviar(FillingExecution(broker)),
        clock=clock, provider=FakeIntradayProvider(session, clock=clock),
    )
    assert code == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": 10.0}


def test_reinicio_tras_caida_con_orden_enviada_reconoce_la_posicion(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    _crash_after_post(tmp_path, session, broker)
    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker),
        clock=clock, provider=FakeIntradayProvider(session, clock=clock),
    )
    assert code == EXIT_OK
    assert len([order for order in broker.orders if order["side"] == "buy"]) == 1
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6


def test_reinicio_tras_caida_sigue_frenando_lo_ajeno(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()
    _crash_after_post(tmp_path, session, broker)
    broker.positions["AAPL"] = 5.0
    notes = _Notes()
    clock = FakeClock(session.open_utc)
    code = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker),
        clock=clock, provider=FakeIntradayProvider(session, clock=clock),
        notifier=notes,
    )
    assert code == EXIT_GUARD
    messages = [alert.message for alert in notes.alerts if alert.exit_code == EXIT_GUARD]
    assert any("símbolos=['AAPL']" in message for message in messages)
    assert broker.positions == {"SPY": 10.0, "AAPL": 5.0}


class _CrashTrasPostear(FillingExecution):
    """Postea y llena en el broker y luego revienta, como el motor el 2026-10-12."""

    def __init__(self, broker: FakeBroker) -> None:
        super().__init__(broker)
        self.armed = True

    def submit(self, intents, *, now, bars):
        fills = super().submit(intents, now=now, bars=bars)
        for order in self.broker.orders:
            order.setdefault("filled_qty", str(order["qty"]))
        if intents and self.armed:
            self.armed = False
            raise ValueError("el fill excede la cantidad")
        return fills


def _drop_submit_events(tmp_path: Path, session: Session) -> None:
    path = tmp_path / "state" / session.date.isoformat() / "events.jsonl"
    kept = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["type"] != "submit"
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def test_relanzamiento_reconoce_la_orden_llena_que_el_journal_no_vio(tmp_path, monkeypatch):
    # HFT-PAPER-14: paper 2026-10-12. La orden h-...-SPY-buy salió y llenó 5 en el
    # mismo paso en que el motor reventó, antes de escribir el journal. Sin rastro
    # local, el preflight tomó SPY por ajena y los 20 relanzamientos salieron 3.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = _BracketBroker()
    execution = _CrashTrasPostear(broker)
    clock = FakeClock(session.open_utc)
    first = _run(
        tmp_path, session,
        broker=broker, execution=execution, clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
    )
    assert first == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": 10.0}
    steps = [row for row in _journal(tmp_path, session) if row["type"] == "step"]
    assert not any(row.get("sent") for row in steps)
    # El write-ahead deja el envío en el journal (evento submit). Sin ese
    # rastro local, la vía del broker tiene que bastar por sí sola.
    _drop_submit_events(tmp_path, session)
    clock2 = FakeClock(session.open_utc)
    second = _run(
        tmp_path, session,
        broker=broker, execution=execution, clock=clock2,
        provider=FakeIntradayProvider(session, clock=clock2),
    )
    assert second == EXIT_OK
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6
    assert not any(item["type"] == "adopt" for item in _journal(tmp_path, session))


class _VendeLoHeredado(_Once):
    """Cierra su posición heredada en la primera barra, como un exit de vwap.

    Mismo id que quien la abrió: al relanzar recupera el dueño de la orden
    ``h-fraccion-…`` (HFT-PAPER-08). Otra estrategia no la vería.
    """

    id = "fraccion"

    def on_bar(self, ctx):
        qty = ctx.exit_qty("SPY")
        if self._done or qty <= 0:
            return []
        self._done = True
        return [order_intent(
            self.id, "SPY", "sell", qty, ctx.bar_ts,
            limit_price=100.0, reason="exit",
        )]


@pytest.mark.parametrize("strategy", [_FraccionUnaVez, _VendeLoHeredado], ids=["flatten", "estrategia"])
def test_relanzamiento_cierra_el_largo_fraccional_heredado(tmp_path, monkeypatch, strategy):
    # HFT-PAPER-17: replay Fase 6 del 2026-10-12. Tras la caída quedaron XLE
    # 152.183838076 y SPY 12.852974821 (vwap_reversion dimensiona fraccional).
    # El AlpacaExecution nuevo arrancaba sin posiciones: la venta que cierra el
    # largo heredado salía como corto, "el corto exige acciones enteras", y
    # cada relanzamiento salía 5. El flatten de fin de día iba por la misma vía.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeAlpaca(activity_lag=1)
    clock = FakeClock(session.open_utc)
    first = _run(
        tmp_path, session,
        broker=broker,
        execution=_CaeTrasEnviar(_alpaca_execution(tmp_path, broker, clock, monkeypatch)),
        clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        strategies=(_FraccionUnaVez(),),
    )
    assert first == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": pytest.approx(15.206812652)}
    clock2 = FakeClock(session.open_utc)
    notes = _Notes()
    second = _run(
        tmp_path, session,
        broker=broker,
        execution=_alpaca_execution(tmp_path, broker, clock2, monkeypatch),
        clock=clock2,
        provider=FakeIntradayProvider(session, clock=clock2),
        strategies=(strategy(),),
        notifier=notes,
    )
    errors = [alert.message for alert in notes.alerts if alert.exit_code == EXIT_UNEXPECTED]
    assert second == EXIT_OK, errors
    sells = [post for post in broker.posts if post["side"] == "sell"]
    assert [Decimal(post["qty"]) for post in sells] == [Decimal("15.206812652")]
    # Cada caso sale por su vía: el exit de la estrategia o el flatten de sesión.
    owner = "flatten" if strategy is _FraccionUnaVez else "fraccion"
    assert sells[0]["client_order_id"].startswith(f"h-{owner}-")
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-9


class _NeverFillsFlatten(FillingExecution):
    """Llena las entradas; el broker acepta el flatten y no lo llena."""

    def submit(self, intents, *, now, bars):
        kept = tuple(intent for intent in intents if intent.strategy_id != "flatten")
        return super().submit(kept, now=now, bars=bars)


def _summary(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "state" / "hft_summary.json").read_text(encoding="utf-8"))


def test_resumen_sin_marcas_no_valora_a_un_dolar(tmp_path, monkeypatch, caplog):
    # Relanzamiento con SPY propio en el broker; el feed responde al probe y
    # después falla siempre: la sesión aplana y sale 3 sin una sola barra. El
    # broker no llena el flatten, así que el libro cierra con SPY abierta y
    # sin marca. La NAV no puede salir valorando SPY a $1/acción.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()

    def crash(payload):
        if payload["sent"]:
            raise RuntimeError("caída")

    clock = FakeClock(session.open_utc)
    first = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        hooks=SessionHooks(after_step=crash),
    )
    assert first == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": 10.0}
    clock2 = FakeClock(session.open_utc)
    notes = _Notes()
    with caplog.at_level("WARNING", logger="hft.session"):
        second = _run(
            tmp_path, session,
            broker=broker, execution=_NeverFillsFlatten(broker), clock=clock2,
            provider=FakeIntradayProvider(session, clock=clock2, fail_after=1),
            notifier=notes,
        )
    assert second == EXIT_BROKER
    summary = _summary(tmp_path)
    assert summary["positions"] == {"SPY": 10.0}
    assert summary["nav"] is None
    assert summary["nav_unmarked"] == ["SPY"]
    assert any(
        record.levelname == "WARNING" and "SPY" in record.getMessage()
        for record in caplog.records
    )
    eod = [alert for alert in notes.alerts if alert.title == "Resumen EOD"]
    assert eod and "SPY" in eod[0].message


def test_resumen_usa_la_ultima_marca_del_box(tmp_path, monkeypatch):
    # TLT abierta al cierre y sin barra en el último minuto (IEX): la fila final
    # de los frames trae NaN para TLT. La NAV usa la marca que el box conservó.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    clock = FakeClock(session.open_utc)
    hole = session.close_utc - timedelta(minutes=1)
    provider = _SparseProvider(session, missing=(hole,), clock=clock)
    broker = FakeBroker()
    code = _run(
        tmp_path, session, symbols=("SPY", "TLT"), strategies=(_OnceTlt(),),
        clock=clock, provider=provider, broker=broker,
        execution=_NeverFillsFlatten(broker),
    )
    assert code == EXIT_OK
    summary = _summary(tmp_path)
    assert summary["positions"] == {"TLT": 10.0}
    assert summary["nav"] == pytest.approx(CONFIG.initial_cash)
    assert summary["nav_unmarked"] == []


class _NoBarsProvider(FakeIntradayProvider):
    """Responde a todo, sin una sola barra."""

    def __init__(self, session: Session, **kwargs) -> None:
        super().__init__(session, **kwargs)
        self._frame = self._frame.iloc[0:0]


def test_relanzamiento_sin_barras_con_posicion_abierta_no_tumba_la_sesion(tmp_path, monkeypatch):
    # Relanzamiento con SPY propio y un feed que todavía no trae barras: el
    # motor no tiene frames ni marcas. Engine.step reventaba con "sin marca
    # para la posición abierta SPY" (exit 5) y no quedaba hft_summary.json.
    # La posición queda sin marca hasta que llegue una barra: el journal no
    # inventa una NAV.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = FakeBroker()

    def crash(payload):
        if payload["sent"]:
            raise RuntimeError("caída")

    clock = FakeClock(session.open_utc)
    first = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock,
        provider=FakeIntradayProvider(session, clock=clock),
        hooks=SessionHooks(after_step=crash),
    )
    assert first == EXIT_UNEXPECTED
    assert broker.positions == {"SPY": 10.0}
    clock2 = FakeClock(session.open_utc)
    second = _run(
        tmp_path, session,
        broker=broker, execution=FillingExecution(broker), clock=clock2,
        provider=_NoBarsProvider(session, clock=clock2),
    )
    assert second == EXIT_OK
    assert _summary(tmp_path)["exit"] == EXIT_OK
    events = _journal(tmp_path, session)
    starts = [i for i, item in enumerate(events) if item["type"] == "session_start"]
    steps = [item for item in events[starts[-1]:] if item["type"] == "step"]
    flat_at = flatten_time(session, CONFIG.risk.flatten_minutes).isoformat()
    held = [item for item in steps if item["ts"] < flat_at]
    assert held
    assert all(item["nav"] is None for item in held)
    assert all(item["nav_unmarked"] == ["SPY"] for item in held)


# HFT-PAPER-08 al relanzar. El dueño de cada posición sale de las órdenes
# ``h-<estrategia>-…`` llenas hoy (las patas del bracket cuentan para su
# padre). Si no cuadra con el broker, el símbolo queda a nombre de
# reconcile: nadie lo gestiona y lo cierra el flatten de sesión.


class _Gestiona:
    """No abre nada. Desde las 9:40 ET vende lo propio con ``exit_qty``."""

    id = "gestiona"
    params = _Params(10.0)
    theory = _card(10.0)

    def __init__(self) -> None:
        self.seen = []

    def required_history(self):
        return (0, 0)

    def on_bar(self, ctx):
        self.seen.append(ctx.position("SPY"))
        qty = ctx.exit_qty("SPY")
        if ctx.bar_ts.astimezone(NY).time() < time(9, 40) or qty <= 0:
            return []
        return [order_intent(
            self.id, "SPY", "sell", qty, ctx.bar_ts, limit_price=100.0, reason="exit",
        )]


class _CierraTodo(_Gestiona):
    """Cierra en cada barra todo lo que ve, como el ``_flatten`` de ORB."""

    id = "cierra"

    def on_bar(self, ctx):
        out = []
        for symbol, _held in ctx.positions:
            qty = ctx.exit_qty(symbol)
            if qty > 0:
                out.append(order_intent(
                    self.id, symbol, "sell", qty, ctx.bar_ts, limit_price=100.0, reason="todo",
                ))
        return out


def _h_order(owner: str, side: str, qty: float, minute: int, **extra) -> dict:
    stamp = _session().open_utc + timedelta(minutes=minute)
    row = {
        "id": f"o-{owner}-{minute}",
        "client_order_id": make_intent_order_id(owner, "SPY", stamp, side),
        "symbol": "SPY",
        "side": side,
        "qty": qty,
        "filled_qty": str(qty),
        "status": "filled",
        "filled_at": stamp.isoformat(),
    }
    row.update(extra)
    return row


def _owners_events(tmp_path: Path, session: Session) -> list:
    return [item for item in _journal(tmp_path, session) if item["type"] == "owners"]


def test_relanzamiento_recupera_el_dueno_y_solo_el_lo_gestiona(tmp_path, monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = _BracketBroker(positions={"SPY": 10.0}, orders=[_h_order("gestiona", "buy", 10.0, 1)])
    mine = _Gestiona()
    code = _run(
        tmp_path, session, strategies=(mine, _CierraTodo()),
        broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    assert mine.seen[0] == 10.0
    sells = [order["client_order_id"] for order in broker.orders if order["side"] == "sell"]
    assert len(sells) == 1 and sells[0].startswith("h-gestiona-")
    assert abs(broker.positions.get("SPY", 0.0)) < 1e-6
    events = _owners_events(tmp_path, session)
    assert events[-1]["owners"] == {"SPY": {"gestiona": 10.0}}
    assert events[-1]["mismatch"] == {}


def test_relanzamiento_sin_cuadre_deja_reconcile_y_lo_cierra_el_flatten(tmp_path, monkeypatch):
    # Las órdenes de hoy suman 6 y el broker tiene 10: no se adivina el dueño.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    broker = _BracketBroker(positions={"SPY": 10.0}, orders=[_h_order("gestiona", "buy", 6.0, 1)])
    mine = _Gestiona()
    code = _run(
        tmp_path, session, strategies=(mine, _CierraTodo()),
        broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    assert set(mine.seen) == {0.0}
    sells = [order["client_order_id"] for order in broker.orders if order["side"] == "sell"]
    assert len(sells) == 1 and sells[0].startswith("h-flatten-")
    events = _owners_events(tmp_path, session)
    assert events[-1]["owners"] == {"SPY": {"reconcile": 10.0}}
    assert events[-1]["mismatch"] == {"SPY": {"broker": 10.0, "recovered": {"gestiona": 6.0}}}


def test_relanzamiento_la_pata_stop_cuenta_para_su_padre(tmp_path, monkeypatch):
    # El stop del bracket de ORB lleva un UUID de Alpaca, sin "h-": cuelga del padre.
    monkeypatch.delenv("HFT_KILL", raising=False)
    session = _session()
    parent = _h_order("gestiona", "buy", 10.0, 1)
    leg = {
        "id": "leg-1", "client_order_id": "6c1e9f2a-uuid-de-alpaca", "symbol": "SPY",
        "side": "sell", "qty": 10.0, "filled_qty": "4", "status": "partially_filled",
        "filled_at": (session.open_utc + timedelta(minutes=3)).isoformat(),
        "_parent": parent["client_order_id"],
    }
    broker = _BracketBroker(positions={"SPY": 6.0}, orders=[parent, leg])
    mine = _Gestiona()
    code = _run(
        tmp_path, session, strategies=(mine,),
        broker=broker, execution=FillingExecution(broker),
    )
    assert code == EXIT_OK
    assert mine.seen[0] == 6.0
    assert _owners_events(tmp_path, session)[-1]["owners"] == {"SPY": {"gestiona": 6.0}}
