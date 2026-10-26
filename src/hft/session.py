"""Sesión intradía (HFT-12).

Backtest, simulación y paper siguen entrando por ``Engine.step``. Este
módulo es el ciclo de una sesión: preflight, calentamiento, un paso por
minuto, flatten y el journal. No lee el reloj de pared: el ``clock`` lo
inyecta el runner (``scripts/run_intraday.py`` puede usar
``datetime.now``; aquí no).

``kill_flattens`` es verdadero. No está en ``config/hft.toml``: esa tabla
es estricta desde HFT-08 y una clave nueva invalidaría el hash ya
versionado. El corte por KILL o por pérdida diaria cancela y aplana.

Al relanzar, el dueño de cada posición HFT sale de las órdenes
``h-<estrategia>-…`` llenas hoy (HFT-PAPER-08). Lo que no cuadra con el
broker queda a nombre de ``reconcile``: ninguna estrategia lo gestiona y
lo cierra el flatten de sesión.

El HTML vive en ``src.hft.report``. Al cierre se compromete
``hft_summary.json`` y ``hft_book.json``; el reporte de la sesión se
arma con ``python -m src.hft.report --date``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hft.attribution import Attribution
from src.hft.book import HftBook
from src.hft.calendar import NY, Session, SessionCalendar, bar_close_time, flatten_time
from src.hft.config import HftConfig, config_symbols
from src.hft.data.bars import IntradayBars, empty_bars
from src.hft.engine import INHERITED_OWNER, Engine, assert_paper_allowed, execution_mode
from src.hft.execution.sim import SimExecution
from src.hft.risk import (
    AssetFlag,
    EngineRiskGate,
    RiskManager,
    RiskState,
    arm_kill,
    limits_from_account,
    refresh_kill,
)
from src.hft.sizing import IntradaySizer
from src.hft.strategies.base import STRATEGIES
from src.hft.types import QTY_TOLERANCE, Fill, OrderIntent, ensure_utc, order_intent
from src.live_state import StateLockedError, StateStore, new_run_id
from src.notifier import Alert, ConsoleNotifier
from src.portfolio_state import PortfolioState
from src.provenance import code_version, run_provenance
from src.settings import SEED
from src.brokers.pre_trade import kill_switch_reason

logger = logging.getLogger("hft.session")

UTC = timezone.utc

EXIT_OK = 0
EXIT_NO_SESSION = 1
EXIT_GUARD = 2
EXIT_BROKER = 3
EXIT_LOCK = 4
EXIT_UNEXPECTED = 5
EXIT_DATA = 7
EXIT_RISK = 8

# Prefijo de ``make_intent_order_id`` (HFT-11). Una orden abierta sin él
# no es de este runner.
HFT_ORDER_PREFIX = "h-"

# Fallos de API seguidos antes de aplanar y salir 3. No es un parámetro
# de la estrategia: es el corte del proceso.
MAX_API_FAILURES = 3

# Minutos después del cierre real para exigir el libro HFT en cero.
RECONCILE_AFTER_CLOSE = timedelta(minutes=5)

# Ver el docstring del módulo. No es una clave del TOML.
KILL_FLATTENS = True

# Estrategias opt-in (``--with``) que se suman al set por defecto. Ninguna
# es ``research_only``: esa puerta sigue siendo ``--allow-research``.
WITH_STRATEGIES = ("hourly_momentum",)

_QTY_EPS = 1e-8
# ADV enorme y σ 0: sin historia el impacto no explota el simulador.
# El piso de spread del modelo sigue cobrándose.
_NO_HISTORY_ADV = 1.0e12


class SessionAbort(Exception):
    """Salida prevista. ``code`` es el exit de la sesión."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)


@dataclass
class SessionHooks:
    """Ganchos de test. ``after_step`` corre después de escribir el journal."""

    after_step: Optional[Callable[[Mapping], None]] = None


@dataclass
class SessionRequest:
    """Todo lo que una sesión necesita. El reloj y los datos vienen de fuera."""

    config: HftConfig
    mode: str
    session_date: date
    clock: object
    calendar: SessionCalendar
    provider: object
    state_dir: Path
    results_dir: Path
    kill_dir: Path
    submit: bool = False
    until: Optional[time] = None
    adopt: bool = False
    allow_research: bool = False
    strategies: Optional[Sequence] = None
    execution: Optional[object] = None
    broker: Optional[object] = None
    notifier: Optional[object] = None
    symbols: Optional[Sequence[str]] = None
    risk_state: Optional[RiskState] = None
    hooks: Optional[SessionHooks] = None
    seed: int = SEED
    max_api_failures: int = MAX_API_FAILURES
    kill_flattens: bool = KILL_FLATTENS
    expected_account_id: Optional[str] = None
    # ConsolidatedCloses (cierres SIP diarios) del momentum. Sin ellos no opera.
    momentum_closes: Optional[object] = None
    # Ids de WITH_STRATEGIES que se suman al set por defecto (``--with``).
    with_strategies: Tuple[str, ...] = ()


@dataclass
class _Queued:
    intent: OrderIntent
    ready_ts: pd.Timestamp


class _DryExecution:
    """Anota intenciones y no llama al broker. Paper sin ``--submit``."""

    def __init__(self) -> None:
        self.submitted: list = []
        self._open: dict = {}

    def submit(self, intents, *, now, bars):
        del now, bars
        fresh = []
        for intent in intents:
            if intent.intent_id in self._open:
                continue
            self._open[intent.intent_id] = intent
            self.submitted.append(intent)
            fresh.append(intent)
        return ()

    def open_orders(self):
        return tuple(self._open.values())

    def cancel(self, intent_id: str) -> bool:
        return self._open.pop(intent_id, None) is not None

    def poll(self):
        return ()


class _RememberingExecution:
    """No reenvía un ``intent_id`` que el journal ya marcó como enviado.

    Lo que va a enviar queda en el journal (evento ``submit``) antes de
    llamar al adaptador. Si el proceso cae entre el POST y el evento
    ``step``, el relanzamiento reconoce la posición como propia y no
    reenvía la orden (2026-10-12: SPY quedó "ajena").
    """

    def __init__(self, inner: object, sent: set, journal_path: Optional[Path] = None) -> None:
        self.inner = inner
        self.sent = sent
        self.journal_path = journal_path
        self.last_sent: list = []

    def submit(self, intents, *, now, bars):
        fresh = [intent for intent in intents if intent.intent_id not in self.sent]
        if fresh and self.journal_path is not None:
            _append(self.journal_path, {
                "type": "submit", "ts": _iso(now), "sent": _intent_rows(fresh),
            })
        fills = self.inner.submit(tuple(fresh), now=now, bars=bars)
        for intent in fresh:
            self.sent.add(intent.intent_id)
        self.last_sent = list(fresh)
        return fills

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class _SessionRisk:
    """Actualiza marcas y el KILL, y delega en la puerta del motor.

    El motor filtra con la hora de decisión de la barra (cierre + gracia).
    Si el reloj de la sesión va más atrasado que ``stale_s`` respecto de
    ese cierre, la barra se presenta como vieja: si no, un catch-up
    aprobaría entradas con datos que ya no sirven. El cierre que cuenta
    es el de la barra evaluada (``now - grace_s``), no el de la última
    del box: al arrancar tarde, el motor recorre el atraso con el box ya
    puesto en la barra más nueva.
    """

    def __init__(
        self,
        gate: EngineRiskGate,
        box: dict,
        kill_dir: Path,
        *,
        stale_s: float,
        grace_s: float,
        arm: bool,
    ) -> None:
        self.gate = gate
        self.box = box
        self.kill_dir = Path(kill_dir)
        self.stale_s = float(stale_s)
        self.grace_s = float(grace_s)
        self.arm = bool(arm)
        self.last = None

    def filter(self, intents, state, now):
        day = ensure_utc(now, "now").astimezone(NY).date()
        pnl_now = float(state.realized_pnl) + float(state.unrealized_pnl)
        start_pnl = self.gate.state.session_start_pnl
        if self.gate.state.session_date != day:
            start_pnl = pnl_now
        close = self.box.get("last_bar_close")
        arrived = self.box.get("arrived")
        if close is not None and arrived is not None:
            evaluated = min(
                ensure_utc(close, "close"),
                ensure_utc(now, "now") - timedelta(seconds=self.grace_s),
            )
            age = (ensure_utc(arrived, "arrived") - evaluated).total_seconds()
            if age > self.stale_s:
                close = ensure_utc(now, "now") - timedelta(seconds=self.stale_s + 1)
        self.gate.state = replace(
            self.gate.state,
            marks=tuple(self.box.get("marks") or ()),
            last_bar_close=close,
            session_start_pnl=start_pnl,
            buying_power=max(0.0, float(state.cash)),
        )
        self.gate.state = refresh_kill(self.gate.state, self.kill_dir)
        result = self.gate.filter(intents, state, now)
        self.last = result
        if result.kill and self.arm:
            arm_kill(result, self.kill_dir, live=True)
        return result


def run_session(request: SessionRequest) -> int:
    """Corre una sesión. Los códigos están en las constantes ``EXIT_*``."""
    notifier = request.notifier if request.notifier is not None else ConsoleNotifier()
    try:
        return _run(request, notifier)
    except SessionAbort as exc:
        _alert(notifier, _level_for(exc.code), "Sesión intradía", str(exc), exc.code)
        return exc.code
    except StateLockedError as exc:
        _alert(notifier, "warning", "Lock intradía", str(exc), EXIT_LOCK)
        return EXIT_LOCK
    except Exception as exc:
        logger.exception("sesión intradía")
        _alert(
            notifier, "critical", "Error inesperado",
            f"{type(exc).__name__}: {exc}", EXIT_UNEXPECTED,
        )
        return EXIT_UNEXPECTED


def decision_times(session: Session, timeframe: str, grace_s: float) -> Tuple[datetime, ...]:
    """Instante en que cada barra de la sesión ya es final (cierre + gracia)."""
    from src.hft.calendar import expected_bars

    out = []
    for stamp in expected_bars(session, timeframe):
        start = pd.Timestamp(stamp).to_pydatetime()
        out.append(bar_close_time(start, timeframe) + timedelta(seconds=float(grace_s)))
    return tuple(out)


def _run(request: SessionRequest, notifier) -> int:
    mode = _runner_mode(request.mode)
    if mode == "paper" and request.allow_research:
        raise SessionAbort(
            EXIT_GUARD,
            "--allow-research está prohibido en paper",
        )
    if request.submit and mode != "paper":
        raise SessionAbort(EXIT_GUARD, "--submit solo aplica a --mode paper")
    try:
        _check_with_strategies(request.with_strategies)
    except ValueError as exc:
        raise SessionAbort(EXIT_GUARD, str(exc)) from exc
    session = _session_or_abort(request)
    store = StateStore(Path(request.state_dir))
    store.acquire()
    try:
        return _locked(request, notifier, store, session, mode)
    finally:
        store.release()


def _locked(request, notifier, store: StateStore, session: Session, mode: str) -> int:
    config = request.config
    symbols = _symbols(request)
    adopted, hft_positions, hft_entries, account, own_fills = _preflight(
        request, session, mode, symbols,
    )
    seed_prices = _seed_prices(request, session, hft_positions, hft_entries)
    journal_path = _journal_path(request.state_dir, session.date)
    sent, owned = _load_journal(journal_path)
    owned.update(hft_positions)
    rng = np.random.default_rng(request.seed)
    strategies = list(request.strategies) if request.strategies is not None else _default_strategies(
        config, request.calendar.sessions, include_research=request.allow_research,
        momentum_closes=request.momentum_closes, with_strategies=request.with_strategies,
    )
    if not strategies:
        raise SessionAbort(EXIT_GUARD, "no hay estrategias para la sesión")
    engine_mode = "paper" if mode == "paper" else "sim"
    try:
        assert_paper_allowed(strategies, engine_mode)
    except ValueError as exc:
        raise SessionAbort(EXIT_GUARD, str(exc)) from exc
    inner = request.execution if request.execution is not None else _build_execution(
        request, config, mode, rng, symbols,
    )
    execution = _RememberingExecution(inner, sent, journal_path)
    gate, _state = _build_risk(request, config, account, symbols, engine_mode)
    box = {"marks": (), "last_bar_close": None, "arrived": None}
    engine = Engine(
        strategies,
        lambda bars: None,
        _SessionRisk(
            gate, box, Path(request.kill_dir),
            stale_s=config.risk.stale_data_s,
            grace_s=config.grace_s,
            arm=(mode == "paper"),
        ),
        IntradaySizer(
            config.sizing,
            max_position_notional=config.risk.max_position_notional,
        ),
        execution,
        request.calendar,
        request.clock,
        book=HftBook(config.initial_cash),
        config=config,
        rng=rng,
        mode=engine_mode,
    )
    owners = _recover_owners(
        own_fills, [strategy.id for strategy in engine.strategies], hft_positions, journal_path,
    )
    _seed_book(engine, owners, seed_prices, session.open_utc)
    _seed_execution(execution, hft_positions)
    _warmup(engine, request, session, symbols)
    provenance = run_provenance(code_version(), data_source=str(config.feed))
    run_id = new_run_id(request.clock.now())
    _append(journal_path, {
        "type": "session_start",
        "run_id": run_id,
        "mode": mode,
        "submit": bool(request.submit),
        "date": session.date.isoformat(),
        "config_hash": config.config_hash(),
        "adopted": sorted(adopted),
        "provenance": provenance,
        "kill_flattens": bool(request.kill_flattens),
    })
    _alert(
        notifier, "info", "Sesión intradía",
        f"{mode} {session.date.isoformat()} submit={bool(request.submit)}",
        None, run_id=run_id,
    )
    code = _loop(
        request, notifier, engine, gate, box, session, symbols,
        adopted, journal_path, mode, run_id,
    )
    _finish(request, store, engine, session, mode, provenance, run_id, code, journal_path, box)
    nav, unmarked = _eod_nav(engine, box)
    _alert(
        notifier,
        "critical" if code == EXIT_OK and _still_open(engine, request, adopted, hft_positions) else "info",
        "Resumen EOD",
        f"exit {code} nav={nav:.2f}" if nav is not None else f"exit {code} nav=null sin marca: {unmarked}",
        code, run_id=run_id,
    )
    return code


def _loop(request, notifier, engine, gate, box, session, symbols, adopted, journal_path, mode, run_id) -> int:
    config = request.config
    decisions = list(decision_times(session, config.timeframe, config.grace_s))
    reconcile_at = session.close_utc + RECONCILE_AFTER_CLOSE
    if not decisions or decisions[-1] < reconcile_at:
        decisions.append(reconcile_at)
    flat_at = flatten_time(session, config.risk.flatten_minutes)
    flattened = False
    failures = 0
    signals: dict = {}
    step_index = 0
    for decision in decisions:
        if request.until is not None and _past_until(decision, session.date, request.until):
            _flatten(request, engine, session, adopted, decision, journal_path, "until")
            break
        _wait(request.clock, decision)
        now = ensure_utc(request.clock.now(), "now")
        box["arrived"] = now
        _arm_session_pnl(gate, engine, session.date)
        reason = _risk_stop(request, gate, engine, now)
        if reason is not None and request.kill_flattens:
            _flatten(request, engine, session, adopted, now, journal_path, reason)
            _alert(notifier, "critical", "Corte por riesgo", reason, EXIT_RISK, run_id=run_id)
            _append(journal_path, {"type": "risk_stop", "ts": _iso(now), "reason": reason})
            return EXIT_RISK
        try:
            bars = _fetch(request, symbols, session, now)
        except Exception as exc:
            failures += 1
            _append(journal_path, {
                "type": "api_error", "ts": _iso(now),
                "error": f"{type(exc).__name__}: {exc}", "failures": failures,
            })
            _alert(notifier, "error", "Fallo de API", str(exc), None, run_id=run_id)
            if failures >= int(request.max_api_failures):
                _flatten(request, engine, session, adopted, now, journal_path, "api")
                _alert(
                    notifier, "error", "Fallo de API",
                    f"{failures} fallos seguidos; flatten y salida",
                    EXIT_BROKER, run_id=run_id,
                )
                return EXIT_BROKER
            _wait(request.clock, now + timedelta(seconds=min(2 ** failures, 30)))
            continue
        failures = 0
        _update_box(box, bars, config.timeframe, config.grace_s, now)
        result = engine.step(now, bars)
        _poll_into_book(engine, mode, request.submit)
        sent_rows = _sent_rows(engine.execution)
        _collect_signals(signals, result)
        nav, unmarked = _eod_nav(engine, box)
        _append(journal_path, {
            "type": "step",
            "ts": _iso(now),
            "step": step_index,
            "sent": sent_rows,
            "rejections": [
                {"intent_id": event.intent_id, "reason": event.reason, "symbol": event.symbol}
                for event in result.rejections
            ],
            # Lo que el motor canceló porque la estrategia lo pidió (HFT-PAPER-16).
            "cancelled": [
                intent_id for event in result.events for intent_id in event.cancelled
            ],
            "nav": nav,
            "nav_unmarked": unmarked,
        })
        _heartbeat(journal_path, now, engine, box)
        if mode == "signals":
            _write_signals(request, session.date, signals)
        hook = None if request.hooks is None else request.hooks.after_step
        if hook is not None:
            hook({"step": step_index, "sent": sent_rows, "now": now})
        step_index += 1
        if now >= flat_at and not flattened:
            _flatten(request, engine, session, adopted, now, journal_path, "flatten_time")
            flattened = True
        if now >= reconcile_at:
            _reconcile(request, notifier, engine, adopted, now, run_id, journal_path)
            break
    if mode == "signals":
        _write_signals(request, session.date, signals)
    return EXIT_OK


def _preflight(request, session, mode, symbols):
    _probe(request, symbols, session)
    if mode != "paper":
        return set(), {}, {}, None, ()
    if request.broker is None:
        raise SessionAbort(EXIT_BROKER, "paper necesita un broker")
    try:
        summary = request.broker.check()
    except Exception as exc:
        raise SessionAbort(EXIT_BROKER, f"check de cuenta: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise SessionAbort(EXIT_BROKER, "check de cuenta no devolvió un mapping")
    if summary.get("account_blocked") or summary.get("trading_blocked"):
        raise SessionAbort(EXIT_BROKER, "cuenta o trading bloqueados")
    expected = (request.expected_account_id or "").strip()
    actual = str(summary.get("account_number") or "").strip()
    if expected and actual != expected:
        raise SessionAbort(
            EXIT_BROKER,
            f"BROKER_ACCOUNT_ID {expected} no coincide con {actual or '(vacío)'}",
        )
    try:
        portfolio = PortfolioState.from_broker(
            request.broker, as_of=pd.Timestamp(session.open_utc),
        )
    except Exception as exc:
        raise SessionAbort(EXIT_BROKER, f"PortfolioState.from_broker: {exc}") from exc
    orders = _open_orders(request.broker)
    legs, filled, own_fills = _own_orders(
        request.broker, datetime.combine(session.date, time(0), tzinfo=NY),
    )
    journal_path = _journal_path(request.state_dir, session.date)
    _, owned = _load_journal(journal_path)
    owned.update(filled)
    foreign_orders = []
    for order in orders:
        cid = str(order.get("client_order_id") or "")
        if cid.startswith(HFT_ORDER_PREFIX) or cid in legs or str(order.get("id") or "") in legs:
            symbol = str(order.get("symbol") or order.get("ticker") or "").upper()
            if symbol:
                owned.add(symbol)
            continue
        foreign_orders.append(order)
    foreign_symbols = []
    hft_positions = {}
    hft_entries = {}
    positions = portfolio.positions
    if positions is not None and len(positions):
        for symbol, qty in positions.items():
            amount = float(qty)
            if abs(amount) <= _QTY_EPS:
                continue
            name = str(symbol).strip().upper()
            if name in owned:
                hft_positions[name] = amount
                hft_entries[name] = _avg_entry(portfolio.avg_entry, symbol)
            else:
                foreign_symbols.append(name)
    adopted = set()
    if foreign_orders or foreign_symbols:
        if not request.adopt:
            # Guarda, no fallo del broker: relanzar no la arregla (HFT-PAPER-05).
            raise SessionAbort(
                EXIT_GUARD,
                "posición u orden ajena al intradía (sin --adopt): "
                f"símbolos={sorted(set(foreign_symbols))} "
                f"órdenes={len(foreign_orders)}",
            )
        adopted.update(foreign_symbols)
        for order in foreign_orders:
            symbol = str(order.get("symbol") or order.get("ticker") or "").upper()
            if symbol:
                adopted.add(symbol)
        _append(_journal_path(request.state_dir, session.date), {
            "type": "adopt",
            "symbols": sorted(adopted),
            "orders": [str(order.get("client_order_id") or "") for order in foreign_orders],
        })
    account = _account_mapping(summary, portfolio)
    return adopted, hft_positions, hft_entries, account, own_fills


def _avg_entry(entries, symbol) -> Optional[float]:
    """``avg_entry`` del broker, o None si falta (Alpaca manda 0.0)."""
    try:
        value = float(entries.get(symbol))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _probe(request, symbols, session: Session) -> None:
    end = session.open_utc + timedelta(minutes=1)
    try:
        request.provider.get_bars(list(symbols), session.open_utc, end, request.config.timeframe)
    except Exception as exc:
        status = getattr(exc, "status", None)
        why = "credenciales rechazadas" if status in (401, 403) else "feed no alcanzable"
        raise SessionAbort(EXIT_DATA, f"{why}: {exc}") from exc


def _fetch(request, symbols, session: Session, now: datetime) -> IntradayBars:
    start = session.open_utc
    end = now if now > start else start + timedelta(minutes=1)
    bars = request.provider.get_bars(list(symbols), start, end, request.config.timeframe)
    if not isinstance(bars, IntradayBars):
        raise TypeError("el proveedor no devolvió IntradayBars")
    return bars


def _warmup(engine: Engine, request, session: Session, symbols) -> None:
    need_bars, need_sessions = engine.required_history()
    if need_bars <= 0 and need_sessions <= 0:
        return
    prior = [item for item in request.calendar.sessions if item.date < session.date]
    if not prior:
        return
    extra = max(int(need_sessions), 1)
    for item in prior[-extra:]:
        try:
            bars = request.provider.get_bars(
                list(symbols), item.open_utc, item.close_utc, request.config.timeframe,
            )
        except Exception as exc:
            raise SessionAbort(EXIT_DATA, f"calentamiento: {exc}") from exc
        if bars.close.empty:
            continue
        ready = item.close_utc + timedelta(seconds=request.config.grace_s)
        engine._ingest(bars, ready)


def _flatten(request, engine, session, adopted, now, journal_path, why: str) -> None:
    execution = engine.execution
    listing = getattr(execution, "open_orders", None)
    cancel = getattr(execution, "cancel", None)
    if callable(listing) and callable(cancel):
        for intent in list(listing()):
            cancel(intent.intent_id)
    broker = request.broker
    # Sin --submit la sesión no envió nada: el dry run no toca la cuenta.
    if broker is not None and request.submit:
        method = getattr(broker, "cancel_all_orders", None)
        if not callable(method):
            method = getattr(broker, "cancel_all", None)
        if callable(method):
            method()
    positions = _flat_targets(engine, request, adopted)
    if not positions:
        _append(journal_path, {"type": "flatten", "ts": _iso(now), "why": why, "sent": []})
        return
    anchor = session.open_utc
    frames = engine._frames
    if frames is not None and not frames["close"].empty:
        anchor = pd.Timestamp(frames["close"].index[-1]).to_pydatetime()
    intents = []
    for seq, symbol in enumerate(sorted(positions)):
        qty = positions[symbol]
        side = "sell" if qty > 0 else "buy"
        intents.append(order_intent(
            "flatten", symbol, side, abs(qty), anchor,
            order_type="market", reason=why, seq=seq,
        ))
    for intent in intents:
        engine._live[intent.intent_id] = intent
    window = _window(engine, request, now)
    fills = execution.submit(tuple(intents), now=ensure_utc(now, "now"), bars=window)
    if fills:
        engine._accept((), fills)
    _append(journal_path, {
        "type": "flatten",
        "ts": _iso(now),
        "why": why,
        "sent": _sent_rows(execution),
    })


def _reconcile(request, notifier, engine, adopted, now, run_id, journal_path) -> None:
    left = _flat_targets(engine, request, adopted)
    _append(journal_path, {
        "type": "reconcile",
        "ts": _iso(now),
        "positions": {symbol: qty for symbol, qty in left.items()},
    })
    if left:
        _alert(
            notifier, "critical", "Reconciliación HFT",
            f"posiciones distintas de cero: {left}",
            None, run_id=run_id,
        )


def _finish(request, store, engine, session, mode, provenance, run_id, code, journal_path, box) -> None:
    nav, unmarked = _eod_nav(engine, box)
    if unmarked:
        logger.warning("NAV EOD sin marca para %s: hft_summary.json la deja en null", unmarked)
    summary = {
        "session_date": session.date.isoformat(),
        "mode": mode,
        "submit": bool(request.submit),
        "exit": int(code),
        "config_hash": request.config.config_hash(),
        "provenance": provenance,
        "kill_flattens": bool(request.kill_flattens),
        "nav": nav,
        "nav_unmarked": unmarked,
        "positions": engine.book.positions(),
        "journal": str(journal_path),
        "report": "HFT-13: python -m src.hft.report --date",
    }
    try:
        with store.transaction(run_id, as_of=session.date.isoformat()):
            store.write_json("hft_summary.json", _jsonable(summary))
            store.write_json("hft_book.json", _jsonable(engine.book.to_dict()))
    except Exception as exc:
        logger.warning("no se pudo comprometer el resumen EOD: %s", exc)


def _build_execution(request, config, mode, rng, symbols):
    if mode == "sim":
        sim = SimExecution.from_config(config, rng=rng)
        sim._adv = {symbol: _NO_HISTORY_ADV for symbol in symbols}
        sim._sigma = {symbol: 0.0 for symbol in symbols}
        return sim
    return _DryExecution()


def _build_risk(request, config, account, symbols, engine_mode):
    if request.risk_state is not None:
        state = request.risk_state
    else:
        equity = float(config.initial_cash)
        buying = equity
        if account is not None:
            equity, buying = limits_from_account(account)
        if engine_mode != "paper":
            flags = tuple((symbol, AssetFlag(True, True)) for symbol in symbols)
        else:
            flags = _broker_asset_flags(request.broker, symbols, request.session_date)
        state = RiskState(
            equity=equity,
            nav=equity,
            buying_power=buying,
            session_start_pnl=0.0,
            cash=equity,
            assets=flags,
        )
    manager = RiskManager(config.risk, request.calendar.sessions)
    gate = EngineRiskGate(manager, state, equity_from_book=(engine_mode != "paper"))
    return gate, state


def _broker_asset_flags(broker, symbols, day) -> Tuple[Tuple[str, AssetFlag], ...]:
    """Fichas de short del día (``GET /v2/assets``) para el riesgo de paper.

    Sin ficha el símbolo queda cerrado a cortos (``short_no_habilitado``):
    un fallo de lectura no abre cortos.
    """
    method = getattr(broker, "get_asset", None)
    if not callable(method):
        return ()
    flags = []
    for symbol in symbols:
        try:
            asset = method(symbol, as_of=day)
        except Exception as exc:
            logger.warning("Ficha de %s no disponible (%s): sin cortos.", symbol, exc)
            continue
        shortable = asset.get("shortable") if isinstance(asset, Mapping) else None
        easy = asset.get("easy_to_borrow") if isinstance(asset, Mapping) else None
        if isinstance(shortable, bool) and isinstance(easy, bool):
            flags.append((symbol, AssetFlag(shortable, easy)))
    return tuple(flags)


def _check_with_strategies(names: Sequence[str]) -> None:
    """ValueError si algún id de ``--with`` no está en ``WITH_STRATEGIES``."""
    unknown = [name for name in names if name not in WITH_STRATEGIES]
    if unknown:
        raise ValueError(f"--with solo admite {list(WITH_STRATEGIES)}; no: {unknown}")


def _default_strategies(
    config, sessions, *, include_research: bool, momentum_closes=None, with_strategies=(),
):
    _check_with_strategies(with_strategies)
    from src.hft.strategies.avellaneda_stoikov import AvellanedaStoikov
    from src.hft.strategies.intraday_momentum import IntradayMomentum
    from src.hft.strategies.orb import OpeningRangeBreakout
    from src.hft.strategies.pairs_ou import PairsOu
    from src.hft.strategies.vwap_reversion import VwapReversion

    if momentum_closes is None:
        logger.warning(
            "intraday_momentum sin cierres SIP diarios: no va a operar en esta sesión.",
        )
    built = [
        OpeningRangeBreakout.from_config(config, sessions),
        IntradayMomentum.from_config(config, sessions, closes=momentum_closes),
        VwapReversion.from_config(config, sessions, shortable=config.universe),
        PairsOu.from_config(config, sessions, shortable=config.universe),
    ]
    if include_research:
        built.append(AvellanedaStoikov.from_config(config, sessions))
    if "hourly_momentum" in with_strategies:
        from src.hft.strategies.hourly_momentum import HourlyMomentum

        built.append(HourlyMomentum.from_config(config, sessions))
    return built


def _session_or_abort(request: SessionRequest) -> Session:
    noon = datetime(
        request.session_date.year, request.session_date.month, request.session_date.day,
        16, 0, tzinfo=NY,
    )
    session = request.calendar.session_for(noon)
    if session is None or session.date != request.session_date:
        raise SessionAbort(
            EXIT_NO_SESSION,
            f"sin sesión el {request.session_date.isoformat()}",
        )
    return session


def _symbols(request: SessionRequest) -> Tuple[str, ...]:
    if request.symbols is not None:
        names = [str(symbol).strip().upper() for symbol in request.symbols]
    else:
        names = list(config_symbols(request.config))
    out = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    if not out:
        raise SessionAbort(EXIT_DATA, "universo vacío")
    return tuple(out)


def _runner_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise SessionAbort(EXIT_GUARD, "mode debe ser str")
    name = mode.strip().lower()
    if name not in ("signals", "sim", "paper"):
        raise SessionAbort(EXIT_GUARD, f"mode {mode!r} no es signals, sim o paper")
    return name


def _arm_session_pnl(gate, engine, day: date) -> None:
    if gate.state.session_date == day:
        return
    pnl = float(engine.book.realized_pnl)
    gate.state = replace(gate.state, session_start_pnl=pnl, session_date=day)


def _risk_stop(request, gate, engine, now: datetime) -> Optional[str]:
    reason = kill_switch_reason(Path(request.kill_dir))
    if reason:
        return reason
    pnl = float(engine.book.realized_pnl)
    if engine.book.positions():
        try:
            pnl += float(engine.book.unrealized_pnl(_marks_from_frames(engine)))
        except ValueError:
            pnl += 0.0
    loss = max(0.0, -(pnl - float(gate.state.session_start_pnl)))
    equity = float(gate.state.equity)
    config = request.config.risk
    if loss + 1e-8 >= config.max_daily_loss_usd:
        return (
            f"max_daily_loss: {loss:.2f} USD >= {config.max_daily_loss_usd:.2f}"
        )
    if equity > 0 and loss / equity + 1e-12 >= config.max_daily_loss_pct:
        return (
            f"max_daily_loss: {loss / equity:.4%} >= {config.max_daily_loss_pct:.4%}"
        )
    return None


def _update_box(box, bars: IntradayBars, timeframe: str, grace_s: float, now: datetime) -> None:
    from src.hft.calendar import is_bar_final

    if bars.close.empty:
        return
    final = []
    for stamp in bars.close.index:
        start = pd.Timestamp(stamp).to_pydatetime()
        if is_bar_final(start, now, timeframe, grace_s):
            final.append(pd.Timestamp(stamp))
    if not final:
        return
    last = max(final)
    box["last_bar_close"] = bar_close_time(last.to_pydatetime(), timeframe)
    row = bars.close.loc[last]
    # IEX deja minutos sin barra (TLT: 71 de 390 el 2026-10-09). El símbolo que
    # no trae barra conserva su última marca; si se borrara, la NAV de una
    # posición abierta no tendría precio (HFT-PAPER-03).
    marks = dict(box.get("marks") or ())
    for symbol in row.index:
        value = row[symbol]
        if pd.notna(value) and float(value) > 0:
            marks[str(symbol)] = float(value)
    if marks:
        box["marks"] = tuple(sorted(marks.items()))


def _poll_into_book(engine, mode: str, submit: bool) -> None:
    if mode != "paper" or not submit:
        return
    poll = getattr(engine.execution, "poll", None)
    if not callable(poll):
        return
    fills = tuple(poll())
    known = [fill for fill in fills if fill.intent_id in engine._live]
    if known:
        engine._accept((), known)


def _seed_prices(request, session: Session, positions: Mapping[str, float], entries) -> dict:
    """Costo con el que el libro adopta las posiciones HFT al relanzar.

    Manda el ``avg_entry`` del broker. Si falta, el primer cierre final de
    la sesión, con aviso. Sin ninguno de los dos sale con ``EXIT_DATA`` (se
    relanza): sembrar a $1 inflaba la NAV y el aplanado realizaba ese P&L
    falso (paper 2026-10-12).
    """
    prices = {symbol: entries[symbol] for symbol in positions if entries.get(symbol) is not None}
    missing = sorted(set(positions) - set(prices))
    if not missing:
        return prices
    now = ensure_utc(request.clock.now(), "now")
    try:
        bars = _fetch(request, missing, session, now)
    except Exception as exc:
        raise SessionAbort(
            EXIT_DATA, f"sin avg_entry ni barras para sembrar {missing}: {exc}",
        ) from exc
    first = _first_final_closes(bars, request.config.timeframe, request.config.grace_s, now)
    lacking = [symbol for symbol in missing if symbol not in first]
    if lacking:
        raise SessionAbort(EXIT_DATA, f"sin avg_entry ni barra final para sembrar {lacking}")
    for symbol in missing:
        logger.warning(
            "%s sin avg_entry del broker: se siembra al primer cierre %.4f",
            symbol, first[symbol],
        )
        prices[symbol] = first[symbol]
    return prices


def _first_final_closes(bars: IntradayBars, timeframe: str, grace_s: float, now: datetime) -> dict:
    from src.hft.calendar import is_bar_final

    close = bars.close.sort_index()
    final = [
        stamp for stamp in close.index
        if is_bar_final(pd.Timestamp(stamp).to_pydatetime(), now, timeframe, grace_s)
    ]
    out = {}
    for symbol in close.columns:
        valid = close.loc[final, symbol]
        valid = valid[valid > 0]
        if len(valid):
            out[str(symbol).strip().upper()] = float(valid.iloc[0])
    return out


def _seed_book(
    engine: Engine,
    owners: Mapping[str, Mapping[str, float]],
    prices: Mapping[str, float],
    ts: datetime,
) -> None:
    """Siembra libro y dueños con ``Engine.seed``, al costo de ``_seed_prices``.

    Una fila por símbolo y dueño. La intención lleva el dueño como
    ``strategy_id``: la estrategia recupera lo suyo y lo gestiona.
    """
    intents = []
    fills = []
    seq = 0
    for symbol in sorted(owners):
        price = float(prices[symbol])
        for owner, qty in sorted(owners[symbol].items()):
            qty = float(qty)
            if abs(qty) <= _QTY_EPS:
                continue
            side = "buy" if qty > 0 else "sell"
            intent = order_intent(owner, symbol, side, abs(qty), ts, reason="reinicio", seq=seq)
            seq += 1
            fills.append(Fill(intent.intent_id, ts, abs(qty), price, 0.0, "taker"))
            intents.append(intent)
    if fills:
        engine.seed(intents, fills)


def _recover_owners(
    rows: Sequence[Tuple[str, str, str, float]],
    strategy_ids: Sequence[str],
    positions: Mapping[str, float],
    journal_path: Path,
) -> dict:
    """Dueño de cada posición HFT heredada: ``símbolo → {dueño: qty}``.

    Reproduce en orden de hora los fills de hoy de las órdenes ``h-`` con
    la regla de ``Attribution``: un ``flatten`` previo reduce a los
    dueños, y un dueño que no corre en esta sesión (hourly sin ``--with``)
    cuenta como de la casa. Si lo recuperado suma lo que tiene el broker,
    siembra una fila por dueño. Si no, el símbolo entero queda a nombre de
    ``reconcile``, con aviso en el log: nadie lo gestiona, bloquea
    entradas ajenas y lo cierra el flatten. El evento ``owners`` del
    journal deja las dos cosas.
    """
    if not positions:
        return {}
    known = set(STRATEGIES) | set(strategy_ids) | {"flatten", INHERITED_OWNER}
    replay = Attribution(strategy_ids)
    for _when, client_id, symbol, qty in sorted(rows, key=lambda row: row[0]):
        replay.move(_owner_of(client_id, known), symbol, qty)
    owners = {}
    mismatch = {}
    for symbol in sorted(positions):
        held = float(positions[symbol])
        found = replay.owners(symbol)
        if found and abs(sum(found.values()) - held) <= QTY_TOLERANCE:
            owners[symbol] = found
        else:
            owners[symbol] = {INHERITED_OWNER: held}
            mismatch[symbol] = {"broker": held, "recovered": found}
    if mismatch:
        logger.warning(
            "las órdenes h- de hoy no cuadran con el broker en %s: quedan a nombre de %s",
            ", ".join(sorted(mismatch)), INHERITED_OWNER,
        )
    _append(journal_path, {"type": "owners", "owners": owners, "mismatch": mismatch})
    return owners


def _owner_of(client_id: str, known) -> str:
    """Dueño de ``h-<id>-…``: el id conocido más largo que calce, o el primer tramo.

    La forma con sha1 de ``make_intent_order_id`` conserva 37 caracteres;
    el prefijo más largo de hoy (``h-avellaneda_stoikov-``) ocupa 21.
    """
    body = client_id[len(HFT_ORDER_PREFIX):]
    matches = [name for name in known if body.startswith(f"{name}-")]
    if matches:
        return max(matches, key=len)
    return body.split("-", 1)[0]


def _seed_execution(execution, positions: Mapping[str, float]) -> None:
    """El adaptador ve la misma posición propia que el libro (HFT-PAPER-17).

    Sin esto, la venta que cierra un largo heredado sale como corto.
    """
    seed = getattr(execution, "seed_positions", None)
    if callable(seed) and positions:
        seed(positions)


def _flat_targets(engine, request, adopted: set) -> dict:
    positions = {}
    for symbol, qty in engine.book.positions().items():
        if symbol in adopted or abs(float(qty)) <= _QTY_EPS:
            continue
        positions[symbol] = float(qty)
    broker = request.broker
    if broker is None:
        return positions
    try:
        portfolio = PortfolioState.from_broker(broker)
    except Exception:
        return positions
    if portfolio.positions is None:
        return positions
    for symbol, qty in portfolio.positions.items():
        name = str(symbol).strip().upper()
        if name in adopted or abs(float(qty)) <= _QTY_EPS:
            continue
        if name not in positions:
            positions[name] = float(qty)
    return positions


def _still_open(engine, request, adopted, hft_positions) -> dict:
    del hft_positions
    return _flat_targets(engine, request, adopted)


def _window(engine, request, now: datetime) -> IntradayBars:
    frames = engine._frames
    if frames is None or frames["close"].empty:
        return empty_bars(
            request.config.universe, source="session", feed=request.config.feed,
            timeframe=request.config.timeframe,
        )
    columns = tuple(str(column) for column in frames["close"].columns)
    meta_frames = {name: frames[name].copy() for name in frames}
    from src.hft.data.bars import IntradayMeta

    meta = IntradayMeta(
        source="session", feed=request.config.feed, timeframe=request.config.timeframe,
        adjustment="raw", requested=columns, received=columns,
    )
    return IntradayBars(meta=meta, **meta_frames)


def _own_orders(broker, since: datetime) -> Tuple[set, set, tuple]:
    """Rastro de las órdenes HFT en el broker desde ``since``.

    Devuelve los ids y client_order_id de las patas de sus brackets, los
    símbolos con alguna orden ``h-`` llena y los fills de hoy de cada
    orden ``h-`` y de sus patas: ``(hora, client_order_id del padre,
    símbolo, qty con signo)``. Alpaca les da a las patas stop y take un
    client_order_id propio (UUID), sin el prefijo ``h-``: solo se
    reconocen por el padre, que con ``nested`` las cuelga en ``legs``
    (HFT-PAPER-04), y su fill cuenta para el dueño del padre
    (HFT-PAPER-08). Una orden que salió y llenó en el paso en que el
    motor reventó no llega al journal; el broker sí la tiene (HFT-PAPER-14).
    """
    method = getattr(broker, "list_orders", None)
    if not callable(method):
        return set(), set(), ()
    try:
        raw = method(status="all", after=since, limit=500, nested=True)
    except TypeError:
        return set(), set(), ()
    ids = set()
    filled = set()
    fills = []
    for order in raw or []:
        client_id = str(order.get("client_order_id") or "")
        if not client_id.startswith(HFT_ORDER_PREFIX):
            continue
        for leg in order.get("legs") or ():
            ids.update(str(leg.get(key) or "") for key in ("id", "client_order_id"))
            row = _filled_row(leg, client_id)
            if row is not None:
                fills.append(row)
        row = _filled_row(order, client_id)
        if row is not None:
            filled.add(row[2])
            fills.append(row)
    ids.discard("")
    return ids, filled, tuple(fills)


def _filled_row(order: Mapping, owner_id: str) -> Optional[Tuple[str, str, str, float]]:
    """``(hora, owner_id, símbolo, qty con signo)`` de lo llenado, o None."""
    try:
        done = float(order.get("filled_qty") or 0.0)
    except (TypeError, ValueError):
        done = 0.0
    symbol = str(order.get("symbol") or "").strip().upper()
    side = str(order.get("side") or "").strip().lower()
    if not symbol or done <= _QTY_EPS or side not in ("buy", "sell"):
        return None
    when = ""
    for key in ("filled_at", "updated_at", "submitted_at", "created_at"):
        if order.get(key):
            stamp = pd.Timestamp(order[key])
            stamp = stamp.tz_localize(UTC) if stamp.tzinfo is None else stamp.tz_convert(UTC)
            # Mismo formato para todas: el orden de texto es el de la hora.
            when = stamp.as_unit("ns").isoformat()
            break
    return when, owner_id, symbol, done if side == "buy" else -done


def _open_orders(broker) -> list:
    method = getattr(broker, "list_orders", None)
    if not callable(method):
        method = getattr(broker, "open_orders", None)
    if not callable(method):
        return []
    try:
        raw = method(status="open", limit=500)
    except TypeError:
        raw = method()
    return list(raw or [])


def _account_mapping(summary: Mapping, portfolio: PortfolioState) -> dict:
    equity = summary.get("equity", portfolio.equity)
    buying = summary.get("buying_power", portfolio.cash)
    return {"equity": equity, "buying_power": buying}


def _journal_path(state_dir: Path, day: date) -> Path:
    return Path(state_dir) / day.isoformat() / "events.jsonl"


def _append(path: Path, event: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(dict(event)), sort_keys=True, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_journal(path: Path) -> Tuple[set, set]:
    sent: set = set()
    owned: set = set()
    if not path.is_file():
        return sent, owned
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        for row in event.get("sent") or []:
            intent_id = str(row.get("intent_id") or "")
            symbol = str(row.get("symbol") or "").upper()
            if intent_id:
                sent.add(intent_id)
            if symbol:
                owned.add(symbol)
        for symbol in event.get("symbols") or []:
            owned.add(str(symbol).upper())
    return sent, owned


def _sent_rows(execution) -> list:
    return _intent_rows(getattr(execution, "last_sent", ()) or ())


def _intent_rows(intents) -> list:
    rows = []
    for intent in intents:
        rows.append({
            "intent_id": intent.intent_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": float(intent.qty),
        })
    return rows


def _collect_signals(signals: dict, result) -> None:
    for event in result.events:
        for intent in event.intents:
            if intent.intent_id in signals:
                continue
            stop = None
            if intent.bracket is not None and intent.bracket.stop is not None:
                stop = float(intent.bracket.stop)
            elif intent.stop_price is not None:
                stop = float(intent.stop_price)
            signals[intent.intent_id] = {
                "symbol": intent.symbol,
                "side": intent.side,
                "qty": float(intent.qty),
                "limit": "" if intent.limit_price is None else float(intent.limit_price),
                "stop": "" if stop is None else stop,
                "reason": intent.reason,
            }
            # En signals la consola solo mostraba heartbeats; la señal iba al CSV.
            logger.info(
                "señal %s %s %g limite=%s stop=%s — %s",
                intent.symbol,
                intent.side,
                float(intent.qty),
                "-" if intent.limit_price is None else f"{float(intent.limit_price):g}",
                "-" if stop is None else f"{stop:g}",
                intent.reason,
            )


def _write_signals(request, day: date, signals: dict) -> None:
    path = Path(request.results_dir) / "hft" / "signals" / f"{day.isoformat()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["symbol,side,qty,limit,stop,reason"]
    for intent_id in sorted(signals):
        row = signals[intent_id]
        reason = str(row["reason"]).replace('"', "'")
        lines.append(
            f"{row['symbol']},{row['side']},{row['qty']},{row['limit']},{row['stop']},\"{reason}\""
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _heartbeat(path: Path, now: datetime, engine, box: dict) -> None:
    _append(path, {
        "type": "heartbeat",
        "ts": _iso(now),
        "cash": float(engine.book.cash),
        "positions": engine.book.positions(),
    })
    logger.info("heartbeat %s cash=%.2f", _iso(now), float(engine.book.cash))


def _wait(clock, target: datetime) -> None:
    target = ensure_utc(target, "target")
    sleeper = getattr(clock, "sleep_until", None)
    if callable(sleeper):
        sleeper(target)
        return
    if ensure_utc(clock.now(), "now") < target:
        raise RuntimeError("el reloj no implementa sleep_until y no alcanzó el instante")


def _past_until(decision: datetime, day: date, until: time) -> bool:
    local = ensure_utc(decision, "decision").astimezone(NY)
    if local.date() != day:
        return False
    return (local.hour, local.minute) > (until.hour, until.minute)


def _marks_dict(box: dict) -> dict:
    return {symbol: price for symbol, price in box.get("marks") or ()}


def _marks_from_frames(engine) -> dict:
    frames = engine._frames
    if frames is None or frames["close"].empty:
        return {}
    last = frames["close"].iloc[-1]
    marks = {}
    for symbol in last.index:
        value = last[symbol]
        if pd.notna(value) and float(value) > 0:
            marks[str(symbol)] = float(value)
    return marks


def _last_marks(engine) -> dict:
    return _marks_from_frames(engine)


def _eod_nav(engine, box: dict) -> Tuple[Optional[float], list]:
    """NAV del libro con la última marca conocida de cada posición abierta.

    El box manda sobre la última fila de los frames: conserva la marca del
    símbolo que se quedó sin barra (HFT-PAPER-03). Si una posición abierta no
    tiene ninguna marca, la NAV es ``None`` y el símbolo va en la lista; nunca
    se valora a $1/acción.
    """
    marks = {**_last_marks(engine), **_marks_dict(box)}
    unmarked = sorted(symbol for symbol in engine.book.positions() if symbol not in marks)
    if unmarked:
        return None, unmarked
    return float(engine.book.nav(marks)), []


def _alert(notifier, level: str, title: str, message: str, code, run_id: Optional[str] = None) -> None:
    try:
        notifier.notify(Alert(level, title, message, run_id=run_id, exit_code=code))
    except Exception as exc:
        logger.warning("alerta no entregada: %s", exc)


def _level_for(code: int) -> str:
    if code in (EXIT_RISK, EXIT_UNEXPECTED):
        return "critical"
    if code in (EXIT_BROKER, EXIT_DATA, EXIT_LOCK):
        return "error"
    return "info"


def _iso(ts: datetime) -> str:
    return ensure_utc(ts, "ts").isoformat()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return float(value)
    return value


def execution_mode_name(mode: str) -> str:
    """``sim`` para signals. Paper se queda en paper. Sirve a los tests."""
    name = _runner_mode(mode)
    if name == "signals":
        return execution_mode("sim")
    return execution_mode(name)


# Reexportado para que un lector vea el modo del motor al lado del runner.
__all__ = [
    "EXIT_BROKER",
    "EXIT_DATA",
    "EXIT_GUARD",
    "EXIT_LOCK",
    "EXIT_NO_SESSION",
    "EXIT_OK",
    "EXIT_RISK",
    "EXIT_UNEXPECTED",
    "HFT_ORDER_PREFIX",
    "KILL_FLATTENS",
    "MAX_API_FAILURES",
    "SessionAbort",
    "SessionHooks",
    "SessionRequest",
    "StateLockedError",
    "config_symbols",
    "decision_times",
    "run_session",
]
