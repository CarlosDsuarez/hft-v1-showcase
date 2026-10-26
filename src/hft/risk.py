"""Riesgo intradía (HFT-08).

``RiskManager.filter`` es puro: mismos intents, mismo ``RiskState`` y
mismo ``now`` dan el mismo resultado. No lee el reloj del sistema, no
escribe el archivo KILL y no llama al broker. El estado (rachas, tasa
de órdenes, buying power que queda, ficha de corto) lo trae el
llamador y se devuelve actualizado en ``FilterResult.state``.

Una entrada es una orden que abre, amplía o cruza por cero. Una salida
solo reduce el absoluto de esa posición. Las salidas no consumen tasa,
buying power ni los topes de exposición. El KILL sí las bloquea: el
cierre lo emite el propio riesgo como ``FLATTEN_ALL``.

La pérdida diaria es el realizado más el no realizado, menos la base
de la sesión (``session_start_pnl``). Si cruza el tope en USD o el
tope en fracción del equity, se rechazan las entradas, se emite
``FLATTEN_ALL`` y ``kill`` queda en True. Escribir el archivo es
``arm_kill(..., live=True)``. El backtest no lo llama.

El buying power es el que devuelve la cuenta. No se leen
``daytrade_count`` ni ``pattern_day_trader`` (retirados en jul-2026).
El corto exige ``shortable`` y ``easy_to_borrow`` en la ficha del
estado (la cache de ``/v2/assets`` la llena HFT-11) y acciones enteras.

Antes de dar por bueno un lote de entradas se pasa ``PreTradeGuard``.
Sus defaults de cartera no se tocan: el guard intradía nace de
``[risk]`` y deja en None el tope de nocional total del runner diario.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.brokers.pre_trade import KILL_FILE, PreTradeGuard, kill_switch_reason
from src.hft.calendar import NY, Session, entry_cutoff, flatten_time, in_session, session_for
from src.hft.config import RiskConfig
from src.hft.types import OrderIntent, RiskEvent, ensure_utc, order_intent

FLATTEN_ALL = "FLATTEN_ALL"
KILL_KIND = "KILL"

# Alpaca los retiró de la cuenta el 2026-07-06. No leerlos.
RETIRED_PDT_FIELDS = ("daytrade_count", "pattern_day_trader")

_EPS = 1e-8


@dataclass(frozen=True)
class AssetFlag:
    """Ficha del día para un símbolo. La llena la cache de HFT-11."""

    shortable: bool
    easy_to_borrow: bool

    def __post_init__(self) -> None:
        if type(self.shortable) is not bool or type(self.easy_to_borrow) is not bool:
            raise TypeError("shortable y easy_to_borrow deben ser bool")


@dataclass(frozen=True)
class RiskState:
    """Foto explícita que ve el riesgo. El manager no guarda otra.

    ``last_bar_close`` es el cierre de la última barra final (inicio +
    timeframe), no el inicio. ``session_start_pnl`` es
    realizado + no realizado en la apertura de la sesión. ``spread_bps``
    es el spread estimado que ya calculó el modelo de costos.
    """

    equity: float
    nav: float
    buying_power: float
    session_start_pnl: float
    cash: float = 0.0
    positions: Tuple[Tuple[str, float], ...] = ()
    marks: Tuple[Tuple[str, float], ...] = ()
    spread_bps: Tuple[Tuple[str, float], ...] = ()
    assets: Tuple[Tuple[str, AssetFlag], ...] = ()
    open_orders: int = 0
    orders_this_minute: int = 0
    orders_today: int = 0
    minute_bucket: Optional[datetime] = None
    session_date: Optional[date] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None
    last_bar_close: Optional[datetime] = None
    kill_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("equity", "nav", "buying_power", "cash"):
            number = _finite(name, getattr(self, name))
            if number < 0 and name != "nav":
                raise ValueError(f"{name} no puede ser negativo")
            object.__setattr__(self, name, number)
        object.__setattr__(self, "session_start_pnl", _finite(
            "session_start_pnl", self.session_start_pnl,
        ))
        object.__setattr__(self, "realized_pnl", _finite("realized_pnl", self.realized_pnl))
        object.__setattr__(self, "unrealized_pnl", _finite(
            "unrealized_pnl", self.unrealized_pnl,
        ))
        object.__setattr__(self, "positions", _number_pairs("positions", self.positions))
        object.__setattr__(self, "marks", _price_pairs(self.marks))
        object.__setattr__(self, "spread_bps", _spread_pairs(self.spread_bps))
        object.__setattr__(self, "assets", _asset_pairs(self.assets))
        for name in ("open_orders", "orders_this_minute", "orders_today", "consecutive_losses"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} debe ser un entero >= 0")
        if self.minute_bucket is not None:
            object.__setattr__(self, "minute_bucket", ensure_utc(self.minute_bucket, "minute_bucket"))
        if self.cooldown_until is not None:
            object.__setattr__(self, "cooldown_until", ensure_utc(self.cooldown_until, "cooldown_until"))
        if self.last_bar_close is not None:
            object.__setattr__(self, "last_bar_close", ensure_utc(self.last_bar_close, "last_bar_close"))
        if self.session_date is not None and not isinstance(self.session_date, date):
            raise TypeError("session_date debe ser datetime.date")
        if self.kill_reason is not None and not str(self.kill_reason).strip():
            raise ValueError("kill_reason vacío")

    def position(self, symbol: str) -> float:
        return _lookup(self.positions, symbol, 0.0)

    def mark(self, symbol: str) -> Optional[float]:
        return _lookup(self.marks, symbol, None)

    def spread(self, symbol: str) -> Optional[float]:
        return _lookup(self.spread_bps, symbol, None)

    def asset(self, symbol: str) -> Optional[AssetFlag]:
        return _lookup(self.assets, symbol, None)

    def position_map(self) -> dict:
        return dict(self.positions)


@dataclass(frozen=True)
class FilterResult:
    """Salida de ``filter``. Se desempaqueta como ``(aprobadas, rechazadas)``.

    El motor de HFT-06 solo lee esas dos. ``flatten``, ``kill`` y
    ``state`` son para el runner: el archivo KILL no se escribe aquí.
    """

    approved: Tuple[OrderIntent, ...]
    rejected: Tuple[Tuple[OrderIntent, str], ...]
    flatten: Tuple[OrderIntent, ...]
    kill: bool
    kill_reason: Optional[str]
    events: Tuple[RiskEvent, ...]
    state: RiskState

    def __iter__(self):
        yield self.approved
        yield self.rejected


def limits_from_account(account: Mapping) -> Tuple[float, float]:
    """``(equity, buying_power)`` de la cuenta.

    Solo esas dos claves. ``RETIRED_PDT_FIELDS`` existe para que un
    lector vea lo que no hay que consultar; esta función no las toca.
    """
    if not isinstance(account, Mapping):
        raise TypeError("account debe ser un mapping")
    equity = _account_float(account, "equity")
    buying_power = _account_float(account, "buying_power")
    if equity < 0 or buying_power < 0:
        raise ValueError("equity y buying_power no pueden ser negativos")
    return equity, buying_power


def guard_for(config: RiskConfig) -> PreTradeGuard:
    """Guard de lote intradía. No usa el tope de nocional total del runner."""
    return PreTradeGuard(
        max_notional_per_order=config.max_position_notional,
        max_total_notional=None,
        max_orders=None,
        block_if_account_blocked=True,
        block_if_market_closed=False,
        max_orders_per_minute=config.max_orders_per_minute,
        max_orders_per_day=config.max_orders_per_day,
        max_open_orders=config.max_open_orders,
        max_gross_exposure=config.max_gross_exposure,
        max_net_exposure=config.max_net_exposure,
    )


def activate_kill(state_dir: Path, reason: str) -> Path:
    """Escribe ``state_dir/KILL`` de forma atómica. No mira el reloj."""
    text = str(reason).strip()
    if not text:
        raise ValueError("el KILL necesita un motivo")
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / KILL_FILE
    temporary = directory / f".{KILL_FILE}.{os.getpid()}.tmp"
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def arm_kill(
    result: FilterResult,
    state_dir: Optional[Path],
    *,
    live: bool,
) -> Optional[Path]:
    """Escribe KILL solo en vivo y solo si el resultado lo pide.

    El backtest pasa ``live=False`` y no crea el archivo aunque la
    pérdida diaria esté rota.
    """
    if not result.kill or not live:
        return None
    if state_dir is None:
        raise ValueError("en vivo hace falta un directorio para activar KILL")
    return activate_kill(state_dir, result.kill_reason or KILL_KIND)


def refresh_kill(
    state: RiskState,
    state_dir: Path,
    env: Optional[Mapping[str, str]] = None,
) -> RiskState:
    """Copia el motivo de ``kill_switch_reason`` al estado. No decide órdenes."""
    reason = kill_switch_reason(Path(state_dir), env=env)
    return replace(state, kill_reason=reason)


class RiskManager:
    """Aplica ``[risk]`` a un lote. Las sesiones vienen del calendario de HFT-03."""

    def __init__(
        self,
        config: RiskConfig,
        sessions: Sequence[Session],
        *,
        guard: Optional[PreTradeGuard] = None,
    ) -> None:
        if not isinstance(config, RiskConfig):
            raise TypeError("config debe ser RiskConfig")
        self.config = config
        self.sessions = tuple(sessions)
        self.guard = guard if guard is not None else guard_for(config)

    def filter(
        self,
        intents: Sequence[OrderIntent],
        state: RiskState,
        now: datetime,
    ) -> FilterResult:
        if not isinstance(state, RiskState):
            raise TypeError("state debe ser RiskState")
        instant = ensure_utc(now, "now")
        current = _advance(self.config, state, instant)
        session = session_for(self.sessions, instant)
        trading = in_session(self.sessions, instant)
        past_flatten = False
        past_cutoff = False
        if session is not None:
            past_flatten = instant >= flatten_time(session, self.config.flatten_minutes)
            past_cutoff = instant >= entry_cutoff(session, self.config.entry_cutoff_minutes)
        loss_usd, loss_pct = _daily_loss(current)
        daily_on = (
            loss_usd + _EPS >= self.config.max_daily_loss_usd
            or loss_pct + _EPS >= self.config.max_daily_loss_pct
        )
        daily_text = (
            f"max_daily_loss: {loss_usd:.2f} USD ({loss_pct:.4%} del equity) "
            f">= {self.config.max_daily_loss_usd:.2f} USD o "
            f"{self.config.max_daily_loss_pct:.4%}"
        )
        cooling = (
            current.cooldown_until is not None and instant < current.cooldown_until
        )
        stale = _stale(self.config, current, instant)
        common = _common_block(
            current, daily_on, daily_text, past_flatten, trading, cooling,
            past_cutoff, stale, self.config, instant,
        )

        rows = []
        positions = current.position_map()
        minute_orders = current.orders_this_minute
        day_orders = current.orders_today
        open_orders = current.open_orders
        buying_power = current.buying_power
        for intent in intents:
            if not isinstance(intent, OrderIntent):
                raise TypeError(f"se esperaba OrderIntent, llegó {type(intent).__name__}")
            entry = not _reduces(positions.get(intent.symbol, 0.0), intent.side, intent.qty)
            price = current.mark(intent.symbol)
            notional = abs(intent.qty * price) if price is not None else None
            if current.kill_reason:
                rows.append(_no(intent, entry, f"kill_switch: {current.kill_reason}", notional))
                continue
            if intent.bracket is not None and not _whole(intent.qty):
                rows.append(_no(
                    intent, entry,
                    f"bracket_fraccional: {intent.symbol} qty {intent.qty:g} "
                    "exige acciones enteras",
                    notional,
                ))
                continue
            if not trading and not past_flatten:
                rows.append(_no(intent, entry, "fuera_de_sesion", notional))
                continue
            if entry and common is not None:
                rows.append(_no(intent, entry, common, notional))
                continue
            reason = None if not entry else _entry_reason(
                self.config, current, intent, positions, price, notional,
                minute_orders, day_orders, open_orders, buying_power,
            )
            if reason is not None:
                rows.append(_no(intent, entry, reason, notional))
                continue
            rows.append(_yes(intent, entry, notional))
            if entry:
                minute_orders += 1
                day_orders += 1
                open_orders += 1
                if notional is not None:
                    buying_power -= notional
            positions = _apply(positions, intent)

        rows = _apply_guard(self, current, rows, positions)
        positions = current.position_map()
        minute_orders = current.orders_this_minute
        day_orders = current.orders_today
        open_orders = current.open_orders
        buying_power = current.buying_power
        for row in rows:
            if not row.ok:
                continue
            if row.entry:
                minute_orders += 1
                day_orders += 1
                open_orders += 1
                if row.notional is not None:
                    buying_power -= row.notional
            positions = _apply(positions, row.intent)

        want_flatten = bool(current.kill_reason) or daily_on or past_flatten
        flatten = _flatten(positions, session, instant) if want_flatten else ()
        approved = tuple(row.intent for row in rows if row.ok)
        rejected = tuple((row.intent, row.reason) for row in rows if not row.ok)
        kill = bool(current.kill_reason) or daily_on
        kill_text = current.kill_reason if current.kill_reason else (daily_text if daily_on else None)
        events = _events(instant, rejected, flatten, kill, kill_text, past_flatten, daily_on)
        new_state = replace(
            current,
            orders_this_minute=minute_orders,
            orders_today=day_orders,
            open_orders=open_orders,
            buying_power=max(0.0, buying_power),
        )
        return FilterResult(
            approved=approved,
            rejected=rejected,
            flatten=flatten,
            kill=kill,
            kill_reason=kill_text,
            events=events,
            state=new_state,
        )


class EngineRiskGate:
    """Lo que el motor llama. Superpone el libro del ``EngineState``.

    Equity, buying power, rachas, spreads y la ficha de corto siguen
    en el ``RiskState`` que guarda esta puerta. El libro (caja, posición,
    P&L, nav) se pisa con lo que el motor acaba de marcar. ``open_orders``
    también: son las órdenes que la ejecución todavía ve vivas, no un
    contador que solo sube (HFT-PAPER-10).
    """

    def __init__(
        self,
        manager: RiskManager,
        state: RiskState,
        *,
        equity_from_book: bool = False,
    ) -> None:
        self.manager = manager
        self.state = state
        self.equity_from_book = equity_from_book

    def filter(self, intents: Sequence[OrderIntent], engine_state: object, now: datetime):
        overlaid = replace(
            self.state,
            cash=float(engine_state.cash),
            positions=tuple(engine_state.positions),
            realized_pnl=float(engine_state.realized_pnl),
            unrealized_pnl=float(engine_state.unrealized_pnl),
            nav=float(engine_state.nav),
            open_orders=int(engine_state.open_orders),
        )
        if self.equity_from_book:
            overlaid = replace(overlaid, equity=float(engine_state.nav))
        result = self.manager.filter(intents, overlaid, now)
        self.state = result.state
        return result


@dataclass(frozen=True)
class _Row:
    intent: OrderIntent
    entry: bool
    ok: bool
    reason: str
    notional: Optional[float]


def _yes(intent: OrderIntent, entry: bool, notional: Optional[float]) -> _Row:
    return _Row(intent, entry, True, "", notional)


def _no(intent: OrderIntent, entry: bool, reason: str, notional: Optional[float]) -> _Row:
    return _Row(intent, entry, False, reason, notional)


def _common_block(
    state, daily_on, daily_text, past_flatten, trading, cooling, past_cutoff,
    stale, config, now,
) -> Optional[str]:
    if daily_on:
        return daily_text
    if past_flatten:
        return (
            f"flatten_time: {now.isoformat()} pasó el flatten "
            f"({config.flatten_minutes:g} min antes del cierre)"
        )
    if not trading:
        return "fuera_de_sesion"
    if cooling:
        return f"cooldown: hasta {state.cooldown_until.isoformat()}"
    if past_cutoff:
        return (
            f"entry_cutoff: {now.isoformat()} pasó el corte "
            f"({config.entry_cutoff_minutes:g} min antes del cierre)"
        )
    if stale:
        seen = "sin barra final" if state.last_bar_close is None else state.last_bar_close.isoformat()
        return f"stale_data: última barra final {seen} > {config.stale_data_s:g} s"
    return None


def _entry_reason(
    config, state, intent, positions, price, notional,
    minute_orders, day_orders, open_orders, buying_power,
) -> Optional[str]:
    symbol = intent.symbol
    if price is None:
        return f"sin_precio: {symbol}"
    if _opens_short(positions.get(symbol, 0.0), intent.side, intent.qty):
        reason = _short_reason(state, symbol, intent.qty)
        if reason is not None:
            return reason
    spread = state.spread(symbol)
    if spread is not None and spread > config.spread_guard_bps:
        return (
            f"spread_guard: {symbol} {spread:g} bps > {config.spread_guard_bps:g} bps"
        )
    held = positions.get(symbol, 0.0)
    delta = intent.qty if intent.side == "buy" else -intent.qty
    projected_qty = held + delta
    projected_notional = abs(projected_qty * price)
    if projected_notional > config.max_position_notional + _EPS:
        return (
            f"max_position_notional: {symbol} ${projected_notional:,.2f} > "
            f"${config.max_position_notional:,.2f}"
        )
    trial = _apply(positions, intent)
    exposures = _exposures(trial, state)
    if exposures is None:
        return f"sin_precio: libro sin marca al evaluar {symbol}"
    gross, net = exposures
    gross_cap = config.max_gross_exposure * state.equity
    net_cap = config.max_net_exposure * state.equity
    if state.equity <= 0 or gross > gross_cap + _EPS:
        return f"max_gross_exposure: ${gross:,.2f} > {config.max_gross_exposure:.2f}x equity"
    if net > net_cap + _EPS:
        return f"max_net_exposure: ${net:,.2f} > {config.max_net_exposure:.2f}x equity"
    if open_orders + 1 > config.max_open_orders:
        return f"max_open_orders: {open_orders + 1} > {config.max_open_orders}"
    if minute_orders + 1 > config.max_orders_per_minute:
        return f"max_orders_per_minute: {minute_orders + 1} > {config.max_orders_per_minute}"
    if day_orders + 1 > config.max_orders_per_day:
        return f"max_orders_per_day: {day_orders + 1} > {config.max_orders_per_day}"
    if notional is None or notional > buying_power + _EPS:
        have = 0.0 if notional is None else buying_power
        need = 0.0 if notional is None else notional
        return f"buying_power: ${need:,.2f} > ${have:,.2f}"
    return None


def _short_reason(state: RiskState, symbol: str, qty: float) -> Optional[str]:
    flag = state.asset(symbol)
    if flag is None or not flag.shortable or not flag.easy_to_borrow:
        shortable = None if flag is None else flag.shortable
        easy = None if flag is None else flag.easy_to_borrow
        return (
            f"short_no_habilitado: {symbol} shortable={shortable} "
            f"easy_to_borrow={easy}"
        )
    if not _whole(qty):
        return f"corto_fraccional: {symbol} qty {qty:g} exige acciones enteras"
    return None


def _apply_guard(manager: RiskManager, state: RiskState, rows: Sequence[_Row], positions: dict):
    entries = [row for row in rows if row.ok and row.entry]
    if not entries:
        return list(rows)
    exposures = _exposures(positions, state)
    summary = {
        "equity": state.equity,
        "account_blocked": False,
        "trading_blocked": False,
        "market_open": True,
        "orders_this_minute": state.orders_this_minute,
        "orders_today": state.orders_today,
        "open_orders": state.open_orders,
    }
    if exposures is not None:
        summary["projected_gross_notional"] = exposures[0]
        summary["projected_net_notional"] = exposures[1]
    plan = pd.DataFrame(
        [
            {
                "ticker": row.intent.symbol,
                "side": row.intent.side,
                "notional": 0.0 if row.notional is None else row.notional,
            }
            for row in entries
        ],
        columns=["ticker", "side", "notional"],
    )
    reasons = manager.guard.evaluate(plan, summary)
    if not reasons:
        return list(rows)
    reason = "pre_trade: " + "; ".join(reasons)
    blocked = {row.intent.intent_id for row in entries}
    out = []
    for row in rows:
        if row.intent.intent_id in blocked:
            out.append(_no(row.intent, row.entry, reason, row.notional))
        else:
            out.append(row)
    return out


def _flatten(positions: dict, session: Optional[Session], now: datetime) -> Tuple[OrderIntent, ...]:
    anchor = session.open_utc if session is not None else now
    orders = []
    for seq, symbol in enumerate(sorted(positions)):
        qty = positions[symbol]
        if abs(qty) <= _EPS:
            continue
        side = "sell" if qty > 0 else "buy"
        orders.append(order_intent(
            "risk",
            symbol,
            side,
            abs(qty),
            anchor,
            reason=FLATTEN_ALL,
            seq=seq,
        ))
    return tuple(orders)


def _events(now, rejected, flatten, kill, kill_text, past_flatten, daily_on):
    events = [
        RiskEvent(
            ts=now,
            kind="rejected",
            reason=reason,
            symbol=intent.symbol,
            intent_id=intent.intent_id,
        )
        for intent, reason in rejected
    ]
    if flatten:
        why = "max_daily_loss" if daily_on else ("flatten_time" if past_flatten else "kill_switch")
        events.append(RiskEvent(ts=now, kind=FLATTEN_ALL, reason=why))
    if kill and kill_text:
        events.append(RiskEvent(ts=now, kind=KILL_KIND, reason=kill_text))
    return tuple(events)


def _advance(config: RiskConfig, state: RiskState, now: datetime) -> RiskState:
    bucket = now.replace(second=0, microsecond=0)
    day = now.astimezone(NY).date()
    minute_orders = state.orders_this_minute if state.minute_bucket == bucket else 0
    day_orders = state.orders_today if state.session_date == day else 0
    losses = state.consecutive_losses
    cooldown = state.cooldown_until
    if cooldown is not None and now >= cooldown:
        losses = 0
        cooldown = None
    elif losses >= config.max_consecutive_losses and cooldown is None:
        cooldown = now + timedelta(minutes=config.cooldown_minutes)
    return replace(
        state,
        minute_bucket=bucket,
        session_date=day,
        orders_this_minute=minute_orders,
        orders_today=day_orders,
        consecutive_losses=losses,
        cooldown_until=cooldown,
    )


def _daily_loss(state: RiskState) -> Tuple[float, float]:
    pnl = state.realized_pnl + state.unrealized_pnl - state.session_start_pnl
    loss = max(0.0, -pnl)
    if state.equity > 0:
        return loss, loss / state.equity
    return loss, (math.inf if loss > 0 else 0.0)


def _stale(config: RiskConfig, state: RiskState, now: datetime) -> bool:
    if state.last_bar_close is None:
        return True
    return (now - state.last_bar_close).total_seconds() > config.stale_data_s


def _reduces(position: float, side: str, qty: float) -> bool:
    if abs(position) <= _EPS:
        return False
    delta = qty if side == "buy" else -qty
    new = position + delta
    if position > 0:
        return delta < 0 and new >= -_EPS and abs(new) <= abs(position) + _EPS
    return delta > 0 and new <= _EPS and abs(new) <= abs(position) + _EPS


def _opens_short(position: float, side: str, qty: float) -> bool:
    if side != "sell":
        return False
    return position - qty < -_EPS


def _apply(positions: dict, intent: OrderIntent) -> dict:
    nxt = dict(positions)
    delta = intent.qty if intent.side == "buy" else -intent.qty
    qty = nxt.get(intent.symbol, 0.0) + delta
    if abs(qty) <= _EPS:
        qty = 0.0
    nxt[intent.symbol] = qty
    return nxt


def _exposures(positions: dict, state: RiskState) -> Optional[Tuple[float, float]]:
    signed = []
    for symbol, qty in positions.items():
        if abs(qty) <= _EPS:
            continue
        price = state.mark(symbol)
        if price is None:
            return None
        signed.append(qty * price)
    gross = sum(abs(value) for value in signed)
    net = abs(sum(signed))
    return gross, net


def _whole(qty: float) -> bool:
    return abs(qty - round(qty)) <= 1e-8


def _lookup(pairs, symbol: str, default):
    wanted = str(symbol).strip().upper()
    for name, value in pairs:
        if name == wanted:
            return value
    return default


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} debe ser finito, llegó {value!r}")
    return number


def _account_float(account: Mapping, key: str) -> float:
    if key not in account:
        raise KeyError(key)
    raw = account[key]
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError as exc:
            raise ValueError(f"{key} no es numérico: {account[key]!r}") from exc
    return _finite(key, raw)


def _number_pairs(name: str, value: object) -> Tuple[Tuple[str, float], ...]:
    return _pairs(name, value, lambda raw: _finite(name, raw))


def _price_pairs(value: object) -> Tuple[Tuple[str, float], ...]:
    def parse(raw: object) -> float:
        number = _finite("mark", raw)
        if number <= 0:
            raise ValueError(f"mark debe ser > 0, llegó {raw!r}")
        return number
    return _pairs("marks", value, parse)


def _spread_pairs(value: object) -> Tuple[Tuple[str, float], ...]:
    def parse(raw: object) -> float:
        number = _finite("spread_bps", raw)
        if number < 0:
            raise ValueError("spread_bps no puede ser negativo")
        return number
    return _pairs("spread_bps", value, parse)


def _asset_pairs(value: object) -> Tuple[Tuple[str, AssetFlag], ...]:
    def parse(raw: object) -> AssetFlag:
        if not isinstance(raw, AssetFlag):
            raise TypeError("assets debe mapear a AssetFlag")
        return raw
    return _pairs("assets", value, parse)


def _pairs(name: str, value: object, parse):
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        items = list(value)
    out = []
    seen = set()
    for key, raw in items:
        symbol = str(key).strip().upper()
        if not symbol:
            raise ValueError(f"{name}: símbolo vacío")
        if symbol in seen:
            raise ValueError(f"{name}: símbolo repetido {symbol}")
        seen.add(symbol)
        out.append((symbol, parse(raw)))
    return tuple(out)
