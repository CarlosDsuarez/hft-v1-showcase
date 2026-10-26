"""Ejecución intradía en Alpaca paper (HFT-11).

Misma firma que ``SimExecution.submit(intents, now=, bars=)``: el motor
no distingue el adaptador. Este módulo no abre sockets; el HTTP vive
en ``src.brokers.alpaca_adapter`` y los tests lo sustituyen.

``submit`` publica órdenes nuevas y devuelve los fills que ya se
puedan leer. ``poll`` repite esa lectura como pronto cada
``poll_interval_s`` (reloj inyectado, nunca ``datetime.now``). Las
fuentes son ``GET /v2/orders`` (``nested=true``) y
``GET /v2/account/activities?activity_types=FILL``.

Con ``stream`` (``src.brokers.alpaca_stream.TradeUpdatesStream``) los
fills salen del websocket ``trade_updates``: no gastan el cupo de la
Trading API y llegan en milisegundos. D-07 quedó revocada el
2026-10-13. El REST pasa a reconciliación: cada
``reconcile_interval_s``, tras cada reconexión y tras un rollback.

Con ``client.gateway.workers > 1`` un lote se publica en paralelo. Las
guardas de cada orden se calculan antes, en orden, suponiendo que las
anteriores entran. Los resultados y el journal se aplican en el orden
de las intenciones, no en el orden en que responde el broker.

El id de cada orden sale de ``make_intent_order_id``. Reiniciar el
proceso con la misma intención produce el mismo ``client_order_id``;
un 422 de duplicado se resuelve contra la orden que ya existe.

Los fills de paper se valoran con ``CostModel.paper_fees``. La
actividad FILL no trae liquidez: market, stop, marketable_limit e
``ioc`` se anotan como taker; un límite ``day`` o ``cls``, como maker.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.brokers.alpaca_adapter import (
    _FINAL_STATUSES,
    AlpacaClient,
    BrokerError,
    IntradayOrderError,
    OrderSpec,
    build_order_payload,
    make_intent_order_id,
)
from src.brokers.pre_trade import PreTradeGuard
from src.hft.calendar import NY
from src.hft.costs import CostModel
from src.hft.data.bars import IntradayBars
from src.hft.risk import RiskState
from src.hft.types import Bracket, Fill, OrderIntent, ensure_utc, order_intent

logger = logging.getLogger("hft.alpaca_exec")

UTC = timezone.utc
_EPS = 1e-8
# Eventos del stream sin orden propia que se guardan para reintentar.
_MAX_UNBOUND = 5000

# OrderIntent.order_type → type de POST /v2/orders. marketable_limit
# no existe en Alpaca: se envía como limit al precio de la intención.
_ALPACA_TYPE = {
    "market": "market",
    "limit": "limit",
    "marketable_limit": "limit",
    "stop": "stop",
}


class AlpacaExecution:
    """Adaptador paper. ``clock`` es la única hora que mira."""

    def __init__(
        self,
        client: AlpacaClient,
        costs: CostModel,
        *,
        clock: Callable[[], datetime],
        risk_state: Optional[RiskState] = None,
        guard: Optional[PreTradeGuard] = None,
        poll_interval_s: float = 5.0,
        journal: Optional[List[dict]] = None,
        use_trade_updates: bool = False,
        stream: Optional[object] = None,
        reconcile_interval_s: float = 60.0,
    ) -> None:
        if not isinstance(client, AlpacaClient):
            raise TypeError("client debe ser AlpacaClient")
        if not isinstance(costs, CostModel):
            raise TypeError("costs debe ser CostModel")
        if not callable(clock):
            raise TypeError("clock debe ser invocable y devolver datetime")
        if not isinstance(use_trade_updates, bool):
            raise TypeError("use_trade_updates debe ser bool")
        if use_trade_updates and stream is None:
            raise ValueError("use_trade_updates exige un stream (TradeUpdatesStream)")
        if stream is not None and not all(
            callable(getattr(stream, name, None)) for name in ("drain", "take_reconnected")
        ):
            raise TypeError("stream debe tener drain() y take_reconnected()")
        if isinstance(reconcile_interval_s, bool) or not isinstance(reconcile_interval_s, (int, float)):
            raise TypeError("reconcile_interval_s debe ser numérico")
        if float(reconcile_interval_s) < 0:
            raise ValueError("reconcile_interval_s no puede ser negativo")
        if isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float)):
            raise TypeError("poll_interval_s debe ser numérico")
        if float(poll_interval_s) < 0:
            raise ValueError("poll_interval_s no puede ser negativo")
        if risk_state is not None and not isinstance(risk_state, RiskState):
            raise TypeError("risk_state debe ser RiskState o None")
        if guard is not None and not isinstance(guard, PreTradeGuard):
            raise TypeError("guard debe ser PreTradeGuard o None")
        self.client = client
        self.costs = costs
        self._clock = clock
        self.risk_state = risk_state
        self.guard = guard
        self.poll_interval_s = float(poll_interval_s)
        self.use_trade_updates = stream is not None
        self._stream = stream
        self.reconcile_interval_s = float(reconcile_interval_s)
        self._last_reconcile: Optional[datetime] = None
        self._force_reconcile = True
        # Eventos del stream de órdenes que todavía no se pueden asociar
        # (patas de bracket antes de la lectura con nested=true).
        self._unbound: List[dict] = []
        # Lectura REST incremental (solo con stream): última actividad vista,
        # último submitted_at visto y actividades de órdenes aún sin asociar.
        self._act_cursor: Optional[str] = None
        self._orders_cursor: Optional[str] = None
        self._unbound_acts: List[dict] = []
        self.journal = journal if journal is not None else []
        self._orders: Dict[str, dict] = {}
        self._by_alpaca: Dict[str, OrderIntent] = {}
        self._by_client: Dict[str, OrderIntent] = {}
        self._legs: Dict[str, Tuple[OrderIntent, ...]] = {}
        self._seen_activities: set = set()
        # Cantidad por intención según cada fuente. Actividades y órdenes ven
        # el mismo llenado a destiempo: lo emitido es el máximo de las dos.
        self._activity_qty: Dict[str, float] = {}
        self._emitted: Dict[str, float] = {}
        self._positions: Dict[str, float] = {}
        self._marks: Dict[str, float] = {}
        self._risk_seen: Optional[RiskState] = None
        self._sent_minute = 0
        self._sent_day = 0
        self._minute_bucket: Optional[datetime] = None
        self._session_date = None
        self._last_poll: Optional[datetime] = None
        self._spawned: List[OrderIntent] = []
        self._snap: Optional[dict] = None
        self._last_posted_ids: List[str] = []
        self._sync_risk()

    def submit(
        self,
        intents: Sequence[OrderIntent],
        *,
        now: datetime,
        bars: Optional[IntradayBars] = None,
    ) -> Tuple[Fill, ...]:
        """Publica intenciones nuevas y devuelve fills ya visibles.

        Una intención ya enviada no se vuelve a postear. La validación
        de todas corre antes del primer POST de este lote.
        """
        moment = ensure_utc(now, "now")
        if bars is not None and not isinstance(bars, IntradayBars):
            raise TypeError("bars debe ser IntradayBars")
        self._sync_risk()
        pre = self._snapshot()
        posted: List[str] = []
        spawned: List[OrderIntent] = []
        try:
            ready = [self._prepare(intent, moment, bars) for intent in intents
                     if intent.intent_id not in self._orders]
            if self.client.gateway.workers > 1 and len(ready) > 1:
                self._post_parallel(ready, moment, bars, posted, spawned)
            else:
                for prepared in ready:
                    result = self._post(prepared, moment, bars)
                    if result.get("status") == "submitted" and result.get("id"):
                        posted.append(str(result["id"]))
                    spawned.extend(prepared.get("children") or ())
        except Exception:
            self._restore(pre)
            self._cancel_posted(posted)
            self._reset_cursors()
            raise
        self._spawned = spawned
        self._snap = pre
        self._last_posted_ids = posted
        try:
            fills = self._read_fills(moment)
        except BrokerError as exc:
            logger.warning("lectura de fills tras el envío falló: %s", exc)
            self._force_reconcile = True
            fills = []
        self._last_poll = moment
        return tuple(fills)

    def poll(self) -> Tuple[Fill, ...]:
        """Fills nuevos. Si no pasó ``poll_interval_s``, no hay HTTP.

        Con stream siempre vacía la cola (sin HTTP) y solo reconcilia por
        REST cuando toca.
        """
        moment = ensure_utc(self._clock(), "now")
        if self._stream is not None:
            self._last_poll = moment
            return tuple(self._read_fills(moment))
        if self._last_poll is not None:
            elapsed = (moment - self._last_poll).total_seconds()
            if elapsed < self.poll_interval_s:
                return ()
        self._last_poll = moment
        return tuple(self._fetch_fills(moment))

    def take_spawned(self) -> Tuple[OrderIntent, ...]:
        """Hijos de bracket creados en el último ``submit``. Se vacía."""
        out = tuple(self._spawned)
        self._spawned = []
        return out

    def commit_last(self) -> None:
        """El libro aceptó el lote: el rollback ya no deshace nada."""
        self._snap = None
        self._last_posted_ids = []

    def rollback_last(self) -> None:
        """Vuelve al estado local previo y cancela lo que este submit publicó.

        Un fill que el broker ya hizo no se puede borrar. Si el cancel
        falla, el id sigue en Alpaca y el reintento cae en el 422 de
        duplicado. Los fills locales se olvidan para poder aplicarlos
        otra vez si el libro los rechazó.
        """
        if self._snap is None:
            return
        posted = list(self._last_posted_ids)
        self._restore(self._snap)
        self._snap = None
        self._last_posted_ids = []
        self._cancel_posted(posted)
        # Lo que el stream ya entregó se olvidó con el restore: el REST lo
        # vuelve a leer entero.
        self._reset_cursors()

    def _reset_cursors(self) -> None:
        self._force_reconcile = True
        self._act_cursor = None
        self._orders_cursor = None
        self._unbound_acts = []

    def seed_positions(self, positions: Mapping[str, float]) -> None:
        """Posición de partida por símbolo: lo que el broker ya tiene y es propio.

        Un adaptador nuevo solo conoce sus fills. Tras un relanzamiento, la
        venta que cierra un largo heredado se clasificaría como corto y un
        fraccional se rechazaría por no ser acciones enteras (HFT-PAPER-17).
        """
        for symbol, qty in positions.items():
            self._positions[str(symbol).strip().upper()] = float(qty)

    def open_orders(self) -> Tuple[OrderIntent, ...]:
        """Órdenes que este adaptador todavía considera vivas."""
        return tuple(
            slot["intent"] for slot in self._orders.values() if slot["open"]
        )

    def cancel(self, intent_id: str) -> bool:
        """Cancela una orden viva por ``intent_id``. False si no está abierta."""
        slot = self._orders.get(intent_id)
        if slot is None or not slot["open"]:
            return False
        alpaca_id = slot.get("alpaca_id")
        if alpaca_id:
            self.client.cancel_order(str(alpaca_id))
        slot["open"] = False
        return True

    def _prepare(self, intent: OrderIntent, now: datetime, bars: Optional[IntradayBars]) -> dict:
        if not isinstance(intent, OrderIntent):
            raise TypeError(f"se esperaba OrderIntent, llegó {type(intent).__name__}")
        if intent.tif not in ("day", "ioc", "cls"):
            raise IntradayOrderError(
                f"time_in_force {intent.tif!r} no está en day|ioc|cls"
            )
        spec = _spec_for(intent, self._position(intent.symbol))
        symbol, not_routable = self.client._route(intent.symbol)
        asset = None
        if symbol is not None:
            try:
                build_order_payload(spec, broker_symbol=symbol, asset=None)
            except IntradayOrderError as exc:
                if not exc.needs_asset:
                    raise
            asset = self.client.get_asset(intent.symbol, as_of=now.astimezone(NY).date())
        elif not_routable:
            asset = None
        return {"intent": intent, "spec": spec, "asset": asset, "children": _children(intent),
                "routable": symbol is not None}

    def _post(self, prepared: dict, now: datetime, bars: Optional[IntradayBars]) -> dict:
        intent = prepared["intent"]
        mark = self._mark(intent.symbol, bars)
        minute, today = self._rates(now)
        gross = self._gross_after(intent.symbol, intent.side, intent.qty, mark)
        result = self.client.submit_order(
            prepared["spec"],
            asset=prepared["asset"],
            mark=mark,
            guard=self.guard,
            risk_state=self.risk_state,
            journal=self.journal,
            orders_this_minute=minute,
            orders_today=today,
            projected_gross=gross,
            lane=_lane(intent, self._positions),
        )
        self._apply_post(prepared, result)
        return result

    def _apply_post(self, prepared: dict, result: Mapping) -> None:
        intent = prepared["intent"]
        if result.get("status") != "submitted":
            prepared["children"] = ()
            return
        self._remember(intent, result)
        delta = intent.qty if intent.side == "buy" else -intent.qty
        self._positions[intent.symbol] = self._position(intent.symbol) + delta
        self._sent_minute += 1
        self._sent_day += 1

    def _post_parallel(
        self,
        ready: Sequence[dict],
        now: datetime,
        bars: Optional[IntradayBars],
        posted: List[str],
        spawned: List[OrderIntent],
    ) -> None:
        """Publica el lote con los hilos del cliente. Aplica en orden de intención.

        Cada orden ve las guardas como si las anteriores del lote hubieran
        entrado: tasa ``+i`` y exposición con las posiciones ya movidas. Es
        a lo sumo más estricto que el lazo en serie. Si alguna falla, las
        publicadas quedan en ``posted`` y la excepción sube: ``submit``
        hace el rollback.
        """
        if any(item.get("routable") for item in ready):
            self.client.ensure_account_verified()
        minute, today = self._rates(now)
        positions = dict(self._positions)
        calls = []
        journals: List[List[dict]] = []
        for index, prepared in enumerate(ready):
            intent = prepared["intent"]
            mark = self._mark(intent.symbol, bars)
            gross = self._gross_after(intent.symbol, intent.side, intent.qty, mark,
                                      positions=positions)
            lane = _lane(intent, positions)
            delta = intent.qty if intent.side == "buy" else -intent.qty
            positions[intent.symbol] = positions.get(intent.symbol, 0.0) + delta
            journal: List[dict] = []
            journals.append(journal)
            calls.append(partial(
                self.client.submit_order,
                prepared["spec"],
                asset=prepared["asset"],
                mark=mark,
                guard=self.guard,
                risk_state=self.risk_state,
                journal=journal,
                orders_this_minute=minute + index,
                orders_today=today + index,
                projected_gross=gross,
                lane=lane,
            ))
        outcomes = self.client.run_many(calls)
        first_error: Optional[BaseException] = None
        for prepared, journal, (result, error) in zip(ready, journals, outcomes):
            self.journal.extend(journal)
            if error is not None:
                prepared["children"] = ()
                if first_error is None:
                    first_error = error
                continue
            if result is None:
                prepared["children"] = ()
                continue
            self._apply_post(prepared, result)
            if result.get("status") == "submitted" and result.get("id"):
                posted.append(str(result["id"]))
            spawned.extend(prepared.get("children") or ())
        if first_error is not None:
            raise first_error

    def _remember(self, intent: OrderIntent, result: Mapping) -> None:
        alpaca_id = result.get("id")
        cid = str(result.get("client_order_id") or "")
        status = result.get("broker_status")
        self._orders[intent.intent_id] = {
            "intent": intent,
            "alpaca_id": alpaca_id,
            "client_order_id": cid,
            "open": status not in _FINAL_STATUSES,
        }
        if alpaca_id:
            self._by_alpaca[str(alpaca_id)] = intent
        if cid:
            self._by_client[cid] = intent
        children = _children(intent)
        if children:
            self._legs[intent.intent_id] = tuple(children)

    def _read_fills(self, now: datetime) -> List[Fill]:
        """Sin stream: la lectura REST de siempre. Con stream: cola + REST si toca."""
        if self._stream is None:
            return self._fetch_fills(now)
        fills = self._drain_stream(now)
        if self._reconcile_due(now):
            self._force_reconcile = True
            fills.extend(self._fetch_fills(now, incremental=True))
            self._force_reconcile = False
            self._last_reconcile = now
            fills.extend(self._retry_unbound(now))
        return fills

    def _reconcile_due(self, now: datetime) -> bool:
        reconnected = bool(self._stream.take_reconnected())
        if reconnected or self._force_reconcile or self._last_reconcile is None:
            return True
        return (now - self._last_reconcile).total_seconds() >= self.reconcile_interval_s

    def _drain_stream(self, now: datetime) -> List[Fill]:
        fills: List[Fill] = []
        for data in self._stream.drain():
            fill, known = self._fill_from_event(data, now)
            if not known:
                self._unbound.append(data)
            elif fill is not None:
                fills.append(fill)
        if len(self._unbound) > _MAX_UNBOUND:
            logger.warning("stream: %d eventos sin orden propia; se descartan los más viejos",
                           len(self._unbound) - _MAX_UNBOUND)
            self._unbound = self._unbound[-_MAX_UNBOUND:]
        return fills

    def _retry_unbound(self, now: datetime) -> List[Fill]:
        events, self._unbound = self._unbound, []
        fills: List[Fill] = []
        for data in events:
            fill, known = self._fill_from_event(data, now)
            if not known:
                self._unbound.append(data)
            elif fill is not None:
                fills.append(fill)
        return fills

    def _fill_from_event(self, data: Mapping, now: datetime) -> Tuple[Optional[Fill], bool]:
        """(fill o None, la orden es nuestra). Cantidad acumulada: sin doble conteo.

        ``filled_qty`` de la orden es acumulado; lo que pase de lo emitido
        (por cualquier fuente) es el fill nuevo. El precio es el de la
        ejecución si cuadra con esa cantidad; si no, el promedio.
        """
        order = data.get("order")
        if not isinstance(order, Mapping):
            return None, True
        self._bind_order(order)
        intent = self._by_alpaca.get(str(order.get("id") or ""))
        if intent is None:
            return None, False
        self._note_status(intent, order.get("status"))
        if data.get("event") not in ("fill", "partial_fill"):
            return None, True
        try:
            filled = float(order.get("filled_qty") or 0.0)
        except (TypeError, ValueError):
            return None, True
        already = self._emitted.get(intent.intent_id, 0.0)
        delta = filled - already
        if delta <= _EPS:
            return None, True
        price = _event_price(data, delta)
        if price is None:
            try:
                price = float(order.get("filled_avg_price") or 0.0)
            except (TypeError, ValueError):
                return None, True
        if price <= 0:
            return None, True
        self._emitted[intent.intent_id] = already + delta
        stamp = _parse_time(data.get("timestamp") or order.get("filled_at"), now)
        return self._make_fill(intent, delta, price, stamp), True

    def _fetch_fills(self, now: datetime, *, incremental: bool = False) -> List[Fill]:
        """Fills por REST: actividades FILL y órdenes del día (``nested=true``).

        ``incremental`` (con stream): sigue desde los cursores en vez de
        releer el día. Una actividad de una orden que todavía no se puede
        asociar (pata de bracket) queda en espera hasta que la lectura de
        órdenes la asocie. Sin cursores (primera vez, tras un rollback) la
        lectura es completa, como sin stream.
        """
        session = now.astimezone(NY).date()
        start = datetime.combine(session, time.min, tzinfo=NY)
        fills: List[Fill] = []
        token = self._act_cursor if incremental else None
        activities = self.client.get_account_activities("FILL", session, page_token=token) \
            if token else self.client.get_account_activities("FILL", session)
        pending_acts: List[dict] = []
        for activity in activities:
            if incremental and str(activity.get("order_id") or "") not in self._by_alpaca:
                pending_acts.append(activity)
                continue
            fill = self._fill_from_activity(activity, now)
            if fill is not None:
                fills.append(fill)
        if incremental and activities:
            last_id = activities[-1].get("id")
            if last_id:
                self._act_cursor = str(last_id)
        cursor: object = start
        if incremental and self._orders_cursor:
            cursor = self._orders_cursor
        newest: Optional[str] = None
        seen_cursors = set()
        for _ in range(10):
            key = str(cursor)
            if key in seen_cursors:
                break
            seen_cursors.add(key)
            if incremental:
                page = self.client.list_orders("all", after=cursor, limit=500, nested=True,
                                               direction="asc")
            else:
                page = self.client.list_orders("all", after=cursor, limit=500, nested=True)
            for order in page:
                self._bind_order(order)
                fill = self._fill_from_order(order, now)
                if fill is not None:
                    fills.append(fill)
            submitted = [row.get("submitted_at") for row in page if row.get("submitted_at")]
            if submitted:
                top = max(str(item) for item in submitted)
                newest = top if newest is None or top > newest else newest
            if len(page) < 500 or not page:
                break
            if not submitted:
                break
            cursor = max(str(item) for item in submitted)
        if incremental:
            if newest is not None:
                self._orders_cursor = _back_one_second(newest)
            retry, self._unbound_acts = self._unbound_acts + pending_acts, []
            for activity in retry:
                if str(activity.get("order_id") or "") not in self._by_alpaca:
                    self._unbound_acts.append(activity)
                    continue
                fill = self._fill_from_activity(activity, now)
                if fill is not None:
                    fills.append(fill)
            if len(self._unbound_acts) > _MAX_UNBOUND:
                self._unbound_acts = self._unbound_acts[-_MAX_UNBOUND:]
        return fills

    def _fill_from_activity(self, activity: Mapping, now: datetime) -> Optional[Fill]:
        act_id = str(activity.get("id") or "")
        if not act_id or act_id in self._seen_activities:
            return None
        kind = str(activity.get("type") or "fill")
        if kind not in ("fill", "partial_fill"):
            self._seen_activities.add(act_id)
            return None
        intent = self._by_alpaca.get(str(activity.get("order_id") or ""))
        if intent is None:
            return None
        try:
            qty = float(activity.get("qty") or 0.0)
            price = float(activity.get("price") or 0.0)
        except (TypeError, ValueError):
            self._seen_activities.add(act_id)
            return None
        if qty <= 0 or price <= 0:
            self._seen_activities.add(act_id)
            return None
        self._seen_activities.add(act_id)
        self._note_status(intent, activity.get("order_status"))
        # La vía de órdenes pudo dar ya esta cantidad con filled_qty: la
        # actividad solo emite lo que pase de lo emitido (HFT-PAPER-09).
        seen = self._activity_qty.get(intent.intent_id, 0.0) + qty
        self._activity_qty[intent.intent_id] = seen
        already = self._emitted.get(intent.intent_id, 0.0)
        delta = seen - already
        if delta <= _EPS:
            return None
        self._emitted[intent.intent_id] = already + delta
        stamp = _parse_time(activity.get("transaction_time"), now)
        return self._make_fill(intent, delta, price, stamp)

    def _fill_from_order(self, order: Mapping, now: datetime) -> Optional[Fill]:
        intent = self._by_alpaca.get(str(order.get("id") or ""))
        if intent is None:
            return None
        self._note_status(intent, order.get("status"))
        try:
            filled = float(order.get("filled_qty") or 0.0)
        except (TypeError, ValueError):
            return None
        already = self._emitted.get(intent.intent_id, 0.0)
        delta = filled - already
        if delta <= _EPS:
            return None
        raw_price = order.get("filled_avg_price")
        if raw_price in (None, ""):
            return None
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return None
        if price <= 0:
            return None
        self._emitted[intent.intent_id] = already + delta
        stamp = _parse_time(order.get("filled_at") or order.get("updated_at"), now)
        return self._make_fill(intent, delta, price, stamp)

    def _bind_order(self, order: Mapping) -> None:
        cid = order.get("client_order_id")
        parent = self._by_client.get(str(cid)) if cid else None
        order_id = order.get("id")
        if parent is not None and order_id:
            self._by_alpaca[str(order_id)] = parent
        children = self._legs.get(parent.intent_id, ()) if parent is not None else ()
        for leg in order.get("legs") or []:
            if not isinstance(leg, Mapping):
                continue
            leg_id = leg.get("id")
            if not leg_id:
                continue
            child = _match_leg(str(leg.get("type") or ""), children)
            if child is not None:
                self._by_alpaca[str(leg_id)] = child
                self._note_status(child, leg.get("status"))

    def _note_status(self, intent: OrderIntent, status: object) -> None:
        slot = self._orders.get(intent.intent_id)
        if slot is not None and status in _FINAL_STATUSES:
            slot["open"] = False

    def _make_fill(self, intent: OrderIntent, qty: float, price: float, stamp: datetime) -> Fill:
        if intent.order_type in ("market", "stop", "marketable_limit") or intent.tif == "ioc":
            liquidity = "taker"
        else:
            liquidity = "maker"
        fees = self.costs.paper_fees(
            side=intent.side, qty=qty, price=price, liquidity=liquidity,
        )
        return Fill(
            intent_id=intent.intent_id,
            ts=stamp,
            qty=qty,
            price=price,
            fees=fees,
            liquidity=liquidity,
        )

    def _position(self, symbol: str) -> float:
        return float(self._positions.get(symbol, 0.0))

    def _mark(self, symbol: str, bars: Optional[IntradayBars]) -> Optional[float]:
        fresh = _last_close(symbol, bars)
        if fresh is not None:
            self._marks[symbol] = fresh
            return fresh
        stored = self._marks.get(symbol)
        if stored is not None and stored > 0:
            return stored
        return None

    def _gross_after(
        self,
        symbol: str,
        side: str,
        qty: float,
        mark: Optional[float],
        *,
        positions: Optional[Mapping[str, float]] = None,
    ) -> Optional[float]:
        if self.guard is None or self.guard.max_gross_exposure is None:
            return None
        positions = dict(self._positions if positions is None else positions)
        marks = dict(self._marks)
        delta = qty if side == "buy" else -qty
        positions[symbol] = positions.get(symbol, 0.0) + delta
        if mark is not None:
            marks[symbol] = mark
        total = 0.0
        for name, pos in positions.items():
            if abs(pos) <= _EPS:
                continue
            price = marks.get(name)
            if price is None or price <= 0:
                return None
            total += abs(pos) * float(price)
        return total

    def _rates(self, now: datetime) -> Tuple[int, int]:
        bucket = now.astimezone(UTC).replace(second=0, microsecond=0)
        session = now.astimezone(NY).date()
        if self._minute_bucket != bucket:
            self._minute_bucket = bucket
            self._sent_minute = 0
        if self._session_date != session:
            self._session_date = session
            self._sent_day = 0
        base_minute = 0
        base_day = 0
        state = self.risk_state
        if state is not None:
            if state.minute_bucket is not None:
                state_bucket = state.minute_bucket.astimezone(UTC).replace(second=0, microsecond=0)
                if state_bucket == bucket:
                    base_minute = state.orders_this_minute
            if state.session_date == session:
                base_day = state.orders_today
        return base_minute + self._sent_minute, base_day + self._sent_day

    def _sync_risk(self) -> None:
        state = self.risk_state
        if state is self._risk_seen:
            return
        self._risk_seen = state
        self._sent_minute = 0
        self._sent_day = 0
        if state is None:
            self._positions = {}
            self._marks = {}
            return
        self._positions = {name: float(qty) for name, qty in state.positions}
        self._marks = {name: float(price) for name, price in state.marks}

    def _snapshot(self) -> dict:
        return {
            "orders": {key: dict(slot) for key, slot in self._orders.items()},
            "by_alpaca": dict(self._by_alpaca),
            "by_client": dict(self._by_client),
            "legs": dict(self._legs),
            "seen_activities": set(self._seen_activities),
            "activity_qty": dict(self._activity_qty),
            "emitted": dict(self._emitted),
            "positions": dict(self._positions),
            "marks": dict(self._marks),
            "sent_minute": self._sent_minute,
            "sent_day": self._sent_day,
            "minute_bucket": self._minute_bucket,
            "session_date": self._session_date,
            "spawned": list(self._spawned),
        }

    def _restore(self, snap: dict) -> None:
        self._orders = {key: dict(slot) for key, slot in snap["orders"].items()}
        self._by_alpaca = dict(snap["by_alpaca"])
        self._by_client = dict(snap["by_client"])
        self._legs = dict(snap["legs"])
        self._seen_activities = set(snap["seen_activities"])
        self._activity_qty = dict(snap["activity_qty"])
        self._emitted = dict(snap["emitted"])
        self._positions = dict(snap["positions"])
        self._marks = dict(snap["marks"])
        self._sent_minute = snap["sent_minute"]
        self._sent_day = snap["sent_day"]
        self._minute_bucket = snap["minute_bucket"]
        self._session_date = snap["session_date"]
        self._spawned = list(snap["spawned"])

    def _cancel_posted(self, ids: Sequence[str]) -> None:
        for order_id in ids:
            try:
                self.client.cancel_order(order_id)
            except BrokerError as exc:
                logger.warning("no se pudo cancelar %s en el rollback: %s", order_id, exc)


def _lane(intent: OrderIntent, positions: Mapping[str, float]) -> str:
    """``exit`` si la orden solo reduce la posición; ``entry`` si abre o agranda."""
    held = float(positions.get(intent.symbol, 0.0))
    if held > _EPS and intent.side == "sell" and intent.qty <= held + _EPS:
        return "exit"
    if held < -_EPS and intent.side == "buy" and intent.qty <= -held + _EPS:
        return "exit"
    return "entry"


def _event_price(data: Mapping, delta: float) -> Optional[float]:
    """Precio de la ejecución del evento si su cantidad es la del fill nuevo."""
    try:
        qty = float(data.get("qty"))
        price = float(data.get("price"))
    except (TypeError, ValueError):
        return None
    if price > 0 and abs(qty - delta) <= 1e-6:
        return price
    return None


def _seq_of(intent: OrderIntent) -> int:
    tail = intent.intent_id.rsplit("-", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return 0


def _opens_short(position: float, side: str, qty: float) -> bool:
    if side != "sell":
        return False
    return position - qty < -_EPS


def _spec_for(intent: OrderIntent, position: float) -> OrderSpec:
    order_class, take, stop_loss = _order_class(intent.bracket)
    alpaca_type = _ALPACA_TYPE[intent.order_type]
    limit_price = intent.limit_price if alpaca_type in ("limit", "stop_limit") else None
    stop_price = intent.stop_price if alpaca_type in ("stop", "stop_limit") else None
    return OrderSpec(
        instrument_id=intent.symbol,
        side=intent.side,
        qty=intent.qty,
        order_type=alpaca_type,
        time_in_force=intent.tif,
        order_class=order_class,
        limit_price=limit_price,
        stop_price=stop_price,
        take_profit_limit_price=take,
        stop_loss_stop_price=stop_loss,
        client_order_id=make_intent_order_id(
            intent.strategy_id, intent.symbol, intent.signal_ts, intent.side, _seq_of(intent),
        ),
        extended_hours=False,
        short=_opens_short(position, intent.side, intent.qty),
    )


def _order_class(bracket: Optional[Bracket]) -> Tuple[str, Optional[float], Optional[float]]:
    if bracket is None or (bracket.take is None and bracket.stop is None):
        return "simple", None, None
    if bracket.take is not None and bracket.stop is not None:
        return "bracket", bracket.take, bracket.stop
    if bracket.stop is not None:
        return "oto", None, bracket.stop
    return "oto", bracket.take, None


def _children(intent: OrderIntent) -> Tuple[OrderIntent, ...]:
    """Stop y take que Alpaca cuelga del padre. No se vuelven a postear."""
    bracket = intent.bracket
    if bracket is None or (bracket.take is None and bracket.stop is None):
        return ()
    exit_side = "sell" if intent.side == "buy" else "buy"
    out = []
    if bracket.stop is not None:
        out.append(order_intent(
            intent.strategy_id,
            intent.symbol,
            exit_side,
            intent.qty,
            intent.signal_ts,
            order_type="stop",
            stop_price=bracket.stop,
            tif="day",
            reason="stop de bracket",
            intent_id=f"{intent.intent_id}-stop",
        ))
    if bracket.take is not None:
        out.append(order_intent(
            intent.strategy_id,
            intent.symbol,
            exit_side,
            intent.qty,
            intent.signal_ts,
            order_type="limit",
            limit_price=bracket.take,
            tif="day",
            reason="take de bracket",
            intent_id=f"{intent.intent_id}-take",
        ))
    return tuple(out)


def _match_leg(leg_type: str, children: Sequence[OrderIntent]) -> Optional[OrderIntent]:
    if leg_type in ("stop", "stop_limit"):
        wanted = "stop"
    elif leg_type == "limit":
        wanted = "limit"
    else:
        return None
    for child in children:
        if child.order_type == wanted:
            return child
    return None


def _last_close(symbol: str, bars: Optional[IntradayBars]) -> Optional[float]:
    if bars is None or symbol not in getattr(bars.close, "columns", ()):
        return None
    series = bars.close[symbol].dropna()
    if series.empty:
        return None
    price = float(series.iloc[-1])
    if price <= 0:
        return None
    return price


def _back_one_second(stamp: str) -> str:
    """El ``after`` de ``GET /v2/orders`` es exclusivo: se relee un segundo de margen."""
    parsed = _parse_time(stamp, None)  # type: ignore[arg-type]
    if parsed is None:
        return stamp
    return (parsed - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_time(value: object, fallback: datetime) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        return fallback
    return parsed.astimezone(UTC)
