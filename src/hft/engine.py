"""Motor único intradía (HFT-06).

Backtest, simulación y paper llaman a ``Engine.step``. Cambian el
proveedor de barras y el adaptador de ejecución, no el camino.

Un paso, para cada barra nueva que ya es final:

1. Incorpora la barra al buffer. Una barra que todavía no cumplió
   cierre + gracia no entra. Una republicación posterior se ignora.
2. Suelta las intenciones cuya barra de ejecución ya llegó
   (``signal_ts + latency_bars``). Con ``latency_bars = 1`` eso es t+1.
3. Calcula features sobre las barras finales ``<= t`` con la función
   inyectada (la misma en backtest y en vivo). La ventana no es toda la
   historia: trae las sesiones previas que se calientan al arrancar
   (``history_sessions``) y la sesión en curso. Un backtest de un año no
   copia el año en cada barra, y el paper ve lo mismo que antes.
4. Pide intenciones a cada estrategia. ``on_bar`` no ve el reloj de
   llegada: el contexto lleva el instante en que la barra quedó final.
   Cada estrategia ve lo suyo (HFT-PAPER-08): sus posiciones, que salen
   de ``Attribution``, y lo que falta llenar de sus órdenes vivas
   (``pending``). El libro entero va aparte, solo para el equity.
   Si la estrategia pide cancelar (``take_cancels``), el motor cancela
   sus órdenes vivas en esos símbolos antes de aprobar lo nuevo.
   Una intención sobre un símbolo de otro dueño (posición u orden viva)
   queda rechazada como ``simbolo_de_otra_estrategia``.
5. El sizer fija cantidades. Lo que omite por falta de cupo queda
   como rechazo ``sizing_sin_cupo``. El riesgo aprueba o rechaza, con motivo.
   Si una posición abierta no tiene marca, no hay NAV que medir: todo
   queda rechazado como ``libro_sin_marca`` y ni el sizer ni el riesgo lo ven.
6. Lo aprobado espera a su barra de ejecución. Lo que ya maduró se
   envía a ``execution.submit`` y los fills entran al libro y a la
   atribución, todo o nada.
   Cada barra final se le enseña al adaptador, aunque no haya órdenes
   nuevas: un límite o un stop que quedó vivo tiene que ver el rango.
   Un bracket puede devolver intenciones hijas (``take_spawned``). Si
   el libro rechaza el lote, ``rollback_last`` deshace el envío.
7. Queda un evento en el journal: instante, intenciones, rechazos,
   órdenes, fills y estado.

``step`` no llama al reloj. Quien corre el motor le pasa ``now``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hft.attribution import Attribution
from src.hft.book import HftBook
from src.hft.calendar import DEFAULT_GRACE_S, NY, SessionCalendar, bar_close_time, is_bar_final
from src.hft.config import HftConfig
from src.hft.data.bars import DEFAULT_FEED, IntradayBars, IntradayMeta, canonical_timeframe
from src.hft.strategies.base import Strategy, StrategyContext, validate_strategy
from src.hft.types import QTY_TOLERANCE, Fill, OrderIntent, RiskEvent, ensure_utc, make_intent_id

logger = logging.getLogger("hft.engine")

# backtest y sim pueden correr research_only. paper no: el paper no
# simula la cola (HFT-09E).
EXECUTION_MODES = ("backtest", "sim", "paper")
# Motivo del rechazo de una intención que el sizer omitió por falta de cupo.
SIZING_NO_ROOM = "sizing_sin_cupo"
# Motivo del rechazo cuando una posición abierta todavía no tiene marca.
UNMARKED_BOOK = "libro_sin_marca"
# Motivo del rechazo de una intención sobre un símbolo de otro dueño (HFT-PAPER-08).
OTHER_OWNER = "simbolo_de_otra_estrategia"
# Dueño de la casa de lo que el libro ya traía al construir el motor.
INHERITED_OWNER = "reconcile"

UTC = timezone.utc
_EPS = 1e-8
_FRAME_NAMES = ("open", "high", "low", "close", "volume", "vwap", "trades")
_ANCHOR = datetime(2024, 1, 2, tzinfo=UTC)


class Clock(Protocol):
    """Reloj inyectado. El motor lo guarda y no lo consulta en ``step``."""

    def now(self) -> datetime:
        """Instante actual, con zona."""


class Sizer(Protocol):
    """Fija cantidades. Devuelve las intenciones en el mismo orden.

    Puede omitir la que ya no cabe (cupo de apalancamiento o tope por
    símbolo). El motor la registra como rechazo ``sizing_sin_cupo``.
    No puede reordenar ni agregar intenciones.
    """

    def size(
        self,
        intents: Sequence[OrderIntent],
        state: "EngineState",
        now: datetime,
    ) -> Sequence[OrderIntent]:
        """Intenciones con la cantidad ya decidida."""


class RiskGate(Protocol):
    """Aprueba o rechaza. El motivo del rechazo entra al journal."""

    def filter(
        self,
        intents: Sequence[OrderIntent],
        state: "EngineState",
        now: datetime,
    ) -> Tuple[Sequence[OrderIntent], Sequence[Tuple[OrderIntent, str]]]:
        """``(aprobadas, rechazadas)``. Cada rechazo es ``(intención, motivo)``."""


class Execution(Protocol):
    """Adaptador de ejecución. El simulador pesimista es HFT-07.

    ``now`` es el instante de decisión de la barra de ejecución (cierre +
    gracia), no la hora a la que el dato apareció. El precio sale de
    ``bars``.
    """

    def submit(
        self,
        intents: Sequence[OrderIntent],
        *,
        now: datetime,
        bars: IntradayBars,
    ) -> Sequence[Fill]:
        """Fills nuevos de este envío, también los de órdenes que seguían
        vivas. Vacío si nada se ejecutó.

        Opcional: ``take_spawned()`` devuelve intenciones hijas (el stop
        y el take de un bracket) creadas en ese ``submit``. Opcional:
        ``rollback_last()`` deshace el envío si el libro lo rechaza, y
        ``commit_last()`` olvida el punto de deshacer cuando el libro
        ya lo aceptó.
        """


FeaturesFn = Callable[[IntradayBars], object]


def execution_mode(mode: Optional[str]) -> str:
    """``backtest``, ``sim`` o ``paper``. None es simulación."""
    if mode is None:
        return "sim"
    if not isinstance(mode, str):
        raise TypeError(f"mode debe ser str, llegó {type(mode).__name__}")
    name = mode.strip().lower()
    if name not in EXECUTION_MODES:
        raise ValueError(f"mode {mode!r} no está en {EXECUTION_MODES}")
    return name


def research_only_ids(strategies: Sequence[object]) -> Tuple[str, ...]:
    """Ids cuya TheoryCard tiene ``research_only``."""
    found = []
    for strategy in strategies:
        theory = getattr(strategy, "theory", None)
        if getattr(theory, "research_only", False):
            found.append(str(getattr(strategy, "id", "?")))
    return tuple(found)


def assert_paper_allowed(strategies: Sequence[object], mode: str) -> None:
    """Paper no corre una estrategia ``research_only``.

    Simulación y backtest sí. Avellaneda–Stoikov existe para ver el
    riesgo de inventario: el paper no simula la cola y no es base para
    capital. ``scripts/run_intraday.py`` llama a esta misma función.
    """
    if execution_mode(mode) != "paper":
        return
    blocked = research_only_ids(strategies)
    if blocked:
        raise ValueError(
            "modo paper rechaza estrategias research_only: "
            + ", ".join(blocked)
            + ". No son base para capital: el paper no simula la cola "
            "y IEX ve una fracción del flujo."
        )


@dataclass(frozen=True)
class EngineState:
    """Foto del libro que ven el sizer y el riesgo.

    ``marks`` es el último cierre de cada símbolo hasta la barra: el sizer
    valora con él el tope por símbolo, igual que el riesgo. No va al journal.
    ``pending`` es lo que falta llenar de las órdenes vivas, una fila por
    símbolo y lado (compras en positivo, ventas en negativo), y
    ``open_orders`` cuántas son (ver ``Engine._working``). Tampoco van
    al journal.

    ``unmarked`` son las posiciones abiertas sin ninguna barra todavía (un
    relanzamiento antes de la primera). Con alguna, ``unrealized_pnl`` y
    ``nav`` son None: no se valoran al costo ni a $1/acción.
    """

    cash: float
    positions: Tuple[Tuple[str, float], ...]
    realized_pnl: float
    unrealized_pnl: Optional[float]
    fees: float
    nav: Optional[float]
    marks: Tuple[Tuple[str, float], ...] = ()
    pending: Tuple[Tuple[str, float], ...] = ()
    open_orders: int = 0
    unmarked: Tuple[str, ...] = ()

    def position(self, symbol: str) -> float:
        wanted = str(symbol).strip().upper()
        for name, qty in self.positions:
            if name == wanted:
                return float(qty)
        return 0.0

    def pending_qty(self, symbol: str) -> float:
        wanted = str(symbol).strip().upper()
        return float(sum(qty for name, qty in self.pending if name == wanted))

    def mark(self, symbol: str) -> Optional[float]:
        wanted = str(symbol).strip().upper()
        for name, price in self.marks:
            if name == wanted:
                return float(price)
        return None

    def as_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {name: qty for name, qty in self.positions},
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees": self.fees,
            "nav": self.nav,
            "unmarked": list(self.unmarked),
        }


@dataclass(frozen=True)
class JournalEvent:
    """Un evento del journal, uno por barra procesada (o uno vacío)."""

    arrived_at: datetime
    bar_ts: Optional[datetime]
    decision_time: Optional[datetime]
    intents: Tuple[OrderIntent, ...]
    rejections: Tuple[RiskEvent, ...]
    orders: Tuple[OrderIntent, ...]
    fills: Tuple[Fill, ...]
    held: Tuple[str, ...]
    state: dict
    # intent_id que el motor canceló en esta barra porque la estrategia lo
    # pidió (``take_cancels``). Incluye lo que esperaba su barra y no salió.
    cancelled: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    """Lo que produjo ``step``. ``arrived_at`` es el ``now`` que pasó el caller."""

    arrived_at: datetime
    events: Tuple[JournalEvent, ...]

    @property
    def intents(self) -> Tuple[OrderIntent, ...]:
        return tuple(intent for event in self.events for intent in event.intents)

    @property
    def fills(self) -> Tuple[Fill, ...]:
        return tuple(fill for event in self.events for fill in event.fills)

    @property
    def rejections(self) -> Tuple[RiskEvent, ...]:
        return tuple(event_ for event in self.events for event_ in event.rejections)


@dataclass
class _Held:
    intent: OrderIntent
    ready_ts: pd.Timestamp


class Engine:
    """Una ruta para backtest, simulación y paper.

    Args:
        strategies: estrategias del paso. Se ordenan por ``id``.
        features_fn: ventana de barras finales → features. Se llama igual
            en los dos modos.
        risk: ``filter(intents, state, now)``.
        sizer: ``size(intents, state, now)``.
        execution: ``submit(intents, now=, bars=)``.
        calendar: sesiones ya cargadas. Solo se decide dentro de sesión.
        clock: se guarda para el runner. ``step`` no lo lee.
        book: libro. Si falta, se abre uno con ``config.initial_cash``.
        config: timeframe, gracia, latencia y feed. Un argumento explícito
            pisa a la config.
        mode: ``backtest``, ``sim`` (default) o ``paper``. En paper una
            estrategia ``research_only`` no se construye.
    """

    def __init__(
        self,
        strategies: Sequence[Strategy],
        features_fn: FeaturesFn,
        risk: RiskGate,
        sizer: Sizer,
        execution: Execution,
        calendar: SessionCalendar,
        clock: Clock,
        *,
        book: Optional[HftBook] = None,
        config: Optional[HftConfig] = None,
        timeframe: Optional[str] = None,
        grace_s: Optional[float] = None,
        latency_bars: Optional[int] = None,
        rng: Optional[object] = None,
        mode: Optional[str] = None,
    ) -> None:
        if not callable(features_fn):
            raise TypeError("features_fn debe ser callable")
        for name, obj, method in (
            ("risk", risk, "filter"),
            ("sizer", sizer, "size"),
            ("execution", execution, "submit"),
            ("clock", clock, "now"),
        ):
            if not callable(getattr(obj, method, None)):
                raise TypeError(f"{name} no implementa {method}")
        if not hasattr(calendar, "in_session"):
            raise TypeError("calendar no implementa in_session")
        ids = [strategy.id for strategy in strategies]
        if len(ids) != len(set(ids)):
            raise ValueError(f"estrategias con id repetido: {ids}")
        for strategy in strategies:
            errors = validate_strategy(strategy)
            if errors:
                raise ValueError(f"estrategia {getattr(strategy, 'id', '?')} inválida: " + "; ".join(errors))
        self.mode = execution_mode(mode)
        assert_paper_allowed(strategies, self.mode)
        self.strategies: Tuple[Strategy, ...] = tuple(sorted(strategies, key=lambda s: s.id))
        self.features_fn = features_fn
        self.risk = risk
        self.sizer = sizer
        self.execution = execution
        self.calendar = calendar
        self.clock = clock
        self.config = config
        if timeframe is None:
            timeframe = config.timeframe if config is not None else "1Min"
        if grace_s is None:
            grace_s = config.grace_s if config is not None else DEFAULT_GRACE_S
        if latency_bars is None:
            latency_bars = config.latency_bars if config is not None else 1
        self.timeframe = canonical_timeframe(timeframe)
        self.grace_s = float(grace_s)
        self.latency_bars = int(latency_bars)
        if self.grace_s < 0:
            raise ValueError("grace_s no puede ser negativo")
        if self.latency_bars < 1:
            raise ValueError("latency_bars debe ser >= 1 (ejecución como pronto en t+1)")
        if book is None:
            if config is None:
                raise ValueError("hace falta book o config para abrir el libro")
            book = HftBook(config.initial_cash)
        self.book = book
        # Mismos fills que el libro: la suma de los dueños es la posición.
        self.attribution = Attribution(ids)
        for symbol, qty in book.positions().items():
            self.attribution.move(INHERITED_OWNER, symbol, qty)
        self._rng = rng
        self.feed = config.feed if config is not None else DEFAULT_FEED
        self._step = bar_close_time(_ANCHOR, self.timeframe) - _ANCHOR
        # Todo lo ingerido: los reportes del backtest y el runner leen el
        # rango entero. Las ventanas salen de ``_recent`` (ver ``_retain``).
        self._frames: Optional[dict] = None
        self._recent: Optional[dict] = None
        # Último cierre de cada símbolo en lo que quedó fuera de ``_recent``.
        self._older_marks: dict = {}
        self._seen: set = set()
        self._pending: list = []
        self._live: dict = {}
        self._filled_qty: dict = {}
        self._children: set = set()
        self.journal: list = []

    def required_history(self) -> Tuple[int, int]:
        """Máximo ``(barras, sesiones)`` que piden las estrategias."""
        bars = 0
        sessions = 0
        for strategy in self.strategies:
            need_bars, need_sessions = strategy.required_history()
            bars = max(bars, int(need_bars))
            sessions = max(sessions, int(need_sessions))
        return bars, sessions

    def history_sessions(self) -> int:
        """Sesiones previas a la barra que la ventana conserva.

        Son las que se calientan al arrancar: el backtest mete
        ``max(sesiones, ⌈barras / sesión⌉)`` y el paper ``max(sesiones, 1)``.
        Así la ventana de cualquier día del backtest tiene la forma de la
        del primer día, y al paper no le falta nada.
        """
        need_bars, need_sessions = self.required_history()
        per_session = max(1, int(round(timedelta(minutes=390) / self._step)))
        return max(int(need_sessions), math.ceil(need_bars / per_session), 1)

    def step(self, now: datetime, final_bars: IntradayBars) -> StepResult:
        """Avanza con las barras que ``now`` ya puede usar.

        ``final_bars`` puede traer historia ya vista y la barra que todavía
        se está formando. Solo entra lo final y no visto.
        """
        arrived = ensure_utc(now, "now")
        # El buffer puede recibir varias barras finales de una vez. Cada
        # una se decide con la ventana ``<= t``: la barra t+1, aunque ya
        # esté guardada, no entra en las features de t.
        new_ts = self._ingest(final_bars, arrived)
        if not new_ts:
            event = self._event(arrived, None, None, (), (), (), (), self._state_at(None))
            self.journal.append(event)
            return StepResult(arrived_at=arrived, events=(event,))
        events = []
        for ts in new_ts:
            events.append(self._on_bar(arrived, ts))
        return StepResult(arrived_at=arrived, events=tuple(events))

    def _on_bar(self, arrived: datetime, ts: pd.Timestamp) -> JournalEvent:
        bar_ts = _as_datetime(ts)
        decision_time = bar_close_time(bar_ts, self.timeframe) + timedelta(seconds=self.grace_s)
        window = self._window_until(ts)
        matured = [item for item in self._pending if item.ready_ts <= ts]
        orders = tuple(item.intent for item in matured)
        # Si el libro rechaza el lote, lo pendiente sigue donde estaba y
        # el adaptador puede deshacer el envío.
        produced = tuple(self.execution.submit(orders, now=decision_time, bars=window))
        try:
            self._accept(orders, produced)
        except Exception:
            rollback = getattr(self.execution, "rollback_last", None)
            if callable(rollback):
                rollback()
            raise
        commit = getattr(self.execution, "commit_last", None)
        if callable(commit):
            commit()
        if matured:
            self._pending = [item for item in self._pending if item.ready_ts > ts]
        fills = produced
        # La cotización vieja ya pudo llenarse en esta barra. Lo que sigue
        # vivo se retira antes de colgar la decisión nueva: si no, el
        # límite anterior se ejecuta junto con el de reemplazo.
        self._supersede_passive_limits()
        intents: Tuple[OrderIntent, ...] = ()
        rejections: Tuple[RiskEvent, ...] = ()
        cancelled: Tuple[str, ...] = ()
        if self.calendar.in_session(bar_ts):
            features = self.features_fn(window)
            # Los fills de esta barra ya están en el libro: la estrategia
            # decide con la posición que resulta de ejecutar lo decidido antes.
            state_now = self._state_at(ts)
            proposed = self._propose(bar_ts, decision_time, window, features, state_now)
            # Antes de aprobar lo nuevo: lo que la estrategia acaba de
            # proponer no entra en la cancelación de su propia barra. Cancelar
            # baja riesgo: corre aunque el libro no tenga marca.
            cancelled = self._cancel_requested()
            proposed, foreign = self._exclusive(proposed, decision_time)
            if state_now.unmarked:
                intents, approved = (), ()
                rejections = foreign + _rejected(proposed, UNMARKED_BOOK, decision_time)
                if proposed:
                    logger.warning(
                        "sin marca para %s: %d intenciones rechazadas",
                        ", ".join(state_now.unmarked), len(proposed),
                    )
            else:
                intents, unsized = self._size(proposed, state_now, decision_time)
                approved, rejections = self._filter(intents, state_now, decision_time)
                rejections = foreign + unsized + rejections
            for intent in approved:
                ready = ts + self._step * self.latency_bars
                self._pending.append(_Held(intent=intent, ready_ts=pd.Timestamp(ready)))
        event = self._event(
            arrived, bar_ts, decision_time, intents, rejections, orders, fills, self._state_at(ts),
            cancelled,
        )
        self.journal.append(event)
        logger.debug(
            "barra %s intenciones=%d rechazos=%d ordenes=%d fills=%d",
            bar_ts.isoformat(),
            len(intents),
            len(rejections),
            len(orders),
            len(fills),
        )
        return event

    def _supersede_passive_limits(self) -> None:
        """Cancela límites pasivos de quien recotiza cada barra.

        Lo pide ``replaces_passive_limits`` (Avellaneda–Stoikov). El
        resto de las estrategias no pierde órdenes. Sin ``cancel`` en
        el adaptador, solo se descartan las que todavía no se enviaron.
        """
        managed = {
            strategy.id
            for strategy in self.strategies
            if getattr(strategy, "replaces_passive_limits", False)
        }
        if not managed:
            return
        self._pending = [
            item
            for item in self._pending
            if not (
                item.intent.strategy_id in managed
                and item.intent.order_type == "limit"
            )
        ]
        cancel = getattr(self.execution, "cancel", None)
        listing = getattr(self.execution, "open_orders", None)
        if not callable(cancel) or not callable(listing):
            return
        for intent in listing():
            if intent.strategy_id in managed and intent.order_type == "limit":
                cancel(intent.intent_id)

    def _propose(self, bar_ts, decision_time, window, features, state) -> Tuple[OrderIntent, ...]:
        # Cada estrategia ve sus posiciones y sus órdenes vivas, no las de
        # las otras: una salida o un flatten no cierra lo ajeno (HFT-PAPER-08).
        working = self._working_intents()
        stamped = []
        for strategy in self.strategies:
            ctx = StrategyContext(
                now=decision_time,
                bar_ts=bar_ts,
                bars=window,
                features=features,
                positions=self.attribution.owned(strategy.id),
                cash=state.cash,
                realized_pnl=state.realized_pnl,
                rng=self._rng,
                pending=self._working(strategy.id, working)[0],
                book_positions=state.positions,
            )
            produced = list(strategy.on_bar(ctx))
            for seq, intent in enumerate(produced):
                if not isinstance(intent, OrderIntent):
                    raise TypeError(
                        f"{strategy.id} devolvió {type(intent).__name__}, se esperaba OrderIntent"
                    )
                stamped.append(replace(
                    intent,
                    strategy_id=strategy.id,
                    symbol=intent.symbol.strip().upper(),
                    signal_ts=bar_ts,
                    intent_id=make_intent_id(
                        strategy.id, intent.symbol, bar_ts, intent.side, seq,
                    ),
                ))
        return tuple(stamped)

    def _exclusive(self, proposed, now) -> Tuple[Tuple[OrderIntent, ...], Tuple[RiskEvent, ...]]:
        """Un dueño por símbolo (HFT-PAPER-08).

        Una intención de ``X`` sobre ``s`` se rechaza como
        ``simbolo_de_otra_estrategia`` si otro dueño tiene posición en
        ``s`` o una orden viva ahí (lo que espera su barra y lo abierto en
        el adaptador, con hijos de bracket). El libro y la cuenta son
        netos: dos dueños en un símbolo cruzan cero o se lavan entre sí.

        Dentro de la barra, las estrategias van en orden de id. La primera
        que pasa con un símbolo lo reclama para el resto de la barra,
        aunque después el sizer o el riesgo la rechacen. Con
        ``atomic_entries``, una entrada rechazada arrastra a todas las
        entradas de esa estrategia en la barra.
        """
        if not proposed:
            return (), ()
        live: dict = {}
        for intent in self._working_intents():
            live.setdefault(intent.symbol, set()).add(intent.strategy_id)
        atomic = {
            strategy.id for strategy in self.strategies
            if getattr(strategy, "atomic_entries", False)
        }
        claimed: dict = {}
        kept = []
        refused = []
        for strategy_id, items in groupby(proposed, key=lambda intent: intent.strategy_id):
            group = list(items)
            blocked = [self._foreign(intent, live, claimed) for intent in group]
            if strategy_id in atomic and any(
                hit and self._is_entry(intent) for intent, hit in zip(group, blocked)
            ):
                blocked = [hit or self._is_entry(intent) for intent, hit in zip(group, blocked)]
            for intent, hit in zip(group, blocked):
                if hit:
                    refused.append(intent)
                    continue
                kept.append(intent)
                claimed.setdefault(intent.symbol, strategy_id)
        if refused:
            logger.debug(
                "símbolo de otra estrategia: %s",
                ", ".join(intent.intent_id for intent in refused),
            )
        return tuple(kept), _rejected(refused, OTHER_OWNER, now)

    def _foreign(self, intent: OrderIntent, live: dict, claimed: dict) -> bool:
        owner = intent.strategy_id
        if any(name != owner for name in self.attribution.owners(intent.symbol)):
            return True
        if any(name != owner for name in live.get(intent.symbol, ())):
            return True
        return claimed.get(intent.symbol, owner) != owner

    def _is_entry(self, intent: OrderIntent) -> bool:
        """La intención aumenta la exposición propia de su estrategia."""
        held = self.attribution.position(intent.strategy_id, intent.symbol)
        return held >= 0 if intent.side == "buy" else held <= 0

    def _cancel_requested(self) -> Tuple[str, ...]:
        """Cancela lo vivo que cada estrategia pidió con ``take_cancels``.

        La estrategia no ve la ejecución (``on_bar`` es pura), así que el
        motor cancela por ella: igual en backtest, sim y paper, cambia solo
        el adaptador. Pasa en la pata colgada de pairs_ou: la viva tiene
        que morir o se llena después y deja la cuenta desnuda
        (HFT-PAPER-16). Solo toca órdenes de quien lo pidió en esos
        símbolos. Lo que espera su barra se descarta sin salir; lo enviado
        va a ``execution.cancel``. Los hijos de un bracket se quedan:
        protegen una posición. Un cancel que falla (la orden ya se llenó)
        queda en el log; el fill llega por el camino normal.
        """
        wanted = set()
        for strategy in self.strategies:
            take = getattr(strategy, "take_cancels", None)
            if not callable(take):
                continue
            for symbol in take():
                if not isinstance(symbol, str) or not symbol.strip():
                    raise TypeError(f"{strategy.id}: take_cancels debe devolver símbolos")
                wanted.add((strategy.id, symbol.strip().upper()))
        if not wanted:
            return ()

        def asked(intent: OrderIntent) -> bool:
            return (
                (intent.strategy_id, intent.symbol) in wanted
                and intent.intent_id not in self._children
            )

        dropped = [item.intent.intent_id for item in self._pending if asked(item.intent)]
        self._pending = [item for item in self._pending if not asked(item.intent)]
        done = list(dropped)
        cancel = getattr(self.execution, "cancel", None)
        listing = getattr(self.execution, "open_orders", None)
        if callable(cancel) and callable(listing):
            for intent in listing():
                if not asked(intent) or intent.intent_id in done:
                    continue
                try:
                    ok = bool(cancel(intent.intent_id))
                except Exception as exc:
                    logger.warning("cancel de %s falló: %s", intent.intent_id, exc)
                    continue
                if ok:
                    done.append(intent.intent_id)
                else:
                    logger.warning("cancel de %s: la orden ya no estaba viva", intent.intent_id)
        else:
            logger.warning("la ejecución no cancela; lo enviado de %s sigue vivo", sorted(wanted))
        if done:
            logger.info("canceladas a pedido de la estrategia: %s", ", ".join(done))
        return tuple(done)

    def _size(self, intents, state, now) -> Tuple[Tuple[OrderIntent, ...], Tuple[RiskEvent, ...]]:
        if not intents:
            return (), ()
        sized = tuple(self.sizer.size(intents, state, now))
        kept = iter(item.intent_id for item in intents)
        if not all(item.intent_id in kept for item in sized):
            raise ValueError("el sizer debe conservar los intent_id y su orden")
        returned = {item.intent_id for item in sized}
        omitted = [intent for intent in intents if intent.intent_id not in returned]
        return sized, _rejected(omitted, SIZING_NO_ROOM, now)

    def _filter(self, intents, state, now):
        if not intents:
            return (), ()
        approved, rejected = self.risk.filter(intents, state, now)
        approved = tuple(approved)
        events = []
        rejected_ids = []
        for item in rejected:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("cada rechazo debe ser (OrderIntent, motivo)")
            intent, reason = item
            if not isinstance(intent, OrderIntent):
                raise TypeError("el rechazo no trae OrderIntent")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("el rechazo necesita un motivo")
            rejected_ids.append(intent.intent_id)
            events.append(RiskEvent(
                ts=now,
                kind="rejected",
                reason=reason,
                symbol=intent.symbol,
                intent_id=intent.intent_id,
            ))
        known = {intent.intent_id for intent in intents}
        approved_ids = [intent.intent_id for intent in approved]
        covered = set(approved_ids) | set(rejected_ids)
        once = len(approved_ids) + len(rejected_ids) == len(known)
        if covered != known or not once:
            raise ValueError("el riesgo debe aprobar o rechazar cada intención, una vez")
        return approved, tuple(events)

    def _accept(self, orders: Sequence[OrderIntent], produced: Sequence[Fill]) -> None:
        live = dict(self._live)
        for intent in orders:
            live[intent.intent_id] = intent
        spawned = self._spawned_intents()
        for intent in spawned:
            live[intent.intent_id] = intent
        if produced:
            self._check_known(produced, live)
            self._apply(produced, live)
        self._live = live
        self._children.update(intent.intent_id for intent in spawned)

    def seed(self, intents: Sequence[OrderIntent], fills: Sequence[Fill]) -> None:
        """Posiciones heredadas al relanzar, sin pasar por la ejecución.

        Entran al libro y a la atribución con el dueño de cada intención
        (``strategy_id``). Una estrategia del motor gestiona lo suyo; lo de
        ``reconcile`` no es de nadie y lo cierra el flatten de sesión.
        """
        live = dict(self._live)
        for intent in intents:
            live[intent.intent_id] = intent
        fills = tuple(fills)
        if fills:
            self._check_known(fills, live)
            self._apply(fills, live)
        self._live = live

    def _apply(self, fills: Sequence[Fill], live: dict) -> None:
        # El libro valida el lote; con el lote ya válido la atribución
        # no falla (mismos fills, mismas intenciones).
        self.book.apply_fills(fills, live)
        self.attribution.apply(fills, live)
        for fill in fills:
            self._filled_qty[fill.intent_id] = self._filled_qty.get(fill.intent_id, 0.0) + fill.qty

    def _spawned_intents(self) -> Tuple[OrderIntent, ...]:
        take = getattr(self.execution, "take_spawned", None)
        if not callable(take):
            return ()
        out = []
        for intent in take():
            if not isinstance(intent, OrderIntent):
                raise TypeError(f"take_spawned devolvió {type(intent).__name__}")
            out.append(intent)
        return tuple(out)

    def _check_known(self, fills: Sequence[Fill], live: dict) -> None:
        batch: dict = {}
        for fill in fills:
            if not isinstance(fill, Fill):
                raise TypeError(f"se esperaba Fill, llegó {type(fill).__name__}")
            intent = live.get(fill.intent_id)
            if intent is None:
                raise ValueError(f"fill de una intención no enviada: {fill.intent_id}")
            batch[fill.intent_id] = batch.get(fill.intent_id, 0.0) + fill.qty
            total = self._filled_qty.get(fill.intent_id, 0.0) + batch[fill.intent_id]
            if total - intent.qty >= QTY_TOLERANCE:
                raise ValueError(f"el fill excede la cantidad de {fill.intent_id}")

    def _ingest(self, bars: IntradayBars, now: datetime) -> list:
        if not isinstance(bars, IntradayBars):
            raise TypeError("final_bars debe ser IntradayBars")
        if bars.close.empty:
            return []
        fresh = []
        for ts in bars.close.index:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                raise ValueError("el índice de barras debe llevar zona")
            key = stamp.isoformat()
            if key in self._seen:
                continue
            py_ts = _as_datetime(stamp)
            if not is_bar_final(py_ts, now, self.timeframe, self.grace_s):
                continue
            fresh.append(stamp)
            self._seen.add(key)
        fresh.sort()
        if not fresh:
            return []
        piece = {name: getattr(bars, name).loc[fresh] for name in _FRAME_NAMES}
        if self._frames is None:
            self._frames = {name: frame.copy() for name, frame in piece.items()}
        else:
            columns = sorted(set(self._frames["close"].columns) | set(piece["close"].columns))
            for name in _FRAME_NAMES:
                old = self._frames[name].reindex(columns=columns)
                new = piece[name].reindex(columns=columns)
                self._frames[name] = pd.concat([old, new])
        self._frames["close"].index.name = "ts"
        self._retain(fresh[0])
        return fresh

    def _retain(self, first: pd.Timestamp) -> None:
        """Deja en ``_recent`` lo que todavía puede pedir una barra ``>= first``.

        Conserva las ``history_sessions()`` sesiones con barras antes del
        día de ``first``, las últimas ``barras`` de ``required_history``
        hasta ``first`` y todo desde ese día. Lo que queda afuera es
        anterior a cualquier barra que falte decidir: deja su último
        cierre en ``_older_marks`` para la marca del libro.
        """
        close = self._frames["close"]
        index = pd.DatetimeIndex(close.index)
        days = index.tz_convert(NY).normalize()
        prior = days[days < first.tz_convert(NY).normalize()].unique().sort_values()
        sessions = self.history_sessions()
        keep = np.ones(len(index), dtype=bool)
        if len(prior) > sessions:
            keep = np.asarray(days >= prior[-sessions])
            need_bars = self.required_history()[0]
            if need_bars > 0:
                # ns como ``first.value``: ``asi8`` va en la unidad del índice.
                stamps = index.as_unit("ns").asi8
                upto = np.sort(stamps[stamps <= first.value])
                keep |= stamps >= upto[-min(need_bars, len(upto))]
        if keep.all():
            self._recent = dict(self._frames)
            self._older_marks = {}
            return
        self._recent = {name: self._frames[name].loc[keep] for name in _FRAME_NAMES}
        older = close.loc[~keep]
        marks = {}
        for symbol in older.columns:
            series = older[symbol].dropna()
            if not series.empty:
                marks[str(symbol)] = float(series.iloc[-1])
        self._older_marks = marks

    def _window_until(self, ts: pd.Timestamp) -> IntradayBars:
        assert self._recent is not None
        index = self._recent["close"].index
        kept = index[index <= ts]
        frames = {name: self._recent[name].loc[kept].copy() for name in _FRAME_NAMES}
        columns = tuple(str(column) for column in frames["close"].columns)
        # Un símbolo cuyas barras quedaron todas afuera igual llegó antes de t.
        received = tuple(
            symbol
            for symbol in columns
            if symbol in self._older_marks or frames["close"][symbol].notna().any()
        )
        meta = IntradayMeta(
            source="engine",
            feed=self.feed,
            timeframe=self.timeframe,
            adjustment="raw",
            requested=columns,
            received=received,
        )
        return IntradayBars(meta=meta, **frames)

    def _marks(self, ts: Optional[pd.Timestamp]) -> dict:
        if self._recent is None:
            return {}
        close = self._recent["close"] if ts is None else self._recent["close"].loc[:ts]
        marks = {}
        for symbol in close.columns:
            series = close[symbol].dropna()
            if series.empty:
                older = self._older_marks.get(str(symbol))
                if older is not None:
                    marks[str(symbol)] = older
                continue
            marks[str(symbol)] = float(series.iloc[-1])
        return marks

    def _state_at(self, ts: Optional[pd.Timestamp]) -> EngineState:
        positions = tuple(self.book.positions().items())
        marks = self._marks(ts)
        unmarked = tuple(symbol for symbol, _qty in positions if symbol not in marks)
        if unmarked:
            unrealized = None
            nav = None
        elif positions:
            unrealized = float(self.book.unrealized_pnl(marks))
            nav = float(self.book.nav(marks))
        else:
            unrealized = 0.0
            nav = float(self.book.cash)
        pending, count = self._working()
        return EngineState(
            cash=float(self.book.cash),
            positions=positions,
            realized_pnl=float(self.book.realized_pnl),
            unrealized_pnl=unrealized,
            fees=float(self.book.fees),
            nav=nav,
            marks=tuple(sorted(marks.items())),
            pending=pending,
            open_orders=count,
            unmarked=unmarked,
        )

    def _working_intents(self) -> Tuple[OrderIntent, ...]:
        """Órdenes vivas, una vez cada una: lo aprobado que espera su barra
        y lo que el adaptador ve abierto, con algo por llenar. Hijos incluidos.
        """
        working = [item.intent for item in self._pending]
        listing = getattr(self.execution, "open_orders", None)
        if callable(listing):
            working.extend(listing())
        out = []
        seen: set = set()
        for intent in working:
            if intent.intent_id in seen:
                continue
            seen.add(intent.intent_id)
            if intent.qty - self._filled_qty.get(intent.intent_id, 0.0) <= _EPS:
                continue
            out.append(intent)
        return tuple(out)

    def _working(
        self,
        strategy_id: Optional[str] = None,
        working: Optional[Sequence[OrderIntent]] = None,
    ) -> Tuple[Tuple[Tuple[str, float], ...], int]:
        """Lo que falta llenar de cada símbolo y lado, y cuántas órdenes son.

        Suma lo aprobado que espera su barra de ejecución y lo que el
        adaptador todavía ve vivo (``open_orders``), menos los fills que
        el libro ya aceptó. Los hijos de un bracket no cuentan: protegen
        una posición, no la abren. En paper un límite puede seguir vivo
        varias barras; sin esto la estrategia ve el libro plano y repite
        la entrada (HFT-PAPER-15). La cuenta es la que ve el riesgo para
        ``max_open_orders`` (HFT-PAPER-10).

        Compras y ventas van en filas separadas, no netas: el resto vivo
        de una compra (+6) y una venta de salida viva (−4) darían +2, y la
        salida se repetiría (HFT-PAPER-16).

        Con ``strategy_id`` cuenta solo las órdenes de esa estrategia: es
        el ``pending`` de su contexto (HFT-PAPER-08). Sin él, todas.
        """
        if working is None:
            working = self._working_intents()
        totals: dict = {}
        count = 0
        for intent in working:
            if intent.intent_id in self._children:
                continue
            if strategy_id is not None and intent.strategy_id != strategy_id:
                continue
            left = intent.qty - self._filled_qty.get(intent.intent_id, 0.0)
            count += 1
            key = (intent.symbol, intent.side)
            totals[key] = totals.get(key, 0.0) + left
        pending = tuple(sorted(
            (symbol, qty if side == "buy" else -qty)
            for (symbol, side), qty in totals.items()
            if qty > _EPS
        ))
        return pending, count

    def _event(
        self, arrived, bar_ts, decision_time, intents, rejections, orders, fills, state,
        cancelled=(),
    ):
        return JournalEvent(
            arrived_at=arrived,
            bar_ts=bar_ts,
            decision_time=decision_time,
            intents=tuple(intents),
            rejections=tuple(rejections),
            orders=tuple(orders),
            fills=tuple(fills),
            held=tuple(item.intent.intent_id for item in self._pending),
            state=state.as_dict(),
            cancelled=tuple(cancelled),
        )


def _rejected(intents: Sequence[OrderIntent], reason: str, now: datetime) -> Tuple[RiskEvent, ...]:
    return tuple(
        RiskEvent(
            ts=now,
            kind="rejected",
            reason=reason,
            symbol=intent.symbol,
            intent_id=intent.intent_id,
        )
        for intent in intents
    )


def _as_datetime(ts: pd.Timestamp) -> datetime:
    py = pd.Timestamp(ts).to_pydatetime()
    if py.tzinfo is None:
        raise ValueError("la barra no trae zona")
    return py.astimezone(UTC)
