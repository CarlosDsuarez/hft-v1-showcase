"""Simulador de ejecución pesimista (HFT-07).

Determinista dado el ``rng``. Los fills salen de ``CostModel``: el mismo
objeto que valora un fill de paper. No hay otro slippage ni otras fees.

Cuándo se llena, con la barra de señal ``t`` y ``latency_bars`` extra
del simulador (0 por defecto, porque el motor ya suelta la orden en
t+1):

- market: apertura de la primera barra ``>= t + (1 + latency_bars)``,
  más o menos (medio spread + impacto). Compra suma, venta resta.
- marketable_limit: si el límite alcanza ese precio con slippage, fill
  a ``min`` (compra) o ``max`` (venta) entre el límite y la apertura
  con slippage. Si no alcanza, queda como límite pasivo.
- limit pasivo, modo ``cross`` (el de defecto): solo si la barra cruza
  el límite por al menos un tick (compra: ``low <= límite − tick``;
  venta: ``high >= límite + tick``). Precio = límite. Cantidad
  ``<= participation_cap × volumen`` de la barra. No hay prioridad de
  cola: el tope es lo máximo, no una garantía de ser el primero.
- modo ``touch``: basta con tocar el límite (``low <= límite`` en
  compra). Es una sensibilidad optimista. ``label`` lo dice
  (``optimistic-touch``); el modo de defecto es ``pessimistic-cross``.
- stop: dispara si high/low cruza el stop. El fill es el precio en
  contra, ``max(stop, apertura)`` en una compra y ``min(stop, apertura)``
  en una venta, más el slippage. Un gap se cobra.
- bracket: al llenarse el padre se arman stop y take. Si los dos caben
  en la misma barra, se asume el stop y se cancela el take. Son OCO
  también en parciales: lo que llena uno (un take topado por volumen) ya
  no lo cubre el otro, que baja lo mismo. Los hijos
  salen por ``take_spawned`` para que el libro los vea como órdenes
  propias (el fill no puede reutilizar el id del padre: el lado es el
  contrario).
- flatten: órdenes market para cerrar. El fill es el de un market, en
  la barra siguiente.

Un parcial (``partial_prob``, 0.10 como el paper) consume el ``rng`` y
deja el resto vivo. Con probabilidad 0 el fill es completo, salvo el
tope de participación de un límite.

``submit`` es idempotente por ``intent_id``. ``rollback_last`` deshace
el último ``submit`` si el libro rechaza el lote.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.hft.calendar import bar_close_time
from src.hft.costs import CostModel, Quote, _finite_or_none
from src.hft.data.bars import IntradayBars, canonical_timeframe
from src.hft.features import corwin_schultz_spread, roll_spread
from src.hft.types import Bracket, Fill, OrderIntent, ensure_utc, order_intent

UTC = timezone.utc
_EPS = 1e-8
_ASSET = Union[str, Mapping[str, str]]
QuoteFn = Callable[[str, pd.Timestamp], Optional[Quote]]

FILL_MODES = ("cross", "touch")


@dataclass
class _Working:
    """Orden viva. ``remaining`` baja con cada fill.

    ``rank`` es su lugar en ``SimExecution._orders``: el orden de los fills.
    """

    intent: OrderIntent
    remaining: float
    kind: str
    ready: pd.Timestamp
    group: Optional[str] = None
    resting: bool = False
    active: bool = True
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    force_full: bool = False
    rank: int = 0


class SimExecution:
    """Adaptador ``submit(intents, now=, bars=)`` del motor.

    Args:
        costs: el único modelo de costos.
        latency_bars: barras extra después de t+1. 0 cuando el motor ya
            esperó su ``latency_bars``.
        fill_mode: ``cross`` (pesimista) o ``touch`` (optimista, etiquetado).
        participation_cap: fracción máxima del volumen de la barra para
            un límite pasivo.
        partial_prob: probabilidad de un parcial. El resto sigue vivo.
        adv: símbolo → ADV en dólares, ya causal (días previos).
        sigma: símbolo → σ diaria, ya causal.
        quote_fn: ``(símbolo, ts) → Quote`` o None. Si hay quote, no se
            usan Roll ni Corwin–Schultz.
        asset_class: ``us_equity``, ``crypto``, o un mapa por símbolo.
        rng: generador explícito. Si falta, ``default_rng(seed)``.
    """

    def __init__(
        self,
        costs: CostModel,
        *,
        latency_bars: int = 0,
        fill_mode: str = "cross",
        participation_cap: float = 0.10,
        partial_prob: float = 0.0,
        adv: Optional[Mapping[str, float]] = None,
        sigma: Optional[Mapping[str, float]] = None,
        quote_fn: Optional[QuoteFn] = None,
        asset_class: _ASSET = "us_equity",
        timeframe: str = "1Min",
        roll_window: int = 30,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not isinstance(costs, CostModel):
            raise TypeError("costs debe ser CostModel")
        if isinstance(latency_bars, bool) or not isinstance(latency_bars, int) or latency_bars < 0:
            raise ValueError("latency_bars del simulador debe ser un int >= 0")
        if fill_mode not in FILL_MODES:
            raise ValueError(f"fill_mode {fill_mode!r} no está en {FILL_MODES}")
        if not 0 < float(participation_cap) <= 1:
            raise ValueError("participation_cap debe estar en (0, 1]")
        if not 0 <= float(partial_prob) <= 1:
            raise ValueError("partial_prob debe estar en [0, 1]")
        self.costs = costs
        self.latency_bars = int(latency_bars)
        self.fill_mode = fill_mode
        self.participation_cap = float(participation_cap)
        self.partial_prob = float(partial_prob)
        self.quote_fn = quote_fn
        self.asset_class = asset_class
        self.timeframe = canonical_timeframe(timeframe)
        self.roll_window = int(roll_window)
        self._adv = _num_map(adv, "adv")
        self._sigma = _num_map(sigma, "sigma")
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        anchor = datetime(2024, 1, 2, tzinfo=UTC)
        self._step = bar_close_time(anchor, self.timeframe) - anchor
        # Todas las órdenes de la corrida, en orden de inserción (el backtest
        # las lee). ``_live`` son las que todavía pueden llenarse, en ese
        # mismo orden: lo único que recorre una barra.
        self._orders: dict = {}
        self._live: dict = {}
        self._seen: set = set()
        self._spawned: list = []
        self._spread_cache: dict = {}
        self._last_snap = None
        # Barras y spreads que agregó el envío en curso, para deshacerlo.
        self._added_seen: list = []
        self._added_spreads: list = []
        self.label = "optimistic-touch" if fill_mode == "touch" else "pessimistic-cross"
        self.optimistic = fill_mode == "touch"

    @classmethod
    def from_config(cls, config: object, **kwargs: object) -> "SimExecution":
        """Simulador con el ``CostModel`` y la sección ``[sim]`` de la config."""
        costs = kwargs.pop("costs", None)
        if costs is None:
            costs = CostModel.from_config(config)
        sim = getattr(config, "sim", None)
        if sim is None:
            raise TypeError("la config no trae sim")
        timeframe = kwargs.pop("timeframe", getattr(config, "timeframe", "1Min"))
        return cls(
            costs,
            latency_bars=int(sim.latency_bars),
            fill_mode=str(sim.fill_mode),
            participation_cap=float(sim.participation_cap),
            partial_prob=float(sim.partial_prob),
            timeframe=str(timeframe),
            **kwargs,
        )

    def submit(
        self,
        intents: Sequence[OrderIntent],
        *,
        now: datetime,
        bars: IntradayBars,
    ) -> Tuple[Fill, ...]:
        """Registra intenciones nuevas y devuelve los fills de las barras
        que todavía no se habían procesado.

        Una intención ya vista no se duplica. ``now`` se valida (zona) y
        no se usa como reloj: el precio sale de las barras.
        """
        ensure_utc(now, "now")
        if not isinstance(bars, IntradayBars):
            raise TypeError("bars debe ser IntradayBars")
        snap = self._snap()
        try:
            fills = self._submit_inner(intents, bars)
        except Exception:
            self._restore(snap)
            raise
        self._last_snap = snap
        return tuple(fills)

    def flatten(
        self,
        positions: Mapping[str, float],
        *,
        now: datetime,
        bars: IntradayBars,
    ) -> Tuple[Fill, ...]:
        """Cierra con market. El fill queda para la barra siguiente.

        ``signal_ts`` es el inicio de la última barra de ``bars``: con
        ``latency_bars`` 0 la orden está lista en la barra que viene.
        """
        ensure_utc(now, "now")
        if bars.close.empty:
            raise ValueError("flatten sin barras")
        bar_ts = pd.Timestamp(bars.close.index[-1]).to_pydatetime()
        intents = []
        seq = 0
        for symbol, qty in sorted(positions.items()):
            amount = float(qty)
            if abs(amount) <= _EPS:
                continue
            side = "sell" if amount > 0 else "buy"
            intents.append(order_intent(
                "flatten",
                symbol,
                side,
                abs(amount),
                bar_ts,
                order_type="market",
                reason="flatten",
                seq=seq,
            ))
            seq += 1
        return self.submit(intents, now=now, bars=bars)

    def take_spawned(self) -> Tuple[OrderIntent, ...]:
        """Intenciones hijas creadas desde el último ``submit``. Se vacía."""
        out = tuple(self._spawned)
        self._spawned = []
        return out

    def commit_last(self) -> None:
        """El libro aceptó el lote: ya no se puede deshacer."""
        self._last_snap = None

    def rollback_last(self) -> None:
        """Deshace el último ``submit`` o ``flatten`` que sí terminó."""
        if self._last_snap is None:
            return
        self._restore(self._last_snap)
        self._last_snap = None

    def open_orders(self) -> Tuple[OrderIntent, ...]:
        """Órdenes que todavía pueden llenarse. Las canceladas no entran."""
        return tuple(order.intent for order in self._live.values())

    def cancel(self, intent_id: str) -> bool:
        """Deja inactiva una orden viva. False si no está o ya no puede llenarse.

        No borra el id: un ``submit`` posterior con el mismo ``intent_id``
        sigue siendo idempotente y no la revive. Sirve para la pata que no
        se llenó en un par (HFT-09D).
        """
        order = self._live.get(intent_id)
        if order is None:
            return False
        self._retire(order)
        return True

    def _insert(self, order: _Working) -> None:
        order.rank = len(self._orders)
        self._orders[order.intent.intent_id] = order
        if order.remaining > _EPS:
            self._live[order.intent.intent_id] = order

    def _retire(self, order: _Working) -> None:
        order.active = False
        self._live.pop(order.intent.intent_id, None)

    def _revive(self, order: _Working) -> None:
        """Vuelve a ``_live`` en su lugar de ``_orders``, no al final."""
        order.active = True
        intent_id = order.intent.intent_id
        if intent_id in self._live:
            return
        self._live[intent_id] = order
        self._live = dict(sorted(self._live.items(), key=lambda item: item[1].rank))

    def _submit_inner(self, intents: Sequence[OrderIntent], bars: IntradayBars) -> list:
        for intent in intents:
            if not isinstance(intent, OrderIntent):
                raise TypeError(f"se esperaba OrderIntent, llegó {type(intent).__name__}")
            if intent.intent_id in self._orders:
                continue
            self._insert(self._primary(intent))
        fills: list = []
        if bars.close.empty:
            return fills
        for ts in bars.close.index:
            stamp = pd.Timestamp(ts)
            key = stamp.isoformat()
            if key in self._seen:
                continue
            fills.extend(self._on_bar(stamp, bars))
            self._seen.add(key)
            self._added_seen.append(key)
        return fills

    def _primary(self, intent: OrderIntent) -> _Working:
        ready = pd.Timestamp(intent.signal_ts) + self._step * (1 + self.latency_bars)
        return _Working(
            intent=intent,
            remaining=float(intent.qty),
            kind="primary",
            ready=pd.Timestamp(ready),
            stop_price=intent.stop_price,
            limit_price=intent.limit_price,
        )

    def _on_bar(self, ts: pd.Timestamp, bars: IntradayBars) -> list:
        fills: list = []
        for order in list(self._live.values()):
            if order.kind != "primary":
                continue
            fills.extend(self._match(order, ts, bars))
        for fill in list(fills):
            self._arm(fill, ts)
        self._cancel_take_if_both(ts, bars)
        for order in list(self._live.values()):
            if order.kind == "primary":
                continue
            fills.extend(self._match(order, ts, bars))
        return fills

    def _cancel_take_if_both(self, ts: pd.Timestamp, bars: IntradayBars) -> None:
        groups: dict = {}
        for order in self._live.values():
            if not order.group or order.kind not in ("stop", "take"):
                continue
            groups.setdefault(order.group, []).append(order)
        for members in groups.values():
            stop = next((item for item in members if item.kind == "stop"), None)
            take = next((item for item in members if item.kind == "take"), None)
            if stop is None or take is None:
                continue
            if self._would(stop, ts, bars) and self._would(take, ts, bars):
                self._retire(take)
                stop.force_full = True

    def _would(self, order: _Working, ts: pd.Timestamp, bars: IntradayBars) -> bool:
        return self._quote(order, ts, bars) is not None

    def _match(self, order: _Working, ts: pd.Timestamp, bars: IntradayBars) -> list:
        if not order.active or order.remaining <= _EPS:
            return []
        plan = self._quote(order, ts, bars)
        if plan is None:
            return []
        if plan[0] == "rest":
            order.resting = True
            plan = self._quote(order, ts, bars)
            if plan is None or plan[0] == "rest":
                return []
        price, liquidity, capped, volume = plan
        qty = self._quantity(order, capped, volume)
        if qty <= _EPS:
            return []
        if price <= 0 or not math.isfinite(price):
            raise ValueError(f"precio de fill no positivo en {order.intent.intent_id}: {price}")
        order.remaining -= qty
        if order.remaining <= _EPS:
            order.remaining = 0.0
            self._retire(order)
        order.force_full = False
        if order.kind in ("stop", "take"):
            self._shrink_sibling(order, qty)
        fees = self.costs.fees(
            side=order.intent.side,
            qty=qty,
            price=price,
            liquidity=liquidity,
            asset_class=self._asset(order.intent.symbol),
        )
        return [Fill(
            intent_id=order.intent.intent_id,
            ts=ts.to_pydatetime(),
            qty=qty,
            price=price,
            fees=fees,
            liquidity=liquidity,
        )]

    def _quantity(self, order: _Working, capped: bool, volume: float) -> float:
        qty = order.remaining
        if order.force_full:
            return qty
        if capped:
            if volume is None or not math.isfinite(volume) or volume <= 0:
                return 0.0
            qty = min(qty, self.participation_cap * float(volume))
        if qty <= _EPS:
            return 0.0
        if self.partial_prob > 0 and float(self._rng.random()) < self.partial_prob:
            qty = qty * float(self._rng.uniform(0.0, 1.0))
        if qty <= _EPS:
            return 0.0
        if order.remaining - qty <= _EPS:
            return order.remaining
        return qty

    def _quote(
        self,
        order: _Working,
        ts: pd.Timestamp,
        bars: IntradayBars,
    ) -> Optional[tuple]:
        """``(precio, liquidez, topado, volumen)`` o ``('rest',)`` o None.

        No consume el rng ni cambia la orden. ``rest`` significa que un
        marketable_limit no alcanza el precio con slippage y pasa a pasivo.
        """
        if not order.active or order.remaining <= _EPS or ts < order.ready:
            return None
        bar = self._ohlcv(bars, order.intent.symbol, ts)
        if bar is None:
            return None
        open_, high, low, volume = bar
        side = order.intent.side
        style = self._style(order)
        if style == "market":
            slip = self._slip(order, ts, bars, open_)
            price = open_ + slip if side == "buy" else open_ - slip
            return (price, "taker", False, volume)
        if style == "marketable":
            slip = self._slip(order, ts, bars, open_)
            limit = float(order.limit_price)
            if side == "buy":
                taker = open_ + slip
                if limit >= taker - 1e-9:
                    return (min(limit, taker), "taker", False, volume)
            else:
                taker = open_ - slip
                if limit <= taker + 1e-9:
                    return (max(limit, taker), "taker", False, volume)
            return ("rest",)
        if style == "passive":
            limit = float(order.limit_price)
            if not self._crossed(side, limit, low, high):
                return None
            return (limit, "maker", True, volume)
        if style == "stop":
            stop = float(order.stop_price)
            if side == "buy":
                if high < stop - 1e-9:
                    return None
                raw = max(stop, open_)
                price = raw + self._slip(order, ts, bars, raw)
            else:
                if low > stop + 1e-9:
                    return None
                raw = min(stop, open_)
                price = raw - self._slip(order, ts, bars, raw)
            return (price, "taker", False, volume)
        return None

    def _style(self, order: _Working) -> str:
        if order.kind == "stop":
            return "stop"
        if order.kind == "take":
            return "passive"
        if order.resting:
            return "passive"
        if order.intent.order_type == "market":
            return "market"
        if order.intent.order_type == "marketable_limit":
            return "marketable"
        if order.intent.order_type == "limit":
            return "passive"
        return "stop"

    def _crossed(self, side: str, limit: float, low: float, high: float) -> bool:
        tick = float(self.costs.tick)
        if self.fill_mode == "touch":
            if side == "buy":
                return low <= limit + 1e-9
            return high + 1e-9 >= limit
        if side == "buy":
            return low <= limit - tick + 1e-9
        return high + 1e-9 >= limit + tick

    def _slip(self, order: _Working, ts: pd.Timestamp, bars: IntradayBars, price: float) -> float:
        half = self._half(order.intent.symbol, ts, bars, price)
        impact = self.costs.impact_per_share(
            price,
            order.intent.qty,
            self._sigma_of(order.intent.symbol),
            self._adv_of(order.intent.symbol),
        )
        return half + impact

    def _half(self, symbol: str, ts: pd.Timestamp, bars: IntradayBars, price: float) -> float:
        quote = self.quote_fn(symbol, ts) if self.quote_fn is not None else None
        roll = None
        corwin = None
        if quote is None and self.costs.use_spread_estimators:
            roll, corwin = self._estimators(symbol, ts, bars)
        return self.costs.half_spread(price, quote=quote, roll=roll, corwin=corwin)

    def _estimators(
        self,
        symbol: str,
        ts: pd.Timestamp,
        bars: IntradayBars,
    ) -> Tuple[Optional[float], Optional[float]]:
        key = (symbol, ts.isoformat())
        cached = self._spread_cache.get(key)
        if cached is not None:
            return cached
        close = bars.close.loc[:ts, [symbol]]
        high = bars.high.loc[:ts, [symbol]]
        low = bars.low.loc[:ts, [symbol]]
        roll = roll_spread(close, n=self.roll_window)
        corwin = corwin_schultz_spread(high, low)
        found = (_finite_or_none(roll.iloc[-1, 0]), _finite_or_none(corwin.iloc[-1, 0]))
        self._spread_cache[key] = found
        self._added_spreads.append(key)
        return found

    def _arm(self, fill: Fill, ts: pd.Timestamp) -> None:
        parent = self._orders.get(fill.intent_id)
        if parent is None or parent.kind != "primary":
            return
        bracket = parent.intent.bracket
        if not isinstance(bracket, Bracket):
            return
        if bracket.stop is not None:
            self._add_child(parent, "stop", float(bracket.stop), fill.qty, ts)
        if bracket.take is not None:
            self._add_child(parent, "take", float(bracket.take), fill.qty, ts)

    def _add_child(
        self,
        parent: _Working,
        kind: str,
        price: float,
        qty: float,
        ts: pd.Timestamp,
    ) -> None:
        child_id = f"{parent.intent.intent_id}:{kind}"
        side = "sell" if parent.intent.side == "buy" else "buy"
        existing = self._orders.get(child_id)
        if existing is not None:
            # Un hijo inactivo ya no cubre nada: se llenó o lo cancelaron (su
            # hermano cerró la posición, perdió contra el stop en la barra, o
            # un cancel). Lo que le quedaba no se suma: revive con el fill
            # nuevo. La intención dice lo llenado más lo que queda.
            filled = existing.intent.qty - existing.remaining
            if not existing.active:
                existing.remaining = 0.0
            existing.remaining += qty
            existing.intent = replace(existing.intent, qty=filled + existing.remaining)
            self._revive(existing)
            self._spawned.append(existing.intent)
            return
        if kind == "stop":
            intent = order_intent(
                parent.intent.strategy_id,
                parent.intent.symbol,
                side,
                qty,
                ts.to_pydatetime(),
                order_type="stop",
                stop_price=price,
                tif=parent.intent.tif,
                reason=f"stop de {parent.intent.intent_id}",
                intent_id=child_id,
            )
            stop_price, limit_price = price, None
        else:
            intent = order_intent(
                parent.intent.strategy_id,
                parent.intent.symbol,
                side,
                qty,
                ts.to_pydatetime(),
                order_type="limit",
                limit_price=price,
                tif=parent.intent.tif,
                reason=f"take de {parent.intent.intent_id}",
                intent_id=child_id,
            )
            stop_price, limit_price = None, price
        self._insert(_Working(
            intent=intent,
            remaining=qty,
            kind=kind,
            ready=pd.Timestamp(ts),
            group=parent.intent.intent_id,
            stop_price=stop_price,
            limit_price=limit_price,
        ))
        self._spawned.append(intent)

    def _shrink_sibling(self, order: _Working, qty: float) -> None:
        """OCO también en parciales: lo que llenó un hijo ya no lo cubre su
        hermano, que baja lo mismo. Su intención también (lo que llenó no
        cambia). Si no le queda nada se retira tal cual: ``_add_child``
        descarta ese resto si el padre lo revive."""
        if not order.group:
            return
        other = "take" if order.kind == "stop" else "stop"
        # Un hermano que ya no está vivo no cambia.
        sibling = self._live.get(f"{order.group}:{other}")
        if sibling is None:
            return
        if sibling.remaining - qty <= _EPS:
            self._retire(sibling)
            return
        sibling.remaining -= qty
        sibling.intent = replace(sibling.intent, qty=sibling.intent.qty - qty)

    def _ohlcv(
        self,
        bars: IntradayBars,
        symbol: str,
        ts: pd.Timestamp,
    ) -> Optional[tuple]:
        if symbol not in bars.close.columns:
            return None
        open_ = bars.open.at[ts, symbol]
        high = bars.high.at[ts, symbol]
        low = bars.low.at[ts, symbol]
        volume = bars.volume.at[ts, symbol]
        if not _ok(open_) or not _ok(high) or not _ok(low):
            return None
        vol = float(volume) if _ok(volume) else float("nan")
        return float(open_), float(high), float(low), vol

    def _adv_of(self, symbol: str) -> float:
        if symbol not in self._adv:
            raise ValueError(f"sin ADV causal para {symbol}")
        return self._adv[symbol]

    def _sigma_of(self, symbol: str) -> float:
        if symbol not in self._sigma:
            raise ValueError(f"sin σ diaria causal para {symbol}")
        return self._sigma[symbol]

    def _asset(self, symbol: str) -> str:
        if isinstance(self.asset_class, str):
            return self.asset_class
        return self.asset_class.get(symbol, "us_equity")

    def _snap(self) -> dict:
        """Lo que un envío puede cambiar, para deshacerlo.

        El backtest envía en cada barra: copiar o recorrer todas las órdenes,
        barras y spreads de la corrida haría cuadrático un año. Una orden
        inactiva no se vuelve a tocar, salvo el hijo de un bracket cuyo padre
        sigue vivo (un parcial del padre lo reactiva). Se copian las vivas y
        esos hijos; de barras y spreads se anota lo que el envío agrega.
        """
        touched = {}
        for intent_id, order in self._live.items():
            touched[intent_id] = copy.deepcopy(order)
            if order.kind == "primary":
                for kind in ("stop", "take"):
                    child = self._orders.get(f"{intent_id}:{kind}")
                    if child is not None and not child.active:
                        touched[child.intent.intent_id] = copy.deepcopy(child)
        self._added_seen = []
        self._added_spreads = []
        return {
            "count": len(self._orders),
            "touched": touched,
            "seen": self._added_seen,
            "spreads": self._added_spreads,
            "spawned": copy.deepcopy(self._spawned),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def _restore(self, snap: dict) -> None:
        # Las órdenes nuevas se agregaron al final: se sacan por el final y
        # las copias vuelven a su lugar sin mover el resto.
        while len(self._orders) > snap["count"]:
            self._orders.popitem()
        touched = snap["touched"]
        self._orders.update(touched)
        # Fuera de las copias nada estaba vivo ni pudo revivir.
        self._live = {
            intent_id: order
            for intent_id, order in sorted(touched.items(), key=lambda item: item[1].rank)
            if order.active and order.remaining > _EPS
        }
        for key in snap["seen"]:
            self._seen.discard(key)
        for key in snap["spreads"]:
            self._spread_cache.pop(key, None)
        self._spawned = snap["spawned"]
        self._rng.bit_generator.state = snap["rng"]


def _num_map(values: Optional[Mapping[str, float]], name: str) -> dict:
    if values is None:
        return {}
    out = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{key}] debe ser numérico")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name}[{key}] debe ser finito y >= 0")
        out[str(key).strip().upper()] = number
    return out


def _ok(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number > 0
