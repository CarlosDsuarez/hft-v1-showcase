"""Tipos inmutables del motor intradía (HFT-06).

``Bar``, ``OrderIntent``, ``Fill``, ``RiskEvent`` y ``TheoryCard`` no
se mutan: un paso del motor reemplaza valores, no los edita. El
``intent_id`` sale de ``make_intent_id`` y depende solo de la estrategia,
el símbolo, la barra de señal, el lado y la secuencia. No entra el reloj
de llegada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from typing import Optional, Tuple

UTC = timezone.utc

ORDER_TYPES = ("market", "marketable_limit", "limit", "stop")
SIDES = ("buy", "sell")
# TIF que Alpaca documenta para acciones. El adaptador (HFT-11) vuelve a
# validar cada combinación contra el tipo de orden.
TIFS = ("day", "gtc", "opg", "cls", "ioc", "fok")
LIQUIDITIES = ("taker", "maker")

# Alpaca acepta hasta 9 decimales en ``qty`` (órdenes fraccionales).
QTY_DECIMALS = 9
QTY_QUANTUM = 10.0 ** -QTY_DECIMALS
# Un fill que se pasa de su intención en medio cuanto o más es un
# exceso real. Por debajo solo queda el error binario de sumar fills.
QTY_TOLERANCE = QTY_QUANTUM / 2
# El ruido binario de una suma de fills vive por debajo de 1e-12.
_QTY_NOISE = Decimal("1e-12")
_QTY_STEP = Decimal(1).scaleb(-QTY_DECIMALS)


def canonical_qty(qty: float) -> float:
    """Cantidad en la grilla de Alpaca: trunca a ``QTY_DECIMALS``.

    El sizer y ``format_qty`` usan esta regla, así la intención, el
    journal y el POST llevan el mismo número. Truncar no pasa un tope;
    antes se quita el ruido binario (``152.06812652099998`` sigue
    siendo ``152.068126521``, no pierde un cuanto).
    """
    value = Decimal(repr(float(qty))).quantize(_QTY_NOISE, rounding=ROUND_HALF_EVEN)
    return float(value.quantize(_QTY_STEP, rounding=ROUND_DOWN))


def ensure_utc(ts: datetime, what: str) -> datetime:
    """Devuelve ``ts`` en UTC. Rechaza un instante sin zona."""
    if not isinstance(ts, datetime):
        raise TypeError(f"{what} debe ser datetime, llegó {type(ts).__name__}")
    if ts.tzinfo is None:
        raise ValueError(f"{what} debe llevar zona horaria")
    return ts.astimezone(UTC)


def make_intent_id(
    strategy_id: str,
    symbol: str,
    signal_ts: datetime,
    side: str,
    seq: int = 0,
) -> str:
    """Id determinista de una intención. El mismo input da el mismo id."""
    stamp = ensure_utc(signal_ts, "signal_ts").strftime("%Y%m%dT%H%M%S")
    name = str(symbol).strip().upper()
    if int(seq) < 0:
        raise ValueError("seq no puede ser negativo")
    return f"{strategy_id}-{stamp}-{name}-{side}-{int(seq)}"


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} debe ser finito y > 0, llegó {value!r}")
    return number


def _non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} debe ser finito y >= 0, llegó {value!r}")
    return number


@dataclass(frozen=True)
class Bar:
    """Una barra de un símbolo. ``ts`` es el inicio del intervalo, en UTC."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[float] = None
    timeframe: str = "1Min"

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol vacío")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "ts", ensure_utc(self.ts, "ts"))
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(self, "volume", _non_negative("volume", self.volume))
        for name in ("vwap", "trades"):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _non_negative(name, value))
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(
                f"OHLC inválido en {symbol} {self.ts.isoformat()}: "
                f"o={self.open} h={self.high} l={self.low} c={self.close}"
            )
        if not str(self.timeframe).strip():
            raise ValueError("timeframe vacío")


@dataclass(frozen=True)
class Bracket:
    """Take y stop opcionales, ambos precios estrictamente positivos."""

    take: Optional[float] = None
    stop: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("take", "stop"):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _positive(name, value))


@dataclass(frozen=True)
class OrderIntent:
    """Intención de orden. La emite la estrategia; el motor sella id y barra.

    ``signal_ts`` es el inicio de la barra en la que se decidió. La
    ejecución no ocurre en esa barra: el motor la suelta como pronto en
    la barra ``t + latency_bars``.
    """

    strategy_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    tif: str
    bracket: Optional[Bracket]
    reason: str
    signal_ts: datetime
    intent_id: str

    def __post_init__(self) -> None:
        if not str(self.strategy_id).strip():
            raise ValueError("strategy_id vacío")
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol vacío")
        object.__setattr__(self, "symbol", symbol)
        if self.side not in SIDES:
            raise ValueError(f"side {self.side!r} no está en {SIDES}")
        object.__setattr__(self, "qty", _positive("qty", self.qty))
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type {self.order_type!r} no está en {ORDER_TYPES}")
        if self.tif not in TIFS:
            raise ValueError(f"tif {self.tif!r} no está en {TIFS}")
        if self.order_type in ("limit", "marketable_limit") and self.limit_price is None:
            raise ValueError(f"{self.order_type} exige limit_price")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop exige stop_price")
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", _positive("limit_price", self.limit_price))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _positive("stop_price", self.stop_price))
        if self.bracket is not None and not isinstance(self.bracket, Bracket):
            raise TypeError("bracket debe ser Bracket o None")
        if not isinstance(self.reason, str):
            raise TypeError("reason debe ser str")
        object.__setattr__(self, "signal_ts", ensure_utc(self.signal_ts, "signal_ts"))
        if not str(self.intent_id).strip():
            raise ValueError("intent_id vacío")


def order_intent(
    strategy_id: str,
    symbol: str,
    side: str,
    qty: float,
    signal_ts: datetime,
    *,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    tif: str = "day",
    bracket: Optional[Bracket] = None,
    reason: str = "",
    seq: int = 0,
    intent_id: Optional[str] = None,
) -> OrderIntent:
    """Atajo para armar una intención con id determinista."""
    if intent_id is None:
        intent_id = make_intent_id(strategy_id, symbol, signal_ts, side, seq)
    return OrderIntent(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        tif=tif,
        bracket=bracket,
        reason=reason,
        signal_ts=signal_ts,
        intent_id=intent_id,
    )


@dataclass(frozen=True)
class Fill:
    """Ejecución de una intención. ``qty`` es siempre positiva; el lado está en la intención."""

    intent_id: str
    ts: datetime
    qty: float
    price: float
    fees: float
    liquidity: str

    def __post_init__(self) -> None:
        if not str(self.intent_id).strip():
            raise ValueError("intent_id vacío")
        object.__setattr__(self, "ts", ensure_utc(self.ts, "ts"))
        object.__setattr__(self, "qty", _positive("qty", self.qty))
        object.__setattr__(self, "price", _positive("price", self.price))
        object.__setattr__(self, "fees", _non_negative("fees", self.fees))
        if self.liquidity not in LIQUIDITIES:
            raise ValueError(f"liquidity {self.liquidity!r} no está en {LIQUIDITIES}")


@dataclass(frozen=True)
class RiskEvent:
    """Rechazo u otro evento de riesgo. ``kind`` lo pone quien lo emite."""

    ts: datetime
    kind: str
    reason: str
    symbol: Optional[str] = None
    intent_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "ts"))
        if not str(self.kind).strip():
            raise ValueError("kind vacío")
        if not str(self.reason).strip():
            raise ValueError("reason vacío")
        if self.symbol is not None:
            object.__setattr__(self, "symbol", str(self.symbol).strip().upper())


@dataclass(frozen=True)
class TheoryParam:
    """Un parámetro ex-ante: valor, por qué ese valor, y de dónde sale."""

    name: str
    value: object
    justification: str
    source: str

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("el parámetro no tiene nombre")
        if not isinstance(self.justification, str) or not isinstance(self.source, str):
            raise TypeError("justificación y fuente deben ser str")


@dataclass(frozen=True)
class TheoryCard:
    """Justificación teórica que viaja con la estrategia.

    ``mechanism`` responde quién paga el edge. ``params`` es la lista
    ``(nombre, valor, justificación, fuente)``. ``research_only`` marca
    lo que no puede ir a paper. ``Engine`` y ``scripts/run_intraday.py``
    lo rechazan en modo paper (HFT-09E).
    """

    hypothesis: str
    mechanism: str
    references: Tuple[str, ...]
    data_requirements: Tuple[str, ...]
    params: Tuple[TheoryParam, ...]
    holding_period: str
    failure_regimes: Tuple[str, ...]
    kill_criteria: Tuple[str, ...]
    research_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "data_requirements", tuple(self.data_requirements))
        object.__setattr__(self, "failure_regimes", tuple(self.failure_regimes))
        object.__setattr__(self, "kill_criteria", tuple(self.kill_criteria))
        object.__setattr__(self, "params", tuple(self.params))
        for param in self.params:
            if not isinstance(param, TheoryParam):
                raise TypeError("params debe ser una secuencia de TheoryParam")
        if not isinstance(self.research_only, bool):
            raise TypeError("research_only debe ser bool")

    def param_map(self) -> dict:
        """Nombre → valor. El último gana si el nombre se repite."""
        return {param.name: param.value for param in self.params}


def theory_errors(card: TheoryCard) -> Tuple[str, ...]:
    """Huecos de una TheoryCard. Vacío significa completa."""
    if not isinstance(card, TheoryCard):
        return ("theory no es TheoryCard",)
    errors = []
    for name in ("hypothesis", "mechanism", "holding_period"):
        text = getattr(card, name)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{name} vacío")
    for name in ("references", "data_requirements", "failure_regimes", "kill_criteria"):
        values = getattr(card, name)
        if not values:
            errors.append(f"{name} vacío")
            continue
        for item in values:
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{name} contiene un texto vacío")
                break
    seen = set()
    for param in card.params:
        if param.name in seen:
            errors.append(f"parámetro duplicado: {param.name}")
        seen.add(param.name)
        if not param.justification.strip():
            errors.append(f"{param.name}: justificación vacía")
        if not param.source.strip():
            errors.append(f"{param.name}: fuente vacía")
    return tuple(errors)
