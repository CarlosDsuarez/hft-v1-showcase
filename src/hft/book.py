"""Libro intradía con signo (HFT-06).

El ``PositionLedger`` legado solo admite largos y no se toca. Aquí una
posición negativa es un corto. El costeo es FIFO: cubrir un corto (o
vender un largo) realiza P&L de precio y, en el mismo momento, las fees
de apertura de esos lotes más las fees de cierre de este fill.

Las fees de un lote que sigue abierto viven en el lote (``fees_open``),
no en ``realized_pnl``. El cash ya las descontó. Con el libro plano y
sin fees abiertas:

    cash = initial_cash + realized_pnl

Con posición abierta, marcando a ``marks``:

    nav = initial_cash + realized_pnl + unrealized_pnl - fees_open

``apply_fills`` valida el lote entero y lo aplica todo o nada, en el
orden recibido (un corto del mismo lote puede financiar una compra
posterior). No reordena ventas y compras: en intradía el orden es el
de la ejecución.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Mapping, Sequence, Tuple

from src.hft.types import Fill, OrderIntent, ensure_utc

BOOK_SCHEMA_VERSION = 1
# Holgura de caja. Solo absorbe ruido de coma flotante.
CASH_TOLERANCE = 1e-6
# Por debajo de esto un lote se considera cerrado.
_QTY_EPS = 1e-9

_TOP_KEYS = (
    "schema_version",
    "initial_cash",
    "cash",
    "realized_pnl",
    "fees",
    "lots",
    "fills",
)
_LOT_KEYS = ("symbol", "qty", "entry_price", "entry_ts", "fees_open")
_FILL_KEYS = (
    "intent_id",
    "ts",
    "symbol",
    "side",
    "qty",
    "price",
    "fees",
    "liquidity",
    "realized_pnl",
)


@dataclass
class Lot:
    """Lote FIFO. ``qty`` > 0 es largo; ``qty`` < 0 es corto."""

    symbol: str
    qty: float
    entry_price: float
    entry_ts: str
    fees_open: float


class HftBook:
    """Posiciones con signo, cash y P&L. Sin conexión a un broker.

    Args:
        initial_cash: Caja de partida, en moneda base. Tiene que ser > 0.
    """

    def __init__(self, initial_cash: float) -> None:
        cash = _finite("initial_cash", initial_cash)
        if cash <= 0:
            raise ValueError("initial_cash debe ser > 0")
        self.initial_cash: float = cash
        self.cash: float = cash
        self.realized_pnl: float = 0.0
        self.fees: float = 0.0
        self._lots: Dict[str, List[Lot]] = {}
        self._fills: List[dict] = []

    def position(self, symbol: str) -> float:
        """Cantidad neta con signo. 0 si no hay lotes."""
        qty = sum(lot.qty for lot in self._lots.get(_symbol(symbol), ()))
        if abs(qty) <= _QTY_EPS:
            return 0.0
        return float(qty)

    def positions(self) -> Dict[str, float]:
        """Símbolo → cantidad con signo. Sin los que están planos."""
        out = {}
        for symbol in sorted(self._lots):
            qty = self.position(symbol)
            if qty != 0.0:
                out[symbol] = qty
        return out

    def fees_open(self) -> float:
        """Fees de apertura todavía sentadas en lotes vivos."""
        return float(sum(lot.fees_open for lots in self._lots.values() for lot in lots))

    def unrealized_pnl(self, marks: Mapping[str, float]) -> float:
        """P&L de precio no realizado. No incluye las fees abiertas.

        Un corto gana cuando la marca baja: ``(mark - entry) * qty`` con
        ``qty`` negativa.

        Raises:
            ValueError: una posición abierta no tiene marca finita > 0.
        """
        total = 0.0
        for symbol, lots in self._lots.items():
            if not lots:
                continue
            mark = _mark(symbol, marks)
            for lot in lots:
                total += (mark - lot.entry_price) * lot.qty
        return float(total)

    def nav(self, marks: Mapping[str, float]) -> float:
        """Cash más valor de mercado (el corto resta)."""
        value = self.cash
        for symbol, qty in self.positions().items():
            value += qty * _mark(symbol, marks)
        return float(value)

    def apply_fills(
        self,
        fills: Sequence[Fill],
        intents: Mapping[str, OrderIntent],
    ) -> float:
        """Aplica el lote entero o no aplica nada.

        Cada fill trae cantidad positiva. El lado y el símbolo salen de
        la intención con ese ``intent_id``. Un fill puede cerrar la
        posición y abrir la contraria con el residuo.

        Returns:
            P&L realizado de este lote (ya neto de las fees que cierran).

        Raises:
            ValueError: intención ausente, lado inválido, o el lote
                dejaría el cash por debajo de ``-CASH_TOLERANCE``.
                El libro queda como estaba.
        """
        plan = _plan(fills, intents)
        lots = {
            symbol: [replace(lot) for lot in book]
            for symbol, book in self._lots.items()
        }
        cash = self.cash
        realized = self.realized_pnl
        fees = self.fees
        recorded: List[dict] = []
        batch = 0.0
        for fill, intent in plan:
            before = realized
            cash, realized, fees = _apply_one(lots, cash, realized, fees, fill, intent)
            _check_cash(cash, fill.intent_id)
            delta = realized - before
            batch += delta
            recorded.append(_fill_row(fill, intent, delta))
        self._lots = {symbol: book for symbol, book in lots.items() if book}
        self.cash = cash
        self.realized_pnl = realized
        self.fees = fees
        self._fills.extend(recorded)
        return float(batch)

    def to_dict(self) -> dict:
        """Estado público. ``schema_version`` es ``BOOK_SCHEMA_VERSION``."""
        lots = []
        for symbol in sorted(self._lots):
            for lot in self._lots[symbol]:
                lots.append({
                    "symbol": lot.symbol,
                    "qty": float(lot.qty),
                    "entry_price": float(lot.entry_price),
                    "entry_ts": lot.entry_ts,
                    "fees_open": float(lot.fees_open),
                })
        return {
            "schema_version": BOOK_SCHEMA_VERSION,
            "initial_cash": float(self.initial_cash),
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "fees": float(self.fees),
            "lots": lots,
            "fills": [dict(row) for row in self._fills],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "HftBook":
        """Reconstruye un libro. No reejecuta los fills: restaura el estado.

        Raises:
            ValueError: esquema desconocido, clave ajena, o un lote inválido.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("el payload del libro debe ser un mapping")
        unknown = sorted(set(payload) - set(_TOP_KEYS))
        missing = sorted(set(_TOP_KEYS) - set(payload))
        if unknown or missing:
            raise ValueError(f"libro inválido: faltan {missing}, desconocidas {unknown}")
        version = payload["schema_version"]
        if version != BOOK_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={version!r} no soportado "
                f"(se espera {BOOK_SCHEMA_VERSION})"
            )
        book = cls(initial_cash=_finite("initial_cash", payload["initial_cash"]))
        book.cash = _finite("cash", payload["cash"])
        book.realized_pnl = _finite("realized_pnl", payload["realized_pnl"])
        book.fees = _finite("fees", payload["fees"])
        if book.fees < 0:
            raise ValueError("fees no puede ser negativo")
        _check_cash(book.cash, "from_dict")
        for row in payload["lots"]:
            lot = _lot_from_row(row)
            book._lots.setdefault(lot.symbol, []).append(lot)
        _check_signs(book._lots)
        for row in payload["fills"]:
            book._fills.append(_fill_from_row(row))
        return book


def _plan(
    fills: Sequence[Fill],
    intents: Mapping[str, OrderIntent],
) -> List[Tuple[Fill, OrderIntent]]:
    plan = []
    for fill in fills:
        if not isinstance(fill, Fill):
            raise TypeError(f"se esperaba Fill, llegó {type(fill).__name__}")
        intent = intents.get(fill.intent_id)
        if intent is None:
            raise ValueError(f"fill sin intención: {fill.intent_id}")
        if not isinstance(intent, OrderIntent):
            raise TypeError(f"intención inválida para {fill.intent_id}")
        if intent.side not in ("buy", "sell"):
            raise ValueError(f"lado inválido en {fill.intent_id}")
        plan.append((fill, intent))
    return plan


def _apply_one(
    lots: Dict[str, List[Lot]],
    cash: float,
    realized: float,
    fees: float,
    fill: Fill,
    intent: OrderIntent,
) -> Tuple[float, float, float]:
    symbol = intent.symbol
    qty = fill.qty
    price = fill.price
    side = intent.side
    book = lots.get(symbol, [])
    pos = sum(lot.qty for lot in book)
    closing = 0.0
    if side == "buy" and pos < 0:
        closing = min(qty, -pos)
    elif side == "sell" and pos > 0:
        closing = min(qty, pos)
    opening = qty - closing
    if closing:
        close_fee = fill.fees * (closing / qty)
    else:
        close_fee = 0.0
    open_fee = fill.fees - close_fee
    if side == "buy":
        cash -= price * qty + fill.fees
    else:
        cash += price * qty - fill.fees
    fees += fill.fees
    if closing:
        realized = _close(lots, symbol, closing, price, close_fee, realized)
    if opening > _QTY_EPS:
        signed = opening if side == "buy" else -opening
        lots.setdefault(symbol, []).append(Lot(
            symbol=symbol,
            qty=signed,
            entry_price=price,
            entry_ts=_iso(fill.ts),
            fees_open=open_fee,
        ))
    _check_signs(lots)
    return cash, realized, fees


def _close(
    lots: Dict[str, List[Lot]],
    symbol: str,
    qty: float,
    price: float,
    close_fee: float,
    realized: float,
) -> float:
    """Consume lotes FIFO. ``qty`` es la cantidad positiva que se cierra."""
    book = lots.get(symbol, [])
    remaining = qty
    fee_left = close_fee
    while remaining > _QTY_EPS and book:
        lot = book[0]
        avail = abs(lot.qty)
        take = min(avail, remaining)
        if lot.qty > 0:
            gross = (price - lot.entry_price) * take
            lot.qty -= take
        else:
            gross = (lot.entry_price - price) * take
            lot.qty += take
        last = remaining - take <= _QTY_EPS
        if abs(lot.qty) <= _QTY_EPS:
            open_fee = lot.fees_open
            book.pop(0)
        else:
            open_fee = lot.fees_open * (take / avail)
            lot.fees_open -= open_fee
        if last:
            fee_take = fee_left
        else:
            fee_take = close_fee * (take / qty)
            fee_left -= fee_take
        realized += gross - open_fee - fee_take
        remaining -= take
    if remaining > 1e-8:
        raise ValueError(f"cierre FIFO incompleto en {symbol}: faltan {remaining}")
    if not book:
        lots.pop(symbol, None)
    return realized


def _check_cash(cash: float, after: str) -> None:
    if cash < -CASH_TOLERANCE:
        raise ValueError(
            f"el lote dejaría cash={cash:.6f} < -{CASH_TOLERANCE} tras {after}"
        )


def _check_signs(lots: Mapping[str, Sequence[Lot]]) -> None:
    for symbol, book in lots.items():
        signs = {1 if lot.qty > 0 else -1 for lot in book if abs(lot.qty) > _QTY_EPS}
        if len(signs) > 1:
            raise ValueError(f"{symbol} tiene lotes largos y cortos a la vez")


def _mark(symbol: str, marks: Mapping[str, float]) -> float:
    if symbol not in marks:
        raise ValueError(f"sin marca para la posición abierta {symbol}")
    mark = _finite(f"marca {symbol}", marks[symbol])
    if mark <= 0:
        raise ValueError(f"marca inválida para {symbol}: {mark!r}")
    return mark


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: se esperaba un número, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} no es finito: {value!r}")
    return number


def _symbol(symbol: str) -> str:
    name = str(symbol).strip().upper()
    if not name:
        raise ValueError("symbol vacío")
    return name


def _iso(ts: datetime) -> str:
    return ensure_utc(ts, "ts").isoformat()


def _fill_row(fill: Fill, intent: OrderIntent, realized: float) -> dict:
    return {
        "intent_id": fill.intent_id,
        "ts": _iso(fill.ts),
        "symbol": intent.symbol,
        "side": intent.side,
        "qty": float(fill.qty),
        "price": float(fill.price),
        "fees": float(fill.fees),
        "liquidity": fill.liquidity,
        "realized_pnl": float(realized),
    }


def _require_keys(row: Mapping, keys: Sequence[str], what: str) -> None:
    if not isinstance(row, Mapping):
        raise ValueError(f"{what} debe ser un mapping")
    unknown = sorted(set(row) - set(keys))
    missing = sorted(set(keys) - set(row))
    if unknown or missing:
        raise ValueError(f"{what}: faltan {missing}, desconocidas {unknown}")


def _lot_from_row(row: Mapping) -> Lot:
    _require_keys(row, _LOT_KEYS, "lote")
    symbol = _symbol(str(row["symbol"]))
    qty = _finite("qty", row["qty"])
    if abs(qty) <= _QTY_EPS:
        raise ValueError(f"lote de {symbol} con qty nula")
    price = _finite("entry_price", row["entry_price"])
    if price <= 0:
        raise ValueError(f"entry_price inválido en {symbol}")
    fees_open = _finite("fees_open", row["fees_open"])
    if fees_open < -_QTY_EPS:
        raise ValueError(f"fees_open negativo en {symbol}")
    entry_ts = row["entry_ts"]
    if not isinstance(entry_ts, str) or not entry_ts:
        raise ValueError(f"entry_ts inválido en {symbol}")
    return Lot(
        symbol=symbol,
        qty=qty,
        entry_price=price,
        entry_ts=entry_ts,
        fees_open=fees_open,
    )


def _fill_from_row(row: Mapping) -> dict:
    _require_keys(row, _FILL_KEYS, "fill")
    side = row["side"]
    if side not in ("buy", "sell"):
        raise ValueError(f"lado inválido en el historial: {side!r}")
    liquidity = row["liquidity"]
    if liquidity not in ("taker", "maker"):
        raise ValueError(f"liquidity inválida en el historial: {liquidity!r}")
    return {
        "intent_id": str(row["intent_id"]),
        "ts": str(row["ts"]),
        "symbol": _symbol(str(row["symbol"])),
        "side": side,
        "qty": _finite("qty", row["qty"]),
        "price": _finite("price", row["price"]),
        "fees": _finite("fees", row["fees"]),
        "liquidity": liquidity,
        "realized_pnl": _finite("realized_pnl", row["realized_pnl"]),
    }
