"""Dueño de cada posición del motor intradía (HFT-PAPER-08).

``HftBook`` es neto por símbolo y no sabe quién abrió qué. Aquí cada
símbolo guarda la cantidad con signo de cada dueño. El dueño de un fill
es el ``strategy_id`` de su intención; los hijos de un bracket llevan el
del padre.

- Si el dueño es una estrategia del motor, el fill suma a su fila.
- Si no lo es (``flatten``, ``reconcile`` o una estrategia que no corre
  en esta sesión), es de la casa. Primero reduce hacia cero a las
  estrategias con signo contrario, en orden de id y sin dar vuelta a
  ninguna. Después a los otros dueños de la casa con signo contrario.
  Lo que sobra queda a su nombre. Así el flatten de sesión deja a todas
  en 0.

Libro y atribución reciben los mismos fills: para todo símbolo, la suma
de sus dueños es ``book.position(símbolo)``.

Solo stdlib: es parte del núcleo ligero.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

from src.hft.types import Fill, OrderIntent

# El mismo umbral que ``HftBook``: por debajo, la fila se considera cerrada.
_QTY_EPS = 1e-9


class Attribution:
    """Cantidad con signo por ``símbolo → {dueño: qty}``.

    Args:
        strategies: ids de las estrategias del motor. Cualquier otro dueño
            es de la casa.
    """

    def __init__(self, strategies: Iterable[str] = ()) -> None:
        self.strategies = frozenset(str(name) for name in strategies)
        self._qty: Dict[str, Dict[str, float]] = {}

    def apply(self, fills: Sequence[Fill], intents: Mapping[str, OrderIntent]) -> None:
        """Aplica el lote entero o no aplica nada, en el orden recibido.

        Raises:
            ValueError: un fill no tiene su intención. La atribución queda
                como estaba.
        """
        staged = {symbol: dict(rows) for symbol, rows in self._qty.items()}
        for fill in fills:
            intent = intents.get(fill.intent_id)
            if intent is None:
                raise ValueError(f"fill sin intención: {fill.intent_id}")
            delta = float(fill.qty) if intent.side == "buy" else -float(fill.qty)
            _move(staged, self.strategies, intent.strategy_id, intent.symbol, delta)
        self._qty = {symbol: rows for symbol, rows in staged.items() if rows}

    def move(self, owner: str, symbol: str, delta: float) -> None:
        """Un movimiento con signo de ``owner`` en ``symbol``, con la misma regla."""
        _move(self._qty, self.strategies, str(owner), _symbol(symbol), float(delta))
        self._qty = {name: rows for name, rows in self._qty.items() if rows}

    def owned(self, owner: str) -> Tuple[Tuple[str, float], ...]:
        """``((símbolo, qty), …)`` de ``owner``, por símbolo."""
        return tuple(
            (symbol, rows[owner])
            for symbol, rows in sorted(self._qty.items())
            if owner in rows
        )

    def owners(self, symbol: str) -> Dict[str, float]:
        """``{dueño: qty}`` de ``symbol``. Vacío si nadie lo tiene."""
        return dict(self._qty.get(_symbol(symbol), {}))

    def position(self, owner: str, symbol: str) -> float:
        """Cantidad con signo de ``owner`` en ``symbol``, o 0."""
        return float(self._qty.get(_symbol(symbol), {}).get(owner, 0.0))

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {symbol: dict(rows) for symbol, rows in sorted(self._qty.items())}


def _move(
    table: Dict[str, Dict[str, float]],
    strategies: frozenset,
    owner: str,
    symbol: str,
    delta: float,
) -> None:
    rows = table.setdefault(symbol, {})
    if owner not in strategies:
        others = sorted(name for name in rows if name in strategies)
        house = sorted(name for name in rows if name not in strategies)
        for name in others + house:
            if abs(delta) <= _QTY_EPS:
                break
            held = rows[name]
            if held * delta >= 0:
                continue
            take = min(abs(held), abs(delta))
            step = take if delta > 0 else -take
            _set(rows, name, held + step)
            delta -= step
    if abs(delta) > _QTY_EPS:
        _set(rows, owner, rows.get(owner, 0.0) + delta)


def _set(rows: Dict[str, float], name: str, qty: float) -> None:
    if abs(qty) <= _QTY_EPS:
        rows.pop(name, None)
    else:
        rows[name] = qty


def _symbol(symbol: str) -> str:
    return str(symbol).strip().upper()
