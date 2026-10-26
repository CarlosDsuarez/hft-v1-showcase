"""Contrato de estrategia (HFT-06).

``on_bar`` es pura: no hace I/O, no lee el reloj y no usa azar salvo el
``rng`` que llega en el contexto. El motor le pasa la ventana de barras
ya finales y el instante de decisión (cierre de la barra más la gracia),
nunca la hora de llegada del dato.

``STRATEGIES`` es el registro del proceso. ``register`` rechaza una
estrategia cuya TheoryCard esté incompleta o cuyos parámetros no
coincidan con el dataclass frozen.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.hft.data.bars import IntradayBars
from src.hft.types import OrderIntent, TheoryCard, theory_errors


@dataclass(frozen=True)
class StrategyContext:
    """Todo lo que ``on_bar`` puede mirar. Nada de esto lee el sistema.

    ``now`` es el instante en que la barra de ``bar_ts`` quedó final
    (cierre + gracia), igual en backtest y en vivo. ``features`` es la
    salida de la misma función en los dos modos. ``positions`` son las
    posiciones **propias**, las que llenaron las órdenes de esta
    estrategia (HFT-PAPER-08). Tienen signo: positivo largo, negativo corto.

    ``pending`` es lo que falta llenar de las órdenes **propias** que
    siguen vivas (enviadas o esperando su barra de ejecución), con el
    mismo signo. Trae una fila por símbolo y lado: lo que falta comprar en
    positivo y lo que falta vender en negativo. ``positions`` solo trae lo
    lleno: una entrada que no repite mira las dos cosas (HFT-PAPER-15).
    Una salida se mide con ``exit_qty`` (HFT-PAPER-16).

    ``book_positions`` es el libro entero, de todas las estrategias. Sirve
    solo para el equity (``book_items``): las salidas y los flatten miran
    lo propio. ``cash`` y ``realized_pnl`` son los del libro.
    """

    now: datetime
    bar_ts: datetime
    bars: IntradayBars
    features: object
    positions: Tuple[Tuple[str, float], ...]
    cash: float
    realized_pnl: float
    rng: Optional[np.random.Generator] = None
    pending: Tuple[Tuple[str, float], ...] = ()
    book_positions: Optional[Tuple[Tuple[str, float], ...]] = None

    def book_items(self) -> Tuple[Tuple[str, float], ...]:
        """El libro entero para valorar el equity. Sin libro, ``positions``.

        Con una sola estrategia lo propio y el libro coinciden.
        """
        if self.book_positions is None:
            return self.positions
        return self.book_positions

    def position(self, symbol: str) -> float:
        """Cantidad propia con signo, o 0 si esta estrategia no tiene el símbolo."""
        wanted = str(symbol).strip().upper()
        for name, qty in self.positions:
            if name == wanted:
                return float(qty)
        return 0.0

    def pending_qty(self, symbol: str) -> float:
        """Cantidad con signo que falta llenar en órdenes vivas, o 0."""
        wanted = str(symbol).strip().upper()
        return float(sum(qty for name, qty in self.pending if name == wanted))

    def exit_qty(self, symbol: str) -> float:
        """Lo que una salida todavía tiene que cerrar, con el signo de la posición.

        Es lo lleno menos lo que ya cubren las órdenes vivas del lado
        contrario. Sin esto una venta que tarda en llenarse se repite cada
        barra y el libro termina corto (HFT-PAPER-16). Lo vivo del mismo
        lado, el resto de una entrada, no suma: si no llega a llenarse,
        una salida que lo incluya también da vuelta el libro.
        """
        wanted = str(symbol).strip().upper()
        held = self.position(wanted)
        rows = [float(qty) for name, qty in self.pending if name == wanted]
        if held > 0:
            return max(0.0, held + sum(qty for qty in rows if qty < 0))
        if held < 0:
            return min(0.0, held + sum(qty for qty in rows if qty > 0))
        return 0.0

    def position_map(self) -> Mapping[str, float]:
        return dict(self.positions)


class Strategy(Protocol):
    """Estrategia intradía. ``params`` es un dataclass frozen."""

    id: str
    theory: TheoryCard
    params: object

    def required_history(self) -> Tuple[int, int]:
        """``(barras, sesiones)`` con índice ``<= t``, como las features.

        La ventana trae lo que piden las estrategias del motor y la sesión
        en curso, nada más (``Engine.history_sessions``). Leer más atrás de
        lo declarado da otro resultado según con quién corra.
        """

    def on_bar(self, ctx: StrategyContext) -> Sequence[OrderIntent]:
        """Intenciones decididas con la barra de ``ctx``. Puede devolver []."""

    # Opcional: ``take_cancels()`` devuelve los símbolos cuyas órdenes vivas
    # de esta estrategia hay que cancelar, y se vacía. ``on_bar`` no ve la
    # ejecución; el motor cancela por ella justo después de esa barra
    # (pata colgada de pairs_ou, HFT-PAPER-16).
    #
    # Opcional: ``atomic_entries = True``. Si el motor rechaza una entrada
    # porque el símbolo es de otra estrategia (``simbolo_de_otra_estrategia``),
    # rechaza todas las entradas de esta estrategia en esa barra (pairs_ou:
    # no abre media pareja).


# Arranca vacío. Cada estrategia de HFT-09 se registra al importar su
# módulo (``orb`` y las que sigan).
STRATEGIES: dict = {}


def validate_strategy(strategy: object) -> Tuple[str, ...]:
    """Errores de contrato. Una tupla vacía significa que puede registrarse."""
    errors = []
    strategy_id = getattr(strategy, "id", None)
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        errors.append("id vacío")
    theory = getattr(strategy, "theory", None)
    errors.extend(theory_errors(theory))
    params = getattr(strategy, "params", None)
    if not dataclasses.is_dataclass(params) or isinstance(params, type):
        errors.append("params debe ser una instancia de dataclass")
    elif not params.__dataclass_params__.frozen:
        errors.append("params debe ser frozen")
    else:
        fields = {
            field.name: getattr(params, field.name)
            for field in dataclasses.fields(params)
        }
        documented = theory.param_map() if isinstance(theory, TheoryCard) else {}
        missing = sorted(set(fields) - set(documented))
        extra = sorted(set(documented) - set(fields))
        if missing or extra:
            errors.append(
                f"params no coinciden con la TheoryCard: faltan {missing}, sobran {extra}"
            )
        else:
            for name, value in fields.items():
                if documented[name] != value:
                    errors.append(
                        f"{name}: el dataclass tiene {value!r} y la TheoryCard "
                        f"{documented[name]!r}"
                    )
    history = getattr(strategy, "required_history", None)
    if not callable(history):
        errors.append("required_history no es callable")
    else:
        try:
            declared = history()
        except Exception as exc:
            errors.append(f"required_history falló: {type(exc).__name__}: {exc}")
        else:
            ok = (
                isinstance(declared, tuple)
                and len(declared) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                        for item in declared)
            )
            if not ok:
                errors.append("required_history debe devolver (barras, sesiones) >= 0")
    on_bar = getattr(strategy, "on_bar", None)
    if not callable(on_bar):
        errors.append("on_bar no es callable")
    return tuple(errors)


def register(strategy: Strategy) -> Strategy:
    """Valida y guarda la estrategia. Un id repetido es error."""
    errors = validate_strategy(strategy)
    if errors:
        raise ValueError("estrategia inválida: " + "; ".join(errors))
    if strategy.id in STRATEGIES:
        raise ValueError(f"estrategia ya registrada: {strategy.id}")
    STRATEGIES[strategy.id] = strategy
    return strategy
