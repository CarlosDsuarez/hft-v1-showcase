"""Tamaño de la orden intradía (HFT-08).

Tres reglas, todas con el capital de riesgo de D-04::

    equity_riesgo = min(equity de la cuenta, risk_capital_usd)

- Riesgo fijo: ``qty = risk_frac · equity_riesgo / |entrada − stop|``.
- Vol targeting: ``nocional = σ_obj · equity_riesgo / σ_activo``.
  Las dos σ van en la misma unidad. Aquí no se anualiza.
- Kelly fraccional: ``f = c · μ / σ²`` con ``c <= 0.25``. μ y σ tienen
  que llegar ya estimados fuera de muestra: este módulo no los calcula
  y rechaza la llamada si el llamador no lo declara.

El redondeo a acciones enteras aplica cuando hay bracket o cuando la
orden abre o amplía un corto. Si no, la cantidad se trunca a la grilla
de Alpaca (``canonical_qty``, la misma regla que el POST). Por debajo
del nocional mínimo el resultado es 0: no hay orden. ``IntradaySizer``
es la puerta que el motor llama (``size``). Si una entrada cae a 0 por
los topes (el apalancamiento que ya gastó el lote, o el tope por
símbolo), la omite y el motor la registra como rechazo
``sizing_sin_cupo``. Si cae a 0 sin topes, lanza: la estrategia tiene
que usar la función pura y no emitir la intención. Una salida no gasta
cupo ni se recorta.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Optional, Sequence

from src.hft.config import SizingConfig

# Mismo techo que sizing.kelly_c en la config. No se recorta: se rechaza.
_MAX_KELLY_C = 0.25
from src.hft.types import OrderIntent, canonical_qty

_EPS = 1e-9

logger = logging.getLogger("hft.sizing")


def equity_at_risk(account_equity: float, risk_capital_usd: float) -> float:
    """Capital sobre el que se dimensiona. Nunca el saldo de paper si es mayor."""
    equity = _positive("equity", account_equity)
    capital = _positive("risk_capital_usd", risk_capital_usd)
    return min(equity, capital)


def fixed_risk_qty(
    equity_riesgo: float,
    risk_frac: float,
    entry: float,
    stop: float,
) -> float:
    """Acciones para arriesgar ``risk_frac`` del capital si el stop se cumple."""
    capital = _positive("equity_riesgo", equity_riesgo)
    frac = _positive("risk_frac", risk_frac)
    if frac > 1:
        raise ValueError("risk_frac debe ser <= 1")
    entry_px = _positive("entry", entry)
    stop_px = _positive("stop", stop)
    distance = abs(entry_px - stop_px)
    if distance <= _EPS:
        raise ValueError("entrada y stop no pueden coincidir")
    return frac * capital / distance


def vol_target_notional(
    equity_riesgo: float,
    sigma_target: float,
    sigma_asset: float,
) -> float:
    """Nocional que deja la posición en ``sigma_target`` si el activo tiene ``sigma_asset``."""
    capital = _positive("equity_riesgo", equity_riesgo)
    target = _positive("sigma_target", sigma_target)
    asset = _positive("sigma_asset", sigma_asset)
    return target * capital / asset


def kelly_fraction(
    mu: float,
    sigma: float,
    c: float,
    *,
    out_of_sample: bool,
) -> float:
    """``c · μ / σ²``. ``c`` por encima de 0.25 es un error, no se recorta.

    ``out_of_sample`` tiene que ser True. μ y σ estimados en la misma
    muestra con la que se opera no entran: sería buscar el tamaño
    mirando el resultado.
    """
    if out_of_sample is not True:
        raise ValueError("Kelly solo con mu y sigma estimados fuera de muestra")
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise TypeError("c debe ser numérico")
    if not math.isfinite(float(c)) or float(c) < 0 or float(c) > _MAX_KELLY_C:
        raise ValueError(f"c debe estar en [0, {_MAX_KELLY_C}]")
    drift = _finite("mu", mu)
    scale = _positive("sigma", sigma)
    return float(c) * drift / (scale ** 2)


def kelly_notional(
    equity_riesgo: float,
    mu: float,
    sigma: float,
    c: float,
    *,
    out_of_sample: bool,
) -> float:
    """Nocional de Kelly. Si μ es negativo, 0: no se opera en contra del signo."""
    capital = _positive("equity_riesgo", equity_riesgo)
    fraction = kelly_fraction(mu, sigma, c, out_of_sample=out_of_sample)
    if fraction <= 0:
        return 0.0
    return fraction * capital


def apply_sizing_caps(
    qty: float,
    price: float,
    *,
    equity_riesgo: float,
    min_notional: float,
    max_leverage: float,
    whole_shares: bool,
    max_notional: Optional[float] = None,
) -> float:
    """Recorta por apalancamiento, redondea si toca y devuelve 0 bajo el mínimo.

    ``max_notional`` es un techo extra (lo que queda del apalancamiento
    después de las órdenes ya aceptadas en el mismo lote).
    """
    shares = _non_negative("qty", qty)
    px = _positive("price", price)
    capital = _non_negative("equity_riesgo", equity_riesgo)
    floor = _non_negative("min_notional", min_notional)
    leverage = _positive("max_leverage", max_leverage)
    cap = leverage * capital
    if max_notional is not None:
        cap = min(cap, _non_negative("max_notional", max_notional))
    if shares * px > cap:
        shares = cap / px if px > 0 else 0.0
    if whole_shares:
        shares = float(math.floor(shares + 1e-9))
    else:
        shares = canonical_qty(shares)
    if shares * px + _EPS < floor:
        return 0.0
    return shares


def size_fixed_risk(
    config: SizingConfig,
    account_equity: float,
    entry: float,
    stop: float,
    *,
    whole_shares: bool = False,
    max_notional: Optional[float] = None,
) -> float:
    """Cantidad final de riesgo fijo, ya con topes. 0 si no llega al mínimo."""
    capital = equity_at_risk(account_equity, config.risk_capital_usd)
    raw = fixed_risk_qty(capital, config.risk_frac, entry, stop)
    return apply_sizing_caps(
        raw,
        entry,
        equity_riesgo=capital,
        min_notional=config.min_notional,
        max_leverage=config.max_leverage,
        whole_shares=whole_shares,
        max_notional=max_notional,
    )


def size_vol_target(
    config: SizingConfig,
    account_equity: float,
    price: float,
    sigma_asset: float,
    *,
    whole_shares: bool = False,
    max_notional: Optional[float] = None,
) -> float:
    """Cantidad final de vol targeting. ``price`` pasa el nocional a acciones."""
    capital = equity_at_risk(account_equity, config.risk_capital_usd)
    notional = vol_target_notional(capital, config.sigma_target, sigma_asset)
    px = _positive("price", price)
    return apply_sizing_caps(
        notional / px,
        px,
        equity_riesgo=capital,
        min_notional=config.min_notional,
        max_leverage=config.max_leverage,
        whole_shares=whole_shares,
        max_notional=max_notional,
    )


def size_kelly(
    config: SizingConfig,
    account_equity: float,
    price: float,
    mu: float,
    sigma: float,
    *,
    out_of_sample: bool,
    whole_shares: bool = False,
    max_notional: Optional[float] = None,
) -> float:
    """Cantidad final de Kelly. 0 si la fracción no es positiva o no llega al mínimo."""
    capital = equity_at_risk(account_equity, config.risk_capital_usd)
    notional = kelly_notional(
        capital, mu, sigma, config.kelly_c, out_of_sample=out_of_sample,
    )
    px = _positive("price", price)
    return apply_sizing_caps(
        notional / px,
        px,
        equity_riesgo=capital,
        min_notional=config.min_notional,
        max_leverage=config.max_leverage,
        whole_shares=whole_shares,
        max_notional=max_notional,
    )


class IntradaySizer:
    """Puerta ``size`` del motor.

    Si la intención trae ``bracket.stop`` y un precio de entrada
    (límite, o la marca del estado), reescribe la cantidad con riesgo
    fijo. Si no, conserva la cantidad de la estrategia y solo aplica
    apalancamiento, nocional mínimo y el redondeo de bracket o corto.

    ``max_position_notional`` es ``risk.max_position_notional``. Con él,
    la posición proyectada por símbolo no pasa de ese nocional, valorada
    al mayor entre la entrada y la marca (el riesgo valora con la marca).
    Sin él, una orden que el apalancamiento deja pasar la rechaza el
    riesgo. Si la posición ya está llena no se recorta: el riesgo la
    rechaza con su motivo.

    ``room`` es el apalancamiento del lote: las entradas lo gastan en
    orden. La que ya no cabe se omite (el motor la registra como
    rechazo). Una salida que solo reduce la posición pasa con su
    cantidad: ni gasta cupo ni la recortan los topes.
    """

    def __init__(
        self,
        config: SizingConfig,
        max_position_notional: Optional[float] = None,
    ) -> None:
        if not isinstance(config, SizingConfig):
            raise TypeError("config debe ser SizingConfig")
        self.config = config
        self.max_position_notional = (
            None if max_position_notional is None
            else _positive("max_position_notional", max_position_notional)
        )

    def size(self, intents: Sequence[OrderIntent], state: object, now: object) -> tuple:
        del now  # el tamaño no depende del reloj
        if not intents:
            return ()
        account = _account_equity(state)
        capital = equity_at_risk(account, self.config.risk_capital_usd)
        room = self.config.max_leverage * capital
        projected = {}
        sized = []
        for intent in intents:
            if not isinstance(intent, OrderIntent):
                raise TypeError(f"se esperaba OrderIntent, llegó {type(intent).__name__}")
            price = _entry_price(intent, state)
            if intent.symbol not in projected:
                projected[intent.symbol] = _position_of(state, intent.symbol)
            position = projected[intent.symbol]
            stop = intent.bracket.stop if intent.bracket is not None else None
            if stop is not None:
                raw = fixed_risk_qty(capital, self.config.risk_frac, price, stop)
            else:
                raw = intent.qty
            if stop is None and _only_reduces(intent, position, raw):
                projected[intent.symbol] = position + (raw if intent.side == "buy" else -raw)
                sized.append(_with_qty(intent, canonical_qty(raw)))
                continue
            whole = _needs_whole(intent, position, raw)
            cap = room
            symbol_cap = self._symbol_cap(intent, position, price, state)
            if symbol_cap is not None:
                cap = min(cap, symbol_cap)
            qty = apply_sizing_caps(
                raw,
                price,
                equity_riesgo=capital,
                min_notional=self.config.min_notional,
                max_leverage=self.config.max_leverage,
                whole_shares=whole,
                max_notional=cap,
            )
            if qty <= 0:
                uncapped = apply_sizing_caps(
                    raw,
                    price,
                    equity_riesgo=capital,
                    min_notional=self.config.min_notional,
                    max_leverage=self.config.max_leverage,
                    whole_shares=whole,
                )
                if uncapped <= 0:
                    raise ValueError(
                        f"{intent.intent_id}: el sizing quedó en 0 "
                        "(nocional mínimo o tope de apalancamiento)"
                    )
                logger.warning(
                    "%s: sin cupo, se omite (quedan %.2f USD de apalancamiento; "
                    "tope por símbolo %s)",
                    intent.intent_id,
                    max(room, 0.0),
                    "n/a" if symbol_cap is None else f"{symbol_cap:.2f} USD",
                )
                continue
            room -= qty * price
            projected[intent.symbol] = position + (qty if intent.side == "buy" else -qty)
            sized.append(_with_qty(intent, qty))
        return tuple(sized)

    def _symbol_cap(
        self, intent: OrderIntent, position: float, price: float, state: object,
    ) -> Optional[float]:
        """Nocional (al precio de entrada) que deja la posición en el tope, o None."""
        if self.max_position_notional is None:
            return None
        mark = _mark_of(state, intent.symbol)
        valued = max(price, mark) if mark is not None else price
        limit = self.max_position_notional / valued
        allowed = limit - position if intent.side == "buy" else limit + position
        if allowed <= _EPS:
            return None
        return allowed * price


def _only_reduces(intent: OrderIntent, position: float, qty: float) -> bool:
    """La orden va contra la posición y no la da vuelta."""
    if intent.side == "sell":
        return position > _EPS and qty <= position + _EPS
    return position < -_EPS and qty <= -position + _EPS


def _needs_whole(intent: OrderIntent, position: float, qty: float) -> bool:
    if intent.bracket is not None:
        return True
    if intent.side == "sell" and position - qty < -_EPS:
        return True
    return False


def _with_qty(intent: OrderIntent, qty: float) -> OrderIntent:
    # Igualdad exacta: la cantidad canónica difiere de la cruda en menos
    # de un cuanto y es la que tiene que llegar al broker.
    if qty == intent.qty:
        return intent
    return replace(intent, qty=qty)


def _entry_price(intent: OrderIntent, state: object) -> float:
    if intent.limit_price is not None:
        return float(intent.limit_price)
    mark = _mark_of(state, intent.symbol)
    if mark is None:
        raise ValueError(f"sin precio para dimensionar {intent.symbol}")
    return mark


def _account_equity(state: object) -> float:
    equity = getattr(state, "equity", None)
    if equity is None:
        equity = getattr(state, "nav", None)
    if equity is None:
        raise ValueError("el estado no trae equity ni nav")
    return float(equity)


def _position_of(state: object, symbol: str) -> float:
    method = getattr(state, "position", None)
    if callable(method):
        return float(method(symbol))
    positions = getattr(state, "positions", ())
    wanted = symbol.strip().upper()
    if isinstance(positions, dict):
        return float(positions.get(wanted, 0.0))
    for name, qty in positions:
        if str(name).strip().upper() == wanted:
            return float(qty)
    return 0.0


def _mark_of(state: object, symbol: str) -> Optional[float]:
    method = getattr(state, "mark", None)
    if callable(method):
        raw = method(symbol)
        return None if raw is None else float(raw)
    marks = getattr(state, "marks", ())
    wanted = symbol.strip().upper()
    if isinstance(marks, dict):
        raw = marks.get(wanted)
        return None if raw is None else float(raw)
    for name, price in marks:
        if str(name).strip().upper() == wanted:
            return float(price)
    return None


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} debe ser finito, llegó {value!r}")
    return number


def _positive(name: str, value: object) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} debe ser > 0, llegó {value!r}")
    return number


def _non_negative(name: str, value: object) -> float:
    number = _finite(name, value)
    if number < 0:
        raise ValueError(f"{name} debe ser >= 0, llegó {value!r}")
    return number
