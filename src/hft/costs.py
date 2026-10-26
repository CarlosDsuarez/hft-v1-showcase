"""Modelo único de costos intradía (HFT-07).

Lo usan el simulador y la valoración de un fill de paper. Paper no
cobra estas tarifas regulatorias: ``paper_valuation_fees`` es el
importe que hay que restar al P&L para que el backtest y el paper
queden en la misma escala. No hay otro sitio que defina el costo.

Medio spread
    Con quote: ``(ask − bid) / 2``. El quote manda, aunque sea más
    estrecho que el piso.
    Sin quote: ``max(1 tick, piso en bps, Roll/2, Corwin–Schultz/2)``.
    Roll (``features.roll_spread``) está en unidades de precio y es el
    spread completo. Corwin–Schultz es una fracción del precio. Un
    estimador NaN o no positivo no entra: no se rellena con 0 para
    luego tratarlo como spread.

Impacto de raíz cuadrada
    ``Y · σ_diaria · sqrt(q / ADV)`` como fracción del precio.
    ``q`` y ``ADV`` van en dólares. ``ADV`` es la media de
    precio × volumen en los días estrictamente anteriores a ``t``,
    la misma cuenta que ``order_execution.compute_adv`` (ese módulo
    no se importa: el núcleo intradía no arrastra el stack diario).
    Tóth et al. (2011); el marco de impacto temporal es Almgren y
    Chriss (2000). ``Y`` por defecto es 1.

Fees de acciones
    Comisión 0 por acción: Alpaca documenta el trading de acciones
    como commission-free. En ventas, SEC Sección 31 sobre el nocional
    y FINRA TAF por acción con tope. Las tasas viven en
    ``config/hft.toml`` con ``verified_on``.

    Sección 31: 20.60 USD por millón de ventas cubiertas desde el
    2026-04-04 (SEC Release 34-104909). Tasa = 0.0000206.

    TAF 2026 del schedule: 0.000195 USD por acción, tope 9.79 USD por
    trade. Si el precio de ejecución es menor que la tasa por acción,
    FINRA no cobra el TAF. Entre el 2026-10-01 y el 2026-12-31 FINRA
    pausó el cobro (SR-FINRA-2026-021). El modelo conserva la tasa del
    schedule, no el cero de esa pausa: un backtest no debe heredar una
    exención de tres meses. En esa ventana el paper cobra menos y el
    backtest queda más pesimista, que es la regla.

    No se redondea al centavo. Alpaca redondea hacia arriba el total
    diario de fees; la diferencia cabe en un centavo por día.

    Alpaca también lista una tarifa CAT en compras y ventas, pero la
    página de regulatory fees no publica el importe. No se inventa.

Cripto
    Tier 1 de Alpaca: 15 bps maker y 25 bps taker sobre el nocional
    (docs de crypto fees). Alpaca lo carga sobre el activo acreditado;
    aquí se aproxima por ``qty * price``. No se suman SEC ni TAF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Tolerancia para un spread estimador "no visto".
_EPS = 1e-15


@dataclass(frozen=True)
class Quote:
    """Mejor bid y ask. ``ask < bid`` no es un quote utilizable."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        for name in ("bid", "ask"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} debe ser finito y > 0, llegó {value!r}")
            object.__setattr__(self, name, number)
        if self.ask < self.bid:
            raise ValueError(f"ask {self.ask} por debajo del bid {self.bid}")


@dataclass(frozen=True)
class CostModel:
    """Parámetros del único modelo. Los números salen de ``HftConfig.costs``."""

    spread_floor_bps: float
    tick: float
    impact_y: float
    adv_window: int
    commission_per_share: float
    sec_section31_rate: float
    sec_section31_verified_on: str
    finra_taf_per_share: float
    finra_taf_cap: float
    finra_taf_verified_on: str
    crypto_maker_bps: float
    crypto_taker_bps: float
    crypto_fees_verified_on: str
    # Sensibilidad: apagar los estimadores deja solo tick y piso.
    use_spread_estimators: bool = True

    def __post_init__(self) -> None:
        if self.tick <= 0:
            raise ValueError("tick debe ser > 0")
        if self.spread_floor_bps < 0 or self.impact_y < 0:
            raise ValueError("spread_floor_bps e impact_y no pueden ser negativos")
        if self.adv_window < 1:
            raise ValueError("adv_window debe ser >= 1")
        if self.commission_per_share < 0 or self.sec_section31_rate < 0:
            raise ValueError("comisión y Sección 31 no pueden ser negativas")
        if self.finra_taf_per_share < 0 or self.finra_taf_cap < 0:
            raise ValueError("TAF no puede ser negativo")
        if self.crypto_maker_bps < 0 or self.crypto_taker_bps < 0:
            raise ValueError("fees de cripto no pueden ser negativas")
        if not isinstance(self.use_spread_estimators, bool):
            raise TypeError("use_spread_estimators debe ser bool")

    @classmethod
    def from_config(cls, config: object) -> "CostModel":
        """Arma el modelo con las tasas de ``config.costs``."""
        costs = getattr(config, "costs", None)
        if costs is None:
            raise TypeError("la config no trae costs")
        return cls(
            spread_floor_bps=float(costs.spread_floor_bps),
            tick=float(costs.tick),
            impact_y=float(costs.impact_y),
            adv_window=int(costs.adv_window),
            commission_per_share=float(costs.commission_per_share),
            sec_section31_rate=float(costs.sec_section31_rate),
            sec_section31_verified_on=str(costs.sec_section31_verified_on),
            finra_taf_per_share=float(costs.finra_taf_per_share),
            finra_taf_cap=float(costs.finra_taf_cap),
            finra_taf_verified_on=str(costs.finra_taf_verified_on),
            crypto_maker_bps=float(costs.crypto_maker_bps),
            crypto_taker_bps=float(costs.crypto_taker_bps),
            crypto_fees_verified_on=str(costs.crypto_fees_verified_on),
        )

    def half_spread(
        self,
        price: float,
        *,
        quote: Optional[Quote] = None,
        roll: Optional[float] = None,
        corwin: Optional[float] = None,
    ) -> float:
        """Medio spread en unidades de precio.

        ``roll`` es el spread completo de Roll. ``corwin`` es la fracción
        de Corwin–Schultz. Sin quote se toma el máximo de los pisos y de
        los estimadores disponibles: el spread más ancho es el pesimista.
        """
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise TypeError(f"price debe ser numérico, llegó {price!r}")
        level = float(price)
        if not math.isfinite(level) or level <= 0:
            raise ValueError(f"price debe ser finito y > 0, llegó {price!r}")
        if quote is not None:
            if not isinstance(quote, Quote):
                raise TypeError("quote debe ser Quote")
            return (quote.ask - quote.bid) / 2.0
        parts = [float(self.tick), level * float(self.spread_floor_bps) / 10_000.0]
        if self.use_spread_estimators:
            if _positive_finite(roll):
                parts.append(float(roll) / 2.0)
            if _positive_finite(corwin):
                parts.append(float(corwin) * level / 2.0)
        return max(parts)

    def impact_per_share(
        self,
        price: float,
        qty: float,
        sigma_daily: float,
        adv: float,
    ) -> float:
        """Impacto en unidades de precio. ``adv`` es el ADV en dólares."""
        fraction = impact_return(sigma_daily, qty, price, adv, self.impact_y)
        return float(price) * fraction

    def fees(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        liquidity: str,
        asset_class: str = "us_equity",
    ) -> float:
        """Comisión más tarifas. En acciones la comisión por defecto es 0
        y las regulatorias caen solo en la venta."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side {side!r} no es buy ni sell")
        if liquidity not in ("taker", "maker"):
            raise ValueError(f"liquidity {liquidity!r} no es taker ni maker")
        amount = _positive("qty", qty)
        level = _positive("price", price)
        if asset_class == "crypto":
            bps = self.crypto_maker_bps if liquidity == "maker" else self.crypto_taker_bps
            return amount * level * float(bps) / 10_000.0
        if asset_class != "us_equity":
            raise ValueError(f"asset_class {asset_class!r} no está soportado")
        commission = amount * float(self.commission_per_share)
        if side != "sell":
            return commission
        sec = amount * level * float(self.sec_section31_rate)
        if level < float(self.finra_taf_per_share):
            taf = 0.0
        else:
            taf = min(amount * float(self.finra_taf_per_share), float(self.finra_taf_cap))
        return commission + sec + taf

    def paper_fees(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        liquidity: str,
        asset_class: str = "us_equity",
    ) -> float:
        """Lo que hay que restar al P&L de un fill de paper.

        Es ``fees``: paper no debita estas tarifas, el modelo sí.
        """
        return self.fees(
            side=side,
            qty=qty,
            price=price,
            liquidity=liquidity,
            asset_class=asset_class,
        )


def paper_valuation_fees(
    model: CostModel,
    *,
    side: str,
    qty: float,
    price: float,
    liquidity: str,
    asset_class: str = "us_equity",
) -> float:
    """Mismas fees que el simulador, para valorar un fill real de paper."""
    if not isinstance(model, CostModel):
        raise TypeError("model debe ser CostModel")
    return model.fees(
        side=side,
        qty=qty,
        price=price,
        liquidity=liquidity,
        asset_class=asset_class,
    )


def impact_return(
    sigma_daily: float,
    qty: float,
    price: float,
    adv: float,
    y: float = 1.0,
) -> float:
    """``Y · σ_diaria · sqrt(q / ADV)``. ``q = qty * price`` y ``ADV`` en dólares.

    ``σ_diaria`` es un desvío de retornos (no un porcentaje). ``adv``
    tiene que ser el de días anteriores: esta función no mira el calendario.
    """
    sigma = _non_negative("sigma_daily", sigma_daily)
    amount = _positive("qty", qty)
    level = _positive("price", price)
    coef = _non_negative("y", y)
    volume = _positive("adv", adv)
    return coef * sigma * math.sqrt((amount * level) / volume)


def implementation_shortfall(
    *,
    side: str,
    qty: float,
    benchmark: float,
    fill_price: float,
    fees: float,
) -> float:
    """Coste en moneda frente al precio de referencia. Positivo es peor.

    Compra: ``(fill − benchmark) · qty + fees``.
    Venta: ``(benchmark − fill) · qty + fees``.
    """
    amount = _positive("qty", qty)
    mid = _positive("benchmark", benchmark)
    level = _positive("fill_price", fill_price)
    cost = _non_negative("fees", fees)
    if side == "buy":
        return (level - mid) * amount + cost
    if side == "sell":
        return (mid - level) * amount + cost
    raise ValueError(f"side {side!r} no es buy ni sell")


def average_daily_value(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: object,
    window: int = 20,
) -> pd.Series:
    """ADV en dólares, solo con filas estrictamente anteriores a ``t``.

    ``ADV = mean(precio × volumen)`` en las últimas ``window`` filas
    anteriores. Es la definición de ``order_execution.compute_adv``
    (allí el orden de los frames es volúmenes, precios).
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(volumes, pd.DataFrame):
        raise TypeError("prices y volumes deben ser DataFrame")
    look = _window(window)
    stamp = pd.Timestamp(t)
    history_v = volumes.loc[volumes.index < stamp].iloc[-look:]
    history_p = prices.loc[prices.index < stamp].iloc[-look:]
    common = history_v.index.intersection(history_p.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    return (history_v.loc[common] * history_p.loc[common]).mean()


def daily_volatility(prices: pd.Series, t: object, window: int = 20) -> float:
    """Desvío (``ddof=1``) de retornos log diarios, solo con días ``< t``.

    Hacen falta al menos dos retornos. Si no hay historia, el resultado
    es NaN: el impacto no se inventa un cero.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices debe ser Series")
    look = _window(window)
    stamp = pd.Timestamp(t)
    history = prices.loc[prices.index < stamp].iloc[-(look + 1):]
    if len(history) < 3:
        return float("nan")
    returns = _log_returns(history).iloc[-look:]
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1))


def _log_returns(prices: pd.Series) -> pd.Series:
    shifted = prices.shift(1)
    ratio = (prices / shifted).where((prices > 0) & (shifted > 0))
    return pd.Series(np.log(ratio.to_numpy(dtype=float)), index=prices.index).dropna()


def _window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"window debe ser int, llegó {window!r}")
    if window < 1:
        raise ValueError("window debe ser >= 1")
    return window


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


def _positive_finite(value: Optional[float]) -> bool:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number > _EPS


def _finite_or_none(value: object) -> Optional[float]:
    """Último estimador utilizable, o None si es NaN o no positivo."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= _EPS:
        return None
    return number
