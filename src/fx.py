"""
fx.py — Conversion de precios en moneda local a la moneda base.

Convencion de la serie FX: ``fx[ccy]`` = unidades de moneda base por 1
unidad de ``ccy`` (base USD: ``fx["COP"]`` ≈ 1/4000). Yahoo cotiza el par
``{base}{ccy}=X`` (``USDCOP=X`` = COP por USD); el adaptador de datos
(`src.market_data.InstrumentProvider.get_fx`) lo invierte.

`to_base` convierte ANTES de estrategia y ledger: ambos trabajan solo en
moneda base. Solo infraestructura: la politica de cobertura FX es de la
estrategia y no se decide aqui.

Puro: sin red ni estado.
"""

from __future__ import annotations

import re
from typing import List, Mapping

import numpy as np
import pandas as pd

from src.settings import BASE_CURRENCY

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def is_currency_code(code: object) -> bool:
    """True si `code` es un codigo ISO 4217 en mayusculas (USD, COP)."""
    return isinstance(code, str) and bool(_CURRENCY_RE.match(code))


class MissingFxError(ValueError):
    """Moneda local sin ninguna tasa FX valida para convertir a base.

    Attributes:
        currencies: Monedas sin serie, ordenadas.
    """

    def __init__(self, currencies, detail: str = "") -> None:
        self.currencies: List[str] = sorted(str(c) for c in currencies)
        msg = f"No FX rate to base currency for {self.currencies}"
        super().__init__(f"{msg}. {detail}" if detail else msg)


def fx_pair_symbol(currency: str, base: str = BASE_CURRENCY) -> str:
    """Simbolo Yahoo del par: ``{base}{currency}=X`` (unidades de `currency` por base)."""
    return f"{base}{currency}=X"


def to_base(
    prices: pd.DataFrame,
    fx: pd.DataFrame,
    currencies: Mapping[str, str],
    base: str = BASE_CURRENCY,
) -> pd.DataFrame:
    """Precios locales (fecha × id) → moneda base.

    Cada fecha usa la ultima tasa valida en o antes de ella (sin look-ahead);
    fechas anteriores a la primera tasa quedan NaN. Tasas <= 0 o no finitas
    se ignoran. Columnas ya en `base` no necesitan FX y no cambian.

    Args:
        prices: Precios en moneda local por id.
        fx: Unidades de `base` por 1 unidad de cada moneda (columnas ISO).
        currencies: id → moneda local (ver `InstrumentMaster.currencies`).
        base: Moneda base.

    Raises:
        ValueError: columna de `prices` sin moneda en `currencies`.
        MissingFxError: moneda local sin ninguna tasa valida en `fx`.
    """
    no_ccy = [c for c in prices.columns if c not in currencies]
    if no_ccy:
        raise ValueError(f"No currency for {no_ccy}: cannot convert to {base}.")
    out = prices.astype(float).copy()
    foreign = sorted({currencies[c] for c in prices.columns} - {base})
    if not foreign:
        return out
    rates = fx.reindex(columns=foreign).apply(pd.to_numeric, errors="coerce")
    rates = rates.where(np.isfinite(rates) & (rates > 0))
    absent = [c for c in foreign if rates[c].isna().all()]
    if absent:
        raise MissingFxError(absent)
    rates = rates[~rates.index.duplicated(keep="last")].sort_index()
    asof = rates.reindex(rates.index.union(prices.index)).ffill().reindex(prices.index)
    for col in prices.columns:
        if currencies[col] != base:
            out[col] = out[col] * asof[currencies[col]]
    return out
