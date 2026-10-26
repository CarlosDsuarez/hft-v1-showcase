"""
sleeve_attribution.py — Paso 5: atribucion de rendimiento por sleeve.

Descompone el retorno diario del portafolio en contribuciones por sleeve
(por defecto "market": us vs co vs other) a partir de pesos diarios y
precios. Replica la convencion del engine (`backtest_engine.run_backtest`,
ruta clasica):

    contrib_sleeve(t) = sum_{i en sleeve} w_{i,t-1} * r_{i,t}

con los pesos del dia OOS anterior y retornos simples: los rebalanceos se
ejecutan al cierre de t, asi que los pesos registrados en t (post-trade)
rentan desde t+1 y el movimiento t-1 -> t es del libro que se tenia.
Los pesos son fraccion del NAV: la caja (`CASH_COL`) renta 0 y no se
renormaliza. Primera fecha OOS en 0 (sin t-1 en OOS).

La suma de sleeves = retorno BRUTO; `daily_records.daily_return` es neto:
bruto(t) - neto(t) = costo(t) (convencion lineal del engine; `result.returns`
compone exacto via NAV y difiere en el termino cruzado g*c de segundo orden).

Sin look-ahead: solo usa w_{t-1} y r_t (R-05).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from src.backtest_engine import CASH_COL

logger = logging.getLogger("sleeve_attribution")


def _sleeve_of(group: str, level: str) -> str:
    if level == "market":
        return str(group).split("/")[0]
    return str(group)


def _sleeve_map(
    columns: pd.Index,
    groups: Union[pd.Series, Dict],
    level: str,
) -> pd.Series:
    """Ticker -> sleeve ("us"/"co"/"other")."""
    if level not in ("market", "group"):
        raise ValueError(f"level debe ser 'market'|'group': {level}")
    g = pd.Series(groups).reindex(columns)
    return g.apply(
        lambda grp: _sleeve_of(grp, level) if pd.notna(grp) else "other"
    )


def sleeve_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    groups: Union[pd.Series, Dict],
    level: str = "market",
) -> pd.DataFrame:
    """Contribuciones diarias por sleeve (fecha × sleeve).

    Args:
        weights: Pesos diarios como fraccion del NAV (fecha × ticker, la
            caja en CASH_COL), ej. `result.weights`.
        prices: Precios diarios (fecha × ticker) del mismo universo.
        groups: Ticker -> grupo "mercado/sector".
        level: "market" (us/co) o "group" (etiqueta completa).

    Returns:
        DataFrame fecha × sleeve con contribuciones simples diarias.
        Suma por fila = retorno bruto reconstruido del portafolio.
    """
    common_cols = weights.columns.intersection(prices.columns)
    if len(common_cols) == 0:
        raise ValueError("weights y prices no comparten tickers.")
    # Retornos simples sobre el historial completo (r_t necesita p_{t-1});
    # luego se recorta al indice de pesos (fechas OOS).
    rets = prices[common_cols].ffill().pct_change()
    w = weights[common_cols].reindex(rets.index).ffill().fillna(0.0)
    w = w.reindex(weights.index).fillna(0.0)
    # Pesos del dia OOS anterior (el libro que se tenia durante el
    # movimiento t-1 -> t), sin renormalizar: la caja renta 0.
    wf = w.shift(1).fillna(0.0)
    contrib_all = wf * rets.reindex(w.index).fillna(0.0)
    contrib_all.iloc[0] = 0.0  # engine: 0.0 el primer dia OOS
    sleeves = _sleeve_map(common_cols, groups, level)
    out = pd.DataFrame(index=w.index)
    for sleeve in sorted(sleeves.unique()):
        cols = sleeves[sleeves == sleeve].index.tolist()
        out[sleeve] = contrib_all[cols].sum(axis=1)
    return out


def sleeve_nav(contrib: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones -> NAV por sleeve normalizado a 1.0."""
    return (1.0 + contrib.fillna(0.0)).cumprod()


def sleeve_avg_weights(
    weights: pd.DataFrame,
    groups: Union[pd.Series, Dict],
    level: str = "market",
) -> pd.Series:
    """Peso promedio en el tiempo por sleeve; la caja es el sleeve "cash"
    (suma 1 si pesos + caja suman 1)."""
    sleeves = _sleeve_map(weights.columns, groups, level)
    sleeves[sleeves.index == CASH_COL] = "cash"
    avg = weights.mean()
    return avg.groupby(sleeves).sum()


def monthly_summary(contrib: pd.DataFrame) -> pd.DataFrame:
    """Tabla mensual: retorno compuesto por sleeve + total + mejor/peor.

    Returns:
        DataFrame index YYYY-MM con columnas por sleeve, "total",
        "best_sleeve" y "worst_sleeve".
    """
    if contrib.empty:
        return pd.DataFrame()
    monthly = (1.0 + contrib.fillna(0.0)).resample("ME").prod() - 1.0
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly["total"] = (1.0 + contrib.sum(axis=1).fillna(0.0)).resample("ME").prod().values - 1.0
    sleeves = list(contrib.columns)
    monthly["best_sleeve"] = monthly[sleeves].idxmax(axis=1)
    monthly["worst_sleeve"] = monthly[sleeves].idxmin(axis=1)
    return monthly


def attribution_table(
    contrib: pd.DataFrame,
    avg_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Tabla de atribucion del periodo por sleeve.

    Columnas: total_contrib (pp), share_of_return (%), ann_vol_contrib.
    """
    total = contrib.sum()
    port_total = float(total.sum())
    table = pd.DataFrame({
        "total_contrib": total,
        "share_of_return": total / port_total if abs(port_total) > 1e-12 else np.nan,
    })
    if avg_weights is not None:
        table["avg_weight"] = avg_weights.reindex(table.index).fillna(0.0)
    table["ann_vol"] = contrib.std() * np.sqrt(252)
    return table.round(6)
