"""
benchmarks.py — Paso 4: referencias USA + COL para evaluar el sleeve mixto.

Registry:
  SPY (USA equity, USD) | EEM (EM, USD) | TLT (bonos USA, USD) |
  ICOLCAP / HCOLSEL (COL equity, COP) | USDCOP=X (FX para pasar USD→COP).

Las metricas relativas (Beta/IR) ya viven en `metrics_calculator`
(`compute_metrics(benchmark_returns=...)`); aqui estan los helpers de
datos: descarga, retornos log, conversion de moneda y tabla comparada.

Convencion de moneda: el NAV normalizado a 1.0 es comparable entre
monedas en el dashboard; para metricas en una sola moneda (ej. todo COP)
usar `to_cop_returns` con la serie USDCOP=X.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.market_data import (
    RESEARCH_POLICY, DataPolicy, MarketDataProvider, build_provider, load_bars,
)
from src.metrics_calculator import PortfolioMetrics, compute_metrics

logger = logging.getLogger("benchmarks")

from src.settings import TRADING_DAYS

# label -> {yahoo, currency, kind}
BENCHMARKS: Dict[str, Dict[str, str]] = {
    "SPY": {"yahoo": "SPY", "currency": "USD", "kind": "equity_us"},
    "EEM": {"yahoo": "EEM", "currency": "USD", "kind": "equity_em"},
    "TLT": {"yahoo": "TLT", "currency": "USD", "kind": "bonds_us"},
    "ICOLCAP": {"yahoo": "ICOLCAP.CL", "currency": "COP", "kind": "equity_co"},
    "HCOLSEL": {"yahoo": "HCOLSEL.CL", "currency": "COP", "kind": "equity_co"},
    "USDCOP": {"yahoo": "USDCOP=X", "currency": "FX", "kind": "fx"},
}

REFERENCE_DEFAULT = "SPY"


def benchmark_labels(kind: Optional[str] = None) -> List[str]:
    """Labels del registry, opcionalmente filtrados por kind."""
    if kind is None:
        return list(BENCHMARKS)
    return [lb for lb, meta in BENCHMARKS.items() if meta["kind"] == kind]


def fetch_benchmark_prices(
    labels: Optional[List[str]] = None,
    start: str = "2019-01-01",
    end: str = "2026-01-01",
    *,
    provider: Optional[MarketDataProvider] = None,
    policy: DataPolicy = RESEARCH_POLICY,
    now=None,
) -> pd.DataFrame:
    """Cierres ajustados de benchmarks via el proveedor unico (src/market_data).

    Args:
        labels: Subset del registry. None → todos salvo USDCOP.
        start: Fecha inicio 'YYYY-MM-DD'.
        end: Fecha fin 'YYYY-MM-DD' (exclusiva).
        provider: Fuente. Default: build_provider() (Yahoo → CSV en data_dir).
        policy: Default RESEARCH_POLICY: benchmark faltante o atrasado solo
            avisa; la barra intradia se descarta.
        now: Reloj para la frescura (default: ahora).

    Returns:
        DataFrame (fecha × label) de precios ajustados.

    Raises:
        ValueError: label desconocido.
        RuntimeError: sin datos (offline o Yahoo caido) — MarketDataError.
    """
    labels = labels or [lb for lb in BENCHMARKS if lb != "USDCOP"]
    unknown = [lb for lb in labels if lb not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Benchmarks desconocidos: {unknown}. Registry: {benchmark_labels()}")
    tickers = [BENCHMARKS[lb]["yahoo"] for lb in labels]

    bars = load_bars(provider or build_provider(), tickers, start, end,
                     policy=policy, now=now)
    yahoo_to_label = {BENCHMARKS[lb]["yahoo"]: lb for lb in labels}
    return bars.close.rename(columns=yahoo_to_label)


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Precios -> retornos log diarios (primera fila fuera)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def to_cop_returns(
    usd_returns: pd.Series,
    fx_prices: pd.Series,
) -> pd.Series:
    """Convierte retornos USD a COP via USDCOP (retornos log aditivos).

    r_COP = r_USD + r_FX, en interseccion de fechas.

    Args:
        usd_returns: Retornos log diarios en USD.
        fx_prices: Precios USDCOP (COP por USD).

    Returns:
        Retornos log diarios en COP, index interseccion.
    """
    fx_rets = np.log(fx_prices / fx_prices.shift(1))
    common = usd_returns.index.intersection(fx_rets.index)
    out = usd_returns.reindex(common).fillna(0.0) + fx_rets.reindex(common).fillna(0.0)
    out.name = (usd_returns.name or "returns")
    return out


def align_to(bench_returns: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Recorta benchmark al rango/fecha de la estrategia (interseccion)."""
    common = target_index.intersection(bench_returns.index)
    return bench_returns.reindex(common).dropna()


def compute_metrics_vs_benchmarks(
    returns: pd.Series,
    nav: pd.Series,
    benchmarks: Dict[str, pd.Series],
    turnover: Optional[pd.Series] = None,
    reference: str = REFERENCE_DEFAULT,
    risk_free_rate: float = 0.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, PortfolioMetrics]:
    """Metricas de la estrategia vs cada benchmark + metricas de benchmarks.

    Args:
        returns: Retornos diarios de la estrategia.
        nav: NAV diario de la estrategia.
        benchmarks: Dict label -> retornos diarios del benchmark.
        turnover: Turnover diario de la estrategia (opcional).
        reference: Benchmark de referencia para Beta/IR de la estrategia
            y de los demas benchmarks. Debe estar en `benchmarks`.
        risk_free_rate: Tasa libre de riesgo anual.
        n_bootstrap: Iteraciones bootstrap (R-06).
        seed: Semilla.

    Returns:
        Dict {label -> PortfolioMetrics} con "strategy" + un key por benchmark.
        Benchmarks reportan Beta/IR vs `reference` (la referencia misma va
        sin benchmark: beta=1.0, IR=0.0).
    """
    if reference not in benchmarks:
        raise ValueError(
            f"reference='{reference}' no esta en benchmarks: {sorted(benchmarks)}."
        )
    ref_rets = benchmarks[reference]
    out: Dict[str, PortfolioMetrics] = {
        "strategy": compute_metrics(
            returns, nav, turnover=turnover, benchmark_returns=ref_rets,
            risk_free_rate=risk_free_rate, n_bootstrap=n_bootstrap, seed=seed,
        )
    }
    for label, b_rets in benchmarks.items():
        b_nav = (1.0 + b_rets.fillna(0.0)).cumprod()
        out[label] = compute_metrics(
            b_rets, b_nav, turnover=None,
            benchmark_returns=None if label == reference else ref_rets,
            risk_free_rate=risk_free_rate, n_bootstrap=n_bootstrap, seed=seed,
        )
    return out


if __name__ == "__main__":
    px = fetch_benchmark_prices(["SPY", "ICOLCAP", "USDCOP"])
    print(px.tail(3).to_string())
    print("\nMonedas:", {lb: BENCHMARKS[lb]["currency"] for lb in px.columns})
