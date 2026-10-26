"""
demo_data.py — Panel sintetico para los demos `python -m src.<modulo>`.

Los demos corren sin red ni archivos en data/ (ver AGENTS.md): generan un
panel reproducible de un factor (SPY = mercado) con np.random.default_rng,
sin tocar el RNG global. Solo para demos; los tests usan sus propios fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import SEED, TRADING_DAYS

MARKET = "SPY"


def synthetic_prices(
    n_days: int = 2 * TRADING_DAYS,
    n_tickers: int = 19,
    seed: int = SEED,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Cierres diarios de un modelo de un factor: `n_tickers` acciones + MARKET.

    r_i = beta_i * r_m + e_i con beta_i ~ U(0.6, 1.4); columnas T00.. y SPY.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    r_m = rng.normal(0.0004, 0.010, n_days)
    betas = rng.uniform(0.6, 1.4, n_tickers)
    idio = rng.normal(0.0001, 0.012, (n_days, n_tickers))
    rets = np.column_stack([r_m[:, None] * betas + idio, r_m])
    cols = [f"T{i:02d}" for i in range(n_tickers)] + [MARKET]
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)),
                        index=dates, columns=cols)


def synthetic_volumes(prices: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Volumen diario lognormal alrededor de un nivel propio por ticker (1e4-1e7)."""
    rng = np.random.default_rng(seed)
    base = 10.0 ** rng.uniform(4, 7, prices.shape[1])
    noise = rng.lognormal(0.0, 0.3, prices.shape)
    return pd.DataFrame(base * noise, index=prices.index, columns=prices.columns)
