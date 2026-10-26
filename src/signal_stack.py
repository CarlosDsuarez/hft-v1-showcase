"""
signal_stack.py — T-34: Motor multi-factor de señales (precios/volúmenes).

Combina factores calculados sobre el universo OHLCV de HFT-V1
(precios ajustados + volúmenes vía yfinance). Sin fundamentales ni opciones.

Factores implementados:
  - Momentum 12-1  (reusa compute_momentum_scores de momentum_replacement)
  - Mean-reversion (z-score precio vs MA)
  - Volatilidad    (rank inverso de vol realizada → tilt low-vol)
  - Liquidez       (rank inverso Amihud / ADV → tilt líquido)

Factores de ai-hedge-fund NO replicables por falta de datos:
  - Factores fundamentales (P/E, earnings surprise, book-to-market, …)
  - Superficie de volatilidad implícita / skew de opciones
  - Sentiment / conviction vía LLM
  - Short interest / borrow rates
  - Datos macro de alta frecuencia no presentes en data_client

Output: compute_signal_scores(prices, volumes, t) → Series normalizado.
compute_signal_scores_with_meta(...) → (Series, meta) con los factores usados.

Historia: ``factor_requirements(cfg)`` / ``required_history(cfg)`` declaran
cuántas filas (índice <= t) necesita cada factor. Sin historia suficiente:
modo "strict" → InsufficientHistoryError; "permissive" → el factor se excluye
y los pesos se renormalizan (meta["factors_used"]). Nunca se rellena con 0.

Restricciones cubiertas: R-05 (sin look-ahead), R-03 (liquidez), R-10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.history_requirements import (
    PERMISSIVE, STRICT, InsufficientHistoryError, missing_requirements,
    validate_mode,
)
from src.momentum_replacement import compute_momentum_scores

logger = logging.getLogger("signal_stack")

from src.settings import SEED, TRADING_DAYS

_DEFAULT_MR_WINDOW = 21
_DEFAULT_VOL_WINDOW = 63
_DEFAULT_LIQ_WINDOW = 63


# ===================================================================
# Config
# ===================================================================
@dataclass
class SignalStackConfig:
    """Configurable factor weights for the composite signal score.

    Weights are re-normalized to sum to 1 at compute time if they
    do not already. SHOWCASE: equal-weight placeholder defaults.
    """

    w_momentum: float = 0.25
    w_mean_reversion: float = 0.25
    w_volatility: float = 0.25
    w_liquidity: float = 0.25
    mr_window: int = _DEFAULT_MR_WINDOW
    vol_window: int = _DEFAULT_VOL_WINDOW
    liq_window: int = _DEFAULT_LIQ_WINDOW
    momentum_lookback: int = 252
    momentum_skip: int = 21

    def weights(self) -> Dict[str, float]:
        return {
            "momentum": self.w_momentum,
            "mean_reversion": self.w_mean_reversion,
            "volatility": self.w_volatility,
            "liquidity": self.w_liquidity,
        }


# ===================================================================
# History requirements (filas con índice <= t; los factores usan < t)
# ===================================================================
def factor_requirements(config: Optional[SignalStackConfig] = None) -> Dict[str, int]:
    """Filas hábiles (índice <= t) que necesita cada factor con peso > 0."""
    config = config or SignalStackConfig()
    need = {
        # compute_momentum_scores exige lookback+skip filas estrictamente < t
        "momentum": config.momentum_lookback + config.momentum_skip + 1,
        # los demás exigen window+1 filas < t
        "mean_reversion": config.mr_window + 2,
        "volatility": config.vol_window + 2,
        "liquidity": config.liq_window + 2,
    }
    return {k: v for k, v in need.items() if config.weights()[k] > 0}


def required_history(config: Optional[SignalStackConfig] = None) -> int:
    """Máximo de ``factor_requirements`` (0 si ningún factor tiene peso)."""
    return max(factor_requirements(config).values(), default=0)


# ===================================================================
# Cross-sectional normalization
# ===================================================================
def _cs_zscore(s: pd.Series, fill: Optional[float] = 0.0) -> pd.Series:
    """Cross-sectional z-score; zeros where defined if std ≈ 0.

    ``fill=None`` conserva NaN (ticker sin dato para ese factor).
    """
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        z = pd.Series(0.0, index=s.index).where(s.notna())
    else:
        z = (s - mu) / sd
    return z if fill is None else z.fillna(fill)


def _cs_rank_to_unit(s: pd.Series) -> pd.Series:
    """Rank to approximately [-1, 1]."""
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=s.index)
    # percentile in (0,1] → map to [-1, 1]
    pct = r / n
    return ((pct - 0.5) * 2.0).fillna(0.0)


# ===================================================================
# Individual factors (causal: data strictly before t)
# ===================================================================
def _mean_reversion_scores(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_MR_WINDOW,
) -> pd.Series:
    """Z-score of price vs rolling MA: negative z → oversold (buy signal).

    Score = -z so that high score prefers mean-reversion longs.
    """
    past = prices[prices.index < t]
    if len(past) < window + 1:
        return pd.Series(np.nan, index=prices.columns)
    ma = past.iloc[-window:].mean()
    std = past.iloc[-window:].std(ddof=1).replace(0, np.nan)
    last = past.iloc[-1]
    z = (last - ma) / std
    return (-z).astype(float)


def _volatility_scores(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Inverse realized-vol rank: high score = low volatility (low-vol tilt)."""
    past = prices[prices.index < t]
    if len(past) < window + 1:
        return pd.Series(np.nan, index=prices.columns)
    rets = past.ffill().pct_change().iloc[-window:]
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    # Invert: low vol → high score
    return (-vol).astype(float)


def _liquidity_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_LIQ_WINDOW,
) -> pd.Series:
    """Liquidity score from inverse Amihud illiquidity (higher = more liquid).

    Amihud_i = mean(|r_i| / dollar_volume_i) over window.
    Score = -Amihud so liquid names score high.

    Volumen faltante (NaN) o cero no se imputa: esos días se omiten y un
    ticker con menos de ``window // 2`` días válidos queda NaN (sin score).
    """
    past_px = prices[prices.index < t]
    past_vol = volumes[volumes.index < t]
    if len(past_px) < window + 1 or past_vol.empty:
        return pd.Series(np.nan, index=prices.columns)

    px = past_px.iloc[-(window + 1):]
    vol = past_vol.reindex(px.index).iloc[-window:]
    rets = px.ffill().pct_change().iloc[-window:]
    # Align columns
    cols = [c for c in prices.columns if c in vol.columns and c in rets.columns]
    if not cols:
        return pd.Series(np.nan, index=prices.columns)

    abs_ret = rets[cols].abs()
    # Dollar volume ≈ price * volume (use mid-window prices)
    mid_px = px[cols].iloc[-window:]
    dvol = (mid_px * vol[cols]).replace(0, np.nan)
    ratio = abs_ret / dvol
    amihud = ratio.mean().where(ratio.notna().sum() >= max(1, window // 2))
    # High liquidity = low Amihud → negate
    out = (-amihud).reindex(prices.columns)
    return out.astype(float)


# ===================================================================
# Composite
# ===================================================================
def compute_signal_scores_with_meta(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    config: Optional[SignalStackConfig] = None,
    mode: str = PERMISSIVE,
) -> Tuple[pd.Series, dict]:
    """Composite multi-factor scores at t plus meta on which factors were used.

    Each factor is cross-sectionally z-scored then combined with its weight.
    A factor lacking history (``factor_requirements``) or with no data at all
    is never zero-filled: in ``mode="strict"`` it raises
    ``InsufficientHistoryError``; in ``"permissive"`` it is excluded and the
    remaining weights are renormalized. Tickers missing an individual factor
    get their weights renormalized over the factors they do have.

    Args:
        prices: Full clean price DataFrame.
        volumes: Volume DataFrame aligned with prices (NaN = unknown volume).
        t: Reference date (exclusive upper bound for factor data, R-05).
        config: SignalStackConfig. None → defaults.
        mode: "strict" or "permissive".

    Returns:
        (scores, meta). scores: Series indexed by ticker (mean≈0, std≈1).
        meta: {"mode", "history_rows", "requirements", "factors_used",
        "factors_excluded" (name → reason), "weights" (renormalized),
        "tickers_without_score"}.
    """
    validate_mode(mode)
    if config is None:
        config = SignalStackConfig()

    requirements = factor_requirements(config)
    history_rows = int((prices.index <= t).sum())
    short = missing_requirements(history_rows, requirements)
    if short and mode == STRICT:
        raise InsufficientHistoryError(
            history_rows, {f"signal_stack.{k}": v for k, v in short.items()},
        )

    compute = {
        "momentum": lambda: compute_momentum_scores(
            prices, t,
            lookback=config.momentum_lookback,
            skip_month=config.momentum_skip,
        ),
        "mean_reversion": lambda: _mean_reversion_scores(
            prices, t, window=config.mr_window),
        "volatility": lambda: _volatility_scores(
            prices, t, window=config.vol_window),
        "liquidity": lambda: _liquidity_scores(
            prices, volumes, t, window=config.liq_window),
    }

    excluded: Dict[str, str] = {
        k: f"history {history_rows} < {v}" for k, v in short.items()
    }
    zscores: Dict[str, pd.Series] = {}
    for name in requirements:
        if name in excluded:
            continue
        z = _cs_zscore(compute[name](), fill=None).reindex(prices.columns)
        if z.notna().sum() == 0:
            excluded[name] = "no data"
            continue
        zscores[name] = z

    if mode == STRICT and excluded:
        # Historia ya validada arriba: aquí solo quedan factores sin datos
        raise InsufficientHistoryError(
            history_rows, {f"signal_stack.{k}": r for k, r in excluded.items()},
        )
    if excluded:
        logger.warning("signal_stack excluded factors at %s: %s", t, excluded)

    w_raw = config.weights()
    w_sum = sum(w_raw[k] for k in zscores)
    weights = {k: w_raw[k] / w_sum for k in zscores} if w_sum > 1e-12 else {}

    if weights:
        z_df = pd.DataFrame({k: zscores[k] for k in weights})
        w = pd.Series(weights)
        num = z_df.mul(w, axis=1).sum(axis=1, min_count=1)
        den = z_df.notna().mul(w, axis=1).sum(axis=1)
        composite = num / den.replace(0.0, np.nan)
    else:
        composite = pd.Series(np.nan, index=prices.columns, dtype=float)
    without = composite.index[composite.isna()].tolist()

    result = _cs_zscore(composite)
    meta = {
        "mode": mode,
        "history_rows": history_rows,
        "requirements": requirements,
        "factors_used": list(weights),
        "factors_excluded": excluded,
        "weights": weights,
        "tickers_without_score": without,
    }
    logger.info(
        "signal_scores at %s: factors=%s mean=%.3f std=%.3f",
        t.date() if hasattr(t, "date") else t, list(weights),
        float(result.mean()), float(result.std(ddof=1) or 0.0),
    )
    return result, meta


def compute_signal_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    config: Optional[SignalStackConfig] = None,
    mode: str = PERMISSIVE,
) -> pd.Series:
    """Composite scores only; see ``compute_signal_scores_with_meta``."""
    return compute_signal_scores_with_meta(prices, volumes, t, config, mode)[0]
