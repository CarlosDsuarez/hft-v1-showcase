"""
mv_optimizer.py — T-12: Optimizador media-varianza clásico.

Maximiza el Sharpe ratio sujeto a:
  sum(w) = 1, w_i >= 0, w_i <= max_weight

Transformación Charnes-Cooper para reformular como SOCP.
Solver: cvxpy + CLARABEL (interior-point SOCP); fallback SCS.

Restricciones cubiertas: R-04 (long-only, cap de pesos).
"""

import logging
import time
from typing import Dict

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger("mv_optimizer")

from src.settings import SEED, TRADING_DAYS
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_RISK_FREE = 0.0
_REG_EPS = 1e-6

# Status when n * max_weight < 1: equal weights returned, solver not called.
CAP_INFEASIBLE = "cap_infeasible"


# ===================================================================
# Helpers
# ===================================================================
def cap_is_infeasible(n: int, max_weight: float) -> bool:
    """True if no long-only w with sum(w)=1 satisfies w_i <= max_weight.

    Summing the n caps gives 1 = sum(w) <= n * max_weight. When that fails,
    equal weight is the portfolio that breaches the cap least (the cap
    relaxed to its minimum feasible value, 1/n). Same tolerance as
    ``src.strategy.enforce_max_weight``.
    """
    return n * max_weight < 1.0 - 1e-12


def _make_pd(sigma: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    """Add ridge regularization to ensure positive-definiteness."""
    return sigma + eps * np.eye(sigma.shape[0])


def _cholesky_safe(sigma: np.ndarray) -> np.ndarray:
    """Cholesky with fallback to eigendecomposition.

    Args:
        sigma: Symmetric PD matrix.

    Returns:
        Lower triangular L s.t. sigma ≈ L @ L.T
    """
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(np.sqrt(eigvals))


# ===================================================================
# Main optimizer
# ===================================================================
def mv_optimize(
    mu: pd.Series,
    sigma: pd.DataFrame,
    max_weight: float = _DEFAULT_MAX_WEIGHT,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> Dict:
    """Maximize Sharpe ratio via Charnes-Cooper SOCP transform.

    Charnes-Cooper: let y = w/sigma_p, t = 1/sigma_p.
    Maximize (mu - rf)' y
    subject to:
      ||L' y||_2 <= 1   (SOCP: sigma_p = 1 at optimum)
      sum(y) >= 0        (t > 0)
      y >= 0             (long-only, R-04)
      y_i <= max_w * sum(y)  (w_i <= max_w, R-04)
    Recover w = y / sum(y).

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        max_weight: Per-asset weight cap (R-04, default 0.30). If
            n * max_weight < 1 no portfolio meets it: logs a warning and
            returns equal weights with status CAP_INFEASIBLE, no solve.
        risk_free_rate: Risk-free rate for Sharpe computation.

    Raises:
        ValueError: ``mu`` is empty.

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal weights summing to 1.
            expected_sharpe (float): In-sample Sharpe ratio.
            status (str): cvxpy solver status, or CAP_INFEASIBLE.
            solve_time (float): Wall-clock seconds.
    """
    tickers = mu.index.tolist()
    n = len(tickers)
    if n == 0:
        raise ValueError("mv_optimize: mu is empty (no assets to optimize).")
    if cap_is_infeasible(n, max_weight):
        logger.warning(
            "mv_optimize: max_weight=%.4f is infeasible for %d assets "
            "(n * max_weight = %.4f < 1). Solver skipped; returning equal "
            "weights (1/n = %.4f).",
            max_weight, n, n * max_weight, 1.0 / n,
        )
        return {
            "weights": pd.Series(np.ones(n) / n, index=tickers),
            "expected_sharpe": 0.0,
            "status": CAP_INFEASIBLE,
            "solve_time": 0.0,
        }
    mu_arr = mu.values.astype(float)
    sigma_arr = _make_pd(sigma.values.astype(float))
    L = _cholesky_safe(sigma_arr)

    # --- Charnes-Cooper variables ---
    y = cp.Variable(n, nonneg=True)
    excess_mu = mu_arr - risk_free_rate

    objective = cp.Maximize(excess_mu @ y)
    constraints = [
        # SOCP: sqrt(y' Sigma y) <= 1
        cp.norm(L.T @ y, 2) <= 1,
        # Max weight: w_i = y_i/sum(y) <= max_weight
        y <= max_weight * cp.sum(y),
        # t = sum(y) = 1/sigma_p > 0
        cp.sum(y) >= 1e-6,
    ]

    prob = cp.Problem(objective, constraints)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except cp.SolverError:
        logger.warning("CLARABEL failed — retrying with SCS.")
        prob.solve(solver=cp.SCS, verbose=False)
    solve_time = time.perf_counter() - t0

    status = prob.status or "unknown"

    # --- Fallback: equal weights if not converged ---
    if y.value is None or status not in (
        cp.OPTIMAL, cp.OPTIMAL_INACCURATE
    ):
        logger.warning(
            "mv_optimize: solver did not converge (status=%s). "
            "Returning equal weights.",
            status,
        )
        w = pd.Series(np.ones(n) / n, index=tickers)
        return {
            "weights": w,
            "expected_sharpe": 0.0,
            "status": status,
            "solve_time": round(solve_time, 4),
        }

    # --- Recover and clean weights ---
    y_val = np.maximum(y.value, 0.0)
    denom = y_val.sum()
    w_arr = y_val / denom if denom > 1e-10 else np.ones(n) / n
    # Enforce bounds numerically (R-04)
    w_arr = np.clip(w_arr, 0.0, max_weight)
    w_arr /= w_arr.sum()
    w = pd.Series(np.round(w_arr, 4), index=tickers)
    w /= w.sum()  # re-normalize after rounding

    port_ret = float(mu_arr @ w.values)
    port_vol = float(np.sqrt(w.values @ sigma_arr @ w.values))
    sharpe = (
        (port_ret - risk_free_rate) / port_vol
        if port_vol > 1e-10 else 0.0
    )

    logger.info(
        "mv_optimize: Sharpe=%.4f, status=%s, solve_time=%.3fs",
        sharpe, status, solve_time,
    )
    return {
        "weights": w,
        "expected_sharpe": round(sharpe, 6),
        "status": status,
        "solve_time": round(solve_time, 4),
    }
