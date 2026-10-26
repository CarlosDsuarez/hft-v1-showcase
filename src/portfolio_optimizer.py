"""
portfolio_optimizer.py — T-17: Módulo principal de optimización.

Interfaz unificada para todos los métodos de optimización de Fase 3.

Métodos disponibles (registro ``OPTIMIZERS``):
  "mv_classic"  — Máximo Sharpe clásico (T-12)
  "robust"      — Robusto SOCP con incertidumbre elipsoidal (T-13)
  "mv_tc"       — MV con costos de transacción (T-14)
  "robust_tc"   — Robusto + costos de transacción
  "mv_turnover" — MV con restricción de turnover (T-15)
  "liquidity"   — MV con topes por volumen (T-16); requiere t y volumes

Contrato:
  ``OptimizationInput``  — todo lo que un método puede consumir.
  ``Optimizer.requires`` — campos Optional de la entrada sin los que el
                           método no corre (validable antes de correr: ver
                           ``StrategyConfig``).
  ``OptimizationResult`` — weights, status, converged, fallback_used, meta.
                           ``fallback_used`` = los pesos NO salen del solver
                           (equal weight); el caller decide qué hacer.

``run_optimizer(method, inp)`` es la ruta tipada; ``optimize(...)`` la
envuelve y devuelve el dict histórico (weights, expected_sharpe, turnover,
cost, converged, solve_time, method, status, caps_applied, fallback_used).

Restricciones cubiertas: R-01, R-02, R-03, R-04, R-10.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Union

import numpy as np
import pandas as pd

from src.mv_optimizer import mv_optimize
from src.robust_optimizer import robust_optimize
from src.optimizer_with_tc import optimize_with_tc
from src.turnover_constraint import optimize_with_turnover
from src.liquidity_constraint import optimize_with_liquidity
from src.sector_constraints import apply_group_caps

logger = logging.getLogger("portfolio_optimizer")

from src.settings import SEED, TRADING_DAYS

_CONVERGED = ("optimal", "optimal_inaccurate")


# ===================================================================
# Contrato
# ===================================================================
@dataclass(frozen=True, eq=False)
class OptimizationInput:
    """Entrada de un optimizador (datos + parámetros).

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        w_prev: Previous portfolio weights (for TC / turnover). None → 0.
        t: Rebalance date ('liquidity', R-05).
        volumes: Volume DataFrame ('liquidity').
        illiquidity: Amihud scores (optional for 'liquidity').
        costs: Per-asset costs Series (for TC methods).
        groups: Ticker -> group mapping (Series or dict, e.g.
            "co/Financials" via `build_mixed_groups`).
        group_caps: Group -> max fraction (keys may be tuples for
            combined caps, e.g. {("co/Financials","co/Energy"): 0.40}).
        max_weight_by_group: Group -> per-asset cap (e.g. 0.15 for
            "co/*" groups). Requires `groups`.
        max_weight: Per-asset weight cap (R-04, default 0.30).
        turnover_limit: Max turnover for 'mv_turnover' method (R-02).
        tc_lambda: TC penalty coefficient for 'mv_tc', 'robust_tc'.
        budget_tc: TC budget for mode='budget'.
        tc_mode: 'penalty' or 'budget' for TC methods.
        turnover_mode: 'hard' or 'soft' for turnover method.
        kappa: Robustness parameter for 'robust' / 'robust_tc'.
        n_obs: Observations used to estimate mu (for robust).
        risk_free_rate: Risk-free rate for Sharpe computation.
        risk_aversion: Risk aversion gamma in MV objective.
    """

    mu: pd.Series
    sigma: pd.DataFrame
    w_prev: Optional[pd.Series] = None
    t: Optional[pd.Timestamp] = None
    volumes: Optional[pd.DataFrame] = None
    illiquidity: Optional[pd.Series] = None
    costs: Optional[pd.Series] = None
    groups: Optional[Union[pd.Series, Dict]] = None
    group_caps: Optional[Dict] = None
    max_weight_by_group: Optional[Dict] = None
    max_weight: float = 0.30
    turnover_limit: float = 0.20
    tc_lambda: float = 0.001
    budget_tc: float = 0.01
    tc_mode: str = "penalty"
    turnover_mode: str = "hard"
    kappa: float = 0.1
    n_obs: int = TRADING_DAYS
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    """Salida de ``run_optimizer``.

    Attributes:
        weights: Pesos long-only (suman 1) tras topes por activo y grupo.
        status: Estado del solver ("optimal", "infeasible", ...).
        converged: status en ("optimal", "optimal_inaccurate").
        fallback_used: Los pesos son equal weight de respaldo, no la
            solución del solver (no convergió o salió degenerada).
        meta: expected_sharpe, turnover, cost, solve_time, method,
            caps_applied + claves extra del optimizador subyacente.
    """

    weights: pd.Series
    status: str
    converged: bool
    fallback_used: bool
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Optimizer:
    """Entrada del registro: cómo correr un método y qué necesita.

    ``run`` recibe la entrada con ``w_prev``/``costs`` ya alineados a
    ``mu.index`` y devuelve el dict del optimizador subyacente (weights,
    status, ...). ``requires`` = campos de ``OptimizationInput`` que no
    pueden ser None.
    """

    run: Callable[[OptimizationInput], Dict]
    requires: FrozenSet[str] = frozenset()


# ===================================================================
# Métodos
# ===================================================================
def _mv_classic(inp: OptimizationInput) -> Dict:
    return mv_optimize(
        mu=inp.mu,
        sigma=inp.sigma,
        max_weight=inp.max_weight,
        risk_free_rate=inp.risk_free_rate,
    )


def _robust(inp: OptimizationInput) -> Dict:
    return robust_optimize(
        mu=inp.mu,
        sigma=inp.sigma,
        n_obs=inp.n_obs,
        kappa=inp.kappa,
        max_weight=inp.max_weight,
        risk_free_rate=inp.risk_free_rate,
    )


def _mv_tc(inp: OptimizationInput) -> Dict:
    return optimize_with_tc(
        mu=inp.mu,
        sigma=inp.sigma,
        w_prev=inp.w_prev,
        mode=inp.tc_mode,
        tc_lambda=inp.tc_lambda,
        budget_tc=inp.budget_tc,
        costs=inp.costs,
        max_weight=inp.max_weight,
        risk_aversion=inp.risk_aversion,
    )


def _robust_tc(inp: OptimizationInput) -> Dict:
    # Robust optimization + TC penalty (combined)
    robust_res = _robust(inp)
    if robust_res["status"] not in _CONVERGED:
        return robust_res
    # If robust converged, apply TC penalty on top via mv_tc
    result = _mv_tc(inp)
    result["robust_sharpe"] = robust_res.get("robust_sharpe", 0.0)
    return result


def _mv_turnover(inp: OptimizationInput) -> Dict:
    return optimize_with_turnover(
        mu=inp.mu,
        sigma=inp.sigma,
        w_prev=inp.w_prev,
        mode=inp.turnover_mode,
        delta=inp.turnover_limit,
        max_weight=inp.max_weight,
        risk_aversion=inp.risk_aversion,
    )


def _liquidity(inp: OptimizationInput) -> Dict:
    return optimize_with_liquidity(
        mu=inp.mu,
        sigma=inp.sigma,
        t=inp.t,
        volumes=inp.volumes,
        illiquidity_scores=inp.illiquidity,
        w_prev=inp.w_prev,
        global_max_weight=inp.max_weight,
        risk_aversion=inp.risk_aversion,
    )


OPTIMIZERS: Dict[str, Optimizer] = {
    "mv_classic": Optimizer(_mv_classic),
    "robust": Optimizer(_robust),
    "mv_tc": Optimizer(_mv_tc),
    "robust_tc": Optimizer(_robust_tc),
    "mv_turnover": Optimizer(_mv_turnover),
    "liquidity": Optimizer(_liquidity, requires=frozenset({"t", "volumes"})),
}

VALID_METHODS = tuple(OPTIMIZERS)


# ===================================================================
# Dispatcher
# ===================================================================
def run_optimizer(method: str, inp: OptimizationInput) -> OptimizationResult:
    """Corre ``method`` sobre ``inp`` y estandariza la salida.

    Tras el solver: topes por activo/grupo (R-04), normalización, turnover,
    costo y Sharpe recalculados sobre los pesos finales.

    Raises:
        ValueError: ``method`` no está en ``OPTIMIZERS``, falta una entrada
            de ``requires``, o topes por grupo sin ``groups``.
    """
    spec = OPTIMIZERS.get(method)
    if spec is None:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid: {list(OPTIMIZERS)}"
        )
    missing = sorted(f for f in spec.requires if getattr(inp, f) is None)
    if missing:
        raise ValueError(
            f"method='{method}' requires {' and '.join(missing)} arguments."
        )

    mu, sigma, max_weight = inp.mu, inp.sigma, inp.max_weight
    tickers = mu.index.tolist()
    n = len(tickers)
    equal = pd.Series(np.ones(n) / n, index=tickers)

    # Default w_prev: zero weights
    if inp.w_prev is None:
        w_prev_aligned = pd.Series(np.zeros(n), index=tickers)
    else:
        w_prev_aligned = inp.w_prev.reindex(tickers).fillna(0.0)

    costs_aligned = (
        inp.costs.reindex(tickers).fillna(0.001)
        if inp.costs is not None
        else None
    )

    t0_total = time.perf_counter()
    result = spec.run(
        dataclasses.replace(inp, w_prev=w_prev_aligned, costs=costs_aligned)
    )
    total_time = time.perf_counter() - t0_total

    # Convergence flag. Every underlying optimizer hands back equal weight
    # when the solver does not converge, so those weights are a fallback.
    status = result.get("status", "unknown")
    converged = status in _CONVERGED
    fallback_used = not converged
    weights = result.get("weights")
    if weights is None:
        weights, fallback_used = equal, True

    # Per-ticker caps by group (e.g. 0.15 en sleeve CO, R-04 extendido)
    ub_by_ticker = None
    if inp.max_weight_by_group is not None:
        if inp.groups is None:
            raise ValueError("max_weight_by_group requires `groups`.")
        gmap = pd.Series(inp.groups).reindex(tickers)
        ub_by_ticker = gmap.map(
            lambda grp: inp.max_weight_by_group.get(grp, max_weight)
        ).astype(float)

    # Ensure weights are properly normalized and long-only (R-04)
    weights = weights.reindex(tickers).fillna(0.0)
    if ub_by_ticker is not None:
        weights = weights.clip(lower=0.0, upper=ub_by_ticker)
    else:
        weights = weights.clip(lower=0.0, upper=max_weight)
    if weights.sum() > 1e-10:
        weights /= weights.sum()
    else:
        weights, fallback_used = equal, True

    # Group/sector caps projection (criterios.md v2)
    caps_applied = False
    if inp.group_caps:
        if inp.groups is None:
            raise ValueError("group_caps requires `groups`.")
        weights = apply_group_caps(
            weights,
            groups=pd.Series(inp.groups).reindex(tickers),
            caps=inp.group_caps,
            max_weight=max_weight,
            max_weight_by_ticker=ub_by_ticker,
        )
        caps_applied = True

    # Compute turnover
    turnover = float(
        np.sum(np.abs(weights.values - w_prev_aligned.values))
    )

    # Estimate TC cost
    c_arr = (
        costs_aligned.values if costs_aligned is not None
        else np.full(n, 0.001)
    )
    cost = float(c_arr @ np.abs(weights.values - w_prev_aligned.values))

    # Expected Sharpe (recompute to ensure consistency)
    mu_arr = mu.values.astype(float)
    sigma_arr = sigma.values.astype(float) + _REG_EPS * np.eye(n)
    wf = weights.values
    port_ret = float(mu_arr @ wf)
    port_vol = float(np.sqrt(wf @ sigma_arr @ wf))
    sharpe = (
        (port_ret - inp.risk_free_rate) / port_vol
        if port_vol > 1e-10 else 0.0
    )

    meta: Dict[str, Any] = {
        "expected_sharpe": round(sharpe, 6),
        "turnover": round(turnover, 6),
        "cost": round(cost, 6),
        "solve_time": round(total_time, 4),
        "method": method,
        "caps_applied": caps_applied,
    }
    # Merge any extra keys from the underlying optimizer
    for k, v in result.items():
        if k not in meta and k not in ("weights", "status"):
            meta[k] = v

    logger.info(
        "optimize(method=%s): Sharpe=%.4f, turnover=%.4f, "
        "cost=%.4f, converged=%s, fallback_used=%s",
        method, sharpe, turnover, cost, converged, fallback_used,
    )
    return OptimizationResult(
        weights=weights,
        status=status,
        converged=converged,
        fallback_used=fallback_used,
        meta=meta,
    )


def optimize(
    mu: pd.Series,
    sigma: pd.DataFrame,
    method: str = "mv_classic",
    w_prev: Optional[pd.Series] = None,
    max_weight: float = 0.30,
    turnover_limit: float = 0.20,
    tc_lambda: float = 0.001,
    budget_tc: float = 0.01,
    tc_mode: str = "penalty",
    turnover_mode: str = "hard",
    kappa: float = 0.1,
    n_obs: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    costs: Optional[pd.Series] = None,
    # Liquidity method extras
    t: Optional[pd.Timestamp] = None,
    volumes: Optional[pd.DataFrame] = None,
    illiquidity_scores: Optional[pd.Series] = None,
    # Group/sector caps (Paso 3, criterios.md v2)
    groups: Optional[Union[pd.Series, Dict]] = None,
    group_caps: Optional[Dict] = None,
    max_weight_by_group: Optional[Dict] = None,
) -> Dict:
    """Unified portfolio optimization interface (dict form).

    Backward-compatible wrapper over ``run_optimizer``: same arguments as
    the fields of ``OptimizationInput`` (``illiquidity_scores`` →
    ``illiquidity``).

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal portfolio weights.
            expected_sharpe (float): In-sample Sharpe ratio.
            turnover (float): sum(|w - w_prev|).
            cost (float): Estimated transaction cost.
            converged (bool): Whether solver converged.
            fallback_used (bool): Weights are the equal-weight fallback.
            solve_time (float): Solver wall-clock time in seconds.
            method (str): Method used.
            status (str): Solver status.
            caps_applied (bool): Whether group caps were projected.
            [extra keys depending on method]

    Raises:
        ValueError: If method is not in OPTIMIZERS or a required input
            (e.g. t / volumes for 'liquidity') is missing.
    """
    res = run_optimizer(method, OptimizationInput(
        mu=mu,
        sigma=sigma,
        w_prev=w_prev,
        t=t,
        volumes=volumes,
        illiquidity=illiquidity_scores,
        costs=costs,
        groups=groups,
        group_caps=group_caps,
        max_weight_by_group=max_weight_by_group,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
        tc_lambda=tc_lambda,
        budget_tc=budget_tc,
        tc_mode=tc_mode,
        turnover_mode=turnover_mode,
        kappa=kappa,
        n_obs=n_obs,
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
    ))
    return {
        "weights": res.weights,
        "converged": res.converged,
        "fallback_used": res.fallback_used,
        "status": res.status,
        **res.meta,
    }


_REG_EPS = 1e-6
