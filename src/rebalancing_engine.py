"""
rebalancing_engine.py — T-24: Motor de reequilibrio integrado.

Función central:
  rebalance(t, w_current, w_target, rule, ...) →
    {w_new, trades, cost, actual_turnover}

Integra:
  - T-19: drift_monitor (detección de deriva)
  - T-20: momentum_replacement
  - T-21: random_replacement
  - T-22: min_variance_replacement
  - T-23: order_execution (TWAP + market impact)

Restricciones cubiertas: R-01 (costos), R-02 (turnover), R-07 (reglas).
Topes por grupo (group_caps, criterios.md v2) son duros sobre w_new: si el
blend de turnover los incumple, se vende solo el exceso (R-02 cede 2x exceso).

El libro actual nunca se recorta al objetivo: se trabaja sobre la union
(tenencias actuales ∪ objetivo); lo que sale del objetivo se vende entero,
con costo (salida forzada, fuera del tope R-02, como en la ruta ledger/vivo),
y la caja (1 − Σ w_current) no se renormaliza.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.drift_monitor import check_drift, DriftReport
from src.momentum_replacement import momentum_replace
from src.random_replacement import random_replace
from src.min_variance_replacement import min_variance_replace
from src.order_execution import execute_rebalance, ExecutionResult
from src.sector_constraints import apply_group_caps, sell_down_to_caps

logger = logging.getLogger("rebalancing_engine")

from src.settings import SEED, TRADING_DAYS
_DEFAULT_DRIFT_THRESHOLD = 0.05
_DEFAULT_TURNOVER_LIMIT = 0.20
from src.settings import INITIAL_CASH as _DEFAULT_PORTFOLIO_VALUE
_DEFAULT_REPLACE_PCT = 0.10


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class RebalanceResult:
    """Output of a full rebalance operation."""

    t: pd.Timestamp
    w_before: pd.Series            # Weights before rebalance (holdings ∪ target)
    w_target: pd.Series            # Optimizer target weights
    w_new: pd.Series               # Final post-rebalance weights
    trades: pd.DataFrame           # Per-asset trade details
    cost: float                    # Total execution cost (dollars)
    cost_bps: float                # Cost in basis points
    actual_turnover: float         # sum(|w_new - w_before|)
    rule_applied: str              # Replacement rule used
    universe_changed: bool         # Whether universe was updated
    removed_tickers: List[str]
    added_tickers: List[str]
    drift_report: Optional[DriftReport]
    execution_result: Optional[ExecutionResult]


# ===================================================================
# Main rebalancing function
# ===================================================================
def rebalance(
    t: pd.Timestamp,
    w_current: pd.Series,
    w_target: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rule: str = "none",
    universe: Optional[List[str]] = None,
    active_tickers: Optional[List[str]] = None,
    sigma: Optional[pd.DataFrame] = None,
    portfolio_value: float = _DEFAULT_PORTFOLIO_VALUE,
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
    turnover_limit: float = _DEFAULT_TURNOVER_LIMIT,
    replace_pct: float = _DEFAULT_REPLACE_PCT,
    is_rebalance_date: bool = True,
    # Order execution params
    n_slices: int = 10,
    commission_bps: float = 5.0,
    spread_bps: float = 2.0,
    impact_coef: float = 0.1,
    seed: int = SEED,
    # Topes por grupo (criterios.md v2), duros sobre w_new
    groups: Optional[Dict[str, str]] = None,
    group_caps: Optional[Dict] = None,
    max_weight: Optional[float] = None,
    max_weight_by_ticker: Optional[pd.Series] = None,
) -> RebalanceResult:
    """Execute a full rebalance with replacement rule and cost modeling.

    Works on the union of current holdings and target: a holding absent
    from ``w_target`` (e.g. it left the active universe) is sold in full and
    its cost paid. ``w_current`` is never renormalized; what it leaves below
    1 is cash.

    Steps:
      1. Check drift (T-19) — determine if rebalance is needed.
      2. Apply replacement rule if triggered (T-20/21/22).
      3. Cap w_new to enforce turnover limit (R-02) on the investable book
         (target tickers), from its current mix; holdings outside it are a
         forced exit, outside the cap (as the ledger/live path). With
         group_caps the caps win: only the over-cap excess is sold (and
         rebought outside the group, toward the target), so turnover may
         exceed the limit by 2x that excess on those dates.
      4. Execute orders via TWAP model (T-23) and record costs.

    Args:
        t: Rebalance date.
        w_current: Current portfolio weights (post-drift), every holding
            included; sum <= 1, the rest is cash.
        w_target: Optimizer-suggested target weights.
        prices: Full clean price DataFrame.
        volumes: Full volume DataFrame.
        rule: Replacement rule: 'momentum', 'random', 'min_variance',
              or 'none'.
        universe: Full ticker universe (for replacement candidates).
        active_tickers: Liquid tickers at t (R-03).
        sigma: Covariance matrix (for min_variance rule).
        portfolio_value: Total portfolio value in dollars.
        drift_threshold: Per-asset drift trigger (R-07).
        turnover_limit: Maximum allowed turnover (R-02, default 20%).
        replace_pct: Fraction of portfolio to replace.
        is_rebalance_date: True if this is a scheduled rebalance.
        n_slices: TWAP slices for order execution.
        commission_bps: Commission rate in bps.
        spread_bps: Half bid-ask spread in bps.
        impact_coef: Linear market impact coefficient.
        seed: Semilla base para la regla 'random' (se deriva un
              np.random.default_rng por fecha; sin RNG global).
        groups: Mapeo ticker -> grupo, dict o Series (requerido con
              group_caps).
        group_caps: Grupo (o tupla de grupos) -> tope. Duro sobre w_new:
              el objetivo se re-proyecta antes del blend si los incumple
              (ej. tras un reemplazo; solo sobre sus nombres con peso > 0,
              sin revivir cierres) y se vende solo el exceso
              (sell_down_to_caps).
        max_weight: Tope por activo al re-proyectar el objetivo (R-04).
        max_weight_by_ticker: Tope por activo especifico (ej. 0.15 en CO).

    Returns:
        RebalanceResult with full audit trail.

    Raises:
        ValueError: If rule is not recognized, group_caps lacks groups, or
            the caps are infeasible (from apply_group_caps).
        Exception: Whatever execute_rebalance raises, unchanged (never
            swallowed into a zero-cost rebalance).
    """
    valid_rules = ("none", "momentum", "random", "min_variance")
    if rule not in valid_rules:
        raise ValueError(
            f"rule must be one of {valid_rules}, got '{rule}'"
        )

    tickers = w_target.index.tolist()
    investable = tickers
    held = w_current.index[w_current.fillna(0.0) != 0.0]
    book = tickers + [tk for tk in held if tk not in w_target.index]
    wc = w_current.reindex(book).fillna(0.0)
    wt = w_target.reindex(book).fillna(0.0)

    # --- Step 1: Drift check (T-19) ---
    drift_report = check_drift(
        t=t,
        w_current=wc,
        w_target=wt,
        drift_threshold=drift_threshold,
        is_rebalance_date=is_rebalance_date,
    )

    removed_tickers: List[str] = []
    added_tickers: List[str] = []
    universe_changed = False

    # --- Step 2: Apply replacement rule (T-20/21/22) ---
    current_portfolio = tickers

    if drift_report.should_rebalance and rule != "none":
        if universe is None:
            universe = tickers

        if rule == "momentum":
            res = momentum_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                prices=prices,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
            )
        elif rule == "random":
            res = random_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
                base_seed=seed,
            )
        else:  # min_variance
            res = min_variance_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                prices=prices,
                w_current=wc.reindex(current_portfolio),
                sigma=sigma,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
            )

        removed_tickers = res.removed_tickers
        added_tickers = res.added_tickers
        universe_changed = len(removed_tickers) > 0 or len(added_tickers) > 0

        # Adjust target weights for new universe
        if universe_changed:
            new_tickers = res.new_universe
            # Re-normalize target weights over new universe
            wt_new = w_target.reindex(new_tickers).fillna(0.0)
            # Removed tickers get 0, added tickers get equal share of removed
            total_removed_w = sum(w_target.get(tk, 0.0) for tk in removed_tickers)
            n_added = len(added_tickers)
            for tk in added_tickers:
                wt_new[tk] = (
                    total_removed_w / n_added if n_added > 0 else 0.0
                )
            s = wt_new.sum()
            wt_new = wt_new / s if s > 1e-10 else pd.Series(
                np.ones(len(new_tickers)) / len(new_tickers),
                index=new_tickers,
            )
            # Holdings outside the new universe stay in the book at target 0
            investable = list(new_tickers)
            book = investable + [tk for tk in book if tk not in wt_new.index]
            wt = wt_new.reindex(book).fillna(0.0)
            wc = wc.reindex(book).fillna(0.0)

    # --- Step 3: Apply turnover cap (R-02) ---
    if drift_report.should_rebalance:
        if group_caps:
            if groups is None:
                raise ValueError("group_caps requires `groups`.")
            groups = pd.Series(groups).reindex(wt.index)
            # Objetivo factible ANTES del blend: el turnover apunta al libro
            # que cumple los topes (no a uno que luego haya que reparar).
            support = wt[wt > 0].index
            wt = apply_group_caps(
                wt[support], groups=groups.reindex(support), caps=group_caps,
                max_weight=max_weight, max_weight_by_ticker=max_weight_by_ticker,
            ).reindex(wt.index).fillna(0.0)
        # Holdings outside the investable book are sold in full (forced exit,
        # outside R-02); the cap throttles the investable book, from its
        # current mix.
        wc_inv = wc.reindex(investable)
        s = float(wc_inv.sum())
        w_new = _apply_turnover_cap(
            wc_inv / s if s > 1e-10 else wc_inv,
            wt.reindex(investable), turnover_limit,
        ).reindex(book).fillna(0.0)
        if group_caps:
            w_new = sell_down_to_caps(w_new, wt, groups, group_caps)
        s = w_new.sum()
        w_new = w_new / s if s > 1e-10 else wt.copy()
    else:
        w_new = wc.copy()

    # --- Step 4: Execute orders (T-23) ---
    execution: Optional[ExecutionResult] = None
    cost = 0.0
    cost_bps = 0.0
    if drift_report.should_rebalance:
        # No try/except: a failed execution must not come back as w_new with
        # cost 0 (it would look traded for free). The caller decides.
        execution = execute_rebalance(
            t=t,
            w_current=wc,
            w_target=w_new,
            portfolio_value=portfolio_value,
            prices=prices,
            volumes=volumes,
            n_slices=n_slices,
            commission_bps=commission_bps,
            spread_bps=spread_bps,
            impact_coef=impact_coef,
        )
        cost = execution.total_cost
        cost_bps = execution.total_cost_bps
        trades_df = execution.order_df
    else:
        trades_df = pd.DataFrame()

    # Actual turnover
    actual_turnover = float((w_new - wc).abs().sum())

    logger.info(
        "rebalance(%s, rule=%s): turnover=%.4f, cost=%.4f "
        "(%.2f bps), universe_changed=%s",
        t.date(), rule, actual_turnover, cost, cost_bps, universe_changed,
    )
    return RebalanceResult(
        t=t,
        w_before=wc,
        w_target=wt,
        w_new=w_new,
        trades=trades_df,
        cost=round(cost, 6),
        cost_bps=round(cost_bps, 4),
        actual_turnover=round(actual_turnover, 6),
        rule_applied=rule,
        universe_changed=universe_changed,
        removed_tickers=removed_tickers,
        added_tickers=added_tickers,
        drift_report=drift_report,
        execution_result=execution,
    )


# ===================================================================
# Helper: turnover cap
# ===================================================================
def _apply_turnover_cap(
    w_current: pd.Series,
    w_target: pd.Series,
    turnover_limit: float,
) -> pd.Series:
    """Blend w_current and w_target to respect turnover cap (R-02).

    If |w_target - w_current|_1 > turnover_limit, interpolate:
      w_new = w_current + alpha * (w_target - w_current)
    where alpha = turnover_limit / actual_turnover.

    Args:
        w_current: Current weights.
        w_target: Target weights.
        turnover_limit: Maximum L1 turnover.

    Returns:
        Blended weight vector respecting the turnover limit.
    """
    tickers = w_target.index
    wc = w_current.reindex(tickers).fillna(0.0)
    delta = w_target - wc
    l1 = float(delta.abs().sum())

    if l1 <= turnover_limit + 1e-8:
        return w_target.copy()

    alpha = turnover_limit / l1
    w_blended = wc + alpha * delta
    w_blended = w_blended.clip(lower=0.0)
    s = w_blended.sum()
    return w_blended / s if s > 1e-10 else w_target.copy()
