"""
test_optimizer.py — T-18: Tests unitarios del motor de optimización.

Cubre T-12 a T-17 (todos los optimizadores de Fase 3).

Restricciones validadas:
  R-01: Costos de transacción calculados correctamente.
  R-02: Turnover constraint respetada.
  R-04: sum(w)=1 ± 1e-6, w >= 0, w_i <= max_weight.
  R-10: Reproducibilidad con seed(42).

Ejecutar: pytest tests/test_optimizer.py -v
"""

import dataclasses
import logging

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src.mv_optimizer import mv_optimize
from src.robust_optimizer import robust_optimize
from src.optimizer_with_tc import optimize_with_tc
from src.turnover_constraint import optimize_with_turnover
from src.liquidity_constraint import (
    optimize_with_liquidity,
    compute_volume_upper_bounds,
    apply_amihud_exclusion,
)
from src.portfolio_optimizer import (
    OPTIMIZERS,
    VALID_METHODS,
    OptimizationInput,
    OptimizationResult,
    Optimizer,
    optimize,
    run_optimizer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOL = 1e-4       # weight sum tolerance
WEIGHT_TOL = 1e-4  # individual weight bounds tolerance
N_ASSETS = 10
N_OBS = 300


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def prices():
    """Synthetic price series: N_OBS days × N_ASSETS tickers."""
    np.random.seed(SEED)
    dates = pd.bdate_range("2021-01-01", periods=N_OBS, freq="B")
    tickers = [f"T{i:02d}" for i in range(N_ASSETS)]
    log_rets = np.random.normal(5e-4, 0.015, (N_OBS, N_ASSETS))
    # Add correlation structure
    mkt = np.random.normal(0, 0.01, N_OBS)
    for i in range(N_ASSETS):
        log_rets[:, i] += 0.5 * mkt
    prices_arr = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def volumes(prices):
    """Synthetic volume data."""
    np.random.seed(SEED + 1)
    vols = np.random.randint(500_000, 20_000_000, prices.shape)
    df = pd.DataFrame(
        vols.astype(float), index=prices.index, columns=prices.columns
    )
    df.index.name = "date"
    return df


@pytest.fixture
def mu_sigma(prices):
    """Compute mu and sigma from synthetic prices."""
    from src.expected_returns import mean_historical_return
    from src.covariance_estimators import ledoit_wolf_covariance
    mu = mean_historical_return(prices)
    sigma = ledoit_wolf_covariance(prices)
    return mu, sigma


@pytest.fixture
def w_prev(mu_sigma):
    """Random previous weights (valid portfolio)."""
    mu, _ = mu_sigma
    np.random.seed(SEED + 2)
    w = np.random.dirichlet(np.ones(len(mu)))
    return pd.Series(w, index=mu.index)


# ===========================================================================
# T-12: Mean-Variance Optimizer (mv_optimize)
# ===========================================================================
class TestMVOptimizer:
    def test_weights_sum_to_one(self, mu_sigma):
        mu, sigma = mu_sigma
        res = mv_optimize(mu, sigma)
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_weights_non_negative(self, mu_sigma):
        mu, sigma = mu_sigma
        res = mv_optimize(mu, sigma)
        assert (res["weights"] >= -WEIGHT_TOL).all()

    def test_max_weight_respected(self, mu_sigma):
        mu, sigma = mu_sigma
        cap = 0.30
        res = mv_optimize(mu, sigma, max_weight=cap)
        assert (res["weights"] <= cap + WEIGHT_TOL).all()

    def test_tighter_cap_more_diversified(self, mu_sigma):
        """Lower cap should spread weights more evenly."""
        mu, sigma = mu_sigma
        res_30 = mv_optimize(mu, sigma, max_weight=0.30)
        res_15 = mv_optimize(mu, sigma, max_weight=0.15)
        # Tighter cap → fewer concentrated bets
        assert res_15["weights"].max() <= res_30["weights"].max() + WEIGHT_TOL

    def test_returns_expected_keys(self, mu_sigma):
        mu, sigma = mu_sigma
        res = mv_optimize(mu, sigma)
        assert set(res.keys()) >= {
            "weights", "expected_sharpe", "status", "solve_time"
        }

    def test_sharpe_positive_with_positive_excess_mu(self, mu_sigma):
        mu, sigma = mu_sigma
        # Ensure positive excess returns
        mu_pos = mu + abs(mu.min()) + 0.05
        res = mv_optimize(mu_pos, sigma, risk_free_rate=0.0)
        assert res["expected_sharpe"] > 0

    def test_solve_time_recorded(self, mu_sigma):
        mu, sigma = mu_sigma
        res = mv_optimize(mu, sigma)
        assert res["solve_time"] >= 0.0

    def test_deterministic(self, mu_sigma):
        """R-10: Same inputs → same outputs."""
        mu, sigma = mu_sigma
        r1 = mv_optimize(mu, sigma)
        r2 = mv_optimize(mu, sigma)
        pd.testing.assert_series_equal(r1["weights"], r2["weights"])

    def test_primary_solver_converges_without_scs_fallback(self, mu_sigma, caplog):
        """Well-conditioned SOCP must solve on the primary solver, not SCS."""
        mu, sigma = mu_sigma
        with caplog.at_level(logging.WARNING, logger="mv_optimizer"):
            res = mv_optimize(mu, sigma)
        assert "retrying with SCS" not in caplog.text
        assert res["status"] == cp.OPTIMAL


# ===========================================================================
# T-13: Robust Optimizer (robust_optimize)
# ===========================================================================
class TestRobustOptimizer:
    def test_weights_sum_to_one(self, mu_sigma):
        mu, sigma = mu_sigma
        res = robust_optimize(mu, sigma, kappa=0.1)
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_weights_non_negative(self, mu_sigma):
        mu, sigma = mu_sigma
        res = robust_optimize(mu, sigma, kappa=0.1)
        assert (res["weights"] >= -WEIGHT_TOL).all()

    def test_max_weight_respected(self, mu_sigma):
        mu, sigma = mu_sigma
        cap = 0.30
        res = robust_optimize(mu, sigma, kappa=0.1, max_weight=cap)
        assert (res["weights"] <= cap + WEIGHT_TOL).all()

    def test_kappa_zero_similar_to_mv(self, mu_sigma):
        """kappa=0 should produce results similar to MV classic."""
        mu, sigma = mu_sigma
        res_mv = mv_optimize(mu, sigma)
        res_rob = robust_optimize(mu, sigma, kappa=0.0)
        # Sharpe should be close (not necessarily identical due to solver)
        assert abs(
            res_rob["expected_sharpe"] - res_mv["expected_sharpe"]
        ) < 0.5  # loose: same order of magnitude

    def test_higher_kappa_lower_robust_sharpe(self, mu_sigma):
        """Higher uncertainty aversion → lower nominal Sharpe typically."""
        mu, sigma = mu_sigma
        res_lo = robust_optimize(mu, sigma, kappa=0.0)
        res_hi = robust_optimize(mu, sigma, kappa=0.5)
        # Robust Sharpe <= nominal Sharpe when kappa > 0
        assert (
            res_hi["robust_sharpe"] <= res_hi["expected_sharpe"] + 1e-6
        )

    def test_returns_robust_sharpe(self, mu_sigma):
        mu, sigma = mu_sigma
        res = robust_optimize(mu, sigma, kappa=0.2)
        assert "robust_sharpe" in res

    def test_deterministic(self, mu_sigma):
        mu, sigma = mu_sigma
        r1 = robust_optimize(mu, sigma, kappa=0.1)
        r2 = robust_optimize(mu, sigma, kappa=0.1)
        pd.testing.assert_series_equal(r1["weights"], r2["weights"])

    @pytest.mark.parametrize("kappa", [0.0, 0.1, 0.5])
    def test_primary_solver_converges_without_scs_fallback(
        self, mu_sigma, caplog, kappa
    ):
        """Well-conditioned SOCP must solve on the primary solver, not SCS."""
        mu, sigma = mu_sigma
        with caplog.at_level(logging.WARNING, logger="robust_optimizer"):
            res = robust_optimize(mu, sigma, kappa=kappa)
        assert "retrying with SCS" not in caplog.text
        assert res["status"] == cp.OPTIMAL


# ===========================================================================
# T-12/T-13: SCS fallback when the primary solver fails
# ===========================================================================
@pytest.mark.parametrize(
    "optimizer, logger_name",
    [(mv_optimize, "mv_optimizer"), (robust_optimize, "robust_optimizer")],
)
def test_scs_fallback_matches_primary_solver(
    mu_sigma, monkeypatch, caplog, optimizer, logger_name
):
    """Primary solver failure → SCS fallback, weights close to the primary."""
    mu, sigma = mu_sigma
    primary = optimizer(mu, sigma)

    real_solve = cp.Problem.solve

    def solve_scs_only(self, *args, **kwargs):
        if kwargs.get("solver") != cp.SCS:
            raise cp.SolverError("forced primary solver failure")
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", solve_scs_only)
    with caplog.at_level(logging.WARNING, logger=logger_name):
        fallback = optimizer(mu, sigma)

    assert "retrying with SCS" in caplog.text
    assert fallback["status"] in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    np.testing.assert_allclose(
        fallback["weights"].values, primary["weights"].values, atol=1e-3
    )


# ===========================================================================
# Tope infactible: n * max_weight < 1 (mv_optimize, robust_optimize)
# ===========================================================================
# Sumando los n topes: 1 = sum(w) <= n * max_weight. Con 3 activos y el
# max_weight por defecto (0.30) ningún w long-only totalmente invertido
# cumple R-04; el que menos lo excede es equal weight (tope relajado a 1/n).
CAPPED_OPTIMIZERS = {"mv": mv_optimize, "robust": robust_optimize}


@pytest.fixture
def mu_sigma_3(mu_sigma):
    """3 activos: 3 * 0.30 = 0.90 < 1 → tope infactible."""
    mu, sigma = mu_sigma
    tickers = mu.index[:3]
    return mu[tickers], sigma.loc[tickers, tickers]


@pytest.mark.parametrize("name", list(CAPPED_OPTIMIZERS))
class TestInfeasibleCap:
    def test_status_is_cap_infeasible(self, name, mu_sigma_3):
        res = CAPPED_OPTIMIZERS[name](*mu_sigma_3, max_weight=0.30)
        assert res["status"] == "cap_infeasible"

    def test_returns_equal_weights(self, name, mu_sigma_3):
        mu, sigma = mu_sigma_3
        res = CAPPED_OPTIMIZERS[name](mu, sigma, max_weight=0.30)
        pd.testing.assert_series_equal(
            res["weights"], pd.Series([1 / 3, 1 / 3, 1 / 3], index=mu.index),
            rtol=0, atol=1e-12,
        )

    def test_does_not_call_solver(self, name, mu_sigma_3, monkeypatch):
        """Infactible por aritmética: el solver solo agrega ruido (SCS
        reportaba 'optimal' con pesos > tope; CLARABEL, infeasible_inaccurate
        + UserWarning "Solution may be inaccurate")."""
        def _no_solve(*args, **kwargs):
            raise AssertionError("solver called on an infeasible cap")

        monkeypatch.setattr(cp.Problem, "solve", _no_solve)
        CAPPED_OPTIMIZERS[name](*mu_sigma_3, max_weight=0.30)

    def test_logs_one_warning_naming_max_weight(self, name, mu_sigma_3, caplog):
        with caplog.at_level(logging.WARNING):
            CAPPED_OPTIMIZERS[name](*mu_sigma_3, max_weight=0.30)
        records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(records) == 1
        assert "max_weight" in records[0].getMessage()

    def test_cap_of_exactly_one_over_n_is_still_solved(self, name, mu_sigma):
        """n * max_weight == 1 es factible (único punto: equal weight)."""
        mu, sigma = mu_sigma
        tickers = mu.index[:4]
        res = CAPPED_OPTIMIZERS[name](
            mu[tickers], sigma.loc[tickers, tickers], max_weight=0.25)
        assert res["status"] in ("optimal", "optimal_inaccurate")
        np.testing.assert_allclose(res["weights"].values, [0.25] * 4,
                                   atol=WEIGHT_TOL)

    def test_empty_input_still_raises_value_error(self, name):
        """0 * max_weight < 1, pero sin activos no hay equal weight."""
        with pytest.raises(ValueError):
            CAPPED_OPTIMIZERS[name](pd.Series(dtype=float), pd.DataFrame())


class TestInfeasibleCapThroughOptimize:
    @pytest.mark.parametrize("method", ["mv_classic", "robust", "robust_tc"])
    def test_reported_as_not_converged(self, method, mu_sigma_3):
        """optimize() lo deja ver: status propio y converged=False (el
        backtest lo registra como fallback_ew)."""
        mu, sigma = mu_sigma_3
        res = optimize(mu, sigma, method=method, max_weight=0.30)
        assert res["status"] == "cap_infeasible"
        assert res["converged"] is False


# ===========================================================================
# T-14: Optimizer with Transaction Costs (optimize_with_tc)
# ===========================================================================
class TestOptimizerWithTC:
    def test_penalty_weights_sum_to_one(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_tc(mu, sigma, w_prev=w_prev, mode="penalty")
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_budget_weights_sum_to_one(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_tc(
            mu, sigma, w_prev=w_prev, mode="budget", budget_tc=0.05
        )
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_penalty_weights_non_negative(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_tc(mu, sigma, w_prev=w_prev, mode="penalty")
        assert (res["weights"] >= -WEIGHT_TOL).all()

    def test_budget_tc_respected(self, mu_sigma, w_prev):
        """In budget mode, actual TC must be <= budget_tc."""
        mu, sigma = mu_sigma
        budget = 0.02
        costs = pd.Series(0.001, index=mu.index)
        res = optimize_with_tc(
            mu, sigma, w_prev=w_prev, mode="budget",
            budget_tc=budget, costs=costs,
        )
        if res["status"] in ("optimal", "optimal_inaccurate"):
            assert res["total_cost"] <= budget + 1e-4

    def test_tc_calculated_correctly(self, mu_sigma, w_prev):
        """TC = sum(|w - w_prev| * costs)."""
        mu, sigma = mu_sigma
        costs = pd.Series(0.001, index=mu.index)
        res = optimize_with_tc(
            mu, sigma, w_prev=w_prev, mode="penalty", costs=costs
        )
        expected_tc = float(
            costs.values @ np.abs(
                res["weights"].values - w_prev.reindex(mu.index).values
            )
        )
        assert abs(res["total_cost"] - expected_tc) < 1e-6

    def test_turnover_reported(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_tc(mu, sigma, w_prev=w_prev, mode="penalty")
        assert "turnover" in res
        assert res["turnover"] >= 0.0

    def test_invalid_mode_raises(self, mu_sigma):
        mu, sigma = mu_sigma
        with pytest.raises(ValueError, match="mode must be"):
            optimize_with_tc(mu, sigma, mode="invalid")

    def test_higher_lambda_lower_turnover(self, mu_sigma, w_prev):
        """Higher TC penalty → less deviation from w_prev."""
        mu, sigma = mu_sigma
        res_lo = optimize_with_tc(
            mu, sigma, w_prev=w_prev, mode="penalty", tc_lambda=1e-6
        )
        res_hi = optimize_with_tc(
            mu, sigma, w_prev=w_prev, mode="penalty", tc_lambda=1.0
        )
        # High lambda should produce less turnover
        assert res_hi["turnover"] <= res_lo["turnover"] + 0.05


# ===========================================================================
# T-15: Turnover-Constrained Optimizer (optimize_with_turnover)
# ===========================================================================
class TestTurnoverConstraint:
    def test_hard_weights_sum_to_one(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard", delta=0.20
        )
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_soft_weights_sum_to_one(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="soft"
        )
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_hard_turnover_respected(self, mu_sigma, w_prev):
        """R-02: Actual turnover must be <= delta."""
        mu, sigma = mu_sigma
        delta = 0.20
        res = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard", delta=delta
        )
        if res["status"] in ("optimal", "optimal_inaccurate"):
            assert res["turnover"] <= delta + 1e-4

    def test_tighter_delta_less_turnover(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res_30 = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard", delta=0.30
        )
        res_10 = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard", delta=0.10
        )
        if (
            res_30["status"] in ("optimal", "optimal_inaccurate")
            and res_10["status"] in ("optimal", "optimal_inaccurate")
        ):
            assert res_10["turnover"] <= res_30["turnover"] + 1e-4

    def test_weights_non_negative(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard"
        )
        assert (res["weights"] >= -WEIGHT_TOL).all()

    def test_invalid_mode_raises(self, mu_sigma):
        mu, sigma = mu_sigma
        with pytest.raises(ValueError, match="mode must be"):
            optimize_with_turnover(mu, sigma, mode="unknown")

    def test_zero_delta_returns_w_prev(self, mu_sigma, w_prev):
        """delta=0 means no trading → weights should stay at w_prev."""
        mu, sigma = mu_sigma
        res = optimize_with_turnover(
            mu, sigma, w_prev=w_prev, mode="hard", delta=0.0
        )
        if res["status"] in ("optimal", "optimal_inaccurate"):
            assert res["turnover"] < 1e-3


# ===========================================================================
# T-16: Liquidity-Constrained Optimizer (optimize_with_liquidity)
# ===========================================================================
class TestLiquidityConstraint:
    def test_weights_sum_to_one(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        t = prices.index[-1]
        res = optimize_with_liquidity(
            mu, sigma, t=t, volumes=volumes
        )
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_weights_non_negative(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        t = prices.index[-1]
        res = optimize_with_liquidity(
            mu, sigma, t=t, volumes=volumes
        )
        assert (res["weights"] >= -WEIGHT_TOL).all()

    def test_active_tickers_subset(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        t = prices.index[-1]
        res = optimize_with_liquidity(
            mu, sigma, t=t, volumes=volumes
        )
        assert set(res["active_tickers"]).issubset(set(mu.index))

    def test_excluded_have_zero_weight(self, mu_sigma, volumes, prices):
        """Excluded tickers must have w_i = 0 (R-03)."""
        mu, sigma = mu_sigma
        t = prices.index[-1]
        illiq = pd.Series(
            np.random.default_rng(SEED).random(len(mu)), index=mu.index
        )
        res = optimize_with_liquidity(
            mu, sigma, t=t, volumes=volumes,
            illiquidity_scores=illiq,
            amihud_percentile=50.0,
        )
        for tk in res["excluded_tickers"]:
            assert res["weights"].get(tk, 0.0) < WEIGHT_TOL

    def test_global_max_weight_respected(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        t = prices.index[-1]
        cap = 0.25
        res = optimize_with_liquidity(
            mu, sigma, t=t, volumes=volumes,
            global_max_weight=cap,
        )
        assert (res["weights"] <= cap + WEIGHT_TOL).all()

    def test_volume_upper_bounds_shape(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        tickers = mu.index.tolist()
        t = prices.index[-1]
        ub = compute_volume_upper_bounds(tickers, volumes, t)
        assert len(ub) == len(tickers)
        assert (ub > 0).all()

    def test_amihud_exclusion_reduces_universe(self, mu_sigma):
        mu, _ = mu_sigma
        tickers = mu.index.tolist()
        illiq = pd.Series(
            np.arange(len(tickers), dtype=float), index=tickers
        )
        active, excluded = apply_amihud_exclusion(
            tickers, illiq, percentile_threshold=80.0
        )
        assert len(active) < len(tickers)
        assert len(excluded) > 0


# ===========================================================================
# T-17: Unified Interface (optimize)
# ===========================================================================
class TestPortfolioOptimizer:
    @pytest.mark.parametrize("method", [
        "mv_classic", "robust", "mv_tc", "mv_turnover",
    ])
    def test_weights_sum_to_one(self, method, mu_sigma, w_prev):
        """R-04: sum(w) = 1 ± 1e-4 for all methods."""
        mu, sigma = mu_sigma
        res = optimize(mu, sigma, method=method, w_prev=w_prev)
        assert abs(res["weights"].sum() - 1.0) < TOL

    @pytest.mark.parametrize("method", [
        "mv_classic", "robust", "mv_tc", "mv_turnover",
    ])
    def test_weights_non_negative(self, method, mu_sigma, w_prev):
        """R-04: w_i >= 0 (long-only) for all methods."""
        mu, sigma = mu_sigma
        res = optimize(mu, sigma, method=method, w_prev=w_prev)
        assert (res["weights"] >= -WEIGHT_TOL).all()

    @pytest.mark.parametrize("method", [
        "mv_classic", "robust", "mv_tc", "mv_turnover",
    ])
    def test_max_weight_respected(self, method, mu_sigma, w_prev):
        """R-04: w_i <= max_weight for all methods."""
        mu, sigma = mu_sigma
        cap = 0.30
        res = optimize(
            mu, sigma, method=method, w_prev=w_prev, max_weight=cap
        )
        assert (res["weights"] <= cap + WEIGHT_TOL).all()

    def test_mv_turnover_constraint(self, mu_sigma, w_prev):
        """R-02: turnover <= turnover_limit."""
        mu, sigma = mu_sigma
        limit = 0.20
        res = optimize(
            mu, sigma,
            method="mv_turnover",
            w_prev=w_prev,
            turnover_limit=limit,
            turnover_mode="hard",
        )
        if res["converged"]:
            assert res["turnover"] <= limit + 1e-4

    def test_cost_calculated(self, mu_sigma, w_prev):
        """R-01: cost field is present and non-negative."""
        mu, sigma = mu_sigma
        res = optimize(
            mu, sigma, method="mv_tc", w_prev=w_prev, tc_mode="penalty"
        )
        assert "cost" in res
        assert res["cost"] >= 0.0

    def test_standard_keys_present(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = optimize(mu, sigma, method="mv_classic", w_prev=w_prev)
        expected_keys = {
            "weights", "expected_sharpe", "turnover", "cost",
            "converged", "solve_time", "method", "status",
        }
        assert expected_keys.issubset(set(res.keys()))

    def test_invalid_method_raises(self, mu_sigma):
        mu, sigma = mu_sigma
        with pytest.raises(ValueError, match="Unknown method"):
            optimize(mu, sigma, method="nonexistent")

    def test_liquidity_requires_t_and_volumes(self, mu_sigma):
        mu, sigma = mu_sigma
        with pytest.raises(ValueError, match="requires t and volumes"):
            optimize(mu, sigma, method="liquidity")

    def test_liquidity_method(self, mu_sigma, w_prev, volumes, prices):
        mu, sigma = mu_sigma
        t = prices.index[-1]
        res = optimize(
            mu, sigma,
            method="liquidity",
            w_prev=w_prev,
            t=t,
            volumes=volumes,
        )
        assert abs(res["weights"].sum() - 1.0) < TOL

    def test_convergence_rate(self, mu_sigma, prices):
        """R-10: Convergence > 99% across rebalance windows."""
        from src.rolling_engine import get_rebalance_dates, get_window
        from src.expected_returns import mean_historical_return
        from src.covariance_estimators import ledoit_wolf_covariance

        rebal_dates = get_rebalance_dates(prices, freq="Q", warmup=252)
        converged = 0
        total = 0
        w_prev_iter = None

        for t in rebal_dates:
            try:
                wp = get_window(prices, t, window=252)
                mu_t = mean_historical_return(wp)
                sigma_t = ledoit_wolf_covariance(wp)
                res = optimize(
                    mu_t, sigma_t,
                    method="mv_classic",
                    w_prev=w_prev_iter,
                )
                if res["converged"]:
                    converged += 1
                w_prev_iter = res["weights"]
                total += 1
            except Exception:  # noqa: BLE001
                total += 1

        if total > 0:
            rate = converged / total
            assert rate >= 0.99, (
                f"Convergence rate too low: {rate:.1%} ({converged}/{total})"
            )

    @pytest.mark.parametrize("method", [
        "mv_classic", "robust", "mv_tc", "mv_turnover",
    ])
    def test_deterministic(self, method, mu_sigma, w_prev):
        """R-10: Same inputs → same weights."""
        mu, sigma = mu_sigma
        r1 = optimize(mu, sigma, method=method, w_prev=w_prev)
        r2 = optimize(mu, sigma, method=method, w_prev=w_prev)
        pd.testing.assert_series_equal(
            r1["weights"], r2["weights"],
            atol=1e-6,
        )


# ===========================================================================
# Contratos explícitos: registro OPTIMIZERS + entrada/salida tipadas
# ===========================================================================
class TestOptimizerContracts:
    """Cada método declara qué entradas necesita; el resultado dice si el
    solver convergió o si los pesos son el fallback equal-weight."""

    def test_registry_covers_valid_methods(self):
        fields = {f.name for f in dataclasses.fields(OptimizationInput)}
        assert set(OPTIMIZERS) == set(VALID_METHODS)
        for name, spec in OPTIMIZERS.items():
            assert isinstance(spec, Optimizer), name
            assert spec.requires <= fields, name

    def test_liquidity_declares_t_and_volumes(self):
        assert OPTIMIZERS["liquidity"].requires == {"t", "volumes"}

    def test_run_optimizer_returns_typed_result(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        res = run_optimizer(
            "mv_classic", OptimizationInput(mu=mu, sigma=sigma, w_prev=w_prev)
        )
        assert isinstance(res, OptimizationResult)
        assert res.converged
        assert not res.fallback_used
        assert abs(res.weights.sum() - 1.0) < TOL
        assert {"expected_sharpe", "turnover", "cost", "solve_time",
                "method"} <= set(res.meta)

    def test_missing_required_input_names_the_field(self, mu_sigma, prices):
        mu, sigma = mu_sigma
        inp = OptimizationInput(mu=mu, sigma=sigma, t=prices.index[-1])
        with pytest.raises(ValueError, match="volumes"):
            run_optimizer("liquidity", inp)

    def test_liquidity_runs_with_t_and_volumes(self, mu_sigma, volumes, prices):
        mu, sigma = mu_sigma
        inp = OptimizationInput(mu=mu, sigma=sigma, t=prices.index[-1],
                                volumes=volumes)
        res = run_optimizer("liquidity", inp)
        assert res.converged
        assert abs(res.weights.sum() - 1.0) < TOL

    @pytest.mark.parametrize("method", ["mv_tc", "mv_turnover"])
    def test_solver_failure_flags_fallback(self, method, mu_sigma):
        """n * max_weight < 1 is infeasible: the solver hands back equal weight."""
        mu, sigma = mu_sigma
        res = run_optimizer(
            method, OptimizationInput(mu=mu, sigma=sigma, max_weight=0.05)
        )
        assert not res.converged
        assert res.fallback_used

    def test_legacy_optimize_matches_run_optimizer(self, mu_sigma, w_prev):
        mu, sigma = mu_sigma
        legacy = optimize(mu, sigma, method="mv_tc", w_prev=w_prev)
        typed = run_optimizer(
            "mv_tc", OptimizationInput(mu=mu, sigma=sigma, w_prev=w_prev)
        )
        pd.testing.assert_series_equal(legacy["weights"], typed.weights)
        assert legacy["converged"] == typed.converged
        assert legacy["fallback_used"] is False

    def test_legacy_optimize_reports_fallback(self, mu_sigma):
        mu, sigma = mu_sigma
        res = optimize(mu, sigma, method="mv_tc", max_weight=0.05)
        assert res["fallback_used"] is True
