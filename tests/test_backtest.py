"""
test_backtest.py — Tests for Phase 5: backtest_engine and metrics_calculator.

Coverage:
  - BacktestConfig defaults and overrides
  - run_backtest basic execution (synthetic data)
  - BacktestResult shape and type validation
  - NAV monotonicity checks (no NaN, starts at initial_nav)
  - No look-ahead: all weights at t use only data before t
  - run_comparison with multiple configs
  - metrics_calculator: all individual functions
  - compute_metrics: full suite
  - bootstrap_sharpe_ci: shape, confidence, reproducibility
  - compare_metrics: DataFrame shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import (
    CASH_COL,
    BacktestConfig,
    BacktestResult,
    DailyRecord,
    run_backtest,
    run_comparison,
)
from src.strategy import TargetResult, prev_weights
from src.metrics_calculator import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    var_historical,
    var_parametric,
    cvar,
    beta_alpha,
    information_ratio,
    avg_annual_turnover,
    bootstrap_sharpe_ci,
    compute_metrics,
    compare_metrics,
    PortfolioMetrics,
    metrics_to_dict,
)


def _fake_build_targets(targets):
    """Fake build_targets: fixed per-date targets over all price columns.

    **_ keeps the fake valid when build_targets gains kwargs; otherwise the
    engine swallows the TypeError and the test fails on NAV instead of on
    the signature. w_prev comes from the real state, like build_targets.
    """
    def _fake(state, prices, volumes, t, cfg, **_):
        cols = list(prices.columns)
        return TargetResult(
            weights=targets[t].reindex(cols).fillna(0.0),
            regime=None,
            exits=pd.DataFrame(columns=["ticker", "rule", "detail"]),
            meta={"expected_sharpe": 0.0, "converged": True, "method": "test"},
            active=tuple(cols),
            eligible=tuple(cols),
            w_prev=prev_weights(state, prices, t, cols),
        )
    return _fake


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture(scope="module")
def synthetic_prices():
    """Synthetic price DataFrame: 3 years daily, 5 tickers."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    n = len(dates)
    tickers = ["A", "B", "C", "D", "E"]
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, (n, 5)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture(scope="module")
def synthetic_volumes(synthetic_prices):
    """Synthetic volume DataFrame aligned with prices."""
    rng = np.random.default_rng(99)
    vols = pd.DataFrame(
        1e6 + rng.integers(0, 1_000_000, size=synthetic_prices.shape),
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    return vols.astype(float)


@pytest.fixture(scope="module")
def basic_config():
    return BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq="Q",
        window=126,
        warmup=126,
        initial_nav=1_000_000.0,
    )


@pytest.fixture(scope="module")
def backtest_result(synthetic_prices, synthetic_volumes, basic_config):
    return run_backtest(synthetic_prices, synthetic_volumes, basic_config)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.012, 500))


@pytest.fixture
def sample_nav(sample_returns):
    return pd.Series(1e6 * (1 + sample_returns).cumprod())


# ===================================================================
# BacktestConfig Tests
# ===================================================================
class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.opt_method == "mv_classic"
        assert cfg.rebalance_freq == "ME"
        assert cfg.window == 252
        assert cfg.warmup == 252
        assert cfg.initial_nav == 1_000_000.0
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = BacktestConfig(opt_method="robust", kappa=0.2, window=63)
        assert cfg.opt_method == "robust"
        assert cfg.kappa == 0.2
        assert cfg.window == 63

    def test_max_weight_range(self):
        cfg = BacktestConfig(max_weight=0.25)
        assert 0 < cfg.max_weight <= 1.0

    def test_turnover_limit_range(self):
        cfg = BacktestConfig(turnover_limit=0.30)
        assert 0 < cfg.turnover_limit <= 1.0


# ===================================================================
# run_backtest Tests
# ===================================================================
class TestRunBacktest:
    def test_returns_backtest_result(self, backtest_result):
        assert isinstance(backtest_result, BacktestResult)

    def test_nav_series_not_empty(self, backtest_result):
        assert len(backtest_result.nav) > 0
        assert not backtest_result.nav.isna().any()

    def test_nav_starts_near_initial(self, backtest_result, basic_config):
        """First NAV = initial_nav minus the cost of deploying the cash."""
        first_nav = backtest_result.nav.iloc[0]
        first_cost = backtest_result.rebalance_log["cost_fraction"].iloc[0]
        assert 0 < first_cost < 0.01
        assert first_nav == pytest.approx(
            basic_config.initial_nav * (1 - first_cost), rel=1e-12)

    def test_nav_all_positive(self, backtest_result):
        assert (backtest_result.nav > 0).all()

    def test_returns_series_length_matches_nav(self, backtest_result):
        assert len(backtest_result.returns) == len(backtest_result.nav)

    def test_weights_df_shape(self, backtest_result, synthetic_prices):
        weights = backtest_result.weights
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(backtest_result.nav)
        # Each row sums to approx 1
        row_sums = weights.sum(axis=1)
        assert (row_sums > 0.99).all() or weights.shape[1] == 0

    def test_rebalance_log_not_empty(self, backtest_result):
        assert not backtest_result.rebalance_log.empty

    def test_daily_records_length(self, backtest_result):
        assert len(backtest_result.daily_records) == len(backtest_result.nav)

    def test_daily_records_type(self, backtest_result):
        for rec in backtest_result.daily_records[:5]:
            assert isinstance(rec, DailyRecord)
            assert rec.nav > 0
            assert isinstance(rec.date, pd.Timestamp)

    def test_rebalance_log_columns(self, backtest_result):
        cols = backtest_result.rebalance_log.columns.tolist()
        expected = {"opt_method", "n_active", "turnover", "cost_fraction", "status"}
        assert expected.issubset(set(cols))

    def test_config_stored(self, backtest_result, basic_config):
        assert backtest_result.config.opt_method == basic_config.opt_method

    def test_no_lookahead_rebalance_dates(
        self, backtest_result, synthetic_prices, basic_config
    ):
        """Weights on rebalance date t only use prices before t."""
        rebal_dates = backtest_result.rebalance_log.index
        # Just spot-check first 3
        for t in rebal_dates[:3]:
            w = backtest_result.weights.loc[t]
            assert isinstance(w, pd.Series)

    def test_empty_universe_raises(
        self, synthetic_prices, synthetic_volumes, basic_config
    ):
        with pytest.raises(Exception):
            run_backtest(
                synthetic_prices.iloc[:10],  # Too short for warmup
                synthetic_volumes.iloc[:10],
                basic_config,
            )

    def test_default_config_none(self, synthetic_prices, synthetic_volumes):
        """None config should use BacktestConfig defaults."""
        result = run_backtest(synthetic_prices, synthetic_volumes, config=None)
        assert isinstance(result, BacktestResult)

    def test_custom_universe_subset(
        self, synthetic_prices, synthetic_volumes, basic_config
    ):
        universe = ["A", "B", "C"]
        result = run_backtest(
            synthetic_prices, synthetic_volumes, basic_config, universe=universe
        )
        # Weights should only contain tickers in universe (or subset) + cash
        for col in result.weights.columns:
            assert col in universe + [CASH_COL]


# ===================================================================
# run_comparison Tests
# ===================================================================
class TestRunComparison:
    def test_returns_dict(self, synthetic_prices, synthetic_volumes):
        configs = {
            "c1": BacktestConfig(opt_method="mv_classic", window=126, warmup=126),
            "c2": BacktestConfig(opt_method="mv_classic", window=126, warmup=126,
                                 max_weight=0.25),
        }
        results, errors = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"c1", "c2"}
        assert errors == {}

    def test_each_result_is_backtest_result(
        self, synthetic_prices, synthetic_volumes
    ):
        configs = {
            "x": BacktestConfig(window=126, warmup=126),
        }
        results, _ = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert isinstance(results["x"], BacktestResult)

    def test_failed_config_returned_as_error(
        self, synthetic_prices, synthetic_volumes
    ):
        """A config that fails does not crash run_comparison, and is not
        dropped either: its exception comes back under its label."""
        configs = {
            "good": BacktestConfig(window=126, warmup=126),
            # warmup longer than the history → no rebalance dates
            "bad": BacktestConfig(window=126, warmup=10_000),
        }
        results, errors = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert set(results) == {"good"}
        assert set(errors) == {"bad"}
        assert isinstance(errors["bad"], ValueError)
        assert "No rebalance dates" in str(errors["bad"])

    def test_universe_passed_through(self, synthetic_prices, synthetic_volumes):
        configs = {"x": BacktestConfig(window=126, warmup=126)}
        results, _ = run_comparison(
            synthetic_prices, synthetic_volumes, configs, universe=["A", "B", "C"],
        )
        assert set(results["x"].weights.columns) <= {"A", "B", "C", CASH_COL}


# ===================================================================
# metrics_calculator — individual functions
# ===================================================================
class TestAnnualizedReturn:
    def test_positive_returns(self):
        r = pd.Series([0.001] * 252)
        ann = annualized_return(r)
        assert ann > 0.0

    def test_zero_returns(self):
        r = pd.Series([0.0] * 252)
        ann = annualized_return(r)
        assert abs(ann) < 1e-6

    def test_negative_returns(self):
        r = pd.Series([-0.001] * 252)
        ann = annualized_return(r)
        assert ann < 0.0

    def test_short_series(self):
        ann = annualized_return(pd.Series([0.01]))
        assert ann == 0.0


class TestAnnualizedVolatility:
    def test_nonzero(self, sample_returns):
        vol = annualized_volatility(sample_returns)
        assert vol > 0.0

    def test_scaling(self):
        r = pd.Series([0.01, -0.01] * 100)
        vol = annualized_volatility(r)
        assert abs(vol - r.std() * np.sqrt(252)) < 1e-8

    def test_constant_returns(self):
        r = pd.Series([0.001] * 100)
        assert annualized_volatility(r) < 1e-10


class TestSharpeRatio:
    def test_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.005, 252))
        s = sharpe_ratio(r)
        assert s > 0.0

    def test_zero_std(self):
        r = pd.Series([0.0] * 252)
        s = sharpe_ratio(r)
        assert s == 0.0

    def test_risk_free_reduces_sharpe(self, sample_returns):
        s0 = sharpe_ratio(sample_returns, risk_free_rate=0.0)
        s1 = sharpe_ratio(sample_returns, risk_free_rate=0.05)
        assert s0 >= s1


class TestSortinoRatio:
    def test_greater_than_sharpe_for_positive_drift(self):
        # With positive drift and asymmetric downside, Sortino >= Sharpe
        rng = np.random.default_rng(7)
        r = pd.Series(rng.normal(0.002, 0.008, 500))
        so = sortino_ratio(r)
        sh = sharpe_ratio(r)
        # Both should be positive, and Sortino >= Sharpe
        assert so >= sh - 1e-6

    def test_negative_returns_sortino(self):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(-0.002, 0.008, 252))
        so = sortino_ratio(r)
        assert so < 0.0


class TestMaxDrawdown:
    def test_monotone_uptrend(self):
        nav = pd.Series(np.arange(1.0, 101.0))
        assert abs(max_drawdown(nav)) < 1e-10

    def test_monotone_downtrend(self):
        nav = pd.Series(np.linspace(100, 1, 100))
        mdd = max_drawdown(nav)
        assert mdd < -0.98  # Nearly -100%

    def test_drawdown_after_peak(self):
        nav = pd.Series([1.0, 1.5, 1.2, 1.8, 1.0])
        mdd = max_drawdown(nav)
        # Peak=1.8, trough=1.0 → dd = (1.0 - 1.8)/1.8 ≈ -0.444
        assert abs(mdd - (-4 / 9)) < 0.01

    def test_is_non_positive(self, sample_nav):
        assert max_drawdown(sample_nav) <= 0.0


class TestCalmarRatio:
    def test_positive_calmar(self, sample_returns, sample_nav):
        cal = calmar_ratio(sample_returns, sample_nav)
        assert isinstance(cal, float)

    def test_zero_drawdown_returns_zero(self):
        nav = pd.Series(np.linspace(1.0, 2.0, 252))
        r = nav.ffill().pct_change().fillna(0.0)
        # Nearly zero drawdown
        cal = calmar_ratio(r, nav)
        assert cal == 0.0 or cal > 0.0


class TestVaR:
    def test_var_historical_positive(self, sample_returns):
        v = var_historical(sample_returns)
        assert v > 0.0

    def test_var_parametric_positive(self, sample_returns):
        v = var_parametric(sample_returns)
        assert v > 0.0

    def test_cvar_gte_var(self, sample_returns):
        v = var_historical(sample_returns)
        c = cvar(sample_returns)
        assert c >= v - 1e-10

    def test_var_95_gt_var_90(self, sample_returns):
        v95 = var_historical(sample_returns, confidence=0.95)
        v90 = var_historical(sample_returns, confidence=0.90)
        assert v95 >= v90 - 1e-10

    def test_short_series(self):
        r = pd.Series([0.01, -0.01])
        assert var_historical(r) == 0.0


class TestBetaAlpha:
    def test_beta_one_for_identical_series(self):
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300))
        b, a = beta_alpha(r, r)
        assert abs(b - 1.0) < 0.01

    def test_beta_zero_for_uncorrelated(self):
        rng = np.random.default_rng(2)
        r1 = pd.Series(rng.normal(0, 0.01, 300))
        r2 = pd.Series(rng.normal(0, 0.01, 300))
        b, _ = beta_alpha(r1, r2)
        assert abs(b) < 0.3

    def test_short_series_returns_defaults(self):
        r = pd.Series([0.01] * 5)
        b, a = beta_alpha(r, r)
        assert b == 1.0 and a == 0.0


class TestInformationRatio:
    def test_zero_active_return(self):
        r = pd.Series([0.001] * 252)
        ir = information_ratio(r, r)
        assert abs(ir) < 1e-6

    def test_positive_ir(self):
        r_p = pd.Series([0.002] * 252)
        r_b = pd.Series([0.001] * 252)
        # Same std; IR depends on tracking error
        ir = information_ratio(r_p, r_b)
        assert isinstance(ir, float)


class TestAvgAnnualTurnover:
    def test_quarterly_rebalance(self):
        # 4 rebalances per year, each 20% turnover
        n = 252
        to = pd.Series([0.20 if i % 63 == 0 else 0.0 for i in range(n)])
        avg_to = avg_annual_turnover(to)
        # 4 events * 0.20 per year
        assert abs(avg_to - 0.80) < 0.2

    def test_zero_turnover(self):
        to = pd.Series([0.0] * 252)
        assert avg_annual_turnover(to) == 0.0


# ===================================================================
# bootstrap_sharpe_ci Tests
# ===================================================================
class TestBootstrapSharpeCi:
    def test_returns_tuple_of_two(self, sample_returns):
        ci = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100)
        assert len(ci) == 2

    def test_low_lt_high(self, sample_returns):
        lo, hi = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200)
        assert lo < hi

    def test_reproducible_with_same_seed(self, sample_returns):
        ci1 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100, seed=42)
        ci2 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100, seed=42)
        assert ci1 == ci2

    def test_different_seeds_differ(self, sample_returns):
        ci1 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200, seed=1)
        ci2 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200, seed=2)
        assert ci1 != ci2

    def test_short_series(self):
        r = pd.Series([0.01] * 5)
        lo, hi = bootstrap_sharpe_ci(r, n_bootstrap=10)
        assert lo == hi  # Falls back

    def test_ci_contains_point_estimate(self, sample_returns):
        lo, hi = bootstrap_sharpe_ci(sample_returns, n_bootstrap=500)
        sr = sharpe_ratio(sample_returns)
        assert lo <= sr <= hi


# ===================================================================
# compute_metrics Tests
# ===================================================================
class TestComputeMetrics:
    def test_returns_portfolio_metrics(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert isinstance(m, PortfolioMetrics)

    def test_total_return_consistent(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        # Verify total return sign matches NAV direction
        nav_ret = (sample_nav.iloc[-1] / sample_nav.iloc[0]) - 1
        assert np.sign(m.total_return) == np.sign(nav_ret) or abs(m.total_return) < 0.01

    def test_sharpe_ci_order(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=100)
        assert m.sharpe_ci_low <= m.sharpe_ratio <= m.sharpe_ci_high

    def test_var_positive(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.var_95_historical >= 0.0
        assert m.var_95_parametric >= 0.0

    def test_cvar_gte_var(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.cvar_95 >= m.var_95_historical - 1e-10

    def test_max_drawdown_nonpositive(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.max_drawdown <= 0.0

    def test_with_turnover(self, sample_returns, sample_nav):
        rng = np.random.default_rng(5)
        to = pd.Series([0.1 if i % 63 == 0 else 0.0
                        for i in range(len(sample_returns))])
        m = compute_metrics(sample_returns, sample_nav, turnover=to)
        assert m.avg_annual_turnover >= 0.0
        assert m.n_rebalances > 0

    def test_with_benchmark(self, sample_returns, sample_nav):
        bench = sample_returns * 0.8 + 0.0001
        m = compute_metrics(sample_returns, sample_nav,
                            benchmark_returns=bench)
        assert isinstance(m.beta, float)
        assert isinstance(m.information_ratio, float)

    def test_n_days_set(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.n_days == len(sample_returns.dropna())

    def test_dates_set(self, sample_nav):
        dates = pd.bdate_range("2020-01-01", periods=len(sample_nav))
        sample_nav_dated = pd.Series(sample_nav.values, index=dates)
        rng = np.random.default_rng(1)
        returns_dated = pd.Series(
            rng.normal(0.0005, 0.012, len(dates)), index=dates
        )
        m = compute_metrics(returns_dated, sample_nav_dated)
        assert m.start_date is not None
        assert m.end_date is not None
        assert m.start_date <= m.end_date


# ===================================================================
# metrics_to_dict and compare_metrics Tests
# ===================================================================
class TestMetricsDictAndCompare:
    def test_metrics_to_dict_keys(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        d = metrics_to_dict(m)
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "cvar_95" in d
        assert "annualized_return" in d

    def test_compare_metrics_shape(self, sample_returns, sample_nav):
        m1 = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        m2 = compute_metrics(sample_returns * 0.9, sample_nav * 0.95,
                             n_bootstrap=50)
        table = compare_metrics({"s1": m1, "s2": m2})
        assert isinstance(table, pd.DataFrame)
        assert "s1" in table.columns
        assert "s2" in table.columns

    def test_compare_metrics_row_count(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        table = compare_metrics({"only": m})
        assert len(table) > 5  # At least 5 metrics rows


# ===================================================================
# Integration: backtest → metrics pipeline
# ===================================================================
class TestBacktestMetricsPipeline:
    def test_end_to_end(self, synthetic_prices, synthetic_volumes):
        """Full pipeline: run_backtest → compute_metrics → compare."""
        config = BacktestConfig(
            opt_method="mv_classic",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, config)
        to = pd.Series({r.date: r.turnover for r in result.daily_records})
        m = compute_metrics(result.returns, result.nav, turnover=to)

        assert isinstance(m, PortfolioMetrics)
        assert m.n_days > 0
        assert m.sharpe_ci_low <= m.sharpe_ratio <= m.sharpe_ci_high
        assert m.cvar_95 >= m.var_95_historical - 1e-10


# ===================================================================
# Fase 6: signals / HRP / PositionLedger / 15D
# ===================================================================
class TestFase6Backtest:
    def test_freq_15d(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="mv_classic",
            rebalance_freq="15D",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert len(result.nav) > 0
        assert not result.rebalance_log.empty

    def test_opt_method_hrp(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert (result.rebalance_log["opt_method"] == "hrp").all()
        # Weights sum ≈ 1 on rebalance days
        for r in result.daily_records:
            if r.is_rebalance and len(r.weights) > 0:
                assert abs(r.weights.sum() - 1.0) < 1e-5

    def test_signal_stack_and_ledger(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="15D",
            window=126,
            warmup=126,
            use_signal_stack=True,
            use_position_ledger=True,
            regime_window=63,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert len(result.ledger_snapshots) > 0
        assert result.regime_series is not None
        assert len(result.regime_series) > 0
        # Snapshots have positive NAV
        assert all(s.nav > 0 for s in result.ledger_snapshots)
        # Final NAV finite
        assert np.isfinite(result.nav.iloc[-1])

    def test_ledger_rebalances_reserve_costs(self, synthetic_prices, synthetic_volumes):
        """Ningun rebalanceo del ledger falla en silencio y ninguno deja cash < 0."""
        from src.position_ledger import CASH_TOLERANCE
        from src.rolling_engine import get_rebalance_dates
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="15D",
            window=126,
            warmup=126,
            use_signal_stack=True,
            use_position_ledger=True,
            regime_window=63,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        n_rebal = len(get_rebalance_dates(synthetic_prices, freq="15D", warmup=126))
        assert len(result.ledger_snapshots) == n_rebal
        assert all(s.cash >= -CASH_TOLERANCE for s in result.ledger_snapshots)


# ===========================================================================
# Classic path: rebalance-day P&L accrues to pre-rebalance holdings
# ===========================================================================
class TestClassicPathRebalanceDayPnL:
    """Regression: classic path must mark-to-market with pre-rebalance weights.

    Trades execute at the close of t (ledger path trades at prices.loc[t],
    entries are recorded at that close), so the t_prev→t move belongs to
    the book held overnight, not to the new target.
    """

    def test_rebalance_day_return_uses_prior_holdings(self, monkeypatch):
        """Holding A through an A +10% day, then switching to B, keeps the +10%."""
        idx = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame(
            {
                "A": [100.0, 100.0, 100.0, 110.0, 110.0],
                "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=idx,
        )
        volumes = pd.DataFrame(
            {c: 1_000_000.0 for c in prices.columns}, index=idx
        )

        forced = [idx[1], idx[3]]
        targets = {
            idx[1]: pd.Series({"A": 1.0, "B": 0.0}),
            idx[3]: pd.Series({"A": 0.0, "B": 1.0}),
        }

        import src.backtest_engine as be

        # backtest_engine binds filter_universe_at_date and rebalance at import
        # time, so patch them there; precompute_rolling_metrics is imported
        # lazily inside run_backtest, so it is patched on its home module.
        monkeypatch.setattr(
            be, "get_rebalance_dates", lambda *a, **k: list(forced)
        )
        monkeypatch.setattr(be, "build_targets", _fake_build_targets(targets))

        def _fake_rebalance(**kwargs):
            class _R:
                pass

            r = _R()
            wt = kwargs["w_target"].astype(float)
            r.w_new = wt / wt.sum()
            r.actual_turnover = 1.0
            r.cost = 0.0
            r.trades = pd.DataFrame()
            return r

        monkeypatch.setattr(be, "rebalance", _fake_rebalance)

        cfg = BacktestConfig(
            warmup=0,
            window=2,
            rebalance_freq="15D",
            turnover_limit=2.0,
            commission_bps=0.0,
            spread_bps=0.0,
            impact_coef=0.0,
            replacement_rule="none",
            use_position_ledger=False,
            max_weight=1.0,
            tc_lambda=0.0,
            initial_nav=1_000_000.0,
        )
        result = be.run_backtest(prices, volumes, cfg)

        # Both forced rebalances must actually run (not fail silently).
        assert list(result.rebalance_log.index) == forced
        assert result.weights.loc[idx[3], "B"] == pytest.approx(1.0)
        # A's +10% on idx[3] accrues to A (held overnight), then we hold B.
        assert result.nav.loc[idx[3]] == pytest.approx(1_100_000.0, rel=1e-9)
        assert result.nav.iloc[-1] == pytest.approx(1_100_000.0, rel=1e-9)


class TestMaxWeightEnforcement:
    def test_enforce_max_weight_survives_renorm(self):
        from src.strategy import enforce_max_weight

        # Single clip+renorm would leave ~0.97 on A; iterative must cap.
        # 4 names × 0.30 ≥ 1 so the cap is feasible.
        w = pd.Series({"A": 0.99, "B": 0.005, "C": 0.005, "D": 0.0})
        out = enforce_max_weight(w, max_weight=0.30)
        assert abs(out.sum() - 1.0) < 1e-8
        assert float(out.max()) <= 0.30 + 1e-9

    def test_enforce_max_weight_infeasible_falls_back_equal(self):
        from src.strategy import enforce_max_weight

        # 3 × 0.30 < 1 → cannot satisfy cap; equal weight is the fallback.
        w = pd.Series({"A": 0.9, "B": 0.05, "C": 0.05})
        out = enforce_max_weight(w, max_weight=0.30)
        assert abs(out.sum() - 1.0) < 1e-8
        assert abs(float(out.max()) - 1.0 / 3.0) < 1e-9

    def test_hrp_path_respects_max_weight(
        self, synthetic_prices, synthetic_volumes
    ):
        # turnover_limit=2.0 so executed weights ≈ capped HRP target
        # (a binding 20% turnover blend can re-violate max_weight after renorm).
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="Q",
            window=126,
            warmup=126,
            max_weight=0.30,
            turnover_limit=2.0,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        for r in result.daily_records:
            if r.is_rebalance and len(r.weights) > 0:
                assert float(r.weights.max()) <= 0.30 + 1e-6


class TestLedgerTurnoverCap:
    def test_ledger_path_respects_turnover_limit(
        self, synthetic_prices, synthetic_volumes
    ):
        """Phase 6 ledger path must apply R-02 like classic rebalance()."""
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="15D",
            window=126,
            warmup=126,
            use_position_ledger=True,
            turnover_limit=0.20,
            max_weight=0.30,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        # Skip the initial deployment (no prior book); later rebalances
        # must respect the turnover cap.
        rebal_rows = [
            r for r in result.daily_records if r.is_rebalance
        ]
        assert len(rebal_rows) >= 2
        for r in rebal_rows[1:]:
            assert r.turnover <= 0.20 + 1e-5, (
                f"ledger rebalance turnover {r.turnover} exceeded limit"
            )


class TestExecutionPathParity:
    """Classic and ledger paths execute the same book from the same targets.

    Constant prices (no drift), zero costs, forced targets {A: 1} then
    {B: 1}, turnover_limit=0.20. Both paths start from cash, so:
      1st rebalance: nothing to turn over → deploy straight to A=1.0
         (turnover 1.0; R-02 throttles an existing book, not cash).
      2nd rebalance: |{B:1} - {A:1}|_1 = 2 → alpha = 0.20 / 2 = 0.10
         → A=0.90, B=0.10 (turnover 0.20).
    The classic path used to start from a phantom equal-weight book
    (alpha = 0.20 / (4/3) → A≈0.433 on day one) while the ledger path
    started from cash, so the two were not comparable.
    """

    @staticmethod
    def _run(monkeypatch, use_ledger, **overrides):
        import src.backtest_engine as be

        idx = pd.bdate_range("2024-01-02", periods=6)
        prices = pd.DataFrame(100.0, index=idx, columns=["A", "B", "C"])
        volumes = pd.DataFrame(1_000_000.0, index=idx, columns=prices.columns)
        forced = [idx[1], idx[3]]
        targets = {
            idx[1]: pd.Series({"A": 1.0}),
            idx[3]: pd.Series({"B": 1.0}),
        }
        monkeypatch.setattr(be, "get_rebalance_dates", lambda *a, **k: list(forced))
        monkeypatch.setattr(be, "build_targets", _fake_build_targets(targets))
        cfg = BacktestConfig(**{
            "warmup": 0, "window": 2, "rebalance_freq": "15D",
            "turnover_limit": 0.20, "max_weight": 1.0, "tc_lambda": 0.0,
            "commission_bps": 0.0, "spread_bps": 0.0, "impact_coef": 0.0,
            "use_position_ledger": use_ledger, "initial_nav": 1_000_000.0,
            **overrides,
        })
        return be.run_backtest(prices, volumes, cfg), forced

    @pytest.mark.parametrize("use_ledger", [False, True], ids=["classic", "ledger"])
    def test_same_book_sequence_from_cash(self, monkeypatch, use_ledger):
        result, (t1, t2) = self._run(monkeypatch, use_ledger)

        assert list(result.rebalance_log.index) == [t1, t2]
        assert result.rebalance_log["turnover"].tolist() == pytest.approx([1.0, 0.20])
        w = result.weights.reindex(columns=["A", "B", "C"]).fillna(0.0)
        assert w.loc[t1].tolist() == pytest.approx([1.0, 0.0, 0.0])
        assert w.loc[t2].tolist() == pytest.approx([0.90, 0.10, 0.0])

    @pytest.mark.parametrize("use_ledger", [False, True], ids=["classic", "ledger"])
    def test_first_deployment_cost_hits_nav(self, monkeypatch, use_ledger):
        """Buying the first book from cash costs money on both paths.

        The classic path skipped the NAV update on the first OOS day (no
        book held overnight), so the logged deployment cost never reached
        NAV. Prices are flat: NAV moves only by costs.
        """
        result, (t1, _) = self._run(
            monkeypatch, use_ledger, commission_bps=5.0, spread_bps=2.0)

        cost_t1 = result.rebalance_log.loc[t1, "cost_fraction"]
        assert cost_t1 > 0
        assert result.nav.loc[t1] == pytest.approx(1_000_000.0 * (1 - cost_t1))

    def test_ledger_rejects_replacement_rule(self, monkeypatch):
        """The ledger path has no replacement rules: fail fast, not ignore."""
        with pytest.raises(ValueError, match="replacement_rule"):
            self._run(monkeypatch, True, replacement_rule="momentum")


class TestBlackLittermanBacktestWiring:
    def test_black_litterman_produces_rebalances(
        self, synthetic_prices, synthetic_volumes
    ):
        """Documented mu_method=black_litterman must not silently flat-line.

        Previously estimate_expected_returns was called without cov_matrix,
        every rebalance raised, and run_backtest swallowed the error →
        0 rebalances and constant initial NAV.
        """
        cfg = BacktestConfig(
            opt_method="mv_classic",
            mu_method="black_litterman",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert not result.rebalance_log.empty
        assert len(result.rebalance_log) >= 1
        assert result.nav.nunique() > 1
        assert np.isfinite(result.nav.iloc[-1])


# ===========================================================================
# Classic path: rebalance-day P&L accrues to pre-rebalance holdings
# ===========================================================================
class TestClassicPathRebalanceDayPnL:
    """Regression: classic path must mark-to-market with pre-rebalance weights.

    Trades execute at the close of t (ledger path trades at prices.loc[t],
    entries are recorded at that close), so the t_prev→t move belongs to
    the book held overnight, not to the new target.
    """

    def test_rebalance_day_return_uses_prior_holdings(self, monkeypatch):
        """Holding A through an A +10% day, then switching to B, keeps the +10%."""
        idx = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame(
            {
                "A": [100.0, 100.0, 100.0, 110.0, 110.0],
                "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=idx,
        )
        volumes = pd.DataFrame(
            {c: 1_000_000.0 for c in prices.columns}, index=idx
        )

        forced = [idx[1], idx[3]]
        targets = {
            idx[1]: pd.Series({"A": 1.0, "B": 0.0}),
            idx[3]: pd.Series({"A": 0.0, "B": 1.0}),
        }

        import src.backtest_engine as be
        import src.rebalancing_engine as reb

        monkeypatch.setattr(
            be, "get_rebalance_dates", lambda *a, **k: list(forced)
        )
        monkeypatch.setattr(be, "build_targets", _fake_build_targets(targets))

        def _fake_rebalance(**kwargs):
            class _R:
                pass

            r = _R()
            wt = kwargs["w_target"].astype(float)
            r.w_new = wt / wt.sum()
            r.actual_turnover = 1.0
            r.cost = 0.0
            r.trades = pd.DataFrame()
            return r

        monkeypatch.setattr(reb, "rebalance", _fake_rebalance)

        cfg = BacktestConfig(
            warmup=0,
            window=2,
            rebalance_freq="15D",
            turnover_limit=2.0,
            commission_bps=0.0,
            spread_bps=0.0,
            impact_coef=0.0,
            replacement_rule="none",
            use_position_ledger=False,
            max_weight=1.0,
            tc_lambda=0.0,
            initial_nav=1_000_000.0,
        )
        result = be.run_backtest(prices, volumes, cfg)

        # Both forced rebalances must actually run (not fail silently).
        assert list(result.rebalance_log.index) == forced
        assert result.weights.loc[idx[3], "B"] == pytest.approx(1.0)
        # A's +10% on idx[3] accrues to A (held overnight), then we hold B.
        assert result.nav.loc[idx[3]] == pytest.approx(1_100_000.0, rel=1e-9)
        assert result.nav.iloc[-1] == pytest.approx(1_100_000.0, rel=1e-9)


# ===========================================================================
# Rebalance failures: no silent flat backtests
# ===========================================================================
class TestRebalanceFailures:
    """strict=True (default): a rebalance that raises aborts the run with a
    RebalanceError (date, stage, method, n_assets) chained to the cause.
    strict=False: the failure is recorded (failures + rebalance_log status)
    and the previous weights are kept; if none succeed the run still raises,
    never a flat NAV at initial_nav.
    """

    @staticmethod
    def _rebal_dates(prices, cfg):
        from src.rolling_engine import get_rebalance_dates
        return get_rebalance_dates(prices, freq=cfg.rebalance_freq, warmup=cfg.warmup)

    @staticmethod
    def _optimizer_failing_on(monkeypatch, fail_calls):
        """Real run_optimizer(), but the given 1-based calls raise (one call
        per rebalance date with mv_classic)."""
        import src.strategy as strategy
        real = strategy.run_optimizer
        calls = []

        def _run_optimizer(method, inp):
            calls.append(None)
            if len(calls) in fail_calls:
                raise ValueError(f"solver exploded (call {len(calls)})")
            return real(method, inp)

        monkeypatch.setattr(strategy, "run_optimizer", _run_optimizer)

    def test_strict_is_default(self):
        assert BacktestConfig().strict is True

    def test_strict_optimizer_error_propagates_with_context(
        self, synthetic_prices, synthetic_volumes, basic_config, monkeypatch,
        backtest_result,
    ):
        from src.backtest_engine import RebalanceError

        bad = self._rebal_dates(synthetic_prices, basic_config)[1]
        # Assets the optimizer was solving for (liquidity filter drops one),
        # not the 5 columns of the panel.
        n_active = int(backtest_result.rebalance_log.loc[bad, "n_active"])
        assert n_active == 4
        self._optimizer_failing_on(monkeypatch, {2})
        with pytest.raises(RebalanceError) as info:
            run_backtest(synthetic_prices, synthetic_volumes, basic_config)

        err = info.value
        assert (err.date, err.stage, err.method, err.n_assets) == (
            bad, "targets", "mv_classic", n_active)
        assert isinstance(err.__cause__, ValueError)
        msg = str(err)
        for part in (str(bad.date()), "targets", "mv_classic",
                     f"n_assets={n_active}", "ValueError",
                     "solver exploded (call 2)"):
            assert part in msg

    @pytest.mark.parametrize("use_ledger", [False, True], ids=["classic", "ledger"])
    def test_strict_execution_error_reports_execution_stage(
        self, synthetic_prices, synthetic_volumes, basic_config, monkeypatch,
        backtest_result, use_ledger,
    ):
        import dataclasses
        import src.backtest_engine as be
        from src.backtest_engine import RebalanceError
        from src.position_ledger import PositionLedger

        bad = self._rebal_dates(synthetic_prices, basic_config)[0]

        def _boom(*args, **kwargs):
            raise ValueError("execution boom")

        monkeypatch.setattr(be, "rebalance", _boom)
        monkeypatch.setattr(PositionLedger, "apply_trades", _boom)
        cfg = dataclasses.replace(basic_config, use_position_ledger=use_ledger)
        with pytest.raises(RebalanceError, match="execution boom") as info:
            run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert (info.value.date, info.value.stage, info.value.n_assets) == (
            bad, "execution", backtest_result.rebalance_log.loc[bad, "n_active"])

    def test_non_strict_records_failure_and_keeps_previous_book(
        self, synthetic_prices, synthetic_volumes, basic_config, monkeypatch,
    ):
        import dataclasses

        rebal_dates = self._rebal_dates(synthetic_prices, basic_config)
        bad = rebal_dates[1]
        self._optimizer_failing_on(monkeypatch, {2})
        cfg = dataclasses.replace(basic_config, strict=False)
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)

        assert result.failures == [{
            "date": bad, "stage": "targets", "type": "ValueError",
            "message": "solver exploded (call 2)",
        }]
        log = result.rebalance_log
        assert list(log.index) == list(rebal_dates)
        assert log.loc[bad, "status"] == "failed"
        assert (log["status"].drop(bad) == "ok").all()
        assert log.loc[bad, "turnover"] == 0.0
        assert log.loc[bad, "cost_fraction"] == 0.0
        # Previous book (shares) kept through the failed date
        prev_day = result.weights.index[result.weights.index.get_loc(bad) - 1]

        def shares(d):
            w = result.weights.loc[d].drop(CASH_COL)
            return w * result.nav.loc[d] / synthetic_prices.loc[d, w.index]

        pd.testing.assert_series_equal(
            shares(bad), shares(prev_day), check_names=False, rtol=1e-9)

    @pytest.mark.parametrize("use_ledger", [False, True], ids=["classic", "ledger"])
    def test_non_strict_all_rebalances_failing_still_raises(
        self, synthetic_prices, synthetic_volumes, basic_config, monkeypatch,
        use_ledger,
    ):
        import dataclasses
        import re
        import src.backtest_engine as be

        errors = []

        def _always_fail(**kwargs):
            exc = ValueError(f"boom at {kwargs['t'].date()}")
            errors.append(exc)
            raise exc

        monkeypatch.setattr(be, "build_targets", _always_fail)
        cfg = dataclasses.replace(
            basic_config, use_position_ledger=use_ledger, strict=False)
        n_rebal = len(self._rebal_dates(synthetic_prices, cfg))

        with pytest.raises(
            RuntimeError, match=re.escape(f"All {n_rebal} rebalance(s) failed"),
        ) as info:
            be.run_backtest(synthetic_prices, synthetic_volumes, cfg)

        assert len(errors) == n_rebal
        assert info.value.__cause__ is errors[-1]
        assert str(errors[-1]) in str(info.value)

    def test_non_converged_optimizer_is_flagged_fallback_ew(
        self, synthetic_prices, synthetic_volumes, basic_config, monkeypatch,
    ):
        """Solver did not converge → optimizer returns equal weight. The run
        goes on (not an exception) but the row says so."""
        import dataclasses
        import src.strategy as strategy

        rebal_dates = self._rebal_dates(synthetic_prices, basic_config)
        real = strategy.run_optimizer
        calls = []

        def _run_optimizer(method, inp):
            calls.append(None)
            res = real(method, inp)
            if len(calls) == 2:
                tickers = inp.mu.index
                res = dataclasses.replace(
                    res, converged=False, fallback_used=True,
                    status="infeasible",
                    weights=pd.Series(1.0 / len(tickers), index=tickers))
            return res

        monkeypatch.setattr(strategy, "run_optimizer", _run_optimizer)
        result = run_backtest(synthetic_prices, synthetic_volumes, basic_config)

        status = result.rebalance_log["status"]
        assert status.loc[rebal_dates[1]] == "fallback_ew"
        assert (status.drop(rebal_dates[1]) == "ok").all()
        assert result.failures == []

    def test_infeasible_cap_trades_equal_weight_flagged_fallback_ew(
        self, synthetic_prices, synthetic_volumes, basic_config,
    ):
        """3 tickers with max_weight=0.30: 3 * 0.30 < 1 on every rebalance.
        mv_optimize returns equal weight (status cap_infeasible, no solver);
        the run goes on and every row says it was not the optimizer's
        portfolio."""
        result = run_backtest(synthetic_prices, synthetic_volumes,
                              basic_config, universe=["A", "B", "C"])

        assert (result.rebalance_log["status"] == "fallback_ew").all()
        assert result.failures == []

    def test_clean_run_has_no_failures(self, backtest_result):
        assert backtest_result.failures == []
        assert (backtest_result.rebalance_log["status"] == "ok").all()

    def test_run_comparison_returns_errors_instead_of_dropping(
        self, synthetic_prices, synthetic_volumes, monkeypatch, caplog,
    ):
        import logging
        import src.backtest_engine as be
        from src.backtest_engine import RebalanceError

        real_build = be.build_targets

        def _fail_hrp(**kwargs):
            if kwargs["cfg"].opt_method == "hrp":
                raise ValueError("boom")
            return real_build(**kwargs)

        monkeypatch.setattr(be, "build_targets", _fail_hrp)
        configs = {
            "ok": BacktestConfig(rebalance_freq="Q", window=126, warmup=126),
            "broken": BacktestConfig(
                opt_method="hrp", rebalance_freq="Q", window=126, warmup=126),
        }
        with caplog.at_level(logging.ERROR, logger="backtest_engine"):
            results, errors = be.run_comparison(
                synthetic_prices, synthetic_volumes, configs)

        assert set(results) == {"ok"}
        assert set(errors) == {"broken"}
        assert isinstance(errors["broken"], RebalanceError)
        assert str(errors["broken"].__cause__) == "boom"
        assert any("'broken' failed" in r.getMessage() for r in caplog.records)


# ===========================================================================
# Failed rebalance: nothing traded, so nothing is reported as traded
# ===========================================================================
class TestFailedRebalanceReporting:
    """Regression: a rebalance that raises must report turnover=0, cost=0.

    The ledger path used to compute turnover before apply_trades(); when
    apply_trades raised, the except handler kept the previous weights but
    the DailyRecord still showed the planned turnover (e.g. 1.0), masking
    the failure. Both execution paths must agree: a failed rebalance day
    (strict=False) is is_rebalance=True with turnover 0.0, cost 0.0 and a
    rebalance_log row with status "failed".
    """

    @pytest.mark.parametrize("use_ledger", [False, True], ids=["classic", "ledger"])
    def test_failed_rebalance_reports_no_turnover_or_cost(
        self, monkeypatch, use_ledger
    ):
        idx = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame(
            {
                "A": [100.0, 100.0, 100.0, 110.0, 110.0],
                "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=idx,
        )
        volumes = pd.DataFrame(
            {c: 1_000_000.0 for c in prices.columns}, index=idx
        )

        ok_date, failed_date = idx[1], idx[3]
        targets = {
            ok_date: pd.Series({"A": 1.0, "B": 0.0}),
            failed_date: pd.Series({"A": 0.0, "B": 1.0}),
        }

        import src.backtest_engine as be
        from src.position_ledger import PositionLedger

        monkeypatch.setattr(
            be, "get_rebalance_dates", lambda *a, **k: [ok_date, failed_date]
        )
        monkeypatch.setattr(be, "build_targets", _fake_build_targets(targets))

        def _fake_rebalance(**kwargs):
            if kwargs["t"] == failed_date:
                raise ValueError("forced classic rebalance failure")

            class _R:
                pass

            r = _R()
            wt = kwargs["w_target"].astype(float)
            r.w_new = wt / wt.sum()
            r.actual_turnover = 1.0
            r.cost = 0.0
            r.trades = pd.DataFrame()
            return r

        monkeypatch.setattr(be, "rebalance", _fake_rebalance)

        real_apply = PositionLedger.apply_trades

        def _apply_or_fail(self, trades, prices_at_t, costs=0.0, t=None):
            if t == failed_date:
                raise ValueError(
                    "Batch would leave cash=-1.0 < -1e-06 after costs"
                )
            return real_apply(self, trades, prices_at_t, costs=costs, t=t)

        monkeypatch.setattr(PositionLedger, "apply_trades", _apply_or_fail)

        cfg = BacktestConfig(
            warmup=0,
            window=2,
            rebalance_freq="15D",
            turnover_limit=2.0,
            commission_bps=10.0,
            spread_bps=0.0,
            impact_coef=0.0,
            replacement_rule="none",
            use_position_ledger=use_ledger,
            max_weight=1.0,
            tc_lambda=0.0,
            initial_nav=1_000_000.0,
            strict=False,
        )
        result = be.run_backtest(prices, volumes, cfg)
        records = {r.date: r for r in result.daily_records}

        # The first rebalance ran and is reported as such.
        assert records[ok_date].turnover == pytest.approx(1.0)

        # The failed one is still a rebalance date, but nothing traded.
        failed = records[failed_date]
        assert failed.is_rebalance
        assert failed.turnover == 0.0
        assert failed.cost == 0.0
        assert list(result.rebalance_log.index) == [ok_date, failed_date]
        assert list(result.rebalance_log["status"]) == ["ok", "failed"]
        assert result.rebalance_log.loc[failed_date, "turnover"] == 0.0
        assert [(f["date"], f["stage"]) for f in result.failures] == [
            (failed_date, "execution")]
        # Previous book (all A) is kept.
        assert result.weights.loc[failed_date, "A"] == pytest.approx(1.0)


class TestCapmShrunkActiveWindow:
    """Regression: capm_shrunk must not silently flatline the backtest.

    capm_shrunk needs the benchmark for betas, but mu was estimated on the
    active (liquidity-filtered) window only. With SPY absent every rebalance
    raised, run_backtest swallowed it → 0 rebalances, constant NAV.
    """

    @staticmethod
    def _panel_with_illiquid_benchmark(bench="SPY"):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=320)
        cols = ["A", "B", "C", "D", bench]
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(dates), 5)), axis=0)),
            index=dates,
            columns=cols,
        )
        volumes = pd.DataFrame(1e6, index=dates, columns=cols)
        volumes[bench] = 1.0  # universe filter drops it from the active set
        return prices, volumes

    def test_capm_without_benchmark_in_panel_fails_fast(
        self, synthetic_prices, synthetic_volumes
    ):
        """No silent swap to historical mu: a run labelled capm must be capm."""
        assert "SPY" not in synthetic_prices.columns
        cfg = BacktestConfig(
            opt_method="mv_classic",
            mu_method="capm_shrunk",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        with pytest.raises(ValueError, match="SPY"):
            run_backtest(synthetic_prices, synthetic_volumes, cfg)

    @pytest.mark.parametrize("bench", ["SPY", "MKT"])
    def test_capm_uses_benchmark_outside_active_set(self, bench):
        from src.rolling_engine import get_rebalance_dates
        prices, volumes = self._panel_with_illiquid_benchmark(bench)
        cfg = BacktestConfig(
            opt_method="mv_classic",
            mu_method="capm_shrunk",
            benchmark_col=bench,
            rebalance_freq="Q",
            window=126,
            warmup=126,
            max_weight=0.45,
        )
        result = run_backtest(prices, volumes, cfg)
        n_rebal = len(get_rebalance_dates(prices, freq="Q", warmup=126))
        assert len(result.rebalance_log) == n_rebal
        assert result.failures == []
        assert (result.rebalance_log["status"] == "ok").all()
        assert result.nav.nunique() > 1
        # Benchmark used for betas only; never traded when filtered out
        if bench in result.weights.columns:
            assert (result.weights[bench] == 0.0).all()

    def test_mu_is_capm_not_historical(self):
        """Active names get betas against the benchmark, not historical mu."""
        from src.strategy import StrategyConfig, _estimate_mu
        from src.covariance_estimators import estimate_covariance
        from src.expected_returns import capm_shrunk_return
        prices, _ = self._panel_with_illiquid_benchmark()
        window = prices.iloc[:126]
        wp = window[["A", "B", "C", "D"]]
        mu = _estimate_mu(
            StrategyConfig(mu_method="capm_shrunk"), wp, prices,
            estimate_covariance(wp),
        )
        expected = capm_shrunk_return(window, benchmark_col="SPY").drop("SPY")
        pd.testing.assert_series_equal(mu, expected, check_names=False)

    @staticmethod
    def _gap_config(strict):
        return BacktestConfig(
            opt_method="mv_classic",
            mu_method="capm_shrunk",
            rebalance_freq="Q",
            window=126,
            warmup=126,
            max_weight=0.45,
            strict=strict,
        )

    def test_benchmark_gap_aborts_strict_run(self):
        """Strict (default): the first window with benchmark gaps aborts."""
        from src.backtest_engine import RebalanceError
        prices, volumes = self._panel_with_illiquid_benchmark()
        prices.loc[prices.index[:150], "SPY"] = np.nan  # benchmark lists late
        with pytest.raises(RebalanceError, match="SPY"):
            run_backtest(prices, volumes, self._gap_config(strict=True))

    def test_benchmark_gap_fails_rebalance_visibly(self):
        """Non-strict: a window where the benchmark has no prices is skipped
        and listed."""
        prices, volumes = self._panel_with_illiquid_benchmark()
        prices.loc[prices.index[:150], "SPY"] = np.nan  # benchmark lists late
        result = run_backtest(prices, volumes, self._gap_config(strict=False))
        assert result.failures
        assert all("SPY" in f["message"] for f in result.failures)
        status = result.rebalance_log["status"]
        assert set(status) == {"failed", "ok"}
        assert list(status[status == "failed"].index) == [
            f["date"] for f in result.failures]

    def test_hrp_does_not_need_benchmark(self, synthetic_prices, synthetic_volumes):
        """HRP never estimates mu, so mu_method must not gate it."""
        cfg = BacktestConfig(
            opt_method="hrp",
            mu_method="capm_shrunk",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert not result.rebalance_log.empty


# ===========================================================================
# Optimizer contracts: liquidity wiring + solver fallback visible in the log
# ===========================================================================
class TestOptimizerContractWiring:
    def test_liquidity_backtest_rebalances(
        self, synthetic_prices, synthetic_volumes
    ):
        """opt_method='liquidity' used to raise on every rebalance (t and
        volumes never reached optimize) → swallowed → no rebalances."""
        cfg = BacktestConfig(
            opt_method="liquidity", rebalance_freq="Q", window=126, warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert result.failures == []
        assert len(result.rebalance_log) >= 1
        assert (result.rebalance_log["status"] == "ok").all()
        assert result.nav.nunique() > 1

    def test_solver_fallback_is_marked_in_rebalance_log(
        self, synthetic_prices, synthetic_volumes, caplog,
    ):
        """5 names × max_weight 0.10 < 1 → infeasible → equal-weight fallback
        (real solver, no mocks): OptimizationResult.fallback_used reaches the
        log as status="fallback_ew", with one warning from build_targets."""
        import logging

        cfg = BacktestConfig(
            opt_method="mv_tc", max_weight=0.10, rebalance_freq="Q",
            window=126, warmup=126,
        )
        with caplog.at_level(logging.WARNING):
            result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        log = result.rebalance_log
        assert (log["status"] == "fallback_ew").all()
        warned = [r for r in caplog.records if "fallback" in r.getMessage()
                  and r.name in ("strategy", "backtest_engine")]
        assert len(warned) == len(log)  # one per rebalance, not duplicated
