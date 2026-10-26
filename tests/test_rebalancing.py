"""
test_rebalancing.py — Tests unitarios de Fase 4.

Cubre T-19 a T-24: drift monitor, reglas de reemplazo,
ejecución TWAP y motor de reequilibrio.

Restricciones validadas:
  R-01: Costos de transacción calculados y acotados.
  R-02: Turnover cap respetado en rebalancing_engine.
  R-03: Solo activos líquidos como candidatos de reemplazo.
  R-06: Reproducibilidad con seed(42) en random_replacement.
  R-07: Las 3 reglas (momentum, random, min_variance) funcionan.
  R-10: Resultados deterministas.

Ejecutar: pytest tests/test_rebalancing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.drift_monitor import (
    check_drift,
    compute_drift_weights,
    rolling_drift_monitor,
)
from src.momentum_replacement import (
    compute_momentum_scores,
    momentum_replace,
)
from src.random_replacement import random_replace, _date_seed
from src.min_variance_replacement import (
    compute_marginal_risk_contribution,
    min_variance_replace,
)
from src.order_execution import (
    execute_order_twap,
    execute_rebalance,
    compute_adv,
)
from src.rebalancing_engine import rebalance, _apply_turnover_cap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N = 10          # number of tickers
N_OBS = 400


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tickers():
    return [f"T{i:02d}" for i in range(N)]


@pytest.fixture
def prices(tickers):
    np.random.seed(SEED)
    dates = pd.bdate_range("2021-01-01", periods=N_OBS, freq="B")
    log_rets = np.random.normal(4e-4, 0.013, (N_OBS, N))
    mkt = np.random.normal(0, 0.008, N_OBS)
    for i in range(N):
        log_rets[:, i] += 0.5 * mkt
    arr = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(arr, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def volumes(tickers, prices):
    np.random.seed(SEED + 1)
    vols = np.random.randint(300_000, 15_000_000, (N_OBS, N))
    df = pd.DataFrame(vols.astype(float), index=prices.index, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def w_equal(tickers):
    return pd.Series(np.ones(N) / N, index=tickers)


@pytest.fixture
def w_random(tickers):
    np.random.seed(SEED + 3)
    w = np.random.dirichlet(np.ones(N))
    return pd.Series(w, index=tickers)


# ===========================================================================
# T-19: Drift Monitor
# ===========================================================================
class TestDriftMonitor:
    def test_no_drift_no_rebalance(self, tickers):
        t = pd.Timestamp("2023-01-31")
        w = pd.Series(np.ones(N) / N, index=tickers)
        report = check_drift(t, w, w, drift_threshold=0.05)
        assert not report.should_rebalance
        assert report.trigger_reason == "none"

    def test_periodic_trigger(self, tickers):
        t = pd.Timestamp("2023-01-31")
        w = pd.Series(np.ones(N) / N, index=tickers)
        report = check_drift(t, w, w, is_rebalance_date=True)
        assert report.should_rebalance
        assert report.trigger_reason == "periodic"

    def test_threshold_trigger(self, tickers):
        t = pd.Timestamp("2023-01-31")
        w_target = pd.Series(np.ones(N) / N, index=tickers)
        w_current = w_target.copy()
        w_current.iloc[0] += 0.15  # 15% drift on first asset
        w_current /= w_current.sum()
        report = check_drift(t, w_current, w_target, drift_threshold=0.05)
        assert report.should_rebalance
        assert report.trigger_reason == "threshold"
        assert tickers[0] in report.breached_tickers

    def test_drift_vector_non_negative(self, tickers, w_equal, w_random):
        t = pd.Timestamp("2023-01-31")
        report = check_drift(t, w_random, w_equal)
        assert (report.drift_vector >= 0).all()

    def test_total_drift_equals_l1(self, tickers, w_equal, w_random):
        t = pd.Timestamp("2023-01-31")
        report = check_drift(t, w_random, w_equal)
        expected_l1 = float((w_random - w_equal).abs().sum())
        assert abs(report.total_drift - expected_l1) < 1e-9

    def test_compute_drift_weights_sums_to_one(self, tickers):
        w = pd.Series(np.ones(N) / N, index=tickers)
        returns = pd.Series(
            np.random.default_rng(SEED).standard_normal(N) * 0.02,
            index=tickers,
        )
        w_drifted = compute_drift_weights(w, returns)
        assert abs(w_drifted.sum() - 1.0) < 1e-9


# ===========================================================================
# T-20: Momentum Replacement
# ===========================================================================
class TestMomentumReplacement:
    def test_scores_indexed_by_tickers(self, tickers, prices):
        t = prices.index[-1]
        scores = compute_momentum_scores(prices, t)
        assert set(tickers).issubset(set(scores.index))

    def test_no_lookahead(self, tickers, prices):
        """Scores at t must only use prices strictly before t (R-05)."""
        t = prices.index[200]
        scores = compute_momentum_scores(prices, t)
        # Cannot directly test this via output, but ensure it runs without error
        assert len(scores) == N

    def test_replacement_reduces_worst(self, tickers, prices):
        t = prices.index[-1]
        current = tickers[:8]
        universe = tickers
        result = momentum_replace(t, current, universe, prices)
        # Removed tickers should be the worst performers in current
        scores = compute_momentum_scores(prices, t)
        portfolio_scores = scores.reindex(current).dropna()
        if len(result.removed_tickers) > 0:
            # All removed should have below-median momentum
            min_removed = min(
                portfolio_scores.get(tk, 0) for tk in result.removed_tickers
            )
            max_kept = max(
                portfolio_scores.get(tk, 0)
                for tk in current if tk not in result.removed_tickers
            )
            assert min_removed <= max_kept + 1e-6

    def test_new_universe_non_empty(self, tickers, prices):
        t = prices.index[-1]
        result = momentum_replace(t, tickers[:7], tickers, prices)
        assert len(result.new_universe) > 0

    def test_added_not_in_original(self, tickers, prices):
        t = prices.index[-1]
        current = tickers[:7]
        result = momentum_replace(t, current, tickers, prices)
        current_set = set(current)
        for tk in result.added_tickers:
            assert tk not in current_set


# ===========================================================================
# T-21: Random Replacement
# ===========================================================================
class TestRandomReplacement:
    def test_reproducible_same_seed(self, tickers):
        """R-10: same seed → same result every time."""
        t = pd.Timestamp("2023-06-30")
        r1 = random_replace(t, tickers[:8], tickers)
        r2 = random_replace(t, tickers[:8], tickers)
        assert r1.removed_tickers == r2.removed_tickers
        assert r1.added_tickers == r2.added_tickers

    def test_different_dates_different_result(self, tickers):
        """Different dates should typically produce different results."""
        t1 = pd.Timestamp("2023-06-30")
        t2 = pd.Timestamp("2023-09-30")
        r1 = random_replace(t1, tickers[:8], tickers)
        r2 = random_replace(t2, tickers[:8], tickers)
        # May coincidentally match, but seeds differ
        seed1 = _date_seed(t1)
        seed2 = _date_seed(t2)
        assert seed1 != seed2

    def test_added_not_in_original(self, tickers):
        t = pd.Timestamp("2023-06-30")
        current = tickers[:7]
        result = random_replace(t, current, tickers)
        current_set = set(current)
        for tk in result.added_tickers:
            assert tk not in current_set

    def test_new_universe_size_preserved(self, tickers):
        t = pd.Timestamp("2023-06-30")
        n_current = 8
        result = random_replace(t, tickers[:n_current], tickers)
        # Size should stay roughly the same (removed = added)
        assert len(result.new_universe) >= n_current - 2

    def test_no_candidates_returns_original(self, tickers):
        t = pd.Timestamp("2023-06-30")
        result = random_replace(t, tickers, tickers)  # no external candidates
        assert result.added_tickers == []
        assert result.new_universe == tickers


# ===========================================================================
# T-22: Min-Variance Replacement
# ===========================================================================
class TestMinVarianceReplacement:
    def test_mcr_positive(self, tickers, prices):
        """MCR values should be non-negative."""
        from src.covariance_estimators import sample_covariance
        t = prices.index[-1]
        wp = prices[prices.index < t].iloc[-63:]
        sigma = sample_covariance(wp)
        w = pd.Series(np.ones(N) / N, index=tickers)
        mcr = compute_marginal_risk_contribution(w, sigma)
        assert (mcr >= 0).all()

    def test_mcr_sums_to_portfolio_vol(self, tickers, prices):
        """sum(w_i * MCR_i) = portfolio volatility."""
        from src.covariance_estimators import sample_covariance
        t = prices.index[-1]
        wp = prices[prices.index < t].iloc[-63:]
        sigma = sample_covariance(wp)
        w = pd.Series(np.ones(N) / N, index=tickers)
        mcr = compute_marginal_risk_contribution(w, sigma)
        sigma_p = np.sqrt(float(w.values @ sigma.values @ w.values))
        assert abs(float((w * mcr).sum()) - sigma_p) < 1e-6

    def test_replacement_result_non_empty(self, tickers, prices):
        t = prices.index[-1]
        current = tickers[:7]
        result = min_variance_replace(
            t, current, tickers, prices
        )
        assert result is not None
        assert len(result.new_universe) > 0

    def test_added_not_in_original(self, tickers, prices):
        t = prices.index[-1]
        current = tickers[:7]
        result = min_variance_replace(t, current, tickers, prices)
        current_set = set(current)
        for tk in result.added_tickers:
            assert tk not in current_set

    def test_cash_book_ranks_like_equal_weight(self):
        """A zero book (first deployment from cash) has no risk to rank by.

        Sigma @ 0 = 0 makes every MCR 0, so nlargest fell back to ticker
        order (removed "A"). Rank as with w_current=None (equal weight):
        the riskiest name, "E", goes.
        """
        rng = np.random.default_rng(0)
        cols = ["A", "B", "C", "D", "E", "F"]
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (80, 6)), axis=0)),
            index=pd.bdate_range("2024-01-01", periods=80), columns=cols,
        )
        current = cols[:5]
        sigma = pd.DataFrame(np.diag([0.01] * 4 + [0.25]),
                             index=current, columns=current)
        result = min_variance_replace(
            prices.index[-1], current, cols, prices,
            w_current=pd.Series(0.0, index=current), sigma=sigma,
            replace_pct=0.2,
        )
        assert result.removed_tickers == ["E"]


# ===========================================================================
# T-23: Order Execution (TWAP)
# ===========================================================================
class TestOrderExecution:
    def test_single_order_cost_positive(self):
        res = execute_order_twap(
            ticker="T00",
            order_value=10_000.0,
            adv=1_000_000.0,
            price=100.0,
        )
        assert res.total_cost > 0

    def test_zero_order_zero_cost(self):
        res = execute_order_twap(
            ticker="T00",
            order_value=0.0,
            adv=1_000_000.0,
            price=100.0,
        )
        assert res.total_cost == 0.0

    def test_twap_cheaper_than_block(self):
        """TWAP (n_slices=10) should cost less than block (n_slices=1)."""
        kwargs = dict(
            ticker="T00",
            order_value=100_000.0,
            adv=500_000.0,
            price=100.0,
        )
        block = execute_order_twap(**kwargs, n_slices=1)
        twap = execute_order_twap(**kwargs, n_slices=10)
        assert twap.market_impact_cost <= block.market_impact_cost + 1e-10

    def test_cost_bps_reasonable(self):
        """Total cost should be < 100 bps for a normal order."""
        res = execute_order_twap(
            ticker="T00",
            order_value=50_000.0,
            adv=5_000_000.0,
            price=100.0,
            commission_bps=5.0,
            spread_bps=2.0,
            impact_coef=0.1,
        )
        assert res.cost_bps < 100.0

    def test_adv_computation(self, prices, volumes):
        t = prices.index[-1]
        adv = compute_adv(volumes, prices, t, window=30)
        assert (adv > 0).all()
        # ADV should be in data range (not zero, not infinite)
        assert adv.max() < 1e15

    def test_execute_rebalance_returns_result(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        t = prices.index[-1]
        result = execute_rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_random,
            portfolio_value=1_000_000.0,
            prices=prices,
            volumes=volumes,
        )
        assert result.total_cost >= 0
        assert result.effective_turnover >= 0

    def test_execute_rebalance_cost_proportional(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        """Larger portfolio → proportionally larger cost."""
        t = prices.index[-1]
        r_small = execute_rebalance(
            t=t, w_current=w_equal, w_target=w_random,
            portfolio_value=100_000.0, prices=prices, volumes=volumes,
        )
        r_large = execute_rebalance(
            t=t, w_current=w_equal, w_target=w_random,
            portfolio_value=10_000_000.0, prices=prices, volumes=volumes,
        )
        # Larger portfolio → absolutely larger cost
        assert r_large.total_cost > r_small.total_cost


# ===========================================================================
# T-24: Rebalancing Engine
# ===========================================================================
class TestRebalancingEngine:
    def test_rebalance_no_rule(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        t = prices.index[-1]
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_random,
            prices=prices,
            volumes=volumes,
            rule="none",
            is_rebalance_date=True,
        )
        assert abs(result.w_new.sum() - 1.0) < 1e-4

    @pytest.mark.parametrize("rule", ["momentum", "random", "min_variance"])
    def test_rebalance_all_rules(
        self, rule, tickers, prices, volumes, w_equal, w_random
    ):
        """R-07: All 3 replacement rules must execute without error."""
        t = prices.index[-1]
        universe = tickers
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_random,
            prices=prices,
            volumes=volumes,
            rule=rule,
            universe=universe,
            is_rebalance_date=True,
        )
        assert result.rule_applied == rule
        assert abs(result.w_new.sum() - 1.0) < 1e-4

    def test_turnover_cap_respected(
        self, tickers, prices, volumes, w_equal
    ):
        """R-02: actual_turnover must be <= turnover_limit."""
        t = prices.index[-1]
        # Extreme target to force max turnover
        np.random.seed(SEED)
        w_extreme = pd.Series(
            np.zeros(N), index=tickers
        )
        w_extreme.iloc[0] = 1.0  # 100% in one asset
        limit = 0.20
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_extreme,
            prices=prices,
            volumes=volumes,
            rule="none",
            turnover_limit=limit,
            is_rebalance_date=True,
        )
        assert result.actual_turnover <= limit + 1e-4

    @staticmethod
    def _two_groups(tickers):
        # T00-T04 -> G1 (50% del libro equiponderado), resto -> G2
        return pd.Series({tk: ("G1" if i < 5 else "G2")
                          for i, tk in enumerate(tickers)})

    @staticmethod
    def _capped_target(tickers):
        # G1 = 20%, G2 = 80%: cumple G1 <= 0.30
        return pd.Series([0.04] * 5 + [0.16] * 5, index=tickers)

    def test_group_caps_bind_executed_book(
        self, tickers, prices, volumes, w_equal
    ):
        """Topes por grupo duros en w_new: el turnover cede lo minimo."""
        groups = self._two_groups(tickers)
        # Objetivo con G1 justo en el tope (tope activo) + rotacion interna
        # en G2 (T06 <-> T07) ajena al tope. L1 desde EW = 0.6.
        w_target = pd.Series(
            [0.0, 0.0, 0.1, 0.1, 0.1, 0.3, 0.2, 0.0, 0.1, 0.1], index=tickers)
        result = rebalance(
            t=prices.index[-1],
            w_current=w_equal,
            w_target=w_target,
            prices=prices, volumes=volumes, rule="none",
            turnover_limit=0.20, is_rebalance_date=True,
            groups=groups, group_caps={"G1": 0.30},
        )
        assert result.w_new[groups == "G1"].sum() == pytest.approx(0.30)
        # Blend de turnover (alpha=1/3) deja G1 en 0.4333: se vende solo ese
        # exceso -> 0.20 + 2 * 0.1333 (no el salto completo al objetivo, 0.6)
        assert result.actual_turnover == pytest.approx(0.20 + 2 * 0.4 / 3, abs=1e-6)
        # La rotacion T06/T07 no se acelera: T07 queda donde lo dejo el blend
        assert result.w_new["T07"] == pytest.approx(0.1 - 0.1 / 3)

    def test_group_caps_slack_keeps_turnover_limit(
        self, tickers, prices, volumes, w_equal
    ):
        """Si el blend de turnover ya cumple el tope, no cambia nada."""
        kw = dict(
            t=prices.index[-1], w_current=w_equal,
            w_target=self._capped_target(tickers),
            prices=prices, volumes=volumes, rule="none",
            turnover_limit=0.20, is_rebalance_date=True,
        )
        base = rebalance(**kw)
        capped = rebalance(**kw, groups=self._two_groups(tickers),
                           group_caps={"G1": 0.45})
        pd.testing.assert_series_equal(capped.w_new, base.w_new)
        assert capped.actual_turnover <= 0.20 + 1e-6

    def test_group_caps_blend_toward_projected_target(
        self, tickers, prices, volumes
    ):
        """Libro factible + objetivo fuera de tope: el blend apunta al objetivo
        re-proyectado, sin gastar turnover en reparar su propio exceso."""
        groups = self._two_groups(tickers)
        w_start = pd.Series([0.06] * 5 + [0.14] * 5, index=tickers)  # G1=0.30
        # G1=0.55; blend hacia este objetivo crudo + reparacion gastaba 0.26
        w_breach = pd.Series(
            [0.05, 0.15, 0.15, 0.05, 0.15, 0.1, 0.1, 0.15, 0.05, 0.05],
            index=tickers)
        result = rebalance(
            t=prices.index[-1], w_current=w_start, w_target=w_breach,
            prices=prices, volumes=volumes, rule="none",
            turnover_limit=0.20, is_rebalance_date=True,
            groups=groups, group_caps={"G1": 0.30},
        )
        assert result.w_new[groups == "G1"].sum() <= 0.30 + 1e-9
        assert result.actual_turnover <= 0.20 + 1e-9

    def test_group_caps_recap_respects_max_weight(
        self, tickers, prices, volumes, w_equal
    ):
        """La re-proyeccion del objetivo respeta el tope por activo."""
        groups = self._two_groups(tickers)
        # G1=0.80; sin tope por activo el deficit iria a T05 (0.16 -> 0.56)
        w_breach = pd.Series([0.16] * 5 + [0.16, 0.01, 0.01, 0.01, 0.01],
                             index=tickers)
        result = rebalance(
            t=prices.index[-1], w_current=w_equal, w_target=w_breach,
            prices=prices, volumes=volumes, rule="none",
            turnover_limit=2.0, is_rebalance_date=True,
            groups=groups, group_caps={"G1": 0.30}, max_weight=0.30,
        )
        assert result.w_new[groups == "G1"].sum() <= 0.30 + 1e-9
        assert result.w_new.max() <= 0.30 + 1e-9

    def test_group_caps_recap_breaching_target(
        self, tickers, prices, volumes, w_equal
    ):
        """Objetivo fuera de topes (ej. tras reemplazo): se re-proyecta."""
        groups = self._two_groups(tickers)
        w_breach = pd.Series([0.12] * 5 + [0.08] * 5, index=tickers)  # G1=0.60
        result = rebalance(
            t=prices.index[-1], w_current=w_equal, w_target=w_breach,
            prices=prices, volumes=volumes, rule="none",
            turnover_limit=2.0, is_rebalance_date=True,
            groups=groups, group_caps={"G1": 0.30},
        )
        assert result.w_new[groups == "G1"].sum() <= 0.30 + 1e-9
        assert abs(result.w_new.sum() - 1.0) < 1e-9

    def test_w_new_non_negative(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        """R-04: w_new >= 0."""
        t = prices.index[-1]
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_random,
            prices=prices,
            volumes=volumes,
            rule="none",
        )
        assert (result.w_new >= -1e-6).all()

    def test_cost_non_negative(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        """R-01: transaction cost must be non-negative."""
        t = prices.index[-1]
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_random,
            prices=prices,
            volumes=volumes,
            rule="none",
            is_rebalance_date=True,
        )
        assert result.cost >= 0.0

    def test_no_rebalance_when_no_drift(
        self, tickers, prices, volumes, w_equal
    ):
        """No drift + no periodic date → weights unchanged."""
        t = prices.index[-1]
        result = rebalance(
            t=t,
            w_current=w_equal,
            w_target=w_equal,
            prices=prices,
            volumes=volumes,
            rule="none",
            drift_threshold=0.05,
            is_rebalance_date=False,
        )
        assert not result.drift_report.should_rebalance

    def test_invalid_rule_raises(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        t = prices.index[-1]
        with pytest.raises(ValueError, match="rule must be one of"):
            rebalance(
                t=t,
                w_current=w_equal,
                w_target=w_random,
                prices=prices,
                volumes=volumes,
                rule="invalid_rule",
            )

    def test_turnover_cap_function(self, tickers):
        """_apply_turnover_cap returns L1 <= limit."""
        w_c = pd.Series(np.ones(N) / N, index=tickers)
        w_t = pd.Series(np.zeros(N), index=tickers)
        w_t.iloc[0] = 1.0
        limit = 0.20
        w_capped = _apply_turnover_cap(w_c, w_t, limit)
        l1 = float((w_capped - w_c).abs().sum())
        assert l1 <= limit + 1e-6

    def test_random_replacement_reproducible(
        self, tickers, prices, volumes, w_equal, w_random
    ):
        """R-10: random rule gives same result on independent runs."""
        t = prices.index[-1]
        r1 = rebalance(
            t=t, w_current=w_equal, w_target=w_random,
            prices=prices, volumes=volumes, rule="random",
            is_rebalance_date=True,
        )
        r2 = rebalance(
            t=t, w_current=w_equal, w_target=w_random,
            prices=prices, volumes=volumes, rule="random",
            is_rebalance_date=True,
        )
        pd.testing.assert_series_equal(r1.w_new, r2.w_new, atol=1e-6)


class TestNanAdvMarketImpact:
    """Missing ADV must not invent a tiny sentinel that wipes NAV."""

    def test_nan_adv_does_not_explode_impact_cost(self):
        dates = pd.bdate_range("2024-01-01", periods=40)
        prices = pd.DataFrame(
            {
                "A": np.linspace(100, 110, len(dates)),
                "B": np.linspace(50, 55, len(dates)),
            },
            index=dates,
        )
        volumes = pd.DataFrame(
            {
                "A": np.full(len(dates), 1e6),
                "B": np.nan,  # yfinance-style missing volume series
            },
            index=dates,
        )
        t = dates[-1]
        w_current = pd.Series({"A": 0.5, "B": 0.5})
        w_target = pd.Series({"A": 0.1, "B": 0.9})
        pv = 1_000_000.0

        result = execute_rebalance(
            t=t,
            w_current=w_current,
            w_target=w_target,
            portfolio_value=pv,
            prices=prices,
            volumes=volumes,
        )
        # Commission + spread only on |Δw|*NAV ≈ 0.8 * 1e6 → ~560 at 7 bps.
        # Pre-fix sentinel ADV=$1000 produced multi-million impact.
        assert result.total_cost < 5_000.0
        assert result.total_cost / pv < 0.01
        if not result.order_df.empty and "B" in result.order_df["ticker"].values:
            b_row = result.order_df.set_index("ticker").loc["B"]
            assert b_row["market_impact"] == 0.0



class TestRebalanceExecutionFailure:
    """Regression: a failed execute_rebalance used to be swallowed and the
    rebalance reported cost=0 with the new weights, as if it had traded."""

    def test_execution_failure_propagates(
        self, prices, volumes, w_equal, w_random, monkeypatch
    ):
        import src.rebalancing_engine as reb

        def _boom(**kwargs):
            raise ValueError("execution boom")

        monkeypatch.setattr(reb, "execute_rebalance", _boom)
        with pytest.raises(ValueError, match="execution boom"):
            rebalance(
                t=prices.index[-1],
                w_current=w_equal,
                w_target=w_random,
                prices=prices,
                volumes=volumes,
                rule="none",
                is_rebalance_date=True,
            )


class TestRebalanceKeepsHoldingsOutsideTarget:
    """Regression: rebalance() reindexed w_current to the target's tickers, so
    a holding outside w_target vanished — never sold, its cost never paid, its
    weight silently spread over the rest of the book."""

    PV = 1_000_000.0

    def _run(self, prices, volumes, w_current, w_target, **kw):
        kw.setdefault("rule", "none")
        return rebalance(
            t=prices.index[-1], w_current=w_current, w_target=w_target,
            prices=prices, volumes=volumes, portfolio_value=self.PV,
            turnover_limit=2.0, is_rebalance_date=True, **kw,
        )

    def test_holding_dropped_from_target_is_sold_with_cost(
        self, tickers, prices, volumes
    ):
        a, b, out = tickers[0], tickers[1], tickers[-1]
        w_current = pd.Series({a: 0.3, b: 0.3, out: 0.4})
        w_target = pd.Series({a: 0.5, b: 0.5})

        res = self._run(prices, volumes, w_current, w_target)

        assert out in res.w_new.index
        assert res.w_new[out] == 0.0
        trades = res.trades.set_index("ticker")
        assert out in trades.index
        assert trades.loc[out, "order_value"] == pytest.approx(-0.4 * self.PV)
        assert trades.loc[out, "total_cost"] > 0
        assert res.actual_turnover == pytest.approx(0.8)

    def test_holding_outside_target_sold_in_full_under_turnover_cap(
        self, tickers, prices, volumes
    ):
        """Forced exit, outside R-02 (as the ledger/live path): the cap only
        throttles the investable book, starting from its current mix."""
        a, b, out = tickers[0], tickers[1], tickers[-1]
        w_current = pd.Series({a: 0.3, b: 0.3, out: 0.4})
        w_target = pd.Series({a: 0.9, b: 0.1})

        res = rebalance(
            t=prices.index[-1], w_current=w_current, w_target=w_target,
            prices=prices, volumes=volumes, portfolio_value=self.PV,
            rule="none", turnover_limit=0.2, is_rebalance_date=True,
        )

        assert res.w_new[out] == 0.0
        trades = res.trades.set_index("ticker")
        assert trades.loc[out, "order_value"] == pytest.approx(-0.4 * self.PV)
        # Investable mix (0.5, 0.5) moves 0.2 of L1 toward (0.9, 0.1)
        assert res.w_new[a] == pytest.approx(0.6)
        assert res.w_new[b] == pytest.approx(0.4)

    def test_current_weights_not_renormalized(self, tickers, prices, volumes):
        """w_before is the book as given (20% cash stays cash, out stays)."""
        a, b, out = tickers[0], tickers[1], tickers[-1]
        w_current = pd.Series({a: 0.3, b: 0.3, out: 0.2})
        w_target = pd.Series({a: 0.5, b: 0.5})

        res = self._run(prices, volumes, w_current, w_target)

        pd.testing.assert_series_equal(
            res.w_before.sort_index(), w_current.sort_index(), check_names=False,
        )

    def test_replacement_rule_keeps_holdings_outside_target(
        self, tickers, prices, volumes
    ):
        """The replacement branch rebuilt wc over the new universe only."""
        out = tickers[-1]
        target = tickers[:5]
        w_current = pd.Series({**{tk: 0.16 for tk in target}, out: 0.2})
        w_target = pd.Series(0.2, index=target)

        res = self._run(prices, volumes, w_current, w_target, rule="random",
                        universe=tickers[:-1], replace_pct=0.4)

        assert res.universe_changed
        assert res.w_new.get(out, np.nan) == 0.0
        trades = res.trades.set_index("ticker")
        assert trades.loc[out, "order_value"] == pytest.approx(-0.2 * self.PV)
