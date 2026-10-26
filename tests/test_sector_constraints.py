"""
test_sector_constraints.py — Paso 3: topes por grupo, presets y cableado.

Offline: datos sinteticos + CSVs locales (sin red).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import BacktestConfig, run_backtest
from src.portfolio_state import PortfolioState
from src.rolling_engine import get_window
from src.strategy import StrategyConfig, build_targets
from src.portfolio_optimizer import optimize
from src.risk_profiles import (
    RISK_PROFILES,
    mixed_sleeve_config,
    profile_config,
)
from src.sector_constraints import (
    apply_group_caps,
    build_mixed_groups,
    co_sleeve_caps,
    min_blend_to_caps,
    sell_down_to_caps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_weights() -> tuple[pd.Series, pd.Series]:
    w = pd.Series(
        {"a": 0.40, "b": 0.30, "c": 0.20, "d": 0.10},
        dtype=float,
    )
    groups = pd.Series(
        {"a": "co/Financials", "b": "co/Energy", "c": "us/Tech", "d": "us/Tech"}
    )
    return w, groups


class TestApplyGroupCaps:
    def test_simple_cap_respected(self):
        w, groups = _toy_weights()
        out = apply_group_caps(w, groups, {"co/Financials": 0.25})
        assert out["a"] <= 0.25 + 1e-9
        assert abs(out.sum() - 1) < 1e-9
        assert (out >= 0).all()

    def test_combined_cap_respected(self):
        w, groups = _toy_weights()
        out = apply_group_caps(
            w, groups, {("co/Financials", "co/Energy"): 0.40}
        )
        assert out[["a", "b"]].sum() <= 0.40 + 1e-9
        assert abs(out.sum() - 1) < 1e-9

    def test_no_caps_passthrough(self):
        w, groups = _toy_weights()
        out = apply_group_caps(w, groups, {})
        pd.testing.assert_series_equal(out, w / w.sum())

    def test_deterministic(self):
        w, groups = _toy_weights()
        caps = {("co/Financials", "co/Energy"): 0.40}
        pd.testing.assert_series_equal(
            apply_group_caps(w, groups, caps), apply_group_caps(w, groups, caps)
        )

    def test_infeasible_raises(self):
        w, groups = _toy_weights()
        with pytest.raises(ValueError, match="[Ii]nfactible"):
            apply_group_caps(
                w, groups,
                {"co/Financials": 0.10, "co/Energy": 0.10, "us/Tech": 0.10},
            )

    def test_max_weight_by_ticker(self):
        w = pd.Series({"a": 0.50, "b": 0.30, "c": 0.20}, dtype=float)
        groups = pd.Series({"a": "co/X", "b": "co/X", "c": "us/Y"})
        out = apply_group_caps(
            w, groups, {"co/X": 0.60},
            max_weight_by_ticker=pd.Series({"a": 0.15, "b": 0.50, "c": 0.50}),
        )
        assert out["a"] <= 0.15 + 1e-9
        assert out[["a", "b"]].sum() <= 0.60 + 1e-9
        assert abs(out.sum() - 1) < 1e-9
        assert (out >= 0).all()

    def test_infeasible_ub_raises(self):
        w = pd.Series({"a": 0.60, "b": 0.40}, dtype=float)
        groups = pd.Series({"a": "co/X", "b": "us/Y"})
        with pytest.raises(ValueError, match="[Ii]nfactible"):
            apply_group_caps(
                w, groups, {"us/Y": 0.50},
                max_weight_by_ticker=pd.Series({"a": 0.15, "b": 0.50}),
            )


def _blend(w_from: pd.Series, w_to: pd.Series, alpha: float) -> pd.Series:
    a = w_from.reindex(w_to.index).fillna(0.0)
    w = a + alpha * (w_to - a)
    return w / w.sum()


class TestMinBlendToCaps:
    """Paso minimo sobre el segmento actual -> objetivo que cumple los topes."""

    GROUPS = pd.Series({"a": "co/Financials", "b": "co/Energy", "c": "us/Tech"})

    def test_zero_when_start_within_caps(self):
        w_from = pd.Series({"a": 0.20, "b": 0.10, "c": 0.70})
        w_to = pd.Series({"a": 0.05, "b": 0.05, "c": 0.90})
        assert min_blend_to_caps(
            w_from, w_to, self.GROUPS, {"co/Financials": 0.30}) == 0.0

    def test_smallest_step_lands_on_cap(self):
        w_from = pd.Series({"a": 0.50, "b": 0.00, "c": 0.50})
        w_to = pd.Series({"a": 0.20, "b": 0.00, "c": 0.80})
        alpha = min_blend_to_caps(
            w_from, w_to, self.GROUPS, {"co/Financials": 0.30})
        assert alpha == pytest.approx(2 / 3)
        assert _blend(w_from, w_to, alpha)["a"] == pytest.approx(0.30)

    def test_combined_cap_on_unnormalized_start(self):
        # w_from suma 0.8 (nombres fuera del objetivo): tope sobre el
        # blend renormalizado, igual que lo ejecuta el rebalanceador.
        w_from = pd.Series({"a": 0.30, "b": 0.20, "c": 0.30})
        w_to = pd.Series({"a": 0.10, "b": 0.10, "c": 0.80})
        caps = {("co/Financials", "co/Energy"): 0.40}
        alpha = min_blend_to_caps(w_from, w_to, self.GROUPS, caps)
        assert 0.0 < alpha < 1.0
        assert _blend(w_from, w_to, alpha)[["a", "b"]].sum() == pytest.approx(0.40)

    def test_target_breaching_cap_raises(self):
        w_from = pd.Series({"a": 0.50, "b": 0.00, "c": 0.50})
        w_to = pd.Series({"a": 0.45, "b": 0.00, "c": 0.55})
        with pytest.raises(ValueError, match="objetivo"):
            min_blend_to_caps(w_from, w_to, self.GROUPS, {"co/Financials": 0.30})


class TestSellDownToCaps:
    """Reparacion de topes con turnover extra minimo, siempre hacia el objetivo."""

    GROUPS = pd.Series({"a": "co/Financials", "b": "co/Energy",
                        "c": "us/Tech", "d": "us/Tech"})
    CAPS = {"co/Financials": 0.30}

    def test_sells_only_excess_toward_target(self):
        w = pd.Series({"a": 0.40, "b": 0.10, "c": 0.30, "d": 0.20})
        target = pd.Series({"a": 0.20, "b": 0.15, "c": 0.25, "d": 0.40})
        out = sell_down_to_caps(w, target, self.GROUPS, self.CAPS)
        assert out["a"] == pytest.approx(0.30)
        # exceso 0.10 vendido y recomprado: turnover extra = 2 * exceso
        assert (out - w).abs().sum() == pytest.approx(0.20)
        # compras a prorrata de la brecha (b: 0.05, d: 0.20); c ya sobre
        # su objetivo no se toca
        assert out["b"] == pytest.approx(0.12)
        assert out["d"] == pytest.approx(0.28)
        assert out["c"] == pytest.approx(0.30)
        # cada trade va hacia el objetivo sin sobrepasarlo
        assert ((out - w) * (target - out) >= -1e-12).all()

    def test_crossed_caps_repaired_without_jumping_to_target(self):
        # Reparar Fin recompra Tech (bajo objetivo) y re-excede Tech: otra
        # pasada vende ese exceso; no salta al objetivo (c quedaria en 0.20).
        w = pd.Series({"a": 0.35, "b": 0.05, "c": 0.45, "d": 0.15})
        target = pd.Series({"a": 0.30, "b": 0.20, "c": 0.20, "d": 0.30})
        caps = {"us/Tech": 0.50, "co/Financials": 0.30}
        out = sell_down_to_caps(w, target, self.GROUPS, caps)
        expected = pd.Series({"a": 0.30, "b": 0.20, "c": 0.3125, "d": 0.1875})
        pd.testing.assert_series_equal(out, expected, atol=1e-6)

    def test_noop_within_caps(self):
        w = pd.Series({"a": 0.25, "b": 0.10, "c": 0.35, "d": 0.30})
        target = pd.Series({"a": 0.20, "b": 0.15, "c": 0.25, "d": 0.40})
        out = sell_down_to_caps(w, target, self.GROUPS, self.CAPS)
        pd.testing.assert_series_equal(out, w)

    def test_target_breaching_cap_raises(self):
        w = pd.Series({"a": 0.40, "b": 0.10, "c": 0.30, "d": 0.20})
        target = pd.Series({"a": 0.35, "b": 0.15, "c": 0.25, "d": 0.25})
        with pytest.raises(ValueError, match="objetivo"):
            sell_down_to_caps(w, target, self.GROUPS, self.CAPS)


class TestHelpers:
    def test_build_mixed_groups(self):
        uni = pd.read_csv(REPO_ROOT / "universe_mixed.csv")
        groups = build_mixed_groups(uni)
        assert len(groups) == 120
        assert groups["EC"] == "co/Energy"
        assert groups["AAPL"] == "us/Technology"

    def test_build_mixed_groups_requires_columns(self):
        with pytest.raises(ValueError, match="sin columnas"):
            build_mixed_groups(pd.DataFrame({"ticker": ["A"]}))

    def test_co_sleeve_caps_is_relative_to_sleeve(self):
        # criterios.md: Fin+Energia <= 40% DEL SLEEVE; sleeve 25% NAV -> 10% NAV
        caps = co_sleeve_caps(0.40, sleeve_weight=0.25)
        assert caps[("co/Financials", "co/Energy")] == pytest.approx(0.10)
        caps = co_sleeve_caps(0.40, sleeve_weight=0.20)
        assert caps[("co/Financials", "co/Energy")] == pytest.approx(0.08)

    def test_co_sleeve_caps_validates(self):
        with pytest.raises(ValueError, match="fuera de"):
            co_sleeve_caps(1.5, sleeve_weight=0.25)
        with pytest.raises(ValueError, match="sleeve_weight"):
            co_sleeve_caps(0.40, sleeve_weight=0.0)
        with pytest.raises(ValueError, match="sleeve_weight"):
            co_sleeve_caps(0.40, sleeve_weight=1.2)


def _toy_mu_sigma(seed: int = 3):
    rng = np.random.default_rng(seed)
    tickers = ["a", "b", "c", "d", "e", "f"]
    mu = pd.Series(rng.normal(0.08, 0.03, len(tickers)), index=tickers)
    base = rng.normal(size=(len(tickers), len(tickers)))
    sigma = pd.DataFrame(
        base @ base.T / len(tickers) + np.eye(len(tickers)) * 0.02,
        index=tickers, columns=tickers,
    )
    return mu, sigma


class TestOptimizeCaps:
    def test_optimize_respects_caps(self):
        mu, sigma = _toy_mu_sigma()
        groups = pd.Series(
            {"a": "G1", "b": "G1", "c": "G2", "d": "G2", "e": "G3", "f": "G3"}
        )
        res = optimize(mu, sigma, method="mv_classic",
                       groups=groups, group_caps={"G1": 0.20})
        assert res["caps_applied"] is True
        assert res["weights"][["a", "b"]].sum() <= 0.20 + 1e-6
        assert abs(res["weights"].sum() - 1) < 1e-9

    def test_optimize_no_caps_flag_false(self):
        mu, sigma = _toy_mu_sigma()
        res = optimize(mu, sigma, method="mv_classic")
        assert res["caps_applied"] is False

    def test_optimize_caps_require_groups(self):
        mu, sigma = _toy_mu_sigma()
        with pytest.raises(ValueError, match="requires `groups`"):
            optimize(mu, sigma, group_caps={"G1": 0.20})


def _synthetic_mixed(seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=150)
    us = [f"US{i}" for i in range(6)]
    co = ["CO_F1", "CO_F2", "CO_E1", "CO_U1"]
    cols = us + co
    prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for c in cols:
        drift = 0.001 if c.startswith("US") else 0.0005
        rets = rng.normal(drift, 0.015, len(dates))
        prices[c] = 100 * np.exp(np.cumsum(rets))
    volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
    return prices, volumes


def _mixed_groups(columns) -> dict:
    co = {"CO_F1": "co/Financials", "CO_F2": "co/Financials",
          "CO_E1": "co/Energy", "CO_U1": "co/Utilities"}
    return {c: ("us/Tech" if c.startswith("US") else co[c]) for c in columns}


class TestBacktestWiring:
    @pytest.mark.parametrize("ledger", [False, True], ids=["classic", "ledger"])
    @pytest.mark.parametrize("method", ["mv_classic", "robust", "hrp"])
    def test_executed_book_respects_cap_under_turnover_limit(self, method, ledger):
        """Topes duros sobre el libro ejecutado, no solo el objetivo.

        Libro de partida equiponderado sobre 9 activos (Fin+Energia = 3/9)
        + turnover_limit=0.20 dejaba 0.3229/0.3119/0.3063 (mv) sobre 0.30.
        """
        prices, volumes = _synthetic_mixed()
        cfg = BacktestConfig(
            opt_method=method, window=30, warmup=30,
            rebalance_freq="M", seed=42,
            groups=_mixed_groups(prices.columns),
            group_caps={("co/Financials", "co/Energy"): 0.30},
            use_position_ledger=ledger,
        )
        res = run_backtest(prices, volumes, cfg)
        rebal_dates = res.rebalance_log.index
        assert len(rebal_dates) == 6  # ningun rebalanceo fallido
        co_fe = res.weights.loc[rebal_dates, ["CO_F1", "CO_F2", "CO_E1"]].sum(axis=1)
        assert (co_fe <= 0.30 + 1e-6).all(), co_fe.round(4).tolist()

    @pytest.mark.parametrize("opt_method", ["mv_classic", "robust", "hrp"])
    def test_backtest_respects_combined_cap(self, opt_method):
        prices, volumes = _synthetic_mixed()
        groups = _mixed_groups(prices.columns)
        cfg = BacktestConfig(
            opt_method=opt_method, window=30, warmup=30,
            rebalance_freq="M", seed=42,
            # No turnover throttle, so the book reaches the capped target;
            # uncapped, CO Fin+Energy peaks at 0.46 (hrp) to 0.60 (mv).
            turnover_limit=2.0,
            groups=groups,
            group_caps={("co/Financials", "co/Energy"): 0.30},
        )
        res = run_backtest(prices, volumes, cfg)
        assert len(res.rebalance_log) > 0
        # The book holds shares: caps bind on the executed book (rebalance
        # dates); between rebalances weights drift with prices.
        rebal_dates = res.rebalance_log.index
        co_fe = res.weights.loc[rebal_dates, ["CO_F1", "CO_F2", "CO_E1"]].sum(axis=1)
        assert (co_fe <= 0.30 + 1e-6).all()

    @staticmethod
    def _target_with_exits(stopped):
        prices, volumes = _synthetic_mixed()
        t = prices.index[60]
        wp = get_window(prices, t, window=30).dropna(axis=1)
        cfg = StrategyConfig(
            opt_method="mv_classic", window=30,
            universe_filter=False,  # todo wp activo: solo topes + salidas
            groups=_mixed_groups(prices.columns),
            group_caps={("co/Financials", "co/Energy"): 0.30},
            use_exit_rules=True, take_profit_pct=None,
        )
        # Libro con todo wp; entrada absurda -> stop-loss seguro en `stopped`
        held = pd.Series(1.0 / wp.shape[1], index=wp.columns)
        state = PortfolioState.from_weights(
            held, prices.loc[t], equity=1e6,
            avg_entry={tk: 1e9 for tk in stopped}, as_of=t)
        res = build_targets(state, prices, volumes, t, cfg)
        return res.weights, {"exits": res.exits["ticker"].tolist()}

    def test_exits_keep_caps_and_stopped_names_out(self):
        """Salidas tras topes: renormalizar inflaba el sleeve sobre el tope."""
        w0, _ = self._target_with_exits([])
        assert w0[["CO_F1", "CO_F2", "CO_E1"]].sum() == pytest.approx(0.30)
        top_us = w0.filter(like="US").idxmax()
        w, meta = self._target_with_exits([top_us])
        assert meta["exits"] == [top_us]
        assert w[top_us] == 0.0
        assert w[["CO_F1", "CO_F2", "CO_E1"]].sum() <= 0.30 + 1e-9
        assert w.sum() == pytest.approx(1.0)

    def test_exits_closing_everything_stay_in_cash(self):
        w, _ = self._target_with_exits(list(_synthetic_mixed()[0].columns))
        assert (w == 0.0).all()

    @pytest.mark.parametrize("groups", [None, {}])
    @pytest.mark.parametrize("opt_method", ["mv_classic", "robust", "hrp"])
    def test_backtest_caps_without_groups_rejected_upfront(
        self, opt_method, groups,
    ):
        # Same config error on every optimizer path, raised before the loop:
        # a plain ValueError, not a per-rebalance failure (flat NAV, or a
        # RuntimeError chained to it) nor HRP's silent equal-weight.
        prices, volumes = _synthetic_mixed()
        cfg = BacktestConfig(
            opt_method=opt_method, window=30, warmup=30,
            rebalance_freq="M", seed=42,
            groups=groups, group_caps={"G": 0.5},
        )
        with pytest.raises(ValueError, match="group_caps requires `groups`"):
            run_backtest(prices, volumes, cfg)

    @pytest.mark.parametrize("ledger", [False, True], ids=["classic", "ledger"])
    @pytest.mark.parametrize("method", ["mv_classic", "hrp"])
    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "lenient"])
    def test_infeasible_caps_fail_rebalance_for_all_methods(
        self, method, ledger, strict,
    ):
        """Topes infactibles: el rebalanceo falla igual en mv y HRP (nunca un
        fallback equiponderado que los incumple). strict -> RebalanceError en
        el primer rebalanceo; no estricto -> infactibles en todas las fechas,
        todos fallan -> RuntimeError. Ambos encadenados al de los topes."""
        from src.backtest_engine import RebalanceError
        prices, volumes = _synthetic_mixed()
        cfg = BacktestConfig(
            opt_method=method, window=30, warmup=30,
            rebalance_freq="M", seed=42,
            groups=_mixed_groups(prices.columns),
            # cubren todo el universo y suman 0.8 < 1
            group_caps={"us/Tech": 0.5, "co/Financials": 0.1,
                        "co/Energy": 0.1, "co/Utilities": 0.1},
            use_position_ledger=ledger,
            strict=strict,
        )
        expected = RebalanceError if strict else RuntimeError
        match = "stage 'targets'" if strict else "All .* rebalance"
        with pytest.raises(expected, match=match) as exc:
            run_backtest(prices, volumes, cfg)
        assert "nfactible" in str(exc.value.__cause__)


class TestRiskProfiles:
    def test_known_profiles(self):
        assert set(RISK_PROFILES) == {"conservador", "moderado", "agresivo"}
        assert profile_config("moderado").kappa == 0.10
        assert profile_config("conservador").max_weight == 0.15
        assert profile_config("agresivo").opt_method == "mv_classic"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="desconocido"):
            profile_config("temerario")

    def test_overrides(self):
        cfg = profile_config("moderado", max_weight=0.10)
        assert cfg.max_weight == 0.10

    def test_mixed_sleeve_config(self):
        cfg = mixed_sleeve_config("moderado", universe_path="universe_mixed.csv")
        assert len(cfg.groups) == 120
        assert cfg.groups["EC"] == "co/Energy"
        assert cfg.group_caps[("co/Financials", "co/Energy")] == pytest.approx(0.10)
        cfg30 = mixed_sleeve_config("moderado", universe_path="universe_mixed.csv",
                                    co_sleeve_weight=0.30)
        assert cfg30.group_caps[("co/Financials", "co/Energy")] == pytest.approx(0.12)
        co_groups = {g for g in cfg.groups.values() if g.startswith("co/")}
        assert set(cfg.max_weight_by_group) == co_groups
        assert set(cfg.max_weight_by_group.values()) == {0.15}

    def test_mixed_sleeve_universe_name_resolves_from_repo_root(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.chdir(tmp_path)  # nunca relativo al cwd
        cfg = mixed_sleeve_config("moderado", universe_path="universe_mixed.csv")
        assert len(cfg.groups) == 120
        assert len(mixed_sleeve_config("moderado").groups) == 120
