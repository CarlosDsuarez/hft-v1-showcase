"""
test_exit_rules.py — Criterios de cierre total (stop, trailing, exclusion).

Offline: datos sinteticos (sin red).
"""

import numpy as np
import pandas as pd
import pytest

from src.exit_rules import (
    ExitConfig,
    apply_exits_to_weights,
    evaluate_exits,
    stop_loss_exits,
    take_profit_exits,
    trailing_stop_exits,
    universe_exits,
)


class TestStopLoss:
    def test_triggers_below_threshold(self):
        px = pd.Series({"A": 90.0, "B": 95.0, "C": 100.0})
        out = stop_loss_exits(px, {"A": 100.0, "B": 100.0, "C": 100.0}, 0.08)
        assert out["ticker"].tolist() == ["A"]
        assert out["rule"].tolist() == ["stop_loss"]

    def test_boundary_exact(self):
        px = pd.Series({"A": 91.99})
        out = stop_loss_exits(px, {"A": 100.0}, 0.08)
        assert out["ticker"].tolist() == ["A"]  # <= -pct dispara

    def test_skips_bad_data(self):
        px = pd.Series({"A": float("nan"), "B": 50.0})
        out = stop_loss_exits(px, {"A": 100.0, "B": 0.0, "C": 100.0}, 0.08)
        assert out.empty  # A NaN, B entrada 0, C sin precio


class TestTakeProfit:
    def test_triggers_above_threshold(self):
        px = pd.Series({"A": 102.0, "B": 101.0})
        out = take_profit_exits(px, {"A": 100.0, "B": 100.0}, 0.02)
        assert out["ticker"].tolist() == ["A"]
        assert out["rule"].tolist() == ["take_profit"]

    def test_scalp_band(self):
        # stop 1% no dispara, take 2% si: la banda 1%/2% del usuario
        px = pd.Series({"A": 102.5})
        assert take_profit_exits(px, {"A": 100.0}, 0.02).shape[0] == 1
        assert stop_loss_exits(px, {"A": 100.0}, 0.01).empty


class TestTrailingStop:
    def _prices(self, tail_vals, n=70):
        dates = pd.bdate_range("2024-01-01", periods=n)
        base = np.linspace(100, 110, n)
        base[-len(tail_vals):] = tail_vals
        return pd.DataFrame({"A": base, "B": np.full(n, 100.0)}, index=dates)

    def test_triggers_on_drawdown(self):
        px = self._prices([110.0, 90.0])  # -18% del pico
        t = px.index[-1]
        out = trailing_stop_exits(px, t, ["A", "B"], 0.12, window=63)
        assert out["ticker"].tolist() == ["A"]

    def test_no_trigger_on_recovery(self):
        px = self._prices([110.0, 90.0, 100.0])  # -9% del pico
        t = px.index[-1]
        out = trailing_stop_exits(px, t, ["A", "B"], 0.12, window=63)
        assert out.empty

    def test_insufficient_history(self):
        px = pd.DataFrame({"A": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1))
        out = trailing_stop_exits(px, px.index[-1], ["A"], 0.12)
        assert out.empty


class TestUniverseExits:
    def test_excluded_tickers(self):
        out = universe_exits(["A", "B", "C"], ["A", "C"])
        assert out["ticker"].tolist() == ["B"]

    def test_none_excluded(self):
        assert universe_exits(["A"], ["A", "B"]).empty


class TestEvaluate:
    def test_combined_and_dedup(self):
        dates = pd.bdate_range("2024-01-01", periods=70)
        crash = np.linspace(100, 110, 70)
        crash[-1] = 80.0  # -27% pico y -20% vs entrada 100
        px = pd.DataFrame({"A": crash, "B": np.full(70, 100.0)}, index=dates)
        t = dates[-1]
        out = evaluate_exits(t, px, ["A", "B"], {"A": 100.0, "B": 100.0},
                             active=["A", "B"],
                             config=ExitConfig(stop_loss_pct=0.08,
                                               trailing_stop_pct=0.12,
                                               exit_on_illiquid=True))
        assert out["ticker"].tolist() == ["A"]  # una fila aunque 2 reglas
        assert out["rule"].tolist() == ["stop_loss"]  # primera regla gana

    def test_empty_held(self):
        px = pd.DataFrame({"A": [1.0, 2.0]})
        assert evaluate_exits(px.index[-1], px, [], {}, active=["A"]).empty

    def test_all_disabled(self):
        dates = pd.bdate_range("2024-01-01", periods=70)
        px = pd.DataFrame({"A": np.full(70, 50.0)}, index=dates)
        cfg = ExitConfig(stop_loss_pct=None, trailing_stop_pct=None,
                         take_profit_pct=None, exit_on_illiquid=False)
        assert cfg.active_rules() == []
        out = evaluate_exits(dates[-1], px, ["A"], {"A": 100.0},
                             active=[], config=cfg)
        assert out.empty

    def test_defaults_scalp_band(self):
        assert ExitConfig().stop_loss_pct == 0.01
        assert ExitConfig().take_profit_pct == 0.02
        assert ExitConfig().trailing_stop_pct is None


class TestApply:
    def test_zero_and_renormalize(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        exits = pd.DataFrame([{"ticker": "A", "rule": "stop_loss", "detail": ""}])
        out = apply_exits_to_weights(w, exits)
        assert out["A"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-12
        assert abs(out["B"] - 0.6) < 1e-12

    def test_no_exits_passthrough(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        pd.testing.assert_series_equal(
            apply_exits_to_weights(w, pd.DataFrame()), w)

    def test_all_closed_zeros(self):
        w = pd.Series({"A": 1.0})
        exits = pd.DataFrame([{"ticker": "A", "rule": "stop_loss", "detail": ""}])
        assert (apply_exits_to_weights(w, exits) == 0.0).all()


class TestBacktestWiring:
    def test_disabled_by_default(self):
        from src.backtest_engine import BacktestConfig
        assert BacktestConfig().use_exit_rules is False

    def test_stop_closes_crasher(self):
        from src.backtest_engine import BacktestConfig, run_backtest
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2022-01-01", periods=250)
        cols = ["GOOD%d" % i for i in range(5)] + ["CRASH"]
        prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
        for c in cols:
            prices[c] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(dates))))
        # Crash -60% a mitad del periodo (se mantiene abajo; varios
        # rebalanceos mensuales caen dentro del crash)
        crash_px = prices["CRASH"].iloc[125] * np.linspace(1.0, 0.4, 125)
        prices.iloc[125:, prices.columns.get_loc("CRASH")] = crash_px
        volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
        # HRP sin filtro de liquidez: CRASH sigue en cartera al empezar la
        # caída, así el cierre lo dispara el stop (no el optimizador ni el
        # filtro). Antes el backtest "cerraba" CRASH ya vendido: las
        # entradas nunca se borraban (salida fantasma + KeyError).
        cfg = BacktestConfig(opt_method="hrp", window=30, warmup=30,
                             rebalance_freq="M", seed=42, use_exit_rules=True,
                             universe_filter=False, max_weight=1.0,
                             stop_loss_pct=0.08, trailing_stop_pct=None,
                             take_profit_pct=None, exit_on_illiquid=False,
                             turnover_limit=2.0)  # sin blending: el cierre ejecuta total
        res = run_backtest(prices, volumes, cfg)  # strict: any failure raises
        assert res.failures == []
        hit = res.rebalance_log[res.rebalance_log["exits"].str.contains("CRASH")]
        assert not hit.empty
        assert (hit.index >= prices.index[125]).all()  # solo tras el crash
        # Cada stop cierra CRASH completo en ese rebalanceo
        assert (res.weights.loc[hit.index, "CRASH"] == 0.0).all()

    def test_no_exits_column_when_disabled(self):
        from src.backtest_engine import BacktestConfig, run_backtest
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2022-01-01", periods=100)
        cols = ["S%d" % i for i in range(5)]
        prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
        for c in cols:
            prices[c] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(dates))))
        volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
        res = run_backtest(prices, volumes, BacktestConfig(
            opt_method="mv_classic", window=30, warmup=30,
            rebalance_freq="M", seed=42))
        assert (res.rebalance_log["exits"] == "").all()
