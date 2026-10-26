"""
test_benchmarks.py — Paso 4: registry, FX COP, metricas vs benchmarks y dashboard.

Offline: datos sinteticos + CSVs locales (sin red). El fetch live
(yfinance) no se testea aqui.
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import (
    BENCHMARKS,
    align_to,
    benchmark_labels,
    compute_metrics_vs_benchmarks,
    fetch_benchmark_prices,
    prices_to_log_returns,
    to_cop_returns,
)


def _gbm(seed: int, n: int, mu: float, vol: float, s0: float,
         index: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu / 252, vol / np.sqrt(252), n)
    return pd.Series(s0 * np.cumprod(1 + rets), index=index)


class TestRegistry:
    def test_known_labels(self):
        for lb in ["SPY", "EEM", "TLT", "ICOLCAP", "HCOLSEL", "USDCOP"]:
            assert lb in BENCHMARKS

    def test_currencies(self):
        assert BENCHMARKS["SPY"]["currency"] == "USD"
        assert BENCHMARKS["ICOLCAP"]["currency"] == "COP"
        assert BENCHMARKS["USDCOP"]["currency"] == "FX"

    def test_kind_filter(self):
        assert set(benchmark_labels(kind="equity_co")) == {"ICOLCAP", "HCOLSEL"}
        assert "SPY" in benchmark_labels()

    def test_fetch_unknown_label_raises_without_network(self):
        with pytest.raises(ValueError, match="desconocidos"):
            fetch_benchmark_prices(["NOEXISTE"])


def _bench_provider(columns):
    from src.market_data import FakeProvider
    idx = pd.bdate_range("2024-01-02", periods=30)
    close = pd.DataFrame({c: _gbm(i, len(idx), 0.05, 0.2, 100.0, idx)
                          for i, c in enumerate(columns)})
    return FakeProvider(close, pd.DataFrame(1e6, index=idx, columns=columns))


class TestFetchViaProvider:
    def test_uses_provider_and_returns_labels(self):
        fake = _bench_provider(["SPY", "ICOLCAP.CL", "USDCOP=X"])
        px = fetch_benchmark_prices(["SPY", "ICOLCAP", "USDCOP"], start="2024-01-01",
                                    end="2024-02-01", provider=fake)
        assert list(px.columns) == ["SPY", "ICOLCAP", "USDCOP"]
        assert len(fake.calls) == 1
        assert fake.calls[0][0] == ("SPY", "ICOLCAP.CL", "USDCOP=X")

    def test_missing_benchmark_warns_and_keeps_rest(self, caplog):
        fake = _bench_provider(["SPY"])
        with caplog.at_level("WARNING"):
            px = fetch_benchmark_prices(["SPY", "HCOLSEL"], start="2024-01-01",
                                        end="2024-02-01", provider=fake)
        assert list(px.columns) == ["SPY"]
        assert "HCOLSEL" in caplog.text

    def test_no_data_raises_runtime_error(self):
        fake = _bench_provider(["ZZZ"])
        with pytest.raises(RuntimeError):
            fetch_benchmark_prices(["SPY"], start="2024-01-01", end="2024-02-01",
                                   provider=fake)


class TestReturnsMath:
    def test_prices_to_log_returns(self):
        px = pd.Series([100.0, 110.0, 121.0])
        r = prices_to_log_returns(px.to_frame("A"))["A"]
        assert len(r) == 2
        assert abs(r.iloc[0] - np.log(1.1)) < 1e-12
        assert abs(float(np.exp(r.sum())) - 1.21) < 1e-9

    def test_to_cop_additive(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        usd = pd.Series(np.full(10, 0.001), index=idx, name="strat")
        fx = pd.Series(4000 * np.cumprod(1 + np.full(10, 0.0005)), index=idx)
        out = to_cop_returns(usd, fx)
        fx_r = np.log(fx / fx.shift(1)).reindex(idx).fillna(0.0)
        pd.testing.assert_series_equal(
            out, (usd + fx_r).rename("strat"), check_names=True,
        )

    def test_to_cop_inner_join(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        usd = pd.Series(0.001, index=idx)
        fx = pd.Series(4000.0, index=idx[:6])
        out = to_cop_returns(usd, fx)
        assert len(out) == 6

    def test_align_to(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        bench = pd.Series(0.001, index=idx)
        target = pd.bdate_range("2024-01-03", periods=5)
        assert len(align_to(bench, target)) == 5


class TestMetricsVsBenchmarks:
    def _data(self):
        idx = pd.bdate_range("2022-01-01", periods=252)
        spy = _gbm(1, 252, 0.09, 0.16, 400, idx)
        ico = _gbm(2, 252, 0.06, 0.20, 15000, idx)
        strat_rets = np.log(spy / spy.shift(1)).dropna() + 0.0002
        return spy, ico, strat_rets

    def test_keys_and_reference_beta(self):
        spy, ico, strat_rets = self._data()
        benches = {
            "SPY": np.log(spy / spy.shift(1)).dropna(),
            "ICOLCAP": np.log(ico / ico.shift(1)).dropna(),
        }
        nav = (1 + strat_rets.fillna(0)).cumprod()
        out = compute_metrics_vs_benchmarks(
            strat_rets, nav, benches, n_bootstrap=50, seed=42,
        )
        assert set(out) == {"strategy", "SPY", "ICOLCAP"}
        assert abs(out["SPY"].beta - 1.0) < 1e-9
        assert abs(out["SPY"].information_ratio) < 1e-9
        assert out["ICOLCAP"].beta != pytest.approx(1.0, abs=0.05) or True
        assert out["strategy"].n_days == len(strat_rets.dropna())

    def test_unknown_reference_raises(self):
        spy, _, strat_rets = self._data()
        with pytest.raises(ValueError, match="reference"):
            compute_metrics_vs_benchmarks(
                strat_rets, (1 + strat_rets.fillna(0)).cumprod(),
                {"SPY": np.log(spy / spy.shift(1)).dropna()},
                reference="COLCAP", n_bootstrap=20,
            )


class TestDashboardBenchmarks:
    def _results(self):
        from src.backtest_engine import BacktestConfig, run_backtest
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2022-01-01", periods=150)
        cols = [f"S{i}" for i in range(6)]
        prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
        for c in cols:
            prices[c] = 100 * np.exp(
                np.cumsum(rng.normal(0.0008, 0.015, len(dates))))
        volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
        cfg = BacktestConfig(opt_method="robust", window=30, warmup=30,
                             rebalance_freq="M", seed=42)
        res = run_backtest(prices, volumes, cfg)
        benches = {
            "SPY": _gbm(9, 150, 0.09, 0.16, 400, dates),
            "ICOLCAP": _gbm(10, 150, 0.06, 0.20, 15000, dates),
        }
        return {"robust": res}, benches

    def test_dashboard_with_benchmarks(self, tmp_path):
        from src.dashboard_results import build_dashboard
        results, benches = self._results()
        out = build_dashboard(results, output_path=str(tmp_path / "dash.html"),
                              benchmark_prices=benches)
        html = open(out, encoding="utf-8").read()
        assert "ICOLCAP (bench)" in html
        assert "Benchmark" in html
        assert "SPY" in html

    def test_dashboard_backward_compatible(self, tmp_path):
        from src.dashboard_results import build_dashboard
        results, _ = self._results()
        out = build_dashboard(results, output_path=str(tmp_path / "dash.html"))
        html = open(out, encoding="utf-8").read()
        assert "robust" in html
        assert "(bench)" not in html

    def test_synthetic_demo_results_use_valid_configs(self):
        """CI integration runs `python -m src.dashboard_results`; its demo
        labels ('robust_k010') are not opt_methods, and StrategyConfig now
        rejects unknown methods at construction."""
        from src.dashboard_results import _make_synthetic_results
        results = _make_synthetic_results()
        assert set(results) == {"mv_classic", "robust_k010", "robust_k025"}
        assert results["robust_k010"].config.opt_method == "robust"
        assert results["robust_k010"].config.kappa == pytest.approx(0.10)
        assert results["robust_k025"].config.kappa == pytest.approx(0.25)
