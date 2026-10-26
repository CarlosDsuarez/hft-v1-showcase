"""
test_sleeve_report.py — Paso 5: atribucion por sleeve y reporte mensual.

Offline: datos sinteticos (sin red).
"""

import numpy as np
import pandas as pd
import pytest

from src.sleeve_attribution import (
    attribution_table,
    monthly_summary,
    sleeve_avg_weights,
    sleeve_nav,
    sleeve_returns,
)


def _mixed_prices(seed: int = 42, n: int = 200):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    us = [f"US{i}" for i in range(6)]
    co = ["CO_F1", "CO_F2", "CO_E1", "CO_U1"]
    cols = us + co
    prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for c in cols:
        drift = 0.0008 if c.startswith("US") else 0.0005
        prices[c] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, len(dates))))
    volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
    groups = {c: ("us/Tech" if c.startswith("US") else
                  {"CO_F1": "co/Financials", "CO_F2": "co/Financials",
                   "CO_E1": "co/Energy", "CO_U1": "co/Utilities"}[c])
              for c in cols}
    return prices, volumes, groups


def _backtested():
    from src.backtest_engine import BacktestConfig, run_backtest
    prices, volumes, groups = _mixed_prices()
    cfg = BacktestConfig(opt_method="robust", window=30, warmup=30,
                         rebalance_freq="M", seed=42, groups=groups,
                         group_caps={("co/Financials", "co/Energy"): 0.40})
    return run_backtest(prices, volumes, cfg), prices, groups


class TestSleeveReturns:
    def test_sleeves_partition_total(self):
        res, prices, groups = _backtested()
        contrib = sleeve_returns(res.weights, prices, groups)
        assert set(contrib.columns) == {"us", "co"}
        w = res.weights.reindex(prices.index).ffill().fillna(0.0)
        w = w.reindex(res.weights.index).fillna(0.0)
        wf = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        wf = wf.shift(1).fillna(0.0)  # r_t is earned by the book held at t-1
        manual = (wf * prices.ffill().pct_change().reindex(w.index).fillna(0.0)).sum(axis=1)
        manual.iloc[0] = 0.0
        pd.testing.assert_series_equal(contrib.sum(axis=1), manual,
                                       check_names=False)

    def test_gross_minus_net_equals_costs(self):
        # Replica exacta del engine (convencion lineal de daily_records:
        # daily_ret = portfolio_ret - cost). Nota: res.returns compone exacto
        # via NAV ((1+g)(1-c)-1) y difiere en el termino cruzado g*c (~1e-6).
        res, prices, groups = _backtested()
        contrib = sleeve_returns(res.weights, prices, groups)
        gross = contrib.sum(axis=1)
        net_lin = pd.Series({r.date: r.daily_return for r in res.daily_records})
        net_lin = net_lin.reindex(gross.index).fillna(0.0)
        costs = pd.Series({r.date: r.cost for r in res.daily_records})
        costs = costs.reindex(gross.index).fillna(0.0)
        # Incluye el dia 0: el despliegue inicial de caja paga su costo.
        assert costs.iloc[0] > 0
        pd.testing.assert_series_equal(
            gross - net_lin, costs, check_names=False, atol=1e-12,
        )

    def test_group_level_and_other(self):
        res, prices, groups = _backtested()
        contrib = sleeve_returns(res.weights, prices, groups, level="group")
        assert "co/Financials" in contrib.columns
        groups_missing = {k: v for k, v in groups.items() if not k.startswith("CO_F1")}
        contrib2 = sleeve_returns(res.weights, prices, groups_missing)
        assert "other" in contrib2.columns

    def test_errors(self):
        res, prices, groups = _backtested()
        with pytest.raises(ValueError, match="no comparten tickers"):
            sleeve_returns(res.weights, prices.rename(columns=lambda c: "X" + c), groups)
        with pytest.raises(ValueError, match="level"):
            sleeve_returns(res.weights, prices, groups, level="region")


class TestSleeveAggregations:
    def test_nav_starts_at_one(self):
        res, prices, groups = _backtested()
        nav = sleeve_nav(sleeve_returns(res.weights, prices, groups))
        assert ((nav.iloc[0] - 1.0).abs() < 1e-12).all()

    def test_monthly_compounds_to_total(self):
        res, prices, groups = _backtested()
        contrib = sleeve_returns(res.weights, prices, groups)
        monthly = monthly_summary(contrib)
        assert {"total", "best_sleeve", "worst_sleeve"}.issubset(monthly.columns)
        total = float((1 + contrib.sum(axis=1)).prod() - 1)
        from_monthly = float((1 + monthly["total"]).prod() - 1)
        assert abs(total - from_monthly) < 1e-9

    def test_avg_weights_sum_to_one(self):
        res, prices, groups = _backtested()
        avg = sleeve_avg_weights(res.weights, groups)
        assert abs(avg.sum() - 1.0) < 1e-9
        assert set(avg.index) == {"us", "co", "cash"}
        assert abs(avg["cash"]) < 1e-9

    def test_attribution_table(self):
        res, prices, groups = _backtested()
        contrib = sleeve_returns(res.weights, prices, groups)
        table = attribution_table(contrib, sleeve_avg_weights(res.weights, groups))
        assert set(table.index) == {"us", "co"}
        assert abs(table["share_of_return"].sum() - 1.0) < 1e-9


class TestBuildReport:
    def test_builds_html(self, tmp_path):
        from src.monthly_report import build_report
        res, prices, groups = _backtested()
        out = build_report({"robusto": res}, prices, groups,
                           output_path=str(tmp_path / "report.html"))
        html = open(out, encoding="utf-8").read()
        for token in ["robusto", "co", "us", "Atribucion", "mensual", "Sharpe"]:
            assert token in html

    def test_empty_results_raises(self, tmp_path):
        from src.monthly_report import build_report
        _, prices, groups = _backtested()
        with pytest.raises(ValueError, match="vacio"):
            build_report({}, prices, groups,
                         output_path=str(tmp_path / "report.html"))
