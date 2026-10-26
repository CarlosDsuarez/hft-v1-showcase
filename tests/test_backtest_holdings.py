"""
test_backtest_holdings.py — The classic backtest never loses holdings when
the active universe changes.

Bug de origen: ``rebalance()`` reindexaba los pesos actuales a los tickers
del objetivo y ``run_backtest`` partía de ``w_prev`` renormalizado sobre los
activos; un activo en cartera que salía del universo activo se evaporaba:
sin venta, sin costo, su peso repartido. El libro de la ruta clásica es
ahora un ``PortfolioState`` (acciones + caja) y los pesos se reportan con
la caja explícita (columna ``CASH_COL``).

Offline/sintético.
"""

import numpy as np
import pandas as pd
import pytest

import src.backtest_engine as be
from src.backtest_engine import CASH_COL, BacktestConfig, RebalanceError
from src.strategy import TargetResult, prev_weights

IDX = pd.bdate_range("2024-01-02", periods=8)
D1, D4 = IDX[1], IDX[4]
NAV0 = 1_000_000.0


def _prices(x_path=None):
    x = [100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0, 120.0]
    if x_path is not None:
        x = x_path
    return pd.DataFrame(
        {"A": 100.0, "B": 100.0, "X": x}, index=IDX, dtype=float,
    )


# D1: A, B, X active; D4: X has left the active universe.
PLAN = {
    D1: (pd.Series(1 / 3, index=["A", "B", "X"]), ("A", "B", "X")),
    D4: (pd.Series(0.5, index=["A", "B"]), ("A", "B")),
}


def _fake_build_targets(state, prices, volumes, t, cfg, **_):
    w, active = PLAN[t]
    return TargetResult(
        weights=w,
        regime=None,
        exits=pd.DataFrame(columns=["ticker", "rule", "detail"]),
        meta={"expected_sharpe": 0.0, "converged": True, "method": "test"},
        active=active,
        eligible=active,
        w_prev=prev_weights(state, prices, t, list(active)),
    )


def _config(**kw):
    base = dict(
        warmup=0, window=2, rebalance_freq="15D", turnover_limit=2.0,
        commission_bps=5.0, spread_bps=2.0, impact_coef=0.0,
        replacement_rule="none", max_weight=1.0, tc_lambda=0.0,
        initial_nav=NAV0,
    )
    base.update(kw)
    return BacktestConfig(**base)


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(be, "get_rebalance_dates", lambda *a, **k: [D1, D4])
    monkeypatch.setattr(be, "build_targets", _fake_build_targets)

    def _run(prices=None, **kw):
        prices = _prices() if prices is None else prices
        volumes = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
        return be.run_backtest(prices, volumes, _config(**kw))
    return _run


class TestAssetLeavingActiveUniverse:
    def test_is_sold_with_cost(self, run):
        res = run()
        assert list(res.rebalance_log.index) == [D1, D4]
        sells = res.trades[(res.trades["date"] == D4) & (res.trades["ticker"] == "X")]
        assert len(sells) == 1
        x_value_pre = res.weights.loc[IDX[3], "X"] * res.nav.loc[IDX[3]]
        assert sells["order_value"].iloc[0] == pytest.approx(-x_value_pre)
        assert sells["cost"].iloc[0] > 0
        assert res.rebalance_log.loc[D4, "cost_fraction"] > 0

    def test_position_is_closed_not_redistributed(self, run):
        res = run()
        assert (res.weights.loc[D4:, "X"] == 0.0).all()
        # turnover = |ΔA| + |ΔB| + w_X = 2 w_X (A and B split X's sale)
        w_x = res.weights.loc[IDX[3], "X"]
        assert w_x > 0
        assert res.rebalance_log.loc[D4, "turnover"] == pytest.approx(2 * w_x)

    def test_weights_plus_cash_sum_to_one_every_date(self, run):
        res = run()
        assert CASH_COL in res.weights.columns
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-12)
        assert (res.weights[CASH_COL] >= -1e-12).all()


class TestBookIsHoldings:
    def test_weights_drift_with_prices_between_rebalances(self, run):
        """Shares are held, not weights: X +20% lifts its weight to 120/320."""
        res = run()
        assert res.weights.loc[IDX[2], "X"] == pytest.approx(1 / 3)
        assert res.weights.loc[IDX[3], "X"] == pytest.approx(120 / 320)

    def test_initial_deployment_from_cash_pays_cost(self, run):
        res = run()
        assert res.rebalance_log.loc[D1, "turnover"] == pytest.approx(1.0)
        assert res.rebalance_log.loc[D1, "cost_fraction"] > 0
        assert res.nav.loc[D1] < NAV0

    def test_held_asset_without_price_at_rebalance_is_kept(self, run):
        """Unpriced holding can't be sold: the rebalance fails, X is kept."""
        x = [100.0, 100.0, 100.0, 120.0, np.nan, 120.0, 120.0, 120.0]
        with pytest.raises(RebalanceError, match="X"):
            run(prices=_prices(x))
        res = run(prices=_prices(x), strict=False)
        assert [f["date"] for f in res.failures] == [D4]
        assert (res.weights.loc[D4:, "X"] > 0).all()
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-12)


class TestLedgerPathReportsCash:
    def test_weights_plus_cash_sum_to_one_every_date(self, run):
        res = run(use_position_ledger=True)
        assert CASH_COL in res.weights.columns
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-9)
