"""
test_position_ledger.py — Tests de T-32: PositionLedger + freq '<N>D'.

Cubre:
  - FIFO P&L realizado
  - NAV mark-to-market
  - Rebalanceo pesos → shares
  - Reconstrucción de historial vía transactions_df
  - get_rebalance_dates con '15D' (días hábiles)

Restricciones validadas: R-01, R-05, R-10.
"""

import numpy as np
import pandas as pd
import pytest

from src.position_ledger import PositionLedger, LedgerSnapshot
from src.rolling_engine import get_rebalance_dates

SEED = 42


@pytest.fixture
def prices_row():
    return pd.Series({"A": 100.0, "B": 50.0, "C": 25.0})


# ===========================================================================
# FIFO P&L
# ===========================================================================
class TestFIFOPnL:
    def test_fifo_realized_pnl(self, prices_row):
        """Sell after two buys at different prices → FIFO cost basis."""
        ledger = PositionLedger(initial_cash=100_000.0)
        t0 = pd.Timestamp("2022-01-03")
        t1 = pd.Timestamp("2022-01-04")
        t2 = pd.Timestamp("2022-01-05")

        # Buy 10 A @ 100
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            prices_row, t=t0,
        )
        # Buy 10 A @ 110
        px1 = prices_row.copy()
        px1["A"] = 110.0
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            px1, t=t1,
        )
        # Sell 15 A @ 120 → 10*(120-100) + 5*(120-110) = 200 + 50 = 250
        px2 = prices_row.copy()
        px2["A"] = 120.0
        pnl = ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": -15.0}]),
            px2, t=t2,
        )
        assert abs(pnl - 250.0) < 1e-8
        assert abs(ledger.realized_pnl - 250.0) < 1e-8
        # Remaining: 5 shares from second lot @ 110
        pos = ledger.positions()
        assert abs(pos["A"] - 5.0) < 1e-8

    def test_sell_more_than_held_raises(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 5.0}]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        with pytest.raises(ValueError):
            ledger.apply_trades(
                pd.DataFrame([{"ticker": "A", "shares": -10.0}]),
                prices_row, t=pd.Timestamp("2022-01-04"),
            )


# ===========================================================================
# NAV
# ===========================================================================
class TestNAV:
    def test_nav_equals_cash_plus_positions(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([
                {"ticker": "A", "shares": 10.0},
                {"ticker": "B", "shares": 20.0},
            ]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        # Spent: 10*100 + 20*50 = 2000; cash = 8000
        # MTM: 8000 + 1000 + 1000 = 10000
        nav = ledger.nav(prices_row)
        assert abs(nav - 10_000.0) < 1e-6
        assert abs(ledger.cash - 8_000.0) < 1e-6

    def test_nav_tracks_price_changes(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        px = prices_row.copy()
        px["A"] = 110.0
        # cash 9000 + 10*110 = 10100
        assert abs(ledger.nav(px) - 10_100.0) < 1e-6

    @pytest.mark.parametrize("marks", [
        pd.Series({"B": 50.0}),            # held ticker missing
        pd.Series({"A": np.nan}),
        pd.Series({"A": np.inf}),
        pd.Series({"A": 0.0}),
        pd.Series({"A": -1.0}),
    ])
    def test_invalid_mark_for_held_ticker_raises(self, prices_row, marks):
        """Missing/invalid marks must not silently value holdings at 0."""
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        with pytest.raises(ValueError, match="A"):
            ledger.nav(marks)
        with pytest.raises(ValueError, match="A"):
            ledger.snapshot(pd.Timestamp("2022-01-04"), marks)
        with pytest.raises(ValueError, match="A"):
            ledger.weights_from_positions(marks)

    def test_invalid_mark_for_unheld_ticker_is_ignored(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        marks = pd.Series({"A": 110.0, "B": np.nan})
        assert abs(ledger.nav(marks) - 10_100.0) < 1e-6


# ===========================================================================
# Rebalance weights → shares
# ===========================================================================
class TestWeightTranslation:
    def test_trades_from_target_weights(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        tw = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        trades = ledger.trades_from_target_weights(tw, prices_row, nav=10_000.0)
        ledger.apply_trades(trades, prices_row, t=pd.Timestamp("2022-01-03"))

        # Target: A=50 shares, B=60, C=80
        pos = ledger.positions()
        assert abs(pos["A"] - 50.0) < 1e-8
        assert abs(pos["B"] - 60.0) < 1e-8
        assert abs(pos["C"] - 80.0) < 1e-8
        # Cash residual ≈ 0
        assert abs(ledger.cash) < 1e-6

    def test_rebalance_to_new_weights(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        tw1 = pd.Series({"A": 1.0})
        t0 = pd.Timestamp("2022-01-03")
        trades1 = ledger.trades_from_target_weights(tw1, prices_row, nav=10_000.0)
        ledger.apply_trades(trades1, prices_row, t=t0)

        tw2 = pd.Series({"B": 1.0})
        t1 = pd.Timestamp("2022-01-04")
        nav = ledger.nav(prices_row)
        trades2 = ledger.trades_from_target_weights(tw2, prices_row, nav=nav)
        ledger.apply_trades(trades2, prices_row, t=t1)

        pos = ledger.positions()
        assert "A" not in pos.index or abs(pos.get("A", 0.0)) < 1e-8
        assert abs(pos["B"] - (nav / 50.0)) < 1e-6

    @pytest.mark.parametrize("bad_px", [None, np.nan, 0.0])
    def test_held_ticker_without_valid_price_raises(self, prices_row, bad_px):
        """A held name with no usable price cannot be silently left unsized."""
        ledger = PositionLedger(initial_cash=10_000.0)
        ledger.apply_trades(
            pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
            prices_row, t=pd.Timestamp("2022-01-03"),
        )
        px = prices_row.drop("A") if bad_px is None else prices_row.copy()
        if bad_px is not None:
            px["A"] = bad_px
        with pytest.raises(ValueError, match="A"):
            ledger.trades_from_target_weights(
                pd.Series({"B": 1.0}), px, nav=10_000.0,
            )

    def test_unheld_ticker_without_price_is_skipped(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        px = prices_row.drop("C")
        trades = ledger.trades_from_target_weights(
            pd.Series({"A": 0.5, "C": 0.5}), px, nav=10_000.0,
        )
        assert trades["ticker"].tolist() == ["A"]


# ===========================================================================
# Snapshot + transaction history reconstruction
# ===========================================================================
class TestHistoryReconstruction:
    def test_snapshot_and_transactions(self, prices_row):
        ledger = PositionLedger(initial_cash=10_000.0)
        t0 = pd.Timestamp("2022-01-03")
        tw = pd.Series({"A": 0.6, "B": 0.4})
        trades = ledger.trades_from_target_weights(
            tw, prices_row, nav=10_000.0, costs=5.0,
        )
        ledger.apply_trades(trades, prices_row, costs=5.0, t=t0)

        snap = ledger.snapshot(t0, prices_row)
        assert isinstance(snap, LedgerSnapshot)
        assert snap.t == t0
        assert snap.nav > 0
        assert snap.realized_pnl == 0.0  # no sells yet

        txn = ledger.transactions_df()
        assert len(txn) >= 2  # buys + cost row
        assert set(txn.columns) >= {
            "t", "ticker", "shares", "price", "proceeds", "realized_pnl", "cost"
        }
        # Reconstruct: sum of buy proceeds + cost ≈ -initial spend
        buy_rows = txn[txn["ticker"] != "__COST__"]
        assert (buy_rows["shares"] > 0).all()


# ===========================================================================
# get_rebalance_dates '<N>D'
# ===========================================================================
class TestRebalanceFreqND:
    @pytest.fixture
    def long_prices(self):
        np.random.seed(SEED)
        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        arr = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, (500, 3)), axis=0))
        return pd.DataFrame(arr, index=dates, columns=["X", "Y", "Z"])

    def test_15d_spacing_is_trading_days(self, long_prices):
        dates = get_rebalance_dates(long_prices, freq="15D", warmup=252)
        assert len(dates) > 0
        # First date must have >= warmup history
        assert (long_prices.index <= dates[0]).sum() >= 252
        # Spacing between consecutive dates == 15 trading days
        idx = long_prices.index
        for a, b in zip(dates[:-1], dates[1:]):
            i = idx.get_loc(a)
            j = idx.get_loc(b)
            assert j - i == 15

    def test_monthly_still_works(self, long_prices):
        dates = get_rebalance_dates(long_prices, freq="M", warmup=252)
        assert len(dates) > 0

    def test_invalid_freq_raises(self, long_prices):
        with pytest.raises(ValueError):
            get_rebalance_dates(long_prices, freq="W", warmup=252)

    @pytest.mark.parametrize("warmup", [0, 1])
    def test_small_warmup_nd_starts_at_first_bar(self, long_prices, warmup):
        """warmup=0 must not use start_idx=-1 (dates[-1] first → collapsed OOS)."""
        dates = get_rebalance_dates(long_prices, freq="15D", warmup=warmup)
        idx = long_prices.index
        assert dates[0] == idx[0]
        assert dates == sorted(dates)
        assert len(dates) == len(range(0, len(idx), 15))
        for a, b in zip(dates[:-1], dates[1:]):
            assert idx.get_loc(b) - idx.get_loc(a) == 15

    def test_negative_warmup_nd_raises(self, long_prices):
        with pytest.raises(ValueError, match="warmup"):
            get_rebalance_dates(long_prices, freq="15D", warmup=-1)
