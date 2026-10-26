"""
test_portfolio_state.py — PortfolioState: vista unica de la cartera, sea el
ledger local (sim) o la cuenta del broker. Sin red: el broker es un doble.
"""

import pandas as pd
import pytest

from src.portfolio_state import PortfolioState, position_drift
from src.position_ledger import PositionLedger

T = pd.Timestamp("2026-09-28")
PX = pd.Series({"AAA": 10.0, "BBB": 20.0, "CCC": 50.0})


class FakeBroker:
    """Doble de lectura de AlpacaClient (sin HTTP); registra llamadas."""

    def __init__(self, account, positions):
        self.account = account
        self.positions = positions
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        return dict(self.account)

    def get_positions(self):
        self.calls.append("get_positions")
        return self.positions.copy()


def _positions(rows):
    """rows: (ticker, qty, avg_entry); market_value marcado a PX."""
    return pd.DataFrame(
        [{"ticker": tk, "qty": q, "market_value": q * PX[tk],
          "avg_entry": e, "side": "long"} for tk, q, e in rows],
        columns=["ticker", "qty", "market_value", "avg_entry", "side"])


def _ledger():
    # cash = 5000 - 100*9 - 50*20 = 3100; NAV a PX = 3100 + 1000 + 1000 = 5100
    lg = PositionLedger(initial_cash=5_000.0)
    lg.apply_trades(pd.DataFrame([{"ticker": "AAA", "shares": 100.0},
                                  {"ticker": "BBB", "shares": 50.0}]),
                    pd.Series({"AAA": 9.0, "BBB": 20.0}), t=T)
    return lg


def _broker(cash="4000"):
    account = {"equity": "10000"} if cash is None else {"equity": "10000", "cash": cash}
    return FakeBroker(account, _positions([("AAA", 100.0, 11.0),
                                           ("CCC", 100.0, 48.0)]))


class TestFromLedger:
    def test_mirrors_ledger_book(self):
        st = PortfolioState.from_ledger(_ledger(), PX, as_of=T)
        assert st.source == "ledger"
        assert st.as_of == T
        assert st.positions.to_dict() == {"AAA": 100.0, "BBB": 50.0}
        assert st.avg_entry.to_dict() == {"AAA": 9.0, "BBB": 20.0}
        assert st.cash == pytest.approx(3_100.0)
        assert st.equity == pytest.approx(5_100.0)

    def test_missing_mark_for_held_ticker_raises(self):
        with pytest.raises(ValueError, match="AAA"):
            PortfolioState.from_ledger(_ledger(), PX.drop("AAA"), as_of=T)


class TestFromBroker:
    def test_reads_only_account_and_positions(self):
        fb = _broker()
        st = PortfolioState.from_broker(fb)
        assert st.source == "broker"
        assert st.positions.to_dict() == {"AAA": 100.0, "CCC": 100.0}
        assert st.avg_entry.to_dict() == {"AAA": 11.0, "CCC": 48.0}
        assert st.cash == 4_000.0
        assert st.equity == 10_000.0
        assert sorted(fb.calls) == ["get_account", "get_positions"]

    def test_flat_account(self):
        fb = FakeBroker({"equity": "10000", "cash": "10000"}, _positions([]))
        st = PortfolioState.from_broker(fb)
        assert st.positions.empty
        assert st.exit_inputs() == ([], {}, {})

    @pytest.mark.parametrize("short_qty", [5.0, -5.0])
    def test_short_side_is_negative_whatever_qty_sign(self, short_qty):
        pos = _positions([("AAA", 100.0, 11.0), ("CCC", short_qty, 48.0)])
        pos.loc[pos["ticker"] == "CCC", "side"] = "short"
        st = PortfolioState.from_broker(
            FakeBroker({"equity": "10000", "cash": "4000"}, pos))
        assert st.positions.to_dict() == {"AAA": 100.0, "CCC": -5.0}
        assert st.exit_inputs()[0] == ["AAA"]
        assert st.to_ledger().positions().to_dict() == {"AAA": 100.0}

    def test_missing_cash_falls_back_to_equity_minus_market_value(self):
        # 10000 - (100*10 + 100*50) = 4000
        assert PortfolioState.from_broker(_broker(cash=None)).cash == 4_000.0


class TestConsumers:
    def test_exit_inputs_long_only_with_valid_entry(self):
        st = PortfolioState(
            positions=pd.Series({"AAA": 10.0, "BBB": -5.0, "CCC": 3.0}),
            avg_entry=pd.Series({"AAA": 9.0, "BBB": 20.0, "CCC": 0.0}),
            cash=0.0, equity=1.0, as_of=T, source="broker")
        held, entries, shares = st.exit_inputs()
        assert held == ["AAA", "CCC"]
        assert entries == {"AAA": 9.0}
        assert shares == {"AAA": 10.0, "CCC": 3.0}

    def test_weights_match_ledger(self):
        lg = _ledger()
        st = PortfolioState.from_ledger(lg, PX, as_of=T)
        pd.testing.assert_series_equal(st.weights(PX),
                                       lg.weights_from_positions(PX))

    def test_broker_weights_use_broker_equity(self):
        w = PortfolioState.from_broker(_broker()).weights(PX)
        assert w.to_dict() == pytest.approx({"AAA": 0.1, "CCC": 0.5})

    def test_unrealized_pnl_matches_ledger_snapshot(self):
        lg = _ledger()
        st = PortfolioState.from_ledger(lg, PX, as_of=T)
        assert st.unrealized_pnl(PX) == pytest.approx(
            lg.snapshot(T, PX).unrealized_pnl)


class TestToLedger:
    def test_broker_mirror_has_broker_book_and_no_history(self):
        st = PortfolioState.from_broker(_broker())
        mirror = st.to_ledger()
        back = PortfolioState.from_ledger(mirror, PX, as_of=T)
        assert back.positions.to_dict() == st.positions.to_dict()
        assert back.avg_entry.to_dict() == pytest.approx(st.avg_entry.to_dict())
        assert back.cash == st.cash
        assert mirror.transactions_df().empty
        assert mirror.realized_pnl == 0.0


class TestPositionDrift:
    def test_lists_tickers_whose_shares_differ(self):
        local = PortfolioState.from_ledger(_ledger(), PX, as_of=T)   # AAA 100, BBB 50
        broker = PortfolioState.from_broker(_broker())               # AAA 100, CCC 100
        drift = position_drift(local, broker)
        assert list(drift.columns) == ["ticker", "local_shares",
                                       "broker_shares", "diff"]
        assert drift.set_index("ticker")["diff"].to_dict() == {
            "BBB": -50.0, "CCC": 100.0}

    def test_empty_when_books_match(self):
        broker = PortfolioState.from_broker(_broker())
        local = PortfolioState.from_ledger(broker.to_ledger(), PX, as_of=T)
        assert position_drift(local, broker).empty
