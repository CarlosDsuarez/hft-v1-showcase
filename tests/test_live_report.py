"""
test_live_report.py — Adaptador LiveResult desde live_state sintético (ARQ-21).

Estado escrito a mano en tmp_path (history + snapshots + transactions +
broker_orders) y, al final, generado por el runner real con precios falsos.
Sin red.
"""

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.backtest_engine import CASH_COL
from src.live_report import (
    LiveResult, load_live_result, main as live_main, mark_to_market, read_live_book,
)
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

D1, D2, D3 = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-22"), pd.Timestamp("2024-04-12")
NAV0 = 100_000.0


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-02-26", "2024-04-30")
    rng = np.random.default_rng(0)
    data = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (len(idx), 3)), axis=0))
    return pd.DataFrame(data, index=idx, columns=["AAA", "BBB", "CCC"])


def _books(px: pd.DataFrame):
    """Cartera tras cada corrida vigente (autofinanciada, sin costos)."""
    s1 = pd.Series({"AAA": 300.0, "BBB": 200.0})
    c1 = NAV0 - float((s1 * px.loc[D1, s1.index]).sum())
    s2 = pd.Series({"BBB": 250.0, "CCC": 100.0})
    trades = s2.reindex(s1.index.union(s2.index), fill_value=0.0) - s1.reindex(
        s1.index.union(s2.index), fill_value=0.0)
    c2 = c1 - float((trades * px.loc[D2, trades.index]).sum())
    return [(D1, "run-1", s1, c1), (D2, "run-3", s2, c2)]


def _snap_rows(date, run_id, shares, cash, px):
    price = px.loc[date, shares.index]
    rows = pd.DataFrame({"ticker": [*shares.index, CASH_COL],
                         "shares": [*shares, cash], "price": [*price, 1.0]})
    rows["value"] = rows["shares"] * rows["price"]
    rows["weight"] = rows["value"] / rows["value"].sum()
    return rows.assign(date=date.date().isoformat(), run_id=run_id, mode="sim")


def _history_row(date, run_id, nav, cash, turnover, status="ok"):
    return {"date": date.date().isoformat(), "mode": "sim", "nav": nav, "cash": cash,
            "turnover": turnover, "cost": 0.0, "run_id": run_id, "status": status}


@pytest.fixture
def state(tmp_path):
    """3 fechas: D1 ok, D2 forzada dos veces (vale la última), D3 partial."""
    px = _prices()
    (d1, r1, s1, c1), (d2, r3, s2, c2) = _books(px)
    nav1 = float((s1 * px.loc[d1, s1.index]).sum()) + c1
    nav2 = float((s2 * px.loc[d2, s2.index]).sum()) + c2
    pd.DataFrame([
        _history_row(d1, r1, nav1, c1, 0.9),
        _history_row(d2, "run-2", nav2, c2, 0.7),      # superseded por run-3
        _history_row(d2, r3, nav2, c2, 0.4),
        _history_row(D3, "run-4", nav2, c2, 0.2, status="partial"),
    ]).to_csv(tmp_path / "rebalance_history.csv", index=False)
    pd.concat([
        _snap_rows(d1, r1, s1, c1, px),
        _snap_rows(d2, "run-2", s1 * 0.5, c1, px),     # nunca debe usarse
        _snap_rows(d2, r3, s2, c2, px),
    ]).to_csv(tmp_path / "snapshots.csv", index=False)
    return tmp_path, px, {d1: (s1, c1), d2: (s2, c2)}


def _expected_nav(px, books, day):
    date = max(d for d in books if d <= day)
    shares, cash = books[date]
    return float((shares * px.loc[day, shares.index]).sum()) + cash


# ---------------------------------------------------------------------------
# Snapshots + precios diarios
# ---------------------------------------------------------------------------
class TestFromSnapshots:
    def test_daily_nav_marks_latest_book_to_prices(self, state):
        root, px, books = state
        res = load_live_result(root, prices=px)
        assert isinstance(res, LiveResult)
        assert res.book_source == "snapshots" and res.marked_daily
        assert res.nav.index[0] == D1 and res.nav.index[-1] == px.index[-1]
        for day in (D1, pd.Timestamp("2024-03-12"), D2, pd.Timestamp("2024-04-29")):
            assert res.nav[day] == pytest.approx(_expected_nav(px, books, day))

    def test_weights_include_cash_and_sum_to_one(self, state):
        root, px, _ = state
        res = load_live_result(root, prices=px)
        assert CASH_COL in res.weights.columns
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)
        assert res.weights.loc[pd.Timestamp("2024-03-12"), "CCC"] == 0.0
        assert res.weights.loc[D2, "AAA"] == 0.0 and res.weights.loc[D2, "CCC"] > 0

    def test_partial_and_superseded_runs_do_not_count(self, state):
        root, px, _ = state
        res = load_live_result(root, prices=px)
        assert list(res.holdings.index) == [D1, D2]
        assert res.rebalance_log["run_id"].tolist() == ["run-1", "run-3"]
        assert len(res.history) == 4  # history completo queda disponible
        assert res.holdings.loc[D2, "BBB"] == 250.0

    def test_daily_records_flag_rebalances_with_their_turnover(self, state):
        root, px, _ = state
        res = load_live_result(root, prices=px)
        rebal = {r.date: r.turnover for r in res.daily_records if r.is_rebalance}
        assert rebal == {D1: 0.9, D2: 0.4}
        assert sum(r.turnover for r in res.daily_records) == pytest.approx(1.3)
        assert len(res.daily_records) == len(res.nav)

    def test_returns_follow_backtest_convention(self, state):
        root, px, _ = state
        res = load_live_result(root, prices=px)
        pd.testing.assert_series_equal(
            res.returns, res.nav.ffill().pct_change().fillna(0.0).rename("returns"))
        assert res.returns.iloc[0] == 0.0

    def test_end_truncates_marking(self, state):
        root, px, _ = state
        res = load_live_result(root, prices=px, end="2024-03-29")
        assert res.nav.index[-1] == pd.Timestamp("2024-03-29")

    def test_open_position_without_price_raises(self, state):
        root, px, _ = state
        with pytest.raises(ValueError, match="sin precio.*CCC"):
            load_live_result(root, prices=px.drop(columns="CCC"))

    def test_without_prices_uses_snapshot_values_on_rebalance_dates(self, state):
        root, px, books = state
        res = load_live_result(root)
        assert not res.marked_daily
        assert list(res.nav.index) == [D1, D2]
        for day in (D1, D2):
            assert res.nav[day] == pytest.approx(_expected_nav(px, books, day))
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Fallbacks: transacciones y estado legado
# ---------------------------------------------------------------------------
def _txn(t, ticker, shares, price, run_id, cost=0.0):
    return {"t": f"{t.date()}T00:00:00", "ticker": ticker, "shares": shares,
            "price": price, "proceeds": -shares * price, "realized_pnl": 0.0,
            "cost": cost, "run_id": run_id}


class TestFallbacks:
    def test_holdings_from_transactions_when_no_snapshots(self, state):
        root, px, books = state
        (root / "snapshots.csv").unlink()
        (d1, s1, c1), (d2, s2, c2) = [(d, *books[d]) for d in (D1, D2)]
        rows = [_txn(d1, tk, sh, px.loc[d1, tk], "run-1") for tk, sh in s1.items()]
        rows.append(_txn(d1, "__COST__", 0.0, 0.0, "run-1", cost=5.0))
        delta = s2.reindex(["AAA", "BBB", "CCC"], fill_value=0.0) - s1.reindex(
            ["AAA", "BBB", "CCC"], fill_value=0.0)
        rows += [_txn(d2, tk, sh, px.loc[d2, tk], "run-3")
                 for tk, sh in delta.items() if sh]
        pd.DataFrame(rows).to_csv(root / "transactions.csv", index=False)

        res = load_live_result(root, prices=px)
        assert res.book_source == "transactions"
        assert res.holdings.loc[D2, "AAA"] == pytest.approx(0.0)
        assert res.holdings.loc[D2, "CCC"] == pytest.approx(100.0)
        assert res.nav[pd.Timestamp("2024-04-29")] == pytest.approx(
            _expected_nav(px, books, pd.Timestamp("2024-04-29")))
        # __COST__ de run-1 repartido pro rata al nocional de sus trades
        first = res.trades[res.trades["date"] == D1]
        assert first["cost"].sum() == pytest.approx(5.0)
        assert (first["order_value"] > 0).all()

    def test_legacy_history_only(self, tmp_path):
        pd.DataFrame({
            "date": ["2024-03-01", "2024-03-01", "2024-03-22"],
            "nav": [100.0, 101.0, 103.0], "cash": [1.0, 1.0, 1.0],
            "turnover": [1.0, 0.1, 0.2], "cost": [0.0, 0.0, 0.0],
        }).to_csv(tmp_path / "rebalance_history.csv", index=False)
        res = load_live_result(tmp_path, prices=_prices())
        assert res.book_source == "none" and not res.marked_daily
        assert res.nav.tolist() == [101.0, 103.0]  # mismo as_of → la última fila
        assert res.weights.empty

    def test_missing_history_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="rebalance_history"):
            load_live_result(tmp_path)

    def test_only_partial_runs_raise(self, tmp_path):
        pd.DataFrame([_history_row(D1, "r", 1.0, 1.0, 0.1, status="partial")]).to_csv(
            tmp_path / "rebalance_history.csv", index=False)
        with pytest.raises(ValueError, match="status=ok"):
            load_live_result(tmp_path)

    def test_broker_fills_become_signed_trades(self, state):
        root, px, _ = state
        pd.DataFrame([
            {"ticker": "AAA", "side": "sell", "qty": 10, "filled_qty": 10.0,
             "filled_avg_price": 99.0, "date": "2024-04-12", "run_id": "run-4"},
            {"ticker": "BBB", "side": "buy", "qty": 5, "filled_qty": None,
             "filled_avg_price": None, "date": "2024-04-12", "run_id": "run-4"},
        ]).to_csv(root / "broker_orders.csv", index=False)
        res = load_live_result(root, prices=px)
        broker = res.trades[res.trades["date"] == D3]
        assert broker[["ticker", "order_value"]].values.tolist() == [["AAA", -990.0]]


def test_mark_to_market_requires_overlapping_prices():
    holdings = pd.DataFrame({"AAA": [1.0], CASH_COL: [0.0]}, index=[D3])
    with pytest.raises(ValueError, match="no cubre"):
        mark_to_market(holdings, _prices(), end="2024-03-01")


# ---------------------------------------------------------------------------
# Compatibilidad con los reportes existentes
# ---------------------------------------------------------------------------
class TestReports:
    def test_dashboard_accepts_live_result(self, state, tmp_path):
        from src.dashboard_results import build_dashboard
        root, px, _ = state
        out = tmp_path / "out" / "live.html"
        path = build_dashboard({"live": load_live_result(root, prices=px)},
                               output_path=out)
        assert out.exists() and path == str(out)
        assert "live" in out.read_text(encoding="utf-8")

    def test_monthly_report_accepts_live_result(self, state, tmp_path):
        from src.monthly_report import build_report
        root, px, _ = state
        out = tmp_path / "out" / "report.html"
        groups = {"AAA": "us/Tech", "BBB": "us/Tech", "CCC": "co/Financials"}
        build_report({"live": load_live_result(root, prices=px)}, px, groups,
                     output_path=out)
        assert out.exists()

    def test_cli_writes_dashboard_from_prices_csv(self, state, tmp_path, capsys):
        root, px, _ = state
        csv = tmp_path / "px.csv"
        px.to_csv(csv)
        out = tmp_path / "cli.html"
        assert live_main(["--state-dir", str(root), "--prices-csv", str(csv),
                          "--output", str(out)]) == 0
        assert out.exists()
        assert str(out) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Estado generado por el runner real
# ---------------------------------------------------------------------------
@pytest.fixture
def runner_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    return home / "data" / "live_state"


def test_runner_state_round_trips_into_live_result(runner_home, monkeypatch):
    use_market_data(monkeypatch, _fake_prices())
    assert runner.run_once(force=True, start="2024-01-01") == 0
    longer = _fake_prices(n_rows=320)
    use_market_data(monkeypatch, longer)
    assert runner.run_once(force=True, start="2024-01-01") == 0

    snaps = pd.read_csv(runner_home / "snapshots.csv")
    hist = pd.read_csv(runner_home / "rebalance_history.csv")
    assert set(snaps["run_id"]) == set(hist["run_id"])
    assert (snaps["ticker"] == CASH_COL).sum() == 2

    book = read_live_book(runner_home)
    assert book.source == "snapshots" and set(book.tickers) <= set(TICKERS)
    res = load_live_result(runner_home, prices=longer)
    # Sim: NAV de history = ledger tras los trades marcado en t = NAV del adaptador
    for row in hist.itertuples():
        assert res.nav[pd.Timestamp(row.date)] == pytest.approx(row.nav, rel=1e-9)
    assert res.nav.index[-1] == longer.index[-1]
    assert not res.trades.empty
