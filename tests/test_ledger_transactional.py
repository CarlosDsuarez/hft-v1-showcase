"""
test_ledger_transactional.py — PositionLedger: persistencia publica y
aplicacion transaccional de lotes de trades.

Cubre:
  - to_dict()/from_dict(): roundtrip identico (incl. JSON), payload legado
    sin schema_version/txns, schema_version desconocida.
  - apply_trades atomico: venta invalida / precio invalido / costos invalidos
    en mitad del lote → error y estado identico al previo.
  - Ventas antes que compras dentro del lote.
  - Invariante cash >= -CASH_TOLERANCE y reserva de costos en
    trades_from_target_weights (cash no queda negativo tras rebalanceo).

Todo sintetico, sin red.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.position_ledger as pl
from src.position_ledger import PositionLedger

T0 = pd.Timestamp("2022-01-03")
T1 = pd.Timestamp("2022-01-04")
T2 = pd.Timestamp("2022-01-05")
COST_BPS = 7.0


@pytest.fixture
def prices():
    return pd.Series({"A": 100.0, "B": 50.0, "C": 25.0, "D": 10.0})


@pytest.fixture
def ledger(prices):
    """Cash 7000 + A=10 @100 + B=20 @50 (via API publica)."""
    lg = PositionLedger(initial_cash=10_000.0)
    lg.apply_trades(
        pd.DataFrame([{"ticker": "A", "shares": 10.0},
                      {"ticker": "B", "shares": 20.0}]),
        prices, t=T0,
    )
    return lg


def _state(lg: PositionLedger, prices: pd.Series) -> dict:
    """Estado observable via queries publicas preexistentes (NaT → None)."""
    snap = lg.snapshot(T0, prices)
    txn = lg.transactions_df().astype(object)
    return {
        "cash": lg.cash,
        "realized_pnl": lg.realized_pnl,
        "positions": snap.positions.sort_index().to_dict(),
        "avg_entry": snap.avg_entry.sort_index().to_dict(),
        "cost_basis": snap.cost_basis.sort_index().to_dict(),
        "txns": txn.where(txn.notna(), None).to_dict("records"),
    }


def _cost(tw: pd.Series, lg: PositionLedger, prices: pd.Series) -> float:
    """Costo estimado como en el runner: turnover L1 * bps * NAV."""
    nav = lg.nav(prices)
    w_prev = lg.weights_from_positions(prices).reindex(tw.index).fillna(0.0)
    return float((tw - w_prev).abs().sum()) * COST_BPS / 10_000.0 * nav


def _rich_ledger(prices: pd.Series) -> PositionLedger:
    """Ledger con compras, venta FIFO parcial, costos y t=NaT."""
    lg = PositionLedger(initial_cash=10_000.0)
    lg.apply_trades(pd.DataFrame([{"ticker": "A", "shares": 10.0}]),
                    prices, t=T0)
    px1 = prices.copy()
    px1["A"] = 110.0
    lg.apply_trades(pd.DataFrame([{"ticker": "A", "shares": 10.0},
                                  {"ticker": "C", "shares": 3.5}]),
                    px1, costs=1.25, t=T1)
    px2 = prices.copy()
    px2["A"] = 120.0
    lg.apply_trades(pd.DataFrame([{"ticker": "A", "shares": -15.0}]),
                    px2, costs=0.5)  # t=None → NaT
    return lg


# ===========================================================================
# Persistencia: to_dict / from_dict
# ===========================================================================
class TestPersistence:
    def test_to_dict_exposes_full_state(self, prices):
        lg = _rich_ledger(prices)
        d = lg.to_dict()
        assert d["schema_version"] == pl.LEDGER_SCHEMA_VERSION
        assert d["cash"] == lg.cash
        assert d["realized_pnl"] == lg.realized_pnl
        # FIFO: quedan 5 de A del segundo lote (@110) y 3.5 de C
        assert [(r["ticker"], r["shares"], r["entry_price"]) for r in d["lots"]] == [
            ("A", 5.0, 110.0), ("C", 3.5, 25.0),
        ]
        assert len(d["txns"]) == len(lg.transactions_df())

    def test_roundtrip_is_identical(self, prices):
        lg = _rich_ledger(prices)
        d = lg.to_dict()
        restored = PositionLedger.from_dict(d)
        assert restored.to_dict() == d
        assert _state(restored, prices) == _state(lg, prices)

    def test_roundtrip_through_json_is_identical(self, prices):
        lg = _rich_ledger(prices)
        d = lg.to_dict()
        restored = PositionLedger.from_dict(json.loads(json.dumps(d)))
        assert restored.to_dict() == d
        assert _state(restored, prices) == _state(lg, prices)

    def test_restored_ledger_is_independent(self, prices):
        lg = _rich_ledger(prices)
        before = lg.to_dict()
        restored = PositionLedger.from_dict(before)
        restored.apply_trades(pd.DataFrame([{"ticker": "A", "shares": -5.0}]),
                              prices, t=T2)
        assert lg.to_dict() == before

    def test_from_dict_accepts_legacy_payload(self):
        legacy = {
            "cash": -1383.08,
            "realized_pnl": -7.5,
            "lots": [
                {"ticker": "A", "shares": 1.5, "entry_price": 100.0,
                 "entry_date": "2026-09-28T00:00:00"},
                {"ticker": "A", "shares": 2.0, "entry_price": 101.0,
                 "entry_date": None},
            ],
        }
        lg = PositionLedger.from_dict(legacy)
        assert lg.cash == -1383.08
        assert lg.realized_pnl == -7.5
        assert lg.positions().to_dict() == {"A": 3.5}
        assert lg.transactions_df().empty
        assert lg.to_dict()["schema_version"] == pl.LEDGER_SCHEMA_VERSION

    def test_from_dict_rejects_unknown_schema_version(self, prices):
        d = _rich_ledger(prices).to_dict()
        d["schema_version"] = pl.LEDGER_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema_version"):
            PositionLedger.from_dict(d)


# ===========================================================================
# apply_trades transaccional
# ===========================================================================
class TestAtomicApply:
    def test_invalid_sell_mid_batch_leaves_state_identical(self, ledger, prices):
        before = _state(ledger, prices)
        batch = pd.DataFrame([
            {"ticker": "C", "shares": 10.0},
            {"ticker": "A", "shares": -5.0},
            {"ticker": "B", "shares": -50.0},   # solo hay 20
            {"ticker": "D", "shares": 4.0},
        ])
        with pytest.raises(ValueError, match="B"):
            ledger.apply_trades(batch, prices, costs=1.0, t=T1)
        assert _state(ledger, prices) == before

    def test_split_oversell_same_ticker_is_rejected(self, ledger, prices):
        before = _state(ledger, prices)
        batch = pd.DataFrame([
            {"ticker": "A", "shares": -6.0},
            {"ticker": "A", "shares": -6.0},    # 12 > 10 en total
        ])
        with pytest.raises(ValueError, match="A"):
            ledger.apply_trades(batch, prices, t=T1)
        assert _state(ledger, prices) == before

    @pytest.mark.parametrize("bad_px", [None, np.nan, 0.0, -5.0, np.inf])
    def test_invalid_price_mid_batch_raises_without_mutation(
        self, ledger, prices, bad_px,
    ):
        before = _state(ledger, prices)
        px = prices.copy()
        if bad_px is None:
            px = px.drop("D")
        else:
            px["D"] = bad_px
        batch = pd.DataFrame([
            {"ticker": "A", "shares": -5.0},
            {"ticker": "D", "shares": 4.0},
            {"ticker": "C", "shares": 1.0},
        ])
        with pytest.raises(ValueError, match="D"):
            ledger.apply_trades(batch, px, t=T1)
        assert _state(ledger, prices) == before

    @pytest.mark.parametrize("bad_shares", [np.nan, np.inf])
    def test_non_finite_shares_raise_without_mutation(
        self, ledger, prices, bad_shares,
    ):
        before = _state(ledger, prices)
        batch = pd.DataFrame([
            {"ticker": "C", "shares": 1.0},
            {"ticker": "A", "shares": bad_shares},
        ])
        with pytest.raises(ValueError, match="A"):
            ledger.apply_trades(batch, prices, t=T1)
        assert _state(ledger, prices) == before

    @pytest.mark.parametrize("bad_cost", [-1.0, np.nan])
    def test_invalid_costs_raise_without_mutation(self, ledger, prices, bad_cost):
        before = _state(ledger, prices)
        batch = pd.DataFrame([{"ticker": "C", "shares": 1.0}])
        with pytest.raises(ValueError, match="cost"):
            ledger.apply_trades(batch, prices, costs=bad_cost, t=T1)
        assert _state(ledger, prices) == before


# ===========================================================================
# Orden: ventas antes que compras
# ===========================================================================
class TestSellsBeforeBuys:
    def test_sells_execute_before_buys(self, prices):
        lg = PositionLedger(initial_cash=10_000.0)
        lg.apply_trades(pd.DataFrame([{"ticker": "A", "shares": 100.0}]),
                        prices, t=T0)
        assert lg.cash == 0.0
        # La compra de B (primera en orden alfabetico) solo es financiable
        # con la venta de A del mismo lote.
        lg.apply_trades(pd.DataFrame([
            {"ticker": "B", "shares": 120.0},
            {"ticker": "A", "shares": -60.0},
        ]), prices, t=T1)
        txn = lg.transactions_df()
        assert txn.loc[txn["t"] == T1, "ticker"].tolist() == ["A", "B"]
        assert abs(lg.cash) < 1e-9


# ===========================================================================
# Invariante de caja + reserva de costos
# ===========================================================================
class TestCashInvariant:
    def test_overspending_batch_raises_without_mutation(self, ledger, prices):
        before = _state(ledger, prices)   # cash 7000
        batch = pd.DataFrame([{"ticker": "C", "shares": 400.0}])  # 10_000
        with pytest.raises(ValueError, match="cash"):
            ledger.apply_trades(batch, prices, t=T1)
        assert _state(ledger, prices) == before

    def test_costs_that_push_cash_negative_raise(self, prices):
        lg = PositionLedger(initial_cash=1_000.0)
        before = _state(lg, prices)
        batch = pd.DataFrame([{"ticker": "A", "shares": 10.0}])  # 1000
        with pytest.raises(ValueError, match="cash"):
            lg.apply_trades(batch, prices, costs=5.0, t=T0)
        assert _state(lg, prices) == before

    def test_trades_from_target_weights_reserves_costs(self, prices):
        lg = PositionLedger(initial_cash=10_000.0)
        tw = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        trades = lg.trades_from_target_weights(tw, prices, nav=10_000.0,
                                               costs=70.0)
        px = prices.reindex(trades["ticker"]).to_numpy()
        spend = float((trades["shares"].to_numpy() * px).sum())
        assert spend == pytest.approx(10_000.0 - 70.0, abs=1e-9)

    def test_rebalances_with_costs_keep_cash_non_negative(self, prices):
        lg = PositionLedger(initial_cash=10_000.0)
        for t, tw in [
            (T0, pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})),
            (T1, pd.Series({"B": 0.1, "C": 0.4, "D": 0.5})),
            (T2, pd.Series({"A": 1.0})),
        ]:
            costs = _cost(tw, lg, prices)
            assert costs > 0
            nav_pre = lg.nav(prices)
            trades = lg.trades_from_target_weights(tw, prices, nav=nav_pre,
                                                   costs=costs)
            lg.apply_trades(trades, prices, costs=costs, t=t)
            assert lg.cash >= -pl.CASH_TOLERANCE
            assert lg.nav(prices) == pytest.approx(nav_pre - costs, abs=1e-6)

    def test_legacy_negative_cash_heals_on_next_rebalance(self, prices):
        lg = PositionLedger.from_dict({
            "cash": -1383.0,
            "realized_pnl": 0.0,
            "lots": [{"ticker": "A", "shares": 50.0, "entry_price": 90.0,
                      "entry_date": None},
                     {"ticker": "B", "shares": 80.0, "entry_price": 55.0,
                      "entry_date": None}],
        })
        tw = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        costs = _cost(tw, lg, prices)
        trades = lg.trades_from_target_weights(tw, prices,
                                               nav=lg.nav(prices), costs=costs)
        lg.apply_trades(trades, prices, costs=costs, t=T1)
        assert lg.cash >= -pl.CASH_TOLERANCE
