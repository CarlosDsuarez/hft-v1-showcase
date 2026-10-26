"""
test_broker_rebalance.py — Cableado rebalanceo periodico -> broker paper.

Todo mockeado (sin red): build_broker_plan es puro y run_broker_step usa
un FakeClient inyectado via monkeypatch.
"""

import pandas as pd
import pytest

from scripts.run_periodic_rebalance import build_broker_plan, run_broker_step


class FakeClient:
    """Doble de AlpacaClient: verifica cuenta, sin HTTP."""

    instances = []

    def __init__(self, config, kill_dir=None):
        self.config = config
        self.submitted = []
        FakeClient.instances.append(self)

    def check(self):
        return {"account_number": "PA0000000002", "equity": "10000",
                "buying_power": "40000", "market_open": True}

    def get_account(self):
        return {"account_number": "PA0000000002", "equity": "10000",
                "cash": "9000"}

    def get_positions(self):
        return pd.DataFrame([{"ticker": "AAPL", "qty": 5.0,
                              "market_value": 1000.0, "avg_entry": 200.0,
                              "side": "long"}])

    def submit_orders(self, plan, dry_run=True, run_id=None, on_result=None):
        self.submitted.append((plan, dry_run))
        results = [{"ticker": r.ticker, "side": r.side, "qty": r.qty,
                    "status": "dry_run" if dry_run else "submitted", "id": None}
                   for r in plan.itertuples()]
        for r in results:
            if on_result is not None:
                on_result(r)
        return results

    def poll_orders(self, ids, timeout=30.0):
        return {}


def _patch(monkeypatch):
    import src.brokers.alpaca_adapter as adapter
    monkeypatch.setattr(adapter, "AlpacaClient", FakeClient)
    monkeypatch.setattr(adapter, "load_broker_config", lambda *a, **k: object())


def _target():
    return pd.Series({"AAPL": 0.6, "MSFT": 0.4})


def _prices():
    return pd.Series({"AAPL": 200.0, "MSFT": 50.0})


class TestBuildBrokerPlan:
    def test_math(self):
        plan = build_broker_plan(_target(), pd.Series({"AAPL": 5.0}),
                                 _prices(), equity=10000.0)
        # AAPL: 30.0-5.0 buy 25 ($5000); MSFT: 80.0-0 buy 80 ($4000)
        assert set(plan["ticker"]) == {"AAPL", "MSFT"}
        row = plan.set_index("ticker").loc["AAPL"]
        assert row["qty"] == 25.0 and row["side"] == "buy"


class TestRunBrokerStep:
    def test_dry_run_writes_csv(self, monkeypatch, tmp_path):
        _patch(monkeypatch)
        t = pd.Timestamp("2026-09-25")
        plan, results, status = run_broker_step(
            _target(), _prices(), t, tmp_path, submit=False)
        assert status == "dry_run"
        assert results and all(r["status"] == "dry_run" for r in results)
        saved = pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")
        assert set(saved["ticker"]) == {"AAPL", "MSFT"}
        assert (saved["status"] == "dry_run").all()
        assert {"order_id", "error"}.issubset(saved.columns)

    def test_submit_sends(self, monkeypatch, tmp_path):
        _patch(monkeypatch)
        FakeClient.instances.clear()
        t = pd.Timestamp("2026-09-25")
        _, results, status = run_broker_step(
            _target(), _prices(), t, tmp_path, submit=True)
        assert status == "submitted"
        assert all(r["status"] == "submitted" for r in results)
        sent_plan, dry = FakeClient.instances[-1].submitted[-1]
        assert dry is False and len(sent_plan) == 2

    def test_broker_error_propagates(self, monkeypatch, tmp_path):
        import src.brokers.alpaca_adapter as adapter

        class BadClient(FakeClient):
            def check(self):
                raise adapter.BrokerError("Cuenta equivocada.")

        monkeypatch.setattr(adapter, "AlpacaClient", BadClient)
        monkeypatch.setattr(adapter, "load_broker_config", lambda *a, **k: object())
        with pytest.raises(adapter.BrokerError, match="Cuenta equivocada"):
            run_broker_step(_target(), _prices(),
                            pd.Timestamp("2026-09-25"), tmp_path, submit=True)
