"""
test_pre_trade_guards.py — Guardas pre-trade explicitas (sin red).

- URL: host exacto paper-api.alpaca.markets + https (no substring "paper").
- PreTradeGuard: topes por orden / por corrida / numero de ordenes y
  bloqueos de cuenta/mercado, evaluados sobre el plan completo antes del
  primer POST.
- Kill-switch: archivo state_dir/KILL o env HFT_KILL=1 bloquea cualquier envio.
"""

import json

import pandas as pd
import pytest

import src.brokers.alpaca_adapter as adapter
from src.brokers.alpaca_adapter import AlpacaClient, BrokerConfig, load_broker_config
from src.brokers.pre_trade import PreTradeBlocked, PreTradeGuard, kill_switch_reason
from tests.test_broker_idempotency import (
    ACCOUNT, BASE, CLOCK, ORDERS, POSITIONS, FakeUrlopen, Seq,
)

KEYS = {"BROKER_API_KEY": "PKTEST1234", "BROKER_SECRET_KEY": "secreto"}


@pytest.fixture(autouse=True)
def _no_env_kill(monkeypatch):
    monkeypatch.delenv("HFT_KILL", raising=False)


# ===================================================================
# 1. URL: host exacto + https
# ===================================================================
class TestPaperUrl:
    @pytest.mark.parametrize("url", [
        "https://api.alpaca.markets/paper",               # live con "paper" en path
        "https://api.alpaca.markets/paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.com",
        "https://paper-api.alpaca.markets@api.alpaca.markets",
        "https://api.alpaca.markets/?h=paper",
        "http://paper-api.alpaca.markets",                # sin TLS
        "https://paper-api.alpaca.markets:8443",
        "https://paper-api.alpaca.markets/paper",         # path extra
    ])
    def test_non_paper_host_rejected_without_http(self, monkeypatch, url):
        fake = FakeUrlopen({})
        monkeypatch.setattr("urllib.request.urlopen", fake)
        with pytest.raises(ValueError, match="solo-paper"):
            load_broker_config(env={**KEYS, "BROKER_BASE_URL": url})
        assert fake.calls == []

    @pytest.mark.parametrize("url", [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/",
        "https://paper-api.alpaca.markets/v2",
        "https://paper-api.alpaca.markets:443/v2/",
    ])
    def test_paper_host_accepted(self, url):
        cfg = load_broker_config(env={**KEYS, "BROKER_BASE_URL": url})
        assert cfg.base_url.startswith("https://paper-api.alpaca.markets")

    def test_client_rejects_live_url_even_without_loader(self):
        with pytest.raises(ValueError, match="solo-paper"):
            AlpacaClient(BrokerConfig("PKTEST1234", "s",
                                      "https://api.alpaca.markets/paper"))


# ===================================================================
# 2. PreTradeGuard (puro)
# ===================================================================
def _plan(*notionals):
    return pd.DataFrame([
        {"ticker": f"T{i}", "side": "buy", "qty": 1.0, "notional": n}
        for i, n in enumerate(notionals)],
        columns=["ticker", "side", "qty", "notional"])


OK_SUMMARY = {"equity": "10000", "account_blocked": False,
              "trading_blocked": False, "market_open": True}


class TestPreTradeGuard:
    def test_plan_within_limits_passes(self):
        guard = PreTradeGuard(max_notional_per_order=5000,
                              max_total_notional=1.0, max_orders=5)
        assert guard.evaluate(_plan(4000, 3000), OK_SUMMARY) == []

    def test_order_over_per_order_cap(self):
        reasons = PreTradeGuard(max_notional_per_order=5000).evaluate(
            _plan(4000, 6000), OK_SUMMARY)
        assert len(reasons) == 1
        assert "max_notional_per_order" in reasons[0] and "T1" in reasons[0]

    def test_total_over_equity_fraction(self):
        reasons = PreTradeGuard(max_total_notional=0.5).evaluate(
            _plan(3000, 3000), OK_SUMMARY)  # 6000 > 0.5 * 10000
        assert len(reasons) == 1 and "max_total_notional" in reasons[0]

    def test_too_many_orders(self):
        reasons = PreTradeGuard(max_orders=2).evaluate(
            _plan(1, 1, 1), OK_SUMMARY)
        assert len(reasons) == 1 and "max_orders" in reasons[0]

    @pytest.mark.parametrize("flag", ["account_blocked", "trading_blocked"])
    def test_blocked_account(self, flag):
        reasons = PreTradeGuard().evaluate(_plan(1), {**OK_SUMMARY, flag: True})
        assert len(reasons) == 1 and flag in reasons[0]

    def test_market_closed_only_blocks_when_configured(self):
        closed = {**OK_SUMMARY, "market_open": False}
        assert PreTradeGuard().evaluate(_plan(1), closed) == []
        reasons = PreTradeGuard(block_if_market_closed=True).evaluate(
            _plan(1), closed)
        assert len(reasons) == 1 and "mercado cerrado" in reasons[0]

    def test_none_disables_limits(self):
        guard = PreTradeGuard(max_notional_per_order=None,
                              max_total_notional=None, max_orders=None)
        assert guard.evaluate(_plan(1e9, 1e9), OK_SUMMARY) == []

    def test_empty_plan_passes(self):
        assert PreTradeGuard().evaluate(_plan(), OK_SUMMARY) == []


# ===================================================================
# 3. Kill-switch
# ===================================================================
class TestKillSwitch:
    def test_off_by_default(self, tmp_path):
        assert kill_switch_reason(tmp_path, env={}) is None

    def test_kill_file(self, tmp_path):
        (tmp_path / "KILL").touch()
        assert "KILL" in kill_switch_reason(tmp_path, env={})

    @pytest.mark.parametrize("value,active", [
        ("1", True), ("true", True), ("0", False), ("", False)])
    def test_env_var(self, tmp_path, value, active):
        reason = kill_switch_reason(tmp_path, env={"HFT_KILL": value})
        assert (reason is not None) == active


def _client(monkeypatch, tmp_path, posts=({"id": "a", "status": "accepted"},)):
    fake = FakeUrlopen({("GET", ACCOUNT): {"account_number": "", "equity": "1"},
                        ("POST", ORDERS): Seq(*posts)})
    monkeypatch.setattr("urllib.request.urlopen", fake)
    client = AlpacaClient(BrokerConfig("PKTEST1234", "s", BASE),
                          kill_dir=tmp_path)
    return client, fake


def _two():
    return pd.DataFrame([{"ticker": "AAPL", "side": "buy", "qty": 1.0},
                         {"ticker": "MSFT", "side": "buy", "qty": 1.0}])


class TestKillSwitchBlocksAnySend:
    def test_kill_file_blocks_before_any_http(self, monkeypatch, tmp_path):
        (tmp_path / "KILL").touch()
        client, fake = _client(monkeypatch, tmp_path)
        with pytest.raises(PreTradeBlocked, match="KILL"):
            client.submit_orders(_two(), dry_run=False, run_id="r")
        assert fake.calls == []

    def test_env_kill_blocks(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HFT_KILL", "1")
        client, fake = _client(monkeypatch, tmp_path)
        with pytest.raises(PreTradeBlocked, match="HFT_KILL"):
            client.submit_orders(_two(), dry_run=False, run_id="r")
        assert fake.posts() == []

    def test_kill_mid_batch_stops_remaining_orders(self, monkeypatch, tmp_path):
        client, fake = _client(monkeypatch, tmp_path)
        seen = []

        def _on_result(r):
            seen.append(r)
            (tmp_path / "KILL").touch()  # operador activa KILL tras la 1a orden

        with pytest.raises(PreTradeBlocked):
            client.submit_orders(_two(), dry_run=False, run_id="r",
                                 on_result=_on_result)
        assert len(fake.posts()) == 1
        assert [r["status"] for r in seen] == ["submitted", "blocked"]

    def test_dry_run_ignores_kill(self, monkeypatch, tmp_path):
        (tmp_path / "KILL").touch()
        client, fake = _client(monkeypatch, tmp_path)
        out = client.submit_orders(_two(), dry_run=True)
        assert [r["status"] for r in out] == ["dry_run", "dry_run"]
        assert fake.calls == []


# ===================================================================
# run_broker_step: guard sobre el plan completo antes del primer POST
# ===================================================================
def _patch_step(monkeypatch, account=None, market_open=True):
    fake = FakeUrlopen({
        ("GET", ACCOUNT): {"account_number": "PA1", "equity": "10000",
                           **(account or {})},
        ("GET", CLOCK): {"is_open": market_open, "timestamp": "t"},
        ("GET", POSITIONS): [],
        ("POST", ORDERS): {"id": "x", "status": "accepted"},
        ("GET", ORDERS + "/x"): {"id": "x", "status": "filled",
                                 "filled_qty": "1", "filled_avg_price": "1"},
    })
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr(adapter, "load_broker_config",
                        lambda *a, **k: BrokerConfig("PKTEST1234", "s", BASE))
    return fake


TARGET = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2})   # 5000/3000/2000
PRICES = pd.Series({"AAPL": 100.0, "MSFT": 50.0, "NVDA": 20.0})
T = pd.Timestamp("2026-09-25")


class TestRunBrokerStepGuard:
    def _run(self, tmp_path, **kw):
        from scripts.run_periodic_rebalance import run_broker_step
        return run_broker_step(TARGET, PRICES, T, tmp_path, **kw)

    def _saved(self, tmp_path):
        return pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")

    def test_plan_over_cap_sends_zero_orders(self, monkeypatch, tmp_path):
        fake = _patch_step(monkeypatch)
        with pytest.raises(PreTradeBlocked, match="max_notional_per_order"):
            self._run(tmp_path, submit=True,
                      guard=PreTradeGuard(max_notional_per_order=4000))
        assert fake.posts() == []
        saved = self._saved(tmp_path)
        assert (saved["status"] == "blocked").all()
        assert saved["blocked_reason"].str.contains("AAPL").all()

    def test_block_is_journaled(self, monkeypatch, tmp_path):
        _patch_step(monkeypatch)
        with pytest.raises(PreTradeBlocked):
            self._run(tmp_path, submit=True,
                      guard=PreTradeGuard(max_orders=2))
        lines = (tmp_path / "broker_journal_2026-09-25.jsonl").read_text().splitlines()
        (record,) = [json.loads(line) for line in lines]
        assert record["event"] == "blocked"
        assert any("max_orders" in r for r in record["reasons"])

    def test_kill_file_in_state_dir_blocks(self, monkeypatch, tmp_path):
        fake = _patch_step(monkeypatch)
        (tmp_path / "KILL").touch()
        with pytest.raises(PreTradeBlocked, match="KILL"):
            self._run(tmp_path, submit=True)
        assert fake.posts() == []
        assert (self._saved(tmp_path)["status"] == "blocked").all()

    def test_account_blocked_from_check_blocks(self, monkeypatch, tmp_path):
        fake = _patch_step(monkeypatch, account={"trading_blocked": True})
        with pytest.raises(PreTradeBlocked, match="trading_blocked"):
            self._run(tmp_path, submit=True)
        assert fake.posts() == []

    def test_dry_run_reports_violation_without_raising(self, monkeypatch, tmp_path):
        _patch_step(monkeypatch)
        self._run(tmp_path, submit=False,
                  guard=PreTradeGuard(max_notional_per_order=4000))
        saved = self._saved(tmp_path)
        assert (saved["status"] == "dry_run").all()
        assert saved["blocked_reason"].str.contains("max_notional_per_order").all()

    def test_within_limits_submits_and_reason_empty(self, monkeypatch, tmp_path):
        fake = _patch_step(monkeypatch)
        self._run(tmp_path, submit=True, poll_timeout=0)
        assert len(fake.posts()) == 3
        assert self._saved(tmp_path)["blocked_reason"].isna().all()


class TestRunOnceExitCode:
    def test_kill_switch_exits_3_broker_and_sends_nothing(self, monkeypatch, tmp_path):
        """Bloqueo pre-trade = fallo de broker (exit 3); 4 es el lock (ARQ-06)."""
        import scripts.run_periodic_rebalance as runner
        from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data
        monkeypatch.setenv("HFT_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HFT_KILL", "1")
        monkeypatch.setattr(runner, "load_universe",
                            lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
        use_market_data(monkeypatch, _fake_prices())
        fake = _patch_step(monkeypatch)
        assert runner.run_once(force=True, start="2024-01-01",
                               mode="broker", submit=True) == runner.EXIT_BROKER
        assert fake.posts() == []
        state = tmp_path / "home" / "data" / "live_state"
        assert not (state / "last_rebalance.txt").exists()
