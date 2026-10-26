"""
test_runner_phases.py — run_once en fases explícitas con commit al final (ARQ-06).

load_state → fetch_data → cartera → build_targets (pesos + salidas) →
build_sim_plan / plan_broker → [execute_broker] → commit_state.

Todo sintético y sin red: precios falsos (tests.test_runner_paths) y un
FakeBroker inyectado vía monkeypatch sobre src.brokers.alpaca_adapter.
Estado en tmp_path vía HFT_HOME.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
import src.brokers.alpaca_adapter as adapter
from src.brokers.alpaca_adapter import BrokerError
from src.brokers.pre_trade import PreTradeBlocked, PreTradeGuard
from src.history_requirements import InsufficientHistoryError
from src.instruments import Instrument, InstrumentMaster
from src.live_state import StateStore
from src.portfolio_state import PortfolioState
from src.position_ledger import PositionLedger
from src.market_data import DataPolicy
from src.strategy import build_targets, load_strategy_config
from tests.test_runner_paths import TICKERS, _fake_prices, _fake_volumes, use_market_data

DRIFT = pd.Series([1.1, 0.9, 1.0, 1.05, 0.95, 1.0], index=TICKERS)
START = "2024-01-01"
LIVE = load_strategy_config()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeBroker:
    """Doble de AlpacaClient con la semántica de submit_orders de ARQ-05.

    Args:
        fail_after: Tras N órdenes enviadas, la siguiente falla con 403
            (se reporta "failed" y se aborta el lote).
        reject: Tickers rechazados por el broker (4xx por orden: se sigue).
    """

    def __init__(self, equity=10_000.0, market_open=True, account=None,
                 fail_after=None, reject=(), check_error=None):
        self.equity = equity
        self.market_open = market_open
        self.account = account or {}
        self.fail_after = fail_after
        self.reject = set(reject)
        self.check_error = check_error
        self.sent = []          # órdenes que llegaron al broker
        self.submit_calls = []  # (n_orders, dry_run)

    def check(self):
        if self.check_error is not None:
            raise self.check_error
        return {"account_number": "PA1", "equity": str(self.equity),
                "account_blocked": False, "trading_blocked": False,
                "market_open": self.market_open, **self.account}

    def get_account(self):
        return {"account_number": "PA1", "equity": str(self.equity),
                "cash": str(self.equity)}

    def get_positions(self):
        return pd.DataFrame(columns=["ticker", "qty"])

    def submit_orders(self, plan, dry_run=True, run_id=None, on_result=None):
        self.submit_calls.append((len(plan), dry_run))
        results = []

        def emit(rec):
            results.append(rec)
            if on_result is not None:
                on_result(rec)

        for r in plan.itertuples():
            base = {"ticker": r.ticker, "side": r.side, "qty": r.qty,
                    "client_order_id": f"{run_id}-{r.ticker}-{r.side}",
                    "broker_status": None, "error": None}
            if dry_run:
                emit({**base, "status": "dry_run", "id": None})
            elif self.fail_after is not None and len(self.sent) >= self.fail_after:
                emit({**base, "status": "failed", "id": None, "error": "403"})
                raise BrokerError("forbidden", status=403)
            elif r.ticker in self.reject:
                emit({**base, "status": "rejected", "id": None, "error": "422"})
            else:
                self.sent.append(r.ticker)
                emit({**base, "status": "submitted", "id": f"id-{r.ticker}",
                      "broker_status": "accepted"})
        return results

    def poll_orders(self, ids, timeout=30.0):
        return {oid: {"final_status": "filled", "filled_qty": 1.0,
                      "filled_avg_price": 100.0} for oid in ids}


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    use_market_data(monkeypatch, _fake_prices())
    return home


@pytest.fixture
def state_dir(home):
    return home / "data" / "live_state"


@pytest.fixture
def broker(monkeypatch):
    fake = FakeBroker()
    monkeypatch.setattr(adapter, "AlpacaClient", lambda *a, **k: fake)
    monkeypatch.setattr(adapter, "load_broker_config", lambda *a, **k: object())
    return fake


def _tree(root: Path) -> dict:
    """Todo archivo bajo root (incl. ocultos/.lock/.staging): bytes + mtime."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _state_tree(root: Path) -> dict:
    """_tree sin .lock (se reescribe con el pid al tomarlo) ni logs/ (runner.log
    y alerts.jsonl registran cada corrida escritora: observabilidad, no estado)."""
    return {k: v for k, v in _tree(root).items()
            if k != ".lock" and Path(k).parts[0] != "logs"}


def _history(state_dir: Path) -> pd.DataFrame:
    return pd.read_csv(state_dir / "rebalance_history.csv")


def _drift(monkeypatch):
    use_market_data(monkeypatch, _fake_prices() * DRIFT)


def _data():
    return runner.MarketData(prices=_fake_prices(), volumes=_fake_volumes())


def _held_ledger(prices_t: pd.Series) -> PositionLedger:
    lg = PositionLedger(initial_cash=100_000.0)
    trades = pd.DataFrame({"ticker": TICKERS, "shares": [100.0] * len(TICKERS)})
    lg.apply_trades(trades, prices_t * 0.5, t=pd.Timestamp("2024-01-02"))
    return lg


# ===========================================================================
# Fases puras / de lectura
# ===========================================================================
class TestLoadState:
    def test_missing_state_dir_gives_fresh_state_without_creating_it(self, tmp_path):
        root = tmp_path / "nope"
        state = runner.load_state(StateStore(root), initial_cash=123.0)
        assert state.last_rebalance is None
        assert state.manifest == {}
        assert state.ledger.nav(pd.Series(dtype=float)) == pytest.approx(123.0)
        assert not root.exists()

    def test_reads_committed_state(self, home, state_dir):
        assert runner.run_once(force=True, start=START) == 0
        state = runner.load_state(StateStore(state_dir), initial_cash=1.0)
        assert state.last_rebalance is not None
        assert state.manifest["run_id"]
        assert state.ledger.to_dict() == json.loads(
            (state_dir / "ledger.json").read_text())


class TestFetchData:
    def test_short_history_raises_insufficient_data(self, home, monkeypatch):
        use_market_data(monkeypatch, _fake_prices().iloc[:20])
        with pytest.raises(InsufficientHistoryError):
            runner.fetch_data(START)

    def test_returns_aligned_prices_and_volumes(self, home):
        data = runner.fetch_data(START)
        assert data.t == _fake_prices().index[-1]
        assert data.volumes.index.equals(data.prices.index)
        assert list(data.volumes.columns) == list(data.prices.columns)


class TestDecideSchedule:
    def _state(self, last=None, manifest=None):
        return runner.LiveState(ledger=PositionLedger(), last_rebalance=last,
                                manifest=manifest or {})

    def test_never_rebalanced_is_due(self):
        sched = runner.decide_schedule(self._state(), _data(), every_n=15)
        assert sched.due and not sched.same_as_of and sched.supersedes is None

    def test_recent_rebalance_not_due(self):
        data = _data()
        sched = runner.decide_schedule(self._state(last=data.prices.index[-3]),
                                       data, every_n=15)
        assert not sched.due

    def test_same_as_of_never_due_and_supersedes_committed_run(self):
        data = _data()
        manifest = {"run_id": "r1", "as_of": data.t.date().isoformat()}
        sched = runner.decide_schedule(self._state(data.t, manifest), data, every_n=0)
        assert not sched.due and sched.same_as_of
        assert sched.supersedes == "r1"

    def test_retry_after_partial_supersedes_it(self):
        """Parcial: manifiesto con este as_of pero last_rebalance viejo."""
        data = _data()
        manifest = {"run_id": "partial-run", "as_of": data.t.date().isoformat()}
        sched = runner.decide_schedule(
            self._state(data.prices.index[-40], manifest), data, every_n=15)
        assert sched.due
        assert sched.supersedes == "partial-run"


class TestTargetsAndExits:
    """Objetivos + salidas salen de src.strategy.build_targets (config viva)."""

    @staticmethod
    def _state(ledger, data):
        prices_t = data.prices.loc[data.t]
        return PortfolioState.from_ledger(ledger, prices_t, as_of=data.t)

    def test_too_few_assets_is_insufficient_data(self):
        prices = _fake_prices()[["AAA", "BBB"]]
        t = prices.index[-1]
        empty = PortfolioState.from_ledger(PositionLedger(), prices.loc[t], as_of=t)
        with pytest.raises(runner.InsufficientDataError):
            build_targets(empty, prices, _fake_volumes()[["AAA", "BBB"]], t, LIVE)

    def test_exits_zero_weights_without_mutating_ledger(self):
        data = _data()
        ledger = _held_ledger(data.prices.loc[data.t])
        before = ledger.to_dict()
        cfg = dataclasses.replace(LIVE, take_profit_dollars=2.0,
                                  stop_loss_dollars=None)
        res = build_targets(self._state(ledger, data), data.prices,
                            data.volumes, data.t, cfg)
        assert set(res.exits["ticker"]) == set(TICKERS)  # +100% sobre la entrada
        assert (res.weights == 0).all()
        assert ledger.to_dict() == before

    def test_exits_off_means_no_exits_and_book_does_not_matter(self):
        data = _data()
        cfg = dataclasses.replace(LIVE, use_exit_rules=False)
        held = build_targets(
            self._state(_held_ledger(data.prices.loc[data.t]), data),
            data.prices, data.volumes, data.t, cfg)
        fresh = build_targets(self._state(PositionLedger(), data),
                              data.prices, data.volumes, data.t, cfg)
        assert held.exits.empty
        assert held.weights.sum() == pytest.approx(1.0)
        pd.testing.assert_series_equal(held.weights, fresh.weights)


class TestBuildSimPlan:
    def test_plan_is_pure_and_sizes_trades(self):
        data = _data()
        prices_t = data.prices.loc[data.t]
        ledger = PositionLedger(initial_cash=100_000.0)
        before = ledger.to_dict()
        w = pd.Series(1.0 / len(TICKERS), index=TICKERS)
        sim = runner.build_sim_plan(ledger, w, prices_t, data.t, LIVE)
        assert ledger.to_dict() == before           # la entrada no se toca
        assert sim.turnover == pytest.approx(1.0)
        # costo = turnover × (commission_bps + spread_bps) de la config
        assert sim.cost == pytest.approx(1.0 * LIVE.cost_bps / 10_000 * 100_000.0)
        assert len(sim.trades) == len(TICKERS)
        assert sim.ledger_after.cash >= 0
        # una compra por ticker + el registro de costos (__COST__)
        assert set(sim.new_txns["ticker"]) == set(TICKERS) | {"__COST__"}
        assert sim.snap_pre.nav == pytest.approx(100_000.0)
        assert sim.snap_after.nav == pytest.approx(100_000.0 - sim.cost)


# ===========================================================================
# Execute (broker) — clasificación del resultado
# ===========================================================================
class TestExecuteBroker:
    T = pd.Timestamp("2026-09-25")
    W = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    PX = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 20.0})

    def _exec(self, fake, tmp_path, guard=None, submit=True):
        bplan = runner.plan_broker(runner.read_broker(fake), self.W, self.PX,
                                   min_notional=1.0)
        return runner.execute_broker(fake, bplan, self.T, tmp_path, submit=submit,
                                     guard=guard, poll_timeout=0)

    def test_all_submitted_is_ok(self, tmp_path):
        ex = self._exec(FakeBroker(), tmp_path)
        assert ex.status == "ok" and ex.error is None
        assert ex.n_submitted == 3
        assert (tmp_path / "broker_plan_2026-09-25.csv").exists()

    def test_abort_after_some_submitted_is_partial(self, tmp_path):
        ex = self._exec(FakeBroker(fail_after=1), tmp_path)
        assert ex.status == "partial"
        assert isinstance(ex.error, BrokerError)
        assert ex.n_submitted == 1
        assert list(ex.orders["status"]) == ["submitted", "failed", "not_sent"]

    def test_abort_before_any_order_is_failed(self, tmp_path):
        ex = self._exec(FakeBroker(fail_after=0), tmp_path)
        assert ex.status == "failed" and ex.n_submitted == 0

    def test_all_rejected_is_failed(self, tmp_path):
        ex = self._exec(FakeBroker(reject={"AAA", "BBB", "CCC"}), tmp_path)
        assert ex.error is None
        assert ex.status == "failed"

    def test_some_rejected_is_ok_and_counted(self, tmp_path):
        ex = self._exec(FakeBroker(reject={"CCC"}), tmp_path)
        assert ex.status == "ok"
        assert (ex.n_submitted, ex.n_rejected) == (2, 1)

    def test_guard_block_sends_nothing(self, tmp_path):
        fake = FakeBroker()
        ex = self._exec(fake, tmp_path, guard=PreTradeGuard(max_orders=1))
        assert isinstance(ex.error, PreTradeBlocked)
        assert ex.status == "failed"
        assert fake.sent == []

    def test_dry_run_never_sends(self, tmp_path):
        fake = FakeBroker()
        ex = self._exec(fake, tmp_path, submit=False)
        assert ex.status == "dry_run"
        assert fake.sent == []
        assert all(dry for _, dry in fake.submit_calls)


class TestSubmitGuard:
    def test_market_closed_blocks_unless_allowed(self):
        guard = runner.submit_guard(None, allow_closed=False)
        assert guard.block_if_market_closed and guard.block_if_account_blocked
        assert not runner.submit_guard(None, allow_closed=True).block_if_market_closed

    def test_allow_closed_never_unblocks_account(self):
        base = PreTradeGuard(block_if_account_blocked=False)
        assert runner.submit_guard(base, allow_closed=True).block_if_account_blocked

    def test_keeps_numeric_limits(self):
        base = PreTradeGuard(max_orders=7)
        assert runner.submit_guard(base, allow_closed=False).max_orders == 7


# ===========================================================================
# Orquestación run_once
# ===========================================================================
class TestExitCodes:
    def test_documented_values(self):
        assert (runner.EXIT_OK, runner.EXIT_NOT_DUE, runner.EXIT_NO_DATA,
                runner.EXIT_BROKER, runner.EXIT_LOCKED) == (0, 1, 2, 3, 4)

    def test_not_due_exits_1_and_writes_nothing(self, home, state_dir):
        assert runner.run_once(start=START) == 0
        before = _state_tree(state_dir)
        assert runner.run_once(start=START) == runner.EXIT_NOT_DUE
        # .lock se reescribe (pid) al tomar el lock y runner.log crece; el estado no
        assert _state_tree(state_dir) == before

    def test_insufficient_history_exits_2_and_writes_nothing(self, home, state_dir, monkeypatch):
        use_market_data(monkeypatch, _fake_prices().iloc[:20])
        assert runner.run_once(force=True, start=START) == runner.EXIT_NO_DATA
        assert not (state_dir / "rebalance_history.csv").exists()

    def test_too_few_assets_exits_2(self, home, state_dir, monkeypatch):
        # Solo llegan 2 tickers (política warn): < 3 activos → exit 2
        use_market_data(monkeypatch, _fake_prices()[["AAA", "BBB"]])
        assert runner.run_once(force=True, start=START,
                               data_policy=DataPolicy(on_missing="warn")) == runner.EXIT_NO_DATA
        assert not (state_dir / "ledger.json").exists()

    def test_pre_trade_block_exits_3(self, home, state_dir, broker):
        broker.account = {"trading_blocked": True}
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        assert broker.sent == []

    def test_sim_only_run_commits_status_ok(self, home, state_dir):
        assert runner.run_once(force=True, start=START) == runner.EXIT_OK
        hist = _history(state_dir)
        assert hist["status"].tolist() == ["ok"]
        assert hist["mode"].tolist() == ["sim"]


class TestNothingPersistedBeforeExecute:
    def test_broker_read_failure_leaves_state_untouched(self, home, state_dir, broker,
                                                        monkeypatch):
        assert runner.run_once(force=True, start=START) == 0
        before = _state_tree(state_dir)
        _drift(monkeypatch)
        broker.check_error = BrokerError("boom", status=500)
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        assert _state_tree(state_dir) == before

    def test_market_closed_submit_aborts_without_commit(self, home, state_dir, broker):
        broker.market_open = False
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        assert broker.sent == []
        assert not (state_dir / "ledger.json").exists()
        assert not (state_dir / "last_rebalance.txt").exists()
        assert not (state_dir / "rebalance_history.csv").exists()

    def test_allow_closed_overrides_market_closed(self, home, state_dir, broker):
        broker.market_open = False
        assert runner.run_once(force=True, start=START, mode="broker", submit=True,
                               allow_closed=True) == runner.EXIT_OK
        assert broker.sent
        assert (state_dir / "last_rebalance.txt").exists()

    def test_submit_ok_commits_everything(self, home, state_dir, broker):
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_OK
        hist = _history(state_dir)
        assert hist["status"].tolist() == ["ok"]
        assert hist["mode"].tolist() == ["broker"]
        assert int(hist["n_submitted"].iloc[0]) == len(broker.sent)
        orders = pd.read_csv(state_dir / "broker_orders.csv")
        assert set(orders["run_id"]) == set(hist["run_id"])
        assert (state_dir / "last_rebalance.txt").exists()


class TestPartialExecution:
    def test_partial_commits_executions_but_not_last_rebalance(self, home, state_dir, broker):
        broker.fail_after = 2
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        hist = _history(state_dir)
        assert hist["status"].tolist() == ["partial"]
        assert int(hist["n_submitted"].iloc[0]) == 2
        assert "forbidden" in hist["error"].iloc[0]
        # Lo efectivamente ejecutado queda comprometido (ARQ-05 → broker_orders)
        orders = pd.read_csv(state_dir / "broker_orders.csv")
        assert (orders["status"] == "submitted").sum() == 2
        assert set(orders["run_id"]) == set(hist["run_id"])
        # ... pero no se marca rebalanceado ni se avanza el ledger simulado
        assert not (state_dir / "last_rebalance.txt").exists()
        assert not (state_dir / "ledger.json").exists()
        assert not (state_dir / "exit_log.csv").exists()

    def test_next_run_retries_without_force(self, home, state_dir, broker):
        broker.fail_after = 2
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        partial_run = _history(state_dir)["run_id"].iloc[0]
        broker.fail_after = None
        # Sin --force: como no hay last_rebalance nuevo, toca reintentar
        assert runner.run_once(start=START, mode="broker", submit=True) == runner.EXIT_OK
        hist = _history(state_dir)
        assert hist["status"].tolist() == ["partial", "ok"]
        assert hist["supersedes"].iloc[1] == partial_run
        assert (state_dir / "last_rebalance.txt").read_text() == hist["date"].iloc[1]
        # Una vez OK, la siguiente corrida ya no toca
        assert runner.run_once(start=START, mode="broker", submit=True) == runner.EXIT_NOT_DUE

    def test_partial_after_ok_keeps_previous_ledger(self, home, state_dir, broker, monkeypatch):
        assert runner.run_once(force=True, start=START) == 0
        ledger_before = (state_dir / "ledger.json").read_bytes()
        last_before = (state_dir / "last_rebalance.txt").read_bytes()
        _drift(monkeypatch)
        broker.fail_after = 1
        assert runner.run_once(force=True, start=START, mode="broker",
                               submit=True) == runner.EXIT_BROKER
        assert (state_dir / "ledger.json").read_bytes() == ledger_before
        assert (state_dir / "last_rebalance.txt").read_bytes() == last_before
        assert _history(state_dir)["status"].tolist() == ["ok", "partial"]


class TestSameDayRetryAfterUncommittedSubmit:
    """Todo se envió pero el commit no llegó (crash/Ctrl-C/error de disco):
    el rerun del mismo día reusa los client_order_id, Alpaca responde 422
    y esas órdenes deben contar como enviadas (no "todo rechazado")."""

    def test_retry_commits_instead_of_failing(self, home, state_dir, monkeypatch):
        from tests.test_broker_idempotency import (
            ACCOUNT, BASE, CLOCK, ORDERS, POSITIONS, FakeUrlopen, Seq,
            _http_error, _lookup,
        )
        prefix = runner.broker_run_id(_fake_prices().index[-1])
        n = len(TICKERS)
        routes = {
            ("GET", ACCOUNT): {"account_number": "PA1", "equity": "10000"},
            ("GET", CLOCK): {"is_open": True, "timestamp": "t"},
            ("GET", POSITIONS): [],  # fills aún no visibles en posiciones
            # 1ª corrida: todas aceptadas; reintento: todas "id duplicado"
            ("POST", ORDERS): Seq(*[{"id": f"o-{tk}", "status": "accepted"}
                                    for tk in TICKERS],
                                  *[_http_error(ORDERS, 422)] * n),
        }
        for tk in TICKERS:
            routes[("GET", _lookup(f"{prefix}-{tk}-buy"))] = {
                "id": f"o-{tk}", "status": "accepted"}
            routes[("GET", f"{ORDERS}/o-{tk}")] = {
                "id": f"o-{tk}", "status": "accepted", "filled_qty": "0"}
        fake = FakeUrlopen(routes)
        monkeypatch.setattr("urllib.request.urlopen", fake)
        monkeypatch.setattr(adapter, "load_broker_config",
                            lambda *a, **k: adapter.BrokerConfig("PKTEST1234", "s", BASE))
        # Tickers sinteticos: ruteables a Alpaca solo en este maestro de prueba.
        master = InstrumentMaster(
            Instrument(id=tk, yahoo_symbol=tk, broker_symbol=tk, market="us",
                       currency="USD", asset_type="stock", sector="Tech",
                       tradable_brokers=("alpaca",))
            for tk in TICKERS)
        monkeypatch.setattr(adapter, "load_instruments", lambda *a, **k: master)

        real_commit = runner.commit_state

        def crash(*_a, **_k):
            raise OSError("crash before commit")

        monkeypatch.setattr(runner, "commit_state", crash)
        with pytest.raises(OSError, match="crash before commit"):
            runner.run_once(force=True, start=START, mode="broker", submit=True,
                            poll_timeout=0)
        assert not (state_dir / "last_rebalance.txt").exists()
        monkeypatch.setattr(runner, "commit_state", real_commit)

        assert runner.run_once(start=START, mode="broker", submit=True,
                               poll_timeout=0) == runner.EXIT_OK
        hist = _history(state_dir)
        assert hist["status"].tolist() == ["ok"]
        assert int(hist["n_submitted"].iloc[0]) == n
        assert (state_dir / "last_rebalance.txt").exists()
        assert len(fake.posts()) == 2 * n  # los reintentos no crearon órdenes


class TestDryRun:
    def test_dry_run_changes_no_file_in_live_state(self, home, state_dir, broker, monkeypatch):
        assert runner.run_once(force=True, start=START) == 0
        before = _tree(state_dir)
        _drift(monkeypatch)
        assert runner.run_once(force=True, start=START, mode="broker",
                               dry_run=True) == runner.EXIT_OK
        assert _tree(state_dir) == before
        assert broker.sent == []

    def test_dry_run_without_state_creates_nothing_in_live_state(self, home, state_dir):
        assert runner.run_once(force=True, start=START, dry_run=True) == runner.EXIT_OK
        assert not state_dir.exists()
        (plan_dir,) = (home / "results" / "plans").iterdir()
        assert (plan_dir / "sim_trades.csv").exists()

    def test_dry_run_saves_plan_under_results_plans(self, home, state_dir, broker):
        assert runner.run_once(force=True, start=START, mode="broker",
                               dry_run=True) == runner.EXIT_OK
        (plan_dir,) = (home / "results" / "plans").iterdir()
        summary = json.loads((plan_dir / "summary.json").read_text())
        assert summary["run_id"] == plan_dir.name
        assert summary["mode"] == "broker" and summary["dry_run"] is True
        for name in ("targets.csv", "exits.csv"):
            assert (plan_dir / name).exists(), name
        assert not (plan_dir / "sim_trades.csv").exists()  # broker: sin sim
        (bplan,) = plan_dir.glob("broker_plan_*.csv")
        assert (pd.read_csv(bplan)["status"] == "dry_run").all()

    def test_dry_run_reports_would_block_reasons(self, home, state_dir, broker):
        broker.market_open = False
        assert runner.run_once(force=True, start=START, mode="broker",
                               dry_run=True) == runner.EXIT_OK
        (plan_dir,) = (home / "results" / "plans").iterdir()
        summary = json.loads((plan_dir / "summary.json").read_text())
        assert any("mercado cerrado" in r for r in summary["blocked_reasons"])

    def test_dry_run_respects_schedule(self, home, state_dir):
        assert runner.run_once(start=START) == 0
        assert runner.run_once(start=START, dry_run=True) == runner.EXIT_NOT_DUE

    def test_dry_run_works_while_state_is_locked(self, home, state_dir):
        assert runner.run_once(force=True, start=START) == 0
        with StateStore(state_dir):
            assert runner.run_once(force=True, start=START,
                                   dry_run=True) == runner.EXIT_OK

    def test_broker_without_submit_is_a_dry_run(self, home, state_dir, broker):
        assert runner.run_once(force=True, start=START, mode="broker") == runner.EXIT_OK
        assert not state_dir.exists()
        assert broker.sent == []
        assert (home / "results" / "plans").is_dir()

    def test_check_only_is_read_only(self, home, state_dir):
        assert runner.run_once(check_only=True, start=START) == runner.EXIT_OK
        assert not state_dir.exists()


# ===========================================================================
# Validación de flags en main
# ===========================================================================
class TestMainFlags:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []
        monkeypatch.setattr(runner, "run_once", lambda **kw: calls.append(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        return calls

    @pytest.mark.parametrize("argv, msg", [
        (["--submit"], "--submit requires --mode broker"),
        (["--broker", "--submit", "--dry-run"], "not allowed with argument --submit"),
        (["--allow-closed"], "--allow-closed requires --mode broker"),
    ])
    def test_invalid_combinations_are_argparse_errors(self, captured, capsys, argv, msg):
        with pytest.raises(SystemExit) as exc:
            runner.main(argv)
        assert exc.value.code == 2
        assert msg in capsys.readouterr().err
        assert captured == []

    def test_dry_run_and_allow_closed_are_forwarded(self, captured):
        assert runner.main(["--mode", "broker", "--submit", "--allow-closed"]) == 0
        assert runner.main(["--mode", "broker", "--dry-run"]) == 0
        assert captured[0]["allow_closed"] is True and captured[0]["dry_run"] is False
        assert captured[0]["mode"] == "broker"
        assert captured[1]["dry_run"] is True and captured[1]["submit"] is False

    def test_run_once_rejects_submit_without_broker(self):
        with pytest.raises(ValueError, match="--submit requires --mode broker"):
            runner.run_once(submit=True)

    def test_unexpected_error_has_its_own_exit_code(self, monkeypatch):
        def boom(**_kw):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(runner, "run_once", boom)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        rc = runner.main([])
        assert rc == runner.EXIT_UNEXPECTED
        assert rc not in (0, 1, 2, 3, 4)


def test_makefile_broker_dry_uses_dry_run_without_force():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    (line,) = [ln for ln in makefile.splitlines()
               if "scripts.run_periodic_rebalance" in ln and "--mode broker" in ln]
    assert "--dry-run" in line
    assert "--force" not in line


def test_random_state_untouched_by_phases():
    state = np.random.get_state()[1].copy()
    runner.build_sim_plan(PositionLedger(10_000.0),
                          pd.Series(1.0 / len(TICKERS), index=TICKERS),
                          _fake_prices().iloc[-1], _fake_prices().index[-1], LIVE)
    assert (np.random.get_state()[1] == state).all()
