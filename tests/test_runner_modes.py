"""
test_runner_modes.py — `--mode {sim,broker}`: una sola fuente de verdad.

broker: la cartera (shares, avg_entry, cash, equity) sale del broker ANTES de
evaluar salidas y dimensionar; ledger.json pasa a ser espejo del broker.
sim: el ledger local es la cartera y nada toca la red.

Sin red: un broker falso se inyecta via `runner.make_broker_client`.
El espejo/respaldo del ledger se publica al comprometer (ARQ-06): los casos
que leen archivos persistidos usan submit=True; el dry-run no escribe nada.
"""

import json
import logging

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src import strategy
from src.position_ledger import PositionLedger
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

PX_T = _fake_prices().iloc[-1]
DAY = _fake_prices().index[-1].date().isoformat()
EQUITY = 10_000.0
# Broker: CCC pierde $10 (10 acc, entrada px+1) → stop_loss_dollars; DDD plano.
BROKER_BOOK = {"CCC": (10.0, PX_T["CCC"] + 1.0), "DDD": (1.0, PX_T["DDD"])}
BROKER_CASH = EQUITY - sum(q * PX_T[tk] for tk, (q, _) in BROKER_BOOK.items())


class FakeBroker:
    """Doble de AlpacaClient (sin HTTP) con un libro DISTINTO al ledger."""

    def __init__(self, fail_check=False):
        self.fail_check = fail_check
        self.calls = []
        self.submitted = []

    def check(self):
        self.calls.append("check")
        if self.fail_check:
            raise RuntimeError("Cuenta equivocada")
        return {"account_number": "PA1", "equity": str(EQUITY),
                "buying_power": str(2 * EQUITY), "market_open": True}

    def get_account(self):
        self.calls.append("get_account")
        return {"account_number": "PA1", "equity": str(EQUITY),
                "cash": str(BROKER_CASH)}

    def get_positions(self):
        self.calls.append("get_positions")
        return pd.DataFrame([
            {"ticker": tk, "qty": q, "market_value": q * PX_T[tk],
             "avg_entry": e, "side": "long"}
            for tk, (q, e) in BROKER_BOOK.items()])

    def submit_orders(self, plan, dry_run=True, run_id=None, on_result=None):
        self.calls.append("submit_orders")
        self.submitted.append((plan, dry_run))
        results = [{"ticker": r.ticker, "side": r.side, "qty": r.qty,
                    "status": "dry_run" if dry_run else "submitted",
                    "id": None if dry_run else f"id-{r.ticker}"}
                   for r in plan.itertuples()]
        for r in results:
            if on_result is not None:
                on_result(r)
        return results

    def poll_orders(self, ids, timeout=30.0):
        self.calls.append("poll_orders")
        return {}


@pytest.fixture
def state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    use_market_data(monkeypatch, _fake_prices())
    # Sizing equal-weight sin tilts: el plan esperado se calcula a mano. Las
    # salidas siguen corriendo dentro de build_targets (config viva).
    monkeypatch.setattr(
        strategy, "hrp_weights",
        lambda rets: pd.Series(1.0 / rets.shape[1], index=rets.columns))
    monkeypatch.setattr(
        strategy, "conviction_to_tilts",
        lambda conviction, base_weights=None, **k: base_weights / base_weights.sum())
    path = home / "data" / "live_state"
    path.mkdir(parents=True)
    return path


def _seed_sim_ledger(state):
    """Ledger simulado DISTINTO al broker: AAA -$10 (stop en sim), BBB plano."""
    lg = PositionLedger(initial_cash=100_000.0)
    entry = PX_T.copy()
    entry["AAA"] += 1.0
    lg.apply_trades(pd.DataFrame([{"ticker": "AAA", "shares": 10.0},
                                  {"ticker": "BBB", "shares": 10.0}]),
                    entry, t=pd.Timestamp("2024-04-01"))
    runner.save_ledger(lg, state / "ledger.json")
    return lg


def _install(monkeypatch, fake):
    created = []

    def factory(state_dir):
        created.append(state_dir)
        return fake

    monkeypatch.setattr(runner, "make_broker_client", factory)
    return created


def _run(**kw):
    return runner.run_once(force=True, start="2024-01-01", **kw)


def _drift_logs(caplog):
    return [r.getMessage() for r in caplog.records if "drift" in r.getMessage().lower()]


class TestBrokerMode:
    def test_exits_evaluated_on_broker_book_not_ledger(self, state, monkeypatch):
        _seed_sim_ledger(state)
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0
        exits = pd.read_csv(state / "exit_log.csv")
        assert exits["ticker"].tolist() == ["CCC"]
        assert exits["rule"].tolist() == ["stop_loss_dollars"]

    def test_plan_sized_on_broker_equity_and_qty(self, state, monkeypatch):
        _seed_sim_ledger(state)
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0
        plan = pd.read_csv(state / f"broker_plan_{DAY}.csv").set_index("ticker")
        w = 1.0 / (len(TICKERS) - 1)          # CCC cerrado → resto renormalizado
        target_w = pd.Series(w, index=TICKERS).mask(lambda s: s.index == "CCC", 0.0)
        # Como el backtest (R-02): CCC (salida) se vende entero; DDD, lo único
        # que se mantiene, rota a lo sumo turnover_limit; la caja (y lo de
        # CCC) va directo al objetivo.
        limit = strategy.load_strategy_config().turnover_limit
        kept = pd.Series(0.0, index=TICKERS).mask(lambda s: s.index == "DDD", 1.0)
        rotated = kept + limit / (target_w - kept).abs().sum() * (target_w - kept)
        r = PX_T["DDD"] / EQUITY              # fracción del NAV que rota
        w_exec = r * rotated + (1.0 - r) * target_w
        held = {tk: q for tk, (q, _) in BROKER_BOOK.items()}
        assert sorted(plan.index) == sorted(TICKERS)
        for tk in TICKERS:
            target = w_exec[tk] * EQUITY / PX_T[tk]
            diff = target - held.get(tk, 0.0)
            assert plan.loc[tk, "side"] == ("buy" if diff > 0 else "sell"), tk
            assert plan.loc[tk, "qty"] == pytest.approx(abs(diff), abs=1e-6), tk

    def test_ledger_becomes_mirror_of_broker(self, state, monkeypatch):
        _seed_sim_ledger(state)
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0
        saved = PositionLedger.from_dict(json.loads((state / "ledger.json").read_text()))
        assert saved.positions().to_dict() == {"CCC": 10.0, "DDD": 1.0}
        assert saved.cash == pytest.approx(BROKER_CASH)
        assert saved.transactions_df().empty   # sin trades simulados

    def test_sim_book_backed_up_before_first_mirror(self, state, monkeypatch):
        lg = _seed_sim_ledger(state)
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0
        backup = state / f"ledger_local_backup_{DAY}.json"
        assert json.loads(backup.read_text()) == lg.to_dict()

    def test_legacy_ledger_without_txns_backed_up(self, state, monkeypatch):
        legacy = {"cash": 50_000.0, "realized_pnl": 0.0, "lots": [
            {"ticker": "AAA", "shares": 10.0, "entry_price": float(PX_T["AAA"]),
             "entry_date": "2024-04-01T00:00:00"}]}
        (state / "ledger.json").write_text(json.dumps(legacy))
        expected = runner.load_ledger(state / "ledger.json").to_dict()
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0
        backup = state / f"ledger_local_backup_{DAY}.json"
        assert json.loads(backup.read_text()) == expected

    def test_no_backup_without_file_or_once_in_sync(self, state, monkeypatch):
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0  # sin ledger.json previo
        assert not list(state.glob("ledger_local_backup_*"))
        assert _run(mode="broker", submit=True) == 0  # ya es espejo, sin drift
        assert not list(state.glob("ledger_local_backup_*"))

    def test_drift_is_logged_when_ledger_differs(self, state, monkeypatch, caplog):
        _seed_sim_ledger(state)
        _install(monkeypatch, FakeBroker())
        with caplog.at_level(logging.INFO, logger="periodic_rebalance"):
            assert _run(mode="broker") == 0
        msgs = _drift_logs(caplog)
        assert msgs, "sin log de drift"
        for tk in ("AAA", "BBB", "CCC", "DDD"):
            assert any(tk in m for m in msgs), tk

    def test_no_drift_once_ledger_mirrors_broker(self, state, monkeypatch, caplog):
        _install(monkeypatch, FakeBroker())
        assert _run(mode="broker", submit=True) == 0  # 1a corrida: espeja
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="periodic_rebalance"):
            assert _run(mode="broker") == 0      # broker sin cambios
        assert _drift_logs(caplog) == []

    def test_broker_read_failure_exits_3_and_changes_nothing(self, state, monkeypatch):
        _seed_sim_ledger(state)
        before = (state / "ledger.json").read_text()
        _install(monkeypatch, FakeBroker(fail_check=True))
        assert _run(mode="broker", submit=True) == 3
        assert (state / "ledger.json").read_text() == before
        assert not (state / f"broker_plan_{DAY}.csv").exists()
        assert not (state / "last_rebalance.txt").exists()

    def test_dry_run_does_not_mark_rebalanced(self, state, monkeypatch):
        fake = FakeBroker()
        _install(monkeypatch, fake)
        assert _run(mode="broker") == 0
        assert fake.submitted[-1][1] is True     # dry_run
        assert not (state / "last_rebalance.txt").exists()
        assert not (state / "rebalance_history.csv").exists()

    def test_submit_marks_rebalanced_with_broker_state(self, state, monkeypatch):
        fake = FakeBroker()
        _install(monkeypatch, fake)
        assert _run(mode="broker", submit=True) == 0
        assert fake.submitted[-1][1] is False
        assert (state / "last_rebalance.txt").read_text() == DAY
        (row,) = pd.read_csv(state / "rebalance_history.csv").to_dict("records")
        assert row["mode"] == "broker"
        assert row["nav"] == pytest.approx(EQUITY)
        assert row["cash"] == pytest.approx(BROKER_CASH)


class TestSimMode:
    def test_sim_never_builds_client_nor_touches_network(self, state, monkeypatch):
        created = _install(monkeypatch, FakeBroker())

        def no_network(*a, **k):
            raise AssertionError("red en modo sim")

        monkeypatch.setattr("urllib.request.urlopen", no_network)
        assert _run() == 0                       # default = sim
        assert created == []
        assert not list(state.glob("broker_*"))
        (row,) = pd.read_csv(state / "rebalance_history.csv").to_dict("records")
        assert row["mode"] == "sim"

    def test_sim_exits_evaluated_on_ledger_book(self, state, monkeypatch):
        _seed_sim_ledger(state)
        fake = FakeBroker()
        _install(monkeypatch, fake)
        assert _run(mode="sim") == 0
        assert pd.read_csv(state / "exit_log.csv")["ticker"].tolist() == ["AAA"]
        assert fake.calls == []

    def test_submit_requires_broker_mode(self, state):
        with pytest.raises(ValueError, match="broker"):
            _run(mode="sim", submit=True)


class TestCli:
    @pytest.fixture
    def seen(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(runner, "run_once",
                            lambda **kw: captured.update(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        return captured

    def test_default_mode_is_sim(self, seen):
        assert runner.main([]) == 0
        assert seen["mode"] == "sim" and seen["submit"] is False

    def test_mode_broker_submit(self, seen):
        assert runner.main(["--mode", "broker", "--submit"]) == 0
        assert seen["mode"] == "broker" and seen["submit"] is True

    def test_submit_without_broker_mode_is_usage_error(self, seen):
        with pytest.raises(SystemExit) as exc:
            runner.main(["--submit"])
        assert exc.value.code == 2
        assert seen == {}

    def test_legacy_broker_flag_means_broker_mode(self, seen):
        assert runner.main(["--broker"]) == 0
        assert seen["mode"] == "broker"
