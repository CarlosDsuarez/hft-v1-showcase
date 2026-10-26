"""
test_live_runner.py — Integración del runner vivo con fakes, sin red (ARQ-18).

Corridas completas de `run_once` / `main` contra FakeMarket (barras
sintéticas deterministas) y FakeBroker (libro con estado, fills y fallos
inyectables), con state_dir en tmp_path. Cada caso es una secuencia de
corridas como las haría el cron: primera corrida, no toca, --force, fallo de
datos, fallo del broker a mitad, rerun del mismo día, ticker en cartera que
desaparece de los datos, dry-run.

Los tests describen el comportamiento deseado de ARQ-02..06 (ver
auditoria_arquitectura_prompts.md). Los que dependen de un fix aún no hecho
están marcados xfail(strict=True) con el ID: cuando el fix llegue, el xpass
rompe la suite y obliga a quitar la marca.

Correr solo esta suite: `make test-integration` (= pytest -m integration).
"""

import json
from collections import Counter

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.live_state import StateStore
from src.market_data import DataUnavailableError, FallbackChain
from src.position_ledger import CASH_TOLERANCE
from tests.integration.fakes import BROKER_EQUITY, OUTSIDER, START

pytestmark = pytest.mark.integration

EVERY_N = runner.REBALANCE_EVERY_N
CENT = 0.01


def _largest_holding(positions: pd.Series) -> str:
    held = positions[positions > 0]
    assert not held.empty, "la corrida previa no dejó posiciones"
    return str(held.idxmax())


def _assert_manifest_consistent(live) -> None:
    """El manifiesto describe exactamente los archivos publicados."""
    with StateStore(live.state_dir) as store:  # además: el lock quedó libre
        assert store.verify() == []
    staging = live.state_dir / ".staging"
    assert not staging.exists() or not any(staging.iterdir())


# ===========================================================================
# 1. Primera corrida
# ===========================================================================
class TestFirstRun:
    def test_sim_commits_complete_and_consistent_state(self, live):
        assert live.run() == runner.EXIT_OK

        for name in ("ledger.json", "transactions.csv", "last_rebalance.txt",
                     "latest_snapshot.csv", "rebalance_history.csv",
                     "state_manifest.json"):
            assert (live.state_dir / name).exists(), name
        hist = live.history()
        assert hist[["mode", "status"]].values.tolist() == [["sim", "ok"]]
        assert not hist["forced"].iloc[0]
        assert pd.isna(hist["supersedes"].iloc[0])
        assert live.last_rebalance() == live.as_of()
        assert live.manifest()["run_id"] == hist["run_id"].iloc[0]
        assert live.manifest()["as_of"] == live.as_of()

        ledger = live.ledger()
        assert not ledger.positions().empty
        assert ledger.cash >= -CASH_TOLERANCE
        assert ledger.nav(live.market.close.iloc[-1]) == pytest.approx(hist["nav"].iloc[0])
        txns = live.csv("transactions.csv")
        assert set(txns["run_id"]) == {hist["run_id"].iloc[0]}
        _assert_manifest_consistent(live)

    def test_broker_submits_plan_sized_on_broker_equity(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK

        assert broker.orders and {o["side"] for o in broker.orders} == {"buy"}
        spent = sum(o["qty"] * o["price"] for o in broker.orders)
        assert spent <= BROKER_EQUITY + CENT  # qty redondeada a 1e-6 acciones
        assert broker.cash >= -CENT
        orders = live.csv("broker_orders.csv")
        sent = orders[orders["status"] == "submitted"]
        assert set(sent["client_order_id"]) == {o["client_order_id"] for o in broker.orders}
        hist = live.history()
        assert hist[["mode", "status"]].values.tolist() == [["broker", "ok"]]
        assert int(hist["n_submitted"].iloc[0]) == len(broker.orders)
        assert live.last_rebalance() == live.as_of()
        # Sin ledger previo no hay nada que respaldar antes del espejo
        assert not list(live.state_dir.glob("ledger_local_backup_*"))
        _assert_manifest_consistent(live)

    def test_next_broker_run_reads_fills_back_from_broker(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        live.market.advance(EVERY_N)
        book = broker.positions()
        assert live.submit() == runner.EXIT_OK
        # ARQ-02: el espejo local es la cartera del broker leída al inicio
        mirror = live.ledger().positions()
        pd.testing.assert_series_equal(mirror.sort_index(), book.sort_index(),
                                       check_names=False)


# ===========================================================================
# 2. Corrida no debida (exit 1)
# ===========================================================================
class TestNotDue:
    def test_exits_1_and_changes_nothing(self, live):
        assert live.run() == runner.EXIT_OK
        before = live.tree()
        live.market.advance(EVERY_N - 1)
        assert live.run() == runner.EXIT_NOT_DUE
        assert live.tree() == before

    def test_due_again_after_every_n_sessions(self, live):
        assert live.run() == runner.EXIT_OK
        live.market.advance(EVERY_N)
        assert live.run() == runner.EXIT_OK
        assert live.history()["forced"].tolist() == [False, False]
        assert live.last_rebalance() == live.as_of()

    def test_check_only_follows_schedule_without_writing(self, live):
        assert live.run() == runner.EXIT_OK
        live.market.advance(EVERY_N - 1)
        before = live.tree(all_files=True)
        assert live.run(check_only=True) == runner.EXIT_NOT_DUE
        live.market.advance(1)
        assert live.run(check_only=True) == runner.EXIT_OK
        assert live.tree(all_files=True) == before

    def test_broker_not_due_never_contacts_broker(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        broker.calls.clear()
        live.market.advance(EVERY_N - 1)
        before = live.tree()
        assert live.submit() == runner.EXIT_NOT_DUE
        assert broker.calls == []
        assert live.tree() == before


# ===========================================================================
# 3. --force
# ===========================================================================
class TestForce:
    def test_force_rebalances_before_due_and_is_recorded(self, live):
        assert live.run() == runner.EXIT_OK
        live.market.advance(5)
        assert live.run(force=True) == runner.EXIT_OK
        hist = live.history()
        assert hist["forced"].tolist() == [False, True]
        assert hist["supersedes"].isna().all()  # otro as_of: no reemplaza nada
        assert live.last_rebalance() == live.as_of()
        _assert_manifest_consistent(live)

    def test_cli_force_flag(self, live, monkeypatch):
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        assert runner.main(["--start", START]) == runner.EXIT_OK
        live.market.advance(5)
        assert runner.main(["--start", START]) == runner.EXIT_NOT_DUE
        assert runner.main(["--force", "--start", START]) == runner.EXIT_OK
        assert live.history()["forced"].tolist() == [False, True]


# ===========================================================================
# 4. Fallo de datos
# ===========================================================================
class _DownSource:
    name = "yahoo-down"

    def get_bars(self, symbols, start, end):
        raise ConnectionError("yahoo: read timed out")


class TestDataFailure:
    def test_all_sources_down_exits_7_and_writes_nothing(self, live):
        assert live.run() == runner.EXIT_OK
        before = live.tree()
        live.market.advance(EVERY_N)
        live.market.fail = DataUnavailableError("All data sources exhausted")
        assert live.run() == runner.EXIT_DATA
        assert live.tree() == before
        assert live.alerts()[-1]["level"] == "error"  # ARQ-21: aviso, no silencio
        # Cuando la fuente vuelve, la corrida pendiente se hace sin --force
        live.market.fail = None
        assert live.run() == runner.EXIT_OK

    def test_fallback_source_completes_the_run(self, live, monkeypatch):
        chain = FallbackChain([_DownSource(), live.market])
        monkeypatch.setattr(runner, "make_market_data_provider", lambda *a, **k: chain)
        assert live.run() == runner.EXIT_OK
        assert live.history()["status"].tolist() == ["ok"]

    def test_universe_symbol_missing_exits_7(self, live):
        live.market.drop.add("BBB")
        assert live.run() == runner.EXIT_DATA
        assert live.tree() == {}

    def test_insufficient_history_exits_2(self, live):
        live.market.upto = 120
        assert live.run() == runner.EXIT_NO_DATA
        assert live.tree() == {}

    def test_provider_crash_releases_lock_and_next_run_recovers(self, live, monkeypatch):
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        assert live.run() == runner.EXIT_OK
        before = live.tree()
        live.market.advance(EVERY_N)
        live.market.fail = ConnectionResetError("connection reset by peer")
        assert runner.main(["--start", START]) == runner.EXIT_UNEXPECTED
        assert live.tree() == before
        live.market.fail = None
        assert live.run() == runner.EXIT_OK
        assert live.history()["status"].tolist() == ["ok", "ok"]

    def test_broker_mode_data_failure_never_contacts_broker(self, live):
        broker = live.connect_broker()
        live.market.fail = DataUnavailableError("All data sources exhausted")
        assert live.submit() == runner.EXIT_DATA
        assert broker.calls == []
        assert live.tree() == {}


# ===========================================================================
# 5. Fallo del broker a mitad
# ===========================================================================
def _forced_partial_after_ok(live):
    """ok en t0; --force 3 sesiones después falla tras la 1ª orden."""
    broker = live.connect_broker()
    assert live.submit() == runner.EXIT_OK
    t0 = live.as_of()
    live.market.advance(3)
    broker.fail_after = 1
    assert live.submit(force=True) == runner.EXIT_BROKER
    broker.fail_after = None
    return broker, t0


class TestBrokerFailureMidway:
    def test_abort_after_two_orders_commits_only_what_was_executed(self, live):
        broker = live.connect_broker()
        broker.fail_after = 2
        assert live.submit() == runner.EXIT_BROKER

        assert len(broker.orders) == 2
        orders = live.csv("broker_orders.csv")
        status = orders["status"].value_counts().to_dict()
        assert status.get("submitted") == 2 and status.get("failed") == 1
        assert set(status) <= {"submitted", "failed", "not_sent"}
        hist = live.history()
        assert hist["status"].tolist() == ["partial"]
        assert int(hist["n_submitted"].iloc[0]) == 2
        assert "forbidden" in hist["error"].iloc[0]
        # No se marca rebalanceado ni se publica el espejo: la próxima reintenta
        assert live.last_rebalance() is None
        assert not (live.state_dir / "ledger.json").exists()
        journal = (live.state_dir / f"broker_journal_{live.as_of()}.jsonl").read_text()
        events = [json.loads(line) for line in journal.splitlines()]
        assert events[-1]["event"] == "abort"
        assert sum(e.get("status") == "submitted" for e in events) == 2

    def test_same_day_retry_completes_without_duplicate_orders(self, live):
        broker = live.connect_broker()
        broker.fail_after = 2
        assert live.submit() == runner.EXIT_BROKER
        partial_run = live.history()["run_id"].iloc[0]
        first_tickers = {o["ticker"] for o in broker.orders}
        broker.fail_after = None

        assert live.submit() == runner.EXIT_OK  # sin --force
        hist = live.history()
        assert hist["status"].tolist() == ["partial", "ok"]
        assert hist["supersedes"].iloc[1] == partial_run
        assert live.last_rebalance() == live.as_of()
        # Cada (ticker, lado) llegó una sola vez al broker entre ambas corridas
        per_order = Counter((o["ticker"], o["side"]) for o in broker.orders)
        assert set(per_order.values()) == {1}
        retried = {o["ticker"] for o in broker.orders[2:]}
        assert retried and retried.isdisjoint(first_tickers)
        # Ya en objetivo: la siguiente corrida no toca
        assert live.submit() == runner.EXIT_NOT_DUE

    def test_account_read_failure_exits_3_and_commits_nothing(self, live):
        broker = live.connect_broker()
        broker.fail_check = RuntimeError("BROKER_ACCOUNT_ID mismatch")
        assert live.submit() == runner.EXIT_BROKER
        assert broker.orders == []
        assert live.tree() == {}

    def test_fill_poll_failure_is_partial_and_rerun_reconciles(self, live):
        broker = live.connect_broker()
        broker.fail_poll = TimeoutError("poll timed out")
        assert live.submit() == runner.EXIT_BROKER
        n_sent = len(broker.orders)
        hist = live.history()
        assert hist["status"].tolist() == ["partial"]
        assert int(hist["n_submitted"].iloc[0]) == n_sent
        broker.fail_poll = None

        assert live.submit() == runner.EXIT_OK
        assert len(broker.orders) == n_sent  # el libro ya estaba en objetivo
        assert live.history()["status"].tolist() == ["partial", "ok"]

    def test_forced_partial_keeps_last_rebalance_and_mirror(self, live):
        _, t0 = _forced_partial_after_ok(live)
        assert live.history()["status"].tolist() == ["ok", "partial"]
        assert live.last_rebalance() == t0

    @pytest.mark.xfail(strict=True, reason=(
        "ARQ-06 (paso 4): tras status=partial 'la siguiente corrida reintenta'. "
        "decide_schedule solo mira last_rebalance: si la corrida parcial fue "
        "--force, el cron siguiente responde 'no toca' (exit 1) y el libro queda "
        "a medio rebalancear hasta cumplir every_n."))
    def test_next_run_retries_a_forced_partial(self, live):
        _forced_partial_after_ok(live)
        rc = live.submit()  # mismo día, cron sin --force
        assert rc == runner.EXIT_OK, f"rc={rc}"
        assert live.history()["status"].tolist() == ["ok", "partial", "ok"]


# ===========================================================================
# 6. Rerun el mismo día
# ===========================================================================
class TestSameDayRerun:
    def test_without_force_is_a_noop(self, live):
        assert live.run() == runner.EXIT_OK
        before = live.tree()
        assert live.run() == runner.EXIT_NOT_DUE
        # La guarda es por as_of comprometido, no por conteo de sesiones
        assert live.run(every_n=0) == runner.EXIT_NOT_DUE
        assert live.tree() == before

    def test_forced_rerun_supersedes_previous_run(self, live):
        assert live.run() == runner.EXIT_OK
        first = live.manifest()["run_id"]
        assert live.run(force=True) == runner.EXIT_OK

        hist = live.history()
        assert hist["run_id"].iloc[0] == first
        assert hist["forced"].tolist() == [False, True]
        assert hist["supersedes"].iloc[1] == first
        assert live.manifest()["run_id"] == hist["run_id"].iloc[1]
        assert set(live.csv("transactions.csv")["run_id"]) <= set(hist["run_id"])
        assert live.ledger().cash >= -CASH_TOLERANCE
        _assert_manifest_consistent(live)

    def test_broker_forced_rerun_sends_no_duplicate_orders(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        n_sent = len(broker.orders)
        assert live.submit() == runner.EXIT_NOT_DUE
        assert live.submit(force=True) == runner.EXIT_OK
        assert len(broker.orders) == n_sent
        hist = live.history()
        assert hist["supersedes"].iloc[1] == hist["run_id"].iloc[0]


# ===========================================================================
# 7. Ticker en cartera que desaparece de los datos
# ===========================================================================
class TestHeldTickerVanishes:
    def test_missing_from_download_aborts_6_without_writing(self, live):
        assert live.run() == runner.EXIT_OK
        held = _largest_holding(live.ledger().positions())
        before = live.tree()
        live.market.advance(EVERY_N)
        live.market.drop.add(held)
        assert live.run() == runner.EXIT_MISSING_PRICE
        assert live.tree() == before

    def test_no_recent_prices_aborts_6_without_writing(self, live):
        assert live.run() == runner.EXIT_OK
        held = _largest_holding(live.ledger().positions())
        before = live.tree()
        live.market.advance(EVERY_N)
        live.market.nan_tail[held] = 10  # suspendido: más allá del ffill
        assert live.run() == runner.EXIT_MISSING_PRICE
        assert live.tree() == before

    def test_ticker_leaving_universe_is_still_priced_and_sold(self, live):
        assert live.run() == runner.EXIT_OK
        held = _largest_holding(live.ledger().positions())
        live.universe.remove(held)
        live.market.advance(EVERY_N)
        assert live.run() == runner.EXIT_OK
        assert held in live.market.requests[-1]  # universo ∪ posiciones
        assert held not in live.ledger().positions().index
        nav = live.history()["nav"]
        assert 0.9 < nav.iloc[1] / nav.iloc[0] < 1.1  # el NAV no colapsa

    def test_broker_held_ticker_missing_aborts_before_any_order(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        live.market.advance(EVERY_N)
        assert live.submit() == runner.EXIT_OK  # espejo = libro con fills
        held = _largest_holding(live.ledger().positions())
        n_sent, book = len(broker.orders), dict(broker.book)
        before = live.tree()
        live.market.advance(EVERY_N)
        live.market.drop.add(held)
        assert live.submit() == runner.EXIT_MISSING_PRICE
        assert len(broker.orders) == n_sent and broker.book == book
        assert live.tree() == before

    def test_broker_only_holding_without_data_aborts_6(self, live):
        px = float(live.market.marks()[OUTSIDER])
        broker = live.connect_broker(cash=BROKER_EQUITY - 10 * px,
                                     positions={OUTSIDER: (10.0, px)})
        live.market.drop.add(OUTSIDER)
        assert live.submit() == runner.EXIT_MISSING_PRICE
        assert broker.orders == []
        assert live.tree() == {}

    @pytest.mark.xfail(strict=True, reason=(
        "ARQ-02/ARQ-03: en modo broker la descarga pide universo ∪ posiciones "
        "del ledger espejo, que va una corrida atrás del broker. Una posición "
        "solo-broker fuera del universo nunca se descarga aunque haya datos → "
        "exit 6 en cada corrida (el espejo nunca se publica) en vez de "
        "valorarla y venderla."))
    def test_broker_only_holding_outside_universe_is_priced_and_sold(self, live):
        px = float(live.market.marks()[OUTSIDER])
        broker = live.connect_broker(cash=BROKER_EQUITY - 10 * px,
                                     positions={OUTSIDER: (10.0, px)})
        rc = live.submit()
        assert rc == runner.EXIT_OK, f"rc={rc}"
        assert OUTSIDER not in broker.book

    @pytest.mark.xfail(strict=True, reason=(
        "ARQ-02/ARQ-03: en la 1ª corrida broker el espejo está vacío, así que "
        "un ticker del broker que falta en la descarga se trata como faltante "
        "del universo (exit 7, datos) y no como posición abierta sin precio "
        "(exit 6)."))
    def test_broker_held_ticker_missing_on_first_mirror_is_exit_6(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        held = _largest_holding(broker.positions())
        live.market.advance(EVERY_N)
        live.market.drop.add(held)
        rc = live.submit()
        assert rc == runner.EXIT_MISSING_PRICE, f"rc={rc}"


# ===========================================================================
# 8. Dry-run no muta
# ===========================================================================
class TestDryRun:
    def test_sim_dry_run_mutates_nothing_and_saves_plan(self, live):
        assert live.run() == runner.EXIT_OK
        live.market.advance(EVERY_N)
        before = live.tree(all_files=True)
        assert live.run(dry_run=True) == runner.EXIT_OK
        assert live.tree(all_files=True) == before

        (plan_dir,) = live.plans_dir.iterdir()
        summary = json.loads((plan_dir / "summary.json").read_text())
        assert summary["dry_run"] is True and summary["as_of"] == live.as_of()
        assert (plan_dir / "sim_trades.csv").exists()
        # No marcó nada: la corrida real sigue tocando
        assert live.run() == runner.EXIT_OK

    def test_broker_dry_run_sends_nothing_and_mutates_nothing(self, live):
        broker = live.connect_broker()
        assert live.submit() == runner.EXIT_OK
        live.market.advance(EVERY_N)
        n_sent, book, cash = len(broker.orders), dict(broker.book), broker.cash
        before = live.tree(all_files=True)

        assert live.run(mode="broker", dry_run=True) == runner.EXIT_OK
        assert live.run(mode="broker") == runner.EXIT_OK  # sin --submit = dry-run
        assert live.tree(all_files=True) == before
        assert len(broker.orders) == n_sent
        assert broker.book == book and broker.cash == cash
        assert len(broker.dry_run_plans) == 2
        plans = sorted(live.plans_dir.iterdir())
        assert len(plans) == 2
        for plan_dir in plans:
            assert list(plan_dir.glob("broker_plan_*.csv"))
            assert not list(plan_dir.glob("broker_journal_*"))

    def test_dry_run_on_fresh_install_creates_no_state(self, live):
        assert live.run(dry_run=True) == runner.EXIT_OK
        assert not live.state_dir.exists()


# ===========================================================================
# Recuperación ante fallo a mitad de commit / lock
# ===========================================================================
class TestRecovery:
    def test_crash_mid_commit_keeps_previous_state_and_next_run_commits(
            self, live, monkeypatch):
        assert live.run() == runner.EXIT_OK
        before = live.tree()
        live.market.advance(EVERY_N)
        real_append = StateStore.append_csv

        def disk_full(self, name, frame, *a, **k):
            if name == "rebalance_history.csv":
                raise OSError("No space left on device")
            return real_append(self, name, frame, *a, **k)

        monkeypatch.setattr(StateStore, "append_csv", disk_full)
        with pytest.raises(OSError, match="No space left"):
            live.run()
        assert live.tree() == before
        monkeypatch.setattr(StateStore, "append_csv", real_append)

        assert live.run() == runner.EXIT_OK
        assert len(live.history()) == 2
        _assert_manifest_consistent(live)

    def test_crash_after_fills_before_commit_is_reconciled_same_day(
            self, live, monkeypatch):
        broker = live.connect_broker()
        real_commit = runner.commit_state

        def crash(*_a, **_k):
            raise OSError("killed before commit")

        monkeypatch.setattr(runner, "commit_state", crash)
        with pytest.raises(OSError, match="killed before commit"):
            live.submit()
        n_sent = len(broker.orders)
        assert n_sent > 0 and live.last_rebalance() is None
        monkeypatch.setattr(runner, "commit_state", real_commit)

        assert live.submit() == runner.EXIT_OK  # mismo día, sin --force
        assert len(broker.orders) == n_sent  # nada se reenvía
        assert live.history()["status"].tolist() == ["ok"]

    def test_lock_held_by_another_run_exits_4_without_reading(self, live):
        with StateStore(live.state_dir):
            assert live.run(force=True) == runner.EXIT_LOCKED
        assert live.market.requests == []
        assert live.tree() == {}
