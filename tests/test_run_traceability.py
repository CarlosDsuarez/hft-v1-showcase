"""
test_run_traceability.py — Trazabilidad de corridas vivas (ARQ-21).

run_id en todos los logs + runner.log rotativo en state_dir/logs/,
procedencia en rebalance_history (git, config, datos, esquemas) y hook de
alertas (Notifier: consola/archivo/email opcional). Sintético, sin red.
"""

import json
import logging
import re
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
import src.provenance as provenance
from src.backtest_engine import CASH_COL
from src.live_state import STATE_SCHEMA_VERSION, StateStore
from src.notifier import (
    Alert, ConsoleNotifier, EmailConfig, EmailNotifier, FileNotifier, MultiNotifier,
    build_notifier, load_email_config,
)
from src.position_ledger import LEDGER_SCHEMA_VERSION
from src.schemas import CONFIG_SCHEMA_VERSION as SCHEMA_VERSION
from src.settings import (
    LOG_DATEFMT, LOG_FORMAT, RunIdFilter, current_run_id, load_settings,
    log_to_file, run_logging,
)
from src.strategy import CONFIG_SCHEMA_VERSION
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

START = "2024-01-01"
RUN_ID = "20240101T000000Z-deadbeef"


class Recorder:
    def __init__(self):
        self.alerts = []

    def notify(self, alert):
        self.alerts.append(alert)


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


def _history(state_dir):
    return pd.read_csv(state_dir / "rebalance_history.csv")


# ---------------------------------------------------------------------------
# run_id en los logs
# ---------------------------------------------------------------------------
class TestRunLogging:
    def _capture(self):
        records = []
        handler = logging.Handler()
        handler.addFilter(RunIdFilter())
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        handler.emit = lambda r: records.append(handler.format(r))
        return handler, records

    def test_filter_tags_records_inside_and_outside_a_run(self):
        handler, records = self._capture()
        log = logging.getLogger("trace_test")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            log.info("fuera")
            with run_logging(RUN_ID):
                assert current_run_id() == RUN_ID
                log.info("dentro")
            assert current_run_id() is None
        finally:
            log.removeHandler(handler)
        assert "[run=-] — fuera" in records[0]
        assert f"[run={RUN_ID}] — dentro" in records[1]

    def test_log_to_file_is_scoped_and_restores_root(self, tmp_path):
        root = logging.getLogger()
        before_handlers, before_level = list(root.handlers), root.level
        with run_logging(RUN_ID), log_to_file(tmp_path / "logs") as path:
            logging.getLogger("trace_test").info("hola archivo")
        assert root.handlers == before_handlers and root.level == before_level
        text = path.read_text(encoding="utf-8")
        assert f"[run={RUN_ID}]" in text and "hola archivo" in text

    def test_log_to_file_rotates(self, tmp_path):
        with log_to_file(tmp_path, max_bytes=400, backup_count=2) as path:
            for i in range(50):
                logging.getLogger("trace_test").warning("linea %03d %s", i, "x" * 40)
        assert path.exists() and (tmp_path / "runner.log.1").exists()
        assert not (tmp_path / "runner.log.3").exists()

    def test_runner_log_lines_all_carry_the_run_id(self, home, state_dir):
        assert runner.run_once(force=True, start=START, run_id=RUN_ID) == 0
        lines = (state_dir / "logs" / "runner.log").read_text(encoding="utf-8").splitlines()
        stamped = [ln for ln in lines if re.match(r"\d{4}-\d\d-\d\d ", ln)]
        assert stamped and all(f"[run={RUN_ID}]" in ln for ln in stamped)
        assert any("Committed run" in ln for ln in stamped)

    def test_run_id_is_shared_by_every_state_file(self, home, state_dir):
        assert runner.run_once(force=True, start=START, run_id=RUN_ID) == 0
        assert _history(state_dir)["run_id"].tolist() == [RUN_ID]
        for name in ("transactions.csv", "snapshots.csv", "exit_log.csv"):
            path = state_dir / name
            if path.exists() and path.stat().st_size:
                assert set(pd.read_csv(path)["run_id"]) <= {RUN_ID}, name
        assert json.loads((state_dir / "state_manifest.json").read_text())["run_id"] == RUN_ID

    def test_default_run_id_format(self, home, state_dir):
        assert runner.run_once(force=True, start=START) == 0
        (run_id,) = _history(state_dir)["run_id"]
        assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run_id)

    def test_read_only_runs_write_no_log_file(self, home, state_dir):
        assert runner.run_once(check_only=True, start=START) == runner.EXIT_OK
        assert runner.run_once(force=True, start=START, dry_run=True) == runner.EXIT_OK
        assert not state_dir.exists()

    def test_crash_traceback_lands_in_runner_log(self, home, state_dir, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("estrategia rota")

        monkeypatch.setattr(runner, "build_targets", boom)
        with pytest.raises(RuntimeError, match="estrategia rota"):
            runner.run_once(force=True, start=START, run_id=RUN_ID)
        text = (state_dir / "logs" / "runner.log").read_text(encoding="utf-8")
        assert "Traceback" in text and "estrategia rota" in text


# ---------------------------------------------------------------------------
# Procedencia en history
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_config_schema_version_reexported_from_strategy(self):
        # Mismo valor; strategy lo reexporta para no romper imports viejos.
        assert SCHEMA_VERSION == CONFIG_SCHEMA_VERSION == 1

    def test_history_records_code_config_data_and_schemas(self, home, state_dir):
        assert runner.run_once(force=True, start=START) == 0
        row = _history(state_dir).iloc[0]
        assert re.fullmatch(r"[0-9a-f]{40}|unknown", str(row["git_sha"]))
        assert row["data_source"] == "fake"          # FakeProvider.name
        assert len(row["config_hash"]) == 64
        assert row["config_schema"] == CONFIG_SCHEMA_VERSION
        assert row["state_schema"] == STATE_SCHEMA_VERSION
        assert row["ledger_schema"] == LEDGER_SCHEMA_VERSION

    def test_dry_run_summary_carries_provenance(self, home):
        assert runner.run_once(force=True, start=START, dry_run=True) == runner.EXIT_OK
        (plan_dir,) = (home / "results" / "plans").iterdir()
        summary = json.loads((plan_dir / "summary.json").read_text())
        assert summary["provenance"]["data_source"] == "fake"
        assert "git_sha" in summary["provenance"]

    def test_code_version_reads_git(self):
        code = provenance.code_version()
        assert re.fullmatch(r"[0-9a-f]{40}", code.sha) or code.sha == "unknown"

    def test_code_version_falls_back_to_env_without_git(self, monkeypatch):
        def no_git(*_a, **_k):
            raise FileNotFoundError("git")

        monkeypatch.setattr(provenance.subprocess, "run", no_git)
        code = provenance.code_version(env={"HFT_GIT_SHA": "abc123"})
        assert (code.sha, code.dirty) == ("abc123", None)
        assert provenance.code_version(env={}).sha == "unknown"

    def test_dirty_flag_and_short_tag(self, monkeypatch):
        outputs = iter(["f" * 40, " M src/x.py"])
        monkeypatch.setattr(
            provenance.subprocess, "run",
            lambda *a, **k: SimpleNamespace(stdout=next(outputs)))
        code = provenance.code_version()
        assert code.dirty is True and code.short == "f" * 12 + "-dirty"

    def test_git_error_is_not_fatal(self, monkeypatch):
        def fail(*_a, **_k):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(provenance.subprocess, "run", fail)
        assert provenance.code_version(env={}).sha == "unknown"


# ---------------------------------------------------------------------------
# Snapshot de la cartera tras la corrida (broker: + fills)
# ---------------------------------------------------------------------------
def test_book_after_applies_broker_fills():
    plan = SimpleNamespace(
        sim=None,
        portfolio=SimpleNamespace(positions=pd.Series({"AAA": 10.0, "BBB": 5.0}),
                                  cash=1_000.0),
        prices_t=pd.Series({"AAA": 10.0, "BBB": 20.0, "CCC": 50.0}),
    )
    orders = pd.DataFrame({
        "ticker": ["AAA", "CCC", "BBB"], "side": ["sell", "buy", "buy"],
        "filled_qty": [10.0, 2.0, None], "filled_avg_price": [11.0, 49.0, None],
    })
    shares, cash = runner.book_after(plan, SimpleNamespace(orders=orders))
    assert shares.get("AAA", 0.0) == 0.0 and shares["BBB"] == 5.0 and shares["CCC"] == 2.0
    assert cash == pytest.approx(1_000.0 + 110.0 - 98.0)


def test_snapshot_frame_has_cash_row_and_weights(home, state_dir):
    assert runner.run_once(force=True, start=START) == 0
    snaps = pd.read_csv(state_dir / "snapshots.csv")
    assert list(snaps.columns[:5]) == ["ticker", "shares", "price", "value", "weight"]
    assert snaps["weight"].sum() == pytest.approx(1.0)
    cash = snaps[snaps["ticker"] == CASH_COL].iloc[0]
    assert cash["price"] == 1.0
    assert snaps["value"].sum() == pytest.approx(_history(state_dir)["nav"].iloc[0])


# ---------------------------------------------------------------------------
# Alertas del runner
# ---------------------------------------------------------------------------
class TestRunnerAlerts:
    def test_committed_run_alerts_info(self, home):
        rec = Recorder()
        assert runner.run_once(force=True, start=START, run_id=RUN_ID, notifier=rec) == 0
        (alert,) = rec.alerts
        assert (alert.level, alert.exit_code, alert.run_id) == ("info", 0, RUN_ID)
        assert alert.details["as_of"] == _fake_prices().index[-1].date().isoformat()

    def test_not_due_and_read_only_ok_do_not_alert(self, home):
        rec = Recorder()
        assert runner.run_once(start=START, notifier=rec) == 0
        rec.alerts.clear()
        assert runner.run_once(start=START, notifier=rec) == runner.EXIT_NOT_DUE
        assert runner.run_once(check_only=True, force=True, start=START,
                               notifier=rec) == runner.EXIT_OK
        assert rec.alerts == []

    def test_data_error_alerts_with_logged_reason(self, home, monkeypatch):
        use_market_data(monkeypatch, _fake_prices()[TICKERS[:-1]])
        rec = Recorder()
        assert runner.run_once(force=True, start=START, notifier=rec) == runner.EXIT_DATA
        (alert,) = rec.alerts
        assert alert.level == "error" and alert.exit_code == runner.EXIT_DATA
        assert TICKERS[-1] in alert.message

    def test_crash_alerts_critical_and_reraises(self, home, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("estrategia rota")

        monkeypatch.setattr(runner, "build_targets", boom)
        rec = Recorder()
        with pytest.raises(RuntimeError):
            runner.run_once(force=True, start=START, notifier=rec)
        (alert,) = rec.alerts
        assert alert.level == "critical" and alert.exit_code == runner.EXIT_UNEXPECTED
        assert "estrategia rota" in alert.message

    def test_lock_contention_alerts_warning(self, home, state_dir):
        rec = Recorder()
        with StateStore(state_dir):
            assert runner.run_once(force=True, start=START,
                                   notifier=rec) == runner.EXIT_LOCKED
        (alert,) = rec.alerts
        assert alert.level == "warning" and alert.exit_code == runner.EXIT_LOCKED

    def test_default_notifier_writes_alerts_jsonl(self, home, state_dir, monkeypatch):
        use_market_data(monkeypatch, _fake_prices()[TICKERS[:-1]])
        assert runner.run_once(force=True, start=START, run_id=RUN_ID) == runner.EXIT_DATA
        (line,) = (state_dir / "logs" / "alerts.jsonl").read_text().splitlines()
        alert = json.loads(line)
        assert (alert["run_id"], alert["exit_code"], alert["level"]) == (RUN_ID, 7, "error")

    def test_failing_notifier_never_breaks_the_run(self, home):
        class Broken:
            def notify(self, alert):
                raise OSError("smtp caído")

        assert runner.run_once(force=True, start=START, notifier=Broken()) == 0

    def test_main_returns_5_on_crash(self, home, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("estrategia rota")

        monkeypatch.setattr(runner, "build_targets", boom)
        assert runner.main(["--force", "--start", START]) == runner.EXIT_UNEXPECTED


# ---------------------------------------------------------------------------
# Notifiers
# ---------------------------------------------------------------------------
class FakeSMTP:
    def __init__(self):
        self.sent, self.logins, self.closed = [], [], False

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.closed = True


class TestNotifiers:
    def test_alert_rejects_unknown_level(self):
        with pytest.raises(ValueError, match="level"):
            Alert("fatal", "t", "m")

    def test_file_notifier_appends_json_lines(self, tmp_path):
        path = tmp_path / "logs" / "alerts.jsonl"
        n = FileNotifier(path)
        n.notify(Alert("error", "a", "uno", run_id="r1", exit_code=7))
        n.notify(Alert("info", "b", "dos", details={"as_of": "2024-01-02"}))
        rows = [json.loads(ln) for ln in path.read_text().splitlines()]
        assert [r["title"] for r in rows] == ["a", "b"]
        assert rows[0]["exit_code"] == 7 and rows[1]["details"] == {"as_of": "2024-01-02"}

    def test_console_notifier_logs_at_alert_level(self, caplog):
        with caplog.at_level(logging.INFO, logger="alerts"):
            ConsoleNotifier().notify(Alert("critical", "Caída", "boom", exit_code=5))
        (rec,) = [r for r in caplog.records if r.name == "alerts"]
        assert rec.levelno == logging.CRITICAL and "Caída" in rec.getMessage()

    def test_multi_notifier_isolates_failures(self, caplog):
        class Broken:
            def notify(self, alert):
                raise RuntimeError("no")

        rec = Recorder()
        with caplog.at_level(logging.WARNING, logger="alerts"):
            MultiNotifier([Broken(), rec]).notify(Alert("error", "t", "m"))
        assert len(rec.alerts) == 1
        assert any("Broken" in r.getMessage() for r in caplog.records)

    def test_email_notifier_filters_by_level_and_sends(self):
        smtp = FakeSMTP()
        cfg = EmailConfig(recipients=("pm@example.com", "ops@example.com"),
                          host="smtp.example.com", sender="hft@example.com",
                          user="hft", password="s3cret", min_level="error")
        n = EmailNotifier(cfg, smtp_factory=lambda _c: smtp)
        n.notify(Alert("warning", "ignorada", "m"))
        assert smtp.sent == []
        n.notify(Alert("error", "Datos", "falló", run_id=RUN_ID, exit_code=7))
        (msg,) = smtp.sent
        assert msg["Subject"] == "[HFT-V1 ERROR] Datos"
        assert msg["To"] == "pm@example.com, ops@example.com"
        assert RUN_ID in msg.get_content()
        assert smtp.logins == [("hft", "s3cret")] and smtp.closed
        assert "s3cret" not in repr(cfg)

    def test_email_config_disabled_without_recipient_or_host(self, caplog):
        assert load_email_config(env={}) is None
        with caplog.at_level(logging.WARNING, logger="alerts"):
            assert load_email_config(env={"ALERT_EMAIL": "pm@example.com"}) is None
        assert any("ALERT_SMTP_HOST" in r.getMessage() for r in caplog.records)

    def test_email_config_from_env(self):
        cfg = load_email_config(env={
            "ALERT_EMAIL": "a@example.com, b@example.com",
            "ALERT_SMTP_HOST": "smtp.example.com", "ALERT_SMTP_PORT": "465",
            "ALERT_SMTP_USER": "u", "ALERT_SMTP_PASSWORD": "p",
            "ALERT_EMAIL_FROM": "hft@example.com", "ALERT_EMAIL_MIN_LEVEL": "WARNING",
        })
        assert cfg.recipients == ("a@example.com", "b@example.com")
        assert (cfg.port, cfg.sender, cfg.min_level) == (465, "hft@example.com", "warning")

    def test_build_notifier_wiring(self, tmp_path, caplog):
        settings = load_settings(env={"HFT_HOME": str(tmp_path)})
        email_env = {"ALERT_EMAIL": "pm@example.com", "ALERT_SMTP_HOST": "smtp.x"}
        kinds = lambda n: [type(x).__name__ for x in n.notifiers]  # noqa: E731
        assert kinds(build_notifier(settings, writable=False, env=email_env)) == [
            "ConsoleNotifier"]
        assert kinds(build_notifier(settings, env={})) == [
            "ConsoleNotifier", "FileNotifier"]
        assert kinds(build_notifier(settings, env=email_env)) == [
            "ConsoleNotifier", "FileNotifier", "EmailNotifier"]
        with caplog.at_level(logging.WARNING, logger="alerts"):
            bad = build_notifier(settings, env={**email_env, "ALERT_SMTP_PORT": "x"})
        assert kinds(bad) == ["ConsoleNotifier", "FileNotifier"]
        file_n = build_notifier(settings, env={}).notifiers[1]
        assert file_n.path == settings.state_dir / "logs" / "alerts.jsonl"
