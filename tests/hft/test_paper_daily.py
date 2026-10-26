"""Supervisor diario de paper: relanza por red, para por riesgo, lock o cierre."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import scripts.hft_paper_daily as daily
from src.hft.session import (
    EXIT_BROKER,
    EXIT_DATA,
    EXIT_LOCK,
    EXIT_NO_SESSION,
    EXIT_OK,
    EXIT_RISK,
    EXIT_UNEXPECTED,
)

UTC = timezone.utc
T0 = datetime(2026, 10, 8, 12, 15, tzinfo=UTC)  # 08:15 en Nueva York
CLOSE = datetime(2026, 10, 8, 20, 0, tzinfo=UTC)  # 16:00 en Nueva York


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.t = start
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += timedelta(seconds=seconds)


def _runner(codes):
    calls = []

    def run():
        calls.append(len(calls))
        return codes[len(calls) - 1]

    return run, calls


def _supervise(run, clock, **kwargs):
    return daily.supervise(run, clock.now, clock.sleep, stop_at=CLOSE, **kwargs)


class TestSupervise:
    def test_sesion_completa_corre_una_vez(self):
        clock = FakeClock(T0)
        run, calls = _runner([EXIT_OK])
        assert _supervise(run, clock) == EXIT_OK
        assert len(calls) == 1
        assert clock.slept == []

    @pytest.mark.parametrize("code", [EXIT_BROKER, EXIT_DATA, EXIT_UNEXPECTED])
    def test_fallo_de_red_datos_o_inesperado_relanza(self, code):
        clock = FakeClock(T0)
        run, calls = _runner([code, EXIT_OK])
        assert _supervise(run, clock, retry_s=60) == EXIT_OK
        assert len(calls) == 2
        assert clock.slept == [60]

    @pytest.mark.parametrize("code", [EXIT_RISK, EXIT_LOCK, EXIT_NO_SESSION, 2])
    def test_riesgo_lock_feriado_o_guarda_no_relanza(self, code):
        clock = FakeClock(T0)
        run, calls = _runner([code, EXIT_OK])
        assert _supervise(run, clock) == code
        assert len(calls) == 1

    def test_senal_del_sistema_no_relanza(self):
        clock = FakeClock(T0)
        run, calls = _runner([-15, EXIT_OK])
        assert _supervise(run, clock) == -15
        assert len(calls) == 1

    def test_no_relanza_despues_del_cierre(self):
        clock = FakeClock(CLOSE - timedelta(seconds=30))
        run, calls = _runner([EXIT_BROKER, EXIT_OK])
        assert _supervise(run, clock, retry_s=60) == EXIT_BROKER
        assert len(calls) == 1

    def test_tope_de_intentos(self):
        clock = FakeClock(T0)
        run, calls = _runner([EXIT_BROKER] * 10)
        assert _supervise(run, clock, retry_s=60, max_runs=3) == EXIT_BROKER
        assert len(calls) == 3


class TestCommand:
    def test_paper_con_submit_y_fecha(self):
        cmd = daily.session_command("/repo/.venv/bin/python", date(2026, 10, 9))
        assert cmd == [
            "/repo/.venv/bin/python", "-m", "scripts.run_intraday",
            "--mode", "paper", "--submit", "--date", "2026-10-09",
        ]

    def test_dry_run_no_envia(self):
        cmd = daily.session_command("py", date(2026, 10, 9), submit=False)
        assert "--submit" not in cmd
        assert cmd[-2:] == ["--date", "2026-10-09"]

    def test_with_se_reenvia_al_runner(self):
        cmd = daily.session_command(
            "py", date(2026, 10, 9), with_strategies=("hourly_momentum",),
        )
        assert cmd == [
            "py", "-m", "scripts.run_intraday", "--mode", "paper", "--submit",
            "--with", "hourly_momentum", "--date", "2026-10-09",
        ]

    def test_with_en_dry_run_tampoco_envia(self):
        cmd = daily.session_command(
            "py", date(2026, 10, 9), submit=False, with_strategies=("hourly_momentum",),
        )
        assert "--submit" not in cmd
        assert cmd[-4:] == ["--with", "hourly_momentum", "--date", "2026-10-09"]

    def test_fecha_es_la_de_nueva_york(self):
        # 23:30 en Nueva York ya es el día siguiente en UTC.
        late = datetime(2026, 10, 9, 3, 30, tzinfo=UTC)
        assert daily.ny_date(late) == date(2026, 10, 8)

    def test_cierre_en_utc(self):
        assert daily.close_utc(date(2026, 10, 8)) == CLOSE
        assert daily.close_utc(date(2026, 12, 8)) == datetime(2026, 12, 8, 21, 0, tzinfo=UTC)


class TestPlist:
    def _plist(self, tmp_path):
        return daily.render_plist(
            tmp_path, tmp_path / ".venv" / "bin" / "python",
            hour=8, minute=15, log_path=tmp_path / "paper.log",
        )

    def test_lunes_a_viernes_a_la_hora(self, tmp_path):
        plist = self._plist(tmp_path)
        assert plist["Label"] == daily.LABEL
        assert plist["StartCalendarInterval"] == [
            {"Weekday": day, "Hour": 8, "Minute": 15} for day in range(1, 6)
        ]
        assert plist["RunAtLoad"] is False

    def test_corre_run_desde_el_repo(self, tmp_path):
        plist = self._plist(tmp_path)
        assert plist["ProgramArguments"] == [
            str(tmp_path / ".venv" / "bin" / "python"), "-m", "scripts.hft_paper_daily", "run",
        ]
        assert plist["WorkingDirectory"] == str(tmp_path)
        assert plist["StandardOutPath"] == str(tmp_path / "paper.log")
        assert plist["StandardErrorPath"] == str(tmp_path / "paper.log")

    def test_sin_secretos_en_el_plist(self, tmp_path):
        env = self._plist(tmp_path).get("EnvironmentVariables", {})
        assert not any(key.startswith(("BROKER_", "APCA_", "ALPACA_")) for key in env)
        assert "/usr/bin" in env["PATH"].split(":")

    def test_parse_at(self):
        assert daily.parse_at("08:15") == (8, 15)
        with pytest.raises(ValueError):
            daily.parse_at("25:00")

    def test_with_solo_agrega_la_bandera_a_run(self, tmp_path):
        base = self._plist(tmp_path)
        plist = daily.render_plist(
            tmp_path, tmp_path / ".venv" / "bin" / "python",
            hour=8, minute=15, log_path=tmp_path / "paper.log",
            with_strategies=("hourly_momentum",),
        )
        assert plist["ProgramArguments"] == [
            *base["ProgramArguments"], "--with", "hourly_momentum",
        ]
        assert {k: v for k, v in plist.items() if k != "ProgramArguments"} == {
            k: v for k, v in base.items() if k != "ProgramArguments"
        }


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)


class TestMain:
    def _calls(self, monkeypatch, code=0):
        calls = []

        def _call(cmd, **kwargs):
            calls.append(list(cmd))
            return code

        # Nada llega a launchctl ni a run_intraday de verdad.
        monkeypatch.setattr(daily.subprocess, "call", _call)
        return calls

    def test_run_reenvia_with(self, monkeypatch, quiet):
        calls = self._calls(monkeypatch)
        monkeypatch.setattr(daily, "_keep_awake", lambda: None)
        monkeypatch.setattr(daily, "supervise", lambda run, *args, **kwargs: run())
        assert daily.main(["run", "--with", "hourly_momentum"]) == 0
        assert calls[-1][-4:-2] == ["--with", "hourly_momentum"]
        assert "--submit" in calls[-1]
        assert daily.main(["run"]) == 0
        assert "--with" not in calls[-1]

    def test_with_otra_estrategia_es_error(self, quiet):
        with pytest.raises(SystemExit) as exc:
            daily.main(["run", "--with", "avellaneda_stoikov"])
        assert exc.value.code == 2

    def _install(self, tmp_path, monkeypatch, argv):
        import plistlib

        repo = tmp_path / "repo"
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
        target = tmp_path / "agents" / f"{daily.LABEL}.plist"
        monkeypatch.setattr(daily, "REPO_ROOT", repo)
        monkeypatch.setattr(daily, "_plist_path", lambda: target)
        monkeypatch.setattr(daily, "_log_path", lambda: tmp_path / "logs" / "paper.log")
        calls = self._calls(monkeypatch)
        assert daily.main(["install", *argv]) == 0
        assert any(cmd[:2] == ["launchctl", "bootstrap"] for cmd in calls)
        return plistlib.loads(target.read_bytes())

    def test_install_sin_with_deja_el_plist_de_hoy(self, tmp_path, monkeypatch, quiet):
        plist = self._install(tmp_path, monkeypatch, [])
        assert plist["ProgramArguments"][-3:] == ["-m", "scripts.hft_paper_daily", "run"]

    def test_install_con_with_lo_reenvia_a_run(self, tmp_path, monkeypatch, quiet):
        plist = self._install(tmp_path, monkeypatch, ["--with", "hourly_momentum"])
        assert plist["ProgramArguments"][-3:] == ["run", "--with", "hourly_momentum"]
