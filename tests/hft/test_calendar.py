"""Calendario de sesión, DST, finalidad de barra y cache (HFT-03)."""

from __future__ import annotations

import ast
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.hft import calendar as cal

UTC = timezone.utc
NY = cal.NY


def _row(day: str, open_hhmm: str = "09:30", close_hhmm: str = "16:00") -> dict:
    return {"date": day, "open": open_hhmm, "close": close_hhmm,
            "session_open": "0400", "settlement_date": day}


def _half_day() -> cal.Session:
    """2024-07-03, cierre temprano 13:00 ET (día previo al 4 de julio)."""
    return cal.sessions_from_alpaca([_row("2024-07-03", close_hhmm="13:00")])[0]


class _Rows:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def get_calendar(self, start, end):
        self.calls += 1
        return list(self.rows)


class _Boom:
    def get_calendar(self, start, end):
        raise OSError("red caida")


def _et(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)


class TestAlpacaHours:
    def test_media_sesion_flatten_cinco_minutos_antes(self):
        session = _half_day()
        assert session.close_et.hour == 13 and session.close_et.minute == 0
        flat = cal.flatten_time(session, 5)
        assert flat.astimezone(NY) == _et(date(2024, 7, 3), 12, 55)
        assert flat == datetime(2024, 7, 3, 16, 55, tzinfo=UTC)
        assert cal.entry_cutoff(session, 15).astimezone(NY) == _et(date(2024, 7, 3), 12, 45)

    def test_cambio_de_horario_marzo_y_noviembre(self):
        # 2024: el reloj adelanta el 10-mar y atrasa el 3-nov. La rueda
        # abre a las 09:30 ET en ambos lados: 14:30 UTC en invierno, 13:30 en verano.
        sessions = cal.sessions_from_alpaca([
            _row("2024-03-08"),
            _row("2024-03-11"),
            _row("2024-11-01"),
            _row("2024-11-04"),
        ])
        by_day = {session.date: session for session in sessions}
        assert by_day[date(2024, 3, 8)].open_utc == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
        assert by_day[date(2024, 3, 8)].close_utc == datetime(2024, 3, 8, 21, 0, tzinfo=UTC)
        assert by_day[date(2024, 3, 11)].open_utc == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
        assert by_day[date(2024, 3, 11)].close_utc == datetime(2024, 3, 11, 20, 0, tzinfo=UTC)
        assert by_day[date(2024, 11, 1)].open_utc == datetime(2024, 11, 1, 13, 30, tzinfo=UTC)
        assert by_day[date(2024, 11, 4)].open_utc == datetime(2024, 11, 4, 14, 30, tzinfo=UTC)

    def test_ignora_campos_extra_y_rechaza_fila_rota(self):
        session = cal.sessions_from_alpaca([_row("2024-07-03", close_hhmm="13:00")])[0]
        assert session.open_et.strftime("%H:%M") == "09:30"
        with pytest.raises(ValueError, match="close"):
            cal.sessions_from_alpaca([{"date": "2024-07-03", "open": "09:30"}])
        with pytest.raises(ValueError, match="HH:MM"):
            cal.sessions_from_alpaca([_row("2024-07-03", open_hhmm="9:30")])


class TestSessionClock:
    def test_bordes_de_sesion_con_reloj_inyectado(self):
        session = _half_day()
        book = (session,)
        assert cal.in_session(book, session.open_utc) is True
        assert cal.in_session(book, session.close_utc) is False
        assert cal.in_session(book, session.close_utc - timedelta(microseconds=1)) is True
        pre = _et(date(2024, 7, 3), 9, 0)
        assert cal.session_for(book, pre) == session
        assert cal.in_session(book, pre) is False
        assert cal.minutes_to_close(book, pre) == pytest.approx(240.0)
        assert cal.minutes_to_close(book, session.close_utc) == pytest.approx(0.0)
        assert cal.minutes_to_close(book, _et(date(2024, 7, 3), 13, 1)) == pytest.approx(-1.0)
        saturday = datetime(2024, 7, 6, 15, 0, tzinfo=UTC)
        assert cal.session_for(book, saturday) is None
        assert cal.in_session(book, saturday) is False
        assert cal.minutes_to_close(book, saturday) is None
        with pytest.raises(ValueError, match="zona"):
            cal.in_session(book, datetime(2024, 7, 3, 14, 0))

    def test_fake_calendar_no_toca_disco(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        fake = cal.FakeCalendar([_half_day()])
        loaded = fake.load(date(2024, 7, 3), date(2024, 7, 3))
        assert loaded == (_half_day(),)
        assert fake.in_session(_half_day().open_utc) is True
        assert not (tmp_path / "data").exists()

    def test_barras_esperadas_en_media_sesion(self):
        session = _half_day()
        bars = cal.expected_bars(session, "1min")
        assert len(bars) == 210
        assert bars[0] == pd.Timestamp(session.open_utc)
        assert bars[-1] == pd.Timestamp(session.close_utc - timedelta(minutes=1))
        assert str(bars.tz) == "UTC"
        five = cal.expected_bars(session, timedelta(minutes=5))
        assert len(five) == 42
        assert five[-1] == pd.Timestamp(_et(date(2024, 7, 3), 12, 55))
        full = cal.sessions_from_alpaca([_row("2024-03-11")])[0]
        assert len(cal.expected_bars(full, "1min")) == 390


class TestBarFinal:
    def test_final_en_el_borde_de_la_gracia(self):
        start = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
        step = timedelta(minutes=1)
        edge = start + step + timedelta(seconds=cal.DEFAULT_GRACE_S)
        assert cal.bar_close_time(start, step) == start + step
        assert cal.DEFAULT_GRACE_S == 5
        assert cal.is_bar_final(start, edge, "1min") is True
        assert cal.is_bar_final(start, edge - timedelta(microseconds=1), "1min") is False
        assert cal.is_bar_final(start, edge, "1min", grace_s=5) is True
        assert cal.is_bar_final(start, start + step, "1min", grace_s=0) is True
        with pytest.raises(ValueError, match="grace_s"):
            cal.is_bar_final(start, edge, "1min", grace_s=-1)

    def test_solo_systemclock_lee_el_reloj(self):
        tree = ast.parse(Path(cal.__file__).read_text(encoding="utf-8"))
        hits = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"now", "time"}:
                    hits.append((tuple(self.stack), func.attr))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert hits == [(("SystemClock", "now"), "now")]

    def test_systemclock_devuelve_utc(self, monkeypatch):
        fixed = datetime(2024, 3, 11, 14, 30, tzinfo=UTC)

        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                assert tz == UTC
                return fixed

        monkeypatch.setattr(cal, "datetime", Frozen)
        assert cal.SystemClock().now() == fixed


class TestSources:
    def test_cache_roundtrip_atomico(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        replaced = []
        real_replace = os.replace

        def _spy(src, dst):
            replaced.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(cal.os, "replace", _spy)
        client = _Rows([
            _row("2024-03-08"),
            _row("2024-03-11"),
        ])
        loaded = cal.SessionCalendar(client=client).load(date(2024, 3, 8), date(2024, 3, 11))
        cache = tmp_path / "data" / "hft" / "calendar.csv"
        assert cache.is_file()
        assert not cache.with_name("calendar.csv.tmp").exists()
        assert replaced and replaced[-1][1] == cache
        assert client.calls == 1

        again = cal.SessionCalendar(client=None).load(date(2024, 3, 8), date(2024, 3, 11))
        assert again == loaded
        assert [session.open_utc.hour for session in again] == [14, 13]

    def test_fin_de_semana_cacheado_no_inventa_sesion(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        client = _Rows([])
        empty = cal.SessionCalendar(client=client).load(date(2024, 3, 9), date(2024, 3, 10))
        assert empty == ()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hft.calendar"):
            again = cal.SessionCalendar(client=None).load(date(2024, 3, 9), date(2024, 3, 10))
        assert again == ()
        assert caplog.records == []

    def test_alpaca_caido_usa_cache_sin_respaldo(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        rows = [_row("2024-07-03", close_hhmm="13:00")]
        first = cal.SessionCalendar(client=_Rows(rows)).load(date(2024, 7, 3), date(2024, 7, 3))
        with caplog.at_level(logging.WARNING, logger="hft.calendar"):
            second = cal.SessionCalendar(client=_Boom()).load(date(2024, 7, 3), date(2024, 7, 3))
        assert second == first
        assert second[0].close_et.hour == 13
        assert not any("medias sesiones" in record.message for record in caplog.records)
        assert any("calendar" in record.message for record in caplog.records)

    def test_fallback_avisa_y_no_conoce_la_media_sesion(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="hft.calendar"):
            sessions = cal.SessionCalendar(client=None).load(date(2024, 11, 27), date(2024, 11, 29))
        assert any(record.message == cal.FALLBACK_WARNING for record in caplog.records)
        assert "medias sesiones" in cal.FALLBACK_WARNING
        by_day = {session.date: session for session in sessions}
        assert date(2024, 11, 28) not in by_day  # Thanksgiving
        friday = by_day[date(2024, 11, 29)]
        assert friday.open_et.strftime("%H:%M") == "09:30"
        assert friday.close_et.strftime("%H:%M") == "16:00"
        assert cal.flatten_time(friday, 5).astimezone(NY).strftime("%H:%M") == "15:55"
        assert not (tmp_path / "data" / "hft" / "calendar.csv").exists()

    def test_fallback_excluye_feriados_nyse(self):
        sessions = cal.fallback_sessions(date(2024, 1, 1), date(2024, 1, 2))
        assert [session.date for session in sessions] == [date(2024, 1, 2)]
        good_friday = cal.fallback_sessions(date(2024, 3, 29), date(2024, 3, 29))
        assert good_friday == ()

    def test_cobertura_incompleta_no_disfraza_el_hueco(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))
        cal.SessionCalendar(client=_Rows([_row("2024-03-11")])).load(
            date(2024, 3, 11), date(2024, 3, 11))
        with caplog.at_level(logging.WARNING, logger="hft.calendar"):
            sessions = cal.SessionCalendar(client=None).load(date(2024, 3, 8), date(2024, 3, 11))
        assert any("medias sesiones" in record.message for record in caplog.records)
        assert date(2024, 3, 8) in {session.date for session in sessions}
        assert all(session.close_et.strftime("%H:%M") == "16:00" for session in sessions)
