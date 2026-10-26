"""Calendario de sesión y finalidad de barra (HFT-03).

Sesiones en UTC (tz-aware). La hora de pantalla es America/New_York
(``zoneinfo``). Ninguna función de este módulo lee el reloj del sistema:
el único punto es ``SystemClock.now``.

Fuentes, en orden:

1. Alpaca ``GET /v2/calendar`` (Trading API). Campos verificados: ``date``
   (YYYY-MM-DD), ``open`` y ``close`` (HH:MM). ``start`` y ``end`` son
   inclusivos; ``date_type=TRADING``. El esquema no declara zona horaria;
   Alpaca confirmó que ``open``/``close`` son hora del Este
   (https://forum.alpaca.markets/t/calendar-endpoint/11190). Las medias
   sesiones llegan con ``close`` 13:00. ``session_open``, ``session_close``
   y ``settlement_date`` se ignoran.
2. Cache en ``<data_dir>/hft/calendar.csv`` (escritura atómica). Cubre un
   rango solo si una descarga previa lo incluyó entero; un día ausente
   dentro de esa cobertura es un día sin sesión, no un hueco.
3. ``src.market_data.NYSEHolidayCalendar`` con 09:30–16:00 ET y un WARNING.
   No modela medias sesiones ni cierres extraordinarios. No se escribe al
   cache: no debe disfrazarse de calendario de Alpaca.

Barras. Alpaca etiqueta ``t`` con el inicio del intervalo y agrega
operaciones en ``[t, t + timeframe)``
(https://alpaca.markets/learn/stock-minute-bars). Por tanto
``bar_close_time(t) = t + timeframe`` y la barra es final cuando
``now >= t + timeframe + grace_s`` (``grace_s`` por defecto 5). Alpaca
recalcula la barra cerca de 30 s después del cierre porque un trade puede
publicarse tarde. En vivo, una corrección que llega después de
``is_bar_final`` se ignora. El backtest usa el histórico ya cerrado. Esa
diferencia se reporta en HFT-13; aquí no se reconcilia.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("hft.calendar")

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# Gracia tras el cierre de la barra antes de darla por final (segundos).
DEFAULT_GRACE_S = 5.0
# Días de calendario cargados antes de la primera sesión. Cubre las 21
# sesiones previas que piden las estrategias (σ de 20 sesiones + 1) con
# margen para feriados. Backtest y vivo cargan la misma ventana.
HISTORY_LOOKBACK_DAYS = 120

FALLBACK_WARNING = (
    "Calendario de respaldo NYSEHolidayCalendar (09:30-16:00 ET). "
    "No modela medias sesiones ni cierres extraordinarios."
)

_OPEN_ET = time(9, 30)
_CLOSE_ET = time(16, 0)
_CACHE_NAME = "calendar.csv"
_COLUMNS = ("date", "open_utc", "close_utc")

DateLike = Union[date, datetime, str, pd.Timestamp]
TimeframeLike = Union[timedelta, pd.Timedelta, str]


class CalendarClient(Protocol):
    """Lo mínimo que ``SessionCalendar`` pide al broker. Solo lectura."""

    def get_calendar(self, start, end):
        """Lista de objetos con date/open/close, o lanza si la red falla."""


class SystemClock:
    """Reloj de pared UTC. Única llamada a ``datetime.now`` de este módulo."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True)
class Session:
    """Una sesión: ``date`` es el día en Nueva York; los instantes van en UTC."""

    date: date
    open_utc: datetime
    close_utc: datetime

    def __post_init__(self) -> None:
        if self.open_utc.tzinfo is None or self.close_utc.tzinfo is None:
            raise ValueError("open_utc y close_utc deben llevar zona horaria")
        open_utc = self.open_utc.astimezone(UTC)
        close_utc = self.close_utc.astimezone(UTC)
        if close_utc <= open_utc:
            raise ValueError("close_utc debe ser posterior a open_utc")
        if open_utc.astimezone(NY).date() != self.date:
            raise ValueError("date debe ser el día de apertura en America/New_York")
        if close_utc.astimezone(NY).date() != self.date:
            raise ValueError("el cierre debe caer en el mismo día de Nueva York")
        object.__setattr__(self, "open_utc", open_utc)
        object.__setattr__(self, "close_utc", close_utc)

    @property
    def open_et(self) -> datetime:
        """Apertura en America/New_York (solo display)."""
        return to_new_york(self.open_utc)

    @property
    def close_et(self) -> datetime:
        """Cierre en America/New_York (solo display)."""
        return to_new_york(self.close_utc)


def to_new_york(ts: datetime) -> datetime:
    """Convierte un instante a America/New_York. La zona interna es UTC."""
    return _as_utc(ts).astimezone(NY)


def session_at(day: date, open_t: time, close_t: time) -> Session:
    """Arma una sesión desde hora de pared en Nueva York (incluye el cambio DST)."""
    if close_t <= open_t:
        raise ValueError("close debe ser posterior a open el mismo día")
    open_local = datetime.combine(day, open_t, tzinfo=NY)
    close_local = datetime.combine(day, close_t, tzinfo=NY)
    return Session(date=day, open_utc=open_local, close_utc=close_local)


def sessions_from_alpaca(rows: Sequence[Mapping]) -> Tuple[Session, ...]:
    """Filas de ``GET /v2/calendar`` → sesiones UTC.

    Exige ``date``, ``open`` y ``close``. El resto de campos se ignora.
    """
    if isinstance(rows, Mapping):
        raise ValueError("el calendario de Alpaca debe ser una lista de sesiones")
    sessions = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("cada fila del calendario debe ser un objeto")
        try:
            day = date.fromisoformat(str(row["date"]))
            open_t = _parse_hhmm(str(row["open"]))
            close_t = _parse_hhmm(str(row["close"]))
        except KeyError as exc:
            raise ValueError(
                f"fila de calendario sin campo {exc.args[0]}"
            ) from None
        session = session_at(day, open_t, close_t)
        if session.date in seen:
            raise ValueError(f"fecha duplicada en el calendario: {session.date.isoformat()}")
        seen.add(session.date)
        sessions.append(session)
    sessions.sort(key=lambda item: item.date)
    return tuple(sessions)


def in_session(sessions: Sequence[Session], now: datetime) -> bool:
    """True si ``now`` cae en ``[open, close)`` de la sesión de ese día ET."""
    session = session_for(sessions, now)
    if session is None:
        return False
    instant = _as_utc(now)
    return session.open_utc <= instant < session.close_utc


def session_for(sessions: Sequence[Session], now: datetime) -> Optional[Session]:
    """Sesión del día ET de ``now``, o None si ese día no está cargado.

    Devuelve la sesión también antes de la apertura y después del cierre.
    No lee el reloj: ``now`` lo inyecta el llamador.
    """
    instant = _as_utc(now)
    day = instant.astimezone(NY).date()
    for session in sessions:
        if session.date == day:
            return session
    return None


def minutes_to_close(sessions: Sequence[Session], now: datetime) -> Optional[float]:
    """Minutos hasta el cierre (negativo si ``now`` ya pasó). None sin sesión."""
    session = session_for(sessions, now)
    if session is None:
        return None
    delta = session.close_utc - _as_utc(now)
    return delta.total_seconds() / 60.0


def entry_cutoff(session: Session, minutes_before_close: float) -> datetime:
    """Último instante UTC en el que todavía se aceptan entradas."""
    return _before_close(session, minutes_before_close)


def flatten_time(session: Session, minutes_before_close: float) -> datetime:
    """Instante UTC en el que el libro debe quedar plano."""
    return _before_close(session, minutes_before_close)


def expected_bars(session: Session, timeframe: TimeframeLike) -> pd.DatetimeIndex:
    """Inicios de barra en ``[open, close)``, UTC. Sirve para ver huecos.

    Una barra cuyo inicio cae antes del cierre entra aunque su fin lo pase.
    """
    step = _as_step(timeframe)
    cursor = _as_utc(session.open_utc)
    end = _as_utc(session.close_utc)
    stamps = []
    while cursor < end:
        stamps.append(cursor)
        cursor = cursor + step
    index = pd.DatetimeIndex(stamps)
    if index.tz is None:
        index = index.tz_localize(UTC)
    else:
        index = index.tz_convert(UTC)
    index.name = "ts"
    return index


def bar_close_time(t: datetime, timeframe: TimeframeLike) -> datetime:
    """Fin de la barra que empieza en ``t``. Alpaca marca ``t`` como inicio."""
    return _as_utc(t) + _as_step(timeframe)


def is_bar_final(
    t: datetime,
    now: datetime,
    timeframe: TimeframeLike,
    grace_s: float = DEFAULT_GRACE_S,
) -> bool:
    """True si ``now >= t + timeframe + grace_s``.

    ``grace_s`` es configurable (default 5). En vivo, una corrección de
    Alpaca posterior a este instante se ignora: la barra ya se usó para
    decidir. El backtest lee el histórico final; la diferencia entre ambos
    queda para el reporte de HFT-13. Alpaca puede republicar la barra
    cerca de 30 s después del minuto porque hay trades que llegan tarde;
    con la gracia por defecto esa republicación queda fuera.
    """
    if grace_s < 0:
        raise ValueError("grace_s no puede ser negativo")
    ready = bar_close_time(t, timeframe) + timedelta(seconds=float(grace_s))
    return _as_utc(now) >= ready


class SessionCalendar:
    """Sesiones cargadas de Alpaca, del cache o del calendario NYSE de respaldo."""

    def __init__(
        self,
        client: Optional[CalendarClient] = None,
        cache_path: Optional[Path] = None,
        sessions: Optional[Sequence[Session]] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            client: objeto con ``get_calendar(start, end)``. None salta Alpaca.
            cache_path: CSV de sesiones. None → ``<data_dir>/hft/calendar.csv``
                resuelto al leer (respeta ``HFT_HOME`` en ese momento).
            sessions: sesiones ya en memoria. Si se pasan, el calendario
                arranca listo y ``load`` las reemplaza.
            use_cache: False no lee ni escribe disco (tests en memoria).
        """
        self._client = client
        self._cache_path = None if cache_path is None else Path(cache_path)
        self._use_cache = use_cache
        self._sessions: Tuple[Session, ...] = tuple(sessions or ())
        self._ready = sessions is not None

    @property
    def sessions(self) -> Tuple[Session, ...]:
        self._require()
        return self._sessions

    def load(self, start: DateLike, end: DateLike) -> Tuple[Session, ...]:
        """Carga ``[start, end]`` (fechas de trading, inclusivas) y lo recuerda."""
        start_d = _as_date(start)
        end_d = _as_date(end)
        if end_d < start_d:
            raise ValueError(f"start {start_d.isoformat()} es posterior a end {end_d.isoformat()}")
        fetched = self._try_alpaca(start_d, end_d)
        if fetched is not None:
            self._remember(fetched)
            return fetched
        if self._use_cache:
            cached = self._read_slice(start_d, end_d)
            if cached is not None:
                self._remember(cached)
                return cached
        logger.warning(FALLBACK_WARNING)
        fallback = fallback_sessions(start_d, end_d)
        self._remember(fallback)
        return fallback

    def in_session(self, now: datetime) -> bool:
        return in_session(self.sessions, now)

    def session_for(self, now: datetime) -> Optional[Session]:
        return session_for(self.sessions, now)

    def minutes_to_close(self, now: datetime) -> Optional[float]:
        return minutes_to_close(self.sessions, now)

    def _remember(self, sessions: Sequence[Session]) -> None:
        self._sessions = tuple(sessions)
        self._ready = True

    def _require(self) -> None:
        if not self._ready:
            raise RuntimeError("cargar sesiones con load() antes de consultar el calendario")

    def _try_alpaca(self, start: date, end: date) -> Optional[Tuple[Session, ...]]:
        if self._client is None:
            return None
        try:
            raw = self._client.get_calendar(start, end)
        except Exception as exc:
            logger.warning(
                "GET /v2/calendar falló (%s: %s); se intenta el cache y luego el respaldo NYSE.",
                type(exc).__name__,
                exc,
            )
            return None
        parsed = sessions_from_alpaca(raw)
        if self._use_cache:
            self._upsert(start, end, parsed)
        return parsed

    def _path(self) -> Path:
        if self._cache_path is not None:
            return self._cache_path
        from src.settings import load_settings
        return load_settings().data_dir / "hft" / _CACHE_NAME

    def _read_slice(self, start: date, end: date) -> Optional[Tuple[Session, ...]]:
        cache = _read_cache(self._path())
        if not any(lo <= start and end <= hi for lo, hi in cache.ranges):
            return None
        return tuple(
            session for session in cache.sessions if start <= session.date <= end
        )

    def _upsert(self, start: date, end: date, fresh: Sequence[Session]) -> None:
        path = self._path()
        current = _read_cache(path)
        by_date = {session.date: session for session in current.sessions}
        fresh_dates = {session.date for session in fresh}
        for day in list(by_date):
            if start <= day <= end and day not in fresh_dates:
                del by_date[day]
        for session in fresh:
            by_date[session.date] = session
        ranges = _merge_ranges((*current.ranges, (start, end)))
        _write_cache(path, _Cache(ranges, tuple(by_date[day] for day in sorted(by_date))))


class FakeCalendar(SessionCalendar):
    """Calendario en memoria. No llama a Alpaca ni toca el disco."""

    def __init__(self, sessions: Sequence[Session]):
        super().__init__(client=None, cache_path=None, sessions=sessions, use_cache=False)
        self._fixed = tuple(sessions)

    def load(self, start: DateLike, end: DateLike) -> Tuple[Session, ...]:
        start_d = _as_date(start)
        end_d = _as_date(end)
        if end_d < start_d:
            raise ValueError(f"start {start_d.isoformat()} es posterior a end {end_d.isoformat()}")
        chosen = tuple(
            session for session in self._fixed if start_d <= session.date <= end_d
        )
        self._remember(chosen)
        return chosen


@dataclass(frozen=True)
class _Cache:
    ranges: Tuple[Tuple[date, date], ...]
    sessions: Tuple[Session, ...]


def fallback_sessions(start: date, end: date) -> Tuple[Session, ...]:
    """Días hábiles NYSE con cierre fijo 16:00 ET. Sin medias sesiones.

    Import diferido: un ``import`` estático de ``src.market_data`` metería
    en la clausura del guardarraíl la dependencia diferida de ese módulo.
    Cargar el módulo no activa esa dependencia.
    """
    import importlib

    market_data = importlib.import_module("src.market_data")
    holidays = {
        pd.Timestamp(stamp).date()
        for stamp in market_data.NYSEHolidayCalendar().holidays(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )
    }
    sessions = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            sessions.append(session_at(day, _OPEN_ET, _CLOSE_ET))
        day += timedelta(days=1)
    return tuple(sessions)


def _before_close(session: Session, minutes_before_close: float) -> datetime:
    minutes = float(minutes_before_close)
    if minutes < 0:
        raise ValueError("minutes_before_close no puede ser negativo")
    return session.close_utc - timedelta(minutes=minutes)


def _as_utc(ts: datetime) -> datetime:
    if not isinstance(ts, datetime) or ts.tzinfo is None:
        raise ValueError("se requiere un datetime con zona horaria")
    return ts.astimezone(UTC)


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(NY).date()
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        stamp = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fecha no reconocida: {value!r}") from exc
    if stamp.tzinfo is not None:
        return stamp.tz_convert(NY).date()
    return stamp.date()


_STEP = re.compile(
    r"^(\d+)\s*(min|mins|minute|minutes|m|h|hr|hour|hours|s|sec|secs|second|seconds)$",
    re.IGNORECASE,
)
_STEP_MINUTES = frozenset({"min", "mins", "minute", "minutes", "m"})
_STEP_HOURS = frozenset({"h", "hr", "hour", "hours"})


def _as_step(timeframe: TimeframeLike) -> timedelta:
    if isinstance(timeframe, timedelta):
        # pd.Timedelta también es timedelta; total_seconds evita el constructor
        # que NumPy marca como obsoleto.
        step = timedelta(seconds=float(timeframe.total_seconds()))
    elif isinstance(timeframe, str):
        match = _STEP.fullmatch(timeframe.strip())
        if match is None:
            raise ValueError(f"timeframe no reconocido: {timeframe!r}")
        count = int(match.group(1))
        unit = match.group(2).lower()
        if unit in _STEP_MINUTES:
            step = timedelta(minutes=count)
        elif unit in _STEP_HOURS:
            step = timedelta(hours=count)
        else:
            step = timedelta(seconds=count)
    else:
        raise TypeError(
            f"timeframe debe ser timedelta o texto, no {type(timeframe).__name__}"
        )
    if step <= timedelta(0):
        raise ValueError("timeframe debe ser positivo")
    return step


def _parse_hhmm(text: str) -> time:
    parts = text.strip().split(":")
    if len(parts) != 2 or not all(len(part) == 2 and part.isdigit() for part in parts):
        raise ValueError(f"hora {text!r} no está en formato HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if hour > 23 or minute > 59:
        raise ValueError(f"hora {text!r} fuera de rango")
    return time(hour, minute)


def _merge_ranges(
    ranges: Sequence[Tuple[date, date]],
) -> Tuple[Tuple[date, date], ...]:
    ordered = sorted(ranges)
    if not ordered:
        return ()
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + timedelta(days=1):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _read_cache(path: Path) -> _Cache:
    if not path.is_file():
        return _Cache((), ())
    covered: list[Tuple[date, date]] = []
    sessions: list[Session] = []
    with path.open(encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    body = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# covered:"):
            covered.extend(_parse_covered(stripped.split(":", 1)[1]))
            continue
        if stripped.startswith("#") or not stripped:
            continue
        body.append(line)
    reader = csv.DictReader(io.StringIO("".join(body)))
    if reader.fieldnames is None or tuple(reader.fieldnames) != _COLUMNS:
        raise ValueError(f"cache de calendario con columnas inesperadas: {path}")
    for row in reader:
        day = date.fromisoformat(row["date"])
        sessions.append(Session(
            date=day,
            open_utc=datetime.fromisoformat(row["open_utc"]),
            close_utc=datetime.fromisoformat(row["close_utc"]),
        ))
    sessions.sort(key=lambda item: item.date)
    if not covered and sessions:
        covered = [(sessions[0].date, sessions[-1].date)]
    return _Cache(tuple(covered), tuple(sessions))


def _parse_covered(text: str) -> list[Tuple[date, date]]:
    ranges = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        start_s, end_s = piece.split("..")
        ranges.append((date.fromisoformat(start_s), date.fromisoformat(end_s)))
    return ranges


def _write_cache(path: Path, cache: _Cache) -> None:
    buffer = io.StringIO()
    if cache.ranges:
        parts = ",".join(f"{start.isoformat()}..{end.isoformat()}" for start, end in cache.ranges)
        buffer.write(f"# covered:{parts}\n")
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for session in cache.sessions:
        writer.writerow({
            "date": session.date.isoformat(),
            "open_utc": session.open_utc.isoformat(timespec="seconds"),
            "close_utc": session.close_utc.isoformat(timespec="seconds"),
        })
    _atomic_write(path, buffer.getvalue())


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
