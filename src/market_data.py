"""
market_data.py — Proveedor unico de datos de mercado con contrato explicito.

Contrato
--------
    MarketDataProvider.get_bars(symbols, start, end) -> Bars(close, volume, meta)

- Una sola descarga OHLCV por llamada: cierres ajustados + volumen juntos.
- Ventana [start, end): `end` exclusivo (como yfinance).
- Ningun simbolo con datos → DataUnavailableError.
- `Bars.meta`: simbolos pedidos vs recibidos, ultima fecha por simbolo,
  fuente usada, sesion esperada, barras parciales descartadas, stale.

Implementaciones
----------------
    YahooProvider   yfinance; retry con backoff exponencial ante errores y
                    reintento solo de los simbolos que llegaron vacios.
    CsvProvider     CSV ancho local (``date`` + una columna por simbolo).
                    Rutas configurables; default ``<data_dir>/raw_prices_backup.csv``
                    y ``raw_volumes_backup.csv`` (src/settings.py, nunca el cwd).
    FakeProvider    frames en memoria (tests y demos offline).
    FallbackChain   prueba proveedores en orden; el primero con datos gana.
    CachedProvider  cache parquet por simbolo: solo descarga barras nuevas,
                    re-descarga si la historia ajustada cambio (split/dividendo)
                    y nunca persiste la barra de una sesion aun abierta.
    InstrumentProvider  adaptador del maestro (src/instruments.py): pide por
                    `id`, descarga el ``yahoo_symbol`` y devuelve columnas por
                    `id` en moneda LOCAL; `get_fx` trae la serie FX a moneda
                    base. `close_in_base` convierte (src.fx.to_base).

`build_provider(sources, cache_dir=...)` arma la cadena (default yahoo → csv).

Politica (`load_bars(provider, symbols, start, end, policy, now)`)
-----------------------------------------------------------------
1. Barra parcial (sesion de hoy sin cerrar segun America/New_York):
   ``on_partial_bar`` = "drop" (default) | "abort" → PartialBarError.
2. Simbolos pedidos sin datos: ``on_missing`` = "error" (default, live)
   → MissingSymbolsError | "warn".
3. Frescura: si la ventana cubre la ultima sesion cerrada, cada simbolo debe
   tener barra en ella: ``on_stale`` = "error" (default) → StaleDataError | "warn".

Calendario: dias habiles de pandas con feriados NYSE (CustomBusinessDay),
cierre 16:00 ET. No modela medias sesiones (cierre 13:00) ni cierres
extraordinarios (p. ej. duelo nacional): ese dia la barra se trata como
parcial hasta las 16:00 y, si falta la sesion esperada, usar on_stale="warn".
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time as _time
from dataclasses import dataclass, replace
from datetime import datetime, time, timezone
from pathlib import Path
from typing import (
    Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

from src.fx import MissingFxError, fx_pair_symbol, to_base
from src.instruments import InstrumentMaster
from src.settings import BASE_CURRENCY, Settings, load_settings

logger = logging.getLogger("market_data")

DateLike = Union[str, pd.Timestamp, datetime]

MARKET_TZ = "America/New_York"
MARKET_CLOSE = time(16, 0)
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # segundos; espera tras el intento k = BACKOFF_BASE ** k
AVAILABLE_SOURCES = ("yahoo", "csv")
DEFAULT_SOURCES = ("yahoo", "csv")
MISSING_POLICIES = ("error", "warn")
PARTIAL_BAR_POLICIES = ("drop", "abort")
STALE_POLICIES = ("error", "warn")


# ===================================================================
# Errores
# ===================================================================
class MarketDataError(RuntimeError):
    """Datos de mercado inutilizables para la corrida."""


class DataUnavailableError(MarketDataError):
    """Ninguna fuente devolvio datos para los simbolos pedidos."""


class MissingSymbolsError(MarketDataError):
    """Simbolos pedidos sin datos (politica on_missing="error")."""

    def __init__(self, missing: Sequence[str], source: str):
        self.missing = tuple(missing)
        super().__init__(
            f"{len(self.missing)} symbol(s) without data from {source}: "
            f"{', '.join(self.missing)}"
        )


class PartialBarError(MarketDataError):
    """Barras de una sesion aun abierta (politica on_partial_bar="abort")."""

    def __init__(self, dates: Sequence[pd.Timestamp], expected: pd.Timestamp):
        self.dates = tuple(dates)
        super().__init__(
            f"Bar(s) {', '.join(_iso(d) for d in self.dates)} belong to a session "
            f"not yet closed (last closed session: {_iso(expected)})"
        )


class StaleDataError(MarketDataError):
    """Simbolos sin barra en la ultima sesion cerrada (on_stale="error")."""

    def __init__(self, stale: Sequence[str], expected: pd.Timestamp,
                 last_date: Mapping[str, pd.Timestamp]):
        self.stale = tuple(stale)
        self.expected = expected
        detail = ", ".join(f"{s} (last {_iso(last_date[s])})" for s in self.stale)
        super().__init__(
            f"Stale data: last closed session is {_iso(expected)} but "
            f"{len(self.stale)} symbol(s) end earlier: {detail}"
        )


# ===================================================================
# Calendario NYSE
# ===================================================================
class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Feriados regulares NYSE (sin medias sesiones ni cierres extraordinarios)."""

    rules = [
        # NYSE no cierra el viernes previo si el 1-ene cae sabado.
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@functools.lru_cache(maxsize=1)
def nyse_session() -> CustomBusinessDay:
    """Offset de sesion NYSE (pandas bdate + feriados NYSE)."""
    return CustomBusinessDay(calendar=NYSEHolidayCalendar())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def last_closed_session(
    now: Optional[DateLike] = None,
    close_time: time = MARKET_CLOSE,
) -> pd.Timestamp:
    """Ultima sesion NYSE cerrada a la hora `now` (naive = UTC).

    Hoy cuenta si es sesion y ya pasaron las `close_time` en Nueva York;
    si no, la sesion habil anterior.
    """
    ts = pd.Timestamp(now if now is not None else _utcnow())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(MARKET_TZ)
    today = pd.Timestamp(local.date())
    session = nyse_session()
    if session.is_on_offset(today) and local.time() >= close_time:
        return today
    return today - session


# ===================================================================
# Contrato: Bars
# ===================================================================
@dataclass(frozen=True)
class BarsMeta:
    """Procedencia y calidad de una descarga."""

    source: str
    requested: Tuple[str, ...]
    received: Tuple[str, ...]
    last_date: Mapping[str, pd.Timestamp]
    expected_session: Optional[pd.Timestamp] = None
    dropped_partial: Tuple[pd.Timestamp, ...] = ()
    stale: Tuple[str, ...] = ()

    @property
    def missing(self) -> Tuple[str, ...]:
        got = set(self.received)
        return tuple(s for s in self.requested if s not in got)


@dataclass(frozen=True)
class Bars:
    """Cierres ajustados y volumen (fecha × simbolo, mismo indice) + meta."""

    close: pd.DataFrame
    volume: pd.DataFrame
    meta: BarsMeta


class MarketDataProvider(Protocol):
    """Fuente de barras diarias. Ver contrato en el docstring del modulo."""

    name: str

    def get_bars(self, symbols: Sequence[str], start: DateLike,
                 end: DateLike) -> Bars: ...


def _iso(d: DateLike) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _unique(symbols: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(symbols))


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Indice de fechas naive (fecha local de la bolsa), ordenado y unico."""
    out = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out.columns.name = None
    return out[~out.index.duplicated(keep="last")].sort_index()


def _window(frame: pd.DataFrame, start: DateLike, end: DateLike) -> pd.DataFrame:
    idx = pd.DatetimeIndex(frame.index)
    return frame.loc[(idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))]


def make_bars(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    *,
    source: str,
    requested: Sequence[str],
) -> Bars:
    """Normaliza close/volume y calcula meta (recibidos = con algun cierre)."""
    requested = tuple(_unique(requested))
    close = _normalize_index(close)
    received = [s for s in requested if s in close.columns and close[s].notna().any()]
    close = close.loc[:, received].astype(float).dropna(how="all")
    if volume is None:
        volume = pd.DataFrame(index=close.index, columns=received, dtype=float)
    volume = _normalize_index(volume).reindex(index=close.index, columns=received)
    last_date = {s: close[s].last_valid_index() for s in received}
    meta = BarsMeta(source=source, requested=requested, received=tuple(received),
                    last_date=last_date)
    return Bars(close=close, volume=volume.astype(float), meta=meta)


# ===================================================================
# Proveedores
# ===================================================================
class FakeProvider:
    """Proveedor en memoria (tests/demos). Registra cada llamada en `calls`."""

    def __init__(self, close: pd.DataFrame, volume: Optional[pd.DataFrame] = None,
                 name: str = "fake"):
        self.close = close
        self.volume = volume
        self.name = name
        self.calls: List[Tuple[Tuple[str, ...], str, str]] = []

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        symbols = _unique(symbols)
        self.calls.append((tuple(symbols), str(start), str(end)))
        close = _window(self.close, start, end)
        cols = [s for s in symbols if s in close.columns]
        if not cols or close[cols].dropna(how="all").empty:
            raise DataUnavailableError(
                f"{self.name}: no data for {len(symbols)} symbol(s) in "
                f"[{_iso(start)}, {_iso(end)})")
        volume = None if self.volume is None else _window(self.volume, start, end)
        return make_bars(close[cols], volume, source=self.name, requested=symbols)


def _yf_download(symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # diferido: solo este proveedor toca la red

    return yf.download(list(symbols), start=start, end=end, auto_adjust=True,
                       progress=False, threads=True, group_by="column")


def _split_ohlcv(raw: Optional[pd.DataFrame],
                 symbols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Salida de yf.download → (close, volume) con un simbolo por columna."""
    if raw is None or raw.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), dtype=float)
        return empty, empty
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = (raw["Volume"] if "Volume" in raw.columns.get_level_values(0)
                  else pd.DataFrame(index=raw.index, columns=close.columns, dtype=float))
    else:
        if len(symbols) != 1:
            raise ValueError(f"flat yfinance frame for {len(symbols)} symbols")
        close = raw[["Close"]].set_axis(list(symbols), axis=1)
        volume = (raw[["Volume"]].set_axis(list(symbols), axis=1) if "Volume" in raw
                  else pd.DataFrame(index=raw.index, columns=list(symbols), dtype=float))
    return _normalize_index(close), _normalize_index(volume)


class YahooProvider:
    """Yahoo Finance: una descarga OHLCV por intento, retry con backoff.

    Reintenta ante cualquier excepcion y, si la descarga llega parcial, solo
    los simbolos sin datos. No duerme tras el ultimo intento.
    """

    name = "yahoo"

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE,
        downloader: Optional[Callable[[Sequence[str], str, str], pd.DataFrame]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._download = downloader or _yf_download
        self._sleep = sleep or _time.sleep

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        requested = _unique(symbols)
        pending = list(requested)
        closes: List[pd.DataFrame] = []
        volumes: List[pd.DataFrame] = []
        reason = "no data"
        for attempt in range(1, self.max_retries + 1):
            try:
                close, volume = _split_ohlcv(
                    self._download(pending, _iso(start), _iso(end)), pending)
            except Exception as exc:  # noqa: BLE001 — red/HTTP/parseo: se reintenta
                reason = f"{type(exc).__name__}: {exc}"
            else:
                got = [s for s in pending if s in close.columns and close[s].notna().any()]
                if got:
                    closes.append(close[got])
                    volumes.append(volume.reindex(columns=got))
                pending = [s for s in pending if s not in got]
                if not pending:
                    break
                reason = f"{len(pending)} symbol(s) without data: {pending[:10]}"
            if attempt < self.max_retries:
                wait = self.backoff_base ** attempt
                logger.warning("yahoo attempt %d/%d failed (%s) — retrying in %.1fs",
                               attempt, self.max_retries, reason, wait)
                self._sleep(wait)
        if not closes:
            raise DataUnavailableError(
                f"yahoo: no data after {self.max_retries} attempt(s) ({reason})")
        if pending:
            logger.warning("yahoo: %d symbol(s) still without data after %d attempt(s): %s",
                           len(pending), self.max_retries, pending)
        return make_bars(pd.concat(closes, axis=1), pd.concat(volumes, axis=1),
                         source=self.name, requested=requested)


def _read_wide_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


class CsvProvider:
    """CSV ancho local (``date`` + una columna por simbolo).

    Rutas resueltas al leer (HFT_HOME vigente), nunca relativas al cwd.
    Sin archivo de volumenes → volumen NaN (el factor de liquidez lo excluye).
    """

    name = "csv"

    def __init__(
        self,
        close_path: Optional[Union[str, Path]] = None,
        volume_path: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ):
        self.close_path = close_path
        self.volume_path = volume_path
        self._settings = settings

    def paths(self) -> Tuple[Path, Path]:
        s = self._settings or load_settings()
        return (Path(self.close_path) if self.close_path else s.price_backup_csv,
                Path(self.volume_path) if self.volume_path else s.volume_backup_csv)

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        requested = _unique(symbols)
        close_path, volume_path = self.paths()
        if not close_path.exists():
            raise DataUnavailableError(f"csv: {close_path} not found")
        close = _window(_read_wide_csv(close_path), start, end)
        cols = [s for s in requested if s in close.columns]
        if not cols or close[cols].dropna(how="all").empty:
            raise DataUnavailableError(
                f"csv: no data for requested symbols in {close_path} "
                f"[{_iso(start)}, {_iso(end)})")
        volume = (_window(_read_wide_csv(volume_path), start, end)
                  if volume_path.exists() else None)
        return make_bars(close[cols], volume, source=self.name, requested=requested)


class FallbackChain:
    """Prueba cada proveedor en orden; devuelve el primero con datos."""

    def __init__(self, providers: Sequence[MarketDataProvider]):
        if not providers:
            raise ValueError("FallbackChain needs at least one provider")
        self.providers = list(providers)
        self.name = ">".join(p.name for p in self.providers)

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        errors: List[str] = []
        for provider in self.providers:
            try:
                bars = provider.get_bars(symbols, start, end)
            except Exception as exc:  # noqa: BLE001 — se registra y se pasa a la siguiente
                logger.warning("Market data source %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
                continue
            if bars.close.empty:
                errors.append(f"{provider.name}: empty")
                continue
            if errors:
                logger.warning("Using fallback market data source %s", provider.name)
            return bars
        raise DataUnavailableError(
            "All data sources exhausted. Check network or provide CSV fallback. "
            + "; ".join(errors))


class CachedProvider:
    """Cache parquet por simbolo delante de otro proveedor.

    Por simbolo: sin cache o historia mas corta que la pedida → descarga
    completa (un simbolo listado despues de `start` se re-descarga en cada
    llamada); cache atrasado → descarga desde su ultima barra (solape). Si el
    solape no coincide (la historia ajustada cambio por split/dividendo) se
    re-descarga completo. Solo se persisten barras de sesiones ya cerradas
    segun `clock`; las demas se devuelven para que `load_bars` decida.
    Si la descarga falla se sirve lo cacheado (la politica de frescura decide).
    """

    name = "cache"

    def __init__(
        self,
        inner: MarketDataProvider,
        cache_dir: Union[str, Path],
        clock: Optional[Callable[[], datetime]] = None,
        close_time: time = MARKET_CLOSE,
        rtol: float = 1e-6,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self._clock = clock or _utcnow
        self.close_time = close_time
        self.rtol = rtol

    def _file(self, symbol: str) -> Path:
        return self.cache_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", symbol) + ".parquet")

    def _read(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._file(symbol)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 — cache corrupto = sin cache
            logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
            return None

    def _write(self, symbol: str, frame: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._file(symbol)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp)
        os.replace(tmp, path)

    def _fetch(self, symbols: Sequence[str], start: DateLike, end: DateLike,
               sources: List[str]) -> Dict[str, pd.DataFrame]:
        try:
            bars = self.inner.get_bars(symbols, start, end)
        except MarketDataError as exc:
            logger.warning("cache: download of %d symbol(s) failed, serving cached "
                           "data: %s", len(symbols), exc)
            return {}
        sources.append(bars.meta.source)
        return {
            s: pd.DataFrame({"close": bars.close[s], "volume": bars.volume[s]})
            .dropna(subset=["close"])
            for s in bars.meta.received
        }

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        requested = _unique(symbols)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        sessions = pd.date_range(start_ts, end_ts, freq=nyse_session(),
                                 inclusive="left")
        if len(sessions) == 0:
            return self.inner.get_bars(requested, start, end)
        settled = last_closed_session(self._clock(), self.close_time)
        last_needed = min(sessions[-1], settled)

        frames = {s: self._read(s) for s in requested}
        full: List[str] = []
        incremental: Dict[str, pd.Timestamp] = {}
        for s, cached in frames.items():
            if cached is None or cached.empty or cached.index[0] > sessions[0]:
                full.append(s)
            elif cached.index[-1] < last_needed:
                incremental[s] = cached.index[-1]

        sources: List[str] = []
        dirty: set = set()
        invalid: set = set()
        if incremental:
            fresh = self._fetch(list(incremental), min(incremental.values()), end, sources)
            for s, new in fresh.items():
                cached = frames[s]
                overlap = cached.index.intersection(new.index)
                if new.empty or overlap.empty or not np.allclose(
                        new.loc[overlap, "close"], cached.loc[overlap, "close"],
                        rtol=self.rtol, atol=0.0):
                    logger.info("cache: adjusted history changed for %s, refetching", s)
                    full.append(s)
                    invalid.add(s)
                    continue
                frames[s] = pd.concat([cached[cached.index < new.index[0]], new])
                dirty.add(s)
        if full:
            fresh = self._fetch(full, start_ts, end, sources)
            for s in full:
                if s in fresh:
                    frames[s] = fresh[s]
                    dirty.add(s)
                elif s in invalid:
                    frames[s] = None  # historia inconsistente: mejor faltante

        for s in sorted(dirty):
            settled_rows = frames[s][frames[s].index <= settled]
            if not settled_rows.empty:
                self._write(s, settled_rows)

        windows = {s: _window(df, start_ts, end_ts) for s, df in frames.items()
                   if df is not None and not df.empty}
        if not windows:
            raise DataUnavailableError(
                f"cache: no data for {len(requested)} symbol(s) in "
                f"[{_iso(start)}, {_iso(end)})")
        close = pd.DataFrame({s: w["close"] for s, w in windows.items()})
        volume = pd.DataFrame({s: w["volume"] for s, w in windows.items()})
        source = "+".join(["cache", *dict.fromkeys(sources)])
        return make_bars(close, volume, source=source, requested=requested)


class InstrumentProvider:
    """Adaptador de datos del maestro de instrumentos (ids ↔ simbolos Yahoo).

    `get_bars(ids)` descarga el ``yahoo_symbol`` de cada id y devuelve las
    columnas por id, en moneda LOCAL. Ids sin definicion en el maestro se
    piden tal cual (compatibilidad con universos/tests sin maestro).
    `get_fx(monedas)` devuelve unidades de moneda base por 1 unidad de cada
    moneda (par ``{base}{ccy}=X`` invertido). Convertir con `src.fx.to_base`.
    """

    def __init__(
        self,
        inner: MarketDataProvider,
        instruments: InstrumentMaster,
        base_currency: str = BASE_CURRENCY,
    ):
        self.inner = inner
        self.instruments = instruments
        self.base_currency = base_currency
        self.name = inner.name

    def _yahoo(self, instrument_id: str) -> str:
        inst = self.instruments.get(instrument_id)
        return inst.yahoo_symbol if inst is not None else instrument_id

    def get_bars(self, symbols: Sequence[str], start: DateLike, end: DateLike) -> Bars:
        ids = _unique(symbols)
        to_yahoo = {i: self._yahoo(i) for i in ids}
        bars = self.inner.get_bars(list(to_yahoo.values()), start, end)
        to_id = {y: i for i, y in to_yahoo.items()}
        return make_bars(bars.close.rename(columns=to_id),
                         bars.volume.rename(columns=to_id),
                         source=bars.meta.source, requested=ids)

    def currencies(self, ids: Sequence[str]) -> Dict[str, str]:
        """id → moneda local; ids sin definicion se asumen en moneda base (WARNING)."""
        known = self.instruments.currencies(ids)
        unknown = [i for i in ids if i not in known]
        if unknown:
            logger.warning("%d id(s) not in the instrument master, assumed in %s: %s",
                           len(unknown), self.base_currency, unknown)
        return {i: known.get(i, self.base_currency) for i in ids}

    def get_fx(self, currencies: Sequence[str], start: DateLike,
               end: DateLike) -> pd.DataFrame:
        """Serie FX (fecha × moneda): unidades de base por 1 unidad de moneda.

        Raises:
            MissingFxError: algun par sin datos en la fuente.
        """
        foreign = [c for c in _unique(currencies) if c != self.base_currency]
        if not foreign:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"), dtype=float)
        pairs = {fx_pair_symbol(c, self.base_currency): c for c in foreign}
        try:
            bars = self.inner.get_bars(list(pairs), start, end)
        except DataUnavailableError as exc:
            raise MissingFxError(foreign, str(exc)) from exc
        if bars.meta.missing:
            raise MissingFxError([pairs[s] for s in bars.meta.missing],
                                 f"source {bars.meta.source}")
        quotes = bars.close.rename(columns=pairs)
        return 1.0 / quotes.where(quotes > 0)


def close_in_base(
    provider: MarketDataProvider,
    close: pd.DataFrame,
    start: DateLike,
    end: DateLike,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Cierres en moneda local → moneda base (src.fx.to_base), antes de
    estrategia y ledger. Universo solo en moneda base → sin descarga FX.

    Solo un `InstrumentProvider` conoce la moneda de cada id; otro proveedor
    (p. ej. FakeProvider en tests o CSV ya en base) se trata como ya en
    moneda base.

    Returns:
        (cierres en base, id → moneda local).

    Raises:
        MissingFxError: moneda local sin serie FX.
    """
    if not isinstance(provider, InstrumentProvider):
        return close, {}
    currencies = provider.currencies(list(close.columns))
    foreign = sorted(set(currencies.values()) - {provider.base_currency})
    if not foreign:
        return close, currencies
    fx = provider.get_fx(foreign, start, end)
    logger.info("Converting %d instrument(s) in %s to %s",
                sum(c != provider.base_currency for c in currencies.values()),
                foreign, provider.base_currency)
    return to_base(close, fx, currencies, base=provider.base_currency), currencies


def build_provider(
    sources: Sequence[str] = DEFAULT_SOURCES,
    *,
    settings: Optional[Settings] = None,
    csv_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    clock: Optional[Callable[[], datetime]] = None,
) -> MarketDataProvider:
    """Cadena de fallback configurable por nombre (``yahoo``, ``csv``).

    Args:
        sources: Orden de las fuentes. Una sola → ese proveedor sin cadena.
        settings: Para rutas default del CSV (default: load_settings() al leer).
        csv_path: Ruta del CSV de precios (default settings.price_backup_csv).
        cache_dir: Si se da, la cadena va detras de un CachedProvider.
        clock: Reloj UTC del cache (tests).

    Raises:
        ValueError: fuente desconocida o lista vacia.
    """
    factories: Dict[str, Callable[[], MarketDataProvider]] = {
        "yahoo": YahooProvider,
        "csv": lambda: CsvProvider(close_path=csv_path, settings=settings),
    }
    names = [str(s).strip().lower() for s in sources]
    unknown = [n for n in names if n not in AVAILABLE_SOURCES]
    if unknown or not names:
        raise ValueError(f"Unknown market data source(s) {unknown or names}; "
                         f"available: {sorted(factories)}")
    providers = [factories[n]() for n in names]
    provider: MarketDataProvider = (providers[0] if len(providers) == 1
                                    else FallbackChain(providers))
    if cache_dir is not None:
        provider = CachedProvider(provider, cache_dir, clock=clock)
    return provider


# ===================================================================
# Politica: faltantes, barra parcial y frescura
# ===================================================================
@dataclass(frozen=True)
class DataPolicy:
    """Que hacer ante simbolos faltantes, barra intradia y datos atrasados."""

    on_missing: str = "error"
    on_partial_bar: str = "drop"
    on_stale: str = "error"
    close_time: time = MARKET_CLOSE

    def __post_init__(self):
        for field_name, allowed in (("on_missing", MISSING_POLICIES),
                                    ("on_partial_bar", PARTIAL_BAR_POLICIES),
                                    ("on_stale", STALE_POLICIES)):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"{field_name}={value!r} not in {allowed}")


# Live: nunca operar con un universo incompleto, barra parcial o datos viejos.
LIVE_POLICY = DataPolicy()
# Investigacion/backtests: se tolera universo parcial e historia atrasada.
RESEARCH_POLICY = DataPolicy(on_missing="warn", on_stale="warn")


def load_bars(
    provider: MarketDataProvider,
    symbols: Sequence[str],
    start: DateLike,
    end: DateLike,
    *,
    policy: DataPolicy = LIVE_POLICY,
    now: Optional[DateLike] = None,
) -> Bars:
    """Descarga via `provider` y aplica `policy` (ver docstring del modulo).

    Args:
        now: Reloj (naive = UTC) para decidir la ultima sesion cerrada.
            Default: ahora.

    Raises:
        PartialBarError, MissingSymbolsError, StaleDataError segun `policy`;
        DataUnavailableError si no queda ningun simbolo con datos.
    """
    requested = _unique(symbols)
    bars = provider.get_bars(requested, start, end)
    expected = last_closed_session(now, policy.close_time)

    close, volume = bars.close, bars.volume
    partial = tuple(close.index[close.index > expected])
    if partial:
        if policy.on_partial_bar == "abort":
            raise PartialBarError(partial, expected)
        logger.warning("Dropping %d bar(s) of sessions not yet closed: %s "
                       "(last closed session %s)", len(partial),
                       [_iso(d) for d in partial], _iso(expected))
        keep = close.index <= expected
        close, volume = close.loc[keep], volume.loc[keep]

    out = make_bars(close, volume, source=bars.meta.source, requested=requested)
    meta = out.meta
    if meta.missing:
        if policy.on_missing == "error":
            raise MissingSymbolsError(meta.missing, meta.source)
        logger.warning("%d of %d symbol(s) without data from %s: %s",
                       len(meta.missing), len(requested), meta.source,
                       list(meta.missing))
    if not meta.received:
        raise DataUnavailableError(f"{meta.source}: no data for requested symbols")

    stale: Tuple[str, ...] = ()
    if pd.Timestamp(end) > expected:  # la ventana cubre la ultima sesion cerrada
        stale = tuple(s for s in meta.received if meta.last_date[s] < expected)
        if stale and policy.on_stale == "error":
            raise StaleDataError(stale, expected, meta.last_date)
        if stale:
            logger.warning("Stale data: %d symbol(s) without bar for %s: %s",
                           len(stale), _iso(expected), list(stale))

    meta = replace(meta, expected_session=expected, dropped_partial=partial,
                   stale=stale)
    logger.info("Market data: %d/%d symbols from %s, %d bars up to %s",
                len(meta.received), len(requested), meta.source, len(out.close),
                _iso(out.close.index[-1]))
    return Bars(close=out.close, volume=out.volume, meta=meta)
