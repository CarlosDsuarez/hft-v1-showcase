"""Cliente REST de datos de Alpaca (solo GET, stdlib).

Host fijo ``https://data.alpaca.markets``. Mismas credenciales que el
broker (headers ``APCA-API-KEY-ID`` y ``APCA-API-SECRET-KEY``; Market Data
FAQ). El secreto no entra en logs ni en excepciones.

Endpoints verificados en docs.alpaca.markets:

- ``GET /v2/stocks/bars`` — histórico multi-símbolo. Params: ``symbols``,
  ``timeframe`` (``1Min``, ``5Min``, ``1Day``, …), ``start``/``end``
  RFC-3339, ``limit`` (default 1000, máximo 10000; cuenta el total, no
  por símbolo), ``page_token``, ``feed`` (``iex`` | ``sip`` | ``boats`` |
  ``otc``), ``adjustment`` (default ``raw``), ``sort``. La respuesta trae
  ``bars`` (objeto símbolo → lista) y ``next_page_token``. Cada barra:
  ``t, o, h, l, c, v, n, vw``. ``t`` es el inicio del intervalo.
- ``GET /v2/stocks/bars/latest`` — última barra de un minuto por símbolo.
  No admite ``adjustment`` (el dato sale como se recibió).
- ``GET /v2/stocks/quotes/latest`` — última cotización (``bp``, ``bs``,
  ``ap``, ``as``, ``t``, …).
- ``GET /v2/stocks/snapshots`` — foto por símbolo. Se devuelve el JSON
  tal cual: el esquema anida trade, quote y barras, y no se reinterpreta.
- ``GET /v1/corporate-actions`` — acciones corporativas. Params:
  ``symbols``, ``types``, ``start``/``end`` (YYYY-MM-DD, por
  ``process_date``), ``limit``, ``page_token``. La respuesta agrupa por
  tipo (``forward_splits``, ``reverse_splits``, …). Documentos viejos
  citan ``/v1beta1/corporate-actions``; el path vigente en la referencia
  actual es ``/v1``. Queda en ``CORPORATE_ACTIONS_PATH``.

Plan Basic. Con ``feed=sip``, un histórico cuyo ``end`` cae después de
``now - 15 min`` se rechaza antes del HTTP (FAQ: el ``end`` tiene que
tener al menos 15 minutos). Las rutas ``/latest`` y ``snapshots`` con
``sip`` también: esas rutas no tienen ``end`` y exigen suscripción para
SIP. ``feed=iex`` no tiene esa ventana.

Paginación. Alpaca ordena por símbolo y después por tiempo, así que una
página de 10000 puntos suele ser de un solo símbolo si ese símbolo llena
el límite. ``estimate_bar_requests`` cuenta con eso.

Rate limit. El plan Basic admite 200 llamadas por minuto. Un token bucket
del cliente no pasa de eso (configurable hacia abajo). Ante 429 o 5xx se
reintenta el GET. Si viene ``Retry-After`` (segundos o fecha HTTP), se
espera eso, con un tope local de 120 s para no colgar el proceso. Si no
viene, se usa ``X-RateLimit-Reset`` (epoch, header documentado en el 429)
y, en su defecto, un backoff 2**intento.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping, Optional, Sequence
from urllib.parse import urlencode, urlparse

import pandas as pd

from src.brokers.alpaca_adapter import BrokerConfig
from src.hft.calendar import SystemClock
from src.hft.data.bars import (
    DEFAULT_FEED,
    IntradayBars,
    as_utc,
    bars_from_records,
    canonical_timeframe,
    check_adjustment,
    check_feed,
    records_from_alpaca,
    to_rfc3339,
    unique_symbols,
)

logger = logging.getLogger("hft.data.alpaca")

DATA_BASE = "https://data.alpaca.markets"
_DATA_HOSTS = frozenset({"data.alpaca.markets", "data.sandbox.alpaca.markets"})
# Tope del plan Basic (About Market Data API). El bucket no lo supera.
MAX_RATE_PER_MIN = 200.0
DEFAULT_RATE_PER_MIN = MAX_RATE_PER_MIN
# Máximo documentado de ``limit`` en /v2/stocks/bars.
PAGE_LIMIT = 10_000
SIP_LAG = timedelta(minutes=15)
# Minutos de una sesión regular 09:30–16:00. Las medias sesiones tienen menos.
REGULAR_SESSION_MINUTES = 390
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_RETRY_AFTER_CAP_S = 120.0
_DEFAULT_TIMEOUT = 30
UTC = timezone.utc

# Tipos que cambian precios o volúmenes históricos. ``adjustment=split``
# cubre splits y spin-offs; ``all`` suma dividendos. Verificado en la doc
# de ``adjustment`` y en los ``types`` de corporate actions.
ADJUSTMENT_ACTION_TYPES = {
    "split": ("forward_split", "reverse_split", "unit_split", "spin_off"),
    "all": (
        "forward_split",
        "reverse_split",
        "unit_split",
        "spin_off",
        "cash_dividend",
        "stock_dividend",
    ),
}

CORPORATE_ACTIONS_PATH = "/v1/corporate-actions"
BARS_PATH = "/v2/stocks/bars"
LATEST_BARS_PATH = "/v2/stocks/bars/latest"
LATEST_QUOTES_PATH = "/v2/stocks/quotes/latest"
SNAPSHOTS_PATH = "/v2/stocks/snapshots"

ClockNow = Callable[[], datetime]
Monotonic = Callable[[], float]
Sleeper = Callable[[float], None]


class DataError(RuntimeError):
    """Fallo de la API de datos. El mensaje no contiene secretos."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class DataAuthError(DataError):
    """401/403: Alpaca rechazó las credenciales. Reintentar no sirve."""


class DataTransientError(DataError):
    """429, 5xx o corte de red. ``retry_after`` en segundos, si vino header."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, status=status)
        self.retry_after = retry_after


class SipWindowError(DataError):
    """``feed=sip`` pediría datos de los últimos 15 minutos (plan Basic)."""


class TokenBucket:
    """Como máximo ``rate_per_min`` adquisiciones por minuto, tope 200."""

    def __init__(
        self,
        rate_per_min: float = DEFAULT_RATE_PER_MIN,
        *,
        monotonic: Optional[Monotonic] = None,
        sleep: Optional[Sleeper] = None,
    ):
        rate = float(rate_per_min)
        if not 0 < rate <= MAX_RATE_PER_MIN:
            raise ValueError(
                f"rate_per_min debe estar en (0, {MAX_RATE_PER_MIN:g}] (recibido: {rate})"
            )
        self.rate_per_min = rate
        self.capacity = rate
        self.tokens = rate
        self._per_sec = rate / 60.0
        self._monotonic = time.monotonic if monotonic is None else monotonic
        self._sleep = time.sleep if sleep is None else sleep
        self._updated = self._monotonic()

    def acquire(self) -> None:
        """Consume un token. Si no hay, espera a que el reloj monotónico avance."""
        while True:
            now = self._monotonic()
            elapsed = max(0.0, now - self._updated)
            self._updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self._per_sec)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            self._sleep((1.0 - self.tokens) / self._per_sec)


def estimate_bar_requests(
    n_symbols: int,
    bars_per_symbol: int,
    *,
    page_limit: int = PAGE_LIMIT,
) -> int:
    """Páginas de ``/v2/stocks/bars`` si cada símbolo trae ``bars_per_symbol``.

    La doc ordena por símbolo y el ``limit`` es el total de la página. Si un
    símbolo solo ya llena el límite, las páginas no se comparten: el total
    es ``n_symbols * ceil(barras / limit)``.
    """
    if n_symbols < 0 or bars_per_symbol < 0:
        raise ValueError("n_symbols y bars_per_symbol no pueden ser negativos")
    if page_limit < 1:
        raise ValueError("page_limit debe ser positivo")
    if n_symbols == 0 or bars_per_symbol == 0:
        return 0
    return n_symbols * math.ceil(bars_per_symbol / page_limit)


def estimate_year_1min(
    n_symbols: int = 10,
    n_sessions: int = 252,
    *,
    minutes_per_session: int = REGULAR_SESSION_MINUTES,
    page_limit: int = PAGE_LIMIT,
) -> int:
    """Techo de llamadas para un año de barras de 1 minuto.

    Usa 390 minutos por sesión (09:30–16:00). Una media sesión tiene menos,
    así que el número real no supera este techo. Con 10 símbolos y 252
    sesiones: 100 páginas, por debajo de las 200 llamadas/minuto del plan.
    """
    if minutes_per_session < 0:
        raise ValueError("minutes_per_session no puede ser negativo")
    return estimate_bar_requests(
        n_symbols,
        n_sessions * minutes_per_session,
        page_limit=page_limit,
    )


def actions_fingerprint(actions: Sequence[Mapping]) -> str:
    """SHA-256 estable de una lista de acciones corporativas."""
    blob = json.dumps(list(actions), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_data_base(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if host.endswith(":443"):
        host = host[:-4]
    if (
        parsed.scheme != "https"
        or host not in _DATA_HOSTS
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            f"host de datos rechazado: {base_url}. Solo se admite "
            f"https://data.alpaca.markets (o el sandbox de partners)."
        )
    return f"https://{host}"


def _scrub(text: str, secrets: Sequence[str]) -> str:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def _header(headers: object, name: str) -> Optional[str]:
    if headers is None or not hasattr(headers, "get"):
        return None
    for key in (name, name.lower(), name.upper()):
        value = headers.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _retry_after_seconds(headers: object, now: datetime) -> Optional[float]:
    """Segundos a esperar. ``Retry-After`` gana; si no, ``X-RateLimit-Reset``."""
    raw = _header(headers, "Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
            except (TypeError, ValueError, IndexError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return max(0.0, (parsed - as_utc(now)).total_seconds())
    reset = _header(headers, "X-RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset) - as_utc(now).timestamp())
        except ValueError:
            return None
    return None


class AlpacaDataClient:
    """GET a la Market Data API. Inyectar ``clock``, ``monotonic`` y ``sleep`` en tests."""

    def __init__(
        self,
        config: BrokerConfig,
        *,
        feed: str = DEFAULT_FEED,
        base_url: str = DATA_BASE,
        rate_per_min: float = DEFAULT_RATE_PER_MIN,
        page_limit: int = PAGE_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        clock: Optional[ClockNow] = None,
        monotonic: Optional[Monotonic] = None,
        sleep: Optional[Sleeper] = None,
        opener: Optional[Callable] = None,
    ):
        if page_limit < 1 or page_limit > PAGE_LIMIT:
            raise ValueError(f"page_limit debe estar en [1, {PAGE_LIMIT}]")
        if max_retries < 1:
            raise ValueError("max_retries debe ser >= 1")
        self.config = config
        self.feed = check_feed(feed)
        self.base_url = _validate_data_base(base_url)
        self.page_limit = int(page_limit)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._clock = SystemClock().now if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        self._opener = urllib.request.urlopen if opener is None else opener
        self._bucket = TokenBucket(rate_per_min, monotonic=monotonic, sleep=self._sleep)
        self.request_count = 0

    def __repr__(self) -> str:
        masked = self.config.api_key[:4] + "…" if self.config.api_key else "(vacía)"
        return (
            f"AlpacaDataClient(api_key={masked}, feed={self.feed}, "
            f"base_url={self.base_url})"
        )

    def get_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        *,
        feed: Optional[str] = None,
        adjustment: str = "raw",
    ) -> dict[str, list]:
        """Todas las páginas de ``/v2/stocks/bars``. ``end`` es exclusivo aquí.

        Alpaca trata ``end`` como inclusivo. Se envía un segundo antes para
        no incluir una barra que caiga exactamente en ``end``.
        """
        wanted = unique_symbols(symbols)
        if not wanted:
            raise ValueError("symbols vacío")
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        if end_utc <= start_utc:
            raise ValueError("end debe ser posterior a start")
        chosen = self.feed if feed is None else check_feed(feed)
        adjust = check_adjustment(adjustment)
        step = canonical_timeframe(timeframe)
        # Inclusivo en Alpaca, exclusivo en este cliente.
        alpaca_end = end_utc - timedelta(seconds=1)
        self._reject_recent_sip(chosen, alpaca_end)
        params = {
            "symbols": ",".join(wanted),
            "timeframe": step,
            "start": to_rfc3339(start_utc),
            "end": to_rfc3339(alpaca_end),
            "limit": str(self.page_limit),
            "adjustment": adjust,
            "feed": chosen,
            "sort": "asc",
        }
        collected: dict[str, list] = {symbol: [] for symbol in wanted}
        token: Optional[str] = None
        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            payload = self._get(BARS_PATH, query)
            bars = payload.get("bars") or {}
            if not isinstance(bars, dict):
                raise DataError("GET /v2/stocks/bars: 'bars' debe ser un objeto")
            for symbol, rows in bars.items():
                if not isinstance(rows, list):
                    raise DataError(f"GET /v2/stocks/bars: barras de {symbol} no son una lista")
                collected.setdefault(str(symbol), []).extend(rows)
            token = payload.get("next_page_token") or None
            if not token:
                break
        return collected

    def get_latest_bars(
        self,
        symbols: Sequence[str],
        *,
        feed: Optional[str] = None,
    ) -> dict[str, Mapping]:
        """``GET /v2/stocks/bars/latest``. Objeto símbolo → una barra, no una lista."""
        chosen = self.feed if feed is None else check_feed(feed)
        self._reject_recent_sip(chosen, None)
        payload = self._get(
            LATEST_BARS_PATH,
            {"symbols": ",".join(unique_symbols(symbols)), "feed": chosen},
        )
        bars = payload.get("bars") or {}
        if not isinstance(bars, dict):
            raise DataError("GET /v2/stocks/bars/latest: 'bars' debe ser un objeto")
        return bars

    def get_latest_quotes(
        self,
        symbols: Sequence[str],
        *,
        feed: Optional[str] = None,
    ) -> dict[str, Mapping]:
        """``GET /v2/stocks/quotes/latest``. Campos de cada quote: los que envíe Alpaca."""
        chosen = self.feed if feed is None else check_feed(feed)
        self._reject_recent_sip(chosen, None)
        payload = self._get(
            LATEST_QUOTES_PATH,
            {"symbols": ",".join(unique_symbols(symbols)), "feed": chosen},
        )
        quotes = payload.get("quotes") or {}
        if not isinstance(quotes, dict):
            raise DataError("GET /v2/stocks/quotes/latest: 'quotes' debe ser un objeto")
        return quotes

    def get_snapshots(
        self,
        symbols: Sequence[str],
        *,
        feed: Optional[str] = None,
    ) -> dict:
        """``GET /v2/stocks/snapshots``. JSON por símbolo, sin reinterpretar campos."""
        chosen = self.feed if feed is None else check_feed(feed)
        self._reject_recent_sip(chosen, None)
        payload = self._get(
            SNAPSHOTS_PATH,
            {"symbols": ",".join(unique_symbols(symbols)), "feed": chosen},
        )
        if not isinstance(payload, dict):
            raise DataError("GET /v2/stocks/snapshots: se esperaba un objeto")
        payload.pop("next_page_token", None)
        return payload

    def get_corporate_actions(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        *,
        types: Sequence[str],
    ) -> list[dict]:
        """Acciones en ``[start, end]`` (fechas de proceso, inclusivas).

        Cada elemento es el objeto de Alpaca más ``type`` (la clave del
        grupo, en singular de la doc: ``forward_split``, …). No se inventan
        campos de precio.
        """
        wanted = unique_symbols(symbols)
        if not wanted:
            raise ValueError("symbols vacío")
        if end < start:
            raise ValueError("end debe ser posterior o igual a start")
        type_list = tuple(str(item).strip() for item in types if str(item).strip())
        if not type_list:
            raise ValueError("types vacío")
        params = {
            "symbols": ",".join(wanted),
            "types": ",".join(type_list),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": "1000",
            "sort": "asc",
        }
        found: list[dict] = []
        token: Optional[str] = None
        while True:
            query = dict(params)
            if token:
                query["page_token"] = token
            payload = self._get(CORPORATE_ACTIONS_PATH, query)
            groups = payload.get("corporate_actions") or {}
            if not isinstance(groups, dict):
                raise DataError("corporate_actions debe ser un objeto")
            for group, rows in groups.items():
                if not isinstance(rows, list):
                    raise DataError(f"corporate_actions.{group} no es una lista")
                kind = _action_type(str(group))
                for row in rows:
                    if not isinstance(row, dict):
                        raise DataError("una acción corporativa no es un objeto")
                    item = dict(row)
                    item["type"] = kind
                    found.append(item)
            token = payload.get("next_page_token") or None
            if not token:
                break
        return found

    def _reject_recent_sip(self, feed: str, end: Optional[datetime]) -> None:
        """Plan Basic: SIP histórico solo si ``end <= now - 15 min``.

        ``end is None`` cubre latest y snapshots, que no tienen ventana y
        exigen suscripción para SIP (FAQ).
        """
        if feed != "sip":
            return
        now = as_utc(self._clock())
        limit = now - SIP_LAG
        if end is None or as_utc(end) > limit:
            raise SipWindowError(
                "feed=sip en el plan Basic exige un end anterior o igual a "
                "now - 15 min. Las rutas latest y snapshots con sip se "
                "rechazan siempre: piden el dato reciente (Market Data FAQ)."
            )

    def _get(self, path: str, params: Mapping[str, str]) -> dict:
        query = urlencode(params)
        url = f"{self.base_url}{path}?{query}" if query else f"{self.base_url}{path}"
        last: Optional[DataTransientError] = None
        for attempt in range(1, self.max_retries + 1):
            self._bucket.acquire()
            try:
                return self._send(url)
            except DataTransientError as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                delay = exc.retry_after
                if delay is None:
                    delay = _BACKOFF_BASE ** attempt
                delay = min(float(delay), _RETRY_AFTER_CAP_S)
                logger.warning(
                    "GET %s intento %s/%s falló (HTTP %s); espera %.1fs",
                    path,
                    attempt,
                    self.max_retries,
                    exc.status,
                    delay,
                )
                self._sleep(delay)
        assert last is not None
        raise DataTransientError(
            f"Reintentos agotados: {last}",
            status=last.status,
            retry_after=last.retry_after,
        )

    def _send(self, url: str) -> dict:
        self.request_count += 1
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:
                detail = ""
            path = urlparse(url).path
            if exc.code in (401, 403):
                # El 401 de Alpaca llega como HTML de nginx: no sirve en el log.
                raise DataAuthError(
                    f"HTTP {exc.code} {path}: credenciales rechazadas por "
                    f"{urlparse(url).netloc}. Revisar BROKER_API_KEY / "
                    f"BROKER_SECRET_KEY en .env (regenerarlas en el dashboard "
                    f"paper de Alpaca si fueron rotadas).",
                    status=exc.code,
                ) from None
            message = _scrub(
                f"HTTP {exc.code} {path} {detail}",
                (self.config.api_key, self.config.secret_key),
            )
            retry_after = _retry_after_seconds(exc.headers, self._clock())
            if exc.code == 429 or 500 <= exc.code < 600:
                raise DataTransientError(message, status=exc.code, retry_after=retry_after) from None
            raise DataError(message, status=exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            # IncompleteRead es HTTPException, no OSError: un body cortado es transitorio.
            path = urlparse(url).path
            raise DataTransientError(
                _scrub(f"red {path}: {type(exc).__name__}", (self.config.api_key, self.config.secret_key))
            ) from None
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
            path = urlparse(url).path
            raise DataTransientError(f"respuesta cortada {path}: {type(exc).__name__}") from None
        if not isinstance(payload, dict):
            raise DataError("respuesta de datos no es un objeto JSON")
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
            "Accept": "application/json",
        }


def _action_type(group: str) -> str:
    """``forward_splits`` → ``forward_split``. Si ya viene en singular, se deja."""
    if group.endswith("s") and not group.endswith("ss"):
        return group[:-1]
    return group


class AlpacaProvider:
    """``IntradayProvider`` sobre ``AlpacaDataClient``. Siempre pide ``adjustment=raw``."""

    name = "alpaca"

    def __init__(self, client: AlpacaDataClient, *, feed: Optional[str] = None):
        self.client = client
        self.feed = client.feed if feed is None else check_feed(feed)

    def get_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> IntradayBars:
        step = canonical_timeframe(timeframe)
        wanted = unique_symbols(symbols)
        raw = self.client.get_bars(
            wanted,
            start,
            end,
            step,
            feed=self.feed,
            adjustment="raw",
        )
        records = records_from_alpaca(raw)
        if not records.empty:
            left = pd.Timestamp(as_utc(start))
            right = pd.Timestamp(as_utc(end))
            records = records[(records["ts"] >= left) & (records["ts"] < right)]
        return bars_from_records(
            records,
            wanted,
            source=f"alpaca:{self.feed}",
            feed=self.feed,
            timeframe=step,
            adjustment="raw",
        )


def slice_window(bars: IntradayBars, start: datetime, end: datetime, symbols: Sequence[str]) -> IntradayBars:
    """Recorta un bloque ya cargado a ``[start, end)`` y a ``symbols``."""
    start_utc = pd.Timestamp(as_utc(start))
    end_utc = pd.Timestamp(as_utc(end))
    wanted = unique_symbols(symbols)
    index = bars.close.index
    mask = (index >= start_utc) & (index < end_utc)
    frames = {}
    for name in ("open", "high", "low", "close", "volume", "vwap", "trades"):
        frame = getattr(bars, name).loc[mask].reindex(columns=wanted)
        frames[name] = frame
    received = tuple(
        symbol
        for symbol in wanted
        if symbol in frames["close"].columns and frames["close"][symbol].notna().any()
    )
    meta = type(bars.meta)(
        source=bars.meta.source,
        feed=bars.meta.feed,
        timeframe=bars.meta.timeframe,
        adjustment=bars.meta.adjustment,
        requested=wanted,
        received=received,
    )
    return IntradayBars(meta=meta, **frames)
