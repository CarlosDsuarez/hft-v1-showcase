"""Contrato de barras intradía.

Frames anchos (columna = símbolo, índice ``ts`` en UTC) para open, high,
low, close, volume, vwap y trades. Una barra ausente queda en NaN: en IEX
un minuto sin trades no trae fila, y rellenar con cero fingiría un precio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

# Feed por defecto del backtest y del vivo (plan Basic, D-02).
DEFAULT_FEED = "iex"
FEEDS = ("iex", "sip", "boats", "otc")

UTC = timezone.utc

# Columnas del CSV y del registro largo. ``vw`` es el VWAP de Alpaca; ``n``
# es el conteo de trades.
LONG_COLUMNS = ("symbol", "ts", "o", "h", "l", "c", "v", "vw", "n")
_PRICE_COLUMNS = ("o", "h", "l", "c", "v", "vw", "n")

_WIDE = (
    ("o", "open"),
    ("h", "high"),
    ("l", "low"),
    ("c", "close"),
    ("v", "volume"),
    ("vw", "vwap"),
    ("n", "trades"),
)


@dataclass(frozen=True)
class IntradayMeta:
    """Procedencia de un bloque de barras. ``adjustment`` es ``raw`` en el árbol principal."""

    source: str
    feed: str
    timeframe: str
    adjustment: str
    requested: Tuple[str, ...]
    received: Tuple[str, ...]

    @property
    def missing(self) -> Tuple[str, ...]:
        got = set(self.received)
        return tuple(symbol for symbol in self.requested if symbol not in got)


@dataclass(frozen=True)
class IntradayBars:
    """OHLCV más VWAP y trades, alineados al mismo índice UTC."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    vwap: pd.DataFrame
    trades: pd.DataFrame
    meta: IntradayMeta

    def frame(self, symbol: str) -> pd.DataFrame:
        """Una columna larga (o, h, l, c, v, vw, n) para ``symbol``."""
        out = pd.DataFrame(
            {
                "o": self.open[symbol],
                "h": self.high[symbol],
                "l": self.low[symbol],
                "c": self.close[symbol],
                "v": self.volume[symbol],
                "vw": self.vwap[symbol],
                "n": self.trades[symbol],
            },
            index=self.close.index,
        )
        return out.dropna(how="all")


class IntradayProvider(Protocol):
    """Fuente de barras. El feed vive en el proveedor, no en cada llamada."""

    def get_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> IntradayBars:
        """Barras con ``ts`` en ``[start, end)``. ``end`` es exclusivo."""


def empty_bars(
    symbols: Sequence[str],
    *,
    source: str,
    feed: str,
    timeframe: str,
    adjustment: str = "raw",
) -> IntradayBars:
    """Bloque vacío con las columnas pedidas y un índice UTC sin filas."""
    requested = unique_symbols(symbols)
    index = pd.DatetimeIndex([], tz=UTC, name="ts")
    frames = {
        name: pd.DataFrame(index=index, columns=requested, dtype=float)
        for _, name in _WIDE
    }
    meta = IntradayMeta(
        source=source,
        feed=feed,
        timeframe=timeframe,
        adjustment=adjustment,
        requested=requested,
        received=(),
    )
    return IntradayBars(meta=meta, **frames)


def bars_from_records(
    records: pd.DataFrame,
    symbols: Sequence[str],
    *,
    source: str,
    feed: str,
    timeframe: str,
    adjustment: str = "raw",
) -> IntradayBars:
    """Registro largo (symbol, ts, o, h, l, c, v, vw, n) → frames anchos.

    Si hay dos filas con el mismo (symbol, ts), gana la última.
    """
    requested = unique_symbols(symbols)
    if records.empty:
        return empty_bars(
            requested,
            source=source,
            feed=feed,
            timeframe=timeframe,
            adjustment=adjustment,
        )
    frame = records.copy()
    missing = [col for col in LONG_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"registros sin columnas {missing}")
    frame["ts"] = frame["ts"].map(ensure_utc_timestamp)
    frame = frame.drop_duplicates(["symbol", "ts"], keep="last")
    index = pd.DatetimeIndex(sorted(frame["ts"].unique()), name="ts")
    if index.tz is None:
        index = index.tz_localize(UTC)
    else:
        index = index.tz_convert(UTC)
    wide: dict[str, pd.DataFrame] = {}
    received: list[str] = []
    for column, name in _WIDE:
        pieces = {}
        for symbol in requested:
            part = frame.loc[frame["symbol"] == symbol, ["ts", column]]
            if part.empty or part[column].notna().sum() == 0:
                pieces[symbol] = pd.Series(np.nan, index=index, dtype=float)
                continue
            series = part.drop_duplicates("ts", keep="last").set_index("ts")[column]
            series.index = pd.DatetimeIndex(series.index, name="ts").tz_convert(UTC)
            pieces[symbol] = series.astype(float).reindex(index)
            if name == "close":
                received.append(symbol)
        wide[name] = pd.DataFrame(pieces, index=index).reindex(columns=requested)
    meta = IntradayMeta(
        source=source,
        feed=feed,
        timeframe=timeframe,
        adjustment=adjustment,
        requested=requested,
        received=tuple(received),
    )
    return IntradayBars(meta=meta, **wide)


def records_from_alpaca(
    by_symbol: Mapping[str, Sequence[Mapping]],
) -> pd.DataFrame:
    """Respuesta ``bars`` de Alpaca (símbolo → lista de objetos) → registro largo.

    Campos verificados en ``GET /v2/stocks/bars``: ``t``, ``o``, ``h``, ``l``,
    ``c``, ``v``, ``n``, ``vw``. ``t`` es el inicio del intervalo (RFC-3339).
    """
    rows = []
    for symbol, bars in by_symbol.items():
        if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
            raise ValueError(f"barras de {symbol} no son una lista")
        for bar in bars:
            rows.append(
                {
                    "symbol": str(symbol),
                    "ts": parse_timestamp(bar["t"]),
                    "o": float(bar["o"]),
                    "h": float(bar["h"]),
                    "l": float(bar["l"]),
                    "c": float(bar["c"]),
                    "v": float(bar["v"]),
                    "vw": float(bar["vw"]),
                    "n": float(bar["n"]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    return pd.DataFrame(rows, columns=list(LONG_COLUMNS))


def unique_symbols(symbols: Sequence[str]) -> Tuple[str, ...]:
    """Símbolos en orden, sin vacíos ni repeticiones. Mayúsculas."""
    out: list[str] = []
    seen = set()
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return tuple(out)


def parse_timestamp(value: object) -> pd.Timestamp:
    """RFC-3339 (con ``Z`` o desplazamiento) → Timestamp UTC."""
    if isinstance(value, pd.Timestamp):
        stamp = value
    elif isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        stamp = pd.Timestamp(text)
    if stamp.tzinfo is None:
        raise ValueError(f"timestamp sin zona: {value!r}")
    return stamp.tz_convert(UTC)


def ensure_utc_timestamp(value: object) -> pd.Timestamp:
    return parse_timestamp(value)


def as_utc(value: datetime) -> datetime:
    """datetime con zona → UTC. Rechaza naive (no se asume la zona del reloj)."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("se requiere un datetime con zona horaria")
    return value.astimezone(UTC)


def to_rfc3339(value: datetime) -> str:
    """UTC con segundos y sufijo ``Z``, como los ejemplos de Alpaca."""
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_timeframe(timeframe: str) -> str:
    """Forma larga de Alpaca (``1Min``, ``5Min``, ``1Day``).

    La doc admite también ``T``/``H``/``D``/``W``/``M``. Se guardan en la
    forma larga para que ``1T`` y ``1Min`` no partan el almacén en dos.
    """
    text = str(timeframe).strip()
    if not text:
        raise ValueError("timeframe vacío")
    number = ""
    i = 0
    while i < len(text) and text[i].isdigit():
        number += text[i]
        i += 1
    unit = text[i:]
    if not number or not unit:
        raise ValueError(f"timeframe no reconocido: {timeframe!r}")
    count = int(number)
    aliases = {
        "Min": "Min",
        "T": "Min",
        "Hour": "Hour",
        "H": "Hour",
        "Day": "Day",
        "D": "Day",
        "Week": "Week",
        "W": "Week",
        "Month": "Month",
        "M": "Month",
    }
    if unit not in aliases:
        raise ValueError(f"timeframe no reconocido: {timeframe!r}")
    long_unit = aliases[unit]
    limits = {
        "Min": range(1, 60),
        "Hour": range(1, 24),
        "Day": (1,),
        "Week": (1,),
        "Month": (1, 2, 3, 4, 6, 12),
    }
    if count not in limits[long_unit]:
        raise ValueError(f"timeframe fuera de rango: {timeframe!r}")
    return f"{count}{long_unit}"


def check_feed(feed: str) -> str:
    token = str(feed).strip().lower()
    if token not in FEEDS:
        raise ValueError(f"feed {feed!r} no está en {FEEDS}")
    return token


def check_adjustment(adjustment: str) -> str:
    """``raw``, ``split``, ``dividend``, ``spin-off``, ``all`` o una combinación.

    Alpaca documenta que se pueden unir con coma (``split,spin-off``).
    ``all`` y ``raw`` no se mezclan con otros tokens.
    """
    text = str(adjustment).strip().lower()
    if not text:
        raise ValueError("adjustment vacío")
    parts = tuple(part.strip() for part in text.split(","))
    allowed = {"raw", "split", "dividend", "spin-off", "all"}
    unknown = [part for part in parts if part not in allowed]
    if unknown:
        raise ValueError(f"adjustment desconocido: {unknown}")
    if ("raw" in parts or "all" in parts) and len(parts) > 1:
        raise ValueError("raw y all no se combinan con otros ajustes")
    return ",".join(parts)


def month_key(ts: pd.Timestamp) -> str:
    """``YYYY-MM`` del instante en UTC."""
    stamp = parse_timestamp(ts)
    return f"{stamp.year:04d}-{stamp.month:02d}"
