"""Descarga reanudable de barras hacia el ``BarStore``.

Solo pide los huecos que el manifiesto todavía no cubre. Una respuesta
vacía también cubre el intervalo: IEX puede no tener trades. El feed por
defecto es ``iex``.

Uso::

    python -m src.hft.data.download --symbols SPY,QQQ \\
        --start 2024-01-02 --end 2024-01-31 --timeframe 1Min --feed iex

Sin ``--symbols`` se usan los id de ``universe_hft.csv``. ``--start`` y
``--end`` aceptan una fecha (UTC, ``end`` exclusivo el día siguiente) o un
instante RFC-3339. El reloj de ``fetched_at`` es ``SystemClock``; los tests
inyectan otro.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Optional, Sequence

import pandas as pd

from src.hft.calendar import SystemClock
from src.hft.data.alpaca_data import (
    ADJUSTMENT_ACTION_TYPES,
    AlpacaDataClient,
    actions_fingerprint,
)
from src.hft.data.bar_store import BarStore, subtract_coverage
from src.hft.data.bars import (
    DEFAULT_FEED,
    canonical_timeframe,
    check_adjustment,
    check_feed,
    records_from_alpaca,
    unique_symbols,
)
from src.settings import load_settings, setup_logging

UTC = timezone.utc
Progress = Callable[[str], None]


@dataclass(frozen=True)
class DownloadResult:
    """Resumen de una pasada. ``calls`` son los GET de esta pasada."""

    calls: int
    rows: int
    requests: int
    symbols: tuple[str, ...]


def hft_symbols() -> tuple[str, ...]:
    """Ids de ``universe_hft.csv`` (raíz del repo, no el cwd)."""
    from src.instruments import load_instruments

    path = load_settings().universe_path("universe_hft.csv")
    master = load_instruments(path)
    return tuple(master)


def download_bars(
    client: AlpacaDataClient,
    store: BarStore,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    timeframe: str = "1Min",
    feed: str = DEFAULT_FEED,
    fetched_at: datetime,
    progress: Optional[Progress] = None,
) -> DownloadResult:
    """Baja los huecos de ``[start, end)`` y los hace upsert raw."""
    step = canonical_timeframe(timeframe)
    chosen = check_feed(feed)
    wanted = unique_symbols(symbols)
    if not wanted:
        raise ValueError("symbols vacío")
    groups = _group_gaps(store, wanted, start, end, timeframe=step, feed=chosen)
    calls_before = client.request_count
    rows = 0
    requests = 0
    report = progress if progress is not None else _stdout
    for gap_start, gap_end, names in groups:
        raw = client.get_bars(names, gap_start, gap_end, step, feed=chosen, adjustment="raw")
        records = records_from_alpaca(raw)
        if not records.empty:
            left = pd.Timestamp(gap_start)
            right = pd.Timestamp(gap_end)
            records = records[(records["ts"] >= left) & (records["ts"] < right)]
        store.upsert(
            records,
            timeframe=step,
            feed=chosen,
            fetched_start=gap_start,
            fetched_end=gap_end,
            fetched_at=fetched_at,
            symbols=names,
        )
        rows += len(records)
        requests += 1
        report(
            f"{','.join(names)} {gap_start.isoformat()} → {gap_end.isoformat()}: "
            f"{len(records)} barras, llamadas={client.request_count - calls_before}"
        )
    return DownloadResult(
        calls=client.request_count - calls_before,
        rows=rows,
        requests=requests,
        symbols=wanted,
    )


def sync_adjusted_daily(
    client: AlpacaDataClient,
    store: BarStore,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    feed: str = DEFAULT_FEED,
    adjustment: str = "split",
    fetched_at: datetime,
) -> tuple[str, ...]:
    """Barras diarias ajustadas. Solo se vuelven a bajar si cambia la huella.

    Devuelve los símbolos que se refrescaron. El árbol es distinto del raw:
    las features intradía no deben leerlo como si fuera la sesión en curso.
    """
    step = "1Day"
    chosen = check_feed(feed)
    adjust = check_adjustment(adjustment)
    if adjust not in ADJUSTMENT_ACTION_TYPES:
        raise ValueError(
            f"adjustment {adjust!r} no tiene tipos de acción asociados; usar 'split' o 'all'"
        )
    wanted = unique_symbols(symbols)
    types = ADJUSTMENT_ACTION_TYPES[adjust]
    actions = client.get_corporate_actions(
        wanted,
        start.astimezone(UTC).date(),
        (end - timedelta(microseconds=1)).astimezone(UTC).date(),
        types=types,
    )
    refreshed: list[str] = []
    for symbol in wanted:
        own = [row for row in actions if str(row.get("symbol", "")).upper() == symbol]
        fingerprint = actions_fingerprint(own)
        if not store.adjusted_stale(
            symbol,
            timeframe=step,
            feed=chosen,
            adjustment=adjust,
            actions_sha256=fingerprint,
            start=start,
            end=end,
        ):
            continue
        raw = client.get_bars(
            [symbol],
            start,
            end,
            step,
            feed=chosen,
            adjustment=adjust,
        )
        records = records_from_alpaca(raw)
        store.replace_adjusted(
            records,
            timeframe=step,
            feed=chosen,
            adjustment=adjust,
            symbols=[symbol],
            fetched_start=start,
            fetched_end=end,
            fetched_at=fetched_at,
            actions_sha256=fingerprint,
        )
        refreshed.append(symbol)
    return tuple(refreshed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga barras de Alpaca al BarStore (reanudable, feed iex por defecto).",
    )
    parser.add_argument("--symbols", default="", help="Lista separada por comas. Default: universe_hft.csv.")
    parser.add_argument("--start", required=True, help="Inicio inclusivo (YYYY-MM-DD o RFC-3339).")
    parser.add_argument("--end", required=True, help="Fin exclusivo si es fecha (el día siguiente).")
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--feed", default=DEFAULT_FEED, choices=("iex", "sip", "boats", "otc"))
    parser.add_argument("--adjusted", default="", help="Si es split o all, refresca el diario ajustado.")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    client: Optional[AlpacaDataClient] = None,
    store: Optional[BarStore] = None,
    clock=None,
) -> int:
    """Entrypoint. ``setup_logging`` solo aquí, no al importar."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    start, end = parse_bounds(args.start, args.end)
    symbols = unique_symbols(args.symbols.split(",")) if args.symbols.strip() else hft_symbols()
    now = (SystemClock() if clock is None else clock).now()
    data_client = client if client is not None else _client_from_env(args.feed)
    bar_store = store if store is not None else BarStore()
    result = download_bars(
        data_client,
        bar_store,
        symbols,
        start,
        end,
        timeframe=args.timeframe,
        feed=args.feed,
        fetched_at=now,
    )
    print(
        f"listo: {result.rows} barras, {result.requests} peticiones, "
        f"{result.calls} llamadas HTTP, feed={args.feed}"
    )
    if args.adjusted:
        refreshed = sync_adjusted_daily(
            data_client,
            bar_store,
            symbols,
            start,
            end,
            feed=args.feed,
            adjustment=args.adjusted,
            fetched_at=now,
        )
        print(f"ajustado {args.adjusted}: {len(refreshed)} símbolos refrescados")
    return 0


def parse_bounds(start: str, end: str) -> tuple[datetime, datetime]:
    """Fecha pelada: start a las 00:00 UTC y end exclusivo el día siguiente."""
    return _parse_bound(start, exclusive_date=False), _parse_bound(end, exclusive_date=True)


def _parse_bound(text: str, *, exclusive_date: bool) -> datetime:
    raw = text.strip()
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        day = datetime.combine(datetime.fromisoformat(raw).date(), time(0, 0), tzinfo=UTC)
        if exclusive_date:
            return day + timedelta(days=1)
        return day
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    stamp = datetime.fromisoformat(raw)
    if stamp.tzinfo is None:
        raise ValueError(f"instante sin zona: {text!r}")
    parsed = stamp.astimezone(UTC)
    if exclusive_date:
        return parsed
    return parsed


def _group_gaps(
    store: BarStore,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    timeframe: str,
    feed: str,
) -> list[tuple[datetime, datetime, tuple[str, ...]]]:
    """Agrupa símbolos que comparten el mismo hueco, para un solo GET."""
    buckets: dict[tuple[datetime, datetime], list[str]] = {}
    for symbol in symbols:
        covered = store.coverage(symbol, timeframe=timeframe, feed=feed)
        for gap_start, gap_end in subtract_coverage(start, end, covered):
            buckets.setdefault((gap_start, gap_end), []).append(symbol)
    grouped = [
        (gap_start, gap_end, tuple(names))
        for (gap_start, gap_end), names in sorted(buckets.items(), key=lambda item: item[0])
    ]
    return grouped


def _client_from_env(feed: str) -> AlpacaDataClient:
    from src.brokers.alpaca_adapter import load_broker_config

    return AlpacaDataClient(load_broker_config(), feed=feed)


def _stdout(message: str) -> None:
    print(message, file=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
