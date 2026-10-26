"""Almacén de barras en CSV.gz (sin parquet).

Layout de las barras ``raw`` (el que lee el backtest por defecto)::

    <root>/<timeframe>/<feed>/<SYMBOL>/<YYYY-MM>.csv.gz

Columnas: ``ts`` (UTC, ISO-8601), ``o``, ``h``, ``l``, ``c``, ``v``, ``vw``,
``n``. El upsert es idempotente: misma ``ts``, gana la última versión.
Escritura atómica (tmp + ``os.replace``). ``manifest.json`` guarda sha256,
filas, rango de datos y ``fetched_at`` por archivo, más la cobertura
pedida (intervalos semiabiertos ``[start, end)``) para poder reanudar.

Las barras diarias ajustadas no entran en ese árbol. Van a::

    <root>/_adjusted/<adjustment>/<timeframe>/<feed>/<SYMBOL>/<YYYY-MM>.csv.gz

con la huella SHA-256 de las acciones corporativas y la fecha de descarga.
``read`` solo devuelve raw. Mezclar los dos niveles en una ventana rompe
stops y rangos: el día en curso coincide en raw y ajustado, pero el
histórico no.

``root`` por defecto es ``settings.bars_dir`` (``<data_dir>/bars``),
resuelto al construir el store (respeta ``HFT_HOME`` en ese momento).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from src.hft.data.bars import (
    DEFAULT_FEED,
    LONG_COLUMNS,
    IntradayBars,
    as_utc,
    bars_from_records,
    canonical_timeframe,
    check_adjustment,
    check_feed,
    month_key,
    parse_timestamp,
    to_rfc3339,
    unique_symbols,
)

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
_FILE_COLUMNS = ("ts", "o", "h", "l", "c", "v", "vw", "n")
_MANIFEST = "manifest.json"
_SCHEMA = 1


@dataclass(frozen=True)
class FileEntry:
    """Una pieza del manifiesto."""

    sha256: str
    rows: int
    start: Optional[str]
    end: Optional[str]
    fetched_at: str


def _symbol_or_raise(symbol: str) -> str:
    token = str(symbol).strip().upper()
    if not _SYMBOL.fullmatch(token):
        raise ValueError(f"símbolo no válido para una ruta: {symbol!r}")
    return token


def _merge_half_open(
    intervals: Sequence[tuple[datetime, datetime]],
) -> tuple[tuple[datetime, datetime], ...]:
    """Une intervalos ``[start, end)`` que se tocan o se solapan."""
    ordered = sorted((as_utc(start), as_utc(end)) for start, end in intervals if end > start)
    if not ordered:
        return ()
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def subtract_coverage(
    start: datetime,
    end: datetime,
    covered: Sequence[tuple[datetime, datetime]],
) -> tuple[tuple[datetime, datetime], ...]:
    """Huecos de ``[start, end)`` que ninguna cobertura incluye."""
    cursor = as_utc(start)
    limit = as_utc(end)
    if limit <= cursor:
        return ()
    gaps: list[tuple[datetime, datetime]] = []
    for left, right in _merge_half_open(covered):
        if right <= cursor:
            continue
        if left >= limit:
            break
        if left > cursor:
            gaps.append((cursor, min(left, limit)))
        cursor = max(cursor, right)
        if cursor >= limit:
            return tuple(gaps)
    if cursor < limit:
        gaps.append((cursor, limit))
    return tuple(gaps)


class BarStore:
    """CSV.gz por símbolo y mes, más manifiesto."""

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            from src.settings import load_settings

            root = load_settings().bars_dir
        self.root = Path(root)

    def upsert(
        self,
        records: pd.DataFrame,
        *,
        timeframe: str,
        feed: str = DEFAULT_FEED,
        fetched_start: datetime,
        fetched_end: datetime,
        fetched_at: datetime,
        symbols: Optional[Sequence[str]] = None,
    ) -> None:
        """Inserta barras ``raw``. La última fila de cada ``ts`` gana.

        Aunque ``records`` venga vacío, ``[fetched_start, fetched_end)``
        queda cubierto: una respuesta vacía de IEX también es una descarga.
        """
        step = canonical_timeframe(timeframe)
        chosen = check_feed(feed)
        start = as_utc(fetched_start)
        end = as_utc(fetched_end)
        if end <= start:
            raise ValueError("fetched_end debe ser posterior a fetched_start")
        fetched = to_rfc3339(as_utc(fetched_at))
        frame = _as_records(records)
        if symbols is None:
            names = unique_symbols(frame["symbol"].tolist()) if not frame.empty else ()
        else:
            names = tuple(_symbol_or_raise(symbol) for symbol in unique_symbols(symbols))
        manifest = _read_manifest(self._manifest_path())
        for symbol in names:
            part = frame.loc[frame["symbol"] == symbol] if not frame.empty else frame
            months = sorted({month_key(ts) for ts in part["ts"]}) if not part.empty else []
            for month in months:
                chunk = part.loc[part["ts"].map(month_key) == month]
                path = self._raw_path(step, chosen, symbol, month)
                merged = _merge_frames(_read_month(path), chunk)
                _write_month(path, merged)
                _put_file(
                    manifest,
                    path,
                    self.root,
                    rows=len(merged),
                    start=_iso_or_none(merged["ts"].iloc[0] if len(merged) else None),
                    end=_iso_or_none(merged["ts"].iloc[-1] if len(merged) else None),
                    fetched_at=fetched,
                )
            _add_coverage(manifest, _coverage_key(step, chosen, symbol), start, end)
        _write_manifest(self._manifest_path(), manifest)

    def read(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
        feed: str = DEFAULT_FEED,
    ) -> IntradayBars:
        """Barras raw con ``ts`` en ``[start, end)``."""
        step = canonical_timeframe(timeframe)
        chosen = check_feed(feed)
        wanted = tuple(_symbol_or_raise(symbol) for symbol in unique_symbols(symbols))
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        if end_utc <= start_utc:
            raise ValueError("end debe ser posterior a start")
        pieces = []
        for symbol in wanted:
            for path in self._months_touching(step, chosen, symbol, start_utc, end_utc, adjusted=None):
                month = _read_month(path)
                if month.empty:
                    continue
                mask = (month["ts"] >= pd.Timestamp(start_utc)) & (month["ts"] < pd.Timestamp(end_utc))
                chunk = month.loc[mask].copy()
                if chunk.empty:
                    continue
                chunk.insert(0, "symbol", symbol)
                pieces.append(chunk)
        if pieces:
            records = pd.concat(pieces, ignore_index=True)
        else:
            records = pd.DataFrame(columns=list(LONG_COLUMNS))
        return bars_from_records(
            records,
            wanted,
            source=f"store:{chosen}",
            feed=chosen,
            timeframe=step,
            adjustment="raw",
        )

    def coverage(
        self,
        symbol: str,
        *,
        timeframe: str = "1Min",
        feed: str = DEFAULT_FEED,
    ) -> tuple[tuple[datetime, datetime], ...]:
        """Intervalos ``[start, end)`` ya pedidos para este símbolo raw."""
        key = _coverage_key(canonical_timeframe(timeframe), check_feed(feed), _symbol_or_raise(symbol))
        manifest = _read_manifest(self._manifest_path())
        return _intervals(manifest.get("coverage", {}).get(key, ()))

    def file_entry(self, relative: str) -> Optional[FileEntry]:
        manifest = _read_manifest(self._manifest_path())
        raw = manifest.get("files", {}).get(relative)
        if not isinstance(raw, dict):
            return None
        return FileEntry(
            sha256=str(raw["sha256"]),
            rows=int(raw["rows"]),
            start=raw.get("start"),
            end=raw.get("end"),
            fetched_at=str(raw["fetched_at"]),
        )

    def replace_adjusted(
        self,
        records: pd.DataFrame,
        *,
        timeframe: str,
        feed: str,
        adjustment: str,
        symbols: Sequence[str],
        fetched_start: datetime,
        fetched_end: datetime,
        fetched_at: datetime,
        actions_sha256: str,
    ) -> None:
        """Sustituye el cache ajustado de ``symbols`` en el rango y guarda la huella."""
        step = canonical_timeframe(timeframe)
        chosen = check_feed(feed)
        adjust = check_adjustment(adjustment)
        if adjust == "raw":
            raise ValueError("el árbol ajustado no guarda raw; usar upsert")
        start = as_utc(fetched_start)
        end = as_utc(fetched_end)
        frame = _as_records(records)
        names = tuple(_symbol_or_raise(symbol) for symbol in unique_symbols(symbols))
        manifest = _read_manifest(self._manifest_path())
        for symbol in names:
            self._drop_adjusted_months(manifest, adjust, step, chosen, symbol, start, end)
            part = frame.loc[frame["symbol"] == symbol] if not frame.empty else frame
            months = sorted({month_key(ts) for ts in part["ts"]}) if not part.empty else []
            for month in months:
                chunk = part.loc[part["ts"].map(month_key) == month]
                path = self._adjusted_path(adjust, step, chosen, symbol, month)
                # El cache ajustado se reemplaza: una acción nueva cambia toda la serie.
                merged = _merge_frames(pd.DataFrame(columns=["ts", *_FILE_COLUMNS[1:]]), chunk)
                _write_month(path, merged)
                _put_file(
                    manifest,
                    path,
                    self.root,
                    rows=len(merged),
                    start=_iso_or_none(merged["ts"].iloc[0] if len(merged) else None),
                    end=_iso_or_none(merged["ts"].iloc[-1] if len(merged) else None),
                    fetched_at=to_rfc3339(as_utc(fetched_at)),
                )
            manifest.setdefault("adjusted", {})[_adjusted_key(step, chosen, adjust, symbol)] = {
                "actions_sha256": actions_sha256,
                "fetched_at": to_rfc3339(as_utc(fetched_at)),
                "start": to_rfc3339(start),
                "end": to_rfc3339(end),
            }
        _write_manifest(self._manifest_path(), manifest)

    def adjusted_stale(
        self,
        symbol: str,
        *,
        timeframe: str,
        feed: str,
        adjustment: str,
        actions_sha256: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """True si hay que volver a bajar: otra huella, u otro rango."""
        key = _adjusted_key(
            canonical_timeframe(timeframe),
            check_feed(feed),
            check_adjustment(adjustment),
            _symbol_or_raise(symbol),
        )
        manifest = _read_manifest(self._manifest_path())
        meta = manifest.get("adjusted", {}).get(key)
        if not isinstance(meta, dict):
            return True
        if meta.get("actions_sha256") != actions_sha256:
            return True
        try:
            have_start = parse_timestamp(meta["start"]).to_pydatetime()
            have_end = parse_timestamp(meta["end"]).to_pydatetime()
        except (KeyError, TypeError, ValueError):
            return True
        return as_utc(start) < have_start or as_utc(end) > have_end

    def read_adjusted(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        timeframe: str,
        feed: str = DEFAULT_FEED,
        adjustment: str = "split",
    ) -> IntradayBars:
        step = canonical_timeframe(timeframe)
        chosen = check_feed(feed)
        adjust = check_adjustment(adjustment)
        wanted = tuple(_symbol_or_raise(symbol) for symbol in unique_symbols(symbols))
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        pieces = []
        for symbol in wanted:
            for path in self._months_touching(
                step, chosen, symbol, start_utc, end_utc, adjusted=adjust
            ):
                month = _read_month(path)
                if month.empty:
                    continue
                mask = (month["ts"] >= pd.Timestamp(start_utc)) & (month["ts"] < pd.Timestamp(end_utc))
                chunk = month.loc[mask].copy()
                if chunk.empty:
                    continue
                chunk.insert(0, "symbol", symbol)
                pieces.append(chunk)
        records = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(LONG_COLUMNS))
        return bars_from_records(
            records,
            wanted,
            source=f"store:{chosen}:{adjust}",
            feed=chosen,
            timeframe=step,
            adjustment=adjust,
        )

    def _drop_adjusted_months(
        self,
        manifest: dict,
        adjustment: str,
        timeframe: str,
        feed: str,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Borra los meses ajustados del rango: la serie nueva los sustituye."""
        for path in self._months_touching(timeframe, feed, symbol, start, end, adjusted=adjustment):
            key = _relative(path, self.root)
            manifest.get("files", {}).pop(key, None)
            path.unlink()

    def _raw_path(self, timeframe: str, feed: str, symbol: str, month: str) -> Path:
        return self.root / timeframe / feed / symbol / f"{month}.csv.gz"

    def _adjusted_path(
        self,
        adjustment: str,
        timeframe: str,
        feed: str,
        symbol: str,
        month: str,
    ) -> Path:
        # La coma de un adjustment compuesto no puede ir en un directorio.
        folder = adjustment.replace(",", "+")
        return self.root / "_adjusted" / folder / timeframe / feed / symbol / f"{month}.csv.gz"

    def _months_touching(
        self,
        timeframe: str,
        feed: str,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: Optional[str],
    ) -> list[Path]:
        if adjusted is None:
            folder = self.root / timeframe / feed / symbol
        else:
            folder = self.root / "_adjusted" / adjusted.replace(",", "+") / timeframe / feed / symbol
        if not folder.is_dir():
            return []
        start_month = month_key(pd.Timestamp(start))
        # end es exclusivo: el mes del instante anterior.
        end_month = month_key(pd.Timestamp(end - timedelta(microseconds=1)))
        found = []
        for path in sorted(folder.glob("*.csv.gz")):
            # ``2024-01.csv.gz``: stem deja el ``.csv``.
            month = path.name.removesuffix(".csv.gz")
            if start_month <= month <= end_month:
                found.append(path)
        return found

    def _manifest_path(self) -> Path:
        return self.root / _MANIFEST


class StoreProvider:
    """``IntradayProvider`` que solo lee el almacén (sin red)."""

    name = "store"

    def __init__(
        self,
        store: BarStore,
        *,
        feed: str = DEFAULT_FEED,
        timeframe: str = "1Min",
    ):
        self.store = store
        self.feed = check_feed(feed)
        self.timeframe = canonical_timeframe(timeframe)

    def get_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: Optional[str] = None,
    ) -> IntradayBars:
        step = self.timeframe if timeframe is None else canonical_timeframe(timeframe)
        return self.store.read(symbols, start, end, timeframe=step, feed=self.feed)


def _as_records(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    missing = [col for col in LONG_COLUMNS if col not in records.columns]
    if missing:
        raise ValueError(f"registros sin columnas {missing}")
    frame = records.loc[:, list(LONG_COLUMNS)].copy()
    frame["symbol"] = frame["symbol"].map(_symbol_or_raise)
    frame["ts"] = frame["ts"].map(parse_timestamp)
    for column in ("o", "h", "l", "c", "v", "vw", "n"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    columns = ["ts", *_FILE_COLUMNS[1:]]
    parts = []
    if old is not None and not old.empty:
        parts.append(old.loc[:, columns])
    if new is not None and not new.empty:
        parts.append(new.loc[:, columns])
    if not parts:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(parts, ignore_index=True)
    frame["ts"] = frame["ts"].map(parse_timestamp)
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts")
    return frame.reset_index(drop=True)


def _read_month(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["ts", *_FILE_COLUMNS[1:]])
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle)
    if list(frame.columns) != list(_FILE_COLUMNS):
        raise ValueError(f"columnas inesperadas en {path}")
    if frame.empty:
        return pd.DataFrame(columns=["ts", *_FILE_COLUMNS[1:]])
    frame["ts"] = frame["ts"].map(parse_timestamp)
    return frame


def _write_month(path: Path, frame: pd.DataFrame) -> None:
    buffer = [",".join(_FILE_COLUMNS)]
    for ts, o, h, l, c, v, vw, n in frame.itertuples(index=False, name=None):
        text_ts = to_rfc3339(parse_timestamp(ts).to_pydatetime())
        values = [text_ts, _format_number(o), _format_number(h), _format_number(l),
                  _format_number(c), _format_number(v), _format_number(vw), _format_number(n)]
        buffer.append(",".join(values))
    text = "\n".join(buffer) + "\n"
    _atomic_gzip(path, text.encode("utf-8"))


def _format_number(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _atomic_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _put_file(
    manifest: dict,
    path: Path,
    root: Path,
    *,
    rows: int,
    start: Optional[str],
    end: Optional[str],
    fetched_at: str,
) -> None:
    manifest.setdefault("files", {})[_relative(path, root)] = {
        "sha256": _sha256(path),
        "rows": int(rows),
        "start": start,
        "end": end,
        "fetched_at": fetched_at,
    }


def _iso_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return to_rfc3339(value.to_pydatetime())
    if isinstance(value, datetime):
        return to_rfc3339(value)
    return None


def _coverage_key(timeframe: str, feed: str, symbol: str) -> str:
    return f"{timeframe}|{feed}|{symbol}"


def _adjusted_key(timeframe: str, feed: str, adjustment: str, symbol: str) -> str:
    return f"{timeframe}|{feed}|{adjustment}|{symbol}"


def _add_coverage(manifest: dict, key: str, start: datetime, end: datetime) -> None:
    current = _intervals(manifest.setdefault("coverage", {}).get(key, ()))
    merged = _merge_half_open((*current, (start, end)))
    manifest["coverage"][key] = [
        [to_rfc3339(left), to_rfc3339(right)] for left, right in merged
    ]


def _intervals(raw: object) -> tuple[tuple[datetime, datetime], ...]:
    if not isinstance(raw, list):
        return ()
    found = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        found.append((parse_timestamp(item[0]).to_pydatetime(), parse_timestamp(item[1]).to_pydatetime()))
    return _merge_half_open(found)


def _empty_manifest() -> dict:
    return {"schema": _SCHEMA, "files": {}, "coverage": {}, "adjusted": {}}


def _read_manifest(path: Path) -> dict:
    if not path.is_file():
        return _empty_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"manifiesto con esquema inesperado: {path}")
    payload.setdefault("files", {})
    payload.setdefault("coverage", {})
    payload.setdefault("adjusted", {})
    return payload


def _write_manifest(path: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

