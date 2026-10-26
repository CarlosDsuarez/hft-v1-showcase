"""Calidad de barras intradía, por símbolo y sesión.

Compara las barras recibidas con ``expected_bars`` del calendario.

- ``sin_trades``: el minuto cae dentro de un intervalo ya descargado y no
  hay fila. En IEX eso es un minuto sin trades, no un agujero del archivo.
- ``descarga_faltante``: el minuto no está en la cobertura del manifiesto
  (nadie pidió ese tramo, o la petición no terminó).
- ``duplicado``: dos filas con el mismo símbolo y la misma ``ts`` en el
  registro largo, antes del upsert que se queda con la última.
- ``ohlc_invalido``: ``l > min(o, c)`` o ``h < max(o, c)``. No se borra.
- ``salto``: ``|r| > k · σ`` con σ calculada solo con retornos anteriores.
  Se marca y no se borra: un salto puede ser un dato real.

La política es ``error`` o ``warn`` por familia, al estilo de
``market_data.DataPolicy``. ``error`` levanta ``QualityError`` con el
reporte completo; ``warn`` lo deja en el log. El reporte no modifica las
barras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.hft.calendar import Session, expected_bars
from src.hft.data.bars import IntradayBars, as_utc, parse_timestamp

logger = logging.getLogger("hft.data.quality")

POLICIES = ("error", "warn")
SIN_TRADES = "sin_trades"
MISSING_DOWNLOAD = "descarga_faltante"
DUPLICATE = "duplicado"
INVALID_OHLC = "ohlc_invalido"
JUMP = "salto"

_KIND_FIELD = {
    SIN_TRADES: "on_gap",
    MISSING_DOWNLOAD: "on_gap",
    DUPLICATE: "on_duplicate",
    INVALID_OHLC: "on_invalid",
    JUMP: "on_jump",
}


@dataclass(frozen=True)
class QualityPolicy:
    """Qué hacer con cada familia de hallazgos."""

    on_gap: str = "warn"
    on_duplicate: str = "warn"
    on_invalid: str = "error"
    on_jump: str = "warn"

    def __post_init__(self) -> None:
        for name in ("on_gap", "on_duplicate", "on_invalid", "on_jump"):
            value = getattr(self, name)
            if value not in POLICIES:
                raise ValueError(f"{name}={value!r} no está en {POLICIES}")

    def action(self, kind: str) -> str:
        return getattr(self, _KIND_FIELD[kind])


@dataclass(frozen=True)
class QualityIssue:
    symbol: str
    kind: str
    ts: Optional[pd.Timestamp] = None
    session: Optional[date] = None
    detail: str = ""


@dataclass
class QualityReport:
    issues: tuple[QualityIssue, ...] = ()
    policy: QualityPolicy = field(default_factory=QualityPolicy)

    def of_kind(self, kind: str) -> tuple[QualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.kind == kind)

    def summary(self) -> str:
        if not self.issues:
            return "calidad: sin hallazgos"
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
        detail = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        return f"calidad: {len(self.issues)} hallazgos ({detail})"


class QualityError(ValueError):
    """La política pidió ``error`` y hay hallazgos de esa familia."""

    def __init__(self, report: QualityReport):
        self.report = report
        super().__init__(report.summary())


def duplicate_timestamps(records: pd.DataFrame) -> tuple[tuple[str, pd.Timestamp], ...]:
    """Pares (símbolo, ts) que aparecen más de una vez."""
    if records.empty or "symbol" not in records.columns or "ts" not in records.columns:
        return ()
    frame = records.loc[:, ["symbol", "ts"]].copy()
    frame["ts"] = frame["ts"].map(parse_timestamp)
    mask = frame.duplicated(["symbol", "ts"], keep=False)
    pairs = {
        (str(row.symbol), row.ts)
        for row in frame.loc[mask].itertuples(index=False)
    }
    return tuple(sorted(pairs, key=lambda item: (item[0], item[1])))


def quality_report(
    bars: IntradayBars,
    sessions: Sequence[Session],
    *,
    coverage: Optional[Mapping[str, Sequence[tuple[datetime, datetime]]]] = None,
    records: Optional[pd.DataFrame] = None,
    policy: Optional[QualityPolicy] = None,
    jump_k: float = 8.0,
    min_obs: int = 20,
) -> QualityReport:
    """Arma el reporte y aplica la política. No altera ``bars``."""
    chosen = policy if policy is not None else QualityPolicy()
    if jump_k <= 0:
        raise ValueError("jump_k debe ser positivo")
    if min_obs < 2:
        raise ValueError("min_obs debe ser >= 2 (hace falta una desviación)")
    issues: list[QualityIssue] = []
    issues.extend(_gap_issues(bars, sessions, coverage))
    issues.extend(_duplicate_issues(records))
    issues.extend(_ohlc_issues(bars))
    issues.extend(_jump_issues(bars, jump_k=jump_k, min_obs=min_obs))
    report = QualityReport(issues=tuple(issues), policy=chosen)
    blocked = [issue for issue in issues if chosen.action(issue.kind) == "error"]
    if blocked:
        raise QualityError(report)
    if issues:
        logger.warning(report.summary())
    return report


def _gap_issues(
    bars: IntradayBars,
    sessions: Sequence[Session],
    coverage: Optional[Mapping[str, Sequence[tuple[datetime, datetime]]]],
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    step = bars.meta.timeframe
    for symbol in bars.meta.requested:
        present = _present_index(bars, symbol)
        intervals = () if coverage is None else tuple(coverage.get(symbol, ()))
        for session in sessions:
            expected = expected_bars(session, step)
            if len(expected) == 0:
                continue
            missing = expected.difference(present)
            for ts in missing:
                stamp = parse_timestamp(ts)
                kind = SIN_TRADES if _inside(stamp, intervals) else MISSING_DOWNLOAD
                label = (
                    "minuto sin trades dentro de una descarga"
                    if kind == SIN_TRADES
                    else "minuto fuera de la cobertura descargada"
                )
                issues.append(
                    QualityIssue(
                        symbol=symbol,
                        kind=kind,
                        ts=stamp,
                        session=session.date,
                        detail=label,
                    )
                )
    return issues


def _present_index(bars: IntradayBars, symbol: str) -> pd.DatetimeIndex:
    if symbol not in bars.close.columns:
        return pd.DatetimeIndex([], tz="UTC")
    series = bars.close[symbol].dropna()
    index = pd.DatetimeIndex(series.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index


def _inside(ts: pd.Timestamp, intervals: Sequence[tuple[datetime, datetime]]) -> bool:
    instant = ts.to_pydatetime()
    for start, end in intervals:
        if as_utc(start) <= instant < as_utc(end):
            return True
    return False


def _duplicate_issues(records: Optional[pd.DataFrame]) -> list[QualityIssue]:
    if records is None:
        return []
    return [
        QualityIssue(symbol=symbol, kind=DUPLICATE, ts=ts, detail="ts repetida; la última versión gana")
        for symbol, ts in duplicate_timestamps(records)
    ]


def _ohlc_issues(bars: IntradayBars) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for symbol in bars.meta.requested:
        if symbol not in bars.close.columns:
            continue
        frame = bars.frame(symbol)
        for ts, row in frame.iterrows():
            o, h, l, c = (row["o"], row["h"], row["l"], row["c"])
            if any(pd.isna(value) for value in (o, h, l, c)):
                continue
            if l > min(o, c) or h < max(o, c):
                issues.append(
                    QualityIssue(
                        symbol=symbol,
                        kind=INVALID_OHLC,
                        ts=parse_timestamp(ts),
                        detail=f"o={o} h={h} l={l} c={c}",
                    )
                )
    return issues


def _jump_issues(bars: IntradayBars, *, jump_k: float, min_obs: int) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for symbol in bars.meta.requested:
        if symbol not in bars.close.columns:
            continue
        close = bars.close[symbol].dropna().astype(float)
        if len(close) < min_obs + 1:
            continue
        values = close.to_numpy()
        stamps = list(close.index)
        log_price = np.log(values)
        returns = np.diff(log_price)
        for i in range(min_obs, len(returns)):
            past = returns[:i]
            sigma = float(np.std(past, ddof=1))
            if not np.isfinite(sigma) or sigma == 0.0:
                continue
            move = float(returns[i])
            if abs(move) > jump_k * sigma:
                issues.append(
                    QualityIssue(
                        symbol=symbol,
                        kind=JUMP,
                        ts=parse_timestamp(stamps[i + 1]),
                        detail=f"|r|={abs(move):.6g} > {jump_k:g}·σ ({sigma:.6g})",
                    )
                )
    return issues
