"""Reporte intradía y reconciliación backtest contra paper (HFT-13).

El HTML es un solo archivo: plantilla de texto, CSS en línea y SVG en
línea. No usa plotly ni ninguna dependencia fuera de stdlib, numpy y
pandas.

La reconciliación vuelve a correr ``src.hft.backtest.run`` sobre las
barras de la sesión (el mismo feed). Compara las intenciones 1:1 y, si
no coinciden, deja la causa:

- ``barra_corregida_tardiamente``: el OHLC que vio el paper no es el
  del histórico (Alpaca puede republicar la barra; el vivo ya decidió);
- ``rechazo_de_riesgo``: un lado la filtró y el otro no;
- ``dato_faltante``: la barra de la señal no está en los dos paneles.

El implementation shortfall de cada fill, en bps, es
``lado × (precio_fill − precio_decisión) / precio_decisión``. El lado
compra suma y el lado venta resta, así que un número positivo es peor
que el precio de la decisión. El slippage real se compara con el modelo
de ``src.hft.costs``. Paper no cobra las tarifas regulatorias:
``paper_valuation_fees`` es el importe sintético que se resta.

Si en 20 sesiones o más el slippage medio real supera al modelado en
más de dos errores estándar, el reporte propone recalibrar. Cambiar
parámetros es una decisión humana y queda en HFT-14: aquí no se toca
la config.
"""

from __future__ import annotations

import argparse
import html
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.hft.costs import CostModel, paper_valuation_fees
from src.hft.data.bars import IntradayBars
from src.hft.data.quality import QualityReport
from src.hft.types import OrderIntent, RiskEvent

CAUSE_LATE_BAR = "barra_corregida_tardiamente"
CAUSE_RISK = "rechazo_de_riesgo"
CAUSE_MISSING = "dato_faltante"
CAUSE_UNEXPLAINED = "sin_explicar"

CAUSES = (CAUSE_LATE_BAR, CAUSE_RISK, CAUSE_MISSING, CAUSE_UNEXPLAINED)

# Debajo de este número de sesiones no hay alerta: el error estándar
# de una muestra más corta no sostiene el umbral.
MIN_SLIPPAGE_SESSIONS = 20
SLIPPAGE_SE_MULT = 2.0

_OHLC = ("open", "high", "low", "close")
_PRICE_EPS = 1e-8
_QTY_EPS = 1e-8


@dataclass(frozen=True)
class PaperFill:
    """Fill de paper. ``fees_charged`` es lo que debitó el broker (casi siempre 0)."""

    intent_id: str
    side: str
    symbol: str
    qty: float
    fill_price: float
    decision_price: float
    fees_charged: float = 0.0
    liquidity: str = "taker"
    asset_class: str = "us_equity"
    # Precio contra el que se mide el slippage del modelo. Si falta, se
    # usa el precio de decisión (el cierre de la barra de la señal).
    reference_price: Optional[float] = None
    adv: Optional[float] = None
    sigma: Optional[float] = None


@dataclass(frozen=True)
class SessionSlippage:
    """Medias de una sesión, en bps. Entran al umbral de 20 sesiones."""

    session_date: str
    real_bps: float
    modeled_bps: float


@dataclass(frozen=True)
class IntentDiff:
    """Una intención que no está en los dos lados con la misma firma."""

    intent_id: str
    where: str
    cause: str
    detail: str
    symbol: str


@dataclass(frozen=True)
class ShortfallRow:
    """Una operación de paper contra el precio de decisión y el modelo."""

    intent_id: str
    symbol: str
    side: str
    qty: float
    decision_price: float
    fill_price: float
    shortfall_bps: float
    real_slippage_bps: float
    modeled_slippage_bps: float
    synthetic_fees: float
    paper_fees: float


@dataclass(frozen=True)
class Reconciliation:
    """Paridad de intenciones, shortfall y la alerta de recalibración."""

    parity: float
    n_paper: int
    n_backtest: int
    n_matched: int
    diffs: Tuple[IntentDiff, ...]
    shortfalls: Tuple[ShortfallRow, ...]
    alert: Optional[str]
    feed: str
    session_date: date

    @property
    def parity_pct(self) -> float:
        return 100.0 * float(self.parity)


@dataclass
class ReportView:
    """Lo que pinta el HTML. Los frames ya vienen agregados."""

    title: str
    start: date
    end: date
    feed: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    by_strategy: pd.DataFrame = field(default_factory=pd.DataFrame)
    blotter: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_events: Sequence[Mapping[str, object]] = ()
    quality_rows: Sequence[Mapping[str, object]] = ()
    cost_rows: Sequence[Mapping[str, object]] = ()
    equity: Sequence[float] = ()
    drawdown: Sequence[float] = ()
    equity_labels: Sequence[str] = ()
    pnl_by_hour: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: Optional[Reconciliation] = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    note: str = ""


def implementation_shortfall_bps(side: str, fill_price: float, decision_price: float) -> float:
    """``lado × (fill − decisión) / decisión``, en bps.

    Compra: el lado es +1. Venta: el lado es −1. Positivo es peor que
    el precio al que se decidió. No incluye fees: esas van aparte.
    """
    sign = _side_sign(side)
    level = _positive("decision_price", decision_price)
    fill = _positive("fill_price", fill_price)
    return sign * (fill - level) / level * 10_000.0


def modeled_slippage_bps(
    model: CostModel,
    *,
    price: float,
    qty: float,
    adv: Optional[float] = None,
    sigma: Optional[float] = None,
) -> float:
    """Medio spread más impacto, en bps del ``price``.

    Sin quote, Roll ni Corwin–Schultz el spread es el piso del modelo
    (tick y piso en bps). Sin ADV positivo el impacto es cero: no se
    inventa volumen.
    """
    if not isinstance(model, CostModel):
        raise TypeError("model debe ser CostModel")
    level = _positive("price", price)
    amount = _positive("qty", qty)
    half = float(model.half_spread(level))
    impact = 0.0
    if adv is not None and float(adv) > 0.0 and sigma is not None and float(sigma) >= 0.0:
        impact = float(model.impact_per_share(level, amount, float(sigma), float(adv)))
    return (half + impact) / level * 10_000.0


def slippage_recalibration_alert(
    observations: Sequence[SessionSlippage],
) -> Optional[str]:
    """Texto de alerta, o None si no se cruza el umbral.

    Hacen falta al menos 20 sesiones. La brecha es el slippage real
    medio menos el modelado. Si la media de esas brechas supera dos
    errores estándar, se propone recalibrar y se deja escrito que el
    cambio de parámetros no lo hace este módulo.
    """
    rows = tuple(observations)
    n = len(rows)
    if n < MIN_SLIPPAGE_SESSIONS:
        return None
    gaps = [float(item.real_bps) - float(item.modeled_bps) for item in rows]
    mean = float(statistics.fmean(gaps))
    if n < 2:
        return None
    stdev = float(statistics.stdev(gaps))
    se = stdev / math.sqrt(n)
    if mean <= SLIPPAGE_SE_MULT * se:
        return None
    return (
        f"En {n} sesiones el slippage real medio supera al modelado "
        f"en {mean:.2f} bps (error estándar {se:.2f}; umbral "
        f"{SLIPPAGE_SE_MULT:.0f}·se = {SLIPPAGE_SE_MULT * se:.2f}). "
        "Se propone recalibrar el modelo de costos. Cambiar los "
        "parámetros es una decisión humana y queda en HFT-14."
    )


def compare_executions(
    paper_intents: Sequence[OrderIntent],
    backtest_intents: Sequence[OrderIntent],
    *,
    paper_bars: IntradayBars,
    replay_bars: IntradayBars,
    paper_fills: Sequence[PaperFill] = (),
    paper_rejections: Sequence[RiskEvent] = (),
    backtest_rejections: Sequence[RiskEvent] = (),
    cost_model: Optional[CostModel] = None,
    slippage_history: Sequence[SessionSlippage] = (),
    feed: str = "iex",
    session_date: Optional[date] = None,
) -> Reconciliation:
    """Compara intenciones ya materializadas y valora los fills de paper.

    No corre el motor. ``reconcile_session`` es la que vuelve a lanzar
    el backtest sobre las mismas barras.
    """
    paper_map = _by_id(paper_intents)
    replay_map = _by_id(backtest_intents)
    ids = sorted(set(paper_map) | set(replay_map))
    diffs = []
    matched = 0
    for intent_id in ids:
        left = paper_map.get(intent_id)
        right = replay_map.get(intent_id)
        if left is not None and right is not None and _same_intent(left, right):
            matched += 1
            continue
        sample = left if left is not None else right
        cause, detail = _explain(
            sample,
            paper_bars,
            replay_bars,
            paper_rejections,
            backtest_rejections,
        )
        if right is None:
            where = "solo_paper"
        elif left is None:
            where = "solo_backtest"
        else:
            where = "distinto"
        diffs.append(IntentDiff(
            intent_id=intent_id,
            where=where,
            cause=cause,
            detail=detail,
            symbol=sample.symbol,
        ))
    parity = 1.0 if not ids else matched / len(ids)
    shortfalls = tuple(
        _shortfall_row(fill, cost_model) for fill in paper_fills
    )
    history = list(slippage_history)
    if shortfalls:
        history.append(SessionSlippage(
            session_date=(session_date or date.min).isoformat(),
            real_bps=float(statistics.fmean(row.real_slippage_bps for row in shortfalls)),
            modeled_bps=float(statistics.fmean(row.modeled_slippage_bps for row in shortfalls)),
        ))
    day = session_date or _guess_day(paper_intents, backtest_intents)
    return Reconciliation(
        parity=float(parity),
        n_paper=len(paper_map),
        n_backtest=len(replay_map),
        n_matched=matched,
        diffs=tuple(diffs),
        shortfalls=shortfalls,
        alert=slippage_recalibration_alert(history),
        feed=str(feed),
        session_date=day,
    )


def reconcile_session(
    config,
    session_date: date,
    paper_bars: IntradayBars,
    paper_intents: Sequence[OrderIntent],
    *,
    replay_bars: Optional[IntradayBars] = None,
    paper_fills: Sequence[PaperFill] = (),
    paper_rejections: Sequence[RiskEvent] = (),
    strategies: Optional[Sequence[object]] = None,
    calendar=None,
    slippage_history: Sequence[SessionSlippage] = (),
    feed: str = "iex",
    seed: int = 1,
) -> Reconciliation:
    """Reejecuta el backtest sobre las barras de la sesión y compara.

    ``replay_bars`` es el histórico final. Si no se pasa, se usan las
    mismas barras que vio el paper: la paridad tiene que ser 1.
    """
    from src.hft.backtest import run

    replay = paper_bars if replay_bars is None else replay_bars
    chosen = str(feed or getattr(config, "feed", "iex"))
    report = run(
        config,
        session_date,
        session_date,
        provider=_PanelProvider(replay),
        feed=chosen,
        strategies=strategies,
        calendar=calendar,
        record_trial=False,
        persist=False,
        seed=seed,
    )
    model = CostModel.from_config(config)
    return compare_executions(
        paper_intents,
        report.intents,
        paper_bars=paper_bars,
        replay_bars=replay,
        paper_fills=paper_fills,
        paper_rejections=paper_rejections,
        backtest_rejections=report.rejections,
        cost_model=model,
        slippage_history=slippage_history,
        feed=chosen,
        session_date=session_date,
    )


def render_html(view: ReportView) -> str:
    """Un documento HTML. No referencia CSS, JS ni imágenes externas."""
    title = html.escape(view.title)
    header = (
        f"<p>Rango {html.escape(view.start.isoformat())} a "
        f"{html.escape(view.end.isoformat())}. Feed {html.escape(view.feed)}.</p>"
    )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="es">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        header,
    ]
    if view.note:
        parts.append(f"<p>{html.escape(view.note)}</p>")
    if view.reconciliation is not None and view.reconciliation.alert:
        parts.append(
            f'<p class="alert">{html.escape(view.reconciliation.alert)}</p>'
        )
    parts.append("<h2>Curva de equity</h2>")
    parts.append(_line_svg(view.equity, "Curva de equity", "#1f4e79"))
    parts.append("<h2>Drawdown</h2>")
    parts.append(_line_svg(view.drawdown, "Drawdown", "#8c2f39"))
    parts.append("<h2>P&amp;L por hora</h2>")
    parts.append(_hour_svg(view.pnl_by_hour))
    parts.append("<h2>Métricas</h2>")
    parts.append(_kv_table(view.metrics))
    parts.append("<h2>Por estrategia</h2>")
    parts.append(_frame_table(view.by_strategy))
    parts.append("<h2>Blotter</h2>")
    parts.append(_frame_table(view.blotter))
    parts.append("<h2>Eventos de riesgo</h2>")
    parts.append(_records_table(view.risk_events))
    parts.append("<h2>Calidad de datos</h2>")
    parts.append(_records_table(view.quality_rows))
    parts.append("<h2>Desglose de costos</h2>")
    parts.append(_records_table(view.cost_rows))
    if view.reconciliation is not None:
        parts.append("<h2>Reconciliación</h2>")
        parts.append(_reconciliation_html(view.reconciliation))
    if view.provenance:
        parts.append("<h2>Procedencia</h2>")
        parts.append(_kv_table(view.provenance))
    parts.append("</body></html>\n")
    return "\n".join(parts)


def write_html(view: ReportView, path: Path) -> Path:
    """Escribe el HTML en ``path``. Un solo archivo."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(view), encoding="utf-8")
    return target


def view_from_backtest(
    report,
    *,
    quality: Optional[QualityReport] = None,
    risk_events: Optional[Sequence[RiskEvent]] = None,
    reconciliation: Optional[Reconciliation] = None,
    title: Optional[str] = None,
) -> ReportView:
    """Arma la vista desde un ``BacktestReport`` y, si hay, la reconciliación."""
    daily = report.daily
    equity = [float(value) for value in daily["nav"]] if not daily.empty else []
    labels = [str(value) for value in daily["date"]] if not daily.empty else []
    events = report.rejections if risk_events is None else risk_events
    metrics = dict(report.metrics or {})
    metrics.update({f"trade_{key}": value for key, value in (report.trade_stats or {}).items()})
    return ReportView(
        title=title or f"HFT {report.start.isoformat()} – {report.end.isoformat()}",
        start=report.start,
        end=report.end,
        feed=str(report.feed),
        metrics=metrics,
        by_strategy=report.attribution_strategy,
        blotter=report.blotter,
        risk_events=_risk_rows(events),
        quality_rows=_quality_rows(quality),
        cost_rows=_cost_rows(report, reconciliation),
        equity=equity,
        drawdown=_drawdown(equity),
        equity_labels=labels,
        pnl_by_hour=report.pnl_by_hour,
        reconciliation=reconciliation,
        provenance=dict(report.provenance or {}),
        note=(
            "Las fees sintéticas de paper salen de costs.py: el broker "
            "de paper no las cobra."
        ),
    )


def synthetic_view(
    *,
    start: date,
    end: date,
    feed: str = "iex",
    equity: Sequence[float] = (),
    pnl_by_hour: Optional[pd.DataFrame] = None,
    metrics: Optional[Mapping[str, object]] = None,
    by_strategy: Optional[pd.DataFrame] = None,
    blotter: Optional[pd.DataFrame] = None,
    risk_events: Sequence[Mapping[str, object]] = (),
    quality_rows: Sequence[Mapping[str, object]] = (),
    cost_rows: Sequence[Mapping[str, object]] = (),
    reconciliation: Optional[Reconciliation] = None,
) -> ReportView:
    """Vista para un estado sintético, sin correr el motor."""
    nav = [float(value) for value in equity]
    hours = pnl_by_hour if pnl_by_hour is not None else pd.DataFrame(
        columns=["hour_et", "pnl", "n"],
    )
    return ReportView(
        title=f"HFT {start.isoformat()} – {end.isoformat()}",
        start=start,
        end=end,
        feed=feed,
        metrics=dict(metrics or {}),
        by_strategy=by_strategy if by_strategy is not None else pd.DataFrame(),
        blotter=blotter if blotter is not None else pd.DataFrame(),
        risk_events=tuple(risk_events),
        quality_rows=tuple(quality_rows),
        cost_rows=tuple(cost_rows),
        equity=nav,
        drawdown=_drawdown(nav),
        pnl_by_hour=hours,
        reconciliation=reconciliation,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. ``--date`` o ``--start`` con ``--end``. No usa la red."""
    from src.hft.backtest import HoldoutLocked, run
    from src.hft.calendar import SessionCalendar
    from src.hft.config import load_hft_config
    from src.settings import load_settings, setup_logging

    parser = argparse.ArgumentParser(description="Reporte intradía HTML (HFT-13)")
    parser.add_argument("--date", default=None, help="Una sesión YYYY-MM-DD")
    parser.add_argument("--start", default=None, help="Primera sesión YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Última sesión YYYY-MM-DD")
    parser.add_argument("--config", default=None, help="TOML estricto (default config/hft.toml)")
    parser.add_argument("--feed", default=None, help="Feed del almacén (default el de la config)")
    parser.add_argument("--output", default=None, help="Ruta del HTML")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.date and (args.start or args.end):
        print("usa --date o --start con --end, no los dos", file=sys.stderr)
        return 2
    if args.date:
        start = end = args.date
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        print("hace falta --date, o --start y --end", file=sys.stderr)
        return 2
    setup_logging()
    config = load_hft_config(args.config)
    feed = args.feed or config.feed
    calendar = SessionCalendar(client=None, use_cache=False)
    try:
        report = run(
            config,
            start,
            end,
            feed=feed,
            calendar=calendar,
            record_trial=False,
            persist=False,
        )
    except HoldoutLocked as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.output:
        target = Path(args.output)
    else:
        name = f"{report.start.isoformat()}_{report.end.isoformat()}.html"
        target = load_settings().results_dir / "hft" / "reports" / name
    written = write_html(view_from_backtest(report), target)
    print(written)
    return 0


class _PanelProvider:
    """Sirve un panel ya en memoria. ``get_bars`` recorta ``[start, end)``."""

    def __init__(self, bars: IntradayBars) -> None:
        self.bars = bars

    def get_bars(self, symbols, start, end, timeframe=None):
        del symbols, timeframe
        index = self.bars.close.index
        if len(index) == 0:
            return self.bars
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (index >= start_ts) & (index < end_ts)
        frames = {
            name: getattr(self.bars, name).loc[mask]
            for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
        }
        return type(self.bars)(meta=self.bars.meta, **frames)


def _shortfall_row(fill: PaperFill, model: Optional[CostModel]) -> ShortfallRow:
    shortfall = implementation_shortfall_bps(fill.side, fill.fill_price, fill.decision_price)
    reference = fill.decision_price if fill.reference_price is None else fill.reference_price
    real = implementation_shortfall_bps(fill.side, fill.fill_price, reference)
    if model is None:
        modeled = 0.0
        synthetic = 0.0
    else:
        modeled = modeled_slippage_bps(
            model,
            price=reference,
            qty=fill.qty,
            adv=fill.adv,
            sigma=fill.sigma,
        )
        synthetic = float(paper_valuation_fees(
            model,
            side=fill.side,
            qty=fill.qty,
            price=fill.fill_price,
            liquidity=fill.liquidity,
            asset_class=fill.asset_class,
        ))
    return ShortfallRow(
        intent_id=fill.intent_id,
        symbol=str(fill.symbol).strip().upper(),
        side=fill.side,
        qty=float(fill.qty),
        decision_price=float(fill.decision_price),
        fill_price=float(fill.fill_price),
        shortfall_bps=float(shortfall),
        real_slippage_bps=float(real),
        modeled_slippage_bps=float(modeled),
        synthetic_fees=synthetic,
        paper_fees=float(fill.fees_charged),
    )


def _explain(
    intent: OrderIntent,
    paper_bars: IntradayBars,
    replay_bars: IntradayBars,
    paper_rejections: Sequence[RiskEvent],
    backtest_rejections: Sequence[RiskEvent],
) -> Tuple[str, str]:
    symbol = intent.symbol
    stamp = pd.Timestamp(intent.signal_ts)
    paper_px = _ohlc_at(paper_bars, symbol, stamp)
    replay_px = _ohlc_at(replay_bars, symbol, stamp)
    if paper_px is None or replay_px is None:
        return CAUSE_MISSING, "la barra de la señal no está en los dos paneles"
    if not _ohlc_equal(paper_px, replay_px):
        return (
            CAUSE_LATE_BAR,
            "el OHLC de la barra de la señal difiere entre el paper y el histórico",
        )
    window = _window_gap(paper_bars, replay_bars, symbol, stamp)
    if window is not None:
        return window
    reason = _rejection_reason(intent, paper_rejections, backtest_rejections)
    if reason is not None:
        return CAUSE_RISK, reason
    session = _session_gap(paper_bars, replay_bars, symbol)
    if session is not None:
        return session
    return CAUSE_UNEXPLAINED, "las barras coinciden y no hay un rechazo con este id"


def _window_gap(paper_bars, replay_bars, symbol: str, stamp: pd.Timestamp):
    stamps = _stamps_upto(paper_bars, replay_bars, symbol, stamp)
    for ts in stamps:
        left = _ohlc_at(paper_bars, symbol, ts)
        right = _ohlc_at(replay_bars, symbol, ts)
        if left is None or right is None:
            return CAUSE_MISSING, f"falta una barra anterior a la señal ({_iso(ts)})"
        if not _ohlc_equal(left, right):
            return (
                CAUSE_LATE_BAR,
                f"una barra anterior a la señal fue corregida ({_iso(ts)})",
            )
    return None


def _session_gap(paper_bars, replay_bars, symbol: str):
    stamps = _symbol_stamps(paper_bars, symbol) | _symbol_stamps(replay_bars, symbol)
    for ts in sorted(stamps):
        left = _ohlc_at(paper_bars, symbol, ts)
        right = _ohlc_at(replay_bars, symbol, ts)
        if left is None or right is None:
            return CAUSE_MISSING, f"falta una barra de la sesión ({_iso(ts)})"
        if not _ohlc_equal(left, right):
            return CAUSE_LATE_BAR, f"una barra de la sesión fue corregida ({_iso(ts)})"
    return None


def _rejection_reason(intent, paper_rejections, backtest_rejections) -> Optional[str]:
    for event in (*paper_rejections, *backtest_rejections):
        if event.intent_id and event.intent_id == intent.intent_id:
            return f"rechazo de riesgo: {event.reason}"
    return None


def _by_id(intents: Sequence[OrderIntent]) -> dict:
    found = {}
    for intent in intents:
        found[intent.intent_id] = intent
    return found


def _same_intent(left: OrderIntent, right: OrderIntent) -> bool:
    return (
        left.strategy_id == right.strategy_id
        and left.symbol == right.symbol
        and left.side == right.side
        and abs(float(left.qty) - float(right.qty)) <= _QTY_EPS
        and left.order_type == right.order_type
        and left.signal_ts == right.signal_ts
    )


def _ohlc_at(bars: IntradayBars, symbol: str, stamp: pd.Timestamp):
    stamp = _as_utc_stamp(stamp)
    close = bars.close
    if symbol not in close.columns or stamp not in close.index:
        return None
    parts = []
    for name in _OHLC:
        frame = getattr(bars, name)
        value = frame.loc[stamp, symbol]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        if pd.isna(value):
            return None
        parts.append(float(value))
    return tuple(parts)


def _ohlc_equal(left: Tuple[float, ...], right: Tuple[float, ...]) -> bool:
    return all(abs(a - b) <= _PRICE_EPS for a, b in zip(left, right))


def _stamps_upto(paper_bars, replay_bars, symbol: str, stamp: pd.Timestamp) -> list:
    stamp = _as_utc_stamp(stamp)
    found = [
        ts for ts in (_symbol_stamps(paper_bars, symbol) | _symbol_stamps(replay_bars, symbol))
        if ts <= stamp
    ]
    return sorted(found)


def _symbol_stamps(bars: IntradayBars, symbol: str) -> set:
    if symbol not in bars.close.columns:
        return set()
    series = bars.close[symbol].dropna()
    return {_as_utc_stamp(ts) for ts in series.index}


def _as_utc_stamp(stamp) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _iso(stamp) -> str:
    return _as_utc_stamp(stamp).isoformat()


def _guess_day(paper_intents, backtest_intents) -> date:
    for intent in (*paper_intents, *backtest_intents):
        return intent.signal_ts.date()
    return date.min


def _side_sign(side: str) -> float:
    if side == "buy":
        return 1.0
    if side == "sell":
        return -1.0
    raise ValueError(f"side {side!r} no es buy ni sell")


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} debe ser finito y > 0, llegó {value!r}")
    return number


def _drawdown(equity: Sequence[float]) -> list:
    peak = None
    out = []
    for value in equity:
        level = float(value)
        peak = level if peak is None else max(peak, level)
        if peak <= 0:
            out.append(0.0)
        else:
            out.append(level / peak - 1.0)
    return out


def _risk_rows(events: Sequence[RiskEvent]) -> tuple:
    rows = []
    for event in events:
        rows.append({
            "ts": event.ts.isoformat(),
            "kind": event.kind,
            "symbol": "" if event.symbol is None else event.symbol,
            "intent_id": "" if event.intent_id is None else event.intent_id,
            "reason": event.reason,
        })
    return tuple(rows)


def _quality_rows(quality: Optional[QualityReport]) -> tuple:
    if quality is None:
        return ()
    rows = []
    for issue in quality.issues:
        rows.append({
            "symbol": issue.symbol,
            "kind": issue.kind,
            "ts": "" if issue.ts is None else str(issue.ts),
            "session": "" if issue.session is None else issue.session.isoformat(),
            "detail": issue.detail,
        })
    return tuple(rows)


def _cost_rows(report, reconciliation: Optional[Reconciliation]) -> tuple:
    blotter = report.blotter
    fees = 0.0
    slip = None
    if blotter is not None and not blotter.empty and "fees" in blotter.columns:
        fees = float(blotter["fees"].sum())
    if (
        blotter is not None
        and not blotter.empty
        and "slippage_bps" in blotter.columns
        and blotter["slippage_bps"].notna().any()
    ):
        slip = float(blotter["slippage_bps"].mean())
    rows = [
        {"concepto": "fees_blotter", "valor": fees},
        {"concepto": "slippage_medio_bps", "valor": "" if slip is None else slip},
    ]
    if reconciliation is not None and reconciliation.shortfalls:
        rows.append({
            "concepto": "fees_sinteticas_paper",
            "valor": float(sum(row.synthetic_fees for row in reconciliation.shortfalls)),
        })
        rows.append({
            "concepto": "fees_cobradas_paper",
            "valor": float(sum(row.paper_fees for row in reconciliation.shortfalls)),
        })
        rows.append({
            "concepto": "slippage_real_medio_bps",
            "valor": float(statistics.fmean(
                row.real_slippage_bps for row in reconciliation.shortfalls
            )),
        })
        rows.append({
            "concepto": "slippage_modelado_medio_bps",
            "valor": float(statistics.fmean(
                row.modeled_slippage_bps for row in reconciliation.shortfalls
            )),
        })
    return tuple(rows)


def _reconciliation_html(item: Reconciliation) -> str:
    union = item.n_paper + item.n_backtest - item.n_matched
    summary = (
        f"<p>Paridad {item.parity_pct:.1f}% "
        f"({item.n_matched} de {union} en la unión; "
        f"paper {item.n_paper}, backtest {item.n_backtest}). "
        f"Feed {html.escape(item.feed)}. Sesión {html.escape(item.session_date.isoformat())}.</p>"
    )
    diffs = [
        {
            "intent_id": row.intent_id,
            "donde": row.where,
            "causa": row.cause,
            "simbolo": row.symbol,
            "detalle": row.detail,
        }
        for row in item.diffs
    ]
    shortfalls = [
        {
            "intent_id": row.intent_id,
            "symbol": row.symbol,
            "side": row.side,
            "qty": row.qty,
            "decision": row.decision_price,
            "fill": row.fill_price,
            "shortfall_bps": row.shortfall_bps,
            "real_bps": row.real_slippage_bps,
            "modelo_bps": row.modeled_slippage_bps,
            "fees_sinteticas": row.synthetic_fees,
            "fees_paper": row.paper_fees,
        }
        for row in item.shortfalls
    ]
    return "\n".join((
        summary,
        "<h3>Diferencias de intenciones</h3>",
        _records_table(diffs),
        "<h3>Implementation shortfall</h3>",
        _records_table(shortfalls),
    ))


def _kv_table(mapping: Mapping[str, object]) -> str:
    rows = [{"campo": key, "valor": _plain(mapping[key])} for key in sorted(mapping, key=str)]
    return _records_table(rows)


def _frame_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "<p>sin filas</p>"
    records = []
    for row in frame.itertuples(index=False):
        records.append({
            column: _plain(value)
            for column, value in zip(frame.columns, row)
        })
    return _records_table(records)


def _records_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "<p>sin filas</p>"
    columns = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(_plain(row.get(column, '')))}</td>" for column in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _plain(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return str(value)


def _line_svg(values: Sequence[float], label: str, color: str) -> str:
    width, height, pad = 640, 200, 28
    title = html.escape(label)
    if not values:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{title}">'
            f'<text x="24" y="100">{title}: sin datos</text></svg>'
        )
    lo = min(values)
    hi = max(values)
    if hi == lo:
        hi = lo + 1.0
    last = max(len(values) - 1, 1)

    def x_at(index: int) -> float:
        if len(values) == 1:
            return width / 2
        return pad + (width - 2 * pad) * index / last

    def y_at(level: float) -> float:
        return pad + (height - 2 * pad) * (1.0 - (level - lo) / (hi - lo))

    points = " ".join(
        f"{x_at(index):.1f},{y_at(float(level)):.1f}" for index, level in enumerate(values)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{title}">'
        f'<text x="8" y="16">{title}</text>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        f"</svg>"
    )


def _hour_svg(frame: pd.DataFrame) -> str:
    width, height, pad = 640, 200, 28
    if frame is None or frame.empty or "pnl" not in frame.columns:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="P&amp;L por hora">'
            f'<text x="24" y="100">P&amp;L por hora: sin datos</text></svg>'
        )
    hours = [int(value) for value in frame["hour_et"]]
    levels = [float(value) for value in frame["pnl"]]
    lo = min(0.0, min(levels))
    hi = max(0.0, max(levels))
    if hi == lo:
        hi = lo + 1.0
    span = hi - lo
    baseline = pad + (height - 2 * pad) * (1.0 - (0.0 - lo) / span)
    slot = (width - 2 * pad) / max(len(levels), 1)
    bars = []
    for index, level in enumerate(levels):
        x = pad + index * slot
        y = pad + (height - 2 * pad) * (1.0 - (level - lo) / span)
        top = min(y, baseline)
        bar_h = max(abs(baseline - y), 1.0)
        color = "#1f4e79" if level >= 0 else "#8c2f39"
        bars.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{max(slot - 4, 1):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}"/>'
        )
    labels = "".join(
        f'<text x="{pad + index * slot:.1f}" y="{height - 6}">{hour}</text>'
        for index, hour in enumerate(hours)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="P&amp;L por hora">'
        f'<text x="8" y="16">P&amp;L por hora (ET)</text>'
        + "".join(bars)
        + labels
        + "</svg>"
    )


_CSS = """
body { font-family: sans-serif; margin: 1.5rem; color: #1c1c1c; }
h1, h2, h3 { font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: left; font-size: 0.85rem; }
th { background: #f4f4f4; }
svg { width: 100%; max-width: 640px; height: auto; background: #fafafa; }
.alert { border: 1px solid #8c2f39; background: #fff4f4; padding: 0.75rem; }
"""


if __name__ == "__main__":
    sys.exit(main())
