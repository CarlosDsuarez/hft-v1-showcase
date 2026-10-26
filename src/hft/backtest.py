"""Backtest intradía (HFT-10).

``run`` recorre las sesiones con ``Engine.step`` y ``SimExecution``.
El proveedor y el adaptador cambian; la estrategia no. La decisión en
``t`` sigue ejecutándose como pronto en ``t+1``.

La salida es un ``BacktestReport``: blotter, P&L diario, equity,
exposición, turnover, atribución y procedencia. Las métricas de
cartera salen de ``src.metrics_calculator.compute_metrics`` sobre los
retornos diarios de la equity. El Sharpe del DSR es el diario, sin
anualizar; el anualizado se guarda solo para mostrarlo.

Cada corrida persistida deja CSV, ``summary.json`` y el HTML
autocontenido de ``src.hft.report`` (HFT-13).

``holdout_start`` bloquea las sesiones de esa fecha en adelante. El
CLI no las corre salvo ``--unlock-holdout``, y ese desbloqueo queda
escrito en ``results/hft/trials.csv``.

Paridad con el paper: se piden los símbolos de ``config_symbols``
(universo, momentum, A–S y patas de los pares). El CLI le da al
momentum los cierres SIP diarios (barras ``1Day`` ``sip`` del BarStore,
o ``--momentum-closes`` con un CSV). Sin ``calendar`` se lee el cache
de ``/v2/calendar``; ``--calendar alpaca`` lo pide y lo actualiza.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.hft.calendar import HISTORY_LOOKBACK_DAYS, SessionCalendar, bar_close_time
from src.hft.config import HftConfig, config_symbols, load_hft_config
from src.hft.costs import CostModel
from src.hft.data.bars import DEFAULT_FEED, IntradayBars, canonical_timeframe
from src.hft.data.bar_store import BarStore, StoreProvider
from src.hft.engine import Engine
from src.hft.execution.sim import SimExecution
from src.hft.risk import AssetFlag, EngineRiskGate, RiskManager, RiskState
from src.hft.sizing import IntradaySizer
from src.hft.validation import (
    TRIAL_FIELDS,
    annualize_sharpe,
    append_trials,
    drop_best_days,
    dsr,
    estimator_variance,
    group_sum,
    min_track_record,
    psr,
    read_trials,
    sample_moments,
    trial_count,
    trial_sharpe_variance,
)
from src.metrics_calculator import compute_metrics, metrics_to_dict
from src.provenance import code_version
from src.settings import SEED, TRADING_DAYS, load_settings

logger = logging.getLogger("hft.backtest")

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
_EPS = 1e-9
# Sin sesiones previas el impacto sqrt(q/ADV) no está definido. Un ADV
# grande y σ = 0 lo dejan en cero; el piso de spread sigue aplicando.
_NO_HISTORY_ADV = 1.0e12

# El blotter vacío igual tiene encabezado: un lector no adivina columnas.
_BLOTTER_COLUMNS = (
    "strategy_id",
    "symbol",
    "side",
    "qty",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "fees",
    "slippage_bps",
    "mae_bps",
    "mfe_bps",
    "holding_minutes",
    "exit_reason",
    "pnl",
    "pnl_bps",
    "r_multiple",
    "closed",
)


class HoldoutLocked(RuntimeError):
    """El rango pide sesiones en o después de ``holdout_start``."""


@dataclass
class BacktestReport:
    """Resultado de una corrida. Los frames ya están en la frecuencia pedida."""

    run_id: str
    config_hash: str
    feed: str
    start: date
    end: date
    holdout_unlocked: bool
    blotter: pd.DataFrame
    daily: pd.DataFrame
    attribution_strategy: pd.DataFrame
    attribution_symbol: pd.DataFrame
    pnl_by_hour: pd.DataFrame
    metrics: Optional[dict]
    trade_stats: dict
    validation: dict
    provenance: dict
    output_dir: Optional[Path] = None
    # Intenciones aprobadas y rechazos del journal. La reconciliación
    # de HFT-13 compara estas intenciones con las del paper.
    intents: tuple = ()
    rejections: tuple = ()


class _Clock:
    """Reloj fijo. ``Engine.step`` no lo consulta; el constructor lo exige."""

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


class _Marks:
    """Último cierre de la ventana que el motor acaba de entregar.

    ``features_fn`` corre antes del filtro, con barras ``<= t``. El
    riesgo necesita una marca para el nocional; el ``EngineState`` no
    la trae.
    """

    def __init__(self) -> None:
        self.latest: dict = {}

    def __call__(self, bars: IntradayBars) -> None:
        close = bars.close
        for symbol in close.columns:
            series = close[symbol].dropna()
            if series.empty:
                continue
            price = float(series.iloc[-1])
            if math.isfinite(price) and price > 0:
                self.latest[str(symbol)] = price
        return None


class _RiskView:
    """Actualiza marcas, caja y la base del día, y delega en el riesgo real."""

    def __init__(self, gate: EngineRiskGate, marks: _Marks) -> None:
        self.gate = gate
        self.marks = marks

    def filter(self, intents, state, now):
        day = now.astimezone(NY).date()
        pnl_now = float(state.realized_pnl) + float(state.unrealized_pnl)
        start_pnl = self.gate.state.session_start_pnl
        # No pisar session_date: lo avanza el manager, y así el conteo
        # diario de órdenes vuelve a cero con el cambio de sesión.
        if self.gate.state.session_date != day:
            start_pnl = pnl_now
        self.gate.state = replace(
            self.gate.state,
            marks=tuple(sorted(self.marks.latest.items())),
            buying_power=max(0.0, float(state.cash)),
            last_bar_close=now,
            session_start_pnl=start_pnl,
        )
        return self.gate.filter(intents, state, now)


def run(
    config: HftConfig,
    start,
    end,
    provider=None,
    feed: str = DEFAULT_FEED,
    *,
    strategies: Optional[Sequence[object]] = None,
    calendar: Optional[SessionCalendar] = None,
    unlock_holdout: bool = False,
    include_research: bool = False,
    cost_multiplier: float = 1.0,
    extra_latency_bars: int = 0,
    seed: int = SEED,
    record_trial: bool = True,
    persist: bool = True,
    rng: Optional[np.random.Generator] = None,
    momentum_closes: Optional[object] = None,
) -> BacktestReport:
    """Corre el motor sesión por sesión y arma el reporte.

    ``provider`` por defecto es un ``StoreProvider`` del almacén de
    barras (sin red). ``feed`` se usa al construir ese proveedor; si el
    llamador pasa un proveedor, se respeta el que él traiga y ``feed``
    queda en la procedencia.

    Sin ``calendar`` se lee el cache de ``GET /v2/calendar`` que llena
    el paper (``<data_dir>/hft/calendar.csv``), con medias sesiones. Si
    no cubre el rango, queda el respaldo NYSE (16:00 fijo) con WARNING.

    ``cost_multiplier`` escala piso de spread, impacto y fees (el tick
    no: es el incremento del mercado, no un costo elegido).
    ``extra_latency_bars`` se suma a la latencia del simulador, encima
    del ``t+1`` del motor.
    """
    if not isinstance(config, HftConfig):
        raise TypeError("config debe ser HftConfig")
    start_d = _as_date(start)
    end_d = _as_date(end)
    if end_d < start_d:
        raise ValueError(f"start {start_d.isoformat()} es posterior a end {end_d.isoformat()}")
    if cost_multiplier <= 0:
        raise ValueError("cost_multiplier debe ser > 0")
    if isinstance(extra_latency_bars, bool) or not isinstance(extra_latency_bars, int):
        raise TypeError("extra_latency_bars debe ser int")
    if extra_latency_bars < 0:
        raise ValueError("extra_latency_bars debe ser >= 0")
    chosen_feed = str(feed).strip().lower()
    if provider is None:
        provider = StoreProvider(
            BarStore(),
            feed=chosen_feed,
            timeframe=config.timeframe,
        )
    cal = calendar if calendar is not None else SessionCalendar(client=None, use_cache=True)
    window_start = start_d - timedelta(days=HISTORY_LOOKBACK_DAYS)
    cal.load(window_start, end_d)
    trading = [session for session in cal.sessions if start_d <= session.date <= end_d]
    _guard_holdout(config, trading, unlock_holdout)
    active = list(strategies) if strategies is not None else _default_strategies(
        config, cal.sessions, include_research=include_research,
        momentum_closes=momentum_closes,
    )
    if not active:
        raise ValueError("no hay estrategias para correr")
    generator = rng if rng is not None else np.random.default_rng(seed)
    engine = _build_engine(
        config,
        active,
        cal,
        generator,
        cost_multiplier=float(cost_multiplier),
        extra_latency_bars=int(extra_latency_bars),
        end=end_d,
    )
    _warmup(engine, provider, cal.sessions, start_d, config)
    tape = _ImpactTape()
    for session in trading:
        _refresh_impact(engine, tape, session.date, config)
        bars = _session_bars(provider, config, session)
        if bars.close.empty:
            continue
        ready = session.close_utc + timedelta(seconds=config.grace_s)
        engine.step(ready, bars)
    report = _report(
        engine,
        config,
        chosen_feed,
        start_d,
        end_d,
        bool(unlock_holdout),
        provider,
        cost_multiplier=float(cost_multiplier),
        extra_latency_bars=int(extra_latency_bars),
    )
    if record_trial:
        _record(report, config)
    if persist:
        report.output_dir = _persist(report)
    return report


def stress_costs(config: HftConfig, start, end, **kwargs) -> BacktestReport:
    """La misma corrida con costos × 2. No entra al registro de pruebas."""
    kwargs["cost_multiplier"] = 2.0
    kwargs["record_trial"] = False
    kwargs["persist"] = False
    return run(config, start, end, **kwargs)


def stress_latency(config: HftConfig, start, end, **kwargs) -> BacktestReport:
    """La misma corrida con una barra más de latencia en el simulador."""
    extra = int(kwargs.pop("extra_latency_bars", 0))
    kwargs["extra_latency_bars"] = extra + 1
    kwargs["record_trial"] = False
    kwargs["persist"] = False
    return run(config, start, end, **kwargs)


def stress_sip(config: HftConfig, start, end, provider, **kwargs) -> Optional[BacktestReport]:
    """Rejuega el rango con el feed SIP si el almacén tiene esas barras.

    Sin ``store`` en el proveedor, o con un SIP vacío, devuelve None:
    no hay sensibilidad que reportar.
    """
    store = getattr(provider, "store", None)
    if store is None:
        return None
    sip = StoreProvider(store, feed="sip", timeframe=config.timeframe)
    start_d = _as_date(start)
    end_d = _as_date(end)
    probe_end = datetime(end_d.year, end_d.month, end_d.day, tzinfo=UTC) + timedelta(days=1)
    probe_start = datetime(start_d.year, start_d.month, start_d.day, tzinfo=UTC)
    probe = sip.get_bars(config.universe, probe_start, probe_end)
    if probe.close.dropna(how="all").empty:
        return None
    kwargs["record_trial"] = False
    kwargs["persist"] = False
    return run(config, start, end, provider=sip, feed="sip", **kwargs)


def momentum_closes_from_store(store: BarStore, config: HftConfig, start, end):
    """Cierres SIP diarios del símbolo del momentum, leídos del BarStore.

    Son las barras ``1Day`` del feed ``sip`` (el paper las pide a la API
    en ``scripts/run_intraday._momentum_closes``). Se bajan sin tocar el
    backtest con ``python -m src.hft.data.download --symbols SPY
    --timeframe 1Day --feed sip``. La ventana termina a la medianoche NY
    de ``end``, exclusiva: el cierre de ``end`` no se lee. Dentro del
    rango, ``ConsolidatedCloses`` solo usa fechas anteriores a cada
    sesión. None si el almacén no trae cierres.
    """
    from src.hft.strategies.intraday_momentum import ConsolidatedCloses

    symbol = config.intraday_momentum.symbol
    first = _as_date(start) - timedelta(days=HISTORY_LOOKBACK_DAYS)
    last = _as_date(end)
    bars = store.read(
        [symbol],
        datetime(first.year, first.month, first.day, tzinfo=NY),
        datetime(last.year, last.month, last.day, tzinfo=NY),
        timeframe="1Day",
        feed="sip",
    )
    if symbol not in bars.close.columns:
        return None
    series = bars.close[symbol].dropna()
    if series.empty:
        return None
    series.index = [stamp.tz_convert(NY).date() for stamp in series.index]
    return ConsolidatedCloses(series, feed="sip")


def momentum_closes_from_csv(path):
    """Cierres SIP diarios desde un CSV con columnas ``date`` (sesión NY) y ``close``."""
    from src.hft.strategies.intraday_momentum import ConsolidatedCloses

    frame = pd.read_csv(path)
    missing = sorted({"date", "close"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}; se esperan date,close")
    series = pd.Series(
        frame["close"].to_numpy(dtype=float),
        index=[date.fromisoformat(str(value)) for value in frame["date"]],
    )
    return ConsolidatedCloses(series, feed="sip")


def statistical_stress(
    daily_pnl: pd.Series,
    spy_prices: Optional[pd.Series] = None,
) -> dict:
    """Quitar los 5 mejores días, partir por año y, si hay SPY, por régimen.

    El régimen sale de ``src.regime_detector.classify_regime`` sobre el
    cierre diario de SPY, en modo permisivo: un historial corto no
    aborta el estrés, degrada la etiqueta.
    """
    from src.regime_detector import classify_regime

    values = daily_pnl.to_numpy(dtype=float)
    dropped = drop_best_days(values, 5)
    years = [int(pd.Timestamp(day).year) for day in daily_pnl.index]
    by_year = group_sum(years, values)
    by_regime = None
    if spy_prices is not None and len(spy_prices.dropna()) > 0:
        frame = spy_prices.dropna().to_frame("SPY")
        frame.index = pd.to_datetime(frame.index)
        labels = []
        for day in daily_pnl.index:
            report = classify_regime(frame, pd.Timestamp(day), mode="permissive")
            labels.append(report.state)
        by_regime = {str(key): value for key, value in group_sum(labels, values).items()}
    return {
        "pnl": float(values.sum()) if values.size else 0.0,
        "pnl_without_best_5_days": float(dropped.sum()) if dropped.size else 0.0,
        "by_year": {str(key): value for key, value in by_year.items()},
        "by_regime": by_regime,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Sale 2 si el rango pisa el holdout y no se pasó el desbloqueo."""
    from src.settings import setup_logging

    parser = argparse.ArgumentParser(description="Backtest intradía (HFT-10)")
    parser.add_argument("--config", default=None, help="TOML estricto (default config/hft.toml)")
    parser.add_argument("--start", required=True, help="Primera sesión YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Última sesión YYYY-MM-DD")
    parser.add_argument("--feed", default=DEFAULT_FEED)
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="Permite sesiones >= holdout_start y lo anota en trials.csv",
    )
    parser.add_argument(
        "--include-research",
        action="store_true",
        help="Incluye estrategias research_only (siguen vetadas en paper)",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Además imprime el estrés estadístico del P&L diario",
    )
    parser.add_argument(
        "--calendar",
        choices=("cache", "alpaca"),
        default="cache",
        help="cache: data/hft/calendar.csv, sin red (respaldo NYSE si no cubre el rango). "
        "alpaca: GET /v2/calendar, que además actualiza el cache",
    )
    parser.add_argument(
        "--momentum-closes",
        default=None,
        metavar="CSV",
        help="Cierres SIP diarios del momentum (date,close). "
        "Default: barras 1Day sip del BarStore",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging()
    config = load_hft_config(args.config)
    if args.momentum_closes is not None:
        closes = momentum_closes_from_csv(args.momentum_closes)
    else:
        closes = momentum_closes_from_store(BarStore(), config, args.start, args.end)
    calendar = None
    if args.calendar == "alpaca":
        calendar = SessionCalendar(client=_calendar_client(), use_cache=True)
    try:
        report = run(
            config,
            args.start,
            args.end,
            feed=args.feed,
            unlock_holdout=args.unlock_holdout,
            include_research=args.include_research,
            momentum_closes=closes,
            calendar=calendar,
        )
    except HoldoutLocked as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.stress and not report.daily.empty:
        view = statistical_stress(report.daily["pnl"], _spy_from_report(report))
        print(json.dumps(view, indent=2, sort_keys=True))
    print(report.output_dir)
    return 0


def _calendar_client():
    """Cliente de la Trading API para ``GET /v2/calendar`` (solo lectura)."""
    from src.brokers.alpaca_adapter import AlpacaClient, load_broker_config

    return AlpacaClient(load_broker_config())


def _guard_holdout(config: HftConfig, sessions, unlock: bool) -> None:
    blocked = [session.date.isoformat() for session in sessions if session.date.isoformat() >= config.holdout_start]
    if blocked and not unlock:
        raise HoldoutLocked(
            f"holdout_start={config.holdout_start} bloquea {blocked[0]}..{blocked[-1]} "
            f"({len(blocked)} sesiones). Volvé a correr con --unlock-holdout; "
            "el desbloqueo queda en trials.csv."
        )


def _default_strategies(config, sessions, *, include_research: bool, momentum_closes=None):
    """Las cuatro de capital. Avellaneda–Stoikov solo con el flag.

    El universo se marca operable en corto dentro de la estrategia. El
    paper (HFT-11) lo va a leer de ``get_asset``; aquí no hay ficha del
    broker y sin el flag las patas cortas no existirían.
    """
    from src.hft.strategies.avellaneda_stoikov import AvellanedaStoikov
    from src.hft.strategies.intraday_momentum import IntradayMomentum
    from src.hft.strategies.orb import OpeningRangeBreakout
    from src.hft.strategies.pairs_ou import PairsOu
    from src.hft.strategies.vwap_reversion import VwapReversion

    if momentum_closes is None:
        logger.warning(
            "intraday_momentum sin cierres SIP diarios (momentum_closes): no opera "
            "en este backtest.",
        )
    built = [
        OpeningRangeBreakout.from_config(config, sessions),
        IntradayMomentum.from_config(config, sessions, closes=momentum_closes),
        VwapReversion.from_config(config, sessions, shortable=config.universe),
        PairsOu.from_config(config, sessions, shortable=config.universe),
    ]
    if include_research:
        built.append(AvellanedaStoikov.from_config(config, sessions))
    return built


def _build_engine(config, strategies, calendar, rng, *, cost_multiplier, extra_latency_bars, end):
    sim = SimExecution.from_config(config, rng=rng)
    sim.costs = _scale_costs(sim.costs, cost_multiplier)
    sim.latency_bars = int(sim.latency_bars) + int(extra_latency_bars)
    # Como el paper en sim: todo símbolo de config_symbols lleva ficha.
    assets = tuple((symbol, AssetFlag(True, True)) for symbol in config_symbols(config))
    risk_state = RiskState(
        equity=config.initial_cash,
        nav=config.initial_cash,
        buying_power=config.initial_cash,
        session_start_pnl=0.0,
        cash=config.initial_cash,
        assets=assets,
    )
    manager = RiskManager(config.risk, calendar.sessions)
    gate = EngineRiskGate(manager, risk_state, equity_from_book=True)
    marks = _Marks()
    instant = datetime(end.year, end.month, end.day, 21, 0, tzinfo=UTC)
    return Engine(
        strategies,
        marks,
        _RiskView(gate, marks),
        IntradaySizer(
            config.sizing,
            max_position_notional=config.risk.max_position_notional,
        ),
        sim,
        calendar,
        _Clock(instant),
        config=config,
        rng=rng,
        mode="backtest",
    )


def _scale_costs(model: CostModel, factor: float) -> CostModel:
    """Piso, impacto y fees × factor. El tick queda: es el mercado."""
    if factor == 1.0:
        return model
    return replace(
        model,
        spread_floor_bps=model.spread_floor_bps * factor,
        impact_y=model.impact_y * factor,
        commission_per_share=model.commission_per_share * factor,
        sec_section31_rate=model.sec_section31_rate * factor,
        finra_taf_per_share=model.finra_taf_per_share * factor,
        finra_taf_cap=model.finra_taf_cap * factor,
        crypto_maker_bps=model.crypto_maker_bps * factor,
        crypto_taker_bps=model.crypto_taker_bps * factor,
    )


def _warmup(engine: Engine, provider, sessions, start: date, config: HftConfig) -> None:
    """Mete historia anterior a ``start`` en el buffer, sin decidir.

    ``_ingest`` marca las barras como vistas. El primer ``step`` de la
    muestra arma la ventana ``<= t`` con esa historia y no reabre
    aquellas barras: no hay operaciones de calentamiento en el blotter.
    """
    need_bars, need_sessions = engine.required_history()
    step = bar_close_time(datetime(2024, 1, 2, tzinfo=UTC), config.timeframe) - datetime(2024, 1, 2, tzinfo=UTC)
    per_session = max(1, int(round(timedelta(minutes=390) / step)))
    extra = max(int(need_sessions), int(math.ceil(need_bars / per_session)))
    if extra <= 0:
        return
    prior = [session for session in sessions if session.date < start][-extra:]
    for session in prior:
        bars = _session_bars(provider, config, session)
        if bars.close.empty:
            continue
        ready = session.close_utc + timedelta(seconds=config.grace_s)
        engine._ingest(bars, ready)


class _ImpactTape:
    """Dólar operado y último cierre por día y símbolo, ya resumidos.

    ``_refresh_impact`` corre antes de cada sesión. Recorrer toda la
    historia en cada una hace cuadrático un año. Cada día se resume
    cuando ya es anterior a la sesión, con el mismo cálculo que antes;
    si después le llegan barras, se vuelve a resumir.
    """

    def __init__(self) -> None:
        self._rows = 0
        self._codes = np.empty(0, dtype="datetime64[D]")
        self._dirty: set = set()
        self._days: dict = {}

    def update(self, frames: dict, before: date) -> None:
        close = frames["close"]
        if len(close.index) > self._rows:
            stamps = pd.DatetimeIndex(close.index[self._rows:]).tz_convert(NY)
            fresh = stamps.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
            self._codes = np.concatenate([self._codes, fresh])
            self._dirty.update(day.item() for day in np.unique(fresh))
            self._rows = len(close.index)
        volume = frames["volume"]
        for day in sorted(item for item in self._dirty if item < before):
            mask = self._codes == np.datetime64(day)
            summary = {}
            for symbol in close.columns:
                pair = pd.DataFrame({
                    "px": close.loc[mask, symbol].to_numpy(dtype=float),
                    "vol": volume.loc[mask, symbol].to_numpy(dtype=float),
                }).dropna()
                pair = pair[(pair["px"] > 0) & (pair["vol"] >= 0)]
                if pair.empty:
                    continue
                summary[str(symbol)] = (
                    float((pair["px"] * pair["vol"]).sum()),
                    float(pair["px"].iloc[-1]),
                )
            self._days[day] = summary
            self._dirty.discard(day)

    def history(self, symbol: str, before: date) -> Tuple[list, list]:
        """``(dólar, último cierre)`` de cada día previo con datos, en orden."""
        dollar = []
        lasts = []
        for day in sorted(item for item in self._days if item < before):
            found = self._days[day].get(symbol)
            if found is None:
                continue
            dollar.append(found[0])
            lasts.append(found[1])
        return dollar, lasts


def _refresh_impact(engine: Engine, tape: _ImpactTape, session_date: date, config: HftConfig) -> None:
    """ADV y σ diarios con barras ya vistas, anteriores a esta sesión.

    El simulador los exige para el impacto. Usar el día que se está
    operando sería look-ahead. Si un símbolo no tiene historia, el
    impacto queda en cero y el costo que queda es el spread. Cubre
    ``config_symbols``, como el paper: las patas de los pares también.
    """
    symbols = [str(symbol) for symbol in config_symbols(config)]
    adv = {symbol: _NO_HISTORY_ADV for symbol in symbols}
    sigma = {symbol: 0.0 for symbol in symbols}
    frames = engine._frames
    window = max(1, int(config.costs.adv_window))
    if frames is not None and not frames["close"].empty:
        tape.update(frames, session_date)
        for symbol in frames["close"].columns:
            name = str(symbol)
            dollar, lasts = tape.history(name, session_date)
            used = [value for value in dollar[-window:] if value > 0]
            if used:
                adv[name] = float(np.mean(used))
            if len(lasts) >= 3:
                rets = np.diff(np.log(np.asarray(lasts, dtype=float)))
                if rets.size >= 2 and np.isfinite(rets).all():
                    std = float(rets.std(ddof=1))
                    if math.isfinite(std) and std >= 0.0:
                        sigma[name] = std
    execution = engine.execution
    if hasattr(execution, "_adv") and hasattr(execution, "_sigma"):
        execution._adv = adv
        execution._sigma = sigma


def _session_bars(provider, config: HftConfig, session) -> IntradayBars:
    """Los mismos símbolos que pide el paper (``config_symbols``)."""
    return provider.get_bars(
        config_symbols(config),
        session.open_utc,
        session.close_utc,
        config.timeframe,
    )


def _report(
    engine: Engine,
    config: HftConfig,
    feed: str,
    start: date,
    end: date,
    unlock: bool,
    provider,
    *,
    cost_multiplier: float,
    extra_latency_bars: int,
) -> BacktestReport:
    frames = engine._frames
    fills = _fills(engine)
    closed, open_rows = _round_trips(fills, frames)
    blotter = pd.DataFrame(closed + open_rows, columns=_BLOTTER_COLUMNS)
    daily = _daily_table(engine, config.initial_cash, fills)
    metrics = _portfolio_metrics(daily)
    trade_stats = _trade_stats(blotter, daily)
    spy = _spy_daily(frames)
    stress = statistical_stress(daily["pnl"], spy) if not daily.empty else {
        "pnl": 0.0,
        "pnl_without_best_5_days": 0.0,
        "by_year": {},
        "by_regime": None,
    }
    code = code_version()
    strategy_ids = tuple(strategy.id for strategy in engine.strategies)
    run_id = _run_id(
        config, start, end, feed, cost_multiplier, extra_latency_bars, unlock, strategy_ids,
    )
    provenance = {
        "git_sha": code.sha,
        "git_dirty": code.dirty,
        "config_hash": config.config_hash(),
        "feed": feed,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "holdout_start": config.holdout_start,
        "holdout_unlocked": unlock,
        "cost_multiplier": cost_multiplier,
        "extra_latency_bars": extra_latency_bars,
        "data": _data_version(provider, config, feed),
        "strategies": list(strategy_ids),
    }
    validation = {
        "portfolio": _score_series(
            daily["return"].to_numpy(dtype=float) if not daily.empty else np.asarray([]),
            n_trials=1,
            sr_variance=None,
        ),
        "by_strategy": {},
        "stress": stress,
        "run_id": run_id,
    }
    report = BacktestReport(
        run_id=run_id,
        config_hash=config.config_hash(),
        feed=feed,
        start=start,
        end=end,
        holdout_unlocked=unlock,
        blotter=blotter,
        daily=daily,
        attribution_strategy=_attribute(blotter, "strategy_id"),
        attribution_symbol=_attribute(blotter, "symbol"),
        pnl_by_hour=_pnl_by_hour(blotter),
        metrics=metrics,
        trade_stats=trade_stats,
        validation=validation,
        provenance=provenance,
        intents=_approved_intents(engine),
        rejections=_rejections(engine),
    )
    report.validation["by_strategy"] = {
        strategy_id: _strategy_score(report, strategy_id) for strategy_id in strategy_ids
    }
    report.validation["by_strategy"]["book"] = _strategy_score(report, "book")
    return report


def _record(report: BacktestReport, config: HftConfig) -> None:
    path = load_settings().results_dir / "hft" / "trials.csv"
    rows = []
    names = list(report.validation["by_strategy"])
    for name in names:
        score = report.validation["by_strategy"][name]
        trades = report.trade_stats["n"] if name == "book" else _trades_of(report.blotter, name)
        rows.append({
            "run_id": report.run_id,
            "config_hash": report.config_hash,
            "strategy": name,
            "sr": score["sr"],
            "n": score["t"],
            "n_trades": trades,
            "date": report.end.isoformat(),
            "feed": report.feed,
            "start": report.start.isoformat(),
            "end": report.end.isoformat(),
            "holdout_unlocked": report.holdout_unlocked,
            "cost_multiplier": report.provenance["cost_multiplier"],
            "extra_latency_bars": report.provenance["extra_latency_bars"],
        })
    recorded = append_trials(path, rows)
    for name, score in report.validation["by_strategy"].items():
        observed = trial_count(recorded, name)
        variance = trial_sharpe_variance(recorded, name)
        if variance is None and score["t"] >= 2:
            try:
                variance = estimator_variance(score["sr"], score["skew"], score["kurtosis"], score["t"])
            except ValueError:
                variance = None
        score["n_trials"] = observed
        score["sr_variance"] = variance
        if variance is not None and score["t"] >= 2:
            score["dsr"] = dsr(
                score["sr"], variance, score["t"], score["skew"], score["kurtosis"], observed,
            )
            score["psr"] = psr(score["sr"], 0.0, score["t"], score["skew"], score["kurtosis"])
        else:
            score["dsr"] = None
            score["psr"] = None


def _persist(report: BacktestReport) -> Path:
    directory = load_settings().results_dir / "hft" / "backtests" / report.run_id
    directory.mkdir(parents=True, exist_ok=True)
    report.blotter.to_csv(directory / "blotter.csv", index=False)
    report.daily.to_csv(directory / "daily.csv", index=False)
    report.attribution_strategy.to_csv(directory / "attribution_strategy.csv", index=False)
    report.attribution_symbol.to_csv(directory / "attribution_symbol.csv", index=False)
    report.pnl_by_hour.to_csv(directory / "pnl_by_hour.csv", index=False)
    summary = {
        "run_id": report.run_id,
        "provenance": report.provenance,
        "metrics": report.metrics,
        "trade_stats": report.trade_stats,
        "validation": report.validation,
        "html": "report.html",
        "html_note": "HTML autocontenido de src.hft.report (HFT-13).",
    }
    text = json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    (directory / "summary.json").write_text(text, encoding="utf-8")
    # Import diferido: report.py llama a run() y no puede cargarse arriba.
    from src.hft.report import render_html, view_from_backtest

    page = render_html(view_from_backtest(report))
    (directory / "report.html").write_text(page, encoding="utf-8")
    return directory


def _approved_intents(engine: Engine) -> tuple:
    """Intenciones que pasaron el riesgo, en orden de decisión."""
    approved = []
    seen = set()
    for event in engine.journal:
        rejected = {item.intent_id for item in event.rejections if item.intent_id}
        for intent in event.intents:
            if intent.intent_id in rejected or intent.intent_id in seen:
                continue
            seen.add(intent.intent_id)
            approved.append(intent)
    return tuple(approved)


def _rejections(engine: Engine) -> tuple:
    return tuple(
        item
        for event in engine.journal
        for item in event.rejections
    )


def _fills(engine: Engine) -> list:
    intents = {}
    for event in engine.journal:
        for intent in (*event.orders, *event.intents):
            intents[intent.intent_id] = intent
    orders = getattr(engine.execution, "_orders", {})
    for working in orders.values():
        intents.setdefault(working.intent.intent_id, working.intent)
    rows = []
    for event in engine.journal:
        for fill in event.fills:
            intent = intents.get(fill.intent_id)
            if intent is None:
                logger.warning("fill sin intención: %s", fill.intent_id)
                continue
            rows.append((fill, intent))
    rows.sort(key=lambda item: (item[0].ts, item[0].intent_id))
    return rows


def _round_trips(pairs, frames) -> Tuple[list, list]:
    books: dict = {}
    closed = []
    for fill, intent in pairs:
        key = (intent.strategy_id, intent.symbol)
        lots = books.setdefault(key, [])
        closed.extend(_consume(lots, fill, intent, frames))
    opened = []
    for (strategy_id, symbol), lots in books.items():
        if lots:
            opened.append(_open_row(strategy_id, symbol, lots, frames))
    return closed, opened


def _consume(lots, fill, intent, frames) -> list:
    sign = 1.0 if intent.side == "buy" else -1.0
    remaining = float(fill.qty)
    fee_per = float(fill.fees) / float(fill.qty)
    decision = _price_at(frames, "close", intent.symbol, intent.signal_ts)
    stop = None if intent.bracket is None else intent.bracket.stop
    done = []
    while remaining > _EPS and lots and _lot_sign(lots[0]["qty"]) != sign:
        lot = lots[0]
        take = min(abs(lot["qty"]), remaining)
        fraction = take / abs(lot["qty"])
        fees = lot["fees"] * fraction + fee_per * take
        side = "long" if lot["qty"] > 0 else "short"
        entry = float(lot["price"])
        exit_price = float(fill.price)
        if side == "long":
            pnl = (exit_price - entry) * take - fees
        else:
            pnl = (entry - exit_price) * take - fees
        entry_slip = _slip_bps("buy" if side == "long" else "sell", entry, lot["decision"])
        exit_slip = _slip_bps("sell" if side == "long" else "buy", exit_price, decision)
        notional = entry * take
        risk = None
        if lot["stop"] is not None and abs(entry - lot["stop"]) > _EPS:
            risk = abs(entry - lot["stop"]) * take
        mae, mfe = _excursions(frames, intent.symbol, lot["ts"], fill.ts, entry, side)
        done.append({
            "strategy_id": intent.strategy_id,
            "symbol": intent.symbol,
            "side": side,
            "qty": take,
            "entry_ts": _iso(lot["ts"]),
            "exit_ts": _iso(fill.ts),
            "entry_price": entry,
            "exit_price": exit_price,
            "fees": fees,
            "slippage_bps": entry_slip + exit_slip,
            "mae_bps": mae,
            "mfe_bps": mfe,
            "holding_minutes": (fill.ts - lot["ts"]).total_seconds() / 60.0,
            "exit_reason": intent.reason,
            "pnl": pnl,
            "pnl_bps": (pnl / notional * 1e4) if notional > 0 else 0.0,
            "r_multiple": (pnl / risk) if risk else None,
            "closed": True,
        })
        lot["qty"] = lot["qty"] - math.copysign(take, lot["qty"])
        lot["fees"] = lot["fees"] * (1.0 - fraction)
        if abs(lot["qty"]) <= _EPS:
            lots.pop(0)
        remaining -= take
    if remaining > _EPS:
        lots.append({
            "qty": sign * remaining,
            "price": float(fill.price),
            "fees": fee_per * remaining,
            "ts": fill.ts,
            "reason": intent.reason,
            "decision": decision,
            "stop": stop,
        })
    return done


def _open_row(strategy_id, symbol, lots, frames) -> dict:
    qty = float(sum(abs(lot["qty"]) for lot in lots))
    notional = float(sum(abs(lot["qty"]) * lot["price"] for lot in lots))
    entry = notional / qty if qty > 0 else 0.0
    fees = float(sum(lot["fees"] for lot in lots))
    side = "long" if lots[0]["qty"] > 0 else "short"
    entry_ts = min(lot["ts"] for lot in lots)
    mark = _last_price(frames, symbol)
    if mark is None:
        pnl = 0.0
    elif side == "long":
        pnl = (mark - entry) * qty - fees
    else:
        pnl = (entry - mark) * qty - fees
    mae, mfe = _excursions(
        frames, symbol, entry_ts, _last_ts(frames) or entry_ts, entry, side,
    )
    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_ts": _iso(entry_ts),
        "exit_ts": "",
        "entry_price": entry,
        "exit_price": mark,
        "fees": fees,
        "slippage_bps": _slip_bps("buy" if side == "long" else "sell", entry, lots[0]["decision"]),
        "mae_bps": mae,
        "mfe_bps": mfe,
        "holding_minutes": None,
        "exit_reason": "abierta",
        "pnl": pnl,
        "pnl_bps": (pnl / notional * 1e4) if notional > 0 else 0.0,
        "r_multiple": None,
        "closed": False,
    }


def _daily_table(engine: Engine, initial_cash: float, fills) -> pd.DataFrame:
    last = {}
    gross_max = {}
    for event in engine.journal:
        if event.bar_ts is None:
            continue
        day = event.bar_ts.astimezone(NY).date()
        last[day] = event
        gross = _gross(event, engine)
        gross_max[day] = max(gross_max.get(day, 0.0), gross)
    traded = {}
    for fill, _intent in fills:
        day = fill.ts.astimezone(NY).date()
        traded[day] = traded.get(day, 0.0) + abs(fill.qty * fill.price)
    rows = []
    prev = float(initial_cash)
    for day in sorted(last):
        state = last[day].state
        nav = float(state["nav"])
        pnl = nav - prev
        ret = (nav / prev - 1.0) if prev > 0 else 0.0
        gross = _gross(last[day], engine)
        net = _net(last[day], engine)
        two_way = traded.get(day, 0.0)
        rows.append({
            "date": day.isoformat(),
            "nav": nav,
            "pnl": pnl,
            "return": ret,
            "gross_exposure": gross,
            "net_exposure": net,
            "max_gross_exposure": gross_max.get(day, gross),
            "traded_notional": two_way,
            "turnover": (two_way / nav) if nav > 0 else 0.0,
            "turnover_one_way": (0.5 * two_way / nav) if nav > 0 else 0.0,
            "fees": float(state["fees"]),
            "realized_pnl": float(state["realized_pnl"]),
        })
        prev = nav
    return pd.DataFrame(rows, columns=[
        "date", "nav", "pnl", "return", "gross_exposure", "net_exposure",
        "max_gross_exposure", "traded_notional", "turnover", "turnover_one_way",
        "fees", "realized_pnl",
    ])


def _gross(event, engine) -> float:
    return _exposure(event, engine, absolute=True)


def _net(event, engine) -> float:
    return _exposure(event, engine, absolute=False)


def _exposure(event, engine, *, absolute: bool) -> float:
    positions = event.state.get("positions") or {}
    if not positions or engine._frames is None:
        return 0.0
    close = engine._frames["close"]
    stamp = pd.Timestamp(event.bar_ts)
    total = 0.0
    for symbol, qty in positions.items():
        price = _price_at_index(close, symbol, stamp)
        if price is None:
            continue
        signed = abs(float(qty)) if absolute else float(qty)
        total += signed * price
    return float(total)


def _portfolio_metrics(daily: pd.DataFrame) -> Optional[dict]:
    if daily.empty:
        return None
    returns = pd.Series(daily["return"].to_numpy(dtype=float), index=pd.to_datetime(daily["date"]))
    nav = pd.Series(daily["nav"].to_numpy(dtype=float), index=returns.index)
    turnover = pd.Series(daily["turnover_one_way"].to_numpy(dtype=float), index=returns.index)
    if len(returns) < 1:
        return None
    metrics = compute_metrics(returns, nav, turnover=turnover, seed=SEED)
    payload = metrics_to_dict(metrics)
    payload["n_days"] = int(metrics.n_days)
    sr, _skew, _kurt, length = sample_moments(returns.to_numpy(dtype=float))
    payload["sharpe_daily"] = sr
    payload["sharpe_annualized_display"] = annualize_sharpe(sr, TRADING_DAYS) if length >= 2 else None
    return payload


def _trade_stats(blotter: pd.DataFrame, daily: pd.DataFrame) -> dict:
    closed = blotter[blotter["closed"] == True] if not blotter.empty else blotter  # noqa: E712
    pnl = closed["pnl"].to_numpy(dtype=float) if not closed.empty else np.asarray([])
    n = int(pnl.size)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    hit = float(wins.size / n) if n else None
    payoff = None
    if wins.size and losses.size:
        payoff = float(wins.mean() / abs(losses.mean()))
    expectancy_bps = float(closed["pnl_bps"].mean()) if n else None
    r_values = closed["r_multiple"].dropna().to_numpy(dtype=float) if n else np.asarray([])
    expectancy_r = float(r_values.mean()) if r_values.size else None
    profit_factor = None
    if losses.size and abs(losses.sum()) > 0:
        profit_factor = float(wins.sum() / abs(losses.sum())) if wins.size else 0.0
    t_stat = None
    if n >= 2:
        std = float(pnl.std(ddof=1))
        if std > 0:
            t_stat = float(pnl.mean() / (std / math.sqrt(n)))
    days = set(daily["date"]) if not daily.empty else set()
    traded_days = set()
    if not blotter.empty:
        for value in blotter["entry_ts"]:
            if value:
                traded_days.add(str(value)[:10])
    share = (len(traded_days & days) / len(days)) if days else None
    return {
        "n": n,
        "hit_rate": hit,
        "payoff": payoff,
        "expectancy_bps": expectancy_bps,
        "expectancy_r": expectancy_r,
        "profit_factor": profit_factor,
        "t_stat": t_stat,
        "pct_days_with_trade": share,
    }


def _strategy_score(report: BacktestReport, strategy_id: str) -> dict:
    if report.daily.empty:
        returns = np.asarray([])
    elif strategy_id == "book":
        returns = report.daily["return"].to_numpy(dtype=float)
    else:
        pnl = _pnl_by_day(report.blotter, strategy_id, report.daily["date"])
        base = report.daily["nav"].iloc[0] if len(report.daily) else 1.0
        # El retorno de la estrategia usa la caja inicial, no el NAV del
        # día: si no, el orden de las estrategias cambiaría el Sharpe.
        initial = float(report.provenance.get("initial_cash", base))
        if "initial_cash" not in report.provenance:
            initial = float(report.daily["nav"].iloc[0] - report.daily["pnl"].iloc[0])
        returns = pnl / initial if initial else pnl * 0.0
    sr, skew, kurt, length = sample_moments(returns)
    min_trl = None
    if length >= 2 and sr != 0.0:
        try:
            min_trl = min_track_record(sr, 0.0, skew, kurt)
        except ValueError:
            min_trl = None
    return {
        "sr": sr,
        "skew": skew,
        "kurtosis": kurt,
        "t": length,
        "sr_annualized": annualize_sharpe(sr, TRADING_DAYS) if length >= 2 else None,
        "min_trl": min_trl,
        "n_trials": None,
        "sr_variance": None,
        "dsr": None,
        "psr": None,
    }


def _score_series(returns: np.ndarray, n_trials: int, sr_variance) -> dict:
    sr, skew, kurt, length = sample_moments(returns)
    return {
        "sr": sr,
        "skew": skew,
        "kurtosis": kurt,
        "t": length,
        "n_trials": n_trials,
        "sr_variance": sr_variance,
    }


def _pnl_by_day(blotter: pd.DataFrame, strategy_id: str, dates: pd.Series) -> np.ndarray:
    totals = {day: 0.0 for day in dates}
    if blotter.empty:
        return np.asarray([totals[day] for day in dates], dtype=float)
    part = blotter[blotter["strategy_id"] == strategy_id]
    for row in part.itertuples(index=False):
        stamp = row.exit_ts or row.entry_ts
        if not stamp:
            continue
        day = str(stamp)[:10]
        if day in totals:
            totals[day] += float(row.pnl)
    return np.asarray([totals[day] for day in dates], dtype=float)


def _trades_of(blotter: pd.DataFrame, strategy_id: str) -> int:
    if blotter.empty:
        return 0
    part = blotter[(blotter["strategy_id"] == strategy_id) & (blotter["closed"] == True)]  # noqa: E712
    return int(len(part))


def _attribute(blotter: pd.DataFrame, column: str) -> pd.DataFrame:
    """P&L por ``column``: lo cerrado más lo abierto a la última marca.

    ``open_pnl`` es la parte abierta. Con ella la suma de ``pnl`` es la del
    daily aunque una posición no se haya cerrado. Antes solo contaban los
    cerrados y lo abierto quedaba escondido (HFT-PAPER-08, vwap +$54 con
    −$293 de IWM afuera). ``n`` y ``slippage_bps`` son de los cerrados.
    """
    columns = ["key", "pnl", "fees", "n", "slippage_bps", "open_pnl"]
    if blotter.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, part in blotter.groupby(column, sort=True):
        done = part["closed"] == True  # noqa: E712
        closed = part[done]
        rows.append({
            "key": key,
            "pnl": float(part["pnl"].sum()),
            "fees": float(part["fees"].sum()),
            "n": int(len(closed)),
            "slippage_bps": float(closed["slippage_bps"].mean()) if len(closed) else None,
            "open_pnl": float(part.loc[~done, "pnl"].sum()),
        })
    return pd.DataFrame(rows, columns=columns)


def _pnl_by_hour(blotter: pd.DataFrame) -> pd.DataFrame:
    columns = ["hour_et", "pnl", "n"]
    if blotter.empty:
        return pd.DataFrame(columns=columns)
    closed = blotter[blotter["closed"] == True]  # noqa: E712
    buckets: dict = {}
    for row in closed.itertuples(index=False):
        if not row.entry_ts:
            continue
        hour = datetime.fromisoformat(row.entry_ts).astimezone(NY).hour
        slot = buckets.setdefault(hour, {"pnl": 0.0, "n": 0})
        slot["pnl"] += float(row.pnl)
        slot["n"] += 1
    rows = [
        {"hour_et": hour, "pnl": buckets[hour]["pnl"], "n": buckets[hour]["n"]}
        for hour in sorted(buckets)
    ]
    return pd.DataFrame(rows, columns=columns)


def _spy_daily(frames) -> Optional[pd.Series]:
    if frames is None or "SPY" not in frames["close"].columns:
        return None
    close = frames["close"]["SPY"].dropna()
    if close.empty:
        return None
    ny = close.index.tz_convert(NY)
    daily = close.groupby(ny.date).last()
    daily.index = pd.to_datetime(daily.index)
    return daily


def _spy_from_report(report: BacktestReport) -> Optional[pd.Series]:
    """El CLI no guarda las barras. El estrés de régimen ya está en el reporte."""
    del report
    return None


def _data_version(provider, config: HftConfig, feed: str) -> dict:
    store = getattr(provider, "store", None)
    if store is None:
        return {"source": type(provider).__name__, "manifest_sha256": None, "files": []}
    path = Path(store.root) / "manifest.json"
    if not path.is_file():
        return {"source": "store", "manifest_sha256": None, "files": []}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        manifest = {}
    files = []
    wanted = {symbol.upper() for symbol in config_symbols(config)}
    step = canonical_timeframe(config.timeframe)
    for relative, meta in sorted((manifest.get("files") or {}).items()):
        parts = str(relative).split("/")
        if step not in parts or feed not in parts:
            continue
        if not any(part.upper() in wanted for part in parts):
            continue
        if not isinstance(meta, dict):
            continue
        files.append({
            "path": relative,
            "sha256": meta.get("sha256"),
            "rows": meta.get("rows"),
            "start": meta.get("start"),
            "end": meta.get("end"),
        })
    return {"source": "store", "manifest_sha256": digest, "files": files}


def _run_id(config, start, end, feed, cost_multiplier, extra, unlock, strategy_ids) -> str:
    payload = json.dumps(
        {
            "config_hash": config.config_hash(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": feed,
            "cost_multiplier": cost_multiplier,
            "extra_latency_bars": extra,
            "unlock_holdout": unlock,
            "strategies": list(strategy_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _excursions(frames, symbol, start, end, entry, side) -> Tuple[float, float]:
    if frames is None or entry <= 0:
        return 0.0, 0.0
    high = frames["high"]
    low = frames["low"]
    if symbol not in high.columns:
        return 0.0, 0.0
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    window_h = high[symbol].loc[lo:hi].dropna()
    window_l = low[symbol].loc[lo:hi].dropna()
    if window_h.empty or window_l.empty:
        return 0.0, 0.0
    if side == "long":
        adverse = float(window_l.min()) - entry
        favorable = float(window_h.max()) - entry
    else:
        adverse = entry - float(window_h.max())
        favorable = entry - float(window_l.min())
    return min(0.0, adverse / entry * 1e4), max(0.0, favorable / entry * 1e4)


def _price_at(frames, column, symbol, ts) -> Optional[float]:
    if frames is None or ts is None:
        return None
    return _price_at_index(frames[column], symbol, pd.Timestamp(ts))


def _price_at_index(frame, symbol, stamp: pd.Timestamp) -> Optional[float]:
    if symbol not in frame.columns:
        return None
    series = frame[symbol]
    if stamp not in series.index:
        return None
    value = series.loc[stamp]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _last_price(frames, symbol) -> Optional[float]:
    if frames is None or symbol not in frames["close"].columns:
        return None
    series = frames["close"][symbol].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _last_ts(frames):
    if frames is None or frames["close"].empty:
        return None
    return frames["close"].index[-1].to_pydatetime()


def _slip_bps(side: str, fill_price: float, decision: Optional[float]) -> float:
    if decision is None or decision <= 0:
        return 0.0
    direction = 1.0 if side == "buy" else -1.0
    return direction * (fill_price - decision) / decision * 1e4


def _lot_sign(qty: float) -> float:
    return 1.0 if qty > 0 else -1.0


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("un instante del blotter no trae zona")
    return ts.astimezone(UTC).isoformat()


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(NY).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    sys.exit(main())
