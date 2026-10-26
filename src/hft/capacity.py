"""Probe de capacidad: cuántas órdenes por segundo aguanta la ruta única.

No es una estrategia de trading. ``CapacityProbe`` emite ``orders_per_bar``
órdenes market de ``qty`` acciones por barra, repartidas por símbolo, y
alterna compra y venta para que el libro vuelva a cero. No abre cortos.
No se registra en ``STRATEGIES`` ni en ``WITH_STRATEGIES``: la usan
``scripts/hft_capacity_bench.py`` y ``scripts/hft_rate_probe.py``.

``probe_config`` sube los topes de riesgo que existen para frenar una
estrategia real (órdenes por minuto y por día, abiertas, pérdida diaria,
racha de pérdidas). La config de producción no cambia.

``run_capacity`` arma el motor con la ejecución que se le pase
(``SimExecution`` o ``AlpacaExecution`` contra el exchange falso o paper)
y mide cada ``Engine.step``: es la misma ruta que backtest, sim y paper.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Sequence, Tuple

from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, Session, bar_close_time
from src.hft.config import HftConfig
from src.hft.data.bars import IntradayBars
from src.hft.engine import Engine
from src.hft.risk import AssetFlag, EngineRiskGate, RiskManager, RiskState
from src.hft.sizing import IntradaySizer
from src.hft.types import TheoryCard, TheoryParam, order_intent

PROBE_ID = "capacity_probe"


@dataclass(frozen=True)
class CapacityParams:
    orders_per_bar: int
    qty: float


def _theory(params: CapacityParams) -> TheoryCard:
    return TheoryCard(
        hypothesis="No hay hipótesis de mercado: mide órdenes por segundo de la ruta única.",
        mechanism="Órdenes market de una acción que se compensan; nadie paga un edge.",
        references=("Plan de capacidad 2026-10-13",),
        data_requirements=("barras ya finales del motor",),
        params=(
            TheoryParam("orders_per_bar", params.orders_per_bar,
                        "Carga del probe, elegida por quien mide.", "plan 2026-10-13"),
            TheoryParam("qty", params.qty,
                        "Tamaño mínimo para no mover el libro.", "plan 2026-10-13"),
        ),
        holding_period="una barra",
        failure_regimes=("cualquier uso como estrategia",),
        kill_criteria=("nunca correrla en una sesión diaria",),
        research_only=False,
    )


class CapacityProbe:
    """Alterna compra y venta por símbolo. Plano al final de cada par."""

    id = PROBE_ID

    def __init__(self, symbols: Sequence[str], orders_per_bar: int, qty: float = 1.0):
        if not symbols:
            raise ValueError("hace falta al menos un símbolo")
        if isinstance(orders_per_bar, bool) or not isinstance(orders_per_bar, int) or orders_per_bar < 1:
            raise ValueError("orders_per_bar debe ser un entero >= 1")
        self.symbols = tuple(str(s).strip().upper() for s in symbols)
        self.params = CapacityParams(orders_per_bar, float(qty))
        self.theory = _theory(self.params)

    def required_history(self) -> Tuple[int, int]:
        return (1, 0)

    def on_bar(self, ctx) -> List:
        planned = {}
        out = []
        for index in range(self.params.orders_per_bar):
            symbol = self.symbols[index % len(self.symbols)]
            if symbol not in planned:
                planned[symbol] = ctx.position(symbol) + ctx.pending_qty(symbol)
            side = "sell" if planned[symbol] > 1e-9 else "buy"
            qty = self.params.qty
            planned[symbol] += qty if side == "buy" else -qty
            out.append(order_intent(
                self.id, symbol, side, qty, ctx.bar_ts, reason="probe de capacidad",
            ))
        return out


def probe_config(base: HftConfig, *, capital: float = 10_000_000.0) -> HftConfig:
    """``base`` con los topes de tasa y de pérdida fuera del camino."""
    big = 10**9
    risk = dataclasses.replace(
        base.risk,
        max_open_orders=big,
        max_orders_per_minute=big,
        max_orders_per_day=big,
        max_daily_loss_usd=float(big),
        max_daily_loss_pct=1.0,
        max_consecutive_losses=big,
        max_position_notional=float(capital),
    )
    sizing = dataclasses.replace(base.sizing, risk_capital_usd=float(capital), min_notional=1.0)
    return dataclasses.replace(base, risk=risk, sizing=sizing, initial_cash=float(capital))


@dataclass
class CapacityResult:
    bars: int = 0
    intents: int = 0
    submitted: int = 0
    fills: int = 0
    rejected: int = 0
    step_s: List[float] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return float(sum(self.step_s))

    def pct(self, q: float) -> float:
        if not self.step_s:
            return 0.0
        ordered = sorted(self.step_s)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    def orders_per_s(self) -> float:
        return self.submitted / self.total_s if self.total_s > 0 else 0.0

    def fills_per_s(self) -> float:
        return self.fills / self.total_s if self.total_s > 0 else 0.0


class _ProbeRisk:
    """Como ``backtest._RiskView``: marcas, caja y última barra antes del riesgo real."""

    def __init__(self, gate: EngineRiskGate) -> None:
        self.gate = gate
        self.marks: dict = {}

    def filter(self, intents, state, now):
        self.gate.state = dataclasses.replace(
            self.gate.state,
            marks=tuple(sorted(self.marks.items())),
            buying_power=max(0.0, float(state.cash)),
            last_bar_close=now,
        )
        return self.gate.filter(intents, state, now)


class _Clock:
    def __init__(self, moment: datetime):
        self.moment = moment

    def now(self) -> datetime:
        return self.moment


def _slice(bars: IntradayBars, end: int) -> IntradayBars:
    frames = {name: getattr(bars, name).iloc[end - 1:end]
              for name in ("open", "high", "low", "close", "volume", "vwap", "trades")}
    return IntradayBars(meta=bars.meta, **frames)


def run_capacity(
    execution: object,
    *,
    config: HftConfig,
    session: Session,
    bars: IntradayBars,
    orders_per_bar: int,
    mode: str = "sim",
    max_bars: Optional[int] = None,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> CapacityResult:
    """Corre ``CapacityProbe`` barra a barra por ``Engine.step`` y mide.

    Cada ``step`` recibe solo la barra nueva, como el vivo después del
    calentamiento. La ejecución sale en t+1, así que la primera barra no
    publica nada.
    """
    symbols = tuple(bars.close.columns)
    probe = CapacityProbe(symbols, orders_per_bar)
    state = RiskState(
        equity=config.initial_cash,
        nav=config.initial_cash,
        buying_power=config.initial_cash,
        session_start_pnl=0.0,
        cash=config.initial_cash,
        assets=tuple((symbol, AssetFlag(True, True)) for symbol in symbols),
    )
    manager = RiskManager(config.risk, (session,))
    risk = _ProbeRisk(EngineRiskGate(manager, state, equity_from_book=(mode != "paper")))
    clock = _Clock(session.open_utc)
    engine = Engine(
        [probe],
        lambda window: None,
        risk,
        IntradaySizer(config.sizing, max_position_notional=config.risk.max_position_notional),
        execution,
        FakeCalendar([session]),
        clock,
        book=HftBook(config.initial_cash),
        config=config,
        mode=mode,
    )
    result = CapacityResult()
    index = bars.close.index
    total = len(index) if max_bars is None else min(len(index), int(max_bars))
    for end in range(1, total + 1):
        ts = index[end - 1].to_pydatetime()
        now = bar_close_time(ts, config.timeframe) + timedelta(seconds=config.grace_s)
        clock.moment = now
        latest = _slice(bars, end)
        for symbol, price in latest.close.iloc[-1].items():
            if price == price and price > 0:
                risk.marks[str(symbol)] = float(price)
        started = time.perf_counter()
        step = engine.step(now, latest)
        elapsed = time.perf_counter() - started
        result.step_s.append(elapsed)
        result.bars += 1
        for event in step.events:
            result.intents += len(event.intents)
            result.submitted += len(event.orders)
            result.fills += len(event.fills)
            result.rejected += len(event.rejections)
        if on_step is not None:
            on_step(end, elapsed)
    return result
