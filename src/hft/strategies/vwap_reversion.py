"""Reversión a VWAP con filtro de régimen (HFT-09C).

Grossman y Miller (1988) describen a quien provee liquidez como el que
absorbe un choque de flujo no informativo y cobra cuando el precio vuelve.
Nagel (2012) muestra que el premio de reversión de corto plazo es mayor
cuando la liquidez se encarece. Esa reversión falla cuando el desvío es
información: el filtro de régimen (pendiente intradía baja y vol realizada
del día por debajo de un percentil) es parte de la hipótesis.

Reglas, fijadas de antemano:

- ``z_t = (C_t − VWAP_t) / σ_dev(N)`` con ``N = 30`` barras. ``σ_dev`` es
  la desviación típica de ``C − VWAP`` y no cruza la apertura. Historia
  corta o ``σ = 0`` es NaN y no hay entrada.
- Solo entre ``open + 15 min`` y ``entry_cutoff``, en el reloj de la
  decisión (``ctx.now``), el mismo corte que el riesgo.
- Régimen: ``|pendiente normalizada| < umbral`` y el percentil de la vol
  realizada del día, contra las sesiones previas y el mismo número de
  retornos desde la apertura, por debajo de un máximo. Cualquier NaN
  bloquea la entrada. La pendiente y el percentil adaptan
  ``src.regime_detector`` sin importarlo: ese módulo devuelve 0 o 50
  cuando falta historia, y aquí la historia corta sigue siendo NaN.
- Largo si ``z < −2``. Corto si ``z > +2`` y el símbolo está en el
  conjunto shortable inyectado (el cache de activos es de HFT-11).
  Salida si ``|z| < 0.5``. Stop si ``|z| > 3.5``. Stop de tiempo de 60
  minutos desde la primera barra en que ve la posición propia. Salidas,
  stop de tiempo y flatten miran solo lo que llenaron sus órdenes
  (``ctx.positions``, HFT-PAPER-08), no lo de otra estrategia.
  Flatten al cierre.
- Entrada ``marketable_limit``. La variante ``passive_limit`` es un
  experimento: ``research_only`` y el motivo lo dice. El paper no simula
  la cola.
- Universo del día: los ``top_k`` de menor spread de Corwin–Schultz. Un
  spread que no se puede estimar no entra.

El tamaño es riesgo fijo de ``sizing.py`` hasta el precio donde ``|z|``
llegaría al stop con el VWAP y la σ de la decisión. No es un bracket: el
stop vive en z y se revisa en cada barra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hft.calendar import Session, entry_cutoff, flatten_time, session_for
from src.hft.config import HftConfig, SizingConfig, VwapReversionConfig
from src.hft.data.bars import IntradayBars, canonical_timeframe
from src.hft.features import (
    DEFAULT_TREND_LOOK,
    DEFAULT_TREND_MA,
    DEFAULT_Z_BARS,
    corwin_schultz_spread,
    session_vwap,
    trend_score,
    vwap_zscore,
)
from src.hft.sizing import size_fixed_risk
from src.hft.strategies.base import StrategyContext, register
from src.hft.types import OrderIntent, TheoryCard, TheoryParam, order_intent

STRATEGY_ID = "vwap_reversion"

# Menos de esto el percentil no se publica. ``regime_detector._vol_percentile``
# pide 5 valores y, si faltan, devuelve 50. Aquí faltan y es NaN.
_MIN_VOL_RETURNS = 5
_DEFAULT_RISK_CAPITAL = 25_000.0
_DEFAULT_SIGMA_TARGET = 0.01
_DEFAULT_KELLY_C = 0.25
_DEFAULT_MIN_NOTIONAL = 100.0
_DEFAULT_MAX_LEVERAGE = 1.0
_DEFAULT_FLATTEN_MINUTES = 5.0
_DEFAULT_TICK = 0.01
_QTY_EPS = 1e-9
_PRICE_EPS = 1e-9
_ENTRY_STYLES = ("marketable_limit", "passive_limit")


@dataclass(frozen=True)
class VwapReversionParams:
    """Parámetros ex-ante. El valor vive también en la TheoryCard."""

    z_n: int = DEFAULT_Z_BARS
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    skip_open_minutes: int = 15
    entry_cutoff_minutes: float = 15.0
    time_stop_minutes: int = 60
    vol_sessions: int = 20
    max_vol_percentile: float = 70.0
    trend_ma: int = DEFAULT_TREND_MA
    trend_look: int = DEFAULT_TREND_LOOK
    trend_threshold: float = 0.5
    top_k: int = 10
    risk_frac: float = 0.01
    limit_through_ticks: int = 1
    entry_style: str = "marketable_limit"

    def __post_init__(self) -> None:
        _int_at_least("z_n", self.z_n, 2)
        _int_at_least("skip_open_minutes", self.skip_open_minutes, 0)
        _int_at_least("time_stop_minutes", self.time_stop_minutes, 1)
        _int_at_least("vol_sessions", self.vol_sessions, 1)
        _int_at_least("trend_ma", self.trend_ma, 1)
        _int_at_least("trend_look", self.trend_look, 2)
        _int_at_least("top_k", self.top_k, 1)
        _int_at_least("limit_through_ticks", self.limit_through_ticks, 0)
        for name in ("entry_z", "exit_z", "stop_z", "trend_threshold"):
            if not _positive_number(getattr(self, name)):
                raise ValueError(f"{name} debe ser un número > 0")
        if not self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("hace falta exit_z < entry_z < stop_z")
        if not _positive_number(self.entry_cutoff_minutes):
            raise ValueError("entry_cutoff_minutes debe ser un número > 0")
        if not _positive_number(self.max_vol_percentile) or self.max_vol_percentile > 100:
            raise ValueError("max_vol_percentile debe estar en (0, 100]")
        if not _positive_number(self.risk_frac) or self.risk_frac > 1:
            raise ValueError("risk_frac debe estar en (0, 1]")
        style = str(self.entry_style).strip().lower()
        if style not in _ENTRY_STYLES:
            raise ValueError(f"entry_style {self.entry_style!r} no está en {_ENTRY_STYLES}")
        object.__setattr__(self, "entry_style", style)


def theory_for(params: VwapReversionParams) -> TheoryCard:
    """TheoryCard cuyos valores son exactamente los del dataclass."""
    documented = []
    for field in fields(params):
        justification, source = _CITATIONS[field.name]
        documented.append(
            TheoryParam(field.name, getattr(params, field.name), justification, source)
        )
    passive = params.entry_style == "passive_limit"
    return TheoryCard(
        hypothesis=(
            "Un desvío del cierre respecto del VWAP de la sesión, medido en "
            "desviaciones de ese desvío, revierte cuando lo causó flujo no "
            "informativo. Se entra a favor de la reversión solo si la pendiente "
            "normalizada del precio intradía está por debajo del umbral y la "
            "vol realizada del día, contra las sesiones previas, queda por "
            "debajo del percentil máximo. La entrada es marketable_limit en la "
            "barra siguiente. La variante passive_limit es un experimento "
            "etiquetado: el paper no simula la cola y no se usa para capital."
        ),
        mechanism=(
            "Paga quien necesita liquidez ya y empuja el precio lejos del VWAP "
            "sin información. Cobra quien absorbe ese inventario y espera la "
            "reversión (Grossman y Miller, 1988). Nagel (2012) documenta que "
            "el premio de reversión de corto plazo aumenta cuando la liquidez "
            "se encarece. El filtro no persigue ese extremo: un día de "
            "tendencia o de vol realizada alta es información, no inventario, "
            "y ahí la reversión falla. Sin el filtro no hay quién pague de "
            "forma distinguible del flujo informado."
        ),
        references=(
            "Grossman, Sanford J. & Miller, Merton H. (1988), Liquidity and "
            "market structure, Journal of Finance 43(3), 617-633",
            "Nagel, Stefan (2012), Evaporating liquidity, Review of Financial "
            "Studies 25(7), 2005-2039",
        ),
        data_requirements=(
            "barras finales del timeframe con high, low, close y volume; "
            "el VWAP de sesión usa el vw de la barra si es positivo y si no "
            "el precio típico",
            "calendario de sesión inyectado, para la apertura, el corte y el flatten",
            "sesiones previas para el percentil de vol realizada; sin ellas no se entra",
            "conjunto shortable inyectado; sin el símbolo no se abre corto",
        ),
        params=tuple(documented),
        holding_period=(
            "De unos minutos a 60 minutos, y siempre plano en flatten_time. "
            "El cuadro de HFT-09C acota el holding entre 5 y 60 minutos. "
            "No se deja posición para la sesión siguiente."
        ),
        failure_regimes=(
            "tendencia intradía: |pendiente normalizada| en el umbral o por "
            "encima. El desvío del VWAP es información y no revierte",
            "vol realizada del día en el percentil máximo o por encima, frente "
            "a las sesiones previas. Nagel (2012) ve un premio mayor con "
            "liquidez cara; ese caso no se opera, porque el prompt deja la "
            "vol por debajo de un máximo y un día de información no es inventario",
            "apertura (primeros 15 minutos) y después del entry_cutoff: el "
            "VWAP todavía no ancla, o no hay tiempo de cobrar la reversión",
            "historia corta de z, de la pendiente o del percentil: NaN, sin entrada",
            "símbolo que no está entre los de menor spread estimado",
        ),
        kill_criteria=(
            "expectativa neta rolling de 60 sesiones < 0 con t < -1",
            "el filtro deja de separar: el P&L de los días con pendiente por "
            "encima del umbral, si se operaran, no es peor que el de los días "
            "que el filtro deja pasar",
        ),
        research_only=passive,
    )


_CITATIONS = {
    "z_n": (
        "N barras para la desviación típica de C − VWAP. 30 es el valor del "
        "prompt y el default de vwap_zscore. No se eligió mirando un resultado.",
        "HFT-09C; src.hft.features.DEFAULT_Z_BARS",
    ),
    "entry_z": (
        "Se entra cuando el cierre está al menos dos desviaciones del VWAP "
        "de la sesión. Es el umbral del prompt, no un cuantil estimado.",
        "HFT-09C",
    ),
    "exit_z": (
        "Se sale cuando el desvío ya se deshizo (|z| < 0.5). Cobrar la "
        "reversión y no esperar a que el precio cruce al otro lado.",
        "HFT-09C",
    ),
    "stop_z": (
        "Si |z| pasa de 3.5 el desvío se amplió: se deja de proveer liquidez. "
        "El prompt lo fija; no es un múltiplo optimizado.",
        "HFT-09C",
    ),
    "skip_open_minutes": (
        "No se entra en los primeros 15 minutos. La apertura mezcla subasta "
        "e información; el VWAP de ese tramo no es el ancla de Grossman y Miller.",
        "HFT-09C, open+15 min",
    ),
    "entry_cutoff_minutes": (
        "Minutos antes del cierre real en los que ya no se abre posición. "
        "15 coincide con risk.entry_cutoff_minutes: la estrategia y el "
        "gate cortan en el mismo instante.",
        "HFT-09C; HFT-08 risk.entry_cutoff_minutes",
    ),
    "time_stop_minutes": (
        "Si a los 60 minutos el desvío no volvió, se cierra. Es el techo del "
        "holding del cuadro de 09C (5–60 min), contado desde la barra en que "
        "el libro ya tiene la posición.",
        "HFT-09C, stop de tiempo de 60 min",
    ),
    "vol_sessions": (
        "El percentil de la vol realizada del día se mide contra las 20 "
        "sesiones previas, el mismo horizonte de barras desde la apertura. "
        "Con menos sesiones el percentil es NaN y no se entra.",
        "HFT-09C, 20 sesiones previas",
    ),
    "max_vol_percentile": (
        "Solo se entra si el percentil queda estrictamente por debajo de 70. "
        "70 es el corte high_vol de regime_detector. Nagel (2012) asocia un "
        "premio mayor a la liquidez cara; el prompt igual pone un máximo, "
        "porque la vol extrema se trata como día de información.",
        "HFT-09C; src.regime_detector._DEFAULT_VOL_PCTILE_HIGH = 70 "
        "(no se importa el módulo)",
    ),
    "trend_ma": (
        "Media móvil sobre la que se mide la pendiente intradía. 50 es la MA "
        "del trend score de regime_detector y el default de features.trend_score.",
        "src.regime_detector._DEFAULT_MA_TREND; src.hft.features.DEFAULT_TREND_MA",
    ),
    "trend_look": (
        "Barras de la recta de mínimos cuadrados sobre esa media. 21 es la "
        "ventana de _trend_score y el default de features.trend_score.",
        "src.regime_detector._trend_score (look = min(21, ...)); "
        "src.hft.features.DEFAULT_TREND_LOOK",
    ),
    "trend_threshold": (
        "Por debajo de 0.5, en valor absoluto, el día no se clasifica como "
        "tendencia. Es el corte de regime_detector. En el umbral o por "
        "encima no se entra: el filtro es estricto.",
        "src.regime_detector._DEFAULT_TREND_THRESHOLD = 0.5 "
        "(no se importa el módulo)",
    ),
    "top_k": (
        "Como máximo los 10 nombres de menor spread de Corwin–Schultz en la "
        "barra. 10 es el tamaño del universo HFT (D-03). Un spread NaN no "
        "compite: sin estimador no hay liquidez que cobrar.",
        "HFT-09C, spread estimado mínimo; D-03",
    ),
    "risk_frac": (
        "Riesgo fijo del 1% del equity de riesgo hasta el precio en el que "
        "|z| llegaría al stop con el VWAP y la σ de la decisión. La misma "
        "fracción que sizing.risk_frac. El prompt no da otro tamaño.",
        "HFT-08 sizing.risk_frac; HFT-09C",
    ),
    "limit_through_ticks": (
        "En marketable_limit el límite cruza un tick el cierre, para alcanzar "
        "el medio spread mínimo del modelo (un tick). En passive_limit el "
        "mismo tick queda del lado que no cruza: es el experimento, no una "
        "entrada que el simulador trate como toma de liquidez.",
        "HFT-09C; HFT-07, piso de spread = 1 tick",
    ),
    "entry_style": (
        "marketable_limit es la entrada del prompt. passive_limit es solo un "
        "experimento etiquetado: research_only pasa a True y el motivo de la "
        "orden lo dice. El paper no simula la cola, así que esa variante no "
        "informa una decisión de capital.",
        "HFT-09C",
    ),
}


def realized_vol_percentile(
    close: pd.DataFrame,
    sessions: Sequence[Session],
    symbol: str,
    as_of: datetime,
    n_prior: int,
    *,
    min_returns: int = _MIN_VOL_RETURNS,
) -> float:
    """Percentil de la vol realizada de la sesión en curso.

    En ``as_of`` la vol es la desviación típica (``ddof=1``) de los
    log-retornos desde la apertura hasta esa barra. Cada sesión previa
    se mide con el mismo número de barras desde su apertura. Hacen falta
    las ``n_prior`` sesiones anteriores, todas con ese largo, y al menos
    ``min_returns`` retornos. Si falta algo, NaN: no se inventa un 50.
    """
    if n_prior < 1 or min_returns < 2:
        return float("nan")
    if symbol not in close.columns or close.empty:
        return float("nan")
    stamp = pd.Timestamp(as_of)
    series = close.loc[:stamp, symbol]
    today, prior = _split_sessions(series, sessions, stamp)
    if today is None:
        return float("nan")
    today_rv = _rv(today.to_numpy(dtype=float), min_returns)
    if today_rv is None:
        return float("nan")
    window = prior[-n_prior:]
    if len(window) < n_prior:
        return float("nan")
    need = len(today)
    sample = []
    for part in window:
        values = part.to_numpy(dtype=float)
        if len(values) < need:
            return float("nan")
        rv = _rv(values[:need], min_returns)
        if rv is None:
            return float("nan")
        sample.append(rv)
    hist = np.asarray(sample, dtype=float)
    return float(np.mean(hist <= today_rv) * 100.0)


def _default_sizing(risk_frac: float) -> SizingConfig:
    return SizingConfig(
        risk_capital_usd=_DEFAULT_RISK_CAPITAL,
        risk_frac=float(risk_frac),
        sigma_target=_DEFAULT_SIGMA_TARGET,
        kelly_c=_DEFAULT_KELLY_C,
        min_notional=_DEFAULT_MIN_NOTIONAL,
        max_leverage=_DEFAULT_MAX_LEVERAGE,
    )


def params_from_config(cfg: VwapReversionConfig) -> VwapReversionParams:
    """Copia los campos de ``[vwap_reversion]``. Los nombres coinciden."""
    payload = {field.name: getattr(cfg, field.name) for field in fields(VwapReversionParams)}
    return VwapReversionParams(**payload)


class VwapReversion:
    """Reversión a VWAP. ``on_bar`` solo lee el contexto y el calendario inyectado.

    Sin sesiones cargadas no opera. ``shortable`` es el conjunto de ids que
    el cache de activos marcó como shortable; vacío significa que no se
    abre ningún corto. ``_opened`` recuerda la primera barra en que el
    libro ya tenía la posición, para el stop de 60 minutos. Se actualiza
    con la posición que el motor ya aplicó, no con el reloj del sistema.
    """

    id = STRATEGY_ID

    def __init__(
        self,
        params: Optional[VwapReversionParams] = None,
        *,
        sessions: Sequence[Session] = (),
        flatten_minutes: float = _DEFAULT_FLATTEN_MINUTES,
        sizing: Optional[SizingConfig] = None,
        timeframe: str = "1Min",
        tick: float = _DEFAULT_TICK,
        shortable: Sequence[str] = (),
    ) -> None:
        self.params = VwapReversionParams() if params is None else params
        if not isinstance(self.params, VwapReversionParams):
            raise TypeError("params debe ser VwapReversionParams")
        self.theory = theory_for(self.params)
        self.sessions = tuple(sessions)
        for session in self.sessions:
            if not isinstance(session, Session):
                raise TypeError("sessions debe ser una secuencia de Session")
        if isinstance(flatten_minutes, bool) or not isinstance(flatten_minutes, (int, float)):
            raise TypeError("flatten_minutes debe ser numérico")
        if float(flatten_minutes) < 0:
            raise ValueError("flatten_minutes debe ser >= 0")
        self.flatten_minutes = float(flatten_minutes)
        if self.params.entry_cutoff_minutes + 1e-12 < self.flatten_minutes:
            raise ValueError(
                "entry_cutoff_minutes debe ser >= flatten_minutes "
                "(el corte de entradas no puede ir después del flatten)"
            )
        self.sizing = _default_sizing(self.params.risk_frac) if sizing is None else sizing
        if not isinstance(self.sizing, SizingConfig):
            raise TypeError("sizing debe ser SizingConfig")
        if abs(self.sizing.risk_frac - self.params.risk_frac) > 1e-12:
            raise ValueError("sizing.risk_frac debe coincidir con params.risk_frac")
        self.timeframe = canonical_timeframe(timeframe)
        self._bar_minutes = _bar_minutes(self.timeframe)
        if isinstance(tick, bool) or not isinstance(tick, (int, float)) or float(tick) <= 0:
            raise ValueError("tick debe ser > 0")
        self.tick = float(tick)
        names = []
        for symbol in shortable:
            text = str(symbol).strip().upper()
            if not text:
                raise ValueError("shortable tiene un símbolo vacío")
            names.append(text)
        self.shortable = frozenset(names)
        self._opened: dict = {}

    @classmethod
    def from_config(
        cls,
        config: HftConfig,
        sessions: Sequence[Session] = (),
        shortable: Sequence[str] = (),
    ) -> "VwapReversion":
        """Instancia con ``[vwap_reversion]``, el sizing, el tick y el flatten."""
        if not isinstance(config, HftConfig):
            raise TypeError("config debe ser HftConfig")
        return cls(
            params_from_config(config.vwap_reversion),
            sessions=sessions,
            flatten_minutes=config.risk.flatten_minutes,
            sizing=config.sizing,
            timeframe=config.timeframe,
            tick=config.costs.tick,
            shortable=shortable,
        )

    def required_history(self) -> Tuple[int, int]:
        signal = max(
            self.params.z_n,
            self.params.trend_ma + self.params.trend_look - 1,
            _MIN_VOL_RETURNS + 1,
        )
        sessions = self.params.vol_sessions + 1
        bars_per_session = math.ceil(390 / self._bar_minutes)
        return (max(signal, bars_per_session * sessions), sessions)

    def on_bar(self, ctx: StrategyContext) -> Sequence[OrderIntent]:
        """Entradas y salidas con barras de inicio ``<= ctx.bar_ts``.

        El futuro de esa barra, si viniera en el frame, se recorta.
        """
        session = session_for(self.sessions, ctx.bar_ts)
        if session is None:
            return ()
        self._sync_opened(ctx)
        if ctx.now >= flatten_time(session, self.flatten_minutes):
            return tuple(self._flatten(ctx))
        bars = _until(ctx.bars, ctx.bar_ts)
        stamp = pd.Timestamp(ctx.bar_ts)
        if bars.close.empty or stamp not in bars.close.index:
            return ()
        z = self._z(bars)
        orders = self._exits(ctx, z, stamp)
        if self._in_window(ctx, session):
            exiting = {order.symbol for order in orders}
            orders.extend(self._entries(ctx, bars, session, z, stamp, exiting))
        return tuple(orders)

    def _sync_opened(self, ctx: StrategyContext) -> None:
        held = set()
        for symbol, qty in ctx.positions:
            if abs(qty) <= _QTY_EPS:
                continue
            held.add(symbol)
            if symbol not in self._opened:
                self._opened[symbol] = ctx.bar_ts
        for symbol in [name for name in self._opened if name not in held]:
            del self._opened[symbol]

    def _in_window(self, ctx: StrategyContext, session: Session) -> bool:
        start = session.open_utc + timedelta(minutes=self.params.skip_open_minutes)
        cutoff = entry_cutoff(session, self.params.entry_cutoff_minutes)
        return start <= ctx.now < cutoff

    def _z(self, bars: IntradayBars) -> pd.DataFrame:
        vwap = session_vwap(
            bars.high,
            bars.low,
            bars.close,
            bars.volume,
            alpaca_vw=bars.vwap,
            sessions=self.sessions,
        )
        return vwap_zscore(
            bars.close,
            vwap,
            n=self.params.z_n,
            by_session=True,
            sessions=self.sessions,
        )

    def _exits(
        self,
        ctx: StrategyContext,
        z: pd.DataFrame,
        stamp: pd.Timestamp,
    ) -> list:
        orders = []
        for symbol, _held in ctx.positions:
            # Una salida que sigue viva ya cubre la posición (HFT-PAPER-16).
            qty = ctx.exit_qty(symbol)
            if abs(qty) <= _QTY_EPS:
                continue
            reason = self._exit_reason(symbol, z, stamp, ctx.bar_ts)
            if reason is None:
                continue
            side = "sell" if qty > 0 else "buy"
            orders.append(order_intent(
                self.id,
                symbol,
                side,
                abs(qty),
                ctx.bar_ts,
                order_type="market",
                tif="day",
                reason=reason,
            ))
        orders.sort(key=lambda item: item.symbol)
        return orders

    def _exit_reason(
        self,
        symbol: str,
        z: pd.DataFrame,
        stamp: pd.Timestamp,
        bar_ts: datetime,
    ) -> Optional[str]:
        value = _cell(z, stamp, symbol)
        if value is not None and abs(value) > self.params.stop_z:
            return f"vwap stop_z z={value:.4g}"
        if value is not None and abs(value) < self.params.exit_z:
            return f"vwap exit_z z={value:.4g}"
        opened = self._opened.get(symbol)
        if opened is not None:
            held = bar_ts - opened
            if held >= timedelta(minutes=self.params.time_stop_minutes):
                return f"vwap time_stop {self.params.time_stop_minutes}min"
        return None

    def _entries(
        self,
        ctx: StrategyContext,
        bars: IntradayBars,
        session: Session,
        z: pd.DataFrame,
        stamp: pd.Timestamp,
        exiting: set,
    ) -> list:
        equity = _equity(ctx, bars)
        if equity <= 0:
            return []
        trend = trend_score(
            bars.close,
            ma_window=self.params.trend_ma,
            look=self.params.trend_look,
            by_session=True,
            sessions=self.sessions,
        )
        liquid = _liquid(bars, stamp, self.params.top_k)
        orders = []
        for symbol in liquid:
            if symbol in exiting or abs(ctx.position(symbol)) > _QTY_EPS:
                continue
            # Una entrada que sigue viva todavía no está en el libro.
            if abs(ctx.pending_qty(symbol)) > _QTY_EPS:
                continue
            if not self._regime_ok(bars, trend, stamp, symbol):
                continue
            value = _cell(z, stamp, symbol)
            if value is None:
                continue
            if value < -self.params.entry_z:
                side = "buy"
            elif value > self.params.entry_z and symbol in self.shortable:
                side = "sell"
            else:
                continue
            order = self._entry(bars, ctx.bar_ts, symbol, side, value, equity)
            if order is not None:
                orders.append(order)
        return orders

    def _regime_ok(
        self,
        bars: IntradayBars,
        trend: pd.DataFrame,
        stamp: pd.Timestamp,
        symbol: str,
    ) -> bool:
        slope = _cell(trend, stamp, symbol)
        if slope is None or abs(slope) >= self.params.trend_threshold:
            return False
        percentile = realized_vol_percentile(
            bars.close,
            self.sessions,
            symbol,
            stamp.to_pydatetime(),
            self.params.vol_sessions,
        )
        if not math.isfinite(percentile):
            return False
        return percentile < self.params.max_vol_percentile

    def _entry(
        self,
        bars: IntradayBars,
        bar_ts: datetime,
        symbol: str,
        side: str,
        z_value: float,
        equity: float,
    ) -> Optional[OrderIntent]:
        stamp = pd.Timestamp(bar_ts)
        close = _cell(bars.close, stamp, symbol)
        vwap = _vwap_at(bars, self.sessions, stamp, symbol)
        if close is None or vwap is None or abs(z_value) <= _PRICE_EPS:
            return None
        sigma = (close - vwap) / z_value
        if not math.isfinite(sigma) or sigma <= 0:
            return None
        if side == "buy":
            stop_raw = vwap - self.params.stop_z * sigma
        else:
            stop_raw = vwap + self.params.stop_z * sigma
        stop = _round_tick(stop_raw, self.tick)
        bump = self.params.limit_through_ticks * self.tick
        passive = self.params.entry_style == "passive_limit"
        if passive:
            raw_limit = close - bump if side == "buy" else close + bump
            order_type = "limit"
        else:
            raw_limit = close + bump if side == "buy" else close - bump
            order_type = "marketable_limit"
        limit = _round_tick(raw_limit, self.tick)
        if limit <= 0 or stop <= 0 or abs(limit - stop) <= _PRICE_EPS:
            return None
        # El stop de un largo tiene que quedar debajo del límite, y el de
        # un corto por encima. Si el redondeo los cruza, no hay distancia.
        if side == "buy" and stop >= limit:
            return None
        if side == "sell" and stop <= limit:
            return None
        qty = size_fixed_risk(
            self.sizing,
            equity,
            limit,
            stop,
            whole_shares=True,
        )
        if qty <= 0:
            return None
        tag = " experimento_pasivo" if passive else ""
        return order_intent(
            self.id,
            symbol,
            side,
            qty,
            bar_ts,
            order_type=order_type,
            limit_price=limit,
            tif="day",
            reason=(
                f"vwap {side} z={z_value:.4g} limite={limit:.4g} "
                f"stop_z={stop:.4g}{tag}"
            ),
        )

    def _flatten(self, ctx: StrategyContext) -> list:
        orders = []
        for symbol, _held in ctx.positions:
            qty = ctx.exit_qty(symbol)
            if abs(qty) <= _QTY_EPS:
                continue
            side = "sell" if qty > 0 else "buy"
            orders.append(order_intent(
                self.id,
                symbol,
                side,
                abs(qty),
                ctx.bar_ts,
                order_type="market",
                tif="day",
                reason="vwap flatten_time",
            ))
        orders.sort(key=lambda item: item.symbol)
        return orders


def _split_sessions(
    series: pd.Series,
    sessions: Sequence[Session],
    stamp: pd.Timestamp,
) -> Tuple[Optional[pd.Series], list]:
    """Sesión que contiene ``stamp`` y las anteriores, en orden."""
    today = None
    prior = []
    for session in sorted(sessions, key=lambda item: item.open_utc):
        start = pd.Timestamp(session.open_utc)
        end = pd.Timestamp(session.close_utc)
        mask = (series.index >= start) & (series.index < end) & (series.index <= stamp)
        part = series.loc[mask]
        if start <= stamp < end:
            today = part
        elif end <= stamp:
            prior.append(part)
    return today, prior


def _rv(values: np.ndarray, min_returns: int) -> Optional[float]:
    if values.size < min_returns + 1:
        return None
    if not np.isfinite(values).all() or np.any(values <= 0):
        return None
    returns = np.diff(np.log(values))
    if returns.size < min_returns:
        return None
    scale = float(np.std(returns, ddof=1))
    if not math.isfinite(scale):
        return None
    return scale


def _liquid(bars: IntradayBars, stamp: pd.Timestamp, top_k: int) -> Tuple[str, ...]:
    spread = corwin_schultz_spread(bars.high, bars.low)
    if stamp not in spread.index:
        return ()
    ranked = []
    row = spread.loc[stamp]
    for symbol in row.index:
        value = _finite_or_none(row[symbol])
        if value is None:
            continue
        ranked.append((float(value), str(symbol)))
    ranked.sort()
    return tuple(symbol for _, symbol in ranked[:top_k])


def _vwap_at(
    bars: IntradayBars,
    sessions: Sequence[Session],
    stamp: pd.Timestamp,
    symbol: str,
) -> Optional[float]:
    vwap = session_vwap(
        bars.high,
        bars.low,
        bars.close,
        bars.volume,
        alpaca_vw=bars.vwap,
        sessions=sessions,
    )
    return _cell(vwap, stamp, symbol)


def _equity(ctx: StrategyContext, bars: IntradayBars) -> float:
    """Caja más valor de mercado al cierre de la barra de decisión."""
    equity = float(ctx.cash)
    last = bars.close.iloc[-1]
    # Libro entero: el sizing no depende de quién tiene cada posición.
    for symbol, qty in ctx.book_items():
        if symbol not in last.index:
            continue
        price = _finite_or_none(last[symbol])
        if price is None:
            continue
        equity += float(qty) * price
    return equity


def _cell(frame: pd.DataFrame, stamp: pd.Timestamp, symbol: str) -> Optional[float]:
    if stamp not in frame.index or symbol not in frame.columns:
        return None
    return _finite_or_none(frame.at[stamp, symbol])


def _until(bars: IntradayBars, bar_ts: datetime) -> IntradayBars:
    stamp = pd.Timestamp(bar_ts)
    mask = bars.close.index <= stamp
    if bool(mask.all()):
        return bars
    frames = {
        name: getattr(bars, name).loc[mask]
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    return IntradayBars(meta=bars.meta, **frames)


def _finite_or_none(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _round_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 0.5) * tick


def _bar_minutes(timeframe: str) -> int:
    text = str(timeframe)
    if text.endswith("Min") and text[:-3].isdigit():
        minutes = int(text[:-3])
        if 1 <= minutes < 60:
            return minutes
    raise ValueError(f"timeframe {timeframe!r} no es NMin")


def _int_at_least(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}")


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _register_default() -> VwapReversion:
    strategy = VwapReversion()
    register(strategy)
    return strategy


DEFAULT_VWAP_REVERSION = _register_default()
