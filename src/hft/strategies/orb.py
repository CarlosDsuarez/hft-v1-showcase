"""Opening Range Breakout en stocks in play (HFT-09A).

El desbalance de los primeros minutos, cuando el volumen es anómalo,
sigue en la misma dirección porque esa información se incorpora con
retraso. Los parámetros salen de Zarattini, Barbon y Aziz (2024,
SSRN 4729284) y no se optimizan:

- Universo (sec. 2.1): precio > 5 USD, ADV(14) >= 1.000.000 acciones,
  ATR(14) > 0.50 USD. Encima, el percentil de Amihud de
  ``src.universe_filter`` (R-03) sobre sesiones ya cerradas.
- Rango: primeros 5 minutos. RVOL = volumen de ese tramo / media del
  mismo tramo en las 14 sesiones previas (sec. 3). Se opera el top 20
  con RVOL >= 1 (sec. 4).
- Dirección: signo(close_5 − open_5). Doji (open = close) no opera
  (sec. 2.1).
- Entrada: al cierre de la barra de las 09:34, ``marketable_limit``.
  El motor la ejecuta en la barra siguiente (09:35). El paper coloca
  un stop en el high o el low del rango y espera la ruptura; esa
  variante no se usa aquí.
- Stop: 10% del ATR(14) diario desde el precio de la orden (sec. 2.1
  y figura 2). Sin take-profit. En paper el bracket viaja con la
  orden, en acciones enteras.
- Salida: ese stop, o una orden market en ``flatten_time``.
- Tamaño: riesgo fijo del 1% de ``equity_riesgo`` (``sizing.py``),
  con el tope de apalancamiento de la config.

Ajuste. El paper usó IQFeed sin ajustar por splits (sec. 2). Aquí el
ATR, el ADV, el Amihud y el volumen histórico del RVOL salen de barras
``adjustment=split``. El rango de apertura del día en curso sale de
raw: ese día es el mismo en raw y en ajustado (HFT-04). Mezclar el
salto del split con el ATR movería el stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hft.calendar import Session, flatten_time, session_for
from src.hft.config import HftConfig, OrbConfig, SizingConfig
from src.hft.data.bars import IntradayBars, canonical_timeframe
from src.hft.features import relative_volume, wilder_atr
from src.hft.sizing import size_fixed_risk
from src.hft.strategies.base import StrategyContext, register
from src.hft.types import Bracket, OrderIntent, TheoryCard, TheoryParam, order_intent
from src.universe_filter import filter_universe_at_date

STRATEGY_ID = "orb"

# Misma caja de riesgo que config/hft.toml [sizing]. ``from_config``
# pisa estos números con el archivo.
_DEFAULT_RISK_CAPITAL = 25_000.0
_DEFAULT_SIGMA_TARGET = 0.01
_DEFAULT_KELLY_C = 0.25
_DEFAULT_MIN_NOTIONAL = 100.0
_DEFAULT_MAX_LEVERAGE = 1.0
_DEFAULT_TICK = 0.01
_DEFAULT_FLATTEN_MINUTES = 5.0

_PRICE_EPS = 1e-9
_QTY_EPS = 1e-9


@dataclass(frozen=True)
class OrbParams:
    """Parámetros ex-ante. El valor vive también en la TheoryCard."""

    opening_minutes: int = 5
    rvol_sessions: int = 14
    min_rvol: float = 1.0
    top_k: int = 20
    min_price: float = 5.0
    min_adv_shares: int = 1000000
    adv_sessions: int = 14
    min_atr: float = 0.5
    atr_sessions: int = 14
    stop_atr_frac: float = 0.1
    risk_frac: float = 0.01
    doji_eps: float = 0.0
    amihud_percentile: float = 90.0
    limit_through_ticks: int = 1

    def __post_init__(self) -> None:
        for name in (
            "opening_minutes",
            "rvol_sessions",
            "top_k",
            "min_adv_shares",
            "adv_sessions",
            "atr_sessions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} debe ser un entero >= 1")
        if not _positive_number(self.min_rvol) or self.min_rvol <= 0:
            raise ValueError("min_rvol debe ser > 0")
        if not _positive_number(self.min_price) or self.min_price <= 0:
            raise ValueError("min_price debe ser > 0")
        if not _positive_number(self.min_atr) or self.min_atr <= 0:
            raise ValueError("min_atr debe ser > 0")
        if not _positive_number(self.stop_atr_frac) or not 0 < self.stop_atr_frac <= 1:
            raise ValueError("stop_atr_frac debe estar en (0, 1]")
        if not _positive_number(self.risk_frac) or not 0 < self.risk_frac <= 1:
            raise ValueError("risk_frac debe estar en (0, 1]")
        if not _positive_number(self.doji_eps) or self.doji_eps < 0:
            raise ValueError("doji_eps debe ser >= 0")
        if not _positive_number(self.amihud_percentile) or not 0 < self.amihud_percentile <= 100:
            raise ValueError("amihud_percentile debe estar en (0, 100]")
        ticks = self.limit_through_ticks
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
            raise ValueError("limit_through_ticks debe ser un entero >= 0")


def _positive_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def theory_for(params: OrbParams) -> TheoryCard:
    """TheoryCard cuyos valores son exactamente los del dataclass."""
    cited = {item[0]: item for item in _CITATIONS}
    documented = []
    for field in fields(params):
        name, justification, source = cited[field.name]
        documented.append(TheoryParam(name, getattr(params, name), justification, source))
    return TheoryCard(
        hypothesis=(
            "El desbalance de la subasta y de los primeros minutos en acciones "
            "con volumen anómalo (stocks in play) refleja información que el "
            "mercado incorpora con retraso, así que el precio sigue en la "
            "dirección de esa apertura. La entrada de este módulo se decide "
            "al cierre de la barra que completa el rango y sale como "
            "marketable_limit en la barra siguiente. El paper, en cambio, "
            "deja un stop en el máximo o el mínimo del rango y espera la "
            "ruptura (sección 2.1, Entry Conditions); esa espera no se opera aquí."
        ),
        mechanism=(
            "Paga quien incorpora tarde la información de la noche y del "
            "desbalance de la apertura. El volumen anómalo marca que el "
            "catalizador todavía no estaba en el precio (Zarattini, Barbon "
            "y Aziz, sección 3). Sin ese retraso no hay momentum que cobrar."
        ),
        references=(
            "Zarattini, Barbon & Aziz (2024), A Profitable Day Trading Strategy "
            "For The U.S. Equity Market, SSRN 4729284, secciones 2.1, 3 y 4",
            "Crabel (1990), Day Trading with Short Term Price Patterns and "
            "Opening Range Breakout",
            "Wilder (1978), New Concepts in Technical Trading Systems, ATR "
            "(nota 1 del paper)",
        ),
        data_requirements=(
            "barras de 1 minuto raw de la sesión en curso para el rango de apertura",
            "ATR y ADV de sesiones previas con adjustment=split",
            "RVOL del tramo de apertura contra las sesiones previas, con el "
            "volumen histórico ajustado por splits",
            "símbolos de universe_hft.csv, ya como columnas de las barras",
        ),
        params=tuple(documented),
        holding_period=(
            "Desde la barra siguiente al cierre del rango hasta el stop o el "
            "flatten_time de la misma sesión. El paper cierra a las 16:00 ET "
            "si el stop no saltó (sección 2.1). Aquí la salida a mercado usa "
            "el flatten del riesgo, por defecto 5 minutos antes del cierre "
            "real, para que la orden quepa en t+1."
        ),
        failure_regimes=(
            "RVOL del rango por debajo de 100%: el paper mide en media -0.02R (sección 3, figura 4)",
            "doji en el rango (open = close): no hay dirección (sección 2.1)",
            "catalizador ya descontado: el volumen no se dispara y el filtro de stocks in play no deja entrar",
            "precio, ADV o ATR por debajo de los mínimos de la sección 2.1",
        ),
        kill_criteria=(
            "expectativa neta rolling de 60 sesiones < 0 con t < -1",
            "deterioro de la tasa de acierto > 2σ respecto del backtest",
        ),
        research_only=False,
    )


# nombre, justificación, fuente. Una entrada por campo de OrbParams.
_CITATIONS: Tuple[Tuple[str, str, str], ...] = (
    (
        "opening_minutes",
        "El rango es el de los primeros 5 minutos. El paper centra el "
        "resultado en ese plazo y muestra que 15, 30 y 60 minutos rinden menos.",
        "Zarattini, Barbon & Aziz (2024), secciones 2 y 4; tabla 3",
    ),
    (
        "rvol_sessions",
        "El denominador del Relative Volume es la media del mismo tramo "
        "en los 14 días previos, sin incluir la sesión en curso.",
        "Zarattini, Barbon & Aziz (2024), sección 3, fórmula de Relative Volume",
    ),
    (
        "min_rvol",
        "Por debajo del 100% la media del trade es -0.02R; desde 100% pasa "
        "a 0.08R. El filtro de la sección 4 exige al menos 100%.",
        "Zarattini, Barbon & Aziz (2024), sección 3 figura 4 y sección 4 punto 4",
    ),
    (
        "top_k",
        "Entre los nombres con RVOL >= 100% se opera un top fijo, no una "
        "frontera optimizada. El paper usa los 20 de mayor Relative Volume.",
        "Zarattini, Barbon & Aziz (2024), sección 4 punto 5",
    ),
    (
        "min_price",
        "El precio de apertura tiene que estar por encima de 5 USD. En "
        "pre-mercado se mira el cierre previo; a la decisión también el "
        "open de la sesión, que ya es conocido.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 requisito 1",
    ),
    (
        "min_adv_shares",
        "El volumen medio de los 14 días previos tiene que ser de al menos "
        "1.000.000 de acciones. Descarta penny stocks y nombres sin liquidez.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 requisito 2",
    ),
    (
        "adv_sessions",
        "La ventana del ADV y del filtro de liquidez es de 14 sesiones, "
        "la misma del requisito de volumen y del ATR.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 requisitos 2 y 3",
    ),
    (
        "min_atr",
        "El ATR de 14 días tiene que ser estrictamente mayor que 0.50 USD. "
        "Sin ese rango el stop de 10% queda dentro del ruido.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 requisito 3",
    ),
    (
        "atr_sessions",
        "Wilder define el ATR con 14 sesiones. El paper usa esa ventana y "
        "solo sesiones ya cerradas: el rango de hoy no entra en el stop de hoy.",
        "Wilder (1978); Zarattini, Barbon & Aziz (2024), sección 2.1 nota 1",
    ),
    (
        "stop_atr_frac",
        "El stop se coloca a un 10% del ATR diario desde el precio de "
        "entrada. No hay take-profit: si el stop no salta, se cierra al final.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 y figura 2",
    ),
    (
        "risk_frac",
        "Si el stop se cumple, la pérdida prevista es el 1% del capital "
        "asignado a la estrategia. El tope de apalancamiento puede dejarla "
        "por debajo de ese 1%, como ya advierte la nota 3 del paper.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 Position Sizing",
    ),
    (
        "doji_eps",
        "Si la primera vela abre y cierra igual, no hay orden. Cero exige "
        "esa igualdad; un epsilon mayor sería otra regla y no está en el paper.",
        "Zarattini, Barbon & Aziz (2024), sección 2.1 Entry Conditions",
    ),
    (
        "amihud_percentile",
        "El percentil 90 de Amihud es el corte de src.universe_filter. "
        "Se aplica solo a sesiones previas. El paper no usa Amihud: su "
        "liquidez es el ADV de 1.000.000 de acciones. Esto añade R-03.",
        "src.universe_filter.filter_universe_at_date, umbral por defecto 90",
    ),
    (
        "limit_through_ticks",
        "Un tick del lado agresivo hace que el marketable_limit cruce el "
        "medio spread mínimo del modelo de costos, que es un tick. El paper "
        "no usa límite: entra con un stop en el extremo del rango. No es "
        "un parámetro buscado.",
        "HFT-09A (marketable_limit en t+1); HFT-07 piso de spread = 1 tick; "
        "Zarattini et al. sección 2.1 usa stop en el high/low, no este límite",
    ),
)


def _default_sizing(risk_frac: float) -> SizingConfig:
    return SizingConfig(
        risk_capital_usd=_DEFAULT_RISK_CAPITAL,
        risk_frac=float(risk_frac),
        sigma_target=_DEFAULT_SIGMA_TARGET,
        kelly_c=_DEFAULT_KELLY_C,
        min_notional=_DEFAULT_MIN_NOTIONAL,
        max_leverage=_DEFAULT_MAX_LEVERAGE,
    )


def params_from_config(orb: OrbConfig) -> OrbParams:
    """Copia los campos de ``[orb]``. Los nombres coinciden a propósito."""
    payload = {field.name: getattr(orb, field.name) for field in fields(OrbParams)}
    return OrbParams(**payload)


class OpeningRangeBreakout:
    """Estrategia ORB. ``on_bar`` solo lee el contexto y el calendario inyectado.

    Sin sesiones cargadas no opera: el cierre real (media sesión incluida)
    no se adivina. ``splits`` mapea la fecha NY del split a la razón
    (2.0 = 2 por 1). Vacío significa que el historial ya viene ajustado
    o que no hubo splits.
    """

    id = STRATEGY_ID

    def __init__(
        self,
        params: Optional[OrbParams] = None,
        *,
        sessions: Sequence[Session] = (),
        flatten_minutes: float = _DEFAULT_FLATTEN_MINUTES,
        sizing: Optional[SizingConfig] = None,
        splits: Optional[Mapping[date, float]] = None,
        timeframe: str = "1Min",
        tick: float = _DEFAULT_TICK,
    ) -> None:
        self.params = OrbParams() if params is None else params
        if not isinstance(self.params, OrbParams):
            raise TypeError("params debe ser OrbParams")
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
        self.sizing = _default_sizing(self.params.risk_frac) if sizing is None else sizing
        if not isinstance(self.sizing, SizingConfig):
            raise TypeError("sizing debe ser SizingConfig")
        if abs(self.sizing.risk_frac - self.params.risk_frac) > 1e-12:
            raise ValueError("sizing.risk_frac debe coincidir con params.risk_frac")
        self.splits = {}
        for day, ratio in (splits or {}).items():
            if not isinstance(day, date) or isinstance(day, datetime):
                raise TypeError("la clave de splits debe ser datetime.date")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError("la razón de split debe ser numérica")
            if not math.isfinite(float(ratio)) or float(ratio) <= 0:
                raise ValueError("la razón de split debe ser finita y > 0")
            self.splits[day] = float(ratio)
        self.timeframe = canonical_timeframe(timeframe)
        self._bar_minutes = _bar_minutes(self.timeframe)
        if self.params.opening_minutes < self._bar_minutes:
            raise ValueError("opening_minutes debe cubrir al menos una barra")
        if isinstance(tick, bool) or not isinstance(tick, (int, float)) or float(tick) <= 0:
            raise ValueError("tick debe ser > 0")
        self.tick = float(tick)

    @classmethod
    def from_config(
        cls,
        config: HftConfig,
        sessions: Sequence[Session] = (),
        splits: Optional[Mapping[date, float]] = None,
    ) -> "OpeningRangeBreakout":
        """Instancia con ``[orb]``, el sizing, el tick y el flatten del archivo."""
        if not isinstance(config, HftConfig):
            raise TypeError("config debe ser HftConfig")
        return cls(
            params_from_config(config.orb),
            sessions=sessions,
            flatten_minutes=config.risk.flatten_minutes,
            sizing=config.sizing,
            splits=splits,
            timeframe=config.timeframe,
            tick=config.costs.tick,
        )

    def required_history(self) -> Tuple[int, int]:
        need = max(
            self.params.atr_sessions,
            self.params.adv_sessions,
            self.params.rvol_sessions,
        ) + 1
        bars_per_session = math.ceil(390 / self._bar_minutes)
        opening_bars = math.ceil(self.params.opening_minutes / self._bar_minutes)
        return (max(opening_bars, bars_per_session * need), need)

    def on_bar(self, ctx: StrategyContext) -> Sequence[OrderIntent]:
        """Entrada solo en la barra que cierra el rango. Salida en el flatten.

        Usa barras con inicio ``<= ctx.bar_ts``. Un frame que traiga el
        futuro de esa barra se recorta.
        """
        session = session_for(self.sessions, ctx.bar_ts)
        if session is None:
            return ()
        if ctx.now >= flatten_time(session, self.flatten_minutes):
            return tuple(self._flatten(ctx, session))
        if not _is_range_close(ctx.bar_ts, session, self.params.opening_minutes, self._bar_minutes):
            return ()
        bars = _until(ctx.bars, ctx.bar_ts)
        # ``DatetimeIndex.__contains__`` acepta Timestamp. Un datetime suelto
        # puede no estar en un set de Timestamp aunque sea el mismo instante.
        stamp = pd.Timestamp(ctx.bar_ts)
        if bars.close.empty or stamp not in bars.close.index:
            return ()
        candidates = self._candidates(bars, session)
        if not candidates:
            return ()
        equity = _equity(ctx, bars)
        if equity <= 0:
            return ()
        orders = []
        for symbol, rvol, atr in candidates:
            if abs(ctx.position(symbol)) > _QTY_EPS:
                continue
            order = self._entry(bars, session, ctx.bar_ts, symbol, rvol, atr, equity)
            if order is not None:
                orders.append(order)
        return tuple(orders)

    def _candidates(
        self,
        bars: IntradayBars,
        session: Session,
    ) -> Tuple[Tuple[str, float, float], ...]:
        """Top K con RVOL >= umbral, ya filtrados por precio, ADV, ATR y Amihud."""
        daily = _daily_ohlcv(bars, self.sessions, before=session.date)
        if daily is None:
            return ()
        open_, high, low, close, volume = adjust_daily_for_splits(
            *daily, splits=self.splits,
        )
        need = max(self.params.atr_sessions, self.params.adv_sessions, self.params.rvol_sessions)
        if len(close.index) < need:
            return ()
        liquid = _liquid_names(self.params, close, volume)
        dates = _bar_dates(bars.close.index, self.sessions)
        adjusted_volume = _adjust_intraday_volume(bars.volume, dates, self.splits)
        rvol = relative_volume(
            adjusted_volume,
            k_minutes=self.params.opening_minutes,
            sessions_back=self.params.rvol_sessions,
            timeframe=self.timeframe,
            sessions=self.sessions,
        )
        # Wilder recorre todos los días previos de la ventana: los que se
        # calientan (``Engine.history_sessions``), en backtest y en paper.
        atr = wilder_atr(high, low, close, n=self.params.atr_sessions).iloc[-1]
        adv = volume.iloc[-self.params.adv_sessions:].mean()
        prior_close = close.iloc[-1]
        ranked = []
        stamp = pd.Timestamp(bars.close.index[-1])
        for symbol in close.columns:
            if symbol not in liquid:
                continue
            info = self._score(
                bars, session, symbol, rvol, atr, adv, prior_close, stamp,
            )
            if info is not None:
                ranked.append(info)
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return tuple(ranked[: self.params.top_k])

    def _score(
        self,
        bars: IntradayBars,
        session: Session,
        symbol: str,
        rvol: pd.DataFrame,
        atr: pd.Series,
        adv: pd.Series,
        prior_close: pd.Series,
        stamp: pd.Timestamp,
    ) -> Optional[Tuple[str, float, float]]:
        if symbol not in bars.close.columns:
            return None
        opened = _session_open(bars, session, symbol)
        if opened is None or not opened > self.params.min_price:
            return None
        last_close = _finite_or_none(prior_close.get(symbol))
        if last_close is None or not last_close > self.params.min_price:
            return None
        adv_shares = _finite_or_none(adv.get(symbol))
        if adv_shares is None or adv_shares < self.params.min_adv_shares:
            return None
        atr_value = _finite_or_none(atr.get(symbol))
        # El paper pide estrictamente más de 0.50 USD.
        if atr_value is None or not atr_value > self.params.min_atr:
            return None
        if stamp not in rvol.index or symbol not in rvol.columns:
            return None
        ratio = _finite_or_none(rvol.at[stamp, symbol])
        if ratio is None or ratio < self.params.min_rvol:
            return None
        return (str(symbol), float(ratio), float(atr_value))

    def _entry(
        self,
        bars: IntradayBars,
        session: Session,
        bar_ts: datetime,
        symbol: str,
        rvol: float,
        atr: float,
        equity: float,
    ) -> Optional[OrderIntent]:
        opened = _session_open(bars, session, symbol)
        closed = _finite_or_none(bars.close.at[pd.Timestamp(bar_ts), symbol])
        if opened is None or closed is None:
            return None
        move = closed - opened
        if abs(move) <= self.params.doji_eps:
            return None
        side = "buy" if move > 0 else "sell"
        bump = self.params.limit_through_ticks * self.tick
        raw_limit = closed + bump if side == "buy" else closed - bump
        limit = _round_tick(raw_limit, self.tick)
        distance = self.params.stop_atr_frac * atr
        raw_stop = limit - distance if side == "buy" else limit + distance
        stop = _round_tick(raw_stop, self.tick)
        if limit <= 0 or stop <= 0 or abs(limit - stop) <= _PRICE_EPS:
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
        return order_intent(
            self.id,
            symbol,
            side,
            qty,
            bar_ts,
            order_type="marketable_limit",
            limit_price=limit,
            tif="day",
            bracket=Bracket(stop=stop),
            reason=(
                f"orb {side} rvol={rvol:.4g} atr={atr:.4g} "
                f"stop={stop:.4g} limite={limit:.4g}"
            ),
        )

    def _flatten(self, ctx: StrategyContext, session: Session) -> list:
        orders = []
        for symbol, _held in ctx.positions:
            # Un flatten que sigue vivo ya cubre la posición. El stop del
            # bracket no cuenta: el motor no lo pone en ``pending``
            # (HFT-PAPER-16).
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
                reason="orb flatten_time",
            ))
        orders.sort(key=lambda item: item.symbol)
        return orders


def adjust_daily_for_splits(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    splits: Mapping[date, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Precios entre el factor y volumen por el factor, solo en fechas anteriores.

    ``splits[día] = 2`` es un 2 por 1 en la apertura de ``día``. Las filas
    estrictamente anteriores se ajustan; el día del split no. Varios
    splits se componen. Sin splits se devuelven los mismos frames.

    El día en curso no entra en estos frames: el rango de apertura se
    lee de las barras raw. Así un split viejo no mueve ni el ATR ni el
    high/low de hoy.
    """
    if not splits:
        return open_, high, low, close, volume
    factor = pd.Series(
        (_split_factor(_index_date(ts), splits) for ts in close.index),
        index=close.index,
        dtype=float,
    )
    return (
        open_.div(factor, axis=0),
        high.div(factor, axis=0),
        low.div(factor, axis=0),
        close.div(factor, axis=0),
        volume.mul(factor, axis=0),
    )


def _split_factor(day: Optional[date], splits: Mapping[date, float]) -> float:
    if day is None:
        return 1.0
    factor = 1.0
    for split_day, ratio in splits.items():
        if day < split_day:
            factor *= float(ratio)
    return factor


def _index_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _liquid_names(params: OrbParams, close: pd.DataFrame, volume: pd.DataFrame) -> set:
    """Nombres que pasan Amihud y volumen medio. Solo hay filas previas."""
    if len(close.index) < params.adv_sessions:
        return set()
    prices = close.copy()
    volumes = volume.reindex_like(prices)
    as_of = pd.Timestamp(prices.index[-1])
    active, _log = filter_universe_at_date(
        as_of,
        prices,
        volumes,
        amihud_percentile_threshold=float(params.amihud_percentile),
        min_volume=float(params.min_adv_shares),
        window=int(params.adv_sessions),
    )
    return set(active)


def _daily_ohlcv(
    bars: IntradayBars,
    sessions: Sequence[Session],
    *,
    before: date,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """OHLC y volumen por sesión ya cerrada. El día ``before`` no entra."""
    dates = _bar_dates(bars.close.index, sessions)
    days = [session for session in sessions if session.date < before]
    symbols = list(bars.close.columns)
    index = []
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for session in days:
        mask = dates.map(lambda day: day == session.date).fillna(False).astype(bool)
        if not bool(mask.any()):
            continue
        index.append(pd.Timestamp(session.date))
        opens.append(_reduce(bars.open.loc[mask], "first"))
        highs.append(_reduce(bars.high.loc[mask], "max"))
        lows.append(_reduce(bars.low.loc[mask], "min"))
        closes.append(_reduce(bars.close.loc[mask], "last"))
        volumes.append(_reduce(bars.volume.loc[mask], "sum"))
    if not index:
        return None
    stamp = pd.DatetimeIndex(index)
    return (
        pd.DataFrame(opens, index=stamp, columns=symbols),
        pd.DataFrame(highs, index=stamp, columns=symbols),
        pd.DataFrame(lows, index=stamp, columns=symbols),
        pd.DataFrame(closes, index=stamp, columns=symbols),
        pd.DataFrame(volumes, index=stamp, columns=symbols),
    )


def _reduce(part: pd.DataFrame, how: str) -> dict:
    out = {}
    for symbol in part.columns:
        series = part[symbol].dropna()
        if series.empty:
            out[symbol] = np.nan
            continue
        if how == "first":
            out[symbol] = float(series.iloc[0])
        elif how == "last":
            out[symbol] = float(series.iloc[-1])
        elif how == "max":
            out[symbol] = float(series.max())
        elif how == "min":
            out[symbol] = float(series.min())
        else:
            out[symbol] = float(series.sum())
    return out


def _bar_dates(index: pd.DatetimeIndex, sessions: Sequence[Session]) -> pd.Series:
    dates = pd.Series(pd.NA, index=index, dtype="object")
    utc = pd.DatetimeIndex(index).tz_convert("UTC")
    for session in sessions:
        open_ts = pd.Timestamp(session.open_utc)
        close_ts = pd.Timestamp(session.close_utc)
        mask = (utc >= open_ts) & (utc < close_ts)
        dates.loc[mask] = session.date
    return dates


def _adjust_intraday_volume(
    volume: pd.DataFrame,
    dates: pd.Series,
    splits: Mapping[date, float],
) -> pd.DataFrame:
    if not splits:
        return volume
    factor = pd.Series(
        (_split_factor(day if day is not pd.NA else None, splits) for day in dates),
        index=volume.index,
        dtype=float,
    )
    return volume.mul(factor, axis=0)


def _session_open(bars: IntradayBars, session: Session, symbol: str) -> Optional[float]:
    idx = bars.open.index
    mask = (idx >= pd.Timestamp(session.open_utc)) & (idx < pd.Timestamp(session.close_utc))
    series = bars.open.loc[mask, symbol].dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


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


def _is_range_close(bar_ts: datetime, session: Session, opening_minutes: int, bar_minutes: int) -> bool:
    delta = bar_ts - session.open_utc
    start = timedelta(minutes=opening_minutes - bar_minutes)
    end = timedelta(minutes=opening_minutes)
    return start <= delta < end


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


def _register_default() -> OpeningRangeBreakout:
    strategy = OpeningRangeBreakout()
    register(strategy)
    return strategy


DEFAULT_ORB = _register_default()
