"""Configuración estricta del intradía (``config/hft.toml``).

Mismo contrato que ``strategy_config_from_dict``: toda clave es
obligatoria, una clave desconocida es error y los tipos se validan.
``config_hash`` es el sha256 de la forma canónica (claves ordenadas).
Los costos (HFT-07) van en ``[costs]`` y ``[sim]``. El riesgo y el
sizing (HFT-08) van en ``[risk]`` y ``[sizing]``. El ORB (HFT-09A) va
en ``[orb]``. El momentum intradía (HFT-09B) va en ``[intraday_momentum]``.
La reversión a VWAP (HFT-09C) va en ``[vwap_reversion]``.
Los pares OU (HFT-09D) van en ``[pairs_ou]``: la lista de pares es
ex-ante y no se descubre minando el universo.
Avellaneda–Stoikov (HFT-09E) va en ``[avellaneda_stoikov]`` y es
``research_only``: el motor no lo acepta en paper.
El momentum horario va en ``[hourly_momentum]``: no trae símbolos y
opera solo ``universe``.
``holdout_start`` (HFT-10) es la primera sesión bloqueada: el backtest
no la usa salvo un desbloqueo explícito que queda en el registro.
Los umbrales del protocolo go/no-go (HFT-14) van en ``[decision]``.
Ninguno tiene default: si falta una clave, la carga falla.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

# Mismas horas de la sesión regular y mismo formato HH:MM que el calendario.
from src.hft.calendar import DEFAULT_GRACE_S, _CLOSE_ET, _OPEN_ET, _parse_hhmm
from src.hft.data.bars import DEFAULT_FEED, FEEDS, canonical_timeframe, check_feed
from src.settings import REPO_ROOT

# Esquema de este archivo. No es CONFIG_SCHEMA_VERSION del legado.
HFT_SCHEMA_VERSION = 1

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "hft.toml"

# Claves de primer nivel, en el orden del archivo.
_FIELDS = (
    "schema_version",
    "timeframe",
    "grace_s",
    "latency_bars",
    "initial_cash",
    "feed",
    "holdout_start",
    "universe",
    "costs",
    "sim",
    "risk",
    "sizing",
    "orb",
    "intraday_momentum",
    "vwap_reversion",
    "pairs_ou",
    "avellaneda_stoikov",
    "hourly_momentum",
    "decision",
)

_COST_FIELDS = (
    "spread_floor_bps",
    "tick",
    "impact_y",
    "adv_window",
    "commission_per_share",
    "sec_section31_rate",
    "sec_section31_verified_on",
    "finra_taf_per_share",
    "finra_taf_cap",
    "finra_taf_verified_on",
    "crypto_maker_bps",
    "crypto_taker_bps",
    "crypto_fees_verified_on",
)

_SIM_FIELDS = (
    "participation_cap",
    "partial_prob",
    "fill_mode",
    "latency_bars",
)

_FILL_MODES = ("cross", "touch")

# Kelly fraccional: el prompt fija c <= 0.25. No es un resultado estimado.
_MAX_KELLY_C = 0.25

_RISK_FIELDS = (
    "max_position_notional",
    "max_gross_exposure",
    "max_net_exposure",
    "max_open_orders",
    "max_orders_per_minute",
    "max_orders_per_day",
    "max_daily_loss_usd",
    "max_daily_loss_pct",
    "max_consecutive_losses",
    "cooldown_minutes",
    "entry_cutoff_minutes",
    "flatten_minutes",
    "stale_data_s",
    "spread_guard_bps",
)

_SIZING_FIELDS = (
    "risk_capital_usd",
    "risk_frac",
    "sigma_target",
    "kelly_c",
    "min_notional",
    "max_leverage",
)

# Parámetros ex-ante del ORB. Los valores del archivo versionado son los
# de Zarattini, Barbon & Aziz (2024); la estrategia no los busca.
_ORB_FIELDS = (
    "opening_minutes",
    "rvol_sessions",
    "min_rvol",
    "top_k",
    "min_price",
    "min_adv_shares",
    "adv_sessions",
    "min_atr",
    "atr_sessions",
    "stop_atr_frac",
    "risk_frac",
    "doji_eps",
    "amihud_percentile",
    "limit_through_ticks",
)

# Parámetros ex-ante del momentum intradía (HFT-09B). Los números del
# archivo versionado citan el paper en la TheoryCard. No se buscan.
_MOMENTUM_FIELDS = (
    "symbol",
    "first_half_minutes",
    "last_half_minutes",
    "require_r12_agreement",
    "use_vol_filter",
    "min_abs_r1_sigma",
    "sigma_sessions",
    "exit_mode",
    "cls_cutoff_minutes",
)

_MOMENTUM_EXITS = ("flatten", "cls")

# Parámetros ex-ante de la reversión a VWAP (HFT-09C). Los números del
# archivo versionado están en la TheoryCard. No se buscan.
_VWAP_FIELDS = (
    "z_n",
    "entry_z",
    "exit_z",
    "stop_z",
    "skip_open_minutes",
    "entry_cutoff_minutes",
    "time_stop_minutes",
    "vol_sessions",
    "max_vol_percentile",
    "trend_ma",
    "trend_look",
    "trend_threshold",
    "top_k",
    "risk_frac",
    "limit_through_ticks",
    "entry_style",
)

_VWAP_STYLES = ("marketable_limit", "passive_limit")

# Parámetros ex-ante de los pares OU (HFT-09D). Los pares están escritos
# en ``pairs``; no hay una clave para buscarlos en el universo.
_PAIRS_FIELDS = (
    "formation_sessions",
    "adf_lags",
    "entry_z",
    "exit_z",
    "stop_z",
    "half_life_min",
    "half_life_max",
    "half_life_stop_mult",
    "leg_timeout_s",
    "risk_frac",
    "limit_through_ticks",
    "entry_cutoff_minutes",
    "pairs",
)

_PAIR_FIELDS = ("leg_a", "leg_b", "link")

# Avellaneda–Stoikov (HFT-09E). research_only: γ sale de la simulación
# del paper; k no es una clave, se estima en la estrategia.
_AS_FIELDS = (
    "symbol",
    "gamma",
    "q_max",
    "quote_qty",
    "sigma_bars",
    "k_bars",
    "min_trades",
    "min_bars",
    "entry_cutoff_minutes",
)

# Momentum horario. Exploratorio: no hay clave de símbolos, la estrategia
# opera solo ``universe``. Las ventanas son horas de Nueva York.
_HOURLY_FIELDS = (
    "first_window_et",
    "last_window_et",
    "step_minutes",
    "lookback_minutes",
    "sigma_sessions",
    "min_abs_z",
    "top_k",
    "max_new_per_window",
    "hold_minutes",
    "stop_atr_frac",
    "atr_sessions",
    "risk_frac",
    "limit_through_ticks",
)

# Umbrales del protocolo go/no-go (HFT-14). Los números del archivo
# versionado son los del prompt. No se estiman.
_DECISION_FIELDS = (
    "min_expectancy",
    "min_t_stat",
    "min_dsr",
    "max_pbo",
    "min_oos_trades",
    "low_frequency_strategies",
    "max_year_pnl_share",
    "best_days_dropped",
    "min_paper_sessions",
    "min_paper_trades",
    "min_slippage_sessions",
    "slippage_se_mult",
    "required_drills",
)

# G5. El archivo tiene que listar estos tres: no se puede bajar el
# simulacro de KILL, de caída o de media sesión editando la lista.
PROTOCOL_DRILLS = ("crash_restart", "kill", "half_session")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class CostConfig:
    """Tasas del modelo único. ``verified_on`` es la fecha en que se miró la fuente."""

    spread_floor_bps: float
    tick: float
    impact_y: float
    adv_window: int
    commission_per_share: float
    sec_section31_rate: float
    sec_section31_verified_on: str
    finra_taf_per_share: float
    finra_taf_cap: float
    finra_taf_verified_on: str
    crypto_maker_bps: float
    crypto_taker_bps: float
    crypto_fees_verified_on: str

    def __post_init__(self) -> None:
        if self.tick <= 0:
            raise ValueError("costs.tick debe ser > 0")
        if self.spread_floor_bps < 0 or self.impact_y < 0:
            raise ValueError("costs.spread_floor_bps e impact_y no pueden ser negativos")
        if self.adv_window < 1:
            raise ValueError("costs.adv_window debe ser >= 1")
        for name in (
            "commission_per_share",
            "sec_section31_rate",
            "finra_taf_per_share",
            "finra_taf_cap",
            "crypto_maker_bps",
            "crypto_taker_bps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"costs.{name} no puede ser negativo")


@dataclass(frozen=True)
class SimConfig:
    """Cómo llena el simulador. ``latency_bars`` son barras extra, no las del motor.

    ``fill_mode = touch`` es la sensibilidad optimista. El archivo
    versionado deja ``cross``.
    """

    participation_cap: float
    partial_prob: float
    fill_mode: str
    latency_bars: int

    def __post_init__(self) -> None:
        if not 0 < self.participation_cap <= 1:
            raise ValueError("sim.participation_cap debe estar en (0, 1]")
        if not 0 <= self.partial_prob <= 1:
            raise ValueError("sim.partial_prob debe estar en [0, 1]")
        if self.fill_mode not in _FILL_MODES:
            raise ValueError(f"sim.fill_mode {self.fill_mode!r} no está en {_FILL_MODES}")
        if self.latency_bars < 0:
            raise ValueError("sim.latency_bars debe ser >= 0")


@dataclass(frozen=True)
class RiskConfig:
    """Límites continuos del intradía. Todos obligatorios en ``[risk]``.

    ``max_gross_exposure`` y ``max_net_exposure`` son múltiplos del
    equity de la cuenta. ``max_daily_loss_pct`` es una fracción de ese
    mismo equity (``0.02`` = 2%), no puntos de porcentaje.
    ``entry_cutoff_minutes`` y ``flatten_minutes`` son minutos antes
    del cierre real de la sesión (media sesión incluida).
    """

    max_position_notional: float
    max_gross_exposure: float
    max_net_exposure: float
    max_open_orders: int
    max_orders_per_minute: int
    max_orders_per_day: int
    max_daily_loss_usd: float
    max_daily_loss_pct: float
    max_consecutive_losses: int
    cooldown_minutes: int
    entry_cutoff_minutes: float
    flatten_minutes: float
    stale_data_s: float
    spread_guard_bps: float

    def __post_init__(self) -> None:
        for name in (
            "max_position_notional",
            "max_gross_exposure",
            "max_net_exposure",
            "max_daily_loss_usd",
            "stale_data_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"risk.{name} debe ser > 0")
        if not 0 < self.max_daily_loss_pct <= 1:
            raise ValueError("risk.max_daily_loss_pct debe estar en (0, 1]")
        if self.spread_guard_bps < 0:
            raise ValueError("risk.spread_guard_bps no puede ser negativo")
        for name in (
            "max_open_orders",
            "max_orders_per_minute",
            "max_orders_per_day",
            "max_consecutive_losses",
            "cooldown_minutes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"risk.{name} debe ser >= 1")
        if self.flatten_minutes < 0 or self.entry_cutoff_minutes < 0:
            raise ValueError("risk.entry_cutoff_minutes y flatten_minutes deben ser >= 0")
        if self.entry_cutoff_minutes < self.flatten_minutes:
            raise ValueError(
                "risk.entry_cutoff_minutes debe ser >= flatten_minutes "
                "(el corte de entradas no puede ir después del flatten)"
            )


@dataclass(frozen=True)
class SizingConfig:
    """Capital de riesgo y las tres reglas de tamaño (HFT-08, D-04).

    ``equity_riesgo`` es ``min(equity de la cuenta, risk_capital_usd)``.
    ``kelly_c`` no puede pasar de 0.25. ``sigma_target`` y la σ del
    activo van en la misma unidad; este módulo no anualiza.
    """

    risk_capital_usd: float
    risk_frac: float
    sigma_target: float
    kelly_c: float
    min_notional: float
    max_leverage: float

    def __post_init__(self) -> None:
        if self.risk_capital_usd <= 0:
            raise ValueError("sizing.risk_capital_usd debe ser > 0")
        if not 0 < self.risk_frac <= 1:
            raise ValueError("sizing.risk_frac debe estar en (0, 1]")
        if self.sigma_target <= 0:
            raise ValueError("sizing.sigma_target debe ser > 0")
        if not 0 < self.kelly_c <= _MAX_KELLY_C:
            raise ValueError(f"sizing.kelly_c debe estar en (0, {_MAX_KELLY_C}]")
        if self.min_notional < 0:
            raise ValueError("sizing.min_notional no puede ser negativo")
        if self.max_leverage <= 0:
            raise ValueError("sizing.max_leverage debe ser > 0")


@dataclass(frozen=True)
class OrbConfig:
    """Parámetros ex-ante del opening-range breakout (HFT-09A).

    No hay defaults en la carga: el TOML trae todos. Los números del
    archivo versionado citan sección del paper en la TheoryCard.
    ``risk_frac`` tiene que coincidir con ``sizing.risk_frac``: el
    tamaño lo calcula ``sizing.py`` y la tarjeta no puede decir otra cosa.
    """

    opening_minutes: int
    rvol_sessions: int
    min_rvol: float
    top_k: int
    min_price: float
    min_adv_shares: int
    adv_sessions: int
    min_atr: float
    atr_sessions: int
    stop_atr_frac: float
    risk_frac: float
    doji_eps: float
    amihud_percentile: float
    limit_through_ticks: int

    def __post_init__(self) -> None:
        for name in (
            "opening_minutes",
            "rvol_sessions",
            "top_k",
            "min_adv_shares",
            "adv_sessions",
            "atr_sessions",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"orb.{name} debe ser >= 1")
        if self.min_rvol <= 0:
            raise ValueError("orb.min_rvol debe ser > 0")
        if self.min_price <= 0 or self.min_atr <= 0:
            raise ValueError("orb.min_price y min_atr deben ser > 0")
        if not 0 < self.stop_atr_frac <= 1:
            raise ValueError("orb.stop_atr_frac debe estar en (0, 1]")
        if not 0 < self.risk_frac <= 1:
            raise ValueError("orb.risk_frac debe estar en (0, 1]")
        if self.doji_eps < 0:
            raise ValueError("orb.doji_eps no puede ser negativo")
        if not 0 < self.amihud_percentile <= 100:
            raise ValueError("orb.amihud_percentile debe estar en (0, 100]")
        if self.limit_through_ticks < 0:
            raise ValueError("orb.limit_through_ticks debe ser >= 0")


@dataclass(frozen=True)
class IntradayMomentumConfig:
    """Parámetros ex-ante del momentum de la última media hora (HFT-09B).

    No hay defaults en la carga: el TOML trae todos. ``symbol`` se guarda
    en mayúsculas. ``exit_mode`` es ``flatten`` o ``cls``.
    """

    symbol: str
    first_half_minutes: int
    last_half_minutes: int
    require_r12_agreement: bool
    use_vol_filter: bool
    min_abs_r1_sigma: float
    sigma_sessions: int
    exit_mode: str
    cls_cutoff_minutes: int

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("intraday_momentum.symbol vacío")
        object.__setattr__(self, "symbol", symbol)
        for name in ("first_half_minutes", "last_half_minutes", "sigma_sessions", "cls_cutoff_minutes"):
            if getattr(self, name) < 1:
                raise ValueError(f"intraday_momentum.{name} debe ser >= 1")
        if self.sigma_sessions < 2:
            raise ValueError("intraday_momentum.sigma_sessions debe ser >= 2")
        if self.last_half_minutes <= self.cls_cutoff_minutes:
            raise ValueError(
                "intraday_momentum.last_half_minutes debe ser > cls_cutoff_minutes"
            )
        if self.min_abs_r1_sigma <= 0:
            raise ValueError("intraday_momentum.min_abs_r1_sigma debe ser > 0")
        mode = str(self.exit_mode).strip().lower()
        if mode not in _MOMENTUM_EXITS:
            raise ValueError(
                f"intraday_momentum.exit_mode {self.exit_mode!r} no está en {_MOMENTUM_EXITS}"
            )
        object.__setattr__(self, "exit_mode", mode)


@dataclass(frozen=True)
class VwapReversionConfig:
    """Parámetros ex-ante de la reversión a VWAP (HFT-09C).

    No hay defaults en la carga: el TOML trae todos. ``entry_style`` es
    ``marketable_limit`` o ``passive_limit``. ``entry_cutoff_minutes`` y
    ``risk_frac`` tienen que coincidir con ``[risk]`` y ``[sizing]``.
    """

    z_n: int
    entry_z: float
    exit_z: float
    stop_z: float
    skip_open_minutes: int
    entry_cutoff_minutes: float
    time_stop_minutes: int
    vol_sessions: int
    max_vol_percentile: float
    trend_ma: int
    trend_look: int
    trend_threshold: float
    top_k: int
    risk_frac: float
    limit_through_ticks: int
    entry_style: str

    def __post_init__(self) -> None:
        for name, minimum in (
            ("z_n", 2),
            ("skip_open_minutes", 0),
            ("time_stop_minutes", 1),
            ("vol_sessions", 1),
            ("trend_ma", 1),
            ("trend_look", 2),
            ("top_k", 1),
            ("limit_through_ticks", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"vwap_reversion.{name} debe ser un entero >= {minimum}")
        if not self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("vwap_reversion exige exit_z < entry_z < stop_z")
        if self.entry_z <= 0 or self.exit_z <= 0 or self.stop_z <= 0:
            raise ValueError("vwap_reversion: entry_z, exit_z y stop_z deben ser > 0")
        if self.entry_cutoff_minutes <= 0:
            raise ValueError("vwap_reversion.entry_cutoff_minutes debe ser > 0")
        if not 0 < self.max_vol_percentile <= 100:
            raise ValueError("vwap_reversion.max_vol_percentile debe estar en (0, 100]")
        if self.trend_threshold <= 0:
            raise ValueError("vwap_reversion.trend_threshold debe ser > 0")
        if not 0 < self.risk_frac <= 1:
            raise ValueError("vwap_reversion.risk_frac debe estar en (0, 1]")
        style = str(self.entry_style).strip().lower()
        if style not in _VWAP_STYLES:
            raise ValueError(
                f"vwap_reversion.entry_style {self.entry_style!r} no está en {_VWAP_STYLES}"
            )
        object.__setattr__(self, "entry_style", style)


@dataclass(frozen=True)
class PairsOuConfig:
    """Parámetros ex-ante de los pares con residuo OU (HFT-09D).

    ``pairs`` es la lista escrita a mano: cada elemento es
    ``(leg_a, leg_b, vínculo)``. ``leg_a`` es la dependiente del OLS.
    No hay defaults en la carga. ``entry_cutoff_minutes`` y
    ``risk_frac`` tienen que coincidir con ``[risk]`` y ``[sizing]``.
    """

    formation_sessions: int
    adf_lags: int
    entry_z: float
    exit_z: float
    stop_z: float
    half_life_min: float
    half_life_max: float
    half_life_stop_mult: float
    leg_timeout_s: float
    risk_frac: float
    limit_through_ticks: int
    entry_cutoff_minutes: float
    pairs: Tuple[Tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        for name, minimum in (
            ("formation_sessions", 1),
            ("adf_lags", 0),
            ("limit_through_ticks", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"pairs_ou.{name} debe ser un entero >= {minimum}")
        if not self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("pairs_ou exige exit_z < entry_z < stop_z")
        for name in ("entry_z", "exit_z", "stop_z", "half_life_stop_mult", "leg_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f"pairs_ou.{name} debe ser un número > 0")
        if not (
            isinstance(self.half_life_min, (int, float))
            and isinstance(self.half_life_max, (int, float))
            and not isinstance(self.half_life_min, bool)
            and not isinstance(self.half_life_max, bool)
            and 0 < float(self.half_life_min) < float(self.half_life_max)
        ):
            raise ValueError("pairs_ou exige 0 < half_life_min < half_life_max")
        if self.entry_cutoff_minutes <= 0:
            raise ValueError("pairs_ou.entry_cutoff_minutes debe ser > 0")
        if not 0 < self.risk_frac <= 1:
            raise ValueError("pairs_ou.risk_frac debe estar en (0, 1]")
        object.__setattr__(self, "pairs", _validated_pairs(self.pairs))


@dataclass(frozen=True)
class AvellanedaStoikovConfig:
    """Parámetros ex-ante de Avellaneda–Stoikov (HFT-09E).

    No hay defaults en la carga. ``k`` no está: se estima con las
    barras. ``entry_cutoff_minutes`` tiene que coincidir con ``[risk]``.
    La estrategia es ``research_only`` aunque esta tabla no lo diga.
    """

    symbol: str
    gamma: float
    q_max: int
    quote_qty: int
    sigma_bars: int
    k_bars: int
    min_trades: int
    min_bars: int
    entry_cutoff_minutes: float

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("avellaneda_stoikov.symbol vacío")
        object.__setattr__(self, "symbol", symbol)
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise TypeError("avellaneda_stoikov.gamma debe ser numérico")
        if float(self.gamma) <= 0:
            raise ValueError("avellaneda_stoikov.gamma debe ser > 0")
        for name, minimum in (
            ("q_max", 1),
            ("quote_qty", 1),
            ("sigma_bars", 2),
            ("min_bars", 2),
            ("min_trades", 1),
            ("k_bars", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"avellaneda_stoikov.{name} debe ser un entero >= {minimum}")
        if self.k_bars < self.min_bars:
            raise ValueError("avellaneda_stoikov.k_bars debe ser >= min_bars")
        if isinstance(self.entry_cutoff_minutes, bool) or not isinstance(
            self.entry_cutoff_minutes, (int, float)
        ):
            raise TypeError("avellaneda_stoikov.entry_cutoff_minutes debe ser numérico")
        if float(self.entry_cutoff_minutes) <= 0:
            raise ValueError("avellaneda_stoikov.entry_cutoff_minutes debe ser > 0")


@dataclass(frozen=True)
class HourlyMomentumConfig:
    """Parámetros ex-ante del momentum horario (``hourly_momentum``).

    No hay defaults en la carga. Las ventanas van de ``first_window_et``
    a ``last_window_et`` cada ``step_minutes``, en horas ``HH:MM`` de
    Nueva York dentro de la sesión regular. No hay clave de símbolos: la
    estrategia opera solo ``universe``. ``risk_frac`` tiene que coincidir
    con ``sizing.risk_frac`` y la última ventana tiene que quedar antes
    de ``risk.entry_cutoff_minutes``.
    """

    first_window_et: str
    last_window_et: str
    step_minutes: int
    lookback_minutes: int
    sigma_sessions: int
    min_abs_z: float
    top_k: int
    max_new_per_window: int
    hold_minutes: int
    stop_atr_frac: float
    atr_sessions: int
    risk_frac: float
    limit_through_ticks: int

    def __post_init__(self) -> None:
        for name, minimum in (
            ("step_minutes", 1),
            ("lookback_minutes", 1),
            ("sigma_sessions", 2),
            ("top_k", 1),
            ("max_new_per_window", 1),
            ("hold_minutes", 1),
            ("atr_sessions", 1),
            ("limit_through_ticks", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"hourly_momentum.{name} debe ser un entero >= {minimum}")
        if self.max_new_per_window > self.top_k:
            raise ValueError("hourly_momentum.max_new_per_window debe ser <= top_k")
        if not _finite(self.min_abs_z) or self.min_abs_z <= 0:
            raise ValueError("hourly_momentum.min_abs_z debe ser finito y > 0")
        if not _finite(self.stop_atr_frac) or not 0 < self.stop_atr_frac <= 1:
            raise ValueError("hourly_momentum.stop_atr_frac debe estar en (0, 1]")
        if not _finite(self.risk_frac) or not 0 < self.risk_frac <= 1:
            raise ValueError("hourly_momentum.risk_frac debe estar en (0, 1]")
        first = _window_clock("first_window_et", self.first_window_et)
        last = _window_clock("last_window_et", self.last_window_et)
        if first > last:
            raise ValueError("hourly_momentum.first_window_et debe ser <= last_window_et")
        if (_clock_minutes(last) - _clock_minutes(first)) % self.step_minutes:
            raise ValueError(
                "hourly_momentum: last_window_et − first_window_et debe ser "
                "múltiplo de step_minutes"
            )
        object.__setattr__(self, "first_window_et", first.strftime("%H:%M"))
        object.__setattr__(self, "last_window_et", last.strftime("%H:%M"))


@dataclass(frozen=True)
class DecisionConfig:
    """Umbrales G1–G5. El veredicto los lee; no activa nada.

    ``min_expectancy`` está en la misma unidad que el P&L por operación
    que se le pase al gate (el backtest usa USD). ``min_t_stat`` es el
    t de esa media (``media / (s/√n)``). ``low_frequency_strategies``
    puede declarar menos de ``min_oos_trades`` solo si la evidencia
    dice que ese conteo es el máximo disponible. ``required_drills``
    es exactamente ``PROTOCOL_DRILLS``.
    """

    min_expectancy: float
    min_t_stat: float
    min_dsr: float
    max_pbo: float
    min_oos_trades: int
    low_frequency_strategies: Tuple[str, ...]
    max_year_pnl_share: float
    best_days_dropped: int
    min_paper_sessions: int
    min_paper_trades: int
    min_slippage_sessions: int
    slippage_se_mult: float
    required_drills: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not _finite(self.min_expectancy):
            raise ValueError("decision.min_expectancy debe ser finito")
        if not _finite(self.min_t_stat) or self.min_t_stat <= 0:
            raise ValueError("decision.min_t_stat debe ser finito y > 0")
        if not _finite(self.min_dsr) or not 0 < self.min_dsr <= 1:
            raise ValueError("decision.min_dsr debe estar en (0, 1]")
        if not _finite(self.max_pbo) or not 0 <= self.max_pbo < 1:
            raise ValueError("decision.max_pbo debe estar en [0, 1)")
        if isinstance(self.min_oos_trades, bool) or not isinstance(self.min_oos_trades, int):
            raise TypeError("decision.min_oos_trades debe ser int")
        if self.min_oos_trades < 1:
            raise ValueError("decision.min_oos_trades debe ser >= 1")
        names = _strategy_ids("decision.low_frequency_strategies", self.low_frequency_strategies)
        object.__setattr__(self, "low_frequency_strategies", names)
        if not _finite(self.max_year_pnl_share) or not 0 < self.max_year_pnl_share <= 1:
            raise ValueError("decision.max_year_pnl_share debe estar en (0, 1]")
        for name in ("best_days_dropped", "min_paper_sessions", "min_paper_trades"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"decision.{name} debe ser un entero >= 1")
        if (
            isinstance(self.min_slippage_sessions, bool)
            or not isinstance(self.min_slippage_sessions, int)
            or self.min_slippage_sessions < 2
        ):
            raise ValueError("decision.min_slippage_sessions debe ser un entero >= 2")
        if not _finite(self.slippage_se_mult) or self.slippage_se_mult <= 0:
            raise ValueError("decision.slippage_se_mult debe ser finito y > 0")
        drills = self.required_drills
        if isinstance(drills, str) or not isinstance(drills, tuple):
            raise TypeError("decision.required_drills debe ser una tupla")
        if set(drills) != set(PROTOCOL_DRILLS) or len(drills) != len(PROTOCOL_DRILLS):
            raise ValueError(
                "decision.required_drills debe listar "
                + ", ".join(PROTOCOL_DRILLS)
            )
        object.__setattr__(self, "required_drills", PROTOCOL_DRILLS)


@dataclass(frozen=True)
class HftConfig:
    """Config efectiva del motor. Los números ya están coercidos."""

    schema_version: int
    timeframe: str
    grace_s: float
    latency_bars: int
    initial_cash: float
    feed: str
    holdout_start: str
    universe: Tuple[str, ...]
    costs: CostConfig
    sim: SimConfig
    risk: RiskConfig
    sizing: SizingConfig
    orb: OrbConfig
    intraday_momentum: IntradayMomentumConfig
    vwap_reversion: VwapReversionConfig
    pairs_ou: PairsOuConfig
    avellaneda_stoikov: AvellanedaStoikovConfig
    hourly_momentum: HourlyMomentumConfig
    decision: DecisionConfig

    def __post_init__(self) -> None:
        if self.schema_version != HFT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={self.schema_version!r} no soportado "
                f"(se espera {HFT_SCHEMA_VERSION})"
            )
        object.__setattr__(self, "timeframe", canonical_timeframe(self.timeframe))
        if self.grace_s < 0:
            raise ValueError("grace_s no puede ser negativo")
        if self.latency_bars < 1:
            raise ValueError("latency_bars debe ser >= 1 (ejecución como pronto en t+1)")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash debe ser > 0")
        object.__setattr__(self, "feed", check_feed(self.feed))
        object.__setattr__(self, "holdout_start", _as_date("holdout_start", self.holdout_start))
        names = tuple(str(symbol).strip().upper() for symbol in self.universe)
        if not names or any(not name for name in names):
            raise ValueError("universe vacío o con un símbolo en blanco")
        if len(names) != len(set(names)):
            raise ValueError("universe tiene símbolos repetidos")
        object.__setattr__(self, "universe", names)
        if not isinstance(self.costs, CostConfig):
            raise TypeError("costs debe ser CostConfig")
        if not isinstance(self.sim, SimConfig):
            raise TypeError("sim debe ser SimConfig")
        if not isinstance(self.risk, RiskConfig):
            raise TypeError("risk debe ser RiskConfig")
        if not isinstance(self.sizing, SizingConfig):
            raise TypeError("sizing debe ser SizingConfig")
        if not isinstance(self.orb, OrbConfig):
            raise TypeError("orb debe ser OrbConfig")
        if not isinstance(self.intraday_momentum, IntradayMomentumConfig):
            raise TypeError("intraday_momentum debe ser IntradayMomentumConfig")
        if not isinstance(self.vwap_reversion, VwapReversionConfig):
            raise TypeError("vwap_reversion debe ser VwapReversionConfig")
        if not isinstance(self.pairs_ou, PairsOuConfig):
            raise TypeError("pairs_ou debe ser PairsOuConfig")
        if not isinstance(self.avellaneda_stoikov, AvellanedaStoikovConfig):
            raise TypeError("avellaneda_stoikov debe ser AvellanedaStoikovConfig")
        if not isinstance(self.hourly_momentum, HourlyMomentumConfig):
            raise TypeError("hourly_momentum debe ser HourlyMomentumConfig")
        if not isinstance(self.decision, DecisionConfig):
            raise TypeError("decision debe ser DecisionConfig")
        if self.intraday_momentum.last_half_minutes <= self.risk.flatten_minutes:
            raise ValueError(
                "intraday_momentum.last_half_minutes debe ser > risk.flatten_minutes "
                "(la decisión tiene que quedar antes del flatten)"
            )
        if abs(self.orb.risk_frac - self.sizing.risk_frac) > 1e-12:
            raise ValueError(
                "orb.risk_frac debe coincidir con sizing.risk_frac "
                "(el tamaño lo pone sizing.py)"
            )
        if abs(self.vwap_reversion.risk_frac - self.sizing.risk_frac) > 1e-12:
            raise ValueError(
                "vwap_reversion.risk_frac debe coincidir con sizing.risk_frac "
                "(el tamaño lo pone sizing.py)"
            )
        if abs(self.vwap_reversion.entry_cutoff_minutes - self.risk.entry_cutoff_minutes) > 1e-12:
            raise ValueError(
                "vwap_reversion.entry_cutoff_minutes debe coincidir con "
                "risk.entry_cutoff_minutes"
            )
        if abs(self.pairs_ou.risk_frac - self.sizing.risk_frac) > 1e-12:
            raise ValueError(
                "pairs_ou.risk_frac debe coincidir con sizing.risk_frac "
                "(el tamaño lo pone el riesgo fijo hasta el stop en z)"
            )
        if abs(self.pairs_ou.entry_cutoff_minutes - self.risk.entry_cutoff_minutes) > 1e-12:
            raise ValueError(
                "pairs_ou.entry_cutoff_minutes debe coincidir con "
                "risk.entry_cutoff_minutes"
            )
        if abs(
            self.avellaneda_stoikov.entry_cutoff_minutes - self.risk.entry_cutoff_minutes
        ) > 1e-12:
            raise ValueError(
                "avellaneda_stoikov.entry_cutoff_minutes debe coincidir con "
                "risk.entry_cutoff_minutes"
            )
        if abs(self.hourly_momentum.risk_frac - self.sizing.risk_frac) > 1e-12:
            raise ValueError(
                "hourly_momentum.risk_frac debe coincidir con sizing.risk_frac "
                "(el tamaño lo pone sizing.py)"
            )
        # En sesión regular. La media sesión la resuelve la estrategia con
        # el cierre real del calendario.
        last = _clock_minutes(_parse_hhmm(self.hourly_momentum.last_window_et))
        if last >= _clock_minutes(_CLOSE_ET) - self.risk.entry_cutoff_minutes:
            raise ValueError(
                "hourly_momentum.last_window_et debe quedar antes del corte de "
                "entradas de una sesión regular (cierre − risk.entry_cutoff_minutes)"
            )

    def to_dict(self) -> dict:
        """Forma canónica. ``1`` y ``1.0`` en un float quedan como ``1.0``."""
        costs = self.costs
        sim = self.sim
        risk = self.risk
        sizing = self.sizing
        orb = self.orb
        momentum = self.intraday_momentum
        vwap = self.vwap_reversion
        pairs = self.pairs_ou
        maker = self.avellaneda_stoikov
        hourly = self.hourly_momentum
        decision = self.decision
        return {
            "schema_version": int(self.schema_version),
            "timeframe": self.timeframe,
            "grace_s": float(self.grace_s),
            "latency_bars": int(self.latency_bars),
            "initial_cash": float(self.initial_cash),
            "feed": self.feed,
            "holdout_start": self.holdout_start,
            "universe": list(self.universe),
            "costs": {
                "spread_floor_bps": float(costs.spread_floor_bps),
                "tick": float(costs.tick),
                "impact_y": float(costs.impact_y),
                "adv_window": int(costs.adv_window),
                "commission_per_share": float(costs.commission_per_share),
                "sec_section31_rate": float(costs.sec_section31_rate),
                "sec_section31_verified_on": costs.sec_section31_verified_on,
                "finra_taf_per_share": float(costs.finra_taf_per_share),
                "finra_taf_cap": float(costs.finra_taf_cap),
                "finra_taf_verified_on": costs.finra_taf_verified_on,
                "crypto_maker_bps": float(costs.crypto_maker_bps),
                "crypto_taker_bps": float(costs.crypto_taker_bps),
                "crypto_fees_verified_on": costs.crypto_fees_verified_on,
            },
            "sim": {
                "participation_cap": float(sim.participation_cap),
                "partial_prob": float(sim.partial_prob),
                "fill_mode": sim.fill_mode,
                "latency_bars": int(sim.latency_bars),
            },
            "risk": {
                "max_position_notional": float(risk.max_position_notional),
                "max_gross_exposure": float(risk.max_gross_exposure),
                "max_net_exposure": float(risk.max_net_exposure),
                "max_open_orders": int(risk.max_open_orders),
                "max_orders_per_minute": int(risk.max_orders_per_minute),
                "max_orders_per_day": int(risk.max_orders_per_day),
                "max_daily_loss_usd": float(risk.max_daily_loss_usd),
                "max_daily_loss_pct": float(risk.max_daily_loss_pct),
                "max_consecutive_losses": int(risk.max_consecutive_losses),
                "cooldown_minutes": int(risk.cooldown_minutes),
                "entry_cutoff_minutes": float(risk.entry_cutoff_minutes),
                "flatten_minutes": float(risk.flatten_minutes),
                "stale_data_s": float(risk.stale_data_s),
                "spread_guard_bps": float(risk.spread_guard_bps),
            },
            "sizing": {
                "risk_capital_usd": float(sizing.risk_capital_usd),
                "risk_frac": float(sizing.risk_frac),
                "sigma_target": float(sizing.sigma_target),
                "kelly_c": float(sizing.kelly_c),
                "min_notional": float(sizing.min_notional),
                "max_leverage": float(sizing.max_leverage),
            },
            "orb": {
                "opening_minutes": int(orb.opening_minutes),
                "rvol_sessions": int(orb.rvol_sessions),
                "min_rvol": float(orb.min_rvol),
                "top_k": int(orb.top_k),
                "min_price": float(orb.min_price),
                "min_adv_shares": int(orb.min_adv_shares),
                "adv_sessions": int(orb.adv_sessions),
                "min_atr": float(orb.min_atr),
                "atr_sessions": int(orb.atr_sessions),
                "stop_atr_frac": float(orb.stop_atr_frac),
                "risk_frac": float(orb.risk_frac),
                "doji_eps": float(orb.doji_eps),
                "amihud_percentile": float(orb.amihud_percentile),
                "limit_through_ticks": int(orb.limit_through_ticks),
            },
            "intraday_momentum": {
                "symbol": momentum.symbol,
                "first_half_minutes": int(momentum.first_half_minutes),
                "last_half_minutes": int(momentum.last_half_minutes),
                "require_r12_agreement": bool(momentum.require_r12_agreement),
                "use_vol_filter": bool(momentum.use_vol_filter),
                "min_abs_r1_sigma": float(momentum.min_abs_r1_sigma),
                "sigma_sessions": int(momentum.sigma_sessions),
                "exit_mode": momentum.exit_mode,
                "cls_cutoff_minutes": int(momentum.cls_cutoff_minutes),
            },
            "vwap_reversion": {
                "z_n": int(vwap.z_n),
                "entry_z": float(vwap.entry_z),
                "exit_z": float(vwap.exit_z),
                "stop_z": float(vwap.stop_z),
                "skip_open_minutes": int(vwap.skip_open_minutes),
                "entry_cutoff_minutes": float(vwap.entry_cutoff_minutes),
                "time_stop_minutes": int(vwap.time_stop_minutes),
                "vol_sessions": int(vwap.vol_sessions),
                "max_vol_percentile": float(vwap.max_vol_percentile),
                "trend_ma": int(vwap.trend_ma),
                "trend_look": int(vwap.trend_look),
                "trend_threshold": float(vwap.trend_threshold),
                "top_k": int(vwap.top_k),
                "risk_frac": float(vwap.risk_frac),
                "limit_through_ticks": int(vwap.limit_through_ticks),
                "entry_style": vwap.entry_style,
            },
            "pairs_ou": {
                "formation_sessions": int(pairs.formation_sessions),
                "adf_lags": int(pairs.adf_lags),
                "entry_z": float(pairs.entry_z),
                "exit_z": float(pairs.exit_z),
                "stop_z": float(pairs.stop_z),
                "half_life_min": float(pairs.half_life_min),
                "half_life_max": float(pairs.half_life_max),
                "half_life_stop_mult": float(pairs.half_life_stop_mult),
                "leg_timeout_s": float(pairs.leg_timeout_s),
                "risk_frac": float(pairs.risk_frac),
                "limit_through_ticks": int(pairs.limit_through_ticks),
                "entry_cutoff_minutes": float(pairs.entry_cutoff_minutes),
                "pairs": [
                    {"leg_a": leg_a, "leg_b": leg_b, "link": link}
                    for leg_a, leg_b, link in pairs.pairs
                ],
            },
            "avellaneda_stoikov": {
                "symbol": maker.symbol,
                "gamma": float(maker.gamma),
                "q_max": int(maker.q_max),
                "quote_qty": int(maker.quote_qty),
                "sigma_bars": int(maker.sigma_bars),
                "k_bars": int(maker.k_bars),
                "min_trades": int(maker.min_trades),
                "min_bars": int(maker.min_bars),
                "entry_cutoff_minutes": float(maker.entry_cutoff_minutes),
            },
            "hourly_momentum": {
                "first_window_et": hourly.first_window_et,
                "last_window_et": hourly.last_window_et,
                "step_minutes": int(hourly.step_minutes),
                "lookback_minutes": int(hourly.lookback_minutes),
                "sigma_sessions": int(hourly.sigma_sessions),
                "min_abs_z": float(hourly.min_abs_z),
                "top_k": int(hourly.top_k),
                "max_new_per_window": int(hourly.max_new_per_window),
                "hold_minutes": int(hourly.hold_minutes),
                "stop_atr_frac": float(hourly.stop_atr_frac),
                "atr_sessions": int(hourly.atr_sessions),
                "risk_frac": float(hourly.risk_frac),
                "limit_through_ticks": int(hourly.limit_through_ticks),
            },
            "decision": {
                "min_expectancy": float(decision.min_expectancy),
                "min_t_stat": float(decision.min_t_stat),
                "min_dsr": float(decision.min_dsr),
                "max_pbo": float(decision.max_pbo),
                "min_oos_trades": int(decision.min_oos_trades),
                "low_frequency_strategies": list(decision.low_frequency_strategies),
                "max_year_pnl_share": float(decision.max_year_pnl_share),
                "best_days_dropped": int(decision.best_days_dropped),
                "min_paper_sessions": int(decision.min_paper_sessions),
                "min_paper_trades": int(decision.min_paper_trades),
                "min_slippage_sessions": int(decision.min_slippage_sessions),
                "slippage_se_mult": float(decision.slippage_se_mult),
                "required_drills": list(decision.required_drills),
            },
        }

    def config_hash(self) -> str:
        """sha256 de la forma canónica. El orden de las claves no importa."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hft_config_from_dict(data: Mapping[str, Any]) -> HftConfig:
    """``HftConfig`` desde un dict (TOML o ``to_dict``). Estricto.

    Raises:
        ValueError: clave faltante o desconocida, tipo inválido, o
            ``schema_version`` distinto de ``HFT_SCHEMA_VERSION``.
    """
    raw = dict(data)
    unknown = sorted(set(raw) - set(_FIELDS))
    missing = sorted(set(_FIELDS) - set(raw))
    if unknown or missing:
        raise ValueError(
            f"Config HFT inválida: faltan {missing}, desconocidas {unknown}."
        )
    version = _as_int("schema_version", raw["schema_version"])
    if version != HFT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version={version!r} no soportado (se espera {HFT_SCHEMA_VERSION})."
        )
    timeframe = raw["timeframe"]
    if not isinstance(timeframe, str):
        raise ValueError(f"timeframe: se esperaba str, llegó {timeframe!r}")
    feed = raw["feed"]
    if not isinstance(feed, str):
        raise ValueError(f"feed: se esperaba str, llegó {feed!r}")
    universe = raw["universe"]
    if isinstance(universe, str) or not isinstance(universe, Sequence):
        raise ValueError(f"universe: se esperaba una lista de símbolos, llegó {universe!r}")
    symbols = []
    for item in universe:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"universe: símbolo inválido {item!r}")
        symbols.append(item.strip())
    return HftConfig(
        schema_version=version,
        timeframe=timeframe,
        grace_s=_as_float("grace_s", raw["grace_s"]),
        latency_bars=_as_int("latency_bars", raw["latency_bars"]),
        initial_cash=_as_float("initial_cash", raw["initial_cash"]),
        feed=feed,
        holdout_start=_as_date("holdout_start", raw["holdout_start"]),
        universe=tuple(symbols),
        costs=_cost_config(raw["costs"]),
        sim=_sim_config(raw["sim"]),
        risk=_risk_config(raw["risk"]),
        sizing=_sizing_config(raw["sizing"]),
        orb=_orb_config(raw["orb"]),
        intraday_momentum=_momentum_config(raw["intraday_momentum"]),
        vwap_reversion=_vwap_config(raw["vwap_reversion"]),
        pairs_ou=_pairs_config(raw["pairs_ou"]),
        avellaneda_stoikov=_as_config(raw["avellaneda_stoikov"]),
        hourly_momentum=_hourly_config(raw["hourly_momentum"]),
        decision=_decision_config(raw["decision"]),
    )


def load_hft_config(path: Optional[PathLike] = None) -> HftConfig:
    """Carga el TOML versionado (default ``config/hft.toml``)."""
    target = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with target.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        return hft_config_from_dict(data)
    except ValueError as exc:
        raise ValueError(f"{target}: {exc}") from exc


def config_symbols(config: HftConfig) -> Tuple[str, ...]:
    """Símbolos que una sesión por defecto pide: universo, momentum, A–S y pares.

    El paper (``src.hft.session``) y el backtest piden esta misma lista.
    """
    names = list(config.universe)
    names.append(config.intraday_momentum.symbol)
    names.append(config.avellaneda_stoikov.symbol)
    for leg_a, leg_b, _link in config.pairs_ou.pairs:
        names.extend((leg_a, leg_b))
    return tuple(dict.fromkeys(name for name in names if name))


def _cost_config(value: object) -> CostConfig:
    raw = _table("costs", value, _COST_FIELDS)
    return CostConfig(
        spread_floor_bps=_as_float("costs.spread_floor_bps", raw["spread_floor_bps"]),
        tick=_as_float("costs.tick", raw["tick"]),
        impact_y=_as_float("costs.impact_y", raw["impact_y"]),
        adv_window=_as_int("costs.adv_window", raw["adv_window"]),
        commission_per_share=_as_float("costs.commission_per_share", raw["commission_per_share"]),
        sec_section31_rate=_as_float("costs.sec_section31_rate", raw["sec_section31_rate"]),
        sec_section31_verified_on=_as_date(
            "costs.sec_section31_verified_on", raw["sec_section31_verified_on"],
        ),
        finra_taf_per_share=_as_float("costs.finra_taf_per_share", raw["finra_taf_per_share"]),
        finra_taf_cap=_as_float("costs.finra_taf_cap", raw["finra_taf_cap"]),
        finra_taf_verified_on=_as_date("costs.finra_taf_verified_on", raw["finra_taf_verified_on"]),
        crypto_maker_bps=_as_float("costs.crypto_maker_bps", raw["crypto_maker_bps"]),
        crypto_taker_bps=_as_float("costs.crypto_taker_bps", raw["crypto_taker_bps"]),
        crypto_fees_verified_on=_as_date(
            "costs.crypto_fees_verified_on", raw["crypto_fees_verified_on"],
        ),
    )


def _risk_config(value: object) -> RiskConfig:
    raw = _table("risk", value, _RISK_FIELDS)
    return RiskConfig(
        max_position_notional=_as_float(
            "risk.max_position_notional", raw["max_position_notional"],
        ),
        max_gross_exposure=_as_float("risk.max_gross_exposure", raw["max_gross_exposure"]),
        max_net_exposure=_as_float("risk.max_net_exposure", raw["max_net_exposure"]),
        max_open_orders=_as_int("risk.max_open_orders", raw["max_open_orders"]),
        max_orders_per_minute=_as_int(
            "risk.max_orders_per_minute", raw["max_orders_per_minute"],
        ),
        max_orders_per_day=_as_int("risk.max_orders_per_day", raw["max_orders_per_day"]),
        max_daily_loss_usd=_as_float("risk.max_daily_loss_usd", raw["max_daily_loss_usd"]),
        max_daily_loss_pct=_as_float("risk.max_daily_loss_pct", raw["max_daily_loss_pct"]),
        max_consecutive_losses=_as_int(
            "risk.max_consecutive_losses", raw["max_consecutive_losses"],
        ),
        cooldown_minutes=_as_int("risk.cooldown_minutes", raw["cooldown_minutes"]),
        entry_cutoff_minutes=_as_float(
            "risk.entry_cutoff_minutes", raw["entry_cutoff_minutes"],
        ),
        flatten_minutes=_as_float("risk.flatten_minutes", raw["flatten_minutes"]),
        stale_data_s=_as_float("risk.stale_data_s", raw["stale_data_s"]),
        spread_guard_bps=_as_float("risk.spread_guard_bps", raw["spread_guard_bps"]),
    )


def _sizing_config(value: object) -> SizingConfig:
    raw = _table("sizing", value, _SIZING_FIELDS)
    return SizingConfig(
        risk_capital_usd=_as_float("sizing.risk_capital_usd", raw["risk_capital_usd"]),
        risk_frac=_as_float("sizing.risk_frac", raw["risk_frac"]),
        sigma_target=_as_float("sizing.sigma_target", raw["sigma_target"]),
        kelly_c=_as_float("sizing.kelly_c", raw["kelly_c"]),
        min_notional=_as_float("sizing.min_notional", raw["min_notional"]),
        max_leverage=_as_float("sizing.max_leverage", raw["max_leverage"]),
    )


def _orb_config(value: object) -> OrbConfig:
    raw = _table("orb", value, _ORB_FIELDS)
    return OrbConfig(
        opening_minutes=_as_int("orb.opening_minutes", raw["opening_minutes"]),
        rvol_sessions=_as_int("orb.rvol_sessions", raw["rvol_sessions"]),
        min_rvol=_as_float("orb.min_rvol", raw["min_rvol"]),
        top_k=_as_int("orb.top_k", raw["top_k"]),
        min_price=_as_float("orb.min_price", raw["min_price"]),
        min_adv_shares=_as_int("orb.min_adv_shares", raw["min_adv_shares"]),
        adv_sessions=_as_int("orb.adv_sessions", raw["adv_sessions"]),
        min_atr=_as_float("orb.min_atr", raw["min_atr"]),
        atr_sessions=_as_int("orb.atr_sessions", raw["atr_sessions"]),
        stop_atr_frac=_as_float("orb.stop_atr_frac", raw["stop_atr_frac"]),
        risk_frac=_as_float("orb.risk_frac", raw["risk_frac"]),
        doji_eps=_as_float("orb.doji_eps", raw["doji_eps"]),
        amihud_percentile=_as_float("orb.amihud_percentile", raw["amihud_percentile"]),
        limit_through_ticks=_as_int("orb.limit_through_ticks", raw["limit_through_ticks"]),
    )


def _momentum_config(value: object) -> IntradayMomentumConfig:
    raw = _table("intraday_momentum", value, _MOMENTUM_FIELDS)
    symbol = raw["symbol"]
    if not isinstance(symbol, str):
        raise ValueError(f"intraday_momentum.symbol: se esperaba str, llegó {symbol!r}")
    mode = raw["exit_mode"]
    if not isinstance(mode, str):
        raise ValueError(f"intraday_momentum.exit_mode: se esperaba str, llegó {mode!r}")
    return IntradayMomentumConfig(
        symbol=symbol,
        first_half_minutes=_as_int(
            "intraday_momentum.first_half_minutes", raw["first_half_minutes"],
        ),
        last_half_minutes=_as_int(
            "intraday_momentum.last_half_minutes", raw["last_half_minutes"],
        ),
        require_r12_agreement=_as_bool(
            "intraday_momentum.require_r12_agreement", raw["require_r12_agreement"],
        ),
        use_vol_filter=_as_bool(
            "intraday_momentum.use_vol_filter", raw["use_vol_filter"],
        ),
        min_abs_r1_sigma=_as_float(
            "intraday_momentum.min_abs_r1_sigma", raw["min_abs_r1_sigma"],
        ),
        sigma_sessions=_as_int(
            "intraday_momentum.sigma_sessions", raw["sigma_sessions"],
        ),
        exit_mode=mode,
        cls_cutoff_minutes=_as_int(
            "intraday_momentum.cls_cutoff_minutes", raw["cls_cutoff_minutes"],
        ),
    )


def _vwap_config(value: object) -> VwapReversionConfig:
    raw = _table("vwap_reversion", value, _VWAP_FIELDS)
    style = raw["entry_style"]
    if not isinstance(style, str):
        raise ValueError(f"vwap_reversion.entry_style: se esperaba str, llegó {style!r}")
    return VwapReversionConfig(
        z_n=_as_int("vwap_reversion.z_n", raw["z_n"]),
        entry_z=_as_float("vwap_reversion.entry_z", raw["entry_z"]),
        exit_z=_as_float("vwap_reversion.exit_z", raw["exit_z"]),
        stop_z=_as_float("vwap_reversion.stop_z", raw["stop_z"]),
        skip_open_minutes=_as_int(
            "vwap_reversion.skip_open_minutes", raw["skip_open_minutes"],
        ),
        entry_cutoff_minutes=_as_float(
            "vwap_reversion.entry_cutoff_minutes", raw["entry_cutoff_minutes"],
        ),
        time_stop_minutes=_as_int(
            "vwap_reversion.time_stop_minutes", raw["time_stop_minutes"],
        ),
        vol_sessions=_as_int("vwap_reversion.vol_sessions", raw["vol_sessions"]),
        max_vol_percentile=_as_float(
            "vwap_reversion.max_vol_percentile", raw["max_vol_percentile"],
        ),
        trend_ma=_as_int("vwap_reversion.trend_ma", raw["trend_ma"]),
        trend_look=_as_int("vwap_reversion.trend_look", raw["trend_look"]),
        trend_threshold=_as_float(
            "vwap_reversion.trend_threshold", raw["trend_threshold"],
        ),
        top_k=_as_int("vwap_reversion.top_k", raw["top_k"]),
        risk_frac=_as_float("vwap_reversion.risk_frac", raw["risk_frac"]),
        limit_through_ticks=_as_int(
            "vwap_reversion.limit_through_ticks", raw["limit_through_ticks"],
        ),
        entry_style=style,
    )


def _pairs_config(value: object) -> PairsOuConfig:
    raw = _table("pairs_ou", value, _PAIRS_FIELDS)
    return PairsOuConfig(
        formation_sessions=_as_int(
            "pairs_ou.formation_sessions", raw["formation_sessions"],
        ),
        adf_lags=_as_int("pairs_ou.adf_lags", raw["adf_lags"]),
        entry_z=_as_float("pairs_ou.entry_z", raw["entry_z"]),
        exit_z=_as_float("pairs_ou.exit_z", raw["exit_z"]),
        stop_z=_as_float("pairs_ou.stop_z", raw["stop_z"]),
        half_life_min=_as_float("pairs_ou.half_life_min", raw["half_life_min"]),
        half_life_max=_as_float("pairs_ou.half_life_max", raw["half_life_max"]),
        half_life_stop_mult=_as_float(
            "pairs_ou.half_life_stop_mult", raw["half_life_stop_mult"],
        ),
        leg_timeout_s=_as_float("pairs_ou.leg_timeout_s", raw["leg_timeout_s"]),
        risk_frac=_as_float("pairs_ou.risk_frac", raw["risk_frac"]),
        limit_through_ticks=_as_int(
            "pairs_ou.limit_through_ticks", raw["limit_through_ticks"],
        ),
        entry_cutoff_minutes=_as_float(
            "pairs_ou.entry_cutoff_minutes", raw["entry_cutoff_minutes"],
        ),
        pairs=_parse_pairs(raw["pairs"]),
    )


def _as_config(value: object) -> AvellanedaStoikovConfig:
    raw = _table("avellaneda_stoikov", value, _AS_FIELDS)
    symbol = raw["symbol"]
    if not isinstance(symbol, str):
        raise ValueError(f"avellaneda_stoikov.symbol: se esperaba str, llegó {symbol!r}")
    return AvellanedaStoikovConfig(
        symbol=symbol,
        gamma=_as_float("avellaneda_stoikov.gamma", raw["gamma"]),
        q_max=_as_int("avellaneda_stoikov.q_max", raw["q_max"]),
        quote_qty=_as_int("avellaneda_stoikov.quote_qty", raw["quote_qty"]),
        sigma_bars=_as_int("avellaneda_stoikov.sigma_bars", raw["sigma_bars"]),
        k_bars=_as_int("avellaneda_stoikov.k_bars", raw["k_bars"]),
        min_trades=_as_int("avellaneda_stoikov.min_trades", raw["min_trades"]),
        min_bars=_as_int("avellaneda_stoikov.min_bars", raw["min_bars"]),
        entry_cutoff_minutes=_as_float(
            "avellaneda_stoikov.entry_cutoff_minutes", raw["entry_cutoff_minutes"],
        ),
    )


def _hourly_config(value: object) -> HourlyMomentumConfig:
    raw = _table("hourly_momentum", value, _HOURLY_FIELDS)
    for name in ("first_window_et", "last_window_et"):
        if not isinstance(raw[name], str):
            raise ValueError(
                f"hourly_momentum.{name}: se esperaba 'HH:MM', llegó {raw[name]!r}"
            )
    return HourlyMomentumConfig(
        first_window_et=raw["first_window_et"],
        last_window_et=raw["last_window_et"],
        step_minutes=_as_int("hourly_momentum.step_minutes", raw["step_minutes"]),
        lookback_minutes=_as_int(
            "hourly_momentum.lookback_minutes", raw["lookback_minutes"],
        ),
        sigma_sessions=_as_int("hourly_momentum.sigma_sessions", raw["sigma_sessions"]),
        min_abs_z=_as_float("hourly_momentum.min_abs_z", raw["min_abs_z"]),
        top_k=_as_int("hourly_momentum.top_k", raw["top_k"]),
        max_new_per_window=_as_int(
            "hourly_momentum.max_new_per_window", raw["max_new_per_window"],
        ),
        hold_minutes=_as_int("hourly_momentum.hold_minutes", raw["hold_minutes"]),
        stop_atr_frac=_as_float("hourly_momentum.stop_atr_frac", raw["stop_atr_frac"]),
        atr_sessions=_as_int("hourly_momentum.atr_sessions", raw["atr_sessions"]),
        risk_frac=_as_float("hourly_momentum.risk_frac", raw["risk_frac"]),
        limit_through_ticks=_as_int(
            "hourly_momentum.limit_through_ticks", raw["limit_through_ticks"],
        ),
    )


def _window_clock(name: str, value: object) -> time:
    if not isinstance(value, str):
        raise ValueError(f"hourly_momentum.{name}: se esperaba 'HH:MM', llegó {value!r}")
    try:
        clock = _parse_hhmm(value)
    except ValueError as exc:
        raise ValueError(f"hourly_momentum.{name}: {exc}") from exc
    if not _OPEN_ET < clock < _CLOSE_ET:
        raise ValueError(
            f"hourly_momentum.{name} {value!r} no cae dentro de la sesión regular "
            f"({_OPEN_ET:%H:%M}–{_CLOSE_ET:%H:%M} ET)"
        )
    return clock


def _clock_minutes(clock: time) -> int:
    return clock.hour * 60 + clock.minute


def _parse_pairs(value: object) -> Tuple[Tuple[str, str, str], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"pairs_ou.pairs: se esperaba una lista, llegó {value!r}")
    parsed = []
    for index, item in enumerate(value):
        raw = _table(f"pairs_ou.pairs[{index}]", item, _PAIR_FIELDS)
        leg_a = _pair_symbol(f"pairs_ou.pairs[{index}].leg_a", raw["leg_a"])
        leg_b = _pair_symbol(f"pairs_ou.pairs[{index}].leg_b", raw["leg_b"])
        link = raw["link"]
        if not isinstance(link, str) or not link.strip():
            raise ValueError(f"pairs_ou.pairs[{index}].link vacío")
        parsed.append((leg_a, leg_b, link.strip()))
    return tuple(parsed)


def _pair_symbol(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: se esperaba str, llegó {value!r}")
    symbol = value.strip().upper()
    if not symbol or any(char.isspace() for char in symbol):
        raise ValueError(f"{name}: símbolo inválido {value!r}")
    return symbol


def _validated_pairs(pairs: object) -> Tuple[Tuple[str, str, str], ...]:
    if isinstance(pairs, str) or not isinstance(pairs, tuple) or not pairs:
        raise ValueError("pairs_ou.pairs vacío: los pares se escriben en la config")
    seen_symbols = set()
    seen_pairs = set()
    cleaned = []
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"pairs_ou.pairs: elemento inválido {item!r}")
        leg_a, leg_b, link = item
        leg_a = _pair_symbol("pairs_ou.pairs.leg_a", leg_a)
        leg_b = _pair_symbol("pairs_ou.pairs.leg_b", leg_b)
        if leg_a == leg_b:
            raise ValueError(f"pairs_ou.pairs: {leg_a} está en las dos patas")
        if not isinstance(link, str) or not link.strip():
            raise ValueError(f"pairs_ou.pairs: {leg_a}/{leg_b} no trae el vínculo")
        key = frozenset((leg_a, leg_b))
        if key in seen_pairs:
            raise ValueError(f"pairs_ou.pairs: {leg_a}/{leg_b} está repetido")
        overlap = ({leg_a, leg_b}) & seen_symbols
        if overlap:
            raise ValueError(
                f"pairs_ou.pairs: {sorted(overlap)} ya está en otro par "
                "(un símbolo en dos pares mezcla las coberturas)"
            )
        seen_pairs.add(key)
        seen_symbols.update((leg_a, leg_b))
        cleaned.append((leg_a, leg_b, link.strip()))
    return tuple(cleaned)


def _decision_config(value: object) -> DecisionConfig:
    raw = _table("decision", value, _DECISION_FIELDS)
    return DecisionConfig(
        min_expectancy=_as_float("decision.min_expectancy", raw["min_expectancy"]),
        min_t_stat=_as_float("decision.min_t_stat", raw["min_t_stat"]),
        min_dsr=_as_float("decision.min_dsr", raw["min_dsr"]),
        max_pbo=_as_float("decision.max_pbo", raw["max_pbo"]),
        min_oos_trades=_as_int("decision.min_oos_trades", raw["min_oos_trades"]),
        low_frequency_strategies=_string_list(
            "decision.low_frequency_strategies", raw["low_frequency_strategies"],
        ),
        max_year_pnl_share=_as_float(
            "decision.max_year_pnl_share", raw["max_year_pnl_share"],
        ),
        best_days_dropped=_as_int("decision.best_days_dropped", raw["best_days_dropped"]),
        min_paper_sessions=_as_int(
            "decision.min_paper_sessions", raw["min_paper_sessions"],
        ),
        min_paper_trades=_as_int("decision.min_paper_trades", raw["min_paper_trades"]),
        min_slippage_sessions=_as_int(
            "decision.min_slippage_sessions", raw["min_slippage_sessions"],
        ),
        slippage_se_mult=_as_float("decision.slippage_se_mult", raw["slippage_se_mult"]),
        required_drills=_string_list("decision.required_drills", raw["required_drills"]),
    )


def _sim_config(value: object) -> SimConfig:
    raw = _table("sim", value, _SIM_FIELDS)
    mode = raw["fill_mode"]
    if not isinstance(mode, str):
        raise ValueError(f"sim.fill_mode: se esperaba str, llegó {mode!r}")
    return SimConfig(
        participation_cap=_as_float("sim.participation_cap", raw["participation_cap"]),
        partial_prob=_as_float("sim.partial_prob", raw["partial_prob"]),
        fill_mode=mode,
        latency_bars=_as_int("sim.latency_bars", raw["latency_bars"]),
    )


def _table(name: str, value: object, fields: Sequence[str]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: se esperaba una tabla, llegó {value!r}")
    raw = dict(value)
    unknown = sorted(set(raw) - set(fields))
    missing = sorted(set(fields) - set(raw))
    if unknown or missing:
        raise ValueError(f"{name}: faltan {missing}, desconocidas {unknown}.")
    return raw


def _as_date(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: se esperaba una fecha ISO, llegó {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name}: fecha ISO inválida {value!r}") from exc
    return value


def _as_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name}: se esperaba bool, llegó {value!r}")
    return value


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: se esperaba int, llegó {value!r}")
    return value


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: se esperaba float, llegó {value!r}")
    return float(value)


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _string_list(name: str, value: object) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name}: se esperaba una lista de str, llegó {value!r}")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}: elemento inválido {item!r}")
        out.append(item.strip())
    if len(out) != len(set(out)):
        raise ValueError(f"{name}: hay repetidos")
    return tuple(out)


def _strategy_ids(name: str, value: object) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} debe ser una secuencia de ids")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}: id inválido {item!r}")
        text = item.strip()
        if not text.replace("_", "").isalnum() or text[0].isdigit():
            raise ValueError(f"{name}: id inválido {item!r}")
        out.append(text)
    if len(out) != len(set(out)):
        raise ValueError(f"{name}: hay repetidos")
    return tuple(out)


# Reexportados para que un lector vea los defaults al lado del esquema.
__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FEED",
    "DEFAULT_GRACE_S",
    "FEEDS",
    "HFT_SCHEMA_VERSION",
    "AvellanedaStoikovConfig",
    "CostConfig",
    "DecisionConfig",
    "HftConfig",
    "PROTOCOL_DRILLS",
    "HourlyMomentumConfig",
    "IntradayMomentumConfig",
    "OrbConfig",
    "PairsOuConfig",
    "VwapReversionConfig",
    "RiskConfig",
    "SimConfig",
    "SizingConfig",
    "config_symbols",
    "hft_config_from_dict",
    "load_hft_config",
]
