"""
strategy.py — Construcción de cartera única para backtest y modo vivo.

``build_targets(state, prices, volumes, t, cfg)`` es la ÚNICA ruta que
convierte datos + cartera en pesos objetivo. La usan
``src.backtest_engine.run_backtest`` y ``scripts/run_periodic_rebalance.py``:
con el mismo ``StrategyConfig`` y los mismos datos, ambos producen los mismos
pesos en t, así que las métricas del backtest son de la estrategia que corre
en vivo.

Pasos (todos causales, R-05):
  1. Universo: columnas de ``universe`` (default: todas) → filtro de liquidez
     ``filter_universe_at_date`` (opt-in ``universe_filter``, R-03).
  2. Ventana ``window`` hasta t (``get_window``); activos con historia
     completa (al menos ``min_assets``, si no ``InsufficientDataError``).
  3. Stack de señales (opt-in): régimen + señales + convicción (+ top-k).
  4. Sizing: HRP u ``run_optimizer()`` (mu/Sigma + t/volúmenes), tope
     ``max_weight``.
  5. Tilts de convicción → tope ``max_weight``.
  6. Salidas (``exit_rules``) sobre la cartera ``state`` → re-tope.
  7. Topes por grupo (``apply_group_caps``).

Configuración: ``StrategyConfig`` (inmutable). El modo vivo la carga de un
archivo versionado (``config/live.toml``, ``load_strategy_config``) y registra
``config_hash()`` en rebalance_history.csv. Los defaults de ``StrategyConfig``
son los del backtest histórico; el archivo vivo declara TODOS los valores
(ninguno se hereda en silencio).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.conviction_scoring import conviction_score, conviction_to_tilts
from src.covariance_estimators import estimate_covariance
from src.exit_rules import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_TRAILING_WINDOW,
    ExitConfig,
    apply_exits_to_weights,
    evaluate_exits,
)
from src.exit_rules import required_history as exit_required_history
from src.expected_returns import estimate_expected_returns
from src.history_requirements import PERMISSIVE
from src.hrp_sizing import hrp_weights
from src.hrp_sizing import required_history as hrp_required_history
from src.portfolio_optimizer import OPTIMIZERS, OptimizationInput, run_optimizer
from src.portfolio_state import PortfolioState
from src.regime_detector import classify_regime
from src.regime_detector import required_history as regime_required_history
from src.rebalancing_engine import _apply_turnover_cap
from src.rolling_engine import get_window
from src.schemas import CONFIG_SCHEMA_VERSION
from src.sector_constraints import apply_group_caps, sell_down_to_caps
from src.settings import REPO_ROOT
from src.signal_stack import (
    SignalStackConfig,
    compute_signal_scores_with_meta,
    factor_requirements,
)
from src.universe_filter import (
    compute_rolling_amihud,
    compute_rolling_volume,
    filter_universe_at_date,
)

logger = logging.getLogger("strategy")

# Archivo versionado del modo vivo (relativo al repo, nunca al cwd).
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "live.toml"
# Reexportada desde src.schemas (mismo valor). provenance no importa este módulo.

# Filtro de liquidez con < 3 activos → se ignora (universo completo).
_MIN_FILTERED = 3
_EXIT_COLUMNS = ["ticker", "rule", "detail"]

# Campos de OptimizationInput que build_targets llena en todo rebalanceo
# (w_prev es None con libro vacío; illiquidity/costs nunca se pasan).
_OPTIMIZER_INPUTS = frozenset({"mu", "sigma", "t", "volumes"})


class InsufficientDataError(RuntimeError):
    """Menos de ``min_assets`` activos con historia completa en la ventana."""


# ===================================================================
# Config
# ===================================================================
@dataclass(frozen=True)
class StrategyConfig:
    """Todo lo que decide los pesos objetivo (sin parámetros de simulación).

    Defaults = backtest histórico (Fase 5/6). El modo vivo no usa defaults:
    ``load_strategy_config`` exige cada campo en el archivo.
    """

    # --- Sizing ---
    opt_method: str = "mv_classic"     # portfolio_optimizer method OR 'hrp'
    window: int = 252                  # ventana de estimación (filas <= t)
    mu_method: str = "historical"      # expected returns estimator
    benchmark_col: str = "SPY"         # market proxy for capm_shrunk betas
    cov_method: str = "ledoit_wolf"    # covariance estimator
    max_weight: float = 0.30           # tope por activo (R-04); 1.0 = sin tope
    turnover_limit: float = 0.20
    tc_lambda: float = 0.001
    kappa: float = 0.10                # robustness (for robust methods)
    # --- Universo (R-03) ---
    universe_filter: bool = True       # filter_universe_at_date en t
    filter_amihud_pctile: float = 90.0
    filter_min_volume: float = 100_000.0
    filter_window: int = 30
    min_assets: int = 1                # < min_assets con historia → InsufficientDataError
    # --- Señales (Fase 6) ---
    use_signal_stack: bool = False
    signal_config: Optional[SignalStackConfig] = None  # None → SignalStackConfig()
    regime_window: int = 63
    conviction_top_k: Optional[int] = None  # None = tilt all active names
    # --- Topes por grupo / sleeve (criterios.md v2) ---
    groups: Optional[Dict[str, str]] = None  # ticker -> "mercado/sector"
    # grupo (o tupla de grupos) -> tope fraccion.
    group_caps: Optional[Dict] = None
    max_weight_by_group: Optional[Dict[str, float]] = None
    # --- Criterios de salida (exit_rules) ---
    use_exit_rules: bool = False
    stop_loss_pct: Optional[float] = DEFAULT_STOP_LOSS_PCT
    take_profit_pct: Optional[float] = DEFAULT_TAKE_PROFIT_PCT
    trailing_stop_pct: Optional[float] = None
    trailing_window: int = DEFAULT_TRAILING_WINDOW
    exit_on_illiquid: bool = True
    take_profit_dollars: Optional[float] = None  # PnL >= +USD → cierra
    stop_loss_dollars: Optional[float] = None    # PnL <= -USD → cierra
    # --- Costos (estimación de turnover × bps) ---
    commission_bps: float = 5.0
    spread_bps: float = 2.0

    def __post_init__(self) -> None:
        # Una sola representación de "factores por defecto" (igualdad y hash).
        if self.signal_config is None:
            object.__setattr__(self, "signal_config", SignalStackConfig())
        _check_opt_method(self.opt_method)

    @property
    def cost_bps(self) -> float:
        return float(self.commission_bps) + float(self.spread_bps)

    def exit_config(self) -> Optional[ExitConfig]:
        """ExitConfig de las salidas, o None si ``use_exit_rules`` es False."""
        if not self.use_exit_rules:
            return None
        return ExitConfig(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            trailing_window=self.trailing_window,
            exit_on_illiquid=self.exit_on_illiquid,
            take_profit_dollars=self.take_profit_dollars,
            stop_loss_dollars=self.stop_loss_dollars,
        )

    def max_weight_by_ticker(self) -> Optional[pd.Series]:
        """Tope por activo segun max_weight_by_group (None si no aplica)."""
        if not (self.max_weight_by_group and self.groups):
            return None
        return pd.Series(self.groups).map(
            lambda grp: self.max_weight_by_group.get(grp, self.max_weight)
        ).astype(float)

    def to_dict(self) -> Dict[str, Any]:
        """Forma canónica JSON (None → null; tuplas de grupos → listas).

        Numéricos normalizados por tipo declarado (1 y 1.0 → 1.0 en un campo
        float): configs iguales tienen el mismo ``config_hash``.
        """
        out: Dict[str, Any] = {}
        for name, hint in typing.get_type_hints(StrategyConfig).items():
            value = getattr(self, name)
            if name == "signal_config":
                hints = typing.get_type_hints(SignalStackConfig)
                value = {k: _canonical(v, hints[k])
                         for k, v in dataclasses.asdict(value).items()}
            elif name == "group_caps" and value is not None:
                value = sorted((
                    {"groups": [key] if isinstance(key, str) else list(key),
                     "cap": float(cap)}
                    for key, cap in value.items()
                ), key=lambda row: row["groups"])
            elif name == "max_weight_by_group" and value is not None:
                value = {str(k): float(v) for k, v in value.items()}
            elif isinstance(value, dict):
                value = dict(value)
            else:
                value = _canonical(value, hint)
            out[name] = value
        return out

    def config_hash(self) -> str:
        """sha256 de la forma canónica: identifica la configuración efectiva."""
        canonical = json.dumps(self.to_dict(), sort_keys=True,
                               separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_opt_method(method: str) -> None:
    """Fail-fast al construir la config, no en la fecha 37 del backtest.

    ``method`` debe ser 'hrp' o estar en ``OPTIMIZERS``, y build_targets
    debe poder darle todas sus entradas requeridas (``Optimizer.requires``).
    """
    if method == "hrp":
        return
    spec = OPTIMIZERS.get(method)
    if spec is None:
        raise ValueError(f"opt_method={method!r} desconocido; válidos: "
                         f"{['hrp', *OPTIMIZERS]}.")
    missing = sorted(spec.requires - _OPTIMIZER_INPUTS)
    if missing:
        raise ValueError(f"opt_method={method!r} requiere {missing}, que "
                         "build_targets no provee.")


def _canonical(value: Any, hint: Any) -> Any:
    """float(value) en campos float / Optional[float]; resto sin cambio."""
    if value is None or isinstance(value, bool):
        return value
    if hint is float or (typing.get_origin(hint) is Union
                         and float in typing.get_args(hint)):
        return float(value)
    return value


# ===================================================================
# Archivo de configuración (config/live.toml)
# ===================================================================
_TABLE_FIELDS = ("signal_config", "groups", "group_caps", "max_weight_by_group")


def _scalar_fields() -> Dict[str, Any]:
    hints = typing.get_type_hints(StrategyConfig)
    return {f.name: hints[f.name] for f in dataclasses.fields(StrategyConfig)
            if f.name not in _TABLE_FIELDS}


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """Valida el tipo de un escalar del archivo; ``false`` = off en Optional."""
    optional = typing.get_origin(hint) is Union and type(None) in typing.get_args(hint)
    base = next(a for a in typing.get_args(hint) if a is not type(None)) if optional else hint
    if optional and (value is None or value is False):
        return None
    if base is bool:
        ok = isinstance(value, bool)
    elif base is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif base is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    else:
        ok = isinstance(value, base)
    if not ok:
        off = " (o false = off)" if optional else ""
        raise ValueError(f"{name}: se esperaba {base.__name__}{off}, "
                         f"llegó {value!r}")
    return value


def strategy_config_from_dict(data: Mapping[str, Any]) -> StrategyConfig:
    """StrategyConfig desde un dict (TOML o ``to_dict()``), estricto.

    Todos los escalares son obligatorios (nada se hereda de los defaults del
    backtest); claves desconocidas son error. Tablas opcionales:
    ``signal_config`` (todas sus claves), ``groups``, ``group_caps``
    (lista de ``{groups = [...], cap = x}``) y ``max_weight_by_group``.
    En un Optional, ``false`` (o null) significa "off" (None).

    Raises:
        ValueError: clave faltante/desconocida, tipo inválido o
            ``schema_version`` distinto de ``CONFIG_SCHEMA_VERSION``.
    """
    data = dict(data)
    version = data.pop("schema_version", CONFIG_SCHEMA_VERSION)
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version={version!r} no soportado "
                         f"(se espera {CONFIG_SCHEMA_VERSION}).")
    scalars = _scalar_fields()
    unknown = sorted(set(data) - set(scalars) - set(_TABLE_FIELDS))
    missing = sorted(set(scalars) - set(data))
    if unknown or missing:
        raise ValueError(f"Config de estrategia inválida: faltan {missing}, "
                         f"desconocidas {unknown}.")
    kwargs: Dict[str, Any] = {name: _coerce(name, data[name], hint)
                              for name, hint in scalars.items()}

    sig = data.get("signal_config")
    if sig is not None:
        sig_fields = typing.get_type_hints(SignalStackConfig)
        bad = sorted(set(sig) ^ set(sig_fields))
        if bad:
            raise ValueError(f"signal_config: claves faltantes/desconocidas {bad}.")
        kwargs["signal_config"] = SignalStackConfig(
            **{k: _coerce(f"signal_config.{k}", v, sig_fields[k]) for k, v in sig.items()})
    if data.get("groups") is not None:
        kwargs["groups"] = {str(k): str(v) for k, v in data["groups"].items()}
    if data.get("group_caps") is not None:
        caps = {}
        for row in data["group_caps"]:
            groups = [str(g) for g in row["groups"]]
            key = groups[0] if len(groups) == 1 else tuple(groups)
            caps[key] = _coerce("group_caps.cap", row["cap"], float)
        kwargs["group_caps"] = caps
    if data.get("max_weight_by_group") is not None:
        kwargs["max_weight_by_group"] = {
            str(k): _coerce(f"max_weight_by_group.{k}", v, float)
            for k, v in data["max_weight_by_group"].items()}
    return StrategyConfig(**kwargs)


def _toml_value(value: Any) -> str:
    if value is None:
        return "false"  # off
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return json.dumps(value)  # int, str (comillas básicas JSON = TOML)


def strategy_config_to_toml(cfg: StrategyConfig) -> str:
    """TOML que ``load_strategy_config`` vuelve a cargar idéntico (mismo hash).

    Útil para pasar a config/live.toml una estrategia ya backtesteada.
    """
    data = cfg.to_dict()
    lines = [f"schema_version = {CONFIG_SCHEMA_VERSION}"]
    lines += [f"{name} = {_toml_value(data[name])}" for name in _scalar_fields()]
    lines += ["", "[signal_config]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in data["signal_config"].items()]
    for table in ("groups", "max_weight_by_group"):
        if data[table] is not None:
            lines += ["", f"[{table}]"]
            lines += [f"{json.dumps(k)} = {_toml_value(v)}"
                      for k, v in data[table].items()]
    for row in data["group_caps"] or []:
        lines += ["", "[[group_caps]]", f"groups = {json.dumps(row['groups'])}",
                  f"cap = {_toml_value(row['cap'])}"]
    return "\n".join(lines) + "\n"


def load_strategy_config(path: Union[str, Path, None] = None) -> StrategyConfig:
    """Carga el TOML versionado (default ``config/live.toml``) a StrategyConfig."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return strategy_config_from_dict(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


# ===================================================================
# Historia requerida
# ===================================================================
def history_requirements(cfg: StrategyConfig) -> Dict[str, int]:
    """Filas hábiles (índice <= t) que exige cada componente activo del stack."""
    reqs: Dict[str, int] = {}
    if cfg.use_signal_stack:
        reqs.update({f"signal_stack.{k}": v
                     for k, v in factor_requirements(cfg.signal_config).items()})
        reqs["regime_detector"] = regime_required_history(window=cfg.regime_window)
    # La ventana alimenta HRP y los estimadores mu/Sigma por igual.
    reqs["sizing"] = hrp_required_history(cfg.window)
    if cfg.universe_filter:
        reqs["universe_filter"] = int(cfg.filter_window)
    exit_cfg = cfg.exit_config()
    if exit_cfg is not None:
        reqs["exit_rules"] = exit_required_history(exit_cfg)
    return reqs


# ===================================================================
# Helpers de construcción
# ===================================================================
def enforce_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """Project long-only weights onto the max-weight simplex.

    A single clip-then-renormalize pass can re-violate ``max_weight``
    (e.g. HRP [0.99, 0.01] with cap 0.30 → clip [0.30, 0.01] → renorm
    ≈ [0.97, 0.03]). Excess from capped names is redistributed equally
    across names still below the cap. If ``n * max_weight < 1``, fall
    back to equal weight.
    """
    w = weights.fillna(0.0).clip(lower=0.0).astype(float)
    if w.empty:
        return w
    n = len(w)
    if max_weight <= 0:
        return pd.Series(1.0 / n, index=w.index)
    if n * max_weight < 1.0 - 1e-12:
        return pd.Series(1.0 / n, index=w.index)

    total = float(w.sum())
    if total <= 1e-12:
        return pd.Series(1.0 / n, index=w.index)
    w = w / total

    for _ in range(n + 5):
        over = w > max_weight + 1e-12
        if not bool(over.any()):
            return w
        excess = float((w[over] - max_weight).sum())
        w = w.copy()
        w[over] = max_weight
        under = w < max_weight - 1e-12
        n_under = int(under.sum())
        if n_under == 0 or excess <= 1e-15:
            break
        # Equal share to all under-cap names (including current zeros)
        # so mass is not trapped on a subset that then re-breaches.
        w[under] = w[under] + excess / n_under

    total = float(w.sum())
    if total <= 1e-12:
        return pd.Series(1.0 / n, index=w.index)
    w = w / total
    # Final safety clip if float dust remains
    if float(w.max()) > max_weight + 1e-9:
        return pd.Series(1.0 / n, index=w.index)
    return w


def turnover_cost(
    w_prev: pd.Series,
    w_target: pd.Series,
    cfg: StrategyConfig,
) -> Tuple[float, float]:
    """(turnover L1 sobre el índice objetivo, costo como fracción del NAV).

    costo = turnover × (commission_bps + spread_bps) / 10_000.
    """
    aligned = w_prev.reindex(w_target.index).fillna(0.0)
    turnover = float((w_target - aligned).abs().sum())
    return turnover, turnover * (cfg.cost_bps / 10_000.0)


def uses_capm(cfg: StrategyConfig) -> bool:
    """True when the run estimates mu with capm_shrunk (HRP never uses mu)."""
    return (
        cfg.opt_method != "hrp"
        and cfg.mu_method.lower().strip() == "capm_shrunk"
    )


def _estimate_mu(
    cfg: StrategyConfig,
    wp: pd.DataFrame,
    prices: pd.DataFrame,
    sigma: pd.DataFrame,
) -> pd.Series:
    """Expected returns for the active window ``wp``.

    ``capm_shrunk`` needs the benchmark for betas, but the active set often
    excludes it (liquidity filter, stock-only universe). Estimating on
    ``wp`` alone raised every rebalance, which run_backtest swallowed →
    silent flat NAV. Borrow the benchmark from ``prices`` on the same
    (pre-t) window for estimation only. A window where the benchmark has
    gaps raises (a strict backtest aborts; strict=False records it as a
    failed rebalance) rather than silently swapping in historical mu.
    """
    method = cfg.mu_method.lower().strip()
    bench = cfg.benchmark_col
    if not uses_capm(cfg) or bench in wp.columns:
        return estimate_expected_returns(
            wp, method=method, cov_matrix=sigma, benchmark_col=bench,
        )

    if bench not in prices.columns:
        raise ValueError(f"mu_method='capm_shrunk' needs benchmark '{bench}' "
                         "in the price panel.")
    bench_px = prices[bench].reindex(wp.index)
    if bench_px.isna().any():
        raise ValueError(
            f"capm_shrunk: benchmark '{bench}' has gaps in the window "
            f"ending {wp.index[-1].date()}."
        )
    mu = estimate_expected_returns(
        wp.assign(**{bench: bench_px}), method=method, benchmark_col=bench,
    )
    return mu.drop(bench)


def _row_major(frame: pd.DataFrame) -> pd.DataFrame:
    """Copia float con el mismo layout de memoria venga de donde venga.

    clean_prices (vivo) devuelve bloques en otro orden que el backtest; numpy
    reduce en distinto orden según el layout → pesos que difieren en ~1e-16.
    Con un layout fijo, backtest y vivo dan pesos idénticos bit a bit.
    """
    return pd.DataFrame(np.ascontiguousarray(frame.to_numpy(dtype=float)),
                        index=frame.index, columns=frame.columns)


def _eligible(
    sp: pd.DataFrame,
    sv: pd.DataFrame,
    t: pd.Timestamp,
    cfg: StrategyConfig,
) -> Tuple[List[str], List[str]]:
    """(elegibles tras el filtro de liquidez en t, excluidos).

    Amihud/volumen rodantes sobre las últimas ``filter_window`` + 1 filas
    <= t: mismo cálculo en backtest y vivo. Con < 3 elegibles el filtro se
    ignora (universo completo).
    """
    cols = list(sp.columns)
    if not cfg.universe_filter:
        return cols, []
    upto = sp.loc[sp.index <= t].iloc[-(cfg.filter_window + 1):]
    vol = sv.reindex(index=upto.index, columns=cols)
    active, log = filter_universe_at_date(
        t, sp, sv,
        amihud_percentile_threshold=cfg.filter_amihud_pctile,
        min_volume=cfg.filter_min_volume,
        window=cfg.filter_window,
        rolling_amihud=compute_rolling_amihud(upto, vol, cfg.filter_window),
        rolling_avg_vol=compute_rolling_volume(vol, cfg.filter_window),
    )
    excluded = log["ticker"].tolist() if not log.empty else []
    if len(active) < _MIN_FILTERED:
        return cols, excluded
    return list(active), excluded


def prev_weights(
    state: PortfolioState,
    prices: pd.DataFrame,
    t: pd.Timestamp,
    active: List[str],
) -> Optional[pd.Series]:
    """Pesos actuales alineados a ``active`` (suma 1); None con libro vacío.

    Posiciones fuera de ``active`` se descartan antes de renormalizar; si
    ninguna está en ``active`` → equal weight.
    """
    prices_t = prices.loc[t] if t in prices.index else pd.Series(dtype=float)
    held = state.weights(prices_t)
    if held.empty:
        return None
    w = held.reindex(active).fillna(0.0)
    s = float(w.sum())
    if s > 1e-10:
        return w / s
    return pd.Series(np.ones(len(active)) / len(active), index=active)


def _universe_window(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    cfg: StrategyConfig,
    universe: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """Pasos 1-2 de build_targets: (sp, sv, eligible, excluidos, wp).

    ``wp`` = ventana hasta t de los activos con historia completa (sus
    columnas son los activos a optimizar).

    Raises:
        InsufficientDataError: < ``cfg.min_assets`` activos con historia.
    """
    cols = (list(prices.columns) if universe is None
            else [tk for tk in universe if tk in prices.columns])
    sp = _row_major(prices[cols])
    sv = _row_major(volumes.reindex(index=prices.index, columns=cols))
    eligible, universe_excluded = _eligible(sp, sv, t, cfg)
    window_prices = get_window(sp, t, window=cfg.window)
    wp = window_prices[eligible].dropna(axis=1, how="any")
    if wp.shape[1] < 2:
        wp = window_prices.dropna(axis=1, how="any")
    if wp.shape[1] < max(cfg.min_assets, 1):
        raise InsufficientDataError(f"Insufficient assets in window at {t.date()}")
    return sp, sv, eligible, universe_excluded, wp


def active_assets(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    cfg: StrategyConfig,
    *,
    universe: Optional[Sequence[str]] = None,
) -> List[str]:
    """Activos que build_targets optimiza en t (universo + ventana, R-03/R-05).

    Mismo cálculo que ``TargetResult.active``; sirve para dar contexto cuando
    build_targets falla después de elegirlos (p.ej. en el optimizador).
    """
    return _universe_window(prices, volumes, t, cfg, universe)[4].columns.tolist()


# ===================================================================
# Construcción
# ===================================================================
@dataclass(frozen=True)
class TargetResult:
    """Salida de ``build_targets``.

    Attributes:
        weights: Pesos objetivo por ticker de ``active`` (suman 1, o 0 si
            todo se cerró → caja).
        regime: Régimen en t (None sin stack de señales).
        exits: Salidas evaluadas [ticker, rule, detail] (incluye posiciones
            fuera de ``active``, que ya van a peso 0).
        meta: expected_sharpe, converged, fallback_used (pesos = equal
            weight de respaldo del optimizador), method, factors_used,
            factors_excluded, regime_breadth, universe_excluded.
        active: Tickers con historia completa en la ventana (lo optimizado).
        eligible: Tickers tras el filtro de universo (antes de exigir historia).
        w_prev: Pesos de ``state`` alineados a ``active`` (None: libro vacío).
    """

    weights: pd.Series
    regime: Optional[str]
    exits: pd.DataFrame
    meta: Dict[str, Any]
    active: Tuple[str, ...] = ()
    eligible: Tuple[str, ...] = ()
    w_prev: Optional[pd.Series] = None


def build_targets(
    state: PortfolioState,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    cfg: StrategyConfig,
    *,
    universe: Optional[Sequence[str]] = None,
    history_mode: str = PERMISSIVE,
) -> TargetResult:
    """Pesos objetivo en t: universo → sizing → señales → salidas → topes.

    Args:
        state: Cartera fuente de verdad en t (ledger, broker o pesos del
            backtest): salidas y pesos previos salen SOLO de aquí.
        prices: Precios (date × ticker). Puede incluir posiciones fuera de
            ``universe`` (solo se usan para valorar y evaluar salidas) y
            filas posteriores a t (se ignoran).
        volumes: Volúmenes alineados (NaN = desconocido).
        t: Fecha de rebalanceo.
        cfg: Estrategia.
        universe: Tickers invertibles (default: columnas de ``prices``).
        history_mode: "strict" (componente sin historia → error) o
            "permissive" (régimen/señales degradan).

    Si el optimizador no converge, los pesos son su equal weight de respaldo:
    se registra un warning y ``meta["fallback_used"]`` queda en True.

    Raises:
        InsufficientDataError: < ``cfg.min_assets`` activos con historia.
        InsufficientHistoryError: history_mode="strict" y falta historia.
        ValueError: ``group_caps`` sin ``groups``; tope infactible.
    """
    if cfg.group_caps and not cfg.groups:
        raise ValueError("group_caps requires `groups` (ticker -> group map).")

    # 1-2. Universo + ventana (R-03, R-05)
    sp, sv, eligible, universe_excluded, wp = _universe_window(
        prices, volumes, t, cfg, universe)
    active = wp.columns.tolist()
    w_prev = prev_weights(state, prices, t, active)

    # 3. Señales
    regime: Optional[str] = None
    conviction = None
    meta: Dict[str, Any] = {"factors_used": [], "factors_excluded": {},
                            "regime_breadth": False,
                            "universe_excluded": universe_excluded}
    if cfg.use_signal_stack:
        report = classify_regime(sp[active], t, window=cfg.regime_window,
                                 mode=history_mode)
        regime = report.state
        scores, sig_meta = compute_signal_scores_with_meta(
            sp, sv, t, config=cfg.signal_config, mode=history_mode,
        )
        meta.update(factors_used=sig_meta["factors_used"],
                    factors_excluded=sig_meta["factors_excluded"],
                    regime_breadth=report.breadth is not None)
        # Tickers sin ningún factor → score neutro
        conviction = conviction_score(scores.reindex(active).fillna(0.0), regime)
        if cfg.conviction_top_k is not None and cfg.conviction_top_k > 0:
            top = conviction.nlargest(min(cfg.conviction_top_k, len(conviction)))
            conviction = conviction.where(conviction.index.isin(top.index), other=-1e9)

    # 4. Sizing
    if cfg.opt_method == "hrp":
        rets = wp.ffill().pct_change().dropna(how="all")
        w = hrp_weights(rets).reindex(active).fillna(0.0)
        s = w.sum()
        w = w / s if s > 1e-10 else pd.Series(
            np.ones(len(active)) / len(active), index=active,
        )
        w = enforce_max_weight(w, cfg.max_weight)
        meta.update(expected_sharpe=0.0, converged=True, fallback_used=False,
                    method="hrp")
    else:
        # Covariance first: black_litterman requires cov_matrix.
        sigma = estimate_covariance(wp, method=cfg.cov_method)
        mu = _estimate_mu(cfg, wp, prices, sigma)
        # Todo lo de _OPTIMIZER_INPUTS, siempre: el contrato se validó en la
        # config. Volúmenes hasta t (R-05) de los activos optimizados.
        opt = run_optimizer(cfg.opt_method, OptimizationInput(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev,
            t=t,
            volumes=sv.loc[sv.index <= t, active],
            max_weight=cfg.max_weight,
            turnover_limit=cfg.turnover_limit,
            tc_lambda=cfg.tc_lambda,
            kappa=cfg.kappa,
            groups=cfg.groups,
            group_caps=cfg.group_caps,
            max_weight_by_group=cfg.max_weight_by_group,
        ))
        w = opt.weights
        if opt.fallback_used:
            logger.warning(
                "%s: optimizer %s used the equal-weight fallback (status=%s).",
                t.date(), cfg.opt_method, opt.status,
            )
        meta.update(expected_sharpe=opt.meta["expected_sharpe"],
                    converged=opt.converged, fallback_used=opt.fallback_used,
                    method=cfg.opt_method)

    # 5. Tilts de convicción
    if conviction is not None:
        w = conviction_to_tilts(conviction, base_weights=w)
        w = enforce_max_weight(w, cfg.max_weight)

    # 6. Salidas sobre la cartera fuente de verdad
    exits = pd.DataFrame(columns=_EXIT_COLUMNS)
    exit_cfg = cfg.exit_config()
    if exit_cfg is not None:
        held, entries, shares = state.exit_inputs()
        exits = evaluate_exits(t, prices, held, entries, active=active,
                               config=exit_cfg, shares=shares)
        # Tickers fuera del objetivo (p.ej. fuera de active) ya van a 0.
        hit = exits[exits["ticker"].isin(w.index)]
        if not hit.empty:
            w = apply_exits_to_weights(w, hit)
            # Re-imponer R-04 tras renormalizar
            w = w.clip(upper=cfg.max_weight)
            s = w.sum()
            w = w / s if s > 1e-10 else w

    # 7. Topes por grupo al final (tras tilts/HRP/salidas): renormalizar tras
    # un cierre puede inflar un grupo. Se proyecta solo sobre nombres sin
    # cierre para no devolverles peso; todo cerrado -> caja (cumple topes).
    # Topes infactibles propagan (igual que optimize() en mv/robust).
    if cfg.group_caps:
        keep = w.index.difference(exits["ticker"].tolist(), sort=False)
        if float(w[keep].sum()) > 1e-10:
            w = apply_group_caps(
                w[keep],
                groups=pd.Series(cfg.groups).reindex(keep),
                caps=cfg.group_caps,
                max_weight=cfg.max_weight,
                max_weight_by_ticker=cfg.max_weight_by_ticker(),
            ).reindex(w.index).fillna(0.0)

    return TargetResult(
        weights=w,
        regime=regime,
        exits=exits.reset_index(drop=True),
        meta=meta,
        active=tuple(active),
        eligible=tuple(eligible),
        w_prev=w_prev,
    )


def execution_weights(
    targets: TargetResult,
    state: PortfolioState,
    cfg: StrategyConfig,
) -> pd.Series:
    """Libro a ejecutar en t (backtest y vivo): objetivo con tope de turnover.

    El libro que se mantiene (``targets.w_prev`` sin las salidas) rota hacia
    el objetivo a lo sumo ``cfg.turnover_limit`` (R-02, ``_apply_turnover_cap``)
    y, con ``group_caps``, vende solo el exceso (``sell_down_to_caps``). La
    caja y lo vendido por salidas no son turnover: van directo al objetivo,
    así un libro vacío (despliegue inicial) recibe el objetivo completo y las
    salidas cierran enteras. Libro invertido sin salidas → solo el blend.

    Returns:
        Pesos sobre el índice de ``targets.weights`` (dimensionar sobre el NAV).
    """
    w_target = targets.weights
    if targets.w_prev is None or state.equity <= 1e-10:
        return w_target
    keep = targets.w_prev.copy()
    keep[keep.index.isin(targets.exits["ticker"])] = 0.0
    kept = float(keep.sum())
    if kept <= 1e-10:
        return w_target
    rotated = _apply_turnover_cap(keep / kept, w_target, cfg.turnover_limit)
    if cfg.group_caps:
        rotated = sell_down_to_caps(rotated, w_target, cfg.groups, cfg.group_caps)
    # Fracción del NAV que rota con tope; el resto (caja + salidas) al objetivo
    invested = min(max(1.0 - state.cash / state.equity, 0.0), 1.0)
    r = invested * kept
    return r * rotated + (1.0 - r) * w_target
