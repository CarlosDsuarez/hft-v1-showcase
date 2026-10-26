"""
backtest_engine.py — T-25: Motor de backtesting rolling OOS.

Loop temporal:
  for t in rebalance_dates:
      1. Cartera en t → PortfolioState (ledger o libro clásico)
      2. Pesos objetivo: src.strategy.build_targets (universo R-03, ventana
         R-05, mu/Sigma u HRP, señales, salidas, topes) — la MISMA ruta que
         usa el runner vivo (scripts/run_periodic_rebalance.py)
      3. Reequilibrar (rebalancing_engine.rebalance, R-07)
         o PositionLedger (Fase 6, simulación sin broker)
      4. Simular NAV diario en periodo OOS hasta siguiente rebalanceo

Configuración: ``BacktestConfig`` = ``StrategyConfig`` (qué cartera se
construye) + parámetros de simulación (frecuencia, warmup, reemplazo, NAV
inicial, impacto, semilla, ledger).

Fase 6 (extensión):
  - freq='15D' (días hábiles) vía get_rebalance_dates
  - opt_method='hrp'
  - use_signal_stack → regime + signals + conviction
  - use_position_ledger → trades/shares/FIFO audit trail

Ruta clásica: el libro es un PortfolioState (acciones + caja) marcado cada
día al último precio válido <= t; los pesos derivan con los precios entre
rebalanceos. rebalance() opera sobre tenencias ∪ objetivo: un activo que sale
del universo activo se vende con costo, nunca se evapora.

Registra por fecha: NAV, pesos (+ caja en CASH_COL), trades, costos, turnover.
Validacion de no look-ahead en cada iteracion (R-05).
Sin RNG global: config.seed se pasa explicito a quien usa aleatoriedad
(random_replacement) para reproducibilidad (R-10).

Restricciones cubiertas: R-01, R-02, R-03, R-05, R-06, R-07, R-09, R-10.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.history_requirements import PERMISSIVE
from src.portfolio_state import PortfolioState
from src.position_ledger import MissingPriceError
from src.rolling_engine import get_rebalance_dates
from src.rebalancing_engine import rebalance
from src.strategy import (
    StrategyConfig,
    TargetResult,
    active_assets,
    build_targets,
    execution_weights,
    turnover_cost,
    uses_capm,
)

if TYPE_CHECKING:
    from src.position_ledger import LedgerSnapshot

logger = logging.getLogger("backtest_engine")

from src.settings import SEED, TRADING_DAYS
from src.settings import INITIAL_CASH as _INITIAL_NAV

_STRATEGY_FIELDS = frozenset(f.name for f in dataclasses.fields(StrategyConfig))

# Columna de caja en BacktestResult.weights: pesos + caja = 1 en cada fecha.
CASH_COL = "__CASH__"
TRADE_COLUMNS = ["date", "ticker", "order_value", "cost"]


# ===================================================================
# Data structures
# ===================================================================
@dataclass(init=False)
class BacktestConfig:
    """Configuration for a single backtest run: strategy + simulation.

    ``strategy`` decide los pesos objetivo (``src.strategy.build_targets``,
    misma ruta que el runner vivo); el resto son parámetros de simulación.

    Compatibilidad: los campos de estrategia siguen aceptándose planos
    (``BacktestConfig(opt_method="hrp", window=60)``) y se leen/escriben
    como atributos (``cfg.window``); van a ``cfg.strategy``. Con
    ``strategy=`` explícito, los campos planos lo sobreescriben.
    Fase 5 defaults preserved.
    """

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    rebalance_freq: str = "ME"         # 'ME'/'QE'/'M'/'Q'/'15D'/...
    warmup: int = 252                  # warmup before first rebalance
    # 'none','momentum','random','min_variance'. Classic path only: with
    # use_position_ledger=True anything but 'none' raises ValueError.
    replacement_rule: str = "none"
    initial_nav: float = _INITIAL_NAV
    impact_coef: float = 0.1
    seed: int = SEED
    # Fase 6: simulate in shares (FIFO). Executes the same book as the
    # classic path — first rebalance deploys cash to the full target, later
    # ones blend under turnover_limit (R-02) then sell down group_caps
    # excess — except exits, which close in full (execution_weights, shared
    # with the live runner). Costs are flat bps (no TWAP impact) and there
    # are no replacement rules.
    use_position_ledger: bool = False
    # True: un rebalanceo que lanza aborta el run (RebalanceError con fecha,
    # etapa, método y n_assets). False: se registra en
    # BacktestResult.failures / rebalance_log.status="failed" y se mantiene
    # el libro previo.
    strict: bool = True

    def __init__(self, strategy: Optional[StrategyConfig] = None, **kwargs: Any):
        for f in dataclasses.fields(self):
            if f.name != "strategy":
                object.__setattr__(self, f.name, kwargs.pop(f.name, f.default))
        base = strategy if strategy is not None else StrategyConfig()
        # Campo desconocido → TypeError de StrategyConfig.__init__
        object.__setattr__(
            self, "strategy", dataclasses.replace(base, **kwargs) if kwargs else base)

    def __getattr__(self, name: str) -> Any:
        # Solo se llama si el atributo no existe: campos planos de estrategia.
        if name in _STRATEGY_FIELDS:
            return getattr(self.strategy, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _STRATEGY_FIELDS:
            value, name = dataclasses.replace(self.strategy, **{name: value}), "strategy"
        object.__setattr__(self, name, value)


@dataclass
class DailyRecord:
    """Single-day record in the backtest."""

    date: pd.Timestamp
    nav: float
    daily_return: float
    weights: pd.Series
    is_rebalance: bool
    turnover: float
    cost: float


@dataclass
class BacktestResult:
    """Full backtest output."""

    config: BacktestConfig
    nav: pd.Series                  # Daily NAV indexed by date
    returns: pd.Series              # Daily gross returns
    weights: pd.DataFrame           # Daily weights (date × ticker + CASH_COL; rows sum 1)
    rebalance_log: pd.DataFrame     # One row per rebalance event
    daily_records: List[DailyRecord] = field(default_factory=list)
    # Fase 6 extensions (additive; empty/None when unused)
    ledger_snapshots: List["LedgerSnapshot"] = field(default_factory=list)
    regime_series: Optional[pd.Series] = None
    # strict=False only: one dict per rebalance that raised and kept the
    # previous weights — {date, stage, type, message}. Empty on a clean run.
    failures: List[dict] = field(default_factory=list)
    # One row per executed order (TRADE_COLUMNS): order_value in currency,
    # signed (< 0 = sell); cost in currency (ledger path: the rebalance cost
    # split pro rata to notional).
    trades: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS))


class RebalanceError(RuntimeError):
    """A rebalance raised in a strict backtest; chained to the original error.

    Attributes:
        date: Rebalance date.
        stage: "targets" (state + build_targets) or "execution" (rebalance()
            or ledger trades).
        method: opt_method of the run.
        n_assets: Assets being optimized at ``date`` (None if that set could
            not be determined, e.g. too few assets with history).
    """

    def __init__(self, date: pd.Timestamp, stage: str, method: str,
                 n_assets: Optional[int], cause: BaseException):
        self.date, self.stage, self.method, self.n_assets = (
            date, stage, method, n_assets)
        super().__init__(
            f"rebalance at {date.date()} failed in stage '{stage}' "
            f"(method={method}, n_assets={n_assets}): "
            f"{type(cause).__name__}: {cause}"
        )


def _n_assets(
    targets: Optional[TargetResult],
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    strategy: StrategyConfig,
    universe: List[str],
) -> Optional[int]:
    """Assets optimized at t, for failure context. build_targets may have
    raised after picking them (optimizer, caps): recompute the same set."""
    if targets is not None:
        return len(targets.active)
    try:
        return len(active_assets(prices, volumes, t, strategy, universe=universe))
    except Exception:  # noqa: BLE001 — context only; the real error is reported
        return None


def _log_row(
    t: pd.Timestamp,
    config: BacktestConfig,
    targets: Optional[TargetResult],
    n_assets: Optional[int],
    turnover: float,
    cost: float,
    status: str,
) -> dict:
    """rebalance_log row; ``targets`` is None when build_targets raised."""
    meta = targets.meta if targets is not None else {}
    return {
        "date": t,
        "opt_method": config.strategy.opt_method,
        "n_active": n_assets,
        "expected_sharpe": meta.get("expected_sharpe", 0.0),
        "turnover": turnover,
        "cost_fraction": cost,
        "rule": config.replacement_rule,
        "converged": meta.get("converged", False),
        "regime": targets.regime if targets is not None else None,
        "exits": ",".join(targets.exits["ticker"]) if targets is not None else "",
        "status": status,
    }


def _mark(book: PortfolioState, marks: pd.Series, t: pd.Timestamp) -> PortfolioState:
    """Libro clásico valorado a ``marks`` (último precio válido <= t)."""
    value = float((book.positions * marks.reindex(book.positions.index)).sum())
    return dataclasses.replace(book, equity=book.cash + value, as_of=t)


def _with_cash(w: pd.Series) -> pd.Series:
    """Pesos por ticker + ``CASH_COL`` = 1 − Σw (la caja es el residuo)."""
    out = w.astype(float).copy()
    out[CASH_COL] = 1.0 - float(out.sum())
    return out


def _unpriced(tickers: pd.Index, prices_t: pd.Series) -> List[str]:
    """Tickers sin precio finito > 0 en ``prices_t`` (no se pueden operar)."""
    px = pd.to_numeric(prices_t.reindex(tickers), errors="coerce").astype(float)
    return px.index[~(np.isfinite(px) & (px > 0))].tolist()


def _trade_rows(t: pd.Timestamp, tickers, order_value, cost) -> pd.DataFrame:
    return pd.DataFrame({"date": t, "ticker": list(tickers),
                         "order_value": np.asarray(order_value, dtype=float),
                         "cost": np.asarray(cost, dtype=float)},
                        columns=TRADE_COLUMNS)


# ===================================================================
# Core backtest loop
# ===================================================================
def run_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    universe: Optional[List[str]] = None,
) -> BacktestResult:
    """Execute a full rolling OOS backtest.

    For each rebalance date:
      1. Current book → PortfolioState (ledger, or the classic book: shares
         + cash, marked at the last valid price <= t).
      2. build_targets (src.strategy) → target weights: universe filter
         (R-03), training window (R-05 strict), mu/Sigma or HRP, signals,
         exits, caps. Same construction as the live runner.
      3. rebalance() or PositionLedger → executed weights with costs. The
         classic path trades holdings ∪ target: a holding that left the
         active universe is sold, with its cost. A holding without a valid
         price at t cannot be traded: the rebalance raises MissingPriceError
         (wrapped in RebalanceError when strict).
      4. Simulate daily NAV in OOS period using actual prices.

    Args:
        prices: Clean price DataFrame (date-indexed, full history).
        volumes: Volume DataFrame aligned with prices.
        config: BacktestConfig object. None → defaults.
        universe: Ticker universe. None → all price columns. Columns
            outside it are only used to price/exit held positions.

    Returns:
        BacktestResult with NAV, weights (fractions of NAV, cash in
        ``CASH_COL``; each row sums to 1), trades, and rebalance log (one row per
        rebalance date, ``status`` "ok", "fallback_ew" — the optimizer did
        not converge and returned equal weight (``fallback_used``) — or
        "failed"). With
        ``config.strict=False`` rebalances that raise are skipped (previous
        weights kept, turnover/cost 0) and listed in ``failures``.

    Raises:
        ValueError: No rebalance dates, group_caps without groups,
            capm_shrunk without its benchmark in the price panel, or a
            replacement_rule with use_position_ledger=True.
        RebalanceError: ``config.strict`` and a rebalance raised. Chained to
            the original error.
        RuntimeError: strict=False and every rebalance raised, so no weights
            were ever set and the NAV would sit flat at initial_nav. Chained
            to the last error.
    """
    if config is None:
        config = BacktestConfig()
    strategy = config.strategy

    if universe is None:
        universe = prices.columns.tolist()

    # Fail fast: caps need the ticker -> group map on every optimizer path;
    # inside the loop it would surface per rebalance (or be recorded and
    # skipped with strict=False) instead of as a config error.
    if strategy.group_caps and not strategy.groups:
        raise ValueError(
            "group_caps requires `groups` in BacktestConfig "
            "(ticker -> group map)."
        )

    # The ledger path trades straight to the (turnover/group-capped) target;
    # it has no replacement rules, so a rule would be silently ignored.
    if config.use_position_ledger and config.replacement_rule != "none":
        raise ValueError(
            f"replacement_rule='{config.replacement_rule}' is not supported "
            "with use_position_ledger=True (classic path only)."
        )

    # Same for capm_shrunk: without the benchmark in the panel every
    # rebalance would raise and the run would only end in a RuntimeError.
    if uses_capm(strategy) and strategy.benchmark_col not in prices.columns:
        raise ValueError(
            f"mu_method='capm_shrunk' needs benchmark '{strategy.benchmark_col}' "
            "in the price panel. Add it or use another mu_method."
        )

    # Rebalance dates (supports M/Q/ME/QE and '<N>D')
    rebal_dates = get_rebalance_dates(
        prices, freq=config.rebalance_freq, warmup=config.warmup
    )
    if not rebal_dates:
        raise ValueError(
            "No rebalance dates generated. "
            "Check warmup / price history length."
        )

    logger.info(
        "Starting backtest: method=%s, freq=%s, window=%d, "
        "signals=%s, ledger=%s, %d rebalance dates [%s → %s]",
        strategy.opt_method, config.rebalance_freq, strategy.window,
        strategy.use_signal_stack, config.use_position_ledger,
        len(rebal_dates),
        rebal_dates[0].date(), rebal_dates[-1].date(),
    )

    nav = config.initial_nav
    # Weights reported for t (tickers + CASH_COL).
    w_current: Optional[pd.Series] = None
    # ticker -> precio de la primera compra (ruta clásica con salidas; el
    # ledger usa su avg_entry)
    entries: Dict[str, float] = {}
    # Classic path book: shares + cash, all cash until the first rebalance.
    # Marked daily at the last valid price <= t.
    book = PortfolioState.from_weights(
        None, prices.iloc[0], equity=config.initial_nav, as_of=rebal_dates[0])
    marks = prices.where(np.isfinite(prices) & (prices > 0)).ffill()

    nav_records: Dict[pd.Timestamp, float] = {}
    weight_records: Dict[pd.Timestamp, pd.Series] = {}
    rebalance_log_rows: List[dict] = []
    trade_frames: List[pd.DataFrame] = []
    failures: List[dict] = []
    last_rebalance_exc: Optional[Exception] = None
    daily_records: List[DailyRecord] = []
    ledger_snapshots: List = []
    regime_records: Dict[pd.Timestamp, str] = {}

    ledger = None
    if config.use_position_ledger:
        from src.position_ledger import PositionLedger
        ledger = PositionLedger(initial_cash=config.initial_nav)

    rebal_set = set(rebal_dates)
    all_dates = prices.index

    # Iterate over all dates after warmup
    first_rebal = rebal_dates[0]
    oos_dates = all_dates[all_dates >= first_rebal]

    for i, t in enumerate(oos_dates):
        is_rebal = t in rebal_set
        turnover_today = 0.0
        cost_today = 0.0
        if ledger is None:
            # The t_prev→t move accrues to the book held overnight: trades
            # run at t's close (the ledger path fills at prices.loc[t]), so
            # the classic book is marked before it rebalances.
            marks_t = marks.loc[t]
            book = _mark(book, marks_t, t)
            gross_ret = book.equity / nav - 1.0 if nav > 1e-10 else 0.0

        # --- Rebalance step ---
        if is_rebal:
            stage, targets = "targets", None
            try:
                state = (book if ledger is None else
                         PortfolioState.from_ledger(ledger, prices.loc[t], as_of=t))
                targets = build_targets(
                    state=state, prices=prices, volumes=volumes, t=t,
                    cfg=strategy, universe=universe, history_mode=PERMISSIVE,
                )
                w_target = targets.weights
                w_prev = targets.w_prev
                active = list(targets.eligible)
                active_clean = list(targets.active)
                regime_state = targets.regime
                if regime_state is not None:
                    regime_records[t] = regime_state

                # --- Execution path ---
                stage = "execution"
                if config.use_position_ledger and ledger is not None:
                    prices_t = prices.loc[t]
                    nav_pre = ledger.nav(prices_t)
                    # Initial cash deployment: no prior book to turn over
                    w_prev_exec = (pd.Series(0.0, index=active_clean)
                                   if w_prev is None else w_prev)
                    # R-02 + topes por grupo: mismo libro ejecutado que el
                    # runner vivo (src.strategy.execution_weights).
                    w_exec = execution_weights(targets, state, strategy)

                    # Approximate turnover cost before applying trades
                    _, cost_frac = turnover_cost(w_prev_exec, w_exec, strategy)
                    cost_abs = cost_frac * max(nav_pre, 1e-6)
                    planned_turnover = float(
                        (w_exec.reindex(w_prev_exec.index).fillna(0.0)
                         - w_prev_exec).abs().sum()
                    )
                    # Also count new names
                    extra = w_exec.index.difference(w_prev_exec.index)
                    planned_turnover += float(
                        w_exec.reindex(extra).fillna(0.0).abs().sum()
                    )

                    trades = ledger.trades_from_target_weights(
                        w_exec, prices_t, nav=nav_pre, costs=cost_abs,
                    )
                    ledger.apply_trades(
                        trades, prices_t, costs=cost_abs, t=t,
                    )
                    # Report only once the batch has executed: apply_trades is
                    # atomic, so if it raised nothing traded and turnover/cost
                    # stay 0 (same as the classic path on failure).
                    turnover_today = planned_turnover
                    cost_today = cost_frac
                    snap = ledger.snapshot(t, prices_t)
                    ledger_snapshots.append(snap)
                    nav = snap.nav
                    if not trades.empty:
                        notional = (trades["shares"].astype(float).to_numpy()
                                    * prices_t.reindex(trades["ticker"]).to_numpy(float))
                        gross = float(np.abs(notional).sum())
                        trade_frames.append(_trade_rows(
                            t, trades["ticker"], notional,
                            cost_abs * np.abs(notional) / gross if gross > 0
                            else np.zeros_like(notional),
                        ))
                else:
                    # Classic path: trade the whole book (holdings ∪ target);
                    # a holding that left the target is sold, with its cost.
                    prices_t = prices.loc[t]
                    w_held = book.weights(marks_t)
                    unpriced = _unpriced(w_held.index, prices_t)
                    if unpriced:
                        raise MissingPriceError(
                            unpriced, f"Cannot sell or resize them at {t.date()}.")
                    equity_pre = book.equity
                    rebal_result = rebalance(
                        t=t,
                        w_current=w_held,
                        w_target=w_target,
                        prices=prices,
                        volumes=volumes,
                        rule=config.replacement_rule,
                        universe=universe,
                        active_tickers=active,
                        portfolio_value=equity_pre,
                        turnover_limit=strategy.turnover_limit,
                        is_rebalance_date=True,
                        commission_bps=strategy.commission_bps,
                        spread_bps=strategy.spread_bps,
                        impact_coef=config.impact_coef,
                        seed=config.seed,
                        groups=strategy.groups,
                        group_caps=strategy.group_caps,
                        max_weight=strategy.max_weight,
                        max_weight_by_ticker=strategy.max_weight_by_ticker(),
                    )
                    w_new = rebal_result.w_new
                    # Entradas para stop-loss: precio de la compra que abre
                    # la posición (solo si exits on)
                    if strategy.use_exit_rules:
                        for tk, w in w_new.items():
                            if w > 1e-9 and (tk not in entries
                                             or float(w_held.get(tk, 0.0)) < 1e-9):
                                px = float(prices_t.get(tk, float("nan")))
                                if px and px == px:
                                    entries[tk] = px
                    book = PortfolioState.from_weights(
                        w_new, prices_t, equity=equity_pre - rebal_result.cost,
                        avg_entry=entries, as_of=t,
                    )
                    turnover_today = rebal_result.actual_turnover
                    cost_today = (
                        rebal_result.cost / equity_pre
                        if equity_pre > 1e-6 else 0.0
                    )
                    orders = rebal_result.trades
                    if not orders.empty:
                        trade_frames.append(_trade_rows(
                            t, orders["ticker"], orders["order_value"],
                            orders["total_cost"],
                        ))

            except Exception as exc:  # noqa: BLE001
                n_assets = _n_assets(targets, prices, volumes, t, strategy, universe)
                if config.strict:
                    raise RebalanceError(
                        t, stage, strategy.opt_method, n_assets, exc) from exc
                logger.warning(
                    "Rebalance at %s failed in stage '%s': %s. "
                    "Keeping previous weights.", t.date(), stage, exc,
                )
                failures.append({
                    "date": t,
                    "stage": stage,
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                rebalance_log_rows.append(_log_row(
                    t, config, targets, n_assets, 0.0, 0.0, "failed"))
                last_rebalance_exc = exc
            else:
                # OptimizationResult.fallback_used: the solver did not
                # converge and handed back equal weight. It trades, but it is
                # not the strategy (build_targets already logged a warning).
                status = ("fallback_ew" if targets.meta.get("fallback_used", False)
                          else "ok")
                rebalance_log_rows.append(_log_row(
                    t, config, targets, len(active_clean),
                    turnover_today, cost_today, status))

        # --- NAV update ---
        if ledger is not None:
            prices_t = prices.loc[t] if t in prices.index else None
            if prices_t is not None:
                nav_new = ledger.nav(prices_t)
                if i > 0 and nav > 1e-10:
                    # Cost already deducted on rebalance day inside ledger
                    daily_ret = (nav_new / nav) - 1.0
                    if is_rebal and cost_today > 0 and abs(daily_ret + cost_today) > 1e-15:
                        # cost already in nav_new; report gross-ish daily_ret
                        pass
                else:
                    daily_ret = 0.0
                nav = nav_new
                w_current = _with_cash(ledger.weights_from_positions(prices_t))
            else:
                daily_ret = 0.0
        else:
            # Classic book: marked at t, net of today's trading cost. NAV
            # compounds exactly, (1 + gross)(1 − cost); daily_ret keeps the
            # linear convention gross − cost.
            daily_ret = gross_ret - cost_today
            nav = book.equity
            w_current = _with_cash(book.weights(marks_t))

        nav_records[t] = nav
        weight_records[t] = (
            w_current.copy()
            if w_current is not None
            else pd.Series({CASH_COL: 1.0})
        )
        daily_records.append(DailyRecord(
            date=t,
            nav=nav,
            daily_return=daily_ret,
            weights=weight_records[t],
            is_rebalance=is_rebal,
            turnover=turnover_today,
            cost=cost_today,
        ))

    # Never hand back a flat NAV built from zero successful rebalances.
    if failures and len(failures) == len(rebalance_log_rows):
        last = failures[-1]
        raise RuntimeError(
            f"All {len(failures)} rebalance(s) failed; no weights "
            f"were ever set, so NAV would stay flat at initial_nav. Last "
            f"failure at {last['date'].date()} ({last['stage']}): "
            f"{last['type']}: {last['message']}"
        ) from last_rebalance_exc

    nav_series = pd.Series(nav_records, name="nav")
    returns_series = nav_series.ffill().pct_change().fillna(0.0)
    returns_series.name = "returns"

    weights_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weights_df = weights_df[
        [c for c in weights_df.columns if c != CASH_COL] + [CASH_COL]]
    weights_df.index.name = "date"

    frames = [f for f in trade_frames if not f.empty]
    trades_df = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=TRADE_COLUMNS))

    rebal_df = pd.DataFrame(rebalance_log_rows)
    if not rebal_df.empty:
        rebal_df = rebal_df.set_index("date")

    regime_series = (
        pd.Series(regime_records, dtype=object) if regime_records else None
    )

    statuses = rebal_df["status"].value_counts().to_dict() if not rebal_df.empty else {}
    logger.info(
        "Backtest complete: %d days, final NAV=%.2f "
        "(%.1f%% total return), %d rebalances %s",
        len(nav_series), nav,
        (nav / config.initial_nav - 1) * 100,
        len(rebalance_log_rows), statuses,
    )
    return BacktestResult(
        config=config,
        nav=nav_series,
        returns=returns_series,
        weights=weights_df,
        rebalance_log=rebal_df,
        daily_records=daily_records,
        ledger_snapshots=ledger_snapshots,
        regime_series=regime_series,
        failures=failures,
        trades=trades_df,
    )


# ===================================================================
# Convenience: run multiple configs for comparison
# ===================================================================
def run_comparison(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    configs: Dict[str, BacktestConfig],
    universe: Optional[List[str]] = None,
) -> Tuple[Dict[str, BacktestResult], Dict[str, Exception]]:
    """Run multiple backtest configurations for comparison.

    A config that raises does not stop the others, and is not dropped
    silently either: its exception comes back under its label.

    Args:
        prices: Clean price DataFrame.
        volumes: Volume DataFrame.
        configs: Dict {label → BacktestConfig}.
        universe: Ticker universe for every run (None → all price columns).

    Returns:
        (results, errors): {label → BacktestResult} for the runs that
        finished, {label → exception} for the ones that raised.
    """
    results: Dict[str, BacktestResult] = {}
    errors: Dict[str, Exception] = {}
    for label, cfg in configs.items():
        logger.info("Running backtest: %s", label)
        try:
            results[label] = run_backtest(prices, volumes, cfg, universe=universe)
        except Exception as exc:  # noqa: BLE001
            logger.error("Backtest '%s' failed: %s", label, exc)
            errors[label] = exc
    return results, errors
