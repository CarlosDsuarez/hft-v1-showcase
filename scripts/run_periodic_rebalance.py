#!/usr/bin/env python3
"""
run_periodic_rebalance.py — Rebalanceo periódico cada 15 días hábiles.

Qué es esto
-----------
NO es un cron automático en la nube. Es un **comando** que tú (o un
scheduler tipo cron) ejecutas cuando toca rebalancear. Usa el mismo motor
de construcción que el backtest (src/strategy.build_targets):

  universo → precios (historia requerida) → build_targets (filtro, régimen/
  señales/HRP, salidas, topes) → cartera

Estrategia: config/live.toml (versionado; ``--config`` para otro archivo)
cargado a ``StrategyConfig``. Cada fila de rebalance_history.csv registra
``config_hash`` (sha256 de la config efectiva) y ``config_source``.

Historia: cada componente declara ``required_history`` (filas hábiles);
el runner descarga max(...) + margen. ``--history-mode strict`` (default)
aborta con exit 2 si falta historia; ``permissive`` excluye los factores sin
historia y lo registra en rebalance_history.csv (``factors_used``).

Una sola fuente de verdad por corrida (`--mode`, ver src/portfolio_state.py):
  - sim (default): la cartera es el PositionLedger local (ledger.json) y los
    trades se simulan sobre él. Nada toca la red.
  - broker: la cartera es la cuenta Alpaca paper (posiciones, avg_entry,
    cash, equity), leída ANTES de evaluar salidas y de dimensionar.
    ledger.json pasa a ser espejo del broker (drift local vs broker al log;
    se reescribe al comprometer la corrida).

Fases (run_once)
----------------
  load_state → fetch_data → decide_schedule → cartera (ledger | broker) →
  build_targets (pesos + salidas) → execution_weights (tope de turnover
  R-02, como el backtest) → build_sim_plan | plan_broker →
  [execute_broker] → commit_state

Todo hasta el plan es lectura/cálculo puro: **nada se persiste antes de un
execute exitoso**. commit_state publica la corrida en una sola transacción.

Estado persistente en ``settings.state_dir`` (src/settings.py):
``<repo>/data/live_state/`` o ``$HFT_HOME/data/live_state/`` si HFT_HOME
está definido. Nunca relativo al cwd: se puede lanzar desde cualquier carpeta.
  - ledger.json (PositionLedger.to_dict: schema_version, lots, cash,
    realized_pnl, txns); en broker, espejo sin txns
  - ledger_local_backup_<fecha>.json (broker: ledger local previo al espejo
    cuando no coincidía con el broker)
  - transactions.csv (audit trail FIFO; filas append con run_id)
  - last_rebalance.txt
  - latest_snapshot.csv
  - snapshots.csv (append por corrida ok: cartera TRAS la corrida, una fila
    por ticker + caja en ``__CASH__``: shares, price, value, weight; sim =
    ledger tras los trades, broker = cartera leída + fills conocidos al
    comprometer). Fuente de `src.live_report.LiveResult`.
  - rebalance_history.csv (run_id, status ok/partial, mode sim/broker,
    history_mode/factors_used, config_hash/config_source, procedencia
    git_sha/git_dirty/data_source/config_schema/state_schema/ledger_schema,
    n_orders/n_submitted/n_rejected, error) /
    exit_log.csv (con run_id)
  - broker_orders.csv (--submit: estado por orden de cada corrida, con run_id)
  - state_manifest.json (run_id, as_of, sha256 por archivo)
  - logs/ (fuera del manifiesto): runner.log rotativo y alerts.jsonl

Trazabilidad (ARQ-21): cada corrida tiene un run_id (``YYYYMMDDTHHMMSSZ-<8
hex>``) desde su inicio; todo log lo lleva (``[run=...]``, consola y
archivo) y es el mismo de history/transactions/plans. Las corridas que
toman el lock loguean además a ``state_dir/logs/runner.log``; --check-only y
--dry-run solo a consola (no escriben en state_dir). Resultado de cada
corrida → `src.notifier` (consola + logs/alerts.jsonl; email opcional con
ALERT_EMAIL/ALERT_SMTP_*). Reporte del modo vivo: ``python -m
src.live_report``.

Escritura transaccional (src/live_state.StateStore): lock exclusivo sobre
``state_dir/.lock`` (si otra corrida lo tiene → exit 4, sin tocar nada);
todo se escribe en ``.staging/<run_id>/`` y se publica en un solo commit.
Un fallo a mitad deja el estado anterior íntegro. Un rerun del mismo as_of
no escribe; con --force se registra como corrida nueva (forced=True,
supersedes=<run_id previo>).

Uso
---
  # Con venv activo (el cwd no importa: rutas ancladas en src/settings.py):
  python -m scripts.run_periodic_rebalance

  # Forzar aunque no hayan pasado 15 días hábiles:
  python -m scripts.run_periodic_rebalance --force

  # Solo informar si toca rebalancear (exit 0 = toca, 1 = aún no; no escribe):
  python -m scripts.run_periodic_rebalance --check-only

  # Dry-run real: todas las fases salvo execute y commit. No toma el lock ni
  # escribe nada en live_state; el plan queda en results/plans/<run_id>/
  # (summary.json, targets.csv, exits.csv, sim_trades.csv | broker_plan_*.csv).
  python -m scripts.run_periodic_rebalance --dry-run [--mode broker]

Broker paper (sleeve USA, opt-in):
  python -m scripts.run_periodic_rebalance --mode broker --dry-run
      # plan vs cuenta paper (lee cuenta/posiciones; no envía ni persiste).
      # --mode broker sin --submit es siempre dry-run (--broker = alias
      # obsoleto de --mode broker).
  python -m scripts.run_periodic_rebalance --mode broker --submit
      # envía órdenes market/day en paper y, si salen, compromete la corrida.
      # --submit exige --mode broker (error de argparse) y excluye --dry-run.
      # Con mercado cerrado o cuenta bloqueada aborta ANTES de enviar
      # (exit 3); --allow-closed permite enviar con mercado cerrado (nunca con
      # cuenta bloqueada).
      # client_order_id = hft-<fecha>-<ticker>-<side>: relanzar el mismo día no
      # duplica. broker_journal_<fecha>.jsonl registra cada orden al enviarla;
      # broker_plan_<fecha>.csv se escribe siempre (status + fills).
      # Guardas pre-trade: --max-order-notional / --max-total-notional /
      # --max-orders; kill-switch con data/live_state/KILL o HFT_KILL=1.
      # Si bloquean: 0 órdenes, nada comprometido, exit 3.
      # Envío parcial (falla tras >=1 orden enviada): se comprometen las
      # órdenes enviadas (broker_orders.csv) + fila status=partial, pero NO
      # last_rebalance ni el ledger → la próxima corrida reintenta sola.

Códigos de salida
-----------------
  0  OK: rebalanceo comprometido (--check-only: toca; --dry-run: plan listo)
  1  No toca: < every_n días hábiles o as_of ya comprometido (nada escrito)
  2  Datos insuficientes: falta la historia requerida (--history-mode) o
     hay < 3 activos (nada escrito)
  3  Broker: lectura/envío falló, guardas/KILL bloquearon, o envío parcial
  4  Lock: otra corrida tiene state_dir (nada leído ni escrito)
  5  Error inesperado (solo desde la CLI; nada comprometido)
  6  Posicion abierta sin precio valido en t (o ausente de la descarga):
     se aborta antes de planificar; nada escrito ni enviado
  7  Datos de mercado inutilizables segun la politica de datos: ticker del
     universo sin datos (--on-missing error), barra de hoy con el mercado
     abierto (--on-partial-bar abort), ultima barra anterior a la ultima
     sesion NYSE cerrada (--on-stale error) o ninguna fuente respondio;
     nada escrito ni enviado

Datos de mercado: una sola descarga OHLCV por corrida via el proveedor unico
(src/market_data.py), cadena ``--data-sources`` (default yahoo,csv) y cache
parquet opcional por simbolo (``--data-cache``, en data/cache/bars/). La barra
de hoy antes del cierre (16:00 ET) se descarta por defecto.

Criterios de salida (peso -> 0 con motivo en exit_log.csv): los define
config/live.toml (sección de salidas). Los flags --stop-loss, --take-profit,
--trailing-stop, --trailing-window, --take-profit-dollars y
--stop-loss-dollars sobreescriben ese campo para la corrida (0 = off) y
activan las salidas; el hash registrado es el de la config efectiva.

Cron ejemplo (cada día a las 16:00; el script decide si ya pasaron 15D):
  0 16 * * 1-5 cd /path/to/HFT-V1 && .venv/bin/python -m scripts.run_periodic_rebalance
  (el ``cd`` solo hace falta para que ``-m scripts...`` resuelva el paquete)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.backtest_engine import CASH_COL
from src.data_client import load_universe, clean_prices
from src.history_requirements import (
    DEFAULT_MARGIN_ROWS, HISTORY_MODES, STRICT,
    InsufficientHistoryError, business_rows_to_calendar_days,
    missing_requirements,
)
from src.fx import MissingFxError
from src.instruments import load_instruments
from src.market_data import (
    AVAILABLE_SOURCES, DEFAULT_SOURCES, LIVE_POLICY, MISSING_POLICIES,
    PARTIAL_BAR_POLICIES, STALE_POLICIES, DataPolicy, InstrumentProvider,
    MarketDataError, MarketDataProvider, MissingSymbolsError, StaleDataError,
    build_provider, close_in_base, load_bars,
)
from src.live_state import (
    MANIFEST_NAME, StateLockedError, StateStore, migrate_ledger_payload, new_run_id,
)
from src.notifier import Alert, Notifier, build_notifier
from src.portfolio_state import PortfolioState, position_drift
from src.position_ledger import LedgerSnapshot, MissingPriceError, PositionLedger
from src.provenance import CodeVersion, code_version, run_provenance
from src.brokers.pre_trade import (
    DEFAULT_MAX_ORDER_NOTIONAL, DEFAULT_MAX_ORDERS, DEFAULT_MAX_TOTAL_NOTIONAL,
    PreTradeBlocked, PreTradeGuard, kill_switch_reason,
)
from src.settings import (
    INITIAL_CASH, SEED, Settings, load_settings, log_to_file, run_logging,
    setup_logging,
)
from src.strategy import (
    DEFAULT_CONFIG_PATH, InsufficientDataError, StrategyConfig,
    build_targets, execution_weights, history_requirements,
    load_strategy_config, turnover_cost,
)

logger = logging.getLogger("periodic_rebalance")

REBALANCE_EVERY_N = 15          # trading days
POLL_TIMEOUT_S = 30.0       # espera de fills tras enviar ordenes paper
PLANS_DIRNAME = "plans"     # --dry-run: results/plans/<run_id>/
MODES = ("sim", "broker")   # fuente de verdad de la cartera
CASH_DRIFT_TOLERANCE = 0.01  # USD: diferencia de caja local vs broker que se loguea

# Códigos de salida (ver docstring del módulo)
EXIT_OK = 0
EXIT_NOT_DUE = 1
EXIT_NO_DATA = 2
EXIT_BROKER = 3
# Otro proceso tiene el lock de state_dir (cron + corrida manual solapadas)
EXIT_LOCKED = 4
EXIT_UNEXPECTED = 5  # solo main(): excepción no prevista
EXIT_MISSING_PRICE = 6  # posicion abierta sin precio valido: nada escrito
EXIT_DATA = 7  # datos de mercado inutilizables (politica de datos): nada escrito


class _FromConfig:
    """Centinela: el campo de salida sale de la config (sin override)."""

    def __repr__(self) -> str:
        return "FROM_CONFIG"


FROM_CONFIG: Any = _FromConfig()
# Flags de salida que sobreescriben el campo homónimo de StrategyConfig.
EXIT_OVERRIDE_FIELDS = ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
                        "trailing_window", "take_profit_dollars", "stop_loss_dollars")


# ===================================================================
# State I/O
# ===================================================================
def _ensure_state_dir(settings: Settings) -> Path:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return settings.state_dir


def save_ledger(ledger: PositionLedger, path: Path) -> None:
    """Persist the full ledger state (PositionLedger.to_dict) to JSON."""
    path.write_text(json.dumps(ledger.to_dict(), indent=2))


def load_ledger(path: Path, initial_cash: float = INITIAL_CASH) -> PositionLedger:
    """Restore PositionLedger from JSON, or create a fresh one.

    El payload pasa por ``migrate_ledger_payload`` (migración explícita del
    ledger legado sin schema_version); el próximo commit lo guarda migrado.
    """
    if not path.exists():
        return PositionLedger(initial_cash=initial_cash)
    return PositionLedger.from_dict(migrate_ledger_payload(json.loads(path.read_text())))


def read_last_rebalance(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return pd.Timestamp(text)


def trading_days_between(
    prices: pd.DataFrame,
    a: pd.Timestamp,
    b: pd.Timestamp,
) -> int:
    """Count trading days in prices index strictly after a, up to and including b."""
    idx = prices.index
    mask = (idx > a) & (idx <= b)
    return int(mask.sum())


def should_rebalance(
    prices: pd.DataFrame,
    last: Optional[pd.Timestamp],
    every_n: int = REBALANCE_EVERY_N,
) -> bool:
    """True if never rebalanced, or >= every_n trading days since last."""
    if last is None:
        return True
    t = prices.index[-1]
    return trading_days_between(prices, last, t) >= every_n


def utcnow() -> datetime:
    """Reloj UTC de la corrida (frescura de datos y ventana de descarga)."""
    return datetime.now(timezone.utc)


def make_market_data_provider(
    settings: Settings,
    sources=DEFAULT_SOURCES,
    cache: bool = False,
) -> MarketDataProvider:
    """Cadena de fuentes (src/market_data.build_provider) detrás del maestro de
    instrumentos: el runner pide ids; la cadena (y su cache en settings) ve
    símbolos Yahoo. Precios en moneda local + FX (ver `fetch_data`)."""
    chain = build_provider(
        sources, settings=settings,
        cache_dir=settings.bars_cache_dir if cache else None, clock=utcnow,
    )
    return InstrumentProvider(chain, load_instruments(),
                              base_currency=settings.base_currency)


def default_start(required_rows: int, margin_rows: int = DEFAULT_MARGIN_ROWS) -> str:
    """Start default: cubre ``required_rows`` + margen filas hábiles hasta hoy."""
    end = utcnow() + timedelta(days=1)
    days = business_rows_to_calendar_days(required_rows + margin_rows)
    return (end - timedelta(days=days)).strftime("%Y-%m-%d")


# ===================================================================
# Estrategia (config/live.toml → StrategyConfig)
# ===================================================================
def apply_exit_overrides(cfg: StrategyConfig, **overrides: Any) -> StrategyConfig:
    """Flags de salida de la CLI sobre la config (FROM_CONFIG = sin cambio).

    Un override con valor (no None) activa ``use_exit_rules``; None apaga esa
    regla. Devuelve la config efectiva (la que se hashea en history).
    """
    unknown = set(overrides) - set(EXIT_OVERRIDE_FIELDS)
    if unknown:
        raise TypeError(f"overrides de salida desconocidos: {sorted(unknown)}")
    changes = {k: v for k, v in overrides.items() if v is not FROM_CONFIG}
    if not changes:
        return cfg
    if any(v is not None for k, v in changes.items() if k != "trailing_window"):
        changes["use_exit_rules"] = True
    logger.info("Overrides de salida sobre la config: %s", changes)
    return replace(cfg, **changes)


# ===================================================================
# Broker paper (sleeve USA, opt-in)
# ===================================================================
def make_broker_client(state_dir: Path):
    """Unico punto del runner que crea el cliente (red) de Alpaca paper."""
    from src.brokers.alpaca_adapter import AlpacaClient, load_broker_config
    return AlpacaClient(load_broker_config(), kill_dir=state_dir)


def _log_drift(local: PortfolioState, broker: PortfolioState) -> bool:
    """Loguea la diferencia ledger local vs broker (acciones y caja), si existe.

    Returns:
        True si ambos libros coinciden (sin drift).
    """
    drift = position_drift(local, broker)
    cash_diff = broker.cash - local.cash
    if drift.empty and abs(cash_diff) < CASH_DRIFT_TOLERANCE:
        logger.info("Ledger local coincide con el broker.")
        return True
    detail = ", ".join(f"{r.ticker} {r.local_shares:g}→{r.broker_shares:g}"
                       for r in drift.itertuples()) or "-"
    logger.warning(
        "Drift ledger local vs broker: %d ticker(s) [%s]; cash local=%.2f "
        "broker=%.2f (%+.2f). Se usa el broker.",
        len(drift), detail, local.cash, broker.cash, cash_diff,
    )
    return False


def mirror_ledger_from_broker(
    ledger: PositionLedger,
    broker: PortfolioState,
    prices_t: pd.Series,
    t: pd.Timestamp,
    ledger_exists: bool,
) -> tuple[PositionLedger, Optional[PositionLedger]]:
    """Ledger local → espejo del broker (fuente de verdad en modo broker).

    Solo calcula y loguea el drift local vs broker: no escribe nada (el
    espejo y el respaldo se publican en commit_state).

    Returns:
        (espejo del broker, ledger local a respaldar en
        ``ledger_local_backup_<fecha>.json`` o None). Se respalda si
        ledger.json existe y no coincide con el broker (libro sim, legado sin
        txns o espejo previo a fills; o no valorable).
    """
    in_sync = False
    try:
        local = PortfolioState.from_ledger(ledger, prices_t, as_of=t)
    except ValueError as exc:
        logger.warning("Ledger local no valorable (%s): drift no calculado.", exc)
    else:
        in_sync = _log_drift(local, broker)
    backup = ledger if (not in_sync and ledger_exists) else None
    return broker.to_ledger(), backup


def build_broker_plan(
    w_target: pd.Series,
    broker_qty: pd.Series,
    prices_t: pd.Series,
    equity: float,
    min_notional: float = 1.0,
) -> pd.DataFrame:
    """Pesos objetivo vs posicion real → plan de ordenes (puro, sin red)."""
    from src.brokers.alpaca_adapter import reconcile_trades
    return reconcile_trades(
        target_weights=w_target.astype(float),
        broker_qty=broker_qty.astype(float),
        prices=prices_t.astype(float),
        portfolio_value=float(equity),
        min_notional=min_notional,
    )


def broker_run_id(t: pd.Timestamp) -> str:
    """run_id determinista por fecha de rebalanceo: relanzar el mismo dia
    reusa los client_order_id y Alpaca rechaza el duplicado (no doble orden)."""
    return f"hft-{pd.Timestamp(t):%Y%m%d}"


def _append_journal(path: Path, record: Dict) -> None:
    """Append-only JSONL con fsync: sobrevive a un crash tras cada orden."""
    line = json.dumps({**record,
                       "logged_at_utc": datetime.now(timezone.utc).isoformat()},
                      default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _plan_with_results(
    plan: pd.DataFrame,
    results: List[Dict],
    fills: Dict[str, Dict],
    default_status: str = "not_sent",
    blocked_reason: Optional[str] = None,
) -> pd.DataFrame:
    """Plan + estado por orden; filas sin resultado → `default_status`."""
    out = plan.copy()
    by_ticker = {r["ticker"]: r for r in results}
    res = [by_ticker.get(tk, {}) for tk in out["ticker"]]
    out["status"] = [r.get("status", default_status) for r in res]
    out["client_order_id"] = [r.get("client_order_id") for r in res]
    out["order_id"] = [r.get("id") for r in res]
    out["broker_status"] = [r.get("broker_status") for r in res]
    out["error"] = [r.get("error") for r in res]
    fill = [fills.get(r.get("id"), {}) for r in res]
    out["filled_qty"] = [f.get("filled_qty") for f in fill]
    out["filled_avg_price"] = [f.get("filled_avg_price") for f in fill]
    out["final_status"] = [f.get("final_status") for f in fill]
    out["blocked_reason"] = blocked_reason
    return out


@dataclass(frozen=True)
class BrokerAccount:
    """Lectura de la cuenta: check (verificación) + cartera del broker."""

    summary: Dict[str, Any]
    portfolio: PortfolioState

    @property
    def market_closed(self) -> bool:
        return self.summary.get("market_open") is False


def read_broker(
    client,
    summary: Optional[Dict[str, Any]] = None,
    portfolio: Optional[PortfolioState] = None,
) -> BrokerAccount:
    """Lee la cuenta del broker (solo lectura): check + PortfolioState.

    Args:
        summary: Resultado de `client.check()` ya hecho (default: se llama aqui;
            verifica BROKER_ACCOUNT_ID antes de leer la cartera).
        portfolio: Cartera del broker ya leida (default
            `PortfolioState.from_broker(client)` tras el check).

    Raises:
        ValueError: `portfolio` no viene del broker.
        RuntimeError: equity del broker <= 0 (o el error del cliente).
    """
    if summary is None:
        summary = client.check()
    if portfolio is None:
        portfolio = PortfolioState.from_broker(client)
    if portfolio.source != "broker":
        raise ValueError(f"El plan broker exige cartera del broker, no "
                         f"source={portfolio.source!r}.")
    if float(portfolio.equity) <= 0:
        raise RuntimeError(f"Equity inválido del broker: {portfolio.equity}")
    if summary.get("market_open") is False:
        logger.warning("Mercado cerrado: las ordenes day quedarían pendientes.")
    return BrokerAccount(summary=summary, portfolio=portfolio)


@dataclass(frozen=True)
class BrokerPlan:
    """Cuenta leída + órdenes para llevar el broker al objetivo."""

    account: BrokerAccount
    orders: pd.DataFrame      # [ticker, side, qty, notional]

    @property
    def summary(self) -> Dict[str, Any]:
        return self.account.summary

    @property
    def equity(self) -> float:
        return float(self.account.portfolio.equity)

    @property
    def market_closed(self) -> bool:
        return self.account.market_closed


def plan_broker(
    account: BrokerAccount,
    w_target: pd.Series,
    prices_t: pd.Series,
    min_notional: float = 1.0,
) -> BrokerPlan:
    """Fase 5b — órdenes dimensionadas SOLO con la cartera del broker (puro)."""
    portfolio = account.portfolio
    orders = build_broker_plan(w_target, portfolio.positions, prices_t,
                               portfolio.equity, min_notional=min_notional)
    return BrokerPlan(account=account, orders=orders)


@dataclass(frozen=True)
class BrokerExecution:
    """Resultado de execute_broker; `orders` es el broker_plan CSV.

    status:
      dry_run  sin envío (submit=False)
      ok       sin error y (>=1 orden enviada o plan vacío); los rechazos
               por orden se cuentan pero no invalidan la corrida
      partial  error tras >=1 orden enviada
      failed   nada llegó al broker (bloqueo, error antes del 1er envío, o
               todas las órdenes rechazadas o no ruteables; un 422 por
               client_order_id ya existente cuenta como enviada, no como rechazo)

    Órdenes "not_routable" (instrumento sin "alpaca" en tradable_brokers del
    maestro) nunca se envían, tampoco en dry-run.
    """

    orders: pd.DataFrame
    results: List[Dict]
    mode: str                              # "dry_run" | "submitted"
    error: Optional[Exception] = None
    blocked_reasons: tuple = ()

    @property
    def n_submitted(self) -> int:
        return int((self.orders["status"] == "submitted").sum())

    @property
    def n_rejected(self) -> int:
        return int((self.orders["status"] == "rejected").sum())

    @property
    def n_not_routable(self) -> int:
        return int((self.orders["status"] == "not_routable").sum())

    @property
    def status(self) -> str:
        if self.mode == "dry_run":
            return "dry_run"
        if self.error is None and (self.orders.empty or self.n_submitted > 0):
            return "ok"
        return "partial" if self.n_submitted > 0 else "failed"


def submit_guard(guard: Optional[PreTradeGuard], allow_closed: bool) -> PreTradeGuard:
    """Guard de --submit: cuenta bloqueada siempre bloquea; mercado cerrado
    bloquea salvo `allow_closed`. Conserva los topes numéricos de `guard`."""
    return replace(guard or PreTradeGuard(), block_if_account_blocked=True,
                   block_if_market_closed=not allow_closed)


def execute_broker(
    client,
    bplan: BrokerPlan,
    t: pd.Timestamp,
    out_dir: Path,
    submit: bool = False,
    guard: Optional[PreTradeGuard] = None,
    poll_timeout: float = POLL_TIMEOUT_S,
    order_prefix: Optional[str] = None,
    kill_dir: Optional[Path] = None,
) -> BrokerExecution:
    """Fase 6 — guardas pre-trade y envío (submit) o solo anotación (dry-run).

    Auditoria (submit=True), en `out_dir`:
      - broker_journal_<fecha>.jsonl: una linea por orden, escrita ANTES de
        enviar la siguiente; si se aborta, la ultima linea es {"event": "abort"}.
      - broker_plan_<fecha>.csv: se escribe SIEMPRE (finally), con status por
        orden (submitted/rejected/failed/not_sent/blocked) y fills consultados
        (filled_qty, filled_avg_price, final_status).

    Guardas pre-trade: `guard` (default PreTradeGuard()) + kill-switch
    (kill_dir/KILL o HFT_KILL=1) se evaluan sobre el plan COMPLETO antes del
    primer POST. Si hay motivos y submit=True no se envia nada: el CSV queda
    con status "blocked" + blocked_reason. En dry-run solo se anotan.

    Los errores del broker/guardas NO se lanzan: se devuelven en `error` y
    `status` resume el efecto real. Ctrl-C/SystemExit se propagan tras dejar
    CSV + journal.

    Args:
        poll_timeout: Segundos esperando estado final de las ordenes (con
            mercado cerrado se hace una sola consulta).
        order_prefix: Prefijo de client_order_id (default `broker_run_id(t)`).
        kill_dir: Carpeta del archivo KILL (default `out_dir`).
    """
    plan = bplan.orders
    mode = "submitted" if submit else "dry_run"
    order_prefix = order_prefix or broker_run_id(t)
    day = pd.Timestamp(t).date().isoformat()
    plan_path = out_dir / f"broker_plan_{day}.csv"
    journal_path = out_dir / f"broker_journal_{day}.jsonl"

    reasons = (guard or PreTradeGuard()).evaluate(plan, bplan.summary)
    kill = kill_switch_reason(kill_dir or out_dir)
    if kill:
        reasons.insert(0, kill)
    blocked_reason = "; ".join(reasons) or None
    default_status = "not_sent"

    results: List[Dict] = []
    fills: Dict[str, Dict] = {}
    error: Optional[Exception] = None

    def _record(result: Dict) -> None:
        results.append(result)
        if submit:
            _append_journal(journal_path, {**result, "run_id": order_prefix})

    try:
        if reasons:
            if submit:
                raise PreTradeBlocked(reasons)
            logger.warning("Guardas pre-trade bloquearian el envio: %s", blocked_reason)
        client.submit_orders(plan, dry_run=not submit, run_id=order_prefix,
                             on_result=_record)
        ids = [r["id"] for r in results if r.get("id")]
        if submit and ids:
            fills = client.poll_orders(
                ids, timeout=0 if bplan.market_closed else poll_timeout)
    except BaseException as exc:  # incl. Ctrl-C: tambien deja envios parciales
        if isinstance(exc, PreTradeBlocked):
            reasons = list(exc.reasons)
            blocked_reason = "; ".join(reasons)
            if not results:
                default_status = "blocked"
        if submit:
            _append_journal(journal_path, (
                {"event": "blocked", "run_id": order_prefix, "reasons": exc.reasons}
                if isinstance(exc, PreTradeBlocked) else
                {"event": "abort", "run_id": order_prefix,
                 "error": f"{type(exc).__name__}: {exc}"}))
        if not isinstance(exc, Exception):
            raise
        error = exc
    finally:
        out = _plan_with_results(plan, results, fills, default_status,
                                 blocked_reason)
        out.to_csv(plan_path, index=False)
    execution = BrokerExecution(orders=out, results=results, mode=mode,
                                error=error, blocked_reasons=tuple(reasons))
    log = logger.info if error is None else logger.error
    log("Broker %s: %d enviadas / %d rechazadas / %d no ruteables / %d en plan "
        "(status=%s%s) → %s",
        mode, execution.n_submitted, execution.n_rejected,
        execution.n_not_routable, len(out),
        execution.status, f", error={error}" if error else "", plan_path)
    return execution


def run_broker_step(
    w_target: pd.Series,
    prices_t: pd.Series,
    t: pd.Timestamp,
    state_dir: Path,
    submit: bool = False,
    min_notional: float = 1.0,
    poll_timeout: float = POLL_TIMEOUT_S,
    run_id: Optional[str] = None,
    guard: Optional[PreTradeGuard] = None,
    client=None,
    portfolio: Optional[PortfolioState] = None,
    summary: Optional[Dict] = None,
) -> tuple[pd.DataFrame, list, str]:
    """read_broker + plan_broker + execute_broker en un paso, sin tocar el
    estado vivo.

    Siempre verifica la cuenta esperada (BROKER_ACCOUNT_ID) antes de
    enviar. Con submit=False solo planifica (dry-run, sin POST de ordenes).
    El plan se dimensiona SOLO con la cartera del broker (qty + equity).
    Auditoría y guardas: ver `execute_broker` (archivos en `state_dir`).

    Args:
        run_id: Prefijo de client_order_id (default `broker_run_id(t)`).
        guard: Topes pre-trade (default `PreTradeGuard()`).
        client: Cliente ya creado (default `make_broker_client(state_dir)`).
        summary / portfolio: Lecturas ya hechas (ver `read_broker`).

    Returns:
        (plan_df, resultados, estado): estado in {"dry_run", "submitted"}.

    Raises:
        ValueError: `portfolio` no viene del broker.
        El error del broker o PreTradeBlocked, tras dejar CSV + journal.
    """
    client = client if client is not None else make_broker_client(state_dir)
    account = read_broker(client, summary=summary, portfolio=portfolio)
    bplan = plan_broker(account, w_target, prices_t, min_notional=min_notional)
    execution = execute_broker(client, bplan, t, state_dir, submit=submit,
                               guard=guard, poll_timeout=poll_timeout,
                               order_prefix=run_id)
    if execution.error is not None:
        raise execution.error
    return execution.orders, execution.results, execution.mode


# ===================================================================
# Fases de run_once
# ===================================================================
@dataclass(frozen=True)
class LiveState:
    """Estado comprometido al inicio de la corrida (solo lectura)."""

    ledger: PositionLedger
    last_rebalance: Optional[pd.Timestamp]
    manifest: Dict[str, Any]


@dataclass(frozen=True)
class MarketData:
    """`prices` incluye las posiciones abiertas (solo para valorar/vender);
    la estrategia ve `strategy_prices`: el universo limpio, como siempre.
    Precios en moneda base; `currencies` = id → moneda local del instrumento
    (vacío: proveedor sin maestro, todo en base)."""

    prices: pd.DataFrame
    volumes: pd.DataFrame
    strategy_cols: Optional[tuple] = None   # None → todas las columnas
    currencies: Mapping[str, str] = field(default_factory=dict)
    source: str = ""        # Bars.meta.source: quién sirvió los precios

    @property
    def strategy_prices(self) -> pd.DataFrame:
        if self.strategy_cols is None:
            return self.prices
        return self.prices[list(self.strategy_cols)]

    @property
    def t(self) -> pd.Timestamp:
        return pd.Timestamp(self.prices.index[-1])


@dataclass(frozen=True)
class Schedule:
    """¿Toca rebalancear en t? `supersedes` = corrida ya comprometida con
    este mismo as_of (rerun con --force o reintento tras un parcial)."""

    t: pd.Timestamp
    last: Optional[pd.Timestamp]
    due: bool
    same_as_of: bool
    supersedes: Optional[str]

    @property
    def as_of(self) -> str:
        return self.t.date().isoformat()


@dataclass(frozen=True)
class SimPlan:
    """Rebalanceo del ledger simulado, calculado sobre una copia."""

    trades: pd.DataFrame          # [ticker, shares]
    turnover: float
    cost: float
    snap_pre: LedgerSnapshot
    ledger_after: PositionLedger
    snap_after: LedgerSnapshot
    new_txns: pd.DataFrame        # transacciones de esta corrida
    positions: pd.DataFrame       # latest_snapshot.csv (shares, avg_entry, weight)


@dataclass(frozen=True)
class RunPlan:
    """Todo lo que haría la corrida; nada aplicado ni persistido aún."""

    run_id: str
    schedule: Schedule
    regime: str
    forced: bool
    w_target: pd.Series
    exits: pd.DataFrame
    prices_t: pd.Series
    portfolio: PortfolioState          # fuente de verdad, antes de operar
    turnover: float
    cost: float
    history_mode: str = STRICT
    history_rows: int = 0
    meta: Optional[Dict[str, Any]] = None  # build_targets: factors_used…
    config_hash: str = ""          # StrategyConfig.config_hash() efectivo
    config_source: str = ""        # archivo de config (ruta)
    # src.provenance.run_provenance: git, fuente de datos, esquemas
    provenance: Mapping[str, Any] = field(default_factory=dict)
    sim: Optional[SimPlan] = None                  # mode="sim"
    broker: Optional[BrokerPlan] = None            # mode="broker"
    mirror: Optional[PositionLedger] = None        # broker: ledger.json espejo
    local_backup: Optional[PositionLedger] = None  # broker: ledger local a respaldar

    @property
    def mode(self) -> str:
        return "broker" if self.broker is not None else "sim"

    @property
    def factors_used(self) -> List[str]:
        return list((self.meta or {}).get("factors_used", []))

    def positions_frame(self) -> pd.DataFrame:
        """latest_snapshot.csv: sim → ledger tras los trades; broker → cartera
        leída antes de enviar (los fills se espejan en la próxima corrida)."""
        if self.sim is not None:
            return self.sim.positions
        p = self.portfolio
        return pd.DataFrame({"shares": p.positions, "avg_entry": p.avg_entry,
                             "weight": p.weights(self.prices_t)})


@dataclass(frozen=True)
class RunOptions:
    force: bool = False
    start: Optional[str] = None
    every_n: int = REBALANCE_EVERY_N
    check_only: bool = False
    mode: str = "sim"
    submit: bool = False
    dry_run: bool = False
    allow_closed: bool = False
    min_notional: float = 1.0
    strategy: StrategyConfig = StrategyConfig()
    config_source: str = ""
    poll_timeout: float = POLL_TIMEOUT_S
    guard: Optional[PreTradeGuard] = None
    history_mode: str = STRICT
    data_sources: tuple = DEFAULT_SOURCES
    data_policy: DataPolicy = LIVE_POLICY
    data_cache: bool = False
    run_id: str = ""                 # "" → new_run_id() al planificar
    code: CodeVersion = CodeVersion()


def load_state(store: StateStore, initial_cash: float = INITIAL_CASH) -> LiveState:
    """Fase 1 — lee el estado comprometido. No escribe ni crea nada."""
    changed = store.verify()
    if changed:
        logger.warning(
            "State files differ from %s (edited outside the runner?): %s",
            MANIFEST_NAME, changed,
        )
    return LiveState(
        ledger=load_ledger(store.path("ledger.json"), initial_cash=initial_cash),
        last_rebalance=read_last_rebalance(store.path("last_rebalance.txt")),
        manifest=store.read_manifest() or {},
    )


def _unpriced(positions: pd.Series, prices_t: pd.Series) -> List[str]:
    """Tickers con posicion abierta (largo o corto) sin precio finito > 0."""
    held = positions.index[positions.abs() > 1e-12]
    px = pd.to_numeric(prices_t.reindex(held), errors="coerce").astype(float)
    return sorted(held[~(np.isfinite(px) & (px > 0))])


def _abort_missing_price(exc: MissingPriceError) -> int:
    """Log + mensaje claro; el caller retorna sin planificar ni escribir."""
    logger.error("ABORT (estado intacto): %s", exc)
    print(
        f"\nABORT: posiciones abiertas sin precio valido: {exc.tickers}.\n"
        f"No se escribio estado ni se opero. Revisar la fuente de precios "
        f"(Yahoo / CSV fallback) antes de reintentar.\n{exc}\n",
        file=sys.stderr,
    )
    return EXIT_MISSING_PRICE


def fetch_data(
    start: str,
    requirements: Optional[Dict[str, int]] = None,
    history_mode: str = STRICT,
    end: Optional[str] = None,
    held: Optional[List[str]] = None,
    *,
    provider: Optional[MarketDataProvider] = None,
    policy: DataPolicy = LIVE_POLICY,
    now: Optional[datetime] = None,
) -> MarketData:
    """Fase 2 — universo + precios/volúmenes limpios (red; no toca estado).

    Args:
        requirements: Filas por componente (default: las de config/live.toml,
            `history_requirements(load_strategy_config())`).
        history_mode: strict exige todas; permissive solo la ventana de sizing.
        held: Posiciones abiertas: se descargan aunque no esten en el
            universo y `clean_prices` nunca las descarta (must_keep).
        provider: Fuente de datos (default: make_market_data_provider()).
        policy: Faltantes / barra intradía / frescura (default LIVE_POLICY).
        now: Reloj de la corrida (default utcnow()).

    Raises:
        InsufficientHistoryError: faltan filas para algún componente exigido.
        MissingPriceError: una posicion abierta no vino en la descarga.
        MarketDataError: datos inutilizables según `policy`.
        MissingFxError: moneda local sin serie FX (ver `close_in_base`).
    """
    requirements = requirements or history_requirements(load_strategy_config())
    logger.info("Loading universe + prices from %s (need %d rows, mode=%s) …",
                start, max(requirements.values()), history_mode)
    tickers = load_universe()["ticker"].tolist()
    universe = set(tickers)
    held = list(held or [])
    # Universo ∪ posiciones abiertas: sin precio una posicion no se puede
    # valorar (NAV falso) ni vender (queda huerfana).
    fetch_tickers = tickers + [tk for tk in held if tk not in universe]
    now = now or utcnow()
    # end exclusivo: mañana para incluir la sesión de hoy si ya cerró
    end = end or (now + timedelta(days=1)).strftime("%Y-%m-%d")
    provider = provider or make_market_data_provider(load_settings())
    try:
        bars = load_bars(provider, fetch_tickers, start, end, policy=policy, now=now)
    except (MissingSymbolsError, StaleDataError) as exc:
        # Posición abierta sin datos o sin barra en la última sesión cerrada:
        # mismo aborto que sin precio en t (exit 6); solo universo → exit 7.
        bad = exc.missing if isinstance(exc, MissingSymbolsError) else exc.stale
        held_bad = [tk for tk in bad if tk in set(held)]
        if held_bad:
            raise MissingPriceError(held_bad, str(exc)) from exc
        raise
    logger.info(
        "Market data: source=%s symbols=%d/%d last_closed_session=%s "
        "dropped_partial=%s", bars.meta.source, len(bars.meta.received),
        len(fetch_tickers), bars.meta.expected_session.date(),
        [d.date().isoformat() for d in bars.meta.dropped_partial],
    )
    close, currencies = close_in_base(provider, bars.close, start, end)
    prices, quality = clean_prices(close, must_keep=held)
    # La estrategia ve exactamente lo de antes: universo limpio. Posiciones
    # fuera del universo o rescatadas por must_keep solo se valoran/venden.
    rescued = set(quality.loc[quality["kept_by_must_keep"], "ticker"])
    strategy_cols = tuple(c for c in prices.columns
                          if c in universe and c not in rescued)
    # Volumen faltante = NaN (el factor de liquidez lo excluye); nunca se imputa
    volumes = bars.volume.reindex(index=prices.index, columns=list(strategy_cols))
    hard = (requirements if history_mode == STRICT
            else {"sizing": requirements["sizing"]})
    missing = missing_requirements(len(prices), hard)
    if missing:
        raise InsufficientHistoryError(len(prices), missing)
    return MarketData(prices=prices, volumes=volumes, strategy_cols=strategy_cols,
                      currencies=currencies, source=bars.meta.source)


def decide_schedule(
    state: LiveState,
    data: MarketData,
    every_n: int = REBALANCE_EVERY_N,
) -> Schedule:
    """Fase 3 — ¿toca? Un as_of ya comprometido solo se reescribe con --force."""
    t = data.t
    last = state.last_rebalance
    same_as_of = last is not None and last.normalize() == t.normalize()
    due = not same_as_of and should_rebalance(data.prices, last, every_n=every_n)
    as_of = t.date().isoformat()
    supersedes = (state.manifest.get("run_id")
                  if state.manifest.get("as_of") == as_of else None)
    return Schedule(t=t, last=last, due=due, same_as_of=same_as_of,
                    supersedes=supersedes)


def estimate_turnover(
    w_prev: pd.Series,
    w_target: pd.Series,
    equity: float,
    cfg: StrategyConfig,
) -> tuple[float, float]:
    """(turnover L1 vs pesos actuales, costo USD = turnover × cfg.cost_bps
    sobre equity; misma fórmula que el backtest, ``strategy.turnover_cost``)."""
    turnover, cost_frac = turnover_cost(w_prev, w_target, cfg)
    return turnover, cost_frac * max(equity, 1e-6)


def build_sim_plan(
    ledger: PositionLedger,
    w_target: pd.Series,
    prices_t: pd.Series,
    t: pd.Timestamp,
    cfg: StrategyConfig,
    currencies: Optional[Mapping[str, str]] = None,
) -> SimPlan:
    """Fase 5a — rebalanceo del ledger simulado sobre una COPIA (puro).

    `currencies` (id → moneda local) se guarda en los lotes nuevos.
    """
    snap_pre = ledger.snapshot(t, prices_t)
    nav_pre = ledger.nav(prices_t)
    turnover, cost = estimate_turnover(ledger.weights_from_positions(prices_t),
                                       w_target, nav_pre, cfg)

    after = PositionLedger.from_dict(ledger.to_dict())
    # Reserva costos: dimensiona sobre nav_pre - cost → cash >= 0 tras pagarlos
    trades = after.trades_from_target_weights(w_target, prices_t, nav=nav_pre,
                                              costs=cost)
    n_txns_before = len(after.transactions_df())
    after.apply_trades(trades, prices_t, costs=cost, t=t, currencies=currencies)
    snap_after = after.snapshot(t, prices_t)
    # Audit trail FIFO: solo las transacciones de esta corrida
    new_txns = pd.DataFrame(after.to_dict()["txns"][n_txns_before:])
    positions = pd.DataFrame({
        "shares": snap_after.positions,
        "avg_entry": snap_after.avg_entry,
        "weight": after.weights_from_positions(prices_t),
    })
    return SimPlan(trades=trades, turnover=turnover, cost=cost,
                   snap_pre=snap_pre, ledger_after=after, snap_after=snap_after,
                   new_txns=new_txns, positions=positions)


def run_status(execution: Optional[BrokerExecution]) -> str:
    """ok/partial/failed/dry_run; sin broker (solo sim) la ejecución es ok."""
    return "ok" if execution is None else execution.status


def history_row(plan: RunPlan, execution: Optional[BrokerExecution]) -> Dict[str, Any]:
    """Fila de rebalance_history.

    sim: NAV/posiciones del ledger que queda comprometido. broker: cartera
    leída antes de enviar (los fills se reconcilian, espejo + drift, al
    inicio de la próxima corrida); realized_pnl no lo expone el broker.
    """
    status = run_status(execution)
    if plan.sim is not None:
        snap = plan.sim.snap_after if status == "ok" else plan.sim.snap_pre
        nav, cash, positions = snap.nav, snap.cash, snap.positions
        realized, unrealized = snap.realized_pnl, snap.unrealized_pnl
    else:
        p = plan.portfolio
        nav, cash, positions = p.equity, p.cash, p.positions
        realized, unrealized = float("nan"), p.unrealized_pnl(plan.prices_t)
    error = execution.error if execution is not None else None
    return {
        "date": plan.schedule.as_of,
        "mode": plan.mode,
        "regime": plan.regime,
        "nav": nav,
        "cash": cash,
        "turnover": plan.turnover,
        "cost": plan.cost,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "n_positions": int((positions > 0).sum()) if len(positions) else 0,
        "forced": plan.forced,
        "history_mode": plan.history_mode,
        "history_rows": plan.history_rows,
        "factors_used": ";".join(plan.factors_used),
        "regime_breadth": bool((plan.meta or {}).get("regime_breadth", False)),
        "config_hash": plan.config_hash,
        "config_source": plan.config_source,
        **plan.provenance,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": plan.run_id,
        "supersedes": plan.schedule.supersedes,
        "status": status,
        "n_orders": len(execution.orders) if execution is not None else None,
        "n_submitted": execution.n_submitted if execution is not None else None,
        "n_rejected": execution.n_rejected if execution is not None else None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
    }


def book_after(
    plan: RunPlan,
    execution: Optional[BrokerExecution] = None,
) -> tuple[pd.Series, float]:
    """(acciones por ticker, caja) de la cartera TRAS la corrida.

    sim: ledger tras los trades (exacto). broker: cartera leída antes de
    enviar + fills conocidos al comprometer (filled_qty × filled_avg_price;
    sin fill → sin cambio). Es una estimación: la próxima corrida espeja el
    broker y la corrige.
    """
    if plan.sim is not None:
        snap = plan.sim.snap_after
        return snap.positions.astype(float), float(snap.cash)
    shares = plan.portfolio.positions.astype(float)
    cash = float(plan.portfolio.cash)
    if execution is None or execution.orders.empty:
        return shares, cash
    orders = execution.orders
    qty = pd.to_numeric(orders["filled_qty"], errors="coerce").fillna(0.0)
    signed = qty.where(orders["side"] == "buy", -qty)
    px = pd.to_numeric(orders["filled_avg_price"], errors="coerce").fillna(
        orders["ticker"].map(plan.prices_t).astype(float))
    delta = signed.groupby(orders["ticker"]).sum()
    shares = shares.add(delta[delta != 0.0], fill_value=0.0)
    cash -= float((signed * px).sum())
    return shares, cash


def snapshot_frame(
    plan: RunPlan,
    execution: Optional[BrokerExecution] = None,
) -> pd.DataFrame:
    """Filas de snapshots.csv: una por ticker con posición + caja (CASH_COL).

    Columnas: ticker, shares, price (precio de t en moneda base; caja = 1),
    value, weight (value / NAV tras la corrida).
    """
    shares, cash = book_after(plan, execution)
    shares = shares[shares.abs() > 1e-12].sort_index()
    price = plan.prices_t.reindex(shares.index).astype(float)
    frame = pd.DataFrame({
        "ticker": [*shares.index, CASH_COL],
        "shares": [*shares.to_numpy(), cash],
        "price": [*price.to_numpy(), 1.0],
    })
    frame["value"] = frame["shares"] * frame["price"]
    nav = float(frame["value"].sum())
    frame["weight"] = frame["value"] / nav if nav else float("nan")
    return frame


def commit_state(
    store: StateStore,
    plan: RunPlan,
    execution: Optional[BrokerExecution] = None,
) -> Dict[str, Any]:
    """Fase 7 — publica la corrida en UNA transacción (requiere el lock).

    ok      ledger (sim: ledger tras los trades + transactions; broker: espejo
            del broker + respaldo del ledger local si no coincidía),
            exit_log, last_rebalance, latest_snapshot, snapshots (cartera
            tras la corrida, `snapshot_frame`), broker_orders (si hubo
            broker) y la fila de history.
    partial solo lo efectivamente ejecutado (broker_orders, estado por orden
            según ARQ-05) + history status=partial. Ledger y last_rebalance
            quedan como estaban → la próxima corrida toca y reintenta.

    Returns:
        El manifiesto escrito.

    Raises:
        ValueError: status sin nada que comprometer (failed/dry_run).
    """
    status = run_status(execution)
    if status not in ("ok", "partial"):
        raise ValueError(f"Nothing to commit for a run with status={status!r}.")
    run_id, as_of = plan.run_id, plan.schedule.as_of
    with store.transaction(run_id, as_of=as_of):
        if status == "ok":
            if plan.sim is not None:
                store.write_json("ledger.json", plan.sim.ledger_after.to_dict())
                store.append_csv("transactions.csv",
                                 plan.sim.new_txns.assign(run_id=run_id))
            else:
                if plan.local_backup is not None:
                    store.write_json(f"ledger_local_backup_{as_of}.json",
                                     plan.local_backup.to_dict())
                store.write_json("ledger.json", plan.mirror.to_dict())
            store.append_csv("exit_log.csv",
                             plan.exits.assign(date=as_of, run_id=run_id))
            store.write_text("last_rebalance.txt", as_of)
            store.write_csv("latest_snapshot.csv", plan.positions_frame(), index=True)
            store.append_csv("snapshots.csv", snapshot_frame(plan, execution).assign(
                date=as_of, run_id=run_id, mode=plan.mode))
        if execution is not None:
            store.append_csv("broker_orders.csv",
                             execution.orders.assign(date=as_of, run_id=run_id))
        store.append_csv("rebalance_history.csv",
                         pd.DataFrame([history_row(plan, execution)]))
    return store.read_manifest() or {}


def save_plan(
    plan_dir: Path,
    plan: RunPlan,
    execution: Optional[BrokerExecution] = None,
) -> Path:
    """Dry-run: deja el plan en `plan_dir` (fuera de live_state)."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    bplan = plan.broker
    summary = {
        "run_id": plan.run_id,
        "as_of": plan.schedule.as_of,
        "mode": plan.mode,
        "dry_run": True,
        "due": plan.schedule.due,
        "forced": plan.forced,
        "supersedes": plan.schedule.supersedes,
        "regime": plan.regime,
        "history_mode": plan.history_mode,
        "history_rows": plan.history_rows,
        "factors_used": plan.factors_used,
        "config_hash": plan.config_hash,
        "config_source": plan.config_source,
        "provenance": dict(plan.provenance),
        "nav_pre": plan.portfolio.equity,
        "turnover": plan.turnover,
        "cost": plan.cost,
        "n_sim_trades": len(plan.sim.trades) if plan.sim is not None else None,
        "exits": plan.exits["ticker"].tolist(),
        "broker": None if bplan is None else {
            "equity": bplan.equity,
            "market_open": bplan.summary.get("market_open"),
            "n_orders": len(bplan.orders),
            "gross_notional": float(bplan.orders["notional"].abs().sum()),
            "ledger_backup_on_commit": plan.local_backup is not None,
        },
        "blocked_reasons": list(execution.blocked_reasons) if execution else [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (plan_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plan.w_target.rename("weight").to_csv(plan_dir / "targets.csv", index_label="ticker")
    plan.exits.to_csv(plan_dir / "exits.csv", index=False)
    if plan.sim is not None:
        plan.sim.trades.to_csv(plan_dir / "sim_trades.csv", index=False)
    return plan_dir


def _print_dry_run(plan: RunPlan, plan_dir: Path,
                   execution: Optional[BrokerExecution]) -> None:
    lines = [
        f"\nDRY-RUN {plan.schedule.as_of} | run={plan.run_id} | mode={plan.mode} "
        f"| regime={plan.regime} | due={plan.schedule.due} forced={plan.forced}",
        f"  Config: {plan.config_source} (sha256 {plan.config_hash[:12]})",
        f"  Cartera ({plan.portfolio.source}): equity={plan.portfolio.equity:,.2f} "
        f"| turnover={plan.turnover:.3f} | cost={plan.cost:,.2f}",
        f"  Salidas: {plan.exits['ticker'].tolist() or '-'}",
    ]
    if plan.sim is not None:
        lines.append(f"  Sim: {len(plan.sim.trades)} trades")
    if plan.broker is not None:
        orders = plan.broker.orders
        lines.append(f"  Broker: {len(orders)} órdenes vs equity "
                     f"{plan.broker.equity:,.2f}")
        if not orders.empty:
            lines.append(orders[["ticker", "side", "qty", "notional"]]
                         .to_string(index=False))
    if execution is not None and execution.blocked_reasons:
        lines.append("  --submit se BLOQUEARÍA: " + "; ".join(execution.blocked_reasons))
    lines.append(f"  Plan → {plan_dir}/ (live_state intacto)\n")
    print("\n".join(lines))


def _run_phases(store: StateStore, settings: Settings, opts: RunOptions) -> int:
    """Orquesta las fases. Solo commit_state escribe en `store` (y el
    journal/plan de execute_broker con --submit)."""
    state = load_state(store, initial_cash=settings.initial_cash)
    cfg = opts.strategy
    logger.info("Estrategia: %s (sha256 %s)", opts.config_source,
                cfg.config_hash()[:12])
    requirements = history_requirements(cfg)
    start = opts.start or default_start(max(requirements.values()))
    try:
        data = fetch_data(
            start, requirements, opts.history_mode,
            held=state.ledger.positions().index.tolist(),
            provider=make_market_data_provider(
                settings, sources=opts.data_sources, cache=opts.data_cache),
            policy=opts.data_policy, now=utcnow(),
        )
    except InsufficientHistoryError as exc:
        logger.error("%s (start=%s, mode=%s)", exc, start, opts.history_mode)
        return EXIT_NO_DATA
    except MissingPriceError as exc:
        return _abort_missing_price(exc)
    except (MarketDataError, MissingFxError) as exc:
        logger.error("Market data unusable, nothing written: %s", exc)
        print(f"\nDATA ERROR (nada cambiado): {exc}\n", file=sys.stderr)
        return EXIT_DATA

    sched = decide_schedule(state, data, every_n=opts.every_n)
    t = sched.t
    logger.info(
        "As-of %s | last_rebalance=%s | trading_days_since=%s | due=%s",
        t.date(),
        sched.last.date() if sched.last is not None else "never",
        trading_days_between(data.prices, sched.last, t)
        if sched.last is not None else "n/a",
        sched.due or opts.force,
    )
    if opts.check_only:
        return EXIT_OK if (sched.due or opts.force) else EXIT_NOT_DUE
    if not sched.due and not opts.force:
        if sched.same_as_of:
            logger.info(
                "Skip: as_of %s already committed (run %s). Use --force to "
                "record a new run that supersedes it.",
                sched.as_of, state.manifest.get("run_id", "legacy"),
            )
        else:
            logger.info(
                "Skip: fewer than %d trading days since last rebalance. "
                "Use --force to override.",
                opts.every_n,
            )
        return EXIT_NOT_DUE

    prices_t = data.prices.loc[t]
    # Sim: la cartera es el ledger local → sin precio en t, abortar ya.
    if opts.mode == "sim":
        unpriced = _unpriced(state.ledger.positions(), prices_t)
        if unpriced:
            return _abort_missing_price(MissingPriceError(
                unpriced, f"Sin precio valido en {t.date()}."))

    # Fuente de verdad ANTES de construir objetivos: salidas y pesos previos
    # salen solo de ella (una por corrida).
    client = account = None
    if opts.mode == "broker":
        try:
            client = make_broker_client(settings.state_dir)
            account = read_broker(client)
        except Exception as exc:  # noqa: BLE001 — nada enviado ni persistido
            logger.error("No se pudo leer la cartera del broker: %s", exc)
            print(f"\nBroker ERROR leyendo cartera (nada enviado ni "
                  f"persistido): {exc}\n", file=sys.stderr)
            return EXIT_BROKER
        portfolio = account.portfolio
        # Posicion del broker sin precio en t: abortar antes de planificar.
        unpriced = _unpriced(portfolio.positions, prices_t)
        if unpriced:
            return _abort_missing_price(MissingPriceError(
                unpriced, f"Sin precio valido en {t.date()}."))
    else:
        portfolio = PortfolioState.from_ledger(state.ledger, prices_t, as_of=t)

    # Misma construcción que el backtest (src/strategy.build_targets). La
    # estrategia ve solo el universo; `data.prices` trae además las
    # posiciones abiertas (valorar / evaluar salidas / vender).
    try:
        targets = build_targets(
            portfolio, data.prices, data.volumes, t, cfg,
            universe=data.strategy_cols, history_mode=opts.history_mode,
        )
    except (InsufficientDataError, InsufficientHistoryError) as exc:
        logger.error("%s (mode=%s)", exc, opts.history_mode)
        return EXIT_NO_DATA
    w_target, exits, regime, meta = (targets.weights, targets.exits,
                                      targets.regime, targets.meta)
    if meta.get("factors_excluded"):
        logger.warning("Factores excluidos (permissive): %s",
                       meta["factors_excluded"])
    if not exits.empty:
        logger.info("Salidas (%s): %s", portfolio.source, exits["ticker"].tolist())

    # Libro a ejecutar = el del backtest: tope de turnover (R-02) + topes por
    # grupo sobre lo que se mantiene; caja y salidas van directo al objetivo.
    w_exec = execution_weights(targets, portfolio, cfg)
    sim = bplan = mirror = local_backup = None
    if account is None:
        sim = build_sim_plan(state.ledger, w_exec, prices_t, t, cfg,
                             currencies=data.currencies)
        turnover, cost = sim.turnover, sim.cost
    else:
        bplan = plan_broker(account, w_exec, prices_t,
                            min_notional=opts.min_notional)
        turnover, cost = estimate_turnover(portfolio.weights(prices_t), w_exec,
                                           portfolio.equity, cfg)
        mirror, local_backup = mirror_ledger_from_broker(
            state.ledger, portfolio, prices_t, t,
            ledger_exists=store.path("ledger.json").exists())

    plan = RunPlan(run_id=opts.run_id or new_run_id(), schedule=sched,
                   regime=regime, forced=bool(opts.force), w_target=w_target,
                   exits=exits, prices_t=prices_t, portfolio=portfolio,
                   turnover=turnover, cost=cost, history_mode=opts.history_mode,
                   history_rows=len(data.prices), meta=meta,
                   config_hash=cfg.config_hash(),
                   config_source=opts.config_source,
                   provenance=run_provenance(opts.code, data_source=data.source),
                   sim=sim, broker=bplan, mirror=mirror, local_backup=local_backup)
    guard = submit_guard(opts.guard, opts.allow_closed)

    if opts.dry_run:
        plan_dir = settings.results_dir / PLANS_DIRNAME / plan.run_id
        plan_dir.mkdir(parents=True, exist_ok=True)
        execution = None
        if bplan is not None:
            execution = execute_broker(client, bplan, t, plan_dir, submit=False,
                                       guard=guard, kill_dir=settings.state_dir)
        save_plan(plan_dir, plan, execution)
        _print_dry_run(plan, plan_dir, execution)
        return EXIT_OK

    execution = None
    if bplan is not None:
        execution = execute_broker(client, bplan, t, store.root, submit=True,
                                   guard=guard, poll_timeout=opts.poll_timeout,
                                   kill_dir=store.root)
        if execution.status == "failed":
            print(f"\nBroker FALLÓ sin ejecutar órdenes (nada comprometido; ver "
                  f"{store.root}/broker_plan_*.csv y broker_journal_*.jsonl): "
                  f"{execution.error or 'todas las órdenes rechazadas'}\n",
                  file=sys.stderr)
            return EXIT_BROKER

    commit_state(store, plan, execution)
    status = run_status(execution)
    if status == "partial":
        print(f"\nPARCIAL {sched.as_of} | run={plan.run_id} | "
              f"{execution.n_submitted}/{len(execution.orders)} órdenes enviadas | "
              f"error: {execution.error}\n"
              f"Comprometido: broker_orders.csv + history status=partial. "
              f"last_rebalance NO se actualizó: la próxima corrida reintenta.\n",
              file=sys.stderr)
        return EXIT_BROKER

    row = history_row(plan, execution)
    logger.info(
        "REBALANCED %s | run=%s | mode=%s | regime=%s | NAV=%.2f | turnover=%.3f "
        "| positions=%d",
        t.date(), plan.run_id, plan.mode, regime, row["nav"], turnover,
        row["n_positions"],
    )
    print(
        f"\nOK rebalance {t.date()} ({plan.mode}) | run={plan.run_id} | "
        f"regime={regime} | NAV={row['nav']:,.2f} | positions={row['n_positions']}\n"
        f"State → {store.root}/\n"
    )
    if execution is not None:
        print(f"Broker submitted: {execution.n_submitted}/{len(execution.orders)} "
              f"órdenes ({execution.n_rejected} rechazadas)\n")
    return EXIT_OK


# ===================================================================
# Main step
# ===================================================================
def run_once(
    force: bool = False,
    start: Optional[str] = None,
    every_n: int = REBALANCE_EVERY_N,
    check_only: bool = False,
    mode: str = "sim",
    submit: bool = False,
    dry_run: bool = False,
    allow_closed: bool = False,
    min_notional: float = 1.0,
    stop_loss_pct: Any = FROM_CONFIG,
    take_profit_pct: Any = FROM_CONFIG,
    trailing_stop_pct: Any = FROM_CONFIG,
    trailing_window: Any = FROM_CONFIG,
    take_profit_dollars: Any = FROM_CONFIG,
    stop_loss_dollars: Any = FROM_CONFIG,
    poll_timeout: float = POLL_TIMEOUT_S,
    guard: Optional[PreTradeGuard] = None,
    settings: Optional[Settings] = None,
    history_mode: str = STRICT,
    data_sources=DEFAULT_SOURCES,
    data_policy: DataPolicy = LIVE_POLICY,
    data_cache: bool = False,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    notifier: Optional[Notifier] = None,
) -> int:
    """Execute one periodic-rebalance check/step (fases: ver docstring del módulo).

    Una sola fuente de verdad por corrida (`PortfolioState`): salidas,
    turnover y dimensionamiento solo leen la cartera de `mode`.

    Con commit (default o --submit) corre bajo el lock exclusivo de
    ``settings.state_dir`` y publica en una sola transacción. --check-only,
    --dry-run y mode="broker" sin submit son de solo lectura: sin lock y sin
    escribir en live_state (--dry-run deja el plan en results/plans/).

    Trazabilidad: todo log de la corrida lleva ``run_id`` (`run_logging`);
    con lock, además a ``settings.logs_dir/runner.log``. El resultado (exit
    code ≠ 1, o excepción) se notifica a ``notifier``.

    Args:
        start: Inicio de la descarga; default = historia requerida + margen.
        mode: "sim" (default): cartera = ledger local; trades simulados sobre
            el; nada toca la red. "broker": cartera = Alpaca paper, leida
            antes de salidas y dimensionamiento; ledger.json pasa a espejo
            del broker al comprometer (drift al log). Sin `submit` es dry-run.
        submit: Solo con mode="broker": envia las ordenes.
        dry_run: Todas las fases salvo execute y commit.
        allow_closed: Con submit, permite enviar con el mercado cerrado.
        settings: Rutas/valores base; default `load_settings()` (HFT_HOME o repo).
        history_mode: "strict" (exit 2 si algún componente no tiene su
            historia) o "permissive" (excluye factores sin historia; solo
            la ventana HRP es obligatoria).
        data_sources: Fuentes de datos en orden de fallback (src/market_data).
        data_policy: Faltantes / barra intradía / frescura. Default
            LIVE_POLICY: faltante o dato atrasado → exit 7; barra de hoy con
            el mercado abierto se descarta.
        data_cache: Cache parquet por símbolo en ``settings.bars_cache_dir``.
        config_path: Config de estrategia (default config/live.toml).
        stop_loss_pct, take_profit_pct, trailing_stop_pct, trailing_window,
        take_profit_dollars, stop_loss_dollars: Overrides del campo homónimo
            de la config para esta corrida (FROM_CONFIG = sin override;
            None = regla off). Ver `apply_exit_overrides`.
        run_id: Id de la corrida (default `new_run_id()`): logs, history,
            transactions, snapshots, plans y alertas.
        notifier: Destino de alertas (default `build_notifier(settings)`:
            consola + logs/alerts.jsonl + email opcional; solo consola en
            corridas de solo lectura).

    Returns:
        Exit code: EXIT_OK (0), EXIT_NOT_DUE (1), EXIT_NO_DATA (2),
        EXIT_BROKER (3), EXIT_LOCKED (4), EXIT_MISSING_PRICE (6),
        EXIT_DATA (7).

    Raises:
        ValueError: `mode`/`history_mode` desconocido, submit sin
            mode="broker", submit junto con dry_run, o config inválida.
    """
    if mode not in MODES:
        raise ValueError(f"mode={mode!r} no soportado (usar {MODES}).")
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"history_mode must be one of {HISTORY_MODES}")
    if submit and mode != "broker":
        raise ValueError("--submit requires --mode broker")
    if submit and dry_run:
        raise ValueError("--submit and --dry-run are mutually exclusive")
    implicit_dry_run = mode == "broker" and not submit and not dry_run
    dry_run = dry_run or implicit_dry_run
    read_only = check_only or dry_run
    settings = settings or load_settings()
    run_id = run_id or new_run_id()

    with run_logging(run_id):
        if notifier is None:
            notifier = build_notifier(settings, writable=not read_only)
        if implicit_dry_run:
            logger.info("--mode broker sin --submit: dry-run (no se envía ni "
                        "persiste nada).")
        code = code_version()
        logger.info("Run %s start: mode=%s%s code=%s", run_id, mode,
                    " (solo lectura)" if read_only else "", code.short)

        def execute(store: StateStore) -> int:
            path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
            strategy = apply_exit_overrides(
                load_strategy_config(path),
                stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct, trailing_window=trailing_window,
                take_profit_dollars=take_profit_dollars,
                stop_loss_dollars=stop_loss_dollars,
            )
            opts = RunOptions(
                force=force, start=start, every_n=every_n, check_only=check_only,
                mode=mode, submit=submit, dry_run=dry_run,
                allow_closed=allow_closed, min_notional=min_notional,
                strategy=strategy, config_source=str(path),
                poll_timeout=poll_timeout, guard=guard, history_mode=history_mode,
                data_sources=tuple(data_sources), data_policy=data_policy,
                data_cache=data_cache, run_id=run_id, code=code,
            )
            return _run_phases(store, settings, opts)

        if read_only:
            # Solo lectura: sin lock ni mkdir (ni log a archivo); lo leído puede
            # estar a medio commit de otra corrida (StateStore publica archivo
            # por archivo).
            store = StateStore(settings.state_dir)
            return _reported_run(lambda: execute(store), run_id, notifier,
                                 store, read_only=True)

        store = StateStore(_ensure_state_dir(settings))
        try:
            store.acquire()
        except StateLockedError as exc:
            logger.error("%s", exc)
            print(f"\nLOCKED: {exc}\n", file=sys.stderr)
            _notify_outcome(notifier, EXIT_LOCKED, run_id, str(exc))
            return EXIT_LOCKED
        try:
            # runner.log solo con el lock: una corrida escritora a la vez rota
            # el archivo.
            with log_to_file(settings.logs_dir):
                try:
                    return _reported_run(lambda: execute(store), run_id,
                                         notifier, store, read_only=False)
                finally:
                    store.close()
        finally:
            store.release()


# Resultado de la corrida → alerta (nivel, título). EXIT_NOT_DUE no alerta;
# EXIT_OK solo en corridas que comprometen (no en --check-only/--dry-run).
_OUTCOME_ALERTS = {
    EXIT_OK: ("info", "Corrida comprometida"),
    EXIT_NO_DATA: ("error", "Datos insuficientes: nada escrito"),
    EXIT_BROKER: ("error", "Broker: fallo, bloqueo o envío parcial"),
    EXIT_LOCKED: ("warning", "state_dir bloqueado por otra corrida"),
    EXIT_UNEXPECTED: ("critical", "Error inesperado: nada comprometido"),
    EXIT_MISSING_PRICE: ("error", "Posición abierta sin precio: nada escrito"),
    EXIT_DATA: ("error", "Datos de mercado inutilizables: nada escrito"),
}


class _LastError(logging.Handler):
    """Recuerda el último ERROR de la corrida (detalle de la alerta)."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.message: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        self.message = record.getMessage()


def _notify_outcome(
    notifier: Notifier,
    rc: int,
    run_id: str,
    message: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    """Alerta según el exit code (ver _OUTCOME_ALERTS); nunca lanza."""
    if rc not in _OUTCOME_ALERTS:
        return
    level, title = _OUTCOME_ALERTS[rc]
    alert = Alert(level=level, title=title, message=message or title,
                  run_id=run_id, exit_code=rc, details=dict(details or {}))
    try:
        notifier.notify(alert)
    except Exception as exc:  # noqa: BLE001 — una alerta nunca rompe la corrida
        logger.warning("Notifier falló (%s: %s).", type(exc).__name__, exc)


def _reported_run(
    run: Any,
    run_id: str,
    notifier: Notifier,
    store: StateStore,
    read_only: bool,
) -> int:
    """Ejecuta ``run()`` y notifica su resultado.

    Una excepción se loguea aquí con traceback (dentro del runner.log de la
    corrida), se notifica como critical y se relanza (main → exit 5).
    """
    last_error = _LastError()
    root = logging.getLogger()
    root.addHandler(last_error)
    try:
        try:
            rc = run()
        except Exception as exc:
            logger.exception("Error inesperado: nada comprometido.")
            _notify_outcome(notifier, EXIT_UNEXPECTED, run_id,
                            f"{type(exc).__name__}: {exc}")
            raise
    finally:
        root.removeHandler(last_error)
    if rc == EXIT_OK and read_only:
        return rc
    details: Dict[str, Any] = {"read_only": read_only}
    if rc == EXIT_OK:
        manifest = store.read_manifest() or {}
        if manifest.get("run_id") == run_id:
            details["as_of"] = manifest.get("as_of")
    _notify_outcome(notifier, rc, run_id, last_error.message, details)
    return rc


def _parse_sources(text: str) -> tuple:
    names = tuple(n.strip().lower() for n in text.split(",") if n.strip())
    unknown = [n for n in names if n not in AVAILABLE_SOURCES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"fuentes desconocidas {unknown or names}; disponibles: {AVAILABLE_SOURCES}")
    return names


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Periodic 15-trading-day portfolio rebalance (sim o broker paper).",
        epilog="Exit codes: 0 ok, 1 not due, 2 insufficient data, 3 broker "
               "(failed/blocked/partial), 4 state locked, 5 unexpected error, "
               "6 open position without price, 7 unusable market data.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebalance even if <15 trading days since last run.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Exit 0 if a rebalance is due, 1 otherwise (read-only).",
    )
    parser.add_argument(
        "--start", default=None,
        help="History start date (default: historia requerida por los "
             "componentes + margen).",
    )
    parser.add_argument(
        "--history-mode", choices=HISTORY_MODES, default=STRICT,
        help="strict (default): aborta (exit 2) si falta historia; "
             "permissive: excluye factores sin historia y lo registra.",
    )
    parser.add_argument(
        "--every-n", type=int, default=REBALANCE_EVERY_N,
        help=f"Trading days between rebalances (default {REBALANCE_EVERY_N}).",
    )
    parser.add_argument(
        "--mode", choices=MODES, default="sim",
        help="Fuente de verdad de la cartera. sim (default): ledger local "
             "simulado, sin red. broker: cuenta Alpaca paper; ledger.json pasa "
             "a espejo del broker (dry-run salvo --submit).",
    )
    parser.add_argument(
        "--broker", action="store_true",
        help="Obsoleto: equivale a --mode broker.",
    )
    send = parser.add_mutually_exclusive_group()
    send.add_argument(
        "--submit", action="store_true",
        help="Requiere --mode broker: envia ordenes market/day en paper y "
             "compromete el estado solo si salen.",
    )
    send.add_argument(
        "--dry-run", action="store_true",
        help="Todas las fases salvo execute y commit; plan en results/plans/ "
             "(no toca live_state).",
    )
    parser.add_argument(
        "--allow-closed", action="store_true",
        help="Requiere --mode broker: con --submit permite enviar con el "
             "mercado cerrado (una cuenta bloqueada siempre aborta).",
    )
    parser.add_argument(
        "--min-notional", type=float, default=1.0,
        help="Omite ordenes menores a este nocional USD (default 1.0).",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Config de estrategia TOML (default config/live.toml). Su hash "
             "queda en rebalance_history.csv.",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=None,
        help="Override: cierre si precio cae esta fraccion bajo entrada "
             "(ej. 0.01; 0=off; default: config).",
    )
    parser.add_argument(
        "--take-profit", type=float, default=None,
        help="Override: cierre si precio sube esta fraccion sobre entrada "
             "(ej. 0.02; 0=off; default: config).",
    )
    parser.add_argument(
        "--trailing-stop", type=float, default=None,
        help="Override: cierre si precio cae esta fraccion bajo maximo movil "
             "(0=off; default: config).",
    )
    parser.add_argument(
        "--trailing-window", type=int, default=None,
        help="Override: ventana del maximo movil en dias (default: config).",
    )
    parser.add_argument(
        "--take-profit-dollars", type=float, default=None,
        help="Override: cierra si PnL >= +USD (0=off; default: config).",
    )
    parser.add_argument(
        "--stop-loss-dollars", type=float, default=None,
        help="Override: cierra si PnL <= -USD (0=off; default: config).",
    )
    parser.add_argument(
        "--max-order-notional", type=float, default=DEFAULT_MAX_ORDER_NOTIONAL,
        help=f"Tope USD por orden (default {DEFAULT_MAX_ORDER_NOTIONAL:,.0f}; 0=off).",
    )
    parser.add_argument(
        "--max-total-notional", type=float, default=DEFAULT_MAX_TOTAL_NOTIONAL,
        help="Tope de nocional bruto por corrida, en x equity "
             f"(default {DEFAULT_MAX_TOTAL_NOTIONAL}; 0=off).",
    )
    parser.add_argument(
        "--max-orders", type=int, default=DEFAULT_MAX_ORDERS,
        help=f"Maximo de ordenes por corrida (default {DEFAULT_MAX_ORDERS}; 0=off).",
    )
    parser.add_argument(
        "--data-sources", type=_parse_sources, default=DEFAULT_SOURCES,
        help="Fuentes de datos separadas por coma, en orden de fallback "
             f"(default {','.join(DEFAULT_SOURCES)}).",
    )
    parser.add_argument(
        "--data-cache", action="store_true",
        help="Cache parquet por símbolo en data/cache/bars (solo descarga barras nuevas).",
    )
    parser.add_argument(
        "--on-missing", choices=MISSING_POLICIES, default=LIVE_POLICY.on_missing,
        help="Símbolo del universo sin datos: error (default, exit 7) o warn (se excluye).",
    )
    parser.add_argument(
        "--on-partial-bar", choices=PARTIAL_BAR_POLICIES,
        default=LIVE_POLICY.on_partial_bar,
        help="Barra de hoy con el mercado abierto: drop (default) o abort (exit 7).",
    )
    parser.add_argument(
        "--on-stale", choices=STALE_POLICIES, default=LIVE_POLICY.on_stale,
        help="Última barra anterior a la última sesión NYSE cerrada: "
             "error (default, exit 7) o warn.",
    )
    parser.add_argument(
        "--poll-timeout", type=float, default=POLL_TIMEOUT_S,
        help=f"Con --submit, segundos esperando fills (default {POLL_TIMEOUT_S}).",
    )
    args = parser.parse_args(argv)
    mode = "broker" if args.broker else args.mode
    if args.submit and mode != "broker":
        parser.error("--submit requires --mode broker")
    if args.allow_closed and mode != "broker":
        parser.error("--allow-closed requires --mode broker")
    setup_logging()
    if args.broker:
        logger.warning("--broker esta obsoleto: usar --mode broker.")
    # Flag ausente → valor de la config; 0 → regla off (None).
    exit_overrides = {
        field: (FROM_CONFIG if value is None else (value or None))
        for field, value in (
            ("stop_loss_pct", args.stop_loss),
            ("take_profit_pct", args.take_profit),
            ("trailing_stop_pct", args.trailing_stop),
            ("take_profit_dollars", args.take_profit_dollars),
            ("stop_loss_dollars", args.stop_loss_dollars),
        )
    }
    exit_overrides["trailing_window"] = (FROM_CONFIG if args.trailing_window is None
                                         else args.trailing_window)
    try:
        return run_once(
            force=args.force,
            start=args.start,
            every_n=args.every_n,
            check_only=args.check_only,
            mode=mode,
            submit=args.submit,
            dry_run=args.dry_run,
            allow_closed=args.allow_closed,
            min_notional=args.min_notional,
            **exit_overrides,
            config_path=args.config,
            poll_timeout=args.poll_timeout,
            history_mode=args.history_mode,
            data_sources=args.data_sources,
            data_cache=args.data_cache,
            data_policy=DataPolicy(
                on_missing=args.on_missing,
                on_partial_bar=args.on_partial_bar,
                on_stale=args.on_stale,
            ),
            guard=PreTradeGuard(
                max_notional_per_order=args.max_order_notional or None,
                max_total_notional=args.max_total_notional or None,
                max_orders=args.max_orders or None,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — no confundir un crash con "no toca" (1)
        # Traceback ya logueado (con run_id, en runner.log) y notificado por
        # run_once; aquí solo el resumen (o el error de validación previo).
        logger.error("Error inesperado (%s: %s): nada comprometido.",
                     type(exc).__name__, exc)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
