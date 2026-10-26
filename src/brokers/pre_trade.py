"""
pre_trade.py — Guardas pre-trade explicitas (sin red, sin estado global).

- `PreTradeGuard`: topes de nocional por orden (USD) y por corrida
  (fraccion de equity, bruto compras+ventas), numero maximo de ordenes y
  bloqueos por cuenta/mercado. Se evalua sobre el plan COMPLETO antes del
  primer POST: si hay motivos, no se envia nada.
- Intradia (HFT-08): topes opcionales de tasa, ordenes abiertas y
  exposicion post-trade. Vienen en None, asi que el runner diario no
  cambia. Quien los enciende es el RiskManager, con los numeros de
  config/hft.toml.
- Kill-switch: `state_dir/KILL` o env `HFT_KILL=1` bloquea cualquier envio
  (tambien a mitad de lote: se comprueba antes de cada POST).

Uso:
    from src.brokers.pre_trade import PreTradeGuard, kill_switch_reason
    reasons = PreTradeGuard().evaluate(plan, client.check())
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

KILL_FILE = "KILL"
KILL_ENV_VAR = "HFT_KILL"
_TRUTHY = {"1", "true", "yes", "on"}

# Defaults conservadores para la cuenta paper (ajustables por CLI):
DEFAULT_MAX_ORDER_NOTIONAL = 25_000.0  # USD por orden
DEFAULT_MAX_TOTAL_NOTIONAL = 1.5       # x equity (primera compra desde caja = 1.0x)
DEFAULT_MAX_ORDERS = 150               # universo USA = 100 tickers


class PreTradeBlocked(RuntimeError):
    """Envio bloqueado por guarda pre-trade o kill-switch (nada enviado)."""

    def __init__(self, reasons: List[str]):
        self.reasons = list(reasons)
        super().__init__("Envio bloqueado pre-trade: " + "; ".join(self.reasons))


def kill_switch_reason(
    state_dir: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Motivo si el kill-switch esta activo (archivo KILL o HFT_KILL), si no None."""
    env = os.environ if env is None else env
    value = (env.get(KILL_ENV_VAR) or "").strip()
    if value.lower() in _TRUTHY:
        return f"KILL switch activo: {KILL_ENV_VAR}={value}"
    path = Path(state_dir) / KILL_FILE
    if path.exists():
        return f"KILL switch activo: existe {path}"
    return None


def _list(items: List[str], limit: int = 5) -> str:
    extra = f" (+{len(items) - limit} mas)" if len(items) > limit else ""
    return ", ".join(items[:limit]) + extra


@dataclass(frozen=True)
class PreTradeGuard:
    """Topes pre-trade. `None` desactiva un tope numerico.

    Los topes de intradía salen en None a propósito. El runner
    de cartera sigue usando los tres defaults de arriba y no ve la tasa
    por minuto, el conteo del día, las órdenes abiertas ni la exposición
    post-trade. Esos números los pone `src.hft.risk` desde `hft.toml`.
    """

    max_notional_per_order: Optional[float] = DEFAULT_MAX_ORDER_NOTIONAL
    max_total_notional: Optional[float] = DEFAULT_MAX_TOTAL_NOTIONAL
    max_orders: Optional[int] = DEFAULT_MAX_ORDERS
    block_if_account_blocked: bool = True
    block_if_market_closed: bool = False
    max_orders_per_minute: Optional[int] = None
    max_orders_per_day: Optional[int] = None
    max_open_orders: Optional[int] = None
    max_gross_exposure: Optional[float] = None
    max_net_exposure: Optional[float] = None

    def evaluate(self, plan: pd.DataFrame, summary: Dict) -> List[str]:
        """Motivos de bloqueo del plan completo ([] = se puede enviar).

        Args:
            plan: DataFrame [ticker, side, notional] (salida de reconcile_trades).
            summary: `AlpacaClient.check()` (equity, account_blocked,
                trading_blocked, market_open).
        """
        if "notional" not in plan.columns:
            raise ValueError("plan sin columna 'notional'")
        reasons: List[str] = []
        if self.block_if_account_blocked:
            for flag in ("account_blocked", "trading_blocked"):
                if summary.get(flag):
                    reasons.append(f"{flag}: el broker reporta la cuenta bloqueada")
        if self.block_if_market_closed and summary.get("market_open") is False:
            reasons.append("mercado cerrado (block_if_market_closed)")
        notional = plan["notional"].astype(float).abs()
        if self.max_orders is not None and len(plan) > self.max_orders:
            reasons.append(f"max_orders: {len(plan)} ordenes > {self.max_orders}")
        if self.max_notional_per_order is not None:
            over = plan[notional > self.max_notional_per_order]
            if not over.empty:
                reasons.append(
                    f"max_notional_per_order ${self.max_notional_per_order:,.2f}: "
                    + _list([f"{r.ticker} {r.side} ${abs(float(r.notional)):,.2f}"
                             for r in over.itertuples()]))
        if self.max_total_notional is not None and len(plan):
            equity = float(summary.get("equity") or 0.0)
            total = float(notional.sum())
            cap = self.max_total_notional * equity
            if equity <= 0 or total > cap:
                ratio = f"{total / equity:.2f}x" if equity > 0 else "equity<=0"
                reasons.append(
                    f"max_total_notional: ${total:,.2f} ({ratio} equity) > "
                    f"{self.max_total_notional:.2f}x equity")
        _rate(reasons, self.max_orders_per_minute, "max_orders_per_minute",
              "orders_this_minute", plan, summary)
        _rate(reasons, self.max_orders_per_day, "max_orders_per_day",
              "orders_today", plan, summary)
        _rate(reasons, self.max_open_orders, "max_open_orders",
              "open_orders", plan, summary)
        _exposure(reasons, self.max_gross_exposure, "max_gross_exposure",
                  "projected_gross_notional", plan, summary)
        _exposure(reasons, self.max_net_exposure, "max_net_exposure",
                  "projected_net_notional", plan, summary)
        return reasons


def _summary_count(summary: Mapping, key: str) -> int:
    value = summary.get(key, 0)
    if value is None or value == "":
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary[{key}] debe ser numerico")
    return int(value)


def _rate(
    reasons: List[str],
    limit: Optional[int],
    name: str,
    key: str,
    plan: pd.DataFrame,
    summary: Mapping,
) -> None:
    """Suma lo ya enviado (`summary[key]`) y el plan. `None` no mira."""
    if limit is None or not len(plan):
        return
    total = _summary_count(summary, key) + len(plan)
    if total > limit:
        reasons.append(f"{name}: {total} ordenes > {limit}")


def _exposure(
    reasons: List[str],
    limit: Optional[float],
    name: str,
    key: str,
    plan: pd.DataFrame,
    summary: Mapping,
) -> None:
    """Exposicion post-trade ya calculada por el llamador.

    Si la clave no viene en el summary, no se inventa: el runner diario
    no la manda y este tope queda en silencio.
    """
    if limit is None or not len(plan) or key not in summary:
        return
    equity = float(summary.get("equity") or 0.0)
    projected = float(summary[key])
    cap = limit * equity
    if equity <= 0 or projected > cap + 1e-6:
        reasons.append(f"{name}: ${projected:,.2f} > {limit:.2f}x equity")
