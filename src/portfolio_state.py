"""
portfolio_state.py — Vista unica de la cartera para el runner vivo.

`PortfolioState` (positions, avg_entry, cash, equity, as_of, source) es lo
unico que consumen las salidas y el dimensionamiento (`src.strategy.build_targets`).
Adaptadores:

  - `PortfolioState.from_ledger(ledger, prices)`: libro local simulado
    (PositionLedger), marcado a `prices`. source="ledger".
  - `PortfolioState.from_broker(client)`: cuenta real via
    `client.get_account()` + `client.get_positions()` (solo lectura).
    source="broker". No valida la cuenta: llamar antes a `client.check()`.
  - `PortfolioState.from_weights(w, prices, equity)`: libro sintético del
    backtest en espacio de pesos. source="weights".

`to_ledger()` construye el espejo local de un estado (un lote por ticker a
avg_entry, sin historial) y `position_drift()` compara dos libros.

Sin red ni imports del adaptador: el cliente se recibe por duck typing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.position_ledger import LEDGER_SCHEMA_VERSION, PositionLedger

logger = logging.getLogger("portfolio_state")

# Diferencia minima de acciones que cuenta como drift (ruido de coma flotante).
DRIFT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioState:
    """Cartera en un instante, venga del ledger simulado o del broker.

    Attributes:
        positions: Acciones por ticker (con signo; largo > 0).
        avg_entry: Precio medio de entrada por ticker.
        cash: Caja en moneda base.
        equity: cash + valor de mercado de las posiciones.
        as_of: Momento de la lectura/marca.
        source: "ledger" o "broker".
    """

    positions: pd.Series
    avg_entry: pd.Series
    cash: float
    equity: float
    as_of: pd.Timestamp
    source: str

    # -----------------------------------------------------------------
    # Adaptadores
    # -----------------------------------------------------------------
    @classmethod
    def from_ledger(
        cls,
        ledger: PositionLedger,
        prices: pd.Series,
        as_of: Optional[pd.Timestamp] = None,
    ) -> "PortfolioState":
        """Estado del libro local marcado a `prices` (equity = NAV del ledger).

        Raises:
            ValueError: Un ticker en cartera no tiene precio valido.
        """
        as_of = _as_of(as_of)
        snap = ledger.snapshot(as_of, prices)
        return cls(positions=snap.positions, avg_entry=snap.avg_entry,
                   cash=float(snap.cash), equity=float(snap.nav),
                   as_of=as_of, source="ledger")

    @classmethod
    def from_broker(
        cls,
        client: Any,
        as_of: Optional[pd.Timestamp] = None,
    ) -> "PortfolioState":
        """Estado real del broker: GET cuenta + posiciones (solo lectura).

        Cortos (`side == "short"`) quedan con acciones negativas aunque el
        payload traiga `qty` sin signo. Sin `cash` en la cuenta se usa
        equity − Σ market_value.
        """
        account = client.get_account()
        pos = client.get_positions()
        equity = float(account.get("equity") or 0.0)
        if pos is None or pos.empty:
            qty = pd.Series(dtype=float)
            entry = pd.Series(dtype=float)
            market_value = 0.0
        else:
            by_ticker = pos.set_index("ticker")
            qty = by_ticker["qty"].astype(float).abs().rename_axis(None)
            if "side" in by_ticker.columns:
                short = (by_ticker["side"].astype(str).str.lower() == "short").to_numpy()
                qty[short] = -qty[short]
            entry = by_ticker["avg_entry"].astype(float).rename_axis(None)
            market_value = float(by_ticker["market_value"].astype(float).sum())
        raw_cash = account.get("cash")
        cash = (float(raw_cash) if raw_cash not in (None, "")
                else equity - market_value)
        return cls(positions=qty, avg_entry=entry, cash=cash, equity=equity,
                   as_of=_as_of(as_of), source="broker")

    @classmethod
    def from_weights(
        cls,
        weights: Optional[pd.Series],
        prices: pd.Series,
        equity: float,
        avg_entry: Optional[Dict[str, float]] = None,
        as_of: Optional[pd.Timestamp] = None,
        min_weight: float = 1e-9,
    ) -> "PortfolioState":
        """Libro sintético del backtest en espacio de pesos (sin acciones reales).

        positions = w × equity / precio para w > ``min_weight`` con precio
        finito > 0; cash = equity − valor invertido. source="weights".
        """
        w = (pd.Series(dtype=float) if weights is None
             else weights.astype(float))
        px = pd.to_numeric(prices.reindex(w.index), errors="coerce")
        ok = (w > min_weight) & np.isfinite(px) & (px > 0)
        qty = (w[ok] * float(equity) / px[ok]).astype(float)
        entries = pd.Series(avg_entry or {}, dtype=float).reindex(qty.index)
        invested = float((w[ok] * float(equity)).sum())
        return cls(positions=qty, avg_entry=entries,
                   cash=float(equity) - invested, equity=float(equity),
                   as_of=_as_of(as_of), source="weights")

    # -----------------------------------------------------------------
    # Consumidores
    # -----------------------------------------------------------------
    def exit_inputs(self) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        """(held, entries, shares) para `exit_rules.evaluate_exits`.

        held: tickers largos; entries: avg_entry > 0 de esos tickers;
        shares: acciones de esos tickers.
        """
        longs = self.positions[self.positions > 0]
        held = longs.index.tolist()
        entries = {tk: float(px) for tk, px in self.avg_entry.items()
                   if tk in longs.index and float(px) > 0}
        shares = {tk: float(sh) for tk, sh in longs.items()}
        return held, entries, shares

    def weights(self, prices: pd.Series) -> pd.Series:
        """Valor de mercado / equity por ticker (tickers sin precio se omiten)."""
        pos = self.positions[self.positions != 0]
        if pos.empty or self.equity < 1e-10:
            return pd.Series(dtype=float)
        px = pd.to_numeric(prices.reindex(pos.index), errors="coerce")
        return ((pos * px).dropna() / self.equity).astype(float)

    def unrealized_pnl(self, prices: pd.Series) -> float:
        """Σ (precio − avg_entry) × acciones (tickers sin precio se omiten)."""
        px = pd.to_numeric(prices.reindex(self.positions.index), errors="coerce")
        entry = self.avg_entry.reindex(self.positions.index)
        return float(((px - entry) * self.positions).dropna().sum())

    def to_ledger(self) -> PositionLedger:
        """Espejo local: un lote por ticker largo a avg_entry, sin historial.

        Posiciones cortas no son representables en el ledger (solo largo):
        se omiten con warning.
        """
        shorts = self.positions[self.positions < 0]
        if not shorts.empty:
            logger.warning("Posiciones cortas omitidas del espejo: %s",
                           shorts.index.tolist())
        entry_date = None if pd.isna(self.as_of) else self.as_of.isoformat()
        lots = [
            {"ticker": str(tk), "shares": float(sh),
             "entry_price": float(self.avg_entry.get(tk, 0.0)),
             "entry_date": entry_date}
            for tk, sh in self.positions.items() if sh > 0
        ]
        return PositionLedger.from_dict({
            "schema_version": LEDGER_SCHEMA_VERSION,
            "cash": float(self.cash),
            "realized_pnl": 0.0,
            "lots": lots,
            "txns": [],
        })


def position_drift(
    local: PortfolioState,
    broker: PortfolioState,
    tol: float = DRIFT_TOLERANCE,
) -> pd.DataFrame:
    """Tickers cuyas acciones difieren entre el libro local y el broker.

    Returns:
        DataFrame [ticker, local_shares, broker_shares, diff] con
        diff = broker − local; vacio si coinciden (dentro de `tol`).
    """
    tickers = sorted(set(local.positions.index) | set(broker.positions.index))
    rows = []
    for tk in tickers:
        a = float(local.positions.get(tk, 0.0))
        b = float(broker.positions.get(tk, 0.0))
        if abs(b - a) > tol:
            rows.append({"ticker": tk, "local_shares": a,
                         "broker_shares": b, "diff": b - a})
    return pd.DataFrame(rows, columns=["ticker", "local_shares",
                                       "broker_shares", "diff"])


def _as_of(value: Optional[pd.Timestamp]) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
