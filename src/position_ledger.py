"""
position_ledger.py — T-32: Libro de posiciones a nivel de lote (FIFO).

Traduce pesos objetivo del optimizador a cantidades de acciones reales
sin conexión a broker externo. Costeo FIFO para P&L realizado.

Métodos clave:
  - trades_from_target_weights(): pesos → deltas de shares (float),
    dimensionando sobre nav − costos estimados
  - apply_trades(): ejecuta un lote de trades con costeo FIFO de forma
    transaccional (valida todo antes; ventas antes que compras; invariante
    cash >= -CASH_TOLERANCE)
  - to_dict() / from_dict(): persistencia pública (schema_version)
  - positions(), nav(), snapshot(), transactions_df(), currency_exposure()

Moneda: todo importe (cash, precios, entry_price, PnL) está en
``base_currency``; el caller convierte antes (src.fx.to_base). Cada lote
guarda además la moneda local de su instrumento (exposición FX), sin decidir
cobertura alguna.

Usa shares flotantes (no enteros) para evitar ruido de redondeo y
preservar determinismo (R-10) en tests y backtests.

Restricciones cubiertas: R-01 (costos), R-05 (precios en t, sin look-ahead
cuando el caller pasa precios correctos), R-10 (determinismo).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("position_ledger")

from src.fx import is_currency_code
from src.settings import BASE_CURRENCY, INITIAL_CASH, SEED

# Versión del payload de to_dict(). 0 = JSON legado del runner (sin
# schema_version ni txns); 1 = sin moneda (lotes en base_currency). Ambos
# los acepta from_dict().
LEDGER_SCHEMA_VERSION = 2
# Invariante de caja: cash >= -CASH_TOLERANCE (moneda) tras cada compra y
# tras descontar costos. Solo absorbe ruido de coma flotante.
CASH_TOLERANCE = 1e-6
# Holgura para vender "todo" pese a ruido de redondeo en shares.
_SELL_TOLERANCE = 1e-8


# ===================================================================
# Errors
# ===================================================================
class MissingPriceError(ValueError):
    """Open position(s) without a finite, positive price to value them.

    Valuing them at 0 would silently collapse NAV (and make a rebalance
    buy to "refill" it); callers must fix the data or opt into stale prices.

    Attributes:
        tickers: Sorted list of the unpriced tickers.
    """

    def __init__(self, tickers, detail: str = "") -> None:
        self.tickers: List[str] = sorted(str(tk) for tk in tickers)
        msg = f"No finite price > 0 for open positions: {self.tickers}"
        super().__init__(f"{msg}. {detail}" if detail else msg)


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class Lot:
    """Single purchase lot for FIFO cost basis tracking.

    ``entry_price`` is in the ledger base currency; ``currency`` is the
    instrument's local (quote) currency, i.e. the FX exposure of the lot.
    """

    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    currency: str = BASE_CURRENCY


@dataclass
class LedgerSnapshot:
    """Point-in-time ledger state for auditability."""

    t: pd.Timestamp
    positions: pd.Series          # shares by ticker
    cash: float
    nav: float
    avg_entry: pd.Series          # average entry price by ticker
    cost_basis: pd.Series         # total cost basis by ticker
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class _TxnRecord:
    """Internal transaction log row."""

    t: pd.Timestamp
    ticker: str
    shares: float          # signed: +buy / -sell
    price: float
    proceeds: float        # signed cash impact before costs
    realized_pnl: float
    cost: float


# ===================================================================
# PositionLedger
# ===================================================================
class PositionLedger:
    """Lot-level position book with FIFO costing (no external broker).

    Args:
        initial_cash: Starting cash balance (in ``base_currency``).
        base_currency: Currency of every amount in the ledger.
    """

    def __init__(
        self,
        initial_cash: float = INITIAL_CASH,
        base_currency: str = BASE_CURRENCY,
    ) -> None:
        if not is_currency_code(base_currency):
            raise ValueError(f"Invalid base_currency={base_currency!r}.")
        self.base_currency: str = base_currency
        self.cash: float = float(initial_cash)
        self._lots: Dict[str, List[Lot]] = {}
        self.realized_pnl: float = 0.0
        self._txns: List[_TxnRecord] = []
        logger.info("PositionLedger init cash=%.2f", self.cash)

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------
    def positions(self) -> pd.Series:
        """Return current share quantities by ticker.

        Returns:
            Series of shares indexed by ticker (empty Series if flat).
        """
        data = {
            tk: sum(lot.shares for lot in lots)
            for tk, lots in self._lots.items()
            if lots and sum(lot.shares for lot in lots) > 1e-12
        }
        if not data:
            return pd.Series(dtype=float)
        return pd.Series(data, dtype=float)

    def unpriced(self, prices: pd.Series) -> List[str]:
        """Open positions without a finite price > 0 in ``prices``.

        Args:
            prices: Series of prices indexed by ticker.

        Returns:
            Sorted list of held tickers that cannot be valued.
        """
        px = self._held_prices(prices)
        return sorted(px.index[~(np.isfinite(px) & (px > 0))])

    def _held_prices(self, prices: pd.Series) -> pd.Series:
        """Prices aligned to open positions (absent/non-numeric → NaN)."""
        return pd.to_numeric(
            prices.reindex(self.positions().index), errors="coerce",
        ).astype(float)

    def _mark_prices(
        self,
        prices: pd.Series,
        stale_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Valuation price per open position; never 0/NaN.

        Held tickers without a valid price in ``prices`` fall back to
        ``stale_prices`` (logged as WARNING) when given; otherwise raise.

        Raises:
            MissingPriceError: If a held ticker has no valid price.
        """
        px = self._held_prices(prices)
        missing = self.unpriced(prices)
        if not missing:
            return px
        if stale_prices is not None:
            stale = stale_prices.reindex(missing).astype(float)
            ok = stale[np.isfinite(stale) & (stale > 0)]
            still_missing = [tk for tk in missing if tk not in ok.index]
            if still_missing:
                raise MissingPriceError(
                    still_missing, "No valid stale price either.")
            logger.warning(
                "Valuing %d position(s) with stale prices: %s",
                len(ok), ok.round(4).to_dict(),
            )
            px.loc[ok.index] = ok
            return px
        raise MissingPriceError(
            missing,
            "Refusing to value them at 0; pass stale_prices= to use the "
            "last known price.",
        )

    def nav(
        self,
        prices: pd.Series,
        stale_prices: Optional[pd.Series] = None,
    ) -> float:
        """Mark-to-market NAV = cash + Σ shares_i * price_i.

        Args:
            prices: Series of prices indexed by ticker.
            stale_prices: Optional last known prices, used (with a WARNING)
                only for held tickers lacking a valid price in ``prices``.

        Returns:
            Total NAV as float.

        Raises:
            MissingPriceError (a ValueError): If an open position has no
                finite price > 0.
        """
        pos = self.positions()
        if pos.empty:
            return float(self.cash)
        px = self._mark_prices(prices, stale_prices)
        return float(self.cash + float((pos * px).sum()))

    def _avg_entry(self) -> pd.Series:
        """Volume-weighted average entry price per ticker."""
        result: Dict[str, float] = {}
        for tk, lots in self._lots.items():
            total_shares = sum(l.shares for l in lots)
            if total_shares <= 1e-12:
                continue
            cost = sum(l.shares * l.entry_price for l in lots)
            result[tk] = cost / total_shares
        return pd.Series(result, dtype=float) if result else pd.Series(dtype=float)

    def _cost_basis(self) -> pd.Series:
        """Total cost basis (shares * entry) per ticker."""
        result: Dict[str, float] = {}
        for tk, lots in self._lots.items():
            cb = sum(l.shares * l.entry_price for l in lots)
            if abs(cb) > 1e-12:
                result[tk] = cb
        return pd.Series(result, dtype=float) if result else pd.Series(dtype=float)

    def _currency_by_ticker(self) -> Dict[str, str]:
        """Local currency of each held ticker (all its lots share it)."""
        return {tk: lots[0].currency for tk, lots in self._lots.items() if lots}

    def currency_exposure(
        self,
        prices: pd.Series,
        stale_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Market value in base currency grouped by local currency (+ cash).

        Args:
            prices: Mark-to-market prices in base currency.
            stale_prices: Optional last known prices (see ``nav``).

        Returns:
            Series currency → value in ``base_currency`` (sorted); cash counts
            as ``base_currency``. Sums to ``nav(prices)``.

        Raises:
            MissingPriceError (a ValueError): If an open position has no
                finite price > 0.
        """
        exposure: Dict[str, float] = {self.base_currency: float(self.cash)}
        pos = self.positions()
        if not pos.empty:
            values = pos * self._mark_prices(prices, stale_prices)
            ccy = self._currency_by_ticker()
            for tk, value in values.items():
                exposure[ccy[tk]] = exposure.get(ccy[tk], 0.0) + float(value)
        return pd.Series(exposure, dtype=float).sort_index()

    def snapshot(
        self,
        t: pd.Timestamp,
        prices: pd.Series,
        stale_prices: Optional[pd.Series] = None,
    ) -> LedgerSnapshot:
        """Capture ledger state at date t for audit trail.

        Args:
            t: Snapshot timestamp.
            prices: Mark-to-market prices by ticker.
            stale_prices: Optional last known prices (see ``nav``).

        Returns:
            LedgerSnapshot with positions, cash, NAV and PnL.

        Raises:
            MissingPriceError (a ValueError): If an open position has no
                finite price > 0.
        """
        pos = self.positions()
        avg = self._avg_entry()
        cb = self._cost_basis()
        nav_val = float(self.cash)
        unrealized = 0.0
        if not pos.empty:
            px = self._mark_prices(prices, stale_prices)
            mkt = (pos * px).sum()
            nav_val = float(self.cash + float(mkt))
            unrealized = float(mkt - cb.reindex(pos.index).fillna(0.0).sum())
        return LedgerSnapshot(
            t=pd.Timestamp(t),
            positions=pos.copy(),
            cash=float(self.cash),
            nav=float(nav_val),
            avg_entry=avg.copy(),
            cost_basis=cb.copy(),
            unrealized_pnl=unrealized,
            realized_pnl=float(self.realized_pnl),
        )

    def transactions_df(self) -> pd.DataFrame:
        """Return full transaction history as a DataFrame.

        Returns:
            DataFrame with columns: t, ticker, shares, price, proceeds,
            realized_pnl, cost. Empty DataFrame if no trades yet.
        """
        if not self._txns:
            return pd.DataFrame(
                columns=[
                    "t", "ticker", "shares", "price",
                    "proceeds", "realized_pnl", "cost",
                ]
            )
        rows = [
            {
                "t": r.t,
                "ticker": r.ticker,
                "shares": r.shares,
                "price": r.price,
                "proceeds": r.proceeds,
                "realized_pnl": r.realized_pnl,
                "cost": r.cost,
            }
            for r in self._txns
        ]
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full ledger state to a JSON-compatible dict.

        Returns:
            Dict with schema_version, base_currency, cash, realized_pnl,
            lots (FIFO order per ticker, with local currency) and txns.
            Timestamps are ISO strings (NaT → None).
        """
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "base_currency": self.base_currency,
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "lots": [
                {
                    "ticker": lot.ticker,
                    "shares": float(lot.shares),
                    "entry_price": float(lot.entry_price),
                    "entry_date": _ts_to_iso(lot.entry_date),
                    "currency": lot.currency,
                }
                for lots in self._lots.values()
                for lot in lots
            ],
            "txns": [
                {**asdict(r), "t": _ts_to_iso(r.t)} for r in self._txns
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PositionLedger":
        """Rebuild a ledger from ``to_dict()`` output (or legacy runner JSON).

        Legacy payloads (no schema_version, no txns) load with an empty
        transaction log; payloads without currencies (v0/v1) are in
        ``BASE_CURRENCY`` with every lot in it. Cash is restored as-is, even
        if negative (legacy state): the invariant is enforced on the next
        ``apply_trades``.

        Args:
            payload: Dict produced by ``to_dict()``.

        Returns:
            New, independent PositionLedger.

        Raises:
            ValueError: If schema_version is not supported.
        """
        version = payload.get("schema_version", 0)
        if version not in range(LEDGER_SCHEMA_VERSION + 1):
            raise ValueError(
                f"Unsupported ledger schema_version={version!r} "
                f"(supported: 0..{LEDGER_SCHEMA_VERSION})."
            )
        ledger = cls(initial_cash=float(payload["cash"]),
                     base_currency=payload.get("base_currency", BASE_CURRENCY))
        ledger.realized_pnl = float(payload.get("realized_pnl", 0.0))
        for row in payload.get("lots", []):
            lot = Lot(
                ticker=str(row["ticker"]),
                shares=float(row["shares"]),
                entry_price=float(row["entry_price"]),
                entry_date=_iso_to_ts(row.get("entry_date")),
                currency=str(row.get("currency", ledger.base_currency)),
            )
            ledger._lots.setdefault(lot.ticker, []).append(lot)
        ledger._txns = [
            _TxnRecord(
                t=_iso_to_ts(r.get("t")),
                ticker=str(r["ticker"]),
                shares=float(r["shares"]),
                price=float(r["price"]),
                proceeds=float(r["proceeds"]),
                realized_pnl=float(r["realized_pnl"]),
                cost=float(r["cost"]),
            )
            for r in payload.get("txns", [])
        ]
        if ledger.cash < -CASH_TOLERANCE:
            logger.warning(
                "Loaded ledger with negative cash=%.2f; next rebalance "
                "(sized on nav − costs) restores cash >= 0.", ledger.cash,
            )
        return ledger

    # -----------------------------------------------------------------
    # Weight → trade translation
    # -----------------------------------------------------------------
    def trades_from_target_weights(
        self,
        target_weights: pd.Series,
        prices_at_t: pd.Series,
        nav: Optional[float] = None,
        costs: float = 0.0,
        return_unpriced: bool = False,
    ):
        """Translate target portfolio weights into share deltas.

        Uses floating-point shares (not integers) for determinism (R-10).

        Args:
            target_weights: Desired weights (should sum ≈ 1). Index = tickers.
            prices_at_t: Prices at rebalance time for valuation.
            nav: Portfolio NAV to size against. None → self.nav(prices_at_t).
            costs: Estimated transaction costs of this rebalance (R-01).
                They are reserved: weights are sized on ``nav - costs`` so
                paying them in ``apply_trades`` leaves cash >= 0.
            return_unpriced: False (default) → raise if an open position has
                no valid price. True → return ``(trades, unpriced)`` and let
                the caller decide; unpriced positions get no trade.

        Returns:
            DataFrame with columns [ticker, shares] where shares is the
            signed delta (positive = buy, negative = sell). Only non-zero
            trades are included. Target names without a usable price are
            skipped. With ``return_unpriced=True``, a tuple
            ``(trades, unpriced_tickers)``.

        Raises:
            MissingPriceError (a ValueError): If an open position has no
                finite price > 0 and ``return_unpriced`` is False, or if
                ``nav`` is None (NAV needs every position priced).
        """
        unpriced = self.unpriced(prices_at_t)
        if unpriced and not return_unpriced:
            raise MissingPriceError(
                unpriced, "Cannot sell or resize them; fix prices or pass "
                "return_unpriced=True to handle them explicitly.")
        if nav is None:
            nav = self.nav(prices_at_t)
        nav = nav - float(costs)

        tw = target_weights.fillna(0.0).astype(float)
        # Include current holdings that may need to be exited
        current = self.positions()
        all_tickers = sorted(set(tw.index) | set(current.index))

        rows: List[dict] = []
        for tk in all_tickers:
            # Unpriced holdings go back to the caller; target-only names
            # without a price are simply not bought.
            if tk in unpriced or tk not in prices_at_t.index:
                continue
            curr_shares = float(current[tk]) if tk in current.index else 0.0
            px = float(prices_at_t[tk])
            if not np.isfinite(px) or px <= 0:
                continue
            w_tgt = float(tw[tk]) if tk in tw.index else 0.0
            target_shares = (w_tgt * nav) / px
            delta = target_shares - curr_shares
            if abs(delta) > 1e-10:
                rows.append({"ticker": tk, "shares": delta})

        trades = (pd.DataFrame(rows) if rows
                  else pd.DataFrame(columns=["ticker", "shares"]))
        if return_unpriced:
            return trades, unpriced
        return trades

    # -----------------------------------------------------------------
    # Trade application (FIFO, transactional)
    # -----------------------------------------------------------------
    def apply_trades(
        self,
        trades: pd.DataFrame,
        prices_at_t: pd.Series,
        costs: float = 0.0,
        t: Optional[pd.Timestamp] = None,
        currencies: Optional[Mapping[str, str]] = None,
    ) -> float:
        """Apply a batch of trades atomically with FIFO lot costing.

        The whole batch is validated before any state changes: every
        non-zero trade needs finite shares and a finite positive price,
        total sells per ticker cannot exceed current holdings, and costs
        must be finite and >= 0. Sells execute before buys (stable order
        within each group) so buys can be funded by same-batch proceeds.
        Cash must stay >= -CASH_TOLERANCE after every buy and after costs
        (R-01). On any error the ledger is left exactly as it was.

        Args:
            trades: DataFrame with columns [ticker, shares] (signed deltas).
            prices_at_t: Execution prices by ticker, in base currency.
            costs: Total transaction cost in base currency units (R-01).
            t: Trade timestamp for the log. Defaults to NaT.
            currencies: Ticker → local currency recorded on new lots
                (e.g. ``InstrumentMaster.currencies()``). Absent tickers keep
                the currency of their open lots, else ``base_currency``.

        Returns:
            Realized PnL generated by this batch of trades.

        Raises:
            ValueError: Invalid shares/price/costs/currency, overselling, a
                buy whose currency differs from the ticker's open lots, or a
                batch that would break the cash invariant. Nothing is mutated.
        """
        t = pd.NaT if t is None else pd.Timestamp(t)
        costs = float(costs)
        if not np.isfinite(costs) or costs < 0:
            raise ValueError(f"Invalid costs={costs!r}: must be finite and >= 0.")
        plan = self._plan_batch(trades, prices_at_t, currencies)

        # Stage on copies; commit only once the whole batch succeeded.
        lots = {tk: [replace(lot) for lot in book] for tk, book in self._lots.items()}
        cash = self.cash
        realized = self.realized_pnl
        txns: List[_TxnRecord] = []
        batch_realized = 0.0
        for tk, delta, px, ccy in plan:
            if delta > 0:
                # Buy: new lot, spend cash
                lots.setdefault(tk, []).append(
                    Lot(ticker=tk, shares=delta, entry_price=px, entry_date=t,
                        currency=ccy)
                )
                cash_impact = -(delta * px)
                cash += cash_impact
                _check_cash(cash, f"buy {tk}")
                txns.append(_TxnRecord(
                    t=t, ticker=tk, shares=delta, price=px,
                    proceeds=cash_impact, realized_pnl=0.0, cost=0.0,
                ))
            else:
                # Sell: FIFO consume lots
                sell_shares = -delta
                pnl = _fifo_sell(lots, tk, sell_shares, px)
                batch_realized += pnl
                realized += pnl
                cash_impact = sell_shares * px
                cash += cash_impact
                txns.append(_TxnRecord(
                    t=t, ticker=tk, shares=delta, price=px,
                    proceeds=cash_impact, realized_pnl=pnl, cost=0.0,
                ))

        if costs > 0:
            cash -= costs
            _check_cash(cash, "costs")
            txns.append(_TxnRecord(
                t=t, ticker="__COST__", shares=0.0, price=0.0,
                proceeds=-costs, realized_pnl=0.0, cost=costs,
            ))

        self._lots = lots
        self.cash = cash
        self.realized_pnl = realized
        self._txns.extend(txns)
        logger.debug(
            "apply_trades: %d trades, realized_pnl=%.4f, cash=%.2f",
            len(plan), batch_realized, self.cash,
        )
        return batch_realized

    def _plan_batch(
        self,
        trades: Optional[pd.DataFrame],
        prices_at_t: pd.Series,
        currencies: Optional[Mapping[str, str]] = None,
    ) -> List[Tuple[str, float, float, str]]:
        """Validate a batch → (ticker, delta, price, lot currency), sells first.

        Raises:
            ValueError: Non-finite shares, missing/invalid price, invalid or
                conflicting currency, or total sells of a ticker above
                current holdings.
        """
        if trades is None or len(trades) == 0:
            return []
        sells: List[Tuple[str, float, float, str]] = []
        buys: List[Tuple[str, float, float, str]] = []
        sell_totals: Dict[str, float] = {}
        held_ccy = self._currency_by_ticker()
        for _, row in trades.iterrows():
            tk = str(row["ticker"])
            delta = float(row["shares"])
            if not np.isfinite(delta):
                raise ValueError(f"Invalid shares for {tk}: {delta!r}.")
            if abs(delta) < 1e-12:
                continue
            px = float(prices_at_t[tk]) if tk in prices_at_t.index else float("nan")
            if not np.isfinite(px) or px <= 0:
                raise ValueError(f"Missing or invalid price for {tk}: {px!r}.")
            ccy = held_ccy.get(tk, self.base_currency)
            if delta > 0:
                ccy = self._buy_currency(tk, held_ccy.get(tk), currencies)
                buys.append((tk, delta, px, ccy))
            else:
                sells.append((tk, delta, px, ccy))
                sell_totals[tk] = sell_totals.get(tk, 0.0) - delta

        held = self.positions()
        for tk, qty in sell_totals.items():
            have = float(held.get(tk, 0.0))
            if qty - have > _SELL_TOLERANCE:
                raise ValueError(
                    f"Cannot sell {qty} of {tk}: "
                    f"insufficient shares (short by {qty - have:.6f})."
                )
        return sells + buys

    def _buy_currency(
        self,
        ticker: str,
        held: Optional[str],
        currencies: Optional[Mapping[str, str]],
    ) -> str:
        """Currency of a new lot: mapping > open lots > base_currency."""
        wanted = (currencies or {}).get(ticker)
        if wanted is None:
            return held or self.base_currency
        if not is_currency_code(wanted):
            raise ValueError(f"Invalid currency {wanted!r} for {ticker}.")
        if held is not None and wanted != held:
            raise ValueError(
                f"{ticker} is held in {held}; cannot add a lot in {wanted}."
            )
        return wanted

    def weights_from_positions(
        self,
        prices: pd.Series,
        stale_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Convert current share holdings to portfolio weights.

        Args:
            prices: Mark-to-market prices.
            stale_prices: Optional last known prices (see ``nav``).

        Returns:
            Weight Series summing to ≤ 1 (cash is the residual).
            If NAV ≈ 0, returns empty Series.

        Raises:
            MissingPriceError (a ValueError): If an open position has no
                finite price > 0.
        """
        pos = self.positions()
        if pos.empty:
            return pd.Series(dtype=float)
        px = self._mark_prices(prices, stale_prices)
        values = pos * px
        nav_val = float(self.cash + float(values.sum()))
        if nav_val < 1e-10:
            return pd.Series(dtype=float)
        return (values / nav_val).astype(float)


# ===================================================================
# Helpers
# ===================================================================
def _fifo_sell(
    book: Dict[str, List[Lot]],
    ticker: str,
    shares: float,
    sell_price: float,
) -> float:
    """Consume lots of ``book`` FIFO (mutates it); return realized PnL.

    Raises:
        ValueError: If trying to sell more shares than held.
    """
    lots = book.get(ticker, [])
    remaining = shares
    pnl = 0.0
    while remaining > 1e-12 and lots:
        lot = lots[0]
        take = min(lot.shares, remaining)
        pnl += take * (sell_price - lot.entry_price)
        lot.shares -= take
        remaining -= take
        if lot.shares <= 1e-12:
            lots.pop(0)

    if remaining > _SELL_TOLERANCE:
        raise ValueError(
            f"Cannot sell {shares} of {ticker}: "
            f"insufficient shares (short by {remaining:.6f})."
        )
    # Clean empty
    if not lots:
        book.pop(ticker, None)
    return float(pnl)


def _check_cash(cash: float, after: str) -> None:
    """Enforce the cash invariant cash >= -CASH_TOLERANCE."""
    if cash < -CASH_TOLERANCE:
        raise ValueError(
            f"Batch would leave cash={cash:.6f} < -{CASH_TOLERANCE} "
            f"after {after}; size trades on nav - costs."
        )


def _ts_to_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    return None if pd.isna(ts) else pd.Timestamp(ts).isoformat()


def _iso_to_ts(value: Optional[str]) -> pd.Timestamp:
    return pd.Timestamp(value) if value else pd.NaT
