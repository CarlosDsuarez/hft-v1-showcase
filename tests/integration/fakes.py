"""
fakes.py — Dobles sin red del runner vivo (ARQ-18).

FakeMarket: barras sintéticas deterministas detrás del contrato
``MarketDataProvider`` (reloj de la corrida = tras el cierre de la última
barra visible), con fallos inyectables por descarga.

FakeBroker: cuenta Alpaca paper en memoria con la interfaz que usa el runner
(check, get_account, get_positions, submit_orders, poll_orders): libro con
estado, registro de órdenes y fallos inyectables. Las órdenes se llenan al
instante al cierre de la última barra (market order, sin comisiones), así la
corrida siguiente ve la cartera real tras los fills.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.brokers.alpaca_adapter import BrokerError
from src.market_data import FakeProvider, nyse_session

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
OUTSIDER = "ZZZ"         # con datos en la fuente, fuera del universo
START = "2024-01-01"
N_ROWS = 300             # >= max required_history del stack (momentum 12-1 = 274)
EXTRA_ROWS = 40          # sesiones futuras que `advance` va revelando
BROKER_EQUITY = 10_000.0


def after_close(day) -> datetime:
    """Reloj UTC a las 17:00 de Nueva York de `day` (sesión ya cerrada)."""
    local = pd.Timestamp(f"{pd.Timestamp(day).date()} 17:00", tz="America/New_York")
    return local.tz_convert("UTC").to_pydatetime()


class FakeMarket:
    """Proveedor de barras sintéticas deterministas con fallos inyectables.

    Solo las primeras `upto` sesiones son visibles; `advance(n)` revela n
    más (el tiempo avanza). Inyección por descarga:
      fail      excepción que lanza cada `get_bars` (fuente caída)
      drop      símbolos que la fuente deja de traer
      nan_tail  símbolo → n últimas barras sin precio (deslistado/suspendido)
    """

    name = "fake-market"

    def __init__(self, tickers=(*TICKERS, OUTSIDER), n_rows: int = N_ROWS,
                 extra_rows: int = EXTRA_ROWS, seed: int = 7):
        # Sesiones NYSE reales: ningún feriado rompe el chequeo de frescura
        idx = pd.date_range(START, periods=n_rows + extra_rows, freq=nyse_session())
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0005, 0.01, (len(idx), len(tickers)))
        self.full = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)),
                                 index=idx, columns=list(tickers))
        self.upto = n_rows
        self.fail: Optional[BaseException] = None
        self.drop: set = set()
        self.nan_tail: Dict[str, int] = {}
        self.requests: List[Tuple[str, ...]] = []

    @property
    def t(self) -> pd.Timestamp:
        return self.full.index[self.upto - 1]

    @property
    def close(self) -> pd.DataFrame:
        frame = self.full.iloc[:self.upto].drop(columns=sorted(self.drop))
        for symbol, n in self.nan_tail.items():
            frame.iloc[-n:, frame.columns.get_loc(symbol)] = np.nan
        return frame

    def marks(self) -> pd.Series:
        """Precio de marca del broker en t (independiente de los fallos)."""
        return self.full.iloc[self.upto - 1]

    def advance(self, sessions: int) -> None:
        if self.upto + sessions > len(self.full):
            raise ValueError("FakeMarket sin sesiones futuras suficientes")
        self.upto += sessions

    def now(self) -> datetime:
        return after_close(self.t)

    def get_bars(self, symbols, start, end):
        self.requests.append(tuple(symbols))
        if self.fail is not None:
            raise self.fail
        close = self.close
        volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
        return FakeProvider(close, volume, name=self.name).get_bars(symbols, start, end)


class FakeBroker:
    """Cuenta paper en memoria: posiciones, equity, órdenes y fallos inyectables.

    client_order_id es único como en Alpaca: reenviar el mismo id no crea
    otra orden (el adaptador real resuelve el 422 a la orden existente y la
    reporta "submitted").

    Fallos inyectables:
      fail_check  excepción en check() (cuenta equivocada, red, 401)
      fail_after  tras N órdenes nuevas en un envío, la siguiente recibe 403
                  y el lote se aborta (BrokerError), como AlpacaClient
      fail_poll   excepción al consultar fills (órdenes ya enviadas)
      reject      tickers rechazados por orden (4xx, el lote sigue)
    """

    def __init__(self, market: FakeMarket, cash: float = BROKER_EQUITY,
                 positions: Optional[Dict[str, Tuple[float, float]]] = None,
                 market_open: bool = True, account_blocked: bool = False):
        self.market = market
        self.cash = float(cash)
        self.book: Dict[str, List[float]] = {
            tk: [float(q), float(e)] for tk, (q, e) in (positions or {}).items()}
        self.market_open = market_open
        self.account_blocked = account_blocked
        self.fail_check: Optional[BaseException] = None
        self.fail_after: Optional[int] = None
        self.fail_poll: Optional[BaseException] = None
        self.reject: set = set()
        self.orders: List[Dict] = []          # órdenes que llegaron (una por id)
        self.dry_run_plans: List[pd.DataFrame] = []
        self.calls: List[str] = []

    # -- lectura --------------------------------------------------------
    def positions(self) -> pd.Series:
        return pd.Series({tk: q for tk, (q, _) in self.book.items()}, dtype=float)

    @property
    def equity(self) -> float:
        px = self.market.marks()
        return self.cash + sum(q * float(px[tk]) for tk, (q, _) in self.book.items())

    def check(self) -> Dict:
        self.calls.append("check")
        if self.fail_check is not None:
            raise self.fail_check
        return {"account_number": "PA-FAKE", "equity": str(self.equity),
                "account_blocked": self.account_blocked, "trading_blocked": False,
                "market_open": self.market_open}

    def get_account(self) -> Dict:
        self.calls.append("get_account")
        return {"account_number": "PA-FAKE", "equity": str(self.equity),
                "cash": str(self.cash)}

    def get_positions(self) -> pd.DataFrame:
        self.calls.append("get_positions")
        px = self.market.marks()
        return pd.DataFrame(
            [{"ticker": tk, "qty": q, "avg_entry": e, "side": "long",
              "market_value": q * float(px[tk])}
             for tk, (q, e) in self.book.items()],
            columns=["ticker", "qty", "avg_entry", "side", "market_value"])

    # -- envío ----------------------------------------------------------
    def _fill(self, ticker: str, side: str, qty: float) -> float:
        px = float(self.market.marks()[ticker])
        held, entry = self.book.get(ticker, [0.0, 0.0])
        if side == "buy":
            new = held + qty
            entry = (held * entry + qty * px) / new
            self.cash -= qty * px
        else:
            if qty > held + 1e-9:
                raise AssertionError(f"plan vende {qty} {ticker} con {held} en cartera")
            new = held - qty
            self.cash += qty * px
        if new > 1e-9:
            self.book[ticker] = [new, entry]
        else:
            self.book.pop(ticker, None)
        return px

    def submit_orders(self, plan, dry_run=True, run_id=None, on_result=None):
        self.calls.append("submit_orders")
        results: List[Dict] = []

        def emit(rec: Dict) -> None:
            results.append(rec)
            if on_result is not None:
                on_result(rec)

        if dry_run:
            self.dry_run_plans.append(plan.copy())
        by_cid = {o["client_order_id"]: o for o in self.orders}
        sent_now = 0
        for r in plan.itertuples():
            cid = f"{run_id}-{r.ticker}-{r.side}"
            base = {"ticker": r.ticker, "side": r.side, "qty": r.qty,
                    "client_order_id": cid, "broker_status": None, "error": None}
            if dry_run:
                emit({**base, "status": "dry_run", "id": None})
            elif cid in by_cid:  # reenvío del mismo id: no hay orden nueva
                emit({**base, "status": "submitted", "id": by_cid[cid]["id"],
                      "broker_status": "duplicate"})
            elif self.fail_after is not None and sent_now >= self.fail_after:
                emit({**base, "status": "failed", "id": None, "error": "403"})
                raise BrokerError("forbidden", status=403)
            elif r.ticker in self.reject:
                emit({**base, "status": "rejected", "id": None, "error": "422"})
            else:
                px = self._fill(r.ticker, r.side, float(r.qty))
                order = {"id": f"ord-{len(self.orders) + 1}", "client_order_id": cid,
                         "ticker": r.ticker, "side": r.side, "qty": float(r.qty),
                         "price": px}
                self.orders.append(order)
                by_cid[cid] = order
                sent_now += 1
                emit({**base, "status": "submitted", "id": order["id"],
                      "broker_status": "accepted"})
        return results

    def poll_orders(self, ids, timeout=30.0):
        self.calls.append("poll_orders")
        if self.fail_poll is not None:
            raise self.fail_poll
        by_id = {o["id"]: o for o in self.orders}
        return {oid: {"final_status": "filled", "filled_qty": by_id[oid]["qty"],
                      "filled_avg_price": by_id[oid]["price"]}
                for oid in ids if oid in by_id}
