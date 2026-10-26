"""
live_report.py — Reporte del modo vivo (ARQ-21): `LiveResult` desde live_state.

`load_live_result(state_dir, prices)` reconstruye, desde los archivos que
publica el runner (scripts/run_periodic_rebalance.py), un `LiveResult` con la
interfaz de `BacktestResult` que leen los reportes (``nav``, ``returns``,
``weights`` con CASH_COL, ``daily_records``; además ``rebalance_log`` y
``trades``). `dashboard_results.build_dashboard` y
`monthly_report.build_report` lo aceptan tal cual.

Fuentes (``state_dir``):
  - rebalance_history.csv: corridas. Cuentan las ``status=ok`` (filas legadas
    sin status = ok); varias con el mismo as_of (--force) → la última.
    Turnover y costo por fecha de rebalanceo.
  - snapshots.csv: cartera TRAS cada corrida (acciones + caja en CASH_COL).
  - transactions.csv: trades sim. Sin snapshots.csv (estado anterior a
    ARQ-21), la cartera de corridas sim se reconstruye acumulando las
    transacciones desde caja (supone un ledger que empezó sin posiciones).
  - broker_orders.csv: fills de corridas broker (trades).

NAV: entre corridas la cartera es la del último snapshot y se marca cada día
con ``prices`` (fecha × ticker, moneda base, ej. ``close_in_base``) desde la
primera corrida. Sin ``prices`` el NAV/pesos son los de cada corrida (precio
del snapshot), solo en fechas de rebalanceo: las métricas anualizan como si
fueran diarias (orientativas). Sin cartera reconstruible (estado legado sin
snapshots ni transacciones con run_id) el NAV es el de history y no hay pesos.

Uso programático:
    from src.live_report import load_live_result
    from src.dashboard_results import build_dashboard
    live = load_live_result(settings.state_dir, prices=prices)
    build_dashboard({"live": live}, title="HFT-V1 Live")

CLI (precios de --prices-csv o de la cadena de datos; red si yahoo):
    python -m src.live_report [--prices-csv P] [--output O] [--sources yahoo,csv]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.backtest_engine import CASH_COL, TRADE_COLUMNS, DailyRecord
from src.settings import Settings, load_settings, setup_logging

logger = logging.getLogger("live_report")

HISTORY_NAME = "rebalance_history.csv"
SNAPSHOTS_NAME = "snapshots.csv"
TRANSACTIONS_NAME = "transactions.csv"
BROKER_ORDERS_NAME = "broker_orders.csv"
COST_TICKER = "__COST__"          # fila de costos del ledger (transactions.csv)
_ZERO = 1e-12

PathLike = Union[str, Path]


@dataclass
class LiveResult:
    """Corridas vivas comprometidas con la interfaz de BacktestResult de los reportes.

    Attributes:
        nav / returns / weights / daily_records: como en BacktestResult
            (``weights`` = fracción del NAV por ticker + CASH_COL).
        rebalance_log: una fila de history por fecha de rebalanceo (la
            corrida ok que quedó vigente), indexada por fecha.
        trades: TRADE_COLUMNS (date, ticker, order_value con signo, cost).
        holdings: acciones (+ caja en CASH_COL) tras cada corrida, fecha × ticker.
        history: rebalance_history.csv completo (ok, partial, superseded).
        book_source: de dónde sale ``holdings`` ("snapshots",
            "transactions" o "none").
        marked_daily: True si el NAV se marcó día a día con ``prices``.
    """

    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    rebalance_log: pd.DataFrame
    daily_records: List[DailyRecord] = field(default_factory=list)
    trades: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS))
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    book_source: str = "none"
    marked_daily: bool = False


@dataclass(frozen=True)
class LiveBook:
    """Lectura de live_state: corridas vigentes + cartera tras cada una."""

    history: pd.DataFrame
    runs: pd.DataFrame                    # una fila ok por fecha (index=date)
    holdings: Optional[pd.DataFrame]      # fecha × ticker + CASH_COL (acciones/caja)
    book_prices: Optional[pd.DataFrame]   # fecha × ticker: precio del snapshot
    source: str

    @property
    def tickers(self) -> List[str]:
        if self.holdings is None:
            return []
        held = self.holdings.drop(columns=[CASH_COL], errors="ignore")
        return sorted(held.columns[(held.abs() > _ZERO).any()])

    @property
    def start(self) -> pd.Timestamp:
        return self.holdings.index[0] if self.holdings is not None else self.runs.index[0]


# ===================================================================
# Lectura
# ===================================================================
def read_history(state_dir: PathLike) -> pd.DataFrame:
    """rebalance_history.csv normalizado (date Timestamp; status/mode por defecto).

    Raises:
        FileNotFoundError: no hay history (ninguna corrida comprometida).
    """
    path = Path(state_dir) / HISTORY_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: ninguna corrida viva comprometida.")
    hist = pd.read_csv(path)
    hist["date"] = pd.to_datetime(hist["date"])
    for col, default in (("status", "ok"), ("mode", "sim")):
        hist[col] = hist[col].fillna(default) if col in hist else default
    if "run_id" not in hist:
        hist["run_id"] = np.nan
    return hist


def committed_runs(history: pd.DataFrame) -> pd.DataFrame:
    """Corridas vigentes: status ok, la última por as_of (orden del archivo =
    orden de commit), indexadas por fecha."""
    ok = history[history["status"] == "ok"]
    ok = ok.drop_duplicates("date", keep="last").sort_values("date", kind="stable")
    return ok.set_index("date")


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _book_from_snapshots(state_dir: Path, runs: pd.DataFrame):
    snaps = _read_csv(state_dir / SNAPSHOTS_NAME)
    if snaps is None:
        return None, None
    snaps = snaps[snaps["run_id"].isin(runs["run_id"].dropna())]
    if snaps.empty:
        return None, None
    snaps = snaps.assign(date=pd.to_datetime(snaps["date"]))
    shares = snaps.pivot(index="date", columns="ticker", values="shares").fillna(0.0)
    prices = snaps.pivot(index="date", columns="ticker", values="price")
    n_missing = int((~runs.index.isin(shares.index)).sum())
    if n_missing:
        logger.info("%d corrida(s) sin snapshot (anteriores a snapshots.csv): "
                    "cartera desde %s.", n_missing, shares.index[0].date())
    if CASH_COL not in shares:
        shares[CASH_COL] = 0.0
    return shares.sort_index(), prices.sort_index()


def _book_from_transactions(state_dir: Path, history: pd.DataFrame,
                            runs: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Acciones acumuladas hasta cada corrida vigente (solo corridas sim con run_id)."""
    txns = _read_csv(state_dir / TRANSACTIONS_NAME)
    if (txns is None or "run_id" not in txns or runs["run_id"].isna().any()
            or not (runs["mode"] == "sim").all()):
        return None
    # Secuencia de commit de cada corrida (todas las filas: las superseded
    # también movieron el ledger).
    seq = {rid: i for i, rid in enumerate(history["run_id"]) if pd.notna(rid)}
    txns = txns[txns["ticker"] != COST_TICKER].assign(seq=txns["run_id"].map(seq))
    if txns["seq"].isna().any():
        logger.warning("transactions.csv tiene run_id sin fila en history: "
                       "cartera no reconstruible desde transacciones.")
        return None
    cum = (txns.pivot_table(index="seq", columns="ticker", values="shares",
                            aggfunc="sum").fillna(0.0).sort_index().cumsum())
    rows = []
    for rid in runs["run_id"]:
        upto = cum[cum.index <= seq[rid]]
        rows.append(upto.iloc[-1] if len(upto) else pd.Series(0.0, index=cum.columns))
    shares = pd.DataFrame(rows, index=runs.index).fillna(0.0)
    shares[CASH_COL] = runs["cash"].astype(float).to_numpy()
    return shares


def read_live_book(state_dir: PathLike) -> LiveBook:
    """History + cartera tras cada corrida (snapshots → transactions → none)."""
    state_dir = Path(state_dir)
    history = read_history(state_dir)
    runs = committed_runs(history)
    if runs.empty:
        raise ValueError(f"{state_dir / HISTORY_NAME}: ninguna corrida con status=ok.")
    holdings, book_prices = _book_from_snapshots(state_dir, runs)
    source = "snapshots"
    if holdings is None:
        holdings, source = _book_from_transactions(state_dir, history, runs), "transactions"
    if holdings is None:
        source = "none"
        logger.warning("Sin snapshots.csv ni transacciones con run_id en %s: NAV de "
                       "history en fechas de rebalanceo, sin pesos.", state_dir)
    return LiveBook(history=history, runs=runs, holdings=holdings,
                    book_prices=book_prices, source=source)


# ===================================================================
# Marcado y armado del resultado
# ===================================================================
def mark_to_market(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    end: Optional[pd.Timestamp] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """NAV y pesos diarios de la cartera vigente marcada a ``prices``.

    Cada día usa la última cartera con fecha <= día; precio = último cierre
    válido <= día. Fechas = ``prices.index`` desde la primera cartera hasta
    ``end`` (default: último precio).

    Raises:
        ValueError: ``prices`` no cubre el periodo, o una posición abierta
            no tiene precio en alguna fecha (no se marca a cero en silencio).
    """
    tickers = [c for c in holdings.columns if c != CASH_COL]
    prices = prices.sort_index()
    end = pd.Timestamp(end) if end is not None else prices.index[-1]
    idx = prices.index[(prices.index >= holdings.index[0]) & (prices.index <= end)]
    if len(idx) == 0:
        raise ValueError(f"prices no cubre {holdings.index[0].date()}..{end.date()}.")
    book = holdings.reindex(holdings.index.union(idx)).ffill().reindex(idx)
    px = prices.reindex(columns=tickers).ffill().reindex(idx)
    unpriced = px.isna() & (book[tickers].abs() > _ZERO)
    if unpriced.any().any():
        bad = unpriced.any()
        first = unpriced.idxmax()[bad]
        raise ValueError(
            "Posiciones abiertas sin precio en `prices`: "
            + ", ".join(f"{tk} (desde {d.date()})" for tk, d in first.items()))
    values = book[tickers] * px.fillna(0.0)
    cash = book[CASH_COL].astype(float)
    nav = (values.sum(axis=1) + cash).rename("nav")
    weights = values.div(nav, axis=0)
    weights[CASH_COL] = cash / nav
    return nav, weights


def _on_or_after(dates: pd.DatetimeIndex, targets: pd.DatetimeIndex):
    """(posición en ``dates`` de la primera fecha >= cada target, máscara de
    los targets que caen dentro de ``dates``)."""
    pos = dates.searchsorted(targets)
    keep = pos < len(dates)
    return pos[keep], keep


def _per_date(runs: pd.DataFrame, col: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Columna de las corridas en la primera fecha de ``dates`` >= su as_of."""
    if col not in runs:
        return pd.Series(0.0, index=dates)
    pos, keep = _on_or_after(dates, runs.index)
    vals = pd.to_numeric(runs[col], errors="coerce").fillna(0.0).to_numpy()[keep]
    out = pd.Series(vals, index=dates[pos]).groupby(level=0).sum()
    return out.reindex(dates, fill_value=0.0)


def _sim_trades(state_dir: Path, ok_ids: set) -> pd.DataFrame:
    txns = _read_csv(state_dir / TRANSACTIONS_NAME)
    if txns is None:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    if "run_id" in txns:
        txns = txns[txns["run_id"].isin(ok_ids)]
    txns = txns.assign(date=pd.to_datetime(txns["t"]).dt.tz_localize(None),
                       key=txns["run_id"] if "run_id" in txns else txns["t"])
    costs = txns[txns["ticker"] == COST_TICKER].groupby("key")["cost"].sum()
    trades = txns[txns["ticker"] != COST_TICKER].copy()
    trades["order_value"] = trades["shares"] * trades["price"]
    # Costo de la corrida repartido pro rata al nocional (como el backtest)
    gross = trades["order_value"].abs().groupby(trades["key"]).transform("sum")
    share = (trades["order_value"].abs() / gross.replace(0.0, np.nan)).fillna(0.0)
    trades["cost"] = trades["key"].map(costs).fillna(0.0) * share
    return trades[TRADE_COLUMNS]


def _broker_trades(state_dir: Path, run_ids: set) -> pd.DataFrame:
    orders = _read_csv(state_dir / BROKER_ORDERS_NAME)
    if orders is None or "filled_qty" not in orders:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    orders = orders[orders["run_id"].isin(run_ids)]
    qty = pd.to_numeric(orders["filled_qty"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(orders["filled_avg_price"], errors="coerce")
    filled = (qty > 0) & px.notna()
    sign = np.where(orders["side"] == "buy", 1.0, -1.0)
    return pd.DataFrame({
        "date": pd.to_datetime(orders["date"]),
        "ticker": orders["ticker"],
        "order_value": qty * px * sign,
        "cost": 0.0,
    })[filled].reset_index(drop=True)


def live_trades(state_dir: PathLike, history: pd.DataFrame) -> pd.DataFrame:
    """Ejecuciones reales: transacciones sim de corridas ok + fills broker
    (ok y partial: lo enviado y llenado en un parcial también ocurrió)."""
    state_dir = Path(state_dir)
    ok_ids = set(history.loc[history["status"] == "ok", "run_id"].dropna())
    sent_ids = set(history.loc[history["status"].isin(["ok", "partial"]),
                               "run_id"].dropna())
    parts = [t for t in (_sim_trades(state_dir, ok_ids),
                         _broker_trades(state_dir, sent_ids)) if not t.empty]
    if not parts:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return (pd.concat(parts, ignore_index=True)
            .sort_values(["date", "ticker"], kind="stable").reset_index(drop=True))


def load_live_result(
    state_dir: Optional[PathLike] = None,
    prices: Optional[pd.DataFrame] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> LiveResult:
    """`LiveResult` de las corridas vivas comprometidas en ``state_dir``.

    Args:
        state_dir: Default ``load_settings().state_dir``.
        prices: Cierres diarios en moneda base (fecha × ticker) que cubran
            las posiciones desde la primera corrida. None → NAV solo en
            fechas de rebalanceo (ver docstring del módulo).
        end: Última fecha marcada (default: último precio).

    Raises:
        FileNotFoundError: sin rebalance_history.csv.
        ValueError: sin corridas ok, o ``prices`` sin precio para una
            posición abierta.
    """
    state_dir = Path(state_dir) if state_dir is not None else load_settings().state_dir
    book = read_live_book(state_dir)
    runs = book.runs
    marked = False
    if book.holdings is not None and prices is not None:
        nav, weights = mark_to_market(book.holdings, prices, end=end)
        marked = True
    elif book.holdings is not None and book.book_prices is not None:
        nav, weights = mark_to_market(book.holdings, book.book_prices, end=end)
    else:
        if prices is not None:
            logger.warning("Sin cartera reconstruible: se ignoran `prices`.")
        nav = runs["nav"].astype(float).rename("nav")
        weights = pd.DataFrame(index=nav.index)
    if not marked:
        logger.warning("NAV solo en fechas de rebalanceo (sin precios diarios): "
                       "métricas anualizadas solo orientativas.")
    nav = nav.rename_axis(None)
    returns = nav.ffill().pct_change().fillna(0.0).rename("returns")
    turnover = _per_date(runs, "turnover", nav.index)
    cost = _per_date(runs, "cost", nav.index)
    rebalance_dates = set(nav.index[_on_or_after(nav.index, runs.index)[0]])
    empty = pd.Series(dtype=float)
    records = [
        DailyRecord(
            date=d, nav=float(nav.iloc[i]), daily_return=float(returns.iloc[i]),
            weights=weights.loc[d] if d in weights.index else empty,
            is_rebalance=d in rebalance_dates,
            turnover=float(turnover.iloc[i]), cost=float(cost.iloc[i]),
        )
        for i, d in enumerate(nav.index)
    ]
    holdings = book.holdings if book.holdings is not None else pd.DataFrame()
    return LiveResult(
        nav=nav, returns=returns, weights=weights, rebalance_log=runs,
        daily_records=records, trades=live_trades(state_dir, book.history),
        holdings=holdings, history=book.history, book_source=book.source,
        marked_daily=marked,
    )


# ===================================================================
# Dashboard del modo vivo (CLI)
# ===================================================================
def default_live_dashboard_path(settings: Optional[Settings] = None) -> Path:
    """`<results_dir>/live_dashboard.html` (nunca relativo al cwd)."""
    return (settings or load_settings()).results_dir / "live_dashboard.html"


def fetch_live_prices(
    tickers: Sequence[str],
    start: pd.Timestamp,
    settings: Optional[Settings] = None,
    sources: Sequence[str] = ("yahoo", "csv"),
) -> pd.DataFrame:
    """Cierres en moneda base de ``tickers`` desde ``start`` (misma cadena de
    datos que el runner: maestro de instrumentos + FX; red si yahoo)."""
    from src.instruments import load_instruments
    from src.market_data import (
        RESEARCH_POLICY, InstrumentProvider, build_provider, close_in_base, load_bars,
    )
    settings = settings or load_settings()
    provider = InstrumentProvider(build_provider(sources, settings=settings),
                                  load_instruments(),
                                  base_currency=settings.base_currency)
    now = datetime.now(timezone.utc)
    start_s = (pd.Timestamp(start) - timedelta(days=7)).strftime("%Y-%m-%d")
    end_s = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    bars = load_bars(provider, list(tickers), start_s, end_s,
                     policy=RESEARCH_POLICY, now=now)
    close, _ = close_in_base(provider, bars.close, start_s, end_s)
    return close


def build_live_dashboard(
    state_dir: Optional[PathLike] = None,
    prices: Optional[pd.DataFrame] = None,
    output_path: Optional[PathLike] = None,
    label: str = "live",
    benchmark_prices=None,
) -> str:
    """Dashboard (src.dashboard_results) del modo vivo; devuelve la ruta."""
    from src.dashboard_results import build_dashboard
    result = load_live_result(state_dir, prices=prices)
    return build_dashboard(
        {label: result},
        output_path=output_path or default_live_dashboard_path(),
        title="HFT-V1 Live Dashboard",
        benchmark_prices=benchmark_prices,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dashboard del modo vivo desde live_state (history + "
                    "snapshots + transactions).")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="Default: settings.state_dir (HFT_HOME o repo).")
    parser.add_argument("--prices-csv", type=Path, default=None,
                        help="Cierres en moneda base (1a columna = fecha). "
                             "Default: descarga con --sources.")
    parser.add_argument("--sources", default="yahoo,csv",
                        help="Cadena de datos si no hay --prices-csv (default yahoo,csv).")
    parser.add_argument("--output", type=Path, default=None,
                        help="HTML (default results/live_dashboard.html).")
    args = parser.parse_args(argv)
    setup_logging()
    settings = load_settings()
    state_dir = args.state_dir or settings.state_dir
    book = read_live_book(state_dir)
    prices = None
    if args.prices_csv is not None:
        prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    elif book.tickers:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        prices = fetch_live_prices(book.tickers, book.start, settings, sources)
    path = build_live_dashboard(state_dir, prices=prices,
                                output_path=args.output or default_live_dashboard_path(settings))
    print(f"Live dashboard → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
