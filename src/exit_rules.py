"""
exit_rules.py — Criterios de cierre total de posiciones.

Antes de este modulo no existia ningun criterio de salida: las posiciones
solo se reducian cuando el peso objetivo bajaba. Aqui se definen los
disparadores de cierre total (peso -> 0), todos causales (R-05):

  1. stop_loss: precio cae >= pct bajo el precio de entrada.
  2. take_profit: precio sube >= pct sobre el precio de entrada.
  3. trailing_stop: precio cae >= pct bajo su maximo movil (ventana).
  4. universe_exit: posicion fuera del universo activo (iliquida/excluida).
  5. take_profit_dollars: PnL no realizado por ticker >= +USD (cierra).
  6. stop_loss_dollars: PnL no realizado por ticker <= -USD (cierra).

Las reglas 5-6 son en dólares absolutos por posición:
  unrealized(tk) = (precio_t - entrada_media) * shares.
La evaluacion es pura y determinista (R-07). El motor (backtest o script
periodico) aplica los cierres y los registra con motivo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("exit_rules")


# Defaults compartidos con BacktestConfig (fuente unica).
DEFAULT_STOP_LOSS_PCT = 0.01
DEFAULT_TAKE_PROFIT_PCT = 0.02
DEFAULT_TRAILING_WINDOW = 63


@dataclass
class ExitConfig:
    """Parametros de salida. pct en fracciones (0.01 = 1%), dólares en USD.

    Defaults estilo scalping: stop 1%, take-profit 2%, sin trailing,
    sin reglas en dólares (None = off; el live las activa con +2/-1).
    """

    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT  # vs precio de entrada (None = off)
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT  # vs precio de entrada (None = off)
    trailing_stop_pct: Optional[float] = None  # vs maximo movil (None = off)
    trailing_window: int = DEFAULT_TRAILING_WINDOW  # ventana del maximo movil (dias)
    exit_on_illiquid: bool = True     # cierra lo fuera del universo activo
    take_profit_dollars: Optional[float] = None  # PnL >= +USD → cierra
    stop_loss_dollars: Optional[float] = None    # PnL <= -USD → cierra

    def active_rules(self) -> List[str]:
        """Nombres de reglas habilitadas."""
        rules = []
        if self.stop_loss_pct is not None:
            rules.append("stop_loss")
        if self.take_profit_pct is not None:
            rules.append("take_profit")
        if self.trailing_stop_pct is not None:
            rules.append("trailing_stop")
        if self.exit_on_illiquid:
            rules.append("universe_exit")
        if self.take_profit_dollars is not None:
            rules.append("take_profit_dollars")
        if self.stop_loss_dollars is not None:
            rules.append("stop_loss_dollars")
        return rules


def required_history(config: Optional[ExitConfig] = None) -> int:
    """Filas (índice <= t): ventana del trailing stop si está activo, si no 1."""
    config = config or ExitConfig()
    return int(config.trailing_window) if config.trailing_stop_pct is not None else 1


def stop_loss_exits(
    prices_t: pd.Series,
    entries: Dict[str, float],
    pct: float,
) -> pd.DataFrame:
    """Tickers con (precio/entrada - 1) <= -pct. Solo fecha t (R-05)."""
    rows = []
    for ticker, entry in entries.items():
        if ticker not in prices_t.index or not entry or entry <= 0:
            continue
        px = float(prices_t[ticker])
        if not px or px != px:
            continue
        dd = px / float(entry) - 1.0
        if dd <= -abs(pct):
            rows.append({"ticker": ticker, "rule": "stop_loss",
                         "detail": f"{dd:.2%} vs entrada {entry:.2f}"})
    return pd.DataFrame(rows, columns=["ticker", "rule", "detail"])


def take_profit_exits(
    prices_t: pd.Series,
    entries: Dict[str, float],
    pct: float,
) -> pd.DataFrame:
    """Tickers con (precio/entrada - 1) >= +pct. Solo fecha t (R-05)."""
    rows = []
    for ticker, entry in entries.items():
        if ticker not in prices_t.index or not entry or entry <= 0:
            continue
        px = float(prices_t[ticker])
        if not px or px != px:
            continue
        gain = px / float(entry) - 1.0
        if gain >= abs(pct):
            rows.append({"ticker": ticker, "rule": "take_profit",
                         "detail": f"+{gain:.2%} vs entrada {entry:.2f}"})
    return pd.DataFrame(rows, columns=["ticker", "rule", "detail"])


def trailing_stop_exits(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    held: List[str],
    pct: float,
    window: int = 63,
) -> pd.DataFrame:
    """Tickers con precio_t < (1-pct) * max(ventana hasta t). Solo datos <= t."""
    past = prices[prices.index <= t]
    if len(past) < 2:
        return pd.DataFrame(columns=["ticker", "rule", "detail"])
    lookback = past.iloc[-window:]
    rows = []
    for ticker in held:
        if ticker not in lookback.columns:
            continue
        series = lookback[ticker].dropna()
        if len(series) < 2:
            continue
        peak = float(series.max())
        last = float(series.iloc[-1])
        if peak > 0 and last < (1.0 - abs(pct)) * peak:
            rows.append({"ticker": ticker, "rule": "trailing_stop",
                         "detail": f"{last:.2f} < {(1.0 - abs(pct)) * peak:.2f} "
                                   f"(pico {window}d)"})
    return pd.DataFrame(rows, columns=["ticker", "rule", "detail"])


def universe_exits(
    held: List[str],
    active: List[str],
) -> pd.DataFrame:
    """Posiciones fuera del universo activo -> cierre total."""
    active_set = set(active)
    rows = [{"ticker": tk, "rule": "universe_exit", "detail": "fuera de activo"}
            for tk in held if tk not in active_set]
    return pd.DataFrame(rows, columns=["ticker", "rule", "detail"])


def dollar_exits(
    prices_t: pd.Series,
    entries: Dict[str, float],
    shares: Dict[str, float],
    take_dollars: Optional[float] = None,
    stop_dollars: Optional[float] = None,
) -> pd.DataFrame:
    """Cierres por PnL no realizado en dólares por ticker.

    unrealized(tk) = (precio_t − entrada_media) × shares.
    - take: unrealized >= +take_dollars → cierra (ganancia).
    - stop: unrealized <= −stop_dollars → cierra (pérdida).
    Solo fecha t (R-05). Sin shares/entradas válidas se omite el ticker.
    """
    rows = []
    for ticker, entry in entries.items():
        if ticker not in prices_t.index or not entry or entry <= 0:
            continue
        sh = float(shares.get(ticker, 0.0)) if shares else 0.0
        if not sh or sh <= 0:
            continue
        px = float(prices_t[ticker])
        if not px or px != px:
            continue
        pnl = (px - float(entry)) * sh
        if take_dollars is not None and pnl >= abs(take_dollars):
            rows.append({"ticker": ticker, "rule": "take_profit_dollars",
                         "detail": f"+${pnl:.2f} >= +${abs(take_dollars):.2f}"})
        elif stop_dollars is not None and pnl <= -abs(stop_dollars):
            rows.append({"ticker": ticker, "rule": "stop_loss_dollars",
                         "detail": f"${pnl:.2f} <= -${abs(stop_dollars):.2f}"})
    return pd.DataFrame(rows, columns=["ticker", "rule", "detail"])


def evaluate_exits(
    t: pd.Timestamp,
    prices: pd.DataFrame,
    held: List[str],
    entries: Dict[str, float],
    active: Optional[List[str]] = None,
    config: Optional[ExitConfig] = None,
    shares: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Evalua todas las reglas habilitadas y devuelve cierres con motivo."""
    config = config or ExitConfig()
    if not held:
        return pd.DataFrame(columns=["ticker", "rule", "detail"])
    frames = []
    if config.stop_loss_pct is not None:
        if t in prices.index:
            frames.append(stop_loss_exits(prices.loc[t], entries,
                                          config.stop_loss_pct))
    if config.take_profit_pct is not None:
        if t in prices.index:
            frames.append(take_profit_exits(prices.loc[t], entries,
                                            config.take_profit_pct))
    if config.trailing_stop_pct is not None:
        frames.append(trailing_stop_exits(prices, t, held,
                                          config.trailing_stop_pct,
                                          config.trailing_window))
    if (config.take_profit_dollars is not None
            or config.stop_loss_dollars is not None):
        if t in prices.index and shares:
            frames.append(dollar_exits(prices.loc[t], entries, shares,
                                       config.take_profit_dollars,
                                       config.stop_loss_dollars))
    if config.exit_on_illiquid and active is not None:
        frames.append(universe_exits(held, active))
    if not frames:
        return pd.DataFrame(columns=["ticker", "rule", "detail"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ticker")
    if not out.empty:
        logger.info("evaluate_exits(%s): cierres=%s",
                    t.date(), out["ticker"].tolist())
    return out


def apply_exits_to_weights(
    weights: pd.Series,
    exits: pd.DataFrame,
) -> pd.Series:
    """Pone en 0 los tickers con cierre y renormaliza a 1."""
    if exits is None or exits.empty:
        return weights
    w = weights.copy().astype(float)
    w[exits["ticker"].tolist()] = 0.0
    w = w.clip(lower=0.0)
    s = w.sum()
    if s <= 1e-10:  # todo cerrado: equal-weight entre no-cerrados es vacio
        return pd.Series(0.0, index=weights.index)
    return w / s
