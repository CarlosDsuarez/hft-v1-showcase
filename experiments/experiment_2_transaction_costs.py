"""
experiment_2_transaction_costs.py — T-28: Impacto de costos de transacción.

Compara cuatro configuraciones de costos:
  1. Sin costos (commission=0, spread=0, impact=0)
  2. Costos bajos (commission=2 bps, spread=1 bps, impact=0.05)
  3. Costos base (commission=5 bps, spread=2 bps, impact=0.10)
  4. Costos altos (commission=10 bps, spread=5 bps, impact=0.20)

Métricas reportadas: Sharpe, net_return, costo_total_acumulado,
turnover_anual, MaxDD.

Genera:
  - Tabla de resultados
  - NAV curves por nivel de costo
  - Scatter: turnover vs net Sharpe

Uso:
  python -m experiments.experiment_2_transaction_costs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.data_client import load_universe, fetch_bars, clean_prices
from src.market_data import RESEARCH_POLICY
from src.settings import LOOKBACK_DAYS, load_settings, setup_logging
from src.backtest_engine import BacktestConfig, run_comparison
from src.metrics_calculator import compute_metrics, compare_metrics

logger = logging.getLogger("exp2")

# Universo canónico: 100 principales del S&P 500 (universe.csv)
TICKERS = load_universe()["ticker"].tolist()
# Ventana Yahoo: solo últimos 3 meses por activo
_END_DT = datetime.now(timezone.utc) + timedelta(days=1)
END = _END_DT.strftime("%Y-%m-%d")
RESULTS_DIR = load_settings().results_dir
START = (_END_DT - pd.Timedelta(LOOKBACK_DAYS, unit="D")).strftime("%Y-%m-%d")
WARMUP = 60
WINDOW = 60
REBAL_FREQ = "15D"

CONFIGS = {
    "no_cost": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        commission_bps=0.0,
        spread_bps=0.0,
        impact_coef=0.0,
    ),
    "low_cost": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        commission_bps=2.0,
        spread_bps=1.0,
        impact_coef=0.05,
    ),
    "base_cost": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.10,
    ),
    "high_cost": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        commission_bps=10.0,
        spread_bps=5.0,
        impact_coef=0.20,
    ),
}


def run() -> pd.DataFrame:
    """Run Experiment 2 and return a metrics comparison DataFrame."""
    logger.info("=== Experiment 2: Transaction Cost Impact ===")

    logger.info("Downloading price data...")
    bars = fetch_bars(TICKERS, start=START, end=END, policy=RESEARCH_POLICY)
    prices, _ = clean_prices(bars.close)
    volumes = bars.volume.reindex(prices.index).ffill().fillna(1e6)

    results, errors = run_comparison(prices, volumes, CONFIGS)
    if errors:
        raise RuntimeError(
            "Backtests failed (comparison would be incomplete): "
            + "; ".join(f"{k}: {v}" for k, v in errors.items())
        ) from next(iter(errors.values()))

    metrics_dict = {}
    for label, res in results.items():
        turnover_s = pd.Series({r.date: r.turnover for r in res.daily_records})
        cost_s = pd.Series({r.date: r.cost for r in res.daily_records})
        m = compute_metrics(
            returns=res.returns,
            nav=res.nav,
            turnover=turnover_s,
        )
        metrics_dict[label] = m
        total_cost_drag = float(cost_s.sum())
        logger.info(
            "%s → Sharpe=%.3f, Ann.Ret=%.1f%%, MaxDD=%.1f%%, "
            "Avg.Annual.TO=%.1f%%, CumCostDrag=%.4f",
            label, m.sharpe_ratio,
            m.annualized_return * 100, m.max_drawdown * 100,
            m.avg_annual_turnover * 100, total_cost_drag,
        )

    table = compare_metrics(metrics_dict)
    print("\n=== Experiment 2 Results ===")
    print(table.T.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.T.to_csv(RESULTS_DIR / "exp2_transaction_costs.csv")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NAV curves
        ax = axes[0]
        colors = ["green", "steelblue", "darkorange", "red"]
        for (label, res), color in zip(results.items(), colors):
            nav_norm = res.nav / res.nav.iloc[0]
            ax.plot(nav_norm.index, nav_norm.values, label=label, color=color)
        ax.set_title("Experiment 2: NAV by Cost Level (Normalized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV (base=1)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Turnover vs Sharpe scatter
        ax2 = axes[1]
        for label, m in metrics_dict.items():
            ax2.scatter(m.avg_annual_turnover, m.sharpe_ratio,
                        s=100, label=label, zorder=5)
            ax2.annotate(
                label,
                (m.avg_annual_turnover, m.sharpe_ratio),
                textcoords="offset points", xytext=(5, 5), fontsize=9,
            )
        ax2.set_title("Annual Turnover vs Net Sharpe")
        ax2.set_xlabel("Avg Annual Turnover")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp2_transaction_costs.png", dpi=120)
        logger.info("Plot saved: %s", RESULTS_DIR / "exp2_transaction_costs.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")

    return table


if __name__ == "__main__":
    setup_logging()
    run()
