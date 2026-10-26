"""
monthly_report.py — Paso 5: reporte mensual auto con atribucion por sleeve.

Genera `results/report.html` autonomo por estrategia:
  1. NAV total vs sleeves (us/co).
  2. Tabla de atribucion del periodo por sleeve.
  3. Barras de retorno mensual + tabla mensual (mejor/peor sleeve).
  4. Fila de metricas (reusa `compute_metrics`).

Uso programatico:
    from src.monthly_report import build_report
    build_report(results, prices, groups)  # -> <results_dir>/report.html

CLI (demo sintetica, sin red):
    python -m src.monthly_report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult
from src.metrics_calculator import compute_metrics
from src.settings import load_settings
from src.sleeve_attribution import (
    attribution_table,
    monthly_summary,
    sleeve_avg_weights,
    sleeve_nav,
    sleeve_returns,
)

if TYPE_CHECKING:  # los reportes aceptan también el modo vivo
    from src.live_report import LiveResult

logger = logging.getLogger("monthly_report")

_CSS = (
    "body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }"
    "h1 { color: #2c3e50; } h2 { color: #34495e; margin-top: 30px; "
    "border-bottom: 1px solid #bdc3c7; }"
    ".card { background: white; border-radius: 8px; "
    "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; padding: 10px; }"
    "table { border-collapse: collapse; width: 100%; font-size: 12px; }"
    "th { background: steelblue; color: white; padding: 6px; text-align: right; }"
    "th:first-child, td:first-child { text-align: left; }"
    "td { border-bottom: 1px solid #eee; padding: 5px 6px; text-align: right; }"
)


def _fig_sleeve_nav(dates, total_nav, sleeves_nav, label: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=total_nav.values, mode="lines",
                             name="total",
                             hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>total</extra>"))
    for sleeve in sleeves_nav.columns:
        fig.add_trace(go.Scatter(x=dates, y=sleeves_nav[sleeve].values, mode="lines",
                                 name=sleeve, line=dict(dash="dash"),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(title=f"NAV total vs sleeves — {label}", template="plotly_white",
                      height=400, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _fig_monthly_bars(monthly: pd.DataFrame, label: str):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=monthly.index.tolist(), y=monthly["total"].values, name="retorno mensual",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(title=f"Retorno mensual — {label}", template="plotly_white",
                      height=350, yaxis_tickformat=".1%")
    return fig


def _df_to_html_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    head = "".join(f"<th>{c}</th>" for c in [""] + list(df.columns))
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                cells.append(f"<td>{floatfmt.format(v)}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append(f"<tr><td><b>{idx}</b></td>{''.join(cells)}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def default_report_path() -> Path:
    """`<results_dir>/report.html` (src/settings.py; nunca relativo al cwd)."""
    return load_settings().results_dir / "report.html"


def build_report(
    results: Dict[str, "BacktestResult | LiveResult"],
    prices: pd.DataFrame,
    groups: Union[pd.Series, Dict],
    output_path: Optional[Union[str, Path]] = None,
    level: str = "market",
    risk_free_rate: float = 0.0,
    title: str = "HFT-V1 Monthly Report — Sleeve Attribution",
) -> str:
    """Construye el reporte mensual con atribucion por sleeve.

    Args:
        results: Dict {estrategia → BacktestResult | LiveResult (modo vivo,
            src.live_report)}.
        prices: Precios diarios del universo (fecha × ticker).
        groups: Ticker -> grupo "mercado/sector".
        output_path: HTML de salida (default: `default_report_path()`).
        level: "market" (us/co) o "group".
        risk_free_rate: Tasa anual para metricas.
        title: Titulo del reporte.

    Returns:
        Ruta del HTML escrito.
    """
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("plotly requerido: pip install plotly") from exc
    if not results:
        raise ValueError("results vacio — nada que reportar.")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>", f"<style>{_CSS}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>Estrategias: {', '.join(results.keys())} | Nivel: {level}</p>",
    ]
    for label, res in results.items():
        contrib = sleeve_returns(res.weights, prices, groups, level=level)
        sleeves_nav = sleeve_nav(contrib)
        total_nav = res.nav / res.nav.iloc[0]
        monthly = monthly_summary(contrib)
        avg_w = sleeve_avg_weights(res.weights, groups, level=level)
        attr = attribution_table(contrib, avg_w)
        turnover_s = pd.Series({r.date: r.turnover for r in res.daily_records})
        m = compute_metrics(res.returns, res.nav, turnover=turnover_s,
                            risk_free_rate=risk_free_rate, n_bootstrap=200)

        parts.append(f"<h2>{label}</h2>")
        parts.append("<div class='card'>"
                     f"<b>Sharpe {m.sharpe_ratio:.3f} [{m.sharpe_ci_low:.3f}, {m.sharpe_ci_high:.3f}]</b>"
                     f" | Ann.Ret {m.annualized_return:.1%} | MaxDD {m.max_drawdown:.1%}"
                     f" | Turnover anual {m.avg_annual_turnover:.1%}</div>")
        parts.append("<h2>1. NAV total vs sleeves</h2><div class='card'>"
                     + _fig_sleeve_nav(total_nav.index, total_nav, sleeves_nav, label)
                     .to_html(full_html=False, include_plotlyjs="cdn") + "</div>")
        parts.append("<h2>2. Atribucion del periodo por sleeve</h2><div class='card'>"
                     + _df_to_html_table(attr) + "</div>")
        parts.append("<h2>3. Retorno mensual</h2><div class='card'>"
                     + _fig_monthly_bars(monthly, label)
                     .to_html(full_html=False, include_plotlyjs=False) + "</div>")
        parts.append("<div class='card'>" + _df_to_html_table(monthly) + "</div>")

    parts.append("</body></html>")
    output_path = Path(output_path) if output_path is not None else default_report_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    logger.info("Reporte escrito en: %s", output_path)
    return str(output_path)


def _demo() -> str:
    """Demo sintetica mixta US/CO (sin red) -> <results_dir>/report.html."""
    from src.backtest_engine import BacktestConfig, run_backtest
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=200)
    us = [f"US{i}" for i in range(6)]
    co = ["CO_F1", "CO_F2", "CO_E1", "CO_U1"]
    cols = us + co
    prices = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for c in cols:
        drift = 0.0008 if c.startswith("US") else 0.0005
        prices[c] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, len(dates))))
    volumes = pd.DataFrame(2_000_000.0, index=dates, columns=cols)
    groups = {c: ("us/Tech" if c.startswith("US") else
                  {"CO_F1": "co/Financials", "CO_F2": "co/Financials",
                   "CO_E1": "co/Energy", "CO_U1": "co/Utilities"}[c])
              for c in cols}
    cfg = BacktestConfig(opt_method="robust", window=30, warmup=30,
                         rebalance_freq="M", seed=42,
                         groups=groups,
                         group_caps={("co/Financials", "co/Energy"): 0.40})
    res = run_backtest(prices, volumes, cfg)
    path = build_report({"robust_mixto": res}, prices, groups)
    print(f"Reporte guardado en: {path}")
    return path


if __name__ == "__main__":
    _demo()
