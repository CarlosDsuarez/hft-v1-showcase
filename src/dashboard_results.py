"""
dashboard_results.py — T-31: Dashboard interactivo con Plotly.

Genera un archivo HTML autónomo con cuatro secciones:
  1. NAV curves comparadas (todos los métodos/configs).
  2. Frontera eficiente ex-post (retorno vs volatilidad anualizada).
  3. Heatmap dinámico de pesos (fecha × ticker).
  4. Scatter turnover vs Sharpe por configuración.

Uso programático:
    from src.dashboard_results import build_dashboard
    build_dashboard(results_dict)  # -> <results_dir>/dashboard.html

Uso desde línea de comandos:
    python -m src.dashboard_results  (ejecuta demo con datos sintéticos)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult
from src.metrics_calculator import compute_metrics, PortfolioMetrics
from src.settings import INITIAL_CASH, load_settings

if TYPE_CHECKING:  # los reportes aceptan también el modo vivo
    from src.live_report import LiveResult

logger = logging.getLogger("dashboard_results")


# ===================================================================
# Helper: compute metrics for a results dict (+ optional benchmarks)
# ===================================================================
def _benchmark_returns(
    benchmark_prices: Optional[Dict[str, pd.Series]],
) -> Dict[str, pd.Series]:
    """Precios de benchmarks -> retornos log diarios por label."""
    out: Dict[str, pd.Series] = {}
    if not benchmark_prices:
        return out
    for label, px in benchmark_prices.items():
        px = pd.Series(px, dtype=float).sort_index()
        out[label] = np.log(px / px.shift(1)).dropna()
    return out


def _compute_all_metrics(
    results: Dict[str, "BacktestResult | LiveResult"],
    risk_free_rate: float = 0.0,
    benchmark_prices: Optional[Dict[str, pd.Series]] = None,
    reference: str = "SPY",
) -> tuple[Dict[str, PortfolioMetrics], Dict[str, pd.Series], List[str]]:
    """Metricas de estrategias (+ benchmarks si se proveen precios).

    Returns:
        (metrics, bench_returns, benchmark_labels): metricas con
        estrategias primero y benchmarks despues; Beta/IR vs `reference`
        cuando esta disponible.
    """
    bench_rets = _benchmark_returns(benchmark_prices)
    ref = bench_rets.get(reference)
    metrics = {}
    for label, res in results.items():
        turnover_s = pd.Series({r.date: r.turnover for r in res.daily_records})
        metrics[label] = compute_metrics(
            returns=res.returns,
            nav=res.nav,
            turnover=turnover_s,
            benchmark_returns=ref,
            risk_free_rate=risk_free_rate,
        )
    for label, r in bench_rets.items():
        b_nav = (1.0 + r.fillna(0.0)).cumprod()
        metrics[label] = compute_metrics(
            returns=r,
            nav=b_nav,
            benchmark_returns=None if label == reference else ref,
            risk_free_rate=risk_free_rate,
        )
    return metrics, bench_rets, list(bench_rets)


# ===================================================================
# Individual chart builders
# ===================================================================
def _nav_chart(
    results: Dict[str, "BacktestResult | LiveResult"],
    benchmark_prices: Optional[Dict[str, pd.Series]] = None,
):
    """Line chart: normalized NAV over time per strategy (+ benchmarks)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    fig = go.Figure()
    for label, res in results.items():
        nav_norm = res.nav / res.nav.iloc[0]
        fig.add_trace(go.Scatter(
            x=nav_norm.index,
            y=nav_norm.values,
            mode="lines",
            name=label,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    for label, px in (benchmark_prices or {}).items():
        px = pd.Series(px, dtype=float).sort_index()
        nav_norm = px / px.iloc[0]
        fig.add_trace(go.Scatter(
            x=nav_norm.index,
            y=nav_norm.values,
            mode="lines",
            name=f"{label} (bench)",
            line=dict(dash="dash", color="gray"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="NAV Curves (Normalized to 1.0)",
        xaxis_title="Date",
        yaxis_title="NAV (base = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        height=450,
    )
    return fig


def _efficient_frontier_chart(
    results: Dict[str, "BacktestResult | LiveResult"],
    metrics: Dict[str, PortfolioMetrics],
    benchmark_labels: Optional[List[str]] = None,
):
    """Scatter: annualized return vs volatility (ex-post efficient frontier)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    bench_set = set(benchmark_labels or [])
    fig = go.Figure()
    for label, m in metrics.items():
        is_bench = label in bench_set
        fig.add_trace(go.Scatter(
            x=[m.annualized_volatility],
            y=[m.annualized_return],
            mode="markers+text",
            marker=dict(
                size=14,
                symbol="x" if is_bench else "circle",
                color="gray" if is_bench else None,
            ),
            text=[label],
            textposition="top center",
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Vol: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                f"Sharpe: {m.sharpe_ratio:.3f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Ex-Post Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    return fig


def _weights_heatmap(
    results: Dict[str, "BacktestResult | LiveResult"],
    strategy: Optional[str] = None,
    top_n: int = 15,
    resample: str = "ME",
):
    """Heatmap: portfolio weights over time for one strategy."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    label = strategy or list(results.keys())[0]
    weights_df = results[label].weights

    if weights_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Weights Heatmap — {label} (no data)")
        return fig

    # Normalise legacy aliases (pandas 2.2+ requires ME/QE instead of M/Q)
    _FREQ_MAP = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}
    resample = _FREQ_MAP.get(resample.upper(), resample)

    # Resample to reduce density
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_resampled = weights_df.resample(resample).last().fillna(0.0)

    # Keep top N tickers by mean weight
    mean_weights = weights_resampled.mean()
    top_tickers = mean_weights.nlargest(top_n).index.tolist()
    w_plot = weights_resampled[top_tickers]

    fig = go.Figure(go.Heatmap(
        z=w_plot.values.T,
        x=w_plot.index.strftime("%Y-%m"),
        y=w_plot.columns.tolist(),
        colorscale="Blues",
        colorbar=dict(title="Weight"),
        hovertemplate="Date: %{x}<br>Ticker: %{y}<br>Weight: %{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Portfolio Weights Over Time — {label}",
        xaxis_title="Date",
        yaxis_title="Ticker",
        template="plotly_white",
        height=500,
        xaxis=dict(tickangle=-45),
    )
    return fig


def _turnover_sharpe_scatter(
    metrics: Dict[str, PortfolioMetrics],
    benchmark_labels: Optional[List[str]] = None,
):
    """Scatter: avg annual turnover vs Sharpe ratio with CI error bars."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    bench_set = set(benchmark_labels or [])
    labels = [l for l in metrics if l not in bench_set]
    sharpes = [metrics[l].sharpe_ratio for l in labels]
    turnovers = [metrics[l].avg_annual_turnover for l in labels]
    ci_lows = [metrics[l].sharpe_ci_low for l in labels]
    ci_highs = [metrics[l].sharpe_ci_high for l in labels]

    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[turnovers[i]],
            y=[sharpes[i]],
            mode="markers+text",
            marker=dict(size=14),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[ci_highs[i] - sharpes[i]],
                arrayminus=[sharpes[i] - ci_lows[i]],
            ),
            text=[label],
            textposition="top center",
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Turnover: %{x:.1%}<br>"
                "Sharpe: %{y:.3f}<br>"
                f"CI: [{ci_lows[i]:.3f}, {ci_highs[i]:.3f}]"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Turnover vs Sharpe (with 95% Bootstrap CI)",
        xaxis_title="Avg Annual Turnover",
        yaxis_title="Sharpe Ratio",
        xaxis_tickformat=".0%",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    return fig


# ===================================================================
# Main dashboard builder
# ===================================================================
def default_dashboard_path() -> Path:
    """`<results_dir>/dashboard.html` (src/settings.py; never cwd-relative)."""
    return load_settings().results_dir / "dashboard.html"


def build_dashboard(
    results: Dict[str, "BacktestResult | LiveResult"],
    output_path: Optional[Union[str, Path]] = None,
    risk_free_rate: float = 0.0,
    heatmap_strategy: Optional[str] = None,
    title: str = "HFT-V1 Backtesting Dashboard",
    benchmark_prices: Optional[Dict[str, pd.Series]] = None,
    reference: str = "SPY",
) -> str:
    """Build a standalone interactive HTML dashboard.

    Sections:
      1. NAV curves (all strategies + benchmarks dashed).
      2. Performance metrics summary (Type: Strategy/Benchmark).
      3. Ex-post efficient frontier (benchmarks as x markers).
      4. Weights heatmap (first or specified strategy).
      5. Turnover vs Sharpe scatter with CI error bars (strategies only).

    Args:
        results: Dict {label → BacktestResult | LiveResult (modo vivo,
            src.live_report.load_live_result)}.
        output_path: Path to output HTML file (default: `default_dashboard_path()`).
        risk_free_rate: Annual risk-free rate for metrics.
        heatmap_strategy: Strategy to show in heatmap. None → first.
        title: Dashboard title.
        benchmark_prices: Optional {label → prices Series} overlay
            (ej. SPY, ICOLCAP). Beta/IR de estrategias vs `reference`.
        reference: Benchmark de referencia para Beta/IR.

    Returns:
        Path to the written HTML file.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for the dashboard. "
            "Install with: pip install plotly"
        )

    if not results:
        raise ValueError("results dict is empty — nothing to plot.")

    logger.info("Computing metrics for %d strategies...", len(results))
    all_metrics, _, bench_labels = _compute_all_metrics(
        results, risk_free_rate,
        benchmark_prices=benchmark_prices, reference=reference,
    )
    bench_set = set(bench_labels)

    # Build individual figures
    fig_nav = _nav_chart(results, benchmark_prices=benchmark_prices)
    fig_ef = _efficient_frontier_chart(results, all_metrics,
                                       benchmark_labels=bench_labels)
    fig_hm = _weights_heatmap(results, strategy=heatmap_strategy)
    fig_scatter = _turnover_sharpe_scatter(all_metrics,
                                           benchmark_labels=bench_labels)

    # Metrics summary table
    metrics_rows = []
    for label, m in all_metrics.items():
        metrics_rows.append({
            "Type": "Benchmark" if label in bench_set else "Strategy",
            "Strategy": label,
            "Ann. Return": f"{m.annualized_return:.1%}",
            "Ann. Vol": f"{m.annualized_volatility:.1%}",
            "Sharpe": f"{m.sharpe_ratio:.3f}",
            "Sharpe CI": f"[{m.sharpe_ci_low:.3f}, {m.sharpe_ci_high:.3f}]",
            "Sortino": f"{m.sortino_ratio:.3f}",
            "Max DD": f"{m.max_drawdown:.1%}",
            "Calmar": f"{m.calmar_ratio:.3f}",
            "VaR 95%": f"{m.var_95_historical:.4f}",
            "CVaR 95%": f"{m.cvar_95:.4f}",
            "Beta": f"{m.beta:.3f}",
            "IR": f"{m.information_ratio:.3f}",
            "Avg TO": f"{m.avg_annual_turnover:.1%}" if label not in bench_set else "—",
        })
    metrics_df = pd.DataFrame(metrics_rows)

    table_trace = go.Table(
        header=dict(
            values=list(metrics_df.columns),
            fill_color="steelblue",
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=[metrics_df[c] for c in metrics_df.columns],
            fill_color=[["white", "#f0f4f8"] * (len(metrics_df) // 2 + 1)][0][:len(metrics_df)],
            align="left",
            font=dict(size=10),
        ),
    )
    fig_table = go.Figure(table_trace)
    fig_table.update_layout(
        title="Performance Metrics Summary",
        height=max(200, 60 + len(metrics_df) * 30),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Assemble HTML
    html_parts = [
        f"<!DOCTYPE html><html><head>",
        f"<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; }",
        ".chart-container { background: white; border-radius: 8px; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; padding: 10px; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>Generated by <b>dashboard_results.py</b> | "
        f"Strategies: {', '.join(results.keys())}"
        + (f" | Benchmarks: {', '.join(bench_labels)}" if bench_labels else "")
        + "</p>",
        "<h2>1. NAV Curves</h2>",
        "<div class='chart-container'>",
        fig_nav.to_html(full_html=False, include_plotlyjs="cdn"),
        "</div>",
        "<h2>2. Performance Metrics Summary</h2>",
        "<div class='chart-container'>",
        fig_table.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>3. Ex-Post Efficient Frontier</h2>",
        "<div class='chart-container'>",
        fig_ef.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>4. Portfolio Weights Heatmap</h2>",
        "<div class='chart-container'>",
        fig_hm.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>5. Turnover vs Sharpe</h2>",
        "<div class='chart-container'>",
        fig_scatter.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "</body></html>",
    ]

    html_content = "\n".join(html_parts)
    output_path = Path(output_path) if output_path is not None else default_dashboard_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Dashboard written to: %s", output_path)
    return str(output_path)


# ===================================================================
# Demo: synthetic data (CLI entry point)
# ===================================================================
def _make_synthetic_results() -> Dict[str, "BacktestResult"]:
    """Generate minimal synthetic BacktestResult objects for demo."""
    from src.backtest_engine import BacktestResult, BacktestConfig, DailyRecord

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2023-12-31")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]

    def _make_result(mu: float, vol: float, **strategy) -> BacktestResult:
        daily_rets = rng.normal(mu / 252, vol / np.sqrt(252), len(dates))
        nav_vals = INITIAL_CASH * np.cumprod(1 + daily_rets)
        nav = pd.Series(nav_vals, index=dates, name="nav")
        returns = pd.Series(daily_rets, index=dates, name="returns")

        n = len(dates)
        w_mat = rng.dirichlet(np.ones(len(tickers)), size=n)
        weights = pd.DataFrame(w_mat, index=dates, columns=tickers)

        config = BacktestConfig(**strategy)
        daily_records = [
            DailyRecord(
                date=d, nav=float(nav.iloc[i]),
                daily_return=float(daily_rets[i]),
                weights=weights.iloc[i],
                is_rebalance=(i % 63 == 0),
                turnover=0.15 if (i % 63 == 0) else 0.0,
                cost=0.0,
            )
            for i, d in enumerate(dates)
        ]
        return BacktestResult(
            config=config,
            nav=nav,
            returns=returns,
            weights=weights,
            rebalance_log=pd.DataFrame(),
            daily_records=daily_records,
        )

    # Etiqueta del dashboard -> config real (opt_method se valida al construir).
    return {
        "mv_classic":  _make_result(mu=0.10, vol=0.15, opt_method="mv_classic"),
        "robust_k010": _make_result(mu=0.09, vol=0.13, opt_method="robust",
                                    kappa=0.10),
        "robust_k025": _make_result(mu=0.08, vol=0.12, opt_method="robust",
                                    kappa=0.25),
    }


def _make_synthetic_benchmarks(dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """Precios sinteticos SPY / ICOLCAP para la demo (sin red)."""
    rng = np.random.default_rng(7)
    spy = 400 * np.cumprod(1 + rng.normal(0.09 / 252, 0.16 / np.sqrt(252), len(dates)))
    ico = 15000 * np.cumprod(1 + rng.normal(0.06 / 252, 0.20 / np.sqrt(252), len(dates)))
    return {
        "SPY": pd.Series(spy, index=dates, name="SPY"),
        "ICOLCAP": pd.Series(ico, index=dates, name="ICOLCAP"),
    }


if __name__ == "__main__":
    synthetic = _make_synthetic_results()
    first = next(iter(synthetic.values()))
    benchmarks = _make_synthetic_benchmarks(first.nav.index)
    path = build_dashboard(synthetic, benchmark_prices=benchmarks)
    print(f"Dashboard saved to: {path}")
