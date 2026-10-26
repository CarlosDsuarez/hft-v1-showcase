"""
data_client.py — T-01..T-05: universo, descarga, limpieza y liquidez.

Descarga: proveedor único de datos de mercado (src/market_data.py).
`fetch_bars` trae cierres ajustados + volumen en UNA descarga OHLCV por la
cadena configurable (default: Yahoo con retry/backoff → CSV local
``<data_dir>/raw_prices_backup.csv``) y aplica la política de datos
(símbolos faltantes, barra intradía, frescura). Por defecto la cadena va
detrás del maestro de instrumentos (src/instruments.py): se pide por id
(ICOLCAP), se descarga el símbolo Yahoo (ICOLCAP.CL) y los cierres salen en
moneda base (src.fx.to_base con la serie FX del proveedor). `fetch_prices` /
`fetch_volumes` son envoltorios de compatibilidad (política de investigación:
avisan en vez de abortar).

Restricciones cubiertas: R-05 (datos para rolling window), R-10 (modular).
"""

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.instruments import load_instruments
from src.market_data import (
    LIVE_POLICY, RESEARCH_POLICY, Bars, DataPolicy, InstrumentProvider,
    MarketDataProvider, build_provider, close_in_base, load_bars,
)

logger = logging.getLogger("data_client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FFILL_MAX_DAYS = 2
MIN_HISTORY_YEARS = 5
# Ventana default de descarga Yahoo: solo últimos 3 meses por activo
from src.settings import LOOKBACK_DAYS as DEFAULT_LOOKBACK_DAYS  # ~3.5 meses calendario
from src.settings import load_settings
from src.position_ledger import MissingPriceError


def default_3m_range() -> tuple[str, str]:
    """Rango default (start, end) = últimos 3 meses hasta mañana UTC.

    end = mañana (para incluir hoy en yfinance, cuyo `end` es exclusivo).
    start = end − DEFAULT_LOOKBACK_DAYS días calendario (src/settings.py).
    """
    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(DEFAULT_LOOKBACK_DAYS, unit="D")).strftime("%Y-%m-%d")
    return start, end


# ===================================================================
# T-01 helper: load universe
# ===================================================================
_HERE = Path(__file__).parent


def load_universe(path: str = None, market: Optional[str] = None) -> pd.DataFrame:
    """
    Load the asset universe from CSV or Excel.

    Args:
        path: Path to universe file (.csv or .xlsx).
              Defaults to universe.csv next to this script, or at the
              repository root (parent of ``src/``).
        market: Optional market filter (``"us"`` | ``"co"``). Only applies
              when the file has a ``market`` column (``universe_co.csv``,
              ``universe_mixed.csv``). ``None`` returns the full file.

    Returns:
        DataFrame with columns [ticker, asset_type, sector, name,
        inclusion_reason] plus [yahoo_ticker, market, currency, adr_proxy]
        when present in the file.
    """
    if path is None:
        # Look next to this module, then at repo root (CM Project/universe.csv)
        candidates = [
            _HERE / "universe.csv",
            _HERE / "universe.xlsx",
            _HERE.parent / "universe.csv",
            _HERE.parent / "universe.xlsx",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"No universe.csv or universe.xlsx found in {_HERE} "
                f"or {_HERE.parent}"
            )

    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    required_cols = {"ticker", "asset_type", "sector", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"universe.csv missing columns: {required_cols - set(df.columns)}"
        )
    logger.info(
        "Universe loaded: %d stocks, %d ETFs",
        (df["asset_type"] == "stock").sum(),
        (df["asset_type"] == "etf").sum(),
    )
    if market is not None:
        if "market" not in df.columns:
            raise ValueError(
                f"market={market!r} requested but {p} has no 'market' column "
                "(use universe_co.csv or universe_mixed.csv)."
            )
        df = df[df["market"] == market].reset_index(drop=True)
        logger.info("Universe filtered to market=%s: %d assets", market, len(df))
    return df


# ===================================================================
# T-02/T-03: descarga via proveedor único (src/market_data.py)
# ===================================================================
def _default_provider(csv_path: Optional[str] = None) -> MarketDataProvider:
    """Cadena Yahoo → CSV detrás del maestro de instrumentos (ids ↔ Yahoo)."""
    return InstrumentProvider(build_provider(csv_path=csv_path), load_instruments())


def fetch_bars(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    provider: Optional[MarketDataProvider] = None,
    policy: DataPolicy = LIVE_POLICY,
    now: Optional[datetime] = None,
) -> Bars:
    """
    Cierres ajustados + volumen en una sola descarga OHLCV.

    Args:
        tickers: Ids del maestro de instrumentos (ticker del universo).
        start: Start date. Default: últimos 3 meses (DEFAULT_LOOKBACK_DAYS).
        end: End date (exclusivo). Default: mañana UTC (incluye hoy).
        provider: Fuente. Default: build_provider() (Yahoo → CSV en data_dir)
            detrás de InstrumentProvider(load_instruments()). Con un
            InstrumentProvider los cierres se convierten a moneda base; con
            otro proveedor se devuelven tal cual.
        policy: Faltantes / barra intradía / frescura. Default LIVE_POLICY
            (símbolo faltante o dato atrasado → error); investigación:
            RESEARCH_POLICY.
        now: Reloj para la frescura (default: ahora).

    Returns:
        Bars(close, volume, meta) — ver src/market_data.py. Columnas = ids.

    Raises:
        MarketDataError (RuntimeError): sin datos ("All data sources
            exhausted"), faltantes, barra parcial o datos atrasados según
            `policy`.
        MissingFxError: moneda local sin serie FX.
    """
    if start is None or end is None:
        _d_start, _d_end = default_3m_range()
        start = start or _d_start
        end = end or _d_end
    provider = provider or _default_provider()
    bars = load_bars(provider, tickers, start, end, policy=policy, now=now)
    close, _ = close_in_base(provider, bars.close, start, end)
    return replace(bars, close=close)


def fetch_prices(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    csv_fallback_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compat: solo los cierres de `fetch_bars` (RESEARCH_POLICY).

    Para precios y volúmenes usar `fetch_bars` (una sola descarga).

    Args:
        tickers: Ticker symbols.
        start: Start date. Default: últimos 3 meses.
        end: End date. Default: mañana UTC (incluye hoy).
        csv_fallback_path: CSV del fallback. Default:
            ``<data_dir>/raw_prices_backup.csv`` (src/settings.py; nunca el cwd).

    Returns:
        DataFrame of adjusted close prices (date index × tickers).

    Raises:
        RuntimeError: If all sources fail.
    """
    provider = _default_provider(csv_fallback_path)
    return fetch_bars(tickers, start, end, provider=provider,
                      policy=RESEARCH_POLICY).close


# ===================================================================
# T-04: Data cleaning and validation
# ===================================================================
def clean_prices(
    prices: pd.DataFrame,
    ffill_max: int = FFILL_MAX_DAYS,
    sigma_threshold: float = 5.0,
    must_keep: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean price data: handle NaN, outliers, alignment.

    Args:
        prices: Raw price DataFrame.
        ffill_max: Max consecutive days for forward-fill.
        sigma_threshold: Outlier threshold in standard deviations
            applied to log-returns.
        must_keep: Tickers never dropped (e.g. open positions). If one
            would be dropped (>5% NaN or residual NaN) it is kept, aligned
            to the clean index, *without* affecting the rows/columns kept
            for the rest (it may carry NaN). Flagged in
            ``kept_by_must_keep``.

    Returns:
        Tuple of (clean_prices, quality_report).
        quality_report has columns: ticker, total_rows, missing_pct,
        outliers_flagged, first_date, last_date, completeness_ok,
        kept_by_must_keep.

    Raises:
        MissingPriceError: A ``must_keep`` ticker is absent from ``prices``
            or has no data at all (it cannot be kept).

    Restricciones: R-05 (temporal alignment), R-09 (no look-ahead).
    """
    must_keep = list(dict.fromkeys(must_keep or []))
    absent = [tk for tk in must_keep
              if tk not in prices.columns or prices[tk].notna().sum() == 0]
    if absent:
        raise MissingPriceError(
            absent, "must_keep tickers have no price data to keep.")

    logger.info("Cleaning prices: %d tickers, %d rows", prices.shape[1], prices.shape[0])

    # --- Step 1: forward-fill limited ---
    filled = prices.ffill(limit=ffill_max)

    # --- Step 2: detect outliers on log-returns ---
    log_rets = np.log(filled / filled.shift(1))
    mu = log_rets.mean()
    sigma = log_rets.std()
    outlier_mask = (log_rets - mu).abs() > sigma_threshold * sigma

    # Replace outlier returns with NaN, then ffill price
    outlier_counts = outlier_mask.sum()
    # Don't remove outliers — flag only. Replace price at outlier points
    # with interpolation to avoid data loss.
    clean = filled.copy()
    for col in clean.columns:
        if outlier_counts[col] > 0:
            bad_idx = outlier_mask.index[outlier_mask[col]]
            clean.loc[bad_idx, col] = np.nan
            clean[col] = clean[col].interpolate(method="linear", limit=2)
            logger.info(
                "Ticker %s: %d outliers interpolated", col, outlier_counts[col]
            )

    # --- Step 3: drop tickers with too many NaN ---
    missing_pct = clean.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    rescued = [tk for tk in bad_tickers if tk in must_keep]
    rescued_cols = clean[rescued]
    dropped = [tk for tk in bad_tickers if tk not in rescued]
    if dropped:
        logger.warning(
            "Dropping tickers with >5%% missing: %s", dropped
        )
    if rescued:
        logger.warning(
            "Keeping must_keep tickers with >5%% missing (may hold NaN): %s",
            rescued,
        )
    if bad_tickers:
        clean = clean.drop(columns=bad_tickers)

    # --- Step 4: residual NaNs — drop incomplete tickers, never dates ---
    # dropna() on rows would remove calendar sessions and make the next
    # surviving dates adjacent. Downstream pct_change / log-returns then
    # treat multi-session gaps as single-day moves (silent return corruption).
    still_nan = clean.columns[clean.isna().any()].tolist()
    kept_nan = [tk for tk in still_nan if tk in must_keep]
    still_nan = [tk for tk in still_nan if tk not in must_keep]
    if still_nan:
        logger.warning(
            "Dropping tickers with residual NaNs after ffill/interpolate: %s",
            still_nan,
        )
        clean = clean.drop(columns=still_nan)
    if kept_nan:
        logger.warning(
            "Keeping must_keep tickers with residual NaNs: %s", kept_nan,
        )
    # must_keep columns never decide which dates survive: set them aside,
    # trim the rest, re-attach on that index (their gaps stay visible). With
    # nothing else left, keep every date instead of wiping or hiding them.
    rescued = rescued + kept_nan
    rescued_cols = pd.concat([rescued_cols, clean[kept_nan]], axis=1)
    clean = clean.drop(columns=kept_nan)
    if len(clean.columns) or not rescued:
        clean = clean.dropna(how="all")
    if rescued:
        clean = clean.join(rescued_cols)
        clean = clean[[c for c in prices.columns if c in clean.columns]]

    # --- Quality report ---
    report_rows = []
    for col in prices.columns:
        total = len(prices)
        missing = prices[col].isna().sum()
        report_rows.append({
            "ticker": col,
            "total_rows": total,
            "missing_count": int(missing),
            "missing_pct": round(missing / total * 100, 2),
            "outliers_flagged": int(outlier_counts.get(col, 0)),
            "first_date": str(prices[col].first_valid_index()),
            "last_date": str(prices[col].last_valid_index()),
            "in_clean_set": col in clean.columns,
            "kept_by_must_keep": col in rescued,
        })
    quality_report = pd.DataFrame(report_rows)

    logger.info(
        "Clean prices: %d tickers retained, %d rows",
        clean.shape[1], clean.shape[0],
    )
    return clean, quality_report


# ===================================================================
# T-04b: Calculate log-returns (helper for downstream)
# ===================================================================
def calc_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns from price DataFrame.

    Args:
        prices: Clean adjusted close prices.

    Returns:
        DataFrame of log-returns (first row dropped).
    """
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets


# ===================================================================
# T-05: Liquidity scoring — Amihud illiquidity ratio
# ===================================================================
def calc_liquidity_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Calculate liquidity scores: average daily volume (30d) and
    Amihud illiquidity ratio per ticker.

    Amihud(i) = mean(|r_t| / volume_t) over trailing window.
    Lower Amihud → more liquid.

    Args:
        prices: Clean adjusted close prices.
        volumes: Volume DataFrame aligned with prices.
        window: Lookback window in trading days.

    Returns:
        DataFrame with columns [ticker, avg_volume_30d,
        amihud_illiquidity, liquidity_rank].

    Restricciones: R-03 (liquidez mínima).
    """
    log_rets = np.log(prices / prices.shift(1)).iloc[-window:]
    vol_window = volumes.iloc[-window:]

    records = []
    for ticker in prices.columns:
        if ticker not in volumes.columns:
            logger.warning("No volume data for %s — skipping", ticker)
            continue

        avg_vol = vol_window[ticker].mean()
        # Amihud: mean(|return| / dollar_volume)
        abs_ret = log_rets[ticker].abs()
        dollar_vol = vol_window[ticker] * prices[ticker].iloc[-window:]
        # Avoid division by zero
        safe_dvol = dollar_vol.replace(0, np.nan)
        amihud = (abs_ret / safe_dvol).mean()

        records.append({
            "ticker": ticker,
            "avg_volume_30d": round(avg_vol, 0),
            "amihud_illiquidity": round(amihud, 12),
        })

    scores = pd.DataFrame(records)
    # Rank: lower Amihud = better liquidity = rank 1
    scores["liquidity_rank"] = scores["amihud_illiquidity"].rank(method="min")
    scores = scores.sort_values("liquidity_rank").reset_index(drop=True)

    logger.info("Liquidity scores computed for %d tickers", len(scores))
    return scores


def fetch_volumes(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compat: solo los volúmenes de `fetch_bars` (RESEARCH_POLICY).

    Para precios y volúmenes usar `fetch_bars` (una sola descarga).

    Args:
        tickers: Ticker symbols.
        start: Start date. Default: últimos 3 meses.
        end: End date. Default: mañana UTC.

    Returns:
        DataFrame of daily volumes (date index × tickers).
    """
    return fetch_bars(tickers, start, end, policy=RESEARCH_POLICY).volume


# ===================================================================
# T-03 + T-04 + T-05: Integrated pipeline runner
# ===================================================================
def run_data_pipeline(
    universe_path: str = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    output_dir: str = "data",
    provider: Optional[MarketDataProvider] = None,
) -> dict:
    """
    Execute the full Phase 1 data pipeline:
      1. Load universe
      2. Fetch prices and volumes
      3. Clean prices
      4. Compute liquidity scores
      5. Save all artifacts

    Args:
        universe_path: Path to universe CSV.
        start: Backtest start date. Default: últimos 3 meses.
        end: Backtest end date. Default: mañana UTC.
        output_dir: Directory for output parquet/CSV files.
        provider: Fuente de datos (default: build_provider()).

    Returns:
        Dict with keys: universe, clean_prices, quality_report,
        liquidity_scores, log_returns.
    """
    if start is None or end is None:
        _d_start, _d_end = default_3m_range()
        start = start or _d_start
        end = end or _d_end
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Universe
    universe = load_universe(universe_path)
    tickers = universe["ticker"].tolist()

    # 2. Fetch
    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA PIPELINE START")
    logger.info("=" * 60)

    bars = fetch_bars(tickers, start, end, provider=provider,
                      policy=RESEARCH_POLICY)
    prices, volumes = bars.close, bars.volume

    # 3. Clean
    clean, quality_report = clean_prices(prices)

    # 4. Liquidity
    # Align volumes to clean prices columns
    common_tickers = [t for t in clean.columns if t in volumes.columns]
    liq_scores = calc_liquidity_scores(
        clean[common_tickers], volumes[common_tickers], window=30
    )

    # 5. Log-returns
    log_rets = calc_log_returns(clean)

    # 6. Save artifacts
    prices.to_parquet(out / "raw_prices.parquet")
    clean.to_parquet(out / "clean_prices.parquet")
    quality_report.to_csv(out / "data_quality_report.csv", index=False)
    liq_scores.to_csv(out / "liquidity_scores.csv", index=False)
    log_rets.to_parquet(out / "log_returns.parquet")

    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA PIPELINE COMPLETE")
    logger.info("Artifacts saved to %s/", output_dir)
    logger.info("=" * 60)

    return {
        "universe": universe,
        "clean_prices": clean,
        "quality_report": quality_report,
        "liquidity_scores": liq_scores,
        "log_returns": log_rets,
    }


if __name__ == "__main__":
    from src.settings import setup_logging
    setup_logging()
    results = run_data_pipeline()
    print("\n--- Quality Report ---")
    print(results["quality_report"].to_string(index=False))
    print("\n--- Liquidity Scores (top 10) ---")
    print(results["liquidity_scores"].head(10).to_string(index=False))
