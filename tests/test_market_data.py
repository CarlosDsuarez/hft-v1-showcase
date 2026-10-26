"""
test_market_data.py — Contrato del proveedor unico de datos de mercado.

100% offline: FakeProvider, downloaders Yahoo inyectados y CSV sinteticos.
Nunca toca la red. Los relojes se fijan en hora de Nueva York.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.market_data import (
    DEFAULT_SOURCES,
    LIVE_POLICY,
    Bars,
    CachedProvider,
    CsvProvider,
    DataPolicy,
    DataUnavailableError,
    FakeProvider,
    FallbackChain,
    MissingSymbolsError,
    PartialBarError,
    StaleDataError,
    YahooProvider,
    build_provider,
    last_closed_session,
    load_bars,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NY = "America/New_York"
SYMS = ["AAA", "BBB", "CCC"]
START = "2026-09-08"  # martes tras Labor Day: ventana sin feriados NYSE


def _ny(ts: str) -> datetime:
    """Hora local de Nueva York -> datetime UTC-aware."""
    return pd.Timestamp(ts, tz=NY).tz_convert("UTC").to_pydatetime()


AFTER_CLOSE = _ny("2026-09-30 16:30")  # miercoles, sesion 09-30 cerrada
INTRADAY = _ny("2026-09-30 11:00")     # misma sesion, mercado abierto


def _frames(start=START, end="2026-09-30", symbols=SYMS, seed=0):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), len(symbols))), axis=0)),
        index=idx, columns=symbols)
    volume = pd.DataFrame(1e6, index=idx, columns=symbols)
    return close, volume


class _Broken:
    name = "broken"

    def get_bars(self, symbols, start, end):
        raise ConnectionError("network down")


# ===================================================================
# Calendario: ultima sesion cerrada (America/New_York)
# ===================================================================
class TestLastClosedSession:
    def test_after_close_is_today(self):
        assert last_closed_session(AFTER_CLOSE) == pd.Timestamp("2026-09-30")

    def test_before_close_is_previous_session(self):
        assert last_closed_session(INTRADAY) == pd.Timestamp("2026-09-29")

    def test_weekend_is_friday(self):
        assert last_closed_session(_ny("2026-10-03 12:00")) == pd.Timestamp("2026-10-02")

    def test_nyse_holidays_are_skipped(self):
        # Good Friday y Labor Day no son sesiones NYSE.
        assert last_closed_session(_ny("2026-04-03 18:00")) == pd.Timestamp("2026-04-02")
        assert last_closed_session(_ny("2026-09-08 09:00")) == pd.Timestamp("2026-09-04")

    def test_utc_clock_is_converted_to_new_york(self):
        # 21:00 UTC = 17:00 EDT (cerrado); 19:00 UTC = 15:00 EDT (abierto)
        closed = datetime(2026, 9, 30, 21, 0, tzinfo=timezone.utc)
        open_ = datetime(2026, 9, 30, 19, 0, tzinfo=timezone.utc)
        assert last_closed_session(closed) == pd.Timestamp("2026-09-30")
        assert last_closed_session(open_) == pd.Timestamp("2026-09-29")


# ===================================================================
# Bars: una descarga OHLCV -> close + volume + meta
# ===================================================================
class TestBarsContract:
    def test_fake_provider_returns_close_volume_and_meta(self):
        close, volume = _frames()
        bars = FakeProvider(close, volume).get_bars(SYMS, START, "2026-10-01")
        assert isinstance(bars, Bars)
        assert list(bars.close.columns) == SYMS
        pd.testing.assert_index_equal(bars.volume.index, bars.close.index)
        assert list(bars.volume.columns) == SYMS
        assert bars.meta.source == "fake"
        assert bars.meta.requested == tuple(SYMS)
        assert bars.meta.received == tuple(SYMS)
        assert bars.meta.missing == ()
        assert bars.meta.last_date["AAA"] == pd.Timestamp("2026-09-30")

    def test_end_is_exclusive(self):
        close, volume = _frames()
        bars = FakeProvider(close, volume).get_bars(SYMS, START, "2026-09-30")
        assert bars.close.index[-1] == pd.Timestamp("2026-09-29")

    def test_meta_reports_requested_vs_received(self):
        close, volume = _frames(symbols=["AAA", "BBB"])
        bars = FakeProvider(close, volume).get_bars(SYMS, START, "2026-10-01")
        assert bars.meta.received == ("AAA", "BBB")
        assert bars.meta.missing == ("CCC",)


# ===================================================================
# Politica de simbolos faltantes
# ===================================================================
class TestMissingSymbolsPolicy:
    def test_live_default_is_error(self):
        assert LIVE_POLICY.on_missing == "error"
        assert DataPolicy().on_missing == "error"

    def test_missing_symbol_raises_by_default(self):
        close, volume = _frames(symbols=["AAA", "BBB"])
        with pytest.raises(MissingSymbolsError, match="CCC") as exc:
            load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                      now=AFTER_CLOSE)
        assert exc.value.missing == ("CCC",)

    def test_missing_symbol_warns_when_configured(self, caplog):
        close, volume = _frames(symbols=["AAA", "BBB"])
        with caplog.at_level(logging.WARNING, logger="market_data"):
            bars = load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                             policy=DataPolicy(on_missing="warn"), now=AFTER_CLOSE)
        assert list(bars.close.columns) == ["AAA", "BBB"]
        assert bars.meta.missing == ("CCC",)
        assert "CCC" in caplog.text

    def test_unknown_policy_value_rejected(self):
        with pytest.raises(ValueError, match="on_missing"):
            DataPolicy(on_missing="ignore")


# ===================================================================
# Frescura: ultima barra = ultima sesion cerrada
# ===================================================================
class TestFreshness:
    def test_intraday_bar_is_dropped(self):
        close, volume = _frames()  # incluye la barra parcial de 2026-09-30
        bars = load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                         now=INTRADAY)
        assert bars.close.index[-1] == pd.Timestamp("2026-09-29")
        assert bars.volume.index[-1] == pd.Timestamp("2026-09-29")
        assert bars.meta.dropped_partial == (pd.Timestamp("2026-09-30"),)
        assert bars.meta.expected_session == pd.Timestamp("2026-09-29")
        assert bars.meta.last_date["AAA"] == pd.Timestamp("2026-09-29")

    def test_intraday_bar_aborts_when_configured(self):
        close, volume = _frames()
        with pytest.raises(PartialBarError, match="2026-09-30"):
            load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                      policy=DataPolicy(on_partial_bar="abort"), now=INTRADAY)

    def test_closed_session_bar_is_kept(self):
        close, volume = _frames()
        bars = load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                         now=AFTER_CLOSE)
        assert bars.close.index[-1] == pd.Timestamp("2026-09-30")
        assert bars.meta.dropped_partial == ()

    def test_stale_data_raises(self):
        close, volume = _frames(end="2026-09-28")
        with pytest.raises(StaleDataError, match="2026-09-30"):
            load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                      now=AFTER_CLOSE)

    def test_single_stale_symbol_raises(self):
        close, volume = _frames()
        close.loc["2026-09-29":, "CCC"] = np.nan
        with pytest.raises(StaleDataError, match="CCC") as exc:
            load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                      now=AFTER_CLOSE)
        assert exc.value.stale == ("CCC",)

    def test_stale_data_warns_when_configured(self):
        close, volume = _frames(end="2026-09-28")
        bars = load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                         policy=DataPolicy(on_stale="warn"), now=AFTER_CLOSE)
        assert bars.meta.stale == tuple(SYMS)

    def test_historical_window_is_not_checked_for_staleness(self):
        close, volume = _frames(end="2026-09-28")
        bars = load_bars(FakeProvider(close, volume), SYMS, START, "2026-09-29",
                         now=AFTER_CLOSE)
        assert bars.close.index[-1] == pd.Timestamp("2026-09-28")

    def test_symbol_with_only_partial_bar_is_missing(self):
        close, volume = _frames()
        close.loc[:"2026-09-29", "CCC"] = np.nan
        with pytest.raises(MissingSymbolsError, match="CCC"):
            load_bars(FakeProvider(close, volume), SYMS, START, "2026-10-01",
                      now=INTRADAY)


# ===================================================================
# Cadena de fallback
# ===================================================================
class TestFallbackChain:
    def test_falls_back_to_next_provider(self):
        close, volume = _frames()
        backup = FakeProvider(close, volume, name="backup")
        bars = load_bars(FallbackChain([_Broken(), backup]), SYMS, START,
                         "2026-10-01", now=AFTER_CLOSE)
        assert bars.meta.source == "backup"
        assert list(bars.close.columns) == SYMS

    def test_first_working_provider_wins(self):
        close, volume = _frames()
        primary = FakeProvider(close, volume, name="primary")
        backup = FakeProvider(close, volume, name="backup")
        bars = FallbackChain([primary, backup]).get_bars(SYMS, START, "2026-10-01")
        assert bars.meta.source == "primary"
        assert backup.calls == []

    def test_all_sources_failing_raises(self):
        empty = FakeProvider(*_frames(symbols=["ZZZ"]), name="empty")
        with pytest.raises(DataUnavailableError,
                           match="All data sources exhausted.*broken.*empty"):
            FallbackChain([_Broken(), empty]).get_bars(SYMS, START, "2026-10-01")


# ===================================================================
# Yahoo: una sola descarga OHLCV con retry/backoff (downloader inyectado)
# ===================================================================
def _yahoo_raw(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Formato de yf.download con varios tickers: columnas (Price, Ticker)."""
    raw = pd.concat({"Close": close, "High": close, "Low": close,
                     "Open": close, "Volume": volume}, axis=1)
    raw.columns.names = ["Price", "Ticker"]
    raw.index.name = "Date"
    return raw


class _Downloader:
    """Simula yf.download: respuestas/excepciones en orden; registra llamadas."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, symbols, start, end):
        self.calls.append(list(symbols))
        out = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(out, Exception):
            raise out
        return out


class TestYahooProvider:
    def test_single_download_for_close_and_volume(self):
        close, volume = _frames()
        dl = _Downloader(_yahoo_raw(close, volume))
        bars = YahooProvider(downloader=dl, sleep=lambda s: None).get_bars(
            SYMS, START, "2026-10-01")
        assert len(dl.calls) == 1
        pd.testing.assert_frame_equal(bars.close, close, check_names=False,
                                      check_freq=False)
        pd.testing.assert_frame_equal(bars.volume, volume, check_names=False,
                                      check_freq=False)
        assert bars.meta.source == "yahoo"

    def test_retries_with_exponential_backoff(self):
        close, volume = _frames()
        sleeps = []
        dl = _Downloader(ConnectionError("x"), ConnectionError("y"),
                         _yahoo_raw(close, volume))
        bars = YahooProvider(downloader=dl, sleep=sleeps.append).get_bars(
            SYMS, START, "2026-10-01")
        assert len(dl.calls) == 3
        assert sleeps == [2.0, 4.0]
        assert bars.meta.received == tuple(SYMS)

    def test_retries_only_missing_symbols(self):
        close, volume = _frames()
        partial = close.copy()
        partial["CCC"] = np.nan
        dl = _Downloader(_yahoo_raw(partial, volume),
                         _yahoo_raw(close[["CCC"]], volume[["CCC"]]))
        bars = YahooProvider(downloader=dl, sleep=lambda s: None).get_bars(
            SYMS, START, "2026-10-01")
        assert dl.calls == [SYMS, ["CCC"]]
        assert bars.meta.received == tuple(SYMS)
        pd.testing.assert_series_equal(bars.close["CCC"], close["CCC"],
                                       check_names=False, check_freq=False)

    def test_exhausted_attempts_raise_without_trailing_sleep(self):
        sleeps = []
        dl = _Downloader(ConnectionError("down"))
        with pytest.raises(DataUnavailableError, match="yahoo"):
            YahooProvider(downloader=dl, sleep=sleeps.append, max_retries=3).get_bars(
                SYMS, START, "2026-10-01")
        assert len(dl.calls) == 3
        assert len(sleeps) == 2

    def test_single_ticker_flat_columns(self):
        close, volume = _frames(symbols=["AAA"])
        raw = pd.DataFrame({"Open": close["AAA"], "Close": close["AAA"],
                            "Volume": volume["AAA"]})
        bars = YahooProvider(downloader=_Downloader(raw), sleep=lambda s: None).get_bars(
            ["AAA"], START, "2026-10-01")
        assert list(bars.close.columns) == ["AAA"]
        assert list(bars.volume.columns) == ["AAA"]


# ===================================================================
# CSV: ruta configurable, anclada a data_dir por defecto
# ===================================================================
def _write_wide(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("date").reset_index().to_csv(path, index=False)
    return path


class TestCsvProvider:
    def test_configurable_path_without_volume_file(self, tmp_path):
        close, _ = _frames()
        path = _write_wide(tmp_path / "px.csv", close)
        bars = CsvProvider(close_path=path).get_bars(["AAA", "BBB"], START, "2026-10-01")
        pd.testing.assert_frame_equal(bars.close, close[["AAA", "BBB"]],
                                      check_names=False, check_freq=False)
        assert bars.volume.isna().all().all()
        assert bars.meta.source == "csv"

    def test_volume_file_is_used(self, tmp_path):
        close, volume = _frames()
        cp = _write_wide(tmp_path / "px.csv", close)
        vp = _write_wide(tmp_path / "vol.csv", volume * 3)
        bars = CsvProvider(close_path=cp, volume_path=vp).get_bars(SYMS, START, "2026-10-01")
        assert (bars.volume == 3e6).all().all()

    def test_default_path_is_data_dir_not_cwd(self, tmp_path, monkeypatch):
        close, _ = _frames()
        home = tmp_path / "home"
        _write_wide(home / "data" / "raw_prices_backup.csv", close)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("HFT_HOME", str(home))
        bars = CsvProvider().get_bars(SYMS, START, "2026-10-01")
        assert len(bars.close) == len(close)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataUnavailableError, match="not found"):
            CsvProvider(close_path=tmp_path / "nope.csv").get_bars(SYMS, START, "2026-10-01")


# ===================================================================
# Cache local parquet por simbolo
# ===================================================================
class TestCachedProvider:
    def test_repeat_request_served_from_cache(self, tmp_path):
        close, volume = _frames()
        inner = FakeProvider(close, volume)
        first = CachedProvider(inner, tmp_path, clock=lambda: AFTER_CLOSE).get_bars(
            SYMS, START, "2026-10-01")
        again = CachedProvider(inner, tmp_path, clock=lambda: AFTER_CLOSE).get_bars(
            SYMS, START, "2026-10-01")
        assert len(inner.calls) == 1
        assert again.meta.source == "cache"
        pd.testing.assert_frame_equal(again.close, first.close, check_freq=False)
        pd.testing.assert_frame_equal(again.volume, first.volume, check_freq=False)
        assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
            "AAA.parquet", "BBB.parquet", "CCC.parquet"]

    def test_only_new_bars_are_downloaded(self, tmp_path):
        close, volume = _frames()
        inner = FakeProvider(close, volume)
        CachedProvider(inner, tmp_path, clock=lambda: _ny("2026-09-25 17:00")).get_bars(
            SYMS, START, "2026-09-26")
        bars = CachedProvider(inner, tmp_path, clock=lambda: AFTER_CLOSE).get_bars(
            SYMS, START, "2026-10-01")
        assert len(inner.calls) == 2
        _, inc_start, _ = inner.calls[1]
        assert pd.Timestamp(inc_start) == pd.Timestamp("2026-09-25")  # solape 1 barra
        pd.testing.assert_frame_equal(bars.close, close, check_names=False,
                                      check_freq=False)

    def test_changed_adjusted_history_triggers_full_refetch(self, tmp_path):
        close, volume = _frames()
        CachedProvider(FakeProvider(close, volume), tmp_path,
                       clock=lambda: _ny("2026-09-25 17:00")).get_bars(
            SYMS, START, "2026-09-26")
        adjusted = close.copy()
        adjusted.loc[:"2026-09-28", "AAA"] /= 2.0  # split: Yahoo reajusta la historia
        inner = FakeProvider(adjusted, volume)
        bars = CachedProvider(inner, tmp_path, clock=lambda: AFTER_CLOSE).get_bars(
            SYMS, START, "2026-10-01")
        assert inner.calls[-1][0] == ("AAA",)
        assert pd.Timestamp(inner.calls[-1][1]) == pd.Timestamp(START)
        pd.testing.assert_frame_equal(bars.close, adjusted, check_names=False,
                                      check_freq=False)

    def test_partial_bar_is_returned_but_not_persisted(self, tmp_path):
        close, volume = _frames()
        bars = CachedProvider(FakeProvider(close, volume), tmp_path,
                              clock=lambda: INTRADAY).get_bars(SYMS, START, "2026-10-01")
        assert bars.close.index[-1] == pd.Timestamp("2026-09-30")
        cached = pd.read_parquet(tmp_path / "AAA.parquet")
        assert cached.index.max() == pd.Timestamp("2026-09-29")

    def test_symbol_file_names_are_sanitized(self, tmp_path):
        close, volume = _frames(symbols=["USDCOP=X", "ICOLCAP.CL"])
        CachedProvider(FakeProvider(close, volume), tmp_path,
                       clock=lambda: AFTER_CLOSE).get_bars(
            ["USDCOP=X", "ICOLCAP.CL"], START, "2026-10-01")
        assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
            "ICOLCAP.CL.parquet", "USDCOP_X.parquet"]


# ===================================================================
# Fabrica configurable
# ===================================================================
class TestBuildProvider:
    def test_default_chain_is_yahoo_then_csv(self):
        assert DEFAULT_SOURCES == ("yahoo", "csv")
        chain = build_provider()
        assert isinstance(chain, FallbackChain)
        assert [p.name for p in chain.providers] == ["yahoo", "csv"]

    def test_single_source_is_not_wrapped(self):
        assert isinstance(build_provider(["csv"]), CsvProvider)

    def test_cache_dir_wraps_chain(self, tmp_path):
        provider = build_provider(["csv"], cache_dir=tmp_path)
        assert isinstance(provider, CachedProvider)

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="alphavantage"):
            build_provider(["yahoo", "alphavantage"])


# ===================================================================
# Un unico punto de acceso a Yahoo
# ===================================================================
class TestSingleProvider:
    def test_yfinance_only_used_by_market_data(self):
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for d in ("src", "scripts", "experiments")
            for p in (REPO_ROOT / d).rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "market_data.py"
            and re.search(r"yf\.download|import yfinance", p.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_data_client_does_not_promise_missing_sources(self):
        text = (REPO_ROOT / "src" / "data_client.py").read_text(encoding="utf-8")
        assert "Alpha Vantage" not in text
        assert "IEX" not in text


# ===================================================================
# data_client: una descarga para precios + volumenes
# ===================================================================
class TestDataClientUsesProvider:
    def test_fetch_bars_single_provider_call(self):
        from src.data_client import fetch_bars
        close, volume = _frames()
        fake = FakeProvider(close, volume)
        bars = fetch_bars(SYMS, START, "2026-10-01", provider=fake, now=AFTER_CLOSE)
        assert len(fake.calls) == 1
        assert bars.meta.received == tuple(SYMS)
        assert (bars.volume == 1e6).all().all()

    def test_fetch_bars_defaults_to_live_policy(self):
        from src.data_client import fetch_bars
        close, volume = _frames(symbols=["AAA"])
        with pytest.raises(MissingSymbolsError):
            fetch_bars(SYMS, START, "2026-10-01", provider=FakeProvider(close, volume),
                       now=AFTER_CLOSE)

    def test_pipeline_downloads_once(self, tmp_path):
        from src.data_client import run_data_pipeline
        close, volume = _frames(start="2026-06-01")
        universe = tmp_path / "universe.csv"
        pd.DataFrame({"ticker": SYMS, "asset_type": "stock", "sector": "X",
                      "name": SYMS}).to_csv(universe, index=False)
        fake = FakeProvider(close, volume)
        out = run_data_pipeline(str(universe), start="2026-06-01", end="2026-10-01",
                                output_dir=str(tmp_path / "out"), provider=fake)
        assert len(fake.calls) == 1
        assert list(out["clean_prices"].columns) == SYMS
        assert set(out["liquidity_scores"]["ticker"]) == set(SYMS)
