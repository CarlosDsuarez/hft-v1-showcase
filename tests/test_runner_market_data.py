"""
test_runner_market_data.py — El runner lee datos solo via el proveedor unico.

Una descarga por corrida, politica live (faltante → abort, barra intradia →
descartada, dato atrasado → abort) y flags de configuracion. FakeProvider:
sin red, estado en tmp_path via HFT_HOME.
"""

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.market_data import CachedProvider, DataPolicy, FallbackChain
from src.settings import load_settings
from tests.test_runner_paths import TICKERS, _fake_prices, after_close, use_market_data

NY = "America/New_York"
STATE_FILES = ("ledger.json", "last_rebalance.txt", "latest_snapshot.csv",
               "rebalance_history.csv")


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    return home / "data" / "live_state"


def _written(state_dir):
    return [n for n in STATE_FILES if (state_dir / n).exists()]


def test_one_download_per_run(state_dir, monkeypatch):
    provider = use_market_data(monkeypatch, _fake_prices())
    assert runner.run_once(force=True, start="2024-01-01") == 0
    assert len(provider.calls) == 1
    assert set(provider.calls[0][0]) == set(TICKERS)


def test_missing_symbol_aborts_without_writing(state_dir, monkeypatch, capsys):
    use_market_data(monkeypatch, _fake_prices().drop(columns="FFF"))
    assert runner.run_once(force=True, start="2024-01-01") == runner.EXIT_DATA
    assert "FFF" in capsys.readouterr().err
    assert _written(state_dir) == []


def test_missing_symbol_tolerated_with_warn_policy(state_dir, monkeypatch):
    use_market_data(monkeypatch, _fake_prices().drop(columns="FFF"))
    rc = runner.run_once(force=True, start="2024-01-01",
                         data_policy=DataPolicy(on_missing="warn"))
    assert rc == 0
    snap = pd.read_csv(state_dir / "latest_snapshot.csv")
    assert "FFF" not in set(snap.iloc[:, 0])


def test_intraday_bar_is_dropped_before_as_of(state_dir, monkeypatch):
    px = _fake_prices()
    last, prev = px.index[-1], px.index[-2]
    intraday = pd.Timestamp(f"{last.date()} 11:00", tz=NY).tz_convert("UTC")
    use_market_data(monkeypatch, px, now=intraday.to_pydatetime())
    assert runner.run_once(force=True, start="2024-01-01") == 0
    as_of = (state_dir / "last_rebalance.txt").read_text().strip()
    assert pd.Timestamp(as_of) == prev


def test_stale_data_aborts(state_dir, monkeypatch):
    px = _fake_prices()
    later = after_close(px.index[-1] + pd.offsets.BDay(3))
    use_market_data(monkeypatch, px, now=later)
    assert runner.run_once(force=True, start="2024-01-01") == runner.EXIT_DATA
    assert _written(state_dir) == []


def test_check_only_reports_data_error(state_dir, monkeypatch):
    use_market_data(monkeypatch, _fake_prices().drop(columns="AAA"))
    assert runner.run_once(check_only=True, start="2024-01-01") == runner.EXIT_DATA


class TestProviderConfig:
    def test_default_chain_is_yahoo_then_csv(self, tmp_path):
        s = load_settings(env={"HFT_HOME": str(tmp_path)})
        chain = runner.make_market_data_provider(s).inner  # tras el maestro
        assert isinstance(chain, FallbackChain)
        assert [p.name for p in chain.providers] == ["yahoo", "csv"]

    def test_cache_lives_in_settings_cache_dir(self, tmp_path):
        s = load_settings(env={"HFT_HOME": str(tmp_path)})
        provider = runner.make_market_data_provider(s, sources=("csv",), cache=True)
        assert isinstance(provider.inner, CachedProvider)
        assert provider.inner.cache_dir == s.bars_cache_dir

    def test_cli_flags_reach_run_once(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(runner, "run_once", lambda **kw: seen.update(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        assert runner.main([
            "--data-sources", "csv,yahoo", "--data-cache", "--on-missing", "warn",
            "--on-partial-bar", "abort", "--on-stale", "warn",
        ]) == 0
        assert seen["data_sources"] == ("csv", "yahoo")
        assert seen["data_cache"] is True
        assert seen["data_policy"] == DataPolicy(
            on_missing="warn", on_partial_bar="abort", on_stale="warn")

    def test_cli_defaults_are_live_policy(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(runner, "run_once", lambda **kw: seen.update(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        assert runner.main([]) == 0
        assert seen["data_sources"] == ("yahoo", "csv")
        assert seen["data_cache"] is False
        assert seen["data_policy"] == DataPolicy()
