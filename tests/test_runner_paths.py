"""
test_runner_paths.py — El runner periodico escribe en settings.state_dir
aunque se ejecute desde otro cwd (datos sinteticos, sin red, sin broker).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.market_data import FakeProvider

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
# >= max required_history del stack (momentum 12-1 = 274 filas)
N_ROWS = 300


def _fake_prices(*_a, n_rows: int = N_ROWS, **_k):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=n_rows)
    data = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (len(idx), len(TICKERS))), axis=0))
    return pd.DataFrame(data, index=idx, columns=TICKERS)


def _fake_volumes(*_a, **_k):
    return pd.DataFrame(1e6, index=_fake_prices().index, columns=TICKERS)


def after_close(day) -> datetime:
    """Reloj UTC a las 17:00 de Nueva York de `day` (sesion ya cerrada)."""
    local = pd.Timestamp(f"{pd.Timestamp(day).date()} 17:00", tz="America/New_York")
    return local.tz_convert("UTC").to_pydatetime()


def use_market_data(monkeypatch, close, volume=None, now=None) -> FakeProvider:
    """El runner lee `close`/`volume` de un FakeProvider (sin red).

    Reloj por defecto: tras el cierre de la ultima barra, asi la barra es la
    ultima sesion cerrada y pasa el chequeo de frescura.
    """
    if volume is None:
        volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    provider = FakeProvider(close, volume)
    clock = now or after_close(close.index[-1])
    monkeypatch.setattr(runner, "make_market_data_provider", lambda *a, **k: provider)
    monkeypatch.setattr(runner, "utcnow", lambda: clock)
    return provider


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    use_market_data(monkeypatch, _fake_prices())
    return home, cwd


def test_runner_from_other_cwd_writes_state_dir(fake_runner):
    home, cwd = fake_runner
    rc = runner.run_once(force=True, start="2024-01-01")
    assert rc == 0
    state = home / "data" / "live_state"
    for name in ("ledger.json", "last_rebalance.txt",
                 "latest_snapshot.csv", "rebalance_history.csv"):
        assert (state / name).exists(), name
    assert not (cwd / "data").exists()


def test_runner_explicit_settings_override(fake_runner, tmp_path):
    from src.settings import load_settings
    home, cwd = fake_runner
    custom = load_settings(env={"HFT_HOME": str(tmp_path / "custom")})
    assert runner.run_once(force=True, start="2024-01-01", settings=custom) == 0
    assert (custom.state_dir / "ledger.json").exists()
    assert not (home / "data").exists()
    assert not (cwd / "data").exists()


def test_runner_fresh_ledger_uses_settings_initial_cash(fake_runner, tmp_path):
    from src.settings import INITIAL_CASH
    ledger = runner.load_ledger(Path(tmp_path / "missing.json"))
    assert ledger.cash == INITIAL_CASH


def test_runner_leaves_global_rng_untouched(fake_runner):
    np.random.seed(2024)
    before = np.random.get_state()
    assert runner.run_once(force=True, start="2024-01-01") == 0
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
