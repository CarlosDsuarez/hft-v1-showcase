"""
test_runner_ledger.py — El runner periodico persiste el ledger solo via la
API publica (to_dict/from_dict) y nunca deja caja negativa tras rebalancear.

Sintetico, sin red, sin broker; estado en tmp_path via HFT_HOME.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
import src.position_ledger as pl
from src.position_ledger import PositionLedger
from src.strategy import load_strategy_config
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

RUNNER_SRC = Path(runner.__file__).read_text()


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    use_market_data(monkeypatch, _fake_prices())
    return home / "data" / "live_state"


def test_runner_does_not_touch_private_ledger_attributes():
    assert not re.search(r"ledger\._\w+", RUNNER_SRC)
    assert "_lots" not in RUNNER_SRC


def test_save_load_ledger_roundtrip_uses_public_api(tmp_path):
    lg = PositionLedger(initial_cash=5_000.0)
    px = pd.Series({"AAA": 10.0, "BBB": 20.0})
    lg.apply_trades(pd.DataFrame([{"ticker": "AAA", "shares": 100.0},
                                  {"ticker": "BBB", "shares": 50.0}]),
                    px, costs=1.0, t=pd.Timestamp("2024-01-02"))
    path = tmp_path / "ledger.json"
    runner.save_ledger(lg, path)
    assert json.loads(path.read_text()) == lg.to_dict()
    assert runner.load_ledger(path).to_dict() == lg.to_dict()


def test_runner_rebalances_keep_cash_non_negative(state_dir, monkeypatch):
    assert runner.run_once(force=True, start="2024-01-01") == 0
    # Precios relativos distintos (retornos iguales): pesos derivan → turnover > 0
    drift = pd.Series([1.1, 0.9, 1.0, 1.05, 0.95, 1.0], index=TICKERS)
    use_market_data(monkeypatch, _fake_prices() * drift)
    assert runner.run_once(force=True, start="2024-01-01") == 0
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert len(hist) == 2
    assert (hist["cost"] > 0).all()
    assert (hist["cash"] >= -pl.CASH_TOLERANCE).all(), hist["cash"].tolist()
    saved = json.loads((state_dir / "ledger.json").read_text())
    assert saved["schema_version"] == pl.LEDGER_SCHEMA_VERSION
    assert saved["cash"] >= -pl.CASH_TOLERANCE


def _composition(ledger: PositionLedger, prices_t: pd.Series) -> pd.Series:
    """Pesos invertidos normalizados a 1 (como el backtest, sin la caja)."""
    w = ledger.weights_from_positions(prices_t)
    return w / w.sum()


def test_sim_rebalance_from_held_book_respects_turnover_limit(state_dir):
    """R-02 en vivo: desde un libro invertido el runner rota a lo sumo
    turnover_limit (como run_backtest), no salta al objetivo completo."""
    px_t = _fake_prices().iloc[-1]
    lg = PositionLedger(initial_cash=100_000.0)
    # Todo en AAA al precio de t: PnL 0 → ninguna salida por dólares.
    lg.apply_trades(pd.DataFrame([{"ticker": "AAA",
                                   "shares": 100_000.0 / px_t["AAA"]}]),
                    px_t, t=pd.Timestamp("2024-04-01"))
    state_dir.mkdir(parents=True)
    runner.save_ledger(lg, state_dir / "ledger.json")
    limit = load_strategy_config().turnover_limit

    assert runner.run_once(force=True, start="2024-01-01") == 0

    after = runner.load_ledger(state_dir / "ledger.json")
    executed = _composition(after, px_t).sub(
        _composition(lg, px_t), fill_value=0.0).abs().sum()
    assert executed == pytest.approx(limit, abs=1e-6)
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert hist["turnover"].iloc[-1] == pytest.approx(limit, abs=1e-6)


def test_initial_deployment_from_cash_is_uncapped(state_dir):
    """Libro vacío: el despliegue inicial va al objetivo completo (sin tope)."""
    assert runner.run_once(force=True, start="2024-01-01") == 0
    hist = pd.read_csv(state_dir / "rebalance_history.csv")
    assert hist["turnover"].iloc[-1] == pytest.approx(1.0)
    after = runner.load_ledger(state_dir / "ledger.json")
    invested = after.weights_from_positions(_fake_prices().iloc[-1]).sum()
    assert invested == pytest.approx(1.0, abs=1e-3)   # solo queda caja = costos


def test_runner_upgrades_legacy_ledger_with_negative_cash(state_dir):
    state_dir.mkdir(parents=True)
    px0 = _fake_prices().iloc[-1]
    legacy = {
        "cash": -1383.0,
        "realized_pnl": 0.0,
        "lots": [
            {"ticker": tk, "shares": 1_000.0, "entry_price": float(px0[tk]),
             "entry_date": "2024-04-01T00:00:00"}
            for tk in TICKERS[:3]
        ],
    }
    (state_dir / "ledger.json").write_text(json.dumps(legacy))
    assert runner.run_once(force=True, start="2024-01-01") == 0
    saved = json.loads((state_dir / "ledger.json").read_text())
    assert saved["schema_version"] == pl.LEDGER_SCHEMA_VERSION
    assert saved["cash"] >= -pl.CASH_TOLERANCE
    assert len(saved["txns"]) > 0
