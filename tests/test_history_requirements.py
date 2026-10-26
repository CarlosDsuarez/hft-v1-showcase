"""
test_history_requirements.py — Cada componente declara su historia mínima
(required_history) y el runner la respeta: strict → error claro, permissive →
excluye factores y lo registra; nunca rellena con 0 en silencio.

Sintético, sin red, sin broker.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataclasses

import scripts.run_periodic_rebalance as runner
from src import exit_rules, hrp_sizing, regime_detector, signal_stack, strategy
from src.history_requirements import (
    InsufficientHistoryError, business_rows_to_calendar_days,
)
from src.signal_stack import (
    SignalStackConfig, compute_signal_scores_with_meta, factor_requirements,
)
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

ALL_FACTORS = ["momentum", "mean_reversion", "volatility", "liquidity"]


def _panel(n_rows: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_rows)
    cols = [f"T{i}" for i in range(8)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_rows, 8)), axis=0)),
        index=idx, columns=cols,
    )
    volumes = pd.DataFrame(rng.integers(1e5, 5e6, (n_rows, 8)).astype(float),
                           index=idx, columns=cols)
    return prices, volumes


# ===================================================================
# Declared requirements
# ===================================================================
class TestDeclaredRequirements:
    def test_signal_stack_per_factor(self):
        req = factor_requirements()
        assert req == {"momentum": 252 + 21 + 1, "mean_reversion": 23,
                       "volatility": 65, "liquidity": 65}
        assert signal_stack.required_history() == 274

    def test_zero_weight_factor_has_no_requirement(self):
        cfg = SignalStackConfig(w_momentum=0.0)
        assert "momentum" not in factor_requirements(cfg)
        assert signal_stack.required_history(cfg) == 65

    def test_regime_needs_breadth_ma(self):
        assert regime_detector.required_history() == 200
        assert regime_detector.required_history(ma_breadth=50, window=63) == 69

    def test_hrp_and_exits(self):
        assert hrp_sizing.required_history(60) == 60
        assert exit_rules.required_history(exit_rules.ExitConfig()) == 1
        cfg = exit_rules.ExitConfig(trailing_stop_pct=0.1, trailing_window=90)
        assert exit_rules.required_history(cfg) == 90

    def test_requirement_is_sufficient_for_momentum(self):
        prices, volumes = _panel(274)
        _, meta = compute_signal_scores_with_meta(
            prices, volumes, prices.index[-1], mode="strict")
        assert meta["factors_used"] == ALL_FACTORS

    def test_calendar_days_cover_business_rows(self):
        end = pd.Timestamp("2026-09-30")
        start = end - pd.Timedelta(business_rows_to_calendar_days(295), unit="D")
        assert len(pd.bdate_range(start=start, end=end)) >= 295


# ===================================================================
# signal_stack strict / permissive
# ===================================================================
class TestSignalStackModes:
    def test_72_rows_strict_raises_clear_error(self):
        prices, volumes = _panel(72)
        with pytest.raises(InsufficientHistoryError) as exc:
            compute_signal_scores_with_meta(prices, volumes, prices.index[-1],
                                            mode="strict")
        assert exc.value.available == 72
        assert exc.value.missing == {"signal_stack.momentum": 274}
        assert "momentum" in str(exc.value) and "72" in str(exc.value)

    def test_72_rows_permissive_excludes_momentum(self):
        prices, volumes = _panel(72)
        scores, meta = compute_signal_scores_with_meta(
            prices, volumes, prices.index[-1], mode="permissive")
        assert "momentum" not in meta["factors_used"]
        assert meta["factors_used"] == ["mean_reversion", "volatility", "liquidity"]
        assert "momentum" in meta["factors_excluded"]
        assert sum(meta["weights"].values()) == pytest.approx(1.0)
        assert scores.notna().all()

    def test_permissive_equals_explicit_reweighting(self):
        """Excluir ≠ rellenar con 0: igual al config sin ese factor."""
        prices, volumes = _panel(72)
        t = prices.index[-1]
        a, _ = compute_signal_scores_with_meta(prices, volumes, t)
        b, _ = compute_signal_scores_with_meta(
            prices, volumes, t, config=SignalStackConfig(w_momentum=0.0))
        pd.testing.assert_series_equal(a, b)

    def test_300_rows_all_factors(self):
        prices, volumes = _panel(300)
        _, meta = compute_signal_scores_with_meta(
            prices, volumes, prices.index[-1], mode="strict")
        assert meta["factors_used"] == ALL_FACTORS
        assert meta["factors_excluded"] == {}

    def test_invalid_mode(self):
        prices, volumes = _panel(80)
        with pytest.raises(ValueError):
            compute_signal_scores_with_meta(prices, volumes, prices.index[-1],
                                            mode="lenient")


# ===================================================================
# Liquidity: volumen faltante = NaN, sin imputar
# ===================================================================
class TestLiquidityNaN:
    def test_ticker_without_volume_has_no_liquidity_score(self):
        prices, volumes = _panel(300)
        volumes["T0"] = np.nan
        liq = signal_stack._liquidity_scores(prices, volumes, prices.index[-1])
        assert np.isnan(liq["T0"])
        assert liq.drop("T0").notna().all()

    def test_no_volume_at_all_strict_raises_permissive_excludes(self):
        prices, volumes = _panel(300)
        volumes[:] = np.nan
        t = prices.index[-1]
        with pytest.raises(InsufficientHistoryError) as exc:
            compute_signal_scores_with_meta(prices, volumes, t, mode="strict")
        # 300 filas cubren la historia: el motivo es falta de datos, no de filas
        assert exc.value.missing == {"signal_stack.liquidity": "no data"}
        assert "needs" not in str(exc.value)
        _, meta = compute_signal_scores_with_meta(prices, volumes, t)
        assert "liquidity" not in meta["factors_used"]
        assert meta["factors_excluded"]["liquidity"] == "no data"

    def test_runner_never_imputes_volume(self):
        src = Path(runner.__file__).read_text()
        assert "fillna(1e6)" not in src


# ===================================================================
# Regime
# ===================================================================
class TestRegimeModes:
    def test_strict_raises_without_breadth_history(self):
        prices, _ = _panel(72)
        with pytest.raises(InsufficientHistoryError, match="regime_detector"):
            regime_detector.classify_regime(prices, prices.index[-1], mode="strict")

    def test_permissive_degrades_breadth(self):
        prices, _ = _panel(72)
        rep = regime_detector.classify_regime(prices, prices.index[-1])
        assert rep.breadth is None

    def test_strict_with_enough_history_has_breadth(self):
        prices, _ = _panel(300)
        rep = regime_detector.classify_regime(prices, prices.index[-1], mode="strict")
        assert rep.breadth is not None


# ===================================================================
# Runner
# ===================================================================
@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))

    def use_rows(n_rows, volumes=None):
        px = _fake_prices(n_rows=n_rows)
        vol = volumes if volumes is not None else pd.DataFrame(
            1e6, index=px.index, columns=TICKERS)
        use_market_data(monkeypatch, px, vol)
        return px

    return home / "data" / "live_state", use_rows


class TestRunner:
    def test_requirements_drive_default_start(self):
        live = strategy.load_strategy_config()
        reqs = runner.history_requirements(live)
        assert reqs["signal_stack.momentum"] == 274
        assert reqs["regime_detector"] == 200
        assert reqs["sizing"] == live.window
        start = pd.Timestamp(runner.default_start(max(reqs.values())))
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        assert len(pd.bdate_range(start, today)) >= 274 + 21

    def test_trailing_window_enters_requirements(self):
        cfg = dataclasses.replace(strategy.load_strategy_config(),
                                  trailing_stop_pct=0.1, trailing_window=400)
        reqs = runner.history_requirements(cfg)
        assert max(reqs.values()) == 400

    def test_72_rows_strict_exits_2_without_trading(self, runner_env, caplog):
        state, use_rows = runner_env
        use_rows(72)
        assert runner.run_once(force=True, start="2024-01-01") == 2
        assert not (state / "ledger.json").exists()
        assert "signal_stack.momentum needs 274" in caplog.text

    def test_72_rows_permissive_records_factors_used(self, runner_env):
        state, use_rows = runner_env
        use_rows(72)
        assert runner.run_once(force=True, start="2024-01-01",
                               history_mode="permissive") == 0
        hist = pd.read_csv(state / "rebalance_history.csv")
        used = hist["factors_used"].iloc[-1].split(";")
        assert "momentum" not in used
        assert hist["history_mode"].iloc[-1] == "permissive"
        assert not bool(hist["regime_breadth"].iloc[-1])

    def test_permissive_still_requires_hrp_window(self, runner_env):
        _, use_rows = runner_env
        use_rows(strategy.load_strategy_config().window - 1)
        assert runner.run_once(force=True, start="2024-01-01",
                               history_mode="permissive") == 2

    def test_300_rows_strict_uses_all_factors(self, runner_env):
        state, use_rows = runner_env
        use_rows(300)
        assert runner.run_once(force=True, start="2024-01-01") == 0
        hist = pd.read_csv(state / "rebalance_history.csv")
        assert hist["factors_used"].iloc[-1].split(";") == ALL_FACTORS
        assert bool(hist["regime_breadth"].iloc[-1])

    def test_missing_volume_stays_nan(self, runner_env, monkeypatch):
        state, use_rows = runner_env
        px = _fake_prices(n_rows=300)
        vol = pd.DataFrame(1e6, index=px.index, columns=TICKERS)
        vol["AAA"] = np.nan
        seen = {}
        real = strategy.compute_signal_scores_with_meta

        def spy(prices, volumes, t, **kw):
            seen["volumes"] = volumes
            return real(prices, volumes, t, **kw)

        use_rows(300, volumes=vol)
        monkeypatch.setattr(strategy, "compute_signal_scores_with_meta", spy)
        assert runner.run_once(force=True, start="2024-01-01") == 0
        assert seen["volumes"]["AAA"].isna().all()
