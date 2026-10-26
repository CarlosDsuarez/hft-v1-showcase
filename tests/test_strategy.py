"""
test_strategy.py — Una sola ruta de construcción de cartera (src/strategy.py).

- config/live.toml → StrategyConfig: estricto, round-trip, hash estable.
- BacktestConfig = StrategyConfig + simulación (campos planos compatibles).
- La config viva reproduce el runner previo a unificar (copia congelada).
- PARIDAD: mismos datos sintéticos + misma config → build_targets llamado
  desde run_backtest y desde run_once da pesos idénticos en t.
- El runner registra en rebalance_history el hash de la config usada.

Todo sintético, sin red (FakeProvider de tests.test_runner_paths).
"""

import dataclasses
import json
import tomllib

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
import src.backtest_engine as be
from src.backtest_engine import BacktestConfig, run_backtest
from src.conviction_scoring import conviction_score, conviction_to_tilts
from src.exit_rules import ExitConfig, apply_exits_to_weights, evaluate_exits
from src.hrp_sizing import hrp_weights
from src.portfolio_state import PortfolioState
from src.position_ledger import PositionLedger
from src.regime_detector import classify_regime
from src.rolling_engine import get_rebalance_dates, get_window
from src.signal_stack import SignalStackConfig, compute_signal_scores_with_meta
from src.strategy import (
    DEFAULT_CONFIG_PATH,
    InsufficientDataError,
    StrategyConfig,
    TargetResult,
    active_assets,
    build_targets,
    execution_weights,
    history_requirements,
    load_strategy_config,
    prev_weights,
    strategy_config_from_dict,
    strategy_config_to_toml,
    turnover_cost,
)
from tests.test_runner_paths import TICKERS, _fake_prices, use_market_data

LIVE = load_strategy_config()
N_ROWS = 330
WARMUP = 280  # >= 274 filas: el runner (strict) tiene historia para todo


def _market(n_rows=N_ROWS, seed=11):
    """Precios lognormales + volúmenes aleatorios (el filtro Amihud discrimina)."""
    prices = _fake_prices(n_rows=n_rows)
    rng = np.random.default_rng(seed)
    volumes = pd.DataFrame(rng.uniform(2e5, 5e6, prices.shape),
                           index=prices.index, columns=prices.columns)
    return prices, volumes


def _empty_state(prices, t):
    return PortfolioState.from_ledger(PositionLedger(), prices.loc[t], as_of=t)


def _toml_dict(cfg):
    return tomllib.loads(strategy_config_to_toml(cfg))


# ===========================================================================
# Config viva
# ===========================================================================
class TestLiveConfig:
    def test_live_file_reproduces_pre_unification_runner_values(self):
        """Valores del runner antes de unificar (WINDOW=60, REGIME_WINDOW=30,
        sin tope/filtro/top-k, salidas +$2/-$1, 7 bps): no se eligen nuevos."""
        assert DEFAULT_CONFIG_PATH.name == "live.toml"
        assert (LIVE.opt_method, LIVE.window, LIVE.regime_window) == ("hrp", 60, 30)
        assert LIVE.use_signal_stack is True
        assert LIVE.signal_config == SignalStackConfig()
        assert LIVE.max_weight == 1.0
        assert LIVE.universe_filter is False
        assert LIVE.conviction_top_k is None
        assert LIVE.min_assets == 3
        assert LIVE.group_caps is None and LIVE.groups is None
        assert LIVE.exit_config() == ExitConfig(
            stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None,
            trailing_window=30, exit_on_illiquid=False,
            take_profit_dollars=2.0, stop_loss_dollars=1.0)
        assert LIVE.cost_bps == 7.0

    @pytest.mark.parametrize("cfg", [
        LIVE,
        StrategyConfig(),
        StrategyConfig(groups={"A": "us/Tech", "B": "co/Energy"},
                       group_caps={("us/Tech", "co/Energy"): 0.5, "us/Tech": 0.3},
                       max_weight_by_group={"us/Tech": 0.2}, conviction_top_k=3),
    ], ids=["live", "backtest_defaults", "groups"])
    def test_toml_round_trip_keeps_config_and_hash(self, cfg):
        back = strategy_config_from_dict(_toml_dict(cfg))
        assert back == cfg
        assert back.config_hash() == cfg.config_hash()

    def test_hash_is_canonical_and_tracks_changes(self):
        assert dataclasses.replace(LIVE, max_weight=1).config_hash() == LIVE.config_hash()
        assert dataclasses.replace(LIVE, window=61).config_hash() != LIVE.config_hash()
        caps = {("us/Tech", "co/Energy"): 0.5, "us/Tech": 0.3}
        a = StrategyConfig(groups={"A": "us/Tech"}, group_caps=caps)
        b = StrategyConfig(groups={"A": "us/Tech"},
                           group_caps=dict(reversed(list(caps.items()))))
        assert a.config_hash() == b.config_hash()  # orden de inserción no importa
        json.dumps(LIVE.to_dict())  # serializable (null para off)

    def test_missing_key_is_error_not_backtest_default(self):
        data = _toml_dict(LIVE)
        del data["window"]
        with pytest.raises(ValueError, match="faltan \\['window'\\]"):
            strategy_config_from_dict(data)

    def test_unknown_key_is_error(self):
        with pytest.raises(ValueError, match="desconocidas \\['windw'\\]"):
            strategy_config_from_dict({**_toml_dict(LIVE), "windw": 60})

    @pytest.mark.parametrize("key,value", [
        ("window", 60.5), ("window", True), ("use_signal_stack", 1),
        ("max_weight", "1.0"), ("universe_filter", "no"),
    ])
    def test_wrong_type_is_error(self, key, value):
        with pytest.raises(ValueError, match=key):
            strategy_config_from_dict({**_toml_dict(LIVE), key: value})

    def test_false_turns_optional_rule_off(self):
        data = {**_toml_dict(LIVE), "take_profit_dollars": False}
        assert strategy_config_from_dict(data).take_profit_dollars is None

    def test_schema_version_mismatch_is_error(self):
        with pytest.raises(ValueError, match="schema_version"):
            strategy_config_from_dict({**_toml_dict(LIVE), "schema_version": 2})

    def test_file_errors_name_the_file(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text(strategy_config_to_toml(LIVE).replace("window = 60\n", ""))
        with pytest.raises(ValueError, match="bad.toml"):
            load_strategy_config(bad)

    def test_history_requirements_follow_enabled_components(self):
        reqs = history_requirements(LIVE)
        assert reqs["sizing"] == 60 and reqs["regime_detector"] == 200
        assert "universe_filter" not in reqs
        assert history_requirements(dataclasses.replace(
            LIVE, universe_filter=True))["universe_filter"] == LIVE.filter_window
        plain = history_requirements(StrategyConfig(universe_filter=False))
        assert set(plain) == {"sizing"}


# ===========================================================================
# BacktestConfig = StrategyConfig + simulación
# ===========================================================================
class TestBacktestConfig:
    def test_flat_strategy_fields_go_to_strategy(self):
        cfg = BacktestConfig(opt_method="hrp", window=60, rebalance_freq="15D")
        assert cfg.strategy == StrategyConfig(opt_method="hrp", window=60)
        assert cfg.rebalance_freq == "15D"
        assert (cfg.window, cfg.opt_method) == (60, "hrp")  # lectura plana

    def test_explicit_strategy_with_flat_overrides(self):
        cfg = BacktestConfig(strategy=LIVE, max_weight=0.3, warmup=10)
        assert cfg.strategy == dataclasses.replace(LIVE, max_weight=0.3)
        assert cfg.warmup == 10

    def test_setting_flat_field_updates_strategy(self):
        cfg = BacktestConfig()
        cfg.max_weight = 0.2
        assert cfg.strategy.max_weight == 0.2
        assert "max_weight" not in vars(cfg)  # nada sombreado fuera de strategy

    def test_unknown_field_is_type_error(self):
        with pytest.raises(TypeError, match="foo"):
            BacktestConfig(foo=1)

    def test_dataclasses_replace_and_equality(self):
        cfg = BacktestConfig(opt_method="hrp")
        new = dataclasses.replace(cfg, use_position_ledger=True, kappa=0.5)
        assert new.use_position_ledger and new.kappa == 0.5
        assert new.opt_method == "hrp"
        assert cfg == BacktestConfig(opt_method="hrp")


# ===========================================================================
# La config viva reproduce el runner previo a unificar
# ===========================================================================
def _legacy_runner_targets(prices, volumes, t, history_mode="strict"):
    """compute_target_weights del runner ANTES de unificar (copia congelada)."""
    wp = get_window(prices, t, window=60).dropna(axis=1, how="any")
    active = wp.columns.tolist()
    base = hrp_weights(wp.ffill().pct_change().dropna(how="all"))
    report = classify_regime(prices[active], t, window=30, mode=history_mode)
    scores, _ = compute_signal_scores_with_meta(prices, volumes, t, mode=history_mode)
    conv = conviction_score(scores.reindex(active).fillna(0.0), report.state)
    w = conviction_to_tilts(conv, base_weights=base.reindex(active).fillna(0.0))
    w = w.clip(lower=0.0)
    return w / w.sum(), report.state


def _legacy_runner_exits(portfolio, prices, t, w):
    """evaluate_exit_rules del runner ANTES de unificar (+$2 / -$1)."""
    held, entries, shares = portfolio.exit_inputs()
    exits = evaluate_exits(t, prices, held, entries, active=None, shares=shares,
                           config=ExitConfig(stop_loss_pct=None, take_profit_pct=None,
                                             trailing_window=30, exit_on_illiquid=False,
                                             take_profit_dollars=2.0,
                                             stop_loss_dollars=1.0))
    return apply_exits_to_weights(w, exits[exits["ticker"].isin(w.index)]), exits


class TestLiveReproducesLegacyRunner:
    @pytest.mark.parametrize("history_mode", ["strict", "permissive"])
    def test_same_weights_and_regime_on_empty_book(self, history_mode):
        prices, volumes = _market()
        t = prices.index[-1]
        res = build_targets(_empty_state(prices, t), prices, volumes, t, LIVE,
                            history_mode=history_mode)
        w_old, regime_old = _legacy_runner_targets(prices, volumes, t, history_mode)
        pd.testing.assert_series_equal(res.weights, w_old, rtol=0, atol=1e-12)
        assert res.regime == regime_old
        assert res.exits.empty

    def test_same_exits_on_held_book(self):
        prices, volumes = _market()
        t = prices.index[-1]
        px = prices.loc[t]
        ledger = PositionLedger(initial_cash=100_000.0)
        # AAA +$5 (take-profit), BBB -$5 (stop), CCC +$0.5 (se queda)
        entry = px.copy()
        entry[["AAA", "BBB", "CCC"]] += [-0.5, 0.5, -0.05]
        ledger.apply_trades(pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                                          "shares": [10.0, 10.0, 10.0]}),
                            entry, t=prices.index[-30])
        state = PortfolioState.from_ledger(ledger, px, as_of=t)
        res = build_targets(state, prices, volumes, t, LIVE, history_mode="strict")
        w_old, _ = _legacy_runner_targets(prices, volumes, t)
        w_old, exits_old = _legacy_runner_exits(state, prices, t, w_old)
        assert sorted(res.exits["ticker"]) == ["AAA", "BBB"]
        pd.testing.assert_frame_equal(res.exits, exits_old.reset_index(drop=True))
        pd.testing.assert_series_equal(res.weights, w_old, rtol=0, atol=1e-12)


# ===========================================================================
# build_targets: piezas
# ===========================================================================
class TestBuildTargets:
    def test_too_few_assets_raises_insufficient_data(self):
        prices, volumes = _market()
        t = prices.index[-1]
        two = prices[["AAA", "BBB"]]
        with pytest.raises(InsufficientDataError):
            build_targets(_empty_state(two, t), two, volumes, t, LIVE)

    def test_exit_on_held_ticker_outside_active_does_not_raise(self):
        """Regresión: el backtest viejo hacía w[['X']] = 0 con X fuera del
        objetivo → KeyError → rebalanceo omitido en silencio."""
        prices, volumes = _market()
        t = prices.index[-1]
        prices = prices.assign(XYZ=prices["AAA"] * 1.01)
        state = PortfolioState.from_weights(
            pd.Series({"AAA": 0.5, "XYZ": 0.5}), prices.loc[t], equity=1e6,
            avg_entry={"AAA": 1e9, "XYZ": 1e9}, as_of=t)  # stop seguro
        cfg = StrategyConfig(opt_method="hrp", window=60, universe_filter=False,
                             use_exit_rules=True, take_profit_pct=None)
        res = build_targets(state, prices, volumes, t, cfg, universe=TICKERS)
        assert set(res.exits["ticker"]) == {"AAA", "XYZ"}
        assert "XYZ" not in res.weights.index
        assert res.weights["AAA"] == 0.0
        assert res.weights.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("universe", [None, TICKERS[:4]])
    def test_active_assets_match_build_targets(self, universe):
        """active_assets (failure context) is the set build_targets optimizes."""
        prices, volumes = _market()
        t = prices.index[-1]
        cfg = dataclasses.replace(LIVE, universe_filter=True)
        res = build_targets(_empty_state(prices, t), prices, volumes, t, cfg,
                            universe=universe)
        assert active_assets(prices, volumes, t, cfg, universe=universe) == list(
            res.active)

    def test_active_assets_too_few_raises_like_build_targets(self):
        prices, volumes = _market()
        t = prices.index[-1]
        two = prices[["AAA", "BBB"]]
        with pytest.raises(InsufficientDataError):
            active_assets(two, volumes, t, LIVE)

    def test_universe_filter_reported(self):
        prices, volumes = _market()
        t = prices.index[-1]
        res = build_targets(_empty_state(prices, t), prices, volumes, t,
                            dataclasses.replace(LIVE, universe_filter=True))
        assert res.meta["universe_excluded"]
        assert set(res.eligible) == set(TICKERS) - set(res.meta["universe_excluded"])
        assert set(res.weights.index) <= set(res.eligible)

    def test_max_weight_cap_applies(self):
        prices, volumes = _market()
        t = prices.index[-1]
        res = build_targets(_empty_state(prices, t), prices, volumes, t,
                            dataclasses.replace(LIVE, max_weight=0.2))
        assert res.weights.max() <= 0.2 + 1e-9
        assert res.weights.sum() == pytest.approx(1.0)

    def test_prev_weights_none_for_empty_book(self):
        prices, _ = _market()
        t = prices.index[-1]
        assert prev_weights(_empty_state(prices, t), prices, t, TICKERS) is None

    def test_from_weights_state_round_trips_weights(self):
        prices, _ = _market()
        t = prices.index[-1]
        w = pd.Series({"AAA": 0.6, "BBB": 0.4, "CCC": 1e-12})
        state = PortfolioState.from_weights(w, prices.loc[t], equity=5e5, as_of=t)
        assert list(state.positions.index) == ["AAA", "BBB"]  # polvo fuera
        pd.testing.assert_series_equal(state.weights(prices.loc[t]),
                                       w[["AAA", "BBB"]], rtol=1e-12)
        assert state.cash == pytest.approx(0.0, abs=1e-6)

    def test_turnover_cost_uses_config_bps(self):
        cfg = StrategyConfig(commission_bps=3.0, spread_bps=1.0)
        turnover, cost = turnover_cost(pd.Series({"A": 1.0}),
                                       pd.Series({"A": 0.5, "B": 0.5}), cfg)
        assert turnover == pytest.approx(1.0)
        assert cost == pytest.approx(4.0 / 10_000)


# ===========================================================================
# Libro ejecutado: tope de turnover (R-02) + topes por grupo
# ===========================================================================
def _targets(weights, w_prev=None, exits=()):
    return TargetResult(
        weights=pd.Series(weights, dtype=float), regime=None,
        exits=pd.DataFrame({"ticker": list(exits), "rule": "stop_loss_dollars",
                            "detail": ""}, columns=["ticker", "rule", "detail"]),
        meta={}, w_prev=None if w_prev is None else pd.Series(w_prev, dtype=float))


def _book(held, cash=0.0, equity=100.0):
    """Libro con `held` (pesos invertidos, suman 1) y fracción `cash` en caja."""
    w = pd.Series(held, dtype=float) * (1.0 - cash)
    return PortfolioState.from_weights(w, pd.Series(1.0, index=w.index), equity)


class TestExecutionWeights:
    CFG = StrategyConfig(turnover_limit=0.20)
    TARGET = {"A": 0.2, "B": 0.4, "C": 0.4}

    def test_empty_book_deploys_full_target(self):
        w = execution_weights(_targets(self.TARGET), _book({}, cash=1.0), self.CFG)
        pd.testing.assert_series_equal(w, pd.Series(self.TARGET))

    def test_invested_book_rotates_at_most_turnover_limit(self):
        # L1 = 1.6 > 0.20 → alpha = 0.125 (mismo blend que run_backtest)
        w = execution_weights(_targets(self.TARGET, w_prev={"A": 1.0}),
                              _book({"A": 1.0}), self.CFG)
        pd.testing.assert_series_equal(w, pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}))

    def test_cash_is_deployed_to_target_not_capped(self):
        # 50% caja: la mitad invertida rota con tope, la caja va al objetivo
        w = execution_weights(_targets(self.TARGET, w_prev={"A": 1.0}),
                              _book({"A": 1.0}, cash=0.5), self.CFG)
        pd.testing.assert_series_equal(
            w, pd.Series({"A": 0.55, "B": 0.225, "C": 0.225}))

    def test_exits_sell_in_full_despite_cap(self):
        # A sale: su mitad se vende entera y va al objetivo; B rota con tope
        target = {"A": 0.0, "B": 0.5, "C": 0.5}
        held = {"A": 0.5, "B": 0.5}
        w = execution_weights(_targets(target, w_prev=held, exits=["A"]),
                              _book(held), self.CFG)
        pd.testing.assert_series_equal(w, pd.Series({"A": 0.0, "B": 0.7, "C": 0.3}))

    def test_everything_exited_goes_to_cash(self):
        target = {"A": 0.0, "B": 0.0}
        w = execution_weights(_targets(target, w_prev={"A": 1.0, "B": 0.0},
                                       exits=["A"]),
                              _book({"A": 1.0}), self.CFG)
        pd.testing.assert_series_equal(w, pd.Series(target))

    def test_group_caps_hold_on_executed_book(self):
        cfg = StrategyConfig(turnover_limit=0.20,
                             groups={"A": "g1", "B": "g1", "C": "g2"},
                             group_caps={"g1": 0.6})
        target = {"A": 0.3, "B": 0.3, "C": 0.4}
        held = {"A": 0.5, "B": 0.5}            # g1 = 100% > tope
        w = execution_weights(_targets(target, w_prev=held), _book(held), cfg)
        assert w[["A", "B"]].sum() <= 0.6 + 1e-9
        assert w.sum() == pytest.approx(1.0)


# ===========================================================================
# PARIDAD backtest ↔ runner
# ===========================================================================
PARITY_CONFIGS = {
    "live": LIVE,
    # Perillas del backtest encendidas: filtro, tope, top-k, salidas pct y
    # por universo. Cubre las ramas que en vivo hoy están apagadas.
    "live_with_backtest_knobs": dataclasses.replace(
        LIVE, universe_filter=True, max_weight=0.30, conviction_top_k=4,
        regime_window=63, exit_on_illiquid=True, stop_loss_pct=0.02,
        take_profit_pct=0.04, take_profit_dollars=None, stop_loss_dollars=None),
}


@pytest.fixture
def parity_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": TICKERS}))
    return home


def _backtest_calls(monkeypatch, prices, volumes, cfg):
    """run_backtest (ledger) espiando build_targets: {t: (state, result)}."""
    calls = {}
    real = be.build_targets

    def spy(**kw):
        res = real(**kw)
        calls[kw["t"]] = (kw["state"], res)
        return res

    monkeypatch.setattr(be, "build_targets", spy)
    run_backtest(prices, volumes, BacktestConfig(
        strategy=cfg, rebalance_freq="15D", warmup=WARMUP,
        use_position_ledger=True, initial_nav=100_000.0))
    monkeypatch.setattr(be, "build_targets", real)
    return calls


def _runner_call(monkeypatch, home, prices, volumes, cfg, t, book):
    """run_once (dry-run, sim) en t con el libro `book`: (prices, result)."""
    close, vol = prices.loc[:t], volumes.loc[:t]
    use_market_data(monkeypatch, close, vol)
    config_path = home / "strategy.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(strategy_config_to_toml(cfg))
    if book.positions.any():
        state_dir = home / "data" / "live_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        runner.save_ledger(book.to_ledger(), state_dir / "ledger.json")
    seen = []
    real = runner.build_targets

    def spy(state, prices_, volumes_, t_, cfg_, **kw):
        res = real(state, prices_, volumes_, t_, cfg_, **kw)
        seen.append((prices_, res))
        return res

    monkeypatch.setattr(runner, "build_targets", spy)
    # Misma historia que el backtest (el régimen usa toda la historia <= t).
    assert runner.run_once(force=True, dry_run=True, config_path=config_path,
                           start=str(prices.index[0].date())) == runner.EXIT_OK
    assert len(seen) == 1
    return seen[0]


class TestBacktestRunnerParity:
    @pytest.mark.parametrize("which", [0, 1], ids=["empty_book", "held_book"])
    @pytest.mark.parametrize("name", list(PARITY_CONFIGS))
    def test_same_weights_at_t(self, parity_env, monkeypatch, name, which):
        cfg = PARITY_CONFIGS[name]
        prices, volumes = _market()
        rebal = get_rebalance_dates(prices, freq="15D", warmup=WARMUP)
        t = rebal[which]
        bt_state, bt = _backtest_calls(monkeypatch, prices, volumes, cfg)[t]
        assert bt_state.positions.any() == bool(which)

        live_prices, live = _runner_call(monkeypatch, parity_env, prices,
                                         volumes, cfg, t, bt_state)

        # Mismos datos de entrada (hasta t) ...
        pd.testing.assert_frame_equal(live_prices, prices.loc[:t],
                                      check_freq=False, check_names=False)
        # ... mismos pesos, régimen, activos y salidas.
        pd.testing.assert_series_equal(live.weights, bt.weights, check_exact=True)
        assert live.regime == bt.regime
        assert live.active == bt.active
        pd.testing.assert_frame_equal(live.exits, bt.exits)
        assert live.weights.sum() == pytest.approx(1.0) or (live.weights == 0).all()

    @pytest.mark.parametrize("name", list(PARITY_CONFIGS))
    def test_same_executed_book_from_held_book(self, parity_env, monkeypatch, name):
        """Tope R-02 + topes por grupo: misma ejecución en backtest y vivo."""
        cfg = PARITY_CONFIGS[name]
        prices, volumes = _market()
        t = get_rebalance_dates(prices, freq="15D", warmup=WARMUP)[1]
        bt_exec = _spy_execution(monkeypatch, be)
        bt_state, _ = _backtest_calls(monkeypatch, prices, volumes, cfg)[t]
        live_exec = _spy_execution(monkeypatch, runner)

        _runner_call(monkeypatch, parity_env, prices, volumes, cfg, t, bt_state)

        pd.testing.assert_series_equal(live_exec[t], bt_exec[t])


def _spy_execution(monkeypatch, module):
    """Espía module.execution_weights: {state.as_of: pesos ejecutados}."""
    seen = {}
    real = module.execution_weights

    def spy(targets, state, cfg):
        seen[state.as_of] = res = real(targets, state, cfg)
        return res

    monkeypatch.setattr(module, "execution_weights", spy)
    return seen


# ===========================================================================
# Runner: hash de la config en history + overrides
# ===========================================================================
class TestRunnerRecordsConfig:
    @pytest.fixture
    def home(self, parity_env, monkeypatch):
        use_market_data(monkeypatch, _fake_prices())
        return parity_env

    def _history(self, home):
        return pd.read_csv(home / "data" / "live_state" / "rebalance_history.csv")

    def test_history_records_live_config_hash(self, home):
        assert runner.run_once(force=True, start="2024-01-01") == 0
        row = self._history(home).iloc[-1]
        assert row["config_hash"] == LIVE.config_hash()
        assert row["config_source"] == str(DEFAULT_CONFIG_PATH)

    def test_exit_override_changes_recorded_hash(self, home):
        assert runner.run_once(force=True, start="2024-01-01",
                               take_profit_dollars=None) == 0
        expected = dataclasses.replace(LIVE, take_profit_dollars=None)
        assert self._history(home).iloc[-1]["config_hash"] == expected.config_hash()

    def test_config_path_selects_file(self, home):
        path = home / "alt.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        alt = dataclasses.replace(LIVE, max_weight=0.5)
        path.write_text(strategy_config_to_toml(alt))
        assert runner.run_once(force=True, start="2024-01-01", config_path=path) == 0
        row = self._history(home).iloc[-1]
        assert row["config_hash"] == alt.config_hash()
        assert row["config_source"] == str(path)

    def test_apply_exit_overrides(self):
        off = dataclasses.replace(LIVE, use_exit_rules=False)
        assert runner.apply_exit_overrides(off) is off
        on = runner.apply_exit_overrides(off, stop_loss_pct=0.05)
        assert on.use_exit_rules and on.stop_loss_pct == 0.05
        with pytest.raises(TypeError):
            runner.apply_exit_overrides(LIVE, window=1)

    def test_cli_config_and_exit_flags(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(runner, "run_once", lambda **kw: seen.update(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        path = tmp_path / "x.toml"
        assert runner.main(["--config", str(path), "--take-profit-dollars", "0",
                            "--stop-loss", "0.03"]) == 0
        assert seen["config_path"] == path
        assert seen["take_profit_dollars"] is None          # 0 = off
        assert seen["stop_loss_pct"] == 0.03                # override
        assert seen["stop_loss_dollars"] is runner.FROM_CONFIG
        assert seen["trailing_window"] is runner.FROM_CONFIG

    def test_cli_defaults_use_live_file(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(runner, "run_once", lambda **kw: seen.update(kw) or 0)
        monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
        assert runner.main([]) == 0
        assert seen["config_path"] == DEFAULT_CONFIG_PATH
        assert all(seen[f] is runner.FROM_CONFIG for f in runner.EXIT_OVERRIDE_FIELDS)


# ===========================================================================
# Contrato del optimizador: se valida al construir la config (fail-fast)
# ===========================================================================
class TestOptimizerContract:
    def test_unknown_opt_method_fails_at_config(self):
        with pytest.raises(ValueError, match="opt_method='nope'"):
            StrategyConfig(opt_method="nope")

    def test_unknown_opt_method_fails_in_backtest_config(self):
        with pytest.raises(ValueError, match="opt_method='nope'"):
            BacktestConfig(opt_method="nope")

    def test_unknown_opt_method_fails_loading_file(self):
        data = dict(_toml_dict(LIVE), opt_method="nope")
        with pytest.raises(ValueError, match="opt_method='nope'"):
            strategy_config_from_dict(data)

    def test_optimizer_needing_input_strategy_lacks_fails_at_config(
        self, monkeypatch,
    ):
        """build_targets nunca pasa illiquidity: un método que la exija no
        puede llegar a la fecha 37 del backtest para fallar ahí."""
        import src.portfolio_optimizer as po
        monkeypatch.setitem(po.OPTIMIZERS, "needs_illiq", po.Optimizer(
            run=po.OPTIMIZERS["mv_classic"].run,
            requires=frozenset({"illiquidity"}),
        ))
        with pytest.raises(ValueError, match="illiquidity"):
            StrategyConfig(opt_method="needs_illiq")

    def test_liquidity_builds_targets(self):
        prices, volumes = _market()
        t = prices.index[-1]
        cfg = StrategyConfig(opt_method="liquidity", window=60,
                             universe_filter=False)
        res = build_targets(_empty_state(prices, t), prices, volumes, t, cfg)
        assert res.weights.sum() == pytest.approx(1.0)
        assert res.meta["converged"]
        assert res.meta["fallback_used"] is False
