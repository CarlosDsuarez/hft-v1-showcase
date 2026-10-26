"""
test_missing_prices.py — Una posicion abierta nunca se valora con precio 0/NaN
en silencio.

Bug de origen: ``PositionLedger.nav`` hacia ``prices.reindex(...).fillna(0.0)``;
si un ticker en cartera desaparecia de ``prices`` (Yahoo no lo trae,
``clean_prices`` lo descarta por >5% NaN o sale del universo) su valor pasaba
a 0, el NAV se desplomaba y el runner compraba para "reponer" con NAV
equivocado (rebalance_history 2026-09-28, corrida 2: NAV 723,858).

Todo offline/sintetico; estado en tmp_path via HFT_HOME.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.brokers.alpaca_adapter import reconcile_trades
from src.data_client import clean_prices
from src.position_ledger import MissingPriceError, PositionLedger
from src.market_data import FakeProvider
from tests.test_runner_paths import after_close

T0 = pd.Timestamp("2026-09-25")


# ===========================================================================
# PositionLedger: nav / snapshot / weights_from_positions
# ===========================================================================
@pytest.fixture
def ledger_xyz():
    """10 A @ 100 + 20 XYZ @ 50 → cash 8,000, NAV 10,000 at entry prices."""
    ledger = PositionLedger(initial_cash=10_000.0)
    ledger.apply_trades(
        pd.DataFrame([{"ticker": "A", "shares": 10.0},
                      {"ticker": "XYZ", "shares": 20.0}]),
        pd.Series({"A": 100.0, "XYZ": 50.0}), t=T0,
    )
    return ledger


@pytest.fixture
def prices_without_xyz():
    return pd.Series({"A": 100.0, "B": 50.0})


class TestLedgerRaisesOnMissingPrice:
    def test_nav_raises_when_held_ticker_absent(self, ledger_xyz, prices_without_xyz):
        with pytest.raises(MissingPriceError) as ei:
            ledger_xyz.nav(prices_without_xyz)
        assert ei.value.tickers == ["XYZ"]
        assert "XYZ" in str(ei.value)

    @pytest.mark.parametrize("bad_px", [np.nan, 0.0, -1.0, np.inf])
    def test_nav_raises_on_invalid_price(self, ledger_xyz, bad_px):
        with pytest.raises(MissingPriceError):
            ledger_xyz.nav(pd.Series({"A": 100.0, "XYZ": bad_px}))

    def test_snapshot_raises(self, ledger_xyz, prices_without_xyz):
        with pytest.raises(MissingPriceError):
            ledger_xyz.snapshot(T0, prices_without_xyz)

    def test_weights_from_positions_raises(self, ledger_xyz, prices_without_xyz):
        with pytest.raises(MissingPriceError):
            ledger_xyz.weights_from_positions(prices_without_xyz)

    def test_missing_price_error_is_value_error(self):
        assert issubclass(MissingPriceError, ValueError)

    def test_flat_ledger_needs_no_prices(self):
        ledger = PositionLedger(initial_cash=5_000.0)
        assert ledger.nav(pd.Series(dtype=float)) == 5_000.0

    def test_unheld_nan_price_is_ignored(self, ledger_xyz):
        px = pd.Series({"A": 100.0, "XYZ": 50.0, "B": np.nan})
        assert ledger_xyz.nav(px) == pytest.approx(10_000.0)


class TestLedgerStalePrices:
    def test_nav_uses_stale_price_and_warns(self, ledger_xyz, prices_without_xyz, caplog):
        with caplog.at_level(logging.WARNING, logger="position_ledger"):
            nav = ledger_xyz.nav(prices_without_xyz,
                                 stale_prices=pd.Series({"XYZ": 45.0}))
        # 8,000 cash + 10*100 + 20*45
        assert nav == pytest.approx(9_900.0)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "XYZ" in warnings[0].getMessage()

    def test_fresh_price_wins_over_stale(self, ledger_xyz, caplog):
        px = pd.Series({"A": 100.0, "XYZ": 60.0})
        with caplog.at_level(logging.WARNING, logger="position_ledger"):
            nav = ledger_xyz.nav(px, stale_prices=pd.Series({"XYZ": 45.0}))
        assert nav == pytest.approx(8_000.0 + 1_000.0 + 1_200.0)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.parametrize("stale", [pd.Series({"OTHER": 1.0}),
                                       pd.Series({"XYZ": np.nan}),
                                       pd.Series({"XYZ": 0.0})])
    def test_invalid_or_absent_stale_still_raises(self, ledger_xyz,
                                                  prices_without_xyz, stale):
        with pytest.raises(MissingPriceError) as ei:
            ledger_xyz.nav(prices_without_xyz, stale_prices=stale)
        assert ei.value.tickers == ["XYZ"]

    def test_snapshot_with_stale(self, ledger_xyz, prices_without_xyz, caplog):
        with caplog.at_level(logging.WARNING, logger="position_ledger"):
            snap = ledger_xyz.snapshot(T0, prices_without_xyz,
                                       stale_prices=pd.Series({"XYZ": 45.0}))
        assert snap.nav == pytest.approx(9_900.0)
        # A flat (0) + XYZ 20 * (45 - 50)
        assert snap.unrealized_pnl == pytest.approx(-100.0)
        assert any("XYZ" in r.getMessage() for r in caplog.records
                   if r.levelno == logging.WARNING)

    def test_weights_with_stale(self, ledger_xyz, prices_without_xyz):
        w = ledger_xyz.weights_from_positions(
            prices_without_xyz, stale_prices=pd.Series({"XYZ": 45.0}))
        assert w["XYZ"] == pytest.approx(900.0 / 9_900.0)
        assert w["A"] == pytest.approx(1_000.0 / 9_900.0)


# ===========================================================================
# trades_from_target_weights / reconcile_trades: unpriced no se omite
# ===========================================================================
class TestTradesUnpriced:
    def test_raises_by_default_even_with_explicit_nav(self, ledger_xyz,
                                                      prices_without_xyz):
        with pytest.raises(MissingPriceError) as ei:
            ledger_xyz.trades_from_target_weights(
                pd.Series({"A": 1.0}), prices_without_xyz, nav=10_000.0)
        assert ei.value.tickers == ["XYZ"]

    def test_return_unpriced_lets_caller_decide(self, ledger_xyz,
                                                prices_without_xyz):
        trades, unpriced = ledger_xyz.trades_from_target_weights(
            pd.Series({"A": 1.0}), prices_without_xyz, nav=10_000.0,
            return_unpriced=True)
        assert unpriced == ["XYZ"]
        assert "XYZ" not in set(trades["ticker"])
        row = trades.set_index("ticker").loc["A"]
        assert row["shares"] == pytest.approx(100.0 - 10.0)

    def test_return_unpriced_empty_when_all_priced(self, ledger_xyz):
        trades, unpriced = ledger_xyz.trades_from_target_weights(
            pd.Series({"A": 1.0}), pd.Series({"A": 100.0, "XYZ": 50.0}),
            return_unpriced=True)
        assert unpriced == []
        assert set(trades["ticker"]) == {"A", "XYZ"}

    def test_unheld_target_without_price_still_skipped(self):
        ledger = PositionLedger(initial_cash=1_000.0)
        trades = ledger.trades_from_target_weights(
            pd.Series({"A": 0.5, "NOPX": 0.5}), pd.Series({"A": 10.0}))
        assert set(trades["ticker"]) == {"A"}


class TestReconcileUnpriced:
    def test_raises_when_broker_holds_unpriced(self):
        with pytest.raises(MissingPriceError) as ei:
            reconcile_trades(pd.Series({"AAPL": 1.0}),
                             pd.Series({"AAPL": 1.0, "XYZ": 3.0}),
                             pd.Series({"AAPL": 100.0}), 1_000.0)
        assert ei.value.tickers == ["XYZ"]

    @pytest.mark.parametrize("bad_px", [np.nan, 0.0])
    def test_return_unpriced(self, bad_px):
        df, unpriced = reconcile_trades(
            pd.Series({"AAPL": 1.0}), pd.Series({"AAPL": 1.0, "XYZ": 3.0}),
            pd.Series({"AAPL": 100.0, "XYZ": bad_px}), 1_000.0,
            return_unpriced=True)
        assert unpriced == ["XYZ"]
        assert list(df["ticker"]) == ["AAPL"]

    def test_zero_qty_without_price_is_not_a_position(self):
        df = reconcile_trades(pd.Series({"AAPL": 1.0}),
                              pd.Series({"AAPL": 0.0, "XYZ": 0.0}),
                              pd.Series({"AAPL": 100.0}), 1_000.0)
        assert list(df["ticker"]) == ["AAPL"]


# ===========================================================================
# clean_prices(must_keep=...)
# ===========================================================================
def _raw_with_gappy_xyz(n=100, gap=10):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2026-01-01", periods=n)
    cols = ["AAA", "BBB", "CCC", "XYZ"]
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(cols))), axis=0))
    raw = pd.DataFrame(data, index=idx, columns=cols)
    raw.iloc[:gap, raw.columns.get_loc("XYZ")] = np.nan  # 10% NaN → dropped
    return raw


class TestCleanPricesMustKeep:
    def test_gappy_ticker_dropped_by_default(self):
        clean, _ = clean_prices(_raw_with_gappy_xyz())
        assert "XYZ" not in clean.columns

    def test_must_keep_never_dropped_and_others_untouched(self):
        raw = _raw_with_gappy_xyz()
        base, _ = clean_prices(raw)
        kept, report = clean_prices(raw, must_keep=["XYZ"])
        assert "XYZ" in kept.columns
        pd.testing.assert_frame_equal(kept.drop(columns="XYZ"), base)
        assert kept["XYZ"].iloc[-1] == pytest.approx(raw["XYZ"].iloc[-1])
        flags = report.set_index("ticker")["kept_by_must_keep"]
        assert bool(flags["XYZ"]) and not flags.drop("XYZ").any()

    @pytest.mark.parametrize("xyz_gap", [slice(0, 10),    # >5% → rescatado
                                         slice(-4, None)])  # NaN residual en t
    def test_must_keep_alone_keeps_every_date(self, xyz_gap):
        """Sin otra columna completa, must_keep no decide (ni borra) fechas."""
        raw = _raw_with_gappy_xyz(gap=0)
        raw.iloc[:20, :3] = np.nan  # todo el resto >5% NaN → descartado
        raw.iloc[xyz_gap, raw.columns.get_loc("XYZ")] = np.nan
        kept, _ = clean_prices(raw, must_keep=["XYZ"])
        assert list(kept.columns) == ["XYZ"]
        pd.testing.assert_index_equal(kept.index, raw.index)
        assert kept["XYZ"].isna().any()  # el hueco sigue visible

    def test_must_keep_preserves_input_column_order(self):
        raw = _raw_with_gappy_xyz()[["XYZ", "AAA", "BBB", "CCC"]]
        kept, _ = clean_prices(raw, must_keep=["XYZ"])
        assert list(kept.columns) == ["XYZ", "AAA", "BBB", "CCC"]

    @pytest.mark.parametrize("make_raw", [
        lambda r: r.drop(columns="XYZ"),
        lambda r: r.assign(XYZ=np.nan),
    ])
    def test_must_keep_absent_raises(self, make_raw):
        with pytest.raises(MissingPriceError) as ei:
            clean_prices(make_raw(_raw_with_gappy_xyz()), must_keep=["XYZ"])
        assert ei.value.tickers == ["XYZ"]


# ===========================================================================
# Runner: descarga universo ∪ posiciones, aborta sin escribir estado
# ===========================================================================
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
STATE_FILES = ("last_rebalance.txt", "latest_snapshot.csv",
               "rebalance_history.csv", "exit_log.csv")


def _synthetic_prices(extra=("XYZ",)):
    rng = np.random.default_rng(7)
    cols = UNIVERSE + list(extra)
    # >= max required_history del stack (momentum 12-1 = 274 filas)
    idx = pd.bdate_range("2024-01-01", periods=300)
    data = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (len(idx), len(cols))), axis=0))
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Runner con HFT_HOME en tmp_path y fuente de precios sintetica mutable."""
    monkeypatch.setenv("HFT_HOME", str(tmp_path / "home"))
    source = {"prices": _synthetic_prices(), "requested": []}

    class SourceProvider:
        """Proveedor sin red que lee `source` al momento de cada descarga."""

        name = "fake"

        def get_bars(self, symbols, start, end):
            source["requested"].append(list(symbols))
            px = source["prices"]
            vol = pd.DataFrame(1e6, index=px.index, columns=px.columns)
            return FakeProvider(px, vol).get_bars(symbols, start, end)

    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": UNIVERSE}))
    monkeypatch.setattr(runner, "make_market_data_provider",
                        lambda *a, **k: SourceProvider())
    monkeypatch.setattr(runner, "utcnow",
                        lambda: after_close(source["prices"].index[-1]))
    from src.settings import load_settings
    settings = load_settings()
    settings.state_dir.mkdir(parents=True)
    return settings, source


def _seed_ledger(settings, entry_px=50.0):
    ledger = PositionLedger(initial_cash=100_000.0)
    ledger.apply_trades(
        pd.DataFrame([{"ticker": "AAA", "shares": 10.0},
                      {"ticker": "XYZ", "shares": 20.0}]),
        pd.Series({"AAA": 100.0, "XYZ": entry_px}), t=pd.Timestamp("2024-03-01"),
    )
    path = settings.state_dir / "ledger.json"
    runner.save_ledger(ledger, path)
    return path


def _assert_state_untouched(settings, ledger_path, ledger_bytes):
    assert ledger_path.read_bytes() == ledger_bytes
    for name in STATE_FILES:
        assert not (settings.state_dir / name).exists(), name


class TestRunnerMissingPrices:
    def test_fetches_universe_union_open_positions(self, live):
        settings, source = live
        _seed_ledger(settings)
        runner.run_once(force=True, start="2024-01-01")
        assert source["requested"], "fetch never called"
        for requested in source["requested"]:
            assert set(UNIVERSE) | {"XYZ"} <= set(requested)

    def test_aborts_without_writing_state_when_held_ticker_not_downloaded(self, live):
        settings, source = live
        source["prices"] = _synthetic_prices(extra=())  # Yahoo no trae XYZ
        ledger_path = _seed_ledger(settings)
        before = ledger_path.read_bytes()
        rc = runner.run_once(force=True, start="2024-01-01")
        assert rc == runner.EXIT_MISSING_PRICE
        _assert_state_untouched(settings, ledger_path, before)

    def test_aborts_when_held_ticker_has_no_price_at_t(self, live):
        settings, source = live
        stale = _synthetic_prices()
        # Deslistado: 10 NaN finales (>5% aun tras ffill) → must_keep lo
        # rescata con NaN en t en vez de recortar filas de todo el universo.
        stale.iloc[-10:, stale.columns.get_loc("XYZ")] = np.nan
        source["prices"] = stale
        ledger_path = _seed_ledger(settings)
        before = ledger_path.read_bytes()
        rc = runner.run_once(force=True, start="2024-01-01")
        assert rc == runner.EXIT_MISSING_PRICE
        _assert_state_untouched(settings, ledger_path, before)

    def test_aborts_when_only_held_ticker_survives_cleaning(self, live):
        """Universo sin datos utiles + XYZ sin precio en t → exit 5, no 2."""
        settings, source = live
        bad = _synthetic_prices()
        bad.iloc[:20, [bad.columns.get_loc(c) for c in UNIVERSE]] = np.nan
        bad.iloc[-4:, bad.columns.get_loc("XYZ")] = np.nan
        source["prices"] = bad
        ledger_path = _seed_ledger(settings)
        before = ledger_path.read_bytes()
        rc = runner.run_once(force=True, start="2024-01-01")
        assert rc == runner.EXIT_MISSING_PRICE
        _assert_state_untouched(settings, ledger_path, before)

    def test_out_of_universe_holding_is_priced_and_sold(self, live, monkeypatch):
        settings, _ = live
        ledger_path = _seed_ledger(settings)
        seen = []
        original = runner.build_targets

        def spy(state, prices, volumes, t, cfg, **k):
            res = original(state, prices, volumes, t, cfg, **k)
            seen.append((list(k["universe"]), list(volumes.columns),
                         list(res.weights.index)))
            return res

        monkeypatch.setattr(runner, "build_targets", spy)
        # Sin reglas de salida: el cierre de XYZ viene solo de peso objetivo 0.
        rc = runner.run_once(force=True, start="2024-01-01",
                             take_profit_dollars=None, stop_loss_dollars=None)
        assert rc == 0
        # La estrategia sigue viendo solo el universo (XYZ solo se valora).
        assert seen == [(UNIVERSE, UNIVERSE, UNIVERSE)]
        after = runner.load_ledger(ledger_path)
        assert "XYZ" not in after.positions().index
        hist = pd.read_csv(settings.state_dir / "rebalance_history.csv")
        assert hist["nav"].iloc[-1] > 99_000.0  # NAV no colapsa

    def test_dollar_exit_on_out_of_universe_holding(self, live):
        settings, _ = live
        ledger_path = _seed_ledger(settings, entry_px=1.0)  # +$ enorme → take-profit
        rc = runner.run_once(force=True, start="2024-01-01")
        assert rc == 0
        exits = pd.read_csv(settings.state_dir / "exit_log.csv")
        assert "XYZ" in set(exits["ticker"])
        assert "XYZ" not in runner.load_ledger(ledger_path).positions().index


class _UnpricedBroker:
    """Alpaca falso (sin HTTP) que tiene ZZZ, ausente de prices_t."""

    def __init__(self, priced="AAPL"):
        self.priced = priced
        self.submitted = []

    def check(self):
        return {"equity": "10000", "market_open": True}

    def get_account(self):
        return {"equity": "10000", "cash": "8500"}

    def get_positions(self):
        return pd.DataFrame([
            {"ticker": self.priced, "qty": 5.0, "market_value": 1000.0,
             "avg_entry": 200.0, "side": "long"},
            {"ticker": "ZZZ", "qty": 3.0, "market_value": 500.0,
             "avg_entry": 150.0, "side": "long"},
        ])

    def submit_orders(self, plan, dry_run=True, **_kw):
        self.submitted.append(plan)
        return []

    def poll_orders(self, ids, timeout=30.0):
        return {}


class TestBrokerStepUnpriced:
    def test_broker_mode_aborts_before_mirror_or_orders(self, live, monkeypatch):
        settings, _ = live
        ledger_path = _seed_ledger(settings)
        before = ledger_path.read_bytes()
        broker = _UnpricedBroker(priced="AAA")
        monkeypatch.setattr(runner, "make_broker_client", lambda _state: broker)
        rc = runner.run_once(force=True, start="2024-01-01",
                             mode="broker", submit=True)
        assert rc == runner.EXIT_MISSING_PRICE
        assert broker.submitted == []
        _assert_state_untouched(settings, ledger_path, before)  # sin espejo
        assert not list(settings.state_dir.glob("ledger_local_backup_*"))
        assert not list(settings.state_dir.glob("broker_plan_*"))

    def test_aborts_before_any_order(self, tmp_path):
        broker = _UnpricedBroker()
        with pytest.raises(MissingPriceError) as ei:
            runner.run_broker_step(pd.Series({"AAPL": 1.0}),
                                   pd.Series({"AAPL": 200.0}),
                                   T0, tmp_path, submit=True, client=broker)
        assert ei.value.tickers == ["ZZZ"]
        assert broker.submitted == []
        assert not list(tmp_path.glob("broker_plan_*.csv"))
