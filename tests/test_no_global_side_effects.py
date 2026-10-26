"""
test_no_global_side_effects.py — La libreria no configura logging ni RNG global.

- Importar modulos de src/ (y el script) no añade handlers al root logger:
  se verifica en un interprete limpio (subprocess), porque pytest ya instala
  sus propios handlers.
- Ejecutar backtest / rebalanceo no modifica el estado de np.random global.
- La semilla llega explicita a quien usa aleatoriedad (random_replacement).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_py(code: str) -> dict:
    """Ejecuta `code` en un interprete limpio desde la raiz; devuelve JSON impreso."""
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
        text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


_ROOT_STATE = (
    "import json, logging; r = logging.getLogger();"
    "print(json.dumps({'handlers': len(r.handlers), 'level': r.level}))"
)


class TestLoggingNotConfiguredOnImport:
    def test_import_data_client_adds_no_root_handlers(self):
        state = _run_py("import src.data_client\n" + _ROOT_STATE)
        assert state == {"handlers": 0, "level": 30}  # WARNING = default

    def test_import_all_library_modules_adds_no_root_handlers(self):
        mods = sorted(
            "src." + ".".join(p.relative_to(REPO_ROOT / "src").with_suffix("").parts)
            for p in (REPO_ROOT / "src").rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "__init__.py"
        ) + ["scripts.run_periodic_rebalance"]
        code = "import importlib\n" + "".join(
            f"importlib.import_module({m!r})\n" for m in mods) + _ROOT_STATE
        assert _run_py(code) == {"handlers": 0, "level": 30}

    def test_no_basicconfig_outside_settings(self):
        """basicConfig solo via settings.setup_logging (nunca directo)."""
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for d in ("src", "scripts", "experiments")
            for p in (REPO_ROOT / d).rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "settings.py"
            and "basicConfig" in p.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestSetupLogging:
    def test_setup_logging_configures_root_once(self):
        state = _run_py(
            "from src.settings import setup_logging\n"
            "setup_logging('DEBUG'); setup_logging('DEBUG')\n" + _ROOT_STATE)
        assert state == {"handlers": 1, "level": 10}

    def test_setup_logging_accepts_int_level(self):
        state = _run_py(
            "import logging\nfrom src.settings import setup_logging\n"
            "setup_logging(logging.WARNING)\n" + _ROOT_STATE)
        assert state == {"handlers": 1, "level": 30}


def _synthetic():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=320)
    cols = [f"T{i}" for i in range(8)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (len(dates), len(cols))), axis=0)),
        index=dates, columns=cols)
    vols = pd.DataFrame(2e6, index=dates, columns=cols)
    return prices, vols


def _same_state(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


class TestNoGlobalRng:
    def test_no_np_random_seed_in_code(self):
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for d in ("src", "scripts", "experiments")
            for p in (REPO_ROOT / d).rglob("*.py")
            if "__pycache__" not in p.parts
            and re.search(r"np\.random\.seed\(|np\.random\.RandomState\(",
                          p.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_run_backtest_leaves_global_rng_untouched(self):
        from src.backtest_engine import BacktestConfig, run_backtest
        prices, vols = _synthetic()
        np.random.seed(12345)
        before = np.random.get_state()
        run_backtest(prices, vols, BacktestConfig(
            window=60, warmup=60, rebalance_freq="QE", replacement_rule="random",
            seed=7))
        assert _same_state(before, np.random.get_state())

    def test_backtest_deterministic_regardless_of_global_rng(self):
        from src.backtest_engine import BacktestConfig, run_backtest
        prices, vols = _synthetic()
        cfg = BacktestConfig(window=60, warmup=60, rebalance_freq="QE",
                             replacement_rule="random")
        np.random.seed(1)
        a = run_backtest(prices, vols, cfg).nav
        np.random.seed(999)
        np.random.random(1000)
        b = run_backtest(prices, vols, cfg).nav
        pd.testing.assert_series_equal(a, b)


class TestSeedPassedExplicitly:
    def test_backtest_seed_reaches_random_replace(self, monkeypatch):
        import src.rebalancing_engine as reb
        from src.backtest_engine import BacktestConfig, run_backtest
        seen = []
        real = reb.random_replace

        def spy(*args, **kwargs):
            seen.append(kwargs.get("base_seed"))
            return real(*args, **kwargs)

        monkeypatch.setattr(reb, "random_replace", spy)
        prices, vols = _synthetic()
        run_backtest(prices, vols, BacktestConfig(
            window=60, warmup=60, rebalance_freq="QE",
            replacement_rule="random", seed=7))
        assert seen and set(seen) == {7}

    def test_rebalance_seed_changes_random_replacement(self):
        from src.rebalancing_engine import rebalance
        prices, vols = _synthetic()
        t = prices.index[200]
        port = ["T0", "T1", "T2", "T3", "T4"]
        w = pd.Series(0.2, index=port)
        kw = dict(t=t, w_current=w, w_target=w, prices=prices, volumes=vols,
                  rule="random", universe=list(prices.columns), replace_pct=0.4)
        runs = {s: tuple(sorted(rebalance(seed=s, **kw).w_new.index)) for s in range(20)}
        # determinista por semilla, y la semilla importa
        assert tuple(sorted(rebalance(seed=3, **kw).w_new.index)) == runs[3]
        assert len(set(runs.values())) > 1

    def test_default_seed_is_settings_seed(self):
        """Sin seed explicito se usa settings.SEED (42): mismo resultado que antes."""
        from src.random_replacement import random_replace
        from src.settings import SEED
        t = pd.Timestamp("2021-03-31")
        port = ["T0", "T1", "T2", "T3", "T4"]
        uni = [f"T{i}" for i in range(8)]
        a = random_replace(t, port, uni, replace_pct=0.4)
        b = random_replace(t, port, uni, replace_pct=0.4, base_seed=SEED)
        assert a.new_universe == b.new_universe
