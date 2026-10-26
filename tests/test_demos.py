"""
test_demos.py — Los demos `python -m src.<modulo>` corren sin red ni data/.

Los demos __main__ de estimacion usan datos sinteticos (src.demo_data); nunca
leen archivos que pueden no existir (p. ej. data/clean_prices.parquet). Se
ejecutan en un interprete limpio con cwd y HFT_HOME vacios, asi una lectura
relativa al cwd o a data/ falla aqui.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.demo_data import synthetic_prices, synthetic_volumes

REPO_ROOT = Path(__file__).resolve().parents[1]

OFFLINE_DEMOS = (
    "src.rolling_engine",
    "src.covariance_estimators",
    "src.expected_returns",
    "src.universe_filter",
)


@pytest.mark.parametrize("module", OFFLINE_DEMOS)
def test_demo_runs_from_empty_dir(module, tmp_path):
    pythonpath = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)
    env = {**os.environ, "HFT_HOME": str(tmp_path), "PYTHONPATH": pythonpath}
    proc = subprocess.run(
        [sys.executable, "-m", module], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip()


class TestSyntheticData:
    def test_prices_deterministic_and_clean(self):
        a = synthetic_prices(seed=7)
        b = synthetic_prices(seed=7)
        assert a.equals(b)
        assert not a.isna().any().any()
        assert (a > 0).all().all()
        assert "SPY" in a.columns  # proxy de mercado para capm_shrunk

    def test_seed_changes_panel(self):
        assert not synthetic_prices(seed=1).equals(synthetic_prices(seed=2))

    def test_volumes_aligned_and_positive(self):
        prices = synthetic_prices()
        vols = synthetic_volumes(prices)
        assert vols.index.equals(prices.index)
        assert vols.columns.equals(prices.columns)
        assert (vols > 0).all().all()

    def test_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        synthetic_volumes(synthetic_prices())
        assert np.random.random() == expected
