"""El núcleo reutilizable importa sin el stack research (HFT-02).

Subproceso limpio: los paquetes pesados quedan en ``sys.modules`` como
``None`` (``import`` lanza ``ImportError``) antes de cargar los módulos.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Nombres de import, no de distribución (scikit-learn → sklearn).
HEAVY = (
    "cvxpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "torch",
    "tensorflow",
    "xgboost",
    "lightgbm",
    "openai",
    "anthropic",
    "langchain",
    "transformers",
    "yfinance",
    "plotly",
    "matplotlib",
    "pyarrow",
    "curl_cffi",
    "lxml",
    "clarabel",
    "osqp",
    "scs",
    "ecos",
    "cvxopt",
)

# Infraestructura que el intradía reutiliza. Si alguno vuelve a arrastrar
# un paquete de HEAVY, el subproceso sale distinto de cero.
CORE_MODULES = (
    "src.hft",
    "src.settings",
    "src.instruments",
    "src.position_ledger",
    "src.brokers.alpaca_adapter",
    "src.brokers.pre_trade",
    "src.live_state",
    "src.notifier",
    "src.provenance",
    "src.metrics_calculator",
    "src.market_data",
    "src.regime_detector",
    "src.exit_rules",
    "src.signal_stack",
    "src.order_execution",
    "src.universe_filter",
)


def test_core_modules_import_with_heavy_packages_blocked():
    script = textwrap.dedent(
        """\
        import sys
        heavy = {heavy!r}
        modules = {modules!r}
        for name in heavy:
            sys.modules[name] = None
        try:
            import scipy  # noqa: F401
        except ImportError:
            pass
        else:
            raise SystemExit("el bloqueo no impide importar scipy")
        failed = []
        for mod in modules:
            try:
                __import__(mod)
            except Exception as exc:
                failed.append(f"{{mod}}: {{type(exc).__name__}}: {{exc}}")
        if failed:
            raise SystemExit("\\n".join(failed))
        """
    ).format(heavy=HEAVY, modules=CORE_MODULES)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("PYTHONSAFEPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
