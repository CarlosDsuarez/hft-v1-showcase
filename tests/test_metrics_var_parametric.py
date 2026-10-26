"""var_parametric sin scipy: mismo cuantil que norm.ppf (HFT-02).

Los valores esperados son los que devolvía ``scipy.stats.norm.ppf``
(scipy 1.18.1) sobre esta serie. Tolerancia 1e-10. No importa scipy:
el núcleo tiene que poder correr el test.
"""

from __future__ import annotations

import pandas as pd

from src.metrics_calculator import var_parametric

# Serie fija (>= 10 observaciones: por debajo, var_parametric devuelve 0).
_RETURNS = pd.Series([
    0.01, -0.02, 0.005, 0.0, -0.01, 0.015, -0.008, 0.012, -0.004, 0.007,
    0.002, -0.011, 0.009, -0.003, 0.006, -0.015, 0.004, 0.001, -0.006, 0.013,
])

# scipy.stats.norm.ppf(1 - confianza) dentro de VaR = -(mu + z * sigma),
# con mu = mean y sigma = std muestral (ddof=1), igual que antes.
_SCIPY_VAR = {
    0.90: 0.01218781646537827,
    0.95: 0.015742113217750674,
    0.99: 0.022409382816525387,
}


def test_var_parametric_matches_previous_scipy_ppf():
    for confidence, expected in _SCIPY_VAR.items():
        got = var_parametric(_RETURNS, confidence=confidence)
        assert abs(got - expected) < 1e-10, (confidence, got, expected)


def test_var_parametric_short_series_is_zero():
    assert var_parametric(_RETURNS.iloc[:9], confidence=0.95) == 0.0
