"""
conviction_scoring.py — T-36: scorer de convicción determinista, basado en reglas.

Combina signal_stack, el régimen y un drift opcional. No se admiten LLM
ni otras APIs de IA, ni en backtest ni en vivo: un modelo de lenguaje no
tiene cutoff de conocimiento mapeado a fechas históricas y rompería R-05
(look-ahead).

``ConvictionScorer`` es el contrato (``__call__(features) -> float``) de
un scorer de reglas. La implementación de referencia es ``conviction_score``.

Restricciones cubiertas: R-05 (sin look-ahead; no llama APIs), R-10.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.drift_monitor import DriftReport
from src.regime_detector import (
    HIGH_VOL_RANGING,
    HIGH_VOL_TRENDING,
    LOW_VOL_RANGING,
    LOW_VOL_TRENDING,
)

logger = logging.getLogger("conviction_scoring")

from src.settings import SEED

# Regime multipliers applied to composite signal scores
_REGIME_MULTIPLIER = {
    LOW_VOL_TRENDING: 1.15,     # favour momentum/trend signals
    LOW_VOL_RANGING: 1.00,
    HIGH_VOL_TRENDING: 0.85,    # dampen but keep direction
    HIGH_VOL_RANGING: 0.60,     # choppy crisis → reduce conviction
}


# ===================================================================
# Contrato del scorer determinista (sin LLM)
# ===================================================================
@runtime_checkable
class ConvictionScorer(Protocol):
    """Contrato de un scorer de convicción determinista, basado en reglas.

    Recibe un diccionario de features fechado en t y devuelve un escalar
    en un rango documentado (p. ej. ``[0, 1]``). No se admiten LLM ni
    otras APIs de IA, ni en backtest ni en vivo. La implementación de
    referencia es ``conviction_score`` (signal_stack + régimen + drift
    opcional): mismas entradas, mismo resultado.

    Ejemplo::

        def rule_score(features: Mapping[str, Any]) -> float:
            return float(features["signal"])

        scorer: ConvictionScorer = rule_score
        value = scorer({"signal": 0.4})
    """

    def __call__(self, features: Mapping[str, Any]) -> float:
        """Devuelve una convicción escalar en un rango documentado (p. ej. [0, 1])."""
        ...


# ===================================================================
# Rule-based proxy
# ===================================================================
def conviction_score(
    signal_scores: pd.Series,
    regime: str,
    drift_report: Optional[DriftReport] = None,
) -> pd.Series:
    """Combine signal scores with regime (and optional drift) adjustment.

    Args:
        signal_scores: Composite scores from signal_stack (ticker-indexed).
        regime: Regime state string from regime_detector.
        drift_report: Optional DriftReport; if present and should_rebalance
            with high total_drift, conviction is slightly reduced (noise).

    Returns:
        Series of conviction-adjusted scores (same index as signal_scores).
        Higher = stronger long conviction. Deterministic; no I/O.
    """
    scores = signal_scores.astype(float).copy()
    mult = _REGIME_MULTIPLIER.get(regime, 1.0)

    # In high_vol_ranging, additionally compress extreme positive momentum-like
    # scores (proxy: scores above +1σ get extra dampening)
    adjusted = scores * mult
    if regime == HIGH_VOL_RANGING:
        sd = float(scores.std(ddof=1) or 1.0)
        if sd > 1e-12:
            extreme = scores > sd
            adjusted = adjusted.where(~extreme, scores * (mult * 0.75))

    if drift_report is not None and drift_report.total_drift > 0.15:
        # High weight drift → slightly less confidence in current tilt
        drift_scale = max(0.7, 1.0 - 0.5 * float(drift_report.total_drift))
        adjusted = adjusted * drift_scale

    logger.debug(
        "conviction_score regime=%s mult=%.2f mean=%.3f",
        regime, mult, float(adjusted.mean()) if len(adjusted) else 0.0,
    )
    return adjusted


def conviction_to_tilts(
    conviction: pd.Series,
    base_weights: Optional[pd.Series] = None,
    long_only: bool = True,
) -> pd.Series:
    """Map conviction scores to portfolio weight tilts.

    If base_weights is provided, tilts them multiplicatively by
    softplus(conviction) and renorm. Otherwise converts conviction
    ranks to a long-only weight vector.

    Args:
        conviction: Conviction scores by ticker.
        base_weights: Optional base weights to tilt.
        long_only: If True, clip negative contributions to a small floor.

    Returns:
        Weight Series summing to 1.
    """
    c = conviction.fillna(0.0).astype(float)
    # Soft positive transform: keep relative ordering, always > 0
    tilt = np.exp(np.clip(c, -3.0, 3.0))  # soft emphasis
    if long_only:
        tilt = tilt.clip(lower=1e-8)

    if base_weights is not None:
        bw = base_weights.reindex(c.index).fillna(0.0)
        raw = bw * tilt
    else:
        raw = tilt

    s = float(raw.sum())
    if s < 1e-12:
        return pd.Series(1.0 / len(c), index=c.index)
    return (raw / s).astype(float)
