"""
risk_profiles.py — Paso 3: presets por perfil de riesgo (uso personal).

EXPERIMENTAL: no conectado a ninguna ruta de ejecución (el runner vivo lee
config/live.toml; los experimentos arman su BacktestConfig). Ver README →
"Módulos experimentales".

Mapea conservador / moderado / agresivo a `BacktestConfig` listos para
correr. El tamano del sleeve COL (20-30% NAV) es decision de universo
(universe_co.csv vs universe_mixed.csv), no del preset.

Uso:
    from src.risk_profiles import profile_config, mixed_sleeve_config
    cfg = profile_config("moderado")
    cfg = mixed_sleeve_config("conservador")  # grupos + tope Fin+Energia CO
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.backtest_engine import BacktestConfig
from src.data_client import load_universe
from src.sector_constraints import build_mixed_groups, co_sleeve_caps
from src.settings import load_settings

RISK_PROFILES: Dict[str, Dict] = {
    "conservador": {
        "opt_method": "robust",
        "kappa": 0.15,
        "max_weight": 0.15,
        "turnover_limit": 0.15,
        "rebalance_freq": "Q",
        "commission_bps": 7.0,
        "spread_bps": 3.0,
    },
    "moderado": {
        "opt_method": "robust",
        "kappa": 0.10,
        "max_weight": 0.20,
        "turnover_limit": 0.20,
        "rebalance_freq": "Q",
        "commission_bps": 5.0,
        "spread_bps": 2.0,
    },
    "agresivo": {
        "opt_method": "mv_classic",
        "kappa": 0.10,
        "max_weight": 0.30,
        "turnover_limit": 0.25,
        "rebalance_freq": "M",
        "commission_bps": 5.0,
        "spread_bps": 2.0,
    },
}


def profile_config(name: str, **overrides) -> BacktestConfig:
    """Devuelve BacktestConfig del perfil indicado mas overrides.

    Raises:
        ValueError: perfil desconocido.
    """
    key = name.strip().lower()
    if key not in RISK_PROFILES:
        raise ValueError(
            f"Perfil '{name}' desconocido. Validos: {sorted(RISK_PROFILES)}."
        )
    params = {**RISK_PROFILES[key], **overrides}
    return BacktestConfig(**params)


def mixed_sleeve_config(
    profile: str = "moderado",
    fin_energy_cap: float = 0.40,
    co_sleeve_weight: float = 0.25,
    co_max_weight: float = 0.15,
    universe_path: Optional[str] = None,
    **overrides,
) -> BacktestConfig:
    """Preset + grupos del universo mixto y topes del sleeve COL.

    - groups desde `universe_mixed.csv` ("us/...", "co/..."). Una ruta
      relativa en `universe_path` se resuelve en la raiz del repo (nunca el cwd).
    - Tope combinado Fin+Energia CO (criterios.md v2): `fin_energy_cap` es
      fraccion del sleeve, escalada por `co_sleeve_weight` a fraccion NAV.
      El peso total del sleeve NO se impone (decision de universo).
    - Tope por activo 0.15 en grupos "co/*".
    """
    cfg = profile_config(profile, **overrides)
    path = Path(universe_path or "universe_mixed.csv")
    if not path.is_absolute():
        path = load_settings().universe_path(str(path))
    universe = load_universe(str(path))
    groups = build_mixed_groups(universe)
    cfg.groups = groups.to_dict()
    cfg.group_caps = co_sleeve_caps(fin_energy_cap, co_sleeve_weight)
    cfg.max_weight_by_group = {
        g: co_max_weight for g in groups.unique() if g.startswith("co/")
    }
    return cfg
