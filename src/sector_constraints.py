"""
sector_constraints.py — Paso 3: topes por grupo (sector / sleeve COL).

Implementa los topes de `criterios.md v2` como proyeccion determinista
post-optimizacion (mismo espiritu que el blending de turnover en
`rebalancing_engine.py`): reduce proporcionalmente los grupos excedidos
y redistribuye el deficit segun headroom, respetando topes por activo.

Soporta topes simples ("co/Financials" <= 0.25) y combinados
(("co/Financials", "co/Energy") <= 0.40, tope Fin+Energia del sleeve).

Grupos tipicos: "us/Technology", "co/Financials" (ver build_mixed_groups).
"""

from __future__ import annotations

import logging
from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("sector_constraints")

CapKey = Union[str, Tuple[str, ...], List[str]]


def build_mixed_groups(universe: pd.DataFrame) -> pd.Series:
    """Construye el mapeo ticker -> grupo "mercado/sector".

    Args:
        universe: DataFrame con columnas [ticker, market, sector]
            (ej. `universe_mixed.csv` via `load_universe`).

    Returns:
        pd.Series indexada por ticker con valores "us/Technology", etc.
    """
    required = {"ticker", "market", "sector"}
    if not required.issubset(universe.columns):
        raise ValueError(f"universe sin columnas {required - set(universe.columns)}")
    groups = {
        row.ticker: f"{row.market}/{row.sector}"
        for row in universe.itertuples()
    }
    return pd.Series(groups)


def co_sleeve_caps(
    fin_energy_cap: float = 0.40, sleeve_weight: float = 0.25
) -> Dict[tuple, float]:
    """Tope combinado Fin+Energia del sleeve COL (criterios.md v2).

    `fin_energy_cap` es fraccion DEL SLEEVE; `apply_group_caps` trabaja en
    fraccion del NAV total, asi que se escala: cap_nav = cap * sleeve_weight
    (0.40 * 0.25 = 0.10 NAV). `sleeve_weight` es el peso objetivo del sleeve
    COL en el NAV (20-30% segun perfil); no se impone aqui.
    """
    if not 0 < fin_energy_cap <= 1:
        raise ValueError(f"fin_energy_cap fuera de (0,1]: {fin_energy_cap}")
    if not 0 < sleeve_weight <= 1:
        raise ValueError(f"sleeve_weight fuera de (0,1]: {sleeve_weight}")
    return {("co/Financials", "co/Energy"): float(fin_energy_cap * sleeve_weight)}


def _normalize_caps(
    caps: Dict[CapKey, float],
    tickers: List[str],
    groups: pd.Series,
) -> List[Tuple[List[str], float]]:
    """Resuelve topes (simples o combinados) a listas de tickers."""
    resolved: List[Tuple[List[str], float]] = []
    for key, cap in caps.items():
        names = list(key) if isinstance(key, (tuple, list, frozenset)) else [key]
        if not 0 <= cap <= 1:
            raise ValueError(f"Tope fuera de [0,1] para {key}: {cap}")
        members = [t for t in tickers if groups.get(t) in names]
        if not members:
            logger.warning("sector_constraints: tope %s sin tickers, se ignora.", key)
            continue
        resolved.append((members, float(cap)))
    return resolved


def min_blend_to_caps(
    w_from: pd.Series,
    w_to: pd.Series,
    groups: Union[pd.Series, Dict[Hashable, str]],
    caps: Dict[CapKey, float],
    tol: float = 1e-9,
) -> float:
    """Menor alpha en [0, 1] tal que el blend renormalizado cumple los topes.

    El blend es ``w_from + alpha * (w_to - w_from)`` (indice de ``w_to``),
    renormalizado a suma 1 como hace el tope de turnover del rebalanceador.
    Cada tope ``sum(w[g]) <= cap * sum(w)`` es lineal en alpha, asi que el
    minimo es exacto: se avanza hacia el objetivo solo lo necesario.

    Raises:
        ValueError: Si ``w_to`` (objetivo) ya incumple algun tope: ningun
            punto del segmento lo arregla.
    """
    tickers = w_to.index.tolist()
    a = pd.Series(w_from, dtype=float).reindex(tickers).fillna(0.0).clip(lower=0.0)
    b = pd.Series(w_to, dtype=float).clip(lower=0.0)
    alpha = 0.0
    for members, cap in _normalize_caps(caps, tickers, pd.Series(groups).reindex(tickers)):
        excess_from = float(a[members].sum()) - cap * float(a.sum())
        excess_to = float(b[members].sum()) - cap * float(b.sum())
        if excess_to > tol:
            raise ValueError(
                f"Pesos objetivo incumplen el tope {cap} en {members}."
            )
        if excess_from > tol:
            alpha = max(alpha, excess_from / (excess_from - excess_to))
    return min(alpha, 1.0)


def sell_down_to_caps(
    w: pd.Series,
    w_target: pd.Series,
    groups: Union[pd.Series, Dict[Hashable, str]],
    caps: Dict[CapKey, float],
    tol: float = 1e-9,
    max_iter: int = 100,
) -> pd.Series:
    """Repara topes incumplidos en ``w`` con el minimo turnover extra.

    Por cada tope excedido vende solo el exceso en miembros por encima de su
    objetivo y lo recompra fuera del grupo en nombres por debajo de su
    objetivo (a prorrata de la brecha): turnover extra = 2 * exceso, todo
    hacia ``w_target`` sin sobrepasarlo (se preservan >= 0 y los topes por
    activo que cumpla el objetivo). Topes cruzados (recomprar en un grupo
    ya reparado) se resuelven con pasadas sucesivas; si tras ``max_iter``
    queda incumplimiento, avanza por el segmento hacia el objetivo
    (``min_blend_to_caps``).

    Returns:
        Pesos (indice de ``w_target``, suma 1) dentro de los topes.

    Raises:
        ValueError: Si ``w_target`` incumple algun tope.
    """
    tickers = w_target.index.tolist()
    b = pd.Series(w_target, dtype=float).clip(lower=0.0)
    b = b / b.sum()
    out = pd.Series(w, dtype=float).reindex(tickers).fillna(0.0).clip(lower=0.0)
    out = out / out.sum()
    g = pd.Series(groups).reindex(tickers)
    constraints = _normalize_caps(caps, tickers, g)
    for members, cap in constraints:
        if float(b[members].sum()) > cap + tol:
            raise ValueError(f"Pesos objetivo incumplen el tope {cap} en {members}.")
    for _ in range(max_iter):
        repaired = False
        for members, cap in constraints:
            excess = float(out[members].sum()) - cap
            if excess <= tol:
                continue
            # Con objetivo dentro del tope, ambas brechas suman >= exceso
            gap = b - out
            in_group = out.index.isin(members)
            sell = (-gap).clip(lower=0.0).where(in_group, 0.0)
            buy = gap.clip(lower=0.0).where(~in_group, 0.0)
            out = out - excess * sell / sell.sum() + excess * buy / buy.sum()
            repaired = True
        if not repaired:
            break
    alpha = min_blend_to_caps(out, b, g, caps, tol)
    return out + alpha * (b - out) if alpha > 0 else out


def apply_group_caps(
    weights: pd.Series,
    groups: Union[pd.Series, Dict[Hashable, str]],
    caps: Dict[CapKey, float],
    max_weight: Optional[float] = None,
    max_weight_by_ticker: Optional[pd.Series] = None,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> pd.Series:
    """Proyecta pesos para cumplir topes por grupo.

    Args:
        weights: Pesos base (se clippean a >= 0 y normalizan a 1).
        groups: Mapeo ticker -> grupo (Series o dict).
        caps: Tope por grupo o tupla de grupos (tope combinado).
        max_weight: Tope global por activo (respeta R-04).
        max_weight_by_ticker: Tope por activo (Series; ej. 0.15 en CO).
        tol: Tolerancia de cumplimiento.
        max_iter: Iteraciones maximas antes de declarar no-factibilidad.

    Returns:
        Pesos que suman 1, long-only, dentro de topes.

    Raises:
        ValueError: Topes inválidos o problema infactible.
        RuntimeError: Sin convergencia (topes casi infactibles).
    """
    if not caps:
        w = weights.clip(lower=0.0).astype(float)
        return w / w.sum() if w.sum() > 0 else w

    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    tickers = w.index.tolist()
    g = pd.Series(groups).reindex(tickers)
    constraints = _normalize_caps(caps, tickers, g)

    # Capacidad total: si TODOS los tickers pertenecen a grupos topados,
    # los topes son simples (grupos disjuntos) y suman < 1, es infactible.
    # Un tope parcial siempre es factible: el resto sin tope absorbe.
    if (
        g.notna().all()
        and constraints
        and all(not isinstance(k, (tuple, list, frozenset)) for k in caps)
    ):
        capped = {t for members, _ in constraints for t in members}
        total = sum(cap for _, cap in constraints)
        if set(tickers) <= capped and total < 1 - tol:
            raise ValueError(
                f"Topes infactibles: cubren todo el universo y suman {total:.4f} < 1."
            )

    if max_weight_by_ticker is not None:
        ub = pd.Series(max_weight_by_ticker, dtype=float).reindex(tickers)
        if max_weight is not None:
            ub = ub.fillna(max_weight)
    elif max_weight is not None:
        ub = pd.Series(max_weight, index=tickers, dtype=float)
    else:
        ub = None

    if w.sum() <= 0:
        w = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
    else:
        w = w / w.sum()

    for it in range(max_iter):
        if ub is not None:
            w = w.clip(upper=ub)
        over = [(m, c) for m, c in constraints if w[m].sum() > c + tol]
        ub_viol = bool(ub is not None and (w > ub + tol).any())
        deficit = 1.0 - float(w.sum())
        if not over and not ub_viol and abs(deficit) <= tol:
            break
        frozen = set()
        for members, cap in over:
            s = float(w[members].sum())
            w[members] *= cap / s
            frozen.update(members)
        deficit = 1.0 - float(w.sum())
        if deficit > tol:
            headroom = pd.Series(np.inf, index=tickers, dtype=float)
            if ub is not None:
                headroom = (ub - w).clip(lower=0.0)
            else:
                headroom = w.copy()
                headroom[headroom <= 0] = 1.0
            # Preferir no-congelados; si no hay capacidad, cualquiera con headroom
            pref = [t for t in tickers if t not in frozen and headroom[t] > tol]
            recipients = pref or [t for t in tickers if headroom[t] > tol]
            room = float(headroom[recipients].sum())
            if room <= tol:
                raise ValueError(
                    "Topes infactibles: sin headroom para redistribuir "
                    f"(deficit={deficit:.6f})."
                )
            w[recipients] += deficit * headroom[recipients] / room
        elif deficit < -tol:
            w /= w.sum()
    else:
        raise RuntimeError(
            "apply_group_caps no convergió: topes casi infactibles o "
            "conflictivos con max_weight."
        )

    # Verificacion final (fail-fast honesto)
    for members, cap in constraints:
        if float(w[members].sum()) > cap + 1e-6:
            raise RuntimeError(f"Tope incumplido tras proyeccion: {members} > {cap}")
    w /= w.sum()
    return w
