"""Validación estadística del backtest intradía (HFT-10).

PSR y DSR de Bailey y López de Prado (2012, 2014), PBO por CSCV de
Bailey, Borwein, López de Prado y Zhu (2017) y la longitud mínima de
track record. Φ y Φ⁻¹ salen de ``statistics.NormalDist``: este módulo
no importa scipy.

El Sharpe que entra en PSR/DSR, la varianza ``V[SR_n]`` y ``T`` van en
la misma frecuencia (aquí, la diaria) y no se anualizan. Anualizar es
solo para mostrar el número, fuera de estas fórmulas.

``γ₄`` es la curtosis de Pearson (una normal vale 3), no el exceso.
``γ_E`` es la constante de Euler–Mascheroni.

El registro ``trials.csv`` es el mecanismo de honestidad: ``N`` del DSR
de una estrategia es el número de filas ya escritas para ella. Reescribir
el mismo ``run_id`` no suma otra prueba.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

# Euler–Mascheroni. ``math`` no la expone en 3.12.
EULER_MASCHERONI = 0.5772156649015328606

# Columnas del registro. ``n`` es T, las observaciones del Sharpe (días).
# ``n_trades`` es el número de operaciones cerradas. ``date`` es el fin
# de la muestra, no el reloj de pared: el módulo no llama a datetime.now.
TRIAL_FIELDS = (
    "run_id",
    "config_hash",
    "strategy",
    "sr",
    "n",
    "n_trades",
    "date",
    "feed",
    "start",
    "end",
    "holdout_unlocked",
    "cost_multiplier",
    "extra_latency_bars",
)

_NORMAL = statistics.NormalDist()


def phi(z: float) -> float:
    """Φ(z), la normal estándar acumulada."""
    return _NORMAL.cdf(float(z))


def phi_inv(p: float) -> float:
    """Φ⁻¹(p). ``p`` tiene que estar en (0, 1)."""
    probability = float(p)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Φ⁻¹ pide una probabilidad en (0, 1), llegó {p!r}")
    return _NORMAL.inv_cdf(probability)


def sharpe_ratio(returns: np.ndarray) -> float:
    """Media / desviación de la serie tal cual. No anualiza.

    Con menos de dos observaciones, o con desviación nula, devuelve 0.
    La desviación es la muestral (``ddof=1``), la misma de los momentos.
    """
    values = _finite_vector(returns, "returns")
    if values.size < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    return float(values.mean() / std)


def sample_moments(returns: np.ndarray) -> Tuple[float, float, float, int]:
    """``(SR, γ₃, γ₄, T)`` de la serie, sin anualizar.

    ``γ₄`` no es exceso: una normal da 3. Los momentos centrales son los
    de la muestra (dividir por ``T``, no por ``T-1``), que es la convención
    de Bailey y López de Prado para el PSR. ``T`` es el tamaño de la serie.
    """
    values = _finite_vector(returns, "returns")
    length = int(values.size)
    sr = sharpe_ratio(values)
    if length < 2:
        return sr, 0.0, 3.0, length
    centered = values - values.mean()
    moment2 = float(np.mean(centered ** 2))
    if moment2 <= 0.0:
        return sr, 0.0, 3.0, length
    moment3 = float(np.mean(centered ** 3))
    moment4 = float(np.mean(centered ** 4))
    skew = moment3 / (moment2 ** 1.5)
    kurtosis = moment4 / (moment2 ** 2)
    return sr, float(skew), float(kurtosis), length


def estimator_variance(sr_hat: float, skew: float, kurtosis: float, t: int) -> float:
    """``V[SR_n]`` analítica, en la misma frecuencia que ``SR̂``.

    ``(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) / (T−1)``. Es la varianza del
    estimador cuando no hay una rejilla observada. Con ``T < 2`` no
    está definida.
    """
    length = int(t)
    if length < 2:
        raise ValueError("V[SR_n] analítica pide T >= 2")
    inside = _sharpe_scale(sr_hat, skew, kurtosis)
    if inside <= 0.0:
        raise ValueError(
            "la combinación de SR, asimetría y curtosis no da una varianza positiva"
        )
    return inside / (length - 1)


def expected_maximum_sr(sr_variance: float, n_trials: int) -> float:
    """``SR₀``, el Sharpe máximo esperado bajo la nula de ``N`` pruebas.

    ``sqrt(V) · ((1−γ_E)·Φ⁻¹(1−1/N) + γ_E·Φ⁻¹(1−1/(N·e)))``.

    Con ``N < 2`` no hay selección: ``SR₀ = 0`` (el PSR contra cero).
    ``Φ⁻¹(1−1/N)`` con ``N = 1`` no está definido.
    """
    trials = int(n_trials)
    if trials < 2:
        return 0.0
    variance = float(sr_variance)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError(f"V[SR_n] debe ser finita y >= 0, llegó {sr_variance!r}")
    if variance == 0.0:
        return 0.0
    gamma = EULER_MASCHERONI
    z_uniform = phi_inv(1.0 - 1.0 / trials)
    z_euler = phi_inv(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(variance) * ((1.0 - gamma) * z_uniform + gamma * z_euler)


def psr(
    sr_hat: float,
    sr_benchmark: float,
    t: int,
    skew: float,
    kurtosis: float,
) -> float:
    """PSR(SR*): probabilidad de que el Sharpe verdadero supere el umbral.

    ``Φ( (SR̂ − SR*)·sqrt(T−1) / sqrt(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) )``.

    Si ``SR̂ = SR*`` el argumento es 0 y el resultado es 0.5, aunque los
    momentos hagan singular el denominador.
    """
    length = int(t)
    if length < 2:
        raise ValueError("PSR pide T >= 2")
    observed = float(sr_hat)
    benchmark = float(sr_benchmark)
    if observed == benchmark:
        return 0.5
    scale = _sharpe_scale(observed, skew, kurtosis)
    if scale <= 0.0:
        raise ValueError("PSR: el denominador no es positivo")
    z = (observed - benchmark) * math.sqrt(length - 1) / math.sqrt(scale)
    return phi(z)


def dsr(
    sr_hat: float,
    sr_variance: float,
    t: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
) -> float:
    """DSR: el PSR evaluado en ``SR₀`` en vez de en un umbral fijo.

    Crece el ``N`` y sube ``SR₀``, así que el DSR baja. Sube la curtosis
    (con la misma ``V[SR_n]``) y el denominador crece, así que el DSR baja.
    ``SR̂``, ``V`` y ``T`` tienen que estar en la misma frecuencia.
    """
    benchmark = expected_maximum_sr(sr_variance, n_trials)
    return psr(sr_hat, benchmark, t, skew, kurtosis)


def min_track_record(
    sr_hat: float,
    sr_benchmark: float,
    skew: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """Mínimo de observaciones para que el PSR llegue a ``confidence``.

    Bailey y López de Prado (2012):
    ``1 + (1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) · (Φ⁻¹(α) / (SR̂ − SR*))²``.
    """
    level = float(confidence)
    if not 0.5 < level < 1.0:
        raise ValueError("confidence debe estar en (0.5, 1)")
    observed = float(sr_hat)
    benchmark = float(sr_benchmark)
    gap = observed - benchmark
    if gap == 0.0:
        return math.inf
    scale = _sharpe_scale(observed, skew, kurtosis)
    if scale <= 0.0:
        raise ValueError("MinTRL: la escala del Sharpe no es positiva")
    z = phi_inv(level)
    return 1.0 + scale * (z / gap) ** 2


def annualize_sharpe(sr: float, periods: int) -> float:
    """Sharpe de display. No entra en el PSR ni en el DSR."""
    count = int(periods)
    if count < 1:
        raise ValueError("periods debe ser >= 1")
    return float(sr) * math.sqrt(count)


def pbo_cscv(returns: np.ndarray, n_blocks: int) -> float:
    """Probability of Backtest Overfitting por CSCV.

    ``returns`` tiene forma ``(T, N)``: una columna por configuración.
    ``n_blocks`` (``S``) es par y parte ``T`` en bloques contiguos del
    mismo largo (el resto final se descarta).

    Cada combinación de ``S/2`` bloques (``itertools`` no hace falta
    importarlo en el llamador: aquí se usa) es la muestra in-sample; el
    complemento es out-of-sample. Se elige la columna de mayor Sharpe
    in-sample y se anota su rango relativo out-of-sample
    ``ω = rango / (N+1)`` con rango 1 = peor. ``λ = ln(ω / (1−ω))``.
    El PBO es la fracción de combinaciones con ``λ <= 0``: la ganadora
    in-sample quedó por debajo de la mediana fuera de muestra.

    Una rejilla de ruido, con semilla fija, cae cerca de 0.5.
    """
    import itertools

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("PBO pide una matriz (T, N)")
    rows, cols = matrix.shape
    blocks = int(n_blocks)
    if blocks < 2 or blocks % 2 != 0:
        raise ValueError("S tiene que ser par y >= 2")
    if cols < 2:
        raise ValueError("PBO pide al menos dos configuraciones")
    if not np.isfinite(matrix).all():
        raise ValueError("PBO no acepta NaN ni infinitos")
    size = rows // blocks
    if size < 2:
        raise ValueError("cada bloque necesita al menos dos observaciones")
    usable = matrix[: size * blocks]
    parts = [usable[i * size : (i + 1) * size] for i in range(blocks)]
    half = blocks // 2
    logits = []
    for chosen in itertools.combinations(range(blocks), half):
        held = [i for i in range(blocks) if i not in chosen]
        in_sample = np.vstack([parts[i] for i in chosen])
        out_sample = np.vstack([parts[i] for i in held])
        in_sharpe = _column_sharpe(in_sample)
        out_sharpe = _column_sharpe(out_sample)
        best = int(np.argmax(in_sharpe))
        # rango 1 = peor. argsort doble: 0 es el menor Sharpe.
        order = np.argsort(np.argsort(out_sharpe))
        rank = int(order[best]) + 1
        omega = rank / (cols + 1)
        logits.append(math.log(omega / (1.0 - omega)))
    if not logits:
        raise ValueError("CSCV no produjo combinaciones")
    return float(np.mean(np.asarray(logits) <= 0.0))


def drop_best_days(pnl: np.ndarray, k: int = 5) -> np.ndarray:
    """P&L diario sin los ``k`` días de mayor P&L.

    Si hay menos de ``k`` días, no queda ninguno. El orden original de
    lo que queda se conserva.
    """
    values = _finite_vector(pnl, "pnl")
    drop = int(k)
    if drop < 0:
        raise ValueError("k no puede ser negativo")
    if drop == 0 or values.size == 0:
        return values.copy()
    if drop >= values.size:
        return values[:0]
    # Los k mayores, estable en empates: se van los primeros índices.
    order = np.argsort(-values, kind="mergesort")
    removed = set(int(i) for i in order[:drop])
    kept = [values[i] for i in range(values.size) if i not in removed]
    return np.asarray(kept, dtype=float)


def group_sum(keys: Sequence[object], pnl: np.ndarray) -> dict:
    """Suma el P&L por clave (año, régimen, lo que pase el llamador)."""
    values = _finite_vector(pnl, "pnl")
    if len(keys) != values.size:
        raise ValueError("keys y pnl tienen que medir lo mismo")
    totals: dict = {}
    for key, value in zip(keys, values):
        totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def read_trials(path: Path) -> list[dict]:
    """Filas del registro. Un archivo ausente es una lista vacía."""
    target = Path(path)
    if not target.is_file():
        return []
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_trials(path: Path, rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Agrega las filas de una corrida. El mismo ``run_id`` no se duplica.

    Todas las filas de la llamada comparten ``run_id``. Si ese id ya
    está en el archivo, el archivo no se toca. Devuelve el registro
    completo después de la operación.
    """
    pending = [_trial_row(row) for row in rows]
    if not pending:
        return read_trials(path)
    run_id = pending[0]["run_id"]
    if any(row["run_id"] != run_id for row in pending):
        raise ValueError("todas las filas de una corrida comparten run_id")
    target = Path(path)
    existing = read_trials(target)
    if any(row.get("run_id") == run_id for row in existing):
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.is_file() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        if write_header:
            writer.writeheader()
        for row in pending:
            writer.writerow(row)
    return read_trials(target)


def trial_count(rows: Sequence[Mapping[str, str]], strategy: str) -> int:
    """``N`` del DSR: pruebas registradas de esa estrategia."""
    name = str(strategy)
    return sum(1 for row in rows if row.get("strategy") == name)


def trial_sharpe_variance(rows: Sequence[Mapping[str, str]], strategy: str) -> Optional[float]:
    """Varianza muestral de los SR registrados de la estrategia.

    ``None`` si hay menos de dos números finitos: el llamador usa entonces
    la varianza analítica del estimador.
    """
    values = []
    name = str(strategy)
    for row in rows:
        if row.get("strategy") != name:
            continue
        text = row.get("sr", "")
        if text is None or text == "":
            continue
        number = float(text)
        if math.isfinite(number):
            values.append(number)
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    variance = float(array.var(ddof=1))
    if variance <= 0.0:
        return None
    return variance


def _sharpe_scale(sr_hat: float, skew: float, kurtosis: float) -> float:
    """``1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²``. El interior de la raíz del PSR."""
    observed = float(sr_hat)
    return 1.0 - float(skew) * observed + ((float(kurtosis) - 1.0) / 4.0) * observed ** 2


def _column_sharpe(block: np.ndarray) -> np.ndarray:
    out = np.empty(block.shape[1], dtype=float)
    for column in range(block.shape[1]):
        out[column] = sharpe_ratio(block[:, column])
    return out


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise ValueError(f"{name} tiene NaN o infinitos")
    return array


def _trial_row(row: Mapping[str, object]) -> dict:
    missing = [field for field in TRIAL_FIELDS if field not in row]
    if missing:
        raise ValueError(f"fila de prueba sin {missing}")
    out = {}
    for field in TRIAL_FIELDS:
        value = row[field]
        if isinstance(value, bool):
            out[field] = "true" if value else "false"
        else:
            out[field] = "" if value is None else str(value)
    if not out["run_id"] or not out["strategy"] or not out["config_hash"]:
        raise ValueError("run_id, strategy y config_hash son obligatorios")
    return out
