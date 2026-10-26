"""PSR, DSR, PBO y el registro de pruebas (HFT-10)."""

from __future__ import annotations

import numpy as np

from src.hft.validation import (
    append_trials,
    drop_best_days,
    dsr,
    min_track_record,
    pbo_cscv,
    psr,
    read_trials,
    trial_count,
)


def _row(run_id: str, strategy: str, sr: float) -> dict:
    return {
        "run_id": run_id,
        "config_hash": "abc",
        "strategy": strategy,
        "sr": sr,
        "n": 20,
        "n_trades": 3,
        "date": "2025-06-30",
        "feed": "iex",
        "start": "2025-01-02",
        "end": "2025-06-30",
        "holdout_unlocked": False,
        "cost_multiplier": 1.0,
        "extra_latency_bars": 0,
    }


def test_el_dsr_baja_cuando_crece_n_y_cuando_crece_la_curtosis():
    # V[SR_n] fijo: el único cambio es N, y con él SR₀.
    pocos = dsr(0.4, 0.01, 250, 0.0, 3.0, 2)
    muchos = dsr(0.4, 0.01, 250, 0.0, 3.0, 100)
    assert muchos < pocos
    # Misma V y mismo N: más curtosis agranda el denominador.
    normal = dsr(0.4, 0.01, 250, 0.0, 3.0, 10)
    colas = dsr(0.4, 0.01, 250, 0.0, 12.0, 10)
    assert colas < normal


def test_el_psr_es_un_medio_cuando_el_sharpe_iguala_el_umbral():
    # Momentos que harían singular el denominador si se evaluara.
    assert psr(0.2, 0.2, 80, 5.0, 40.0) == 0.5


def test_el_pbo_de_ruido_puro_queda_cerca_de_un_medio():
    rng = np.random.default_rng(23)
    # S = 8 par, T divisible por 8. Semilla fija: el PBO de este ruido es ~0.5.
    returns = rng.normal(0.0, 0.01, size=(160, 6))
    pbo = pbo_cscv(returns, 8)
    assert abs(pbo - 0.5) < 0.15


def test_el_registro_no_duplica_un_run_id(tmp_path):
    path = tmp_path / "trials.csv"
    first = append_trials(path, [_row("run-a", "book", 0.1), _row("run-a", "agenda", 0.2)])
    again = append_trials(path, [_row("run-a", "book", 9.0)])
    assert again == first
    assert read_trials(path) == first
    assert trial_count(first, "book") == 1
    second = append_trials(path, [_row("run-b", "book", 0.3)])
    assert trial_count(second, "book") == 2
    assert len(read_trials(path)) == 3


def test_mintrl_y_quitar_los_mejores_dias():
    length = min_track_record(0.2, 0.0, 0.0, 3.0, confidence=0.95)
    assert length > 1.0
    assert min_track_record(0.1, 0.1, 0.0, 3.0) == float("inf")
    pnl = np.asarray([1.0, 5.0, 2.0, 9.0, 3.0, 8.0])
    kept = drop_best_days(pnl, 2)
    assert list(kept) == [1.0, 5.0, 2.0, 3.0]
    assert kept.sum() < pnl.sum()
