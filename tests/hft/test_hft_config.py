"""Carga estricta de config/hft.toml (HFT-06)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.hft.calendar import DEFAULT_GRACE_S
from src.hft.config import (
    HFT_SCHEMA_VERSION,
    hft_config_from_dict,
    load_hft_config,
)
from src.settings import REPO_ROOT

UNIVERSE = REPO_ROOT / "universe_hft.csv"

# sha256 de la forma canónica de cada sección, antes de [hourly_momentum].
# Si uno cambia, cambió la config de una estrategia existente (o de su
# riesgo, sizing o costos): eso va en su propio PR, no en este.
SECTION_HASHES = {
    "costs": "f4d359186212b65929214a4a433a7f2523cb6a637493441e738277964bbdd45a",
    "sim": "c8d009c2a3460f7a9e1d8e83cb27e4e10112e35c4ea62584fb52c4622fadac57",
    "risk": "d88104ecb5f5b4078536d8784c4ee50c6bfaf42101eade556be0830b7b02e9e2",
    "sizing": "d87a3853412ae080a643f74bb833500b3c5afbcc6ddac1654f17902a85207c41",
    "orb": "3624d2fbfa3ce380a3ae6105bce73026e3b5564968a7cd68ee30855392ac7227",
    "intraday_momentum": "9b6ce50316f518de9ff8c039b9a2151c08e29248a5f1d0b0af59325bedb348b1",
    "vwap_reversion": "c7e1a1f956be6907149fc1e2c5a73aa8c30bbf8d3e85d8a00ce92ddaceb2eae7",
    "pairs_ou": "96abadee905c4dc786087de0a132036d14374fadc04072460537edd00e4f34c2",
    "avellaneda_stoikov": "afa55c9c63c5d9566cc10d7aedefc7db3a25526850c05e6f996312ec82c63a78",
    "decision": "3256626ab111943bec001047951ef3b234818b4c420d6bed6087a4b7b55c89c6",
}
# config_hash() de HftConfig antes de [hourly_momentum].
HASH_BEFORE_HOURLY = "871f853cc543d1d8e5c3c179daf39e59a4f2115c07e4e14afbe6b36a22f6add4"

HOURLY_KEYS = (
    "first_window_et",
    "last_window_et",
    "step_minutes",
    "lookback_minutes",
    "sigma_sessions",
    "min_abs_z",
    "top_k",
    "max_new_per_window",
    "hold_minutes",
    "stop_atr_frac",
    "atr_sessions",
    "risk_frac",
    "limit_through_ticks",
)


def _base() -> dict:
    cfg = load_hft_config().to_dict()
    return cfg


def _sha(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _with_hourly(raw: dict, **changes) -> dict:
    return {**raw, "hourly_momentum": {**raw["hourly_momentum"], **changes}}


def test_el_archivo_versionado_es_estricto_y_estable():
    cfg = load_hft_config()
    again = load_hft_config(REPO_ROOT / "config" / "hft.toml")
    assert cfg.schema_version == HFT_SCHEMA_VERSION == 1
    assert cfg.timeframe == "1Min"
    assert cfg.grace_s == DEFAULT_GRACE_S == 5
    assert cfg.latency_bars == 1
    assert cfg.feed == "iex"
    assert cfg.holdout_start == "2026-01-01"
    assert cfg.initial_cash == 100_000
    assert cfg.costs.commission_per_share == 0
    assert cfg.costs.sec_section31_rate == pytest.approx(0.0000206)
    assert cfg.costs.sec_section31_verified_on == "2026-04-04"
    assert cfg.costs.finra_taf_per_share == pytest.approx(0.000195)
    assert cfg.costs.finra_taf_cap == pytest.approx(9.79)
    assert cfg.costs.crypto_maker_bps == 15
    assert cfg.costs.crypto_taker_bps == 25
    assert cfg.costs.impact_y == 1
    assert cfg.sim.fill_mode == "cross"
    assert cfg.sim.participation_cap == pytest.approx(0.10)
    assert cfg.sim.partial_prob == pytest.approx(0.10)
    assert cfg.sim.latency_bars == 0
    assert cfg.risk.max_position_notional == 10_000
    assert cfg.risk.max_gross_exposure == 1
    assert cfg.risk.max_net_exposure == 1
    assert cfg.risk.max_daily_loss_usd == 500
    assert cfg.risk.max_daily_loss_pct == pytest.approx(0.02)
    assert cfg.risk.max_consecutive_losses == 3
    assert cfg.risk.cooldown_minutes == 15
    assert cfg.risk.entry_cutoff_minutes == 15
    assert cfg.risk.flatten_minutes == 5
    assert cfg.risk.entry_cutoff_minutes >= cfg.risk.flatten_minutes
    assert cfg.sizing.risk_capital_usd == 25_000
    assert cfg.sizing.risk_frac == pytest.approx(0.01)
    assert cfg.sizing.kelly_c == pytest.approx(0.25)
    assert cfg.sizing.max_leverage == 1
    assert cfg.config_hash() == again.config_hash()
    assert hft_config_from_dict(cfg.to_dict()).config_hash() == cfg.config_hash()
    assert len(cfg.config_hash()) == 64
    with UNIVERSE.open(newline="", encoding="utf-8") as handle:
        tickers = [row["ticker"] for row in csv.DictReader(handle)]
    assert list(cfg.universe) == tickers


def test_el_hash_no_depende_del_orden_y_cambia_con_el_valor():
    raw = _base()
    flipped = {key: raw[key] for key in reversed(list(raw))}
    assert hft_config_from_dict(flipped).config_hash() == hft_config_from_dict(raw).config_hash()
    other = dict(raw)
    other["grace_s"] = 6
    assert hft_config_from_dict(other).config_hash() != hft_config_from_dict(raw).config_hash()
    alias = dict(raw)
    alias["timeframe"] = "1T"
    alias["feed"] = "IEX"
    assert hft_config_from_dict(alias).config_hash() == hft_config_from_dict(raw).config_hash()


def test_falta_clave_o_sobra_clave():
    raw = _base()
    missing = dict(raw)
    del missing["latency_bars"]
    with pytest.raises(ValueError, match="faltan"):
        hft_config_from_dict(missing)
    with pytest.raises(ValueError, match="desconocidas"):
        hft_config_from_dict({**raw, "risk_capital_usd": 1})


def test_tipos_y_schema_invalidos():
    raw = _base()
    with pytest.raises(ValueError, match="schema_version"):
        hft_config_from_dict({**raw, "schema_version": 2})
    with pytest.raises(ValueError, match="latency_bars"):
        hft_config_from_dict({**raw, "latency_bars": 1.5})
    with pytest.raises(ValueError, match="latency_bars"):
        hft_config_from_dict({**raw, "latency_bars": True})
    with pytest.raises(ValueError, match="grace_s"):
        hft_config_from_dict({**raw, "grace_s": "5"})
    with pytest.raises(ValueError, match="t\\+1"):
        hft_config_from_dict({**raw, "latency_bars": 0})


def test_risk_y_sizing_son_tablas_estrictas():
    raw = _base()
    broken = dict(raw)
    risk = dict(broken["risk"])
    del risk["max_daily_loss_usd"]
    broken["risk"] = risk
    with pytest.raises(ValueError, match="max_daily_loss_usd"):
        hft_config_from_dict(broken)
    with pytest.raises(ValueError, match="desconocidas"):
        hft_config_from_dict({**raw, "risk": {**raw["risk"], "daytrade_count": 1}})
    sizing = dict(raw["sizing"])
    sizing["kelly_c"] = 0.5
    with pytest.raises(ValueError, match="kelly_c"):
        hft_config_from_dict({**raw, "sizing": sizing})
    cutoff = dict(raw["risk"])
    cutoff["entry_cutoff_minutes"] = 1
    cutoff["flatten_minutes"] = 5
    with pytest.raises(ValueError, match="entry_cutoff"):
        hft_config_from_dict({**raw, "risk": cutoff})


def test_toml_con_basura_menciona_la_ruta(tmp_path: Path):
    path = tmp_path / "hft.toml"
    path.write_text('schema_version = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        load_hft_config(path)




def test_hourly_momentum_valores_del_archivo():
    cfg = load_hft_config()
    hourly = cfg.hourly_momentum
    assert hourly.first_window_et == "10:00"
    assert hourly.last_window_et == "14:00"
    assert hourly.step_minutes == 60
    assert hourly.lookback_minutes == 60
    assert hourly.sigma_sessions == 20
    assert hourly.min_abs_z == pytest.approx(1.0)
    assert hourly.top_k == 3
    assert hourly.max_new_per_window == 2
    assert hourly.hold_minutes == 60
    assert hourly.stop_atr_frac == pytest.approx(0.25)
    assert hourly.atr_sessions == 14
    assert hourly.risk_frac == pytest.approx(cfg.sizing.risk_frac) == pytest.approx(0.01)
    assert hourly.limit_through_ticks == 1
    # Sin clave de símbolos: la estrategia opera solo [universe].
    assert tuple(cfg.to_dict()["hourly_momentum"]) == HOURLY_KEYS


def test_hourly_momentum_es_tabla_estricta():
    raw = _base()
    without = {key: value for key, value in raw.items() if key != "hourly_momentum"}
    with pytest.raises(ValueError, match="faltan \\['hourly_momentum'\\]"):
        hft_config_from_dict(without)
    missing = dict(raw["hourly_momentum"])
    del missing["hold_minutes"]
    with pytest.raises(ValueError, match="hourly_momentum: faltan \\['hold_minutes'\\]"):
        hft_config_from_dict({**raw, "hourly_momentum": missing})
    with pytest.raises(ValueError, match="desconocidas \\['symbol'\\]"):
        hft_config_from_dict(_with_hourly(raw, symbol="SPY"))
    with pytest.raises(ValueError, match="hourly_momentum.top_k"):
        hft_config_from_dict(_with_hourly(raw, top_k=True))
    with pytest.raises(ValueError, match="hourly_momentum.step_minutes"):
        hft_config_from_dict(_with_hourly(raw, step_minutes=60.0))
    with pytest.raises(ValueError, match="hourly_momentum.first_window_et"):
        hft_config_from_dict(_with_hourly(raw, first_window_et=10))
    with pytest.raises(ValueError, match="hourly_momentum.min_abs_z"):
        hft_config_from_dict(_with_hourly(raw, min_abs_z="1.0"))


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"first_window_et": "10:00:00"}, "hourly_momentum.first_window_et"),
        ({"first_window_et": "25:00"}, "hourly_momentum.first_window_et"),
        ({"last_window_et": "1400"}, "hourly_momentum.last_window_et"),
        ({"first_window_et": "09:30"}, "sesión regular"),
        ({"last_window_et": "16:00"}, "sesión regular"),
        ({"first_window_et": "14:00", "last_window_et": "10:00"}, "first_window_et"),
        ({"step_minutes": 45}, "múltiplo de step_minutes"),
        ({"step_minutes": 0}, "hourly_momentum.step_minutes"),
        ({"lookback_minutes": 0}, "hourly_momentum.lookback_minutes"),
        ({"sigma_sessions": 1}, "hourly_momentum.sigma_sessions"),
        ({"atr_sessions": 0}, "hourly_momentum.atr_sessions"),
        ({"top_k": 0}, "hourly_momentum.top_k"),
        ({"max_new_per_window": 0}, "hourly_momentum.max_new_per_window"),
        ({"max_new_per_window": 4}, "max_new_per_window debe ser <= top_k"),
        ({"hold_minutes": 0}, "hourly_momentum.hold_minutes"),
        ({"min_abs_z": 0.0}, "hourly_momentum.min_abs_z"),
        ({"min_abs_z": float("inf")}, "hourly_momentum.min_abs_z"),
        ({"stop_atr_frac": 0.0}, "hourly_momentum.stop_atr_frac"),
        ({"stop_atr_frac": 1.5}, "hourly_momentum.stop_atr_frac"),
        ({"risk_frac": 0.0}, "hourly_momentum.risk_frac"),
        ({"limit_through_ticks": -1}, "hourly_momentum.limit_through_ticks"),
    ],
)
def test_hourly_momentum_rechaza_valores_invalidos(changes, match):
    with pytest.raises(ValueError, match=match):
        hft_config_from_dict(_with_hourly(_base(), **changes))


def test_hourly_momentum_coincide_con_risk_y_sizing():
    raw = _base()
    with pytest.raises(ValueError, match="hourly_momentum.risk_frac debe coincidir con sizing.risk_frac"):
        hft_config_from_dict(_with_hourly(raw, risk_frac=0.02))
    # Corte de entradas a 15 min del cierre regular: 15:45 ET.
    with pytest.raises(ValueError, match="risk.entry_cutoff_minutes"):
        hft_config_from_dict(
            _with_hourly(raw, first_window_et="10:45", last_window_et="15:45")
        )
    late = hft_config_from_dict(
        _with_hourly(raw, first_window_et="10:44", last_window_et="15:44")
    )
    assert late.hourly_momentum.last_window_et == "15:44"
