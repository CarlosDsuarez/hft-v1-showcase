"""Backtest intradía: una sesión, holdout y estrés (HFT-10)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, time, timedelta

import pandas as pd
import pytest

from src.hft.backtest import HoldoutLocked, _attribute, main, run, stress_costs, stress_latency
from src.hft.calendar import NY, FakeCalendar, session_at
from src.hft.config import hft_config_from_dict, load_hft_config
from src.hft.data.bars import bars_from_records
from src.hft.strategies.orb import OpeningRangeBreakout
from src.hft.types import TheoryCard, TheoryParam, order_intent
from src.hft.validation import read_trials
from src.settings import load_settings

DAY = date(2025, 1, 6)
HOLDOUT_DAY = date(2026, 1, 2)
SESSION = session_at(DAY, time(9, 30), time(16, 0))
HOLDOUT_SESSION = session_at(HOLDOUT_DAY, time(9, 30), time(16, 0))
PRICE = 200.0


@dataclass(frozen=True)
class _Qty:
    qty: float = 1.0


def _theory(qty: float) -> TheoryCard:
    return TheoryCard(
        hypothesis="Compra y vende una vez para probar el backtest, sin edge.",
        mechanism="Nadie paga: el test mira el blotter, no un resultado.",
        references=("HFT-10",),
        data_requirements=("barras 1Min ya finales",),
        params=(TheoryParam(
            "qty", qty, "Cantidad fija del test, no estimada.", "HFT-10",
        ),),
        holding_period="unas barras",
        failure_regimes=("no aplica",),
        kill_criteria=("no usarla fuera del test",),
        research_only=True,
    )


class _IdaYVuelta:
    """Compra en la segunda barra y cierra en la sexta. Límites holgados."""

    id = "idayvuelta"

    def __init__(self) -> None:
        self.params = _Qty(1.0)
        self.theory = _theory(1.0)

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        n = len(ctx.bars.close.index)
        if n == 2:
            return [order_intent(
                self.id, "SPY", "buy", 1.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE + 50.0,
                reason="entrada de prueba",
            )]
        if n == 6:
            return [order_intent(
                self.id, "SPY", "sell", 1.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE - 50.0,
                reason="salida de prueba",
            )]
        return []


class _Memoria:
    def __init__(self, bars) -> None:
        self.bars = bars

    def get_bars(self, symbols, start, end, timeframe=None):
        del symbols, timeframe
        index = self.bars.close.index
        mask = (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
        frames = {
            name: getattr(self.bars, name).loc[mask]
            for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
        }
        return type(self.bars)(meta=self.bars.meta, **frames)


def _bars(session, n: int = 10):
    rows = []
    start = session.open_utc
    for i in range(n):
        px = PRICE + (0.05 if i % 2 else 0.0)
        rows.append({
            "symbol": "SPY",
            "ts": start + timedelta(minutes=i),
            "o": px,
            "h": px + 0.05,
            "l": px - 0.05,
            "c": px,
            "v": 50_000,
            "vw": px,
            "n": 20,
        })
    return bars_from_records(
        pd.DataFrame(rows), ["SPY"], source="test", feed="iex", timeframe="1Min",
    )


def _config():
    raw = load_hft_config().to_dict()
    raw["sim"]["partial_prob"] = 0.0
    return hft_config_from_dict(raw)


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    return load_settings()


def _run(tmp_path, monkeypatch, **kwargs):
    _home(tmp_path, monkeypatch)
    defaults = dict(
        provider=_Memoria(_bars(SESSION)),
        strategies=[_IdaYVuelta()],
        calendar=FakeCalendar([SESSION]),
        seed=1,
    )
    defaults.update(kwargs)
    return run(_config(), DAY, DAY, **defaults)


def test_una_sesion_cierra_un_roundtrip_y_no_duplica_el_registro(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch)
    closed = report.blotter[report.blotter["closed"] == True]  # noqa: E712
    assert len(closed) == 1
    row = closed.iloc[0]
    assert row["symbol"] == "SPY"
    assert row["side"] == "long"
    assert row["qty"] == pytest.approx(1.0)
    assert row["fees"] >= 0.0
    assert report.metrics is not None
    assert report.trade_stats["n"] == 1
    assert (report.output_dir / "blotter.csv").is_file()
    assert (report.output_dir / "daily.csv").is_file()
    summary = json.loads((report.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["html"] == "report.html"
    page = (report.output_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in page
    assert "plotly" not in page.lower()
    assert summary["provenance"]["config_hash"] == report.config_hash
    assert summary["provenance"]["feed"] == "iex"

    path = load_settings().results_dir / "hft" / "trials.csv"
    first = read_trials(path)
    ids = {item["run_id"] for item in first}
    assert ids == {report.run_id}
    assert any(item["strategy"] == "book" for item in first)
    again = _run(tmp_path, monkeypatch)
    assert again.run_id == report.run_id
    assert read_trials(path) == first


class _Grande(_IdaYVuelta):
    """Pide 100 SPY a 250 ($25k): más que risk.max_position_notional ($10k)."""

    id = "grande"

    def on_bar(self, ctx):
        if len(ctx.bars.close.index) == 2:
            return [order_intent(
                self.id, "SPY", "buy", 100.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE + 50.0,
                reason="entrada de prueba",
            )]
        return []


def test_el_sizer_del_backtest_respeta_el_tope_por_simbolo(tmp_path, monkeypatch):
    # Antes el sizer no veía el tope y el riesgo rechazaba la orden entera.
    report = _run(
        tmp_path, monkeypatch, strategies=[_Grande()], record_trial=False, persist=False,
    )
    spy = report.blotter[report.blotter["symbol"] == "SPY"]
    assert len(spy) == 1
    cap = _config().risk.max_position_notional
    assert 0 < spy.iloc[0]["qty"] <= cap / (PRICE + 50.0)


class _CortoGrande(_IdaYVuelta):
    """Corto de 100 SPY con límite 150: la marca (200) vale más que el límite."""

    id = "cortogrande"

    def on_bar(self, ctx):
        if len(ctx.bars.close.index) == 2:
            return [order_intent(
                self.id, "SPY", "sell", 100.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE - 50.0,
                reason="corto de prueba",
            )]
        return []


def test_el_sizer_valora_el_tope_con_la_marca_del_riesgo(tmp_path, monkeypatch):
    # Con solo el límite (150) el sizer daba 66 acciones = $13.3k a la marca
    # y el riesgo rechazaba. La marca llega al sizer en EngineState.
    report = _run(
        tmp_path, monkeypatch, strategies=[_CortoGrande()], record_trial=False, persist=False,
    )
    spy = report.blotter[report.blotter["symbol"] == "SPY"]
    assert len(spy) == 1
    assert spy.iloc[0]["side"] == "short"
    cap = _config().risk.max_position_notional
    assert 0 < spy.iloc[0]["qty"] <= cap / PRICE


def test_costos_al_doble_cobran_mas_y_la_latencia_atrasa_el_fill(tmp_path, monkeypatch):
    base = _run(tmp_path, monkeypatch, record_trial=False, persist=False)
    dear = stress_costs(
        _config(), DAY, DAY,
        provider=_Memoria(_bars(SESSION)),
        strategies=[_IdaYVuelta()],
        calendar=FakeCalendar([SESSION]),
        seed=1,
    )
    late = stress_latency(
        _config(), DAY, DAY,
        provider=_Memoria(_bars(SESSION)),
        strategies=[_IdaYVuelta()],
        calendar=FakeCalendar([SESSION]),
        seed=1,
    )
    base_fee = float(base.blotter.loc[base.blotter["closed"] == True, "fees"].iloc[0])  # noqa: E712
    dear_fee = float(dear.blotter.loc[dear.blotter["closed"] == True, "fees"].iloc[0])  # noqa: E712
    assert dear_fee > base_fee
    assert late.blotter.loc[late.blotter["closed"] == True, "entry_ts"].iloc[0] > (  # noqa: E712
        base.blotter.loc[base.blotter["closed"] == True, "entry_ts"].iloc[0]  # noqa: E712
    )
    path = load_settings().results_dir / "hft" / "trials.csv"
    assert not path.exists()


def test_el_holdout_bloquea_salvo_desbloqueo_anotado(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    cfg = _config()
    kwargs = dict(
        provider=_Memoria(_bars(HOLDOUT_SESSION)),
        strategies=[_IdaYVuelta()],
        calendar=FakeCalendar([HOLDOUT_SESSION]),
        seed=1,
    )
    with pytest.raises(HoldoutLocked):
        run(cfg, HOLDOUT_DAY, HOLDOUT_DAY, **kwargs)
    path = load_settings().results_dir / "hft" / "trials.csv"
    assert not path.exists()

    opened = run(cfg, HOLDOUT_DAY, HOLDOUT_DAY, unlock_holdout=True, **kwargs)
    rows = read_trials(path)
    assert rows
    assert all(item["holdout_unlocked"] == "true" for item in rows)
    assert all(item["run_id"] == opened.run_id for item in rows)

    code = main(["--start", "2026-01-02", "--end", "2026-01-02"])
    assert code == 2


# HFT-PAPER-08. Backtest 2025-12-01..12 con barras IEX: el flatten de ORB a
# las 15:56 ET compró 40 IWM que cerraban el corto de vwap_reversion de las
# 10:17 ET. El blotter dejó las dos patas "abierta" y la atribución (solo
# cerrados) mostró vwap +$54 con −$293 de IWM escondidos. El NAV cuadraba.

SHORT_SESSION = session_at(DAY, time(9, 30), time(10, 30))


def _bars_multi(session, symbols=("SPY", "IWM")):
    rows = []
    start = session.open_utc
    n = int((session.close_utc - session.open_utc).total_seconds() // 60)
    for symbol in symbols:
        for i in range(n):
            # IWM sube: el corto pierde y la pérdida tiene que verse.
            px = PRICE + (0.02 * i if symbol == "IWM" else (0.05 if i % 2 else 0.0))
            rows.append({
                "symbol": symbol,
                "ts": start + timedelta(minutes=i),
                "o": px,
                "h": px + 0.05,
                "l": px - 0.05,
                "c": px,
                "v": 50_000,
                "vw": px,
                "n": 20,
            })
    return bars_from_records(
        pd.DataFrame(rows), list(symbols), source="test", feed="iex", timeframe="1Min",
    )


class _CortoIwm:
    """Como vwap_reversion: corto de 40 IWM y su propio flatten más tarde que ORB."""

    id = "vwap_stub"

    def __init__(self) -> None:
        self.params = _Qty(40.0)
        self.theory = _theory(40.0)

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        clock = ctx.bar_ts.astimezone(NY).time()
        if clock == time(9, 47):
            return [order_intent(
                self.id, "IWM", "sell", 40.0, ctx.bar_ts,
                order_type="marketable_limit", limit_price=PRICE - 50.0, reason="vwap entrada",
            )]
        qty = ctx.exit_qty("IWM")
        if clock >= time(10, 27) and abs(qty) > 1e-9:
            return [order_intent(
                self.id, "IWM", "sell" if qty > 0 else "buy", abs(qty), ctx.bar_ts,
                reason="vwap flatten_time",
            )]
        return []


def _orb():
    return OpeningRangeBreakout.from_config(_config(), (SHORT_SESSION,))


def _regresion(tmp_path, monkeypatch, strategies):
    return _run(
        tmp_path, monkeypatch,
        provider=_Memoria(_bars_multi(SHORT_SESSION)),
        calendar=FakeCalendar([SHORT_SESSION]),
        strategies=strategies,
        record_trial=False, persist=False,
    )


def test_el_flatten_de_orb_no_cierra_el_corto_de_otra_y_la_atribucion_cuadra(tmp_path, monkeypatch):
    report = _regresion(tmp_path, monkeypatch, [_orb(), _CortoIwm()])
    assert [intent for intent in report.intents if intent.strategy_id == "orb"] == []
    blotter = report.blotter
    assert list(blotter["exit_reason"]) == ["vwap flatten_time"]
    assert bool(blotter.iloc[0]["closed"]) and blotter.iloc[0]["strategy_id"] == "vwap_stub"
    assert blotter.iloc[0]["pnl"] < 0
    by_strategy = report.attribution_strategy.set_index("key")
    assert by_strategy.loc["vwap_stub", "pnl"] == pytest.approx(blotter["pnl"].sum())
    assert report.attribution_strategy["pnl"].sum() == pytest.approx(report.daily["pnl"].sum())


def test_agregar_el_corto_no_cambia_lo_que_emite_orb(tmp_path, monkeypatch):
    # Fase 6(d) con ORB real: el corto ajeno no le cambia ni un intent.
    def orb_of(report):
        return (
            [intent.intent_id for intent in report.intents if intent.strategy_id == "orb"],
            [item.intent_id for item in report.rejections if item.intent_id.startswith("orb-")],
        )

    solo = _regresion(tmp_path, monkeypatch, [_orb()])
    juntas = _regresion(tmp_path, monkeypatch, [_orb(), _CortoIwm()])
    assert orb_of(juntas) == orb_of(solo)


def test_la_atribucion_suma_lo_abierto_a_la_marca():
    # Una posición que quedó abierta no esconde su P&L: entra en pnl y en
    # open_pnl, y no cuenta como trade.
    blotter = pd.DataFrame([
        {"strategy_id": "a", "pnl": 10.0, "fees": 1.0, "slippage_bps": 2.0, "closed": True},
        {"strategy_id": "a", "pnl": -4.0, "fees": 0.5, "slippage_bps": 6.0, "closed": False},
        {"strategy_id": "b", "pnl": -3.0, "fees": 0.2, "slippage_bps": 1.0, "closed": False},
    ])
    table = _attribute(blotter, "strategy_id").set_index("key")
    assert table.loc["a", "pnl"] == pytest.approx(6.0)
    assert table.loc["a", "open_pnl"] == pytest.approx(-4.0)
    assert table.loc["a", "n"] == 1
    assert table.loc["a", "fees"] == pytest.approx(1.5)
    assert table.loc["a", "slippage_bps"] == pytest.approx(2.0)
    assert table.loc["b", "pnl"] == pytest.approx(-3.0)
    assert table.loc["b", "n"] == 0
    assert table["pnl"].sum() == pytest.approx(blotter["pnl"].sum())
