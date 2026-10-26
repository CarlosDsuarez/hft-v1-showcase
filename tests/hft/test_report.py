"""Reporte HTML y reconciliación backtest contra paper (HFT-13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta

import pandas as pd
import pytest

from src.hft.backtest import run
from src.hft.calendar import FakeCalendar, session_at
from src.hft.config import hft_config_from_dict, load_hft_config
from src.hft.costs import paper_valuation_fees
from src.hft.data.bars import bars_from_records
from src.hft.report import (
    CAUSE_LATE_BAR,
    CAUSE_MISSING,
    CAUSE_RISK,
    MIN_SLIPPAGE_SESSIONS,
    PaperFill,
    SessionSlippage,
    compare_executions,
    implementation_shortfall_bps,
    main,
    reconcile_session,
    render_html,
    slippage_recalibration_alert,
    synthetic_view,
    write_html,
)
from src.hft.types import RiskEvent, TheoryCard, TheoryParam, order_intent
from src.settings import load_settings

DAY = date(2025, 1, 6)
SESSION = session_at(DAY, time(9, 30), time(16, 0))
PRICE = 200.0
UTC_TS = SESSION.open_utc


@dataclass(frozen=True)
class _Qty:
    qty: float = 1.0


def _theory() -> TheoryCard:
    return TheoryCard(
        hypothesis="Compra una vez si el cierre de la tercera barra pasa un umbral fijo.",
        mechanism="Nadie paga: el test mira la paridad, no un resultado.",
        references=("HFT-13",),
        data_requirements=("barras 1Min ya finales",),
        params=(TheoryParam(
            "qty", 1.0, "Cantidad fija del test, no estimada.", "HFT-13",
        ),),
        holding_period="unas barras",
        failure_regimes=("no aplica",),
        kill_criteria=("no usarla fuera del test",),
        research_only=True,
    )


class _SegunCierre:
    """Emite una compra solo si el cierre de la tercera barra es >= 150."""

    id = "seguncierre"

    def __init__(self) -> None:
        self.params = _Qty(1.0)
        self.theory = _theory()

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        close = ctx.bars.close
        if "SPY" not in close.columns or len(close.index) != 3:
            return []
        last = float(close["SPY"].iloc[-1])
        if last < 150.0:
            return []
        return [order_intent(
            self.id, "SPY", "buy", 1.0, ctx.bar_ts,
            order_type="marketable_limit", limit_price=PRICE + 50.0,
            reason="cierre por encima del umbral",
        )]


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


def _bars(session, *, override=None):
    rows = []
    start = session.open_utc
    for i in range(8):
        px = PRICE + (0.05 if i % 2 else 0.0)
        if override and i in override:
            px = float(override[i])
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


def _replay(tmp_path, monkeypatch, bars):
    _home(tmp_path, monkeypatch)
    return run(
        _config(),
        DAY,
        DAY,
        provider=_Memoria(bars),
        strategies=[_SegunCierre()],
        calendar=FakeCalendar([SESSION]),
        seed=1,
        record_trial=False,
        persist=False,
    )


def test_el_html_sale_de_un_estado_sintetico_y_es_un_solo_archivo(tmp_path):
    view = synthetic_view(
        start=DAY,
        end=DAY,
        equity=(100_000.0, 100_040.0, 99_900.0),
        pnl_by_hour=pd.DataFrame([
            {"hour_et": 10, "pnl": 40.0, "n": 1},
            {"hour_et": 11, "pnl": -20.0, "n": 1},
        ]),
        metrics={"sharpe_daily": 0.4, "n_days": 1},
        by_strategy=pd.DataFrame([{
            "key": "seguncierre", "pnl": 12.0, "fees": 0.01, "n": 1, "slippage_bps": 3.0,
        }]),
        blotter=pd.DataFrame([{
            "strategy_id": "seguncierre", "symbol": "SPY", "side": "long", "pnl": 12.0,
        }]),
        risk_events=({
            "ts": "2025-01-06T15:00:00+00:00",
            "kind": "max_orders",
            "reason": "tope de prueba",
        },),
        quality_rows=({
            "symbol": "SPY", "kind": "sin_trades", "detail": "minuto sin trades en IEX",
        },),
        cost_rows=({
            "concepto": "fees_sinteticas_paper", "valor": 0.02,
        },),
    )
    target = tmp_path / "reporte.html"
    written = write_html(view, target)
    assert written == target
    assert written.is_file()
    text = written.read_text(encoding="utf-8")
    assert text == render_html(view)
    assert text.count("<svg") == 3
    assert "Curva de equity" in text
    assert "Drawdown" in text
    assert "P&amp;L por hora" in text
    assert "Métricas" in text
    assert "Por estrategia" in text
    assert "Blotter" in text
    assert "Eventos de riesgo" in text
    assert "Calidad de datos" in text
    assert "Desglose de costos" in text
    assert "seguncierre" in text
    assert "tope de prueba" in text
    assert "sin_trades" in text
    assert "fees_sinteticas_paper" in text
    assert "plotly" not in text.lower()
    assert "<script" not in text
    assert 'rel="stylesheet"' not in text
    assert list(tmp_path.glob("*.html")) == [target]


def test_shortfall_en_bps_y_fees_sinteticas_de_una_venta():
    assert implementation_shortfall_bps("buy", 100.10, 100.0) == pytest.approx(10.0)
    assert implementation_shortfall_bps("sell", 99.90, 100.0) == pytest.approx(10.0)
    assert implementation_shortfall_bps("sell", 100.10, 100.0) == pytest.approx(-10.0)
    config = _config()
    from src.hft.costs import CostModel

    model = CostModel.from_config(config)
    synthetic = paper_valuation_fees(
        model, side="sell", qty=10.0, price=100.0, liquidity="taker",
    )
    assert synthetic > 0.0


def test_intenciones_identicas_dan_paridad_total(tmp_path, monkeypatch):
    bars = _bars(SESSION)
    first = _replay(tmp_path, monkeypatch, bars)
    assert first.intents, "el umbral alto tiene que emitir una intención"
    result = reconcile_session(
        _config(),
        DAY,
        bars,
        first.intents,
        strategies=[_SegunCierre()],
        calendar=FakeCalendar([SESSION]),
        feed="iex",
        seed=1,
        paper_fills=(PaperFill(
            intent_id=first.intents[0].intent_id,
            side="sell",
            symbol="SPY",
            qty=10.0,
            fill_price=199.80,
            decision_price=200.0,
            fees_charged=0.0,
        ),),
    )
    assert result.parity == pytest.approx(1.0)
    assert result.diffs == ()
    assert result.n_paper == result.n_backtest == result.n_matched
    row = result.shortfalls[0]
    assert row.shortfall_bps == pytest.approx(10.0)
    assert row.paper_fees == 0.0
    assert row.synthetic_fees > 0.0
    assert result.alert is None


def test_una_barra_alterada_se_detecta_y_se_explica(tmp_path, monkeypatch):
    original = _bars(SESSION)
    first = _replay(tmp_path, monkeypatch, original)
    assert len(first.intents) == 1
    # La tercera barra (índice 2) es la de la señal. El histórico final
    # la republica por debajo del umbral: el backtest ya no entra.
    corrected = _bars(SESSION, override={2: 10.0})
    result = reconcile_session(
        _config(),
        DAY,
        original,
        first.intents,
        replay_bars=corrected,
        strategies=[_SegunCierre()],
        calendar=FakeCalendar([SESSION]),
        feed="iex",
        seed=1,
    )
    assert result.parity < 1.0
    assert result.diffs
    assert all(item.cause == CAUSE_LATE_BAR for item in result.diffs)
    assert "difiere" in result.diffs[0].detail or "corregida" in result.diffs[0].detail
    page = render_html(synthetic_view(
        start=DAY, end=DAY, equity=(100_000.0,), reconciliation=result,
    ))
    assert CAUSE_LATE_BAR in page
    assert "Paridad" in page


def test_rechazo_de_riesgo_y_dato_faltante_tienen_causa(tmp_path, monkeypatch):
    bars = _bars(SESSION)
    first = _replay(tmp_path, monkeypatch, bars)
    intent = first.intents[0]
    rejection = RiskEvent(
        ts=intent.signal_ts,
        kind="max_orders",
        reason="tope de órdenes de la prueba",
        symbol=intent.symbol,
        intent_id=intent.intent_id,
    )
    blocked = compare_executions(
        (),
        (intent,),
        paper_bars=bars,
        replay_bars=bars,
        paper_rejections=(rejection,),
        session_date=DAY,
    )
    assert blocked.diffs[0].cause == CAUSE_RISK
    assert "tope de órdenes" in blocked.diffs[0].detail

    empty = bars_from_records(
        pd.DataFrame(columns=["symbol", "ts", "o", "h", "l", "c", "v", "vw", "n"]),
        ["SPY"],
        source="test",
        feed="iex",
        timeframe="1Min",
    )
    missing = compare_executions(
        (intent,),
        (),
        paper_bars=bars,
        replay_bars=empty,
        session_date=DAY,
    )
    assert missing.diffs[0].cause == CAUSE_MISSING


def test_la_alerta_pide_veinte_sesiones_y_no_cambia_parametros():
    quiet = tuple(
        SessionSlippage(f"2025-01-{index:02d}", real_bps=1.0, modeled_bps=1.0)
        for index in range(1, MIN_SLIPPAGE_SESSIONS + 1)
    )
    assert slippage_recalibration_alert(quiet) is None
    assert slippage_recalibration_alert(quiet[:-1]) is None
    loud = tuple(
        SessionSlippage(f"s{index}", real_bps=8.0, modeled_bps=1.0)
        for index in range(MIN_SLIPPAGE_SESSIONS)
    )
    text = slippage_recalibration_alert(loud)
    assert text is not None
    assert "recalibrar" in text
    assert "HFT-14" in text


def test_el_cli_escribe_el_html_sin_red(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    target = tmp_path / "cli.html"
    code = main(["--date", "2025-01-06", "--output", str(target)])
    assert code == 0
    text = target.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "plotly" not in text.lower()
    assert main(["--start", "2025-01-06"]) == 2
