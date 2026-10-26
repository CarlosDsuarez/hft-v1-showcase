"""Cada regla de riesgo intradía, aislada (HFT-08)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.brokers.pre_trade import PreTradeGuard, kill_switch_reason
from src.hft.book import HftBook
from src.hft.calendar import FakeCalendar, bar_close_time, session_at
from src.hft.config import RiskConfig
from src.hft.data.bars import bars_from_records
from src.hft.engine import Engine
from src.hft.risk import (
    FLATTEN_ALL,
    RETIRED_PDT_FIELDS,
    AssetFlag,
    EngineRiskGate,
    RiskManager,
    RiskState,
    activate_kill,
    arm_kill,
    limits_from_account,
    refresh_kill,
)
from src.hft.types import Bracket, Fill, TheoryCard, TheoryParam, order_intent

UTC = timezone.utc
DAY = date(2024, 7, 2)
HALF_DAY = date(2024, 7, 3)
FULL = session_at(DAY, time(9, 30), time(16, 0))
HALF = session_at(HALF_DAY, time(9, 30), time(13, 0))
NOW = FULL.open_utc + timedelta(minutes=30)


def _risk(**overrides) -> RiskConfig:
    base = dict(
        max_position_notional=1_000_000.0,
        max_gross_exposure=10.0,
        max_net_exposure=10.0,
        max_open_orders=100,
        max_orders_per_minute=100,
        max_orders_per_day=1_000,
        max_daily_loss_usd=1_000_000.0,
        max_daily_loss_pct=1.0,
        max_consecutive_losses=100,
        cooldown_minutes=15,
        entry_cutoff_minutes=30.0,
        flatten_minutes=5.0,
        stale_data_s=10_000.0,
        spread_guard_bps=10_000.0,
    )
    base.update(overrides)
    return RiskConfig(**base)


def _state(now: datetime = NOW, **overrides) -> RiskState:
    base = dict(
        equity=100_000.0,
        nav=100_000.0,
        buying_power=100_000.0,
        session_start_pnl=0.0,
        last_bar_close=now,
        marks={"SPY": 100.0, "QQQ": 100.0},
        minute_bucket=now.replace(second=0, microsecond=0),
        session_date=now.astimezone(ZoneInfo("America/New_York")).date(),
    )
    base.update(overrides)
    return RiskState(**base)


def _mgr(sessions=(FULL,), **overrides) -> RiskManager:
    return RiskManager(_risk(**overrides), sessions)


def _buy(qty: float = 10, symbol: str = "SPY", ts: datetime = NOW, **kw):
    return order_intent("s", symbol, "buy", qty, ts, **kw)


def _sell(qty: float = 10, symbol: str = "SPY", ts: datetime = NOW, **kw):
    return order_intent("s", symbol, "sell", qty, ts, **kw)


def _at(session, hour: int, minute: int) -> datetime:
    return datetime.combine(
        session.date, time(hour, minute), tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)


def test_los_defaults_del_runner_no_encienden_el_intradia():
    plan = pd.DataFrame(
        [{"ticker": "T0", "side": "buy", "notional": 100.0}],
        columns=["ticker", "side", "notional"],
    )
    summary = {
        "equity": 10_000, "account_blocked": False, "trading_blocked": False,
        "market_open": True, "orders_this_minute": 10_000, "orders_today": 10_000,
        "open_orders": 10_000,
    }
    assert PreTradeGuard().evaluate(plan, summary) == []


def test_el_guard_de_tasa_suma_lo_ya_enviado():
    plan = pd.DataFrame(
        [{"ticker": "T0", "side": "buy", "notional": 100.0}],
        columns=["ticker", "side", "notional"],
    )
    summary = {
        "equity": 10_000, "account_blocked": False, "trading_blocked": False,
        "market_open": True, "orders_this_minute": 2,
    }
    reasons = PreTradeGuard(max_orders_per_minute=2).evaluate(plan, summary)
    assert len(reasons) == 1 and "max_orders_per_minute" in reasons[0]


def test_buying_power_ignora_los_campos_pdt():
    assert "daytrade_count" in RETIRED_PDT_FIELDS
    assert "pattern_day_trader" in RETIRED_PDT_FIELDS

    class _Cuenta(dict):
        def __getitem__(self, key):
            if key in RETIRED_PDT_FIELDS:
                raise AssertionError(key)
            return dict.__getitem__(self, key)

        def __contains__(self, key):
            if key in RETIRED_PDT_FIELDS:
                raise AssertionError(key)
            return dict.__contains__(self, key)

        def get(self, key, default=None):
            if key in RETIRED_PDT_FIELDS:
                raise AssertionError(key)
            return dict.get(self, key, default)

    account = _Cuenta(equity="100000", buying_power="4000")
    account["daytrade_count"] = 3
    account["pattern_day_trader"] = True
    assert limits_from_account(account) == (100_000.0, 4_000.0)


def test_una_entrada_dentro_de_limites_pasa():
    result = _mgr().filter([_buy()], _state(), NOW)
    approved, rejected = result
    assert rejected == ()
    assert [item.qty for item in approved] == [10]
    assert result.flatten == ()
    assert result.kill is False
    assert result.state.orders_this_minute == 1
    assert result.state.buying_power == pytest.approx(99_000.0)


def test_nocional_por_simbolo():
    mgr = _mgr(max_position_notional=10_000)
    state = _state(positions={"SPY": 80})
    assert mgr.filter([_buy(10)], state, NOW).rejected == ()
    rejected = mgr.filter([_buy(30)], state, NOW).rejected
    assert len(rejected) == 1 and "max_position_notional" in rejected[0][1]


def test_exposicion_bruta_y_neta():
    gross = _mgr(max_gross_exposure=1.0).filter(
        [_buy(300)], _state(positions={"SPY": 800}), NOW,
    )
    assert "max_gross_exposure" in gross.rejected[0][1]
    net = _mgr(max_gross_exposure=10, max_net_exposure=0.5).filter(
        [_buy(200, symbol="QQQ")], _state(positions={"SPY": 400}), NOW,
    )
    assert "max_net_exposure" in net.rejected[0][1]


def test_ordenes_abiertas_dejan_salir():
    mgr = _mgr(max_open_orders=2)
    state = _state(open_orders=2, positions={"SPY": 10})
    blocked = mgr.filter([_buy()], state, NOW)
    assert "max_open_orders" in blocked.rejected[0][1]
    allowed = mgr.filter([_sell(4)], state, NOW)
    assert allowed.rejected == ()
    assert allowed.approved[0].side == "sell"


def test_tasa_por_minuto_y_por_dia():
    minute = _mgr(max_orders_per_minute=2)
    blocked = minute.filter([_buy()], _state(orders_this_minute=2), NOW)
    assert "max_orders_per_minute" in blocked.rejected[0][1]
    later = NOW + timedelta(minutes=1)
    opened = minute.filter([_buy()], _state(orders_this_minute=2), later)
    assert opened.rejected == ()
    assert opened.state.orders_this_minute == 1

    day = _mgr(max_orders_per_day=1)
    full = day.filter([_buy()], _state(orders_today=1), NOW)
    assert "max_orders_per_day" in full.rejected[0][1]
    nxt = session_at(HALF_DAY, time(9, 30), time(16, 0))
    when = nxt.open_utc + timedelta(minutes=30)
    reset = RiskManager(_risk(max_orders_per_day=1), [FULL, nxt]).filter(
        [_buy(ts=when)], _state(now=when, orders_today=50, session_date=DAY), when,
    )
    assert reset.rejected == ()
    assert reset.state.orders_today == 1
    assert reset.state.session_date == HALF_DAY


def test_buying_power_se_gasta_dentro_del_lote():
    mgr = _mgr()
    first, second = _buy(4, seq=0), _buy(7, seq=1)
    result = mgr.filter([first, second], _state(buying_power=1_000), NOW)
    assert [item.intent_id for item in result.approved] == [first.intent_id]
    assert "buying_power" in result.rejected[0][1]
    assert result.state.buying_power == pytest.approx(600.0)


def test_perdida_diaria_aplana_y_bloquea_entradas(tmp_path):
    mgr = _mgr(max_daily_loss_usd=500, max_daily_loss_pct=0.02)
    state = _state(positions={"SPY": 10}, realized_pnl=-400, unrealized_pnl=-200)
    result = mgr.filter([_sell(4, seq=0), _buy(1, seq=1)], state, NOW)
    assert [item.side for item in result.approved] == ["sell"]
    assert result.approved[0].qty == 4
    assert "max_daily_loss" in result.rejected[0][1]
    assert len(result.flatten) == 1
    assert result.flatten[0].side == "sell"
    assert result.flatten[0].qty == 6
    assert result.flatten[0].reason == FLATTEN_ALL
    assert result.kill is True
    assert arm_kill(result, tmp_path, live=False) is None
    assert not (tmp_path / "KILL").exists()
    path = arm_kill(result, tmp_path, live=True)
    assert path is not None and path.exists()
    assert "KILL" in kill_switch_reason(tmp_path, env={})
    again = mgr.filter([_buy()], result.state, NOW)
    assert again.flatten[0].intent_id == result.flatten[0].intent_id


def test_la_perdida_en_porcentaje_tambien_rompe():
    mgr = _mgr(max_daily_loss_usd=1_000_000, max_daily_loss_pct=0.01)
    state = _state(equity=10_000, positions={"SPY": 8}, realized_pnl=-200)
    result = mgr.filter([_buy()], state, NOW)
    assert result.kill is True
    assert result.flatten[0].qty == 8
    assert "max_daily_loss" in result.rejected[0][1]


def test_por_debajo_de_los_dos_topes_no_hay_kill():
    mgr = _mgr(max_daily_loss_usd=500, max_daily_loss_pct=0.02)
    state = _state(realized_pnl=-400, unrealized_pnl=0)
    result = mgr.filter([_buy()], state, NOW)
    assert result.rejected == ()
    assert result.kill is False
    assert result.flatten == ()


def test_el_corte_se_mueve_con_la_media_sesion():
    mgr = RiskManager(_risk(entry_cutoff_minutes=30, flatten_minutes=5), [FULL, HALF])
    half_cut = _at(HALF, 12, 40)
    full_cut = _at(FULL, 12, 40)
    blocked = mgr.filter([_buy(ts=half_cut)], _state(now=half_cut), half_cut)
    assert "entry_cutoff" in blocked.rejected[0][1]
    assert blocked.flatten == ()
    opened = mgr.filter([_buy(ts=full_cut)], _state(now=full_cut), full_cut)
    assert opened.rejected == ()

    half_flat = _at(HALF, 12, 56)
    full_flat = _at(FULL, 12, 56)
    flat = mgr.filter(
        [_buy(ts=half_flat)], _state(now=half_flat, positions={"SPY": 3}), half_flat,
    )
    assert flat.flatten[0].qty == 3
    assert flat.flatten[0].side == "sell"
    assert "flatten_time" in flat.rejected[0][1]
    assert flat.kill is False
    still = mgr.filter(
        [_buy(ts=full_flat)], _state(now=full_flat, positions={"SPY": 3}), full_flat,
    )
    assert still.rejected == ()
    assert still.flatten == ()


def test_cooldown_tras_rachas_y_su_vencimiento():
    mgr = _mgr(max_consecutive_losses=3, cooldown_minutes=15)
    state = _state(consecutive_losses=3, positions={"SPY": 5})
    first = mgr.filter([_buy(seq=0), _sell(2, seq=1)], state, NOW)
    assert "cooldown" in first.rejected[0][1]
    assert first.approved[0].side == "sell"
    assert first.state.cooldown_until == NOW + timedelta(minutes=15)
    second = mgr.filter([_buy()], first.state, NOW + timedelta(minutes=5))
    assert "cooldown" in second.rejected[0][1]
    later = NOW + timedelta(minutes=15)
    third = mgr.filter([_buy(ts=later)], first.state, later)
    assert third.rejected == ()
    assert third.state.consecutive_losses == 0
    assert third.state.cooldown_until is None


def test_dato_viejo_y_spread():
    stale = _mgr(stale_data_s=90)
    fresh = NOW - timedelta(seconds=90)
    assert stale.filter([_buy()], _state(last_bar_close=fresh), NOW).rejected == ()
    old = NOW - timedelta(seconds=91)
    blocked = stale.filter([_buy()], _state(last_bar_close=old), NOW)
    assert "stale_data" in blocked.rejected[0][1]
    missing = stale.filter([_buy()], _state(last_bar_close=None), NOW)
    assert "stale_data" in missing.rejected[0][1]

    wide = _mgr(spread_guard_bps=15)
    ok = wide.filter([_buy()], _state(spread_bps={"SPY": 15}), NOW)
    assert ok.rejected == ()
    bad = wide.filter([_buy()], _state(spread_bps={"SPY": 15.1}), NOW)
    assert "spread_guard" in bad.rejected[0][1]


def test_cortos_solo_si_la_ficha_lo_permite_y_son_enteros():
    mgr = _mgr()
    naked = mgr.filter([_sell(1)], _state(), NOW)
    assert "short_no_habilitado" in naked.rejected[0][1]
    flag = AssetFlag(shortable=True, easy_to_borrow=False)
    hard = mgr.filter([_sell(1)], _state(assets={"SPY": flag}), NOW)
    assert "short_no_habilitado" in hard.rejected[0][1]
    easy = AssetFlag(shortable=True, easy_to_borrow=True)
    frac = mgr.filter([_sell(1.5)], _state(assets={"SPY": easy}), NOW)
    assert "corto_fraccional" in frac.rejected[0][1]
    whole = mgr.filter([_sell(2)], _state(assets={"SPY": easy}), NOW)
    assert whole.rejected == ()
    cover = mgr.filter([_sell(1.5)], _state(positions={"SPY": 2}), NOW)
    assert cover.rejected == ()
    bracket = mgr.filter(
        [_buy(1.5, bracket=Bracket(stop=99.0))], _state(), NOW,
    )
    assert "bracket_fraccional" in bracket.rejected[0][1]


def test_el_guard_puede_vetar_un_lote_que_la_regla_suelta_dejo_pasar():
    guard = PreTradeGuard(max_orders_per_minute=1)
    mgr = RiskManager(_risk(max_orders_per_minute=10), [FULL], guard=guard)
    result = mgr.filter([_buy(seq=0), _buy(symbol="QQQ", seq=1)], _state(), NOW)
    assert result.approved == ()
    assert all(reason.startswith("pre_trade:") for _, reason in result.rejected)


def test_kill_ya_activo_aplana_y_no_deja_salir(tmp_path):
    state = refresh_kill(_state(positions={"SPY": 4}), tmp_path, env={})
    assert state.kill_reason is None
    activate_kill(tmp_path, "operador")
    state = refresh_kill(state, tmp_path, env={})
    result = _mgr().filter([_sell(4), _buy()], state, NOW)
    assert result.approved == ()
    assert all("kill_switch" in reason for _, reason in result.rejected)
    assert result.flatten[0].qty == 4
    assert result.kill is True


def test_antes_de_la_apertura_no_hay_flatten():
    early = FULL.open_utc - timedelta(minutes=30)
    result = _mgr().filter([_buy(ts=early)], _state(now=early, positions={"SPY": 5}), early)
    assert result.rejected[0][1] == "fuera_de_sesion"
    assert result.flatten == ()
    assert result.kill is False


@dataclass(frozen=True)
class _Qty:
    qty: float = 1.0


def _theory() -> TheoryCard:
    return TheoryCard(
        hypothesis="Compra de prueba del journal, sin hipótesis de mercado.",
        mechanism="Nadie paga: el test mira el rechazo, no un edge.",
        references=("HFT-08",),
        data_requirements=("una barra final",),
        params=(TheoryParam("qty", 1.0, "Cantidad fija del test.", "HFT-08"),),
        holding_period="una barra",
        failure_regimes=("no aplica",),
        kill_criteria=("no usarla fuera del test",),
        research_only=True,
    )


class _Compra:
    id = "una"
    params = _Qty(1.0)
    theory = _theory()

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        if len(ctx.bars.close.index) != 1:
            return []
        return [order_intent(self.id, "SPY", "buy", 1.0, ctx.bar_ts, reason="primera")]


class _Identidad:
    def size(self, intents, state, now):
        return list(intents)


class _Nada:
    def submit(self, intents, *, now, bars):
        return []


class _Reloj:
    def now(self):
        raise AssertionError("el motor leyó el reloj")


def test_el_rechazo_llega_al_journal_del_motor():
    start = FULL.open_utc
    bars = bars_from_records(
        pd.DataFrame([{
            "symbol": "SPY", "ts": start, "o": 100, "h": 101, "l": 99, "c": 100.5,
            "v": 1_000, "vw": 100.2, "n": 10,
        }]),
        ["SPY"], source="test", feed="iex", timeframe="1Min",
    )
    gate = EngineRiskGate(
        _mgr(),
        _state(kill_reason="KILL switch activo: test"),
    )
    engine = Engine(
        [_Compra()],
        lambda window: None,
        gate,
        _Identidad(),
        _Nada(),
        FakeCalendar([FULL]),
        _Reloj(),
        book=HftBook(100_000),
        timeframe="1Min",
        grace_s=5,
    )
    ready = bar_close_time(start, "1Min") + timedelta(seconds=5)
    engine.step(ready, bars)
    assert engine.book.position("SPY") == 0
    reasons = [event.reason for event in engine.journal[-1].rejections]
    assert len(reasons) == 1 and reasons[0].startswith("kill_switch:")


class _AgendaDeEntradas:
    """Compra en la barra 1, vende en la 3 y vuelve a comprar en la 5."""

    id = "agenda"
    params = _Qty(1.0)
    theory = _theory()

    def required_history(self):
        return (1, 0)

    def on_bar(self, ctx):
        side = {1: "buy", 3: "sell", 5: "buy"}.get(len(ctx.bars.close.index))
        if side is None:
            return []
        return [order_intent(self.id, "SPY", side, 1.0, ctx.bar_ts, reason="agenda")]


class _LlenaAlCierre:
    def submit(self, intents, *, now, bars):
        price = float(bars.close["SPY"].iloc[-1])
        return [Fill(intent.intent_id, now, intent.qty, price, 0.0, "taker") for intent in intents]

    def open_orders(self):
        return ()


class _NuncaLlena:
    def __init__(self):
        self._open = {}

    def submit(self, intents, *, now, bars):
        for intent in intents:
            self._open[intent.intent_id] = intent
        return []

    def open_orders(self):
        return tuple(self._open.values())


def _agenda_engine(execution):
    start = FULL.open_utc
    bars = bars_from_records(
        pd.DataFrame([
            {
                "symbol": "SPY", "ts": start + timedelta(minutes=i), "o": 100, "h": 101,
                "l": 99, "c": 100, "v": 1_000, "vw": 100, "n": 10,
            }
            for i in range(7)
        ]),
        ["SPY"], source="test", feed="iex", timeframe="1Min",
    )
    engine = Engine(
        [_AgendaDeEntradas()],
        lambda window: None,
        EngineRiskGate(_mgr(max_open_orders=1), _state()),
        _Identidad(),
        execution,
        FakeCalendar([FULL]),
        _Reloj(),
        book=HftBook(100_000),
        timeframe="1Min",
        grace_s=5,
    )
    for i, ts in enumerate(bars.close.index):
        ready = bar_close_time(ts.to_pydatetime(), "1Min") + timedelta(seconds=5)
        engine.step(ready, _first(bars, i + 1))
    return engine


def _first(bars, n):
    frames = {
        name: getattr(bars, name).iloc[:n]
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades")
    }
    return type(bars)(meta=bars.meta, **frames)


def test_open_orders_baja_cuando_la_orden_ya_se_lleno():
    # HFT-PAPER-10: open_orders solo subía. Con max_open_orders=1 la
    # segunda entrada del día se rechazaba aunque la primera ya estuviera llena.
    engine = _agenda_engine(_LlenaAlCierre())
    reasons = [item.reason for event in engine.journal for item in event.rejections]
    assert reasons == []
    assert engine.book.position("SPY") == 1


def test_una_orden_que_sigue_viva_cuenta_para_max_open_orders():
    engine = _agenda_engine(_NuncaLlena())
    capped = [
        item.intent_id
        for event in engine.journal
        for item in event.rejections
        if item.reason.startswith("max_open_orders")
    ]
    # La compra de la barra 1 sigue abierta: la de la barra 5 no cabe.
    assert len(capped) == 1 and "-buy-" in capped[0]
