"""Contrato de estrategia y TheoryCard (HFT-06)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.hft.strategies.base import STRATEGIES, StrategyContext, register, validate_strategy
from src.hft.types import TheoryCard, TheoryParam, make_intent_id, order_intent

UTC = timezone.utc
TS = datetime(2024, 1, 3, 14, 31, tzinfo=UTC)


@dataclass(frozen=True)
class _Params:
    qty: float = 1.0


def _card(**overrides) -> TheoryCard:
    payload = dict(
        hypothesis="Regla fija para comprobar el contrato.",
        mechanism="Nadie paga: no es una estrategia de capital.",
        references=("HFT-06",),
        data_requirements=("barras finales <= t",),
        params=(TheoryParam(
            "qty", 1.0, "Cantidad fija del test, elegida de antemano.", "HFT-06",
        ),),
        holding_period="una barra",
        failure_regimes=("cualquier régimen: no hay hipótesis de mercado",),
        kill_criteria=("no promoverla a paper",),
        research_only=True,
    )
    payload.update(overrides)
    return TheoryCard(**payload)


class _Estrategia:
    id = "contrato"
    theory = _card()
    params = _Params()

    def required_history(self):
        return (2, 1)

    def on_bar(self, ctx):
        return []


@pytest.fixture
def registro():
    saved = dict(STRATEGIES)
    STRATEGIES.clear()
    yield STRATEGIES
    STRATEGIES.clear()
    STRATEGIES.update(saved)


def test_el_registro_real_cumple_el_contrato():
    """Cada estrategia registrada (ORB y las que sigan) tiene que estar completa."""
    for strategy in STRATEGIES.values():
        assert validate_strategy(strategy) == ()


def test_una_estrategia_completa_se_registra(registro):
    strategy = _Estrategia()
    register(strategy)
    assert registro["contrato"] is strategy
    assert validate_strategy(strategy) == ()
    with pytest.raises(ValueError, match="ya registrada"):
        register(strategy)


def test_theory_card_incompleta_no_entra(registro):
    rota = _Estrategia()
    rota.theory = _card(mechanism="  ")
    assert any("mechanism" in item for item in validate_strategy(rota))
    with pytest.raises(ValueError, match="mechanism"):
        register(rota)

    sin_fuente = _Estrategia()
    sin_fuente.theory = _card(params=(TheoryParam("qty", 1.0, "hay motivo", "  "),))
    assert any("fuente" in item for item in validate_strategy(sin_fuente))

    sin_justificacion = _Estrategia()
    sin_justificacion.theory = _card(params=(TheoryParam("qty", 1.0, "", "HFT-06"),))
    assert any("justificación" in item for item in validate_strategy(sin_justificacion))


def test_params_tienen_que_coincidir_y_ser_frozen():
    distinta = _Estrategia()
    distinta.params = _Params(qty=2.0)
    assert any("TheoryCard" in item for item in validate_strategy(distinta))

    @dataclass
    class Mutable:
        qty: float = 1.0

    suelta = _Estrategia()
    suelta.params = Mutable()
    assert any("frozen" in item for item in validate_strategy(suelta))


def test_intent_id_determinista():
    first = make_intent_id("agenda", "spy", TS, "buy", 0)
    second = make_intent_id("agenda", "SPY", TS, "buy", 0)
    assert first == second
    assert first != make_intent_id("agenda", "SPY", TS, "buy", 1)
    intent = order_intent("agenda", "spy", "buy", 1, TS)
    assert intent.intent_id == first
    assert intent.symbol == "SPY"
    assert intent.signal_ts == TS


def _ctx(position: float, *pending) -> StrategyContext:
    return StrategyContext(
        now=TS,
        bar_ts=TS,
        bars=None,
        features=None,
        positions=(("SPY", position),) if position else (),
        cash=0.0,
        realized_pnl=0.0,
        pending=tuple(("SPY", qty) for qty in pending),
    )


def test_pending_qty_suma_compras_y_ventas_vivas():
    assert _ctx(0.0, 6.0, -4.0).pending_qty("spy") == pytest.approx(2.0)
    assert _ctx(0.0).pending_qty("SPY") == 0.0


@pytest.mark.parametrize(
    ("position", "pending", "expected"),
    [
        (10.0, (), 10.0),
        (10.0, (-10.0,), 0.0),
        (10.0, (-4.0,), 6.0),
        (-8.0, (8.0,), 0.0),
        (-8.0, (3.0,), -5.0),
        (0.0, (-5.0,), 0.0),
        # El resto de la entrada no cubre la salida: si no se llena y la
        # salida lo incluye, el libro queda corto (HFT-PAPER-16).
        (4.0, (6.0,), 4.0),
        (4.0, (6.0, -4.0), 0.0),
    ],
)
def test_exit_qty_descuenta_solo_lo_vivo_del_lado_contrario(position, pending, expected):
    assert _ctx(position, *pending).exit_qty("SPY") == pytest.approx(expected)
