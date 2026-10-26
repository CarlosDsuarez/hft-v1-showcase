"""Libro con signo: FIFO, corto, fees y roundtrip (HFT-06)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.hft.book import BOOK_SCHEMA_VERSION, HftBook
from src.hft.types import Fill, order_intent

UTC = timezone.utc
T0 = datetime(2024, 1, 3, 14, 31, tzinfo=UTC)
T1 = datetime(2024, 1, 3, 14, 32, tzinfo=UTC)
T2 = datetime(2024, 1, 3, 14, 33, tzinfo=UTC)


def _fill(intent_id: str, ts: datetime, qty: float, price: float, fees: float) -> Fill:
    return Fill(
        intent_id=intent_id,
        ts=ts,
        qty=qty,
        price=price,
        fees=fees,
        liquidity="taker",
    )


def _intent(intent_id: str, side: str, qty: float, ts: datetime = T0):
    return order_intent("test", "SPY", side, qty, ts, intent_id=intent_id)


def _nav_identity(book: HftBook, mark: float) -> float:
    """initial + realizado + no realizado − fees todavía abiertas."""
    return (
        book.initial_cash
        + book.realized_pnl
        + book.unrealized_pnl({"SPY": mark})
        - book.fees_open()
    )


def test_cubrir_un_corto_realiza_pnl_neto_de_fees():
    book = HftBook(10_000)
    book.apply_fills(
        [_fill("s", T0, 10, 100, 0.5)],
        {"s": _intent("s", "sell", 10)},
    )
    assert book.position("SPY") == -10
    assert book.cash == 10_999.5
    assert book.realized_pnl == 0.0
    assert book.fees == 0.5
    assert book.unrealized_pnl({"SPY": 90}) == 100
    assert book.nav({"SPY": 90}) == _nav_identity(book, 90)

    book.apply_fills(
        [_fill("b", T1, 10, 90, 0.5)],
        {"b": _intent("b", "buy", 10, T1)},
    )
    assert book.position("SPY") == 0
    assert book.positions() == {}
    assert book.realized_pnl == 99
    assert book.fees == 1
    assert book.fees_open() == 0
    assert book.cash == 10_099
    assert book.nav({"SPY": 90}) == 10_099


def test_fifo_de_cortos_reparte_fees_de_apertura():
    book = HftBook(10_000)
    book.apply_fills(
        [_fill("s1", T0, 4, 100, 0.5)],
        {"s1": _intent("s1", "sell", 4)},
    )
    book.apply_fills(
        [_fill("s2", T1, 4, 102, 0.5)],
        {"s2": _intent("s2", "sell", 4, T1)},
    )
    book.apply_fills(
        [_fill("b", T2, 6, 90, 0.75)],
        {"b": _intent("b", "buy", 6, T2)},
    )
    assert book.position("SPY") == -2
    assert book.realized_pnl == 62.5
    assert book.fees_open() == 0.25
    assert book.cash == 10_266.25
    assert book.nav({"SPY": 90}) == _nav_identity(book, 90)


def test_fifo_de_largos_y_cruce_a_corto():
    book = HftBook(10_000)
    book.apply_fills(
        [_fill("b", T0, 10, 100, 0.5)],
        {"b": _intent("b", "buy", 10)},
    )
    assert book.position("SPY") == 10
    assert book.realized_pnl == 0.0
    assert book.nav({"SPY": 100}) == _nav_identity(book, 100)

    book.apply_fills(
        [_fill("s", T1, 15, 110, 0.75)],
        {"s": _intent("s", "sell", 15, T1)},
    )
    assert book.position("SPY") == -5
    assert book.realized_pnl == 99
    assert book.fees_open() == 0.25
    assert book.cash == 10_648.75
    assert book.nav({"SPY": 110}) == _nav_identity(book, 110)

    book.apply_fills(
        [_fill("c", T2, 5, 100, 0.25)],
        {"c": _intent("c", "buy", 5, T2)},
    )
    # El corto de 5 a 110, cubierto a 100: +50 de precio, menos 0.25 abiertas y 0.25 de cierre.
    assert book.position("SPY") == 0
    assert book.realized_pnl == 99 + 50 - 0.25 - 0.25
    assert book.fees_open() == 0
    assert book.cash == book.initial_cash + book.realized_pnl


def test_apply_fills_es_transaccional():
    book = HftBook(1_000)
    book.apply_fills(
        [_fill("ok", T0, 4, 100, 0)],
        {"ok": _intent("ok", "buy", 4)},
    )
    before = json.dumps(book.to_dict(), sort_keys=True)
    good = _fill("a", T1, 2, 100, 0)
    too_big = _fill("b", T1, 20, 100, 0)
    with pytest.raises(ValueError, match="cash"):
        book.apply_fills(
            [good, too_big],
            {"a": _intent("a", "buy", 2, T1), "b": _intent("b", "buy", 20, T1)},
        )
    assert json.dumps(book.to_dict(), sort_keys=True) == before

    with pytest.raises(ValueError, match="sin intención"):
        book.apply_fills([_fill("falta", T1, 1, 100, 0)], {})
    assert json.dumps(book.to_dict(), sort_keys=True) == before


def test_roundtrip_identico_incluyendo_json():
    book = HftBook(10_000)
    book.apply_fills(
        [_fill("s", T0, 10, 100, 0.5), _fill("b", T1, 4, 90, 0.25)],
        {
            "s": _intent("s", "sell", 10),
            "b": _intent("b", "buy", 4, T1),
        },
    )
    payload = json.loads(json.dumps(book.to_dict()))
    restored = HftBook.from_dict(payload)
    assert restored.to_dict() == book.to_dict()
    assert restored.position("SPY") == book.position("SPY")
    assert restored.nav({"SPY": 95}) == book.nav({"SPY": 95})


def test_schema_desconocida_y_clave_ajena():
    book = HftBook(10_000)
    payload = book.to_dict()
    assert payload["schema_version"] == BOOK_SCHEMA_VERSION
    with pytest.raises(ValueError, match="schema_version"):
        HftBook.from_dict({**payload, "schema_version": 2})
    with pytest.raises(ValueError, match="desconocidas"):
        HftBook.from_dict({**payload, "extra": 1})


def test_un_lote_vacio_no_cambia_nada():
    book = HftBook(10_000)
    before = book.to_dict()
    assert book.apply_fills([], {}) == 0
    assert book.to_dict() == before
