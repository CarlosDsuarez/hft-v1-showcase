"""Adaptador Alpaca intradía (HFT-11). HTTP falso: ningún test abre red."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from src.brokers.alpaca_adapter import (
    AlpacaClient,
    BrokerConfig,
    IntradayOrderError,
    OrderSpec,
    format_qty,
    format_tick_price,
    make_intent_order_id,
)
from src.brokers.pre_trade import PreTradeBlocked, PreTradeGuard
from src.hft.costs import CostModel
from src.hft.execution.alpaca_exec import AlpacaExecution
from src.hft.types import Bracket, canonical_qty, order_intent

UTC = timezone.utc
PAPER = "https://paper-api.alpaca.markets"
SECRET = "secreto-falso"
TS = datetime(2026, 10, 5, 14, 30, tzinfo=UTC)


class _Resp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(url, code, body=b"{}"):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body))


def _install(monkeypatch, handler):
    calls = []

    def _fake(req, timeout=None):
        body = json.loads(req.data.decode()) if req.data else None
        parsed = urlparse(req.full_url)
        query = parse_qs(parsed.query)
        calls.append((req.get_method(), parsed.path, query, body))
        result = handler(req.get_method(), parsed.path, query, body)
        if isinstance(result, Exception):
            raise result
        return _Resp(result)

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


def _client(tmp_path, monkeypatch, handler):
    calls = _install(monkeypatch, handler)
    cfg = BrokerConfig("PKTEST1234", SECRET, PAPER)
    return AlpacaClient(cfg, kill_dir=tmp_path), calls


def _spec(**overrides) -> OrderSpec:
    raw = dict(
        instrument_id="AAPL",
        side="buy",
        qty=10,
        order_type="market",
        time_in_force="day",
        order_class="simple",
        client_order_id="h-orb-202610051430-AAPL-buy",
        extended_hours=False,
        short=False,
    )
    raw.update(overrides)
    return OrderSpec(**raw)


def _asset(**overrides):
    raw = {
        "id": "asset-aapl",
        "class": "us_equity",
        "exchange": "NASDAQ",
        "symbol": "AAPL",
        "name": "Apple",
        "status": "active",
        "tradable": True,
        "marginable": True,
        "shortable": True,
        "easy_to_borrow": True,
        "fractionable": True,
    }
    raw.update(overrides)
    return raw


def _account():
    return {
        "account_number": "",
        "equity": "100000",
        "buying_power": "400000",
        "account_blocked": False,
        "trading_blocked": False,
    }


def _posted(order_id="o1", status="accepted", **extra):
    raw = {"id": order_id, "status": status, "filled_qty": "0"}
    raw.update(extra)
    return raw


class _Clock:
    def __init__(self, moment):
        self.moment = moment

    def __call__(self):
        return self.moment


def _costs() -> CostModel:
    return CostModel(
        spread_floor_bps=1.0,
        tick=0.01,
        impact_y=1.0,
        adv_window=20,
        commission_per_share=0.0,
        sec_section31_rate=0.0000206,
        sec_section31_verified_on="2026-04-04",
        finra_taf_per_share=0.000195,
        finra_taf_cap=9.79,
        finra_taf_verified_on="2026-10-04",
        crypto_maker_bps=15.0,
        crypto_taker_bps=25.0,
        crypto_fees_verified_on="2026-10-04",
    )


def _guard(**overrides) -> PreTradeGuard:
    raw = dict(
        max_notional_per_order=None,
        max_total_notional=None,
        max_orders=None,
        block_if_account_blocked=False,
        block_if_market_closed=False,
    )
    raw.update(overrides)
    return PreTradeGuard(**raw)


class TestIntentIds:
    def test_forma_corta(self):
        assert make_intent_order_id("orb", "aapl", TS, "buy", 0) == \
            "h-orb-202610051430-AAPL-buy"

    def test_seq_distingue(self):
        assert make_intent_order_id("orb", "AAPL", TS, "buy", 3) == \
            "h-orb-202610051430-AAPL-buy-3"

    def test_quinientos_unicos_y_reinicio(self):
        def _batch():
            return [
                make_intent_order_id(
                    "orb",
                    f"S{i % 30}",
                    TS + timedelta(minutes=i % 180),
                    "buy" if i % 2 == 0 else "sell",
                    i,
                )
                for i in range(500)
            ]

        first = _batch()
        assert len(first) == len(set(first)) == 500
        assert all(len(item) <= 48 for item in first)
        assert _batch() == first

    def test_nombre_largo_cae_a_sha1_determinista(self):
        strat = "estrategia_con_nombre_muy_largo_para_forzar_hash"
        body = f"h-{strat}-202610051430-BRK.B-entry-99"
        assert len(body) > 48
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
        expected = f"{body[:48 - 11]}-{digest}"
        assert make_intent_order_id(strat, "BRK.B", TS, "entry", 99) == expected
        assert len(expected) == 48


class TestPayloads:
    def _send(self, tmp_path, monkeypatch, spec, asset=None):
        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "POST" and path == "/v2/orders":
                return _posted()
            raise AssertionError(f"HTTP inesperado {method} {path}")

        client, calls = _client(tmp_path, monkeypatch, handler)
        out = client.submit_order(spec, asset=asset, journal=[])
        posts = [call for call in calls if call[0] == "POST"]
        return out, posts, calls

    def test_market_day(self, tmp_path, monkeypatch):
        out, posts, _ = self._send(tmp_path, monkeypatch, _spec())
        body = posts[0][3]
        assert body == {
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "h-orb-202610051430-AAPL-buy",
            "extended_hours": False,
        }
        assert "order_class" not in body
        assert out["status"] == "submitted" and out["id"] == "o1"

    def test_limit_stop_stop_limit_y_tif(self, tmp_path, monkeypatch):
        cases = [
            _spec(order_type="limit", limit_price=290.123, time_in_force="day"),
            _spec(order_type="stop", stop_price=180.129, time_in_force="day"),
            _spec(order_type="stop_limit", limit_price=181, stop_price=180.129),
            _spec(order_type="market", time_in_force="ioc"),
            _spec(order_type="market", time_in_force="cls"),
        ]
        seen = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "POST" and path == "/v2/orders":
                seen.append(body)
                return _posted()
            raise AssertionError(method + path)

        client, _ = _client(tmp_path, monkeypatch, handler)
        for spec in cases:
            assert client.submit_order(spec)["status"] == "submitted"
        limit, stop, stop_limit, ioc, cls = seen
        assert limit["type"] == "limit" and limit["limit_price"] == "290.12"
        assert "290.123" not in json.dumps(limit)
        assert stop == {
            **{k: stop[k] for k in ("symbol", "qty", "side", "client_order_id", "extended_hours")},
            "type": "stop",
            "time_in_force": "day",
            "stop_price": "180.13",
        }
        assert stop_limit["type"] == "stop_limit"
        assert stop_limit["limit_price"] == "181"
        assert stop_limit["stop_price"] == "180.13"
        assert ioc["time_in_force"] == "ioc" and ioc["type"] == "market"
        assert cls["time_in_force"] == "cls" and cls["type"] == "market"

    def test_bracket_oto_oco(self, tmp_path, monkeypatch):
        seen = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "POST" and path == "/v2/orders":
                seen.append(body)
                return _posted()
            raise AssertionError(method + path)

        client, _ = _client(tmp_path, monkeypatch, handler)
        client.submit_order(_spec(
            order_class="bracket",
            take_profit_limit_price=301,
            stop_loss_stop_price=299,
        ))
        client.submit_order(_spec(
            order_class="oto",
            stop_loss_stop_price=299,
            client_order_id="h-orb-202610051430-AAPL-buy-1",
        ))
        client.submit_order(_spec(
            order_type="limit",
            limit_price=300,
            order_class="oco",
            side="sell",
            take_profit_limit_price=299,
            stop_loss_stop_price=301,
            client_order_id="h-orb-202610051430-AAPL-sell",
        ))
        bracket, oto, oco = seen
        assert bracket["order_class"] == "bracket"
        assert bracket["take_profit"] == {"limit_price": "301"}
        assert bracket["stop_loss"] == {"stop_price": "299"}
        assert bracket["extended_hours"] is False
        assert oto["order_class"] == "oto"
        assert "take_profit" not in oto
        assert oto["stop_loss"] == {"stop_price": "299"}
        assert oco["order_class"] == "oco" and oco["type"] == "limit"
        assert oco["take_profit"]["limit_price"] == "299"
        assert oco["stop_loss"]["stop_price"] == "301"

    def test_precio_bajo_un_dolar_usa_cuatro_decimales(self):
        assert format_tick_price(0.12345) == "0.1235"
        assert format_tick_price(290.123) == "290.12"

    def test_422_duplicado_resuelve_la_orden_existente(self, tmp_path, monkeypatch):
        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "POST" and path == "/v2/orders":
                raise _http_error(PAPER + "/v2/orders", 422, b'{"code":42210000}')
            if method == "GET" and path == "/v2/orders:by_client_order_id":
                return {"id": "ya-esta", "status": "accepted",
                        "client_order_id": query["client_order_id"][0]}
            raise AssertionError(method + path)

        client, calls = _client(tmp_path, monkeypatch, handler)
        out = client.submit_order(_spec())
        assert out["status"] == "submitted" and out["id"] == "ya-esta"
        assert len([call for call in calls if call[0] == "POST"]) == 1


class TestCantidadCanonica:
    """Una sola regla de cantidad para el sizer y el POST (9 decimales)."""

    def test_format_qty_envia_la_cantidad_canonica(self):
        # 2026-10-12: el intent traía 152.06812652068126 y salió "152.068126521".
        assert format_qty(152.06812652068126) == "152.06812652"
        assert float(format_qty(152.06812652068126)) == canonical_qty(152.06812652068126)

    def test_el_ruido_binario_no_resta_un_cuanto(self):
        noisy = 152.06812652099998
        assert canonical_qty(noisy) == 152.068126521
        assert format_qty(noisy) == "152.068126521"

    def test_los_enteros_siguen_enteros(self):
        assert canonical_qty(112.0) == 112.0
        assert format_qty(112.0) == "112"


class TestValidacionSinHttp:
    def test_fraccional_con_bracket_no_abre_red(self, tmp_path, monkeypatch):
        client, calls = _client(
            tmp_path, monkeypatch, lambda *a: (_ for _ in ()).throw(AssertionError("red")),
        )
        with pytest.raises(IntradayOrderError, match="fraccional"):
            client.submit_order(_spec(
                qty=1.5,
                order_class="bracket",
                take_profit_limit_price=120,
                stop_loss_stop_price=80,
            ))
        assert calls == []

    def test_fraccional_stop_y_extendido_no_abren_red(self, tmp_path, monkeypatch):
        client, calls = _client(
            tmp_path, monkeypatch, lambda *a: (_ for _ in ()).throw(AssertionError("red")),
        )
        with pytest.raises(IntradayOrderError, match="fraccional"):
            client.submit_order(_spec(qty=0.5, order_type="stop", stop_price=10))
        with pytest.raises(IntradayOrderError, match="extendido"):
            client.submit_order(_spec(extended_hours=True))
        with pytest.raises(IntradayOrderError, match="corto"):
            client.submit_order(_spec(side="sell", qty=1.5, short=True))
        assert calls == []

    def test_corto_sin_shortable_ni_easy_to_borrow(self, tmp_path, monkeypatch):
        client, calls = _client(
            tmp_path, monkeypatch, lambda *a: (_ for _ in ()).throw(AssertionError("red")),
        )
        with pytest.raises(IntradayOrderError, match="shortable"):
            client.submit_order(
                _spec(side="sell", short=True),
                asset=_asset(shortable=False),
            )
        with pytest.raises(IntradayOrderError, match="easy_to_borrow"):
            client.submit_order(
                _spec(side="sell", short=True),
                asset=_asset(easy_to_borrow=False),
            )
        with pytest.raises(IntradayOrderError, match="ficha"):
            client.submit_order(_spec(side="sell", short=True))
        assert calls == []

    def test_no_ruteable_sin_http(self, tmp_path, monkeypatch):
        client, calls = _client(
            tmp_path, monkeypatch, lambda *a: (_ for _ in ()).throw(AssertionError("red")),
        )
        out = client.submit_order(_spec(
            instrument_id="ICOLCAP",
            client_order_id="h-orb-202610051430-ICOLCAP-buy",
        ))
        assert out["status"] == "not_routable"
        assert calls == []


class TestGuardas:
    def test_kill_anota_y_no_postea(self, tmp_path, monkeypatch):
        (tmp_path / "KILL").write_text("corte\n", encoding="utf-8")
        journal = []

        def handler(method, path, query, body):
            raise AssertionError("el KILL no debe abrir red")

        client, calls = _client(tmp_path, monkeypatch, handler)
        with pytest.raises(PreTradeBlocked):
            client.submit_order(_spec(), journal=journal)
        assert calls == []
        assert journal[0]["status"] == "blocked"
        assert "KILL" in journal[0]["error"]

    def test_tasa_nocional_y_exposicion(self, tmp_path, monkeypatch):
        posts = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "POST" and path == "/v2/orders":
                posts.append(body)
                return _posted()
            raise AssertionError(method + path)

        client, _ = _client(tmp_path, monkeypatch, handler)
        journal = []
        guard = _guard(max_orders_per_minute=1)
        assert client.submit_order(
            _spec(), guard=guard, orders_this_minute=0, journal=journal,
        )["status"] == "submitted"
        blocked = client.submit_order(
            _spec(client_order_id="h-orb-202610051431-AAPL-buy"),
            guard=guard, orders_this_minute=1, journal=journal,
        )
        assert blocked["status"] == "blocked"
        assert "max_orders_per_minute" in blocked["error"]
        assert len(posts) == 1

        journal.clear()
        pricey = client.submit_order(
            _spec(order_type="limit", limit_price=50, qty=100,
                  client_order_id="h-orb-202610051432-AAPL-buy"),
            guard=_guard(max_notional_per_order=1000),
            journal=journal,
        )
        assert pricey["status"] == "blocked"
        assert "max_notional_per_order" in pricey["error"]

        class _Equity:
            equity = 100_000.0

        exposed = client.submit_order(
            _spec(client_order_id="h-orb-202610051433-AAPL-buy"),
            guard=_guard(max_gross_exposure=0.5),
            risk_state=_Equity(),
            projected_gross=60_000,
            mark=100,
            journal=journal,
        )
        assert exposed["status"] == "blocked"
        assert "max_gross_exposure" in exposed["error"]
        assert len(posts) == 1


class TestLecturas:
    def test_cancel_replace_lista_y_actividades(self, tmp_path, monkeypatch):
        def handler(method, path, query, body):
            if method == "DELETE" and path == "/v2/orders/o1":
                return {}
            if method == "DELETE" and path == "/v2/orders":
                return [{"id": "o1", "status": 200}]
            if method == "PATCH" and path == "/v2/orders/o1":
                assert body == {"qty": "4", "limit_price": "150.5", "stop_price": "140"}
                return {"id": "o1", "status": "accepted", **body}
            if method == "GET" and path == "/v2/orders":
                assert query["status"] == ["open"]
                assert query["limit"] == ["25"]
                assert query["after"] == ["2026-10-05T13:30:00Z"]
                assert query["nested"] == ["true"]
                return [{"id": "o1", "status": "new"}]
            if method == "GET" and path == "/v2/account/activities":
                assert query["activity_types"] == ["FILL"]
                assert query["date"] == ["2026-10-05"]
                assert query["direction"] == ["asc"]
                if "page_token" not in query:
                    assert query["page_size"] == ["2"]
                    return [
                        {"id": "p1", "activity_type": "FILL"},
                        {"id": "p2", "activity_type": "FILL"},
                    ]
                assert query["page_token"] == ["p2"]
                return [{"id": "p3", "activity_type": "FILL"}]
            raise AssertionError(method + path + str(query))

        client, _ = _client(tmp_path, monkeypatch, handler)
        assert client.cancel_order("o1") == {}
        assert client.cancel_all_orders()[0]["id"] == "o1"
        replaced = client.replace_order("o1", qty=4, limit_price=150.5, stop_price=140)
        assert replaced["qty"] == "4"
        listed = client.list_orders(
            "open", after=datetime(2026, 10, 5, 13, 30, tzinfo=UTC), limit=25, nested=True,
        )
        assert listed[0]["id"] == "o1"
        fills = client.get_account_activities("FILL", datetime(2026, 10, 5).date(), page_size=2)
        assert [row["id"] for row in fills] == ["p1", "p2", "p3"]

    def test_cache_diaria_de_asset(self, tmp_path, monkeypatch):
        gets = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/assets/AAPL":
                gets.append(path)
                return _asset()
            raise AssertionError(method + path)

        client, _ = _client(tmp_path, monkeypatch, handler)
        day = datetime(2026, 10, 5, tzinfo=UTC).date()
        first = client.get_asset("AAPL", as_of=day)
        second = client.get_asset("AAPL", as_of=day)
        assert first["symbol"] == "AAPL" and second is not None
        assert gets == ["/v2/assets/AAPL"]
        client.get_asset("AAPL", as_of=datetime(2026, 10, 6, tzinfo=UTC).date())
        assert len(gets) == 2


class TestAlpacaExecution:
    def _execution(self, tmp_path, monkeypatch, handler, **kwargs):
        client, calls = _client(tmp_path, monkeypatch, handler)
        clock = kwargs.pop("clock", _Clock(TS))
        execution = AlpacaExecution(client, _costs(), clock=clock, **kwargs)
        return execution, calls, clock

    def test_submit_mapea_el_intent_y_el_reinicio_reusa_el_id(self, tmp_path, monkeypatch):
        posts = []
        looked = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "GET" and path == "/v2/account/activities":
                return []
            if method == "GET" and path == "/v2/orders":
                return []
            if method == "POST" and path == "/v2/orders":
                posts.append(body)
                if len(posts) == 1:
                    return _posted(client_order_id=body["client_order_id"])
                raise _http_error(PAPER + "/v2/orders", 422)
            if method == "GET" and path == "/v2/orders:by_client_order_id":
                looked.append(query["client_order_id"][0])
                return {"id": "o1", "status": "accepted",
                        "client_order_id": query["client_order_id"][0]}
            raise AssertionError(method + path)

        execution, _, _ = self._execution(tmp_path, monkeypatch, handler)
        intent = order_intent("orb", "AAPL", "buy", 10, TS, seq=0)
        assert execution.submit([intent], now=TS) == ()
        body = posts[0]
        assert body["client_order_id"] == "h-orb-202610051430-AAPL-buy"
        assert body["type"] == "market" and body["extended_hours"] is False
        # Reinicio: otra instancia, misma intención, mismo id. Alpaca 422.
        again, _, _ = self._execution(tmp_path, monkeypatch, handler)
        assert again.submit([intent], now=TS) == ()
        assert posts[1]["client_order_id"] == body["client_order_id"]
        assert looked == [body["client_order_id"]]

    def test_fraccional_bracket_en_el_adaptador_no_abre_red(self, tmp_path, monkeypatch):
        execution, calls, _ = self._execution(
            tmp_path, monkeypatch, lambda *a: (_ for _ in ()).throw(AssertionError("red")),
        )
        intent = order_intent(
            "orb", "AAPL", "buy", 1.5, TS,
            bracket=Bracket(take=120, stop=80),
        )
        with pytest.raises(IntradayOrderError, match="fraccional"):
            execution.submit([intent], now=TS)
        assert calls == []

    @staticmethod
    def _sells(posts):
        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "GET" and path in ("/v2/account/activities", "/v2/orders"):
                return []
            if method == "POST" and path == "/v2/orders":
                posts.append(body)
                return _posted(client_order_id=body["client_order_id"])
            raise AssertionError(method + path)

        return handler

    def test_vender_el_largo_fraccional_sembrado_no_es_corto(self, tmp_path, monkeypatch):
        # HFT-PAPER-17: tras un relanzamiento el adaptador no vio el fill de
        # la compra. La sesión le siembra la posición propia del broker.
        posts = []
        execution, _, _ = self._execution(tmp_path, monkeypatch, self._sells(posts))
        execution.seed_positions({"AAPL": 1.5})
        intent = order_intent("vwap_reversion", "AAPL", "sell", 1.5, TS)
        execution.submit([intent], now=TS)
        assert [(body["side"], body["qty"]) for body in posts] == [("sell", "1.5")]

    def test_vender_mas_de_lo_sembrado_en_fraccion_sigue_siendo_corto(self, tmp_path, monkeypatch):
        posts = []
        execution, _, _ = self._execution(tmp_path, monkeypatch, self._sells(posts))
        execution.seed_positions({"AAPL": 1.5})
        intent = order_intent("vwap_reversion", "AAPL", "sell", 2.5, TS)
        with pytest.raises(IntradayOrderError, match="el corto exige acciones enteras"):
            execution.submit([intent], now=TS)
        assert posts == []

    def test_poll_entrega_el_fill_una_vez(self, tmp_path, monkeypatch):
        state = {"fills": []}

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "POST" and path == "/v2/orders":
                return _posted()
            if method == "GET" and path == "/v2/account/activities":
                return list(state["fills"])
            if method == "GET" and path == "/v2/orders":
                qty = "2" if state["fills"] else "0"
                return [{
                    "id": "o1",
                    "status": "filled" if state["fills"] else "accepted",
                    "filled_qty": qty,
                    "filled_avg_price": "190.5" if state["fills"] else None,
                    "client_order_id": "h-orb-202610051430-AAPL-buy",
                    "symbol": "AAPL",
                }]
            raise AssertionError(method + path)

        clock = _Clock(TS)
        execution, calls, _ = self._execution(tmp_path, monkeypatch, handler, clock=clock)
        intent = order_intent("orb", "AAPL", "buy", 2, TS)
        assert execution.submit([intent], now=TS) == ()
        state["fills"] = [{
            "id": "20261005000000000::fill-1",
            "activity_type": "FILL",
            "type": "fill",
            "order_id": "o1",
            "qty": "2",
            "price": "190.5",
            "side": "buy",
            "symbol": "AAPL",
            "transaction_time": "2026-10-05T14:31:00Z",
        }]
        before = len(calls)
        clock.moment = TS + timedelta(seconds=1)
        assert execution.poll() == ()
        assert len(calls) == before
        clock.moment = TS + timedelta(seconds=5)
        fills = execution.poll()
        assert len(fills) == 1
        assert fills[0].intent_id == intent.intent_id
        assert fills[0].qty == 2
        assert fills[0].price == 190.5
        assert fills[0].liquidity == "taker"
        assert fills[0].fees == 0
        again = execution.poll()
        # Dentro del intervalo no hay otra lectura.
        assert again == ()
        clock.moment = TS + timedelta(seconds=10)
        assert execution.poll() == ()

    def test_actividad_tardia_no_duplica_lo_que_ya_dio_la_orden(self, tmp_path, monkeypatch):
        # HFT-PAPER-09: paper 2026-10-12, XLE 152.068126521 (aquí AAPL, mismo
        # reparto). Alpaca llenó en dos actividades: 152 y la fracción 0.068126521.
        # La primera lectura vio la de 152 y la orden ya llena, y la vía de órdenes
        # dio la fracción. La actividad fraccional llegó en la lectura siguiente y
        # se volvía a contar: "el fill excede la cantidad" y exit 5.
        qty = "152.068126521"
        whole = {
            "id": "20261012104307231::a1", "type": "fill", "order_id": "o1",
            "qty": "152", "price": "65.73", "side": "buy", "symbol": "AAPL",
            "transaction_time": "2026-10-12T14:43:07.231694Z", "order_status": "filled",
        }
        fraction = dict(
            whole, id="20261012104307233::a2", type="partial_fill",
            qty="0.068126521", transaction_time="2026-10-12T14:43:07.233444Z",
            order_status="partially_filled",
        )
        state = {"activities": [whole]}

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "POST" and path == "/v2/orders":
                return _posted(client_order_id=body["client_order_id"])
            if method == "GET" and path == "/v2/account/activities":
                return list(state["activities"])
            if method == "GET" and path == "/v2/orders":
                return [{
                    "id": "o1", "status": "filled", "filled_qty": qty,
                    "filled_avg_price": "65.73", "symbol": "AAPL",
                    "client_order_id": "h-vwap_reversion-202610051430-AAPL-buy",
                    "filled_at": "2026-10-12T14:43:07.231694Z",
                }]
            raise AssertionError(method + path)

        clock = _Clock(TS)
        execution, _, _ = self._execution(tmp_path, monkeypatch, handler, clock=clock)
        intent = order_intent(
            "vwap_reversion", "AAPL", "buy", 152.06812652068126, TS,
            order_type="marketable_limit", limit_price=65.76,
        )
        first = execution.submit([intent], now=TS)
        assert sum(fill.qty for fill in first) == pytest.approx(float(qty), abs=1e-12)
        state["activities"] = [whole, fraction]
        clock.moment = TS + timedelta(minutes=1)
        assert execution.poll() == ()

    def test_tasa_en_el_lote_bloquea_la_segunda(self, tmp_path, monkeypatch):
        posts = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path.startswith("/v2/assets/"):
                return _asset(symbol=path.rsplit("/", 1)[-1])
            if method == "POST" and path == "/v2/orders":
                posts.append(body["symbol"])
                return _posted(order_id=f"o-{len(posts)}")
            if method == "GET" and path in ("/v2/account/activities", "/v2/orders"):
                return []
            raise AssertionError(method + path)

        execution, _, _ = self._execution(
            tmp_path, monkeypatch, handler, guard=_guard(max_orders_per_minute=1),
        )
        intents = [
            order_intent("orb", "AAPL", "buy", 1, TS, seq=0),
            order_intent("orb", "MSFT", "buy", 1, TS, seq=1),
        ]
        execution.submit(intents, now=TS)
        assert posts == ["AAPL"]
        blocked = [row for row in execution.journal if row["status"] == "blocked"]
        assert blocked and "max_orders_per_minute" in blocked[0]["error"]

    def test_trade_updates_exige_stream(self, tmp_path, monkeypatch):
        # D-07 se revocó el 2026-10-13: el websocket se acepta, pero sin
        # un stream que vaciar no hay de dónde leer los fills.
        client, calls = _client(tmp_path, monkeypatch, lambda *a: {})
        with pytest.raises(ValueError, match="stream"):
            AlpacaExecution(client, _costs(), clock=_Clock(TS), use_trade_updates=True)
        with pytest.raises(TypeError, match="drain"):
            AlpacaExecution(client, _costs(), clock=_Clock(TS), stream=object())
        assert calls == []

    def test_misma_instancia_no_reenvia(self, tmp_path, monkeypatch):
        posts = []

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "POST" and path == "/v2/orders":
                posts.append(body)
                return _posted()
            if method == "GET" and path in ("/v2/account/activities", "/v2/orders"):
                return []
            raise AssertionError(method + path)

        execution, _, _ = self._execution(tmp_path, monkeypatch, handler)
        intent = order_intent("orb", "AAPL", "buy", 1, TS)
        execution.submit([intent], now=TS)
        execution.submit([intent], now=TS + timedelta(minutes=1))
        assert len(posts) == 1

    @pytest.mark.parametrize("visibles", [0, 1], ids=["ninguna", "solo_la_entera"])
    def test_actividad_tardia_no_repite_lo_que_ya_dio_la_orden(
        self, tmp_path, monkeypatch, visibles,
    ):
        """2026-10-12, XLE 152.068126521: ``GET /v2/orders`` ya mostraba
        ``filled_qty`` completo y las actividades FILL (la parte entera y
        la fracción, por separado) llegaron en la lectura siguiente. El
        fill salía dos veces y el motor cortaba la sesión."""
        acts = [
            {
                "id": "20261005103100001::entera", "type": "fill", "order_id": "o1",
                "qty": "152", "price": "65.73", "side": "buy", "symbol": "AAPL",
                "transaction_time": "2026-10-05T14:31:00.001Z",
            },
            {
                "id": "20261005103100002::fraccion", "type": "partial_fill", "order_id": "o1",
                "qty": "0.068126521", "price": "65.73", "side": "buy", "symbol": "AAPL",
                "transaction_time": "2026-10-05T14:31:00.002Z",
            },
        ]
        state = {"order": None, "acts": []}

        def handler(method, path, query, body):
            if method == "GET" and path == "/v2/account":
                return _account()
            if method == "GET" and path == "/v2/assets/AAPL":
                return _asset()
            if method == "POST" and path == "/v2/orders":
                return _posted(client_order_id=body["client_order_id"])
            if method == "GET" and path == "/v2/account/activities":
                return list(state["acts"])
            if method == "GET" and path == "/v2/orders":
                return [state["order"]] if state["order"] else []
            raise AssertionError(method + path)

        clock = _Clock(TS)
        execution, _, _ = self._execution(tmp_path, monkeypatch, handler, clock=clock)
        intent = order_intent("vwap_reversion", "AAPL", "buy", 152.068126521, TS)
        assert execution.submit([intent], now=TS) == ()
        state["order"] = {
            "id": "o1", "status": "filled", "filled_qty": "152.068126521",
            "filled_avg_price": "65.73", "symbol": "AAPL",
            "client_order_id": "h-vwap_reversion-202610051430-AAPL-buy",
        }
        state["acts"] = acts[:visibles]
        clock.moment = TS + timedelta(seconds=5)
        first = execution.poll()
        state["acts"] = acts
        clock.moment = TS + timedelta(seconds=10)
        second = execution.poll()
        fills = first + second
        assert {fill.intent_id for fill in fills} == {intent.intent_id}
        assert sum(fill.qty for fill in fills) == pytest.approx(152.068126521, abs=1e-12)
