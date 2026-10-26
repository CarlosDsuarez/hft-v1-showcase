"""Gateway de la Trading API: transporte, cupo, paralelo, ids y stream.

Todo contra el exchange falso de loopback (``src.brokers.fake_alpaca``)
o un ``urlopen`` guionizado. Ningún test abre red externa.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import urllib.error
from datetime import datetime, timezone

import pytest

from src.brokers.alpaca_adapter import (
    AlpacaClient,
    BrokerConfig,
    BrokerError,
    GatewayConfig,
    OrderSpec,
    make_intent_order_id,
)
from src.brokers.alpaca_stream import TradeUpdatesStream, validate_stream_url
from src.brokers.fake_alpaca import FakeAlpaca
from src.brokers.http_transport import PooledTransport, UrllibTransport, validate_transport_url
from src.brokers.rate_budget import TradingBudget
from src.instruments import Instrument, InstrumentMaster

UTC = timezone.utc
PAPER = "https://paper-api.alpaca.markets"
SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")


def _instruments() -> InstrumentMaster:
    return InstrumentMaster([
        Instrument(id=s, yahoo_symbol=s, broker_symbol=s, market="US", currency="USD",
                   asset_type="etf", sector="test", tradable_brokers=("alpaca",))
        for s in SYMBOLS
    ])


def _spec(cid: str, symbol: str = "SPY", side: str = "buy") -> OrderSpec:
    return OrderSpec(instrument_id=symbol, side=side, qty=1, order_type="market",
                     time_in_force="day", order_class="simple", client_order_id=cid,
                     extended_hours=False, short=False)


def _client(fake: FakeAlpaca, *, workers: int = 1, budget=None) -> AlpacaClient:
    return AlpacaClient(
        BrokerConfig(fake.api_key, fake.secret_key, PAPER),
        kill_dir=tempfile.mkdtemp(), instruments=_instruments(),
        gateway=GatewayConfig(pooled=True, workers=workers),
        transport=PooledTransport(fake.base_url, 5), budget=budget,
    )


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# --- transporte ---
class TestTransport:
    @pytest.mark.parametrize("url", [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/",
        "http://127.0.0.1:8123",
        "http://localhost:9",
    ])
    def test_acepta_paper_y_loopback(self, url):
        validate_transport_url(url)

    @pytest.mark.parametrize("url", [
        "https://api.alpaca.markets",
        "http://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.com",
        "https://user@paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/v2",
        "http://10.0.0.5:80",
    ])
    def test_rechaza_lo_demas(self, url):
        with pytest.raises(ValueError):
            validate_transport_url(url)

    def test_pool_reusa_una_conexion_por_hilo(self):
        with FakeAlpaca() as fake:
            client = _client(fake)
            for i in range(5):
                client.submit_order(_spec(f"t-{i}"), mark=500.0)
            assert client.transport.connections_opened == 1
            client.close()

    def test_urlopen_abre_una_por_peticion(self):
        with FakeAlpaca() as fake:
            client = AlpacaClient(
                BrokerConfig(fake.api_key, fake.secret_key, PAPER),
                kill_dir=tempfile.mkdtemp(), instruments=_instruments(),
                transport=UrllibTransport(fake.base_url, 5),
            )
            client.get_account()
            client.get_account()
            assert client.transport.connections_opened == 2

    def test_gateway_por_defecto_es_el_de_siempre(self):
        client = AlpacaClient(BrokerConfig("PK", "s", PAPER), kill_dir=tempfile.mkdtemp(),
                              instruments=_instruments())
        assert isinstance(client.transport, UrllibTransport)
        assert client.budget is None
        assert client.gateway.workers == 1


# --- cupo ---
class TestBudget:
    def test_capacidad_y_recarga(self):
        clock = _Clock()
        budget = TradingBudget(60, capacity=3, reserve_frac=0.0, clock=clock)
        assert [budget.try_acquire("entry") for _ in range(3)] == [0.0, 0.0, 0.0]
        assert budget.try_acquire("entry") == pytest.approx(1.0)
        clock.t += 1.0
        assert budget.try_acquire("entry") == 0.0

    def test_la_entrada_respeta_la_reserva_de_salidas(self):
        budget = TradingBudget(60, capacity=10, reserve_frac=0.2, clock=_Clock())
        taken = 0
        while budget.try_acquire("entry") == 0.0:
            taken += 1
        assert taken == 8
        assert budget.try_acquire("exit") == 0.0
        assert budget.try_acquire("exit") == 0.0
        assert budget.try_acquire("exit") > 0.0

    def test_429_frena_hasta_el_segundo_siguiente_al_reset_entero(self):
        # Reset en segundos enteros: el servidor pudo truncar y el cupo
        # vuelve dentro del segundo siguiente (5007 → hasta 5008).
        clock = _Clock()
        budget = TradingBudget(200, clock=clock, wall=lambda: 5000.0)
        budget.observe(429, {"x-ratelimit-reset": "5007", "x-ratelimit-remaining": "0"})
        assert budget.try_acquire("exit") == pytest.approx(8.0)
        clock.t += 7.0
        assert budget.try_acquire("exit") == pytest.approx(1.0)
        clock.t += 1.0
        assert budget.try_acquire("exit") == 0.0
        assert budget.stats.throttled_429 == 1

    def test_remaining_cero_no_despierta_antes_del_segundo_siguiente(self):
        # El caso del bench: el 200º POST dice Remaining 0 y Reset 5060
        # (ventana real hasta 5060.9). Despertar en 5060 da 429 en ráfaga.
        clock = _Clock()
        budget = TradingBudget(200, clock=clock, wall=lambda: 5059.5)
        budget.observe(200, {"x-ratelimit-reset": "5060", "x-ratelimit-remaining": "0"})
        clock.t += 0.5
        assert budget.try_acquire("exit") == pytest.approx(1.0)
        clock.t += 1.0
        assert budget.try_acquire("exit") == 0.0

    def test_reset_con_decimales_se_obedece_tal_cual(self):
        clock = _Clock()
        budget = TradingBudget(200, clock=clock, wall=lambda: 5000.0)
        budget.observe(429, {"x-ratelimit-reset": "5007.25"})
        assert budget.try_acquire("exit") == pytest.approx(7.25)
        clock.t += 7.25
        assert budget.try_acquire("exit") == 0.0

    def test_429_con_retry_after_y_sin_headers(self):
        clock = _Clock()
        budget = TradingBudget(200, clock=clock)
        budget.observe(429, {"retry-after": "3"})
        assert budget.try_acquire("exit") == pytest.approx(3.0)
        other = TradingBudget(200, clock=clock)
        other.observe(429, {})
        assert other.try_acquire("exit") == pytest.approx(1.0)

    def test_remaining_del_servidor_baja_las_fichas(self):
        budget = TradingBudget(200, clock=_Clock())
        budget.observe(200, {"x-ratelimit-remaining": "5"})
        assert budget.tokens == pytest.approx(5.0)
        budget.observe(200, {"x-ratelimit-remaining": "150"})
        assert budget.tokens == pytest.approx(5.0)

    def test_reset_absurdo_no_se_obedece(self):
        clock = _Clock()
        budget = TradingBudget(200, clock=clock, wall=lambda: 5000.0)
        budget.observe(200, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "999999"})
        clock.t += 0.5
        assert budget.try_acquire("exit") == 0.0

    def test_una_salida_esperando_pasa_antes_que_una_entrada(self):
        budget = TradingBudget(6000)
        with budget._cond:
            budget._waiting["exit"] += 1
        try:
            assert budget.acquire("entry", timeout=0.1) is False
        finally:
            with budget._cond:
                budget._waiting["exit"] -= 1
        assert budget.acquire("entry", timeout=0.1) is True

    def test_carril_desconocido(self):
        with pytest.raises(ValueError):
            TradingBudget(200).try_acquire("vip")


# --- paralelo ---
class TestRunMany:
    def test_en_serie_se_detiene_en_el_primer_error(self):
        client = AlpacaClient(BrokerConfig("PK", "s", PAPER), kill_dir=tempfile.mkdtemp(),
                              instruments=_instruments())
        ran = []

        def ok(i):
            ran.append(i)
            return i

        def boom():
            raise RuntimeError("x")

        out = client.run_many([lambda: ok(1), boom, lambda: ok(3)])
        assert ran == [1]
        assert out[0] == (1, None)
        assert isinstance(out[1][1], RuntimeError)
        assert out[2] == (None, None)

    def test_en_paralelo_conserva_el_orden(self):
        client = AlpacaClient(BrokerConfig("PK", "s", PAPER), kill_dir=tempfile.mkdtemp(),
                              instruments=_instruments(), gateway=GatewayConfig(workers=8))

        def slow(i):
            time.sleep(0.01 * (10 - i))
            return i

        out = client.run_many([lambda i=i: slow(i) for i in range(10)])
        assert [res for res, _err in out] == list(range(10))
        client.close()

    def test_ordenes_en_paralelo_sin_duplicados(self):
        with FakeAlpaca(latency_s=0.02) as fake:
            client = _client(fake, workers=16)
            client.ensure_account_verified()
            out = client.run_many([lambda i=i: client.submit_order(_spec(f"p-{i}"), mark=500.0)
                                   for i in range(64)])
            assert all(err is None and res["status"] == "submitted" for res, err in out)
            assert fake.stats.posts == 64
            assert len(fake.state.by_client) == 64
            assert client.sent_orders == 64
            client.close()

    @pytest.mark.parametrize("workers", [0, -1])
    def test_workers_invalido(self, workers):
        with pytest.raises(ValueError):
            GatewayConfig(workers=workers)


# --- 429 en POST ---
class _Resp:
    def __init__(self, payload, status=200):
        self._body = json.dumps(payload).encode()
        self.status = status
        self.headers = {"X-RateLimit-Remaining": "100"}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestPost429:
    def test_con_cupo_reenvia_sin_consultar(self, monkeypatch):
        calls = []

        def fake(req, timeout=None):
            calls.append((req.get_method(), req.full_url.split("?")[0]))
            if req.get_method() == "POST" and sum(1 for m, _ in calls if m == "POST") == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "slow",
                                             {"Retry-After": "0.01"}, io.BytesIO(b"{}"))
            if req.full_url.endswith("/v2/account"):
                return _Resp({"account_number": ""})
            return _Resp({"id": "o1", "status": "accepted"})

        monkeypatch.setattr("urllib.request.urlopen", fake)
        client = AlpacaClient(BrokerConfig("PK", "s", PAPER), kill_dir=tempfile.mkdtemp(),
                              instruments=_instruments(),
                              gateway=GatewayConfig(trading_rpm=6000))
        out = client.submit_order(_spec("c-429"), mark=500.0)
        assert out["status"] == "submitted"
        posts = [c for c in calls if c[0] == "POST"]
        lookups = [c for c in calls if "by_client_order_id" in c[1]]
        assert len(posts) == 2
        assert lookups == []


# --- ids ---
class TestIds:
    def test_barra_de_minuto_no_cambia(self):
        ts = datetime(2026, 10, 5, 14, 30, tzinfo=UTC)
        assert make_intent_order_id("orb", "AAPL", ts, "buy") == "h-orb-202610051430-AAPL-buy"

    def test_segundos_distintos_dan_ids_distintos(self):
        a = make_intent_order_id("p", "SPY", datetime(2026, 10, 5, 14, 30, 5, tzinfo=UTC), "buy")
        b = make_intent_order_id("p", "SPY", datetime(2026, 10, 5, 14, 30, 10, tzinfo=UTC), "buy")
        assert a == "h-p-20261005143005-SPY-buy"
        assert a != b
        assert make_intent_order_id(
            "p", "SPY", datetime(2026, 10, 5, 14, 30, 5, tzinfo=UTC), "buy") == a


# --- stream ---
def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestStream:
    def test_url(self):
        validate_stream_url("wss://paper-api.alpaca.markets/stream")
        validate_stream_url("ws://127.0.0.1:9/stream")
        for bad in ("wss://api.alpaca.markets/stream", "ws://paper-api.alpaca.markets/stream",
                    "wss://evil.example/stream"):
            with pytest.raises(ValueError):
                validate_stream_url(bad)

    def test_recibe_new_y_fill(self):
        with FakeAlpaca(stream=True) as fake:
            stream = TradeUpdatesStream(fake.api_key, fake.secret_key, url=fake.stream_url)
            assert stream.start(5)
            client = _client(fake)
            client.submit_order(_spec("s-1"), mark=500.0)
            events = []
            assert _wait(lambda: events.extend(stream.drain()) or len(events) >= 2)
            assert [e["event"] for e in events] == ["new", "fill"]
            assert events[1]["order"]["client_order_id"] == "s-1"
            assert stream.take_reconnected() is True
            assert stream.take_reconnected() is False
            assert fake.secret_key not in repr(stream)
            stream.stop()
            client.close()

    def test_claves_malas_no_reintenta(self):
        with FakeAlpaca(stream=True) as fake:
            stream = TradeUpdatesStream(fake.api_key, "otra", url=fake.stream_url)
            assert stream.start(5) is False
            assert stream.auth_error
            time.sleep(0.3)
            assert stream.connections == 0
            stream.stop()

    def test_reconecta_y_avisa(self):
        with FakeAlpaca(stream=True) as fake:
            stream = TradeUpdatesStream(fake.api_key, fake.secret_key, url=fake.stream_url,
                                        max_backoff_s=0.5)
            assert stream.start(5)
            assert stream.take_reconnected()
            assert _wait(lambda: len(fake.state.listeners) == 1)
            fake.drop_stream_clients()
            assert _wait(lambda: stream.connections >= 2, timeout=10)
            assert stream.take_reconnected()
            stream.stop()


def test_fake_tope_devuelve_429_con_headers():
    with FakeAlpaca(rpm_limit=2) as fake:
        client = AlpacaClient(
            BrokerConfig(fake.api_key, fake.secret_key, PAPER),
            kill_dir=tempfile.mkdtemp(), instruments=_instruments(),
            transport=PooledTransport(fake.base_url, 5),
        )
        client._send("GET", "/v2/clock", None)
        client._send("GET", "/v2/clock", None)
        with pytest.raises(BrokerError) as info:
            client._send("GET", "/v2/clock", None)
        assert info.value.status == 429
        assert fake.stats.throttled == 1


def test_contador_de_peticiones_con_hilos():
    with FakeAlpaca() as fake:
        client = _client(fake, workers=8)
        before = client.request_count
        threads = [threading.Thread(target=client.get_clock) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert client.request_count - before == 20
        client.close()
