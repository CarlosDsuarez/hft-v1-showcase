"""
test_broker_idempotency.py — Envio de ordenes idempotente y auditable (sin red).

urllib.request.urlopen se reemplaza por un fake con respuestas guionizadas:
cada ruta (metodo, url) devuelve una respuesta fija o una secuencia `Seq`
que se consume en orden (timeouts, 5xx, 403...).
"""

import http.client
import io
import json
import urllib.error

import pandas as pd
import pytest

import src.brokers.alpaca_adapter as adapter
from src.brokers.alpaca_adapter import (
    AlpacaClient,
    BrokerConfig,
    BrokerError,
    BrokerTransientError,
    make_client_order_id,
)

BASE = "https://paper-api.alpaca.markets"
ORDERS = BASE + "/v2/orders"
ACCOUNT = BASE + "/v2/account"
CLOCK = BASE + "/v2/clock"
POSITIONS = BASE + "/v2/positions"


def _lookup(cid):
    return f"{BASE}/v2/orders:by_client_order_id?client_order_id={cid}"


class Seq:
    """Respuestas consumidas en orden (se agota → AssertionError)."""

    def __init__(self, *items):
        self.items = list(items)


class FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class CutResp(FakeResp):
    """Body cortado: read() lanza `error` o devuelve bytes que no son JSON."""

    def __init__(self, raw=b"", error=None):
        self._body = raw
        self._error = error

    def read(self):
        if self._error is not None:
            raise self._error
        return self._body


def _incomplete():
    # Lo que vio paper el 2026-10-06: "IncompleteRead(58546 bytes read)".
    return CutResp(error=http.client.IncompleteRead(b'{"id": "o', 100))


def _truncated_json():
    return CutResp(raw=b'{"id": "o1", "status": "acc')


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", {},
                                  io.BytesIO(b'{"message":"x"}'))


class FakeUrlopen:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []  # (method, url, body_dict|None)

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode()) if req.data else None
        self.calls.append((req.get_method(), req.full_url, body))
        key = (req.get_method(), req.full_url)
        if key not in self.routes:
            raise AssertionError(f"HTTP inesperado: {key}")
        outcome = self.routes[key]
        if isinstance(outcome, Seq):
            if not outcome.items:
                raise AssertionError(f"Secuencia agotada: {key}")
            outcome = outcome.items.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, FakeResp):
            return outcome
        return FakeResp(outcome)

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda s: None)


def _client(monkeypatch, routes):
    fake = FakeUrlopen({("GET", ACCOUNT): {"account_number": "", "equity": "1"},
                        **routes})
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return AlpacaClient(BrokerConfig("PKTEST1234", "secreto", BASE)), fake


def _one(ticker="AAPL", side="buy", qty=1.0):
    return pd.DataFrame([{"ticker": ticker, "side": side, "qty": qty}])


class TestClientOrderId:
    def test_format_is_run_ticker_side(self):
        assert make_client_order_id("hft-20260925", "AAPL", "buy") == \
            "hft-20260925-AAPL-buy"

    def test_longer_than_48_chars_rejected(self):
        with pytest.raises(ValueError, match="48"):
            make_client_order_id("r" * 40, "AAPL", "sell")

    def test_post_carries_deterministic_client_order_id(self, monkeypatch):
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): {"id": "o1", "status": "accepted"}})
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        (_, _, body), = fake.posts()
        assert body["client_order_id"] == "run1-AAPL-buy"
        assert out[0]["client_order_id"] == "run1-AAPL-buy"

    def test_too_long_id_fails_before_any_http(self, monkeypatch):
        client, fake = _client(monkeypatch, {})
        with pytest.raises(ValueError, match="48"):
            client.submit_orders(_one(), dry_run=False, run_id="r" * 45)
        assert fake.calls == []


class TestPostNotBlindlyRetried:
    @pytest.mark.parametrize("failure", [
        TimeoutError("timed out"),
        _http_error(ORDERS, 503),
    ], ids=["timeout", "http503"])
    def test_ambiguous_failure_with_existing_order_does_not_duplicate(
            self, monkeypatch, no_sleep, failure):
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): Seq(failure),
            ("GET", _lookup("run1-AAPL-buy")): {
                "id": "o1", "status": "accepted",
                "client_order_id": "run1-AAPL-buy"},
        })
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        assert len(fake.posts()) == 1
        assert out[0]["status"] == "submitted" and out[0]["id"] == "o1"
        assert client.sent_orders == 1

    def test_ambiguous_failure_without_order_resends_same_id(
            self, monkeypatch, no_sleep):
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): Seq(TimeoutError("timed out"),
                                  {"id": "o2", "status": "accepted"}),
            ("GET", _lookup("run1-AAPL-buy")): _http_error(
                _lookup("run1-AAPL-buy"), 404),
        })
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        posts = fake.posts()
        assert len(posts) == 2
        assert {p[2]["client_order_id"] for p in posts} == {"run1-AAPL-buy"}
        assert out[0]["id"] == "o2" and client.sent_orders == 1

    def test_duplicate_422_on_resend_resolves_to_existing_order(
            self, monkeypatch, no_sleep):
        # 1er POST llego pero aun no era visible; el reenvio choca con 422.
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): Seq(TimeoutError("timed out"),
                                  _http_error(ORDERS, 422)),
            ("GET", _lookup("run1-AAPL-buy")): Seq(
                _http_error(_lookup("run1-AAPL-buy"), 404),
                {"id": "o3", "status": "new"}),
        })
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        assert out[0]["status"] == "submitted" and out[0]["id"] == "o3"
        assert len(fake.posts()) == 2

    def test_first_post_422_duplicate_resolves_to_existing_order(
            self, monkeypatch, no_sleep):
        # Rerun el mismo dia tras enviar todo sin comprometer el estado: el id
        # ya existe en Alpaca → la orden cuenta como enviada, no como rechazo.
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): _http_error(ORDERS, 422),
            ("GET", _lookup("run1-AAPL-buy")): {"id": "o4", "status": "accepted"},
        })
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        assert out[0]["status"] == "submitted" and out[0]["id"] == "o4"
        assert len(fake.posts()) == 1

    def test_post_5xx_exhausted_aborts(self, monkeypatch, no_sleep):
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): _http_error(ORDERS, 503),
            ("GET", _lookup("run1-AAPL-buy")): _http_error(
                _lookup("run1-AAPL-buy"), 404),
        })
        with pytest.raises(BrokerError, match="Reintentos agotados"):
            client.submit_orders(_one(), dry_run=False, run_id="run1")
        assert len(fake.posts()) == adapter._MAX_RETRIES

    def test_get_is_retried_automatically(self, monkeypatch, no_sleep):
        client, fake = _client(monkeypatch, {
            ("GET", CLOCK): Seq(_http_error(CLOCK, 503),
                                {"is_open": True, "timestamp": "t"}),
        })
        assert client.get_clock()["is_open"] is True
        assert len([c for c in fake.calls if c[1] == CLOCK]) == 2


class TestTruncatedBody:
    """Body cortado = fallo ambiguo: no sabemos que respondio el broker."""

    @pytest.mark.parametrize("cut", [_incomplete, _truncated_json],
                             ids=["incomplete_read", "truncated_json"])
    def test_get_is_retried(self, monkeypatch, no_sleep, cut):
        client, fake = _client(monkeypatch, {
            ("GET", CLOCK): Seq(cut(), {"is_open": True, "timestamp": "t"}),
        })
        assert client.get_clock()["is_open"] is True
        assert len([c for c in fake.calls if c[1] == CLOCK]) == 2

    @pytest.mark.parametrize("cut", [_incomplete, _truncated_json],
                             ids=["incomplete_read", "truncated_json"])
    def test_get_exhausted_is_transient(self, monkeypatch, no_sleep, cut):
        client, fake = _client(monkeypatch, {
            ("GET", CLOCK): Seq(*[cut() for _ in range(adapter._MAX_RETRIES)]),
        })
        with pytest.raises(BrokerTransientError, match="Reintentos agotados"):
            client.get_clock()
        assert len([c for c in fake.calls if c[1] == CLOCK]) == adapter._MAX_RETRIES

    @pytest.mark.parametrize("cut", [_incomplete, _truncated_json],
                             ids=["incomplete_read", "truncated_json"])
    def test_post_checks_client_order_id_before_resending(
            self, monkeypatch, no_sleep, cut):
        # El POST llego (Alpaca acepto) pero el body se corto: no reenviar.
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): Seq(cut()),
            ("GET", _lookup("run1-AAPL-buy")): {
                "id": "o1", "status": "accepted",
                "client_order_id": "run1-AAPL-buy"},
        })
        out = client.submit_orders(_one(), dry_run=False, run_id="run1")
        assert len(fake.posts()) == 1
        assert out[0]["status"] == "submitted" and out[0]["id"] == "o1"


class TestIncrementalResults:
    def _three(self):
        return pd.DataFrame([
            {"ticker": "AAPL", "side": "buy", "qty": 1.0},
            {"ticker": "MSFT", "side": "buy", "qty": 2.0},
            {"ticker": "NVDA", "side": "sell", "qty": 3.0},
        ])

    def test_each_result_reported_before_next_post(self, monkeypatch):
        client, fake = _client(monkeypatch, {
            ("POST", ORDERS): Seq({"id": "a", "status": "accepted"},
                                  {"id": "b", "status": "accepted"},
                                  {"id": "c", "status": "accepted"}),
        })
        seen = []
        client.submit_orders(
            self._three(), dry_run=False, run_id="run1",
            on_result=lambda r: seen.append((r["ticker"], len(fake.posts()))))
        assert seen == [("AAPL", 1), ("MSFT", 2), ("NVDA", 3)]

    def test_403_on_third_reports_two_submitted_plus_failure(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("POST", ORDERS): Seq({"id": "a", "status": "accepted"},
                                  {"id": "b", "status": "accepted"},
                                  _http_error(ORDERS, 403)),
        })
        seen = []
        with pytest.raises(BrokerError, match="HTTP 403"):
            client.submit_orders(self._three(), dry_run=False, run_id="run1",
                                 on_result=seen.append)
        assert [r["status"] for r in seen] == ["submitted", "submitted", "failed"]
        assert "HTTP 403" in seen[2]["error"]
        assert seen[2]["client_order_id"] == "run1-NVDA-sell"

    def test_invalid_row_sends_nothing(self, monkeypatch):
        client, fake = _client(monkeypatch, {})
        trades = pd.DataFrame([
            {"ticker": "AAPL", "side": "buy", "qty": 1.0},
            {"ticker": "MSFT", "side": "hold", "qty": 1.0},
        ])
        with pytest.raises(ValueError, match="nvalida"):
            client.submit_orders(trades, dry_run=False, run_id="run1")
        assert fake.posts() == []


class TestPollOrders:
    def test_polls_until_final_status(self, monkeypatch, no_sleep):
        client, _ = _client(monkeypatch, {
            ("GET", ORDERS + "/o1"): Seq(
                {"id": "o1", "status": "new", "filled_qty": "0",
                 "filled_avg_price": None},
                {"id": "o1", "status": "filled", "filled_qty": "2",
                 "filled_avg_price": "101.5"}),
        })
        out = client.poll_orders(["o1"], timeout=60)
        assert out["o1"] == {"final_status": "filled", "filled_qty": 2.0,
                             "filled_avg_price": 101.5}

    def test_timeout_returns_last_seen_status(self, monkeypatch, no_sleep):
        client, fake = _client(monkeypatch, {
            ("GET", ORDERS + "/o1"): {"id": "o1", "status": "accepted",
                                      "filled_qty": "0",
                                      "filled_avg_price": None},
        })
        out = client.poll_orders(["o1"], timeout=0)
        assert out["o1"] == {"final_status": "accepted", "filled_qty": 0.0,
                             "filled_avg_price": None}
        assert len([c for c in fake.calls if c[1] == ORDERS + "/o1"]) == 1


# ===================================================================
# run_broker_step: AlpacaClient real + urllib falso
# ===================================================================
def _step_routes(posts):
    return {
        ("GET", CLOCK): {"is_open": True, "timestamp": "t"},
        ("GET", POSITIONS): [],
        ("POST", ORDERS): Seq(*posts),
    }


def _patch_step(monkeypatch, routes):
    fake = FakeUrlopen({("GET", ACCOUNT): {"account_number": "PA1",
                                           "equity": "10000"}, **routes})
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr(adapter, "load_broker_config",
                        lambda *a, **k: BrokerConfig("PKTEST1234", "s", BASE))
    return fake


def _target():
    # Nocionales 5000/3000/2000 → orden del plan AAPL, MSFT, NVDA.
    return pd.Series({"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2})


def _prices():
    return pd.Series({"AAPL": 100.0, "MSFT": 50.0, "NVDA": 20.0})


class TestRunBrokerStepAudit:
    T = pd.Timestamp("2026-09-25")

    def test_403_on_third_order_still_writes_plan_csv(self, monkeypatch, tmp_path):
        from scripts.run_periodic_rebalance import run_broker_step
        _patch_step(monkeypatch, _step_routes([
            {"id": "a", "status": "accepted"},
            {"id": "b", "status": "accepted"},
            _http_error(ORDERS, 403)]))
        with pytest.raises(BrokerError, match="HTTP 403"):
            run_broker_step(_target(), _prices(), self.T, tmp_path, submit=True)

        saved = pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")
        assert list(saved["ticker"]) == ["AAPL", "MSFT", "NVDA"]
        assert list(saved["status"]) == ["submitted", "submitted", "failed"]
        assert list(saved["order_id"][:2]) == ["a", "b"]
        assert "HTTP 403" in saved.loc[2, "error"]
        assert saved.loc[0, "client_order_id"] == "hft-20260925-AAPL-buy"

    def test_journal_has_sent_orders_and_abort(self, monkeypatch, tmp_path):
        from scripts.run_periodic_rebalance import run_broker_step
        _patch_step(monkeypatch, _step_routes([
            {"id": "a", "status": "accepted"},
            {"id": "b", "status": "accepted"},
            _http_error(ORDERS, 403)]))
        with pytest.raises(BrokerError):
            run_broker_step(_target(), _prices(), self.T, tmp_path, submit=True)

        lines = (tmp_path / "broker_journal_2026-09-25.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        assert [r.get("status") for r in records[:3]] == [
            "submitted", "submitted", "failed"]
        assert records[-1]["event"] == "abort"
        assert "HTTP 403" in records[-1]["error"]

    def test_ctrl_c_mid_submit_still_audited(self, monkeypatch, tmp_path):
        from scripts.run_periodic_rebalance import run_broker_step
        _patch_step(monkeypatch, _step_routes([
            {"id": "a", "status": "accepted"}, KeyboardInterrupt()]))
        with pytest.raises(KeyboardInterrupt):
            run_broker_step(_target(), _prices(), self.T, tmp_path, submit=True)

        saved = pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")
        assert list(saved["status"]) == ["submitted", "not_sent", "not_sent"]
        last = (tmp_path / "broker_journal_2026-09-25.jsonl").read_text().splitlines()[-1]
        assert json.loads(last)["event"] == "abort"

    def test_fills_polled_into_plan_csv(self, monkeypatch, tmp_path, no_sleep):
        from scripts.run_periodic_rebalance import run_broker_step
        routes = _step_routes([{"id": "a", "status": "accepted"},
                               {"id": "b", "status": "accepted"},
                               {"id": "c", "status": "accepted"}])
        for oid, qty, px in [("a", "50", "100.1"), ("b", "60", "50.2"),
                             ("c", "100", "20.3")]:
            routes[("GET", f"{ORDERS}/{oid}")] = {
                "id": oid, "status": "filled", "filled_qty": qty,
                "filled_avg_price": px}
        _patch_step(monkeypatch, routes)
        run_broker_step(_target(), _prices(), self.T, tmp_path, submit=True,
                        poll_timeout=5)

        saved = pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")
        assert {"filled_qty", "filled_avg_price", "final_status"}.issubset(
            saved.columns)
        assert list(saved["final_status"]) == ["filled"] * 3
        assert saved.set_index("ticker").loc["MSFT", "filled_avg_price"] == 50.2

    def test_dry_run_writes_no_journal(self, monkeypatch, tmp_path):
        from scripts.run_periodic_rebalance import run_broker_step
        fake = _patch_step(monkeypatch, _step_routes([]))
        run_broker_step(_target(), _prices(), self.T, tmp_path, submit=False)
        assert fake.posts() == []
        assert not (tmp_path / "broker_journal_2026-09-25.jsonl").exists()
        saved = pd.read_csv(tmp_path / "broker_plan_2026-09-25.csv")
        assert (saved["status"] == "dry_run").all()
