"""
test_alpaca_adapter.py — Paso 7: broker paper con HTTP mockeado (sin red).
"""

import io
import json
import urllib.error

import pandas as pd
import pytest

from src.brokers import alpaca_adapter
from src.brokers.alpaca_adapter import (
    AlpacaClient,
    BrokerConfig,
    BrokerError,
    load_broker_config,
    reconcile_trades,
)
from src.instruments import load_instruments

PAPER = {
    "BROKER_API_KEY": "PKTEST1234",
    "BROKER_SECRET_KEY": "secreto-falso",
    "BROKER_BASE_URL": "https://paper-api.alpaca.markets",
}
SECRET = "secreto-falso"


class FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _fake_urlopen_factory(calls, routes, bodies=None):
    def _fake(req, timeout=None):
        calls.append((req.full_url, req.get_method()))
        if bodies is not None and req.data:
            bodies.append(json.loads(req.data))
        key = (req.get_method(), req.full_url)
        if key not in routes:
            raise AssertionError(f"HTTP inesperado: {key}")
        payload = routes[key]
        if isinstance(payload, Exception):
            raise payload
        return FakeResp(payload)
    return _fake


def _client(monkeypatch, routes, bodies=None, instruments=None):
    calls = []
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen_factory(calls, routes, bodies))
    return AlpacaClient(BrokerConfig(PAPER["BROKER_API_KEY"], SECRET,
                                     PAPER["BROKER_BASE_URL"]),
                        instruments=instruments), calls


SUBMIT_ROUTES = {
    ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
        "account_number": "", "equity": "1"},
    ("POST", PAPER["BROKER_BASE_URL"] + "/v2/orders"): {
        "id": "abc", "status": "accepted"},
}


class TestConfig:
    def test_paper_ok(self):
        cfg = load_broker_config(env=dict(PAPER), dotenv_path="/no/existe.env")
        assert cfg.base_url == PAPER["BROKER_BASE_URL"]

    def test_missing_keys_raise(self):
        with pytest.raises(ValueError, match="Faltan"):
            load_broker_config(env={}, dotenv_path="/no/existe.env")

    def test_base_url_v2_suffix_normalized(self):
        cfg = load_broker_config(
            env={**PAPER, "BROKER_BASE_URL": "https://paper-api.alpaca.markets/v2/"},
            dotenv_path="/no/existe.env")
        assert cfg.base_url == "https://paper-api.alpaca.markets"

    def test_live_url_rejected_without_http(self, monkeypatch):
        calls = []
        monkeypatch.setattr("urllib.request.urlopen",
                            _fake_urlopen_factory(calls, {}))
        with pytest.raises(ValueError, match="solo-paper"):
            load_broker_config(env={**PAPER,
                                    "BROKER_BASE_URL": "https://api.alpaca.markets"})
        assert calls == []

    def test_secret_never_in_repr(self):
        assert SECRET not in repr(
            BrokerConfig("PKTEST1234", SECRET, PAPER["BROKER_BASE_URL"]))

    def test_expected_account_id_loaded(self):
        cfg = load_broker_config(
            env={**PAPER, "BROKER_ACCOUNT_ID": "PA0000000001"},
            dotenv_path="/no/existe.env")
        assert cfg.account_id == "PA0000000001"


class TestAccountGuard:
    def _guard_client(self, monkeypatch, actual_number):
        calls = []
        monkeypatch.setattr("urllib.request.urlopen",
                            _fake_urlopen_factory(calls, {
                                ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
                                    "account_number": actual_number,
                                    "equity": "100000", "buying_power": "200000",
                                    "account_blocked": False, "trading_blocked": False},
                                ("GET", PAPER["BROKER_BASE_URL"] + "/v2/clock"): {
                                    "is_open": True, "timestamp": "t"},
                                ("POST", PAPER["BROKER_BASE_URL"] + "/v2/orders"): {
                                    "id": "abc", "status": "accepted"},
                            }))
        cfg = BrokerConfig(PAPER["BROKER_API_KEY"], SECRET,
                           PAPER["BROKER_BASE_URL"], account_id="PA0000000001")
        return AlpacaClient(cfg), calls

    def test_matching_account_ok(self, monkeypatch):
        client, _ = self._guard_client(monkeypatch, "PA0000000001")
        assert client.check()["account_number"] == "PA0000000001"

    def test_mismatch_aborts_check(self, monkeypatch):
        client, _ = self._guard_client(monkeypatch, "OTRACUENTA")
        with pytest.raises(BrokerError, match="Cuenta equivocada.*PA0000000001"):
            client.check()

    def test_mismatch_blocks_orders_before_post(self, monkeypatch):
        client, calls = self._guard_client(monkeypatch, "OTRACUENTA")
        trades = pd.DataFrame([{"ticker": "AAPL", "side": "buy", "qty": 1.0}])
        with pytest.raises(BrokerError, match="Nada fue enviado"):
            client.submit_orders(trades, dry_run=False)
        posts = [c for c in calls if c[0] == "POST"]
        assert posts == [] and client.sent_orders == 0


class TestReads:
    def test_check(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
                "equity": "100000", "buying_power": "200000",
                "account_blocked": False, "trading_blocked": False},
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/clock"): {
                "is_open": True, "timestamp": "2026-01-01T00:00:00Z"},
        })
        out = client.check()
        assert out["equity"] == "100000" and out["market_open"] is True

    def test_positions(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/positions"): [
                {"symbol": "AAPL", "qty": "10", "market_value": "2000",
                 "avg_entry_price": "190", "side": "long"}],
        })
        df = client.get_positions()
        assert df.loc[0, "ticker"] == "AAPL" and df.loc[0, "qty"] == 10.0

    def test_http_error_hides_secret(self, monkeypatch):
        err = urllib.error.HTTPError(
            PAPER["BROKER_BASE_URL"] + "/v2/account", 401, "Unauthorized",
            {}, io.BytesIO(b'{"message":"forbidden"}'))
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): err})
        with pytest.raises(BrokerError) as excinfo:
            client.get_account()
        assert SECRET not in str(excinfo.value)
        assert "401" in str(excinfo.value)


class TestOrders:
    def test_dry_run_makes_no_http(self, monkeypatch):
        client, calls = _client(monkeypatch, {})
        trades = pd.DataFrame([{"ticker": "AAPL", "side": "buy", "qty": 1.5}])
        out = client.submit_orders(trades, dry_run=True)
        assert calls == [] and out[0]["status"] == "dry_run"

    def test_submit_posts_market_day(self, monkeypatch):
        client, calls = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
                "account_number": "", "equity": "1"},
            ("POST", PAPER["BROKER_BASE_URL"] + "/v2/orders"): {
                "id": "abc", "status": "accepted"},
        })
        trades = pd.DataFrame([{"ticker": "AAPL", "side": "sell", "qty": 2.0}])
        out = client.submit_orders(trades, dry_run=False)
        assert out[0]["id"] == "abc" and client.sent_orders == 1
        posts = [c for c in calls if c[1] == "POST"]
        assert len(posts) == 1 and calls[0][1] == "GET"  # verifica antes de enviar

    def test_invalid_orders_raise(self, monkeypatch):
        client, _ = _client(monkeypatch, {})
        with pytest.raises(ValueError, match="columnas"):
            client.submit_orders(pd.DataFrame([{"ticker": "A"}]))
        with pytest.raises(ValueError, match="nvalida"):
            client.submit_orders(
                pd.DataFrame([{"ticker": "A", "side": "hold", "qty": 1.0}]))


class TestSymbolMapping:
    def test_alias_table_replaced_by_master(self):
        assert not hasattr(alpaca_adapter, "ALPACA_ALIASES")

    def test_orders_use_master_broker_symbol(self, monkeypatch):
        bodies = []
        client, _ = _client(monkeypatch, SUBMIT_ROUTES, bodies)
        trades = pd.DataFrame([{"ticker": "BRK-B", "side": "buy", "qty": 1.0},
                               {"ticker": "AAPL", "side": "buy", "qty": 1.0}])
        out = client.submit_orders(trades, dry_run=False, run_id="r1")
        assert [b["symbol"] for b in bodies] == ["BRK.B", "AAPL"]
        # El plan sigue en ids del sistema.
        assert [r["ticker"] for r in out] == ["BRK-B", "AAPL"]
        assert out[0]["client_order_id"] == "r1-BRK-B-buy"

    def test_injected_master_is_used(self, monkeypatch, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text("ticker,yahoo_ticker,broker_symbol,market,currency,"
                        "tradable_brokers,asset_type,sector\n"
                        "XYZ,XYZ-U,XYZ.U,us,USD,alpaca,stock,Tech\n")
        bodies = []
        client, _ = _client(monkeypatch, SUBMIT_ROUTES, bodies,
                            instruments=load_instruments(path))
        client.submit_orders(
            pd.DataFrame([{"ticker": "XYZ", "side": "buy", "qty": 1.0}]),
            dry_run=False)
        assert bodies[0]["symbol"] == "XYZ.U"

    def test_positions_normalized(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/positions"): [
                {"symbol": "BRK.B", "qty": "1", "market_value": "400",
                 "avg_entry_price": "390", "side": "long"}],
        })
        df = client.get_positions()
        assert df.loc[0, "ticker"] == "BRK-B"


class TestRouting:
    """Instrumentos sin "alpaca" en tradable_brokers: nunca se envian."""

    def _trades(self, *tickers):
        return pd.DataFrame([{"ticker": tk, "side": "buy", "qty": 1.0}
                             for tk in tickers])

    def test_co_local_not_routable_in_dry_run_plan(self, monkeypatch):
        client, calls = _client(monkeypatch, {})
        out = client.submit_orders(self._trades("ICOLCAP", "AAPL"), dry_run=True)
        assert [(r["ticker"], r["status"]) for r in out] == [
            ("ICOLCAP", "not_routable"), ("AAPL", "dry_run")]
        assert "alpaca" in out[0]["error"]
        assert out[0]["id"] is None and calls == []

    def test_co_local_never_posted_on_submit(self, monkeypatch):
        bodies = []
        client, calls = _client(monkeypatch, SUBMIT_ROUTES, bodies)
        out = client.submit_orders(self._trades("ICOLCAP", "AAPL"),
                                   dry_run=False)
        assert [r["status"] for r in out] == ["not_routable", "submitted"]
        assert [b["symbol"] for b in bodies] == ["AAPL"]
        assert client.sent_orders == 1

    def test_only_non_routable_orders_make_no_http(self, monkeypatch):
        client, calls = _client(monkeypatch, {})
        out = client.submit_orders(self._trades("ICOLCAP", "ECOPETROL"),
                                   dry_run=False)
        assert {r["status"] for r in out} == {"not_routable"}
        assert calls == [] and client.sent_orders == 0

    def test_unknown_instrument_not_routable(self, monkeypatch):
        client, calls = _client(monkeypatch, {})
        out = client.submit_orders(self._trades("ZZZZ"), dry_run=False)
        assert out[0]["status"] == "not_routable"
        assert "instrument master" in out[0]["error"]
        assert calls == []


class TestOrderResilience:
    def _http_error(self, code):
        return urllib.error.HTTPError(
            PAPER["BROKER_BASE_URL"] + "/v2/orders", code, "err",
            {}, io.BytesIO(b'{"message":"x"}'))

    def test_422_rejected_and_continues(self, monkeypatch):
        base = PAPER["BROKER_BASE_URL"]
        lookup = base + "/v2/orders:by_client_order_id?client_order_id=r1-{}-buy"
        not_found = urllib.error.HTTPError(base, 404, "err", {},
                                           io.BytesIO(b'{"message":"x"}'))
        client, calls = _client(monkeypatch, {
            ("GET", base + "/v2/account"): {
                "account_number": "", "equity": "1"},
            ("POST", base + "/v2/orders"): self._http_error(422),
            # 422 sin orden previa con ese client_order_id = rechazo real
            ("GET", lookup.format("BRK-B")): not_found,
            ("GET", lookup.format("AAPL")): not_found,
        })
        trades = pd.DataFrame([
            {"ticker": "BRK-B", "side": "buy", "qty": 1.0},
            {"ticker": "AAPL", "side": "buy", "qty": 1.0},
        ])
        out = client.submit_orders(trades, dry_run=False, run_id="r1")
        assert [r["status"] for r in out] == ["rejected", "rejected"]
        assert client.sent_orders == 0
        assert len([c for c in calls if c[1] == "POST"]) == 2

    def test_401_aborts(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
                "account_number": "", "equity": "1"},
            ("POST", PAPER["BROKER_BASE_URL"] + "/v2/orders"):
                self._http_error(401),
        })
        trades = pd.DataFrame([{"ticker": "AAPL", "side": "buy", "qty": 1.0}])
        with pytest.raises(BrokerError, match="HTTP 401"):
            client.submit_orders(trades, dry_run=False)


class TestClosePositions:
    def _pos_client(self, monkeypatch):
        return _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/account"): {
                "account_number": "", "equity": "1"},
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/positions"): [
                {"symbol": "AAPL", "qty": "2", "market_value": "400",
                 "avg_entry_price": "190", "side": "long"},
                {"symbol": "BRK.B", "qty": "1", "market_value": "500",
                 "avg_entry_price": "490", "side": "long"}],
            ("POST", PAPER["BROKER_BASE_URL"] + "/v2/orders"): {
                "id": "x", "status": "accepted"},
        })

    def test_close_all_dry_run_no_post(self, monkeypatch):
        client, calls = self._pos_client(monkeypatch)
        out = client.close_positions(dry_run=True)
        assert [(r["ticker"], r["side"]) for r in out] == [
            ("AAPL", "sell"), ("BRK-B", "sell")]
        assert all(r["status"] == "dry_run" for r in out)
        assert [c for c in calls if c[1] == "POST"] == []

    def test_close_subset(self, monkeypatch):
        client, _ = self._pos_client(monkeypatch)
        out = client.close_positions(["AAPL"], dry_run=True)
        assert [r["ticker"] for r in out] == ["AAPL"]

    def test_close_all_confirmed_posts(self, monkeypatch):
        client, calls = self._pos_client(monkeypatch)
        out = client.close_positions(dry_run=False)
        assert all(r["status"] == "submitted" for r in out)
        assert all(r["broker_status"] == "accepted" for r in out)
        assert len([c for c in calls if c[1] == "POST"]) == 2

    def test_close_empty(self, monkeypatch):
        client, _ = _client(monkeypatch, {
            ("GET", PAPER["BROKER_BASE_URL"] + "/v2/positions"): []})
        assert client.close_positions() == []


class TestReconcile:
    def test_basic_math(self):
        target = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
        held = pd.Series({"AAPL": 5.0, "MSFT": 0.0})
        prices = pd.Series({"AAPL": 100.0, "MSFT": 50.0})
        df = reconcile_trades(target, held, prices, portfolio_value=1000.0)
        # AAPL: 6.0 - 5.0 = buy 1.0 ($100); MSFT: 8.0 - 0 = buy 8.0 ($400)
        assert set(df["ticker"]) == {"AAPL", "MSFT"}
        assert (df["side"] == "buy").all()
        assert df.set_index("ticker").loc["MSFT", "qty"] == 8.0

    def test_sell_and_dust_skipped(self):
        target = pd.Series({"AAPL": 0.0, "MSFT": 1.0})
        held = pd.Series({"AAPL": 10.0, "MSFT": 19.99})
        prices = pd.Series({"AAPL": 100.0, "MSFT": 50.0})
        df = reconcile_trades(target, held, prices, portfolio_value=1000.0,
                              min_notional=1.0)
        assert df.set_index("ticker").loc["AAPL", "side"] == "sell"
        assert "MSFT" not in set(df["ticker"])  # diff $0.50 < $1 polvo

    def test_missing_price_skipped(self):
        target = pd.Series({"AAPL": 1.0})
        df = reconcile_trades(target, pd.Series(dtype=float),
                              pd.Series(dtype=float), 1000.0)
        assert df.empty


def _calendar_url(start, end):
    return (PAPER["BROKER_BASE_URL"]
            + f"/v2/calendar?start={start}&end={end}&date_type=TRADING")


class TestCalendar:
    def test_get_calendar_pide_date_open_close(self, monkeypatch):
        url = _calendar_url("2024-07-03", "2024-07-03")
        payload = [{
            "date": "2024-07-03", "open": "09:30", "close": "13:00",
            "session_open": "0700", "session_close": "1700",
            "settlement_date": "2024-07-05",
        }]
        client, calls = _client(monkeypatch, {("GET", url): payload})
        assert client.get_calendar("2024-07-03", "2024-07-03") == payload
        assert calls == [(url, "GET")]
        assert SECRET not in url

    def test_rango_invertido_no_hace_http(self, monkeypatch):
        client, calls = _client(monkeypatch, {})
        with pytest.raises(ValueError, match="posterior"):
            client.get_calendar("2024-03-12", "2024-03-11")
        assert calls == []

    def test_respuesta_que_no_es_lista(self, monkeypatch):
        url = _calendar_url("2024-03-11", "2024-03-11")
        client, _ = _client(monkeypatch, {("GET", url): {"calendar": []}})
        with pytest.raises(BrokerError, match="lista"):
            client.get_calendar("2024-03-11", "2024-03-11")

    def test_get_reintenta_429(self, monkeypatch):
        monkeypatch.setattr(alpaca_adapter.time, "sleep", lambda _s: None)
        url = _calendar_url("2024-03-11", "2024-03-11")
        payload = [{"date": "2024-03-11", "open": "09:30", "close": "16:00"}]
        calls = []
        err = urllib.error.HTTPError(url, 429, "slow", {}, io.BytesIO(b"{}"))

        def _fake(req, timeout=None):
            calls.append(req.get_method())
            if len(calls) == 1:
                raise err
            return FakeResp(payload)

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        client = AlpacaClient(BrokerConfig(
            PAPER["BROKER_API_KEY"], SECRET, PAPER["BROKER_BASE_URL"]))
        assert client.get_calendar("2024-03-11", "2024-03-11") == payload
        assert calls == ["GET", "GET"]
