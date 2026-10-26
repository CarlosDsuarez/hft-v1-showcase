"""Exchange falso de la Trading API de Alpaca, en loopback.

Sirve al bench de capacidad (``scripts/hft_capacity_bench.py``), al probe
en modo motor y a los tests. Nunca a una sesión real: escucha solo en
127.0.0.1 y el transporte del cliente acepta loopback o paper-api.

Imita lo que el cliente usa:

- ``POST /v2/orders`` (422 si el ``client_order_id`` se repite),
  ``GET /v2/orders`` (status, after, limit, nested), ``GET /v2/orders/{id}``,
  ``GET /v2/orders:by_client_order_id``, ``DELETE /v2/orders/{id}`` y
  ``DELETE /v2/orders``.
- ``GET /v2/account``, ``/v2/clock``, ``/v2/positions``, ``/v2/assets/{sym}``
  y ``/v2/account/activities`` (FILL, con ``page_token`` ascendente).
- ``latency_s`` de espera por petición, como el viaje de ida y vuelta.
- ``rpm_limit``: ventana fija de 60 s con 429 y headers
  ``X-RateLimit-Limit/Remaining/Reset`` (epoch). None: sin tope.
- Stream ``trade_updates`` por websocket (``websockets.sync.server``),
  con auth y listen como Alpaca y frames binarios como paper.

``fill_mode``: ``immediate`` llena market y limit al precio del símbolo
(``prices``, 100.0 si falta); ``never`` deja todo vivo (probe sin fills).
``reject_symbols``: el POST de esos símbolos devuelve 403 (prueba de rollback).
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

UTC = timezone.utc
_FINAL = frozenset({"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"})


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class FakeStats:
    requests: int = 0
    posts: int = 0
    deletes: int = 0
    gets: int = 0
    throttled: int = 0
    fills: int = 0
    by_second: Dict[int, int] = field(default_factory=dict)
    by_path: Dict[str, int] = field(default_factory=dict)


class _State:
    """Libro del exchange. Un lock: el servidor atiende en hilos."""

    def __init__(self, *, fill_mode: str, prices: Mapping[str, float],
                 account_number: str, rpm_limit: Optional[int],
                 reject_symbols: Iterable[str] = ()) -> None:
        if fill_mode not in ("immediate", "never"):
            raise ValueError("fill_mode debe ser immediate o never")
        self.lock = threading.Lock()
        self.fill_mode = fill_mode
        self.prices = {str(k).upper(): float(v) for k, v in prices.items()}
        self.account_number = account_number
        self.rpm_limit = rpm_limit
        self.reject_symbols = frozenset(str(s).upper() for s in reject_symbols)
        self.orders: Dict[str, dict] = {}
        self.by_client: Dict[str, str] = {}
        self.activities: List[dict] = []
        self.positions: Dict[str, float] = {}
        self.window_start = 0.0
        self.window_count = 0
        self.stats = FakeStats()
        self._ids = itertools.count(1)
        self.listeners: List = []

    # --- Tope ---
    def rate(self) -> Tuple[bool, Dict[str, str]]:
        """(permitido, headers). Ventana fija de 60 s, como el reset de Alpaca."""
        now = time.time()
        with self.lock:
            self.stats.requests += 1
            second = int(now)
            self.stats.by_second[second] = self.stats.by_second.get(second, 0) + 1
            if self.rpm_limit is None:
                return True, {}
            if now - self.window_start >= 60.0:
                self.window_start = now
                self.window_count = 0
            reset = int(self.window_start + 60.0)
            if self.window_count >= self.rpm_limit:
                self.stats.throttled += 1
                return False, {
                    "X-RateLimit-Limit": str(self.rpm_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                }
            self.window_count += 1
            return True, {
                "X-RateLimit-Limit": str(self.rpm_limit),
                "X-RateLimit-Remaining": str(self.rpm_limit - self.window_count),
                "X-RateLimit-Reset": str(reset),
            }

    # --- Órdenes ---
    def create(self, body: Mapping) -> Tuple[int, object]:
        cid = str(body.get("client_order_id") or "")
        symbol = str(body.get("symbol") or "").upper()
        side = str(body.get("side") or "")
        try:
            qty = float(body.get("qty"))
        except (TypeError, ValueError):
            return 422, {"code": 40010001, "message": "qty invalida"}
        if symbol in self.reject_symbols:
            return 403, {"code": 40310000, "message": "forbidden (reject_symbols)"}
        events = []
        with self.lock:
            if not cid:
                cid = f"auto-{next(self._ids)}"
            if cid in self.by_client:
                return 422, {"code": 40010001, "message": "client_order_id must be unique"}
            order_id = f"fake-{next(self._ids):08d}"
            stamp = _now_iso()
            order = {
                "id": order_id,
                "client_order_id": cid,
                "symbol": symbol,
                "side": side,
                "qty": body.get("qty"),
                "type": body.get("type"),
                "time_in_force": body.get("time_in_force"),
                "order_class": body.get("order_class") or "simple",
                "limit_price": body.get("limit_price"),
                "stop_price": body.get("stop_price"),
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
                "submitted_at": stamp,
                "updated_at": stamp,
                "filled_at": None,
                "legs": None,
            }
            self.orders[order_id] = order
            self.by_client[cid] = order_id
            self.stats.posts += 1
            events.append(("new", dict(order), None, None))
            if self.fill_mode == "immediate" and order["type"] in ("market", "limit"):
                price = self.prices.get(symbol, 100.0)
                filled_at = _now_iso()
                order.update({
                    "status": "filled",
                    "filled_qty": _fmt(qty),
                    "filled_avg_price": _fmt(price),
                    "filled_at": filled_at,
                    "updated_at": filled_at,
                })
                signed = qty if side == "buy" else -qty
                self.positions[symbol] = self.positions.get(symbol, 0.0) + signed
                self.activities.append({
                    "id": f"act-{len(self.activities) + 1:010d}",
                    "activity_type": "FILL",
                    "type": "fill",
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": _fmt(qty),
                    "price": _fmt(price),
                    "order_status": "filled",
                    "transaction_time": filled_at,
                })
                self.stats.fills += 1
                events.append(("fill", dict(order), price, qty))
            snapshot = dict(order)
        for event in events:
            self.broadcast(*event)
        return 200, snapshot

    def cancel(self, order_id: str) -> Tuple[int, object]:
        with self.lock:
            self.stats.deletes += 1
            order = self.orders.get(order_id)
            if order is None:
                return 404, {"message": "order not found"}
            if order["status"] in _FINAL:
                return 422, {"message": f"order is already in {order['status']} state"}
            order["status"] = "canceled"
            order["updated_at"] = _now_iso()
            snapshot = dict(order)
        self.broadcast("canceled", snapshot, None, None)
        return 204, None

    def cancel_all(self) -> Tuple[int, object]:
        with self.lock:
            open_ids = [oid for oid, o in self.orders.items() if o["status"] not in _FINAL]
        out = []
        for oid in open_ids:
            status, _ = self.cancel(oid)
            out.append({"id": oid, "status": 200 if status == 204 else status})
        return 207, out

    def list_orders(self, query: Mapping[str, List[str]]) -> List[dict]:
        status = (query.get("status") or ["open"])[0]
        limit = int((query.get("limit") or ["50"])[0])
        after = (query.get("after") or [None])[0]
        with self.lock:
            rows = [dict(o) for o in self.orders.values()]
        if status == "open":
            rows = [o for o in rows if o["status"] not in _FINAL]
        elif status == "closed":
            rows = [o for o in rows if o["status"] in _FINAL]
        if after:
            rows = [o for o in rows if o["submitted_at"] > after]
        rows.sort(key=lambda o: o["submitted_at"],
                  reverse=(query.get("direction") or ["desc"])[0] != "asc")
        return rows[:limit]

    def list_activities(self, query: Mapping[str, List[str]]) -> List[dict]:
        size = int((query.get("page_size") or ["100"])[0])
        token = (query.get("page_token") or [None])[0]
        with self.lock:
            rows = list(self.activities)
        if token:
            rows = [a for a in rows if a["id"] > token]
        return rows[:size]

    # --- Stream ---
    def add_listener(self, ws) -> None:
        with self.lock:
            self.listeners.append(ws)

    def drop_listener(self, ws) -> None:
        with self.lock:
            if ws in self.listeners:
                self.listeners.remove(ws)

    def broadcast(self, event: str, order: dict, price: Optional[float],
                  qty: Optional[float]) -> None:
        with self.lock:
            listeners = list(self.listeners)
        if not listeners:
            return
        data = {"event": event, "order": order, "timestamp": _now_iso()}
        if price is not None:
            data["price"] = _fmt(price)
            data["qty"] = _fmt(qty)
            data["execution_id"] = f"exec-{order['id']}-{order['filled_qty']}"
        frame = json.dumps({"stream": "trade_updates", "data": data}).encode()
        for ws in listeners:
            try:
                ws.send(frame)
            except Exception:
                self.drop_listener(ws)


def _fmt(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


class _Server(ThreadingHTTPServer):
    # El default (5) resetea conexiones con muchos hilos cliente a la vez.
    request_queue_size = 256
    daemon_threads = True


def _handler(state: _State, latency_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # silencio
            return None

        def _reply(self, status: int, payload: object, extra: Mapping[str, str]) -> None:
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if latency_s > 0:
                time.sleep(latency_s)
            allowed, limit_headers = state.rate()
            if not allowed:
                self._reply(429, {"message": "rate limit exceeded"}, limit_headers)
                return
            if self.headers.get("APCA-API-KEY-ID") is None:
                self._reply(401, {"message": "unauthorized"}, limit_headers)
                return
            parsed = urlparse(self.path)
            status, payload = self._route(method, parsed.path, parse_qs(parsed.query), raw)
            self._reply(status, payload, limit_headers)

        def _route(self, method: str, path: str, query, raw: bytes) -> Tuple[int, object]:
            key = f"{method} {path}"
            with state.lock:
                state.stats.by_path[key] = state.stats.by_path.get(key, 0) + 1
                if method == "GET":
                    state.stats.gets += 1
            if method == "POST" and path == "/v2/orders":
                try:
                    body = json.loads(raw.decode() or "{}")
                except ValueError:
                    return 400, {"message": "json invalido"}
                return state.create(body)
            if method == "DELETE" and path == "/v2/orders":
                return state.cancel_all()
            if method == "DELETE" and path.startswith("/v2/orders/"):
                return state.cancel(unquote(path[len("/v2/orders/"):]))
            if method != "GET":
                return 405, {"message": "metodo no soportado"}
            if path == "/v2/account":
                return 200, {"account_number": state.account_number, "equity": "1000000",
                             "buying_power": "4000000", "account_blocked": False,
                             "trading_blocked": False, "status": "ACTIVE"}
            if path == "/v2/clock":
                return 200, {"timestamp": _now_iso(), "is_open": True}
            if path == "/v2/positions":
                with state.lock:
                    rows = [{"symbol": sym, "qty": _fmt(qty), "market_value": "0",
                             "avg_entry_price": "0", "side": "long" if qty > 0 else "short"}
                            for sym, qty in state.positions.items() if abs(qty) > 1e-9]
                return 200, rows
            if path.startswith("/v2/assets/"):
                symbol = unquote(path[len("/v2/assets/"):]).upper()
                return 200, {"id": f"asset-{symbol}", "class": "us_equity", "symbol": symbol,
                             "status": "active", "tradable": True, "marginable": True,
                             "shortable": True, "easy_to_borrow": True, "fractionable": True}
            if path == "/v2/orders":
                return 200, state.list_orders(query)
            if path == "/v2/orders:by_client_order_id":
                cid = (query.get("client_order_id") or [""])[0]
                with state.lock:
                    order_id = state.by_client.get(cid)
                    order = dict(state.orders[order_id]) if order_id else None
                return (200, order) if order else (404, {"message": "not found"})
            if path.startswith("/v2/orders/"):
                with state.lock:
                    order = state.orders.get(unquote(path[len("/v2/orders/"):]))
                    snapshot = dict(order) if order else None
                return (200, snapshot) if snapshot else (404, {"message": "not found"})
            if path == "/v2/account/activities":
                return 200, state.list_activities(query)
            return 404, {"message": f"ruta desconocida {path}"}

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

    return Handler


class FakeAlpaca:
    """Servidor HTTP (y stream opcional) en loopback. Usar como context manager."""

    def __init__(
        self,
        *,
        latency_s: float = 0.0,
        rpm_limit: Optional[int] = None,
        fill_mode: str = "immediate",
        prices: Optional[Mapping[str, float]] = None,
        account_number: str = "",
        stream: bool = False,
        api_key: str = "PKFAKE",
        secret_key: str = "fake-secret",
        reject_symbols: Iterable[str] = (),
    ) -> None:
        self.latency_s = float(latency_s)
        self.state = _State(fill_mode=fill_mode, prices=prices or {},
                            account_number=account_number, rpm_limit=rpm_limit,
                            reject_symbols=reject_symbols)
        self.api_key = api_key
        self.secret_key = secret_key
        self._want_stream = bool(stream)
        self._http: Optional[ThreadingHTTPServer] = None
        self._ws_server = None
        self._threads: List[threading.Thread] = []
        self.base_url = ""
        self.stream_url = ""

    @property
    def stats(self) -> FakeStats:
        return self.state.stats

    def start(self) -> "FakeAlpaca":
        server = _Server(("127.0.0.1", 0), _handler(self.state, self.latency_s))
        self._http = server
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        thread = threading.Thread(target=server.serve_forever, name="fake-alpaca-http",
                                  daemon=True)
        thread.start()
        self._threads.append(thread)
        if self._want_stream:
            self._start_stream()
        return self

    def _start_stream(self) -> None:
        from websockets.sync.server import serve

        state = self.state
        key, secret = self.api_key, self.secret_key

        def handle(ws) -> None:
            try:
                auth = json.loads(_text(ws.recv()))
                ok = (auth.get("action") == "auth" and auth.get("key") == key
                      and auth.get("secret") == secret)
                status = "authorized" if ok else "unauthorized"
                ws.send(json.dumps({"stream": "authorization",
                                    "data": {"status": status, "action": "authenticate"}}).encode())
                if not ok:
                    return
                listen = json.loads(_text(ws.recv()))
                streams = (listen.get("data") or {}).get("streams") or []
                ws.send(json.dumps({"stream": "listening",
                                    "data": {"streams": list(streams)}}).encode())
                if "trade_updates" in streams:
                    state.add_listener(ws)
                for _ in ws:
                    pass
            except Exception:
                pass
            finally:
                state.drop_listener(ws)

        server = serve(handle, "127.0.0.1", 0)
        self._ws_server = server
        port = server.socket.getsockname()[1]
        self.stream_url = f"ws://127.0.0.1:{port}/stream"
        thread = threading.Thread(target=server.serve_forever, name="fake-alpaca-ws",
                                  daemon=True)
        thread.start()
        self._threads.append(thread)

    def drop_stream_clients(self) -> None:
        """Corta los websockets abiertos (prueba de reconexión)."""
        with self.state.lock:
            listeners = list(self.state.listeners)
            self.state.listeners.clear()
        for ws in listeners:
            try:
                ws.close()
            except Exception:
                pass

    def stop(self) -> None:
        if self._ws_server is not None:
            self.drop_stream_clients()
            self._ws_server.shutdown()
            self._ws_server = None
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []

    def __enter__(self) -> "FakeAlpaca":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _text(message: object) -> str:
    if isinstance(message, bytes):
        return message.decode("utf-8")
    return str(message)
