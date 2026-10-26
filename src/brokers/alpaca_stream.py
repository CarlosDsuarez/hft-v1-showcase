"""Stream ``trade_updates`` de la Trading API de Alpaca (websocket).

Los fills llegan en milisegundos y no gastan el cupo de 200 req/min de
la Trading API, a diferencia del polling de ``/v2/account/activities`` y
``/v2/orders``. Revoca D-07 (2026-10-13): la dependencia ``websockets``
está aprobada.

Protocolo (docs de Alpaca, "Websocket Streaming"):

1. ``{"action": "auth", "key": ..., "secret": ...}`` y respuesta
   ``{"stream": "authorization", "data": {"status": "authorized"}}``.
2. ``{"action": "listen", "data": {"streams": ["trade_updates"]}}`` y
   respuesta ``{"stream": "listening", ...}``.
3. Mensajes ``{"stream": "trade_updates", "data": {"event", "order",
   "timestamp", "price", "qty", ...}}``. Paper los manda como frames
   binarios: se decodifican igual.

Un hilo recibe y deja cada ``data`` en una cola. ``drain()`` la vacía
sin bloquear. Una caída reconecta con backoff y marca
``take_reconnected()``: el llamador reconcilia por REST, porque lo que
pasó mientras estuvo caído no se reenvía. Un "unauthorized" no se
reintenta: las claves están mal y reintentar no lo arregla.

El host se valida: wss://paper-api.alpaca.markets/stream o un loopback
del exchange falso.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Callable, List, Optional
from urllib.parse import urlparse

from src.brokers.http_transport import LOOPBACK_HOSTS, PAPER_HOST

logger = logging.getLogger("alpaca_stream")

PAPER_STREAM_URL = f"wss://{PAPER_HOST}/stream"
_RECV_TIMEOUT_S = 1.0
_OPEN_TIMEOUT_S = 10.0
_MAX_BACKOFF_S = 30.0


class StreamAuthError(RuntimeError):
    """Alpaca rechazó las claves del stream."""


def validate_stream_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "wss" and host == PAPER_HOST and parsed.port in (None, 443):
        return
    if parsed.scheme in ("ws", "wss") and host in LOOPBACK_HOSTS:
        return
    raise ValueError(f"stream rechazado: {url}. Solo {PAPER_STREAM_URL} o un loopback.")


def _text(message: object) -> str:
    if isinstance(message, (bytes, bytearray)):
        return bytes(message).decode("utf-8")
    return str(message)


class TradeUpdatesStream:
    """Cliente del stream en un hilo propio. ``start()`` / ``stop()``."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        url: str = PAPER_STREAM_URL,
        connect: Optional[Callable] = None,
        max_backoff_s: float = _MAX_BACKOFF_S,
    ) -> None:
        validate_stream_url(url)
        if not api_key or not secret_key:
            raise ValueError("faltan claves para el stream")
        self.url = url
        self._key = api_key
        self._secret = secret_key
        if connect is None:
            from websockets.sync.client import connect as ws_connect
            connect = ws_connect
        self._connect = connect
        self.max_backoff_s = float(max_backoff_s)
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._reconnected = False
        self.connected = False
        self.connections = 0
        self.received = 0
        self.auth_error: Optional[str] = None

    def __repr__(self) -> str:  # nunca el secreto
        return f"TradeUpdatesStream(url={self.url}, connected={self.connected})"

    def start(self, wait_s: float = _OPEN_TIMEOUT_S) -> bool:
        """Arranca el hilo. True si quedó escuchando antes de ``wait_s``."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="alpaca-trade-updates",
                                            daemon=True)
            self._thread.start()
        return self._ready.wait(wait_s) and self.auth_error is None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_RECV_TIMEOUT_S * 3)
            self._thread = None

    def drain(self) -> List[dict]:
        """Eventos recibidos desde la última llamada, en orden de llegada."""
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def take_reconnected(self) -> bool:
        """True una vez por cada (re)conexión: hay que reconciliar por REST."""
        with self._lock:
            flag, self._reconnected = self._reconnected, False
        return flag

    # --- Hilo ---
    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._session()
                backoff = 0.5
            except StreamAuthError as exc:
                self.auth_error = str(exc)
                logger.error("stream trade_updates sin autorización: %s", exc)
                self._ready.set()
                return
            except Exception as exc:  # noqa: BLE001 - red: se reconecta
                logger.warning("stream trade_updates caído (%s); reconecta en %.1fs",
                               type(exc).__name__, backoff)
            self.connected = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2.0, self.max_backoff_s)

    def _session(self) -> None:
        with self._connect(self.url, open_timeout=_OPEN_TIMEOUT_S) as ws:
            ws.send(json.dumps({"action": "auth", "key": self._key, "secret": self._secret}))
            reply = json.loads(_text(ws.recv(timeout=_OPEN_TIMEOUT_S)))
            status = (reply.get("data") or {}).get("status")
            if reply.get("stream") != "authorization" or status != "authorized":
                raise StreamAuthError(f"respuesta de auth: {status or reply.get('stream')}")
            ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
            reply = json.loads(_text(ws.recv(timeout=_OPEN_TIMEOUT_S)))
            if reply.get("stream") != "listening":
                raise ConnectionError(f"listen sin confirmar: {reply.get('stream')}")
            self.connections += 1
            self.connected = True
            with self._lock:
                self._reconnected = True
            self._ready.set()
            logger.info("stream trade_updates escuchando (%s)", self.url)
            while not self._stop.is_set():
                try:
                    message = ws.recv(timeout=_RECV_TIMEOUT_S)
                except TimeoutError:
                    continue
                try:
                    payload = json.loads(_text(message))
                except ValueError:
                    logger.warning("stream: mensaje que no es JSON, se ignora")
                    continue
                if payload.get("stream") == "trade_updates" and isinstance(payload.get("data"), dict):
                    self.received += 1
                    self._queue.put(payload["data"])
