"""Transporte HTTP del broker: una petición, una respuesta cruda.

``AlpacaClient._send`` clasifica la respuesta (2xx, 4xx, 429/5xx, red).
Este módulo solo la mueve. Hay dos transportes con la misma firma:

- ``UrllibTransport``: lo de siempre. ``urllib.request.urlopen`` por
  petición, con una conexión TLS nueva cada vez (~300 ms medidos contra
  paper-api el 2026-10-13). Los tests lo sustituyen parcheando
  ``urllib.request.urlopen``.
- ``PooledTransport``: una conexión keep-alive por hilo
  (``http.client``). Medido igual: ~90 ms por petición. Varios hilos
  dan varias peticiones en vuelo.

El host se valida aquí también: paper-api de Alpaca, o loopback para el
exchange falso de ``src.brokers.fake_alpaca`` (bench y tests). Nada más.
"""

from __future__ import annotations

import http.client
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

PAPER_HOST = "paper-api.alpaca.markets"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# Una conexión ociosa más tiempo que esto se reabre antes de usarla. El
# servidor puede haberla cerrado, y un POST sobre un socket muerto es un
# fallo ambiguo que cuesta un GET de consulta.
DEFAULT_MAX_IDLE_S = 20.0


@dataclass(frozen=True)
class RawResponse:
    """Respuesta sin interpretar. ``headers`` lleva las claves en minúscula."""

    status: int
    headers: Mapping[str, str]
    body: bytes


def validate_transport_url(base_url: str) -> None:
    """Acepta https://paper-api.alpaca.markets o un loopback (http o https).

    Un loopback solo sirve para el exchange falso. Cualquier otra cosa
    (live, un host parecido, userinfo@) se rechaza sin abrir red.
    """
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    clean = parsed.path in ("", "/") and not (
        parsed.params or parsed.query or parsed.fragment or parsed.username
    )
    if parsed.scheme == "https" and host == PAPER_HOST and clean and parsed.port in (None, 443):
        return
    if parsed.scheme in ("http", "https") and host in LOOPBACK_HOSTS and clean:
        return
    raise ValueError(
        f"transporte rechazado: {base_url}. Solo https://{PAPER_HOST} o un "
        "loopback del exchange falso."
    )


def _lower(headers: object) -> Dict[str, str]:
    if headers is None:
        return {}
    items = headers.items() if hasattr(headers, "items") else headers
    try:
        return {str(key).lower(): str(value) for key, value in items}
    except (TypeError, ValueError):
        return {}


class UrllibTransport:
    """``urlopen`` por petición. Mismo comportamiento que antes del pool."""

    pooled = False

    def __init__(self, base_url: str, timeout: float) -> None:
        validate_transport_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connections_opened = 0

    def request(self, method: str, path: str, data: Optional[bytes],
                headers: Mapping[str, str]) -> RawResponse:
        """Una petición. Errores de red suben tal cual (OSError, HTTPException)."""
        req = urllib.request.Request(f"{self.base_url}{path}", data=data,
                                     headers=dict(headers), method=method)
        self.connections_opened += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = getattr(resp, "status", None)
                return RawResponse(int(status) if isinstance(status, int) else 200,
                                   _lower(getattr(resp, "headers", None)), raw)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            except Exception:
                body = b""
            return RawResponse(int(exc.code), _lower(getattr(exc, "headers", None)),
                               body if isinstance(body, bytes) else b"")

    def close(self) -> None:
        return None


class PooledTransport:
    """Una conexión keep-alive por hilo.

    ``http.client`` no es thread-safe por conexión: cada hilo tiene la
    suya (``threading.local``). Un fallo de red cierra la conexión del
    hilo y sube; la siguiente petición abre otra. No reintenta: quien
    llama decide (GET reintenta, POST consulta por client_order_id).
    """

    pooled = True

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        max_idle_s: float = DEFAULT_MAX_IDLE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_transport_url(base_url)
        parsed = urlparse(base_url)
        self.base_url = base_url.rstrip("/")
        self._https = parsed.scheme == "https"
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self.timeout = float(timeout)
        self.max_idle_s = float(max_idle_s)
        self._clock = clock
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all: list = []
        self.connections_opened = 0

    def _open(self) -> http.client.HTTPConnection:
        if self._https:
            conn = http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout)
        else:
            conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
        with self._lock:
            self._all.append(conn)
            self.connections_opened += 1
        return conn

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        last = getattr(self._local, "last", None)
        if conn is not None and last is not None and self._clock() - last > self.max_idle_s:
            self._drop()
            conn = None
        if conn is None:
            conn = self._open()
            self._local.conn = conn
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        self._local.last = None
        if conn is not None:
            with self._lock:
                if conn in self._all:
                    self._all.remove(conn)
            try:
                conn.close()
            except Exception:
                pass

    def request(self, method: str, path: str, data: Optional[bytes],
                headers: Mapping[str, str]) -> RawResponse:
        conn = self._conn()
        try:
            conn.request(method, path, body=data, headers=dict(headers))
            resp = conn.getresponse()
            body = resp.read()
        except BaseException:
            self._drop()
            raise
        self._local.last = self._clock()
        if resp.will_close:
            self._drop()
        return RawResponse(int(resp.status), _lower(resp.getheaders()), body)

    def close(self) -> None:
        with self._lock:
            conns, self._all = self._all, []
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
