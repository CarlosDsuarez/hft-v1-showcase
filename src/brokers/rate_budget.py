"""Cupo de la Trading API de Alpaca, compartido por todas las llamadas.

Alpaca limita la Trading API por cuenta: 200 peticiones por minuto
(foro de Alpaca, staff, y headers ``X-RateLimit-Limit: 200``). El tope
cuenta POST, DELETE, PATCH y GET por igual, y un plan de datos pago no
lo sube. Pasarse devuelve 429.

``TradingBudget`` es un token bucket: ``capacity`` fichas que se
rellenan a ``per_minute / 60`` por segundo. Cada respuesta lo corrige
con lo que dice el servidor (``X-RateLimit-Remaining`` y
``X-RateLimit-Reset``, segundos epoch). Un 429 frena todo hasta el
reset, o hasta ``Retry-After``, o un segundo si no viene ninguno. Un
reset entero cuenta hasta el segundo siguiente: el servidor pudo
truncarlo, y despertar en el entero da 429 en ráfaga.

Hay tres carriles, en orden de prioridad:

- ``exit``: cancelaciones, flatten y órdenes que reducen posición.
- ``read``: lecturas de reconciliación y de estado.
- ``entry``: órdenes que abren o agrandan posición.

Una ficha sale para el carril de mayor prioridad que esté esperando.
Además ``entry`` no puede bajar el bucket de ``reserve`` fichas, y
``read`` no lo baja de la mitad: una salida nunca queda sin cupo detrás
de una ráfaga de entradas.

``try_acquire`` es puro (reloj inyectado) y lo usan los tests.
``acquire`` bloquea con una ``threading.Condition``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

LANES = ("exit", "read", "entry")
_PRIORITY = {lane: index for index, lane in enumerate(LANES)}
DEFAULT_PER_MINUTE = 200
# Fracción del cupo que una entrada no puede consumir.
DEFAULT_RESERVE_FRAC = 0.10
# Espera tras un 429 sin headers de reset.
FALLBACK_429_WAIT_S = 1.0
# Un reset más lejano que esto es un header raro y no se obedece.
_MAX_RESET_S = 120.0


@dataclass
class BudgetStats:
    """Contadores para el bench y el informe."""

    acquired: Dict[str, int]
    waited_s: Dict[str, float]
    throttled_429: int = 0
    server_corrections: int = 0


class TradingBudget:
    """Token bucket con carriles. Thread-safe."""

    def __init__(
        self,
        per_minute: float = DEFAULT_PER_MINUTE,
        *,
        capacity: Optional[float] = None,
        reserve_frac: float = DEFAULT_RESERVE_FRAC,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(per_minute, bool) or not isinstance(per_minute, (int, float)):
            raise TypeError("per_minute debe ser numérico")
        if per_minute <= 0:
            raise ValueError("per_minute debe ser positivo")
        if not 0.0 <= float(reserve_frac) < 1.0:
            raise ValueError("reserve_frac debe estar en [0, 1)")
        self.per_minute = float(per_minute)
        self.capacity = float(capacity) if capacity is not None else self.per_minute
        if self.capacity < 1:
            raise ValueError("capacity debe ser >= 1")
        self.rate = self.per_minute / 60.0
        self.reserve = math.floor(self.capacity * float(reserve_frac))
        self._clock = clock
        self._wall = wall
        self._tokens = self.capacity
        self._stamp = clock()
        self._blocked_until = 0.0
        self._waiting = {lane: 0 for lane in LANES}
        self._cond = threading.Condition()
        self.stats = BudgetStats(
            acquired={lane: 0 for lane in LANES},
            waited_s={lane: 0.0 for lane in LANES},
        )

    # --- Núcleo puro ---
    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._stamp)
        self._stamp = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def _floor(self, lane: str) -> float:
        if lane == "entry":
            return float(self.reserve)
        if lane == "read":
            return float(self.reserve) / 2.0
        return 0.0

    def _higher_waiting(self, lane: str) -> bool:
        mine = _PRIORITY[lane]
        return any(self._waiting[other] for other in LANES if _PRIORITY[other] < mine)

    def try_acquire(self, lane: str) -> float:
        """0.0 si tomó una ficha; si no, segundos sugeridos de espera.

        No mira otros hilos: ``acquire`` lo envuelve con la prioridad.
        """
        if lane not in _PRIORITY:
            raise ValueError(f"carril {lane!r} no está en {LANES}")
        with self._cond:
            now = self._clock()
            self._refill(now)
            if now < self._blocked_until:
                return self._blocked_until - now
            floor = self._floor(lane)
            if self._tokens - floor >= 1.0:
                self._tokens -= 1.0
                self.stats.acquired[lane] += 1
                return 0.0
            missing = floor + 1.0 - self._tokens
            return max(missing / self.rate, 0.001)

    # --- Bloqueante ---
    def acquire(self, lane: str = "entry", *, timeout: Optional[float] = None) -> bool:
        """Espera una ficha. False si vence ``timeout`` (None: sin tope)."""
        if lane not in _PRIORITY:
            raise ValueError(f"carril {lane!r} no está en {LANES}")
        start = self._clock()
        with self._cond:
            self._waiting[lane] += 1
            try:
                while True:
                    if self._higher_waiting(lane):
                        wait = 0.05
                    else:
                        wait = self.try_acquire(lane)
                        if wait == 0.0:
                            self.stats.waited_s[lane] += self._clock() - start
                            return True
                    if timeout is not None:
                        left = timeout - (self._clock() - start)
                        if left <= 0:
                            self.stats.waited_s[lane] += self._clock() - start
                            return False
                        wait = min(wait, left)
                    self._cond.wait(wait)
            finally:
                self._waiting[lane] -= 1
                self._cond.notify_all()

    # --- Lo que dice el servidor ---
    def observe(self, status: int, headers: Mapping[str, str]) -> None:
        """Corrige el bucket con la respuesta. ``headers`` en minúscula."""
        with self._cond:
            now = self._clock()
            self._refill(now)
            reset_at = self._reset_monotonic(headers, now)
            remaining = _as_float(headers.get("x-ratelimit-remaining"))
            if remaining is not None and remaining < self._tokens:
                self._tokens = max(0.0, remaining)
                self.stats.server_corrections += 1
            if remaining is not None and remaining <= 0 and reset_at is not None:
                self._blocked_until = max(self._blocked_until, reset_at)
            if int(status) == 429:
                self.stats.throttled_429 += 1
                self._tokens = 0.0
                retry_after = _as_float(headers.get("retry-after"))
                if reset_at is not None:
                    until = reset_at
                elif retry_after is not None and 0.0 <= retry_after <= _MAX_RESET_S:
                    until = now + retry_after
                else:
                    until = now + FALLBACK_429_WAIT_S
                self._blocked_until = max(self._blocked_until, until)
            self._cond.notify_all()

    def _reset_monotonic(self, headers: Mapping[str, str], now: float) -> Optional[float]:
        epoch = _as_float(headers.get("x-ratelimit-reset"))
        if epoch is None:
            return None
        if epoch.is_integer():
            # Segundos enteros: el servidor pudo truncar y el cupo vuelve
            # dentro del segundo siguiente. Despertar en el entero da 429.
            epoch += 1.0
        delta = epoch - self._wall()
        if delta > _MAX_RESET_S:
            return None
        return now + max(0.0, delta)

    @property
    def tokens(self) -> float:
        with self._cond:
            self._refill(self._clock())
            return self._tokens


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out
