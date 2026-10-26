"""Probe del techo real de la Trading API de Alpaca paper.

Dos modos:

- ``ceiling`` (fuera de horario, sin fills): límites de compra de SPY muy
  por debajo del mercado (``--limit-price``), que Alpaca acepta y encola
  pero no llena, y su cancelación. Mide los headers de rate limit,
  ráfagas de K órdenes concurrentes, la tasa sostenida con el cupo local
  a 200/min y qué pasa por encima (``--over-rpm``).
- ``crypto`` (24/7, con fills): compras market BTC/USD por ``--notional``
  USD en ráfaga, fills por el stream ``trade_updates``, y ventas del 99%
  de cada fill. El resto (comisión cobrada en BTC) se liquida al final si
  la cuenta arrancó sin BTC.

Seguridad:

- Sin ``--confirm`` solo lee (cuenta, reloj, posiciones, órdenes abiertas).
- Toma el lock del estado intradía (``--state-dir``, por defecto el de
  ``settings``): una sesión paper no puede arrancar mientras corre, y si
  hay una corriendo el probe sale con 4 sin enviar nada.
- Ids propios ``p-probe-<run>-...``: la sesión HFT (prefijo ``h-``) no
  los adopta. Al final cancela solo esos ids y verifica por API que no
  queden órdenes abiertas suyas ni posiciones distintas a las de partida.
- ``--max-orders`` es un tope duro por corrida. Respeta el KILL.
- ``ceiling`` se niega con el mercado abierto (``--allow-open`` lo fuerza).

Uso (claves del ``.env`` principal exportadas al proceso):
    python -m scripts.hft_rate_probe --mode ceiling             # solo lectura
    python -m scripts.hft_rate_probe --mode ceiling --confirm
    python -m scripts.hft_rate_probe --mode crypto --confirm --orders 40
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.brokers.alpaca_adapter import (
    AlpacaClient,
    BrokerError,
    BrokerTransientError,
    GatewayConfig,
    OrderSpec,
    load_broker_config,
)
from src.brokers.alpaca_stream import TradeUpdatesStream
from src.brokers.http_transport import PooledTransport
from src.brokers.pre_trade import kill_switch_reason
from src.brokers.rate_budget import TradingBudget

logger = logging.getLogger("hft_rate_probe")

EXIT_OK = 0
EXIT_GUARD = 2
EXIT_BROKER = 3
EXIT_LOCK = 4
EXIT_DIRTY = 6
CRYPTO_SYMBOL = "BTC/USD"
CRYPTO_POSITION = "BTCUSD"


@dataclass
class Record:
    wall: float
    elapsed: float
    method: str
    path: str
    status: Optional[int]
    limit: Optional[str]
    remaining: Optional[str]
    reset: Optional[str]
    error: Optional[str] = None


class RecordingTransport:
    """Envuelve el transporte y anota cada petición (sin cuerpos ni claves)."""

    pooled = True

    def __init__(self, inner) -> None:
        self.inner = inner
        self.records: List[Record] = []
        self._lock = threading.Lock()

    @property
    def connections_opened(self) -> int:
        return getattr(self.inner, "connections_opened", 0)

    def request(self, method, path, data, headers):
        wall = time.time()
        started = time.perf_counter()
        base = path.split("?", 1)[0]
        try:
            resp = self.inner.request(method, path, data, headers)
        except Exception as exc:
            with self._lock:
                self.records.append(Record(wall, time.perf_counter() - started, method, base,
                                           None, None, None, None, type(exc).__name__))
            raise
        with self._lock:
            self.records.append(Record(
                wall, time.perf_counter() - started, method, base, resp.status,
                resp.headers.get("x-ratelimit-limit"), resp.headers.get("x-ratelimit-remaining"),
                resp.headers.get("x-ratelimit-reset"),
            ))
        return resp

    def last_headers(self) -> Optional[Record]:
        with self._lock:
            for record in reversed(self.records):
                if record.reset is not None:
                    return record
        return None

    def close(self) -> None:
        self.inner.close()


@dataclass
class Phase:
    name: str
    sent: int = 0
    accepted: int = 0
    throttled: int = 0
    failed: int = 0
    seconds: float = 0.0
    latencies: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def row(self) -> str:
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] * 1000 if lat else float("nan")
        p99 = lat[min(len(lat) - 1, int(0.99 * (len(lat) - 1)))] * 1000 if lat else float("nan")
        rate = self.accepted / self.seconds if self.seconds > 0 else 0.0
        notes = "; ".join(dict.fromkeys(self.notes))
        return (f"| {self.name} | {self.sent} | {self.accepted} | {self.throttled} | {self.failed} | "
                f"{self.seconds:.2f} | {rate:.2f} | {p50:.0f} | {p99:.0f} | {notes} |")


class Probe:
    def __init__(self, args, client: AlpacaClient, recorder: RecordingTransport, run: str) -> None:
        self.args = args
        self.client = client
        self.recorder = recorder
        self.run = run
        self.counter = 0
        self.mine: Dict[str, Optional[str]] = {}  # client_order_id -> alpaca id
        self._lock = threading.Lock()
        self.phases: List[Phase] = []

    # --- utilidades ---
    def _cid(self, tag: str) -> str:
        with self._lock:
            self.counter += 1
            if self.counter > self.args.max_orders:
                raise RuntimeError(f"tope --max-orders {self.args.max_orders} alcanzado")
            return f"p-probe-{self.run}-{tag}-{self.counter}"

    def _guard_kill(self) -> None:
        reason = kill_switch_reason(self.client.kill_dir)
        if reason:
            raise RuntimeError(reason)

    def wait_reset(self, need: int) -> None:
        """Duerme hasta que el servidor declare cupo para ``need`` peticiones."""
        last = self.recorder.last_headers()
        if last is None or last.remaining is None:
            return
        try:
            remaining = int(float(last.remaining))
            reset = float(last.reset)
        except (TypeError, ValueError):
            return
        if remaining >= need:
            return
        wait = max(0.0, reset - time.time()) + 1.0
        logger.info("esperando el reset del cupo: %.1fs", wait)
        time.sleep(min(wait, 75.0))

    def _limit_spec(self, cid: str) -> OrderSpec:
        return OrderSpec(instrument_id=self.args.symbol, side="buy", qty=1, order_type="limit",
                         time_in_force="day", order_class="simple",
                         limit_price=self.args.limit_price, client_order_id=cid,
                         extended_hours=False, short=False)

    def _post_limit(self, phase: Phase, tag: str) -> Optional[str]:
        self._guard_kill()
        cid = self._cid(tag)
        with self._lock:
            self.mine[cid] = None
        started = time.perf_counter()
        try:
            result = self.client.submit_order(self._limit_spec(cid), mark=self.args.limit_price)
        except BrokerTransientError as exc:
            with self._lock:
                if exc.status == 429:
                    phase.throttled += 1
                else:
                    phase.failed += 1
                    phase.notes.append(str(exc)[:80])
            return None
        except BrokerError as exc:
            with self._lock:
                phase.failed += 1
                phase.notes.append(str(exc)[:80])
            return None
        elapsed = time.perf_counter() - started
        with self._lock:
            phase.sent += 1
            if result.get("status") == "submitted":
                phase.accepted += 1
                phase.latencies.append(elapsed)
                self.mine[cid] = result.get("id")
            else:
                phase.failed += 1
                phase.notes.append(str(result.get("error"))[:80])
        return result.get("id")

    def _cancel(self, order_id: str) -> bool:
        """Cancela con reintento propio ante 429 (DELETE no reintenta)."""
        for _ in range(6):
            try:
                self.client.cancel_order(order_id)
                return True
            except BrokerTransientError as exc:
                if exc.status == 429:
                    self.wait_reset(1)
                    continue
                time.sleep(1.0)
            except BrokerError as exc:
                if exc.status in (404, 422):  # ya final
                    return True
                logger.warning("cancel %s: %s", order_id, exc)
                return False
        return False

    # --- fases ceiling ---
    def burst(self, size: int) -> None:
        phase = Phase(f"ráfaga {size} POST")
        self.wait_reset(size * 2 + 5)
        started = time.perf_counter()
        ids = self.client.run_many([lambda: self._post_limit(phase, f"b{size}")
                                    for _ in range(size)])
        phase.seconds = time.perf_counter() - started
        self.phases.append(phase)
        cancel = Phase(f"cancelar ráfaga {size}")
        started = time.perf_counter()
        order_ids = [oid for oid, err in ids if err is None and oid]
        done = self.client.run_many([lambda oid=oid: self._cancel(oid) for oid in order_ids])
        cancel.seconds = time.perf_counter() - started
        cancel.sent = len(order_ids)
        cancel.accepted = sum(1 for ok, err in done if err is None and ok)
        cancel.failed = cancel.sent - cancel.accepted
        self.phases.append(cancel)

    def sustained(self, name: str, rpm: float, seconds: float) -> None:
        """Alta + cancelación en lazo, con el cupo local a ``rpm``."""
        phase = Phase(f"{name} ({rpm:g}/min, alta+cancel)")
        self.wait_reset(int(rpm))
        self.client.budget = TradingBudget(rpm, capacity=max(1.0, rpm / 6.0))
        deadline = time.monotonic() + seconds
        started = time.perf_counter()

        def worker() -> None:
            while time.monotonic() < deadline:
                try:
                    order_id = self._post_limit(phase, "s")
                except RuntimeError as exc:
                    phase.notes.append(str(exc)[:60])
                    return
                if order_id:
                    self._cancel(order_id)

        threads = [threading.Thread(target=worker) for _ in range(min(8, self.args.workers))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        phase.seconds = time.perf_counter() - started
        stats = self.client.budget.stats
        phase.notes.append(f"429 vistos por el cupo={stats.throttled_429}")
        self.client.budget = None
        self.phases.append(phase)

    # --- cripto ---
    def crypto(self, stream: TradeUpdatesStream) -> None:
        count = self.args.orders
        buys = Phase(f"compras BTC/USD x{count} (${self.args.notional:g})")
        sent_at: Dict[str, float] = {}
        filled_at: Dict[str, float] = {}
        filled_qty: Dict[str, str] = {}

        def post(side: str, tag: str, extra: dict, phase: Phase) -> None:
            self._guard_kill()
            cid = self._cid(tag)
            payload = {"symbol": CRYPTO_SYMBOL, "side": side, "type": "market",
                       "time_in_force": "gtc", "client_order_id": cid, **extra}
            with self._lock:
                self.mine[cid] = None
                sent_at[cid] = time.time()
            started = time.perf_counter()
            try:
                resp = self.client._post_order(payload, lane="entry" if side == "buy" else "exit")
            except BrokerError as exc:
                with self._lock:
                    phase.failed += 1
                    if exc.status == 429:
                        phase.throttled += 1
                    phase.notes.append(str(exc)[:80])
                return
            with self._lock:
                phase.sent += 1
                phase.accepted += 1
                phase.latencies.append(time.perf_counter() - started)
                self.mine[cid] = resp.get("id")

        def collect(cids: List[str], timeout: float) -> None:
            deadline = time.monotonic() + timeout
            wanted = set(cids)
            while time.monotonic() < deadline and not wanted.issubset(filled_at):
                for data in stream.drain():
                    order = data.get("order") or {}
                    cid = str(order.get("client_order_id") or "")
                    if cid in wanted and data.get("event") == "fill":
                        filled_at.setdefault(cid, float(data.get("_recv_ts") or time.time()))
                        filled_qty[cid] = str(order.get("filled_qty"))
                time.sleep(0.005)

        self.wait_reset(count * 2 + 10)
        started = time.perf_counter()
        before = set(sent_at)
        notional = f"{self.args.notional:.2f}"
        self.client.run_many([lambda: post("buy", "cb", {"notional": notional}, buys)
                              for _ in range(count)])
        buy_cids = [cid for cid in sent_at if cid not in before]
        collect(buy_cids, 30.0)
        buys.seconds = time.perf_counter() - started
        fill_lat = [filled_at[c] - sent_at[c] for c in buy_cids if c in filled_at]
        buys.notes.append(f"fills por stream={len(fill_lat)}/{len(buy_cids)}")
        if fill_lat:
            buys.notes.append(f"POST→fill p50={statistics.median(fill_lat) * 1000:.0f}ms "
                              f"max={max(fill_lat) * 1000:.0f}ms")
        self.phases.append(buys)

        sells = Phase(f"ventas BTC/USD x{len(filled_qty)} (99% de cada fill)")
        started = time.perf_counter()
        before = set(sent_at)
        jobs = []
        for qty in filled_qty.values():
            try:
                amount = math.floor(float(qty) * 0.99 * 1e9) / 1e9
            except ValueError:
                continue
            if amount > 0:
                jobs.append(lambda amount=amount: post("sell", "cs", {"qty": f"{amount:.9f}"}, sells))
        self.client.run_many(jobs)
        sell_cids = [cid for cid in sent_at if cid not in before]
        collect(sell_cids, 30.0)
        sells.seconds = time.perf_counter() - started
        fill_lat = [filled_at[c] - sent_at[c] for c in sell_cids if c in filled_at]
        sells.notes.append(f"fills por stream={len(fill_lat)}/{len(sell_cids)}")
        if fill_lat:
            sells.notes.append(f"POST→fill p50={statistics.median(fill_lat) * 1000:.0f}ms "
                               f"max={max(fill_lat) * 1000:.0f}ms")
        self.phases.append(sells)

    # --- limpieza ---
    def cleanup(self) -> List[str]:
        """Cancela lo propio que siga abierto. Devuelve los ids que no se pudo."""
        prefix = f"p-probe-{self.run}-"
        stuck: List[str] = []
        for _ in range(3):
            try:
                open_orders = self.client.list_orders("open", limit=500)
            except BrokerError as exc:
                logger.warning("no se pudo listar órdenes abiertas: %s", exc)
                self.wait_reset(5)
                continue
            mine = [o for o in open_orders if str(o.get("client_order_id", "")).startswith(prefix)]
            if not mine:
                return []
            stuck = [str(o["id"]) for o in mine if not self._cancel(str(o["id"]))]
            time.sleep(1.0)
        return stuck


def _positions(client: AlpacaClient) -> Dict[str, float]:
    raw = client._request("GET", "/v2/positions")
    return {str(p.get("symbol")): float(p.get("qty") or 0.0) for p in (raw or [])}


def _stamp_stream(stream: TradeUpdatesStream) -> None:
    """Sella la hora de llegada de cada evento (latencia POST→fill)."""
    original = stream._queue.put

    def put(item, *a, **k):
        if isinstance(item, dict):
            item["_recv_ts"] = time.time()
        return original(item, *a, **k)

    stream._queue.put = put  # type: ignore[assignment]


def _report(probe: Probe, args, account: dict, clock: dict, before: dict, after: dict,
            stuck: List[str]) -> str:
    records = probe.recorder.records
    limits = sorted({r.limit for r in records if r.limit})
    statuses: Dict[str, int] = {}
    for record in records:
        key = str(record.status) if record.status is not None else (record.error or "error")
        statuses[key] = statuses.get(key, 0) + 1
    ok = [r for r in records if r.status is not None and r.status < 400]
    by_second: Dict[int, int] = {}
    by_minute: Dict[int, int] = {}
    for record in ok:
        by_second[int(record.wall)] = by_second.get(int(record.wall), 0) + 1
        by_minute[int(record.wall // 60)] = by_minute.get(int(record.wall // 60), 0) + 1
    peak = max(by_second.values()) if by_second else 0
    best = 0
    walls = sorted(r.wall for r in ok)
    left = 0
    for right, wall in enumerate(walls):
        while wall - walls[left] >= 60.0:
            left += 1
        best = max(best, right - left + 1)
    lines = [
        f"# Probe de rate limit Alpaca paper ({probe.run})",
        "",
        f"- Modo: `{args.mode}`; cuenta {account.get('account_number')}; "
        f"mercado abierto: {clock.get('is_open')}",
        f"- Peticiones: {len(records)}; estados: {json.dumps(statuses, sort_keys=True)}",
        f"- `X-RateLimit-Limit` visto: {', '.join(limits) or 'ninguno'}",
        f"- Máximo de peticiones aceptadas en un mismo segundo: {peak}",
        f"- Máximo aceptado en cualquier ventana deslizante de 60 s: {best}",
        f"- Aceptadas por minuto de reloj (top 5): {sorted(by_minute.values(), reverse=True)[:5]}",
        f"- Conexiones abiertas: {probe.recorder.connections_opened}",
        f"- Órdenes propias sin cancelar al final: {stuck or 'ninguna'}",
        f"- Posiciones antes: {before}; después: {after}",
        "",
        "| fase | enviadas | aceptadas | 429 | fallidas | segundos | aceptadas/s | p50 ms | p99 ms | notas |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(phase.row() for phase in probe.phases)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("ceiling", "crypto"), required=True)
    parser.add_argument("--confirm", action="store_true", help="sin esto solo lee")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="estado intradía cuyo lock se toma (default: settings.state_dir/hft)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-orders", type=int, default=600)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--limit-price", type=float, default=100.0,
                        help="límite de compra muy por debajo del mercado (no se llena)")
    parser.add_argument("--bursts", default="25,50,100")
    parser.add_argument("--sustained-s", type=float, default=70.0)
    parser.add_argument("--over-rpm", type=float, default=260.0)
    parser.add_argument("--over-s", type=float, default=45.0)
    parser.add_argument("--orders", type=int, default=40, help="compras en modo crypto")
    parser.add_argument("--notional", type=float, default=25.0, help="USD por compra cripto")
    parser.add_argument("--allow-open", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    from src.live_state import StateLockedError, StateStore
    from src.settings import load_settings, setup_logging

    setup_logging(logging.INFO)
    for noisy in ("alpaca_adapter", "alpaca_stream"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings()
    state_dir = Path(args.state_dir) if args.state_dir else settings.state_dir / "hft"
    run = datetime.now(timezone.utc).strftime("%m%d%H%M%S")

    store = StateStore(state_dir)
    try:
        store.acquire()
    except StateLockedError as exc:
        print(f"probe: {exc}", file=sys.stderr)
        return EXIT_LOCK
    stream = None
    client = None
    try:
        try:
            broker_config = load_broker_config()
        except ValueError as exc:
            print(f"probe: {exc}", file=sys.stderr)
            return EXIT_BROKER
        from scripts.run_intraday import _hft_instruments

        recorder = RecordingTransport(PooledTransport(broker_config.base_url, 15))
        # El KILL del intradía vive en settings.state_dir, el padre de /hft.
        client = AlpacaClient(broker_config, kill_dir=state_dir.parent,
                              instruments=_hft_instruments(),
                              gateway=GatewayConfig(pooled=True, workers=args.workers),
                              transport=recorder)
        account = client._verify_account()
        clock = client.get_clock()
        before = _positions(client)
        open_now = client.list_orders("open", limit=500)
        print(f"cuenta {account.get('account_number')} equity={account.get('equity')} "
              f"mercado abierto={clock.get('is_open')} posiciones={before} "
              f"órdenes abiertas={len(open_now)}")
        last = recorder.last_headers()
        if last is not None:
            print(f"X-RateLimit-Limit={last.limit} Remaining={last.remaining} Reset={last.reset}")
        if args.mode == "ceiling" and clock.get("is_open") and not args.allow_open:
            print("probe: ceiling no corre con el mercado abierto (--allow-open)", file=sys.stderr)
            return EXIT_GUARD
        if not args.confirm:
            print("solo lectura: agregar --confirm para enviar órdenes")
            return EXIT_OK
        probe = Probe(args, client, recorder, run)
        try:
            if args.mode == "ceiling":
                for size in [int(x) for x in args.bursts.split(",") if x.strip()]:
                    probe.burst(size)
                if args.sustained_s > 0:
                    probe.sustained("sostenido", 200.0, args.sustained_s)
                if args.over_s > 0:
                    probe.sustained("por encima", args.over_rpm, args.over_s)
            else:
                stream = TradeUpdatesStream(broker_config.api_key, broker_config.secret_key)
                _stamp_stream(stream)
                if not stream.start(15):
                    print(f"probe: el stream no arrancó ({stream.auth_error})", file=sys.stderr)
                    return EXIT_BROKER
                probe.crypto(stream)
        except RuntimeError as exc:
            print(f"probe detenido: {exc}", file=sys.stderr)
        stuck = probe.cleanup()
        if args.mode == "crypto":
            time.sleep(2.0)
            now_pos = _positions(client)
            residual = now_pos.get(CRYPTO_POSITION, 0.0) - before.get(CRYPTO_POSITION, 0.0)
            if residual > 0 and before.get(CRYPTO_POSITION, 0.0) == 0.0:
                print(f"liquidando resto {residual:.9f} {CRYPTO_POSITION}")
                client._request("DELETE", f"/v2/positions/{CRYPTO_POSITION}", lane="exit")
                time.sleep(2.0)
        after = _positions(client)
        text = _report(probe, args, account, clock, before, after, stuck)
        print(text)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            path = args.out_dir / f"probe-{args.mode}-{run}.md"
            path.write_text(text, encoding="utf-8")
            raw = args.out_dir / f"probe-{args.mode}-{run}.jsonl"
            with raw.open("w", encoding="utf-8") as handle:
                for record in recorder.records:
                    handle.write(json.dumps(record.__dict__) + "\n")
            print(f"informe: {path}")
        dirty = bool(stuck) or {k: round(v, 9) for k, v in after.items()} != \
            {k: round(v, 9) for k, v in before.items()}
        if dirty:
            print("ATENCIÓN: quedaron órdenes o posiciones distintas a las de partida",
                  file=sys.stderr)
            return EXIT_DIRTY
        return EXIT_OK
    finally:
        if stream is not None:
            stream.stop()
        if client is not None:
            client.close()
        store.release()


if __name__ == "__main__":
    sys.exit(main())
