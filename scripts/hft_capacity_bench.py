"""Bench local de capacidad de órdenes del intradía. Sin red externa.

Mide cada capa por separado y escribe una tabla markdown:

1. ``engine-sim``: ``Engine.step`` + ``SimExecution`` con ``CapacityProbe``
   (CPU pura: motor, riesgo, sizer y simulador).
2. ``client``: ``AlpacaClient.submit_order`` contra el exchange falso de
   loopback, con el transporte de siempre (urlopen) y con el pool
   keep-alive, en serie y con hilos, a 0 ms y a ``--latency-ms``.
3. ``engine-fake``: la ruta entera (``Engine.step`` → ``AlpacaExecution``
   → pool → exchange falso → fills por el stream ``trade_updates``).
4. ``engine-fake-200``: lo mismo con el tope de 200 req/min de Alpaca y
   el cupo local (``TradingBudget``). Tarda ~2.5 minutos: ``--with-limit``.

Uso:
    python -m scripts.hft_capacity_bench
    python -m scripts.hft_capacity_bench --with-limit --out results/hft/bench.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import List

from src.brokers.alpaca_adapter import AlpacaClient, BrokerConfig, GatewayConfig, OrderSpec
from src.brokers.alpaca_stream import TradeUpdatesStream
from src.brokers.fake_alpaca import FakeAlpaca
from src.brokers.http_transport import PooledTransport, UrllibTransport
from src.hft.calendar import session_at
from src.hft.capacity import probe_config, run_capacity
from src.hft.config import load_hft_config
from src.hft.costs import CostModel
from src.hft.data.synthetic import synthetic_bars
from src.hft.execution.alpaca_exec import AlpacaExecution
from src.hft.execution.sim import SimExecution

PAPER = "https://paper-api.alpaca.markets"
SESSION_DAY = date(2025, 3, 3)


def _instruments():
    from scripts.run_intraday import _hft_instruments
    return _hft_instruments()


def _setup():
    config = probe_config(load_hft_config())
    session = session_at(SESSION_DAY, dtime(9, 30), dtime(16, 0))
    bars = synthetic_bars([session], list(config.universe), start_price=200.0)
    return config, session, bars


def bench_engine_sim(rows: List[str]) -> None:
    config, session, bars = _setup()
    universe = list(config.universe)
    for per_bar in (10, 100, 1000):
        sim = SimExecution(
            CostModel.from_config(config), partial_prob=0.0, participation_cap=1.0,
            adv={s: 1e9 for s in universe}, sigma={s: 0.01 for s in universe},
        )
        result = run_capacity(sim, config=config, session=session, bars=bars,
                              orders_per_bar=per_bar, max_bars=60)
        rows.append(
            f"| engine-sim | {per_bar}/barra | - | {result.submitted} | {result.fills} | "
            f"{result.pct(0.5) * 1000:.1f} | {result.pct(0.99) * 1000:.1f} | "
            f"{result.orders_per_s():.0f} | {result.fills_per_s():.0f} |"
        )


def _spec(cid: str) -> OrderSpec:
    return OrderSpec(instrument_id="SPY", side="buy", qty=1, order_type="market",
                     time_in_force="day", order_class="simple", client_order_id=cid,
                     extended_hours=False, short=False)


def bench_client(rows: List[str], latency_ms: float) -> None:
    instruments = _instruments()
    kill = tempfile.mkdtemp(prefix="hft-bench-kill-")
    for latency in (0.0, latency_ms / 1000.0):
        for name, workers in (("urlopen", 1), ("pool", 1), ("pool", 8), ("pool", 16),
                              ("pool", 32), ("pool", 64)):
            count = 200 if (latency == 0 or workers > 1) else 40
            with FakeAlpaca(latency_s=latency) as fake:
                kind = PooledTransport if name == "pool" else UrllibTransport
                client = AlpacaClient(
                    BrokerConfig(fake.api_key, fake.secret_key, PAPER),
                    kill_dir=kill, instruments=instruments,
                    gateway=GatewayConfig(pooled=name == "pool", workers=workers),
                    transport=kind(fake.base_url, 15),
                )
                client.ensure_account_verified()
                tag = f"{name}{workers}-{int(latency * 1000)}"
                calls = [lambda i=i: client.submit_order(_spec(f"b-{tag}-{i}"), mark=500.0)
                         for i in range(count)]
                started = time.perf_counter()
                outcomes = client.run_many(calls)
                elapsed = time.perf_counter() - started
                ok = sum(1 for res, err in outcomes if err is None and res["status"] == "submitted")
                rows.append(
                    f"| client {name} | {workers} hilos | {latency * 1000:.0f} ms | {ok}/{count} | - | "
                    f"- | - | {count / elapsed:.1f} | - |"
                )
                client.close()


def _engine_fake(rows: List[str], label: str, workers: int, per_bar: int, bars_n: int,
                 latency: float, rpm) -> None:
    config, session, bars = _setup()
    with FakeAlpaca(latency_s=latency, rpm_limit=rpm, stream=True) as fake:
        client = AlpacaClient(
            BrokerConfig(fake.api_key, fake.secret_key, PAPER),
            kill_dir=tempfile.mkdtemp(prefix="hft-bench-kill-"), instruments=_instruments(),
            gateway=GatewayConfig(pooled=True, workers=workers, trading_rpm=rpm),
            transport=PooledTransport(fake.base_url, 15),
        )
        stream = TradeUpdatesStream(fake.api_key, fake.secret_key, url=fake.stream_url)
        if not stream.start(5):
            raise RuntimeError("el stream falso no arrancó")
        execution = AlpacaExecution(client, CostModel.from_config(config),
                                    clock=lambda: datetime.now(timezone.utc), stream=stream)
        result = run_capacity(execution, config=config, session=session, bars=bars,
                              orders_per_bar=per_bar, mode="paper", max_bars=bars_n)
        throttled = fake.stats.throttled
        rows.append(
            f"| {label} | {workers} hilos, {per_bar}/barra, tope {rpm or '-'} | "
            f"{latency * 1000:.0f} ms | {result.submitted} | {result.fills} | "
            f"{result.pct(0.5) * 1000:.0f} | {result.pct(0.99) * 1000:.0f} | "
            f"{result.orders_per_s():.1f} | {result.fills_per_s():.1f} |"
            + (f" 429={throttled}" if throttled else "")
        )
        stream.stop()
        client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latency-ms", type=float, default=90.0,
                        help="RTT inyectado en el exchange falso (medido a paper-api: ~90 ms)")
    parser.add_argument("--with-limit", action="store_true",
                        help="incluye la ruta con tope de 200 req/min (~2.5 min)")
    parser.add_argument("--out", type=Path, default=None, help="archivo markdown de salida")
    args = parser.parse_args(argv)
    from src.settings import setup_logging

    setup_logging(logging.ERROR)
    latency = args.latency_ms / 1000.0
    rows = [
        "| capa | carga | RTT | enviadas | fills | p50 paso ms | p99 paso ms | órdenes/s | fills/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    bench_engine_sim(rows)
    bench_client(rows, args.latency_ms)
    _engine_fake(rows, "engine-fake", 16, 100, 6, latency, None)
    _engine_fake(rows, "engine-fake", 32, 200, 6, latency, None)
    _engine_fake(rows, "engine-fake", 64, 400, 6, latency, None)
    if args.with_limit:
        _engine_fake(rows, "engine-fake-200", 16, 150, 4, latency, 200)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"# Bench de capacidad ({stamp})\n\n" + "\n".join(rows) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
