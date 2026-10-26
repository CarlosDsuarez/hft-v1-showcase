"""Capacidad de órdenes: probe por la ruta única, envío en paralelo y fills por stream.

Contra ``SimExecution`` y contra el exchange falso de loopback. Ningún
test abre red externa.
"""

from __future__ import annotations

import tempfile
import time
from datetime import date, datetime, time as dtime, timedelta, timezone

import pytest

from src.brokers.alpaca_adapter import AlpacaClient, BrokerConfig, GatewayConfig
from src.brokers.alpaca_stream import TradeUpdatesStream
from src.brokers.fake_alpaca import FakeAlpaca
from src.brokers.http_transport import PooledTransport
from src.hft.calendar import session_at
from src.hft.capacity import PROBE_ID, CapacityProbe, probe_config, run_capacity
from src.hft.config import load_hft_config
from src.hft.costs import CostModel
from src.hft.data.synthetic import synthetic_bars
from src.hft.execution.alpaca_exec import AlpacaExecution
from src.hft.execution.sim import SimExecution
from src.hft.strategies.base import STRATEGIES
from src.hft.types import order_intent
from src.instruments import Instrument, InstrumentMaster

UTC = timezone.utc
PAPER = "https://paper-api.alpaca.markets"
SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")
T0 = datetime(2026, 10, 5, 14, 30, tzinfo=UTC)


def _instruments() -> InstrumentMaster:
    return InstrumentMaster([
        Instrument(id=s, yahoo_symbol=s, broker_symbol=s, market="US", currency="USD",
                   asset_type="etf", sector="test", tradable_brokers=("alpaca",))
        for s in SYMBOLS
    ])


def _client(fake: FakeAlpaca, workers: int) -> AlpacaClient:
    return AlpacaClient(
        BrokerConfig(fake.api_key, fake.secret_key, PAPER),
        kill_dir=tempfile.mkdtemp(), instruments=_instruments(),
        gateway=GatewayConfig(pooled=True, workers=workers),
        transport=PooledTransport(fake.base_url, 5),
    )


def _costs() -> CostModel:
    return CostModel.from_config(load_hft_config())


def _intents(ts: datetime, n: int = 8):
    return [order_intent("probe", SYMBOLS[i % len(SYMBOLS)], "buy", 1, ts, seq=i)
            for i in range(n)]


def _wait(predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestProbe:
    def test_no_se_registra(self):
        CapacityProbe(SYMBOLS, 4)
        assert PROBE_ID not in STRATEGIES

    def test_config_de_produccion_intacta(self):
        base = load_hft_config()
        probe = probe_config(base)
        assert base.risk.max_orders_per_minute == 8
        assert base.risk.max_orders_per_day == 100
        assert probe.risk.max_orders_per_minute > 10**6

    def test_ruta_unica_con_simulador(self):
        config = probe_config(load_hft_config())
        session = session_at(date(2025, 3, 3), dtime(9, 30), dtime(16, 0))
        symbols = list(config.universe)[:4]
        bars = synthetic_bars([session], symbols, start_price=200.0)
        sim = SimExecution(CostModel.from_config(config), partial_prob=0.0, participation_cap=1.0,
                           adv={s: 1e9 for s in symbols}, sigma={s: 0.01 for s in symbols})
        result = run_capacity(sim, config=config, session=session, bars=bars,
                              orders_per_bar=8, max_bars=6)
        assert result.rejected == 0
        assert result.intents == 8 * 6
        # La barra t se ejecuta en t+1: la última barra queda pendiente.
        assert result.submitted == 8 * 5
        assert result.fills == result.submitted


class TestParallelSubmit:
    def test_journal_en_orden_de_intencion(self):
        with FakeAlpaca(latency_s=0.01) as fake:
            client = _client(fake, workers=8)
            execution = AlpacaExecution(client, _costs(), clock=lambda: T0)
            intents = _intents(T0)
            execution.submit(intents, now=T0)
            journal = [row["client_order_id"] for row in execution.journal]
            expected = [f"h-probe-202610051430-{i.symbol}-buy" + (f"-{n}" if n else "")
                        for n, i in enumerate(intents)]
            assert journal == expected
            assert fake.stats.posts == 8
            assert all(row["status"] == "submitted" for row in execution.journal)
            client.close()

    def test_un_fallo_deshace_el_lote(self):
        with FakeAlpaca(fill_mode="never", reject_symbols=("IWM",)) as fake:
            client = _client(fake, workers=8)
            execution = AlpacaExecution(client, _costs(), clock=lambda: T0)
            with pytest.raises(Exception):
                execution.submit(_intents(T0), now=T0)
            accepted = list(fake.state.orders.values())
            assert accepted, "las demás órdenes sí llegaron al broker"
            assert all(o["status"] == "canceled" for o in accepted)
            assert execution.open_orders() == ()
            client.close()

    def test_un_worker_es_el_lazo_de_siempre(self):
        with FakeAlpaca() as fake:
            client = _client(fake, workers=1)
            execution = AlpacaExecution(client, _costs(), clock=lambda: T0)
            fills = execution.submit(_intents(T0, 4), now=T0)
            assert sum(f.qty for f in fills) == 4
            client.close()


class TestStreamFills:
    def test_fills_por_stream_sin_doble_conteo(self):
        with FakeAlpaca(stream=True) as fake:
            client = _client(fake, workers=4)
            stream = TradeUpdatesStream(fake.api_key, fake.secret_key, url=fake.stream_url)
            assert stream.start(5)
            moments = {"now": T0}
            execution = AlpacaExecution(client, _costs(), clock=lambda: moments["now"],
                                        stream=stream, reconcile_interval_s=60.0)
            total = sum(f.qty for f in execution.submit(_intents(T0, 4), now=T0))
            assert _wait(lambda: stream.received >= 8)
            moments["now"] = T0 + timedelta(seconds=10)
            total += sum(f.qty for f in execution.poll())
            assert total == 4
            reads_after_first = fake.stats.by_path.get("GET /v2/account/activities", 0)
            later = T0 + timedelta(seconds=20)
            total += sum(f.qty for f in execution.submit(_intents(later, 4), now=later))
            assert _wait(lambda: stream.received >= 16)
            moments["now"] = T0 + timedelta(seconds=30)
            total += sum(f.qty for f in execution.poll())
            assert total == 8
            # Dentro del intervalo no hubo otra lectura REST de actividades.
            assert fake.stats.by_path.get("GET /v2/account/activities", 0) == reads_after_first
            assert execution.open_orders() == ()
            stream.stop()
            client.close()

    def test_reconexion_fuerza_reconciliacion(self):
        with FakeAlpaca(stream=True) as fake:
            client = _client(fake, workers=2)
            stream = TradeUpdatesStream(fake.api_key, fake.secret_key, url=fake.stream_url,
                                        max_backoff_s=0.5)
            assert stream.start(5)
            moments = {"now": T0}
            execution = AlpacaExecution(client, _costs(), clock=lambda: moments["now"],
                                        stream=stream, reconcile_interval_s=3600.0)
            execution.submit(_intents(T0, 2), now=T0)
            reads = fake.stats.by_path.get("GET /v2/account/activities", 0)
            fake.drop_stream_clients()
            assert _wait(lambda: stream.connections >= 2, timeout=10)
            moments["now"] = T0 + timedelta(seconds=5)
            execution.poll()
            assert fake.stats.by_path.get("GET /v2/account/activities", 0) == reads + 1
            stream.stop()
            client.close()
