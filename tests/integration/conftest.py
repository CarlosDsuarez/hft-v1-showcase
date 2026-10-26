"""
Fixtures de la suite de integración del runner vivo (sin red).

`live` arma un entorno completo: HFT_HOME en tmp_path (state_dir y
results_dir aislados), universo sintético, FakeMarket como única fuente de
datos (reloj = tras el cierre de la última barra) y, en modo broker, un
FakeBroker inyectado en `runner.make_broker_client`. En modo sim, construir un
cliente de broker es un error del test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

import scripts.run_periodic_rebalance as runner
from src.live_state import LOCK_NAME
from src.notifier import ALERTS_FILE_NAME
from src.position_ledger import PositionLedger
from src.settings import LOGS_DIRNAME
from tests.integration.fakes import START, TICKERS, FakeBroker, FakeMarket


class LiveEnv:
    """Runner vivo con fakes + helpers para inspeccionar state_dir."""

    def __init__(self, home: Path, market: FakeMarket, monkeypatch):
        self.home = home
        self.market = market
        self.universe: List[str] = list(TICKERS)
        self.broker: Optional[FakeBroker] = None
        self._monkeypatch = monkeypatch

    @property
    def state_dir(self) -> Path:
        return self.home / "data" / "live_state"

    @property
    def plans_dir(self) -> Path:
        return self.home / "results" / "plans"

    def connect_broker(self, **kwargs) -> FakeBroker:
        self.broker = FakeBroker(self.market, **kwargs)
        self._monkeypatch.setattr(runner, "make_broker_client",
                                  lambda _state_dir: self.broker)
        return self.broker

    def run(self, **kwargs) -> int:
        """`run_once` con la historia sintética y sin espera de fills."""
        kwargs.setdefault("start", START)
        kwargs.setdefault("poll_timeout", 0.0)
        return runner.run_once(**kwargs)

    def submit(self, **kwargs) -> int:
        return self.run(mode="broker", submit=True, **kwargs)

    # -- inspección -----------------------------------------------------
    def tree(self, all_files: bool = False) -> Dict[str, tuple]:
        """Archivos de estado bajo state_dir (incl. .staging): bytes + mtime.

        Por defecto excluye lo que toda corrida con lock escribe aunque aborte:
        `.lock` (pid del dueño) y `logs/` (runner.log + alerts.jsonl, fuera del
        manifiesto, ARQ-21). Las de solo lectura (dry-run, check-only) no los
        tocan → `all_files=True`.
        """
        if not self.state_dir.exists():
            return {}
        out = {}
        for p in sorted(self.state_dir.rglob("*")):
            rel = p.relative_to(self.state_dir)
            if not p.is_file():
                continue
            if not all_files and (p.name == LOCK_NAME or rel.parts[0] == LOGS_DIRNAME):
                continue
            out[str(rel)] = (p.read_bytes(), p.stat().st_mtime_ns)
        return out

    def alerts(self) -> List[dict]:
        path = self.state_dir / LOGS_DIRNAME / ALERTS_FILE_NAME
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def history(self) -> pd.DataFrame:
        return pd.read_csv(self.state_dir / "rebalance_history.csv")

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.state_dir / name)

    def ledger(self) -> PositionLedger:
        return runner.load_ledger(self.state_dir / "ledger.json")

    def manifest(self) -> dict:
        return json.loads((self.state_dir / "state_manifest.json").read_text())

    def last_rebalance(self) -> Optional[str]:
        path = self.state_dir / "last_rebalance.txt"
        return path.read_text().strip() if path.exists() else None

    def as_of(self) -> str:
        return self.market.t.date().isoformat()


@pytest.fixture
def market() -> FakeMarket:
    return FakeMarket()


@pytest.fixture
def live(tmp_path, monkeypatch, market) -> LiveEnv:
    home = tmp_path / "home"
    monkeypatch.setenv("HFT_HOME", str(home))
    monkeypatch.delenv("HFT_KILL", raising=False)
    env = LiveEnv(home, market, monkeypatch)
    monkeypatch.setattr(runner, "load_universe",
                        lambda *a, **k: pd.DataFrame({"ticker": list(env.universe)}))
    monkeypatch.setattr(runner, "make_market_data_provider", lambda *a, **k: market)
    monkeypatch.setattr(runner, "utcnow", market.now)

    def no_broker_in_sim(_state_dir):
        raise AssertionError("mode=sim no debe crear un cliente de broker")

    monkeypatch.setattr(runner, "make_broker_client", no_broker_in_sim)
    return env
