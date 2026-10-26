"""
test_settings.py — Configuracion centralizada y rutas ancladas.

Todo offline: tmp_path para HFT_HOME y cwd; nunca toca .env ni
data/live_state reales.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def other_cwd(tmp_path, monkeypatch):
    """cwd ajeno al repo (sin .env, sin data/, sin universe*.csv)."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


class TestSettingsPaths:
    def test_paths_anchored_to_repo_not_cwd(self, other_cwd, monkeypatch):
        from src.settings import load_settings
        monkeypatch.delenv("HFT_HOME", raising=False)
        s = load_settings()
        assert s.repo_root == REPO_ROOT
        assert s.data_dir == REPO_ROOT / "data"
        assert s.state_dir == REPO_ROOT / "data" / "live_state"
        assert s.results_dir == REPO_ROOT / "results"
        for p in (s.repo_root, s.data_dir, s.state_dir, s.results_dir):
            assert p.is_absolute()
            assert other_cwd not in p.parents

    def test_hft_home_overrides_runtime_dirs(self, tmp_path, other_cwd, monkeypatch):
        from src.settings import load_settings
        home = tmp_path / "home"
        monkeypatch.setenv("HFT_HOME", str(home))
        s = load_settings()
        assert s.data_dir == home / "data"
        assert s.state_dir == home / "data" / "live_state"
        assert s.results_dir == home / "results"
        # Codigo y universos siguen en el repo
        assert s.repo_root == REPO_ROOT

    def test_relative_hft_home_is_rejected(self, monkeypatch):
        from src.settings import load_settings
        monkeypatch.setenv("HFT_HOME", "relative/dir")
        with pytest.raises(ValueError, match="HFT_HOME"):
            load_settings()

    def test_default_values_unchanged(self):
        from src.settings import load_settings
        s = load_settings(env={})
        assert s.seed == 42
        assert s.base_currency == "USD"
        assert s.initial_cash == 1_000_000.0
        assert s.broker_env == "paper"
        assert s.trading_days == 252
        assert s.lookback_days == 105


class TestNoDuplicatedConstants:
    """SEED/TRADING_DAYS/INITIAL_CASH/105 solo se definen en src/settings.py."""

    _PATTERNS = [
        re.compile(r"^\s*_?\w*SEED\s*=\s*\d+", re.M),
        re.compile(r"^\s*TRADING_DAYS\s*=\s*\d+", re.M),
        re.compile(r"1_000_000(\.0)?\b"),
        re.compile(r"\b105\b"),
    ]

    def _offenders(self):
        out = []
        files = [*(REPO_ROOT / "src").rglob("*.py"),
                 *(REPO_ROOT / "scripts").glob("*.py"),
                 *(REPO_ROOT / "experiments").glob("*.py")]
        for f in files:
            if f.name == "settings.py" or "__pycache__" in f.parts:
                continue
            code = "\n".join(
                ln for ln in f.read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#"))
            for pat in self._PATTERNS:
                for m in pat.finditer(code):
                    out.append(f"{f.relative_to(REPO_ROOT)}: {m.group(0).strip()}")
        return out

    def test_no_literal_duplicates_outside_settings(self):
        assert self._offenders() == []

    def test_modules_share_settings_values(self):
        from src import settings
        from src import backtest_engine, data_client, position_ledger
        from scripts import run_periodic_rebalance as runner
        assert backtest_engine.SEED == settings.SEED
        assert backtest_engine.BacktestConfig().initial_nav == settings.INITIAL_CASH
        assert position_ledger.PositionLedger().cash == settings.INITIAL_CASH
        assert data_client.DEFAULT_LOOKBACK_DAYS == settings.LOOKBACK_DAYS
        assert runner.SEED == settings.SEED

    def test_backtest_exit_defaults_come_from_exit_rules(self):
        from src.backtest_engine import BacktestConfig
        from src.exit_rules import ExitConfig
        b, e = BacktestConfig(), ExitConfig()
        assert (b.stop_loss_pct, b.take_profit_pct, b.trailing_window) == (
            e.stop_loss_pct, e.take_profit_pct, e.trailing_window)


class TestDotenv:
    def test_read_dotenv_returns_dict_without_mutating_environ(self, tmp_path, monkeypatch):
        from src.settings import read_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("# c\nHFT_TEST_KEY='abc'\nBAD LINE\nOTHER=1\n")
        monkeypatch.delenv("HFT_TEST_KEY", raising=False)
        values = read_dotenv(env_file)
        assert values == {"HFT_TEST_KEY": "abc", "OTHER": "1"}
        assert "HFT_TEST_KEY" not in os.environ

    def test_read_dotenv_missing_file_is_empty(self, tmp_path):
        from src.settings import read_dotenv
        assert read_dotenv(tmp_path / "nope.env") == {}

    def test_cwd_dotenv_is_ignored(self, other_cwd, monkeypatch):
        """Un .env en el cwd no debe filtrarse a la config del broker."""
        import src.settings as settings_mod
        from src.brokers.alpaca_adapter import load_broker_config
        (other_cwd / ".env").write_text(
            "BROKER_API_KEY=cwdkey\nBROKER_SECRET_KEY=cwdsecret\n")
        # repo_root sin .env: apuntamos la busqueda a un dir vacio
        empty_root = other_cwd.parent / "fake_root"
        empty_root.mkdir()
        monkeypatch.setattr(settings_mod, "REPO_ROOT", empty_root)
        for k in ("BROKER_API_KEY", "BROKER_SECRET_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(ValueError, match="Faltan"):
            load_broker_config()
        assert "BROKER_API_KEY" not in os.environ

    def test_broker_config_reads_repo_dotenv_without_mutating_environ(
            self, tmp_path, monkeypatch):
        import src.settings as settings_mod
        from src.brokers.alpaca_adapter import load_broker_config
        root = tmp_path / "root"
        root.mkdir()
        (root / ".env").write_text(
            "BROKER_API_KEY=PKFILE\nBROKER_SECRET_KEY=s3cr3t\n"
            "BROKER_BASE_URL=https://paper-api.alpaca.markets\n")
        monkeypatch.setattr(settings_mod, "REPO_ROOT", root)
        for k in ("BROKER_API_KEY", "BROKER_SECRET_KEY", "BROKER_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        cfg = load_broker_config()
        assert cfg.api_key == "PKFILE"
        assert "BROKER_API_KEY" not in os.environ

    def test_real_environment_wins_over_dotenv(self, tmp_path, monkeypatch):
        import src.settings as settings_mod
        from src.brokers.alpaca_adapter import load_broker_config
        root = tmp_path / "root"
        root.mkdir()
        (root / ".env").write_text("BROKER_API_KEY=PKFILE\nBROKER_SECRET_KEY=x\n")
        monkeypatch.setattr(settings_mod, "REPO_ROOT", root)
        monkeypatch.setenv("BROKER_API_KEY", "PKENV")
        monkeypatch.delenv("BROKER_BASE_URL", raising=False)
        assert load_broker_config().api_key == "PKENV"


class TestEnvExample:
    def test_every_key_is_read_by_code(self):
        keys = [
            ln.lstrip("# ").split("=", 1)[0].strip()
            for ln in (REPO_ROOT / ".env.example").read_text().splitlines()
            if re.match(r"^#?\s*[A-Z][A-Z0-9_]*=", ln)
        ]
        code = "\n".join(
            f.read_text(encoding="utf-8")
            for d in ("src", "scripts") for f in (REPO_ROOT / d).rglob("*.py"))
        unused = [k for k in keys if f'"{k}"' not in code]
        assert unused == []


class TestAnchoredDefaults:
    def test_fetch_prices_csv_fallback_uses_data_dir(self, tmp_path, other_cwd, monkeypatch):
        import src.data_client as dc
        home = tmp_path / "home"
        (home / "data").mkdir(parents=True)
        idx = pd.bdate_range("2024-01-01", periods=5)
        pd.DataFrame({"date": idx, "AAA": np.arange(5.0) + 1}).to_csv(
            home / "data" / "raw_prices_backup.csv", index=False)
        monkeypatch.setenv("HFT_HOME", str(home))

        def yahoo_down(self, *a, **k):
            raise ConnectionError("offline")

        monkeypatch.setattr("src.market_data.YahooProvider.get_bars", yahoo_down)
        out = dc.fetch_prices(["AAA"], start="2024-01-01", end="2024-01-31")
        assert len(out) == 5

    def test_mixed_sleeve_config_from_other_cwd(self, other_cwd):
        from src.risk_profiles import mixed_sleeve_config
        cfg = mixed_sleeve_config("moderado")
        assert cfg.groups

    def test_report_default_outputs_in_results_dir(self, tmp_path, other_cwd, monkeypatch):
        from src.monthly_report import default_report_path
        from src.dashboard_results import default_dashboard_path
        home = tmp_path / "home"
        monkeypatch.setenv("HFT_HOME", str(home))
        assert default_report_path() == home / "results" / "report.html"
        assert default_dashboard_path() == home / "results" / "dashboard.html"
