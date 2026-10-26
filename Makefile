# HFT-V1 — atajos operativos (requiere .venv/; ver README).
# Versión de Python fijada en .python-version (3.12); soportado 3.11–3.12.
PYTHON ?= python$(shell cat .python-version)
PY := .venv/bin/python
PIP := .venv/bin/pip
PYCORE := .venv-core/bin/python
# tests/hft más la infraestructura reutilizable que no importa el stack research.
CORE_TESTS := tests/hft \
	tests/test_metrics_var_parametric.py \
	tests/test_settings.py \
	tests/test_instruments.py \
	tests/test_fx.py \
	tests/test_multicurrency.py \
	tests/test_position_ledger.py \
	tests/test_ledger_transactional.py \
	tests/test_portfolio_state.py \
	tests/test_alpaca_adapter.py \
	tests/test_broker_idempotency.py \
	tests/test_broker_gateway.py \
	tests/test_pre_trade_guards.py \
	tests/test_live_state.py \
	tests/test_regime_detector.py \
	tests/test_exit_rules.py \
	tests/test_signal_stack.py \
	tests/test_conviction_scoring.py \
	tests/test_rebalancing.py \
	tests/test_universe.py \
	tests/test_universe_mixed.py \
	tests/test_data_pipeline.py \
	tests/test_repo_config.py
# Estos nodos importan el runner o el backtest (→ hrp_sizing → scipy).
CORE_DESELECT := \
	--deselect tests/test_settings.py::TestNoDuplicatedConstants::test_modules_share_settings_values \
	--deselect tests/test_settings.py::TestNoDuplicatedConstants::test_backtest_exit_defaults_come_from_exit_rules \
	--deselect tests/test_settings.py::TestAnchoredDefaults::test_mixed_sleeve_config_from_other_cwd \
	--deselect tests/test_settings.py::TestAnchoredDefaults::test_report_default_outputs_in_results_dir \
	--deselect tests/test_broker_idempotency.py::TestRunBrokerStepAudit \
	--deselect tests/test_pre_trade_guards.py::TestRunBrokerStepGuard \
	--deselect tests/test_pre_trade_guards.py::TestRunOnceExitCode \
	--deselect tests/test_exit_rules.py::TestBacktestWiring

.PHONY: help install install-core lock lint test test-integration test-hft test-core compile demo \
        integration dashboard report probe broker-dry live-report docker-build docker-test \
        hft-download hft-backtest hft-signals hft-paper-dry hft-paper hft-report \
        hft-paper-daily hft-schedule-install hft-schedule-uninstall hft-schedule-status

help:
	@echo "install      crea .venv con $(PYTHON) e instala el lock (requirements.txt)"
	@echo "install-core crea .venv-core e instala solo requirements-core.txt"
	@echo "lock         regenera requirements-core.txt y requirements.txt (hashes; requiere uv)"
	@echo "lint         ruff check (solo errores; config en pyproject.toml)"
	@echo "test         suite completa offline"
	@echo "test-integration  runner vivo con fakes (pytest -m integration; lista xfail ARQ-*)"
	@echo "test-hft         suite de src/hft (pytest tests/hft -q)"
	@echo "test-core        tests/hft + infraestructura ligera, con .venv-core"
	@echo "compile      chequeo de sintaxis src/experiments/tests"
	@echo "integration  demos sinteticos end-to-end (dashboard, reporte mensual)"
	@echo "dashboard    dashboard sintetico -> results/dashboard.html"
	@echo "report       reporte mensual sintetico -> results/report.html"
	@echo "probe        probe Yahoo universo CO -> research/"
	@echo "broker-dry   plan sim + paper sin enviar ni persistir -> results/plans/ (requiere .env)"
	@echo "live-report  dashboard del modo vivo (live_state) -> results/live_dashboard.html (red: precios)"
	@echo "docker-build construye imagen (requiere docker)"
	@echo "docker-test  corre la suite dentro de docker"
	@echo "hft-download barras Alpaca al BarStore (red; HFT_START HFT_END; opc. HFT_SYMBOLS HFT_FEED)"
	@echo "hft-backtest backtest sobre el almacén local (HFT_START HFT_END; opc. HFT_FEED)"
	@echo "hft-signals  sesión en modo signals, sin POST (opc. HFT_DATE; default hoy NY)"
	@echo "hft-paper-dry paper sin --submit: journal, cero POST (opc. HFT_DATE; default hoy NY)"
	@echo "hft-paper    paper CON --submit: envía órdenes a Alpaca paper (opc. HFT_DATE; default hoy NY)"
	@echo "hft-report   HTML del intradía (HFT_DATE, o HFT_START y HFT_END)"
	@echo "hft-paper-daily paper CON --submit de hoy; relanza si cae la red; caffeinate"
	@echo "hft-schedule-install LaunchAgent: hft-paper-daily lun-vie a HFT_AT (default 08:15 local)"
	@echo "hft-schedule-uninstall / hft-schedule-status quitar / ver el LaunchAgent"

install:
	$(PYTHON) -m venv .venv && $(PIP) install -U pip && $(PIP) install -r requirements.txt

install-core:
	$(PYTHON) -m venv .venv-core && $(PYCORE) -m pip install -U pip \
		&& $(PYCORE) -m pip install -r requirements-core.txt

# Locks universales (macOS/Linux/Windows) resueltos para el piso soportado (3.11),
# válidos en 3.11–3.12. El completo incluye -r requirements-core.in.
lock:
	uv pip compile requirements-core.in --universal --python-version 3.11 \
		--generate-hashes -o requirements-core.txt
	uv pip compile requirements.in --universal --python-version 3.11 \
		--generate-hashes -o requirements.txt

lint:
	$(PY) -m ruff check src experiments tests scripts

test:
	$(PY) -m pytest tests/ -q

test-integration:
	$(PY) -m pytest tests/ -q -m integration -rxX

test-hft:
	$(PY) -m pytest tests/hft -q

test-core:
	$(PYCORE) -m pytest $(CORE_TESTS) $(CORE_DESELECT) -q

compile:
	$(PY) -m compileall src experiments tests

# Demos end-to-end sin red ni data/ (los de estimación, p. ej. src.rolling_engine,
# también son sintéticos y los ejecuta tests/test_demos.py).
integration:
	$(PY) -m src.dashboard_results
	$(PY) -m src.monthly_report
	test -s results/dashboard.html && test -s results/report.html

# Intraday engine end to end on synthetic 1-minute bars (no network, no data/).
demo:
	$(PY) scripts/demo_synthetic.py

dashboard:
	$(PY) -m src.dashboard_results

report:
	$(PY) -m src.monthly_report

probe:
	$(PY) research/probe_tickers.py --start 2019-01-01 --end 2026-01-01

broker-dry:
	$(PY) -m scripts.run_periodic_rebalance --mode broker --dry-run

live-report:
	$(PY) -m src.live_report

docker-build:
	docker build --build-arg GIT_SHA=$$(git rev-parse HEAD 2>/dev/null || echo unknown) \
		-t hft-v1:latest .

docker-test:
	docker run --rm hft-v1:latest

# Atajos del intradía. La suite no los invoca. hft-download, hft-signals y
# hft-paper* usan red. Fecha de sesión: HFT_DATE o hoy en Nueva York. Sin
# --date, --mode paper a secas cae en la guarda HFT-09E (exit 2) y no abre sesión.
HFT_SESSION_DATE = $(or $(HFT_DATE),$(shell TZ=America/New_York date +%F))

hft-download:
	@test -n "$(HFT_START)" && test -n "$(HFT_END)" || { echo "uso: make hft-download HFT_START=YYYY-MM-DD HFT_END=YYYY-MM-DD"; exit 2; }
	$(PY) -m src.hft.data.download --start $(HFT_START) --end $(HFT_END) $(if $(HFT_SYMBOLS),--symbols $(HFT_SYMBOLS),) $(if $(HFT_FEED),--feed $(HFT_FEED),)

hft-backtest:
	@test -n "$(HFT_START)" && test -n "$(HFT_END)" || { echo "uso: make hft-backtest HFT_START=YYYY-MM-DD HFT_END=YYYY-MM-DD"; exit 2; }
	$(PY) -m src.hft.backtest --start $(HFT_START) --end $(HFT_END) $(if $(HFT_FEED),--feed $(HFT_FEED),)

hft-signals:
	$(PY) -m scripts.run_intraday --mode signals --date $(HFT_SESSION_DATE)

hft-paper-dry:
	$(PY) -m scripts.run_intraday --mode paper --date $(HFT_SESSION_DATE)

hft-paper:
	$(PY) -m scripts.run_intraday --mode paper --submit --date $(HFT_SESSION_DATE)

# Paper de hoy con relanzamiento por red/datos y caffeinate (scripts/hft_paper_daily.py).
hft-paper-daily:
	$(PY) -m scripts.hft_paper_daily run

# LaunchAgent de macOS: hft-paper-daily de lunes a viernes a HFT_AT (hora local, default 08:15).
# Instalar desde el checkout principal, no desde un worktree.
hft-schedule-install:
	$(PY) -m scripts.hft_paper_daily install $(if $(HFT_AT),--at $(HFT_AT),)

hft-schedule-uninstall:
	$(PY) -m scripts.hft_paper_daily uninstall

hft-schedule-status:
	$(PY) -m scripts.hft_paper_daily status

hft-report:
	@if [ -n "$(HFT_DATE)" ]; then \
		$(PY) -m src.hft.report --date $(HFT_DATE); \
	elif [ -n "$(HFT_START)" ] && [ -n "$(HFT_END)" ]; then \
		$(PY) -m src.hft.report --start $(HFT_START) --end $(HFT_END); \
	else \
		echo "uso: make hft-report HFT_DATE=YYYY-MM-DD"; \
		echo "  o: make hft-report HFT_START=YYYY-MM-DD HFT_END=YYYY-MM-DD"; \
		exit 2; \
	fi
