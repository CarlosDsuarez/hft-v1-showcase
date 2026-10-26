# HFT-V1 — imagen reproducible (Python 3.12 slim, CPU; = .python-version, en matriz CI).
# Construye sin credenciales: tests y demos sinteticas son offline.
# Live (yfinance/Alpaca) requiere red + .env en runtime, no en build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Deps del sistema para scipy/cvxpy (compilacion si algun wheel falta)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios; /app le pertenece (pytest/compileall escriben caches)
RUN useradd --create-home --uid 10001 app && chown app:app /app

# Lock con hashes: pip exige --require-hashes automaticamente
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app experiments/ ./experiments/
COPY --chown=app:app scripts/ ./scripts/
# Estrategia viva versionada (src/strategy.DEFAULT_CONFIG_PATH)
COPY --chown=app:app config/ ./config/
COPY --chown=app:app universe.csv universe_co.csv universe_mixed.csv universe_hft.csv universe_hft_pairs.csv ./
COPY --chown=app:app research/ ./research/
# Config que validan tests/test_repo_config.py y tests/test_settings.py
COPY --chown=app:app .python-version requirements.in requirements.txt \
     requirements-core.in requirements-core.txt pyproject.toml \
     Makefile Dockerfile .env.example ./
COPY --chown=app:app .github/ ./.github/
# Docs que valida tests/test_project_identity.py
COPY --chown=app:app README.md LICENSE ./

# Sin .git en la imagen: el SHA llega como build-arg (make docker-build) y
# src/provenance.code_version lo lee de HFT_GIT_SHA (git_sha en history).
# Al final para no invalidar la cache de las capas anteriores.
ARG GIT_SHA=unknown
ENV HFT_GIT_SHA=$GIT_SHA

USER app

# Smoke test en build: sintaxis + suite offline (toda sintetica/local)
RUN python -m compileall src experiments tests \
 && python -m pytest tests/ -q

CMD ["python", "-m", "pytest", "tests/", "-q"]
