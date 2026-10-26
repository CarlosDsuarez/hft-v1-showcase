"""
test_repo_config.py — Guards del entorno reproducible (sin red).

Verifica: intérprete soportado, versión fijada en .python-version, locks con
hashes (requirements-core.in -> requirements-core.txt y requirements.in ->
requirements.txt), matriz CI, job core, .PHONY completo en Makefile,
Dockerfile no-root con la misma versión de Python, ruff mínimo y sin módulos
duplicados de src/ o tests/ en la raíz.
"""

import re
import sys
import tomllib
from pathlib import Path

# Root files the image must carry (README and the files the suite reads).
IDENTITY_FILES = ["README.md", "Makefile", "Dockerfile", ".env.example", ".github/workflows/ci.yml"]

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = ("3.11", "3.12")  # pandas 3 exige >= 3.11
EXPERIMENTAL = ("tail_risk_overlay", "risk_profiles")  # no conectados (README)


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _pinned_version() -> str:
    return _read(".python-version").strip()


def _lock_entries(name: str = "requirements.txt") -> list[str]:
    """Lock como líneas lógicas (une continuaciones `\\`)."""
    text = _read(name).replace("\\\n", " ")
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith(("#", "-"))]


def _direct_req_names(name: str) -> set[str]:
    """Nombres de paquetes directos, siguiendo ``-r`` (sin aflojar el conjunto)."""
    names: set[str] = set()
    for ln in _read(name).splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(("-r ", "--requirement ")):
            names |= _direct_req_names(s.split(None, 1)[1].strip())
            continue
        if s.startswith("-"):
            continue
        names.add(_pkg_name(s))
    return names


_HEAVY = {
    "plotly", "cvxpy", "scipy", "scikit-learn", "matplotlib", "pyarrow", "yfinance",
}
_LOCKS = ("requirements.txt", "requirements-core.txt")


def _pkg_name(entry: str) -> str:
    return re.split(r"[=<>!~;\s\[]", entry, maxsplit=1)[0].lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Intérprete
# ---------------------------------------------------------------------------
class TestPython:
    def test_interpreter_in_supported_range(self):
        assert (3, 11) <= sys.version_info[:2] <= (3, 12), (
            f"Python {sys.version.split()[0]} fuera de 3.11–3.12; "
            "recrear .venv (ver README)")

    def test_python_version_file_is_supported(self):
        assert _pinned_version() in SUPPORTED


# ---------------------------------------------------------------------------
# Lock de dependencias
# ---------------------------------------------------------------------------
class TestLock:
    def test_requirements_in_keeps_plotly(self):
        # El -r se resuelve: numpy/pandas/pytest siguen en el conjunto directo.
        resolved = _direct_req_names("requirements.in")
        assert {"plotly", "numpy", "pandas", "cvxpy", "pytest"} <= resolved
        research = resolved - _direct_req_names("requirements-core.in")
        assert {"plotly", "cvxpy"} <= research

    def test_core_requirements_stay_light(self):
        core = _direct_req_names("requirements-core.in")
        assert {"numpy", "pandas", "pytest", "ruff"} <= core
        assert not core & _HEAVY, core & _HEAVY

    def test_lock_is_pinned_with_hashes(self):
        for name in _LOCKS:
            entries = _lock_entries(name)
            assert entries, f"{name} vacío"
            for entry in entries:
                req = entry.split("--hash")[0]
                assert "==" in req, f"{name} sin versión exacta: {req.strip()}"
                assert "--hash=sha256:" in entry, f"{name} sin hash: {req.strip()}"

    def test_lock_contains_direct_deps(self):
        full = {_pkg_name(e) for e in _lock_entries("requirements.txt")}
        core_lock = {_pkg_name(e) for e in _lock_entries("requirements-core.txt")}
        direct = _direct_req_names("requirements.in")
        core_direct = _direct_req_names("requirements-core.in")
        assert direct <= full, f"faltan en requirements.txt: {direct - full}"
        assert core_direct <= core_lock, (
            f"faltan en requirements-core.txt: {core_direct - core_lock}")
        assert core_direct <= full, f"el lock completo no cubre el núcleo: {core_direct - full}"

    def test_core_lock_excludes_research_stack(self):
        names = {_pkg_name(e) for e in _lock_entries("requirements-core.txt")}
        assert not names & _HEAVY, names & _HEAVY


# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------
class TestCI:
    ci = property(lambda self: _read(".github/workflows/ci.yml"))

    def test_matrix_covers_supported_versions(self):
        matrices = re.findall(r"python-version:\s*\[([^\]]*)\]", self.ci)
        assert matrices, "sin matriz python-version"
        versions = set(re.findall(r"\d+\.\d+", " ".join(matrices)))
        assert set(SUPPORTED) <= versions

    def test_deps_cached_and_hash_locked(self):
        # uv cachea las ruedas (setup-uv en GitHub; ~/.cache/uv en self-hosted)
        assert "astral-sh/setup-uv@" in self.ci
        assert "cache-dependency-glob" in self.ci
        installs = re.findall(r"pip install[^\n]*", self.ci)
        assert installs and all("--require-hashes" in i for i in installs)

    def test_has_integration_job(self):
        assert re.search(r"^\s{2}integration:\s*$", self.ci, re.M)

    def test_core_job_installs_only_core_lock(self):
        match = re.search(r"^  core:\n(.*?)(?=^  \S|\Z)", self.ci, re.M | re.S)
        assert match, "sin job core"
        body = match.group(1)
        assert "requirements-core.txt" in body
        assert "requirements.txt" not in body.replace("requirements-core.txt", "")
        assert "test-core" in body
        assert "tests/hft/test_core_imports.py" in body

    def test_runs_compile_pytest_and_ruff(self):
        for cmd in ("compileall", "pytest", "ruff check"):
            assert cmd in self.ci, f"CI no corre {cmd}"


# ---------------------------------------------------------------------------
# Makefile
# ---------------------------------------------------------------------------
class TestMakefile:
    def test_all_targets_are_phony(self):
        text = _read("Makefile").replace("\\\n", " ")  # une continuaciones
        targets = {m.group(1) for m in
                   re.finditer(r"^([A-Za-z0-9_-]+):(?!=)", text, re.M)}
        phony = set()
        for m in re.finditer(r"^\.PHONY:(.*)$", text, re.M):
            phony |= set(m.group(1).split())
        assert targets, "sin targets"
        assert targets <= phony, f"targets fuera de .PHONY: {targets - phony}"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------
class TestDockerfile:
    df = property(lambda self: _read("Dockerfile"))

    def test_base_image_matches_pinned_python(self):
        m = re.search(r"^FROM\s+python:(\d+\.\d+)", self.df, re.M)
        assert m and m.group(1) == _pinned_version()

    def test_pinned_python_is_in_ci_matrix(self):
        assert f'"{_pinned_version()}"' in _read(".github/workflows/ci.yml")

    def test_runs_as_non_root(self):
        users = re.findall(r"^USER\s+(\S+)", self.df, re.M)
        assert users, "Dockerfile sin USER"
        assert users[-1] not in ("root", "0", "0:0")

    def _copied_sources(self) -> set[str]:
        """Orígenes de todos los COPY (sin flags ni destino)."""
        flat = self.df.replace("\\\n", " ")
        sources: set[str] = set()
        for ln in flat.splitlines():
            if ln.startswith("COPY"):
                args = [a for a in ln.split()[1:] if not a.startswith("--")]
                sources |= set(args[:-1])
        return sources

    def test_copies_repo_files_read_by_suite(self):
        # El build corre la suite: test_settings.py lee .env.example y
        # The suite reads these root files; if they are not in the image the build fails.
        required = [".env.example", *IDENTITY_FILES]
        sources = self._copied_sources()
        dirs = [s for s in sources if s.endswith("/")]
        missing = [p for p in required
                   if p not in sources and not any(p.startswith(d) for d in dirs)]
        assert not missing, f"Dockerfile no copia archivos que lee la suite: {missing}"
        ignore = ROOT / ".dockerignore"
        if ignore.exists():  # no se copia a la imagen: solo se valida en el repo
            ignored = set(ignore.read_text(encoding="utf-8").split())
            assert not ignored & set(required), ignored & set(required)


# ---------------------------------------------------------------------------
# Ruff (solo errores, no formato)
# ---------------------------------------------------------------------------
class TestRuff:
    def test_minimal_error_only_config(self):
        cfg = tomllib.loads(_read("pyproject.toml"))
        select = cfg["tool"]["ruff"]["lint"]["select"]
        assert select
        # E9 = errores de sintaxis/IO; F = pyflakes. Nada de estilo (E1-E7, W, I...).
        assert all(code.startswith(("E9", "F")) for code in select), select


# ---------------------------------------------------------------------------
# Layout: src.* es canónico (AGENTS.md)
# ---------------------------------------------------------------------------
class TestLayout:
    def test_no_root_modules_shadow_src_or_tests(self):
        # Copias sueltas en la raíz divergen en silencio de src/ y tests/
        # (data_client.py raíz saltaba src.market_data).
        canonical = {p.name for d in ("src", "tests") for p in (ROOT / d).glob("*.py")}
        shadows = sorted(p.name for p in ROOT.glob("*.py") if p.name in canonical)
        assert not shadows, f"duplicados en la raíz (usar src.*/tests/): {shadows}"

    def test_experimental_modules_unwired_and_documented(self):
        # Ninguna ruta de ejecución (src/, scripts/, experiments/) los importa
        # y README los lista como experimentales. Al integrar uno (p. ej. vía
        # StrategyConfig), sacarlo de EXPERIMENTAL y de esa sección del README.
        code = {p: p.read_text(encoding="utf-8")
                for d in ("src", "scripts", "experiments")
                for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts}
        readme = _read("README.md")
        assert "### Experimental modules" in readme
        section = readme.split("### Experimental modules", 1)[1].split("\n## ", 1)[0]
        for mod in EXPERIMENTAL:
            importers = sorted(str(p.relative_to(ROOT)) for p, src in code.items()
                               if p.stem != mod and re.search(rf"\bsrc\.{mod}\b", src))
            assert not importers, f"{mod} ya está conectado: {importers}"
            assert f"src/{mod}.py" in section, mod
            assert "EXPERIMENTAL" in (ROOT / "src" / f"{mod}.py").read_text(encoding="utf-8")
