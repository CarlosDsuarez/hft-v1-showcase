"""Guardarraíles de arquitectura de src/hft (HFT-01).

Offline, por AST. El paquete puede estar vacío: las reglas se comprueban
sobre el árbol real y, en tmp_path, sobre un módulo que debe hacerlas fallar.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
HFT_DIR = SRC_DIR / "hft"

# Dependencias que src/hft no puede arrastrar, ni directo ni vía src.*.
FORBIDDEN_ROOTS = frozenset({
    "cvxpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "torch",
    "tensorflow",
    "xgboost",
    "lightgbm",
    "openai",
    "anthropic",
    "langchain",
    "transformers",
    "yfinance",
    "plotly",
    "matplotlib",
    "pyarrow",
})

NETWORK_ROOTS = frozenset({"urllib", "socket", "http"})

# Llamadas de reloj de pared. ``datetime.datetime.now`` cubre
# ``from datetime import datetime; datetime.now()``.
CLOCK_CALLS = frozenset({
    "datetime.now",
    "datetime.datetime.now",
    "time.time",
})

# Las mismas prohibiciones que tests/test_no_global_side_effects.py.
GLOBAL_CALLS = frozenset({
    "numpy.random.seed",
    "numpy.random.RandomState",
    "logging.basicConfig",
})

_BARE_ROOTS = {
    "datetime": "datetime",
    "time": "time",
    "logging": "logging",
    "numpy": "numpy",
    "np": "numpy",
}

_DOC_FRASES = (
    "1–5 min",
    "Alpaca paper",
    "plano al cierre",
    "market making real",
    "sub-segundo",
    "LLM",
    "1. Sin IA.",
    "2. Ligero.",
    "3. Causalidad.",
    "4. Una sola ruta.",
    "5. Pesimismo.",
    "Engine.step",
    "t+1",
)


def _py_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _module_name(path: Path, src_dir: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(src_dir.resolve())
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem
    if not parts:
        return "src"
    return "src." + ".".join(parts)


def _src_file(dotted: str, src_dir: Path) -> Path | None:
    if dotted != "src" and not dotted.startswith("src."):
        return None
    parts = dotted.split(".")[1:]
    base = src_dir.joinpath(*parts) if parts else src_dir
    if not parts:
        init = src_dir / "__init__.py"
        return init if init.is_file() else None
    module = base.with_suffix(".py")
    if module.is_file():
        return module
    init = base / "__init__.py"
    if init.is_file():
        return init
    return None


def _src_chain(dotted: str, src_dir: Path) -> tuple[str, ...]:
    """Módulo ``dotted`` y los paquetes padre que sí tienen archivo."""
    parts = dotted.split(".")
    found: list[str] = []
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if _src_file(name, src_dir) is not None:
            found.append(name)
    return tuple(found)


def _absolute_from_relative(
    current: str, is_package: bool, level: int, submodule: str | None,
) -> str:
    parts = current.split(".")
    if not is_package and parts:
        parts = parts[:-1]
    up = level - 1
    if up > len(parts):
        return ""
    base = parts[: len(parts) - up]
    if submodule:
        base.extend(submodule.split("."))
    return ".".join(base)


class _Import:
    __slots__ = ("root", "module", "lineno", "follow")

    def __init__(self, root: str, module: str, lineno: int, follow: tuple[str, ...]):
        self.root = root
        self.module = module
        self.lineno = lineno
        self.follow = follow


def _import_of(dotted: str, lineno: int, src_dir: Path, names: tuple[str, ...]) -> _Import:
    root = dotted.split(".", 1)[0] if dotted else ""
    follow: list[str] = []
    if root == "src":
        follow.extend(_src_chain(dotted, src_dir))
        for name in names:
            if name == "*":
                continue
            child = f"{dotted}.{name}"
            if _src_file(child, src_dir) is not None:
                follow.extend(_src_chain(child, src_dir))
    # dict.fromkeys conserva el orden y quita duplicados.
    return _Import(root, dotted, lineno, tuple(dict.fromkeys(follow)))


def _imports_in(tree: ast.AST, path: Path, src_dir: Path) -> list[_Import]:
    current = _module_name(path, src_dir)
    is_package = path.name == "__init__.py"
    found: list[_Import] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(_import_of(alias.name, node.lineno, src_dir, ()))
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            if node.level == 0:
                if node.module:
                    found.append(_import_of(node.module, node.lineno, src_dir, names))
                continue
            if current is None:
                continue
            absolute = _absolute_from_relative(current, is_package, node.level, node.module)
            if absolute:
                found.append(_import_of(absolute, node.lineno, src_dir, names))
    return found


def forbidden_imports(entries: list[Path], src_dir: Path) -> list[str]:
    """Imports prohibidos en ``entries`` o en la clausura estática de ``src.*``.

    Recorre también los imports escritos dentro de funciones. Un archivo se
    visita una sola vez (hay ciclos legítimos entre módulos).
    """
    violations: list[str] = []
    seen: set[Path] = set()
    queue: deque[tuple[Path, tuple[str, ...]]] = deque((path, ()) for path in entries)
    while queue:
        path, chain = queue.popleft()
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for hit in _imports_in(tree, path, src_dir):
            where = f"{_display(path)}:{hit.lineno}"
            if hit.root in FORBIDDEN_ROOTS:
                detail = f"{where}: import {hit.module}"
                if chain:
                    detail = " → ".join(chain) + " → " + detail
                violations.append(detail)
            for dotted in hit.follow:
                dest = _src_file(dotted, src_dir)
                if dest is None:
                    continue
                dest = dest.resolve()
                if dest in seen:
                    continue
                step = f"{where}: import {dotted}"
                queue.append((dest, chain + (step,)))
    return violations


def _network_allowed(path: Path, hft_dir: Path) -> bool:
    try:
        rel = path.resolve().relative_to(hft_dir.resolve()).as_posix()
    except ValueError:
        return False
    return rel == "execution/alpaca_exec.py" or rel.startswith("data/")


def network_imports(files: list[Path], hft_dir: Path) -> list[str]:
    """urllib, socket o http fuera de data/** y execution/alpaca_exec.py."""
    violations: list[str] = []
    for path in files:
        if _network_allowed(path, hft_dir):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for hit in _imports_in(tree, path, hft_dir):
            if hit.root in NETWORK_ROOTS:
                violations.append(f"{_display(path)}:{hit.lineno}: import {hit.module}")
    return violations


def _bindings(tree: ast.AST) -> dict[str, str]:
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[alias.name.split(".", 1)[0]] = alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                bound[local] = f"{node.module}.{alias.name}"
    return bound


def _dotted(node: ast.AST, bound: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in bound:
            return bound[node.id]
        return _BARE_ROOTS.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, bound)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _call_target(func: ast.AST, bound: dict[str, str]) -> str | None:
    if isinstance(func, ast.Name):
        # Un nombre suelto solo cuenta si un import lo resolvió
        # (``from time import time; time()``). No se asume ``time`` a ciegas.
        return bound.get(func.id)
    if isinstance(func, ast.Attribute):
        return _dotted(func, bound)
    return None


def _calls(tree: ast.AST) -> list[tuple[int, str]]:
    bound = _bindings(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _call_target(node.func, bound)
            if target:
                found.append((node.lineno, target))
    return found


def _clock_allowed(path: Path, hft_dir: Path, repo_root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(hft_dir.resolve()).as_posix()
    except ValueError:
        rel = ""
    if rel == "calendar.py":
        return True
    try:
        repo_rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    return repo_rel == "scripts/run_intraday.py"


def clock_calls(files: list[Path], hft_dir: Path, repo_root: Path) -> list[str]:
    """``datetime.now()`` / ``time.time()`` fuera de calendar.py y el runner."""
    violations: list[str] = []
    for path in files:
        if _clock_allowed(path, hft_dir, repo_root):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, target in _calls(tree):
            if target in CLOCK_CALLS:
                violations.append(f"{_display(path)}:{lineno}: {target}()")
    return violations


def global_side_effects(files: list[Path]) -> list[str]:
    """``np.random.seed``, ``np.random.RandomState`` y ``logging.basicConfig``."""
    violations: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, target in _calls(tree):
            if target in GLOBAL_CALLS:
                violations.append(f"{_display(path)}:{lineno}: {target}()")
    return violations


def _hft_sources() -> list[Path]:
    files = _py_files(HFT_DIR)
    script = REPO_ROOT / "scripts" / "run_intraday.py"
    if script.is_file():
        files.append(script)
    return files


class TestHftPackageDoc:
    def test_init_es_solo_docstring(self):
        source = (HFT_DIR / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        assert doc
        for node in tree.body:
            assert isinstance(node, ast.Expr)
            assert isinstance(node.value, ast.Constant)
            assert isinstance(node.value.value, str)
        for phrase in _DOC_FRASES:
            assert phrase in doc, phrase

    def test_el_paquete_importa(self):
        import src.hft as hft

        assert hft.__doc__
        assert "plano al cierre" in hft.__doc__


class TestForbiddenImports:
    def test_src_hft_no_arrastra_dependencias_pesadas(self):
        assert forbidden_imports(_py_files(HFT_DIR), SRC_DIR) == []

    def test_numpy_pandas_y_stdlib_estan_permitidos(self, tmp_path: Path):
        module = tmp_path / "ligero.py"
        module.write_text("import numpy as np\nimport pandas as pd\nimport math\n", encoding="utf-8")
        assert forbidden_imports([module], SRC_DIR) == []

    def test_import_prohibido_en_tmp_path_falla(self, tmp_path: Path):
        """Un import prohibido, incluso dentro de una función, pone el guardia en rojo."""
        module = tmp_path / "intruso.py"
        module.write_text("def cargar():\n    import cvxpy\n", encoding="utf-8")
        violations = forbidden_imports([module], SRC_DIR)
        assert violations
        assert any("cvxpy" in item for item in violations)

    def test_cierre_transitiva_via_src(self, tmp_path: Path):
        src = tmp_path / "src"
        package = src / "hft"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""vacio."""\n', encoding="utf-8")
        # El import prohibido vive dentro de una función del módulo legado.
        (src / "pesado.py").write_text(
            "def cargar():\n    import scipy.stats\n",
            encoding="utf-8",
        )
        entry = package / "estrategia.py"
        entry.write_text("def usar():\n    from ..pesado import cargar\n", encoding="utf-8")
        violations = forbidden_imports([entry], src)
        assert any("scipy" in item for item in violations)

    def test_init_de_paquete_src_entra_en_la_clausura(self, tmp_path: Path):
        src = tmp_path / "src"
        legacy = src / "legado"
        legacy.mkdir(parents=True)
        (legacy / "__init__.py").write_text("import lightgbm\n", encoding="utf-8")
        (legacy / "sub.py").write_text("x = 1\n", encoding="utf-8")
        entry = tmp_path / "cliente.py"
        entry.write_text("import src.legado.sub\n", encoding="utf-8")
        violations = forbidden_imports([entry], src)
        assert any("lightgbm" in item for item in violations)

    def test_ciclo_de_imports_termina(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("import src.b\nimport torch\n", encoding="utf-8")
        (src / "b.py").write_text("import src.a\n", encoding="utf-8")
        violations = forbidden_imports([src / "a.py"], src)
        assert any("torch" in item for item in violations)


class TestNetworkImports:
    def test_src_hft_no_importa_red(self):
        assert network_imports(_py_files(HFT_DIR), HFT_DIR) == []

    def test_solo_data_y_alpaca_exec_pueden_importar_red(self, tmp_path: Path):
        hft = tmp_path / "hft"
        (hft / "data").mkdir(parents=True)
        (hft / "execution").mkdir()
        (hft / "engine.py").write_text("def f():\n    import urllib.request\n", encoding="utf-8")
        (hft / "execution" / "sim.py").write_text("import socket\n", encoding="utf-8")
        (hft / "data" / "alpaca_data.py").write_text("import socket\n", encoding="utf-8")
        (hft / "execution" / "alpaca_exec.py").write_text(
            "import http.client\n", encoding="utf-8",
        )
        violations = network_imports(_py_files(hft), hft)
        assert len(violations) == 2
        joined = "\n".join(violations)
        assert "engine.py" in joined and "urllib" in joined
        assert "execution/sim.py" in joined and "socket" in joined
        assert "alpaca_data.py" not in joined
        assert "alpaca_exec.py" not in joined


class TestClock:
    def test_src_hft_no_lee_el_reloj(self):
        assert clock_calls(_hft_sources(), HFT_DIR, REPO_ROOT) == []

    def test_now_y_time_solo_en_calendar_y_runner(self, tmp_path: Path):
        hft = tmp_path / "src" / "hft"
        scripts = tmp_path / "scripts"
        hft.mkdir(parents=True)
        scripts.mkdir()
        (hft / "engine.py").write_text(
            "import datetime\ndatetime.datetime.now()\n", encoding="utf-8",
        )
        (hft / "calendar.py").write_text(
            "from datetime import datetime\ndatetime.now()\n", encoding="utf-8",
        )
        (scripts / "run_intraday.py").write_text(
            "import time\ntime.time()\n", encoding="utf-8",
        )
        files = _py_files(hft) + [scripts / "run_intraday.py"]
        violations = clock_calls(files, hft, tmp_path)
        assert len(violations) == 1
        assert "engine.py" in violations[0]
        assert "datetime.datetime.now" in violations[0]

    def test_patrones_de_reloj(self, tmp_path: Path):
        samples = (
            "import datetime\ndatetime.datetime.now()\n",
            "from datetime import datetime\ndatetime.now()\n",
            "import datetime as dt\ndt.datetime.now()\n",
            "import time\ntime.time()\n",
            "from time import time\ntime()\n",
        )
        for source in samples:
            module = tmp_path / "reloj.py"
            module.write_text(source, encoding="utf-8")
            violations = clock_calls([module], tmp_path / "hft-ausente", tmp_path / "repo-ausente")
            assert violations, source


class TestGlobalSideEffects:
    def test_src_hft_no_toca_rng_ni_logging_global(self):
        assert global_side_effects(_py_files(HFT_DIR)) == []

    def test_seed_randomstate_y_basicconfig(self, tmp_path: Path):
        seed = tmp_path / "seed.py"
        seed.write_text("import numpy as np\ndef f():\n    np.random.seed(1)\n", encoding="utf-8")
        state = tmp_path / "state.py"
        state.write_text("import numpy as np\nnp.random.RandomState(0)\n", encoding="utf-8")
        logging_mod = tmp_path / "log.py"
        logging_mod.write_text(
            "from logging import basicConfig\nbasicConfig()\n", encoding="utf-8",
        )
        clean = tmp_path / "limpio.py"
        clean.write_text("import numpy as np\nrng = np.random.default_rng(1)\n", encoding="utf-8")
        assert global_side_effects([clean]) == []
        assert any("numpy.random.seed" in item for item in global_side_effects([seed]))
        assert any("RandomState" in item for item in global_side_effects([state]))
        assert any("basicConfig" in item for item in global_side_effects([logging_mod]))


class TestConvictionScorerDoc:
    def test_docstring_es_determinista_y_rechaza_llm(self):
        from src.conviction_scoring import ConvictionScorer

        doc = ConvictionScorer.__doc__ or ""
        assert "determinista" in doc
        assert "reglas" in doc
        assert "No se admiten LLM" in doc
        assert "llm_score" not in doc
        assert "backtest" in doc and "vivo" in doc
