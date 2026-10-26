"""
test_pandas3_idioms.py — Lo que pandas 3 cambia sin avisar.

pandas 3 infiere la resolución de los datetime ("us" para strings, ``datetime``,
``date_range`` y ``read_csv``) en vez de fijar "ns", y ``pct_change()`` ya no
rellena NaN antes de dividir. Ninguno de los dos falla: dan otros números.
Este chequeo estático (src/scripts/experiments/tests) exige la forma explícita:

- ``.as_unit("ns").asi8``: ``asi8`` va en la unidad del índice y
  ``Timestamp.value`` siempre en ns; mezclarlos corre los tiempos x1000.
- ``.ffill().pct_change()``: la semántica de pandas 2 (el hueco rinde 0 y el
  movimiento cae el día que vuelve el precio). ``fill_method=None`` también
  vale, como elección explícita de no rellenar.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_method_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def _implicit_calls(path: Path) -> list:
    """``archivo:línea`` de cada ``asi8`` sin ``as_unit`` y ``pct_change`` sin relleno explícito."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    where = path.relative_to(REPO_ROOT)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "asi8":
            if not _is_method_call(node.value, "as_unit"):
                hits.append(f"{where}:{node.lineno} asi8")
        if _is_method_call(node, "pct_change"):
            explicit = _is_method_call(node.func.value, "ffill") or any(
                k.arg == "fill_method" for k in node.keywords
            )
            if not explicit:
                hits.append(f"{where}:{node.lineno} pct_change")
    return hits


def test_unit_and_fill_are_explicit():
    offenders = [
        hit
        for d in ("src", "scripts", "experiments", "tests")
        for p in sorted((REPO_ROOT / d).rglob("*.py"))
        if "__pycache__" not in p.parts
        for hit in _implicit_calls(p)
    ]
    assert offenders == [], (
        "pandas 3: usar .as_unit(\"ns\").asi8 y .ffill().pct_change() "
        f"(ver docstring): {offenders}"
    )
