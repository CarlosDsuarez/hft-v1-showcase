"""
test_timedelta_units.py — Timedeltas con unidad explícita.

Con pandas 2.3.3 y NumPy >= 2.5, ``pd.Timedelta`` con kwargs (``days=1``) o con
un string (``"1D"``) construye ``np.timedelta64`` con la unidad 'generic', que
NumPy deprecó y convertirá en error. La forma limpia es
``pd.Timedelta(n, unit="D")``. pyproject.toml vuelve error ese warning en el
código que corren los tests; este chequeo estático cubre también lo que no
corren (data_client con red, experiments).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_UNIT_KWARGS = {
    "weeks", "days", "hours", "minutes", "seconds",
    "milliseconds", "microseconds", "nanoseconds",
}


def _generic_unit_calls(path: Path) -> list:
    """``archivo:línea`` de cada ``Timedelta(...)`` con kwargs de unidad o string."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "Timedelta":
            continue
        unit_kwargs = {k.arg for k in node.keywords} & _UNIT_KWARGS
        string_arg = (
            bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
        if unit_kwargs or string_arg:
            hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


def test_timedelta_has_explicit_unit():
    offenders = [
        hit
        for d in ("src", "scripts", "experiments", "tests")
        for p in sorted((REPO_ROOT / d).rglob("*.py"))
        if "__pycache__" not in p.parts
        for hit in _generic_unit_calls(p)
    ]
    assert offenders == [], (
        "pd.Timedelta con kwargs o string usa la unidad 'generic' de NumPy "
        f"(deprecada); usar pd.Timedelta(n, unit=\"D\"): {offenders}"
    )
