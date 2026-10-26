"""Protocolo go/no-go y registro de decisión (HFT-14).

Cinco gates por estrategia. Los umbrales salen de ``[decision]`` en
``config/hft.toml``. G2 exige el backtest sobre IEX con el simulador
pesimista (``fill_mode=cross``, costos sin multiplicar, sin barra extra
de latencia). El t por operación es ``media / (s/√n)``, el de la
sección 2.1 del plan. El DSR usa el ``N`` que ya trae el registro de
pruebas. El PBO solo se exige si hubo rejilla. 09B puede quedar por
debajo de 300 operaciones si la evidencia dice que ese conteo es el
máximo disponible.

G3 pide que el P&L siga positivo con costos × 2 y con una barra más de
latencia, que ningún año aporte más de la fracción configurada, que
quitar los mejores días (``drop_best_days``) deje el P&L positivo, y
que el holdout se haya desbloqueado una sola vez y siga en positivo.

G4 compara el slippage con la misma regla del reporte: la brecha media
``real − modelado`` no puede superar ``slippage_se_mult`` errores
estándar. Una diferencia de intenciones pasa si trae causa; ``sin_explicar``
no cuenta. G5 exige los tres simulacros.

El módulo imprime el veredicto y escribe
``results/hft/decisions/<estrategia>-<fecha>.md``. No firma, no cambia
la config y no activa paper: ``promoted`` no puede ser verdadero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from src.hft.config import PROTOCOL_DRILLS, DecisionConfig, load_hft_config
from src.hft.types import TheoryCard, theory_errors
from src.hft.validation import drop_best_days
from src.settings import load_settings

VERDICTS = ("GO", "NO-GO")
UNEXPLAINED_CAUSE = "sin_explicar"
_GATES = ("G1", "G2", "G3", "G4", "G5")

_EVIDENCE_KEYS = ("backtest", "robustness", "paper", "drills", "links")
_BACKTEST_KEYS = (
    "feed",
    "fill_mode",
    "cost_multiplier",
    "extra_latency_bars",
    "trade_pnl",
    "n_oos_trades",
    "dsr",
    "n_trials",
    "pbo",
    "oos_is_maximum_available",
    "oos_maximum_note",
)
_ROBUST_KEYS = (
    "pnl_costs_x2",
    "pnl_extra_latency",
    "pnl_by_year",
    "daily_pnl",
    "holdout_unlock_count",
    "holdout_pnl",
)
_PAPER_KEYS = (
    "n_sessions",
    "n_trades",
    "differences",
    "slippage_real_bps",
    "slippage_modeled_bps",
    "unexplained_risk_incidents",
)
_LINK_KEYS = ("backtests", "trials", "reports")


@dataclass(frozen=True)
class EvidenceLinks:
    """Rutas que el registro cita. No se abren."""

    backtests: Tuple[str, ...] = ()
    trials: str = ""
    reports: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BacktestEvidence:
    """Backtest base, no el de estrés. El PBO vacío significa que no hubo rejilla."""

    feed: str
    fill_mode: str
    cost_multiplier: float
    extra_latency_bars: int
    trade_pnl: Tuple[float, ...]
    n_oos_trades: int
    dsr: float
    n_trials: int
    pbo: Optional[float] = None
    oos_is_maximum_available: bool = False
    oos_maximum_note: str = ""


@dataclass(frozen=True)
class RobustnessEvidence:
    """Estrés ya calculado. ``daily_pnl`` es el P&L del backtest base."""

    pnl_costs_x2: float
    pnl_extra_latency: float
    pnl_by_year: Tuple[Tuple[str, float], ...]
    daily_pnl: Tuple[float, ...]
    holdout_unlock_count: int
    holdout_pnl: float


@dataclass(frozen=True)
class IntentDifference:
    """Una intención que no coincidió. ``cause`` vacío no explica nada."""

    intent_id: str
    cause: str


@dataclass(frozen=True)
class PaperEvidence:
    """Sesiones de paper ya reconciliadas. El slippage va en bps, una fila por sesión."""

    n_sessions: int
    n_trades: int
    differences: Tuple[IntentDifference, ...] = ()
    slippage_real_bps: Tuple[float, ...] = ()
    slippage_modeled_bps: Tuple[float, ...] = ()
    unexplained_risk_incidents: int = 0


@dataclass(frozen=True)
class DrillEvidence:
    """Simulacros. Un ``False`` es un simulacro que no se dio por pasado."""

    crash_restart: bool
    kill: bool
    half_session: bool


@dataclass(frozen=True)
class DecisionInput:
    """Todo lo que miran los gates. ``as_of`` lo pone el llamador, no el reloj."""

    strategy_id: str
    as_of: str
    theory: TheoryCard
    backtest: BacktestEvidence
    robustness: RobustnessEvidence
    paper: PaperEvidence
    drills: DrillEvidence
    links: EvidenceLinks = EvidenceLinks()


@dataclass(frozen=True)
class GateResult:
    """Un gate. ``notes`` guarda los números; ``failures`` es lo que lo tira."""

    gate_id: str
    title: str
    passed: bool
    failures: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """Veredicto sin firma y sin promoción. Las dos últimas no las escribe este módulo."""

    strategy_id: str
    as_of: str
    verdict: str
    gates: Tuple[GateResult, ...]
    links: EvidenceLinks
    signature: str = ""
    promoted: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"veredicto {self.verdict!r} no está en {VERDICTS}")
        if self.signature:
            raise ValueError("la firma la escribe el operador, no este módulo")
        if self.promoted:
            raise ValueError("el sistema no promueve una estrategia")
        ids = tuple(gate.gate_id for gate in self.gates)
        if ids != _GATES:
            raise ValueError(f"se esperaban los gates {_GATES}, llegó {ids}")


def expectancy_and_t(trade_pnl: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Media del P&L por operación y su t.

    ``t = media / (s / √n)`` con ``s`` muestral (``ddof=1``). Con menos
    de dos operaciones el t no está definido (``None``). Si ``s`` es 0
    y la media es positiva, el t es infinito.
    """
    values = _finite_floats("trade_pnl", trade_pnl)
    count = len(values)
    if count == 0:
        return 0.0, None
    mean = float(statistics.fmean(values))
    if count < 2:
        return mean, None
    stdev = float(statistics.stdev(values))
    if stdev == 0.0:
        return mean, math.inf if mean > 0.0 else 0.0
    return mean, mean / (stdev / math.sqrt(count))


def evaluate(evidence: DecisionInput, thresholds: DecisionConfig) -> DecisionRecord:
    """Calcula los cinco gates. Uno solo que falle deja el veredicto en NO-GO."""
    if not isinstance(evidence, DecisionInput):
        raise TypeError("evidence debe ser DecisionInput")
    if not isinstance(thresholds, DecisionConfig):
        raise TypeError("thresholds debe ser DecisionConfig")
    strategy_id = _strategy_id(evidence.strategy_id)
    as_of = _iso_date(evidence.as_of)
    gates = (
        _gate_theory(evidence.theory),
        _gate_backtest(strategy_id, evidence.backtest, thresholds),
        _gate_robustness(evidence.robustness, thresholds),
        _gate_paper(evidence.paper, thresholds),
        _gate_drills(evidence.drills, thresholds),
    )
    verdict = "GO" if all(gate.passed for gate in gates) else "NO-GO"
    return DecisionRecord(
        strategy_id=strategy_id,
        as_of=as_of,
        verdict=verdict,
        gates=gates,
        links=evidence.links,
    )


def render_markdown(record: DecisionRecord) -> str:
    """Plantilla del registro. La firma queda en blanco."""
    if not isinstance(record, DecisionRecord):
        raise TypeError("record debe ser DecisionRecord")
    lines = [
        f"# Decisión go/no-go: {record.strategy_id}",
        "",
        f"Fecha: {record.as_of}",
        "",
        f"**Veredicto: {record.verdict}**",
        "",
        "El sistema no promueve la estrategia. Este registro espera la firma",
        "del operador. Completar la firma no cambia `config/hft.toml` y no",
        "envía órdenes.",
        "",
        "## Gates",
        "",
    ]
    for gate in record.gates:
        state = "PASS" if gate.passed else "FAIL"
        lines.append(f"### {gate.gate_id} {gate.title} — {state}")
        lines.append("")
        for note in gate.notes:
            lines.append(f"- {note}")
        for failure in gate.failures:
            lines.append(f"- Falla: {failure}")
        if not gate.notes and not gate.failures:
            lines.append("- (sin detalle)")
        lines.append("")
    lines.append("## Evidencia")
    lines.append("")
    lines.extend(_evidence_lines(record.links))
    lines.extend([
        "",
        "## Firma del operador",
        "",
        "Nombre:",
        "",
        "Fecha:",
        "",
        "Firma:",
        "",
    ])
    return "\n".join(lines)


def decision_path(strategy_id: str, as_of: str, root: Optional[Path] = None) -> Path:
    """``<root>/<estrategia>-<fecha>.md``. Sin ``root``, cae en ``results/hft/decisions``."""
    name = _strategy_id(strategy_id)
    day = _iso_date(as_of)
    base = Path(root) if root is not None else load_settings().results_dir / "hft" / "decisions"
    return base / f"{name}-{day}.md"


def write_decision(record: DecisionRecord, path: Path) -> Path:
    """Escribe el markdown. No pisa un registro ya existente."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"{target} ya existe. El sistema no pisa un registro: la firma la pone el operador."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(render_markdown(record), encoding="utf-8")
    os.replace(temporary, target)
    return target


def theory_of(strategy_id: str) -> TheoryCard:
    """TheoryCard de la estrategia registrada. Importa el paquete de estrategias."""
    from src.hft.strategies import STRATEGIES

    name = _strategy_id(strategy_id)
    found = STRATEGIES.get(name)
    if found is None:
        known = ", ".join(sorted(STRATEGIES)) or "(ninguna)"
        raise ValueError(f"estrategia no registrada: {name}. Hay {known}.")
    card = getattr(found, "theory", None)
    if not isinstance(card, TheoryCard):
        raise TypeError(f"{name} no tiene TheoryCard")
    return card


def parse_evidence(
    payload: Mapping[str, object],
    strategy_id: str,
    as_of: str,
    theory: TheoryCard,
) -> DecisionInput:
    """Arma la evidencia desde el JSON del CLI. Clave desconocida es error."""
    raw = _table("evidence", payload, _EVIDENCE_KEYS)
    return DecisionInput(
        strategy_id=strategy_id,
        as_of=as_of,
        theory=theory,
        backtest=_backtest(raw["backtest"]),
        robustness=_robustness(raw["robustness"]),
        paper=_paper(raw["paper"]),
        drills=_drills(raw["drills"]),
        links=_links(raw["links"]),
    )


def verdict_line(record: DecisionRecord) -> str:
    """Texto que imprime el CLI. El GO sigue pendiente de firma."""
    if record.verdict == "GO":
        return (
            "Veredicto: GO. Pendiente de firma. El sistema no promueve la estrategia."
        )
    return "Veredicto: NO-GO. El sistema no promueve la estrategia."


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Imprime el veredicto y escribe el registro. Sale 2 si es NO-GO.

    Sale 0 solo cuando los cinco gates pasan. Eso no activa la estrategia:
    la firma del archivo queda vacía.
    """
    from src.settings import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(
        description="Protocolo go/no-go (HFT-14). No promueve la estrategia.",
    )
    parser.add_argument("--strategy", required=True, help="Id registrado (orb, ...)")
    parser.add_argument("--date", required=True, help="Fecha del registro, YYYY-MM-DD")
    parser.add_argument("--evidence", required=True, help="JSON con G2–G5 y los enlaces")
    parser.add_argument("--config", default=None, help="TOML (default config/hft.toml)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta del markdown (default results/hft/decisions)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_hft_config(args.config)
        theory = theory_of(args.strategy)
        payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("el JSON de evidencia debe ser un objeto")
        evidence = parse_evidence(payload, args.strategy, args.date, theory)
        record = evaluate(evidence, config.decision)
        root = Path(args.output_dir) if args.output_dir else None
        path = write_decision(record, decision_path(args.strategy, args.date, root))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(verdict_line(record))
    print(path)
    return 0 if record.verdict == "GO" else 2


def _gate_theory(theory: object) -> GateResult:
    failures = list(theory_errors(theory))
    notes = []
    if isinstance(theory, TheoryCard):
        notes.append(f"mecanismo: {theory.mechanism.strip()}")
        notes.append(f"kill criteria: {len(theory.kill_criteria)}")
        if theory.research_only:
            failures.append("research_only no admite una decisión de capital")
    return _result("G1", "Teoría", failures, notes)


def _gate_backtest(
    strategy_id: str,
    backtest: BacktestEvidence,
    thresholds: DecisionConfig,
) -> GateResult:
    if not isinstance(backtest, BacktestEvidence):
        raise TypeError("backtest debe ser BacktestEvidence")
    failures = []
    notes = []
    feed = str(backtest.feed).strip().lower()
    if feed != "iex":
        failures.append(f"feed {backtest.feed!r}: G2 exige iex")
    else:
        notes.append("feed=iex")
    mode = str(backtest.fill_mode).strip().lower()
    if mode != "cross":
        failures.append("fill_mode debe ser cross (costos pesimistas)")
    else:
        notes.append("fill_mode=cross")
    multiplier = _plain_float("cost_multiplier", backtest.cost_multiplier)
    if not math.isclose(multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12):
        failures.append(f"cost_multiplier={multiplier}: G2 usa el modelo base, no el estrés")
    else:
        notes.append("cost_multiplier=1")
    extra = backtest.extra_latency_bars
    if isinstance(extra, bool) or not isinstance(extra, int) or extra < 0:
        raise ValueError("extra_latency_bars debe ser un entero >= 0")
    if extra != 0:
        failures.append("G2 no incluye la barra extra de latencia")
    else:
        notes.append("extra_latency_bars=0")
    pnl = _finite_floats("trade_pnl", backtest.trade_pnl)
    mean, tstat = expectancy_and_t(pnl)
    notes.append(
        f"expectativa={mean:.6f} t={_format_t(tstat)} n={len(pnl)}"
    )
    if not (mean > thresholds.min_expectancy):
        failures.append(
            f"expectativa {mean:.6f} no es > {thresholds.min_expectancy}"
        )
    if tstat is None or not (tstat >= thresholds.min_t_stat):
        shown = _format_t(tstat)
        failures.append(f"t={shown} no llega a {thresholds.min_t_stat}")
    _oos_clause(strategy_id, backtest, thresholds, len(pnl), failures, notes)
    dsr = _plain_float("dsr", backtest.dsr)
    if not 0.0 <= dsr <= 1.0:
        raise ValueError(f"dsr debe estar en [0, 1], llegó {backtest.dsr!r}")
    trials = backtest.n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("n_trials debe ser un entero >= 1 (N del registro)")
    notes.append(f"dsr={dsr:.6f} N={trials}")
    if not (dsr >= thresholds.min_dsr):
        failures.append(f"DSR {dsr:.6f} < {thresholds.min_dsr} (N={trials})")
    if backtest.pbo is None:
        notes.append("sin rejilla: el PBO no aplica")
    else:
        pbo = _plain_float("pbo", backtest.pbo)
        if not 0.0 <= pbo <= 1.0:
            raise ValueError(f"pbo debe estar en [0, 1], llegó {backtest.pbo!r}")
        notes.append(f"pbo={pbo:.6f}")
        if pbo > thresholds.max_pbo:
            failures.append(f"PBO {pbo:.6f} > {thresholds.max_pbo}")
    return _result("G2", "Backtest", failures, notes)


def _oos_clause(strategy_id, backtest, thresholds, n_pnl, failures, notes) -> None:
    count = backtest.n_oos_trades
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("n_oos_trades debe ser un entero >= 0")
    if count != n_pnl:
        failures.append(
            f"n_oos={count} no coincide con las {n_pnl} operaciones del t"
        )
    low_frequency = strategy_id in thresholds.low_frequency_strategies
    if count >= thresholds.min_oos_trades:
        notes.append(f"{count} operaciones OOS >= {thresholds.min_oos_trades}")
        return
    note = backtest.oos_maximum_note
    if not isinstance(note, str):
        raise TypeError("oos_maximum_note debe ser str")
    if not isinstance(backtest.oos_is_maximum_available, bool):
        raise TypeError("oos_is_maximum_available debe ser bool")
    explicit = backtest.oos_is_maximum_available and bool(note.strip()) and count >= 1
    if low_frequency and explicit:
        notes.append(
            f"{count} es el máximo disponible, por debajo de "
            f"{thresholds.min_oos_trades}: {note.strip()}"
        )
        return
    if low_frequency:
        failures.append(
            f"{count} operaciones OOS < {thresholds.min_oos_trades} "
            "y el máximo disponible no está explicitado"
        )
        return
    failures.append(f"{count} operaciones OOS < {thresholds.min_oos_trades}")


def _gate_robustness(robust: RobustnessEvidence, thresholds: DecisionConfig) -> GateResult:
    if not isinstance(robust, RobustnessEvidence):
        raise TypeError("robustness debe ser RobustnessEvidence")
    failures = []
    notes = []
    costs = _plain_float("pnl_costs_x2", robust.pnl_costs_x2)
    latency = _plain_float("pnl_extra_latency", robust.pnl_extra_latency)
    notes.append(f"pnl_costos_x2={costs:.6f}")
    notes.append(f"pnl_latencia_extra={latency:.6f}")
    if not (costs > 0.0):
        failures.append("con costos × 2 el P&L no es > 0")
    if not (latency > 0.0):
        failures.append("con +1 barra de latencia el P&L no es > 0")
    years = _years(robust.pnl_by_year)
    total = float(sum(pnl for _, pnl in years))
    notes.append(f"pnl_por_año={total:.6f}")
    if not years or not (total > 0.0):
        failures.append("el P&L sumado por año no es positivo")
    else:
        limit = thresholds.max_year_pnl_share
        breached = False
        for year, pnl in years:
            share = pnl / total
            if share > limit:
                breached = True
                failures.append(
                    f"{year} aporta {share:.2%} del P&L (tope {limit:.2%})"
                )
        if not breached:
            notes.append(f"ningún año supera {limit:.2%} del P&L")
    daily = _finite_floats("daily_pnl", robust.daily_pnl)
    kept = drop_best_days(np.asarray(daily, dtype=float), thresholds.best_days_dropped)
    kept_sum = float(kept.sum()) if kept.size else 0.0
    notes.append(
        f"pnl_sin_{thresholds.best_days_dropped}_mejores_días={kept_sum:.6f}"
    )
    if not (kept_sum > 0.0):
        failures.append(
            f"sin los {thresholds.best_days_dropped} mejores días el P&L no es > 0"
        )
    unlocks = robust.holdout_unlock_count
    if isinstance(unlocks, bool) or not isinstance(unlocks, int) or unlocks < 0:
        raise ValueError("holdout_unlock_count debe ser un entero >= 0")
    holdout = _plain_float("holdout_pnl", robust.holdout_pnl)
    notes.append(f"holdout_desbloqueos={unlocks} holdout_pnl={holdout:.6f}")
    if unlocks != 1:
        failures.append(f"el holdout se desbloqueó {unlocks} veces; se exige una")
    elif not (holdout > 0.0):
        failures.append("el holdout desbloqueado una vez no es consistente (P&L <= 0)")
    return _result("G3", "Robustez", failures, notes)


def _gate_paper(paper: PaperEvidence, thresholds: DecisionConfig) -> GateResult:
    if not isinstance(paper, PaperEvidence):
        raise TypeError("paper debe ser PaperEvidence")
    failures = []
    notes = []
    sessions = _count("n_sessions", paper.n_sessions)
    trades = _count("n_trades", paper.n_trades)
    incidents = _count("unexplained_risk_incidents", paper.unexplained_risk_incidents)
    notes.append(f"sesiones={sessions} operaciones={trades}")
    if sessions < thresholds.min_paper_sessions:
        failures.append(f"{sessions} sesiones < {thresholds.min_paper_sessions}")
    if trades < thresholds.min_paper_trades:
        failures.append(f"{trades} operaciones < {thresholds.min_paper_trades}")
    diffs = paper.differences
    if isinstance(diffs, str) or not isinstance(diffs, tuple):
        raise TypeError("differences debe ser una tupla")
    unexplained = []
    for item in diffs:
        if not isinstance(item, IntentDifference):
            raise TypeError("differences debe ser IntentDifference")
        if not str(item.intent_id).strip():
            raise ValueError("intent_id de una diferencia vacío")
        cause = item.cause.strip() if isinstance(item.cause, str) else ""
        if not isinstance(item.cause, str) or not cause or cause == UNEXPLAINED_CAUSE:
            unexplained.append(item.intent_id)
    if unexplained:
        joined = ", ".join(unexplained)
        failures.append(f"diferencias de intents sin explicar: {joined}")
    elif diffs:
        notes.append(f"{len(diffs)} diferencias de intents, todas explicadas")
    else:
        notes.append("paridad de intents 100%")
    real = _finite_floats("slippage_real_bps", paper.slippage_real_bps)
    modeled = _finite_floats("slippage_modeled_bps", paper.slippage_modeled_bps)
    if len(real) != len(modeled):
        raise ValueError("slippage real y modelado no miden lo mismo")
    need = thresholds.min_slippage_sessions
    if len(real) < need:
        failures.append(f"slippage: {len(real)} sesiones < {need}")
    else:
        gaps = [left - right for left, right in zip(real, modeled)]
        mean = float(statistics.fmean(gaps))
        stdev = float(statistics.stdev(gaps))
        se = stdev / math.sqrt(len(gaps))
        limit = thresholds.slippage_se_mult * se
        notes.append(f"brecha_slippage_bps={mean:.6f} umbral={limit:.6f}")
        if mean > limit:
            failures.append(
                f"slippage real supera al modelo en {mean:.6f} bps "
                f"(umbral {limit:.6f})"
            )
    if incidents:
        failures.append(f"{incidents} incidentes de riesgo sin explicar")
    else:
        notes.append("cero incidentes de riesgo sin explicar")
    return _result("G4", "Paper", failures, notes)


def _gate_drills(drills: DrillEvidence, thresholds: DecisionConfig) -> GateResult:
    if not isinstance(drills, DrillEvidence):
        raise TypeError("drills debe ser DrillEvidence")
    flags = {
        "crash_restart": drills.crash_restart,
        "kill": drills.kill,
        "half_session": drills.half_session,
    }
    failures = []
    notes = []
    for name in thresholds.required_drills:
        if name not in flags:
            raise ValueError(f"simulacro desconocido en la config: {name}")
        value = flags[name]
        if not isinstance(value, bool):
            raise TypeError(f"simulacro {name} debe ser bool")
        if value:
            notes.append(f"simulacro {name} pasado")
        else:
            failures.append(f"simulacro {name} no está pasado")
    missing = [name for name in PROTOCOL_DRILLS if name not in thresholds.required_drills]
    if missing:
        failures.append("faltan simulacros en la config: " + ", ".join(missing))
    return _result("G5", "Operación", failures, notes)


def _result(gate_id: str, title: str, failures: Sequence[str], notes: Sequence[str]) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        title=title,
        passed=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
    )


def _evidence_lines(links: EvidenceLinks) -> list:
    lines = []
    for item in links.backtests:
        lines.append(f"- Backtest: `{item}`")
    if links.trials:
        lines.append(f"- Registro de pruebas: `{links.trials}`")
    for item in links.reports:
        lines.append(f"- Reporte: `{item}`")
    if not lines:
        lines.append("- (sin enlaces)")
    return lines


def _years(pairs: Sequence[Tuple[str, float]]) -> Tuple[Tuple[str, float], ...]:
    if isinstance(pairs, str) or not isinstance(pairs, tuple):
        raise TypeError("pnl_by_year debe ser una tupla de (año, pnl)")
    out = []
    seen = set()
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"pnl_by_year: elemento inválido {item!r}")
        year, pnl = item
        if not isinstance(year, str) or not year.strip():
            raise ValueError(f"año inválido {year!r}")
        key = year.strip()
        if key in seen:
            raise ValueError(f"año repetido {key}")
        seen.add(key)
        out.append((key, _plain_float(f"pnl {key}", pnl)))
    return tuple(out)


def _backtest(value: object) -> BacktestEvidence:
    raw = _table("backtest", value, _BACKTEST_KEYS)
    pbo = raw["pbo"]
    if pbo is not None:
        pbo = _plain_float("pbo", pbo)
    return BacktestEvidence(
        feed=_text("backtest.feed", raw["feed"]),
        fill_mode=_text("backtest.fill_mode", raw["fill_mode"]),
        cost_multiplier=_plain_float("cost_multiplier", raw["cost_multiplier"]),
        extra_latency_bars=_count("extra_latency_bars", raw["extra_latency_bars"]),
        trade_pnl=_finite_floats("trade_pnl", raw["trade_pnl"]),
        n_oos_trades=_count("n_oos_trades", raw["n_oos_trades"]),
        dsr=_plain_float("dsr", raw["dsr"]),
        n_trials=_count("n_trials", raw["n_trials"]),
        pbo=pbo,
        oos_is_maximum_available=_bool(
            "oos_is_maximum_available", raw["oos_is_maximum_available"],
        ),
        oos_maximum_note=_text("oos_maximum_note", raw["oos_maximum_note"], allow_empty=True),
    )


def _robustness(value: object) -> RobustnessEvidence:
    raw = _table("robustness", value, _ROBUST_KEYS)
    years = raw["pnl_by_year"]
    if not isinstance(years, Mapping):
        raise ValueError(f"pnl_by_year: se esperaba un objeto, llegó {years!r}")
    pairs = tuple(
        (str(key), _plain_float(f"pnl {key}", years[key]))
        for key in years
    )
    return RobustnessEvidence(
        pnl_costs_x2=_plain_float("pnl_costs_x2", raw["pnl_costs_x2"]),
        pnl_extra_latency=_plain_float("pnl_extra_latency", raw["pnl_extra_latency"]),
        pnl_by_year=pairs,
        daily_pnl=_finite_floats("daily_pnl", raw["daily_pnl"]),
        holdout_unlock_count=_count("holdout_unlock_count", raw["holdout_unlock_count"]),
        holdout_pnl=_plain_float("holdout_pnl", raw["holdout_pnl"]),
    )


def _paper(value: object) -> PaperEvidence:
    raw = _table("paper", value, _PAPER_KEYS)
    diffs = raw["differences"]
    if isinstance(diffs, str) or not isinstance(diffs, Sequence):
        raise ValueError(f"differences: se esperaba una lista, llegó {diffs!r}")
    parsed = []
    for index, item in enumerate(diffs):
        if not isinstance(item, Mapping):
            raise ValueError(f"differences[{index}] no es un objeto")
        row = _table(f"differences[{index}]", item, ("intent_id", "cause"))
        parsed.append(IntentDifference(
            intent_id=_text(f"differences[{index}].intent_id", row["intent_id"]),
            cause=_text(f"differences[{index}].cause", row["cause"], allow_empty=True),
        ))
    return PaperEvidence(
        n_sessions=_count("n_sessions", raw["n_sessions"]),
        n_trades=_count("n_trades", raw["n_trades"]),
        differences=tuple(parsed),
        slippage_real_bps=_finite_floats("slippage_real_bps", raw["slippage_real_bps"]),
        slippage_modeled_bps=_finite_floats(
            "slippage_modeled_bps", raw["slippage_modeled_bps"],
        ),
        unexplained_risk_incidents=_count(
            "unexplained_risk_incidents", raw["unexplained_risk_incidents"],
        ),
    )


def _drills(value: object) -> DrillEvidence:
    raw = _table("drills", value, PROTOCOL_DRILLS)
    return DrillEvidence(
        crash_restart=_bool("crash_restart", raw["crash_restart"]),
        kill=_bool("kill", raw["kill"]),
        half_session=_bool("half_session", raw["half_session"]),
    )


def _links(value: object) -> EvidenceLinks:
    raw = _table("links", value, _LINK_KEYS)
    trials = raw["trials"]
    if not isinstance(trials, str):
        raise ValueError(f"links.trials: se esperaba str, llegó {trials!r}")
    return EvidenceLinks(
        backtests=_path_list("links.backtests", raw["backtests"]),
        trials=trials.strip(),
        reports=_path_list("links.reports", raw["reports"]),
    )


def _path_list(name: str, value: object) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name}: se esperaba una lista, llegó {value!r}")
    out = []
    for item in value:
        text = _text(name, item)
        if any(char in text for char in "\n\r`"):
            raise ValueError(f"{name}: ruta inválida {item!r}")
        out.append(text)
    return tuple(out)


def _table(name: str, value: object, fields: Sequence[str]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: se esperaba un objeto, llegó {value!r}")
    raw = dict(value)
    unknown = sorted(set(raw) - set(fields))
    missing = sorted(set(fields) - set(raw))
    if unknown or missing:
        raise ValueError(f"{name}: faltan {missing}, desconocidas {unknown}.")
    return raw


def _finite_floats(name: str, value: object) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} debe ser una secuencia de números")
    out = []
    for item in value:
        out.append(_plain_float(name, item))
    return tuple(out)


def _plain_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser numérico, llegó {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} debe ser finito, llegó {value!r}")
    return number


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} debe ser un entero >= 0, llegó {value!r}")
    return value


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} debe ser bool, llegó {value!r}")
    return value


def _text(name: str, value: object, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} debe ser str, llegó {value!r}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} vacío")
    return value.strip()


def _strategy_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"strategy_id inválido {value!r}")
    text = value.strip()
    if not text.replace("_", "").isalnum() or text[0].isdigit():
        raise ValueError(f"strategy_id inválido {value!r}")
    return text


def _iso_date(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"fecha inválida {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"fecha ISO inválida {value!r}") from exc
    return value


def _format_t(tstat: Optional[float]) -> str:
    if tstat is None:
        return "indefinido"
    if math.isinf(tstat):
        return "inf"
    return f"{tstat:.6f}"


if __name__ == "__main__":
    sys.exit(main())
