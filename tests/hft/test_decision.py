"""Protocolo go/no-go (HFT-14). Los números son sintéticos: no hay red."""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from src.hft.config import hft_config_from_dict, load_hft_config
from src.hft.decision import (
    BacktestEvidence,
    DecisionRecord,
    DrillEvidence,
    EvidenceLinks,
    IntentDifference,
    PaperEvidence,
    RobustnessEvidence,
    decision_path,
    evaluate,
    expectancy_and_t,
    main,
    parse_evidence,
    render_markdown,
    theory_of,
    write_decision,
)
from src.hft.types import TheoryCard, TheoryParam


def _theory(**overrides) -> TheoryCard:
    payload = dict(
        hypothesis="El precio sigue la apertura cuando el volumen es anómalo.",
        mechanism="Paga el agente que incorpora tarde la información de la apertura.",
        references=("Zarattini, Barbon & Aziz (2024), SSRN 4729284",),
        data_requirements=("barras de 1 minuto del feed iex",),
        params=(
            TheoryParam("opening_minutes", 5, "El paper usa 5 minutos.", "sec. 2.1"),
        ),
        holding_period="de la apertura al cierre",
        failure_regimes=("día con noticia que no revierte",),
        kill_criteria=("expectativa neta de 60 sesiones < 0 con t < -1",),
        research_only=False,
    )
    payload.update(overrides)
    return TheoryCard(**payload)


def _trades(n: int = 300) -> tuple:
    return tuple(2.0 if i % 2 == 0 else 0.0 for i in range(n))


def _backtest(**overrides) -> BacktestEvidence:
    pnl = _trades()
    payload = dict(
        feed="iex",
        fill_mode="cross",
        cost_multiplier=1.0,
        extra_latency_bars=0,
        trade_pnl=pnl,
        n_oos_trades=len(pnl),
        dsr=0.97,
        n_trials=3,
        pbo=None,
        oos_is_maximum_available=False,
        oos_maximum_note="",
    )
    payload.update(overrides)
    return BacktestEvidence(**payload)


def _robust(**overrides) -> RobustnessEvidence:
    payload = dict(
        pnl_costs_x2=12.0,
        pnl_extra_latency=4.0,
        pnl_by_year=(("2022", 40.0), ("2023", 30.0), ("2024", 30.0)),
        daily_pnl=(1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7),
        holdout_unlock_count=1,
        holdout_pnl=5.0,
    )
    payload.update(overrides)
    return RobustnessEvidence(**payload)


def _paper(**overrides) -> PaperEvidence:
    zeros = tuple(0.0 for _ in range(20))
    payload = dict(
        n_sessions=20,
        n_trades=100,
        differences=(),
        slippage_real_bps=zeros,
        slippage_modeled_bps=zeros,
        unexplained_risk_incidents=0,
    )
    payload.update(overrides)
    return PaperEvidence(**payload)


def _input(**overrides):
    from src.hft.decision import DecisionInput

    payload = dict(
        strategy_id="orb",
        as_of="2026-10-06",
        theory=_theory(),
        backtest=_backtest(),
        robustness=_robust(),
        paper=_paper(),
        drills=DrillEvidence(crash_restart=True, kill=True, half_session=True),
        links=EvidenceLinks(
            backtests=("results/hft/backtests/run-1",),
            trials="results/hft/trials.csv",
            reports=("results/hft/reports/2026-10-06.html",),
        ),
    )
    payload.update(overrides)
    return DecisionInput(**payload)


def _thresholds(**overrides):
    base = load_hft_config().decision
    return replace(base, **overrides) if overrides else base


def _gate(record, gate_id: str):
    return next(gate for gate in record.gates if gate.gate_id == gate_id)


def test_umbrales_versionados():
    cfg = load_hft_config().decision
    assert cfg.min_expectancy == 0
    assert cfg.min_t_stat == 2
    assert cfg.min_dsr == pytest.approx(0.95)
    assert cfg.max_pbo == pytest.approx(0.2)
    assert cfg.min_oos_trades == 300
    assert cfg.low_frequency_strategies == ("intraday_momentum",)
    assert cfg.max_year_pnl_share == pytest.approx(0.5)
    assert cfg.best_days_dropped == 5
    assert cfg.min_paper_sessions == 20
    assert cfg.min_paper_trades == 100
    assert cfg.min_slippage_sessions == 20
    assert cfg.slippage_se_mult == 2
    assert cfg.required_drills == ("crash_restart", "kill", "half_session")


def test_la_tabla_decision_es_estricta():
    raw = load_hft_config().to_dict()
    missing = dict(raw)
    del missing["decision"]
    with pytest.raises(ValueError, match="decision"):
        hft_config_from_dict(missing)
    unknown = dict(raw["decision"])
    unknown["grid_search"] = True
    with pytest.raises(ValueError, match="desconocidas"):
        hft_config_from_dict({**raw, "decision": unknown})
    drills = [name for name in raw["decision"]["required_drills"] if name != "kill"]
    with pytest.raises(ValueError, match="required_drills"):
        hft_config_from_dict({**raw, "decision": {**raw["decision"], "required_drills": drills}})


def test_go_cuando_los_cinco_gates_pasan():
    record = evaluate(_input(), _thresholds())
    assert record.verdict == "GO"
    assert record.signature == ""
    assert record.promoted is False
    assert all(gate.passed for gate in record.gates)


def test_g2_calcula_expectativa_y_t():
    pnl = (1.0, 2.0, 3.0, 4.0)
    mean, tstat = expectancy_and_t(pnl)
    assert mean == pytest.approx(statistics.fmean(pnl))
    assert tstat == pytest.approx(mean / (statistics.stdev(pnl) / len(pnl) ** 0.5))
    evidence = _input(backtest=_backtest(trade_pnl=pnl, n_oos_trades=len(pnl)))
    note = " ".join(_gate(evaluate(evidence, _thresholds()), "G2").notes)
    assert f"expectativa={mean:.6f}" in note
    assert f"t={tstat:.6f}" in note


def test_el_umbral_de_t_sale_de_la_config():
    pnl = (1.0, 2.0, 3.0, 4.0)
    _mean, tstat = expectancy_and_t(pnl)
    evidence = _input(backtest=_backtest(trade_pnl=pnl, n_oos_trades=len(pnl)))
    loose = _thresholds(min_t_stat=tstat, min_oos_trades=len(pnl))
    assert _gate(evaluate(evidence, loose), "G2").passed
    tight = _thresholds(min_t_stat=tstat + 1e-6, min_oos_trades=len(pnl))
    failed = _gate(evaluate(evidence, tight), "G2")
    assert not failed.passed
    assert any(item.startswith("t=") for item in failed.failures)


def test_un_gate_fallido_bloquea_el_veredicto_go():
    evidence = _input(robustness=_robust(pnl_costs_x2=-1.0))
    record = evaluate(evidence, _thresholds())
    assert record.verdict == "NO-GO"
    assert record.promoted is False
    assert not _gate(record, "G3").passed
    assert _gate(record, "G1").passed
    assert _gate(record, "G2").passed
    assert _gate(record, "G4").passed
    assert _gate(record, "G5").passed
    with pytest.raises(ValueError, match="no promueve"):
        DecisionRecord(
            strategy_id="orb",
            as_of="2026-10-06",
            verdict="GO",
            gates=record.gates,
            links=record.links,
            promoted=True,
        )


def test_expectativa_no_positiva_dsr_y_pbo():
    flat = (1.0, -1.0, 1.0, -1.0)
    negative = evaluate(
        _input(backtest=_backtest(trade_pnl=flat, n_oos_trades=len(flat))),
        _thresholds(min_oos_trades=len(flat), min_t_stat=0.01),
    )
    assert any("expectativa" in item for item in _gate(negative, "G2").failures)
    low_dsr = evaluate(
        _input(backtest=_backtest(dsr=0.94)),
        _thresholds(),
    )
    assert low_dsr.verdict == "NO-GO"
    assert any("DSR" in item for item in _gate(low_dsr, "G2").failures)
    no_grid = evaluate(_input(), _thresholds())
    assert any("sin rejilla" in item for item in _gate(no_grid, "G2").notes)
    bad_grid = evaluate(_input(backtest=_backtest(pbo=0.21)), _thresholds())
    assert any("PBO" in item for item in _gate(bad_grid, "G2").failures)
    ok_grid = evaluate(_input(backtest=_backtest(pbo=0.2)), _thresholds())
    assert _gate(ok_grid, "G2").passed
    assert ok_grid.verdict == "GO"


def test_conteo_oos_y_la_excepcion_de_baja_frecuencia():
    short = _trades(80)
    base = _backtest(
        trade_pnl=short,
        n_oos_trades=80,
        oos_is_maximum_available=True,
        oos_maximum_note="histórico IEX desde 2016: 80 cierres, no hay más",
    )
    orb = evaluate(_input(backtest=base), _thresholds())
    assert any("300" in item for item in _gate(orb, "G2").failures)
    momentum = evaluate(
        _input(strategy_id="intraday_momentum", backtest=base),
        _thresholds(),
    )
    assert _gate(momentum, "G2").passed
    assert momentum.verdict == "GO"
    silent = _backtest(trade_pnl=short, n_oos_trades=80, oos_is_maximum_available=True)
    denied = evaluate(
        _input(strategy_id="intraday_momentum", backtest=silent),
        _thresholds(),
    )
    assert any("explicitado" in item for item in _gate(denied, "G2").failures)
    mismatch = _backtest(trade_pnl=short, n_oos_trades=300)
    broken = evaluate(_input(backtest=mismatch), _thresholds())
    assert any("no coincide" in item for item in _gate(broken, "G2").failures)


def test_g2_exige_iex_y_costos_pesimistas():
    sip = evaluate(_input(backtest=_backtest(feed="sip")), _thresholds())
    assert any("iex" in item for item in _gate(sip, "G2").failures)
    touch = evaluate(_input(backtest=_backtest(fill_mode="touch")), _thresholds())
    assert any("cross" in item for item in _gate(touch, "G2").failures)
    stressed = evaluate(_input(backtest=_backtest(cost_multiplier=2.0)), _thresholds())
    assert any("cost_multiplier" in item for item in _gate(stressed, "G2").failures)
    late = evaluate(_input(backtest=_backtest(extra_latency_bars=1)), _thresholds())
    assert any("latencia" in item for item in _gate(late, "G2").failures)
    assert all(item.verdict == "NO-GO" for item in (sip, touch, stressed, late))


def test_robustez_anio_dias_y_holdout():
    half = evaluate(
        _input(robustness=_robust(pnl_by_year=(("2022", 50.0), ("2023", 50.0)))),
        _thresholds(),
    )
    assert _gate(half, "G3").passed
    dominated = evaluate(
        _input(robustness=_robust(pnl_by_year=(("2022", 90.0), ("2023", 10.0)))),
        _thresholds(),
    )
    assert any("2022" in item for item in _gate(dominated, "G3").failures)
    best = evaluate(
        _input(robustness=_robust(daily_pnl=(10.0, 10.0, 10.0, 10.0, 10.0, -1.0))),
        _thresholds(),
    )
    assert any("mejores días" in item for item in _gate(best, "G3").failures)
    for count, pnl, ok in ((0, 5.0, False), (2, 5.0, False), (1, 0.0, False), (1, 3.0, True)):
        record = evaluate(
            _input(robustness=_robust(holdout_unlock_count=count, holdout_pnl=pnl)),
            _thresholds(),
        )
        assert _gate(record, "G3").passed is ok
    slow = evaluate(
        _input(robustness=_robust(pnl_extra_latency=-0.1)),
        _thresholds(),
    )
    assert slow.verdict == "NO-GO"


def test_paper_paridad_slippage_e_incidentes():
    explained = evaluate(
        _input(paper=_paper(differences=(
            IntentDifference("abc", "barra_corregida_tardiamente"),
        ))),
        _thresholds(),
    )
    assert _gate(explained, "G4").passed
    assert explained.verdict == "GO"
    missing = evaluate(
        _input(paper=_paper(differences=(IntentDifference("abc", "sin_explicar"),))),
        _thresholds(),
    )
    assert any("sin explicar" in item for item in _gate(missing, "G4").failures)
    short = evaluate(_input(paper=_paper(n_sessions=19, n_trades=99)), _thresholds())
    assert short.verdict == "NO-GO"
    worse = tuple(5.0 for _ in range(20))
    modeled = tuple(0.0 for _ in range(20))
    slip = evaluate(
        _input(paper=_paper(slippage_real_bps=worse, slippage_modeled_bps=modeled)),
        _thresholds(),
    )
    assert any("slippage" in item for item in _gate(slip, "G4").failures)
    incident = evaluate(
        _input(paper=_paper(unexplained_risk_incidents=1)),
        _thresholds(),
    )
    assert any("incidentes" in item for item in _gate(incident, "G4").failures)


def test_un_simulacro_faltante_bloquea_go():
    record = evaluate(
        _input(drills=DrillEvidence(crash_restart=True, kill=False, half_session=True)),
        _thresholds(),
    )
    assert record.verdict == "NO-GO"
    assert any("kill" in item for item in _gate(record, "G5").failures)
    assert _gate(record, "G1").passed
    assert _gate(record, "G2").passed


def test_research_only_y_teoria_incompleta_bloquean():
    from src.hft.strategies import STRATEGIES

    card = STRATEGIES["avellaneda_stoikov"].theory
    assert card.research_only is True
    research = evaluate(
        _input(strategy_id="avellaneda_stoikov", theory=card),
        _thresholds(),
    )
    assert research.verdict == "NO-GO"
    assert any("research_only" in item for item in _gate(research, "G1").failures)
    empty = evaluate(_input(theory=_theory(mechanism="  ")), _thresholds())
    assert any("mechanism" in item for item in _gate(empty, "G1").failures)
    no_kill = evaluate(_input(theory=_theory(kill_criteria=())), _thresholds())
    assert any("kill_criteria" in item for item in _gate(no_kill, "G1").failures)


def test_el_registro_deja_la_firma_en_blanco(tmp_path: Path):
    record = evaluate(_input(), _thresholds())
    path = write_decision(record, decision_path("orb", "2026-10-06", tmp_path))
    assert path == tmp_path / "orb-2026-10-06.md"
    text = path.read_text(encoding="utf-8")
    assert "**Veredicto: GO**" in text
    assert "El sistema no promueve la estrategia" in text
    assert "results/hft/backtests/run-1" in text
    assert "results/hft/trials.csv" in text
    assert "results/hft/reports/2026-10-06.html" in text
    assert "### G1 Teoría — PASS" in text
    assert text.strip().endswith("Firma:")
    assert "Nombre:" in text
    with pytest.raises(FileExistsError):
        write_decision(record, path)
    failed = evaluate(_input(robustness=_robust(pnl_costs_x2=-1.0)), _thresholds())
    blocked = render_markdown(failed)
    assert "**Veredicto: NO-GO**" in blocked
    assert "**Veredicto: GO**" not in blocked


def test_cli_imprime_el_veredicto_y_no_promueve(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    payload = _payload()
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    code = main([
        "--strategy", "orb",
        "--date", "2026-10-06",
        "--evidence", str(evidence),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0].startswith("Veredicto: GO")
    assert "no promueve" in out
    written = tmp_path / "results" / "hft" / "decisions" / "orb-2026-10-06.md"
    assert written.is_file()
    assert written.read_text(encoding="utf-8").strip().endswith("Firma:")
    card = theory_of("orb")
    assert card.research_only is False
    parsed = parse_evidence(payload, "orb", "2026-10-06", card)
    assert evaluate(parsed, _thresholds()).verdict == "GO"


def test_cli_no_go_sale_2(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    payload = _payload()
    payload["drills"]["kill"] = False
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    code = main(["--strategy", "orb", "--date", "2026-10-06", "--evidence", str(evidence)])
    out = capsys.readouterr().out
    assert code == 2
    assert out.splitlines()[0] == "Veredicto: NO-GO. El sistema no promueve la estrategia."
    assert "Veredicto: GO" not in out


def _payload() -> dict:
    pnl = list(_trades())
    return {
        "backtest": {
            "feed": "iex",
            "fill_mode": "cross",
            "cost_multiplier": 1,
            "extra_latency_bars": 0,
            "trade_pnl": pnl,
            "n_oos_trades": len(pnl),
            "dsr": 0.97,
            "n_trials": 3,
            "pbo": None,
            "oos_is_maximum_available": False,
            "oos_maximum_note": "",
        },
        "robustness": {
            "pnl_costs_x2": 12,
            "pnl_extra_latency": 4,
            "pnl_by_year": {"2022": 40, "2023": 30, "2024": 30},
            "daily_pnl": [1, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7],
            "holdout_unlock_count": 1,
            "holdout_pnl": 5,
        },
        "paper": {
            "n_sessions": 20,
            "n_trades": 100,
            "differences": [],
            "slippage_real_bps": [0] * 20,
            "slippage_modeled_bps": [0] * 20,
            "unexplained_risk_incidents": 0,
        },
        "drills": {
            "crash_restart": True,
            "kill": True,
            "half_session": True,
        },
        "links": {
            "backtests": ["results/hft/backtests/run-1"],
            "trials": "results/hft/trials.csv",
            "reports": ["results/hft/reports/2026-10-06.html"],
        },
    }
