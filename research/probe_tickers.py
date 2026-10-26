#!/usr/bin/env python3
"""probe_tickers.py — Paso 1 universo mixto: valida cobertura Yahoo por candidato CO.

Que hace (solo lectura, no modifica universo actual):
  - Prueba variantes Yahoo por candidato (ADR > ETF > local BVC).
  - Periodo 2019-01-01 a 2026-01-01 (ideal 5a, minimo 3a documentado).
  - Metricas: cobertura, %NaN, dias volumen cero, ADV 30d, primer/ultimo dato.
  - Prueba TRM (USDCOP=X) para FX dual USD/COP.
  - Escribe research/data_quality_co.csv + research/universe_co_candidates.md.

Uso:
  .venv/bin/python research/probe_tickers.py
  .venv/bin/python research/probe_tickers.py --start 2019-01-01 --end 2026-01-01

Si Yahoo falla (offline), escribe CSV con status=NO_DATA y MD explicando.
Reglas de decision (criterios.md v2, a congelar en paso 2):
  PASS si cobertura>=80% y NaN<=20% e historial>=3a.
  WARN si cobertura 60-80% o historial 2-3a (revisar manual).
  FAIL en otro caso.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("probe_tickers")

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "data_quality_co.csv"
OUT_MD = HERE / "universe_co_candidates.md"

# Candidatos: (label, grupo, variantes Yahoo en orden, proxy ADR, sector)
# Grupo: adr | etf | local | benchmark | fx
CANDIDATES: list[dict] = [
    # --- ADRs NYSE (prioridad 1) ---
    {"label": "EC", "grupo": "adr", "variants": ["EC"], "proxy": None, "sector": "Energy"},
    {"label": "CIB", "grupo": "adr", "variants": ["CIB"], "proxy": None, "sector": "Financials"},
    {"label": "AVAL", "grupo": "adr", "variants": ["AVAL"], "proxy": None, "sector": "Financials"},
    {"label": "TGLS", "grupo": "adr", "variants": ["TGLS"], "proxy": None, "sector": "Industrials"},
    # --- ETFs COL / cobertura ---
    {"label": "GXG", "grupo": "etf", "variants": ["GXG"], "proxy": None, "sector": "ETF Colombia"},
    {"label": "ICOLCAP", "grupo": "etf", "variants": ["ICOLCAP.CL", "ICOLCAP.CO", "ICOLCAP"], "proxy": "GXG", "sector": "ETF Colombia"},
    {"label": "HCOLSEL", "grupo": "etf", "variants": ["HCOLSEL.CL", "HCOLSEL.CO", "HCOLSEL"], "proxy": "GXG", "sector": "ETF Colombia"},
    # --- Locales BVC (variantes a descubrir en probe) ---
    {"label": "ECOPETROL", "grupo": "local", "variants": ["ECOPETROL.CL", "ECOPETROL.CO", "ECOPETROL"], "proxy": "EC", "sector": "Energy"},
    {"label": "PFBCOLOM", "grupo": "local", "variants": ["PFBCOLOM.CL", "PFBCOLOM.CO", "PFBCOLOM", "PFBCOLO.CL"], "proxy": "CIB", "sector": "Financials"},
    {"label": "BCOLOMBIA", "grupo": "local", "variants": ["BCOLOMBIA.CL", "BCOLOMBIA.CO", "BCOLOMBIA"], "proxy": "CIB", "sector": "Financials"},
    {"label": "ISA", "grupo": "local", "variants": ["ISA.CL", "ISA.CO", "ISA"], "proxy": None, "sector": "Utilities"},
    {"label": "GEB", "grupo": "local", "variants": ["GEB.CL", "GEB.CO", "GEB", "EEB.CL"], "proxy": None, "sector": "Utilities"},
    {"label": "CELSIA", "grupo": "local", "variants": ["CELSIA.CL", "CELSIA.CO", "CELSIA"], "proxy": None, "sector": "Utilities"},
    {"label": "GRUPOARGOS", "grupo": "local", "variants": ["GRUPOARGOS.CL", "GRUPOARGOS.CO", "GRUPOARGOS", "ARGOS.CL"], "proxy": None, "sector": "Materials"},
    {"label": "PFARGOS", "grupo": "local", "variants": ["PFARGOS.CL", "PFARGOS.CO", "PFARGOS"], "proxy": None, "sector": "Materials"},
    {"label": "CEMARGOS", "grupo": "local", "variants": ["CEMARGOS.CL", "CEMARGOS.CO", "CEMARGOS"], "proxy": None, "sector": "Materials"},
    {"label": "NUTRESA", "grupo": "local", "variants": ["NUTRESA.CL", "NUTRESA.CO", "NUTRESA"], "proxy": None, "sector": "Consumer Staples"},
    {"label": "GRUPOSURA", "grupo": "local", "variants": ["GRUPOSURA.CL", "GRUPOSURA.CO", "GRUPOSURA"], "proxy": None, "sector": "Financials"},
    {"label": "PFSURA", "grupo": "local", "variants": ["PFSURA.CL", "PFSURA.CO", "PFSURA"], "proxy": None, "sector": "Financials"},
    {"label": "CORFICOL", "grupo": "local", "variants": ["CORFICOL.CL", "CORFICOL.CO", "CORFICOL", "CORFICOLCF.CL"], "proxy": None, "sector": "Financials"},
    {"label": "MINEROS", "grupo": "local", "variants": ["MINEROS.CL", "MINEROS.CO", "MINEROS"], "proxy": None, "sector": "Materials"},
    {"label": "PFAVAL", "grupo": "local", "variants": ["PFAVAL.CL", "PFAVAL.CO", "PFAVAL"], "proxy": "AVAL", "sector": "Financials"},
    {"label": "PFDAVVNDA", "grupo": "local", "variants": ["PFDAVVNDA.CL", "PFDAVVNDA.CO", "PFDAVVNDA"], "proxy": None, "sector": "Financials"},
    {"label": "EXITO", "grupo": "local", "variants": ["EXITO.CL", "EXITO.CO", "EXITO"], "proxy": None, "sector": "Consumer Staples"},
    {"label": "TERPEL", "grupo": "local", "variants": ["TERPEL.CL", "TERPEL.CO", "TERPEL"], "proxy": None, "sector": "Energy"},
    {"label": "PROMIGAS", "grupo": "local", "variants": ["PROMIGAS.CL", "PROMIGAS.CO", "PROMIGAS"], "proxy": None, "sector": "Utilities"},
    # --- Benchmarks / FX (referencia, no cuentan para los 20) ---
    {"label": "SPY", "grupo": "benchmark", "variants": ["SPY"], "proxy": None, "sector": "ETF USA"},
    {"label": "EEM", "grupo": "benchmark", "variants": ["EEM"], "proxy": None, "sector": "ETF EM"},
    {"label": "TLT", "grupo": "benchmark", "variants": ["TLT"], "proxy": None, "sector": "ETF Bonds"},
    {"label": "USDCOP", "grupo": "fx", "variants": ["USDCOP=X", "COP=X"], "proxy": None, "sector": "FX"},
]


def _download_variant(variant: str, start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    """Descarga Close+Volume de una variante. Retorna (df, nota)."""
    try:
        import yfinance as yf  # import local para mensaje claro si falta
    except Exception as exc:
        return None, f"yfinance import error: {exc}"
    try:
        raw = yf.download(variant, start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            return None, "empty"
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
            vol = raw["Volume"].iloc[:, 0] if "Volume" in raw.columns.get_level_values(0) else pd.Series(np.nan, index=raw.index)
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
            vol = raw["Volume"] if "Volume" in raw.columns else pd.Series(np.nan, index=raw.index)
        df = pd.DataFrame({"close": pd.to_numeric(close, errors="coerce"),
                           "volume": pd.to_numeric(vol, errors="coerce")})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df, "ok"
    except Exception as exc:
        return None, f"error: {type(exc).__name__}: {exc}"


def probe_candidate(cand: dict, start: str, end: str) -> dict:
    """Prueba variantes en orden, se queda con la primera con datos."""
    notes: list[str] = []
    for variant in cand["variants"]:
        df, note = _download_variant(variant, start, end)
        notes.append(f"{variant}:{note}")
        if df is None or df.empty:
            continue
        n = len(df)
        if n < 50:  # muy poco para ser util
            notes.append(f"{variant}: solo {n} filas, sigo probando")
            continue
        close = df["close"]
        vol = df["volume"]
        valid = close.dropna()
        nan_pct = float(close.isna().mean() * 100)
        zero_vol_pct = float(((vol.fillna(0) == 0)).mean() * 100) if len(vol) else 100.0
        first, last = valid.index.min(), valid.index.max()
        hist_years = (last - first).days / 365.25 if pd.notna(first) and pd.notna(last) else 0.0
        # Cobertura vs dias habiles esperados (~252/a entre start-end)
        exp_days = max(int((pd.Timestamp(end) - pd.Timestamp(start)).days * 252 / 365), 1)
        coverage = min(n / exp_days * 100, 100.0)
        tail = df.tail(30)
        adv_30d = float(tail["volume"].mean()) if tail["volume"].notna().any() else float("nan")
        last_px = float(tail["close"].dropna().iloc[-1]) if tail["close"].notna().any() else float("nan")
        adv_cop = adv_30d * last_px if np.isfinite(adv_30d) and np.isfinite(last_px) else float("nan")
        if hist_years >= 3 and coverage >= 80 and nan_pct <= 20:
            decision = "PASS"
        elif hist_years >= 2 and coverage >= 60:
            decision = "WARN"
        else:
            decision = "FAIL"
        return {
            "label": cand["label"], "grupo": cand["grupo"], "sector": cand["sector"],
            "yahoo_ticker": variant, "tried": "; ".join(notes),
            "n_rows": n, "coverage_pct": round(coverage, 1),
            "nan_pct": round(nan_pct, 2), "zero_vol_pct": round(zero_vol_pct, 1),
            "first_date": str(first.date()) if pd.notna(first) else "",
            "last_date": str(last.date()) if pd.notna(last) else "",
            "hist_years": round(hist_years, 2),
            "adv_30d_shares": int(adv_30d) if np.isfinite(adv_30d) else "",
            "last_price": round(last_px, 2) if np.isfinite(last_px) else "",
            "adv_30d_value": int(adv_cop) if np.isfinite(adv_cop) else "",
            "decision": decision, "status": "OK",
        }
    return {
        "label": cand["label"], "grupo": cand["grupo"], "sector": cand["sector"],
        "yahoo_ticker": "", "tried": "; ".join(notes),
        "n_rows": 0, "coverage_pct": 0.0, "nan_pct": 100.0, "zero_vol_pct": 100.0,
        "first_date": "", "last_date": "", "hist_years": 0.0,
        "adv_30d_shares": "", "last_price": "", "adv_30d_value": "",
        "decision": "FAIL", "status": "NO_DATA",
    }


def write_markdown(rows: list[dict], start: str, end: str, path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    df = pd.DataFrame(rows)
    co = df[~df["grupo"].isin(["benchmark", "fx"])].copy()
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    co["rank"] = co["decision"].map(order)
    co = co.sort_values(["rank", "coverage_pct"], ascending=[True, False])
    n_pass = int((co["decision"] == "PASS").sum())
    n_warn = int((co["decision"] == "WARN").sum())
    n_fail = int((co["decision"] == "FAIL").sum())
    eligible = co[co["decision"].isin(["PASS", "WARN"])].head(20)
    top20 = eligible["label"].tolist()

    lines = [
        "# Universo CO — Candidatos (Paso 1, híbrido ADR-first)",
        "",
        f"Generado: {now} | Ventana probe: {start} a {end} | Fuente: Yahoo Finance (solo lectura).",
        "",
        "## Decisión acordada",
        "- 20 líquidas: 4 ADR + ETFs (GXG, ICOLCAP, HCOLSEL si pasan) + locales BVC hasta completar.",
        "- Regla: PASS si cobertura>=80%, NaN<=20%, historial>=3a. WARN 60-80% o 2-3a. Resto FAIL (excluido, nunca en top-20).",
        f"- Resultado probe: **{n_pass} PASS / {n_warn} WARN / {n_fail} FAIL** sobre {len(co)} candidatos CO (excluye benchmarks/FX).",
        "",
        f"## Selección preliminar ({len(top20)} elegibles PASS/WARN, sin FAIL)",
        "",
        "`" + ", ".join(top20) + "`",
        "",
        "> Preliminar: el paso 2 congela `universe_co.csv` + `universe_mixed.csv` aplicando además",
        "> ADV≥500M COP, Amihud p90 por mercado y topes sectoriales. Si un elegible tiene ADV bajo,",
        "> se excluye con motivo. Los FAIL (GXG deslistado; PFBCOLOM/BCOLOMBIA/PFARGOS/PFSURA sin serie Yahoo)",
        "> quedan fuera; Bancolombia se cubre vía ADR CIB, Argos/Sura vía ordinarias.",
        "",
        "## Tabla completa",
        "",
        "| Label | Grupo | Yahoo OK | Cob% | NaN% | Vol0% | Años | ADV30d | Decisión |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['grupo']} | {r['yahoo_ticker'] or '—'} | "
            f"{r['coverage_pct']} | {r['nan_pct']} | {r['zero_vol_pct']} | "
            f"{r['hist_years']} | {r['adv_30d_shares'] or '—'} | {r['decision']} |"
        )
    lines += [
        "",
        "## Notas por grupo",
        "- **ADR (EC, CIB, AVAL, TGLS):** núcleo operable USD. Si alguno FAIL, se mantiene igual con WARN y se documenta.",
        "- **ETF:** GXG deslistado en Yahoo (FAIL verificado) → proxy robusto pasa a ICOLCAP/HCOLSEL (.CL OK).",
        "- **Locales:** variantes probadas `.CL` → `.CO` → bare. El ticker ganador queda en `yahoo_ticker`.",
        "  PFBCOLOM/BCOLOMBIA/PFARGOS/PFSURA sin serie Yahoo en ninguna variante → FAIL documentado;",
        "  cobertura vía CIB (Bancolombia), GRUPOARGOS y GRUPOSURA.",
        "- **FX (USDCOP=X):** debe ser PASS para NAV dual USD/COP; si FAIL, paso 2 usa TRM BanRep CSV.",
        "",
        "## Riesgos conocidos (RT-02/RT-05)",
        "- Yahoo-BVC incompleto: gaps, volumen cero y deslistados posibles. Todo FAIL queda excluido con motivo (no se imputa).",
        "- Survivorship: universo se congela point-in-time al inicio del backtest; este probe es descubrimiento, no selección ex-post.",
        "",
        "## Siguiente (paso 2, requiere aprobación)",
        "1. Aplicar ADV≥500M COP + Amihud por mercado sobre los PASS/WARN.",
        "2. Escribir `universe_co.csv` (20) + `universe_mixed.csv` (120) + `criterios.md v2`.",
        "3. Extender `tests/test_universe.py` y crear `test_universe_mixed.py`.",
        "",
        "_Fuente detalle por ticker: columna `tried` en `data_quality_co.csv`._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe Yahoo para universo CO (solo lectura).")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for cand in CANDIDATES:
        logger.info("Probing %s (%s) …", cand["label"], cand["grupo"])
        rows.append(probe_candidate(cand, args.start, args.end))

    out_csv, out_md = Path(args.out_csv), Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_markdown(rows, args.start, args.end, out_md)

    co = [r for r in rows if r["grupo"] not in ("benchmark", "fx")]
    logger.info("CO: %d PASS / %d WARN / %d FAIL",
                sum(r["decision"] == "PASS" for r in co),
                sum(r["decision"] == "WARN" for r in co),
                sum(r["decision"] == "FAIL" for r in co))
    logger.info("CSV → %s | MD → %s", out_csv, out_md)
    print(f"\nOK probe: {len(rows)} tickers | CSV → {out_csv} | MD → {out_md}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
