"""Offline intraday backtest on synthetic 1-minute bars (no network, no data/).

Builds a fake NYSE session calendar, generates seeded synthetic 1-minute bars
for every symbol the config trades (``src.hft.data.synthetic.synthetic_bars``:
random walk with mild mean reversion and rare jumps), and runs the real
backtest engine (strategies → risk → pessimistic fill simulator → reports)
over it. Nothing is written to the repo: runtime paths go to a temp HFT_HOME,
``persist=False`` and ``record_trial=False``.

The numbers it prints are meaningless as performance claims; they show the
pipeline running end to end. Dates are before the config's holdout start.

    python scripts/demo_synthetic.py [--sessions 20] [--warmup 15] [--seed 7] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", type=int, default=20, help="total synthetic sessions")
    ap.add_argument("--warmup", type=int, default=15, help="history-only sessions before the backtest window")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="show per-order risk/sizing log lines")
    args = ap.parse_args(argv)
    if args.warmup >= args.sessions:
        ap.error("--warmup must be smaller than --sessions")

    if not args.verbose:
        logging.disable(logging.WARNING)
    os.environ.setdefault("HFT_HOME", tempfile.mkdtemp(prefix="hft-demo-"))

    from src.hft.backtest import run
    from src.hft.calendar import FakeCalendar, session_at
    from src.hft.config import load_hft_config
    from src.hft.data.synthetic import FakeIntradayProvider, synthetic_bars
    from src.hft.session import config_symbols

    days = list(pd.bdate_range("2025-09-02", periods=args.sessions).date)
    sessions = [session_at(day, time(9, 30), time(16, 0)) for day in days]
    config = load_hft_config()
    symbols = config_symbols(config)
    bars = synthetic_bars(
        sessions, symbols, seed=args.seed, sigma=0.0008,
        ou_kappa=0.02, jump_prob=0.0005, jump_scale=0.004,
    )
    report = run(
        config, days[args.warmup], days[-1],
        provider=FakeIntradayProvider(bars),
        calendar=FakeCalendar(sessions),
        record_trial=False, persist=False, seed=args.seed,
    )

    closed = report.blotter[report.blotter["closed"] == True] if len(report.blotter) else report.blotter  # noqa: E712
    print("\nSYNTHETIC intraday backtest — illustrative only, not a performance claim")
    print(f"  window     {report.start} → {report.end}  ({args.sessions - args.warmup} sessions, "
          f"{len(symbols)} symbols, 1-minute bars)")
    print(f"  intents    {len(report.intents)}   rejections {len(report.rejections)}   "
          f"closed trades {len(closed)}")
    if len(report.attribution_strategy):
        print("\n  P&L attribution by strategy (synthetic):")
        print(report.attribution_strategy.to_string(index=False, max_cols=8))
    if report.metrics:
        keys = [k for k in ("sharpe", "max_drawdown", "n_days") if k in report.metrics]
        if keys:
            print("\n  " + "   ".join(f"{k}={report.metrics[k]}" for k in keys))
    print(f"\n  config hash {report.config_hash}  run id {report.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
