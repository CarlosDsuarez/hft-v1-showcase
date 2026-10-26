"""Paper intradía todos los días hábiles, sin que alguien tenga que lanzarlo.

``run`` corre ``scripts.run_intraday --mode paper --submit`` con la fecha de
Nueva York y lo relanza si la sesión muere por red, datos o un error
inesperado (exit 3, 5, 7) mientras el mercado sigue abierto. El journal del
día hace seguro el relanzamiento: las posiciones HFT vuelven a ser propias y
los intents ya enviados no se repiten. Riesgo (8), lock (4), guarda (2),
feriado (1), fin normal (0) o una señal del sistema no relanzan.

En macOS, ``run`` sostiene ``caffeinate -i -s`` mientras vive: el Mac no se
duerme por inactividad. Con la tapa cerrada macOS duerme igual y la sesión
pierde la red; por eso falló el paper del 2026-10-07.

``install`` deja un LaunchAgent (``~/Library/LaunchAgents``) que llama
``run`` de lunes a viernes a la hora local ``--at`` (default 08:15, antes de
la apertura en Colombia todo el año). Si el Mac duerme a esa hora, launchd lo
corre al despertar. ``uninstall`` lo quita y ``status`` lo muestra.

``run --with hourly_momentum`` reenvía ``--with`` a ``run_intraday``: el set
por defecto más esa estrategia. ``install --with hourly_momentum`` lo deja en
el plist. Sin ``--with`` el comando y el plist son los de siempre.
"""

from __future__ import annotations

import argparse
import logging
import os
import plistlib
import shutil
import subprocess
import sys
import time as time_mod
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

from src.hft.session import (
    EXIT_GUARD,
    EXIT_LOCK,
    EXIT_NO_SESSION,
    EXIT_OK,
    EXIT_RISK,
    WITH_STRATEGIES,
)

logger = logging.getLogger("hft.paper_daily")

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.hftv1.paper"
STOP_CODES = frozenset({EXIT_OK, EXIT_NO_SESSION, EXIT_GUARD, EXIT_LOCK, EXIT_RISK})
RETRY_S = 60.0
MAX_RUNS = 20
CLOSE_NY = time(16, 0)
DEFAULT_AT = "08:15"


def ny_date(now: datetime) -> date:
    """Fecha de sesión: el día en Nueva York."""
    return now.astimezone(NY).date()


def close_utc(day: date) -> datetime:
    """Cierre regular (16:00 Nueva York) en UTC; tope para relanzar."""
    return datetime.combine(day, CLOSE_NY, tzinfo=NY).astimezone(UTC)


def session_command(
    python: str, day: date, *, submit: bool = True, with_strategies: Sequence[str] = (),
) -> list[str]:
    """Línea de ``run_intraday`` para la sesión paper de ``day``."""
    cmd = [str(python), "-m", "scripts.run_intraday", "--mode", "paper"]
    if submit:
        cmd.append("--submit")
    return cmd + _with_flags(with_strategies) + ["--date", day.isoformat()]


def _with_flags(with_strategies: Sequence[str]) -> list[str]:
    return [item for name in with_strategies for item in ("--with", name)]


def supervise(
    run: Callable[[], int],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    *,
    stop_at: datetime,
    retry_s: float = RETRY_S,
    max_runs: int = MAX_RUNS,
) -> int:
    """Corre ``run`` y lo relanza por fallos transitorios hasta ``stop_at``."""
    runs = 0
    while True:
        code = run()
        runs += 1
        if code < 0 or code in STOP_CODES:
            logger.info("sesión terminó con exit %s; no se relanza", code)
            return code
        if runs >= max_runs:
            logger.error("exit %s y %d intentos: se rinde por hoy", code, runs)
            return code
        if now().timestamp() + retry_s >= stop_at.timestamp():
            logger.error("exit %s después del cierre: no se relanza", code)
            return code
        logger.warning("exit %s; relanza en %.0f s (intento %d)", code, retry_s, runs + 1)
        sleep(retry_s)


def parse_at(text: str) -> tuple[int, int]:
    """``HH:MM`` local a ``(hora, minuto)``."""
    hour, minute = (int(part) for part in text.split(":"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"hora fuera de rango: {text}")
    return hour, minute


def render_plist(
    repo_root: Path,
    python: Path,
    *,
    hour: int,
    minute: int,
    log_path: Path,
    with_strategies: Sequence[str] = (),
) -> dict:
    """LaunchAgent que corre ``run`` de lunes a viernes. Sin secretos: el
    runner lee ``.env`` de la raíz del repo. ``with_strategies`` va a
    ``run`` como ``--with``; sin él, el plist es el de siempre."""
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python), "-m", "scripts.hft_paper_daily", "run", *_with_flags(with_strategies),
        ],
        "WorkingDirectory": str(repo_root),
        "StartCalendarInterval": [
            {"Weekday": day, "Hour": hour, "Minute": minute} for day in range(1, 6)
        ],
        "RunAtLoad": False,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    }


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "hft-v1" / "paper-daily.log"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _keep_awake() -> None:
    """``caffeinate`` atado a este proceso: muere cuando ``run`` termina."""
    tool = shutil.which("caffeinate")
    if sys.platform == "darwin" and tool:
        subprocess.Popen([tool, "-i", "-s", "-w", str(os.getpid())])
    else:
        logger.warning("sin caffeinate: el sistema puede dormirse durante la sesión")


def _run_main(args) -> int:
    def clock() -> datetime:
        return datetime.now(UTC)

    day = ny_date(clock())
    cmd = session_command(
        sys.executable, day, submit=not args.dry_run,
        with_strategies=tuple(dict.fromkeys(args.with_strategies)),
    )
    _keep_awake()
    logger.info("paper diario %s: %s", day.isoformat(), " ".join(cmd))
    return supervise(
        lambda: subprocess.call(cmd, cwd=REPO_ROOT),
        clock,
        time_mod.sleep,
        stop_at=close_utc(day),
    )


def _install(args) -> int:
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        print(f"falta {python}: corre make install en {REPO_ROOT}", file=sys.stderr)
        return 2
    if ".claude/worktrees/" in REPO_ROOT.as_posix():
        print(f"{REPO_ROOT} es un worktree: instala desde el checkout principal", file=sys.stderr)
        return 2
    hour, minute = parse_at(args.at)
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plist = _plist_path()
    plist.parent.mkdir(parents=True, exist_ok=True)
    with_strategies = tuple(dict.fromkeys(args.with_strategies))
    plist.write_bytes(plistlib.dumps(render_plist(
        REPO_ROOT, python, hour=hour, minute=minute, log_path=log_path,
        with_strategies=with_strategies,
    )))
    subprocess.call(["launchctl", "bootout", f"{_domain()}/{LABEL}"], stderr=subprocess.DEVNULL)
    code = subprocess.call(["launchctl", "bootstrap", _domain(), str(plist)])
    if code != 0:
        print(f"launchctl bootstrap falló ({code})", file=sys.stderr)
        return code
    print(f"instalado {plist}: lunes a viernes {hour:02d}:{minute:02d} hora local")
    if with_strategies:
        print(f"set por defecto + {', '.join(with_strategies)}")
    print(f"log: {log_path}")
    return 0


def _uninstall(_args) -> int:
    subprocess.call(["launchctl", "bootout", f"{_domain()}/{LABEL}"], stderr=subprocess.DEVNULL)
    _plist_path().unlink(missing_ok=True)
    print(f"desinstalado {LABEL}")
    return 0


def _status(_args) -> int:
    plist = _plist_path()
    if not plist.is_file():
        print(f"{LABEL} no está instalado")
        return 1
    print(f"plist: {plist}")
    print(f"log: {_log_path()}")
    return subprocess.call(["launchctl", "print", f"{_domain()}/{LABEL}"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper intradía diario (launchd).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Sesión paper de hoy con relanzamiento.")
    run.add_argument("--dry-run", action="store_true", help="Sin --submit: no envía órdenes.")
    install = sub.add_parser("install", help="LaunchAgent de lunes a viernes.")
    install.add_argument("--at", default=DEFAULT_AT, help="HH:MM hora local. Default 08:15.")
    for command in (run, install):
        command.add_argument(
            "--with",
            dest="with_strategies",
            action="append",
            choices=WITH_STRATEGIES,
            default=[],
            help="Estrategia opt-in que run_intraday suma al set por defecto. Repetible.",
        )
    sub.add_parser("uninstall", help="Quita el LaunchAgent.")
    sub.add_parser("status", help="Estado del LaunchAgent.")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    from src.settings import setup_logging

    setup_logging()
    handlers = {"run": _run_main, "install": _install, "uninstall": _uninstall, "status": _status}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
