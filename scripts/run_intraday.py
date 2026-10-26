"""Runner intradía (HFT-12) y guarda ``research_only`` (HFT-09E).

El ciclo de sesión vive en ``src.hft.session``. Este script es la línea
de comandos: modos ``signals``, ``sim`` y ``paper``, preflight y los
códigos de salida. El reloj de pared solo se lee aquí.

``--strategy`` conserva la guarda de HFT-09E (no abre sesión). Sin esa
bandera, ``--mode paper`` a secas también: así los tests de la guarda
siguen viendo el exit 2. Una sesión de verdad lleva ``--date``,
``--submit``, ``--adopt``, ``--until``, ``--allow-research``,
``--config`` o ``--with``, o arranca en ``signals`` (el default).

``--with hourly_momentum`` suma esa estrategia al set por defecto de la
sesión (``src.hft.session.WITH_STRATEGIES``). Sin ``--with`` el set no
cambia. No es ``research_only``; en paper sigue sin enviar nada sin
``--submit``.

``kill_flattens`` es verdadero y no es una clave de ``config/hft.toml``.
El HTML de HFT-13 no se genera aquí.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Sequence

from src.hft.engine import assert_paper_allowed

UTC = timezone.utc

_SESSION_FLAGS = (
    "--date",
    "--submit",
    "--adopt",
    "--until",
    "--allow-research",
    "--config",
    "--with",
    "--gateway",
    "--http-workers",
    "--trading-rpm",
    "--trade-stream",
)


def enforce_research_policy(strategies: Sequence[object], mode: str) -> None:
    """Rechaza paper si alguna estrategia es ``research_only``.

    ``mode`` es ``backtest``, ``sim`` o ``paper``. Otro valor es error,
    el mismo que el del motor.
    """
    assert_paper_allowed(strategies, mode)


class _WallClock:
    """Reloj de pared. Único lugar del runner que llama ``datetime.now``."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep_until(self, target: datetime) -> None:
        if target.tzinfo is None:
            raise ValueError("sleep_until exige un datetime con zona")
        delay = (target.astimezone(UTC) - self.now()).total_seconds()
        if delay > 0:
            import time as time_mod

            time_mod.sleep(delay)


def main(argv: Sequence[str] | None = None) -> int:
    """Guarda HFT-09E o una sesión intradía, según los argumentos."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if _is_guard_invocation(raw):
        return _guard_main(raw)
    return _session_main(raw)


def _is_guard_invocation(argv: Sequence[str]) -> bool:
    """True si la línea es la guarda de HFT-09E y no una sesión."""
    if _has_flag(argv, "--strategy"):
        return True
    if any(_has_flag(argv, flag) for flag in _SESSION_FLAGS):
        return False
    mode = _flag_value(argv, "--mode")
    if mode is None:
        return False
    return mode.strip().lower() != "signals"


def _guard_main(argv: Sequence[str]) -> int:
    """Comprueba la guarda y termina. No abre sesión ni habla con el broker."""
    parser = argparse.ArgumentParser(
        description=(
            "Guarda HFT-09E: paper rechaza estrategias research_only. "
            "No corre la sesión intradía."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="backtest, sim o paper. paper no admite research_only.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Id registrado. Repetible. En paper, uno research_only termina en 2.",
    )
    args = parser.parse_args(list(argv))
    from src.hft.strategies import STRATEGIES
    from src.hft.strategies.hourly_momentum import DEFAULT_HOURLY_MOMENTUM

    # hourly_momentum se registra al importar su módulo, no el paquete.
    known = {**STRATEGIES, DEFAULT_HOURLY_MOMENTUM.id: DEFAULT_HOURLY_MOMENTUM}
    missing = [name for name in args.strategy if name not in known]
    if not args.strategy or missing:
        print(
            "hace falta --strategy con un id registrado"
            + (f"; no están: {missing}" if missing else ""),
            file=sys.stderr,
        )
        return 2
    selected = [known[name] for name in args.strategy]
    try:
        enforce_research_policy(selected, args.mode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _session_main(argv: Sequence[str]) -> int:
    """Una sesión. El default es ``signals`` y no hace POST."""
    from src.hft.config import DEFAULT_CONFIG_PATH, load_hft_config
    from src.hft.session import (
        EXIT_BROKER,
        EXIT_GUARD,
        WITH_STRATEGIES,
        SessionRequest,
        run_session,
    )
    from src.settings import load_settings, setup_logging

    parser = argparse.ArgumentParser(description="Sesión intradía (HFT-12).")
    parser.add_argument(
        "--mode",
        choices=("signals", "sim", "paper"),
        default="signals",
        help="signals (default, CSV manual), sim o paper.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="En paper, POST de las órdenes. Sin esto, paper solo escribe el journal.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="TOML estricto. Default: config/hft.toml.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD. Default: hoy en America/New_York.",
    )
    parser.add_argument(
        "--until",
        type=_hhmm,
        default=None,
        help="HH:MM en hora de Nueva York. Corta y aplana.",
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="Posiciones u órdenes ajenas se excluyen del riesgo HFT y quedan en el journal.",
    )
    parser.add_argument(
        "--allow-research",
        action="store_true",
        help="Incluye Avellaneda–Stoikov. Prohibido con --mode paper.",
    )
    parser.add_argument(
        "--with",
        dest="with_strategies",
        action="append",
        choices=WITH_STRATEGIES,
        default=[],
        help="Suma una estrategia opt-in al set por defecto. Repetible. "
             "En paper sigue haciendo falta --submit para enviar.",
    )
    parser.add_argument(
        "--gateway",
        choices=("legacy", "pooled"),
        default="legacy",
        help="legacy (default): urlopen en serie. pooled: keep-alive, hilos y cupo "
             "local de la Trading API (ver --http-workers y --trading-rpm).",
    )
    parser.add_argument(
        "--http-workers",
        type=int,
        default=8,
        help="Con --gateway pooled: POST en vuelo a la vez.",
    )
    parser.add_argument(
        "--trading-rpm",
        type=float,
        default=200.0,
        help="Con --gateway pooled: cupo por minuto de la Trading API (Alpaca: 200).",
    )
    parser.add_argument(
        "--trade-stream",
        action="store_true",
        help="Fills por el websocket trade_updates (D-07 revocada). REST queda de reconciliación.",
    )
    args = parser.parse_args(list(argv))
    setup_logging()
    settings = load_settings()
    try:
        config = load_hft_config(args.config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_GUARD
    clock = _WallClock()
    session_date = _session_date(args.date, clock)
    if args.mode == "paper" and args.allow_research:
        print("--allow-research está prohibido en paper", file=sys.stderr)
        return EXIT_GUARD
    if args.submit and args.mode != "paper":
        print("--submit solo aplica a --mode paper", file=sys.stderr)
        return EXIT_GUARD
    if args.http_workers < 1 or args.trading_rpm <= 0:
        print("--http-workers >= 1 y --trading-rpm > 0", file=sys.stderr)
        return EXIT_GUARD
    if args.trade_stream and args.mode != "paper":
        print("--trade-stream solo aplica a --mode paper", file=sys.stderr)
        return EXIT_GUARD

    from src.hft.calendar import HISTORY_LOOKBACK_DAYS, NY, SessionCalendar

    # Las estrategias agrupan la historia por las sesiones del calendario y
    # el calentamiento pide las previas: cargar solo el día deja todo frío.
    history_start = session_date - timedelta(days=HISTORY_LOOKBACK_DAYS)
    cache = settings.data_dir / "hft" / "calendar.csv"
    calendar = SessionCalendar(client=None, cache_path=cache, use_cache=True)
    try:
        calendar.load(history_start, session_date)
    except Exception as exc:
        print(f"calendario: {exc}", file=sys.stderr)
        return EXIT_BROKER
    broker = None
    execution = None
    expected_account = None
    if args.mode == "paper" and _has_session(calendar, session_date):
        built = _paper_broker(
            config, clock,
            gateway=_gateway(args), trade_stream=bool(args.trade_stream and args.submit),
        )
        if isinstance(built, int):
            return built
        broker, execution, expected_account = built
        if not args.submit:
            execution = None
        try:
            live_calendar = SessionCalendar(client=broker, cache_path=cache, use_cache=True)
            live_calendar.load(history_start, session_date)
            calendar = live_calendar
        except Exception as exc:
            print(f"calendario del broker no disponible ({exc}); sigue el de respaldo", file=sys.stderr)
    api, api_error = _api_provider(config, clock)
    if api is None and args.mode in ("signals", "paper") and _has_session(calendar, session_date):
        from src.hft.session import EXIT_DATA

        # Sin API solo queda el BarStore, que no tiene las barras de hoy:
        # la sesión latiría hasta el cierre sin emitir una señal.
        print(
            f"--mode {args.mode} necesita la API de datos de Alpaca: {api_error}",
            file=sys.stderr,
        )
        return EXIT_DATA
    provider = _provider(config, settings, api)
    momentum_closes = (
        _momentum_closes(config, api.client, calendar, session_date) if api is not None else None
    )
    return run_session(SessionRequest(
        config=config,
        mode=args.mode,
        session_date=session_date,
        clock=clock,
        calendar=calendar,
        provider=provider,
        state_dir=settings.state_dir / "hft",
        results_dir=settings.results_dir,
        kill_dir=settings.state_dir,
        submit=bool(args.submit),
        until=args.until,
        adopt=bool(args.adopt),
        allow_research=bool(args.allow_research),
        execution=execution,
        broker=broker,
        expected_account_id=expected_account,
        momentum_closes=momentum_closes,
        with_strategies=tuple(dict.fromkeys(args.with_strategies)),
    ))


_HFT_UNIVERSES = ("universe_hft.csv", "universe_hft_pairs.csv")


def _hft_instruments():
    """Maestro del intradía: los ETF de ``universe_hft.csv`` y las patas de los pares.

    El default de ``AlpacaClient`` es ``universe_mixed.csv`` (runner diario),
    que no trae los ETF: con él toda orden intradía sale ``not_routable``.
    """
    from src.instruments import InstrumentMaster, load_instruments
    from src.settings import load_settings

    settings = load_settings()
    items = []
    for name in _HFT_UNIVERSES:
        items.extend(load_instruments(settings.universe_path(name)).values())
    return InstrumentMaster(items)


def _gateway(args):
    """``GatewayConfig`` de los flags. ``legacy`` es el de siempre."""
    from src.brokers.alpaca_adapter import GatewayConfig

    if args.gateway == "legacy":
        return GatewayConfig()
    return GatewayConfig(pooled=True, workers=int(args.http_workers),
                         trading_rpm=float(args.trading_rpm))


def _paper_broker(config, clock: _WallClock, *, gateway=None, trade_stream: bool = False):
    """Cliente paper y, si hay que enviar, el adaptador.

    Exit 3 si faltan claves o si algún símbolo de la sesión no se puede
    rutear: un par con una pata sin ruta quedaría desnudo. Con
    ``trade_stream`` el stream tiene que quedar escuchando: si no, exit 3.
    """
    from src.hft.session import EXIT_BROKER, config_symbols

    try:
        from src.brokers.alpaca_adapter import AlpacaClient, load_broker_config
        from src.hft.costs import CostModel
        from src.hft.execution.alpaca_exec import AlpacaExecution

        broker_config = load_broker_config()
        client = AlpacaClient(broker_config, instruments=_hft_instruments(), gateway=gateway)
        stream = None
        if trade_stream:
            from src.brokers.alpaca_stream import TradeUpdatesStream

            stream = TradeUpdatesStream(broker_config.api_key, broker_config.secret_key)
            if not stream.start():
                reason = stream.auth_error or "sin respuesta"
                stream.stop()
                raise RuntimeError(f"stream trade_updates no arrancó: {reason}")
        execution = AlpacaExecution(
            client,
            CostModel.from_config(config),
            clock=clock.now,
            stream=stream,
        )
    except Exception as exc:
        print(f"broker paper: {exc}", file=sys.stderr)
        return EXIT_BROKER
    unroutable = [
        f"{symbol} ({reason})"
        for symbol in config_symbols(config)
        for _broker_symbol, reason in [client._route(symbol)]
        if reason is not None
    ]
    if unroutable:
        print(
            "broker paper: símbolos sin ruta a Alpaca: " + "; ".join(unroutable)
            + f". Agregarlos a {' o '.join(_HFT_UNIVERSES)}.",
            file=sys.stderr,
        )
        return EXIT_BROKER
    # load_broker_config ya mezcla el .env de la raíz y el entorno.
    account = broker_config.account_id or None
    return client, execution, account


def _provider(config, settings, api):
    """BarStore primero. Si hay claves, la API completa los huecos."""
    from src.hft.data.bar_store import BarStore, StoreProvider

    store = StoreProvider(
        BarStore(settings.bars_dir),
        feed=config.feed,
        timeframe=config.timeframe,
    )
    if api is None:
        return store
    return _ChainedProvider(store, api)


class _ChainedProvider:
    """Lee el almacén. Si no hay filas, completa con la API."""

    def __init__(self, store, api) -> None:
        self.store = store
        self.api = api

    def get_bars(self, symbols, start, end, timeframe):
        bars = self.store.get_bars(symbols, start, end, timeframe)
        if bars.close.empty:
            return self.api.get_bars(symbols, start, end, timeframe)
        return bars


def _api_provider(config, clock: _WallClock):
    """``(provider, None)`` si hay claves; si no, ``(None, motivo)``."""
    try:
        from src.brokers.alpaca_adapter import load_broker_config
        from src.hft.data.alpaca_data import AlpacaDataClient, AlpacaProvider

        broker_config = load_broker_config()
    except Exception as exc:
        return None, str(exc)
    client = AlpacaDataClient(broker_config, feed=config.feed, clock=clock.now)
    return AlpacaProvider(client, feed=config.feed), None


def _momentum_closes(config, client, calendar, day: date):
    """Cierres SIP diarios del símbolo del momentum, de las sesiones previas a ``day``.

    El momentum mide r1 contra el cierre consolidado previo y la σ con
    ``sigma_sessions`` retornos diarios. Sin esto ``on_bar`` devuelve
    () en cada barra. El fin es la medianoche NY de ``day`` (exclusivo):
    no entra nada del día y no se pide SIP de los últimos 15 minutos.
    None, con WARNING, si falla la API o faltan sesiones.
    """
    import logging

    from src.hft.calendar import NY
    from src.hft.data.alpaca_data import AlpacaProvider
    from src.hft.strategies.intraday_momentum import ConsolidatedCloses

    logger = logging.getLogger("hft.run_intraday")
    params = config.intraday_momentum
    needed = int(params.sigma_sessions) + 1
    prior = [item.date for item in calendar.sessions if item.date < day]
    if len(prior) < needed:
        logger.warning("intraday_momentum: el calendario trae %d sesiones previas y hacen "
                       "falta %d; sin cierres no opera.", len(prior), needed)
        return None
    first = prior[-needed]
    # Un día antes: la barra diaria viene sellada a la medianoche NY y no
    # depende de que el corte ``>= start`` caiga justo en ese instante.
    start = datetime(first.year, first.month, first.day, tzinfo=NY) - timedelta(days=1)
    end = datetime(day.year, day.month, day.day, tzinfo=NY)
    try:
        bars = AlpacaProvider(client, feed="sip").get_bars([params.symbol], start, end, "1Day")
    except Exception as exc:
        logger.warning("intraday_momentum: cierres SIP diarios no disponibles (%s); no opera.", exc)
        return None
    if params.symbol not in bars.close.columns:
        series = None
    else:
        series = bars.close[params.symbol].dropna()
        series.index = [pd_timestamp.tz_convert(NY).date() for pd_timestamp in series.index]
    if series is None or len(series) < needed:
        logger.warning("intraday_momentum: %d cierres SIP de %s y hacen falta %d; no opera.",
                       0 if series is None else len(series), params.symbol, needed)
        return None
    return ConsolidatedCloses(series, feed="sip")


def _has_session(calendar, day: date) -> bool:
    from src.hft.calendar import NY

    noon = datetime(day.year, day.month, day.day, 16, 0, tzinfo=NY)
    try:
        session = calendar.session_for(noon)
    except Exception:
        return False
    return session is not None and session.date == day


def _session_date(text: Optional[str], clock: _WallClock) -> date:
    from src.hft.calendar import NY

    if text is None:
        return clock.now().astimezone(NY).date()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(f"fecha no reconocida: {text}") from exc


def _hhmm(text: str) -> time:
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--until es HH:MM en hora de Nueva York") from exc
    return time(parsed.hour, parsed.minute)


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    if flag in argv:
        return True
    prefix = flag + "="
    return any(item.startswith(prefix) for item in argv)


def _flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    prefix = flag + "="
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


if __name__ == "__main__":
    sys.exit(main())
