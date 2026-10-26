"""Línea de comandos del runner intradía, sin abrir una sesión con datos."""

from __future__ import annotations

import scripts.run_intraday as runner


def test_sabado_sale_1(tmp_path, monkeypatch):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    assert runner.main(["--mode", "signals", "--date", "2024-01-06"]) == 1


def test_allow_research_en_paper_sale_2(tmp_path, monkeypatch):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    assert runner.main(["--mode", "paper", "--allow-research", "--date", "2024-01-06"]) == 2


def test_submit_fuera_de_paper_sale_2(tmp_path, monkeypatch):
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    assert runner.main(["--mode", "sim", "--submit", "--date", "2024-01-03"]) == 2


def test_mode_paper_sin_banderas_de_sesion_sigue_en_la_guarda():
    assert runner.main(["--mode", "paper"]) == 2


def test_signals_sin_claves_sale_7_en_vez_de_latir(tmp_path, monkeypatch, capsys):
    def _sin_claves(*args, **kwargs):
        raise ValueError("Faltan BROKER_API_KEY / BROKER_SECRET_KEY.")

    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.brokers.alpaca_adapter.load_broker_config", _sin_claves)
    assert runner.main(["--mode", "signals", "--date", "2024-01-03"]) == 7
    assert "BROKER_API_KEY" in capsys.readouterr().err


def test_signals_carga_las_sesiones_previas_para_el_calentamiento(tmp_path, monkeypatch):
    """El vivo carga la misma ventana que el backtest; con solo el día, nada calienta."""
    from datetime import date

    from src.brokers.alpaca_adapter import BrokerConfig

    captured = {}

    def _capture(request):
        captured["request"] = request
        return 0

    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.brokers.alpaca_adapter.load_broker_config",
        lambda *args, **kwargs: BrokerConfig("PKTEST1234", "secreto-falso"),
    )
    monkeypatch.setattr(runner, "_momentum_closes", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.hft.session.run_session", _capture)
    assert runner.main(["--mode", "signals", "--date", "2024-03-01"]) == 0
    calendar = captured["request"].calendar
    prior = [item for item in calendar.sessions if item.date < date(2024, 3, 1)]
    assert len(prior) >= 30
    assert calendar.sessions[-1].date == date(2024, 3, 1)


def _broker_config(account_id: str = ""):
    from src.brokers.alpaca_adapter import BrokerConfig

    return BrokerConfig("PKTEST1234", "secreto-falso", account_id=account_id)


def test_el_maestro_hft_rutea_todos_los_simbolos_de_la_sesion():
    """Sin esto el paper cargaba universe_mixed.csv: SPY..TLT salían not_routable."""
    from src.hft.config import load_hft_config
    from src.hft.session import config_symbols

    master = runner._hft_instruments()
    missing = [
        symbol for symbol in config_symbols(load_hft_config())
        if symbol not in master or not master[symbol].routable("alpaca")
    ]
    assert missing == []




def test_paper_sin_ruta_para_un_simbolo_sale_3(tmp_path, monkeypatch, capsys):
    from src.hft.config import load_hft_config
    from src.instruments import InstrumentMaster

    full = runner._hft_instruments()
    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "src.brokers.alpaca_adapter.load_broker_config",
        lambda *args, **kwargs: _broker_config(),
    )
    monkeypatch.setattr(
        runner, "_hft_instruments",
        lambda: InstrumentMaster(item for key, item in full.items() if key != "TLT"),
    )
    assert runner._paper_broker(load_hft_config(), runner._WallClock()) == 3
    assert "TLT" in capsys.readouterr().err


class _DailyClient:
    """Cliente de datos falso: barras diarias de SPY, una por sesión."""

    feed = "iex"

    def __init__(self, days):
        self.days = list(days)
        self.calls = []

    def get_bars(self, symbols, start, end, timeframe, *, feed=None, adjustment="raw"):
        self.calls.append({"symbols": list(symbols), "start": start, "end": end,
                           "timeframe": timeframe, "feed": feed})
        from datetime import datetime, timezone

        from src.hft.calendar import NY

        def _midnight_ny(day):
            stamp = datetime(day.year, day.month, day.day, tzinfo=NY).astimezone(timezone.utc)
            return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = [{"t": _midnight_ny(day), "o": 1.0, "h": 1.0, "l": 1.0,
                 "c": 500.0 + index, "v": 1_000_000, "vw": 1.0, "n": 1}
                for index, day in enumerate(self.days)]
        return {"SPY": rows}


def test_momentum_recibe_los_cierres_sip_diarios_previos():
    """Sin cierres el momentum devolvía () en cada barra: nunca operaba."""
    from datetime import date, datetime

    from src.hft.calendar import NY, SessionCalendar
    from src.hft.config import load_hft_config

    config = load_hft_config()
    day = date(2024, 3, 1)
    calendar = SessionCalendar(client=None, use_cache=False)
    calendar.load(date(2023, 12, 1), day)
    prior = [item.date for item in calendar.sessions if item.date < day]
    needed = config.intraday_momentum.sigma_sessions + 1
    client = _DailyClient(prior[-needed:])
    closes = runner._momentum_closes(config, client, calendar, day)
    assert closes is not None
    assert closes.feed == "sip"
    assert list(closes.close.index) == prior[-needed:]
    call = client.calls[0]
    assert call["symbols"] == [config.intraday_momentum.symbol]
    assert call["timeframe"] == "1Day"
    assert call["feed"] == "sip"
    # Fin exclusivo a medianoche NY del día: nada del día en curso y la
    # ventana SIP de 15 minutos no se toca aunque se corra antes de abrir.
    assert call["end"] == datetime(2024, 3, 1, tzinfo=NY)


def test_momentum_sin_cierres_suficientes_queda_apagado_y_avisa(caplog):
    from datetime import date

    from src.hft.calendar import SessionCalendar
    from src.hft.config import load_hft_config

    config = load_hft_config()
    day = date(2024, 3, 1)
    calendar = SessionCalendar(client=None, use_cache=False)
    calendar.load(date(2023, 12, 1), day)
    client = _DailyClient([date(2024, 2, 29)])
    with caplog.at_level("WARNING"):
        assert runner._momentum_closes(config, client, calendar, day) is None
    assert "intraday_momentum" in caplog.text


def test_la_sesion_pasa_los_cierres_al_momentum(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def _capture(request):
        captured["request"] = request
        return 0

    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.brokers.alpaca_adapter.load_broker_config",
        lambda *args, **kwargs: _broker_config(),
    )
    monkeypatch.setattr(runner, "_momentum_closes", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr("src.hft.session.run_session", _capture)
    assert runner.main(["--mode", "signals", "--date", "2024-03-01"]) == 0
    assert captured["request"].momentum_closes is sentinel


def test_default_strategies_entrega_los_cierres_y_avisa_sin_ellos(caplog):
    from datetime import date

    import pandas as pd

    from src.hft.config import load_hft_config
    from src.hft.session import _default_strategies
    from src.hft.strategies.intraday_momentum import ConsolidatedCloses, IntradayMomentum

    config = load_hft_config()
    closes = ConsolidatedCloses(pd.Series([1.0, 2.0], index=[date(2024, 2, 28), date(2024, 2, 29)]))
    built = _default_strategies(config, (), include_research=False, momentum_closes=closes)
    momentum = [item for item in built if isinstance(item, IntradayMomentum)]
    assert momentum[0].closes is closes
    with caplog.at_level("WARNING"):
        _default_strategies(config, (), include_research=False)
    assert "intraday_momentum" in caplog.text


# Set de la sesión paper antes de --with (Fase 5 del plan horario).
_SET_DE_HOY = ("orb", "intraday_momentum", "vwap_reversion", "pairs_ou")


def test_sin_with_el_set_de_la_sesion_es_el_de_hoy():
    from src.hft.config import load_hft_config
    from src.hft.session import _default_strategies

    built = _default_strategies(load_hft_config(), (), include_research=False)
    assert tuple(strategy.id for strategy in built) == _SET_DE_HOY


def test_with_hourly_momentum_agrega_solo_esa_al_final():
    from datetime import date

    from src.hft.calendar import SessionCalendar
    from src.hft.config import load_hft_config
    from src.hft.session import _default_strategies

    config = load_hft_config()
    calendar = SessionCalendar(client=None, use_cache=False)
    calendar.load(date(2024, 1, 2), date(2024, 3, 1))
    sessions = calendar.sessions
    base = _default_strategies(config, sessions, include_research=False)
    built = _default_strategies(
        config, sessions, include_research=False, with_strategies=("hourly_momentum",),
    )
    assert tuple(strategy.id for strategy in built) == (*_SET_DE_HOY, "hourly_momentum")
    for before, after in zip(base, built):
        assert type(after) is type(before)
        assert after.params == before.params
    hourly = built[-1]
    assert hourly.sessions == tuple(sessions)
    assert hourly.universe == tuple(config.universe)


def _capture_session(tmp_path, monkeypatch) -> dict:
    """``run_session`` falso: guarda el ``SessionRequest`` y sale 0."""
    captured = {}

    def _capture(request):
        captured["request"] = request
        return 0

    monkeypatch.setenv("HFT_HOME", str(tmp_path))
    monkeypatch.delenv("HFT_KILL", raising=False)
    monkeypatch.setattr("src.settings.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.brokers.alpaca_adapter.load_broker_config",
        lambda *args, **kwargs: _broker_config(),
    )
    monkeypatch.setattr(runner, "_momentum_closes", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.hft.session.run_session", _capture)
    return captured


def test_sin_with_la_sesion_no_pide_extras(tmp_path, monkeypatch):
    captured = _capture_session(tmp_path, monkeypatch)
    assert runner.main(["--mode", "signals", "--date", "2024-03-01"]) == 0
    assert captured["request"].with_strategies == ()


def test_with_hourly_momentum_llega_a_la_sesion(tmp_path, monkeypatch):
    captured = _capture_session(tmp_path, monkeypatch)
    argv = ["--mode", "signals", "--with", "hourly_momentum", "--date", "2024-03-01"]
    assert runner.main(argv) == 0
    assert captured["request"].with_strategies == ("hourly_momentum",)


def test_with_otra_estrategia_es_error_de_argumentos(tmp_path, monkeypatch, capsys):
    import pytest

    _capture_session(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--mode", "sim", "--with", "avellaneda_stoikov", "--date", "2024-03-01"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_paper_con_with_sigue_exigiendo_submit(tmp_path, monkeypatch):
    """hourly_momentum no es research_only, pero sin --submit paper no envía."""
    captured = _capture_session(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_paper_broker", lambda *args, **kwargs: (object(), "EXEC", "PA3TEST"))
    argv = ["--mode", "paper", "--with", "hourly_momentum", "--date", "2024-03-01"]
    assert runner.main(argv) == 0
    request = captured["request"]
    assert request.with_strategies == ("hourly_momentum",)
    assert request.submit is False
    assert request.execution is None
    assert runner.main([*argv, "--submit"]) == 0
    assert captured["request"].submit is True
    assert captured["request"].execution == "EXEC"


def test_paper_con_with_es_sesion_y_no_la_guarda():
    assert not runner._is_guard_invocation(["--mode", "paper", "--with", "hourly_momentum"])


def test_la_guarda_deja_hourly_momentum_en_paper(monkeypatch):
    from src.hft.strategies import STRATEGIES

    # Proceso nuevo: importar el paquete no registra hourly_momentum.
    monkeypatch.delitem(STRATEGIES, "hourly_momentum", raising=False)
    assert runner.main(["--mode", "paper", "--strategy", "hourly_momentum"]) == 0
    assert runner.main(["--mode", "paper", "--strategy", "avellaneda_stoikov"]) == 2


def test_with_no_admite_otra_estrategia():
    """A–S no entra por --with: la puerta de research sigue siendo --allow-research."""
    import pytest

    from src.hft.config import load_hft_config
    from src.hft.session import _default_strategies

    with pytest.raises(ValueError, match="avellaneda_stoikov"):
        _default_strategies(
            load_hft_config(), (), include_research=False,
            with_strategies=("avellaneda_stoikov",),
        )
