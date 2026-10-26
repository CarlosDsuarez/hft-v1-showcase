"""Datos intradía (HFT-04): cliente Alpaca falso, almacén, calidad y sintético.

Sin red: urllib se sustituye o ni siquiera se construye.
"""

from __future__ import annotations

import http.client
import io
import json
import math
import urllib.error
from datetime import date, datetime, time, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.hft.calendar import expected_bars, session_at
from src.hft.data.alpaca_data import (
    MAX_RATE_PER_MIN,
    AlpacaDataClient,
    AlpacaProvider,
    DataAuthError,
    DataError,
    SipWindowError,
    TokenBucket,
    estimate_year_1min,
)
from src.hft.data.bar_store import BarStore, subtract_coverage
from src.hft.data.bars import DEFAULT_FEED
from src.hft.data.download import build_parser, download_bars, hft_symbols, main, sync_adjusted_daily
from src.hft.data.quality import (
    DUPLICATE,
    INVALID_OHLC,
    JUMP,
    MISSING_DOWNLOAD,
    SIN_TRADES,
    QualityError,
    QualityPolicy,
    quality_report,
)
from src.hft.data.synthetic import FakeIntradayProvider, synthetic_bars
from src.instruments import load_instruments
from src.settings import load_settings

UTC = timezone.utc
NOW = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
SESSION = session_at(date(2024, 1, 3), time(9, 30), time(16, 0))
SECRET = "secreto-falso"
KEY = "PKTEST1234"


def _bar(ts: str, price: float = 100.0, volume: float = 10.0) -> dict:
    return {"t": ts, "o": price, "h": price, "l": price, "c": price, "v": volume, "n": 1, "vw": price}


class _Resp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _CutResp(_Resp):
    """Body cortado: read() lanza ``error`` o devuelve bytes que no son JSON."""

    def __init__(self, raw=b"", error=None):
        self._body = raw
        self._error = error

    def read(self):
        if self._error is not None:
            raise self._error
        return self._body


class _Opener:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        if not self.script:
            raise AssertionError(f"HTTP inesperado: {request.full_url}")
        outcome = self.script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, _Resp):
            return outcome
        return _Resp(outcome)


class _Mono:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _http_error(url, code, body=b"{}", headers=None):
    return urllib.error.HTTPError(url, code, "err", headers if headers is not None else {}, io.BytesIO(body))


def _client(opener, *, feed="iex", clock=None, **kwargs):
    from src.brokers.alpaca_adapter import BrokerConfig

    return AlpacaDataClient(
        BrokerConfig(KEY, SECRET),
        feed=feed,
        clock=clock or (lambda: NOW),
        monotonic=_Mono().monotonic,
        sleep=lambda _seconds: None,
        opener=opener,
        **kwargs,
    )


def _query(request):
    parsed = urlparse(request.full_url)
    return parsed.path, parse_qs(parsed.query)


class TestAlpacaDataClient:
    def test_paginacion_sigue_next_page_token(self):
        opener = _Opener([
            {"bars": {"SPY": [_bar("2024-01-03T14:30:00Z", 10)]}, "next_page_token": "abc="},
            {"bars": {"SPY": [_bar("2024-01-03T14:31:00Z", 11)]}, "next_page_token": None},
        ])
        client = _client(opener)
        raw = client.get_bars(
            ["SPY"],
            datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
            "1Min",
        )
        assert [bar["c"] for bar in raw["SPY"]] == [10, 11]
        assert len(opener.calls) == 2
        _, second = _query(opener.calls[1])
        assert second["page_token"] == ["abc="]
        path, first = _query(opener.calls[0])
        assert path == "/v2/stocks/bars"
        assert "page_token" not in first
        assert first["feed"] == ["iex"]
        assert first["adjustment"] == ["raw"]
        assert first["timeframe"] == ["1Min"]
        headers = {name.lower(): value for name, value in opener.calls[0].header_items()}
        assert headers["apca-api-key-id"] == KEY
        assert headers["apca-api-secret-key"] == SECRET

    @pytest.mark.parametrize("cut", [
        # Paper 2026-10-06 20:11Z: get_bars cayó con "IncompleteRead(58546 bytes read)".
        lambda: _CutResp(error=http.client.IncompleteRead(b'{"bars": {', 58546)),
        lambda: _CutResp(raw=b'{"bars": {"SPY": [{"t": "2024-01-03T14:3'),
    ], ids=["incomplete_read", "truncated_json"])
    def test_body_cortado_se_reintenta(self, cut):
        opener = _Opener([
            cut(),
            {"bars": {"SPY": [_bar("2024-01-03T14:30:00Z")]}, "next_page_token": None},
        ])
        client = _client(opener)
        raw = client.get_bars(
            ["SPY"],
            datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
            "1Min",
        )
        assert len(raw["SPY"]) == 1
        assert client.request_count == 2

    def test_429_respeta_retry_after(self):
        headers = EmailMessage()
        headers["Retry-After"] = "3"
        mono = _Mono()
        opener = _Opener([
            _http_error("https://data.alpaca.markets/v2/stocks/bars", 429, headers=headers),
            {"bars": {"SPY": [_bar("2024-01-03T14:30:00Z")]}, "next_page_token": None},
        ])
        from src.brokers.alpaca_adapter import BrokerConfig

        client = AlpacaDataClient(
            BrokerConfig(KEY, SECRET),
            clock=lambda: NOW,
            monotonic=mono.monotonic,
            sleep=mono.sleep,
            opener=opener,
        )
        raw = client.get_bars(
            ["SPY"],
            datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
            "1Min",
        )
        assert len(raw["SPY"]) == 1
        assert mono.sleeps == [3]
        assert client.request_count == 2

    def test_error_no_lleva_el_secreto(self):
        opener = _Opener([
            _http_error(
                "https://data.alpaca.markets/v2/stocks/bars",
                400,
                body=f'{{"message":"no {SECRET}"}}'.encode(),
            ),
        ])
        client = _client(opener)
        with pytest.raises(DataError) as caught:
            client.get_bars(
                ["SPY"],
                datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
                datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
                "1Min",
            )
        assert SECRET not in str(caught.value)
        assert "***" in str(caught.value)
        assert SECRET not in repr(client)

    @pytest.mark.parametrize("code", [401, 403])
    def test_credenciales_rechazadas_dan_error_claro(self, code):
        nginx = (
            b"<html>\r\n<head><title>401 Authorization Required</title></head>"
            b"<body><center><h1>401 Authorization Required</h1></center></body></html>"
        )
        opener = _Opener([
            _http_error("https://data.alpaca.markets/v2/stocks/bars", code, body=nginx),
        ])
        client = _client(opener)
        with pytest.raises(DataAuthError) as caught:
            client.get_bars(
                ["SPY"],
                datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
                datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
                "1Min",
            )
        text = str(caught.value)
        assert caught.value.status == code
        assert "BROKER_API_KEY" in text
        assert "<html>" not in text
        assert SECRET not in text
        assert len(opener.calls) == 1

    def test_sip_reciente_no_llama(self):
        opener = _Opener([])
        client = _client(opener, feed="sip")
        with pytest.raises(SipWindowError):
            client.get_bars(
                ["SPY"],
                NOW - timedelta(hours=1),
                NOW,
                "1Min",
                feed="sip",
            )
        assert opener.calls == []
        with pytest.raises(SipWindowError):
            client.get_latest_bars(["SPY"], feed="sip")
        with pytest.raises(SipWindowError):
            client.get_latest_quotes(["SPY"], feed="sip")
        with pytest.raises(SipWindowError):
            client.get_snapshots(["SPY"], feed="sip")
        assert opener.calls == []

    def test_sip_con_quince_minutos_si_llama(self):
        opener = _Opener([
            {"bars": {"SPY": []}, "next_page_token": None},
        ])
        client = _client(opener, feed="sip")
        client.get_bars(
            ["SPY"],
            NOW - timedelta(hours=2),
            NOW - timedelta(minutes=15),
            "1Min",
            feed="sip",
        )
        assert len(opener.calls) == 1

    def test_iex_hasta_ahora_si_llama(self):
        opener = _Opener([{"bars": {}, "next_page_token": None}])
        client = _client(opener, feed="iex")
        client.get_bars(["SPY"], NOW - timedelta(minutes=5), NOW, "1Min")
        assert len(opener.calls) == 1

    def test_proveedor_devuelve_frames_raw(self):
        opener = _Opener([
            {
                "bars": {"QQQ": [_bar("2024-01-03T14:30:00Z", 20)]},
                "next_page_token": None,
            },
        ])
        provider = AlpacaProvider(_client(opener))
        assert provider.feed == DEFAULT_FEED
        bars = provider.get_bars(
            ["QQQ"],
            datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
            datetime(2024, 1, 3, 14, 31, tzinfo=UTC),
            "1Min",
        )
        assert bars.meta.adjustment == "raw"
        assert bars.meta.feed == "iex"
        assert bars.close.loc[pd.Timestamp("2024-01-03T14:30:00Z"), "QQQ"] == 20

    def test_acciones_corporativas_paginan(self):
        opener = _Opener([
            {
                "corporate_actions": {"forward_splits": [{"symbol": "SPY", "ex_date": "2024-01-02"}]},
                "next_page_token": "p2",
            },
            {
                "corporate_actions": {"reverse_splits": [{"symbol": "QQQ", "ex_date": "2024-01-03"}]},
                "next_page_token": None,
            },
        ])
        client = _client(opener)
        actions = client.get_corporate_actions(
            ["SPY", "QQQ"],
            date(2024, 1, 1),
            date(2024, 1, 31),
            types=("forward_split", "reverse_split"),
        )
        assert [row["type"] for row in actions] == ["forward_split", "reverse_split"]
        assert len(opener.calls) == 2

    def test_un_ano_de_1min_cabe_en_el_rate_limit(self):
        pages = estimate_year_1min(10, 252)
        assert pages == 100
        assert pages <= MAX_RATE_PER_MIN

    def test_token_bucket_no_pasa_de_200(self):
        mono = _Mono()
        bucket = TokenBucket(200, monotonic=mono.monotonic, sleep=mono.sleep)
        for _ in range(200):
            bucket.acquire()
        assert mono.sleeps == []
        bucket.acquire()
        assert mono.sleeps
        assert mono.sleeps[0] == pytest.approx(60 / 200)
        with pytest.raises(ValueError):
            TokenBucket(201)


class TestBarStore:
    def _records(self, prices=(100.0, 101.0)):
        rows = []
        for i, price in enumerate(prices):
            rows.append({
                "symbol": "SPY",
                "ts": datetime(2024, 1, 3, 14, 30 + i, tzinfo=UTC),
                "o": price, "h": price + 0.5, "l": price - 0.5, "c": price,
                "v": 1000 + i, "vw": price, "n": 3,
            })
        return pd.DataFrame(rows)

    def test_upsert_idempotente_y_la_ultima_version_gana(self, tmp_path):
        store = BarStore(tmp_path)
        start = datetime(2024, 1, 3, 14, 0, tzinfo=UTC)
        end = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
        fetched = datetime(2024, 6, 1, tzinfo=UTC)
        records = self._records()
        store.upsert(records, timeframe="1Min", feed="iex", fetched_start=start,
                     fetched_end=end, fetched_at=fetched, symbols=["SPY"])
        path = tmp_path / "1Min" / "iex" / "SPY" / "2024-01.csv.gz"
        first = path.read_bytes()
        entry = store.file_entry("1Min/iex/SPY/2024-01.csv.gz")
        assert entry is not None
        assert entry.rows == 2
        assert entry.sha256
        store.upsert(records, timeframe="1Min", feed="iex", fetched_start=start,
                     fetched_end=end, fetched_at=fetched, symbols=["SPY"])
        assert path.read_bytes() == first
        changed = records.copy()
        changed.loc[0, "c"] = 77.0
        store.upsert(changed, timeframe="1Min", feed="iex", fetched_start=start,
                     fetched_end=end, fetched_at=fetched, symbols=["SPY"])
        bars = store.read(["SPY"], start, end, timeframe="1Min", feed="iex")
        assert len(bars.close) == 2
        assert bars.close.iloc[0]["SPY"] == pytest.approx(77.0)
        assert not list(tmp_path.rglob("*.tmp"))

    def test_roundtrip_gz(self, tmp_path):
        store = BarStore(tmp_path)
        start = datetime(2024, 1, 3, 14, 0, tzinfo=UTC)
        end = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
        store.upsert(
            self._records(),
            timeframe="1T",
            feed="iex",
            fetched_start=start,
            fetched_end=end,
            fetched_at=datetime(2024, 6, 1, tzinfo=UTC),
            symbols=["spy"],
        )
        bars = store.read(["SPY"], start, end, timeframe="1Min", feed="iex")
        assert bars.meta.timeframe == "1Min"
        assert list(bars.close.columns) == ["SPY"]
        assert bars.volume.iloc[0]["SPY"] == pytest.approx(1000)
        assert bars.vwap.iloc[1]["SPY"] == pytest.approx(101.0)
        assert bars.close.index.tz is not None
        assert str(bars.close.index.tz) == "UTC"

    def test_cobertura_resta_huecos(self):
        start = datetime(2024, 1, 3, 14, 0, tzinfo=UTC)
        mid = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
        end = datetime(2024, 1, 3, 18, 0, tzinfo=UTC)
        assert subtract_coverage(start, end, [(start, mid)]) == ((mid, end),)


class TestQualityAndSynthetic:
    def test_sintetico_respeta_calendario_y_semilla(self):
        bars = synthetic_bars([SESSION], ["SPY"], seed=7, gap_prob=0.0)
        again = synthetic_bars([SESSION], ["SPY"], seed=7, gap_prob=0.0)
        expected = expected_bars(SESSION, "1Min")
        assert len(bars.close) == len(expected) == 390
        assert bars.close.index.equals(expected)
        pd.testing.assert_frame_equal(bars.close, again.close)
        other = synthetic_bars([SESSION], ["SPY"], seed=8, gap_prob=0.0)
        assert not bars.close.equals(other.close)

    def test_volumen_en_u_y_ou_con_edge_conocido(self):
        bars = synthetic_bars([SESSION], ["SPY"], seed=1, gap_prob=0.0, sigma=0.0, mu=0.0)
        volume = bars.volume["SPY"]
        edges = pd.concat([volume.iloc[:30], volume.iloc[-30:]]).mean()
        middle = volume.iloc[180:210].mean()
        assert edges > middle
        pulled = synthetic_bars(
            [SESSION],
            ["SPY"],
            seed=1,
            gap_prob=0.0,
            sigma=0.0,
            mu=0.0,
            ou_kappa=0.5,
            ou_theta=0.0,
            x0=1.0,
        )
        log_close = (pulled.close["SPY"] / 100.0).map(math.log)
        assert abs(float(log_close.iloc[-1])) < abs(float(log_close.iloc[0]))

    def test_calidad_separa_huecos_iex_de_descarga(self):
        bars = synthetic_bars([SESSION], ["SPY"], seed=3, gap_prob=0.15)
        present = bars.close["SPY"].dropna().index
        assert len(present) < 390
        covered = {"SPY": [(SESSION.open_utc, SESSION.close_utc)]}
        report = quality_report(
            bars,
            [SESSION],
            coverage=covered,
            policy=QualityPolicy(on_invalid="warn", on_jump="warn"),
        )
        assert report.of_kind(SIN_TRADES)
        assert report.of_kind(MISSING_DOWNLOAD) == ()
        missing = quality_report(
            bars,
            [SESSION],
            coverage={"SPY": []},
            policy=QualityPolicy(on_invalid="warn", on_jump="warn"),
        )
        assert missing.of_kind(MISSING_DOWNLOAD)
        assert missing.of_kind(SIN_TRADES) == ()

    def test_ohlc_invalido_salta_y_no_borra(self):
        bars = synthetic_bars([SESSION], ["SPY"], seed=1, gap_prob=0.0, sigma=0.0, mu=0.0)
        bars.high.iloc[0, 0] = 0.01
        with pytest.raises(QualityError) as caught:
            quality_report(bars, [SESSION], coverage={"SPY": [(SESSION.open_utc, SESSION.close_utc)]})
        assert caught.value.report.of_kind(INVALID_OHLC)
        assert bars.high.iloc[0, 0] == 0.01
        series = [100.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(25)]
        series.append(180.0)
        index = expected_bars(SESSION, "1Min")[: len(series)]
        frame = synthetic_bars([SESSION], ["SPY"], seed=1, gap_prob=0.0)
        frame.close.iloc[: len(series), 0] = series
        frame.open.iloc[: len(series), 0] = series
        frame.high.iloc[: len(series), 0] = [value + 0.01 for value in series]
        frame.low.iloc[: len(series), 0] = [value - 0.01 for value in series]
        quiet = QualityPolicy(on_invalid="warn", on_jump="warn", on_gap="warn")
        trimmed = _trim(frame, index)
        report = quality_report(
            trimmed,
            [SESSION],
            coverage={"SPY": [(SESSION.open_utc, SESSION.close_utc)]},
            policy=quiet,
        )
        assert report.of_kind(JUMP)
        assert trimmed.close.iloc[-1, 0] == 180.0

    def test_duplicados_se_marcan(self):
        bars = synthetic_bars([SESSION], ["SPY"], seed=1, gap_prob=0.0, sigma=0.0)
        ts = bars.close.index[0]
        records = pd.DataFrame([
            {"symbol": "SPY", "ts": ts, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "vw": 1, "n": 1},
            {"symbol": "SPY", "ts": ts, "o": 2, "h": 2, "l": 2, "c": 2, "v": 2, "vw": 2, "n": 2},
        ])
        report = quality_report(
            bars,
            [SESSION],
            records=records,
            coverage={"SPY": [(SESSION.open_utc, SESSION.close_utc)]},
            policy=QualityPolicy(on_invalid="warn", on_jump="warn", on_duplicate="warn"),
        )
        assert report.of_kind(DUPLICATE)
        assert len(bars.close) == 390


def _trim(bars, index):
    frames = {}
    for name in ("open", "high", "low", "close", "volume", "vwap", "trades"):
        frames[name] = getattr(bars, name).loc[index]
    meta = type(bars.meta)(
        source=bars.meta.source,
        feed=bars.meta.feed,
        timeframe=bars.meta.timeframe,
        adjustment=bars.meta.adjustment,
        requested=bars.meta.requested,
        received=bars.meta.received,
    )
    return type(bars)(meta=meta, **frames)


class TestDownloadAndUniverse:
    def test_reanuda_y_agrupa(self, tmp_path):
        store = BarStore(tmp_path)
        start = datetime(2024, 1, 3, 14, 0, tzinfo=UTC)
        mid = datetime(2024, 1, 3, 16, 0, tzinfo=UTC)
        end = datetime(2024, 1, 3, 18, 0, tzinfo=UTC)
        fetched = datetime(2024, 6, 1, tzinfo=UTC)
        store.upsert(
            pd.DataFrame(columns=["symbol", "ts", "o", "h", "l", "c", "v", "vw", "n"]),
            timeframe="1Min",
            feed="iex",
            fetched_start=start,
            fetched_end=mid,
            fetched_at=fetched,
            symbols=["SPY"],
        )

        class Fake:
            def __init__(self):
                self.request_count = 0
                self.calls = []

            def get_bars(self, symbols, start, end, timeframe, *, feed, adjustment):
                self.request_count += 1
                self.calls.append((tuple(symbols), start, end, feed, adjustment, timeframe))
                return {symbol: [] for symbol in symbols}

        client = Fake()
        messages = []
        result = download_bars(
            client, store, ["SPY", "QQQ"], start, end,
            timeframe="1Min", feed="iex", fetched_at=fetched, progress=messages.append,
        )
        assert result.calls == 2
        spy = [call for call in client.calls if call[0] == ("SPY",)]
        qqq = [call for call in client.calls if "QQQ" in call[0] and "SPY" not in call[0]]
        assert spy and spy[0][1:] == (mid, end, "iex", "raw", "1Min")
        assert qqq and qqq[0][1] == start and qqq[0][2] == end
        assert messages
        # Segunda pasada: ya está cubierto, cero llamadas.
        again = Fake()
        download_bars(again, store, ["SPY", "QQQ"], start, end, fetched_at=fetched, progress=lambda _line: None)
        assert again.calls == []

    def test_ajustado_se_refresca_si_cambia_la_huella(self, tmp_path):
        store = BarStore(tmp_path)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)
        bar = _bar("2024-01-03T05:00:00Z", 50)

        class Fake:
            def __init__(self, actions):
                self.actions = actions
                self.bars = 0

            def get_corporate_actions(self, symbols, start, end, *, types):
                return list(self.actions)

            def get_bars(self, symbols, start, end, timeframe, *, feed, adjustment):
                self.bars += 1
                assert adjustment == "split"
                assert timeframe == "1Day"
                return {symbols[0]: [bar]}

        client = Fake([{"symbol": "SPY", "type": "forward_split", "ex_date": "2024-01-10"}])
        fetched = datetime(2024, 6, 1, tzinfo=UTC)
        first = sync_adjusted_daily(client, store, ["SPY"], start, end, adjustment="split", fetched_at=fetched)
        assert first == ("SPY",)
        assert client.bars == 1
        second = sync_adjusted_daily(client, store, ["SPY"], start, end, adjustment="split", fetched_at=fetched)
        assert second == ()
        assert client.bars == 1
        client.actions.append({"symbol": "SPY", "type": "forward_split", "ex_date": "2024-01-20"})
        third = sync_adjusted_daily(client, store, ["SPY"], start, end, adjustment="split", fetched_at=fetched)
        assert third == ("SPY",)
        assert client.bars == 2
        loaded = store.read_adjusted(["SPY"], start, end, timeframe="1Day", adjustment="split")
        assert loaded.close.iloc[0]["SPY"] == pytest.approx(50)

    def test_parser_default_iex_y_universo(self):
        args = build_parser().parse_args(["--start", "2024-01-02", "--end", "2024-01-31"])
        assert args.feed == "iex"
        assert args.timeframe == "1Min"
        symbols = hft_symbols()
        assert symbols == ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "GLD", "TLT")
        master = load_instruments(load_settings().universe_path("universe_hft.csv"))
        assert len(master) == 10
        for instrument in master.values():
            assert instrument.routable("alpaca")
            assert instrument.currency == "USD"
            assert instrument.broker_symbol

    def test_main_no_toca_la_red(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HFT_HOME", str(tmp_path))

        class Fake:
            def __init__(self):
                self.request_count = 0

            def get_bars(self, symbols, start, end, timeframe, *, feed, adjustment):
                self.request_count += 1
                return {symbol: [] for symbol in symbols}

        class Clock:
            def now(self):
                return datetime(2024, 6, 1, tzinfo=UTC)

        store = BarStore(tmp_path / "bars")
        code = main(
            ["--symbols", "SPY", "--start", "2024-01-03", "--end", "2024-01-04", "--feed", "iex"],
            client=Fake(),
            store=store,
            clock=Clock(),
        )
        assert code == 0

    def test_rutas_siguen_hft_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HFT_HOME", str(tmp_path / "home"))
        settings = load_settings()
        assert settings.hft_dir == tmp_path / "home" / "data" / "hft"
        assert settings.bars_dir == tmp_path / "home" / "data" / "bars"
        assert settings.bars_dir.is_absolute()
        assert BarStore().root == settings.bars_dir

    def test_fake_provider_recorta(self):
        bars = synthetic_bars([SESSION], ["SPY", "QQQ"], seed=2, gap_prob=0.0)
        provider = FakeIntradayProvider(bars)
        start = SESSION.open_utc + timedelta(minutes=10)
        end = start + timedelta(minutes=5)
        sliced = provider.get_bars(["QQQ"], start, end, "1Min")
        assert list(sliced.close.columns) == ["QQQ"]
        assert len(sliced.close) == 5
        assert provider.calls
