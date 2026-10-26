"""Features intradía: truncamiento, fórmulas cerradas y reinicio de sesión."""

from __future__ import annotations

import ast
import math
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from src.hft.calendar import Session, session_at
from src.hft.features import (
    FEATURES,
    amihud,
    amihud_session,
    corwin_schultz_spread,
    garman_klass_variance,
    log_return,
    lookback_log_return,
    microprice,
    opening_range,
    order_flow_imbalance,
    parkinson_variance,
    prior_session_atr,
    prior_sessions_vol,
    quoted_spread_bps,
    realized_vol,
    relative_volume,
    required_history,
    roll_spread,
    session_vwap,
    trend_score,
    vwap_zscore,
    wilder_atr,
)
from src.regime_detector import _trend_score
from src.settings import SEED

UTC = "UTC"
NY = "America/New_York"


def _minutes(day: str, n: int, hour: int = 9, minute: int = 30) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{day} {hour:02d}:{minute:02d}", tz=NY)
    return pd.date_range(start, periods=n, freq="1min").tz_convert(UTC)


def _sessions(*days: str) -> tuple[Session, ...]:
    return tuple(session_at(date.fromisoformat(day), time(9, 30), time(16, 0)) for day in days)


def _frame(values, index, columns) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=columns)


def _panel(days: tuple[str, ...], bars: int, seed: int, columns=("SPY", "QQQ")):
    chunks = [_minutes(day, bars) for day in days]
    index = chunks[0].append(chunks[1:]) if len(chunks) > 1 else chunks[0]
    rng = np.random.default_rng(seed)
    n, k = len(index), len(columns)
    rets = rng.normal(0.0, 0.001, size=(n, k))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = np.vstack([close[0], close[:-1]])
    wick = np.abs(rng.normal(0.0, 0.0004, size=(n, k))) + 1.0e-4
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    volume = rng.uniform(1.0e3, 2.0e3, size=(n, k))
    return (
        _frame(open_, index, columns),
        _frame(high, index, columns),
        _frame(low, index, columns),
        _frame(close, index, columns),
        _frame(volume, index, columns),
    )


def _slice(frame: pd.DataFrame, t) -> pd.DataFrame:
    return frame.loc[:t]


def _compute(open_, high, low, close, volume, sessions=None):
    vwap = session_vwap(high, low, close, volume, sessions=sessions)
    bid = close * 0.999
    ask = close * 1.001
    bid_size = volume
    ask_size = volume * 0.5
    return {
        "log_return": log_return(close),
        "realized_vol": realized_vol(close, n=5, sessions=sessions),
        "parkinson_variance": parkinson_variance(high, low),
        "garman_klass_variance": garman_klass_variance(open_, high, low, close),
        "session_vwap": vwap,
        "vwap_zscore": vwap_zscore(close, vwap, n=5, sessions=sessions),
        "opening_range": opening_range(high, low, k_minutes=5, sessions=sessions),
        "relative_volume": relative_volume(
            volume, k_minutes=5, sessions_back=2, sessions=sessions,
        ),
        "trend_score": trend_score(close, ma_window=5, look=4, sessions=sessions),
        "roll_spread": roll_spread(close, n=8),
        "corwin_schultz_spread": corwin_schultz_spread(high, low),
        "amihud": amihud(close, volume, sessions=sessions),
        "amihud_session": amihud_session(close, volume, sessions=sessions),
        "microprice": microprice(bid, ask, bid_size, ask_size),
        "quoted_spread_bps": quoted_spread_bps(bid, ask),
        "order_flow_imbalance": order_flow_imbalance(bid, ask, bid_size, ask_size),
        "lookback_log_return": lookback_log_return(open_, close, minutes=5, sessions=sessions),
        "prior_sessions_vol": prior_sessions_vol(close, n_sessions=2, sessions=sessions),
    }


def _daily():
    index = pd.bdate_range("2024-01-02", periods=8)
    high = pd.DataFrame(
        {
            "SPY": [10, 11, 12, 13, 14, 15, 16, 40],
            "QQQ": [20, 21, 22, 23, 24, 25, 26, 27],
        },
        index=index,
    )
    low = high - 2.0
    close = high - 1.0
    return high, low, close


def _at(result, t):
    if isinstance(result, tuple):
        return tuple(part.loc[t] for part in result)
    return result.loc[t]


def _assert_row(full, part):
    if isinstance(full, tuple):
        for left, right in zip(full, part):
            pd.testing.assert_series_equal(left, right, rtol=1e-12, atol=1e-12)
        return
    pd.testing.assert_series_equal(full, part, rtol=1e-12, atol=1e-12)


class TestTruncation:
    def test_prefijo_hasta_t_coincide_con_la_serie_completa(self):
        days = ("2024-01-02", "2024-01-03", "2024-01-04")
        open_, high, low, close, volume = _panel(days, 40, seed=SEED)
        sessions = _sessions(*days)
        full = _compute(open_, high, low, close, volume, sessions)
        daily_h, daily_l, daily_c = _daily()
        full["wilder_atr"] = wilder_atr(daily_h, daily_l, daily_c, n=3)
        full["prior_session_atr"] = prior_session_atr(daily_h, daily_l, daily_c, n=3)
        assert set(full) == set(FEATURES)

        rng = np.random.default_rng(SEED)
        picks = rng.choice(len(close.index) - 12, size=6, replace=False) + 10
        finite = 0
        for pos in picks:
            t = close.index[int(pos)]
            part = _compute(
                _slice(open_, t), _slice(high, t), _slice(low, t),
                _slice(close, t), _slice(volume, t), sessions,
            )
            for name, result in full.items():
                if name in ("wilder_atr", "prior_session_atr"):
                    continue
                _assert_row(_at(result, t), _at(part[name], t))
                row = result.loc[t] if not isinstance(result, tuple) else result[0].loc[t]
                finite += int(np.isfinite(row.to_numpy(dtype=float)).sum())
        assert finite > 0

        for pos in (3, 5, 7):
            t = daily_c.index[pos]
            atr = wilder_atr(daily_h.loc[:t], daily_l.loc[:t], daily_c.loc[:t], n=3)
            prior = prior_session_atr(daily_h.loc[:t], daily_l.loc[:t], daily_c.loc[:t], n=3)
            pd.testing.assert_series_equal(atr.loc[t], full["wilder_atr"].loc[t])
            pd.testing.assert_series_equal(prior.loc[t], full["prior_session_atr"].loc[t])

    def test_rvol_no_incluye_volumen_posterior_del_mismo_rango(self):
        # Dos marcas dentro de [open+k−barra, open+k). El volumen de la
        # segunda no puede cambiar el ratio ya publicado en la primera.
        start = pd.Timestamp("2024-01-02 09:30", tz=NY)
        day1 = pd.date_range(start, periods=12, freq="30s").tz_convert(UTC)
        day2 = pd.date_range(start + pd.Timedelta(1, unit="D"), periods=12, freq="30s").tz_convert(UTC)
        index = day1.append(day2)
        volume = pd.DataFrame(10.0, index=index, columns=["SPY", "QQQ"])
        volume.loc[day2, :] = 1.0
        early = day2[8]  # 09:34:00
        late = day2[9]  # 09:34:30
        volume.loc[late, "SPY"] = 1000.0

        kwargs = {"k_minutes": 5, "sessions_back": 1, "timeframe": "1Min"}
        full = relative_volume(volume, **kwargs)
        part = relative_volume(volume.loc[:early], **kwargs)
        # 9 impresiones de 1 frente a 10×10 de la sesión previa.
        assert full.loc[early, "SPY"] == pytest.approx(9.0 / 100.0)
        assert full.loc[early, "QQQ"] == pytest.approx(9.0 / 100.0)
        pd.testing.assert_series_equal(part.loc[early], full.loc[early])
        assert full.loc[late, "SPY"] == pytest.approx(1009.0 / 100.0)
        assert full.loc[late, "QQQ"] == pytest.approx(10.0 / 100.0)

        # timeframe más grueso que la rejilla: 09:30 no ve el resto del rango.
        minutes = _minutes("2024-01-02", 6).append(_minutes("2024-01-03", 6))
        coarse = pd.DataFrame({"SPY": [10.0] * 6 + [1.0, 1.0, 1.0, 1.0, 1000.0, 1.0]}, index=minutes)
        opened = minutes[6]
        coarse_full = relative_volume(coarse, k_minutes=5, sessions_back=1, timeframe="5Min")
        coarse_part = relative_volume(coarse.loc[:opened], k_minutes=5, sessions_back=1, timeframe="5Min")
        assert coarse_full.loc[opened, "SPY"] == pytest.approx(1.0 / 50.0)
        assert coarse_part.loc[opened, "SPY"] == pytest.approx(coarse_full.loc[opened, "SPY"])

    def test_series_y_dataframe_coinciden(self):
        _, high, _, close, _ = _panel(("2024-01-02",), 15, seed=3, columns=("SPY",))
        pd.testing.assert_series_equal(
            log_return(close["SPY"]),
            log_return(close)["SPY"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            parkinson_variance(high["SPY"], high["SPY"] * 0 + high["SPY"]),
            parkinson_variance(high, high)["SPY"],
            check_names=False,
        )


class TestClosedForm:
    def test_vwap_con_volumen_constante_es_la_media_del_precio_tipico(self):
        index = _minutes("2024-01-02", 4)
        columns = ["SPY"]
        # tp = (h+l+c)/3 = 3, 4, 5, 6. Con volumen constante el VWAP es la media.
        high = _frame([[3], [6], [9], [12]], index, columns)
        low = _frame([[3], [3], [3], [3]], index, columns)
        close = _frame([[3], [3], [3], [3]], index, columns)
        volume = _frame([[5], [5], [5], [5]], index, columns)
        vwap = session_vwap(high, low, close, volume)
        np.testing.assert_allclose(vwap["SPY"].to_numpy(), [3.0, 3.5, 4.0, 4.5])

    def test_vwap_de_alpaca_sustituye_al_precio_tipico(self):
        index = _minutes("2024-01-02", 2)
        columns = ["SPY"]
        high = _frame([[10], [10]], index, columns)
        low = close = high.copy()
        volume = _frame([[1], [1]], index, columns)
        vw = _frame([[7], [np.nan]], index, columns)
        vwap = session_vwap(high, low, close, volume, alpaca_vw=vw)
        # Barra 0 usa vw=7. Barra 1 no tiene vw y cae a tp=10. Acumulado (7+10)/2.
        np.testing.assert_allclose(vwap["SPY"].to_numpy(), [7.0, 8.5])

    def test_roll_recupera_el_spread_del_rebote(self):
        rng = np.random.default_rng(SEED)
        spread = 0.10
        signs = rng.choice([-1.0, 1.0], size=20_000)
        price = pd.Series(50.0 + (spread / 2.0) * signs)
        est = roll_spread(price, n=8_000).iloc[-1]
        assert est == pytest.approx(spread, rel=0.10)
        # Covarianza no negativa: el estimador no inventa un 0.
        flat = pd.Series(np.linspace(10.0, 11.0, 40))
        assert roll_spread(flat, n=10).isna().all()

    def test_parkinson_en_gbm_se_acerca_a_sigma(self):
        sigma = 0.01
        n_bars = 8_000
        sub = 2_048
        rng = np.random.default_rng(SEED)
        shocks = rng.normal(0.0, sigma / math.sqrt(sub), size=(n_bars, sub))
        path = np.cumsum(shocks, axis=1)
        full = np.concatenate([np.zeros((n_bars, 1)), path], axis=1)
        # Precios relativos a la apertura. ln(H/L) = max(log) − min(log).
        high = pd.Series(np.exp(full.max(axis=1)))
        low = pd.Series(np.exp(full.min(axis=1)))
        est = parkinson_variance(high, low)
        # El camino es un paseo fino, no el extremo continuo: el sesgo es
        # de unos pocos puntos porcentuales. Una constante mal puesta (sin
        # el 4, o varianza en vez de la fórmula) cae muy fuera.
        assert est.mean() == pytest.approx(sigma ** 2, rel=0.08)
        exact = (0.01 ** 2) / (4.0 * math.log(2.0))
        one = parkinson_variance(pd.Series([math.exp(0.01)]), pd.Series([1.0])).iloc[0]
        assert one == pytest.approx(exact)
        assert parkinson_variance(pd.Series([1.0]), pd.Series([1.0])).iloc[0] == 0.0
        assert math.isnan(parkinson_variance(pd.Series([1.0]), pd.Series([2.0])).iloc[0])

    def test_garman_klass_cerrado(self):
        # O=1, H=e^0.02, L=e^{-0.01}, C=e^0.005. El cierre y la apertura caen dentro.
        open_ = pd.Series([1.0])
        high = pd.Series([math.exp(0.02)])
        low = pd.Series([math.exp(-0.01)])
        close = pd.Series([math.exp(0.005)])
        log_hl = 0.03
        log_co = 0.005
        expected = 0.5 * log_hl ** 2 - (2.0 * math.log(2.0) - 1.0) * log_co ** 2
        assert garman_klass_variance(open_, high, low, close).iloc[0] == pytest.approx(expected)
        broken = garman_klass_variance(
            pd.Series([1.0]), pd.Series([1.01]), pd.Series([0.99]), pd.Series([1.5]),
        )
        assert math.isnan(broken.iloc[0])

    def test_corwin_schultz_dos_barras(self):
        high = pd.Series([10.0, 11.0])
        low = pd.Series([9.0, 10.0])
        beta = math.log(10.0 / 9.0) ** 2 + math.log(11.0 / 10.0) ** 2
        gamma = math.log(11.0 / 9.0) ** 2
        den = 3.0 - 2.0 * math.sqrt(2.0)
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / den - math.sqrt(gamma / den)
        expected = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        got = corwin_schultz_spread(high, low)
        assert math.isnan(got.iloc[0])
        assert got.iloc[1] == pytest.approx(expected)

    def test_atr_de_wilder_y_la_sesion_en_curso_no_entra(self):
        high = pd.Series([10.0, 12.0, 14.0, 20.0])
        low = pd.Series([8.0, 11.0, 12.0, 10.0])
        close = pd.Series([9.0, 11.5, 13.0, 18.0])
        # TR: 2, max(1, |12-9|, |11-9|)=3, max(2, |14-11.5|, |12-11.5|)=2.5
        # ATR en la tercera fila = media(2, 3, 2.5) = 2.5
        # ATR siguiente = (2*2.5 + TR4) / 3, con TR4 = max(10, |20-13|, |10-13|)=10
        atr = wilder_atr(high, low, close, n=3)
        assert math.isnan(atr.iloc[0]) and math.isnan(atr.iloc[1])
        assert atr.iloc[2] == pytest.approx(2.5)
        assert atr.iloc[3] == pytest.approx((2.0 * 2.5 + 10.0) / 3.0)
        prior = prior_session_atr(high, low, close, n=3)
        assert math.isnan(prior.iloc[2])
        assert prior.iloc[3] == pytest.approx(2.5)
        blown = high.copy()
        blown.iloc[-1] = 100.0
        assert prior_session_atr(blown, low, close, n=3).iloc[-1] == pytest.approx(prior.iloc[-1])
        assert wilder_atr(blown, low, close, n=3).iloc[-1] != pytest.approx(atr.iloc[-1])

    def test_cierre_en_cero_no_deja_un_retorno_infinito(self):
        close = pd.Series([0.0, 10.0, 11.0])
        returns = log_return(close)
        assert math.isnan(returns.iloc[1])
        assert returns.iloc[2] == pytest.approx(math.log(1.1))

        index = _minutes("2024-01-02", 3)
        prices = pd.DataFrame({"SPY": [0.0, 10.0, 11.0]}, index=index)
        volume = pd.DataFrame({"SPY": [100.0, 100.0, 100.0]}, index=index)
        bar = amihud(prices, volume)
        session = amihud_session(prices, volume)
        assert math.isnan(float(bar["SPY"].iloc[1]))
        assert math.isnan(float(session["SPY"].iloc[1]))
        expected = abs(math.log(1.1)) / (11.0 * 100.0)
        assert bar["SPY"].iloc[2] == pytest.approx(expected)
        assert session["SPY"].iloc[2] == pytest.approx(expected)

    def test_microprice_spread_y_ofi(self):
        bid = pd.Series([10.0, 10.0, 10.01])
        ask = pd.Series([10.02, 10.02, 10.02])
        bid_size = pd.Series([100.0, 150.0, 40.0])
        ask_size = pd.Series([100.0, 80.0, 80.0])
        micro = microprice(bid, ask, bid_size, ask_size)
        # I = 300/400 no es esta fila. Fila 0: I=0.5 → midpoint 10.01.
        assert micro.iloc[0] == pytest.approx(10.01)
        weight = 150.0 / 230.0
        assert micro.iloc[1] == pytest.approx(weight * 10.02 + (1.0 - weight) * 10.0)
        bps = quoted_spread_bps(bid, ask)
        assert bps.iloc[0] == pytest.approx(0.02 / 10.01 * 1.0e4)
        locked = quoted_spread_bps(pd.Series([5.0]), pd.Series([5.0]))
        assert locked.iloc[0] == 0.0
        ofi = order_flow_imbalance(bid, ask, bid_size, ask_size)
        assert math.isnan(ofi.iloc[0])
        # Precios quietos: Δq_bid − Δq_ask = 50 − (80−100) = 70.
        assert ofi.iloc[1] == pytest.approx(70.0)
        # Bid sube: entra el tamaño nuevo del bid (40) y el ask no cambia.
        assert ofi.iloc[2] == pytest.approx(40.0)


class TestSessionReset:
    def test_el_vwap_empieza_de_cero_en_la_sesion_siguiente(self):
        index = _minutes("2024-01-02", 2).append(_minutes("2024-01-03", 2))
        columns = ["SPY"]
        price_1 = _frame([[10], [10], [30], [30]], index, columns)
        volume = _frame([[1], [1], [1], [1]], index, columns)
        vwap = session_vwap(price_1, price_1, price_1, volume)
        np.testing.assert_allclose(vwap["SPY"].to_numpy(), [10.0, 10.0, 30.0, 30.0])

    def test_la_vol_no_arrastra_el_cierre_de_ayer(self):
        # Sesión 1 plana. Sesión 2 alterna 100 y 110. La vol de la segunda
        # no se diluye con los ceros de la primera.
        s1 = _minutes("2024-01-02", 8)
        s2 = _minutes("2024-01-03", 8)
        index = s1.append(s2)
        prices = [100.0] * 8 + [100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0]
        close = pd.DataFrame({"SPY": prices}, index=index)
        vol = realized_vol(close, n=4)
        assert math.isnan(vol["SPY"].iloc[8])
        a = math.log(1.1)
        # Retornos intradía desde la segunda barra: +a, −a, +a, −a. El
        # primero de la sesión (contra el cierre de ayer) no entra.
        expected = float(np.std([a, -a, a, -a], ddof=1))
        assert vol["SPY"].iloc[12] == pytest.approx(expected)
        assert vol["SPY"].iloc[7] == pytest.approx(0.0)

    def test_rango_de_apertura_solo_despues_de_cerrar_y_sin_el_resto(self):
        index = _minutes("2024-01-02", 8)
        high = pd.DataFrame({"SPY": [1, 2, 3, 4, 5, 9, 9, 9]}, index=index, dtype=float)
        low = pd.DataFrame({"SPY": [1, 1, 1, 1, 0.5, 0.1, 0.1, 0.1]}, index=index)
        hi, lo = opening_range(high, low, k_minutes=5)
        assert hi["SPY"].iloc[:4].isna().all()
        assert hi["SPY"].iloc[4:].eq(5.0).all()
        assert lo["SPY"].iloc[4:].eq(0.5).all()

    def test_rvol_usa_solo_sesiones_previas(self):
        days = ("2024-01-02", "2024-01-03", "2024-01-04")
        index = _minutes(days[0], 6).append([_minutes(day, 6) for day in days[1:]])
        volume = pd.DataFrame(
            {"SPY": [10.0] * 6 + [10.0] * 6 + [20.0] * 6},
            index=index,
        )
        rvol = relative_volume(volume, k_minutes=5, sessions_back=2)
        third = rvol["SPY"].iloc[12:]
        assert third.iloc[:4].isna().all()
        assert third.iloc[4] == pytest.approx(2.0)
        # Una sesión futura no cambia el RVOL ya publicado.
        bumped = volume.copy()
        bumped.iloc[-1, 0] = 1.0e6
        again = relative_volume(bumped, k_minutes=5, sessions_back=2)
        assert again["SPY"].iloc[16] == pytest.approx(2.0)
        assert rvol["SPY"].iloc[:12].isna().all()

    def test_fuera_de_la_sesion_explicita_no_entra(self):
        morning = _minutes("2024-07-03", 3)
        after = _minutes("2024-07-03", 2, hour=13, minute=0)
        index = morning.append(after)
        price = pd.DataFrame({"SPY": [10.0, 10.0, 10.0, 99.0, 99.0]}, index=index)
        volume = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=index)
        half = session_at(date(2024, 7, 3), time(9, 30), time(13, 0))
        vwap = session_vwap(price, price, price, volume, sessions=(half,))
        assert vwap["SPY"].iloc[:3].eq(10.0).all()
        assert vwap["SPY"].iloc[3:].isna().all()


class TestLookbackReturn:
    def test_retorno_de_n_minutos_y_recorte_en_la_apertura(self):
        index = _minutes("2024-01-02", 6)
        open_ = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=index)
        close = pd.DataFrame({"SPY": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]}, index=index)
        out = lookback_log_return(open_, close, minutes=3)
        # 09:34: la base es el cierre de 09:31 (tres minutos antes, misma sesión).
        assert out["SPY"].iloc[4] == pytest.approx(math.log(105.0 / 102.0))
        # 09:33: la base es el cierre de 09:30, la primera barra.
        assert out["SPY"].iloc[3] == pytest.approx(math.log(104.0 / 101.0))
        # 09:31: 09:28 cae antes de la apertura. La base es el open de 09:30.
        assert out["SPY"].iloc[1] == pytest.approx(math.log(102.0 / 100.0))
        assert out["SPY"].iloc[0] == pytest.approx(math.log(101.0 / 100.0))

    def test_no_cruza_la_sesion_ni_salta_un_hueco(self):
        day1 = _minutes("2024-01-02", 4)
        day2 = _minutes("2024-01-03", 4)
        index = day1.append(day2)
        open_ = pd.DataFrame({"SPY": [10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0]}, index=index)
        close = pd.DataFrame({"SPY": [10.0, 10.0, 10.0, 10.0, 55.0, 60.0, 60.0, 60.0]}, index=index)
        out = lookback_log_return(open_, close, minutes=2)
        # 09:31 del segundo día: la base es su propio open, no el cierre de ayer.
        assert out["SPY"].iloc[5] == pytest.approx(math.log(60.0 / 50.0))
        # Sin la barra de 09:31, 09:33 no tiene base: NaN, no la fila anterior.
        gap = index.delete(5)
        holes = lookback_log_return(open_.loc[gap], close.loc[gap], minutes=2)
        assert math.isnan(holes["SPY"].loc[day2[3]])

    def test_fuera_de_la_sesion_es_nan_y_los_minutos_cubren_barras(self):
        morning = _minutes("2024-07-03", 3)
        after = _minutes("2024-07-03", 2, hour=13, minute=0)
        index = morning.append(after)
        price = pd.DataFrame({"SPY": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=index)
        half = session_at(date(2024, 7, 3), time(9, 30), time(13, 0))
        out = lookback_log_return(price, price, minutes=1, sessions=(half,))
        assert out["SPY"].iloc[:3].notna().all()
        assert out["SPY"].iloc[3:].isna().all()
        with pytest.raises(ValueError, match="minutes"):
            lookback_log_return(price, price, minutes=3, timeframe="2Min")


class TestPriorSessionsVol:
    def test_mezcla_las_sesiones_previas_sin_la_en_curso(self):
        days = ("2024-01-02", "2024-01-03", "2024-01-04")
        p1 = [100.0, 101.0, 100.0, 102.0]
        p2 = [50.0, 51.0, 50.0, 49.0]
        p3 = [10.0, 20.0, 5.0, 30.0]
        index = _minutes(days[0], 4).append([_minutes(day, 4) for day in days[1:]])
        close = pd.DataFrame({"SPY": p1 + p2 + p3}, index=index)
        r1 = np.diff(np.log(p1))
        r2 = np.diff(np.log(p2))
        vol = prior_sessions_vol(close, n_sessions=2)
        assert vol["SPY"].iloc[:8].isna().all()
        expected = float(np.std(np.concatenate([r1, r2]), ddof=1))
        np.testing.assert_allclose(vol["SPY"].iloc[8:].to_numpy(), expected, rtol=1e-12)
        one = prior_sessions_vol(close, n_sessions=1)
        np.testing.assert_allclose(one["SPY"].iloc[4:8].to_numpy(), np.std(r1, ddof=1), rtol=1e-12)
        np.testing.assert_allclose(one["SPY"].iloc[8:].to_numpy(), np.std(r2, ddof=1), rtol=1e-12)
        # La sesión en curso no entra: cambiar hoy no mueve el valor de hoy.
        bumped = close.copy()
        bumped.iloc[9:, 0] = 999.0
        pd.testing.assert_series_equal(prior_sessions_vol(bumped, n_sessions=2)["SPY"], vol["SPY"])

    def test_una_sesion_del_calendario_sin_barras_es_nan(self):
        sessions = _sessions("2024-01-02", "2024-01-03", "2024-01-04")
        index = _minutes("2024-01-02", 4).append(_minutes("2024-01-04", 4))
        close = pd.DataFrame({"SPY": [100.0, 101.0, 100.0, 102.0] * 2}, index=index)
        strict = prior_sessions_vol(close, n_sessions=1, sessions=sessions)
        assert strict["SPY"].isna().all()
        # Sin calendario, las sesiones son las fechas presentes en el frame.
        loose = prior_sessions_vol(close, n_sessions=1)
        expected = float(np.std(np.diff(np.log([100.0, 101.0, 100.0, 102.0])), ddof=1))
        np.testing.assert_allclose(loose["SPY"].iloc[4:].to_numpy(), expected, rtol=1e-12)

    def test_un_hueco_no_cuenta_como_retorno_de_una_barra(self):
        index = _minutes("2024-01-02", 6).delete(2).append(_minutes("2024-01-03", 2))
        # 09:30, 09:31, 09:33, 09:34, 09:35 y dos barras del día siguiente.
        close = pd.DataFrame({"SPY": [100.0, 102.0, 101.0, 104.0, 103.0, 50.0, 51.0]}, index=index)
        vol = prior_sessions_vol(close, n_sessions=1)
        # 09:33 no tiene la barra de 09:32: ese retorno de dos minutos no entra.
        kept = [math.log(102.0 / 100.0), math.log(104.0 / 101.0), math.log(103.0 / 104.0)]
        expected = float(np.std(kept, ddof=1))
        np.testing.assert_allclose(vol["SPY"].iloc[5:].to_numpy(), expected, rtol=1e-12)


class TestTrend:
    def test_reimplementa_la_pendiente_y_no_rellena_con_cero(self):
        rng = np.random.default_rng(SEED)
        index = pd.bdate_range("2020-01-02", periods=80)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=80))), index=index)
        legacy = _trend_score(prices, ma_window=20)
        ours = trend_score(prices, ma_window=20, look=21, by_session=False).iloc[-1]
        assert ours == pytest.approx(legacy, abs=1e-8)
        assert ours > -2.0
        short = trend_score(prices.iloc[:10], ma_window=20, look=21, by_session=False)
        assert short.isna().all()
        assert _trend_score(prices.iloc[:10], ma_window=20) == 0.0
        up = pd.Series(np.linspace(100.0, 130.0, 60))
        assert trend_score(up, ma_window=10, look=10, by_session=False).iloc[-1] > 0

    def test_el_modulo_no_importa_regime_detector(self):
        from pathlib import Path

        source = Path(__file__).resolve().parents[2] / "src" / "hft" / "features.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any("regime_detector" in name for name in imported)


class TestHistory:
    def test_cada_feature_declara_historia(self):
        assert required_history("log_return") == (2, 0)
        assert required_history("realized_vol", n=30) == (31, 1)
        assert required_history("realized_vol", n=30, by_session=False) == (31, 0)
        assert required_history("parkinson_variance") == (1, 0)
        assert required_history("garman_klass_variance") == (1, 0)
        assert required_history("wilder_atr", n=14) == (14, 0)
        assert required_history("prior_session_atr", n=14) == (15, 15)
        assert required_history("session_vwap") == (1, 1)
        assert required_history("vwap_zscore", n=30) == (30, 1)
        assert required_history("opening_range", k_minutes=5, timeframe="1Min") == (5, 1)
        assert required_history("relative_volume", k_minutes=5, sessions_back=14) == (5, 15)
        assert required_history("trend_score", ma_window=50, look=21) == (70, 1)
        assert required_history("trend_score", ma_window=50, look=21, by_session=False) == (70, 0)
        assert required_history("roll_spread", n=30) == (32, 0)
        assert required_history("corwin_schultz_spread") == (2, 0)
        assert required_history("amihud") == (2, 1)
        assert required_history("amihud_session") == (2, 1)
        assert required_history("microprice") == (1, 0)
        assert required_history("quoted_spread_bps") == (1, 0)
        assert required_history("order_flow_imbalance") == (2, 0)
        assert required_history("lookback_log_return", minutes=60, timeframe="1Min") == (61, 1)
        assert required_history("lookback_log_return", minutes=60, timeframe="5Min") == (13, 1)
        assert required_history("prior_sessions_vol", n_sessions=20) == (1, 21)
        with pytest.raises(ValueError, match="desconocida"):
            required_history("sharpe")

    def test_la_historia_corta_es_nan_y_no_cero(self):
        index = _minutes("2024-01-02", 3)
        close = pd.DataFrame({"SPY": [10.0, 11.0, 12.0]}, index=index)
        volume = pd.DataFrame({"SPY": [0.0, 0.0, 0.0]}, index=index)
        assert math.isnan(log_return(close)["SPY"].iloc[0])
        assert realized_vol(close, n=5).isna().all().all()
        assert session_vwap(close, close, close, volume).isna().all().all()
        assert amihud(close, volume).isna().all().all()
        assert not (log_return(close).fillna(1.0) == 0.0).any().any()


class TestIndexUnit:
    """pandas 3 infiere la unidad ("us" desde strings o read_csv): no cambia nada."""

    @staticmethod
    def _ns(result):
        if isinstance(result, tuple):
            return tuple(TestIndexUnit._ns(part) for part in result)
        return result.set_axis(result.index.as_unit("ns"))

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    @pytest.mark.parametrize("with_sessions", [False, True])
    def test_la_unidad_del_indice_no_cambia_los_features(self, unit, with_sessions):
        days = ("2024-01-02", "2024-01-03", "2024-01-04")
        frames = _panel(days, 40, SEED)
        sessions = _sessions(*days) if with_sessions else None
        base = _compute(*(f.set_axis(f.index.as_unit("ns")) for f in frames), sessions=sessions)
        other = _compute(*(f.set_axis(f.index.as_unit(unit)) for f in frames), sessions=sessions)
        for name, expected in base.items():
            got = self._ns(other[name])
            if isinstance(expected, tuple):
                for left, right in zip(expected, got):
                    pd.testing.assert_frame_equal(left, right, obj=name)
            else:
                pd.testing.assert_frame_equal(expected, got, obj=name)


class TestPerformance:
    def test_un_ano_de_un_minuto_cabe_en_segundos(self):
        import time

        idx = pd.date_range("2024-01-02 09:30", "2024-12-31 16:00", freq="1min", tz=NY)
        minutes = idx.hour * 60 + idx.minute
        mask = (idx.dayofweek < 5) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
        index = idx[mask].tz_convert(UTC)
        rng = np.random.default_rng(SEED)
        n, k = len(index), 30
        columns = [f"S{i:02d}" for i in range(k)]
        rets = rng.normal(0.0, 0.0008, size=(n, k))
        close_a = 100.0 * np.exp(np.cumsum(rets, axis=0))
        open_a = np.vstack([close_a[0], close_a[:-1]])
        wick = np.abs(rng.normal(0.0, 0.0003, size=(n, k))) + 1.0e-4
        high_a = np.maximum(open_a, close_a) * (1.0 + wick)
        low_a = np.minimum(open_a, close_a) * (1.0 - wick)
        volume_a = rng.lognormal(10.0, 0.2, size=(n, k))
        open_ = _frame(open_a, index, columns)
        high = _frame(high_a, index, columns)
        low = _frame(low_a, index, columns)
        close = _frame(close_a, index, columns)
        volume = _frame(volume_a, index, columns)
        started = time.perf_counter()
        returns = log_return(close)
        realized_vol(close, n=30)
        parkinson_variance(high, low)
        garman_klass_variance(open_, high, low, close)
        vwap = session_vwap(high, low, close, volume)
        vwap_zscore(close, vwap, n=30)
        opening_range(high, low, k_minutes=5)
        relative_volume(volume, k_minutes=5, sessions_back=14)
        trend_score(close, ma_window=50, look=21)
        roll_spread(close, n=30)
        corwin_schultz_spread(high, low)
        amihud(close, volume)
        amihud_session(close, volume)
        lookback_log_return(open_, close, minutes=60)
        prior_sessions_vol(close, n_sessions=20)
        elapsed = time.perf_counter() - started
        assert returns.shape == close.shape
        assert math.isnan(returns.iloc[0, 0])
        # 30 símbolos × ~252 sesiones × 390 minutos. El presupuesto deja
        # margen al CI; el tiempo medido va en el PR.
        assert elapsed < 30.0, f"{elapsed:.2f}s para {n} filas × {k} símbolos"
