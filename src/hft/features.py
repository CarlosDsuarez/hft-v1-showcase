"""Features intradía puras y causales (HFT-05).

Cada función recibe un ``DataFrame`` (columna = símbolo) o una ``Series`` y
devuelve el mismo tipo, alineado al índice. El valor en ``t`` usa solo
filas ``<= t``. Si falta historia el resultado es NaN: no se rellena con 0.
Un 0 solo aparece cuando la fórmula da cero (rango nulo, spread cotizado
nulo).

``required_history(nombre, **params)`` declara esa historia al estilo de
``src.history_requirements``: una tupla ``(barras, sesiones)``.

- ``barras``: filas con índice ``<= t`` (t incluido) en una rejilla sin
  huecos.
- ``sesiones``: sesiones con fecha ``<=`` la de ``t``, incluida la sesión
  de ``t`` cuando la feature la lee. 0 si la feature no corta por sesión.

Las features de sesión exigen un ``DatetimeIndex`` con zona. Sin una
secuencia de ``Session``, la etiqueta es la fecha en America/New_York y
la apertura es la primera barra de esa fecha presente en el frame (no
incluye premarket si el frame no lo trae). Con sesiones de
``src.hft.calendar``, la barra entra solo en ``[open, close)`` y la
apertura es la del calendario.

HFT-04 guarda barras, no un histórico de quotes. ``microprice``,
``quoted_spread_bps`` y ``order_flow_imbalance`` operan sobre series que
el llamador alinea. El cliente solo expone ``GET /v2/stocks/quotes/latest``
(``bp``, ``bs``, ``ap``, ``as``).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.hft.calendar import NY, Session

Frame = Union[pd.Series, pd.DataFrame]

DEFAULT_VOL_BARS = 30
DEFAULT_ATR_PERIOD = 14
DEFAULT_OPENING_MINUTES = 5
DEFAULT_RVOL_SESSIONS = 14
DEFAULT_ROLL_BARS = 30
DEFAULT_TREND_MA = 50
DEFAULT_TREND_LOOK = 21
DEFAULT_Z_BARS = 30
DEFAULT_CLIP = 2.0
DEFAULT_LOOKBACK_MINUTES = 60
DEFAULT_SIGMA_SESSIONS = 20

# Nombres que acepta ``required_history``. El orden es el de este módulo.
FEATURES: Tuple[str, ...] = (
    "log_return",
    "realized_vol",
    "parkinson_variance",
    "garman_klass_variance",
    "wilder_atr",
    "prior_session_atr",
    "session_vwap",
    "vwap_zscore",
    "opening_range",
    "relative_volume",
    "trend_score",
    "roll_spread",
    "corwin_schultz_spread",
    "amihud",
    "amihud_session",
    "microprice",
    "quoted_spread_bps",
    "order_flow_imbalance",
    "lookback_log_return",
    "prior_sessions_vol",
)


def required_history(feature: str, **params) -> Tuple[int, int]:
    """``(barras, sesiones)`` con índice ``<= t`` para que ``feature`` no sea NaN.

    Los defaults coinciden con los de cada función. ``sessions_back`` es
    ``D`` de RVOL. ``by_session=False`` pone sesiones en 0 en la vol, el
    Amihud y la tendencia.
    """
    if feature not in FEATURES:
        raise ValueError(f"feature desconocida: {feature!r}")
    by_session = bool(params.get("by_session", True))
    if feature == "log_return":
        return (2, 0)
    if feature == "realized_vol":
        n = _positive_int("n", params.get("n", DEFAULT_VOL_BARS), minimum=2)
        return (n + 1, 1 if by_session else 0)
    if feature in ("parkinson_variance", "garman_klass_variance", "microprice", "quoted_spread_bps"):
        return (1, 0)
    if feature == "wilder_atr":
        n = _positive_int("n", params.get("n", DEFAULT_ATR_PERIOD), minimum=1)
        return (n, 0)
    if feature == "prior_session_atr":
        n = _positive_int("n", params.get("n", DEFAULT_ATR_PERIOD), minimum=1)
        return (n + 1, n + 1)
    if feature == "session_vwap":
        return (1, 1)
    if feature == "vwap_zscore":
        n = _positive_int("n", params.get("n", DEFAULT_Z_BARS), minimum=2)
        return (n, 1 if by_session else 0)
    if feature == "opening_range":
        return (_opening_bars(params), 1)
    if feature == "relative_volume":
        d = _positive_int(
            "sessions_back", params.get("sessions_back", DEFAULT_RVOL_SESSIONS), minimum=1,
        )
        return (_opening_bars(params), d + 1)
    if feature == "trend_score":
        ma = _positive_int("ma_window", params.get("ma_window", DEFAULT_TREND_MA), minimum=1)
        look = _positive_int("look", params.get("look", DEFAULT_TREND_LOOK), minimum=2)
        return (ma + look - 1, 1 if by_session else 0)
    if feature == "roll_spread":
        n = _positive_int("n", params.get("n", DEFAULT_ROLL_BARS), minimum=2)
        return (n + 2, 0)
    if feature == "corwin_schultz_spread":
        return (2, 0)
    if feature == "amihud":
        return (2, 1 if by_session else 0)
    if feature == "amihud_session":
        return (2, 1)
    if feature == "order_flow_imbalance":
        return (2, 0)
    if feature == "lookback_log_return":
        minutes = _positive_int(
            "minutes", params.get("minutes", DEFAULT_LOOKBACK_MINUTES), minimum=1,
        )
        bar_minutes = _bar_minutes(params.get("timeframe", "1Min"))
        return (_lookback_bars(minutes, bar_minutes) + 1, 1)
    if feature == "prior_sessions_vol":
        n = _positive_int(
            "n_sessions", params.get("n_sessions", DEFAULT_SIGMA_SESSIONS), minimum=1,
        )
        return (1, n + 1)
    raise ValueError(f"feature desconocida: {feature!r}")


def log_return(close: Frame) -> Frame:
    """Retorno logarítmico por barra: ``r_t = ln(C_t / C_{t-1})``.

    La primera fila es NaN. El salto usa la fila anterior del frame, que
    puede ser el cierre de la sesión previa si el frame la trae. Historia:
    ``required_history("log_return") == (2, 0)``.
    """
    frame, squeeze = _box(close)
    ratio = frame / frame.shift(1)
    # Un cierre previo en 0 hace el ratio infinito. Eso no es un retorno:
    # log(inf) contaminaría el Amihud de toda la sesión.
    usable = ratio.where(np.isfinite(ratio) & (ratio > 0))
    return _restore(np.log(usable), squeeze)


def realized_vol(
    close: Frame,
    n: int = DEFAULT_VOL_BARS,
    *,
    by_session: bool = True,
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """Volatilidad realizada: desviación típica (``ddof=1``) de ``n`` retornos log.

    Con ``by_session=True`` (default) la ventana no cruza la apertura y el
    retorno que usa el cierre de la sesión anterior se descarta: no es un
    retorno de un minuto. Historia: ``n + 1`` barras; 1 sesión si corta
    por sesión.
    """
    n = _positive_int("n", n, minimum=2)
    returns = log_return(close)
    if by_session:
        labels = _labels(close.index, sessions)
        returns = returns.where(_same_session(labels, 2))
    vol = returns.rolling(n, min_periods=n).std(ddof=1)
    if by_session:
        vol = vol.where(_same_session(labels, n + 1))
    return vol


def parkinson_variance(high: Frame, low: Frame) -> Frame:
    """Varianza de Parkinson (1980) por barra.

    ``σ²_t = (ln(H_t / L_t))² / (4 ln 2)``.

    Estima la varianza del movimiento browniano de la barra si el máximo
    y el mínimo se observan de forma continua. ``H < L`` o un precio no
    positivo es NaN. ``H == L`` es 0 (no hubo rango). Historia: 1 barra.
    """
    high_f, squeeze = _box(high)
    low_f, _ = _box(low)
    _same_axes(high_f, low_f, "low")
    hl = (high_f / low_f).where((high_f > 0) & (low_f > 0) & (high_f >= low_f))
    out = np.log(hl) ** 2 / (4.0 * np.log(2.0))
    return _restore(out, squeeze)


def garman_klass_variance(
    open_: Frame,
    high: Frame,
    low: Frame,
    close: Frame,
) -> Frame:
    """Varianza de Garman–Klass (1980) por barra.

    ``σ² = ½ (ln H/L)² − (2 ln 2 − 1) (ln C/O)²``.

    Se anula la barra si el OHLC no contiene a la apertura y al cierre
    (``H < max(O, C)`` o ``L > min(O, C)``) o si algún precio no es
    positivo. Historia: 1 barra.
    """
    open_f, squeeze = _box(open_)
    high_f, _ = _box(high)
    low_f, _ = _box(low)
    close_f, _ = _box(close)
    _same_axes(open_f, high_f, "high")
    _same_axes(open_f, low_f, "low")
    _same_axes(open_f, close_f, "close")
    upper = open_f.where(open_f >= close_f, close_f)
    lower = open_f.where(open_f <= close_f, close_f)
    contained = (high_f >= upper) & (low_f <= lower)
    valid = contained & (open_f > 0) & (high_f > 0) & (low_f > 0) & (close_f > 0) & (high_f >= low_f)
    log_hl = np.log((high_f / low_f).where(valid))
    log_co = np.log((close_f / open_f).where(valid))
    out = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return _restore(out.where(valid), squeeze)


def wilder_atr(
    high: Frame,
    low: Frame,
    close: Frame,
    n: int = DEFAULT_ATR_PERIOD,
) -> Frame:
    """ATR de Wilder (1978) sobre el frame que recibe (una fila por sesión).

    ``TR_t = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)``. Sin cierre
    previo, ``TR = H − L``. El primer ATR es la media de ``n`` true range
    consecutivos; después ``ATR_t = ((n − 1) ATR_{t−1} + TR_t) / n``. En
    ``t`` entra la barra ``t``. Un hueco corta la recursión y vuelve a
    sembrar con las ``n`` barras finitas siguientes: no se rellena.

    Para stops de ORB usar ``prior_session_atr``: este valor incluye la
    sesión en curso. Historia: ``n`` filas.
    """
    n = _positive_int("n", n, minimum=1)
    high_f, squeeze = _box(high)
    low_f, _ = _box(low)
    close_f, _ = _box(close)
    _same_axes(high_f, low_f, "low")
    _same_axes(high_f, close_f, "close")
    tr = _true_range(high_f, low_f, close_f)
    out = pd.DataFrame(np.nan, index=tr.index, columns=tr.columns, dtype=float)
    for col in tr.columns:
        out[col] = _wilder_1d(tr[col].to_numpy(dtype=float), n)
    return _restore(out, squeeze)


def prior_session_atr(
    high: Frame,
    low: Frame,
    close: Frame,
    n: int = DEFAULT_ATR_PERIOD,
) -> Frame:
    """ATR de Wilder de las sesiones estrictamente anteriores.

    El valor en la fila ``t`` es el ATR calculado con filas ``<= t−1``.
    El frame es diario (una fila por sesión, en orden). La sesión en curso
    no entra: el rango de hoy no mueve el stop de la apertura de hoy.
    Historia: ``n + 1`` filas y ``n + 1`` sesiones.
    """
    return wilder_atr(high, low, close, n=n).shift(1)


def session_vwap(
    high: Frame,
    low: Frame,
    close: Frame,
    volume: Frame,
    *,
    alpaca_vw: Optional[Frame] = None,
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """VWAP de sesión, reiniciado en cada apertura.

    ``VWAP_t = Σ_{i en la sesión, i<=t} (tp_i v_i) / Σ v_i``.

    Precio típico ``tp = (H + L + C) / 3``. Si ``alpaca_vw`` trae el campo
    ``vw`` de la barra y es positivo, esa barra usa ``vw`` en lugar de
    ``tp``. No son lo mismo: ``vw`` es el VWAP de los trades de la barra
    (``Σ price·size / Σ size``); ``tp`` reparte el rango ``H/L/C`` sin ver
    el tape. El acumulado ``Σ (vw·v) / Σ v`` recupera el VWAP de trades
    de la sesión cuando ``vw`` y ``v`` son los de esos trades. Una barra
    sin ``vw`` cae al precio típico. Volumen no positivo no suma. Una
    barra sin precio conserva el VWAP ya acumulado en la sesión; si todavía
    no hubo volumen, queda NaN. Historia: 1 barra de la sesión en curso.
    """
    high_f, squeeze = _box(high)
    low_f, _ = _box(low)
    close_f, _ = _box(close)
    volume_f, _ = _box(volume)
    _same_axes(high_f, low_f, "low")
    _same_axes(high_f, close_f, "close")
    _same_axes(high_f, volume_f, "volume")
    typical = (high_f + low_f + close_f) / 3.0
    if alpaca_vw is not None:
        vw, _ = _box(alpaca_vw)
        _same_axes(high_f, vw, "alpaca_vw")
        typical = vw.where(vw > 0, typical)
    labels = _labels(high_f.index, sessions)
    usable = typical.notna() & volume_f.notna() & (volume_f > 0)
    vol = volume_f.where(usable)
    traded = typical.where(usable) * vol
    num = _group_cumsum(traded, labels)
    den = _group_cumsum(vol, labels)
    ratio = num / den.where(den > 0)
    out = ratio.groupby(labels, sort=False).ffill()
    return _restore(out.reindex(high_f.index), squeeze)


def vwap_zscore(
    close: Frame,
    vwap: Frame,
    n: int = DEFAULT_Z_BARS,
    *,
    by_session: bool = True,
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """Desvío del cierre respecto del VWAP: ``z_t = (C_t − VWAP_t) / σ_dev(n)``.

    ``σ_dev`` es la desviación típica (``ddof=1``) de ``C − VWAP`` en las
    últimas ``n`` barras. Con ``by_session`` la ventana no cruza la
    apertura. ``σ = 0`` o VWAP ausente → NaN. Historia: ``n`` barras.
    """
    n = _positive_int("n", n, minimum=2)
    close_f, squeeze = _box(close)
    vwap_f, _ = _box(vwap)
    _same_axes(close_f, vwap_f, "vwap")
    dev = close_f - vwap_f
    sigma = dev.rolling(n, min_periods=n).std(ddof=1)
    if by_session:
        labels = _labels(close_f.index, sessions)
        sigma = sigma.where(_same_session(labels, n))
    out = dev / sigma.where(sigma > 0)
    return _restore(out, squeeze)


def opening_range(
    high: Frame,
    low: Frame,
    k_minutes: int = DEFAULT_OPENING_MINUTES,
    *,
    timeframe: str = "1Min",
    sessions: Optional[Sequence[Session]] = None,
) -> Tuple[Frame, Frame]:
    """Máximo y mínimo de los primeros ``k_minutes`` de la sesión.

    Solo se publican cuando el rango ya cerró: en la barra cuyo inicio es
    ``>= open + k_minutes − duración``, y en las siguientes. Antes, NaN.
    El máximo no se actualiza con barras posteriores al rango. Un minuto
    sin fila (hueco IEX) no se imputa; el rango usa las barras presentes.
    Historia: las barras del tramo, y 1 sesión.
    """
    k_minutes = _positive_int("k_minutes", k_minutes, minimum=1)
    bar_minutes = _bar_minutes(timeframe)
    if k_minutes < bar_minutes:
        raise ValueError("k_minutes debe cubrir al menos una barra")
    high_f, squeeze = _box(high)
    low_f, _ = _box(low)
    _same_axes(high_f, low_f, "low")
    labels, opens = _labels_and_opens(high_f.index, sessions)
    in_or, available = _opening_mask(high_f.index, opens, k_minutes, bar_minutes)
    hi = _hold_opening(high_f.where(in_or), labels, how="max")
    lo = _hold_opening(low_f.where(in_or), labels, how="min")
    hi = hi.where(available)
    lo = lo.where(available)
    return _restore(hi, squeeze), _restore(lo, squeeze)


def relative_volume(
    volume: Frame,
    k_minutes: int = DEFAULT_OPENING_MINUTES,
    sessions_back: int = DEFAULT_RVOL_SESSIONS,
    *,
    timeframe: str = "1Min",
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """RVOL del rango de apertura frente a las ``D`` sesiones previas.

    ``RVOL = V_rango(sesión t) / media(V_rango de las D sesiones anteriores)``.

    ``V_rango`` en ``t`` suma solo el volumen de las barras con inicio en
    ``[open, open + k_minutes)`` y marca ``<= t``. La media no incluye la
    sesión en curso: sale del total de sesiones ya terminadas. El ratio se
    publica cuando empieza la última barra del ``timeframe`` que todavía
    abre dentro del rango. Si la rejilla es más fina que ese timeframe, esa
    ventana sigue recibiendo impresiones y el ratio se actualiza; cuando ya
    no quedan barras del rango, queda fijo el resto de la sesión. Menos de
    ``D`` sesiones previas, o una media nula, es NaN. ``D`` cuenta sesiones
    presentes en el frame, no días de calendario. Historia: barras del
    tramo y ``D + 1`` sesiones.
    """
    k_minutes = _positive_int("k_minutes", k_minutes, minimum=1)
    sessions_back = _positive_int("sessions_back", sessions_back, minimum=1)
    bar_minutes = _bar_minutes(timeframe)
    if k_minutes < bar_minutes:
        raise ValueError("k_minutes debe cubrir al menos una barra")
    volume_f, squeeze = _box(volume)
    labels, opens = _labels_and_opens(volume_f.index, sessions)
    in_or, available = _opening_mask(volume_f.index, opens, k_minutes, bar_minutes)
    piece = volume_f.where(in_or)
    # Suma expansiva: la barra t no ve volumen del rango con timestamp > t.
    # El 0 solo anula las barras fuera del tramo; un volumen ausente dentro
    # del tramo no cuenta (``seen``).
    seen = piece.notna().groupby(labels, sort=False).cumsum()
    running = piece.fillna(0.0).groupby(labels, sort=False).cumsum().where(seen > 0)
    # La media de referencia usa el total de sesiones ya terminadas. El
    # shift las deja fuera de su propia sesión, así que el total completo
    # de una sesión pasada no es look-ahead en la sesión en curso.
    totals = piece.groupby(labels, sort=False).sum(min_count=1).sort_index()
    prior = totals.shift(1).rolling(sessions_back, min_periods=sessions_back).mean()
    prior_at = _broadcast(prior, labels, volume_f.index, volume_f.columns)
    out = running / prior_at.where(prior_at > 0)
    return _restore(out.where(available), squeeze)


def trend_score(
    close: Frame,
    ma_window: int = DEFAULT_TREND_MA,
    look: int = DEFAULT_TREND_LOOK,
    *,
    by_session: bool = True,
    sessions: Optional[Sequence[Session]] = None,
    clip: float = DEFAULT_CLIP,
) -> Frame:
    """Pendiente normalizada de la media móvil, en torno a ``[-clip, clip]``.

    ``MA`` es la media de ``ma_window`` cierres. La pendiente es la de la
    recta de mínimos cuadrados sobre los últimos ``look`` valores de esa
    media (eje 0..look−1). ``score = clip(pendiente / MA_t × 100, ±clip)``.
    Positivo es alza.

    No se llama a ``src.regime_detector._trend_score``. Esa función
    devuelve un escalar del tramo final, comprime huecos con ``dropna`` y,
    si falta historia, devuelve 0. Aquí la salida va alineada al índice,
    un hueco invalida la ventana y la historia corta es NaN. Con
    ``by_session=True`` la media no cruza la apertura (en una serie diaria
    de una fila por día hay que pasar ``by_session=False``). Historia:
    ``ma_window + look − 1`` barras.
    """
    ma_window = _positive_int("ma_window", ma_window, minimum=1)
    look = _positive_int("look", look, minimum=2)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip debe ser un número positivo")
    close_f, squeeze = _box(close)
    ma = close_f.rolling(ma_window, min_periods=ma_window).mean()
    if by_session:
        labels = _labels(close_f.index, sessions)
        ma = ma.where(_same_session(labels, ma_window))
    slope = _rolling_slope(ma, look)
    level = ma.where(ma.abs() > 0)
    raw = slope / level * 100.0
    score = raw.clip(lower=-float(clip), upper=float(clip))
    if by_session:
        score = score.where(_same_session(labels, ma_window + look - 1))
    return _restore(score, squeeze)


def roll_spread(price: Frame, n: int = DEFAULT_ROLL_BARS) -> Frame:
    """Spread de Roll (1984), en las mismas unidades que el precio.

    ``S_t = 2 √(−Cov(Δp_t, Δp_{t−1}))`` si la covarianza muestral
    (``ddof=1``) de la ventana de ``n`` cambios es negativa. Si es ``>= 0``
    el rebote no está identificado y el valor es NaN, no 0. La ventana
    puede cruzar la apertura: el salto overnight no es rebote bid-ask.
    Historia: ``n + 2`` filas (``n`` pares de cambios consecutivos).
    """
    n = _positive_int("n", n, minimum=2)
    frame, squeeze = _box(price)
    changes = frame.diff()
    cov = changes.rolling(n, min_periods=n).cov(changes.shift(1))
    out = 2.0 * np.sqrt(-(cov.where(cov < 0)))
    return _restore(out, squeeze)


def corwin_schultz_spread(high: Frame, low: Frame) -> Frame:
    """Spread de Corwin–Schultz (2012) con dos barras consecutivas.

    ``β = (ln H_t/L_t)² + (ln H_{t−1}/L_{t−1})²``,
    ``γ = (ln H*/L*)²`` con ``H* = max(H_t, H_{t−1})`` y
    ``L* = min(L_t, L_{t−1})``,
    ``α = (√(2β) − √β) / (3 − 2√2) − √(γ / (3 − 2√2))``,
    ``S = 2 (e^α − 1) / (1 + e^α)``.

    ``S`` es una fracción del precio, no un importe como el de Roll. El
    paper pone ``α`` negativo en 0; aquí un ``α`` negativo deja un spread
    negativo (el estimador no ve spread) en vez de escribir un 0. Historia
    insuficiente o OHLC inválido sigue siendo NaN. Historia: 2 barras.
    """
    high_f, squeeze = _box(high)
    low_f, _ = _box(low)
    _same_axes(high_f, low_f, "low")
    valid = (high_f > 0) & (low_f > 0) & (high_f >= low_f)
    log_hl_sq = (np.log(high_f / low_f) ** 2).where(valid)
    beta = log_hl_sq + log_hl_sq.shift(1)
    prev_h = high_f.shift(1)
    prev_l = low_f.shift(1)
    high_2 = high_f.combine(prev_h, np.maximum)
    low_2 = low_f.combine(prev_l, np.minimum)
    both = (valid & valid.shift(1)).fillna(False)
    gamma = (np.log(high_2 / low_2) ** 2).where(both & (low_2 > 0) & (high_2 >= low_2))
    den = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return _restore(spread.where(both), squeeze)


def amihud(
    close: Frame,
    volume: Frame,
    *,
    by_session: bool = True,
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """Amihud intradía por barra: ``|r_t| / (C_t · v_t)``.

    ``r_t = ln(C_t / C_{t−1})``, la misma razón que
    ``universe_filter.compute_rolling_amihud`` (log |r| sobre dólar de
    volumen), aquí sin promediar y sin importar ese módulo. Con
    ``by_session`` el retorno que cruza la apertura no entra. Precio o
    volumen no positivos → NaN. Historia: 2 barras.
    """
    close_f, squeeze = _box(close)
    volume_f, _ = _box(volume)
    _same_axes(close_f, volume_f, "volume")
    returns = log_return(close_f).abs()
    if by_session:
        labels = _labels(close_f.index, sessions)
        returns = returns.where(_same_session(labels, 2))
    dollar = (close_f * volume_f).where((close_f > 0) & (volume_f > 0))
    return _restore(returns / dollar, squeeze)


def amihud_session(
    close: Frame,
    volume: Frame,
    *,
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """Media expansiva, dentro de la sesión, del Amihud por barra.

    En ``t`` promedia las barras de la sesión con índice ``<= t`` y Amihud
    finito. Una barra sin retorno no entra en la media ni la pone a 0; si
    todavía no hay ninguna, el agregado es NaN. Historia: 2 barras de la
    sesión en curso.
    """
    bar = amihud(close, volume, by_session=True, sessions=sessions)
    frame, squeeze = _box(bar)
    frame = frame.where(np.isfinite(frame))
    labels = _labels(frame.index, sessions)
    # El 0 solo anula el aporte de una barra sin Amihud dentro de la suma.
    # El conteo no la incluye, así que la media publicada no es un 0 inventado.
    count = frame.notna().astype(float).groupby(labels, sort=False).cumsum()
    total = frame.fillna(0.0).groupby(labels, sort=False).cumsum()
    mean = total / count.where(count > 0)
    return _restore(mean.reindex(frame.index), squeeze)


def microprice(
    bid: Frame,
    ask: Frame,
    bid_size: Frame,
    ask_size: Frame,
) -> Frame:
    """Microprice (weighted mid): ``I·P_ask + (1−I)·P_bid``, ``I = Q_bid / (Q_bid + Q_ask)``.

    Más tamaño en el bid empuja el precio hacia el ask. Ask por debajo del
    bid, tamaño no positivo o precio no positivo → NaN. Ask igual al bid
    deja el propio precio. Historia: 1 snapshot.
    """
    bid_f, squeeze = _box(bid)
    ask_f, _ = _box(ask)
    bid_q, _ = _box(bid_size)
    ask_q, _ = _box(ask_size)
    _same_axes(bid_f, ask_f, "ask")
    _same_axes(bid_f, bid_q, "bid_size")
    _same_axes(bid_f, ask_q, "ask_size")
    ok = (bid_f > 0) & (ask_f > 0) & (ask_f >= bid_f) & (bid_q > 0) & (ask_q > 0)
    weight = bid_q / (bid_q + ask_q)
    mid = weight * ask_f + (1.0 - weight) * bid_f
    return _restore(mid.where(ok), squeeze)


def quoted_spread_bps(bid: Frame, ask: Frame) -> Frame:
    """Spread cotizado en bps: ``10_000 · (ask − bid) / mid``, ``mid = (ask + bid) / 2``.

    Ask igual al bid es 0. Ask por debajo del bid, o un precio no positivo,
    es NaN. Historia: 1 snapshot.
    """
    bid_f, squeeze = _box(bid)
    ask_f, _ = _box(ask)
    _same_axes(bid_f, ask_f, "ask")
    mid = (ask_f + bid_f) / 2.0
    ok = (bid_f > 0) & (ask_f >= bid_f) & (mid > 0)
    out = (ask_f - bid_f) / mid * 1.0e4
    return _restore(out.where(ok), squeeze)


def order_flow_imbalance(
    bid: Frame,
    ask: Frame,
    bid_size: Frame,
    ask_size: Frame,
) -> Frame:
    """OFI de Cont, Kukanov y Stoikov (2014) entre snapshots consecutivos.

    ``e_t = 1_{b_t ≥ b_{t−1}} q^b_t − 1_{b_t ≤ b_{t−1}} q^b_{t−1}
    − 1_{a_t ≤ a_{t−1}} q^a_t + 1_{a_t ≥ a_{t−1}} q^a_{t−1}``.

    Con un snapshot por barra, ``e_t`` es el flujo de esa barra respecto
    de la anterior. No suma los eventos dentro de la barra: HFT-04 no
    guarda el libro evento a evento. La primera fila, o un quote inválido,
    es NaN. Historia: 2 snapshots.
    """
    bid_f, squeeze = _box(bid)
    ask_f, _ = _box(ask)
    bid_q, _ = _box(bid_size)
    ask_q, _ = _box(ask_size)
    _same_axes(bid_f, ask_f, "ask")
    _same_axes(bid_f, bid_q, "bid_size")
    _same_axes(bid_f, ask_q, "ask_size")
    prev_b = bid_f.shift(1)
    prev_a = ask_f.shift(1)
    prev_bq = bid_q.shift(1)
    prev_aq = ask_q.shift(1)
    flow = (
        (bid_f >= prev_b).astype(float) * bid_q
        - (bid_f <= prev_b).astype(float) * prev_bq
        - (ask_f <= prev_a).astype(float) * ask_q
        + (ask_f >= prev_a).astype(float) * prev_aq
    )
    ok = (
        bid_f.notna() & ask_f.notna() & bid_q.notna() & ask_q.notna()
        & prev_b.notna() & prev_a.notna() & prev_bq.notna() & prev_aq.notna()
        & (bid_f > 0) & (ask_f > 0) & (ask_f >= bid_f)
        & (bid_q >= 0) & (ask_q >= 0)
    )
    return _restore(flow.where(ok), squeeze)


def lookback_log_return(
    open_: Frame,
    close: Frame,
    minutes: int = DEFAULT_LOOKBACK_MINUTES,
    *,
    timeframe: str = "1Min",
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """Retorno log de los ``minutes`` que terminan al cierre de la barra ``t``.

    ``r_t = ln(C_t / C_{t−minutes})``. La base es el cierre de la barra que
    empieza ``minutes`` antes, en la misma sesión. El tiempo es exacto: si
    esa barra falta, NaN; no se toma la fila anterior. Si ``t − minutes``
    cae antes de la apertura, la base es el open de la primera barra de la
    sesión y el tramo se acorta a ``t + barra − apertura``. Fuera de sesión
    o con un precio no positivo, NaN. Historia: ``minutes / barra + 1``
    barras y la sesión de ``t``.
    """
    bar_minutes = _bar_minutes(timeframe)
    minutes = _positive_int("minutes", minutes, minimum=1)
    _lookback_bars(minutes, bar_minutes)
    open_f, squeeze = _box(open_)
    close_f, _ = _box(close)
    _same_axes(close_f, open_f, "open_")
    index = _require_time(close_f.index)
    labels, opens = _labels_and_opens(index, sessions)
    span = pd.Timedelta(minutes, unit="min")
    starts = pd.Series(index.tz_convert("UTC"), index=index)
    inside = labels.notna()
    clipped = ((starts - span) < opens).fillna(False) & inside
    # ``shift(freq=)`` mueve el reloj, no las filas: un hueco queda en NaN.
    base_close = close_f.shift(freq=span).reindex(index)
    base_labels = labels.shift(freq=span).reindex(index)
    same = labels.eq(base_labels) & inside & ~clipped
    first = starts.eq(opens) & inside
    first_open = (
        open_f.where(first, axis=0)
        .groupby(labels, sort=False)
        .transform("max")
        .reindex(index)[close_f.columns]
    )
    base = np.where(
        same.to_numpy()[:, None],
        base_close.to_numpy(dtype=float),
        np.where(clipped.to_numpy()[:, None], first_open.to_numpy(dtype=float), np.nan),
    )
    base_f = pd.DataFrame(base, index=close_f.index, columns=close_f.columns)
    ratio = (close_f / base_f).where((close_f > 0) & (base_f > 0))
    usable = ratio.where(np.isfinite(ratio))
    return _restore(np.log(usable), squeeze)


def prior_sessions_vol(
    close: Frame,
    n_sessions: int = DEFAULT_SIGMA_SESSIONS,
    *,
    timeframe: str = "1Min",
    sessions: Optional[Sequence[Session]] = None,
) -> Frame:
    """σ (``ddof=1``) de los retornos log de una barra en las sesiones previas.

    Toda la sesión de ``t`` comparte el valor: lo fijan las ``n_sessions``
    sesiones anteriores y la sesión en curso no entra, como en
    ``prior_session_atr``. Un retorno es ``ln(C_s / C_{s−barra})`` con las
    dos barras en la misma sesión y a una barra exacta: el salto de la
    noche y un hueco no cuentan. Con ``sessions`` las previas son las del
    calendario y una sin barras deja NaN. Sin ``sessions`` son las fechas
    de Nueva York presentes en el frame. Cada sesión previa aporta al
    menos un retorno y el total al menos dos. Historia: la barra ``t`` y
    ``n_sessions + 1`` sesiones.
    """
    n = _positive_int("n_sessions", n_sessions, minimum=1)
    step = pd.Timedelta(_bar_minutes(timeframe), unit="min")
    frame, squeeze = _box(close)
    index = _require_time(frame.index)
    labels = _labels(index, sessions)
    prev = frame.shift(freq=step).reindex(index)
    prev_labels = labels.shift(freq=step).reindex(index)
    same = labels.eq(prev_labels) & labels.notna()
    ratio = (frame / prev).where((frame > 0) & (prev > 0))
    returns = np.log(ratio.where(np.isfinite(ratio))).where(same, axis=0)
    if sessions is None:
        order = np.sort(labels.dropna().unique())
    else:
        order = np.array(sorted(float(item.date.toordinal()) for item in sessions))
    count = returns.notna().groupby(labels).sum().reindex(order, fill_value=0)
    total = returns.groupby(labels).sum().reindex(order, fill_value=0.0)
    square = (returns ** 2).groupby(labels).sum().reindex(order, fill_value=0.0)
    # Ventana de n sesiones que termina en la anterior: shift(1) deja fuera la de hoy.
    prior_count = count.shift(1).rolling(n, min_periods=n).sum()
    prior_total = total.shift(1).rolling(n, min_periods=n).sum()
    prior_square = square.shift(1).rolling(n, min_periods=n).sum()
    present = (count > 0).astype(float).shift(1).rolling(n, min_periods=n).sum()
    var = (prior_square - prior_total ** 2 / prior_count) / (prior_count - 1.0)
    ok = present.eq(float(n)) & (prior_count >= 2)
    vol = np.sqrt(var.clip(lower=0.0)).where(ok)
    out = _broadcast(vol, labels, index, frame.columns)
    return _restore(out, squeeze)


def _lookback_bars(minutes: int, bar_minutes: int) -> int:
    if minutes % bar_minutes:
        raise ValueError(
            f"minutes={minutes} debe ser múltiplo de la barra ({bar_minutes} min)"
        )
    return minutes // bar_minutes


def _opening_bars(params: dict) -> int:
    k_minutes = _positive_int(
        "k_minutes", params.get("k_minutes", DEFAULT_OPENING_MINUTES), minimum=1,
    )
    bar_minutes = _bar_minutes(params.get("timeframe", "1Min"))
    if k_minutes < bar_minutes:
        raise ValueError("k_minutes debe cubrir al menos una barra")
    return math.ceil(k_minutes / bar_minutes)


def _positive_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} debe ser un entero >= {minimum}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}")
    return number


def _bar_minutes(timeframe: object) -> int:
    text = str(timeframe).strip()
    if text.endswith("Min") and text[:-3].isdigit():
        minutes = int(text[:-3])
        if 1 <= minutes < 60:
            return minutes
    raise ValueError(f"timeframe {timeframe!r} no es NMin")


def _box(obj: Frame) -> Tuple[pd.DataFrame, bool]:
    if isinstance(obj, pd.Series):
        return obj.to_frame(), True
    if isinstance(obj, pd.DataFrame):
        return obj, False
    raise TypeError("se espera una Series o un DataFrame")


def _restore(frame: pd.DataFrame, squeeze: bool) -> Frame:
    if squeeze:
        return frame.iloc[:, 0]
    return frame


def _same_axes(left: pd.DataFrame, right: pd.DataFrame, what: str) -> None:
    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError(f"{what} debe compartir índice y columnas")


def _require_time(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("las features de sesión exigen un DatetimeIndex")
    if index.tz is None:
        raise ValueError("el índice debe llevar zona horaria")
    if not index.is_monotonic_increasing:
        raise ValueError("el índice de tiempo debe ser creciente")
    if index.has_duplicates:
        raise ValueError("el índice de tiempo no puede repetir timestamps")
    return index


def _labels(index: pd.Index, sessions: Optional[Sequence[Session]]) -> pd.Series:
    labels, _ = _labels_and_opens(index, sessions)
    return labels


def _labels_and_opens(
    index: pd.Index,
    sessions: Optional[Sequence[Session]],
) -> Tuple[pd.Series, pd.Series]:
    dated = _require_time(index)
    utc = dated.tz_convert("UTC")
    if sessions is None:
        local = utc.tz_convert(NY)
        keys = local.year * 10000 + local.month * 100 + local.day
        labels = pd.Series(keys, index=dated)
        stamps = pd.Series(utc, index=dated)
        opens = stamps.groupby(labels, sort=False).transform("min")
        return labels, opens
    ordered = tuple(sorted(sessions, key=lambda item: item.open_utc))
    if not ordered:
        raise ValueError("sessions está vacío")
    for session in ordered:
        if not isinstance(session, Session):
            raise TypeError("sessions debe ser una secuencia de Session")
    opens_ns = np.array([pd.Timestamp(item.open_utc).value for item in ordered], dtype=np.int64)
    closes_ns = np.array([pd.Timestamp(item.close_utc).value for item in ordered], dtype=np.int64)
    ords = np.array([item.date.toordinal() for item in ordered], dtype=np.int64)
    # ``asi8`` va en la unidad del índice (pandas 3 infiere "us"); ``.value`` en ns.
    ts = utc.as_unit("ns").asi8
    pos = np.searchsorted(opens_ns, ts, side="right") - 1
    clipped = np.clip(pos, 0, len(ordered) - 1)
    inside = (pos >= 0) & (ts >= opens_ns[clipped]) & (ts < closes_ns[clipped])
    labels = pd.Series(np.where(inside, ords[clipped], np.nan), index=dated, dtype="float64")
    open_ns = np.where(inside, opens_ns[clipped], 0)
    opens = pd.Series(pd.to_datetime(open_ns, utc=True), index=dated).where(inside)
    return labels, opens


def _same_session(labels: pd.Series, span: int) -> pd.Series:
    """True en t si las ``span`` filas que terminan en t comparten etiqueta."""
    if span <= 1:
        return labels.notna()
    prev = labels.shift(span - 1)
    same = labels.eq(prev) & labels.notna() & prev.notna()
    return same.fillna(False)


def _group_cumsum(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    out = frame.groupby(labels, sort=False).cumsum()
    return out.reindex(frame.index)


def _opening_mask(
    index: pd.DatetimeIndex,
    opens: pd.Series,
    k_minutes: int,
    bar_minutes: int,
) -> Tuple[pd.Series, pd.Series]:
    starts = pd.Series(index.tz_convert("UTC"), index=index)
    delta = starts - opens
    # unit explícita: Timedelta(minutes=) construye una unidad genérica que
    # NumPy ya deprecó.
    zero = pd.Timedelta(0, unit="ns")
    width = pd.Timedelta(int(k_minutes), unit="min")
    ready_after = pd.Timedelta(int(k_minutes - bar_minutes), unit="min")
    in_range = ((delta >= zero) & (delta < width)).fillna(False)
    ready = (delta >= ready_after).fillna(False)
    return in_range, ready


def _hold_opening(masked: pd.DataFrame, labels: pd.Series, how: str) -> pd.DataFrame:
    grouped = masked.groupby(labels, sort=False)
    extreme = grouped.cummax() if how == "max" else grouped.cummin()
    # cummax deja NaN en las filas fuera del rango; el último extremo se
    # arrastra solo dentro de la misma sesión.
    held = extreme.groupby(labels, sort=False).ffill()
    return held.reindex(masked.index)


def _broadcast(
    by_session: pd.DataFrame,
    labels: pd.Series,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    data = {col: labels.map(by_session[col]) for col in columns}
    return pd.DataFrame(data, index=index)


def _true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    hl = high - low
    hc = (high - prev).abs()
    lc = (low - prev).abs()
    stacked = np.maximum(hl.to_numpy(dtype=float), np.maximum(hc.to_numpy(dtype=float), lc.to_numpy(dtype=float)))
    use_hl = prev.isna().to_numpy()
    values = np.where(use_hl, hl.to_numpy(dtype=float), stacked)
    out = pd.DataFrame(values, index=high.index, columns=high.columns)
    invalid = high.isna() | low.isna() | (high < low)
    return out.where(~invalid)


def _wilder_1d(tr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(tr.shape[0], np.nan, dtype=float)
    alpha = 1.0 / n
    keep = 1.0 - alpha
    i = 0
    size = tr.shape[0]
    while i < size:
        if not np.isfinite(tr[i]):
            i += 1
            continue
        end = i + n
        if end > size:
            break
        window = tr[i:end]
        if np.isfinite(window).all():
            prev = float(window.mean())
            out[end - 1] = prev
            j = end
            while j < size and np.isfinite(tr[j]):
                prev = keep * prev + alpha * float(tr[j])
                out[j] = prev
                j += 1
            i = j + 1
        else:
            i += 1
    return out


def _rolling_slope(values: pd.DataFrame, look: int) -> pd.DataFrame:
    """Pendiente OLS de una ventana fija. NaN si la ventana tiene un hueco."""
    x = np.arange(look, dtype=float)
    centered = x - x.mean()
    var_x = float(np.dot(centered, centered))
    arr = values.to_numpy(dtype=float)
    n_rows, n_cols = arr.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    if n_rows < look or var_x == 0.0:
        return pd.DataFrame(out, index=values.index, columns=values.columns)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    kernel = np.ones(look, dtype=float)
    for col in range(n_cols):
        corr = np.correlate(filled[:, col], centered, mode="valid")
        complete = np.convolve(finite[:, col].astype(float), kernel, mode="valid") == look
        slope = corr / var_x
        slope = np.where(complete, slope, np.nan)
        out[look - 1:, col] = slope
    return pd.DataFrame(out, index=values.index, columns=values.columns)
