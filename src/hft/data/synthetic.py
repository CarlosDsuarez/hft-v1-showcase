"""Sesiones sintéticas sembradas, para tests y demos sin red.

Por barra, en el log-precio ``x``::

    x_t = x_{t-1} + μ + κ (θ − x_{t-1}) + σ Z_t + saltos

``Z_t`` es normal estándar (``np.random.default_rng(seed)``, sin tocar el
RNG global). El cierre es ``S0 · exp(x_t)``. Con ``κ = 0`` es un paseo
geométrico de deriva ``μ`` y volatilidad ``σ`` por barra. Con ``κ > 0``
hay un término Ornstein–Uhlenbeck hacia ``θ``: un test de reversión puede
usar un edge conocido (``σ = 0`` decae en línea recta hacia ``θ``).

El volumen sigue una U intradía (más alto en la apertura y en el cierre).
``gap_prob`` omite minutos al azar, como IEX cuando no hay trades. Los
saltos suman ``±jump_scale`` al retorno de la barra. El high/low se arman
alrededor de open y close para que el OHLC sea válido.

Solo se emiten inicios de ``expected_bars``: el generador no inventa
minutos fuera de la sesión.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.hft.calendar import Session, expected_bars
from src.hft.data.bars import (
    IntradayBars,
    bars_from_records,
    canonical_timeframe,
)
from src.settings import SEED


def synthetic_bars(
    sessions: Sequence[Session],
    symbols: Sequence[str],
    *,
    timeframe: str = "1Min",
    seed: int = SEED,
    start_price: float = 100.0,
    x0: float = 0.0,
    mu: float = 0.0,
    sigma: float = 0.0008,
    ou_kappa: float = 0.0,
    ou_theta: float = 0.0,
    jump_prob: float = 0.0,
    jump_scale: float = 0.0,
    gap_prob: float = 0.0,
    feed: str = "synthetic",
) -> IntradayBars:
    """Un bloque ``IntradayBars`` para las sesiones, en orden.

    Cada símbolo sale de una semilla hija, así que cambiar la lista no
    reordena el camino de los anteriores.
    """
    if start_price <= 0:
        raise ValueError("start_price debe ser positivo")
    if sigma < 0 or gap_prob < 0 or gap_prob > 1 or jump_prob < 0 or jump_prob > 1:
        raise ValueError("sigma, gap_prob y jump_prob fuera de rango")
    if ou_kappa < 0:
        raise ValueError("ou_kappa no puede ser negativo")
    step = canonical_timeframe(timeframe)
    names = tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    if not names:
        raise ValueError("symbols vacío")
    children = np.random.SeedSequence(seed).spawn(len(names))
    rows: list[dict] = []
    for symbol, child in zip(names, children):
        rng = np.random.default_rng(child)
        rows.extend(
            _symbol_rows(
                rng,
                symbol,
                sessions,
                step,
                start_price=start_price,
                x0=x0,
                mu=mu,
                sigma=sigma,
                ou_kappa=ou_kappa,
                ou_theta=ou_theta,
                jump_prob=jump_prob,
                jump_scale=jump_scale,
                gap_prob=gap_prob,
            )
        )
    records = pd.DataFrame(rows)
    return bars_from_records(
        records,
        names,
        source="synthetic",
        feed=feed,
        timeframe=step,
        adjustment="raw",
    )


class FakeIntradayProvider:
    """Proveedor en memoria. Recorta al ventana pedida y anota las llamadas."""

    name = "fake"

    def __init__(self, bars: IntradayBars, *, timeframe: Optional[str] = None):
        self.bars = bars
        self.timeframe = canonical_timeframe(timeframe or bars.meta.timeframe)
        self.calls: list[tuple[tuple[str, ...], datetime, datetime, str]] = []

    def get_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> IntradayBars:
        step = canonical_timeframe(timeframe)
        if step != self.timeframe:
            raise ValueError(f"timeframe {step} no coincide con {self.timeframe}")
        wanted = tuple(str(symbol).strip().upper() for symbol in symbols)
        self.calls.append((wanted, start, end, step))
        start_utc = pd.Timestamp(start)
        end_utc = pd.Timestamp(end)
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start y end deben llevar zona")
        index = self.bars.close.index
        mask = (index >= start_utc) & (index < end_utc)
        frames = {}
        for name in ("open", "high", "low", "close", "volume", "vwap", "trades"):
            frames[name] = getattr(self.bars, name).loc[mask].reindex(columns=list(wanted))
        received = tuple(
            symbol
            for symbol in wanted
            if symbol in frames["close"].columns and frames["close"][symbol].notna().any()
        )
        meta = type(self.bars.meta)(
            source=self.bars.meta.source,
            feed=self.bars.meta.feed,
            timeframe=step,
            adjustment=self.bars.meta.adjustment,
            requested=wanted,
            received=received,
        )
        return type(self.bars)(meta=meta, **frames)


def _symbol_rows(
    rng: np.random.Generator,
    symbol: str,
    sessions: Sequence[Session],
    timeframe: str,
    *,
    start_price: float,
    x0: float,
    mu: float,
    sigma: float,
    ou_kappa: float,
    ou_theta: float,
    jump_prob: float,
    jump_scale: float,
    gap_prob: float,
) -> list[dict]:
    rows: list[dict] = []
    x_prev = float(x0)
    prev_close = start_price * float(np.exp(x_prev))
    for session in sessions:
        index = expected_bars(session, timeframe)
        n = len(index)
        if n == 0:
            continue
        z = rng.normal(0.0, 1.0, n)
        opens = np.empty(n)
        closes = np.empty(n)
        highs = np.empty(n)
        lows = np.empty(n)
        for i in range(n):
            shock = mu + ou_kappa * (ou_theta - x_prev) + sigma * float(z[i])
            if jump_prob and float(rng.random()) < jump_prob:
                sign = 1.0 if float(rng.random()) < 0.5 else -1.0
                shock += jump_scale * sign
            x_prev = x_prev + shock
            close = start_price * float(np.exp(x_prev))
            open_ = prev_close
            wobble = abs(float(rng.normal(0.0, max(sigma, 1e-8) * 0.25)))
            high = max(open_, close) * (1.0 + wobble)
            low = min(open_, close) * (1.0 - wobble)
            opens[i] = open_
            closes[i] = close
            highs[i] = high
            lows[i] = low
            prev_close = close
        frac = np.linspace(0.0, 1.0, n)
        shape = 1.0 + 4.0 * (frac - 0.5) ** 2
        base = float(rng.uniform(1.0e4, 5.0e4))
        volume = base * shape * rng.lognormal(0.0, 0.05, n)
        keep = rng.random(n) >= gap_prob
        for i in range(n):
            if not keep[i]:
                continue
            typical = (highs[i] + lows[i] + closes[i]) / 3.0
            rows.append(
                {
                    "symbol": symbol,
                    "ts": index[i],
                    "o": float(opens[i]),
                    "h": float(highs[i]),
                    "l": float(lows[i]),
                    "c": float(closes[i]),
                    "v": float(volume[i]),
                    "vw": float(typical),
                    "n": float(max(1, int(volume[i] // 100))),
                }
            )
    return rows
