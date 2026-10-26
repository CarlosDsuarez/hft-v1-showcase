"""Datos intradía (HFT-04): Alpaca, almacén CSV.gz, calidad y sintético.

El backtest y el vivo usan el mismo feed. El default es ``iex`` (plan
Basic). ``sip`` queda para una sensibilidad y, en ese plan, no se pide si
el extremo de la ventana cae en los últimos 15 minutos.

Ajuste. El ``BarStore`` guarda barras ``raw`` (inmutables, cacheables).
Las features de una sesión usan raw: el día en curso es idéntico en raw y
en ajustado. ATR y RVOL, que miran varios días, piden barras diarias
``adjustment=split`` (o ``all``) aparte. Ese cache vive en otro árbol, con
la fecha de descarga y la huella de las acciones corporativas; si la huella
cambia, se vuelve a bajar. Mezclar raw y ajustado en la misma ventana
desplaza stops y rangos, así que ``read`` no las junta.

Universo. ``universe_hft.csv`` (raíz del repo) trae los 10 ETF iniciales.
Para ampliarlo: una fila nueva con las columnas de
``src.instruments.REQUIRED_COLUMNS``, ``tradable_brokers=alpaca`` y el
``broker_symbol`` de Alpaca. El plan Basic limita el WebSocket a 30
símbolos; no pasar de ahí. No hace falta un diccionario de alias.
"""

from src.hft.data.bars import (
    DEFAULT_FEED,
    IntradayBars,
    IntradayMeta,
    IntradayProvider,
)

__all__ = [
    "DEFAULT_FEED",
    "IntradayBars",
    "IntradayMeta",
    "IntradayProvider",
]
