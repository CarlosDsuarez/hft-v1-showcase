"""
instruments.py — Maestro de instrumentos: un `id` interno por activo.

Tres espacios de nombres conviven: el `id` del sistema (columna ``ticker``
de los universos), el simbolo de Yahoo (``ICOLCAP.CL``) y el del broker
(``BRK.B`` en Alpaca). Todo el sistema usa `id`; solo los adaptadores
traducen con este maestro:

  - datos: `src.market_data.InstrumentProvider` (id ↔ yahoo_symbol).
  - broker: `src.brokers.alpaca_adapter.AlpacaClient` (id ↔ broker_symbol,
    y rechaza antes de enviar lo que no sea ruteable a Alpaca).

Fuente: CSV de universo con las columnas de `REQUIRED_COLUMNS`
(``universe_mixed.csv`` / ``universe_co.csv``; ``universe.csv`` es solo una
lista de ids y no basta como maestro). `tradable_brokers` admite varios
brokers separados por ";"; vacio = no ruteable a ningun broker.

Uso:
    from src.instruments import load_instruments
    master = load_instruments()          # <repo_root>/universe_mixed.csv
    master["ICOLCAP"].yahoo_symbol       # "ICOLCAP.CL"
    master["ICOLCAP"].routable("alpaca") # False
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import pandas as pd

from src.fx import is_currency_code
from src.settings import load_settings

DEFAULT_INSTRUMENTS_FILE = "universe_mixed.csv"
REQUIRED_COLUMNS = ("ticker", "yahoo_ticker", "broker_symbol", "market",
                    "currency", "tradable_brokers", "asset_type", "sector")
# Brokers con adaptador en src/brokers/ (un typo en el CSV falla al cargar).
KNOWN_BROKERS = ("alpaca",)
_NON_BLANK = ("ticker", "yahoo_ticker", "market", "asset_type", "sector")


@dataclass(frozen=True)
class Instrument:
    """Definicion de un activo; `id` es la unica clave interna."""

    id: str
    yahoo_symbol: str
    broker_symbol: Optional[str]
    market: str
    currency: str                       # ISO 4217 de cotizacion (moneda local)
    asset_type: str
    sector: str
    tradable_brokers: Tuple[str, ...] = ()

    def routable(self, broker: str) -> bool:
        """True si este instrumento puede enviarse a `broker`."""
        return broker in self.tradable_brokers


class InstrumentMaster(Mapping[str, Instrument]):
    """Mapping inmutable id → Instrument con busqueda inversa por simbolo."""

    def __init__(self, instruments: Iterable[Instrument]):
        self._by_id: Dict[str, Instrument] = {i.id: i for i in instruments}
        self._by_yahoo = {i.yahoo_symbol: i for i in self._by_id.values()}
        self._by_broker = {i.broker_symbol: i for i in self._by_id.values()
                           if i.broker_symbol}

    def __getitem__(self, instrument_id: str) -> Instrument:
        return self._by_id[instrument_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def by_yahoo_symbol(self, symbol: str) -> Optional[Instrument]:
        return self._by_yahoo.get(symbol)

    def by_broker_symbol(self, symbol: str) -> Optional[Instrument]:
        return self._by_broker.get(symbol)

    def currencies(self, ids: Optional[Iterable[str]] = None) -> Dict[str, str]:
        """id → moneda local (ids desconocidos se omiten)."""
        wanted = self._by_id if ids is None else ids
        return {i: self._by_id[i].currency for i in wanted if i in self._by_id}


def load_instruments(path: Optional[Union[str, Path]] = None) -> InstrumentMaster:
    """Carga y valida el maestro desde un CSV (o Excel) de universo.

    Args:
        path: Archivo de universo. Default: ``universe_mixed.csv`` en la raiz
            del repo (src/settings.py; nunca relativo al cwd).

    Raises:
        ValueError: faltan columnas de `REQUIRED_COLUMNS`, campos obligatorios
            vacios, moneda no ISO, broker desconocido, instrumento ruteable sin
            broker_symbol, o ticker / yahoo_ticker / broker_symbol repetidos.
    """
    p = Path(path) if path is not None else load_settings().universe_path(
        DEFAULT_INSTRUMENTS_FILE)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(p, dtype=str, keep_default_na=False)
    else:
        raw = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{p}: instrument master missing columns {missing} "
                         f"(required: {list(REQUIRED_COLUMNS)}).")
    frame = raw[list(REQUIRED_COLUMNS)].fillna("").astype(str).apply(
        lambda col: col.str.strip())

    errors: List[str] = []
    instruments: List[Instrument] = []
    for n, row in enumerate(frame.to_dict("records"), start=2):  # linea CSV
        where = f"line {n} ({row['ticker'] or '?'})"
        blank = [c for c in _NON_BLANK if not row[c]]
        if blank:
            errors.append(f"{where}: empty {blank}")
            continue
        if not is_currency_code(row["currency"]):
            errors.append(f"{where}: currency {row['currency']!r} is not an "
                          "ISO 4217 code (e.g. USD, COP)")
        brokers = tuple(b.strip().lower()
                        for b in row["tradable_brokers"].split(";") if b.strip())
        unknown = [b for b in brokers if b not in KNOWN_BROKERS]
        if unknown:
            errors.append(f"{where}: unknown broker(s) {unknown} "
                          f"(known: {list(KNOWN_BROKERS)})")
        broker_symbol = row["broker_symbol"] or None
        if brokers and broker_symbol is None:
            errors.append(f"{where}: tradable on {list(brokers)} but "
                          "broker_symbol is empty")
        instruments.append(Instrument(
            id=row["ticker"], yahoo_symbol=row["yahoo_ticker"],
            broker_symbol=broker_symbol, market=row["market"].lower(),
            currency=row["currency"], asset_type=row["asset_type"],
            sector=row["sector"], tradable_brokers=brokers,
        ))

    for label, values in (
        ("ticker", [i.id for i in instruments]),
        ("yahoo_ticker", [i.yahoo_symbol for i in instruments]),
        ("broker_symbol", [i.broker_symbol for i in instruments if i.broker_symbol]),
    ):
        dupes = sorted(v for v, n in Counter(values).items() if n > 1)
        if dupes:
            errors.append(f"duplicate {label}: {dupes}")
    if errors:
        raise ValueError(f"{p}: invalid instrument master:\n  - "
                         + "\n  - ".join(errors))
    return InstrumentMaster(instruments)
