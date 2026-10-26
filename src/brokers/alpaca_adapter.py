"""
alpaca_adapter.py — Paso 7: broker Alpaca SOLO-PAPER (stdlib, sin SDK).

Lee credenciales UNICAMENTE de entorno / `.env` (nunca hardcodeadas).
Seguridad:
  - Solo https://paper-api.alpaca.markets (host exacto, no substring);
    cualquier otra URL se rechaza fail-fast, sin HTTP.
  - Kill-switch (`<kill_dir>/KILL` o HFT_KILL=1): bloquea todo envio, se
    comprueba antes de cada POST (ver src/brokers/pre_trade.py).
  - `dry_run=True` por defecto: planifica sin enviar nada.
  - El secreto jamas se loguea ni aparece en excepciones.

Endpoints usados: /v2/account, /v2/clock, /v2/calendar, /v2/positions,
/v2/orders (POST, GET, DELETE, PATCH /{id}), /v2/orders:by_client_order_id,
/v2/assets/{symbol}, /v2/account/activities.

Idempotencia: cada orden lleva `client_order_id` determinista y solo los GET
se reintentan; un POST ambiguo (timeout/5xx) se consulta antes de reenviar.

Simbolos: el sistema usa el `id` del maestro de instrumentos
(src/instruments.py); este adaptador lo traduce a `broker_symbol` (BRK-B →
BRK.B) y de vuelta al leer posiciones. Un id fuera del maestro o sin
"alpaca" en `tradable_brokers` (p. ej. acciones BVC en COP) se rechaza ANTES
de enviar: queda en el plan con status "not_routable", sin HTTP.

Uso:
    from src.brokers.alpaca_adapter import load_broker_config, AlpacaClient
    client = AlpacaClient(load_broker_config())  # lee .env de la raiz
    print(client.check())                        # cuenta + reloj + buying power
    plan = client.submit_orders(trades_df)       # dry-run por defecto

CLI:
    python -m src.brokers.alpaca_adapter --check
    python -m src.brokers.alpaca_adapter --positions
    python -m src.brokers.alpaca_adapter --close NEE            # dry-run
    python -m src.brokers.alpaca_adapter --close NEE --confirm  # vende NEE
    python -m src.brokers.alpaca_adapter --close-all --confirm  # liquida todo
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse

import pandas as pd

from src.brokers.http_transport import PooledTransport, UrllibTransport
from src.brokers.pre_trade import PreTradeBlocked, kill_switch_reason
from src.brokers.rate_budget import TradingBudget
from src.hft.types import canonical_qty
from src.instruments import InstrumentMaster, load_instruments
from src.position_ledger import MissingPriceError

logger = logging.getLogger("alpaca_adapter")

BROKER_NAME = "alpaca"  # valor en `tradable_brokers` del maestro
_PAPER_HOST = "paper-api.alpaca.markets"
_DEFAULT_TIMEOUT = 15
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_MAX_CLIENT_ORDER_ID = 48
# Estados sin mas cambios (done_for_day: no recibe fills hasta el dia siguiente).
_FINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected",
                             "replaced", "done_for_day"})

def make_client_order_id(run_id: str, ticker: str, side: str) -> str:
    """Id idempotente `{run_id}-{ticker}-{side}` (Alpaca lo exige unico).

    Determinista: reintentar/relanzar la misma corrida produce el mismo id,
    asi Alpaca rechaza el duplicado en vez de ejecutar dos veces.

    Raises:
        ValueError: supera _MAX_CLIENT_ORDER_ID caracteres.
    """
    cid = f"{run_id}-{ticker}-{side}"
    if len(cid) > _MAX_CLIENT_ORDER_ID:
        raise ValueError(
            f"client_order_id '{cid}' supera {_MAX_CLIENT_ORDER_ID} caracteres "
            "(acortar run_id)."
        )
    return cid


# Tipos, TIF y clases que este adaptador envia. Verificados en
# https://docs.alpaca.markets/docs/orders-at-alpaca (2026-10-05):
# type market|limit|stop|stop_limit; time_in_force day|ioc|cls entre
# los que el intradía usa; order_class simple|bracket|oto|oco.
_ORDER_TYPES = ("market", "limit", "stop", "stop_limit")
_ORDER_TIFS = ("day", "ioc", "cls")
_ORDER_CLASSES = ("simple", "bracket", "oto", "oco")
# GET /v2/orders. status y limit salen de
# https://docs.alpaca.markets/docs/working-with-orders ; el tope 500
# es el del trading API (por encima Alpaca rechaza la consulta).
_LIST_STATUSES = ("open", "closed", "all")
_MAX_LIST_LIMIT = 500
# GET /v2/account/activities: page_size maximo 100 (OpenAPI, 2026-10-05).
_ACTIVITY_PAGE_MAX = 100
_ACTIVITY_PAGES = 50
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class IntradayOrderError(ValueError):
    """La orden no se envia: la validacion fallo antes de cualquier HTTP.

    ``needs_asset`` es True solo cuando falta la ficha de un corto: el
    llamador puede hacer ``get_asset`` y reintentar. El resto de los
    rechazos no se arreglan con un GET.
    """

    def __init__(self, message: str, *, needs_asset: bool = False):
        super().__init__(message)
        self.needs_asset = needs_asset


def _id_token(value: object, what: str) -> str:
    text = _ID_UNSAFE.sub("", str(value).strip())
    if not text:
        raise ValueError(f"{what} vacio para client_order_id")
    return text


def make_intent_order_id(
    strategy_id: str,
    symbol: str,
    signal_ts: datetime,
    leg: str,
    seq: int = 0,
) -> str:
    """Id por intencion, el mismo entre reinicios.

    Forma corta ``h-<estrategia>-<YYYYMMDDHHMM>-<symbol>-<leg>`` y, si
    ``seq`` > 0, ``-<seq>``. Si ``signal_ts`` trae segundos (barras de
    menos de un minuto), el sello es ``YYYYMMDDHHMMSS``: dos barras del
    mismo minuto no comparten id, y el 422 de duplicado no se traga la
    segunda. Con barras de minuto el id no cambia. El OpenAPI de ``POST /v2/orders`` (2026-10-05)
    declara ``client_order_id`` con maxLength 128. Este adaptador
    conserva el tope de 48 de ``make_client_order_id``: si la forma
    corta no cabe, el id es un prefijo mas ``sha1(texto)[:10]``. El
    hash no es un secreto; solo hace corto y determinista el id, y
    Alpaca rechaza el duplicado.
    """
    if not isinstance(signal_ts, datetime):
        raise TypeError("signal_ts debe ser datetime")
    if signal_ts.tzinfo is None:
        raise ValueError("signal_ts debe llevar zona horaria")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("seq debe ser un entero >= 0")
    moment = signal_ts.astimezone(timezone.utc)
    stamp = moment.strftime("%Y%m%d%H%M%S" if moment.second else "%Y%m%d%H%M")
    body = "h-{strat}-{stamp}-{symbol}-{leg}".format(
        strat=_id_token(strategy_id, "strategy_id"),
        stamp=stamp,
        symbol=_id_token(str(symbol).upper(), "symbol"),
        leg=_id_token(leg, "leg"),
    )
    if seq:
        body = f"{body}-{seq}"
    if len(body) <= _MAX_CLIENT_ORDER_ID:
        return body
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
    room = _MAX_CLIENT_ORDER_ID - len(digest) - 1
    return f"{body[:room]}-{digest}"


def format_tick_price(price: float) -> str:
    """Precio al incremento que Alpaca acepta en limit y stop.

    Doc ``orders-at-alpaca`` (sub-penny, 2026-10-05): precio >= 1 USD,
    maximo 2 decimales; por debajo de 1 USD, maximo 4. ``290.123`` se
    rechaza con codigo 42210000. La enmienda de 2024 a la Regla 612
    (tick de medio centavo en ciertos nombres) no es lo que esta API
    acepta: no se envia un sub-penny por encima de 1 USD.
    """
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise TypeError(f"precio no numerico: {price!r}")
    if not math.isfinite(float(price)) or float(price) <= 0:
        raise ValueError(f"precio debe ser finito y > 0, llego {price!r}")
    value = Decimal(str(price))
    tick = Decimal("0.01") if value >= 1 else Decimal("0.0001")
    rounded = value.quantize(tick, rounding=ROUND_HALF_UP)
    if rounded >= 1 and tick == Decimal("0.0001"):
        rounded = rounded.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded <= 0:
        raise ValueError("precio redondeado no positivo")
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def whole_shares(qty: float) -> bool:
    """True si ``qty`` es un entero de acciones >= 1 (tolera ruido binario)."""
    if isinstance(qty, bool) or not isinstance(qty, (int, float)):
        return False
    if not math.isfinite(float(qty)) or float(qty) <= 0:
        return False
    nearest = round(float(qty))
    return abs(float(qty) - nearest) <= 1e-9 and nearest >= 1


def format_qty(qty: float) -> str:
    """Cantidad como la pide ``POST /v2/orders`` (string).

    Misma regla que el sizer (``canonical_qty``): lo que sale es la
    cantidad de la intención, no un redondeo propio.
    """
    if isinstance(qty, bool) or not isinstance(qty, (int, float)):
        raise TypeError("qty no numerica")
    if not math.isfinite(float(qty)) or float(qty) <= 0:
        raise ValueError("qty debe ser > 0")
    if whole_shares(qty):
        return str(int(round(float(qty))))
    return f"{canonical_qty(qty):.9f}".rstrip("0").rstrip(".")


def asset_flags(raw: Mapping) -> Dict[str, object]:
    """``tradable``, ``shortable``, ``easy_to_borrow``, ``fractionable``.

    ``GET /v2/assets/{symbol}`` (2026-10-05) exige ``tradable``,
    ``shortable`` y ``fractionable``. ``easy_to_borrow`` sigue en el
    ejemplo; el campo de properties es ``borrow_status``
    (``easy_to_borrow`` | ``hard_to_borrow``). Si falta el booleano se
    usa ``borrow_status``. Si faltan los dos, no se considera facil
    de tomar prestado y el corto se rechaza.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("asset debe ser un mapping")
    if "easy_to_borrow" in raw:
        easy = bool(raw.get("easy_to_borrow"))
    elif raw.get("borrow_status") == "easy_to_borrow":
        easy = True
    else:
        easy = False
    return {
        "tradable": bool(raw.get("tradable")),
        "shortable": bool(raw.get("shortable")),
        "easy_to_borrow": easy,
        "fractionable": bool(raw.get("fractionable")),
        "symbol": raw.get("symbol"),
    }


@dataclass(frozen=True)
class OrderSpec:
    """Orden intradía lista para ``POST /v2/orders``. Sin HTTP.

    ``instrument_id`` es el id del maestro. ``order_type`` se serializa
    como ``type``. ``extended_hours`` por defecto es False y un True
    se rechaza: este adaptador no opera fuera de la sesion regular.
    ``short`` lo pone quien conoce la posicion (una venta que abre o
    agranda un corto).
    """

    instrument_id: str
    side: str
    qty: float
    order_type: str
    time_in_force: str = "day"
    order_class: str = "simple"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_limit_price: Optional[float] = None
    stop_loss_stop_price: Optional[float] = None
    client_order_id: str = ""
    extended_hours: bool = False
    short: bool = False


def coerce_spec(spec: object) -> OrderSpec:
    """``OrderSpec`` o un mapping. La clave ``type`` equivale a ``order_type``."""
    if isinstance(spec, OrderSpec):
        return spec
    if isinstance(spec, Mapping):
        data = dict(spec)
        if "type" in data and "order_type" not in data:
            data["order_type"] = data.pop("type")
        known = {item.name for item in fields(OrderSpec)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"campos desconocidos en spec: {sorted(unknown)}")
        return OrderSpec(**data)
    raise TypeError("spec debe ser OrderSpec o un mapping")


def _reference_price(spec: OrderSpec, mark: Optional[float]) -> Optional[float]:
    """Precio para el nocional: limite si hay, si no la marca, si no el stop."""
    if spec.order_type in ("limit", "stop_limit", "marketable_limit") and spec.limit_price:
        return float(spec.limit_price)
    if mark is not None and math.isfinite(float(mark)) and float(mark) > 0:
        return float(mark)
    if spec.stop_price:
        return float(spec.stop_price)
    if spec.limit_price:
        return float(spec.limit_price)
    return None


def build_order_payload(
    spec: OrderSpec,
    *,
    broker_symbol: str,
    asset: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """JSON de ``POST /v2/orders``. No abre red.

    Fraccional solo ``market`` o ``limit`` con ``day`` (HFT-11). El
    texto del OpenAPI dice "fractionable for only market and day"; la
    tabla HTML de fraccionales marca DAY en los cuatro tipos. Se
    aplica la regla mas estricta del prompt. Bracket, oto, oco y
    corto exigen acciones enteras. El corto ademas exige
    ``shortable`` y ``easy_to_borrow`` en la ficha (HFT-08); si no
    hay ficha, no se envia.
    """
    if not isinstance(spec, OrderSpec):
        raise TypeError("spec debe ser OrderSpec")
    if not broker_symbol or not str(broker_symbol).strip():
        raise IntradayOrderError("broker_symbol vacio")
    if not isinstance(spec.extended_hours, bool) or not isinstance(spec.short, bool):
        raise TypeError("extended_hours y short deben ser bool")
    if spec.extended_hours:
        raise IntradayOrderError("horario extendido deshabilitado")
    side = str(spec.side).strip().lower()
    order_type = str(spec.order_type).strip().lower()
    tif = str(spec.time_in_force).strip().lower()
    order_class = str(spec.order_class).strip().lower()
    if side not in ("buy", "sell"):
        raise IntradayOrderError(f"side {spec.side!r} no es buy ni sell")
    if order_type not in _ORDER_TYPES:
        raise IntradayOrderError(f"type {spec.order_type!r} no esta en {_ORDER_TYPES}")
    if tif not in _ORDER_TIFS:
        raise IntradayOrderError(f"time_in_force {spec.time_in_force!r} no esta en {_ORDER_TIFS}")
    if order_class not in _ORDER_CLASSES:
        raise IntradayOrderError(f"order_class {spec.order_class!r} no esta en {_ORDER_CLASSES}")
    cid = str(spec.client_order_id).strip()
    if not cid or len(cid) > _MAX_CLIENT_ORDER_ID:
        raise IntradayOrderError(
            f"client_order_id vacio o mayor de {_MAX_CLIENT_ORDER_ID}"
        )
    if isinstance(spec.qty, bool) or not isinstance(spec.qty, (int, float)):
        raise IntradayOrderError("qty no numerica")
    qty = float(spec.qty)
    if not math.isfinite(qty) or qty <= 0:
        raise IntradayOrderError("qty debe ser finita y > 0")
    fractional = not whole_shares(qty)
    advanced = order_class in ("bracket", "oto", "oco")
    if fractional and advanced:
        raise IntradayOrderError(
            "fraccional con bracket/oto/oco: esas clases exigen acciones enteras"
        )
    if fractional and (order_type not in ("market", "limit") or tif != "day"):
        raise IntradayOrderError(
            "fraccional solo market o limit con time_in_force day"
        )
    if spec.short and not whole_shares(qty):
        raise IntradayOrderError("el corto exige acciones enteras")
    if order_type in ("stop", "stop_limit") and tif in ("ioc", "cls"):
        raise IntradayOrderError(
            f"{order_type} no admite time_in_force {tif} "
            "(tabla whole-qty de orders-at-alpaca)"
        )
    if advanced and tif != "day":
        raise IntradayOrderError(
            f"{order_class} exige time_in_force day "
            "(la doc admite day o gtc; este adaptador no envia gtc)"
        )
    if order_class == "oco" and order_type != "limit":
        raise IntradayOrderError("oco exige type limit")
    if order_type in ("limit", "stop_limit") and spec.limit_price is None:
        raise IntradayOrderError(f"{order_type} exige limit_price")
    if order_type in ("stop", "stop_limit") and spec.stop_price is None:
        raise IntradayOrderError(f"{order_type} exige stop_price")
    take = spec.take_profit_limit_price
    stop_loss = spec.stop_loss_stop_price
    if order_class == "bracket" and (take is None or stop_loss is None):
        raise IntradayOrderError(
            "bracket exige take_profit.limit_price y stop_loss.stop_price"
        )
    if order_class == "oto" and (take is None) == (stop_loss is None):
        raise IntradayOrderError("oto exige take_profit o stop_loss, no los dos ni ninguno")
    if order_class == "oco" and (take is None or stop_loss is None):
        raise IntradayOrderError("oco exige take_profit y stop_loss")
    if order_class == "simple" and (take is not None or stop_loss is not None):
        raise IntradayOrderError("simple no lleva take_profit ni stop_loss")
    if take is not None and stop_loss is not None:
        if side == "buy" and not float(take) > float(stop_loss):
            raise IntradayOrderError("en compra el take debe quedar por encima del stop")
        if side == "sell" and not float(take) < float(stop_loss):
            raise IntradayOrderError("en venta el take debe quedar por debajo del stop")
    flags = asset_flags(asset) if asset is not None else None
    if spec.short:
        if flags is None:
            raise IntradayOrderError(
                "corto sin ficha de activo (get_asset); no se envia",
                needs_asset=True,
            )
        if not flags["shortable"]:
            raise IntradayOrderError("corto rechazado: el activo no es shortable")
        if not flags["easy_to_borrow"]:
            raise IntradayOrderError(
                "corto rechazado: el activo no es easy_to_borrow"
            )
    if flags is not None:
        if not flags["tradable"]:
            raise IntradayOrderError("el activo no es tradable en Alpaca")
        if fractional and not flags["fractionable"]:
            raise IntradayOrderError("el activo no es fractionable")
    payload: Dict[str, Any] = {
        "symbol": str(broker_symbol),
        "qty": format_qty(qty),
        "side": side,
        "type": order_type,
        "time_in_force": tif,
        "client_order_id": cid,
        "extended_hours": False,
    }
    if order_class != "simple":
        payload["order_class"] = order_class
    if order_type in ("limit", "stop_limit"):
        payload["limit_price"] = format_tick_price(float(spec.limit_price))
    if order_type in ("stop", "stop_limit"):
        payload["stop_price"] = format_tick_price(float(spec.stop_price))
    if take is not None:
        payload["take_profit"] = {"limit_price": format_tick_price(float(take))}
    if stop_loss is not None:
        payload["stop_loss"] = {"stop_price": format_tick_price(float(stop_loss))}
    return payload


def _rfc3339(value: object) -> str:
    """Instante UTC sin microsegundos, o el texto que ya vino armado."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("after sin zona horaria")
        stamp = value.astimezone(timezone.utc).replace(microsecond=0)
        return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(value).strip()
    if not text:
        raise ValueError("after vacio")
    return text


def _activity_day(value: object) -> str:
    """YYYY-MM-DD para ``date`` de activities. Un datetime usa su fecha de pared."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("date de activities sin zona horaria")
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return _calendar_day(value)


def _remember(journal: Optional[List[Dict]], record: Dict) -> None:
    if journal is not None:
        journal.append(record)


# ===================================================================
# Config (solo env / .env)
# ===================================================================
@dataclass
class BrokerConfig:
    """Credenciales Alpaca paper (secretos solo en memoria)."""

    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    account_id: str = ""  # ej. PA0000000001; si se fija, se verifica

    def __repr__(self) -> str:  # nunca exponer el secreto
        masked = self.api_key[:4] + "…" if self.api_key else "(vacia)"
        return (f"BrokerConfig(api_key={masked}, base_url={self.base_url}, "
                f"account_id={self.account_id or '(sin fijar)'})")


@dataclass(frozen=True)
class GatewayConfig:
    """Cómo habla el cliente con la Trading API. El default es lo de siempre.

    - ``pooled``: conexión keep-alive por hilo (``PooledTransport``) en
      vez de ``urlopen`` por petición.
    - ``workers``: POST en vuelo a la vez desde ``submit_many``. 1 es en
      serie, como antes.
    - ``trading_rpm``: cupo local de la Trading API por minuto
      (``TradingBudget``). None: sin cupo local, y un 429 se trata como
      antes (fallo transitorio).
    - ``reserve_frac``: fracción del cupo que una entrada no consume.
    """

    pooled: bool = False
    workers: int = 1
    trading_rpm: Optional[float] = None
    reserve_frac: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.pooled, bool):
            raise TypeError("pooled debe ser bool")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers debe ser un entero >= 1")
        if self.trading_rpm is not None:
            if isinstance(self.trading_rpm, bool) or not isinstance(self.trading_rpm, (int, float)):
                raise TypeError("trading_rpm debe ser numérico o None")
            if self.trading_rpm <= 0:
                raise ValueError("trading_rpm debe ser positivo")


def _load_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """Lee `.env` (default: raiz del repo, nunca el cwd) sin mutar os.environ."""
    from src import settings
    values = settings.read_dotenv(path)
    if values:
        logger.info("Variables leidas desde %s", path or settings.REPO_ROOT / ".env")
    else:
        logger.info("Sin archivo .env; usando entorno del sistema.")
    return values


def _validate_paper_url(base_url: str) -> None:
    """Fail-fast (sin HTTP) si base_url no es https://paper-api.alpaca.markets.

    Host exacto, no substring: una URL live con "paper" en el path, un
    subdominio parecido o userinfo@host se rechazan.
    """
    parsed = urlparse(base_url)
    if (parsed.scheme != "https"
            or parsed.netloc.lower() not in (_PAPER_HOST, f"{_PAPER_HOST}:443")
            or parsed.path not in ("", "/")
            or parsed.params or parsed.query or parsed.fragment):
        raise ValueError(
            f"URL rechazada (solo-paper): {base_url}. Solo se admite "
            f"https://{_PAPER_HOST}; este adaptador jamas opera en live."
        )


def load_broker_config(
    env: Optional[Dict[str, str]] = None,
    dotenv_path: Optional[str] = None,
) -> BrokerConfig:
    """Construye config desde entorno (+ `.env` de la raiz como fallback).

    No muta os.environ: el `.env` se lee a un dict y el entorno real gana.

    Raises:
        ValueError: faltan keys o la URL no es paper.
    """
    # Prioridad: env explicito > entorno real > `.env` de la raiz del repo.
    source = env if env is not None else {**_load_dotenv(dotenv_path), **os.environ}
    api_key = (source.get("BROKER_API_KEY") or "").strip()
    secret = (source.get("BROKER_SECRET_KEY") or "").strip()
    account_id = (source.get("BROKER_ACCOUNT_ID") or "").strip()
    base_url = (source.get("BROKER_BASE_URL")
                or "https://paper-api.alpaca.markets").strip()
    # Acepta con o sin sufijo: "...markets" y "...markets/v2" valen igual
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v2"):
        base_url = base_url[:-len("/v2")]
    if not api_key or not secret:
        raise ValueError(
            "Faltan BROKER_API_KEY / BROKER_SECRET_KEY. "
            "Copiar .env.example -> .env y rellenar (ver README)."
        )
    _validate_paper_url(base_url)
    return BrokerConfig(api_key=api_key, secret_key=secret,
                        base_url=base_url, account_id=account_id)


# ===================================================================
# Cliente REST minimo (urllib)
# ===================================================================
class BrokerError(RuntimeError):
    """Error del broker (sin secretos en el mensaje). `status`: codigo HTTP."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class BrokerTransientError(BrokerError):
    """Fallo ambiguo (red/timeout/429/5xx): el broker pudo o no procesarlo."""


def _calendar_day(value) -> str:
    """Fecha de trading YYYY-MM-DD para GET /v2/calendar."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) < 10:
        raise ValueError(f"fecha de calendario no reconocida: {value!r}")
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"fecha de calendario no reconocida: {value!r}") from exc


class AlpacaClient:
    """Cliente Alpaca paper: cuenta, posiciones y ordenes.

    ``submit_orders`` sigue enviando market/day del runner diario.
    ``submit_order`` cubre limit, stop, stop_limit, bracket, oto y oco
    del intradía (HFT-11).
    """

    def __init__(
        self,
        config: BrokerConfig,
        timeout: int = _DEFAULT_TIMEOUT,
        kill_dir: Optional[Path] = None,
        instruments: Optional[InstrumentMaster] = None,
        *,
        gateway: Optional[GatewayConfig] = None,
        transport: Optional[object] = None,
        budget: Optional[TradingBudget] = None,
    ):
        """
        Args:
            kill_dir: Carpeta donde un archivo `KILL` bloquea todo envio
                (default: `settings.state_dir`).
            instruments: Maestro id ↔ broker_symbol (default
                `load_instruments()`, universe_mixed.csv de la raiz).
            gateway: Transporte, hilos y cupo (default: urlopen en serie,
                sin cupo local, como siempre).
            transport: Transporte ya armado. Solo el bench y los tests lo
                pasan, para apuntar al exchange falso de loopback.
            budget: Cupo ya armado (si no, sale de ``gateway.trading_rpm``).
        """
        _validate_paper_url(config.base_url)  # tambien sin load_broker_config
        if kill_dir is None:
            from src.settings import load_settings
            kill_dir = load_settings().state_dir
        self.instruments = (instruments if instruments is not None
                            else load_instruments())
        self.config = config
        self.kill_dir = Path(kill_dir)
        self.timeout = timeout
        self.gateway = gateway if gateway is not None else GatewayConfig()
        if transport is None:
            kind = PooledTransport if self.gateway.pooled else UrllibTransport
            transport = kind(config.base_url, timeout)
        self.transport = transport
        if budget is None and self.gateway.trading_rpm is not None:
            budget = TradingBudget(self.gateway.trading_rpm,
                                   reserve_frac=self.gateway.reserve_frac)
        self.budget = budget
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.sent_orders = 0  # contador de POST reales (auditoria)
        self.request_count = 0  # peticiones HTTP de trading (todas)
        self._account_verified = False
        # symbol del broker → (dia, respuesta de GET /v2/assets). El dia
        # lo elige el llamador (``as_of``); sin el, es el dia UTC.
        self._asset_cache: Dict[str, Tuple[date, Dict]] = {}

    def _route(self, instrument_id: str) -> Tuple[Optional[str], Optional[str]]:
        """(simbolo Alpaca, None) o (None, motivo por el que no es ruteable)."""
        inst = self.instruments.get(instrument_id)
        if inst is None:
            return None, f"{instrument_id} is not in the instrument master"
        if not inst.routable(BROKER_NAME):
            return None, (f"{instrument_id} ({inst.market}/{inst.currency}) is not "
                          f"tradable on {BROKER_NAME} "
                          f"(tradable_brokers={list(inst.tradable_brokers)})")
        return inst.broker_symbol, None

    def _instrument_id(self, symbol: str) -> str:
        """Simbolo Alpaca → id del maestro (desconocido: el simbolo, con WARNING)."""
        inst = self.instruments.by_broker_symbol(symbol)
        if inst is None:
            logger.warning("Posicion %s fuera del maestro de instrumentos: se "
                           "usa el simbolo del broker como id.", symbol)
            return symbol
        return inst.id

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
            "Content-Type": "application/json",
        }

    def _send(self, method: str, path: str, data: Optional[bytes],
              lane: Optional[str] = None) -> Dict:
        """Un solo intento HTTP. Fallos ambiguos → BrokerTransientError.

        Un body cortado (``IncompleteRead``, JSON truncado) tambien es
        ambiguo: el broker pudo procesar la peticion y no sabemos que
        respondio. GET se reintenta; POST consulta antes de reenviar.

        Con cupo local, la peticion espera su ficha en ``lane`` (``exit``,
        ``read`` o ``entry``; por defecto DELETE es ``exit``, GET ``read``
        y el resto ``entry``) y la respuesta corrige el cupo.
        """
        if lane is None:
            lane = "exit" if method == "DELETE" else "read" if method == "GET" else "entry"
        if self.budget is not None:
            self.budget.acquire(lane)
        with self._lock:
            self.request_count += 1
        try:
            resp = self.transport.request(method, path, data, self._headers())
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            raise BrokerTransientError(
                f"red {path}: {type(exc).__name__}") from None
        if self.budget is not None:
            self.budget.observe(resp.status, resp.headers)
        if resp.status >= 400:
            try:
                detail = resp.body.decode("utf-8")[:300]
            except Exception:
                detail = ""
            msg = f"HTTP {resp.status} {path} {detail}"
            if resp.status == 429 or 500 <= resp.status < 600:
                raise BrokerTransientError(msg, status=resp.status) from None
            raise BrokerError(msg, status=resp.status) from None
        try:
            return json.loads(resp.body.decode("utf-8") or "{}")
        except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
            raise BrokerTransientError(
                f"respuesta cortada {path}: {type(exc).__name__}") from None

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict] = None,
        *,
        lane: Optional[str] = None,
    ) -> Any:
        """HTTP con reintentos SOLO para GET (idempotente).

        Un POST fallido pudo haber llegado al broker: se intenta una vez y el
        llamador decide (ver `_post_order`, que consulta antes de reenviar).
        Con cupo local, un 429 no duerme aqui: el cupo ya frena hasta el reset.
        """
        data = json.dumps(payload).encode() if payload is not None else None
        attempts = _MAX_RETRIES if method == "GET" else 1
        for attempt in range(1, attempts + 1):
            try:
                return self._send(method, path, data, lane)
            except BrokerTransientError as exc:
                if attempts == 1:
                    raise
                last = exc
                if attempt < attempts and not (self.budget is not None and exc.status == 429):
                    time.sleep(_BACKOFF_BASE ** attempt)
        raise BrokerTransientError(f"Reintentos agotados: {last}",
                                   status=last.status)

    def close(self) -> None:
        """Cierra el pool de hilos y las conexiones keep-alive."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def _pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.gateway.workers,
                    thread_name_prefix="alpaca-gw",
                )
            return self._executor

    def run_many(self, calls: List[Callable[[], Any]]) -> List[Tuple[Any, Optional[BaseException]]]:
        """Corre ``calls`` con ``gateway.workers`` hilos. Resultado en el orden de entrada.

        Cada elemento es ``(resultado, None)`` o ``(None, excepcion)``. Con
        un worker corre en serie en este hilo y se detiene en la primera
        excepcion: lo que sigue queda como ``(None, None)`` sin ejecutar,
        igual que el lazo de antes.
        """
        if self.gateway.workers <= 1 or len(calls) <= 1:
            out: List[Tuple[Any, Optional[BaseException]]] = []
            for index, call in enumerate(calls):
                try:
                    out.append((call(), None))
                except BaseException as exc:  # noqa: BLE001 - se devuelve al llamador
                    out.append((None, exc))
                    out.extend((None, None) for _ in calls[index + 1:])
                    break
            return out
        futures = [self._pool().submit(call) for call in calls]
        results: List[Tuple[Any, Optional[BaseException]]] = []
        for future in futures:
            try:
                results.append((future.result(), None))
            except BaseException as exc:  # noqa: BLE001
                results.append((None, exc))
        return results

    # --- Lectura ---
    def get_account(self) -> Dict:
        """Cuenta: equity, buying_power, bloqueos. (GET /v2/account)"""
        return self._request("GET", "/v2/account")

    def get_clock(self) -> Dict:
        """Reloj de mercado. (GET /v2/clock)"""
        return self._request("GET", "/v2/clock")

    def get_calendar(self, start, end) -> List:
        """Sesiones del mercado (GET /v2/calendar). Solo lectura.

        Reintenta con el mismo camino GET de `_request` (429/5xx/red). No
        envia ordenes ni cambia otros metodos.

        Doc verificada (Trading API, ``GET /v2/calendar``, paper y live):
        ``start`` y ``end`` inclusivos. Se envia ``date_type=TRADING``
        (valor por defecto documentado: el rango filtra por fecha de
        negociacion, no de liquidacion). Cada elemento trae ``date``
        (YYYY-MM-DD), ``open`` y ``close`` (HH:MM). El esquema no declara
        zona; Alpaca confirmo que esas horas son del Este
        (America/New_York), no UTC. Pueden venir ``session_open``,
        ``session_close`` y ``settlement_date``; aqui no se interpretan.

        Raises:
            ValueError: ``start`` posterior a ``end``, o fecha ilegible.
            BrokerError: la respuesta no es una lista.
        """
        start_s = _calendar_day(start)
        end_s = _calendar_day(end)
        if start_s > end_s:
            raise ValueError(f"start {start_s} es posterior a end {end_s}")
        query = urlencode({
            "start": start_s,
            "end": end_s,
            "date_type": "TRADING",
        })
        raw = self._request("GET", f"/v2/calendar?{query}")
        if not isinstance(raw, list):
            raise BrokerError(
                "GET /v2/calendar: se esperaba una lista de sesiones, "
                f"llego {type(raw).__name__}"
            )
        return raw

    def get_positions(self) -> pd.DataFrame:
        """Posiciones abiertas -> DataFrame [ticker, qty, market_value, ...]."""
        raw = self._request("GET", "/v2/positions")
        rows = [{
            "ticker": self._instrument_id(p.get("symbol", "")),
            "qty": float(p.get("qty", 0.0)),
            "market_value": float(p.get("market_value", 0.0)),
            "avg_entry": float(p.get("avg_entry_price", 0.0) or 0.0),
            "side": p.get("side", "long"),
        } for p in (raw or [])]
        df = pd.DataFrame(rows, columns=["ticker", "qty", "market_value",
                                         "avg_entry", "side"])
        return df if not df.empty else df

    def get_order(self, order_id: str) -> Dict:
        """Orden por id de Alpaca. (GET /v2/orders/{id})"""
        return self._request("GET", f"/v2/orders/{quote(order_id, safe='')}")

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict]:
        """Orden por client_order_id, o None si Alpaca no la tiene (404)."""
        path = ("/v2/orders:by_client_order_id?client_order_id="
                + quote(client_order_id, safe=""))
        try:
            return self._request("GET", path)
        except BrokerError as exc:
            if exc.status == 404:
                return None
            raise

    def poll_orders(
        self,
        ids: List[str],
        timeout: float = 30.0,
        interval: float = 1.0,
    ) -> Dict[str, Dict]:
        """Consulta ordenes hasta estado final o `timeout` segundos (solo lectura).

        Siempre hace al menos una pasada (timeout=0 → foto instantanea).

        Returns:
            {id: {"final_status", "filled_qty", "filled_avg_price"}}. Si vence
            el timeout, final_status es el ultimo estado visto (ej. "new");
            None si la orden no se pudo leer.
        """
        out: Dict[str, Dict] = {oid: {"final_status": None, "filled_qty": None,
                                      "filled_avg_price": None} for oid in ids}
        pending = list(out)
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while pending:
            for oid in list(pending):
                try:
                    order = self.get_order(oid)
                except BrokerError as exc:
                    logger.warning("No se pudo leer la orden %s: %s", oid, exc)
                    continue
                price = order.get("filled_avg_price")
                out[oid] = {
                    "final_status": order.get("status"),
                    "filled_qty": float(order.get("filled_qty") or 0.0),
                    "filled_avg_price": float(price) if price not in (None, "") else None,
                }
                if order.get("status") in _FINAL_STATUSES:
                    pending.remove(oid)
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(interval)
        if pending:
            logger.warning("Ordenes sin estado final tras %ss: %s", timeout, pending)
        return out

    def _verify_account(self) -> Dict:
        """Confirma que la API responde con la cuenta esperada (1 GET).

        Si `BROKER_ACCOUNT_ID` esta fijado y no coincide, aborta ANTES de
        cualquier orden. Sin secretos en el mensaje.
        """
        with self._lock:
            account = self.get_account()
            expected = (self.config.account_id or "").strip()
            actual = str(account.get("account_number", ""))
            if expected and actual != expected:
                raise BrokerError(
                    f"Cuenta equivocada: espero {expected}, la API responde "
                    f"{actual or '(sin numero)'}. Revisar BROKER_ACCOUNT_ID / "
                    "keys. Nada fue enviado."
                )
            self._account_verified = True
            return account

    def ensure_account_verified(self) -> None:
        """Verifica la cuenta una sola vez, antes de un lote en paralelo."""
        with self._lock:
            if not self._account_verified:
                self._verify_account()

    def close_positions(
        self,
        tickers: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> List[Dict]:
        """Cierra posiciones abiertas (vende el 100% de cada una).

        Args:
            tickers: Subset en formato universo (None = todas).
            dry_run: True (default) → planifica sin enviar.

        Returns:
            Igual que `submit_orders` (status dry_run/submitted/rejected).
        """
        positions = self.get_positions()
        if positions.empty:
            logger.info("Sin posiciones abiertas.")
            return []
        if tickers is not None:
            wanted = {str(t) for t in tickers}
            positions = positions[positions["ticker"].isin(wanted)]
        long_pos = positions[positions["qty"] > 0]
        if long_pos.empty:
            logger.info("Nada que cerrar para el filtro dado.")
            return []
        trades = pd.DataFrame([{
            "ticker": row.ticker, "side": "sell", "qty": round(float(row.qty), 6),
        } for row in long_pos.itertuples()])
        logger.info("Cierre: %d posiciones (dry_run=%s).", len(trades), dry_run)
        return self.submit_orders(trades, dry_run=dry_run)

    def check(self) -> Dict:
        """Chequeo de conectividad: cuenta + reloj (solo lectura)."""
        account = self._verify_account()
        clock = self.get_clock()
        summary = {
            "account_number": account.get("account_number"),
            "equity": account.get("equity"),
            "buying_power": account.get("buying_power"),
            "account_blocked": account.get("account_blocked", False),
            "trading_blocked": account.get("trading_blocked", False),
            "market_open": clock.get("is_open"),
            "timestamp": clock.get("timestamp"),
        }
        logger.info("check: equity=%s open=%s", summary["equity"],
                    summary["market_open"])
        return summary

    # --- Ordenes ---
    def _post_order(self, payload: Dict, lane: str = "entry") -> Dict:
        """POST /v2/orders idempotente via `client_order_id`.

        Ante un fallo ambiguo (red/timeout/5xx) NO reenvia a ciegas:
        consulta la orden por client_order_id y solo reenvia si no existe.
        Un 429 no es ambiguo (el limitador lo corta antes de procesarlo):
        con cupo local se reenvia cuando el cupo lo deja, sin consultar.

        Raises:
            BrokerTransientError: sigue ambiguo tras _MAX_RETRIES intentos.
            BrokerError: rechazo definitivo (4xx).
        """
        cid = payload["client_order_id"]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._request("POST", "/v2/orders", payload, lane=lane)
            except BrokerTransientError as exc:
                last = exc
                if exc.status == 429 and self.budget is not None:
                    logger.warning("POST %s frenado por 429; se reenvia con el cupo.", cid)
                    continue
            except BrokerError as exc:
                # 422 puede ser "id duplicado": el POST anterior si llego
                # (reenvio tras fallo ambiguo, o rerun del mismo dia cuyo
                # estado no se comprometio). Si la orden existe, ya esta enviada.
                if exc.status == 422:
                    existing = self.get_order_by_client_id(cid)
                    if existing is not None:
                        logger.info("Orden %s ya existe (id=%s): no se reenvia.",
                                    cid, existing.get("id"))
                        return existing
                raise
            logger.warning("POST %s ambiguo (%s); consultando antes de reenviar.",
                           cid, last)
            existing = self.get_order_by_client_id(cid)
            if existing is not None:
                logger.info("Orden %s ya existe (id=%s): no se reenvia.",
                            cid, existing.get("id"))
                return existing
            if attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_BASE ** attempt)
        raise BrokerTransientError(f"Reintentos agotados: {last}",
                                   status=last.status)

    def submit_orders(
        self,
        trades: pd.DataFrame,
        dry_run: bool = True,
        run_id: Optional[str] = None,
        on_result: Optional[Callable[[Dict], None]] = None,
    ) -> List[Dict]:
        """Envia ordenes market (o planifica con dry_run).

        Args:
            trades: DataFrame [ticker, side, qty] (qty>0; side buy/sell).
                Tickers = ids del maestro (BRK-B se envia como BRK.B).
            dry_run: True (default) → no hay HTTP, devuelve el plan.
            run_id: Prefijo del client_order_id `{run_id}-{ticker}-{side}`.
                Reusar el mismo run_id hace el reenvio idempotente.
                Default: marca de tiempo UTC (unico por corrida).
            on_result: Callback con cada resultado ANTES de pasar a la
                siguiente orden (journal incremental). Si se aborta, recibe
                la orden fallida (status "failed" + error) antes del raise.

        Returns:
            Lista de dicts {ticker, side, qty, client_order_id, status, id,
            broker_status, error}; status in submitted / dry_run / rejected /
            not_routable. not_routable (fuera del maestro o sin "alpaca" en
            tradable_brokers) se decide antes de enviar, sin HTTP, tambien en
            dry_run. Ante un rechazo 4xx (ej. activo no operable) se registra
            y se CONTINUA; auth (401/403) y fallos ambiguos persistentes abortan.

        Raises:
            ValueError: fila invalida o client_order_id > 48 (nada enviado).
            PreTradeBlocked: kill-switch activo (antes del primer POST o a
                mitad de lote: la orden en curso se reporta como "blocked").
        """
        required = {"ticker", "side", "qty"}
        if not required.issubset(trades.columns):
            raise ValueError(f"trades sin columnas {required - set(trades.columns)}")
        run_id = run_id or datetime.now(timezone.utc).strftime("hft-%Y%m%d%H%M%S")
        # Valida TODO antes del primer POST: nunca abortar a mitad por datos.
        orders = []
        for row in trades.itertuples():
            qty = round(float(row.qty), 6)
            side = str(row.side).lower()
            if qty <= 0 or side not in ("buy", "sell"):
                raise ValueError(f"Orden invalida: {row.ticker} {side} {qty}")
            ticker = str(row.ticker)
            symbol, not_routable = self._route(ticker)
            orders.append((ticker, side, qty,
                           make_client_order_id(run_id, ticker, side),
                           symbol, not_routable))
        cids = [o[3] for o in orders]
        if len(set(cids)) != len(cids):
            raise ValueError("client_order_id duplicado en el plan (ticker+side repetido).")
        if not dry_run:
            kill = kill_switch_reason(self.kill_dir)
            if kill:
                raise PreTradeBlocked([kill])  # antes de cualquier HTTP
        routable = any(o[4] is not None for o in orders)
        if not self._account_verified and not dry_run and routable:
            self._verify_account()  # aborta antes del primer POST si mismatch

        results: List[Dict] = []

        def _emit(record: Dict) -> None:
            results.append(record)
            if on_result is not None:
                on_result(record)

        for ticker, side, qty, cid, symbol, not_routable in orders:
            base = {"ticker": ticker, "side": side, "qty": qty,
                    "client_order_id": cid}
            if not_routable:
                logger.warning("Orden no ruteable (no se envia): %s %s %s — %s",
                               side, qty, ticker, not_routable)
                _emit({**base, "status": "not_routable", "id": None,
                       "broker_status": None, "error": not_routable})
                continue
            if dry_run:
                _emit({**base, "status": "dry_run", "id": None,
                       "broker_status": None, "error": None})
                continue
            kill = kill_switch_reason(self.kill_dir)  # KILL a mitad de lote
            if kill:
                _emit({**base, "status": "blocked", "id": None,
                       "broker_status": None, "error": kill})
                raise PreTradeBlocked([kill])
            try:
                resp = self._post_order({
                    "symbol": symbol, "qty": str(qty), "side": side,
                    "type": "market", "time_in_force": "day",
                    "client_order_id": cid,
                })
            except BrokerError as exc:
                msg = str(exc)
                if isinstance(exc, BrokerTransientError) or exc.status in (401, 403):
                    _emit({**base, "status": "failed", "id": None,
                           "broker_status": None, "error": msg})
                    raise
                logger.warning("Orden rechazada %s %s %s: %s",
                               side, qty, symbol, msg)
                _emit({**base, "status": "rejected", "id": None,
                       "broker_status": None, "error": msg})
                continue
            self.sent_orders += 1
            _emit({**base, "status": "submitted", "id": resp.get("id"),
                   "broker_status": resp.get("status"), "error": None})
            logger.info("Orden enviada: %s %s %s (id=%s, cid=%s)",
                        side, qty, symbol, resp.get("id"), cid)
        return results

    def submit_order(
        self,
        spec: object,
        *,
        asset: Optional[Mapping] = None,
        mark: Optional[float] = None,
        guard: Optional[object] = None,
        risk_state: Optional[object] = None,
        journal: Optional[List[Dict]] = None,
        orders_this_minute: Optional[int] = None,
        orders_today: Optional[int] = None,
        projected_gross: Optional[float] = None,
        lane: str = "entry",
    ) -> Dict:
        """Envia una orden intradía (``POST /v2/orders``).

        ``lane`` es el carril del cupo: ``exit`` para una orden que
        reduce posición, ``entry`` para el resto.

        La validacion de tipo, TIF, tick, fraccional, bracket y corto
        ocurre antes de cualquier HTTP. Un simbolo no ruteable vuelve
        con status ``not_routable`` y tampoco abre red. El KILL se
        anota en ``journal`` y lanza ``PreTradeBlocked``. El resto de
        guardas (tasa, nocional, exposicion) anota el motivo y devuelve
        status ``blocked`` sin POST.

        Un 422 de id duplicado lo resuelve ``_post_order``: consulta
        por ``client_order_id`` y, si la orden existe, no se reenvia.

        Returns:
            Dict con ticker, side, qty, client_order_id, status, id,
            broker_status y error. status in submitted / blocked /
            rejected / not_routable.
        """
        order = coerce_spec(spec)
        symbol, not_routable = self._route(str(order.instrument_id))
        base = {
            "ticker": str(order.instrument_id),
            "side": str(order.side).strip().lower(),
            "qty": order.qty,
            "client_order_id": str(order.client_order_id).strip(),
        }
        if not_routable:
            record = {**base, "status": "not_routable", "id": None,
                      "broker_status": None, "error": not_routable}
            logger.warning("Orden no ruteable (no se envia): %s — %s",
                           order.instrument_id, not_routable)
            _remember(journal, record)
            return record
        # Valida el payload ANTES de cuenta, KILL-HTTP y POST. Un
        # fraccional con bracket falla aqui, con el contador de red en cero.
        payload = build_order_payload(order, broker_symbol=symbol, asset=asset)
        base["client_order_id"] = payload["client_order_id"]
        base["side"] = payload["side"]
        kill = kill_switch_reason(self.kill_dir)
        if kill:
            record = {**base, "status": "blocked", "id": None,
                      "broker_status": None, "error": kill}
            _remember(journal, record)
            raise PreTradeBlocked([kill])
        blocked = self._intraday_block(
            order, guard=guard, risk_state=risk_state, mark=mark,
            orders_this_minute=orders_this_minute, orders_today=orders_today,
            projected_gross=projected_gross,
        )
        if blocked:
            record = {**base, "status": "blocked", "id": None,
                      "broker_status": None, "error": blocked}
            logger.warning("Orden bloqueada pre-trade %s: %s",
                           payload["client_order_id"], blocked)
            _remember(journal, record)
            return record
        if not self._account_verified:
            self.ensure_account_verified()
        try:
            resp = self._post_order(payload, lane=lane)
        except BrokerError as exc:
            msg = str(exc)
            if isinstance(exc, BrokerTransientError) or exc.status in (401, 403):
                record = {**base, "status": "failed", "id": None,
                          "broker_status": None, "error": msg}
                _remember(journal, record)
                raise
            logger.warning("Orden rechazada %s: %s", payload["client_order_id"], msg)
            record = {**base, "status": "rejected", "id": None,
                      "broker_status": None, "error": msg}
            _remember(journal, record)
            return record
        with self._lock:
            self.sent_orders += 1
        record = {**base, "status": "submitted", "id": resp.get("id"),
                  "broker_status": resp.get("status"), "error": None}
        _remember(journal, record)
        logger.info("Orden intradía enviada: %s %s %s (id=%s, cid=%s)",
                    payload["side"], payload["qty"], symbol, resp.get("id"),
                    payload["client_order_id"])
        return record

    def _intraday_block(
        self,
        spec: OrderSpec,
        *,
        guard: Optional[object],
        risk_state: Optional[object],
        mark: Optional[float],
        orders_this_minute: Optional[int],
        orders_today: Optional[int],
        projected_gross: Optional[float],
    ) -> Optional[str]:
        """Motivo de guarda, o None. No abre red."""
        if guard is None:
            return None
        price = _reference_price(spec, mark)
        per_order = getattr(guard, "max_notional_per_order", None)
        if per_order is not None and price is None:
            return "sin precio de referencia para el nocional"
        gross_cap = getattr(guard, "max_gross_exposure", None)
        if gross_cap is not None and projected_gross is None:
            return "exposicion bruta post-trade no calculada"
        notional = abs(float(spec.qty) * price) if price is not None else 0.0
        equity = 0.0
        if risk_state is not None and getattr(risk_state, "equity", None) is not None:
            equity = float(risk_state.equity)
        summary: Dict[str, object] = {
            "equity": equity,
            "account_blocked": False,
            "trading_blocked": False,
            "market_open": True,
        }
        if orders_this_minute is not None:
            summary["orders_this_minute"] = int(orders_this_minute)
        elif risk_state is not None:
            summary["orders_this_minute"] = int(getattr(risk_state, "orders_this_minute", 0))
        if orders_today is not None:
            summary["orders_today"] = int(orders_today)
        elif risk_state is not None:
            summary["orders_today"] = int(getattr(risk_state, "orders_today", 0))
        if projected_gross is not None:
            summary["projected_gross_notional"] = float(projected_gross)
        plan = pd.DataFrame([{
            "ticker": spec.instrument_id,
            "side": spec.side,
            "notional": notional,
        }])
        reasons = guard.evaluate(plan, summary)
        if not reasons:
            return None
        return "; ".join(reasons)

    def cancel_order(self, order_id: str) -> Dict:
        """Cancela una orden (``DELETE /v2/orders/{order_id}``).

        204 llega como ``{}``: el cuerpo vacio no es un error.
        """
        if not str(order_id).strip():
            raise ValueError("order_id vacio")
        raw = self._request("DELETE", f"/v2/orders/{quote(str(order_id), safe='')}")
        if raw is None or raw == {}:
            return {}
        if isinstance(raw, dict):
            return raw
        raise BrokerError(
            f"DELETE /v2/orders/{{id}}: se esperaba un objeto, llego {type(raw).__name__}"
        )

    def cancel_all_orders(self) -> List:
        """Cancela las ordenes abiertas (``DELETE /v2/orders``).

        La respuesta documentada es una lista (multi-status). Un cuerpo
        vacio se trata como lista vacia.
        """
        raw = self._request("DELETE", "/v2/orders")
        if raw is None or raw == {}:
            return []
        if isinstance(raw, list):
            return raw
        raise BrokerError(
            f"DELETE /v2/orders: se esperaba una lista, llego {type(raw).__name__}"
        )

    def list_orders(
        self,
        status: str = "open",
        after: object = None,
        limit: int = 50,
        *,
        nested: bool = False,
        direction: Optional[str] = None,
    ) -> List:
        """Ordenes (``GET /v2/orders``).

        ``status`` es ``open``, ``closed`` o ``all``. ``after`` es un
        instante RFC3339 (exclusivo, ordenes posteriores). ``limit`` va
        de 1 a 500. ``nested=true`` incluye ``legs`` de bracket/oco.
        ``direction`` es ``asc`` o ``desc`` (default de Alpaca: ``desc``,
        las mas nuevas primero). Paginar con ``after`` exige ``asc``.
        """
        if status not in _LIST_STATUSES:
            raise ValueError(f"status {status!r} no esta en {_LIST_STATUSES}")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit debe ser un entero entre 1 y {_MAX_LIST_LIMIT}")
        if not isinstance(nested, bool):
            raise TypeError("nested debe ser bool")
        if direction not in (None, "asc", "desc"):
            raise ValueError(f"direction {direction!r} no esta en asc|desc")
        params = {"status": status, "limit": str(limit)}
        if after is not None:
            params["after"] = _rfc3339(after)
        if nested:
            params["nested"] = "true"
        if direction is not None:
            params["direction"] = direction
        raw = self._request("GET", "/v2/orders?" + urlencode(params))
        if not isinstance(raw, list):
            raise BrokerError(
                f"GET /v2/orders: se esperaba una lista, llego {type(raw).__name__}"
            )
        return raw

    def replace_order(
        self,
        order_id: str,
        *,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """Reemplaza qty o precios (``PATCH /v2/orders/{order_id}``).

        Cada campo presente pisa el de la orden viva. Los precios pasan
        por el mismo tick que un alta. No es un GET: no se reintenta.
        """
        if not str(order_id).strip():
            raise ValueError("order_id vacio")
        if qty is None and limit_price is None and stop_price is None:
            raise ValueError("replace exige qty, limit_price o stop_price")
        payload: Dict[str, str] = {}
        if qty is not None:
            payload["qty"] = format_qty(qty)
        if limit_price is not None:
            payload["limit_price"] = format_tick_price(limit_price)
        if stop_price is not None:
            payload["stop_price"] = format_tick_price(stop_price)
        raw = self._request(
            "PATCH", f"/v2/orders/{quote(str(order_id), safe='')}", payload,
        )
        if not isinstance(raw, dict):
            raise BrokerError(
                f"PATCH /v2/orders/{{id}}: se esperaba un objeto, llego {type(raw).__name__}"
            )
        return raw

    def get_asset(self, symbol: str, *, as_of: Optional[object] = None) -> Dict:
        """Ficha (``GET /v2/assets/{symbol}``) con cache de un dia.

        ``symbol`` puede ser el id del maestro o el simbolo del broker.
        ``as_of`` es el dia de la cache (date, o datetime: se usa su
        fecha de pared). Sin ``as_of``, el dia es UTC de este proceso:
        el intradía pasa el dia de la sesion para no depender del reloj
        aqui. Un dia distinto vuelve a pedir la ficha.
        """
        day = self._asset_day(as_of)
        inst = self.instruments.get(str(symbol).strip())
        broker = inst.broker_symbol if inst is not None and inst.broker_symbol else str(symbol).strip()
        if not broker:
            raise ValueError("symbol vacio")
        with self._lock:
            cached = self._asset_cache.get(broker)
        if cached is not None and cached[0] == day:
            return cached[1]
        raw = self._request("GET", f"/v2/assets/{quote(broker, safe='')}")
        if not isinstance(raw, dict):
            raise BrokerError(
                f"GET /v2/assets: se esperaba un objeto, llego {type(raw).__name__}"
            )
        with self._lock:
            self._asset_cache[broker] = (day, raw)
        return raw

    @staticmethod
    def _asset_day(as_of: Optional[object]) -> date:
        if as_of is None:
            return datetime.now(timezone.utc).date()
        if isinstance(as_of, datetime):
            return as_of.date()
        if isinstance(as_of, date):
            return as_of
        raise TypeError("as_of debe ser date o datetime")

    def get_account_activities(
        self,
        activity_type: str = "FILL",
        date: object = None,
        *,
        page_size: int = _ACTIVITY_PAGE_MAX,
        page_token: Optional[str] = None,
    ) -> List:
        """Fills u otras actividades (``GET /v2/account/activities``).

        El query documentado es ``activity_types`` (plural), no
        ``activity_type``. ``date`` filtra por ``created_at``
        (YYYY-MM-DD), no por liquidacion. Se pagina con ``page_token``
        = id de la ultima fila, ``page_size`` <= 100, en orden
        ascendente, hasta ``_ACTIVITY_PAGES`` paginas. ``page_token``
        arranca despues de ese id (lectura incremental).
        """
        kind = str(activity_type).strip().upper()
        if not kind or not kind.replace("_", "").isalnum():
            raise ValueError(f"activity_type no reconocido: {activity_type!r}")
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise TypeError("page_size debe ser int")
        if not 1 <= page_size <= _ACTIVITY_PAGE_MAX:
            raise ValueError(f"page_size debe estar entre 1 y {_ACTIVITY_PAGE_MAX}")
        params: Dict[str, str] = {
            "activity_types": kind,
            "page_size": str(page_size),
            "direction": "asc",
        }
        if date is not None:
            params["date"] = _activity_day(date)
        out: List = []
        token: Optional[str] = str(page_token) if page_token else None
        for _ in range(_ACTIVITY_PAGES):
            query = dict(params)
            if token:
                query["page_token"] = token
            page = self._request("GET", "/v2/account/activities?" + urlencode(query))
            if not isinstance(page, list):
                raise BrokerError(
                    "GET /v2/account/activities: se esperaba una lista, "
                    f"llego {type(page).__name__}"
                )
            out.extend(page)
            if len(page) < page_size or not page:
                break
            token = str(page[-1].get("id") or "")
            if not token:
                break
        return out


# ===================================================================
# Reconciliacion: pesos objetivo -> ordenes en acciones
# ===================================================================
def reconcile_trades(
    target_weights: pd.Series,
    broker_qty: pd.Series,
    prices: pd.Series,
    portfolio_value: float,
    min_notional: float = 1.0,
    return_unpriced: bool = False,
):
    """Diferencia entre cartera objetivo y posicion real del broker.

    Args:
        target_weights: Pesos objetivo (suman 1) por ticker.
        broker_qty: Acciones actuales en broker por ticker (0 si ausente).
        prices: Ultimo precio por ticker.
        portfolio_value: Equity para convertir pesos -> acciones.
        min_notional: Omite ordenes menores (polvo).
        return_unpriced: False (default) → MissingPriceError si el broker
            tiene una posicion sin precio valido. True → devuelve
            ``(plan, unpriced)`` y el caller decide (esas no se operan).

    Returns:
        DataFrame [ticker, side, qty, notional] solo con diferencias.
        Con ``return_unpriced=True``, tupla ``(plan, unpriced_tickers)``.

    Raises:
        MissingPriceError: Posicion en broker sin precio finito > 0 y
            ``return_unpriced`` False (no se omite en silencio: quedaria
            huerfana, imposible de vender).
    """
    tickers = target_weights.index.union(broker_qty.index).union(prices.index)
    rows = []
    unpriced: List[str] = []
    for ticker in tickers:
        w = float(target_weights.get(ticker, 0.0))
        px = float(prices.get(ticker, float("nan")))
        if not math.isfinite(px) or px <= 0:  # sin precio no se opera
            if abs(float(broker_qty.get(ticker, 0.0))) > 1e-12:
                unpriced.append(str(ticker))
            continue
        target_qty = w * portfolio_value / px
        current = float(broker_qty.get(ticker, 0.0))
        diff = target_qty - current
        notional = abs(diff) * px
        if notional < min_notional:
            continue
        rows.append({
            "ticker": ticker,
            "side": "buy" if diff > 0 else "sell",
            "qty": round(abs(diff), 6),
            "notional": round(notional, 2),
        })
    if unpriced and not return_unpriced:
        raise MissingPriceError(
            unpriced, "Posiciones del broker sin precio: no se planifica "
            "ninguna orden (pasar return_unpriced=True para decidir).")
    df = pd.DataFrame(rows, columns=["ticker", "side", "qty", "notional"])
    df = df.sort_values("notional", ascending=False).reset_index(drop=True)
    if return_unpriced:
        return df, sorted(unpriced)
    return df


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Alpaca paper: operar con red de seguridad.")
    parser.add_argument("--check", action="store_true", help="Cuenta + reloj.")
    parser.add_argument("--positions", action="store_true", help="Lista posiciones.")
    parser.add_argument("--close", action="append", default=[],
                        help="Cierra un ticker (repetible). Sin --confirm es dry-run.")
    parser.add_argument("--close-all", action="store_true",
                        help="Cierra TODAS las posiciones. Sin --confirm es dry-run.")
    parser.add_argument("--confirm", action="store_true",
                        help="Con --close/--close-all, envia de verdad (verifica cuenta).")
    args = parser.parse_args(argv)
    from src.settings import setup_logging
    setup_logging()
    try:
        client = AlpacaClient(load_broker_config())
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Conectado: {client.config}")
    if args.check:
        print(json.dumps(client.check(), indent=2, default=str))
    if args.positions:
        print(client.get_positions().to_string(index=False) or "(sin posiciones)")
    if args.close or args.close_all:
        tickers = args.close or None
        if args.close_all:
            tickers = None
        results = client.close_positions(tickers, dry_run=not args.confirm)
        if not results:
            print("(nada que cerrar)")
        for r in results:
            print(f"{r['status']:>9} {r['side']:>4} {r['qty']:<12} {r['ticker']}"
                  + (f" id={r['id']}" if r.get("id") else "")
                  + (f" ERR={r['error']}" if r.get("error") else ""))
        if not args.confirm:
            print("Dry-run: nada enviado. Agrega --confirm para cerrar de verdad.")
    if not args.check and not args.positions and not args.close and not args.close_all:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
