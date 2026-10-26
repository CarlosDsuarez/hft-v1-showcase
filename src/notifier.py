"""
notifier.py — Hook de alertas del runner vivo (ARQ-21).

Interfaz `Notifier` (``notify(alert) -> None``) e implementaciones:

  - `ConsoleNotifier`: al logger ``alerts`` (consola + ``runner.log`` de la
    corrida, con su run_id), nivel según la alerta.
  - `FileNotifier`: una línea JSON por alerta (append + fsync), por defecto
    ``state_dir/logs/alerts.jsonl``.
  - `EmailNotifier`: SMTP (stdlib), opcional: solo se activa con
    ``ALERT_EMAIL`` + ``ALERT_SMTP_HOST`` (entorno o ``.env`` de la raíz; el
    entorno gana). Solo envía alertas >= ``ALERT_EMAIL_MIN_LEVEL``
    (default "error").
  - `MultiNotifier`: reparte a varios; un notifier que falla se loguea y
    nunca rompe la corrida ni a los demás.

`build_notifier(settings)` arma consola + archivo (+ email si está
configurado). Con ``writable=False`` (corridas de solo lectura:
--check-only / --dry-run) solo consola: no escribe en ``state_dir``.

Uso:
    notifier = build_notifier(load_settings())
    notifier.notify(Alert("error", "Datos inutilizables", "...", run_id=run_id))
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from src.settings import Settings, read_dotenv

logger = logging.getLogger("alerts")

LEVELS = ("info", "warning", "error", "critical")
_LOG_LEVELS = {"info": logging.INFO, "warning": logging.WARNING,
               "error": logging.ERROR, "critical": logging.CRITICAL}
ALERTS_FILE_NAME = "alerts.jsonl"
DEFAULT_EMAIL_MIN_LEVEL = "error"
DEFAULT_SMTP_PORT = 587
SMTP_SSL_PORT = 465
SMTP_TIMEOUT_S = 10.0


def _check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"level={level!r} no soportado (usar {LEVELS}).")
    return level


def level_at_least(level: str, minimum: str) -> bool:
    return LEVELS.index(_check_level(level)) >= LEVELS.index(_check_level(minimum))


@dataclass(frozen=True)
class Alert:
    """Evento notable de una corrida (resultado, error, bloqueo...)."""

    level: str                       # info | warning | error | critical
    title: str
    message: str
    run_id: Optional[str] = None
    exit_code: Optional[int] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _check_level(self.level)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["details"] = dict(self.details)
        return out


class Notifier(Protocol):
    """Destino de alertas. No debe lanzar por fallos propios (ver MultiNotifier)."""

    def notify(self, alert: Alert) -> None: ...


class ConsoleNotifier:
    """Alerta → logger ``alerts`` (consola y runner.log de la corrida)."""

    def notify(self, alert: Alert) -> None:
        exit_code = "" if alert.exit_code is None else f" (exit {alert.exit_code})"
        logger.log(_LOG_LEVELS[alert.level], "ALERT %s%s: %s — %s",
                   alert.level.upper(), exit_code, alert.title, alert.message)


class FileNotifier:
    """Alerta → una línea JSON en ``path`` (append + fsync)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def notify(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(alert.to_dict(), default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


@dataclass(frozen=True)
class EmailConfig:
    """SMTP de alertas (``ALERT_*``). La contraseña nunca va al repr/log."""

    recipients: tuple
    host: str
    port: int = DEFAULT_SMTP_PORT
    sender: str = ""
    user: str = ""
    password: str = field(default="", repr=False)
    min_level: str = DEFAULT_EMAIL_MIN_LEVEL

    def __post_init__(self) -> None:
        _check_level(self.min_level)


def load_email_config(env: Optional[Mapping[str, str]] = None) -> Optional[EmailConfig]:
    """EmailConfig desde el entorno (+ ``.env`` de la raíz), o None si el
    email no está configurado (falta ``ALERT_EMAIL`` o ``ALERT_SMTP_HOST``).

    Raises:
        ValueError: ``ALERT_SMTP_PORT`` o ``ALERT_EMAIL_MIN_LEVEL`` inválidos.
    """
    source = env if env is not None else {**read_dotenv(), **os.environ}
    recipients = tuple(r.strip() for r in (source.get("ALERT_EMAIL") or "").split(",")
                       if r.strip())
    host = (source.get("ALERT_SMTP_HOST") or "").strip()
    if not recipients:
        return None
    if not host:
        logger.warning("ALERT_EMAIL definido sin ALERT_SMTP_HOST: alertas por "
                       "email desactivadas.")
        return None
    port_text = (source.get("ALERT_SMTP_PORT") or "").strip()
    try:
        port = int(port_text) if port_text else DEFAULT_SMTP_PORT
    except ValueError:
        raise ValueError(f"ALERT_SMTP_PORT={port_text!r} no es un entero.") from None
    user = (source.get("ALERT_SMTP_USER") or "").strip()
    return EmailConfig(
        recipients=recipients, host=host, port=port,
        sender=(source.get("ALERT_EMAIL_FROM") or "").strip() or user or recipients[0],
        user=user,
        password=source.get("ALERT_SMTP_PASSWORD") or "",
        min_level=(source.get("ALERT_EMAIL_MIN_LEVEL") or DEFAULT_EMAIL_MIN_LEVEL
                   ).strip().lower(),
    )


def _smtp_client(config: EmailConfig) -> smtplib.SMTP:
    if config.port == SMTP_SSL_PORT:
        return smtplib.SMTP_SSL(config.host, config.port, timeout=SMTP_TIMEOUT_S)
    client = smtplib.SMTP(config.host, config.port, timeout=SMTP_TIMEOUT_S)
    client.starttls()
    return client


class EmailNotifier:
    """Alerta >= ``config.min_level`` → email (SMTP_SSL en 465; si no, STARTTLS).

    Args:
        smtp_factory: ``config -> cliente smtplib`` (tests: doble sin red).
    """

    def __init__(
        self,
        config: EmailConfig,
        smtp_factory: Optional[Callable[[EmailConfig], Any]] = None,
    ) -> None:
        self.config = config
        self._smtp_factory = smtp_factory or _smtp_client

    def message(self, alert: Alert) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = f"[HFT-V1 {alert.level.upper()}] {alert.title}"
        msg["From"] = self.config.sender
        msg["To"] = ", ".join(self.config.recipients)
        lines = [alert.message, "",
                 f"run_id: {alert.run_id or '-'}",
                 f"exit_code: {alert.exit_code if alert.exit_code is not None else '-'}",
                 f"created_at_utc: {alert.created_at_utc}"]
        lines += [f"{k}: {v}" for k, v in alert.details.items()]
        msg.set_content("\n".join(lines))
        return msg

    def notify(self, alert: Alert) -> None:
        if not level_at_least(alert.level, self.config.min_level):
            return
        client = self._smtp_factory(self.config)
        try:
            if self.config.user:
                client.login(self.config.user, self.config.password)
            client.send_message(self.message(alert))
        finally:
            client.quit()


class MultiNotifier:
    """Reparte cada alerta; el fallo de uno se loguea y no frena al resto."""

    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self.notifiers: List[Notifier] = list(notifiers)

    def notify(self, alert: Alert) -> None:
        for notifier in self.notifiers:
            try:
                notifier.notify(alert)
            except Exception as exc:  # noqa: BLE001 — una alerta nunca rompe la corrida
                logger.warning("Notifier %s falló (%s: %s); alerta '%s' no entregada.",
                               type(notifier).__name__, type(exc).__name__, exc,
                               alert.title)


def build_notifier(
    settings: Settings,
    writable: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> MultiNotifier:
    """Consola siempre; con ``writable``, además ``logs/alerts.jsonl`` y email
    si ``ALERT_EMAIL`` + ``ALERT_SMTP_HOST`` están definidos.

    Un ``ALERT_*`` inválido desactiva el email con un warning (no aborta la
    corrida por la configuración de alertas).
    """
    notifiers: List[Notifier] = [ConsoleNotifier()]
    if writable:
        notifiers.append(FileNotifier(settings.logs_dir / ALERTS_FILE_NAME))
        try:
            email = load_email_config(env)
        except ValueError as exc:
            logger.warning("Config de alertas por email inválida (%s): email "
                           "desactivado.", exc)
            email = None
        if email is not None:
            notifiers.append(EmailNotifier(email))
    return MultiNotifier(notifiers)
