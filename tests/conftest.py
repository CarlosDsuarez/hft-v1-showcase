"""
conftest.py — Aislamiento global de la suite (offline, sin efectos externos).

- Alertas por email apagadas: un ``.env`` real del desarrollador con
  ``ALERT_EMAIL`` + ``ALERT_SMTP_HOST`` no debe enviar correos cuando un test
  ejecuta el runner (el entorno gana sobre ``.env``; vacío = desactivado).
"""

import pytest


@pytest.fixture(autouse=True)
def _no_alert_email(monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL", "")
