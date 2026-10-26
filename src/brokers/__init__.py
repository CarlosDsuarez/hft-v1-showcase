"""Alpaca adapter (paper-only). Import perezoso para evitar doble import con -m."""

__all__ = ["AlpacaClient", "BrokerConfig", "load_broker_config", "reconcile_trades"]


def __getattr__(name):
    if name in __all__:
        from src.brokers import alpaca_adapter as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
