# src/trading/__init__.py
"""Trading module for edge calculation, bet sizing, and execution."""

__all__ = [
    "EdgeCalculator",
    "remove_vig",
    "odds_to_prob",
    "rolling_backtest",
    "KellyCalculator",
    "kelly_stake",
]


def __getattr__(name):
    """Lazy imports to avoid circular import issues."""
    if name in ["EdgeCalculator", "remove_vig", "odds_to_prob", "rolling_backtest"]:
        from src.trading import edge

        return getattr(edge, name)
    if name in ["KellyCalculator", "kelly_stake"]:
        from src.trading import kelly

        return getattr(kelly, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
