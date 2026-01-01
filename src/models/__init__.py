# src/models/__init__.py
"""Models for sports prediction."""

__all__ = ["PoissonModel"]


def __getattr__(name):
    """Lazy imports to avoid circular import issues."""
    if name in __all__:
        from src.models import poisson

        return getattr(poisson, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
