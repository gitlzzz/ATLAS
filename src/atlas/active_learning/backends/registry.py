"""Registry mapping a ``model_type`` string to its :class:`MLIPBackend`."""

from __future__ import annotations

from atlas.active_learning.backends.base import MLIPBackend

_REGISTRY: dict[str, type[MLIPBackend]] = {}


def register_backend(name: str, cls: type[MLIPBackend]) -> None:
    """Register a backend class under ``name`` (case-insensitive)."""
    _REGISTRY[name.lower()] = cls


def available_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def get_backend(model_type: str = 'mace') -> MLIPBackend:
    """Return an instance of the backend for ``model_type`` (defaults to MACE)."""
    key = (model_type or 'mace').lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown MLIP model_type '{model_type}'. "
            f'Registered backends: {available_backends()}'
        )
    return _REGISTRY[key]()
