"""Pluggable MLIP backends for ATLAS.

The active-learning loop resolves a backend from the ``model_type`` config field
via :func:`get_backend`. MACE is registered by default; additional frameworks
(NequIP, Allegro, DeepMD, EquiformerV2/V3, Orb, LASP) register themselves here.
"""

from atlas.active_learning.backends.base import MLIPBackend
from atlas.active_learning.backends.mace import MaceBackend
from atlas.active_learning.backends.registry import (
    available_backends,
    get_backend,
    register_backend,
)

register_backend('mace', MaceBackend)

__all__ = [
    'MLIPBackend',
    'MaceBackend',
    'available_backends',
    'get_backend',
    'register_backend',
]
