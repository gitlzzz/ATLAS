"""Pluggable MLIP backends for ATLAS.

The active-learning loop resolves a backend from the ``model_type`` config field
via :func:`get_backend`. MACE is fully supported; NequIP, Allegro, DeepMD,
EquiformerV2/V3, Orb, and LASP are registered as (guarded) backends built on the
universal ASE-calculator interface — they light up once their framework package
is installed. See each module for the (currently unverified) framework API used.
"""

from atlas.active_learning.backends.allegro import AllegroBackend
from atlas.active_learning.backends.ase_backend import ASECalculatorBackend
from atlas.active_learning.backends.base import MLIPBackend
from atlas.active_learning.backends.containers import resolve_container_settings
from atlas.active_learning.backends.deepmd import DeepMDBackend
from atlas.active_learning.backends.equiformer import EquiformerBackend
from atlas.active_learning.backends.lasp import LASPBackend
from atlas.active_learning.backends.mace import MaceBackend
from atlas.active_learning.backends.nequip import NequIPBackend
from atlas.active_learning.backends.orb import OrbBackend
from atlas.active_learning.backends.registry import (
    available_backends,
    get_backend,
    register_backend,
)

register_backend('mace', MaceBackend)
register_backend('nequip', NequIPBackend)
register_backend('allegro', AllegroBackend)
register_backend('deepmd', DeepMDBackend)
register_backend('equiformer', EquiformerBackend)
register_backend('orb', OrbBackend)
register_backend('lasp', LASPBackend)

__all__ = [
    'MLIPBackend',
    'ASECalculatorBackend',
    'MaceBackend',
    'NequIPBackend',
    'AllegroBackend',
    'DeepMDBackend',
    'EquiformerBackend',
    'OrbBackend',
    'LASPBackend',
    'available_backends',
    'get_backend',
    'register_backend',
    'resolve_container_settings',
]
