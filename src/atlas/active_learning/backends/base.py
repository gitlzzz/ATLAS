"""The MLIP backend interface.

An `MLIPBackend` encapsulates everything framework-specific about a machine-learned
interatomic potential so the active-learning loop can stay model-agnostic:

- building an ASE calculator from a trained model or a foundation-model id,
- preparing the training dataset in the framework's expected format,
- naming the AiiDA training CalcJob,
- (optionally) extracting descriptors and exporting a LAMMPS model.

Concrete backends (MACE today; NequIP/Allegro/DeepMD/Equiformer/Orb/LASP as
follow-ups) subclass this and register themselves via
``atlas.active_learning.backends.registry.register_backend``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class MLIPBackend(ABC):
    """Framework-specific hooks for an MLIP, consumed by the AL loop."""

    #: Short backend name, matched against the ``model_type`` config field.
    name: str = 'base'
    #: File extension of a trained model for this backend.
    model_extension: str = '.model'

    # --- required ---------------------------------------------------------

    @abstractmethod
    def build_calculator(
        self,
        model=None,
        *,
        device: str = 'cpu',
        dtype: str = 'float64',
        enable_cueq: bool = False,
        **kwargs,
    ):
        """Return an ASE calculator for a trained model path or a foundation id."""

    @abstractmethod
    def train_calcjob_entry_point(self) -> str:
        """AiiDA CalcJob entry-point name that trains a model of this backend."""

    # --- optional (sensible defaults) ------------------------------------

    def parse_foundation_id(self, model) -> tuple[str, str] | None:
        """Return ``(family, variant)`` if ``model`` names a foundation model, else None."""
        return None

    def prepare_training_input(self, structures, out_path, **kwargs) -> Path:
        """Write the framework's training dataset to ``out_path``; return the path.

        Defaults to unsupported; extxyz-based backends override this.
        """
        raise NotImplementedError(
            f'{self.name} backend does not implement prepare_training_input.'
        )

    def get_descriptors(self, *args, **kwargs):
        """Return per-structure descriptors, or raise to signal a SOAP fallback."""
        raise NotImplementedError(
            f'{self.name} backend has no native descriptors; fall back to SOAP.'
        )

    def get_lammps_model(self, model_path):
        """Return a LAMMPS-compatible model path, or None if unsupported."""
        return None
