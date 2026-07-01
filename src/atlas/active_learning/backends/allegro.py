"""Allegro backend.

Allegro is a strictly-local equivariant model in the NequIP ecosystem: it uses
the same ASE calculator and `nequip-train` CLI, with the Allegro architecture
selected in the training config. Reuses the NequIP backend's calculator logic.
UNVERIFIED against a live install.
"""

from __future__ import annotations

from atlas.active_learning.backends.nequip import NequIPBackend


class AllegroBackend(NequIPBackend):
    """Allegro models (NequIP ecosystem)."""

    name = 'allegro'
    architecture = 'allegro'
    # The calculator comes from `nequip`; the Allegro architecture needs the
    # `allegro` package installed at training time.
    required_package = 'nequip'

    def train_calcjob_entry_point(self) -> str:
        return 'allegro-train'
