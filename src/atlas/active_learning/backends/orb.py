"""Orb backend (orb-models).

Orb ships pretrained universal potentials with an ASE calculator; ``model`` may
name a pretrained potential (e.g. ``"orb-v2"``) or point to a checkpoint.
Training/fine-tuning support in orb-models is limited. UNVERIFIED against a live
install.
"""

from __future__ import annotations

from atlas.active_learning.backends.ase_backend import ASECalculatorBackend


class OrbBackend(ASECalculatorBackend):
    """Orb graph-network potentials (pretrained-focused)."""

    name = 'orb'
    model_extension = '.ckpt'
    required_package = 'orb_models'

    def parse_foundation_id(self, model):
        if isinstance(model, str) and model.startswith('orb'):
            return ('pretrained', model)
        return None

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        foundation = self.parse_foundation_id(model)
        if foundation is not None:
            _, variant = foundation
            factory = getattr(pretrained, variant.replace('-', '_'), None)
            if factory is None:
                raise ValueError(f"Unknown Orb pretrained potential '{variant}'.")
            orbff = factory(device=device)
        elif model:
            orbff = pretrained.orb_v2(weights_path=str(model), device=device)
        else:
            orbff = pretrained.orb_v2(device=device)
        return ORBCalculator(orbff, device=device)

    def train_calcjob_entry_point(self) -> str:
        return 'orb-train'
