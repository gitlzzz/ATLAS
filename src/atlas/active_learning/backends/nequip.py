"""NequIP backend.

API verified against nequip 0.18.0 (calculator import path + classmethod
signature). Note: NequIP requires ``e3nn>=0.6``, which is INCOMPATIBLE with the
``e3nn==0.4.4`` MACE pins, so NequIP must run in its own environment/container
(not co-installed with MACE). A live single-point needs a compiled model
(``nequip-train`` -> ``nequip-compile``), so only the API is verified here.
"""

from __future__ import annotations

from atlas.active_learning.backends.ase_backend import ASECalculatorBackend


class NequIPBackend(ASECalculatorBackend):
    """E(3)-equivariant NequIP models, used via the NequIP ASE calculator."""

    name = 'nequip'
    model_extension = '.nequip.pt2'
    required_package = 'nequip'
    #: model architecture selected in the nequip training config
    architecture = 'nequip'

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        # nequip >= 0.7 moved the calculator to nequip.integrations.ase.
        try:
            from nequip.integrations.ase import NequIPCalculator
        except ImportError:
            from nequip.ase import NequIPCalculator

        # >= 0.7 loads compiled models; <= 0.6 loaded deployed .pth models.
        if hasattr(NequIPCalculator, 'from_compiled_model'):
            return NequIPCalculator.from_compiled_model(str(model), device=device)
        return NequIPCalculator.from_deployed_model(model_path=str(model), device=device)

    def train_calcjob_entry_point(self) -> str:
        # The matching AiiDA CalcJob (wrapping `nequip-train`) is added in the
        # NequIP training-integration PR.
        return 'nequip-train'
