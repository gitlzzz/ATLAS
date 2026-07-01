"""NequIP backend.

NOTE: the NequIP calculator/training API is version-sensitive and this wrapper
is UNVERIFIED against a live install (nequip is not a hard dependency). The
import paths below follow the documented API and should be confirmed when the
`nequip` package is available.
"""

from __future__ import annotations

from atlas.active_learning.backends.ase_backend import ASECalculatorBackend


class NequIPBackend(ASECalculatorBackend):
    """E(3)-equivariant NequIP models, used via the NequIP ASE calculator."""

    name = 'nequip'
    model_extension = '.pth'
    required_package = 'nequip'
    #: model architecture selected in the nequip training config
    architecture = 'nequip'

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        from nequip.ase import NequIPCalculator

        # nequip >= 0.7 uses compiled models; <= 0.6 uses deployed .pth models.
        if hasattr(NequIPCalculator, 'from_compiled_model'):
            try:
                return NequIPCalculator.from_compiled_model(str(model), device=device)
            except Exception:
                pass
        return NequIPCalculator.from_deployed_model(model_path=str(model), device=device)

    def train_calcjob_entry_point(self) -> str:
        # The matching AiiDA CalcJob (wrapping `nequip-train`) is added in the
        # NequIP training-integration PR.
        return 'nequip-train'
