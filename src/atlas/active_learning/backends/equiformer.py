"""EquiformerV2 / V3 backend (via the fairchem ecosystem).

Equiformer models are trained/served through `fairchem` (formerly OCP). The
calculator API changed across fairchem versions (OCPCalculator -> FAIRChemCalculator),
so both are attempted.

UNVERIFIED: fairchem-core was not installed (it downgrades torch to 2.8, pulls
e3nn 0.6 [breaks MACE], numpy 2 [breaks ATLAS] and click 8.4 [breaks aiida], plus
a very large dependency tree). It must run in its own environment/container;
confirm the calculator import there.
"""

from __future__ import annotations

from atlas.active_learning.backends.ase_backend import ASECalculatorBackend


class EquiformerBackend(ASECalculatorBackend):
    """EquiformerV2/V3 models via fairchem."""

    name = 'equiformer'
    model_extension = '.pt'
    required_package = 'fairchem'

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        cpu = device == 'cpu'
        try:
            from fairchem.core import OCPCalculator

            return OCPCalculator(checkpoint_path=str(model), cpu=cpu)
        except ImportError:
            # Newer fairchem exposes FAIRChemCalculator.
            from fairchem.core import FAIRChemCalculator

            return FAIRChemCalculator(str(model))

    def train_calcjob_entry_point(self) -> str:
        return 'equiformer-train'
