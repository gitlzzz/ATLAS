"""DeepMD (DeePMD-kit) backend.

Uses DeePMD-kit's ASE calculator and the `dp train` CLI (JSON config). DeepMD
models run natively in LAMMPS, so `get_lammps_model` returns the model path
directly.

API verified against deepmd-kit 3.1.3: ``deepmd.calculator.DP(model, ...)``
matches the construction below. Note: deepmd-kit requires ``numpy>=2``, which is
INCOMPATIBLE with ATLAS's ``numpy<2`` pin, so it must run in its own
environment/container. A live single-point needs a frozen model (`dp freeze`),
so only the API is verified here.
"""

from __future__ import annotations

from atlas.active_learning.backends.ase_backend import ASECalculatorBackend


class DeepMDBackend(ASECalculatorBackend):
    """DeePMD-kit models."""

    name = 'deepmd'
    model_extension = '.pb'
    required_package = 'deepmd'

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        from deepmd.calculator import DP

        return DP(model=str(model))

    def train_calcjob_entry_point(self) -> str:
        return 'deepmd-train'

    def get_lammps_model(self, model_path):
        # A frozen DeepMD graph (.pb) is consumed directly by LAMMPS pair_style deepmd.
        return model_path
