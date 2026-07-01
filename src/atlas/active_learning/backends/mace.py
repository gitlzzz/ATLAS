"""MACE backend — wraps ATLAS's existing MACE calculator/training/LAMMPS logic."""

from __future__ import annotations

from atlas.active_learning.backends.base import MLIPBackend

_MP_VARIANTS = ('small', 'medium', 'large', 'medium-mpa-0')
_OFF_VARIANTS = ('small', 'medium', 'large')


class MaceBackend(MLIPBackend):
    """MLIP backend for MACE models (trained ``.model`` files or foundation ids)."""

    name = 'mace'
    model_extension = '.model'

    def parse_foundation_id(self, model):
        if not isinstance(model, str):
            return None
        if model.startswith('mace:mp-'):
            return ('mp', model.split('mace:mp-')[-1])
        if model.startswith('mace:off-'):
            return ('off', model.split('mace:off-')[-1])
        return None

    def build_calculator(
        self, model=None, *, device='cpu', dtype='float64', enable_cueq=False, **kwargs
    ):
        """Return a MACE ASE calculator.

        ``model`` may be a trained model path or a foundation id such as
        ``"mace:mp-small"`` / ``"mace:off-medium"``.
        """
        foundation = self.parse_foundation_id(model)
        if foundation is not None:
            family, variant = foundation
            if family == 'mp':
                from mace.calculators import mace_mp

                return mace_mp(model=variant, device=device, default_dtype=dtype)
            from mace.calculators import mace_off

            return mace_off(model=variant, device=device, default_dtype=dtype)

        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=model,
            device=device,
            default_dtype=dtype,
            enable_cueq=enable_cueq,
        )

    def train_calcjob_entry_point(self) -> str:
        return 'mace-train'

    def prepare_training_input(self, structures, out_path, **kwargs):
        from pathlib import Path

        from atlas.active_learning import conversion as atl_conv

        atl_conv.gen_mace_train_structure_list(structures, str(out_path), **kwargs)
        return Path(out_path)

    def get_lammps_model(self, model_path):
        from atlas.active_learning.active_learning_utils import create_mace_lammps_model

        return create_mace_lammps_model(model_path)
