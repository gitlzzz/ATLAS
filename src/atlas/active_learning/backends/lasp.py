"""LASP backend (scaffold).

LASP (SSW global optimization + its own NN potential) does not expose a standard
ASE calculator; it is driven through its own executables and I/O files. This is a
scaffold: it registers the ``lasp`` model_type and names its training CalcJob, but
building a calculator requires a dedicated LASP driver (added in the LASP PR).
"""

from __future__ import annotations

from atlas.active_learning.backends.base import MLIPBackend


class LASPBackend(MLIPBackend):
    """LASP models — scaffold pending a dedicated LASP driver."""

    name = 'lasp'
    model_extension = '.pot'

    def build_calculator(self, model=None, *, device='cpu', dtype='float64', enable_cueq=False, **kwargs):
        raise NotImplementedError(
            'LASP does not provide a standard ASE calculator; a dedicated LASP '
            'driver (file-based I/O) is required. This backend is currently a scaffold.'
        )

    def train_calcjob_entry_point(self) -> str:
        return 'lasp-train'
