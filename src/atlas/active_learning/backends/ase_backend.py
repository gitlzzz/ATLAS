"""Shared base for backends that expose their model through an ASE calculator.

The ASE ``Calculator`` is the universal inference interface across MLIP
frameworks, so most backends only need to say *how to construct their ASE
calculator*. This base implements :meth:`build_calculator` on top of a single
subclass hook, :meth:`_make_ase_calculator`, and turns a missing framework into
a clear, actionable error instead of an opaque ``ImportError``.
"""

from __future__ import annotations

from atlas.active_learning.backends.base import MLIPBackend


class ASECalculatorBackend(MLIPBackend):
    """Base for backends whose model is used via an ASE calculator."""

    #: Import name of the python package the backend needs (for error messages).
    required_package: str = ''

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        """Construct and return the framework's ASE calculator. Override in subclasses."""
        raise NotImplementedError

    def build_calculator(
        self, model=None, *, device='cpu', dtype='float64', enable_cueq=False, **kwargs
    ):
        try:
            return self._make_ase_calculator(
                model, device=device, dtype=dtype, **kwargs
            )
        except ImportError as exc:
            pkg = self.required_package or self.name
            raise ImportError(
                f"The '{self.name}' MLIP backend requires the '{pkg}' package, which "
                f"is not installed. Install it to use model_type='{self.name}'. "
                f'Original import error: {exc}'
            ) from exc
