"""Orb backend (orb-models).

Orb ships pretrained universal potentials with an ASE calculator; ``model`` may
name a pretrained potential (e.g. ``"orb-v2"``, ``"orb-d3-xs-v2"``) or point to a
checkpoint. Verified against orb-models v0.5.5.
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

    def _resolve_pretrained_factory(self, variant):
        """Return the orb-models loader for a pretrained potential name."""
        from orb_models.forcefield import pretrained

        # v3 potentials are registered in ORB_PRETRAINED_MODELS (dashed keys);
        # older ones (orb_v2, orb_d3_*_v2) are module-level functions (underscores).
        registry = getattr(pretrained, 'ORB_PRETRAINED_MODELS', {})
        if variant in registry:
            return registry[variant]
        func_name = variant.replace('-', '_')
        if hasattr(pretrained, func_name):
            return getattr(pretrained, func_name)
        raise ValueError(
            f"Unknown Orb pretrained potential '{variant}'. "
            f'Available: {sorted(registry)} plus module loaders like orb_v2.'
        )

    def _make_ase_calculator(self, model=None, *, device='cpu', dtype='float64', **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # torch.compile is opt-in here: it is faster on GPU but needs a working
        # C++ toolchain (torch inductor), which isn't guaranteed on every host.
        compile_model = kwargs.get('compile', False)

        foundation = self.parse_foundation_id(model)
        if foundation is not None:
            _, variant = foundation
            factory = self._resolve_pretrained_factory(variant)
            orbff = factory(device=device, compile=compile_model)
        elif model:
            orbff = pretrained.orb_v2(
                weights_path=str(model), device=device, compile=compile_model
            )
        else:
            orbff = pretrained.orb_v2(device=device, compile=compile_model)
        return ORBCalculator(orbff, device=device)

    def train_calcjob_entry_point(self) -> str:
        return 'orb-train'
