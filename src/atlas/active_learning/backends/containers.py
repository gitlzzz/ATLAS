"""Resolve per-backend container settings.

Because MLIP frameworks have mutually-incompatible dependency stacks (e.g. MACE
needs e3nn 0.4.4 while NequIP needs e3nn>=0.6; DeepMD/fairchem need numpy>=2),
each backend's train/MD code should run in its own container image. This helper
lets ``container_settings`` carry an optional per-backend override:

```toml
[active_learning.container_settings]
use_container = true
image_name = "atlas_mace.sif"          # default / MACE image
engine_command = "singularity exec --bind .:/atl_data --nv {image_name}"

[active_learning.container_settings.images.nequip]
image_name = "atlas_nequip.sif"

[active_learning.container_settings.images.deepmd]
image_name = "atlas_deepmd.sif"
engine_command = "singularity exec --bind .:/atl_data --nv {image_name}"
```

Configs without an ``images`` section behave exactly as before (single image).
"""

from __future__ import annotations


def resolve_container_settings(
    container_settings: dict | None, model_type: str = 'mace'
) -> dict:
    """Return the effective container settings for ``model_type``.

    Per-backend overrides under ``container_settings['images'][model_type]`` are
    merged over the global settings; the ``images`` key itself is dropped from
    the result. When there is no override, the global settings are returned
    unchanged (minus ``images``), preserving the previous single-image behavior.
    """
    settings = dict(container_settings or {})
    images = settings.pop('images', None) or {}
    override = images.get(model_type)
    if override:
        settings.update(override)
    return settings
