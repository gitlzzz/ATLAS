"""Tests for per-backend container-settings resolution."""

from atlas.active_learning.backends import resolve_container_settings

GLOBAL = {
    'use_container': True,
    'image_name': 'atlas_mace.sif',
    'engine_command': 'singularity exec --bind .:/atl_data {image_name}',
    'prepend_text': 'module load singularity',
}


class TestResolveContainerSettings:
    def test_no_images_section_is_passthrough(self):
        # Previous single-image behavior: settings unchanged (minus 'images').
        assert resolve_container_settings(GLOBAL, 'mace') == GLOBAL

    def test_none_input(self):
        assert resolve_container_settings(None, 'mace') == {}

    def test_backend_without_override_uses_global(self):
        cfg = dict(GLOBAL, images={'nequip': {'image_name': 'atlas_nequip.sif'}})
        # 'mace' has no override -> global image, and 'images' key is stripped.
        resolved = resolve_container_settings(cfg, 'mace')
        assert resolved['image_name'] == 'atlas_mace.sif'
        assert 'images' not in resolved

    def test_backend_override_merges_over_global(self):
        cfg = dict(
            GLOBAL,
            images={
                'nequip': {'image_name': 'atlas_nequip.sif'},
                'deepmd': {
                    'image_name': 'atlas_deepmd.sif',
                    'engine_command': 'docker run {image_name}',
                },
            },
        )
        nq = resolve_container_settings(cfg, 'nequip')
        assert nq['image_name'] == 'atlas_nequip.sif'
        # non-overridden keys fall back to the global settings
        assert nq['engine_command'] == GLOBAL['engine_command']
        assert nq['use_container'] is True
        assert 'images' not in nq

        dp = resolve_container_settings(cfg, 'deepmd')
        assert dp['image_name'] == 'atlas_deepmd.sif'
        assert dp['engine_command'] == 'docker run {image_name}'

    def test_does_not_mutate_input(self):
        cfg = dict(GLOBAL, images={'nequip': {'image_name': 'x.sif'}})
        _ = resolve_container_settings(cfg, 'nequip')
        assert 'images' in cfg  # original dict untouched
