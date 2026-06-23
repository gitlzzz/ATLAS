"""Smoke tests for the StructureViewer widget (headless)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(scope='module')
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture()
def cu_atoms():
    from ase.build import bulk
    return bulk('Cu', 'fcc', a=3.6)


def test_viewer_placeholder_visible(qapp):
    from atlas.core.gui.widgets.structure_viewer import StructureViewer

    viewer = StructureViewer()
    assert not viewer._placeholder.isHidden()
    assert viewer._canvas.isHidden()


def test_viewer_shows_atoms(qapp, cu_atoms):
    from atlas.core.gui.widgets.structure_viewer import StructureViewer

    viewer = StructureViewer()
    viewer.set_atoms(cu_atoms, info={'phase': 'fcc', 'struct_type': 'bulk'})
    assert viewer._placeholder.isHidden()
    assert 'Cu' in viewer._info_label.text()
    assert 'fcc' in viewer._info_label.text()


def test_viewer_rotation_change(qapp, cu_atoms):
    from atlas.core.gui.widgets.structure_viewer import StructureViewer

    viewer = StructureViewer()
    viewer.set_atoms(cu_atoms)
    viewer._rotation_combo.setCurrentText('Top (z)')
    assert viewer._placeholder.isHidden()


def test_viewer_clear(qapp, cu_atoms):
    from atlas.core.gui.widgets.structure_viewer import StructureViewer

    viewer = StructureViewer()
    viewer.set_atoms(cu_atoms)
    viewer.clear()
    assert not viewer._placeholder.isHidden()
