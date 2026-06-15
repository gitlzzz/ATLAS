"""Tests for ActiveLearningPage wiring (headless)."""

from __future__ import annotations

import os

import pytest
import yaml

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytestmark = pytest.mark.skip(
    reason='Skipping GUI tests over SSH session due to D-Bus timeout'
)


@pytest.fixture(scope='module')
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        # 2. Force headless mode natively via Qt's internal argv
        # (The first string is a dummy program name required by Qt)
        app = QApplication(['dummy_test_app', '-platform', 'offscreen'])
    return app


@pytest.fixture()
def project(tmp_path):
    from atlas.core.gui.project import Project

    return Project.create(tmp_path, 'al_test')


@pytest.fixture()
def schema_data():
    from importlib.resources import files

    schema_path = files('atlas.data').joinpath('config_schema.yaml')
    data = yaml.safe_load(schema_path.read_text())
    return data


@pytest.fixture()
def al_page(qapp, project, schema_data):
    from PySide6.QtGui import QFont

    from atlas.core.gui.pages.active_learning import ActiveLearningPage

    return ActiveLearningPage(
        project=project,
        schema_data=schema_data,
        application_font=QFont(),
        log=lambda msg: None,
        navigate=lambda key: None,
    )


def test_al_page_has_tabs(al_page):
    assert al_page.tabs.count() == 4
    assert al_page.tabs.tabText(0) == 'Config'
    assert al_page.tabs.tabText(1) == 'Monitor'
    assert al_page.tabs.tabText(2) == 'Outputs'
    assert al_page.tabs.tabText(3) == 'Runs'


def test_al_config_has_sub_tabs(al_page):
    from atlas.core.gui.pages.active_learning import AL_TABS

    panel = al_page.config_panel
    assert panel._form_tabs is not None
    assert panel._form_tabs.count() == len(AL_TABS)
    for i, (label, _) in enumerate(AL_TABS):
        assert panel._form_tabs.tabText(i) == label


def test_al_prereq_banner_visible_no_db(al_page):
    al_page._update_prereq_banner()
    assert not al_page.prereq_banner.isHidden()


def test_al_preflight_catches_missing_fields():
    from atlas.core.gui.pages.active_learning import _preflight_check

    problems = _preflight_check({}, '/nonexistent')
    assert any('aiida_profile' in p for p in problems)
    assert any('run_name' in p for p in problems)


def test_al_preflight_catches_missing_db_path():
    from atlas.core.gui.pages.active_learning import _preflight_check

    parsed = {
        'active_learning': {
            'aiida_profile': 'test',
            'run_name': 'run1',
            'results_dir': '/tmp',
            'init_db_path': '/nonexistent/path',
        },
    }
    problems = _preflight_check(parsed, '/tmp')
    assert any('does not exist' in p for p in problems)


def test_al_runs_panel_empty(al_page):
    panel = al_page._runs_panel
    assert panel._table.rowCount() == 0
    assert 'No AL runs' in panel._summary.text()


def test_al_run_recording(al_page):
    proj = al_page.project
    run_id = proj.record_al_submission(
        base_workchain_pk=None,
        config_snapshot_id=None,
    )
    assert run_id >= 1

    proj.update_al_run_status(run_id, 'completed', finished=True)

    runs = proj.list_al_runs()
    assert len(runs) == 1
    assert runs[0]['status'] == 'completed'
    assert runs[0]['finished_at'] is not None

    al_page._runs_panel.refresh()
    assert al_page._runs_panel._table.rowCount() == 1


def test_al_run_counts(al_page):
    proj = al_page.project
    proj.record_al_submission(base_workchain_pk=None, config_snapshot_id=None)
    proj.record_al_submission(base_workchain_pk=None, config_snapshot_id=None)

    counts = proj.al_run_counts()
    assert counts.get('submitted', 0) >= 2
