"""Reports page — training report, error analysis, latent space gallery."""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.widgets.prereq_banner import PrereqBanner


class ReportsPage(WorkflowPage):
    """AL training report, error analysis, and latent space gallery."""

    DISPLAY_NAME = 'Reports'
    NAVIGATION_KEY = 'reports'

    def __init__(
        self,
        project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        parent=None,
    ):
        super().__init__(
            project,
            schema_data,
            application_font,
            log,
            navigate,
            notification,
            parent,
        )

        from atlas.core.gui.widgets.error_analysis_panel import ErrorAnalysisPanel
        from atlas.core.gui.widgets.latent_space_gallery import LatentSpaceGallery
        from atlas.core.gui.widgets.training_report_panel import TrainingReportPanel

        self._training_panel = TrainingReportPanel(project)
        self._error_panel = ErrorAnalysisPanel(project)
        self._gallery_panel = LatentSpaceGallery(project)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._training_panel, 'Training Report')
        self.tabs.addTab(self._error_panel, 'Error Analysis')
        self.tabs.addTab(self._gallery_panel, 'Latent Space')

        self.prereq_banner = PrereqBanner(
            message=(
                'No AL runs to report on yet. Start an active learning loop '
                'to populate this page.'
            ),
            action_label='Go to Active Learning',
            on_action=lambda: self._navigate('al'),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.prereq_banner)
        layout.addWidget(self.tabs, 1)

        self._update_prereq_banner()

    def on_shown(self) -> None:
        self._update_prereq_banner()
        self._training_panel.refresh()
        self._error_panel.refresh()
        self._gallery_panel.refresh()

    def _update_prereq_banner(self) -> None:
        state = self.project.workflow_state()
        al_status = state['stages']['al']['status']
        self.prereq_banner.setVisible(al_status in ('empty',))
