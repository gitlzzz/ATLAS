"""Latent space gallery, browse saved plots from AL iterations.

Scans the project directory for latent space plot images generated
during active learning and displays them in a navigable gallery.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}


class LatentSpaceGallery(QWidget):
    """Browse latent space images from AL iterations."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._images: list[Path] = []

        # --- image list ---
        self._list = QListWidget()
        self._list.setMaximumWidth(280)
        self._list.currentRowChanged.connect(self._on_selected)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        list_panel = QWidget()
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel('Latent Space Plots'))
        list_layout.addWidget(self._list, 1)
        list_layout.addWidget(refresh_btn)

        # --- image viewer ---
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet('background: white;')

        self._info_label = QLabel()
        self._info_label.setStyleSheet('color: #555; padding: 4px 8px;')

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self._image_label, 1)
        viewer_layout.addWidget(self._info_label)

        # --- splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(list_panel)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- empty state ---
        self._empty_label = QLabel(
            'No latent space plots found.\n\n'
            'Latent space images are generated during active learning\n'
            'iterations and saved in the project directory.\n'
            'Run an AL loop to generate them.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            'color: #6c757d; padding: 40px; font-size: 13px;'
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._empty_label)
        self._splitter = splitter

    # ---------------------------------------------------------- public

    def refresh(self) -> None:
        self._scan_images()
        self._list.clear()

        if not self._images:
            self._splitter.setVisible(False)
            self._empty_label.setVisible(True)
            self._image_label.clear()
            self._info_label.clear()
            return

        self._splitter.setVisible(True)
        self._empty_label.setVisible(False)

        for img_path in self._images:
            item = QListWidgetItem(img_path.name)
            item.setToolTip(str(img_path))
            self._list.addItem(item)

        self._list.setCurrentRow(0)

    # ---------------------------------------------------------- internals

    def _scan_images(self) -> None:
        self._images = []
        search_dirs = [
            self._project.dir,
            self._project.dir / 'databases',
            self._project.dir / 'results',
        ]

        seen = set()
        for base in search_dirs:
            if not base.exists():
                continue
            for path in sorted(base.rglob('*')):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                name_lower = path.name.lower()
                if not any(
                    kw in name_lower
                    for kw in (
                        'latent',
                        'descriptor',
                        'concave',
                        'boundary',
                        'domain',
                        'ls_plot',
                        'latent_space',
                    )
                ):
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    self._images.append(path)

        self._images.sort(key=lambda p: p.name)

    def _on_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._images):
            self._image_label.clear()
            self._info_label.clear()
            return

        path = self._images[row]
        self._info_label.setText(f'{path.name} ,  {path.parent}')

        if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self._image_label.setPixmap(scaled)
            else:
                self._image_label.setText('Could not load image.')
        else:
            self._image_label.setText(
                f'Preview not available for {path.suffix} files.\nFile: {path}'
            )
