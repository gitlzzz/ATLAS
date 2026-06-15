"""Interactive latent-space boundary panel.

Loads ``latent_space.npz`` (produced by ``descriptors_concave_hull``)
and renders an interactive matplotlib scatter plot with the boundary
overlay (concave hull or morphological closing).  Pan, zoom, and hover
are provided via the standard matplotlib navigation toolbar.

Clicking a point in the scatter plot loads the corresponding structure
in a 3D viewer on the right-hand side.

Falls back to a static PNG viewer for projects generated before the
``.npz`` format was introduced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

LATENT_SPACE_FILE = 'latent_space.npz'
HULL_GLOBS = ['concave_hull*.png', 'comp_plot*.png', 'descriptors_concave_hull*.png']

_HOVER_RADIUS = 0.02
_CLICK_RADIUS = 0.03


class BoundaryPanel(QWidget):
    """Interactive latent-space scatter + hull boundary plot with structure detail."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._init_db_cache = None

        # Hover state
        self._hover_cid = None
        self._press_cid = None
        self._release_cid = None
        self._press_xy: tuple[float, float] | None = None
        self._hover_idx: int | None = None
        self._hover_highlight = None
        self._hover_annot = None
        self._hover_tree = None
        self._hover_ls = None
        self._hover_ids = None
        self._hover_ax = None

        # Selection state (click)
        self._sel_idx: int | None = None
        self._sel_highlight = None
        self._sel_annot = None

        # --- Interactive plot widgets ---
        self._fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._plot_container = QWidget()
        plot_layout = QVBoxLayout(self._plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas, 1)

        # --- Structure detail panel (right side) ---
        from atlas.core.gui.widgets.structure_viewer import StructureViewer

        self._viewer = StructureViewer()
        self._viewer.setMinimumWidth(280)

        self._detail_placeholder = QLabel(
            'Click a point in the latent space\nto view its structure.'
        )
        self._detail_placeholder.setAlignment(Qt.AlignCenter)
        self._detail_placeholder.setWordWrap(True)
        self._detail_placeholder.setStyleSheet('padding: 30px; font-size: 13px;')

        self._detail_container = QWidget()
        detail_layout = QVBoxLayout(self._detail_container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self._viewer)
        detail_layout.addWidget(self._detail_placeholder)
        self._viewer.setVisible(False)

        # Splitter: plot (left) + detail (right)
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(self._plot_container)
        self._main_splitter.addWidget(self._detail_container)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)

        # --- Legacy image viewer (for old projects without .npz) ---
        self._legacy_container = QWidget()
        legacy_layout = QVBoxLayout(self._legacy_container)
        legacy_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_legacy_image)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet('')
        scroll = QScrollArea()
        scroll.setWidget(self._image_label)
        scroll.setWidgetResizable(True)

        legacy_splitter = QSplitter(Qt.Horizontal)
        legacy_splitter.addWidget(self._list)
        legacy_splitter.addWidget(scroll)
        legacy_splitter.setStretchFactor(0, 1)
        legacy_splitter.setStretchFactor(1, 4)
        legacy_layout.addWidget(legacy_splitter, 1)

        # --- Empty state ---
        self._empty_label = QLabel(
            'No boundary data found.\n'
            'Run database generation with boundary determination enabled\n'
            'to see the latent-space plot here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('padding: 40px;')

        # --- Main layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter, 1)
        layout.addWidget(self._legacy_container, 1)
        layout.addWidget(self._empty_label)

        self._legacy_images: list[Path] = []

        self.refresh()

    # ---------------------------------------------------------- public API

    def refresh(self) -> None:
        self._init_db_cache = None

        npz_path = self._find_npz()
        if npz_path is not None:
            self._load_and_draw(npz_path)
            self._main_splitter.setVisible(True)
            self._legacy_container.setVisible(False)
            self._empty_label.setVisible(False)
            self._clear_selection()
            return

        legacy = self._find_legacy_images()
        if legacy:
            self._show_legacy_list(legacy)
            self._main_splitter.setVisible(False)
            self._legacy_container.setVisible(True)
            self._empty_label.setVisible(False)
            return

        self._main_splitter.setVisible(False)
        self._legacy_container.setVisible(False)
        self._empty_label.setVisible(True)

    def has_content(self) -> bool:
        if self._find_npz() is not None:
            return True
        return bool(self._find_legacy_images())

    # --------------------------------------------------------- npz loading

    def _find_npz(self) -> Path | None:
        for parent in (self._project.dir / 'databases', self._project.dir):
            p = parent / LATENT_SPACE_FILE
            if p.is_file():
                return p
        return None

    def _load_and_draw(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        ls = data['latent_space']
        unique_ids = data['unique_ids']

        method = 'concave_hull'
        if 'boundary_method' in data:
            method = str(data['boundary_method'][0])

        if method == 'morphological_closing':
            coords = data['boundary_coords']
            counts = data['boundary_counts']
            boundaries = np.split(coords, np.cumsum(counts)[:-1])
            disk_size = (
                int(data['morph_disk_size'][0]) if 'morph_disk_size' in data else 10
            )
            self._draw_morph(ls, boundaries, unique_ids, disk_size)
        else:
            hull = data['hull_vertices']
            alpha = float(data['alpha'][0])
            self._draw(ls, hull, unique_ids, alpha)

    # ------------------------------------------------------- info box placement

    @staticmethod
    def _best_corner(ax, ls: np.ndarray) -> tuple[float, float, str, str]:
        xmid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        ymid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

        corners = [
            (ls[:, 0] < xmid) & (ls[:, 1] >= ymid),
            (ls[:, 0] >= xmid) & (ls[:, 1] >= ymid),
            (ls[:, 0] < xmid) & (ls[:, 1] < ymid),
            (ls[:, 0] >= xmid) & (ls[:, 1] < ymid),
        ]
        positions = [
            (0.03, 0.97, 'top', 'left'),
            (0.97, 0.97, 'top', 'right'),
            (0.03, 0.03, 'bottom', 'left'),
            (0.97, 0.03, 'bottom', 'right'),
        ]
        counts = [int(np.sum(mask)) for mask in corners]
        best = int(np.argmin(counts))
        x, y, va, ha = positions[best]
        return x, y, va, ha

    # --------------------------------------------------------- hover + click

    def _setup_interaction(self, ax, ls, unique_ids) -> None:
        from scipy.spatial import cKDTree

        self._hover_tree = cKDTree(ls)
        self._hover_ls = ls
        self._hover_ids = unique_ids
        self._hover_ax = ax
        self._hover_idx = None
        self._sel_idx = None

        # Hover ring (orange, no fill)
        (self._hover_highlight,) = ax.plot(
            [],
            [],
            'o',
            ms=10,
            mec='#e85d04',
            mfc='none',
            mew=2.0,
            zorder=5,
            visible=False,
        )

        # Hover tooltip
        self._hover_annot = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=mpl.rcParams['figure.facecolor'],
                ec='#adb5bd',
                alpha=0.95,
                lw=0.8,
            ),
            arrowprops=dict(arrowstyle='->', color='#6c757d', lw=0.8),
            zorder=6,
            visible=False,
        )

        # Selection marker (teal, persistent until next click)
        (self._sel_highlight,) = ax.plot(
            [],
            [],
            'o',
            ms=12,
            mec='#2a9d8f',
            mfc='#2a9d8f44',
            mew=2.0,
            zorder=7,
            visible=False,
        )

        # Selection tooltip (persistent, same style but teal border)
        self._sel_annot = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=mpl.rcParams['figure.facecolor'],
                ec='#2a9d8f',
                alpha=0.95,
                lw=1.0,
            ),
            arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=1.0),
            zorder=8,
            visible=False,
        )

        # Disconnect old handlers
        for cid in (self._hover_cid, self._press_cid, self._release_cid):
            if cid is not None:
                self._canvas.mpl_disconnect(cid)

        self._hover_cid = self._canvas.mpl_connect(
            'motion_notify_event',
            self._on_mouse_move,
        )
        self._press_cid = self._canvas.mpl_connect(
            'button_press_event',
            self._on_mouse_press,
        )
        self._release_cid = self._canvas.mpl_connect(
            'button_release_event',
            self._on_mouse_release,
        )

        def format_coord(x, y):
            dist, idx = self._hover_tree.query([x, y])
            uid = unique_ids[idx] if idx < len(unique_ids) else '?'
            return f'x={x:.4f}  y={y:.4f}  [{uid}]'

        ax.format_coord = format_coord

    def _annot_offset(self, x: float, y: float) -> tuple[float, float]:
        """Return xytext offset that keeps the tooltip inside the axes."""
        ax = self._hover_ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xfrac = (x - xlim[0]) / (xlim[1] - xlim[0])
        yfrac = (y - ylim[0]) / (ylim[1] - ylim[0])
        ox = -14 if xfrac > 0.75 else 14
        oy = -14 if yfrac > 0.80 else 14
        return ox, oy

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is not self._hover_ax or self._hover_tree is None:
            if self._hover_idx is not None:
                self._hover_highlight.set_visible(False)
                self._hover_annot.set_visible(False)
                self._hover_idx = None
                self._canvas.draw_idle()
            return

        dist, idx = self._hover_tree.query([event.xdata, event.ydata])

        xlim = self._hover_ax.get_xlim()
        ylim = self._hover_ax.get_ylim()
        diag = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if dist > diag * _HOVER_RADIUS:
            if self._hover_idx is not None:
                self._hover_highlight.set_visible(False)
                self._hover_annot.set_visible(False)
                self._hover_idx = None
                self._canvas.draw_idle()
            return

        if idx == self._hover_idx:
            return

        self._hover_idx = idx
        x, y = self._hover_ls[idx]
        uid = self._hover_ids[idx] if idx < len(self._hover_ids) else '?'

        self._hover_highlight.set_data([x], [y])
        self._hover_highlight.set_visible(True)

        ox, oy = self._annot_offset(x, y)
        self._hover_annot.set_position((ox, oy))
        self._hover_annot.xy = (x, y)
        self._hover_annot.set_text(f'{uid}\nx={x:.4f}  y={y:.4f}')
        self._hover_annot.set_visible(True)

        self._canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        if event.button == 1 and event.xdata is not None:
            self._press_xy = (event.x, event.y)
        else:
            self._press_xy = None

    def _on_mouse_release(self, event) -> None:
        if event.button != 1 or self._press_xy is None:
            return
        if event.inaxes is not self._hover_ax or self._hover_tree is None:
            self._press_xy = None
            return

        dx = event.x - self._press_xy[0]
        dy = event.y - self._press_xy[1]
        self._press_xy = None
        if np.hypot(dx, dy) > 5:
            return

        dist, idx = self._hover_tree.query([event.xdata, event.ydata])

        xlim = self._hover_ax.get_xlim()
        ylim = self._hover_ax.get_ylim()
        diag = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if dist > diag * _CLICK_RADIUS:
            return

        self._sel_idx = idx
        x, y = self._hover_ls[idx]
        uid = str(self._hover_ids[idx]) if idx < len(self._hover_ids) else '?'

        self._sel_highlight.set_data([x], [y])
        self._sel_highlight.set_visible(True)

        ox, oy = self._annot_offset(x, y)
        self._sel_annot.set_position((ox, oy))
        self._sel_annot.xy = (x, y)
        self._sel_annot.set_text(f'{uid}\nx={x:.4f}  y={y:.4f}')
        self._sel_annot.set_visible(True)

        self._canvas.draw_idle()

        self._load_structure_for_id(uid, x, y)

    def _clear_selection(self) -> None:
        self._sel_idx = None
        if self._sel_highlight is not None:
            self._sel_highlight.set_visible(False)
        if self._sel_annot is not None:
            self._sel_annot.set_visible(False)
        self._viewer.clear()
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)

    # ----------------------------------------------------- structure loading

    def _load_structure_for_id(self, atl_id: str, lx: float, ly: float) -> None:
        try:
            df = self._get_init_db()
            if df is None:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            id_col = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
            match = df[df[id_col] == atl_id]
            if match.empty:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            row = match.iloc[0]
            struct = row.get('structure')
            if struct is None:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(struct)
            info = {
                'atl_id': atl_id,
                'phase': row.get('phase', ''),
                'struct_type': row.get('struct_type', ''),
                'calc_energy': row.get('calc_energy'),
            }
            self._detail_placeholder.setVisible(False)
            self._viewer.setVisible(True)
            self._viewer.set_atoms(atoms, info)

        except Exception:
            self._show_detail_fallback(atl_id, lx, ly)

    def _show_detail_fallback(self, atl_id: str, lx: float, ly: float) -> None:
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)
        self._detail_placeholder.setText(
            f'Structure: {atl_id}\n'
            f'Latent coords: ({lx:.4f}, {ly:.4f})\n\n'
            f'Could not load 3D structure.\n'
            f'The database may not be available.'
        )

    def _get_init_db(self):
        if self._init_db_cache is not None:
            return self._init_db_cache
        path = self._project.init_db_path()
        if not path.exists():
            return None
        from atlas.core.initial_db import InitialDatabase

        loaded = InitialDatabase.load_database(path)
        df = getattr(loaded, 'df', loaded)
        if df is None or getattr(df, 'empty', True):
            return None
        self._init_db_cache = df
        return df

    # ----------------------------------------------------------- plotting

    def _draw(
        self,
        ls: np.ndarray,
        hull: np.ndarray,
        unique_ids: np.ndarray,
        alpha: float,
    ) -> None:
        self._fig.clear()
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        ax = self._fig.add_subplot(111)

        ax.scatter(
            ls[:, 0],
            ls[:, 1],
            s=4,
            c='steelblue',
            alpha=0.6,
            edgecolors='none',
            label=f'Structures ({len(ls):,})',
            rasterized=True,
            zorder=1,
        )

        ax.plot(
            hull[:, 0],
            hull[:, 1],
            '-',
            color='#cc241d',
            lw=2,
            label='Concave hull',
            zorder=2,
        )

        try:
            from shapely.geometry import Polygon

            hull_area = Polygon(hull).area
            area_str = f'\nHull area: {hull_area:.2e}'
        except Exception:
            area_str = ''

        info_text = f'Alpha: {alpha:.2f}{area_str}\nStructures: {len(ls):,}'
        bx, by, va, ha = self._best_corner(ax, ls)
        ax.text(
            bx,
            by,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(
                boxstyle='round', facecolor=mpl.rcParams['figure.facecolor'], alpha=0.85
            ),
            zorder=4,
        )

        ax.set_xlabel('Embedded dimension 1')
        ax.set_ylabel('Embedded dimension 2')
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._setup_interaction(ax, ls, unique_ids)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def _draw_morph(
        self,
        ls: np.ndarray,
        boundaries: list[np.ndarray],
        unique_ids: np.ndarray,
        disk_size: int,
    ) -> None:
        self._fig.clear()
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        ax = self._fig.add_subplot(111)

        ax.scatter(
            ls[:, 0],
            ls[:, 1],
            s=4,
            c='steelblue',
            alpha=0.6,
            edgecolors='none',
            label=f'Structures ({len(ls):,})',
            rasterized=True,
            zorder=1,
        )

        for i, contour in enumerate(boundaries):
            if len(contour) < 2:
                continue
            closed = np.vstack([contour, contour[:1]])
            label = 'Boundary' if i == 0 else None
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                '-',
                color='#cc241d',
                lw=2,
                label=label,
                zorder=2,
            )

        area_str = ''
        try:
            from shapely.geometry import Polygon

            total_area = sum(Polygon(c).area for c in boundaries if len(c) >= 3)
            area_str = f'\nTotal area: {total_area:.2e}'
        except Exception:
            pass

        info_text = (
            f'Method: morphological closing\n'
            f'Disk size: {disk_size}{area_str}\n'
            f'Regions: {len(boundaries)}\n'
            f'Structures: {len(ls):,}'
        )
        bx, by, va, ha = self._best_corner(ax, ls)
        ax.text(
            bx,
            by,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(
                boxstyle='round', facecolor=mpl.rcParams['figure.facecolor'], alpha=0.85
            ),
            zorder=4,
        )

        ax.set_xlabel('Embedded dimension 1')
        ax.set_ylabel('Embedded dimension 2')
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._setup_interaction(ax, ls, unique_ids)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    # ------------------------------------------------ legacy image viewer

    def _find_legacy_images(self) -> list[Path]:
        images: list[Path] = []
        for parent in (self._project.dir / 'databases', self._project.dir):
            if parent.is_dir():
                for pattern in HULL_GLOBS:
                    for p in sorted(parent.glob(pattern)):
                        if p not in images:
                            images.append(p)
        return images

    def _show_legacy_list(self, images: list[Path]) -> None:
        self._list.clear()
        self._legacy_images = images
        self._image_label.clear()
        for img in images:
            self._list.addItem(img.name)
        self._list.setCurrentRow(0)

    def _show_legacy_image(self, row: int) -> None:
        if row < 0 or row >= len(self._legacy_images):
            return
        path = self._legacy_images[row]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._image_label.setText(f'Cannot load: {path.name}')
            return
        available = self._image_label.parent().size()
        scaled = pixmap.scaled(
            available,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
