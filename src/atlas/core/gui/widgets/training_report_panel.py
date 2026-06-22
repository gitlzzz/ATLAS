"""Training report panel, cross-run comparison and export.

Renders an AL training report with run selector, summary table,
overlaid RMSE charts for comparing runs, and export-to-PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project
from atlas.core.gui.widgets.al_monitor_panel import parse_al_log

COLORS = ['#6aa1f4', '#ffa6c8', '#3ed04e', '#e50e3f', '#fed65b', '#7532c8']
CHEM_ACC_E = 43.37


class _Chart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 3.5), dpi=100, tight_layout=True)
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    @property
    def fig(self) -> Figure:
        return self._fig

    def redraw(self) -> None:
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()


class TrainingReportPanel(QWidget):
    """Cross-run training report with comparison charts and summary table."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        # --- run selector ---
        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(200)
        self._run_combo.currentIndexChanged.connect(lambda _: self._reload())

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        export_btn = QPushButton('Export PNG…')
        export_btn.setFixedWidth(120)
        export_btn.clicked.connect(self._export_png)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel('AL Run:'))
        top_bar.addWidget(self._run_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(export_btn)
        top_bar.addWidget(refresh_btn)

        # --- summary table (all runs) ---
        self._summary_table = QTableWidget()
        self._summary_table.setColumnCount(7)
        self._summary_table.setHorizontalHeaderLabels(
            [
                'Run',
                'Status',
                'Iterations',
                'Final RMSE E',
                'Final RMSE F',
                'Final DB Size',
                'DB Growth',
            ]
        )
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._summary_table.setAlternatingRowColors(True)
        self._summary_table.setMaximumHeight(180)

        # --- 4-panel charts ---
        self._chart_db = _Chart()
        self._chart_rmse_e = _Chart()
        self._chart_rmse_f = _Chart()
        self._chart_combined = _Chart()

        chart_top = QSplitter(Qt.Horizontal)
        chart_top.addWidget(self._chart_db)
        chart_top.addWidget(self._chart_combined)

        chart_bottom = QSplitter(Qt.Horizontal)
        chart_bottom.addWidget(self._chart_rmse_e)
        chart_bottom.addWidget(self._chart_rmse_f)

        chart_grid = QSplitter(Qt.Vertical)
        chart_grid.addWidget(chart_top)
        chart_grid.addWidget(chart_bottom)

        # --- iteration detail table ---
        self._iter_table = QTableWidget()
        self._iter_table.setColumnCount(7)
        self._iter_table.setHorizontalHeaderLabels(
            [
                'Iteration',
                'Train DB',
                'Seed DB',
                'RMSE E (meV)',
                'RMSE F (meV/A)',
                'DB Delta',
                'Status',
            ]
        )
        self._iter_table.horizontalHeader().setStretchLastSection(True)
        self._iter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._iter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._iter_table.setAlternatingRowColors(True)

        # --- layout ---
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(chart_grid)
        main_splitter.addWidget(self._iter_table)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._summary_table)
        content_layout.addWidget(main_splitter, 1)

        self._empty_label = QLabel(
            'No AL runs available.\n'
            'Complete an active learning run to generate a training report.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('color: #6c757d; padding: 40px;')

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._content, 1)
        layout.addWidget(self._empty_label)

        self._all_stats: dict[int, dict] = {}

    # ---------------------------------------------------------- public

    def refresh(self) -> None:
        self._refresh_run_combo()
        self._refresh_summary_table()
        self._reload()

    # ---------------------------------------------------------- internals

    def _refresh_run_combo(self) -> None:
        current_id = self._run_combo.currentData()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        self._run_combo.addItem('All runs (compare)', -1)
        runs = self._project.list_al_runs()
        for r in runs:
            rid = r['id']
            status = r.get('status', '')
            started = (r.get('started_at') or '')[:16]
            self._run_combo.addItem(f'#{rid}  {status}  ({started})', rid)
        if current_id is not None:
            for i in range(self._run_combo.count()):
                if self._run_combo.itemData(i) == current_id:
                    self._run_combo.setCurrentIndex(i)
                    break
        self._run_combo.blockSignals(False)

    def _refresh_summary_table(self) -> None:
        runs = self._project.list_al_runs()
        self._all_stats.clear()

        self._summary_table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            rid = r['id']
            stats = self._gather_stats(rid)
            self._all_stats[rid] = stats

            finished = sorted(
                (s for s in stats.values() if s.get('finished')),
                key=lambda s: s['it_idx'],
            )

            n_it = len([s for s in finished if s['it_idx'] > 0])
            final_e = finished[-1].get('mace_e') if finished else None
            final_f = finished[-1].get('mace_f') if finished else None
            final_db = finished[-1].get('train_db_size') if finished else None
            first_db = finished[0].get('train_db_size') if finished else None

            if final_db and first_db and first_db > 0:
                growth = f'+{final_db - first_db:,}'
            else:
                growth = '—'

            self._summary_table.setItem(i, 0, _item(f'#{rid}'))
            status_item = _item(r.get('status', ''))
            if r.get('status') == 'completed':
                status_item.setForeground(Qt.darkGreen)
            elif r.get('status') in ('errored', 'stopped'):
                status_item.setForeground(Qt.red)
            self._summary_table.setItem(i, 1, status_item)
            self._summary_table.setItem(i, 2, _item(str(n_it)))
            self._summary_table.setItem(
                i,
                3,
                _item(f'{final_e:.2f}' if final_e else '—'),
            )
            self._summary_table.setItem(
                i,
                4,
                _item(f'{final_f:.1f}' if final_f else '—'),
            )
            self._summary_table.setItem(
                i,
                5,
                _item(f'{final_db:,}' if final_db else '—'),
            )
            self._summary_table.setItem(i, 6, _item(growth))

    def _reload(self) -> None:
        if not self._all_stats:
            self._content.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._content.setVisible(True)
        self._empty_label.setVisible(False)

        selected_id = self._run_combo.currentData()
        if selected_id == -1:
            self._draw_comparison()
            self._iter_table.setRowCount(0)
        else:
            stats = self._all_stats.get(selected_id, {})
            self._draw_single_run(selected_id, stats)
            self._populate_iter_table(stats)

    def _gather_stats(self, run_id: int) -> dict:
        log_path = self._find_log_for_run(run_id)
        if log_path and log_path.exists():
            stats = parse_al_log(log_path)
            if stats:
                return stats
        rows = self._project.list_al_iterations(run_id)
        if not rows:
            return {}
        stats = {}
        for r in rows:
            it = r['iteration']
            stats[it] = {
                'it_idx': it,
                'mace_e': r.get('rmse_e'),
                'mace_f': r.get('rmse_f'),
                'train_db_size': r.get('db_size'),
                'seed_gen_db_size': None,
                'finished': True,
            }
        return stats

    def _find_log_for_run(self, run_id: int) -> Path | None:
        log_dir = self._project.dir / 'logs'
        if not log_dir.is_dir():
            return None
        candidates = sorted(
            log_dir.glob('atl_active_learning.*.log'),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    # ---------------------------------------------------------- comparison

    def _draw_comparison(self) -> None:
        run_ids = sorted(self._all_stats.keys())
        if not run_ids:
            return

        # DB evolution, bar chart of final DB sizes per run
        fig = self._chart_db.fig
        fig.clear()
        ax = fig.add_subplot(111)
        labels, sizes = [], []
        for rid in run_ids:
            stats = self._all_stats[rid]
            finished = sorted(
                (s for s in stats.values() if s.get('train_db_size') is not None),
                key=lambda s: s['it_idx'],
            )
            if finished:
                labels.append(f'#{rid}')
                sizes.append(finished[-1]['train_db_size'])
        if sizes:
            x = np.arange(len(labels))
            colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
            ax.bar(x, sizes, color=colors, edgecolor='#282828', linewidth=0.8)
            ax.set_xticks(x, labels=labels)
            ax.set_ylabel('Final DB Size')
            ax.set_title('Final Database Size', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_db.redraw()

        # RMSE E, overlaid lines per run
        fig = self._chart_rmse_e.fig
        fig.clear()
        ax = fig.add_subplot(111)
        has_data = False
        for i, rid in enumerate(run_ids):
            stats = self._all_stats[rid]
            entries = sorted(
                (
                    s
                    for s in stats.values()
                    if s.get('mace_e') is not None and s['it_idx'] > 0
                ),
                key=lambda s: s['it_idx'],
            )
            if entries:
                has_data = True
                x = [s['it_idx'] for s in entries]
                e = [s['mace_e'] for s in entries]
                ax.plot(
                    x,
                    e,
                    'o-',
                    color=COLORS[i % len(COLORS)],
                    markersize=4,
                    label=f'#{rid}',
                )
        if has_data:
            ax.axhline(y=CHEM_ACC_E, color='#28282855', ls='--', lw=1)
            ax.text(
                ax.get_xlim()[0], CHEM_ACC_E + 1, 'Chem. Acc.', fontsize=7, color='#888'
            )
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('RMSE E [meV/atom]')
            ax.set_title('Energy RMSE', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_rmse_e.redraw()

        # RMSE F, overlaid lines per run
        fig = self._chart_rmse_f.fig
        fig.clear()
        ax = fig.add_subplot(111)
        has_data = False
        for i, rid in enumerate(run_ids):
            stats = self._all_stats[rid]
            entries = sorted(
                (
                    s
                    for s in stats.values()
                    if s.get('mace_f') is not None and s['it_idx'] > 0
                ),
                key=lambda s: s['it_idx'],
            )
            if entries:
                has_data = True
                x = [s['it_idx'] for s in entries]
                f = [s['mace_f'] for s in entries]
                ax.plot(
                    x,
                    f,
                    's-',
                    color=COLORS[i % len(COLORS)],
                    markersize=4,
                    label=f'#{rid}',
                )
        if has_data:
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('RMSE F [meV/A]')
            ax.set_title('Force RMSE', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_rmse_f.redraw()

        # Combined, iteration count per run
        fig = self._chart_combined.fig
        fig.clear()
        ax = fig.add_subplot(111)
        labels, iters = [], []
        for rid in run_ids:
            stats = self._all_stats[rid]
            n_it = len(
                [s for s in stats.values() if s.get('finished') and s['it_idx'] > 0]
            )
            labels.append(f'#{rid}')
            iters.append(n_it)
        if iters:
            x = np.arange(len(labels))
            colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
            ax.bar(x, iters, color=colors, edgecolor='#282828', linewidth=0.8)
            ax.set_xticks(x, labels=labels)
            ax.set_ylabel('Completed Iterations')
            ax.set_title('Iterations per Run', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_major_locator(
                __import__('matplotlib.ticker', fromlist=['MaxNLocator']).MaxNLocator(
                    integer=True
                )
            )
        self._chart_combined.redraw()

    # ---------------------------------------------------------- single run

    def _draw_single_run(self, run_id: int, stats: dict) -> None:
        if not stats:
            for chart in (
                self._chart_db,
                self._chart_rmse_e,
                self._chart_rmse_f,
                self._chart_combined,
            ):
                chart.fig.clear()
                ax = chart.fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    'No data',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    color='#6c757d',
                )
                ax.set_axis_off()
                chart.redraw()
            return

        entries = sorted(
            (s for s in stats.values() if s.get('train_db_size') is not None),
            key=lambda s: s['it_idx'],
        )
        it_idx = [s['it_idx'] for s in entries]
        train = [s['train_db_size'] for s in entries]
        seed = [s.get('seed_gen_db_size') for s in entries]

        # DB evolution
        fig = self._chart_db.fig
        fig.clear()
        if it_idx:
            ax = fig.add_subplot(111)
            x = np.arange(len(it_idx))
            has_seed = any(s is not None for s in seed)
            w = max(0.25, 0.8 / (2 if has_seed else 1))
            ax.bar(
                x,
                train,
                width=w,
                label='Train DB',
                color=COLORS[0],
                edgecolor='#282828',
                linewidth=0.8,
            )
            if has_seed:
                seed_c = [s if s is not None else 0 for s in seed]
                ax.bar(
                    x + w,
                    seed_c,
                    width=w,
                    label='Seed DB',
                    color=COLORS[1],
                    edgecolor='#282828',
                    linewidth=0.8,
                )
            ax.set_xticks(x + w / 2, labels=it_idx)
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('Structures')
            ax.set_title(f'Run #{run_id}, DB Evolution', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_db.redraw()

        # RMSE E
        e_entries = sorted(
            (
                s
                for s in stats.values()
                if s.get('mace_e') is not None and s['it_idx'] > 0
            ),
            key=lambda s: s['it_idx'],
        )
        fig = self._chart_rmse_e.fig
        fig.clear()
        if e_entries:
            ax = fig.add_subplot(111)
            x = [s['it_idx'] for s in e_entries]
            e = [s['mace_e'] for s in e_entries]
            ax.plot(x, e, 'o-', color=COLORS[2], markersize=5)
            ax.axhline(y=CHEM_ACC_E, color='#28282855', ls='--', lw=1)
            ax.text(x[0], CHEM_ACC_E + 1, 'Chem. Acc.', fontsize=7, color='#888')
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('RMSE E [meV/atom]')
            ax.set_title('Energy RMSE', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_rmse_e.redraw()

        # RMSE F
        f_entries = sorted(
            (
                s
                for s in stats.values()
                if s.get('mace_f') is not None and s['it_idx'] > 0
            ),
            key=lambda s: s['it_idx'],
        )
        fig = self._chart_rmse_f.fig
        fig.clear()
        if f_entries:
            ax = fig.add_subplot(111)
            x = [s['it_idx'] for s in f_entries]
            f_vals = [s['mace_f'] for s in f_entries]
            ax.plot(x, f_vals, 's-', color=COLORS[3], markersize=5)
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('RMSE F [meV/A]')
            ax.set_title('Force RMSE', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_rmse_f.redraw()

        # DB delta
        fig = self._chart_combined.fig
        fig.clear()
        if len(it_idx) >= 2:
            ax = fig.add_subplot(111)
            delta = [0] + [train[i] - train[i - 1] for i in range(1, len(train))]
            x = np.arange(len(it_idx))
            ax.bar(x, delta, color=COLORS[0], edgecolor='#282828', linewidth=0.8)
            ax.axhline(y=0, color='#28282855', ls='--')
            ax.set_xticks(x, labels=it_idx)
            ax.set_xlabel('AL Iteration')
            ax.set_ylabel('Δ Structures')
            ax.set_title('Per-Iteration Change', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        self._chart_combined.redraw()

    # ---------------------------------------------------------- iter table

    def _populate_iter_table(self, stats: dict) -> None:
        entries = sorted(stats.values(), key=lambda s: s['it_idx'])
        self._iter_table.setRowCount(len(entries))
        for i, s in enumerate(entries):
            it = s['it_idx']
            train = s.get('train_db_size')
            seed = s.get('seed_gen_db_size')
            e = s.get('mace_e')
            f = s.get('mace_f')

            prev = [
                p
                for p in entries
                if p['it_idx'] < it and p.get('train_db_size') is not None
            ]
            if prev and train is not None:
                delta = train - prev[-1]['train_db_size']
                delta_str = f'+{delta}' if delta >= 0 else str(delta)
            else:
                delta_str = '—'

            status = 'Done' if s.get('finished') else 'In progress'

            self._iter_table.setItem(i, 0, _item(str(it)))
            self._iter_table.setItem(
                i, 1, _item(f'{train:,}' if train is not None else '—')
            )
            self._iter_table.setItem(
                i, 2, _item(f'{seed:,}' if seed is not None else '—')
            )
            self._iter_table.setItem(i, 3, _item(f'{e:.2f}' if e is not None else '—'))
            self._iter_table.setItem(i, 4, _item(f'{f:.1f}' if f is not None else '—'))
            self._iter_table.setItem(i, 5, _item(delta_str))
            status_item = _item(status)
            if status == 'Done':
                status_item.setForeground(Qt.darkGreen)
            else:
                status_item.setForeground(Qt.darkYellow)
            self._iter_table.setItem(i, 6, status_item)

    # ---------------------------------------------------------- export

    def _export_png(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            'Export Report',
            'training_report.png',
            'PNG Images (*.png);;PDF (*.pdf);;All files (*)',
        )
        if not filepath:
            return
        path = Path(filepath)

        fig = Figure(figsize=(16, 10), dpi=150)
        fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])

        axes = fig.subplots(2, 2)
        for src_chart, ax in zip(
            [
                self._chart_db,
                self._chart_rmse_e,
                self._chart_rmse_f,
                self._chart_combined,
            ],
            axes.flat,
            strict=False,
        ):
            src_fig = src_chart.fig
            if src_fig.axes:
                src_ax = src_fig.axes[0]
                for line in src_ax.get_lines():
                    ax.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        color=line.get_color(),
                        ls=line.get_linestyle(),
                        marker=line.get_marker(),
                        ms=line.get_markersize(),
                        label=line.get_label(),
                    )
                for container in src_ax.containers:
                    for patch in container:
                        ax.bar(
                            patch.get_x(),
                            patch.get_height(),
                            width=patch.get_width(),
                            color=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                        )
                ax.set_xlabel(src_ax.get_xlabel())
                ax.set_ylabel(src_ax.get_ylabel())
                ax.set_title(src_ax.get_title())
                if src_ax.get_legend():
                    ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(str(path), dpi=150, bbox_inches='tight')


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
