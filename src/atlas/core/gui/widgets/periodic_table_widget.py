"""Periodic table element picker with pill display."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ELEMENTS = [
    (1, 'H', 'Hydrogen', 1, 1, 'nonmetal'),
    (2, 'He', 'Helium', 1, 18, 'noble'),
    (3, 'Li', 'Lithium', 2, 1, 'alkali'),
    (4, 'Be', 'Beryllium', 2, 2, 'alkaline'),
    (5, 'B', 'Boron', 2, 13, 'metalloid'),
    (6, 'C', 'Carbon', 2, 14, 'nonmetal'),
    (7, 'N', 'Nitrogen', 2, 15, 'nonmetal'),
    (8, 'O', 'Oxygen', 2, 16, 'nonmetal'),
    (9, 'F', 'Fluorine', 2, 17, 'halogen'),
    (10, 'Ne', 'Neon', 2, 18, 'noble'),
    (11, 'Na', 'Sodium', 3, 1, 'alkali'),
    (12, 'Mg', 'Magnesium', 3, 2, 'alkaline'),
    (13, 'Al', 'Aluminium', 3, 13, 'post_transition'),
    (14, 'Si', 'Silicon', 3, 14, 'metalloid'),
    (15, 'P', 'Phosphorus', 3, 15, 'nonmetal'),
    (16, 'S', 'Sulfur', 3, 16, 'nonmetal'),
    (17, 'Cl', 'Chlorine', 3, 17, 'halogen'),
    (18, 'Ar', 'Argon', 3, 18, 'noble'),
    (19, 'K', 'Potassium', 4, 1, 'alkali'),
    (20, 'Ca', 'Calcium', 4, 2, 'alkaline'),
    (21, 'Sc', 'Scandium', 4, 3, 'transition'),
    (22, 'Ti', 'Titanium', 4, 4, 'transition'),
    (23, 'V', 'Vanadium', 4, 5, 'transition'),
    (24, 'Cr', 'Chromium', 4, 6, 'transition'),
    (25, 'Mn', 'Manganese', 4, 7, 'transition'),
    (26, 'Fe', 'Iron', 4, 8, 'transition'),
    (27, 'Co', 'Cobalt', 4, 9, 'transition'),
    (28, 'Ni', 'Nickel', 4, 10, 'transition'),
    (29, 'Cu', 'Copper', 4, 11, 'transition'),
    (30, 'Zn', 'Zinc', 4, 12, 'post_transition'),
    (31, 'Ga', 'Gallium', 4, 13, 'post_transition'),
    (32, 'Ge', 'Germanium', 4, 14, 'metalloid'),
    (33, 'As', 'Arsenic', 4, 15, 'metalloid'),
    (34, 'Se', 'Selenium', 4, 16, 'nonmetal'),
    (35, 'Br', 'Bromine', 4, 17, 'halogen'),
    (36, 'Kr', 'Krypton', 4, 18, 'noble'),
    (37, 'Rb', 'Rubidium', 5, 1, 'alkali'),
    (38, 'Sr', 'Strontium', 5, 2, 'alkaline'),
    (39, 'Y', 'Yttrium', 5, 3, 'transition'),
    (40, 'Zr', 'Zirconium', 5, 4, 'transition'),
    (41, 'Nb', 'Niobium', 5, 5, 'transition'),
    (42, 'Mo', 'Molybdenum', 5, 6, 'transition'),
    (43, 'Tc', 'Technetium', 5, 7, 'transition'),
    (44, 'Ru', 'Ruthenium', 5, 8, 'transition'),
    (45, 'Rh', 'Rhodium', 5, 9, 'transition'),
    (46, 'Pd', 'Palladium', 5, 10, 'transition'),
    (47, 'Ag', 'Silver', 5, 11, 'transition'),
    (48, 'Cd', 'Cadmium', 5, 12, 'post_transition'),
    (49, 'In', 'Indium', 5, 13, 'post_transition'),
    (50, 'Sn', 'Tin', 5, 14, 'post_transition'),
    (51, 'Sb', 'Antimony', 5, 15, 'metalloid'),
    (52, 'Te', 'Tellurium', 5, 16, 'metalloid'),
    (53, 'I', 'Iodine', 5, 17, 'halogen'),
    (54, 'Xe', 'Xenon', 5, 18, 'noble'),
    (55, 'Cs', 'Caesium', 6, 1, 'alkali'),
    (56, 'Ba', 'Barium', 6, 2, 'alkaline'),
    (57, 'La', 'Lanthanum', 9, 3, 'lanthanide'),
    (58, 'Ce', 'Cerium', 9, 4, 'lanthanide'),
    (59, 'Pr', 'Praseodymium', 9, 5, 'lanthanide'),
    (60, 'Nd', 'Neodymium', 9, 6, 'lanthanide'),
    (61, 'Pm', 'Promethium', 9, 7, 'lanthanide'),
    (62, 'Sm', 'Samarium', 9, 8, 'lanthanide'),
    (63, 'Eu', 'Europium', 9, 9, 'lanthanide'),
    (64, 'Gd', 'Gadolinium', 9, 10, 'lanthanide'),
    (65, 'Tb', 'Terbium', 9, 11, 'lanthanide'),
    (66, 'Dy', 'Dysprosium', 9, 12, 'lanthanide'),
    (67, 'Ho', 'Holmium', 9, 13, 'lanthanide'),
    (68, 'Er', 'Erbium', 9, 14, 'lanthanide'),
    (69, 'Tm', 'Thulium', 9, 15, 'lanthanide'),
    (70, 'Yb', 'Ytterbium', 9, 16, 'lanthanide'),
    (71, 'Lu', 'Lutetium', 6, 3, 'lanthanide'),
    (72, 'Hf', 'Hafnium', 6, 4, 'transition'),
    (73, 'Ta', 'Tantalum', 6, 5, 'transition'),
    (74, 'W', 'Tungsten', 6, 6, 'transition'),
    (75, 'Re', 'Rhenium', 6, 7, 'transition'),
    (76, 'Os', 'Osmium', 6, 8, 'transition'),
    (77, 'Ir', 'Iridium', 6, 9, 'transition'),
    (78, 'Pt', 'Platinum', 6, 10, 'transition'),
    (79, 'Au', 'Gold', 6, 11, 'transition'),
    (80, 'Hg', 'Mercury', 6, 12, 'post_transition'),
    (81, 'Tl', 'Thallium', 6, 13, 'post_transition'),
    (82, 'Pb', 'Lead', 6, 14, 'post_transition'),
    (83, 'Bi', 'Bismuth', 6, 15, 'post_transition'),
    (84, 'Po', 'Polonium', 6, 16, 'post_transition'),
    (85, 'At', 'Astatine', 6, 17, 'halogen'),
    (86, 'Rn', 'Radon', 6, 18, 'noble'),
    (87, 'Fr', 'Francium', 7, 1, 'alkali'),
    (88, 'Ra', 'Radium', 7, 2, 'alkaline'),
    (89, 'Ac', 'Actinium', 10, 3, 'actinide'),
    (90, 'Th', 'Thorium', 10, 4, 'actinide'),
    (91, 'Pa', 'Protactinium', 10, 5, 'actinide'),
    (92, 'U', 'Uranium', 10, 6, 'actinide'),
    (93, 'Np', 'Neptunium', 10, 7, 'actinide'),
    (94, 'Pu', 'Plutonium', 10, 8, 'actinide'),
    (95, 'Am', 'Americium', 10, 9, 'actinide'),
    (96, 'Cm', 'Curium', 10, 10, 'actinide'),
    (97, 'Bk', 'Berkelium', 10, 11, 'actinide'),
    (98, 'Cf', 'Californium', 10, 12, 'actinide'),
    (99, 'Es', 'Einsteinium', 10, 13, 'actinide'),
    (100, 'Fm', 'Fermium', 10, 14, 'actinide'),
    (101, 'Md', 'Mendelevium', 10, 15, 'actinide'),
    (102, 'No', 'Nobelium', 10, 16, 'actinide'),
    (103, 'Lr', 'Lawrencium', 7, 3, 'actinide'),
    (104, 'Rf', 'Rutherfordium', 7, 4, 'transition'),
    (105, 'Db', 'Dubnium', 7, 5, 'transition'),
    (106, 'Sg', 'Seaborgium', 7, 6, 'transition'),
    (107, 'Bh', 'Bohrium', 7, 7, 'transition'),
    (108, 'Hs', 'Hassium', 7, 8, 'transition'),
    (109, 'Mt', 'Meitnerium', 7, 9, 'transition'),
    (110, 'Ds', 'Darmstadtium', 7, 10, 'transition'),
    (111, 'Rg', 'Roentgenium', 7, 11, 'transition'),
    (112, 'Cn', 'Copernicium', 7, 12, 'post_transition'),
    (113, 'Nh', 'Nihonium', 7, 13, 'post_transition'),
    (114, 'Fl', 'Flerovium', 7, 14, 'post_transition'),
    (115, 'Mc', 'Moscovium', 7, 15, 'post_transition'),
    (116, 'Lv', 'Livermorium', 7, 16, 'post_transition'),
    (117, 'Ts', 'Tennessine', 7, 17, 'halogen'),
    (118, 'Og', 'Oganesson', 7, 18, 'noble'),
]

CATEGORY_COLORS = {
    'alkali': '#FF6666',
    'alkaline': '#FFDEAD',
    'transition': '#87CEEB',
    'post_transition': '#90EE90',
    'metalloid': '#DAA520',
    'nonmetal': '#FFFACD',
    'halogen': '#AFEEEE',
    'noble': '#E6E6FA',
    'lanthanide': '#FFB6C1',
    'actinide': '#D8BFD8',
}

SELECTED_STYLE = (
    'QPushButton {{ background-color: {bg}; color: white; '
    'border: 2px solid {bg}; border-radius: 3px; '
    'font-weight: bold; font-size: 11px; padding: 2px; }}'
)
NORMAL_STYLE = (
    'QPushButton {{ background-color: {bg}; color: #333; '
    'border: 1px solid #aaa; border-radius: 3px; '
    'font-size: 11px; padding: 2px; }}'
    'QPushButton:hover {{ border: 2px solid #555; }}'
)


class PeriodicTableDialog(QDialog):
    """Dialog showing a periodic table for multi-element selection."""

    def __init__(
        self,
        selected: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle('Select Elements')
        self.setMinimumSize(750, 420)
        self._selected: set[str] = set(selected or [])
        self._buttons: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)

        grid = QGridLayout()
        grid.setSpacing(2)

        for z, symbol, name, row, col, _category in ELEMENTS:
            btn = QPushButton(symbol)
            btn.setFixedSize(38, 32)
            btn.setToolTip(f'{z} — {name}')
            btn.setCheckable(True)
            btn.setChecked(symbol in self._selected)
            btn.clicked.connect(lambda _, s=symbol: self._toggle(s))
            self._buttons[symbol] = btn
            grid.addWidget(btn, row - 1, col - 1)

        # Spacer row between main table and lanthanides/actinides
        spacer = QLabel()
        spacer.setFixedHeight(8)
        grid.addWidget(spacer, 7, 0, 1, 18)

        self._update_styles()
        outer.addLayout(grid)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        outer.addWidget(btn_box)

    def _toggle(self, symbol: str) -> None:
        if symbol in self._selected:
            self._selected.discard(symbol)
        else:
            self._selected.add(symbol)
        self._update_styles()

    def _update_styles(self) -> None:
        elem_map = {e[1]: e for e in ELEMENTS}
        for symbol, btn in self._buttons.items():
            cat = elem_map[symbol][5]
            bg = CATEGORY_COLORS.get(cat, '#DDD')
            if symbol in self._selected:
                from atlas.core.gui.themes import saved_global_theme, theme_colors

                primary = theme_colors(saved_global_theme()).get('primary', '#2962ff')
                btn.setStyleSheet(SELECTED_STYLE.format(bg=primary))
                btn.setChecked(True)
            else:
                btn.setStyleSheet(NORMAL_STYLE.format(bg=bg))
                btn.setChecked(False)

    def selected_elements(self) -> list[str]:
        order = [e[1] for e in ELEMENTS]
        return [s for s in order if s in self._selected]


class _ElementPill(QFrame):
    """Small rounded pill showing an element symbol with a remove button."""

    removed = Signal(str)

    def __init__(self, symbol: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.symbol = symbol
        self.setFrameShape(QFrame.StyledPanel)

        elem_map = {e[1]: e for e in ELEMENTS}
        cat = elem_map.get(symbol, (0, '', '', 0, 0, 'nonmetal'))[5]
        bg = CATEGORY_COLORS.get(cat, '#DDD')
        self.setStyleSheet(
            f'QFrame {{ background-color: {bg}; border-radius: 10px;'
            f' border: 1px solid #aaa; padding: 1px 4px; }}'
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 2, 2)
        layout.setSpacing(2)

        lbl = QLabel(symbol)
        lbl.setStyleSheet(
            'font-weight: bold; font-size: 12px; border: none; background: transparent;'
        )
        layout.addWidget(lbl)

        close_btn = QPushButton('x')
        close_btn.setFixedSize(16, 16)
        close_btn.setStyleSheet(
            'QPushButton { border: none; font-size: 10px; color: #666;'
            ' background: transparent; padding: 0; }'
            'QPushButton:hover { color: #c00; }'
        )
        close_btn.clicked.connect(lambda: self.removed.emit(self.symbol))
        layout.addWidget(close_btn)


class ElementPickerField(QWidget):
    """Combines an element pill bar with a periodic table picker button.

    Drop-in replacement for a QLineEdit that edits a comma-separated
    element list.  Emits ``elements_changed`` whenever the selection
    changes.
    """

    elements_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._elements: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Pill bar
        self._pill_bar = QWidget()
        self._pill_layout = _FlowLayout(self._pill_bar)
        self._pill_layout.setSpacing(4)
        layout.addWidget(self._pill_bar)

        # Pick button
        pick_btn = QPushButton('Pick Elements...')
        pick_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pick_btn.clicked.connect(self._open_picker)
        layout.addWidget(pick_btn)

    def elements(self) -> list[str]:
        return list(self._elements)

    def set_elements(self, elements: list[str]) -> None:
        self._elements = list(elements)
        self._rebuild_pills()
        self.elements_changed.emit(self._elements)

    def text(self) -> str:
        return ', '.join(self._elements)

    def setText(self, text: str) -> None:
        elems = [e.strip() for e in text.split(',') if e.strip()]
        self.set_elements(elems)

    def _open_picker(self) -> None:
        dlg = PeriodicTableDialog(self._elements, self)
        if dlg.exec() == QDialog.Accepted:
            self.set_elements(dlg.selected_elements())

    def _rebuild_pills(self) -> None:
        while self._pill_layout.count():
            item = self._pill_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for symbol in self._elements:
            pill = _ElementPill(symbol)
            pill.removed.connect(self._remove_element)
            self._pill_layout.addWidget(pill)

    def _remove_element(self, symbol: str) -> None:
        if symbol in self._elements:
            self._elements.remove(symbol)
            self._rebuild_pills()
            self.elements_changed.emit(self._elements)


class _FlowLayout(QVBoxLayout):
    """Simple flow layout that wraps pills into rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QWidget] = []
        self._row_layout: QHBoxLayout | None = None
        self._ensure_row()

    def _ensure_row(self) -> None:
        self._row_layout = QHBoxLayout()
        self._row_layout.setSpacing(4)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.addStretch()
        super().addLayout(self._row_layout)

    def addWidget(self, widget: QWidget) -> None:
        if self._row_layout is None:
            self._ensure_row()
        idx = self._row_layout.count() - 1
        self._row_layout.insertWidget(idx, widget)
        self._items.append(widget)

    def count(self) -> int:
        return len(self._items)

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            widget = self._items.pop(index)
            return type('LayoutItem', (), {'widget': lambda self=widget: self})()
        return None
