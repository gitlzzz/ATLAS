"""Application themes for the ATLAS GUI.

Each theme is derived from a Gogh terminal color scheme and mapped onto
Qt widget roles.  The mapping uses the 16 ANSI colours plus background,
foreground, and cursor from the Gogh palette:

    background  → window / base background
    foreground  → text colour
    color_01    → dark accent (borders, subtle text in dark themes)
    color_02    → red / error / danger
    color_03    → green / success / done
    color_04    → yellow / warning / in-progress
    color_05    → blue / primary / links
    color_06    → magenta / accent
    color_07    → cyan / info
    color_08    → light grey (alt row, muted text in dark themes)
    color_09–16 → bright variants of 01–08

``build_stylesheet(theme_name)`` returns a full QSS string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:  # noqa
    name: str
    variant: str  # 'light' or 'dark'
    background: str
    foreground: str
    cursor: str
    c01: str
    c02: str
    c03: str
    c04: str  # noqa: E702
    c05: str
    c06: str
    c07: str
    c08: str  # noqa: E702
    c09: str
    c10: str
    c11: str
    c12: str  # noqa: E702
    c13: str
    c14: str
    c15: str
    c16: str  # noqa: E702


def _t(
    name,
    variant,
    bg,
    fg,
    cur,
    c01,
    c02,
    c03,
    c04,
    c05,
    c06,
    c07,
    c08,
    c09,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    c16,
) -> Theme:
    return Theme(
        name,
        variant,
        bg,
        fg,
        cur,
        c01,
        c02,
        c03,
        c04,
        c05,
        c06,
        c07,
        c08,
        c09,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
    )


THEMES: dict[str, Theme] = {}


def _register(*themes: Theme):
    for t in themes:
        THEMES[t.name] = t


_register(
    _t(
        'Default (Light)',
        'light',
        '#ffffff',
        '#1a1a2e',
        '#1a1a2e',
        '#2b2b2b',
        '#dc322f',
        '#2e8b57',
        '#b8860b',
        '#2962ff',
        '#8b2fc9',
        '#0891b2',
        '#6c757d',
        '#495057',
        '#e74c3c',
        '#28a745',
        '#e5c07b',
        '#4a90d9',
        '#a855f7',
        '#22d3ee',
        '#adb5bd',
    ),
    _t(
        'Mimi (Light)',
        'light',
        '#FFF0F5',
        '#4A3040',
        '#4A3040',
        '#6B4C5A',
        '#D94E6E',
        '#7BAE7F',
        '#C9956B',
        '#B06098',
        '#D4789C',
        '#6BA5A5',
        '#B8A0AC',
        '#8A6878',
        '#E8648A',
        '#8EC493',
        '#DAAD82',
        '#C87AB0',
        '#E895B5',
        '#82BFBF',
        '#D4C0CC',
    ),
    _t(
        'Gruvbox Dark',
        'dark',
        '#282828',
        '#EBDBB2',
        '#EBDBB2',
        '#282828',
        '#CC241D',
        '#98971A',
        '#D79921',
        '#458588',
        '#B16286',
        '#689D6A',
        '#A89984',
        '#928374',
        '#FB4934',
        '#B8BB26',
        '#FABD2F',
        '#83A598',
        '#D3869B',
        '#8EC07C',
        '#EBDBB2',
    ),
    _t(
        'Catppuccin Mocha',
        'dark',
        '#1E1E2E',
        '#CDD6F4',
        '#CDD6F4',
        '#45475A',
        '#F38BA8',
        '#A6E3A1',
        '#F9E2AF',
        '#89B4FA',
        '#F5C2E7',
        '#94E2D5',
        '#BAC2DE',
        '#585B70',
        '#F38BA8',
        '#A6E3A1',
        '#F9E2AF',
        '#89B4FA',
        '#F5C2E7',
        '#94E2D5',
        '#A6ADC8',
    ),
    _t(
        'Catppuccin Latte',
        'light',
        '#EFF1F5',
        '#4C4F69',
        '#4C4F69',
        '#5C5F77',
        '#D20F39',
        '#40A02B',
        '#DF8E1D',
        '#1E66F5',
        '#EA76CB',
        '#179299',
        '#ACB0BE',
        '#6C6F85',
        '#D20F39',
        '#40A02B',
        '#DF8E1D',
        '#1E66F5',
        '#EA76CB',
        '#179299',
        '#BCC0CC',
    ),
    _t(
        'Dracula',
        'dark',
        '#282A36',
        '#F8F8F2',
        '#F8F8F2',
        '#262626',
        '#E64747',
        '#42E66C',
        '#E4F34A',
        '#9B6BDF',
        '#E356A7',
        '#75D7EC',
        '#F8F8F2',
        '#7A7A7A',
        '#FF5555',
        '#50FA7B',
        '#F1FA8C',
        '#BD93F9',
        '#FF79C6',
        '#8BE9FD',
        '#F9F9FB',
    ),
    _t(
        'Nord',
        'dark',
        '#2E3440',
        '#D8DEE9',
        '#D8DEE9',
        '#3B4252',
        '#BF616A',
        '#A3BE8C',
        '#EBCB8B',
        '#81A1C1',
        '#B48EAD',
        '#88C0D0',
        '#E5E9F0',
        '#4C566A',
        '#BF616A',
        '#A3BE8C',
        '#EBCB8B',
        '#81A1C1',
        '#B48EAD',
        '#8FBCBB',
        '#ECEFF4',
    ),
    _t(
        'Solarized Dark',
        'dark',
        '#002B36',
        '#839496',
        '#839496',
        '#073642',
        '#DC322F',
        '#859900',
        '#CF9A6B',
        '#268BD2',
        '#D33682',
        '#2AA198',
        '#EEE8D5',
        '#657B83',
        '#CB4B16',
        '#859900',
        '#CF9A6B',
        '#6c71c4',
        '#D33682',
        '#2AA198',
        '#FDF6E3',
    ),
    _t(
        'Solarized Light',
        'light',
        '#FDF6E3',
        '#657B83',
        '#657B83',
        '#EEE8D5',
        '#DC322F',
        '#859900',
        '#B58900',
        '#268BD2',
        '#D33682',
        '#2AA198',
        '#002B36',
        '#657b83',
        '#CB4B16',
        '#859900',
        '#B58900',
        '#6C71C4',
        '#D33682',
        '#2AA198',
        '#073642',
    ),
    _t(
        'Tokyo Night',
        'dark',
        '#1A1B26',
        '#C0CAF5',
        '#C0CAF5',
        '#414868',
        '#F7768E',
        '#9ECE6A',
        '#E0AF68',
        '#7AA2F7',
        '#BB9AF7',
        '#7DCFFF',
        '#A9B1D6',
        '#414868',
        '#F7768E',
        '#9ECE6A',
        '#E0AF68',
        '#7AA2F7',
        '#BB9AF7',
        '#7DCFFF',
        '#C0CAF5',
    ),
    _t(
        'Tokyo Night Light',
        'light',
        '#D5D6DB',
        '#565A6E',
        '#565A6E',
        '#0F0F14',
        '#8C4351',
        '#485E30',
        '#8F5E15',
        '#34548A',
        '#5A4A78',
        '#0F4B6E',
        '#343B58',
        '#9699A3',
        '#8C4351',
        '#485E30',
        '#8F5E15',
        '#34548A',
        '#5A4A78',
        '#0F4B6E',
        '#343B58',
    ),
    _t(
        'Everforest Dark',
        'dark',
        '#272E33',
        '#D3C6AA',
        '#D3C6AA',
        '#2E383C',
        '#E67E80',
        '#A7C080',
        '#DBBC7F',
        '#7FBBB3',
        '#D699B6',
        '#83C092',
        '#D3C6AA',
        '#5C6A72',
        '#F85552',
        '#8DA101',
        '#DFA000',
        '#3A94C5',
        '#DF69BA',
        '#35A77C',
        '#DFDDC8',
    ),
    _t(
        'Everforest Light',
        'light',
        '#FFFBEF',
        '#5C6A72',
        '#5C6A72',
        '#5C6A72',
        '#F85552',
        '#8DA101',
        '#DFA000',
        '#3A94C5',
        '#DF69BA',
        '#35A77C',
        '#DFDDC8',
        '#2E383C',
        '#E67E80',
        '#A7C080',
        '#DBBC7F',
        '#7FBBB3',
        '#D699B6',
        '#83C092',
        '#D3C6AA',
    ),
    _t(
        'Kanagawa Wave',
        'dark',
        '#1F1F28',
        '#DCD7BA',
        '#DCD7BA',
        '#090618',
        '#C34043',
        '#76946A',
        '#C0A36E',
        '#7E9CD8',
        '#957FB8',
        '#6A9589',
        '#C8C093',
        '#727169',
        '#E82424',
        '#98BB6C',
        '#E6C384',
        '#7FB4CA',
        '#938AA9',
        '#7AA89F',
        '#DCD7BA',
    ),
    _t(
        'Kanagawa Lotus',
        'light',
        '#f2ecbc',
        '#545464',
        '#43436c',
        '#1f1f28',
        '#c84053',
        '#6f894e',
        '#77713f',
        '#4d699b',
        '#b35b79',
        '#597b75',
        '#545464',
        '#8a8980',
        '#d7474b',
        '#6e915f',
        '#836f4a',
        '#6693bf',
        '#624c83',
        '#5e857a',
        '#43436c',
    ),
    _t(
        'Monokai Dark',
        'dark',
        '#272822',
        '#F8F8F2',
        '#F8F8F2',
        '#75715E',
        '#F92672',
        '#A6E22E',
        '#F4BF75',
        '#66D9EF',
        '#AE81FF',
        '#2AA198',
        '#F9F8F5',
        '#272822',
        '#F92672',
        '#A6E22E',
        '#F4BF75',
        '#66D9EF',
        '#AE81FF',
        '#2AA198',
        '#F8F8F2',
    ),
    _t(
        'One Dark',
        'dark',
        '#1E2127',
        '#ABB2BF',
        '#ABB2BF',
        '#000000',
        '#E06C75',
        '#98C379',
        '#D19A66',
        '#61AFEF',
        '#C678DD',
        '#56B6C2',
        '#ABB2BF',
        '#5C6370',
        '#E06C75',
        '#98C379',
        '#D19A66',
        '#61AFEF',
        '#C678DD',
        '#56B6C2',
        '#FFFEFE',
    ),
)

DEFAULT_THEME = 'Default (Light)'


def _mix(hex_a: str, hex_b: str, t: float = 0.5) -> str:
    """Linearly interpolate two hex colours. *t* = 0 → a, 1 → b."""
    ra, ga, ba = int(hex_a[1:3], 16), int(hex_a[3:5], 16), int(hex_a[5:7], 16)
    rb, gb, bb = int(hex_b[1:3], 16), int(hex_b[3:5], 16), int(hex_b[5:7], 16)
    r = int(ra + (rb - ra) * t)
    g = int(ga + (gb - ga) * t)
    b = int(ba + (bb - ba) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _alpha(hex_color: str, alpha: int) -> str:
    """Return ``rgba(r,g,b,a)`` from a hex colour + 0–255 alpha."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'


def build_stylesheet(theme_name: str) -> str:
    """Generate a full QSS stylesheet for the given theme."""
    t = THEMES.get(theme_name)
    if t is None:
        return ''

    is_dark = t.variant == 'dark'

    bg = t.background
    fg = t.foreground
    base = bg
    alt_bg = _mix(bg, fg, 0.04) if is_dark else _mix(bg, fg, 0.05)
    surface = _mix(bg, fg, 0.07) if is_dark else _mix(bg, fg, 0.06)
    border = _mix(bg, fg, 0.15) if is_dark else _mix(bg, fg, 0.16)
    border_light = _mix(bg, fg, 0.10) if is_dark else _mix(bg, fg, 0.10)
    muted = _mix(bg, fg, 0.45)
    hover = _mix(bg, t.c05, 0.15) if is_dark else _mix(bg, t.c05, 0.10)
    selected_bg = _mix(bg, t.c05, 0.25) if is_dark else _mix(bg, t.c05, 0.15)
    selected_fg = t.c13 if is_dark else t.c05

    primary = t.c05 if not is_dark else t.c13

    btn_bg = _mix(bg, fg, 0.08) if not is_dark else surface
    btn_hover = _mix(btn_bg, fg, 0.08)
    btn_pressed = _mix(btn_bg, fg, 0.14)

    _mix(primary, fg, 0.15)

    input_bg = base if is_dark else '#ffffff'
    input_border = border

    from atlas.core.gui.icons import ensure_arrow_svgs

    arrow_up, arrow_down = ensure_arrow_svgs(fg)

    scrollbar_bg = alt_bg
    scrollbar_handle = _mix(bg, fg, 0.20)
    scrollbar_handle_hover = _mix(bg, fg, 0.30)

    tab_bg = alt_bg
    tab_selected_bg = bg
    tab_hover_bg = _mix(alt_bg, fg, 0.05)

    return f"""
/* ====== Global ====== */
QWidget {{
    background-color: {bg};
    color: {fg};
    font-size: 13px;
}}

/* ====== Menu Bar ====== */
QMenuBar {{
    background-color: {surface};
    color: {fg};
    border-bottom: 1px solid {border};
}}
QMenuBar::item:selected {{
    background-color: {hover};
}}
QMenu {{
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
}}
QMenu::item:selected {{
    background-color: {selected_bg};
    color: {selected_fg};
}}
QMenu::separator {{
    height: 1px;
    background: {border_light};
    margin: 4px 8px;
}}

/* ====== Sidebar (QListWidget) ====== */
QListWidget {{
    background-color: {alt_bg};
    border: none;
    border-right: 1px solid {border};
    padding: 4px;
}}
QListWidget::item {{
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 4px;
    color: {fg};
}}
QListWidget::item:selected {{
    background-color: {selected_bg};
    color: {selected_fg};
    font-weight: bold;
}}
QListWidget::item:hover:!selected {{
    background-color: {hover};
}}

/* ====== Tabs ====== */
QTabWidget::pane {{
    border: 1px solid {border};
    border-top: none;
    background-color: {bg};
}}
QTabBar::tab {{
    background-color: {tab_bg};
    color: {muted};
    padding: 8px 16px;
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {tab_selected_bg};
    color: {fg};
    border-bottom: 2px solid {primary};
}}
QTabBar::tab:hover:!selected {{
    background-color: {tab_hover_bg};
    color: {fg};
}}

/* ── Page-level (outer) tab bar ── */
QTabWidget#pageTab > QTabBar::tab {{
    padding: 9px 18px;
    font-weight: 600;
}}
QTabWidget#pageTab > QTabBar::tab:disabled {{
    min-width: 0px;
    max-width: 2px;
    padding: 4px 0px;
    margin-left: 6px;
    margin-right: 6px;
    background-color: {border};
    border: none;
    border-radius: 1px;
}}
QTabWidget#pageTab::pane {{
    border-top: 2px solid {border};
}}

/* ====== Buttons ====== */
QPushButton {{
    background-color: {btn_bg};
    color: {fg};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {btn_hover};
    border-color: {_mix(border, fg, 0.1)};
}}
QPushButton:pressed {{
    background-color: {btn_pressed};
}}
QPushButton:disabled {{
    color: {muted};
    background-color: {alt_bg};
    border-color: {border_light};
}}

/* ====== Input Fields ====== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {input_bg};
    color: {fg};
    border: 1px solid {input_border};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {primary};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: url({arrow_down});
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
    selection-background-color: {selected_bg};
    selection-color: {selected_fg};
}}

/* ====== Spin Box Buttons ====== */
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {border_light};
    border-top-right-radius: 4px;
    background: {btn_bg};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: {btn_hover};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid {border_light};
    border-bottom-right-radius: 4px;
    background: {btn_bg};
}}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {btn_hover};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({arrow_up});
    width: 12px;
    height: 12px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({arrow_down});
    width: 12px;
    height: 12px;
}}

/* ====== Tables ====== */
QTableWidget, QTableView {{
    background-color: {base};
    color: {fg};
    gridline-color: {border_light};
    border: 1px solid {border};
    alternate-background-color: {alt_bg};
    selection-background-color: {selected_bg};
    selection-color: {selected_fg};
}}
QHeaderView::section {{
    background-color: {surface};
    color: {fg};
    padding: 4px 8px;
    border: none;
    border-bottom: 1px solid {border};
    border-right: 1px solid {border_light};
    font-weight: bold;
}}

/* ====== Group Boxes ====== */
QGroupBox {{
    background-color: {bg};
    color: {fg};
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {fg};
}}

/* ====== Frames ====== */
QFrame {{
    color: {fg};
}}

/* ====== Splitter ====== */
QSplitter::handle {{
    background-color: {border_light};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ====== Scroll Area ====== */
QScrollArea {{
    border: none;
    background-color: {bg};
}}

/* ====== Log Viewer (QTextEdit) ====== */
QTextEdit {{
    background-color: {base};
    color: {fg};
    border: 1px solid {border};
    selection-background-color: {selected_bg};
    selection-color: {selected_fg};
}}

/* ====== Dock Widget ====== */
QDockWidget {{
    color: {fg};
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background-color: {surface};
    padding: 6px;
    border-bottom: 1px solid {border};
}}

/* ====== Scrollbars ====== */
QScrollBar:vertical {{
    background: {scrollbar_bg};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {scrollbar_handle};
    min-height: 30px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {scrollbar_handle_hover};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {scrollbar_bg};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {scrollbar_handle};
    min-width: 30px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {scrollbar_handle_hover};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ====== Progress Bar ====== */
QProgressBar {{
    border: 1px solid {border};
    border-radius: 4px;
    background: {alt_bg};
    text-align: center;
    color: {fg};
}}
QProgressBar::chunk {{
    background: {primary};
    border-radius: 3px;
}}

/* ====== Tooltips ====== */
QToolTip {{
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
    padding: 4px;
}}

/* ====== Status Bar ====== */
QStatusBar {{
    background-color: {surface};
    color: {fg};
    border-top: 1px solid {border};
}}

/* ====== Labels ====== */
QLabel {{
    background-color: transparent;
    color: {fg};
}}

/* ====== Disabled Group Boxes (optional sections) ====== */
QGroupBox[section_off="true"] {{
    background-color: {alt_bg};
    border-style: dashed;
}}
QGroupBox[section_off="true"] QLabel,
QGroupBox[section_off="true"] QLineEdit,
QGroupBox[section_off="true"] QSpinBox,
QGroupBox[section_off="true"] QDoubleSpinBox,
QGroupBox[section_off="true"] QComboBox,
QGroupBox[section_off="true"] QCheckBox,
QGroupBox[section_off="true"] QTableWidget,
QGroupBox[section_off="true"] QPushButton {{
    color: {muted};
}}
QGroupBox[section_off="true"] QLineEdit,
QGroupBox[section_off="true"] QSpinBox,
QGroupBox[section_off="true"] QDoubleSpinBox,
QGroupBox[section_off="true"] QComboBox,
QGroupBox[section_off="true"] QCheckBox,
QGroupBox[section_off="true"] QTableWidget {{
    background-color: {alt_bg};
    border-color: {border_light};
}}
"""


def theme_names() -> list[str]:
    """Return theme names, light themes first, then dark."""
    light = sorted(n for n, t in THEMES.items() if t.variant == 'light')
    dark = sorted(n for n, t in THEMES.items() if t.variant == 'dark')
    return light + dark


def theme_variant(name: str) -> str:
    """Return 'light' or 'dark' for the given theme name."""
    t = THEMES.get(name)
    return t.variant if t else 'light'


def theme_colors(name: str) -> dict[str, str]:
    """Return derived colors for the theme (bg, fg, surface, border, etc.)."""
    t = THEMES.get(name)
    if t is None:
        return {
            'bg': '#ffffff',
            'fg': '#1a1a2e',
            'surface': '#f5f5f5',
            'border': '#d0d4db',
            'axes_bg': '#ffffff',
            'muted': '#888888',
            'primary': '#4a90d9',
        }
    is_dark = t.variant == 'dark'
    bg = t.background
    fg = t.foreground
    surface = _mix(bg, fg, 0.07) if is_dark else _mix(bg, fg, 0.04)
    border = _mix(bg, fg, 0.15) if is_dark else _mix(bg, fg, 0.12)
    axes_bg = _mix(bg, fg, 0.03) if is_dark else '#ffffff'
    muted = _mix(bg, fg, 0.45)
    primary = t.c05 if not is_dark else t.c13
    return {
        'bg': bg,
        'fg': fg,
        'surface': surface,
        'border': border,
        'axes_bg': axes_bg,
        'muted': muted,
        'primary': primary,
    }


def apply_theme_to_app(theme_name: str, *, force: bool = False) -> None:
    """Apply theme QSS to QApplication and set OS color scheme hint."""
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return

    settings = QSettings('ATLAS', 'GUI')
    if not force and settings.value('app_theme') == theme_name and app.styleSheet():
        return

    qss = build_stylesheet(theme_name)
    app.setStyleSheet(qss)

    settings.setValue('app_theme', theme_name)

    t = THEMES.get(theme_name)
    variant = theme_variant(theme_name)

    # Set palette so the window manager can infer dark/light preference.
    from PySide6.QtGui import QColor, QPalette

    if t is not None:
        palette = app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(t.background))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(t.foreground))
        palette.setColor(QPalette.ColorRole.Base, QColor(t.background))
        palette.setColor(QPalette.ColorRole.Text, QColor(t.foreground))
        app.setPalette(palette)

    # Qt 6.8+ exposes an explicit color scheme hint for title bars.
    try:
        scheme = Qt.ColorScheme.Dark if variant == 'dark' else Qt.ColorScheme.Light
        app.styleHints().setColorScheme(scheme)
    except AttributeError:
        pass


def saved_global_theme() -> str:
    """Return the globally persisted theme name (via QSettings)."""
    from PySide6.QtCore import QSettings

    return QSettings('ATLAS', 'GUI').value('app_theme', DEFAULT_THEME)
