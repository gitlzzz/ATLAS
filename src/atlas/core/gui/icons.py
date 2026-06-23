"""Theme-adaptive icon loading for the ATLAS GUI.

SVG icons ship with ``fill="#000"`` and the fill is replaced at load time
to match the active theme's foreground colour.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICONS_DIR = Path(__file__).resolve().parent / 'assets' / 'icons'
ASSETS_DIR = Path(__file__).resolve().parent / 'assets'

_arrow_cache: dict[str, Path] = {}
_icon_cache: dict[tuple[str, str, int], QIcon] = {}


def themed_icon(name: str, color: str, size: int = 24) -> QIcon:
    """Return a ``QIcon`` for *name* recoloured to *color*."""
    key = (name, color, size)
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached

    svg_path = ICONS_DIR / f'{name}.svg'
    if not svg_path.exists():
        return QIcon()
    svg_data = svg_path.read_bytes().replace(b'fill="#000"', f'fill="{color}"'.encode())
    renderer = QSvgRenderer(svg_data)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    icon = QIcon(pixmap)
    _icon_cache[key] = icon
    return icon


def _tray_icon_path() -> Path | None:
    """Return the SVG path for the system tray icon.

    The tray icon always uses the light version with a blue accent
    because system trays render on their own background (often dark)
    and we can't control it.
    """
    return ASSETS_DIR / 'atlas_atom_tray.svg'


def app_icon() -> QIcon:
    """Return the application icon for window/tray use."""
    from atlas.core.gui.themes import saved_global_theme, theme_variant

    variant = theme_variant(saved_global_theme())
    dark = ASSETS_DIR / 'atlas_atom_dark.svg'
    light = ASSETS_DIR / 'atlas_atom.svg'
    if variant == 'dark' and dark.exists():
        return QIcon(str(dark))
    if light.exists():
        return QIcon(str(light))
    logo = ASSETS_DIR / 'atlas_logo_light.png'
    if logo.exists():
        return QIcon(str(logo))
    return QIcon.fromTheme('applications-science')


def tray_icon() -> QIcon:
    """Return the icon for the system tray.

    Always uses the light version with a blue accent since system
    trays render on an independent background.
    """
    tray_svg = _tray_icon_path()
    if tray_svg and tray_svg.exists():
        return QIcon(str(tray_svg))
    return QIcon.fromTheme('applications-science')


def ensure_arrow_svgs(fg_color: str) -> tuple[str, str]:
    """Generate themed spinbox arrow SVGs and return (up_path, down_path).

    Results are cached per colour so the files are written at most once
    per theme.
    """
    key = fg_color.lstrip('#')
    if key in _arrow_cache:
        cache_dir = _arrow_cache[key]
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix='atlas_arrows_'))
        _arrow_cache[key] = cache_dir

    up_path = cache_dir / 'up.svg'
    down_path = cache_dir / 'down.svg'

    if not up_path.exists():
        up_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<path fill="{fg_color}" d="M7.41 15.41L12 10.83l4.59 4.58L18 '
            '14l-6-6-6 6z"/></svg>'
        )
    if not down_path.exists():
        down_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<path fill="{fg_color}" d="M7.41 8.59L12 13.17l4.59-4.58L18 '
            '10l-6 6-6-6z"/></svg>'
        )
    return str(up_path), str(down_path)
