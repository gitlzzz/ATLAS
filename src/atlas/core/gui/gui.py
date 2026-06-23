"""
A PyQt-based GUI for the ATLAS library.

This application provides a user-friendly interface to configure and run
ATLAS processes. It uses a YAML schema as a template to dynamically
build the GUI, and outputs the configuration in TOML format.
"""

import os
import string
import subprocess
import sys
from functools import partial

import tomli
import tomli_w
import yaml
from PySide6.QtCore import QEvent, QRegularExpression, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from atlas.core.command_line import command_line_utils as cli_utils


# --- TOML Syntax Highlighter ---
class TomlHighlighter(QSyntaxHighlighter):
    """
    A syntax highlighter for TOML format that can be applied to a QTextDocument.
    It supports multiple color themes.
    """

    THEMES = {
        'Default': {
            'background': None,
            'section': '#107c10',
            'comment': '#808080',
            'keyword': '#0033cc',
            'string': '#A31515',
            'number': '#c500c5',
        },
        'Gruvbox Light': {
            'background': None,
            'section': '#79740e',
            'comment': '#928374',
            'keyword': '#427b58',
            'string': '#9d0006',
            'number': '#8f3f71',
        },
        'Catppuccin Latte': {
            'background': None,
            'section': '#40a02b',
            'comment': '#9ca0b0',
            'keyword': '#8839ef',
            'string': '#d20f39',
            'number': '#e64553',
        },
        'Solarized Light': {
            'background': '#fdf6e3',
            'section': '#859900',
            'comment': '#93a1a1',
            'keyword': '#d33682',
            'string': '#dc322f',
            'number': '#cb4b16',
        },
    }

    def __init__(self, parent=None, theme='Default'):
        super().__init__(parent)
        self.highlighting_rules = []

        palette = self.THEMES.get(theme, self.THEMES['Default'])

        # Section format: [section]
        section_format = QTextCharFormat()
        section_format.setForeground(QColor(palette['section']))
        section_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append(
            (QRegularExpression(r'\[([^]]*)\]'), section_format)
        )

        # Comment format: # comment
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(palette['comment']))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression(r'#[^\n]*'), comment_format))

        # Keyword format: true, false
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(palette['keyword']))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = ['true', 'false']
        for word in keywords:
            pattern = QRegularExpression(rf'\b{word}\b')
            self.highlighting_rules.append((pattern, keyword_format))

        # String format: "string"
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(palette['string']))
        self.highlighting_rules.append((QRegularExpression(r'"[^"]*"'), string_format))

        # Number format
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(palette['number']))
        self.highlighting_rules.append(
            (QRegularExpression(r'\b-?\d+(\.\d+)?\b'), number_format)
        )

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class ApplicationParameters:
    """Holds application-wide parameters."""

    FONT_FAMILIES_REGULAR = ['Noto Sans', 'Sans Serif', 'Arial']
    FONT_FAMILIES_MONOSPACE = ['Fira Code', 'monospace']


# --- Launcher Dialog ---
class LauncherDialog(QDialog):
    """A modal dialog that serves as the application's entry point."""

    def __init__(self):
        super().__init__()
        self.selected_index = 0  # Default to the main view
        self.setWindowTitle('ATLAS Hub')
        self.setMinimumSize(400, 350)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_path = os.path.join(
            os.path.dirname(__file__), '..', 'assets', 'atlas_logo_light.png'
        )
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaledToWidth(300, Qt.SmoothTransformation))
            layout.addWidget(logo_label, alignment=Qt.AlignCenter)
        else:
            print(f'Warning: Logo file not found at {logo_path}')

        title_font = QFont()
        title_font.setFamilies(['Noto Sans', 'Sans Serif', 'Arial'])
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label = QLabel('Welcome to ATLAS')
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel('Click below to start managing your input files.')
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        button_font = QFont()
        button_font.setFamilies(['Noto Sans', 'Sans Serif', 'Arial'])
        button_font.setPointSize(12)

        self.start_button = QPushButton('Manage Input Files')
        self.start_button.setFont(button_font)
        self.start_button.setMinimumHeight(60)
        self.start_button.clicked.connect(self.accept)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addStretch()
        layout.addWidget(self.start_button)
        layout.addStretch()


# --- Worker Thread for Running External Processes ---
class ProcessWorker(QThread):
    """Run a command in a separate thread to keep the GUI responsive."""

    log_message = Signal(str)
    process_finished = Signal(int)

    def __init__(self, command_args):
        super().__init__()
        self.command_args = command_args
        self.process = None

    def run(self):
        try:
            self.log_message.emit(f'🚀 Running command: {" ".join(self.command_args)}')
            self.process = subprocess.Popen(
                self.command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
            )
            for line in iter(self.process.stdout.readline, ''):
                self.log_message.emit(line.strip())
            self.process.stdout.close()
            return_code = self.process.wait()
            self.process_finished.emit(return_code)
        except FileNotFoundError:
            self.log_message.emit(f"Error: Command not found '{self.command_args[0]}'.")
            self.process_finished.emit(-1)
        except Exception as e:
            self.log_message.emit(f'An unexpected error occurred: {e}')
            self.process_finished.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.log_message.emit('Process terminated by user.')


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """The main window of the application."""

    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1600, 900)
        self.setWindowTitle('ATLAS - Input File Management')
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.input_file_view = InputFileTab()
        main_layout.addWidget(self.input_file_view)

        self._create_menu_bar()

        log_group = QGroupBox('Logs & Output')
        log_layout = QVBoxLayout(log_group)
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        log_layout.addWidget(self.log_viewer)
        main_layout.addWidget(log_group)
        main_layout.setStretch(0, 5)
        main_layout.setStretch(1, 1)

        self.worker = None

        schema_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'config_schema.yaml'
        )
        self.input_file_view.load_schema(schema_path)

    def _create_menu_bar(self):
        """Creates the main menu bar for the application."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu('&File')

        load_action = QAction('&Load TOML...', self)
        load_action.triggered.connect(self.input_file_view.load_toml_from_dialog)
        file_menu.addAction(load_action)

        save_action = QAction('&Save TOML...', self)
        save_action.triggered.connect(self.input_file_view.save_to_file)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit Menu (Placeholder)
        edit_menu = menu_bar.addMenu('&Edit')
        undo_action = QAction('&Undo', self)
        undo_action.setEnabled(False)
        edit_menu.addAction(undo_action)
        redo_action = QAction('&Redo', self)
        redo_action.setEnabled(False)
        edit_menu.addAction(redo_action)

        # Format Menu
        format_menu = menu_bar.addMenu('F&ormat')
        theme_menu = format_menu.addMenu('Syntax Highlighting Theme')
        theme_group = QActionGroup(self)

        for theme_name in TomlHighlighter.THEMES:
            action = QAction(theme_name, self, checkable=True)
            if theme_name == 'Default':
                action.setChecked(True)
            action.triggered.connect(
                partial(self.input_file_view.set_highlighter_theme, theme_name)
            )
            theme_group.addAction(action)
            theme_menu.addAction(action)

    def log(self, message):
        self.log_viewer.append(message)

    def on_process_finished(self, return_code):
        self.log(f'\n✅ Process finished with exit code: {return_code}\n')


# --- Input File Management View ---
class InputFileTab(QWidget):
    """Manages UI for creating and editing TOML files based on a schema."""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.schema_data = None
        self.widgets_map = {}
        self.dynamic_widgets = {}
        self._last_highlight_path = None
        self._highlight_color = QColor('#ffe8a3')
        self._highlight_text_color = QColor('#1b1b1b')
        self._highlight_format = QTextCharFormat()
        self._highlight_format.setBackground(self._highlight_color)
        self._highlight_format.setForeground(self._highlight_text_color)
        self._highlight_format.setProperty(QTextFormat.FullWidthSelection, True)

        top_bar_layout = QHBoxLayout()
        top_bar_layout.addWidget(QLabel('Configuration Section:'))
        self.schema_selector = QComboBox()
        self.schema_selector.currentIndexChanged.connect(self.on_schema_selected)
        top_bar_layout.addWidget(self.schema_selector)
        top_bar_layout.addStretch()
        self.layout.addLayout(top_bar_layout)

        splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(splitter)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.container = QWidget()
        scroll_area.setWidget(self.container)
        splitter.addWidget(scroll_area)

        description_group = QGroupBox('Description')
        description_layout = QVBoxLayout(description_group)
        self.description_viewer = QTextEdit()
        self.description_viewer.setReadOnly(True)
        self.description_viewer.setPlaceholderText(
            'Welcome to the Input File Management tool!\n\n'
            'Select a configuration section to begin.\n'
            'Click on any input field to see its description here.'
        )
        description_layout.addWidget(self.description_viewer)
        splitter.addWidget(description_group)

        toml_group = QGroupBox('Live TOML Preview')
        toml_layout = QVBoxLayout(toml_group)
        self.toml_previewer = QTextEdit()
        self.toml_previewer.setReadOnly(False)
        previewer_font = QFont('monospace')
        previewer_font.setFamilies(ApplicationParameters.FONT_FAMILIES_MONOSPACE)
        self.toml_previewer.setFont(previewer_font)
        self.toml_previewer.setLineWrapMode(QTextEdit.NoWrap)
        self.highlighter = TomlHighlighter(self.toml_previewer.document())
        toml_layout.addWidget(self.toml_previewer)
        splitter.addWidget(toml_group)

        splitter.setSizes([600, 300, 400])

        self._updating_preview = False
        self._preview_modified = False
        self.toml_previewer.textChanged.connect(self.on_toml_text_changed)

        button_layout = QHBoxLayout()
        load_toml_button = QPushButton('Load TOML File')
        load_toml_button.clicked.connect(self.load_toml_from_dialog)
        save_button = QPushButton('Save Config TOML')
        save_button.clicked.connect(self.save_to_file)
        self.check_button = QPushButton('Check Config')
        self.check_button.clicked.connect(self.check_configuration)
        self.run_button = QPushButton('🚀 Run')
        self.run_button.clicked.connect(self.run_process)
        self.run_button.setEnabled(False)
        button_layout.addWidget(load_toml_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(self.check_button)
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        self.layout.addLayout(button_layout)

    def set_highlighter_theme(self, theme_name):
        """Creates a new highlighter with the selected theme and applies it."""
        theme_palette = TomlHighlighter.THEMES.get(theme_name, {})
        background_color = theme_palette.get('background')

        if background_color:
            self.toml_previewer.setStyleSheet(f'background-color: {background_color};')
        else:
            self.toml_previewer.setStyleSheet('')

        self.highlighter = TomlHighlighter(self.toml_previewer.document(), theme_name)
        self.highlighter.rehighlight()

    def eventFilter(self, source, event):
        if event.type() == QEvent.FocusIn:
            description = source.property('description')
            value_type = source.property('value_type')
            if description:
                display_text = (
                    f'{description}<br><br><b>Required type:</b> {value_type}'
                )
                self.description_viewer.setHtml(display_text)
            schema_path = source.property('schema_path')
            if schema_path:
                self._highlight_toml_entry(list(schema_path))
        return super().eventFilter(source, event)

    def load_schema(self, filepath):
        main_win = self.window()
        try:
            with open(filepath, encoding='utf-8') as f:
                self.schema_data = yaml.safe_load(f)
            self.schema_selector.clear()
            self.schema_selector.addItems(self.schema_data.keys())
            if main_win:
                main_win.log(f'✅ Schema loaded successfully from {filepath}')
        except Exception as e:
            if main_win:
                main_win.log(f"❌ Error loading schema '{filepath}': {e}")

    def on_schema_selected(self, index):
        if index == -1:
            return

        selected_key = self.schema_selector.currentText()
        schema_section = self.schema_data[selected_key]

        self._clear_layout()
        self.container = QWidget()
        scroll_area = self.layout.itemAt(1).widget().widget(0)
        scroll_area.setWidget(self.container)

        self.widgets_map = {}
        self.dynamic_widgets = {}
        self._last_highlight_path = None
        self._clear_preview_highlight()
        self._build_widgets_recursively(
            self.container, schema_section, self.widgets_map, 0, []
        )
        self.update_toml_preview()

        if selected_key == 'database_generation':
            self.run_button.setText('🚀 Run Database Generation')
            self.run_button.setEnabled(True)
        else:
            self.run_button.setText('🚀 Run')
            self.run_button.setEnabled(False)

    def _build_widgets_recursively(
        self,
        parent_widget,
        config_level,
        widget_storage,
        depth,
        path_prefix=None,
    ):
        path_prefix = list(path_prefix or [])
        layout = QFormLayout(parent_widget)

        for key, item in config_level.items():
            if not isinstance(item, dict):
                continue

            current_path = path_prefix + [key]

            if 'type' in item:
                label_text = item.get('name', key.replace('_', ' ').title())
                label = QLabel(label_text)
                label_font = QFont(application_font)

                is_mandatory = item.get('mandatory', False)
                if is_mandatory:
                    label_font.setBold(True)
                else:
                    label.setText(f'{label_text} (optional)')
                label.setFont(label_font)

                widget = self._create_widget(item)
                widget.installEventFilter(self)
                layout.addRow(label, widget)
                widget_storage[key] = widget
                widget.setProperty('schema_key', key)
                widget.setProperty('schema_path', tuple(current_path))

                if isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self.update_toml_preview)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(self.update_toml_preview)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
                    if isinstance(widget, QComboBox):
                        widget.currentIndexChanged.connect(self.update_toml_preview)
                    else:
                        widget.valueChanged.connect(self.update_toml_preview)

            elif 'description' in item:
                if item.get('dynamic_keys'):
                    self._handle_dynamic_section(
                        layout, key, item, widget_storage, depth, current_path
                    )
                else:
                    group_box = QGroupBox(
                        item.get('name_pretty', key.replace('_', ' ').title())
                    )
                    font = group_box.font()
                    base_size = application_font.pointSize() + 4
                    new_size = max(base_size - depth, application_font.pointSize() - 1)
                    font.setPointSize(new_size)
                    font.setBold(True)
                    group_box.setFont(font)

                    is_optional = not item.get('mandatory', True)
                    if is_optional:
                        group_box.setCheckable(True)
                        group_box.setChecked(True)
                        group_box.toggled.connect(self.update_toml_preview)

                    widget_storage[key] = {'_group': group_box}
                    self._build_widgets_recursively(
                        group_box,
                        item,
                        widget_storage[key],
                        depth + 1,
                        current_path,
                    )
                    layout.addRow(group_box)

    def _handle_dynamic_section(
        self, parent_layout, key, item, widget_storage, depth, path_prefix
    ):
        container_group = QGroupBox(key.replace('_', ' ').title())
        container_layout = QVBoxLayout(container_group)
        self.dynamic_widgets[key] = {
            'layout': container_layout,
            'template': item.get('schema', {}),
            'widgets': [],
            'path': tuple(path_prefix),
        }

        font = container_group.font()
        base_size = application_font.pointSize() + 4
        new_size = max(base_size - depth, application_font.pointSize() - 1)
        font.setPointSize(new_size)
        font.setBold(True)
        container_group.setFont(font)

        add_button = QPushButton(f'Add New {key.replace("_", " ").title()}')
        add_button.setFont(application_font)
        add_button.clicked.connect(lambda: self.add_dynamic_item(key))
        container_layout.addWidget(add_button)

        parent_layout.addRow(container_group)

    def add_dynamic_item(self, key, name='new_item', data=None):
        if key not in self.dynamic_widgets:
            return

        dyn_info = self.dynamic_widgets[key]
        item_widget_container = QWidget()
        h_layout = QHBoxLayout(item_widget_container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        item_group = QGroupBox(name)
        item_group.setCheckable(True)
        item_group.setChecked(True)
        item_group.toggled.connect(self.update_toml_preview)

        widget_storage = {}
        item_group.setProperty('widget_storage', widget_storage)
        self._build_widgets_recursively(
            item_group,
            dyn_info['template'],
            widget_storage,
            0,
            list(dyn_info['path']) + [name],
        )

        remove_button = QPushButton('Remove')
        remove_button.setFont(application_font)
        remove_button.clicked.connect(
            lambda: (
                item_widget_container.deleteLater(),
                self.update_toml_preview(),
                dyn_info['widgets'].remove(item_widget_container),
            )
        )

        h_layout.addWidget(item_group, 1)
        h_layout.addWidget(remove_button)

        dyn_info['layout'].insertWidget(
            dyn_info['layout'].count() - 1, item_widget_container
        )
        dyn_info['widgets'].append(item_widget_container)
        self.update_toml_preview()

    def update_toml_preview(self):
        if not self.widgets_map:
            return

        output_data = self._collect_data_recursively(self.widgets_map)

        try:
            toml_string = tomli_w.dumps(output_data or {})
            self._updating_preview = True
            self.toml_previewer.blockSignals(True)
            self.toml_previewer.setPlainText(toml_string)
            self.toml_previewer.blockSignals(False)
            self._updating_preview = False
            self._preview_modified = False
            self._refresh_preview_highlight()
        except Exception as e:
            self.toml_previewer.setPlainText(f'# Error generating TOML preview:\n# {e}')
            self._updating_preview = False
            self._preview_modified = False
            self._clear_preview_highlight()

    def on_toml_text_changed(self):
        if self._updating_preview:
            return
        self._preview_modified = True
        self._refresh_preview_highlight()

    def _parse_preview_dict(self):
        text = self.toml_previewer.toPlainText()
        if not text.strip():
            return None, 'TOML preview is empty.'
        try:
            parsed = tomli.loads(text)
        except tomli.TOMLDecodeError as exc:
            return None, f'TOML parsing error: {exc}'
        return parsed, None

    def _infer_section_from_data(self, data):
        if not isinstance(data, dict) or not self.schema_data:
            return None

        data_keys = set(data.keys())
        best_match = None
        best_score = (-1, float('-inf'))

        for section_name, section_schema in self.schema_data.items():
            if not isinstance(section_schema, dict):
                continue

            expected_keys = {
                key for key, value in section_schema.items() if isinstance(value, dict)
            }
            if not expected_keys:
                continue

            matches = len(data_keys & expected_keys)
            unexpected = len(data_keys - expected_keys)
            score = (matches, -unexpected)

            if score > best_score:
                best_match = section_name
                best_score = score

        if best_score[0] <= 0:
            return None
        return best_match

    def _highlight_toml_entry(self, path):
        if not path:
            self._clear_preview_highlight()
            return

        self._last_highlight_path = tuple(path)
        text = self.toml_previewer.toPlainText()
        index = self._locate_toml_index(text, path)
        if index is None:
            self._last_highlight_path = None
            self._clear_preview_highlight()
            return

        doc = self.toml_previewer.document()
        block = doc.findBlock(index)
        if not block.isValid():
            self._last_highlight_path = None
            self._clear_preview_highlight()
            return

        cursor = QTextCursor(block)
        cursor.select(QTextCursor.LineUnderCursor)

        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        selection.format = QTextCharFormat(self._highlight_format)

        self.toml_previewer.setExtraSelections([selection])
        self._ensure_cursor_visible(cursor)
        self.toml_previewer.viewport().update()

    def _clear_preview_highlight(self):
        self.toml_previewer.setExtraSelections([])
        self.toml_previewer.viewport().update()

    def _refresh_preview_highlight(self):
        if self._last_highlight_path:
            self._highlight_toml_entry(list(self._last_highlight_path))
        else:
            self._clear_preview_highlight()

    def _ensure_cursor_visible(self, cursor):
        self.toml_previewer.setTextCursor(cursor)
        self.toml_previewer.ensureCursorVisible()

    def _locate_toml_index(self, text, path):
        if not text or not path:
            return None

        section_start = 0
        split_index = 0

        for i in range(1, len(path)):
            header = f'[{".".join(path[:i])}]'
            header_idx = text.find(header, section_start)
            if header_idx == -1:
                break
            split_index = i
            newline_idx = text.find('\n', header_idx)
            if newline_idx == -1:
                section_start = len(text)
                break
            section_start = newline_idx + 1
        else:
            split_index = len(path) - 1

        section_end = text.find('\n[', section_start)
        if section_end == -1:
            section_end = len(text)
        segment = text[section_start:section_end]

        key_segments = path[split_index:]
        if not key_segments:
            key_segments = [path[-1]]

        key_expression = self._build_toml_dotted_key(key_segments)
        for pattern in (f'{key_expression} =', f'{key_expression}='):
            rel_idx = segment.find(pattern)
            if rel_idx != -1:
                return section_start + rel_idx

        for pattern in (f'{key_expression} =', f'{key_expression}='):
            rel_idx = text.find(pattern)
            if rel_idx != -1:
                return rel_idx

        return None

    def _build_toml_dotted_key(self, segments):
        parts = []
        for segment in segments:
            escaped = segment.replace('"', '\\"')
            if self._requires_toml_quotes(segment):
                parts.append(f'"{escaped}"')
            else:
                parts.append(segment)
        return '.'.join(parts)

    @staticmethod
    def _requires_toml_quotes(segment):
        if not segment:
            return True
        allowed = set(string.ascii_letters + string.digits + '_-')
        if segment[0].isdigit():
            return True
        return any(char not in allowed for char in segment)

    def _get_current_section_payload(self):
        parsed, error = self._parse_preview_dict()
        if error:
            return None, None, error

        selected_key = self.schema_selector.currentText()
        if not selected_key:
            return parsed, None, 'No configuration section selected.'

        if not isinstance(parsed, dict):
            return (
                parsed,
                selected_key,
                'TOML preview must define a configuration table.',
            )

        return parsed, selected_key, None

    def check_configuration(self):
        config_data, section_key, error = self._get_current_section_payload()
        section_label = section_key or '(none)'

        if error:
            self._report_validation_result(
                success=False,
                section=section_label,
                errors=[error],
            )
            return

        main_win = self.window() if isinstance(self.window(), MainWindow) else None
        if main_win:
            main_win.log(f'🔍 Checking configuration for section "{section_label}"...')

        is_valid, errors = self._validate_configuration(config_data, section_key)
        self._report_validation_result(
            success=is_valid, section=section_label, errors=errors
        )

    def save_to_file(self, filepath=None):
        parsed, error = self._parse_preview_dict()
        if error:
            if self.window():
                self.window().log(f'❌ Cannot save configuration: {error}')
            self._show_message('Save Failed', error, QMessageBox.Critical)
            return False

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                self, 'Save TOML Config', '', 'TOML Files (*.toml)'
            )
            if not filepath:
                return False

        with open(filepath, 'wb') as f:
            tomli_w.dump(parsed, f)
        if self.window():
            self.window().log(f'💾 Configuration saved to {filepath}')
        self._preview_modified = False
        return True

    def load_toml_from_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load TOML File', '', 'TOML Files (*.toml)'
        )
        if not filepath:
            return

        try:
            with open(filepath, 'rb') as f:
                data = tomli.load(f)

            if not data:
                raise ValueError('TOML file is empty.')

            section_key = self._infer_section_from_data(data)
            if not section_key:
                raise ValueError('Unable to determine configuration section for file.')

            self.schema_selector.setCurrentText(section_key)
            QApplication.processEvents()
            self._populate_widgets_recursively(self.widgets_map, data)
            self.update_toml_preview()
            if self.window():
                self.window().log(f'📄 Loaded TOML from {filepath}')

        except Exception as e:
            if self.window():
                self.window().log(f'❌ Error loading TOML file: {e}')

    def _populate_widgets_recursively(self, widget_level, data_level):
        for key, item in widget_level.items():
            if key not in data_level:
                continue

            value = data_level[key]
            if isinstance(item, dict) and '_group' in item:
                group_box = item['_group']
                if group_box.isCheckable():
                    group_box.setChecked(True)
                sub_widgets = {k: v for k, v in item.items() if k != '_group'}
                self._populate_widgets_recursively(sub_widgets, value)
            elif isinstance(item, QWidget):
                self._set_widget_value(item, value)

    def _set_widget_value(self, widget, value):
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QLineEdit):
            if isinstance(value, list):
                widget.setText(', '.join(map(str, value)))
            else:
                widget.setText(str(value))

    def _collect_data_recursively(self, widget_level):
        data = {}
        for key, item in widget_level.items():
            if isinstance(item, dict) and '_group' in item:
                group_box = item['_group']
                if group_box.isCheckable() and not group_box.isChecked():
                    continue
                sub_data_widgets = {k: v for k, v in item.items() if k != '_group'}
                data[key] = self._collect_data_recursively(sub_data_widgets)
            elif isinstance(item, QWidget):
                value = self._get_widget_value(item)
                if value is not None and value != '':
                    data[key] = value

        # Handle dynamic sections
        for key, dyn_info in self.dynamic_widgets.items():
            dyn_data = {}
            for item_widget in dyn_info['widgets']:
                group_box = item_widget.findChild(QGroupBox)
                if group_box and group_box.isChecked():
                    item_name = group_box.title()
                    widget_storage = group_box.property('widget_storage')
                    if widget_storage:
                        dyn_data[item_name] = self._collect_data_recursively(
                            widget_storage
                        )
            if dyn_data:
                data[key] = dyn_data

        return data

    def _validate_configuration(self, section_data, section_key):
        try:
            return cli_utils.validate_config_file(
                config_type=section_key,
                config_dict=section_data,
                allow_deprecated=False,
                run_mode='script',
            )
        except Exception as exc:
            return False, [str(exc)]

    def _collect_validation_errors(self, section_data, section_key):
        try:
            schema = cli_utils.get_schema()
            config_schema = schema.get(section_key)
            if not config_schema:
                return [f'Unknown configuration type: {section_key}']

            migrated_config, deprecation_warnings = cli_utils.check_deprecated_keys(
                section_data, config_schema
            )

            errors = cli_utils.validate_section_recursive(
                migrated_config,
                config_schema,
                root_config_data=migrated_config,
                original_schema_dict=config_schema,
            )

            if deprecation_warnings:
                errors.extend(deprecation_warnings)
                errors.append(
                    'Configuration contains deprecated keys. '
                    'Please update your file to use the new key names.'
                )
            return errors
        except Exception as exc:
            return [str(exc)]

    def _report_validation_result(self, success, section, errors):
        main_win = self.window() if isinstance(self.window(), MainWindow) else None

        if success:
            message = f'Configuration for section "{section}" is valid.'
            if main_win:
                main_win.log(f'✅ {message}')
            self._show_message('Configuration Valid', message, QMessageBox.Information)
            return

        errors = errors or ['Validation failed for unknown reasons.']
        header = f'Configuration validation failed for section "{section}".'
        if main_win:
            main_win.log(f'❌ {header}')
            for err in errors:
                main_win.log(f'   - {err}')

        details = '\n'.join(errors)
        self._show_message(
            'Configuration Invalid',
            header,
            QMessageBox.Critical,
            details=details,
        )

    def _show_message(self, title, text, icon, details=None):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        if details:
            msg_box.setDetailedText(details)
        msg_box.exec()

    def run_process(self):
        selected_key = self.schema_selector.currentText()
        if selected_key != 'database_generation':
            return

        config_data, _, error = self._get_current_section_payload()
        if error:
            window = self.window()
            if window:
                window.log(f'❌ Cannot run database generation: {error}')
            self._show_message('Run Aborted', error, QMessageBox.Critical)
            return

        temp_config_path = 'temp_db_gen_config.toml'
        try:
            with open(temp_config_path, 'wb') as temp_file:
                tomli_w.dump(config_data, temp_file)
        except Exception as exc:
            window = self.window()
            if window:
                window.log(f'❌ Failed to prepare configuration: {exc}')
            self._show_message(
                'Run Aborted',
                f'Failed to prepare temporary configuration file: {exc}',
                QMessageBox.Critical,
            )
            return

        command = ['atl_gen_init_db', 'generate', '-c', temp_config_path]
        main_win = self.window()
        main_win.log_viewer.clear()
        main_win.worker = ProcessWorker(command)
        main_win.worker.log_message.connect(main_win.log)
        main_win.worker.process_finished.connect(main_win.on_process_finished)
        main_win.worker.start()

    def _create_widget(self, item_def):
        widget_type_str = item_def.get('type', 'str')
        default_value = item_def.get('default')
        choices = item_def.get('choices')

        default_font = QFont()

        if default_value == 'None':
            default_value = None

        if choices:
            widget = QComboBox()
            widget.addItems(map(str, choices))
            if default_value is not None:
                widget.setCurrentText(str(default_value))
        elif 'list' in widget_type_str:
            widget = QLineEdit(', '.join(map(str, default_value or [])))
        elif 'bool' in widget_type_str:
            widget = QCheckBox()
            if default_value is not None:
                widget.setChecked(bool(default_value))
        elif 'int' in widget_type_str:
            widget = QSpinBox()
            widget.setRange(-1_000_000, 1_000_000_000)
            if default_value is not None:
                widget.setValue(int(default_value))
        elif 'float' in widget_type_str:
            widget = QDoubleSpinBox()
            widget.setRange(-1e9, 1e9)
            widget.setDecimals(5)
            if default_value is not None:
                widget.setValue(float(default_value))
        else:  # str, PosixPath
            widget = QLineEdit(str(default_value or ''))

        widget.setFont(default_font)
        widget.setProperty(
            'description', item_def.get('description', 'No description available.')
        )
        widget.setProperty('value_type', item_def.get('type', 'N/A'))
        desc = item_def.get('description', '')
        if desc:
            widget.setToolTip(f'<p>{desc}</p>')
        return widget

    def _get_widget_value(self, widget):
        value_type = widget.property('value_type')

        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            text = widget.currentText()
            try:
                if 'int' in value_type:
                    return int(text)
                if 'float' in value_type:
                    return float(text)
            except (ValueError, TypeError):
                pass
            return text
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if not text:
                return None

            if 'list' in value_type:
                items = [item.strip() for item in text.split(',')]
                # Basic type inference for list items
                typed_items = []
                for item in items:
                    try:
                        typed_items.append(int(item))
                    except ValueError:
                        try:
                            typed_items.append(float(item))
                        except ValueError:
                            typed_items.append(item)
                return typed_items

            try:
                if 'int' in value_type:
                    return int(text)
                if 'float' in value_type:
                    return float(text)
            except (ValueError, TypeError):
                pass
            return text
        return None

    def _clear_layout(self):
        if self.container is not None:
            new_container = QWidget()
            scroll_area = self.layout.itemAt(1).widget().widget(0)
            scroll_area.setWidget(new_container)
            self.container = new_container
        self.widgets_map = {}
        self.dynamic_widgets = {}
        self._last_highlight_path = None
        self._clear_preview_highlight()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    application_font = app.font()
    application_font.setFamilies(ApplicationParameters.FONT_FAMILIES_REGULAR)
    launcher = LauncherDialog()

    if launcher.exec() == QDialog.Accepted:
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
