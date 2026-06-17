from pathlib import Path


SURFACE_BASE = "#0f131a"
SURFACE_ELEVATED = "#171d27"
SURFACE_ALT = "#111722"
SURFACE_INTERACTIVE = "#1f2734"
SURFACE_INTERACTIVE_HOVER = "#273243"
BORDER_SUBTLE = "#2c3748"
TEXT_PRIMARY = "#e6edf7"
TEXT_MUTED = "#a9b7cb"
ACCENT = "#59b6ff"
ACCENT_SOFT = "#2f6b98"
TIMELINE_BACKGROUND = "#0c1118"
TIMELINE_GRID = "#2a3444"
TIMELINE_PLAYHEAD = "#ffb454"

CHECKBOX_CHECK_ICON = Path(__file__).with_name("checkbox_check.svg").as_posix()


WINDOW_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SURFACE_BASE};
    color: {TEXT_PRIMARY};
    font-size: 12px;
}}
QLabel {{
    color: {TEXT_PRIMARY};
}}
QLabel[role="muted"] {{
    color: {TEXT_MUTED};
}}
QLabel[role="inspectorTitle"] {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 600;
}}
QLabel[role="presetName"] {{
    color: {TEXT_PRIMARY};
    font-weight: 600;
}}
QWidget#presetPanel {{
    background-color: {SURFACE_ELEVATED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}
QWidget#channelPanel {{
    background-color: {SURFACE_ELEVATED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}
QWidget#thresholdPanel {{
    background-color: {SURFACE_ELEVATED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}
QWidget#centerWorkspace {{
    background-color: {SURFACE_BASE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}
QWidget#transportPanel {{
    background-color: {SURFACE_BASE};
}}
QWidget#transportBox {{
    background-color: {SURFACE_ELEVATED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}
QListWidget#presetList {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    outline: none;
}}
QListWidget#presetList::item {{
    padding: 4px 8px;
    min-height: 22px;
}}
QListWidget#presetList::item:hover {{
    background-color: {SURFACE_INTERACTIVE};
}}
QListWidget#presetList::item:selected {{
    background-color: {ACCENT_SOFT};
    color: {TEXT_PRIMARY};
}}
QPushButton {{
    background-color: {SURFACE_INTERACTIVE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {SURFACE_INTERACTIVE_HOVER};
}}
QPushButton:pressed {{
    background-color: {SURFACE_ALT};
}}
QToolButton {{
    background-color: {SURFACE_INTERACTIVE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 24px;
    min-width: 96px;
}}
QToolButton:hover {{
    background-color: {SURFACE_INTERACTIVE_HOVER};
}}
QToolButton:pressed {{
    background-color: {SURFACE_ALT};
}}
QToolButton[role="transport"] {{
    background-color: #2f3a4d;
    color: #ffffff;
    border: 1px solid #7f91ad;
    border-radius: 4px;
    min-width: 18px;
    min-height: 18px;
    max-width: 22px;
    max-height: 22px;
    padding: 0;
    font-size: 11px;
    font-weight: 700;
}}
QToolButton[role="transport"]:hover {{
    background-color: #3d4b62;
    border-color: #b5c7e2;
}}
QToolButton[role="transport"]:pressed {{
    background-color: #52627c;
}}
QToolButton[role="presetIcon"], QToolButton[role="presetGlobal"] {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    min-width: 20px;
    min-height: 20px;
    max-width: 22px;
    max-height: 22px;
    padding: 0;
}}
QToolButton[role="presetIcon"]:hover, QToolButton[role="presetGlobal"]:hover {{
    background-color: {SURFACE_INTERACTIVE};
    border-color: {BORDER_SUBTLE};
}}
QToolButton[role="presetIcon"]:pressed, QToolButton[role="presetGlobal"]:pressed {{
    background-color: {SURFACE_ALT};
}}
QToolButton#transportPlayButton {{
    min-width: 20px;
    min-height: 18px;
    max-width: 24px;
    max-height: 22px;
}}
QToolButton#transportLoopButton {{
    min-width: 40px;
    max-width: 40px;
}}
QSlider::groove:horizontal {{
    background: {SURFACE_INTERACTIVE};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QComboBox, QLineEdit, QDoubleSpinBox {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    min-height: 24px;
    padding: 2px 8px;
    selection-background-color: {ACCENT_SOFT};
}}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover {{
    border-color: {ACCENT_SOFT};
}}
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QCheckBox {{
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    background-color: {SURFACE_INTERACTIVE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 3px;
}}
QCheckBox::indicator:hover {{
    background-color: {SURFACE_INTERACTIVE_HOVER};
    border-color: {ACCENT_SOFT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: url("{CHECKBOX_CHECK_ICON}");
}}
QCheckBox::indicator:disabled {{
    background-color: {SURFACE_ALT};
    border-color: {BORDER_SUBTLE};
}}
QTableWidget {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    gridline-color: {BORDER_SUBTLE};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT_PRIMARY};
}}
QTableWidget#xarrayEventsTable {{
    alternate-background-color: {SURFACE_INTERACTIVE};
}}
QTableView::item {{
    padding: 3px 6px;
    border: none;
}}
QHeaderView::section {{
    background-color: {SURFACE_ELEVATED};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER_SUBTLE};
    padding: 5px 8px;
    font-weight: 600;
}}
QSplitter::handle {{
    background: {SURFACE_ELEVATED};
}}
QSplitter::handle:vertical {{
    min-height: 8px;
    border-top: 1px solid {BORDER_SUBTLE};
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
QSplitter::handle:vertical:hover {{
    background: {ACCENT_SOFT};
}}
QSplitter::handle:horizontal {{
    min-width: 8px;
    border-left: 1px solid {BORDER_SUBTLE};
    border-right: 1px solid {BORDER_SUBTLE};
}}
QSplitter::handle:horizontal:hover {{
    background: {ACCENT_SOFT};
}}
QMenu {{
    background-color: {SURFACE_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
}}
QMenu::item {{
    padding: 8px 18px;
    min-height: 22px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_SOFT};
}}
"""
