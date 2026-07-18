"""Application-wide open-source Qt control theme integration."""

from __future__ import annotations

from importlib import import_module

from PySide6.QtWidgets import QApplication

from ui.runtime_ui_tokens import (
    BTN_DANGER_BG,
    BTN_DANGER_BORDER,
    BTN_PRIMARY_BG,
    BTN_PRIMARY_BORDER,
    BTN_PRIMARY_HOVER,
    INPUT_BG,
    PANEL_BG_DARK,
    PANEL_BG_SOFT,
    STROKE_MAIN,
    STROKE_MUTED,
    TEXT_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

THEME_PACKAGE = "pyqtdarktheme"
THEME_IMPORT = "qdarktheme"
THEME_VERSION = "2.1.0"
THEME_ID = f"{THEME_PACKAGE}-{THEME_VERSION}"
PRIMARY_COLOR = "#35D7FF"

# The library owns geometry, icons, arrows, indicators, and base states. These
# rules only map its standard controls onto the existing industrial palette.
SEMANTIC_CONTROL_QSS = f"""
QPushButton {{
    min-height: 40px;
    padding: 4px 12px;
    color: {TEXT_PRIMARY};
    background-color: {PANEL_BG_SOFT};
    border-color: {STROKE_MAIN};
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #19345F;
    border-color: {TEXT_ACCENT};
}}
QPushButton:pressed {{
    background-color: {PANEL_BG_DARK};
    border-color: {TEXT_ACCENT};
}}
QPushButton:focus {{ border-color: {TEXT_ACCENT}; }}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: #0A1428;
    border-color: {STROKE_MUTED};
}}
QPushButton[buttonRole="primary"] {{
    color: #FFFFFF;
    background-color: {BTN_PRIMARY_BG};
    border-color: {BTN_PRIMARY_BORDER};
    font-weight: 700;
}}
QPushButton[buttonRole="primary"]:hover {{
    background-color: {BTN_PRIMARY_HOVER};
    border-color: {TEXT_ACCENT};
}}
QPushButton[buttonRole="danger"] {{
    color: #FFFFFF;
    background-color: {BTN_DANGER_BG};
    border-color: {BTN_DANGER_BORDER};
    font-weight: 700;
}}
QPushButton[buttonRole="danger"]:hover {{
    background-color: #A82838;
    border-color: #FF9AA3;
}}
QPushButton[navigation="true"] {{
    text-align: left;
    padding-left: 12px;
}}
QPushButton[navigation="true"]:checked {{
    color: {TEXT_ACCENT};
    background-color: #17345F;
    border-color: {TEXT_ACCENT};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 38px;
    color: {TEXT_PRIMARY};
    background-color: {INPUT_BG};
    border-color: {STROKE_MAIN};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {TEXT_ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {TEXT_MUTED};
    background-color: {PANEL_BG_DARK};
    border-color: {STROKE_MUTED};
}}
QComboBox QAbstractItemView {{
    color: {TEXT_PRIMARY};
    background-color: {PANEL_BG_DARK};
    border-color: {STROKE_MAIN};
    selection-background-color: #174D8F;
}}
"""


def require_theme_module():
    """Load the required theme module with an actionable failure message."""
    try:
        return import_module(THEME_IMPORT)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"缺少界面主题依赖 {THEME_PACKAGE}=={THEME_VERSION}，请重新安装 requirements.txt"
        ) from exc


def apply_application_theme(app: QApplication) -> None:
    """Apply the shared dark control theme before any window is created."""
    theme = require_theme_module()
    theme.setup_theme(
        theme="dark",
        corner_shape="rounded",
        custom_colors={"primary": PRIMARY_COLOR},
        additional_qss=SEMANTIC_CONTROL_QSS,
    )
    app.setProperty("openSourceControlTheme", THEME_ID)
