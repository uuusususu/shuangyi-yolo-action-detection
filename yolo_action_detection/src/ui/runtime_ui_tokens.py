"""Shared runtime UI sizing and style helpers."""

from __future__ import annotations

FONT_FAMILY = "Microsoft YaHei UI"

PAGE_BG = "#0B1636"
PANEL_BG = "#101F43"
PANEL_BG_DARK = "#09132A"
PANEL_BG_MID = "#102247"
PANEL_BG_SOFT = "#11244C"
PANEL_BG_ALT = "#122A56"
INPUT_BG = "#19345F"
CHIP_BG = "#17345E"
HEADER_BG = "#12386F"
TRACK_BG = "#164B7A"
VIEWPORT_BG = "#0A1228"

STROKE_MAIN = "#2876D9"
STROKE_SOFT = "#2E6DC2"
STROKE_MUTED = "#274C77"

TEXT_PRIMARY = "#D7F8FF"
TEXT_SECONDARY = "#8FEFFF"
TEXT_ACCENT = "#73F2FF"
TEXT_MUTED = "#4C78A6"
TEXT_SUCCESS = "#ACFFC6"
TEXT_WARNING = "#FFD34D"
TEXT_DANGER = "#FF6A75"

# 步骤卡片状态色（科技风深色底 + 状态描边/文字）
STEP_WAITING_BG = "#0E1C3F"
STEP_WAITING_BORDER = "#274C77"
STEP_WAITING_TEXT = "#6E8FB5"

STEP_ACTIVE_BG = "#0F2A5C"
STEP_ACTIVE_BORDER = "#35D7FF"
STEP_ACTIVE_TEXT = "#9BEFFF"

STEP_PASS_BG = "#0F3A28"
STEP_PASS_BORDER = "#52E896"
STEP_PASS_TEXT = "#ACFFC6"

STEP_NG_BG = "#3A1320"
STEP_NG_BORDER = "#FF6A75"
STEP_NG_TEXT = "#FFB3BC"

STEP_LOCKED_BG = "#0A1428"
STEP_LOCKED_BORDER = "#1B2C45"
STEP_LOCKED_TEXT = "#3A5575"

# 主按钮状态色
BTN_PRIMARY_BG = "#1668C9"
BTN_PRIMARY_BORDER = "#4FA3F5"
BTN_PRIMARY_HOVER = "#1B79E6"
BTN_SECONDARY_BG = "#19345F"
BTN_SECONDARY_BORDER = "#2E6DC2"
BTN_SECONDARY_HOVER = "#21416E"
BTN_DANGER_BG = "#8E2230"
BTN_DANGER_BORDER = "#FF6A75"

PAGE_MARGIN = 20
PAGE_SPACING = 16
CARD_PADDING = 16
CARD_SPACING = 12
CARD_RADIUS = 12

FONT_TITLE = 18
FONT_SECTION = 16
FONT_BODY = 14
FONT_SMALL = 12
FONT_MICRO = 11
FONT_METRIC = 26
FONT_METRIC_LARGE = 42
FONT_TAG = 14

MIN_BUTTON_HEIGHT = 34
TAG_BUTTON_HEIGHT = 34
ROW_HEIGHT = 42
TABLE_ROW_HEIGHT = 36

DEFAULT_WINDOW_WIDTH = 1366
DEFAULT_WINDOW_HEIGHT = 920


def frame_style(background: str = PANEL_BG, *, border: str = STROKE_MAIN, radius: int = CARD_RADIUS) -> str:
    return (
        "QFrame { "
        f"background-color: {background}; "
        f"border: 1px solid {border}; "
        f"border-radius: {radius}px; "
        "}"
    )


def text_style(
    color: str = TEXT_PRIMARY,
    *,
    size: int = FONT_BODY,
    weight: int | str = 500,
    extra: str = "",
) -> str:
    extra_rules = f" {extra.strip()}" if extra.strip() else ""
    return (
        f"color: {color}; "
        f"font-size: {size}px; "
        f"font-weight: {weight}; "
        "background: transparent; border: none;"
        f"{extra_rules}"
    )


def chip_button_style(
    *,
    background: str,
    border: str,
    color: str,
    font_size: int = FONT_BODY,
    min_height: int = TAG_BUTTON_HEIGHT,
    padding: str = "6px 12px",
    radius: int = 8,
    disabled_background: str | None = None,
    disabled_border: str | None = None,
    disabled_color: str = TEXT_MUTED,
) -> str:
    disabled_bg = disabled_background or background
    disabled_border_color = disabled_border or border
    return (
        "QPushButton { "
        f"background-color: {background}; "
        f"border: 2px solid {border}; "
        f"color: {color}; "
        f"font-size: {font_size}px; "
        "font-weight: 700; "
        f"padding: {padding}; "
        f"border-radius: {radius}px; "
        f"min-height: {min_height}px; "
        "}"
        "QPushButton:hover { "
        f"border: 2px solid {border}; "
        "}"
        "QPushButton:disabled { "
        f"background-color: {disabled_bg}; "
        f"border: 2px solid {disabled_border_color}; "
        f"color: {disabled_color}; "
        "}"
    )


def table_style() -> str:
    return f"""
            QTableWidget {{
                background-color: {PANEL_BG_MID};
                border: 1px solid {STROKE_MAIN};
                border-radius: 8px;
                color: {TEXT_PRIMARY};
                gridline-color: #234E88;
                font-size: {FONT_SMALL}px;
            }}
            QHeaderView::section {{
                background-color: #143562;
                color: {TEXT_ACCENT};
                padding: 7px 8px;
                border: 1px solid {STROKE_MAIN};
                font-size: {FONT_SMALL}px;
                font-weight: 700;
            }}
            """


def scroll_area_style() -> str:
    return (
        "QScrollArea { background: transparent; border: none; }"
        f"QScrollBar:vertical {{ background: {PANEL_BG_DARK}; width: 8px; border-radius: 4px; }}"
        f"QScrollBar::handle:vertical {{ background: {STROKE_MAIN}; border-radius: 4px; min-height: 28px; }}"
        f"QScrollBar::handle:vertical:hover {{ background: {TEXT_ACCENT}; }}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
    )


def group_box_style() -> str:
    return (
        "QGroupBox { "
        f"background-color: {PANEL_BG}; "
        f"border: 1px solid {STROKE_MAIN}; "
        "border-radius: 10px; "
        "margin-top: 18px; "
        "padding: 14px 12px 12px 12px; "
        f"color: {TEXT_PRIMARY}; "
        f"font-family: '{FONT_FAMILY}'; "
        f"font-size: {FONT_BODY}px; "
        "font-weight: 700; "
        "}"
        "QGroupBox::title { "
        "subcontrol-origin: margin; "
        "left: 12px; "
        "padding: 0 8px; "
        f"color: {TEXT_ACCENT}; "
        f"background-color: {PANEL_BG}; "
        "}"
    )


def form_label_style() -> str:
    return text_style(TEXT_SECONDARY, size=FONT_BODY, weight=600)


def hint_label_style() -> str:
    return text_style(TEXT_MUTED, size=FONT_SMALL, weight=500)


def input_style() -> str:
    return (
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { "
        f"background-color: {INPUT_BG}; "
        f"border: 1px solid {STROKE_MUTED}; "
        "border-radius: 7px; "
        f"color: {TEXT_PRIMARY}; "
        f"font-family: '{FONT_FAMILY}'; "
        f"font-size: {FONT_BODY}px; "
        "padding: 6px 8px; "
        "min-height: 30px; "
        "}"
        "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { "
        f"border: 1px solid {TEXT_ACCENT}; "
        "}"
        "QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { "
        f"background-color: {PANEL_BG_DARK}; "
        f"color: {TEXT_MUTED}; "
        "}"
        "QComboBox::drop-down { "
        f"border-left: 1px solid {STROKE_MUTED}; "
        "width: 24px; "
        "}"
        "QComboBox QAbstractItemView { "
        f"background-color: {PANEL_BG_DARK}; "
        f"border: 1px solid {STROKE_MAIN}; "
        f"color: {TEXT_PRIMARY}; "
        "selection-background-color: #174D8F; "
        "}"
    )


def checkbox_style() -> str:
    return (
        "QCheckBox { "
        f"color: {TEXT_PRIMARY}; "
        f"font-family: '{FONT_FAMILY}'; "
        f"font-size: {FONT_BODY}px; "
        "spacing: 8px; "
        "}"
        "QCheckBox::indicator { width: 18px; height: 18px; }"
        "QCheckBox::indicator:unchecked { "
        f"background-color: {PANEL_BG_DARK}; "
        f"border: 1px solid {STROKE_MUTED}; "
        "border-radius: 4px; "
        "}"
        "QCheckBox::indicator:checked { "
        f"background-color: {BTN_PRIMARY_BG}; "
        f"border: 1px solid {TEXT_ACCENT}; "
        "border-radius: 4px; "
        "}"
    )


def config_button_style(variant: str = "primary") -> str:
    if variant == "primary":
        return chip_button_style(
            background=BTN_PRIMARY_BG,
            border=BTN_PRIMARY_BORDER,
            color="#FFFFFF",
            font_size=FONT_BODY,
            min_height=36,
            padding="7px 14px",
            radius=8,
            disabled_background="#123258",
            disabled_border="#1B3A5C",
        )
    if variant == "danger":
        return chip_button_style(
            background=BTN_DANGER_BG,
            border=BTN_DANGER_BORDER,
            color="#FFFFFF",
            font_size=FONT_BODY,
            min_height=36,
            padding="7px 14px",
            radius=8,
            disabled_background="#2A1018",
            disabled_border="#3A1C24",
        )
    return chip_button_style(
        background=BTN_SECONDARY_BG,
        border=BTN_SECONDARY_BORDER,
        color=TEXT_PRIMARY,
        font_size=FONT_BODY,
        min_height=36,
        padding="7px 14px",
        radius=8,
        disabled_background=PANEL_BG_DARK,
        disabled_border=STEP_LOCKED_BORDER,
    )
