"""Shared runtime UI sizing and style helpers."""

from __future__ import annotations

FONT_FAMILY = "Microsoft YaHei UI"

PAGE_BG = "#071226"
PANEL_BG = "#0D1D39"
PANEL_BG_DARK = "#09172F"
PANEL_BG_MID = "#11284C"
PANEL_BG_SOFT = "#10213F"
PANEL_BG_ALT = "#102442"
INPUT_BG = "#08162D"
CHIP_BG = "#102442"
HEADER_BG = "#0B1B35"
TRACK_BG = "#17385F"
VIEWPORT_BG = "#050D1B"

STROKE_MAIN = "#24558A"
STROKE_SOFT = "#315A82"
STROKE_MUTED = "#263C58"

TEXT_PRIMARY = "#EDF6FF"
TEXT_SECONDARY = "#A9C0DA"
TEXT_ACCENT = "#35D7FF"
TEXT_MUTED = "#87A2C2"
TEXT_SUCCESS = "#7CE9BE"
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
BTN_PRIMARY_BG = "#1764B8"
BTN_PRIMARY_BORDER = "#2C7BD2"
BTN_PRIMARY_HOVER = "#1C72C9"
BTN_DANGER_BG = "#8E2230"
BTN_DANGER_BORDER = "#FF6A75"

PAGE_MARGIN = 14
PAGE_SPACING = 12
CARD_PADDING = 16
CARD_SPACING = 10
CARD_RADIUS = 8
CONFIG_NAV_WIDTH = 230
TOP_BAR_HEIGHT = 56
PRIMARY_CONTROL_HEIGHT = 52

FONT_TITLE = 18
FONT_SECTION = 16
FONT_BODY = 14
FONT_SMALL = 12
FONT_MICRO = 11
FONT_METRIC = 26
FONT_METRIC_LARGE = 42
FONT_TAG = 14

MIN_BUTTON_HEIGHT = 40
ROW_HEIGHT = 44
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
        "border-radius: 8px; "
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
