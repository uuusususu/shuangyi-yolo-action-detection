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
