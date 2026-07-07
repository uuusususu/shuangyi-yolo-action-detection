"""Native Qt panel widgets used by runtime pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy

from ui.runtime_ui_tokens import (
    CHIP_BG,
    FONT_BODY,
    FONT_SECTION,
    FONT_SMALL,
    PANEL_BG_ALT,
    STROKE_MAIN,
    STROKE_SOFT,
    TEXT_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    frame_style,
    text_style,
)


class PanelSection(QFrame):
    """Container panel with optional title."""

    def __init__(self, title: str, *, background: str = PANEL_BG_ALT, parent=None):
        super().__init__(parent)
        self.setStyleSheet(frame_style(background, border=STROKE_MAIN, radius=12))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 14, 14, 14)
        self._layout.setSpacing(10)
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(text_style(TEXT_ACCENT, size=FONT_SECTION, weight=700))
        self._layout.addWidget(self.title_label)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._layout


class MetricCard(QFrame):
    """Title + value card."""

    def __init__(self, title: str, value: str = "0", parent=None):
        super().__init__(parent)
        self.setStyleSheet(frame_style(CHIP_BG, border=STROKE_SOFT, radius=8))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(0)
        self.title_label = QLabel(title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(text_style(TEXT_SECONDARY, size=10, weight=700))
        self.value_label = QLabel(value, self)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet(text_style(TEXT_PRIMARY, size=14, weight=800))
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def replace_value_widget(self, widget: QLabel) -> None:
        layout = self.layout()
        if layout is None:
            return
        layout.removeWidget(self.value_label)
        self.value_label.deleteLater()
        self.value_label = widget
        widget.setParent(self)
        layout.addWidget(widget)


class DefectChip(QFrame):
    """Name + percentage chip."""

    def __init__(self, name: str, value: str = "0.00%", parent=None):
        super().__init__(parent)
        self.setStyleSheet(frame_style(CHIP_BG, border=STROKE_SOFT, radius=8))
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        self.name_label = QLabel(name, self)
        self.name_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))
        self.value_label = QLabel(value, self)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.value_label)


class StatusBadge(QLabel):
    """Read-only status badge."""

    def __init__(self, text: str = "READY", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(92)
        self.setFixedHeight(28)
        self.set_status(text)

    def set_status(self, text: str, *, active: bool = False) -> None:
        text = str(text)
        if active or text.upper() == "PASS":
            background = "#1C8E4F"
            border = "#52E896"
            color = "#E8FFF0"
        elif text.upper() == "TESTING":
            background = "#9B6A00"
            border = "#FFD34D"
            color = "#FFF5C7"
        else:
            background = "#155084"
            border = "#35D7FF"
            color = TEXT_PRIMARY
        self.setText(text)
        self.setStyleSheet(
            f"background-color: {background}; border: 1px solid {border}; border-radius: 6px; "
            f"color: {color}; font-size: 13px; font-weight: 700; padding: 2px 10px;"
        )


class ResultRowWidget(QFrame):
    """Three-column result row."""

    def __init__(self, index: int, step_name: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: transparent; }")
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.index_label = QLabel(str(index), self)
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_label.setFixedHeight(28)
        self.index_label.setStyleSheet(
            "background-color: #1299A7; color: #FFFFFF; border-radius: 6px; "
            "font-size: 14px; font-weight: 800;"
        )

        self.step_label = QLabel(step_name, self)
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.step_label.setFixedHeight(28)
        self.step_label.setStyleSheet(
            f"background-color: {PANEL_BG_ALT}; border: 1px solid {STROKE_MAIN}; border-radius: 6px; "
            f"color: {TEXT_PRIMARY}; font-size: 15px; padding: 4px 12px; font-weight: 700;"
        )

        self.result_badge = StatusBadge("READY", self)

        layout.addWidget(self.index_label, 1)
        layout.addWidget(self.step_label, 5)
        layout.addWidget(self.result_badge, 2)


class ThumbnailTile(QFrame):
    """Thumbnail + status tile."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(frame_style(PANEL_BG_ALT, border=STROKE_SOFT, radius=8))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel(title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))

        self.preview_label = QLabel("等待画面", self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(72)
        self.preview_label.setStyleSheet(
            f"background-color: #0E1C3F; border: 1px dashed {STROKE_SOFT}; color: #4C78A6; "
            f"border-radius: 6px; font-size: {FONT_SMALL}px;"
        )

        self.status_badge = StatusBadge("READY", self)
        self.status_badge.setMinimumWidth(0)

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.status_badge)
