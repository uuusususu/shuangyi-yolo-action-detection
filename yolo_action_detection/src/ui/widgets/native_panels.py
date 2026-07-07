"""Native Qt panel widgets used by runtime pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy

from ui.runtime_ui_tokens import (
    BTN_DANGER_BG,
    BTN_DANGER_BORDER,
    BTN_PRIMARY_BG,
    BTN_PRIMARY_BORDER,
    BTN_PRIMARY_HOVER,
    BTN_SECONDARY_BG,
    BTN_SECONDARY_BORDER,
    BTN_SECONDARY_HOVER,
    CHIP_BG,
    FONT_BODY,
    FONT_METRIC,
    FONT_SECTION,
    FONT_SMALL,
    FONT_TITLE,
    PANEL_BG,
    PANEL_BG_ALT,
    PANEL_BG_DARK,
    STEP_ACTIVE_BG,
    STEP_ACTIVE_BORDER,
    STEP_ACTIVE_TEXT,
    STEP_LOCKED_BG,
    STEP_LOCKED_BORDER,
    STEP_LOCKED_TEXT,
    STEP_NG_BG,
    STEP_NG_BORDER,
    STEP_NG_TEXT,
    STEP_PASS_BG,
    STEP_PASS_BORDER,
    STEP_PASS_TEXT,
    STEP_WAITING_BG,
    STEP_WAITING_BORDER,
    STEP_WAITING_TEXT,
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


# ---------------------------------------------------------------------------
# 步骤检测卡片（运行时主界面核心控件）
# ---------------------------------------------------------------------------

# 步骤卡片状态映射到 (背景, 描边, 文字, 状态文案, 辅助提示)
_STEP_STATE_STYLE = {
    "waiting": (STEP_WAITING_BG, STEP_WAITING_BORDER, STEP_WAITING_TEXT, "等待", "等待上一步通过"),
    "active": (STEP_ACTIVE_BG, STEP_ACTIVE_BORDER, STEP_ACTIVE_TEXT, "检测中", "请执行当前步骤"),
    "pass": (STEP_PASS_BG, STEP_PASS_BORDER, STEP_PASS_TEXT, "PASS", "已完成"),
    "ng": (STEP_NG_BG, STEP_NG_BORDER, STEP_NG_TEXT, "NG", "请重新执行当前步骤"),
    "locked": (STEP_LOCKED_BG, STEP_LOCKED_BORDER, STEP_LOCKED_TEXT, "锁定", "等待纠正前置步骤"),
}


class StepCard(QFrame):
    """大尺寸步骤检测卡片。

    显示步骤序号(01/02/...)、名称、状态徽章和辅助提示。
    通过 ``set_step_state`` 更新状态，状态色由统一 token 驱动。
    """

    def __init__(self, index: int, name: str, parent=None):
        super().__init__(parent)
        self._index = index
        self.setMinimumHeight(104)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(14)

        # 序号圆角块
        self._index_label = QLabel(f"{index:02d}", self)
        self._index_label.setFixedSize(56, 56)
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._index_label, 0)

        # 名称 + 提示
        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(4)
        self._name_label = QLabel(name, self)
        self._name_label.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_SECTION, weight=700))
        self._hint_label = QLabel("等待上一步通过", self)
        self._hint_label.setStyleSheet(text_style(STEP_WAITING_TEXT, size=FONT_SMALL, weight=500))
        text_box.addWidget(self._name_label)
        text_box.addWidget(self._hint_label)
        outer.addLayout(text_box, 1)

        # 状态徽章
        self._status_label = QLabel("等待", self)
        self._status_label.setFixedSize(96, 40)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._status_label, 0)

        self.set_step_state("waiting")

    def set_step_state(self, state: str, *, hint: str | None = None) -> None:
        """更新步骤卡片状态。state 取值: waiting/active/pass/ng/locked。"""
        bg, border, color, status_text, default_hint = _STEP_STATE_STYLE.get(
            state, _STEP_STATE_STYLE["waiting"]
        )
        min_height = {
            "active": 116,
            "ng": 116,
            "waiting": 98,
            "pass": 84,
            "locked": 84,
        }.get(state, 98)
        border_width = 3 if state in {"active", "ng"} else 2
        index_size = 56 if state in {"active", "ng"} else 48
        status_height = 40 if state in {"active", "ng"} else 34
        name_weight = 800 if state in {"active", "ng"} else 700
        name_color = TEXT_PRIMARY if state not in {"pass", "locked"} else color
        self.setMinimumHeight(min_height)
        self._index_label.setFixedSize(index_size, index_size)
        self._status_label.setFixedSize(96, status_height)
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: {border_width}px solid {border}; border-radius: 12px; }}"
        )
        self._index_label.setStyleSheet(
            f"background-color: {border}; color: #0B1636; border-radius: 10px; "
            f"font-size: {22 if state in {'active', 'ng'} else 18}px; font-weight: 800;"
        )
        self._name_label.setStyleSheet(text_style(name_color, size=FONT_SECTION, weight=name_weight))
        self._hint_label.setStyleSheet(text_style(color, size=FONT_SMALL, weight=700 if state == "ng" else 500))
        self._hint_label.setText(hint if hint is not None else default_hint)
        self._status_label.setStyleSheet(
            f"background-color: {border}; color: #0B1636; border-radius: 8px; "
            f"font-size: 15px; font-weight: 800;"
        )
        self._status_label.setText(status_text)


class KpiRow(QFrame):
    """单行 KPI 显示：今日产能、良率、OK、NG。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(72)
        self.setStyleSheet(frame_style(PANEL_BG_DARK, border=STROKE_MAIN, radius=10))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self._cells: dict[str, QLabel] = {}
        for key, title in [("output", "今日产能"), ("yield", "良率"), ("ok", "OK"), ("ng", "NG")]:
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
            t = QLabel(title, self)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))
            v = QLabel("0", self)
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color = TEXT_DANGER_COLOR if key == "ng" else TEXT_ACCENT
            v.setStyleSheet(text_style(color, size=FONT_METRIC, weight=800))
            cell.addWidget(t)
            cell.addWidget(v)
            wrapper = QFrame(self)
            wrapper.setLayout(cell)
            wrapper.setStyleSheet("background: transparent; border: none;")
            layout.addWidget(wrapper, 1)
            self._cells[key] = v

    def set_value(self, key: str, text: str) -> None:
        if key in self._cells:
            self._cells[key].setText(text)


# 良率/OK 用青色，NG 用红色
TEXT_DANGER_COLOR = "#FF6A75"


def main_button_style(variant: str = "primary") -> str:
    """主操作按钮科技风样式。variant: primary(相机/检测) / secondary(配置/关闭)。"""
    if variant == "primary":
        return (
            f"QPushButton {{ background-color: {BTN_PRIMARY_BG}; border: 2px solid {BTN_PRIMARY_BORDER}; "
            f"color: #FFFFFF; font-size: 16px; font-weight: 700; padding: 10px 16px; "
            f"border-radius: 10px; min-height: 44px; }}"
            f"QPushButton:hover {{ background-color: {BTN_PRIMARY_HOVER}; border: 2px solid {BTN_PRIMARY_BORDER}; }}"
            f"QPushButton:disabled {{ background-color: #123258; border: 2px solid #1B3A5C; color: #4C78A6; }}"
        )
    if variant == "danger":
        return (
            f"QPushButton {{ background-color: {BTN_DANGER_BG}; border: 2px solid {BTN_DANGER_BORDER}; "
            f"color: #FFFFFF; font-size: 16px; font-weight: 700; padding: 10px 16px; "
            f"border-radius: 10px; min-height: 44px; }}"
            f"QPushButton:hover {{ background-color: #A82838; border: 2px solid {BTN_DANGER_BORDER}; }}"
            f"QPushButton:disabled {{ background-color: #2A1018; border: 2px solid #3A1C24; color: #4C78A6; }}"
        )
    return (
        f"QPushButton {{ background-color: {BTN_SECONDARY_BG}; border: 2px solid {BTN_SECONDARY_BORDER}; "
        f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700; padding: 8px 14px; "
        f"border-radius: 10px; min-height: 40px; }}"
        f"QPushButton:hover {{ background-color: {BTN_SECONDARY_HOVER}; border: 2px solid {BTN_SECONDARY_BORDER}; }}"
        f"QPushButton:disabled {{ background-color: #0E1C3F; border: 2px solid #1B2C45; color: #4C78A6; }}"
    )
