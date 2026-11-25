"""Native Qt panel widgets used by runtime pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.runtime_ui_tokens import (
    BANNER_ACTIVE_BG,
    BANNER_ACTIVE_BORDER,
    BANNER_ACTIVE_SUB,
    BANNER_ACTIVE_TITLE,
    BANNER_IDLE_BG,
    BANNER_IDLE_BORDER,
    BANNER_IDLE_SUB,
    BANNER_IDLE_TITLE,
    BANNER_NG_BG,
    BANNER_NG_BORDER,
    BANNER_NG_SUB,
    BANNER_NG_TITLE,
    BANNER_PASS_BG,
    BANNER_PASS_BORDER,
    BANNER_PASS_SUB,
    BANNER_PASS_TITLE,
    BANNER_WARN_BG,
    BANNER_WARN_BORDER,
    BANNER_WARN_SUB,
    BANNER_WARN_TITLE,
    FONT_BODY,
    FONT_METRIC,
    FONT_MICRO,
    FONT_SECTION,
    FONT_SMALL,
    MONO_FAMILY,
    PAGE_BG,
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
    STEP_PROGRESS_BG,
    STEP_PROGRESS_BORDER,
    STEP_PROGRESS_TEXT,
    STEP_WAITING_BG,
    STEP_WAITING_BORDER,
    STEP_WAITING_TEXT,
    STROKE_MAIN,
    STROKE_SOFT,
    STROKE_SUBTLE,
    TEXT_ACCENT,
    TEXT_DANGER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_SUCCESS,
    TEXT_WARNING,
    frame_style,
    text_style,
)


class StatusBadge(QLabel):
    """Read-only status badge."""

    def __init__(self, text: str = "待机", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(84)
        self.setFixedHeight(28)
        self.set_status(text)

    def set_status(self, text: str, *, active: bool = False) -> None:
        text = str(text)
        upper = text.upper()
        if active or upper == "PASS" or text == "合格":
            background = "#1C8E4F"
            border = "#52E896"
            color = "#E8FFF0"
        elif upper == "TESTING" or text in ("确认中", "检测中"):
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

        self.result_badge = StatusBadge("待机", self)

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

        self.status_badge = StatusBadge("待机", self)
        self.status_badge.setMinimumWidth(0)

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.status_badge)


class ConfigNavButton(QPushButton):
    """配置页左侧分区导航按钮。"""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setProperty("navigation", True)


class FormGroup(QFrame):
    """带标题的配置表单分组卡片。"""

    def __init__(self, title: str, parent=None, number: str = "", subtitle: str = ""):
        super().__init__(parent)
        self.setObjectName("formGroup")
        self.setStyleSheet(frame_style(PANEL_BG, border=STROKE_SUBTLE, radius=10))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)

        head_wrap = QVBoxLayout()
        head_wrap.setContentsMargins(0, 0, 0, 0)
        head_wrap.setSpacing(3)

        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_SECTION, weight=700))
        self.number_label: QLabel | None = None
        if number:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(8)
            self.number_label = QLabel(number, self)
            self.number_label.setStyleSheet(
                text_style(
                    TEXT_ACCENT,
                    size=FONT_MICRO,
                    weight=800,
                    extra=f"font-family: '{MONO_FAMILY}';",
                )
            )
            head.addWidget(self.number_label, 0, Qt.AlignmentFlag.AlignVCenter)
            head.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignVCenter)
            head.addStretch(1)
            head_wrap.addLayout(head)
        else:
            head_wrap.addWidget(self.title_label)

        self.subtitle_label = QLabel(subtitle, self)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(text_style(TEXT_MUTED, size=FONT_SMALL, weight=500))
        self.subtitle_label.setVisible(bool(subtitle))
        head_wrap.addWidget(self.subtitle_label)
        outer.addLayout(head_wrap)

        divider = QFrame(self)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"QFrame {{ background-color: {STROKE_SUBTLE}; border: none; }}")
        outer.addWidget(divider)

        self._content = QWidget(self)
        self._content.setObjectName("formGroupContent")
        self._content.setStyleSheet("QWidget#formGroupContent { background: transparent; border: none; }")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(12)
        outer.addWidget(self._content)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


class CollapsibleSection(QFrame):
    """默认折叠的高级/诊断区块，使用原生 QPushButton 作为折叠开关。"""

    def __init__(self, title: str, parent=None, *, expanded: bool = False):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self.setStyleSheet(frame_style(PANEL_BG_DARK, border=STROKE_SUBTLE, radius=10))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._base_title = str(title)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        self.toggle_button = QPushButton(self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setMinimumHeight(40)
        self.toggle_button.setProperty("collapsible", True)
        self.toggle_button.toggled.connect(self._sync_expanded)
        outer.addWidget(self.toggle_button)

        self._body = QWidget(self)
        self._body.setObjectName("collapsibleBody")
        self._body.setStyleSheet("QWidget#collapsibleBody { background: transparent; border: none; }")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(6, 4, 6, 4)
        self._body_layout.setSpacing(10)
        outer.addWidget(self._body)

        self.set_expanded(expanded)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._body_layout

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(bool(expanded))
        self._sync_expanded(bool(expanded))

    def _sync_expanded(self, expanded: bool) -> None:
        prefix = "[-] " if expanded else "[+] "
        self.toggle_button.setText(f"{prefix}{self._base_title}")
        self._body.setVisible(expanded)


class FormField(QWidget):
    """可见标签、输入控件和可选帮助文字组成的字段。"""

    def __init__(self, title: str, control: QWidget, helper: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("formField")
        self.setStyleSheet("QWidget#formField { background: transparent; border: none; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.label = QLabel(title, self)
        self.label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))
        layout.addWidget(self.label)
        control.setParent(self)
        layout.addWidget(control)
        self.control = control
        self.helper_label = QLabel(helper, self)
        self.helper_label.setWordWrap(True)
        self.helper_label.setStyleSheet(text_style(TEXT_MUTED, size=11, weight=500))
        self.helper_label.setVisible(bool(helper))
        layout.addWidget(self.helper_label)


class SwitchRow(QFrame):
    """标题、帮助文字和原生复选框组成的设置行。"""

    def __init__(self, title: str, helper: str, parent=None):
        super().__init__(parent)
        self.setObjectName("switchRow")
        self.setStyleSheet(
            f"QFrame#switchRow {{ background: transparent; border: none; "
            f"border-bottom: 1px solid {STROKE_SUBTLE}; border-radius: 6px; }}"
            "QFrame#switchRow:hover { background-color: rgba(25, 52, 95, 90); }"
        )
        self.setMinimumHeight(58)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 12, 8)
        layout.setSpacing(16)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_BODY, weight=700))
        self.helper_label = QLabel(helper, self)
        self.helper_label.setWordWrap(True)
        self.helper_label.setStyleSheet(text_style(TEXT_MUTED, size=FONT_SMALL, weight=500))
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.helper_label)
        layout.addLayout(text_layout, 1)
        self.checkbox = QCheckBox(self)
        self.checkbox.setAccessibleName(title)
        layout.addWidget(self.checkbox, 0, Qt.AlignmentFlag.AlignVCenter)


class FooterActionBar(QFrame):
    """配置页固定底部操作栏，含未保存变更提示与保存反馈。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("footerActionBar")
        self.setStyleSheet(
            f"QFrame#footerActionBar {{ background-color: {PANEL_BG_DARK}; border: none; "
            f"border-top: 1px solid {STROKE_SUBTLE}; }}"
        )
        self.setFixedHeight(66)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 10, 24, 10)
        layout.setSpacing(12)

        self.dirty_label = QLabel("● 有未保存的修改", self)
        self.dirty_label.setStyleSheet(text_style(TEXT_WARNING, size=FONT_SMALL, weight=700))
        self.dirty_label.setVisible(False)

        self.saved_label = QLabel("✓ 配置已保存并生效", self)
        self.saved_label.setStyleSheet(text_style(STEP_PASS_TEXT, size=FONT_SMALL, weight=700))
        self.saved_label.setVisible(False)

        self.cancel_button = QPushButton("返回主屏", self)
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setMinimumWidth(104)

        self.save_button = QPushButton("保存并生效", self)
        self.save_button.setMinimumHeight(40)
        self.save_button.setMinimumWidth(120)
        self.save_button.setProperty("buttonRole", "primary")
        self.save_button.setDefault(True)

        layout.addWidget(self.dirty_label)
        layout.addWidget(self.saved_label)
        layout.addStretch(1)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.save_button)

    def set_dirty(self, dirty: bool) -> None:
        self.dirty_label.setVisible(bool(dirty))
        if dirty:
            self.saved_label.setVisible(False)


# ---------------------------------------------------------------------------
# Hero 判定状态横幅（主检测界面核心视觉焦点）
# ---------------------------------------------------------------------------

_BANNER_PALETTE = {
    "idle": (BANNER_IDLE_BG, BANNER_IDLE_BORDER, BANNER_IDLE_TITLE, BANNER_IDLE_SUB, "当前状态"),
    "active": (BANNER_ACTIVE_BG, BANNER_ACTIVE_BORDER, BANNER_ACTIVE_TITLE, BANNER_ACTIVE_SUB, "实时检测"),
    "pass": (BANNER_PASS_BG, BANNER_PASS_BORDER, BANNER_PASS_TITLE, BANNER_PASS_SUB, "判定通过"),
    "ng": (BANNER_NG_BG, BANNER_NG_BORDER, BANNER_NG_TITLE, BANNER_NG_SUB, "判定异常"),
    "warn": (BANNER_WARN_BG, BANNER_WARN_BORDER, BANNER_WARN_TITLE, BANNER_WARN_SUB, "操作提示"),
}


class ResultBannerWidget(QFrame):
    """主界面右侧顶部高可视度判定状态横幅（待机 / 检测中 / PASS / FAIL / 警告）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kind = "idle"
        self.setObjectName("resultBannerWidget")
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        self._eyebrow_label = QLabel("当前状态", self)
        self._meta_label = QLabel("", self)
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(self._eyebrow_label)
        top_row.addStretch(1)
        top_row.addWidget(self._meta_label)
        layout.addLayout(top_row)

        self._title_label = QLabel("待机中", self)
        self._title_label.setWordWrap(False)
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("打开相机后点击「开始检测」进入作业防错", self)
        self._subtitle_label.setWordWrap(True)
        layout.addWidget(self._subtitle_label)

        self.set_banner_state("idle", "待机中", "打开相机后点击「开始检测」进入作业防错")

    def set_banner_state(
        self,
        kind: str,
        title: str,
        subtitle: str = "",
        meta: str = "",
        *,
        eyebrow: str | None = None,
    ) -> None:
        normalized = kind if kind in _BANNER_PALETTE else "idle"
        self._kind = normalized
        bg, border, title_color, sub_color, default_eyebrow = _BANNER_PALETTE[normalized]
        self.setStyleSheet(
            f"QFrame#resultBannerWidget {{ background-color: {bg}; border: 1px solid {border}; "
            "border-radius: 10px; }"
        )
        self._eyebrow_label.setText(default_eyebrow if eyebrow is None else eyebrow)
        self._eyebrow_label.setStyleSheet(text_style(sub_color, size=FONT_MICRO, weight=700))

        self._meta_label.setText(meta)
        self._meta_label.setVisible(bool(meta))
        self._meta_label.setStyleSheet(
            text_style(title_color, size=FONT_SMALL, weight=700, extra=f"font-family: '{MONO_FAMILY}';")
        )

        title_size = 28 if normalized in ("pass", "ng") else 20
        self._title_label.setText(title)
        self._title_label.setStyleSheet(text_style(title_color, size=title_size, weight=800))

        self._subtitle_label.setText(subtitle)
        self._subtitle_label.setVisible(bool(subtitle))
        self._subtitle_label.setStyleSheet(text_style(sub_color, size=FONT_SMALL, weight=600))

    def kind_name(self) -> str:
        return self._kind

    def title_text(self) -> str:
        return self._title_label.text()

    def subtitle_text(self) -> str:
        return self._subtitle_label.text()

    def meta_text(self) -> str:
        return self._meta_label.text()


# ---------------------------------------------------------------------------
# 步骤检测卡片（运行时主界面核心控件）
# ---------------------------------------------------------------------------

# 步骤卡片状态映射到 (背景, 描边, 文字, 状态文案, 辅助提示)
_STEP_STATE_STYLE = {
    "waiting": (STEP_WAITING_BG, STEP_WAITING_BORDER, STEP_WAITING_TEXT, "待执行", "等待前置步骤完成"),
    "active": (STEP_ACTIVE_BG, STEP_ACTIVE_BORDER, STEP_ACTIVE_TEXT, "请操作", "请执行当前步骤"),
    "progress": (STEP_PROGRESS_BG, STEP_PROGRESS_BORDER, STEP_PROGRESS_TEXT, "确认中", "动作持续确认中"),
    "pass": (STEP_PASS_BG, STEP_PASS_BORDER, STEP_PASS_TEXT, "合格", "已完成"),
    "ng": (STEP_NG_BG, STEP_NG_BORDER, STEP_NG_TEXT, "异常", "请检查当前步骤"),
    "locked": (STEP_LOCKED_BG, STEP_LOCKED_BORDER, STEP_LOCKED_TEXT, "已锁定", "等待纠正前置步骤"),
}


class RecognitionListItem(QFrame):
    """紧凑识别进度项，显示顺序、类别、数量与统一中文语义状态。"""

    def __init__(self, index: int, name: str, required_count: int = 1, parent=None):
        super().__init__(parent)
        self._index = index
        self._state = "waiting"
        self._required_count = max(1, int(required_count))
        self.setObjectName("recognitionListItem")
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 12, 8)
        layout.setSpacing(12)

        self._index_label = QLabel(f"{index:02d}", self)
        self._index_label.setFixedSize(34, 34)
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._index_label)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        self._name_label = QLabel(name, self)
        self._name_label.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_BODY, weight=700))
        self._status_label = QLabel("待执行", self)
        self._status_label.setStyleSheet(text_style(TEXT_MUTED, size=FONT_SMALL, weight=600))
        text_layout.addWidget(self._name_label)
        text_layout.addWidget(self._status_label)
        layout.addLayout(text_layout, 1)

        self._quantity_label = QLabel(f"0 / {self._required_count}", self)
        self._quantity_label.setMinimumWidth(68)
        self._quantity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._quantity_label)
        self.set_step_state("waiting")

    def set_step_state(self, state: str, *, hint: str | None = None) -> None:
        bg, border, color, status_text, _default_hint = _STEP_STATE_STYLE.get(
            state, _STEP_STATE_STYLE["waiting"]
        )
        self._state = state if state in _STEP_STATE_STYLE else "waiting"
        border_width = 2 if self._state in ("active", "ng") else 1
        self.setStyleSheet(
            f"QFrame#recognitionListItem {{ background-color: {bg}; "
            f"border: {border_width}px solid {border}; border-radius: 8px; }}"
        )
        self._index_label.setStyleSheet(
            f"background-color: {border}; color: {PAGE_BG}; border: none; border-radius: 17px; "
            "font-size: 12px; font-weight: 800;"
        )
        self._name_label.setStyleSheet(
            text_style(TEXT_PRIMARY if self._state != "locked" else color, size=FONT_BODY, weight=700)
        )
        self._status_label.setText(hint if hint is not None else status_text)
        self._status_label.setStyleSheet(text_style(color, size=FONT_SMALL, weight=700))
        self._quantity_label.setStyleSheet(
            f"background-color: {PANEL_BG_DARK}; border: 1px solid {border}; border-radius: 6px; "
            f"color: {color}; font-size: {FONT_SMALL}px; font-weight: 800; padding: 4px 8px; "
            f"font-family: '{MONO_FAMILY}';"
        )

    def set_quantity_progress(self, current: int, required: int) -> None:
        self._required_count = max(1, int(required))
        self._quantity_label.setText(f"{max(0, int(current))} / {self._required_count}")

    def quantity_text(self) -> str:
        return self._quantity_label.text()

    def status_text(self) -> str:
        return self._status_label.text()

    def state_name(self) -> str:
        return self._state


_KPI_SPEC = (
    ("output", "总计", TEXT_PRIMARY),
    ("yield", "良率", TEXT_ACCENT),
    ("ok", "OK", TEXT_SUCCESS),
    ("ng", "NG", TEXT_DANGER),
)


class KpiRow(QFrame):
    """四色语义 KPI 卡片：总计、良率、OK、NG。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("kpiRow")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(74)
        self.setStyleSheet(frame_style(PANEL_BG_DARK, border=STROKE_SUBTLE, radius=10))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self._cells: dict[str, QLabel] = {}
        for idx, (key, title, color) in enumerate(_KPI_SPEC):
            if idx > 0:
                sep = QFrame(self)
                sep.setFixedWidth(1)
                sep.setStyleSheet(f"QFrame {{ background-color: {STROKE_SUBTLE}; border: none; }}")
                layout.addWidget(sep)
            cell = QVBoxLayout()
            cell.setContentsMargins(6, 0, 6, 0)
            cell.setSpacing(2)
            t = QLabel(title, self)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(text_style(TEXT_MUTED, size=FONT_MICRO, weight=700))
            v = QLabel("0.0%" if key == "yield" else "0", self)
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setStyleSheet(
                text_style(color, size=FONT_METRIC, weight=800, extra=f"font-family: '{MONO_FAMILY}';")
            )
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


TEXT_DANGER_COLOR = TEXT_DANGER
