"""ROI 检测 FAIL 时的模态告警弹窗：显示 FAIL 帧 + 多行文字摘要，回车关闭。"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class RoiFailAlertDialog(QDialog):
    """FAIL 告警弹窗：画面 + 失败摘要，模态但保持事件循环运转。

    - show() 弹出，不阻塞事件循环（检测继续）。
    - 每轮 FAIL 调 refresh() 刷新画面、追加一行摘要。
    - 回车键关闭并触发 alert_closed 信号，由调用方停止循环声音。
    """

    alert_closed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI 检测异常")
        self.setModal(True)
        self.setMinimumSize(640, 520)

        layout = QVBoxLayout(self)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(320)
        self._image_label.setStyleSheet("background-color: #101820;")
        layout.addWidget(self._image_label, stretch=3)

        self._summary_scroll = QScrollArea()
        self._summary_scroll.setWidgetResizable(True)
        self._summary_inner = QLabel()
        self._summary_inner.setWordWrap(True)
        self._summary_inner.setStyleSheet(
            "color: #F44336; font-size: 15px; font-weight: bold; padding: 8px;"
        )
        self._summary_scroll.setWidget(self._summary_inner)
        layout.addWidget(self._summary_scroll, stretch=1)

        self._hint_label = QLabel("按下回车键确认并关闭")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet("color: #9BA8B4; font-size: 13px; padding: 4px;")
        layout.addWidget(self._hint_label)

        self._lines: list[str] = []

    def refresh(self, frame: Optional[np.ndarray], summary: str) -> None:
        """刷新画面并追加一行摘要。frame 可为 None（画面保持上次）。"""
        if frame is not None:
            self._show_frame(frame)
        self._lines.append(summary)
        self._summary_inner.setText("\n".join(self._lines))
        # 滚动到底部，显示最新一行。
        bar = self._summary_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_and_close(self) -> None:
        """清空内容并关闭窗口（不发信号，由调用方处理后续）。"""
        self._lines.clear()
        self._summary_inner.setText("")
        self._image_label.clear()
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clear_and_close()
            self.alert_closed.emit()
            return
        super().keyPressEvent(event)

    def _show_frame(self, frame: np.ndarray) -> None:
        """显示 BGR 帧为 QPixmap，保持宽高比。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
