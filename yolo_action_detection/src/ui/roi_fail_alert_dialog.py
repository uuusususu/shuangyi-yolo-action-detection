"""ROI 检测 FAIL 时的纯图像模态告警弹窗。"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class RoiFailAlertDialog(QDialog):
    """FAIL 告警弹窗：只显示冻结画面，保持事件循环运转。

    - show() 弹出，不阻塞事件循环（检测继续）。
    - 每轮 FAIL 调 refresh() 替换冻结画面。
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
        layout.addWidget(self._image_label)
        self._source_pixmap = QPixmap()

    def refresh(self, frame: Optional[np.ndarray]) -> None:
        """用新的 NG 同帧图像替换当前画面。"""
        if frame is not None:
            self._show_frame(frame)

    def clear_and_close(self) -> None:
        """清空内容并关闭窗口（不发信号，由调用方处理后续）。"""
        self._source_pixmap = QPixmap()
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
        self._source_pixmap = QPixmap.fromImage(image)
        self._scale_pixmap()

    def _scale_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        super().resizeEvent(event)
        self._scale_pixmap()
