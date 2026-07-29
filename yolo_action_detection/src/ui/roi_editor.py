"""静态相机帧上的三类 ROI 与模板创建结果编辑器。"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from roi_inspection.geometry import NormalizedRect


class RoiKind(str, Enum):
    LOCATOR = "locator"
    INSPECTION = "inspection"
    SEARCH = "search"


_ROI_LABELS = {
    RoiKind.LOCATOR: "模板框选区域",
    RoiKind.INSPECTION: "识别 ROI",
    RoiKind.SEARCH: "搜索 ROI",
}
_ROI_COLORS = {
    RoiKind.LOCATOR: QColor("#35D7FF"),
    RoiKind.INSPECTION: QColor("#7ED957"),
    RoiKind.SEARCH: QColor("#FFB547"),
}


class RoiEditorWidget(QWidget):
    """维护模板、搜索、识别 ROI，并显示一次性模板创建结果。"""

    rect_changed = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 340)
        self.setMouseTracking(True)
        self._frame: Optional[np.ndarray] = None
        self._rects: Dict[RoiKind, NormalizedRect] = {}
        self._active_kind = RoiKind.LOCATOR
        self._drag_mode = ""
        self._drag_origin: Optional[QPointF] = None
        self._drag_start_rect: Optional[NormalizedRect] = None
        self._template_feedback: Optional[dict] = None

    @property
    def has_frame(self) -> bool:
        return self._frame is not None

    @property
    def frame_shape(self):
        return None if self._frame is None else tuple(self._frame.shape)

    @property
    def rects(self) -> Dict[RoiKind, NormalizedRect]:
        return dict(self._rects)

    @property
    def template_feedback(self) -> Optional[dict]:
        if self._template_feedback is None:
            return None
        return dict(self._template_feedback)

    def frame_copy(self) -> Optional[np.ndarray]:
        return None if self._frame is None else self._frame.copy()

    def capture_rect_frame(self, kind: RoiKind | str) -> Optional[np.ndarray]:
        """截取指定归一化矩形对应的静态帧区域。"""
        if self._frame is None:
            return None
        rect = self.roi_rect(kind)
        if rect is None:
            return None
        height, width = self._frame.shape[:2]
        x1 = min(width, max(0, int(round(rect.x * width))))
        y1 = min(height, max(0, int(round(rect.y * height))))
        x2 = min(width, max(x1, int(round((rect.x + rect.width) * width))))
        y2 = min(height, max(y1, int(round((rect.y + rect.height) * height))))
        crop = self._frame[y1:y2, x1:x2]
        return None if crop.size == 0 else crop.copy()

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self._frame = None
        else:
            self._frame = frame.copy()
        self._template_feedback = None
        self.update()

    def set_active_kind(self, kind: RoiKind | str) -> None:
        self._active_kind = RoiKind(kind)

    def roi_rect(self, kind: RoiKind | str) -> Optional[NormalizedRect]:
        return self._rects.get(RoiKind(kind))

    def set_rect(self, kind: RoiKind | str, rect: NormalizedRect) -> None:
        normalized_kind = RoiKind(kind)
        if normalized_kind in (RoiKind.LOCATOR, RoiKind.SEARCH):
            self._template_feedback = None
        self._rects[normalized_kind] = rect
        self._active_kind = normalized_kind
        self.rect_changed.emit(normalized_kind.value, rect)
        self.update()

    def delete_rect(self, kind: RoiKind | str) -> None:
        normalized_kind = RoiKind(kind)
        if normalized_kind in (RoiKind.LOCATOR, RoiKind.SEARCH):
            self._template_feedback = None
        if self._rects.pop(normalized_kind, None) is not None:
            self.rect_changed.emit(normalized_kind.value, None)
            self.update()

    def set_template_feedback(
        self,
        *,
        success: bool,
        score: float | None = None,
        box=None,
        contours=(),
        message: str = "",
    ) -> None:
        """显示本次保存回查的轮廓、外框、得分或失败原因。"""
        self._template_feedback = {
            "success": bool(success),
            "score": None if score is None else float(score),
            "box": tuple(box) if box else None,
            "contours": tuple(np.asarray(contour).copy() for contour in (contours or ())),
            "message": str(message),
        }
        self.update()

    def move_rect(self, kind: RoiKind | str, dx: float, dy: float) -> None:
        normalized_kind = RoiKind(kind)
        rect = self._rects.get(normalized_kind)
        if rect is None:
            return
        x = min(max(0.0, rect.x + float(dx)), 1.0 - rect.width)
        y = min(max(0.0, rect.y + float(dy)), 1.0 - rect.height)
        self.set_rect(
            normalized_kind,
            NormalizedRect(_stable(x), _stable(y), rect.width, rect.height),
        )

    def resize_rect(self, kind: RoiKind | str, width: float, height: float) -> None:
        normalized_kind = RoiKind(kind)
        rect = self._rects.get(normalized_kind)
        if rect is None:
            return
        new_width = min(max(0.01, float(width)), 1.0 - rect.x)
        new_height = min(max(0.01, float(height)), 1.0 - rect.y)
        self.set_rect(
            normalized_kind,
            NormalizedRect(rect.x, rect.y, _stable(new_width), _stable(new_height)),
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101820"))
        content = self._content_rect()
        if self._frame is None or content is None:
            painter.setPen(QColor("#9BA8B4"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "没有可用相机画面")
            return

        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        painter.drawImage(content, image)
        for kind, normalized in self._rects.items():
            display = self._display_rect(normalized)
            color = _ROI_COLORS[kind]
            painter.setPen(QPen(color, 3 if kind == self._active_kind else 2))
            painter.drawRect(display)
        self._draw_template_feedback(painter, content)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = self._normalized_point(event.position())
        if point is None:
            return
        existing = self._rects.get(self._active_kind)
        self._drag_origin = point
        self._drag_start_rect = existing
        if existing and _near_bottom_right(existing, point):
            self._drag_mode = "resize"
        elif existing and _contains(existing, point):
            self._drag_mode = "move"
        else:
            self._drag_mode = "create"
            start = QPointF(min(point.x(), 0.99), min(point.y(), 0.99))
            self._drag_origin = start
            self.set_rect(self._active_kind, NormalizedRect(start.x(), start.y(), 0.01, 0.01))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._drag_mode or self._drag_origin is None:
            return
        point = self._normalized_point(event.position(), clamp=True)
        if point is None:
            return
        if self._drag_mode == "move" and self._drag_start_rect is not None:
            self._rects[self._active_kind] = self._drag_start_rect
            self.move_rect(
                self._active_kind,
                point.x() - self._drag_origin.x(),
                point.y() - self._drag_origin.y(),
            )
        elif self._drag_mode == "resize" and self._drag_start_rect is not None:
            self.resize_rect(
                self._active_kind,
                point.x() - self._drag_start_rect.x,
                point.y() - self._drag_start_rect.y,
            )
        else:
            x1, y1 = self._drag_origin.x(), self._drag_origin.y()
            x2, y2 = point.x(), point.y()
            x, y = min(x1, x2), min(y1, y2)
            width, height = max(0.01, abs(x2 - x1)), max(0.01, abs(y2 - y1))
            width = min(width, 1.0 - x)
            height = min(height, 1.0 - y)
            self.set_rect(
                self._active_kind,
                NormalizedRect(_stable(x), _stable(y), _stable(width), _stable(height)),
            )

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag_mode = ""
        self._drag_origin = None
        self._drag_start_rect = None

    def _content_rect(self) -> Optional[QRectF]:
        if self._frame is None:
            return None
        frame_height, frame_width = self._frame.shape[:2]
        scale = min(self.width() / frame_width, self.height() / frame_height)
        width = frame_width * scale
        height = frame_height * scale
        return QRectF((self.width() - width) / 2.0, (self.height() - height) / 2.0, width, height)

    def _display_rect(self, rect: NormalizedRect) -> QRectF:
        content = self._content_rect()
        assert content is not None
        return QRectF(
            content.left() + rect.x * content.width(),
            content.top() + rect.y * content.height(),
            rect.width * content.width(),
            rect.height * content.height(),
        )

    def _normalized_point(self, position: QPointF, clamp: bool = False) -> Optional[QPointF]:
        content = self._content_rect()
        if content is None:
            return None
        x = (position.x() - content.left()) / content.width()
        y = (position.y() - content.top()) / content.height()
        if clamp:
            return QPointF(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            return None
        return QPointF(x, y)

    def _draw_template_feedback(self, painter: QPainter, content: QRectF) -> None:
        feedback = self._template_feedback
        if not feedback or self._frame is None:
            return
        color = QColor("#35D06F") if feedback["success"] else QColor("#FF5A67")
        painter.setPen(QPen(color, 2))
        for contour in feedback["contours"]:
            points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            if len(points) < 2:
                continue
            painter.drawPolyline(QPolygonF([self._source_to_display(point) for point in points]))
        if feedback["box"]:
            box_points = [self._source_to_display(point) for point in feedback["box"]]
            if box_points:
                box_points.append(box_points[0])
                painter.drawPolyline(QPolygonF(box_points))
        if feedback["success"] and feedback["score"] is not None:
            text = f"创建得分 {feedback['score']:.3f}"
        else:
            text = feedback["message"] or "模板创建失败"
        painter.drawText(
            QRectF(content.left() + 8, content.top() + 8, max(1.0, content.width() - 16), 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _source_to_display(self, point) -> QPointF:
        content = self._content_rect()
        assert content is not None and self._frame is not None
        frame_height, frame_width = self._frame.shape[:2]
        return QPointF(
            content.left() + float(point[0]) / frame_width * content.width(),
            content.top() + float(point[1]) / frame_height * content.height(),
        )


def _contains(rect: NormalizedRect, point: QPointF) -> bool:
    return rect.x <= point.x() <= rect.x + rect.width and rect.y <= point.y() <= rect.y + rect.height


def _near_bottom_right(rect: NormalizedRect, point: QPointF) -> bool:
    return abs(point.x() - (rect.x + rect.width)) <= 0.03 and abs(point.y() - (rect.y + rect.height)) <= 0.03


def _stable(value: float) -> float:
    return round(float(value), 6)
