"""ROI 归一化坐标和显示映射。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

logger = logging.getLogger("yolo_action_detection.runtime")


Size = Tuple[int, int]
PixelRect = Tuple[float, float, float, float]


def _validated_size(size: Size, name: str) -> tuple[float, float]:
    width, height = float(size[0]), float(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name}尺寸必须大于 0")
    return width, height


@dataclass(frozen=True)
class NormalizedRect:
    """相对于原始图像宽高的矩形，所有值均在 0..1。"""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("ROI 坐标必须为数字")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI 起点不能小于 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI 必须具有正面积")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("ROI 必须完整位于 0..1 范围内")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NormalizedRect":
        if not isinstance(value, Mapping):
            raise ValueError("ROI 必须是包含 x/y/width/height 的对象")
        try:
            return cls(
                x=float(value["x"]),
                y=float(value["y"]),
                width=float(value["width"]),
                height=float(value["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ROI 必须包含有效的 x/y/width/height") from exc

    def to_mapping(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    def to_pixel_rect(self, frame_size: Size) -> PixelRect:
        frame_width, frame_height = _validated_size(frame_size, "原始帧")
        return (
            self.x * frame_width,
            self.y * frame_height,
            self.width * frame_width,
            self.height * frame_height,
        )

    def to_display_rect(self, frame_size: Size, display_size: Size) -> PixelRect:
        """映射到保持宽高比且居中的显示区域。"""
        frame_width, frame_height = _validated_size(frame_size, "原始帧")
        display_width, display_height = _validated_size(display_size, "显示区域")
        scale = min(display_width / frame_width, display_height / frame_height)
        content_width = frame_width * scale
        content_height = frame_height * scale
        offset_x = (display_width - content_width) / 2.0
        offset_y = (display_height - content_height) / 2.0
        x, y, width, height = self.to_pixel_rect(frame_size)
        return (
            offset_x + x * scale,
            offset_y + y * scale,
            width * scale,
            height * scale,
        )

    def contains_pixel_point(self, point: tuple[float, float], frame_size: Size) -> bool:
        x, y, width, height = self.to_pixel_rect(frame_size)
        point_x, point_y = float(point[0]), float(point[1])
        return x <= point_x <= x + width and y <= point_y <= y + height


def detection_center_in_rect(detection, rect: NormalizedRect, frame_size: Size) -> bool:
    """按检测框中心点判断目标是否位于 ROI 内。"""
    return rect.contains_pixel_point(detection.center, frame_size)


FULL_FRAME_RECT = NormalizedRect(0.0, 0.0, 1.0, 1.0)


def usable_normalized_rect(
    value,
    *,
    warn_label: Optional[str] = None,
) -> Optional[NormalizedRect]:
    """有效归一化矩形；None、空对象、校验失败都当空。"""
    if value is None:
        return None
    try:
        return NormalizedRect.from_mapping(value)
    except (TypeError, ValueError) as exc:
        if warn_label:
            logger.warning("%s 无效，按空处理: %s value=%s", warn_label, exc, value)
        return None
