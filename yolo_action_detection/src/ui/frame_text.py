"""在 OpenCV 画面上绘制中文。

``cv2.putText`` 的 Hershey 字体不含汉字，类别名会变成问号。
这里用 Windows 系统字体（微软雅黑/黑体/宋体）经 Qt 画回 BGR 图。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QImage, QPainter

from ui.runtime_ui_tokens import FONT_FAMILY

_FONT_FILES = (
    "msyh.ttc",
    "msyh.ttf",
    "msyhl.ttc",
    "msyhbd.ttc",
    "simhei.ttf",
    "simsun.ttc",
    "simsun.ttf",
)
_PREFERRED_FAMILIES = (
    FONT_FAMILY,
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "微软雅黑",
    "SimHei",
    "黑体",
    "SimSun",
    "宋体",
)


def _windows_font_dir() -> Path:
    return Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


@lru_cache(maxsize=1)
def cjk_font_family() -> str:
    """选出能画出常用汉字的字体。优先加载 Windows Fonts 里的文件。"""
    for filename in _FONT_FILES:
        path = _windows_font_dir() / filename
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return str(families[0])

    available = set(QFontDatabase.families())
    for name in _PREFERRED_FAMILIES:
        if name in available and _family_supports_cjk(name):
            return name
    for name in QFontDatabase.families():
        if _family_supports_cjk(name):
            return name
    return FONT_FAMILY


def _family_supports_cjk(family: str) -> bool:
    metrics = QFontMetrics(QFont(family, 16))
    if hasattr(metrics, "inFontUcs4"):
        return bool(metrics.inFontUcs4(0x70ED))  # 热
    return metrics.boundingRect("导热").width() > metrics.boundingRect("??").width() + 4


def draw_bgr_texts(
    frame: np.ndarray,
    items: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
    *,
    pixel_size: int = 16,
) -> np.ndarray:
    """在 BGR 画面上一次画出多条文本。坐标 y 与 ``cv2.putText`` 一样是基线。"""
    if frame is None or frame.size == 0 or not items:
        return frame
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    font = QFont(cjk_font_family())
    font.setPixelSize(max(10, int(pixel_size)))
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    painter.setFont(font)
    for text, (x, y), color in items:
        if not text:
            continue
        blue, green, red = (int(channel) for channel in color)
        painter.setPen(QColor(red, green, blue))
        painter.drawText(int(x), int(y), str(text))
    painter.end()

    bits = qimg.constBits()
    raw = np.frombuffer(bits, dtype=np.uint8, count=qimg.sizeInBytes())
    row_bytes = qimg.bytesPerLine()
    painted = raw.reshape((height, row_bytes))[:, : width * 3].reshape((height, width, 3))
    frame[:] = cv2.cvtColor(np.ascontiguousarray(painted), cv2.COLOR_RGB2BGR)
    return frame
