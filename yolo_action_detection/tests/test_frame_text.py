"""相机叠加中文不得退化成问号。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication

from ui.frame_text import cjk_font_family, draw_bgr_texts


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_cjk_font_family_can_draw_chinese(qapp):
    from ui.frame_text import cjk_font_family as resolve_family

    resolve_family.cache_clear()
    family = resolve_family()
    assert family
    from PySide6.QtGui import QFont, QFontMetrics

    metrics = QFontMetrics(QFont(family, 16))
    if hasattr(metrics, "inFontUcs4"):
        assert metrics.inFontUcs4(0x70ED)
    else:
        assert metrics.boundingRect("热").width() > 0


def test_draw_bgr_texts_does_not_fall_back_to_hershey_question_marks(qapp):
    qt_frame = np.zeros((80, 260, 3), dtype=np.uint8)
    hershey = qt_frame.copy()
    draw_bgr_texts(qt_frame, [("导热垫", (12, 48), (0, 255, 128))], pixel_size=20)
    cv2.putText(
        hershey,
        "导热垫",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 128),
        1,
    )
    assert int(qt_frame.sum()) > 0
    assert not np.array_equal(qt_frame, hershey)
