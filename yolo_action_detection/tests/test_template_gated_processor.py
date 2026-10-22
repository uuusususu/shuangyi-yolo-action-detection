"""形状匹配门控处理器单元测试。"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from template_matching.shape_model import (  # noqa: E402
    ShapeFailure,
    ShapeMatchResult,
    ShapeModel,
    ShapeModelSettings,
)
from yolo_runtime.template_gated_processor import TemplateGatedProcessor  # noqa: E402
from yolo_runtime.yolo_result_models import DetectionOverlayState  # noqa: E402


class LocatorStub:
    """只替代 find()，其余属性用真实模型的形状，保证诊断字段也能构造。"""

    def __init__(self, result: ShapeMatchResult, model: ShapeModel) -> None:
        self.result = result
        self.calls = []
        self.angle_curve_requests = []
        self._model = model

    @property
    def settings(self):
        return self._model.settings

    @property
    def template_size(self):
        return self._model.template_size

    @property
    def level_count(self):
        return self._model.level_count

    @property
    def point_counts(self):
        return self._model.point_counts

    @property
    def base_angle_step(self):
        return self._model.base_angle_step

    def find(self, frame, search_rect, *, collect_angle_curve=True):
        self.calls.append((frame, search_rect))
        self.angle_curve_requests.append(bool(collect_angle_curve))
        return self.result


class ProcessorSpy:
    def __init__(self) -> None:
        self.calls = []

    def get_class_names(self):
        return {0: "散热片"}

    def process_frame(self, frame, source_frame_id=0, round_id=0):
        self.calls.append((frame, source_frame_id, round_id))
        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            round_id=round_id,
            status="ok",
        )


def _shape_template():
    image = np.full((64, 80, 3), 30, np.uint8)
    cv2.rectangle(image, (6, 6), (73, 57), (235, 235, 235), 2)
    cv2.line(image, (14, 12), (14, 52), (235, 235, 235), 3)
    cv2.circle(image, (56, 26), 9, (120, 120, 120), 2)
    cv2.line(image, (20, 50), (60, 14), (180, 180, 180), 2)
    return image


def _real_model(angle_range=0.0):
    return ShapeModel(_shape_template(), ShapeModelSettings.from_legacy(
        match_threshold=0.6, angle_range=angle_range))


def _roi_for(model, cx, cy, travel, width, height):
    need_w, need_h = model.min_search_size(travel)
    return {"x": (cx - need_w / 2) / width, "y": (cy - need_h / 2) / height,
            "width": need_w / width, "height": need_h / height}


def test_gate_skips_yolo_when_shape_does_not_match():
    model = _real_model()
    locator = LocatorStub(
        ShapeMatchResult(matched=False, score=0.25,
                         failure=ShapeFailure.SCORE_BELOW_MIN), model)
    inner = ProcessorSpy()
    search_rect = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.6}
    processor = TemplateGatedProcessor(inner, locator, search_rect)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    overlay = processor.process_frame(frame, source_frame_id=7, round_id=3)

    assert locator.calls == [(frame, search_rect)]
    assert inner.calls == []
    assert overlay.source_frame_id == 7
    assert overlay.detections == []
    assert overlay.template_matched is False
    assert overlay.template_score == 0.25
    # 失败原因要带进 status，现场才能区分"没到位"和"过曝没轮廓"
    assert overlay.status == "template_not_matched:score_below_min_score"


def test_gate_runs_yolo_and_attaches_pose():
    model = _real_model()
    locator = LocatorStub(
        ShapeMatchResult(matched=True, score=0.94, center=(120.0, 80.0), angle=4.5),
        model)
    inner = ProcessorSpy()
    search_rect = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.6}
    processor = TemplateGatedProcessor(inner, locator, search_rect)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    overlay = processor.process_frame(frame, source_frame_id=8, round_id=4)

    assert len(inner.calls) == 1
    assert inner.calls[0][0] is frame
    assert inner.calls[0][1:] == (8, 4)
    assert overlay.template_matched is True
    assert overlay.template_score == 0.94
    assert overlay.template_center == (120.0, 80.0)
    assert overlay.template_angle == 4.5
    assert processor.get_class_names() == {0: "散热片"}
    assert processor.template_model is locator
    assert processor.search_rect is search_rect


class _RecorderStub:
    def __init__(self, active: bool) -> None:
        self.active = active
        self.records = []

    def record(self, payload, sample_image=None):
        self.records.append(payload)


def test_angle_curve_is_requested_only_when_diagnostics_active(monkeypatch):
    """角度曲线是纯诊断输出，诊断没开就不该为它付采样代价。"""
    model = _real_model()
    locator = LocatorStub(
        ShapeMatchResult(matched=True, score=0.9, center=(30.0, 20.0), angle=0.0),
        model)
    inner = ProcessorSpy()
    processor = TemplateGatedProcessor(
        inner, locator, {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.6})
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    idle = _RecorderStub(active=False)
    monkeypatch.setattr(
        "yolo_runtime.template_gated_processor.get_recorder", lambda: idle)
    processor.process_frame(frame, source_frame_id=1)

    assert locator.angle_curve_requests == [False]
    assert idle.records == []

    live = _RecorderStub(active=True)
    monkeypatch.setattr(
        "yolo_runtime.template_gated_processor.get_recorder", lambda: live)
    processor.process_frame(frame, source_frame_id=2)

    assert locator.angle_curve_requests == [False, True]
    assert len(live.records) == 1


def test_real_shape_gate_runs_yolo_on_full_frame():
    model = _real_model()
    template = _shape_template()
    frame = np.full((240, 320, 3), 30, np.uint8)
    frame[88:152, 120:200] = template
    inner = ProcessorSpy()
    processor = TemplateGatedProcessor(
        inner, model, _roi_for(model, 160, 120, 40, 320, 240))

    overlay = processor.process_frame(frame, source_frame_id=11, round_id=6)

    assert overlay.template_matched is True, overlay.status
    assert overlay.template_score > 0.8
    assert len(inner.calls) == 1
    assert inner.calls[0][0] is frame


def test_real_shape_gate_skips_yolo_when_target_outside_search_roi():
    model = _real_model()
    template = _shape_template()
    frame = np.full((240, 320, 3), 30, np.uint8)
    frame[160:224, 220:300] = template          # 目标在右下角
    inner = ProcessorSpy()
    processor = TemplateGatedProcessor(
        inner, model, _roi_for(model, 90, 70, 40, 320, 240))   # 搜索框在左上角

    overlay = processor.process_frame(frame, source_frame_id=12, round_id=6)

    assert overlay.template_matched is False
    assert inner.calls == []


def test_real_shape_gate_tolerates_rotated_target():
    """同一产品小幅旋转仍要命中——这是旧实现失效的核心场景。"""
    model = _real_model(angle_range=10.0)
    template = _shape_template()
    canvas = np.full((240, 320, 3), 30, np.uint8)
    canvas[88:152, 120:200] = template
    matrix = cv2.getRotationMatrix2D((160.0, 120.0), 6.0, 1.0)
    frame = cv2.warpAffine(canvas, matrix, (320, 240), flags=cv2.INTER_LINEAR)
    inner = ProcessorSpy()
    processor = TemplateGatedProcessor(
        inner, model, _roi_for(model, 160, 120, 40, 320, 240))

    overlay = processor.process_frame(frame, source_frame_id=13, round_id=7)

    assert overlay.template_matched is True, overlay.status
    assert overlay.template_score > 0.75
    assert len(inner.calls) == 1
