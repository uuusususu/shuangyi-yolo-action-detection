"""模板匹配现场诊断埋点的回归测试。

形状匹配本身的验收在 test_shape_model.py；这里只覆盖诊断记录、抽样、
异常安全，以及处理器把形状匹配的诊断字段完整落盘。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from template_matching.match_diagnostics import TemplateMatchDiagnostics  # noqa: E402
from template_matching.shape_model import ShapeModel, ShapeModelSettings  # noqa: E402
from yolo_runtime.template_gated_processor import TemplateGatedProcessor  # noqa: E402
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection  # noqa: E402


def make_scene(width=320, height=240, cx=160, cy=120):
    image = np.full((height, width, 3), 30, np.uint8)
    cv2.rectangle(image, (cx - 40, cy - 32), (cx + 40, cy + 32), (235, 235, 235), 2)
    cv2.line(image, (cx - 32, cy - 26), (cx - 32, cy + 26), (235, 235, 235), 3)
    cv2.circle(image, (cx + 20, cy - 10), 9, (120, 120, 120), 2)
    cv2.line(image, (cx - 26, cy + 24), (cx + 14, cy - 12), (180, 180, 180), 2)
    return image


@pytest.fixture()
def scene_and_model():
    scene = make_scene()
    template = scene[84:156, 112:208].copy()
    model = ShapeModel(template, ShapeModelSettings.from_legacy(
        match_threshold=0.6, angle_range=4.0))
    need_w, need_h = model.min_search_size(30)
    rect = {"x": (160 - need_w / 2) / 320, "y": (120 - need_h / 2) / 240,
            "width": need_w / 320, "height": need_h / 240}
    return scene, model, rect


class _StubInner:
    def __init__(self):
        self.calls = 0

    def get_class_names(self):
        return {0: "散热片"}

    def process_frame(self, frame, source_frame_id=0, round_id=0):
        self.calls += 1
        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            timestamp=1.0,
            detections=[
                ObbDetection(0, "散热片", 0.9, None, [], (0, 0, 10, 10), (50.0, 50.0)),
                ObbDetection(0, "散热片", 0.8, None, [], (0, 0, 10, 10), (60.0, 60.0)),
            ],
            latency_ms=12.0,
        )


# ----------------------------------------------------------------------
# 记录器
# ----------------------------------------------------------------------
def test_recorder_is_noop_when_disabled(tmp_path):
    recorder = TemplateMatchDiagnostics()
    recorder.configure(enabled=False, log_dir=tmp_path)
    assert recorder.start_session() == ""
    assert recorder.active is False
    assert recorder.record({"score": 0.5}) is False
    assert recorder.stop_session() == {}
    assert not list(tmp_path.glob("*.ndjson"))


def test_recorder_writes_records_and_summary(tmp_path):
    recorder = TemplateMatchDiagnostics()
    recorder.configure(
        enabled=True, log_dir=tmp_path, sample_frames=0, threaded=False, shadow_enabled=False
    )
    session = recorder.start_session(context={"angle_range": 12.0})
    assert session
    for index in range(6):
        angle = 0.0 if index % 2 == 0 else 4.0
        recorder.record({
            "frame_id": index,
            "matched": index % 2 == 0,
            "score": 0.9 if index % 2 == 0 else 0.4,
            "angle": angle,
            "center": [100.0 + index, 200.0],
            "failure": "" if index % 2 == 0 else "score_below_min_score",
            "angle_scores": [[0.0, 0.9], [4.0, 0.4]],
            "skipped_angles": [12.0] if index == 5 else [],
            "total_ms": 5.0 + index,
            "search_stats": {"mean": 60.0, "std": 20.0, "edge_pixels": 1000},
            "yolo": {"ran": index % 2 == 0, "det_count": 2, "labels": {"散热片": 2}},
        })
    summary = recorder.stop_session()

    lines = [
        json.loads(line)
        for line in Path(recorder.log_path).read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 6
    assert lines[0]["schema_version"] == "template_match_log.v1"

    assert summary["frames"] == 6
    assert summary["matched_frames"] == 3
    assert summary["matched_rate"] == pytest.approx(0.5)
    assert summary["failures"]["score_below_min_score"] == 3
    assert summary["frames_with_skipped_angles"] == 1
    assert summary["best_angle_histogram"] == {"0.0": 3, "4.0": 3}
    assert summary["angle_score_curve"]["0.0"]["mean"] == pytest.approx(0.9)
    rotation = summary["rotation"]
    assert rotation["zero_angle_mean_score"] == pytest.approx(0.9)
    assert rotation["nonzero_angle_mean_score"] == pytest.approx(0.4)
    assert summary["yolo_label_counts"]["散热片"]["total"] == 6
    assert Path(recorder.summary_path).is_file()


def test_recorder_samples_frames_within_limit(tmp_path):
    recorder = TemplateMatchDiagnostics()
    recorder.configure(enabled=True, log_dir=tmp_path, sample_frames=3, threaded=False)
    recorder.start_session()
    frame = make_scene()
    for index in range(20):
        recorder.record(
            {
                "frame_id": index,
                "score": index / 20.0,
                "matched": False,
                "search_rect_px": [60, 40, 260, 200],
            },
            sample_image=frame,
        )
    summary = recorder.stop_session()
    assert summary["sampled_frames"] == 3
    saved = list((tmp_path / f"{summary['session_id']}_frames").glob("*.png"))
    assert len(saved) == 3


def test_recorder_never_raises_on_bad_payload(tmp_path):
    recorder = TemplateMatchDiagnostics()
    recorder.configure(enabled=True, log_dir=tmp_path, sample_frames=0, threaded=False)
    recorder.start_session()
    recorder.record({"score": float("nan"), "angle": None, "angle_scores": [["x", "y"]]})
    recorder.record({"score": np.float32(0.5), "center": np.array([1.0, 2.0])})
    summary = recorder.stop_session()
    assert summary["frames"] == 2
    assert summary["dropped"] == 0


# ----------------------------------------------------------------------
# 处理器接线
# ----------------------------------------------------------------------
def test_gated_processor_records_shape_diagnostics(tmp_path, monkeypatch, scene_and_model):
    scene, model, rect = scene_and_model
    recorder = TemplateMatchDiagnostics()
    recorder.configure(enabled=True, log_dir=tmp_path, sample_frames=0, threaded=False)
    monkeypatch.setattr(
        "yolo_runtime.template_gated_processor.get_recorder", lambda: recorder
    )
    recorder.start_session()

    inner = _StubInner()
    processor = TemplateGatedProcessor(inner, model, rect)
    overlay = processor.process_frame(scene, source_frame_id=7)
    summary = recorder.stop_session()

    assert overlay.template_matched is True, overlay.status
    assert inner.calls == 1
    record = json.loads(Path(recorder.log_path).read_text(encoding="utf-8").splitlines()[0])
    assert record["frame_id"] == 7
    assert record["matched"] is True
    assert record["matcher"] == "shape"
    assert record["angle_scores"]
    assert record["level_candidates"]
    assert record["levels_used"] >= 1
    # 余量诊断必须落盘：现场"随机不触发"就是靠它定位的
    assert record["center_travel_px"][0] > 0
    assert record["center_travel_px"][1] > 0
    assert record["required_roi_px"] == list(model.min_search_size(0))
    assert record["rotated_bbox_px"] == list(model.rotated_bbox(0))
    assert record["template"]["point_counts"] == list(model.point_counts)
    assert record["search_stats"]["area_px"] > 0
    assert record["yolo"]["ran"] is True
    assert record["yolo"]["labels"] == {"散热片": 2}
    assert summary["frames"] == 1


def test_gated_processor_works_without_diagnostics(scene_and_model):
    scene, _model, rect = scene_and_model
    tight = ShapeModel(
        scene[84:156, 112:208].copy(),
        ShapeModelSettings.from_legacy(match_threshold=0.999, angle_range=0.0),
    )
    inner = _StubInner()
    processor = TemplateGatedProcessor(inner, tight, rect)
    overlay = processor.process_frame(scene, source_frame_id=1)
    assert overlay.template_matched is False
    assert inner.calls == 0
    assert overlay.status.startswith("template_not_matched")


# ----------------------------------------------------------------------
# 离线分析器
# ----------------------------------------------------------------------
def test_analyzer_runs_on_generated_session(tmp_path):
    recorder = TemplateMatchDiagnostics()
    recorder.configure(enabled=True, log_dir=tmp_path, sample_frames=0, threaded=False)
    recorder.start_session(context={"angle_range": 12.0, "angle_step": 0.0})
    for index in range(10):
        tilt = index % 3 != 0
        recorder.record({
            "frame_id": index,
            "matched": not tilt,
            "score": 0.35 if tilt else 0.95,
            "threshold": 0.7,
            "angle": 4.0 if tilt else 0.0,
            "center": [500.0, 400.0],
            "failure": "score_below_min_score" if tilt else "",
            "angle_scores": [[0.0, 0.95], [4.0, 0.35], [-4.0, 0.33]],
            "total_ms": 6.0,
            "search_rect_px": [100, 100, 500, 460],
            "center_travel_px": [120, 120],
            "template": {"size": [200, 156], "angle_range": 12.0},
            "search_stats": {"mean": 55.0, "std": 18.0, "edge_pixels": 2000},
            "yolo": {"ran": not tilt, "det_count": 3, "labels": {"散热片": 3}},
        })
    recorder.stop_session()

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import analyze_template_log

    assert analyze_template_log.main([recorder.log_path]) == 0
