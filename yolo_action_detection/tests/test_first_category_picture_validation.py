import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ConfigManager
from validate_onnx_picture import evaluate_first_category_region, run_validation, write_image
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


def _detection(label, x, y, width, height, class_id):
    return ObbDetection(
        class_id=class_id,
        label=label,
        conf=0.9,
        track_id=None,
        polygon=[
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ],
        box=(x, y, x + width, y + height),
        center=(x + width / 2, y + height / 2),
    )


def test_picture_region_validation_checks_parent_before_inside_children():
    config = ConfigManager()
    config.first_category_region_check_enabled = True
    config.category_names = ["pcb", "脚垫", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.action_pass_stable_frames = 2
    detections = [_detection("pcb", 100, 100, 300, 300, 1)] + [
        _detection("脚垫", 140 + index * 50, 160, 20, 20, 0)
        for index in range(4)
    ] + [
        _detection("脚垫", 700, 700, 20, 20, 0)
    ]

    summary = evaluate_first_category_region(
        detections,
        image_size=(1000, 1000),
        config=config,
        frame_repetitions=2,
    )

    assert summary["ok"] is True
    assert summary["status"] == "pass"
    assert summary["parent_class"] == "pcb"
    assert summary["parent_detection_count"] == 1
    assert summary["parents"][0]["track_id"] < 0
    assert summary["parents"][0]["observed_counts"] == {"脚垫": 4}
    assert summary["parents"][0]["required_counts"] == {"脚垫": 4}


def test_picture_region_validation_counts_zero_children_as_ng_after_stability():
    config = ConfigManager()
    config.first_category_region_check_enabled = True
    config.category_names = ["产品主体", "脚垫", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.action_ng_stable_frames = 3

    summary = evaluate_first_category_region(
        [_detection("产品主体", 100, 100, 300, 300, 1)],
        image_size=(1000, 1000),
        config=config,
        frame_repetitions=3,
    )

    assert summary["ok"] is False
    assert summary["status"] == "ng"
    assert summary["parent_detection_count"] == 1
    assert summary["parents"][0]["observed_counts"] == {"脚垫": 0}
    assert summary["parents"][0]["slot_statuses"] == {"脚垫": "ng_latched"}
    assert summary["events"][0]["result"] == "fail"


def test_picture_region_validation_reports_two_parent_pass_events_when_interval_is_zero():
    config = ConfigManager()
    config.first_category_region_check_enabled = True
    config.category_names = ["pcb", "脚垫", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.round_cooldown_seconds = 0.0
    detections = (
        [_detection("pcb", 100, 100, 250, 250, 1)]
        + [_detection("脚垫", 130 + index * 40, 140, 20, 20, 0) for index in range(4)]
        + [_detection("pcb", 500, 100, 250, 250, 1)]
        + [_detection("脚垫", 530 + index * 40, 140, 20, 20, 0) for index in range(4)]
    )

    summary = evaluate_first_category_region(
        detections,
        image_size=(1000, 1000),
        config=config,
        frame_repetitions=1,
    )

    assert summary["ok"] is True
    assert summary["status"] == "pass"
    assert summary["parent_detection_count"] == 2
    assert [event["result"] for event in summary["events"]] == ["pass", "pass"]
    assert len({event["track_id"] for event in summary["events"]}) == 2


def test_run_validation_includes_first_category_region_summary(tmp_path):
    detections = [_detection("pcb", 100, 100, 300, 300, 1)] + [
        _detection("脚垫", 140 + index * 50, 160, 20, 20, 0)
        for index in range(4)
    ]

    class StaticProcessor:
        def __init__(self, **_kwargs):
            pass

        def load(self):
            pass

        def process_frame(self, _frame, source_frame_id=0, round_id=0):
            return DetectionOverlayState(
                source_frame_id=source_frame_id,
                round_id=round_id,
                detections=detections,
                task_type="onnx_obb",
            )

        def get_class_names(self):
            return {0: "脚垫", 1: "pcb"}

    image_path = tmp_path / "脚垫.jpg"
    model_path = tmp_path / "model.onnx"
    config_path = tmp_path / "config.json"
    assert write_image(image_path, np.zeros((1000, 1000, 3), dtype=np.uint8))
    model_path.write_bytes(b"test model placeholder")
    config = ConfigManager()
    config.first_category_region_check_enabled = True
    config.category_names = ["pcb", "脚垫", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.action_pass_stable_frames = 2
    config.save(config_path)

    summary = run_validation(
        image_path=image_path,
        model_path=model_path,
        output_dir=tmp_path / "outputs",
        processor_factory=StaticProcessor,
        region_config_path=config_path,
        region_frame_repetitions=2,
    )

    assert summary["ok"] is True
    assert summary["first_category_region"]["status"] == "pass"
    assert summary["first_category_region"]["parents"][0]["observed_counts"] == {"脚垫": 4}


def test_picture_region_validation_reports_missing_parent_class():
    config = ConfigManager()
    config.first_category_region_check_enabled = True
    config.category_names = ["pcb", "脚垫", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]

    summary = evaluate_first_category_region(
        [_detection("脚垫", 140, 160, 20, 20, 0)],
        image_size=(1000, 1000),
        config=config,
        frame_repetitions=1,
    )

    assert summary["ok"] is False
    assert summary["status"] == "no_parent"
    assert summary["error"] == "未识别到父类: pcb"
