import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ConfigManager
from validate_onnx_picture import evaluate_roi_inspection, run_validation, write_image
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


def _detection(label, x, y, class_id=0):
    return ObbDetection(
        class_id=class_id,
        label=label,
        conf=0.9,
        track_id=None,
        polygon=[(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5), (x - 5, y + 5)],
        box=(x - 5, y - 5, x + 5, y + 5),
        center=(x, y),
    )


def _config():
    config = ConfigManager()
    config.roi_inspection_enabled = True
    config.roi_template_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.category_names = ["脚垫", "", "", "", "", ""]
    config.category_counts = [4, 1, 1, 1, 1, 1]
    return config


def test_picture_roi_validation_uses_inspection_roi_and_exact_count():
    config = _config()
    config.action_pass_stable_frames = 2
    detections = [
        _detection("脚垫", 600 + index * 50, 100) for index in range(4)
    ] + [_detection("脚垫", 100, 500)]

    summary = evaluate_roi_inspection(
        detections,
        image_size=(1000, 1000),
        config=config,
        model_class_names=["脚垫"],
        frame_repetitions=2,
    )

    assert summary["ok"] is True
    assert summary["status"] == "pass"
    assert summary["categories"]["脚垫"]["observed_count"] == 4
    assert summary["events"] == [{
        "round_id": 1,
        "result": "pass",
        "observed_counts": {"脚垫": 4},
        "required_counts": {"脚垫": 4},
        "failed_categories": [],
    }]


def test_picture_roi_validation_reports_exact_count_ng():
    config = _config()
    config.action_ng_stable_frames = 2
    detections = [
        _detection("脚垫", 600 + index * 50, 100) for index in range(3)
    ]

    summary = evaluate_roi_inspection(
        detections,
        image_size=(1000, 1000),
        config=config,
        model_class_names=["脚垫"],
        frame_repetitions=2,
    )

    assert summary["status"] == "ng"
    assert summary["events"][0]["failed_categories"] == ["脚垫"]


def test_run_validation_includes_roi_summary(tmp_path):
    detections = [
        _detection("脚垫", 600 + index * 50, 100) for index in range(4)
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
            return {0: "脚垫", 1: "定位"}

    image_path = tmp_path / "脚垫.jpg"
    model_path = tmp_path / "model.onnx"
    config_path = tmp_path / "config.json"
    assert write_image(image_path, np.zeros((1000, 1000, 3), dtype=np.uint8))
    model_path.write_bytes(b"test model placeholder")
    config = _config()
    config.save(config_path)

    summary = run_validation(
        image_path=image_path,
        model_path=model_path,
        output_dir=tmp_path / "outputs",
        processor_factory=StaticProcessor,
        roi_config_path=config_path,
    )

    assert summary["ok"] is True
    assert summary["roi_inspection"]["status"] == "pass"
    assert "first_category_region" not in summary
