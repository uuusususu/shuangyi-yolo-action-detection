"""ROI 定位检测主窗口集成测试。"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

from app_state import AppState
from config import ConfigManager
from detection_logging.fail_evidence import FailEvidenceContext, FailEvidenceSaver
from roi_inspection.models import InspectionOutcome, RoundPhase
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow


class DummyProcessor:
    def get_class_names(self):
        return {0: "A", 1: "B"}


class SoundSpy:
    def __init__(self):
        self.enabled = True
        self.pass_calls = 0
        self.fail_calls = 0
        self.fail_loop_calls = 0
        self.stop_loop_calls = 0

    def play_pass(self):
        self.pass_calls += 1

    def play_fail(self):
        self.fail_calls += 1

    def play_fail_loop(self):
        self.fail_loop_calls += 1

    def stop_loop(self):
        self.stop_loop_calls += 1


class EvidenceSpy:
    def __init__(self, result=Path("evidence")):
        self.enabled = True
        self.calls = []
        self.result = result

    def save_fail_event(self, frame, context):
        self.calls.append((frame, context))
        return self.result


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _roi_config(tmp_path: Path) -> ConfigManager:
    """无模板定位模式：识别 ROI 内任意类别触发。"""
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.roi_inspection_enabled = True
    config.roi_template_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    config.action_pass_stable_frames = 1
    config.action_ng_stable_frames = 2
    config.round_cooldown_seconds = 60.0
    config.pass_sound_enabled = True
    config.fail_sound_enabled = True
    config.fail_evidence_enabled = True
    return config


def _enable_template(config: ConfigManager) -> ConfigManager:
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.15, "y": 0.15, "width": 0.1, "height": 0.1}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    return config


def _window(qapp, config, *, sound=None, evidence=None):
    return MainWindow(
        config,
        AppState(),
        processor=DummyProcessor(),
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.get_runtime_category_counts(),
        ),
        sound_feedback=sound,
        fail_evidence_saver=evidence,
        processor_factory=lambda *args, **kwargs: DummyProcessor(),
    )


def _det(label, x, y):
    return {
        "class_id": 0,
        "label": label,
        "conf": 0.9,
        "track_id": None,
        "polygon": [[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]],
        "box": [x - 5, y - 5, x + 5, y + 5],
        "center": [x, y],
    }


def _payload(
    detections,
    frame_id=1,
    *,
    template_matched=None,
    template_score=None,
    source_frame=None,
):
    payload = {
        "source_frame_id": frame_id,
        "timestamp": time.monotonic(),
        "model_path": "model.onnx",
        "task_type": "ultralytics_obb",
        "detections": detections,
        "image_size": (1000, 1000),
        "source_frame": (
            source_frame
            if source_frame is not None
            else np.full((1000, 1000, 3), frame_id % 255, dtype=np.uint8)
        ),
    }
    if template_matched is not None:
        payload["template_matched"] = template_matched
        payload["template_score"] = template_score or 0.0
    return payload


def _fail_result(frame_id=42, round_id=1):
    return SimpleNamespace(
        round_id=round_id,
        failed_categories=["B"],
        source_frame_id=frame_id,
        timestamp=time.time(),
    )


def test_roi_mode_builds_only_roi_engine_and_normal_mode_keeps_step_engine(qapp, tmp_path):
    roi_window = _window(qapp, _roi_config(tmp_path))
    assert roi_window._roi_engine is not None

    normal_config = ConfigManager()
    normal_config.category_names = ["A", "", "", "", "", ""]
    normal = _window(qapp, normal_config)
    assert normal._roi_engine is None
    assert normal.step_engine is not None


def test_roi_pass_updates_cards_sound_and_stats_once(qapp, tmp_path):
    sound = SoundSpy()
    window = _window(qapp, _roi_config(tmp_path), sound=sound)
    passing = [_det("A", 600, 100), _det("B", 700, 100)]

    window._on_overlay_updated(_payload(passing, 1))
    window._on_overlay_updated(_payload(passing, 2))

    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ok == 1
    assert sound.pass_calls == 1
    assert window._result_label.text() == "PASS"
    assert [card.quantity_text() for card in window._step_cards] == ["1 / 1", "1 / 1"]
    assert all(card.state_name() == "pass" for card in window._step_cards)


def test_roi_runtime_uses_display_frame_size_for_legacy_payload_without_image_size(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = _window(qapp, _roi_config(tmp_path))
    passing = [_det("A", 600, 100), _det("B", 700, 100)]
    payload = _payload(passing, 1)
    payload.pop("image_size")
    monkeypatch.setattr(
        window.worker,
        "get_display_frame",
        lambda: (np.zeros((1000, 1000, 3), dtype=np.uint8), 1),
    )

    window._on_overlay_updated(payload)

    assert window._stats_manager.batch.ok == 1


def test_roi_fail_updates_stats_sound_and_evidence_once(qapp, tmp_path, monkeypatch):
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    latest_frame = np.full((1000, 1000, 3), 199, dtype=np.uint8)
    monkeypatch.setattr(window.worker, "get_display_frame", lambda: (latest_frame.copy(), 99))
    monkeypatch.setattr(window.worker, "get_latest_preview_frame", lambda: latest_frame.copy())
    failing = [_det("A", 600, 100)]
    ng_frame = np.full((1000, 1000, 3), 42, dtype=np.uint8)

    window._on_overlay_updated(_payload(failing, 1, source_frame=ng_frame))
    window._on_overlay_updated(_payload(failing, 1, source_frame=ng_frame))

    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ng == 1
    assert sound.fail_loop_calls == 1
    assert len(evidence.calls) == 1
    saved_frame, context = evidence.calls[0]
    assert context.round_id == 1
    assert context.source_frame_id == 1
    assert context.detections == [
        {
            **failing[0],
            "polygon": [[95, 95], [105, 95], [105, 105], [95, 105]],
            "box": [95, 95, 105, 105],
            "center": [100, 100],
        }
    ]
    assert context.source_roi == {
        "x": 500,
        "y": 0,
        "width": 500,
        "height": 1000,
        "source_width": 1000,
        "source_height": 1000,
    }
    assert "B" in context.step_name
    assert saved_frame.shape == (1000, 500, 3)
    assert tuple(int(value) for value in saved_frame[900, 400]) == (42, 42, 42)
    assert tuple(int(value) for value in saved_frame[95, 95]) == (0, 255, 128)
    assert not np.array_equal(saved_frame, latest_frame)


def test_roi_ng_evidence_respects_switch_and_pass_does_not_save(qapp, tmp_path):
    disabled_config = _roi_config(tmp_path)
    disabled_config.action_ng_stable_frames = 1
    disabled_config.fail_evidence_enabled = False
    disabled_evidence = EvidenceSpy()
    disabled_window = _window(
        qapp,
        disabled_config,
        sound=SoundSpy(),
        evidence=disabled_evidence,
    )

    disabled_window._on_overlay_updated(_payload([_det("A", 600, 100)], 1))

    assert disabled_evidence.calls == []
    assert disabled_config.fail_evidence_enabled is False

    pass_config = _roi_config(tmp_path)
    pass_config.action_pass_stable_frames = 1
    pass_evidence = EvidenceSpy()
    pass_window = _window(qapp, pass_config, sound=SoundSpy(), evidence=pass_evidence)
    pass_window._on_overlay_updated(
        _payload([_det("A", 600, 100), _det("B", 700, 100)], 2)
    )

    assert pass_evidence.calls == []
    assert pass_config.fail_evidence_enabled is True


def test_roi_ng_evidence_crops_roi_and_draws_only_roi_local_obb(qapp, tmp_path):
    evidence = EvidenceSpy()
    window = _window(qapp, _roi_config(tmp_path), evidence=evidence)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    inside = {
        **_det("inside-label", 100, 60),
        "conf": 0.97,
        "track_id": 17,
        "center_px": [100, 60],
        "polygon": [[70, 50], [110, 45], [115, 65], [75, 70]],
        "box": [70, 45, 115, 70],
    }
    outside = {
        **_det("outside-label", 30, 35),
        "conf": 0.88,
        "track_id": 18,
        "polygon": [[20, 30], [35, 20], [45, 40], [30, 50]],
    }
    invalid = {
        **_det("invalid", 120, 90),
        "polygon": [[115, 85], [125, 85], [125, 95]],
    }
    payload = _payload(
        [inside, outside, invalid],
        42,
        source_frame=frame,
    )

    window._save_roi_fail_evidence(_fail_result(), payload)

    assert len(evidence.calls) == 1
    saved_frame, context = evidence.calls[0]
    expected = frame[:, 80:160].copy()
    local_inside_polygon = [[-10, 50], [30, 45], [35, 65], [-5, 70]]
    cv2.polylines(
        expected,
        [np.asarray(local_inside_polygon, dtype=np.int32)],
        True,
        (0, 255, 128),
        2,
    )
    assert np.array_equal(saved_frame, expected)
    assert saved_frame.shape == (120, 80, 3)
    assert context.source_frame_id == 42
    assert [item["label"] for item in context.detections] == ["inside-label", "invalid"]
    assert context.detections[0]["polygon"] == local_inside_polygon
    assert context.detections[0]["box"] == [-10, 45, 35, 70]
    assert context.detections[0]["center"] == [20, 60]
    assert context.detections[0]["center_px"] == [20, 60]
    assert context.source_roi == {
        "x": 80,
        "y": 0,
        "width": 80,
        "height": 120,
        "source_width": 160,
        "source_height": 120,
    }
    assert outside["center"] == [30, 35]
    assert inside["center"] == [100, 60]
    assert inside["polygon"] == [[70, 50], [110, 45], [115, 65], [75, 70]]


def test_roi_ng_evidence_writes_roi_local_metadata(qapp, tmp_path):
    evidence_root = tmp_path / "roi-evidence"
    saver = FailEvidenceSaver(enabled=True, base_dir=evidence_root)
    window = _window(qapp, _roi_config(tmp_path), evidence=saver)
    frame = np.full((100, 200, 3), 31, dtype=np.uint8)
    inside = _det("A", 130, 40)
    outside = _det("outside", 30, 40)
    payload = _payload([inside, outside], 42, source_frame=frame)
    payload["image_size"] = (200, 100)
    payload["timestamp"] = time.time()

    window._save_roi_fail_evidence(_fail_result(), payload)

    metadata_paths = list(evidence_root.rglob("metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["source_frame_id"] == 42
    assert metadata["source_roi"] == {
        "x": 100,
        "y": 0,
        "width": 100,
        "height": 100,
        "source_width": 200,
        "source_height": 100,
    }
    assert [item["label"] for item in metadata["detections"]] == ["A"]
    assert metadata["detections"][0]["center"] == [30, 40]
    saved = cv2.imdecode(
        np.frombuffer((metadata_paths[0].parent / "frame.jpg").read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert saved.shape == (100, 100, 3)


def test_legacy_fail_evidence_metadata_does_not_add_source_roi(tmp_path):
    saver = FailEvidenceSaver(enabled=True, base_dir=tmp_path / "legacy-evidence")
    folder = saver.save_fail_event(
        np.zeros((20, 30, 3), dtype=np.uint8),
        FailEvidenceContext(
            round_id=3,
            action_ng_step=0,
            step_name="legacy",
            source_frame_id=7,
            timestamp=time.time(),
        ),
    )

    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert "source_roi" not in metadata


def test_roi_ng_evidence_saves_same_frame_without_boxes_when_detections_empty(
    qapp,
    tmp_path,
):
    evidence = EvidenceSpy()
    window = _window(qapp, _roi_config(tmp_path), evidence=evidence)
    frame = np.full((40, 60, 3), 73, dtype=np.uint8)

    window._save_roi_fail_evidence(
        _fail_result(),
        _payload([], 42, source_frame=frame),
    )

    assert len(evidence.calls) == 1
    saved_frame, context = evidence.calls[0]
    assert np.array_equal(saved_frame, frame[:, 30:60])
    assert saved_frame.shape == (40, 30, 3)
    assert saved_frame is not frame
    assert context.detections == []
    assert context.source_roi["width"] == 30


def test_roi_ng_evidence_rejects_invalid_or_empty_roi_without_full_frame_fallback(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _roi_config(tmp_path)
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=SoundSpy(), evidence=evidence)
    errors = []
    monkeypatch.setattr(
        window._runtime_logger,
        "error",
        lambda message, *args: errors.append((message, args)),
    )
    payload = _payload([], 42, source_frame=np.zeros((20, 20, 3), dtype=np.uint8))

    config.roi_inspection_rect = None
    window._save_roi_fail_evidence(_fail_result(), payload)
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.001, "height": 1.0}
    window._save_roi_fail_evidence(_fail_result(), payload)

    assert evidence.calls == []
    assert len(errors) == 2
    assert "invalid_inspection_roi" in errors[0][0]
    assert "empty_inspection_roi" in errors[1][0]


def test_invalid_roi_ng_keeps_stats_and_uses_one_shot_fail_sound(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    errors = []
    monkeypatch.setattr(
        window._runtime_logger,
        "error",
        lambda message, *args: errors.append((message, args)),
    )
    config.roi_inspection_rect = None

    window._on_overlay_updated(_payload([_det("A", 600, 100)], 42))

    assert window._stats_manager.batch.ng == 1
    assert evidence.calls == []
    assert window._roi_fail_alert is None
    assert sound.fail_calls == 1
    assert sound.fail_loop_calls == 0
    assert any("invalid_inspection_roi" in message for message, _args in errors)


def test_roi_ng_evidence_rejects_mismatch_or_missing_frame_without_fallback(
    qapp,
    tmp_path,
    monkeypatch,
):
    evidence = EvidenceSpy()
    window = _window(qapp, _roi_config(tmp_path), evidence=evidence)
    fallback_calls = []
    errors = []
    monkeypatch.setattr(
        window.worker,
        "get_display_frame",
        lambda: (fallback_calls.append("display") or np.zeros((20, 20, 3), dtype=np.uint8), 99),
    )
    monkeypatch.setattr(
        window.worker,
        "get_latest_preview_frame",
        lambda: fallback_calls.append("preview") or np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        window._runtime_logger,
        "error",
        lambda message, *args: errors.append((message, args)),
    )

    window._save_roi_fail_evidence(
        _fail_result(),
        _payload([], 43, source_frame=np.zeros((20, 20, 3), dtype=np.uint8)),
    )
    missing_payload = _payload([], 42)
    missing_payload["source_frame"] = None
    window._save_roi_fail_evidence(_fail_result(), missing_payload)

    assert evidence.calls == []
    assert fallback_calls == []
    assert len(errors) == 2
    assert "frame_id_mismatch" in errors[0][0]
    assert errors[0][1] == (42, 43)
    assert "source_frame_unavailable" in errors[1][0]
    assert errors[1][1] == (42,)


def test_roi_ng_evidence_save_failure_is_logged_without_blocking_feedback(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    evidence = EvidenceSpy(result=None)
    sound = SoundSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    errors = []
    monkeypatch.setattr(
        window._runtime_logger,
        "error",
        lambda message, *args: errors.append((message, args)),
    )

    window._on_overlay_updated(_payload([_det("A", 600, 100)], 42))

    assert len(evidence.calls) == 1
    assert window._stats_manager.batch.ng == 1
    assert window._result_label.text() == "NG"
    assert window._roi_fail_alert is not None
    assert sound.fail_loop_calls == 1
    assert any("save_failed" in message for message, _args in errors)


def test_legacy_action_ng_evidence_still_uses_existing_feedback_frame(
    qapp,
    tmp_path,
    monkeypatch,
):
    evidence = EvidenceSpy()
    window = _window(qapp, _roi_config(tmp_path), evidence=evidence)
    legacy_frame = np.full((20, 30, 3), 91, dtype=np.uint8)
    monkeypatch.setattr(window, "_get_feedback_frame", lambda: (legacy_frame.copy(), 7))
    window._last_overlay_data = {
        "source_frame_id": 7,
        "detections": [_det("A", 10, 10)],
        "model_path": "legacy.onnx",
        "timestamp": time.time(),
    }

    window._save_fail_evidence(SimpleNamespace(round_id=3, action_ng_step=0))

    assert len(evidence.calls) == 1
    saved_frame, context = evidence.calls[0]
    assert np.array_equal(saved_frame, legacy_frame)
    assert context.source_frame_id == 7
    assert context.model_path == "legacy.onnx"


def test_invalid_roi_configuration_blocks_detection_start(qapp, tmp_path, monkeypatch):
    config = _roi_config(tmp_path)
    config.roi_inspection_rect = None
    config.category_counts[0] = "invalid"
    window = _window(qapp, config)
    window.state.camera_on = True
    calls = []
    monkeypatch.setattr(window.worker, "start_inference", lambda: calls.append(True))

    window._on_toggle_detect()

    assert calls == []
    assert "待检数量" in window._status_label.text()
    assert "识别 ROI" in window._notice_label.text()
    assert "待检数量" in window._notice_label.text()


def test_manual_classes_are_checked_against_loaded_model_only_when_detection_starts(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _roi_config(tmp_path)
    config.category_names = ["手填类别", "", "", "", "", ""]
    window = _window(qapp, config)
    window.state.camera_on = True
    calls = []
    monkeypatch.setattr(window.worker, "start_inference", lambda: calls.append(True))

    window._on_toggle_detect()

    assert calls == []
    assert "待检类别 '手填类别' 不属于当前模型" in window._notice_label.text()


def test_config_page_receives_latest_frame(qapp, tmp_path, monkeypatch):
    window = _window(qapp, _roi_config(tmp_path))
    monkeypatch.setattr(
        window.worker,
        "get_latest_preview_frame",
        lambda: np.zeros((480, 640, 3), dtype=np.uint8),
    )

    window._on_config_clicked()
    window._config_page._refresh_roi_frame()

    assert window._config_page._roi_editor.has_frame is True


def test_runtime_reset_clears_roi_round_but_preserves_production_stats(qapp, tmp_path):
    window = _window(qapp, _roi_config(tmp_path))
    passing = [_det("A", 600, 100), _det("B", 700, 100)]
    window._on_overlay_updated(_payload(passing, 1))
    assert window._roi_engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert window._stats_manager.batch.total == 1

    window._clear_camera_session_visuals()

    assert window._roi_engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert window._stats_manager.batch.total == 1


def test_runtime_overlay_draws_only_template_search_roi(qapp, tmp_path):
    config = _enable_template(_roi_config(tmp_path))
    window = _window(qapp, config)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    rendered = window._draw_roi_overlay(frame)

    # 搜索 ROI 边界可见，识别 ROI 边界及内部得分/数量文字均不绘制。
    assert tuple(int(value) for value in rendered[100, 100]) != (0, 0, 0)
    assert tuple(int(value) for value in rendered[0, 500]) == (0, 0, 0)
    assert tuple(int(value) for value in rendered[150, 150]) == (0, 0, 0)


def test_successful_detection_start_clears_stale_missing_roi_notice(qapp, tmp_path, monkeypatch):
    window = _window(qapp, _roi_config(tmp_path))
    window.state.camera_on = True
    window._set_notice("识别 ROI 未配置")
    monkeypatch.setattr(window.worker, "start_inference", lambda: None)

    window._on_toggle_detect()

    assert window._notice_label.text() == ""


def _failing_window(qapp, tmp_path, monkeypatch):
    """构造一个 ROI 模式窗口，FAIL 帧稳定帧=1。"""
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    return window, sound


def test_roi_fail_opens_alert_dialog_and_starts_loop_sound(qapp, tmp_path, monkeypatch):
    """ROI FAIL 时弹出告警窗口并启动循环声音。"""
    window, sound = _failing_window(qapp, tmp_path, monkeypatch)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))

    assert window._roi_fail_alert is not None
    assert window._roi_fail_alert.isVisible()
    assert sound.fail_loop_calls == 1


def test_roi_fail_alert_replaces_image_for_new_round_while_open(qapp, tmp_path, monkeypatch):
    """弹窗已开时，新一轮 FAIL 只替换同帧图片，不重启声音。"""
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    config.round_cooldown_seconds = 0.0
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(
        _payload(failing, 1, source_frame=np.full((1000, 1000, 3), 10, dtype=np.uint8))
    )
    assert sound.fail_loop_calls == 1
    first_pixmap_key = window._roi_fail_alert._image_label.pixmap().cacheKey()

    window._on_overlay_updated(
        _payload(failing, 2, source_frame=np.full((1000, 1000, 3), 20, dtype=np.uint8))
    )
    assert sound.fail_loop_calls == 1
    second_pixmap_key = window._roi_fail_alert._image_label.pixmap().cacheKey()
    assert second_pixmap_key != first_pixmap_key


def test_roi_fail_alert_enter_key_closes_and_stops_sound(qapp, tmp_path, monkeypatch):
    """回车键关闭弹窗并停止循环声音。"""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    window, sound = _failing_window(qapp, tmp_path, monkeypatch)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))
    assert window._roi_fail_alert.isVisible()

    dialog = window._roi_fail_alert
    event = QKeyEvent(QEvent.Type.KeyPress, int(Qt.Key.Key_Return), Qt.KeyboardModifier.NoModifier)
    dialog.keyPressEvent(event)
    qapp.processEvents()

    assert sound.stop_loop_calls == 1
    assert window._roi_fail_alert is None


def test_roi_fail_alert_content_contains_only_the_image(qapp, tmp_path, monkeypatch):
    """弹窗内容区只保留图像，不再显示摘要或操作提示。"""
    window, _ = _failing_window(qapp, tmp_path, monkeypatch)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))
    dialog = window._roi_fail_alert

    assert dialog.layout().count() == 1
    assert dialog.findChildren(QScrollArea) == []
    assert dialog.findChildren(QLabel) == [dialog._image_label]
    assert not hasattr(dialog, "_summary_inner")
    assert not hasattr(dialog, "_hint_label")


def test_roi_ng_alert_uses_event_frame_even_when_worker_frames_advanced(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    window = _window(qapp, config, sound=SoundSpy(), evidence=EvidenceSpy())
    latest_frame = np.full((1000, 1000, 3), 199, dtype=np.uint8)
    monkeypatch.setattr(window.worker, "get_display_frame", lambda: (latest_frame.copy(), 99))
    monkeypatch.setattr(window.worker, "get_latest_preview_frame", lambda: latest_frame.copy())

    class AlertSpy:
        def __init__(self):
            self.frames = []

        def isVisible(self):
            return True

        def refresh(self, frame):
            self.frames.append(frame.copy())

    alert = AlertSpy()
    window._roi_fail_alert = alert
    ng_frame = np.full((1000, 1000, 3), 42, dtype=np.uint8)

    window._on_overlay_updated(
        _payload([_det("A", 600, 100)], 42, source_frame=ng_frame)
    )

    assert len(alert.frames) == 1
    assert alert.frames[0].shape == (1000, 500, 3)
    assert tuple(int(value) for value in alert.frames[0][900, 400]) == (42, 42, 42)
    assert not np.array_equal(alert.frames[0], latest_frame)


def test_roi_ng_alert_draws_only_all_valid_green_obb_outlines(qapp, tmp_path):
    config = _enable_template(_roi_config(tmp_path))
    window = _window(qapp, config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    inside = {
        **_det("inside-label", 100, 60),
        "conf": 0.97,
        "track_id": 17,
        "polygon": [[90, 50], [110, 45], [115, 65], [95, 70]],
    }
    outside = {
        **_det("outside-label", 30, 35),
        "conf": 0.88,
        "track_id": 18,
        "polygon": [[20, 30], [35, 20], [45, 40], [30, 50]],
    }
    invalid = {
        **_det("invalid", 70, 90),
        "polygon": [[65, 85], [75, 85], [75, 95]],
    }

    rendered = window._draw_roi_ng_detection_outlines(
        frame,
        [inside, outside, invalid],
    )
    expected = frame.copy()
    for detection in (inside, outside):
        points = np.asarray(detection["polygon"], dtype=np.int32)
        cv2.polylines(expected, [points], True, (0, 255, 128), 2)

    assert np.array_equal(rendered, expected)
    assert tuple(int(value) for value in rendered[50, 90]) == (0, 255, 128)
    assert tuple(int(value) for value in rendered[30, 20]) == (0, 255, 128)
    assert tuple(int(value) for value in rendered[10, 10]) == (0, 0, 0)


def test_roi_ng_alert_rejects_mismatched_or_missing_event_frame(qapp, tmp_path, monkeypatch):
    window = _window(qapp, _roi_config(tmp_path), sound=SoundSpy(), evidence=EvidenceSpy())
    errors = []
    monkeypatch.setattr(window._runtime_logger, "error", lambda message, *args: errors.append((message, args)))
    result = SimpleNamespace(source_frame_id=42)
    window._show_roi_fail_alert(
        result,
        {
            "source_frame_id": 43,
            "source_frame": np.zeros((20, 20, 3), dtype=np.uint8),
            "detections": [],
        },
    )
    window._show_roi_fail_alert(
        result,
        {"source_frame_id": 42, "source_frame": None, "detections": []},
    )

    assert window._roi_fail_alert is None
    assert len(errors) == 2


def test_detection_display_uses_same_inference_frame_not_latest_preview(qapp, tmp_path):
    """检测中框必须画在被推理的同帧上，不能叠到目标已离开的最新预览。"""
    from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection

    window = _window(qapp, _roi_config(tmp_path))
    infer_frame = np.full((40, 60, 3), (10, 20, 30), dtype=np.uint8)
    preview_frame = np.full((40, 60, 3), (200, 200, 200), dtype=np.uint8)
    shown: list[np.ndarray] = []
    window._show_frame = lambda frame: shown.append(frame.copy())
    window.state.set_camera_on(True)
    assert window.state.set_inference_on(True)
    window.worker.get_display_frame = lambda: (infer_frame.copy(), 7)
    window.worker.get_latest_preview_frame = lambda: preview_frame.copy()
    window._latest_runtime_overlay = DetectionOverlayState(
        source_frame_id=7,
        timestamp=time.time(),
        detections=[
            ObbDetection(
                class_id=0,
                label="A",
                conf=0.9,
                track_id=None,
                polygon=[(8, 8), (18, 8), (18, 18), (8, 18)],
                box=(8, 8, 18, 18),
                center=(13, 13),
            )
        ],
    )

    window._update_display()

    assert shown
    assert float(shown[0].mean()) < 80

    shown.clear()
    window.state.set_inference_on(False)
    window._update_display()
    assert shown
    assert float(shown[0].mean()) > 150


def test_roi_pass_does_not_open_alert_dialog(qapp, tmp_path):
    """PASS 时不弹出告警窗口。"""
    sound = SoundSpy()
    config = _roi_config(tmp_path)
    config.action_pass_stable_frames = 1
    window = _window(qapp, config, sound=sound)
    passing = [_det("A", 600, 100), _det("B", 700, 100)]

    window._on_overlay_updated(_payload(passing, 1))

    assert window._roi_fail_alert is None
    assert sound.fail_loop_calls == 0


def test_template_match_trigger_via_overlay(qapp, tmp_path):
    """模板匹配 overlay 携带 matched=True 时触发本轮。"""
    config = _enable_template(_roi_config(tmp_path))
    sound = SoundSpy()
    window = _window(qapp, config, sound=sound)
    passing = [_det("A", 600, 100), _det("B", 700, 100)]

    # 模板匹配成功 + YOLO 检测到 A 和 B -> PASS
    window._on_overlay_updated(_payload(passing, 1, template_matched=True, template_score=0.75))

    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ok == 1
    assert window._roi_engine.snapshot.template_matched is True


def test_template_not_matched_does_not_trigger(qapp, tmp_path):
    """模板匹配未命中时不触发本轮。"""
    config = _enable_template(_roi_config(tmp_path))
    sound = SoundSpy()
    window = _window(qapp, config, sound=sound)
    passing = [_det("A", 600, 100), _det("B", 700, 100)]

    window._on_overlay_updated(_payload(passing, 1, template_matched=False, template_score=0.3))

    assert window._stats_manager.batch.total == 0
    assert window._roi_engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert window._roi_engine.snapshot.template_matched is False
