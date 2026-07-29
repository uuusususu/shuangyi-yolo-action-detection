"""ROI 定位检测主窗口集成测试。"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from app_state import AppState
from config import ConfigManager
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
    def __init__(self):
        self.enabled = True
        self.calls = []

    def save_fail_event(self, frame, context):
        self.calls.append((frame, context))


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


def _payload(detections, frame_id=1, *, template_matched=None, template_score=None):
    payload = {
        "source_frame_id": frame_id,
        "timestamp": time.monotonic(),
        "model_path": "model.onnx",
        "task_type": "ultralytics_obb",
        "detections": detections,
        "image_size": (1000, 1000),
    }
    if template_matched is not None:
        payload["template_matched"] = template_matched
        payload["template_score"] = template_score or 0.0
    return payload


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
    monkeypatch.setattr(
        window,
        "_get_feedback_frame",
        lambda: (np.zeros((20, 20, 3), dtype=np.uint8), 1),
    )
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))
    window._on_overlay_updated(_payload(failing, 2))

    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ng == 1
    assert sound.fail_loop_calls == 1
    assert len(evidence.calls) == 1
    assert evidence.calls[0][1].round_id == 1
    assert "B" in evidence.calls[0][1].step_name


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
    """构造一个 ROI 模式窗口，FAIL 帧稳定帧=1，反馈帧返回固定画面。"""
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    monkeypatch.setattr(
        window,
        "_get_feedback_frame",
        lambda: (np.zeros((20, 20, 3), dtype=np.uint8), 1),
    )
    return window, sound


def test_roi_fail_opens_alert_dialog_and_starts_loop_sound(qapp, tmp_path, monkeypatch):
    """ROI FAIL 时弹出告警窗口并启动循环声音。"""
    window, sound = _failing_window(qapp, tmp_path, monkeypatch)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))

    assert window._roi_fail_alert is not None
    assert window._roi_fail_alert.isVisible()
    assert sound.fail_loop_calls == 1


def test_roi_fail_alert_appends_new_round_while_open(qapp, tmp_path, monkeypatch):
    """弹窗已开时，新一轮 FAIL 刷新画面并追加一行，不重启声音。"""
    config = _roi_config(tmp_path)
    config.action_ng_stable_frames = 1
    config.round_cooldown_seconds = 0.0
    sound = SoundSpy()
    evidence = EvidenceSpy()
    window = _window(qapp, config, sound=sound, evidence=evidence)
    monkeypatch.setattr(
        window,
        "_get_feedback_frame",
        lambda: (np.zeros((20, 20, 3), dtype=np.uint8), 1),
    )
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))
    assert sound.fail_loop_calls == 1
    first_text = window._roi_fail_alert._summary_inner.text()
    assert "第1轮" in first_text

    window._on_overlay_updated(_payload(failing, 2))
    assert sound.fail_loop_calls == 1
    second_text = window._roi_fail_alert._summary_inner.text()
    assert "第1轮" in second_text
    assert "第2轮" in second_text


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


def test_roi_fail_alert_summary_format(qapp, tmp_path, monkeypatch):
    """失败摘要格式：第N轮 NG: 类别B 0/1。"""
    window, _ = _failing_window(qapp, tmp_path, monkeypatch)
    failing = [_det("A", 600, 100)]

    window._on_overlay_updated(_payload(failing, 1))
    text = window._roi_fail_alert._summary_inner.text()

    assert "第1轮 NG:" in text
    assert "B 0/1" in text or "B 0 / 1" in text
    assert "A" not in text


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
