from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QLabel

from app_state import AppState
from config import ConfigManager
from roi_inspection.geometry import NormalizedRect
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.config_page import ConfigPage
from ui.main_window import MainWindow
from ui.roi_editor import RoiEditorWidget, RoiKind
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection
import ui.widgets.native_panels as native_panels
from ui.widgets.native_panels import (
    ConfigNavButton,
    FooterActionBar,
    FormGroup,
    KpiRow,
    RecognitionListItem,
    SwitchRow,
)


class DummyProcessor:
    def get_class_names(self):
        return {0: "A", 1: "B"}


def _detection(label: str) -> ObbDetection:
    return ObbDetection(
        class_id=0,
        label=label,
        conf=0.9,
        track_id=1,
        polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
        box=(10, 10, 30, 30),
        center=(20, 20),
    )


def _overlay_payload(frame_id: int, label: str = "A") -> dict:
    return {
        "source_frame_id": frame_id,
        "timestamp": time.time(),
        "model_path": "test.onnx",
        "task_type": "onnx_obb",
        "detections": [
            {
                "class_id": 0,
                "label": label,
                "conf": 0.9,
                "track_id": None,
                "polygon": [(10, 10), (30, 10), (30, 30), (10, 30)],
                "box": (10, 10, 30, 30),
                "center": (20, 20),
            }
        ],
        "image_size": (60, 40),
        "status": "",
        "error": "",
        "round_id": 0,
        "latency_ms": 5.0,
    }


def _window(config: ConfigManager) -> MainWindow:
    return MainWindow(
        config,
        AppState(),
        processor=DummyProcessor(),
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.get_runtime_category_counts(),
        ),
        processor_factory=lambda *args, **kwargs: DummyProcessor(),
    )


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_config_navigation_preserves_unsaved_values(qapp):
    page = ConfigPage(ConfigManager())

    page._model_path_input.setText("unsaved-model.onnx")
    page._show_section(1)
    page._show_section(0)

    assert page._section_stack.count() == 6
    assert [button.text() for button in page._nav_buttons] == [
        "模型与步骤",
        "动作判定",
        "工业相机",
        "显示与反馈",
        "ROI 配置",
        "生产统计",
    ]
    assert page._model_path_input.text() == "unsaved-model.onnx"


def test_round_interval_has_single_action_judgement_control(qapp):
    page = ConfigPage(ConfigManager())

    assert page._action_page.isAncestorOf(page._cooldown_input)
    assert not page._region_page.isAncestorOf(page._cooldown_input)
    assert "PCB" not in " ".join(label.text() for label in page._region_page.findChildren(type(page._region_title_label)))


def test_recognition_item_displays_quantity_and_semantic_states(qapp):
    item = RecognitionListItem(1, "脚垫", required_count=4)

    item.set_step_state("active")
    item.set_quantity_progress(3, 4)
    assert item.quantity_text() == "3 / 4"
    assert item.state_name() == "active"
    assert item.status_text() == "检测中"

    item.set_step_state("pass")
    item.set_quantity_progress(4, 4)
    assert item.quantity_text() == "4 / 4"
    assert item.state_name() == "pass"
    assert item.status_text() == "PASS"

    item.set_step_state("ng")
    assert item.state_name() == "ng"
    assert item.status_text() == "NG"


def test_runtime_and_config_use_live_widgets_not_retired_shell(qapp):
    config = ConfigManager()
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [2, 3, 1, 1, 1, 1]
    window = _window(config)
    page = ConfigPage(config)

    assert window.findChildren(KpiRow)
    assert window.findChildren(RecognitionListItem)
    assert page.findChildren(ConfigNavButton)
    assert page.findChildren(FormGroup)
    assert page.findChildren(SwitchRow)
    assert page.findChildren(FooterActionBar)
    assert page.findChildren(RoiEditorWidget)
    for retired in ("StepCard", "PanelSection", "MetricCard", "DefectChip"):
        assert not hasattr(native_panels, retired)


def test_sequence_waiting_items_stay_hidden_until_detected(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [2, 3, 1, 1, 1, 1]
    window = _window(config)

    assert [card.isHidden() for card in window._step_cards] == [True, True]
    assert window._step_cards[0].quantity_text() == "0 / 2"
    assert window._step_cards[1].quantity_text() == "0 / 3"


def test_sequence_items_appear_in_detection_order(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine.start_round()

    first = window.step_engine.update([_detection("A")])
    window._update_step_display(first)
    assert window._step_cards[0].isHidden() is False
    assert window._step_cards[1].isHidden() is True


def test_lstm_label_turns_yellow_then_green_and_clears_after_the_round(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = True
    config.category_names = ["包装", "封装", "", "", "", ""]
    window = _window(config)
    window.step_engine.start_round()

    state = window.step_engine.get_state()
    window._update_step_display(state, live_action="包装")
    assert window._step_cards[0].isHidden() is False
    assert window._step_cards[0]._state == "progress"
    assert window._step_cards[1].isHidden() is True

    window._update_step_display(state, live_action="封装")
    assert window._step_cards[0]._state == "progress"
    assert window._step_cards[1].isHidden() is True

    state = window.step_engine.apply_action_event({
        "session_id": 1,
        "event_id": 1,
        "name": "包装",
        "start_frame": 1,
        "missing_parts": [],
    })
    window._update_step_display(state, live_action="封装")
    assert window._step_cards[0]._state == "pass"
    assert window._step_cards[0].isHidden() is False
    assert window._step_cards[1]._state == "progress"
    assert window._step_cards[1].isHidden() is False

    state = window.step_engine.apply_action_event({
        "session_id": 1,
        "event_id": 2,
        "name": "封装",
        "start_frame": 20,
        "missing_parts": [],
    })
    window._update_step_display(state, live_action="")
    assert window._step_cards[0]._state == "pass"
    assert window._step_cards[1]._state == "pass"
    assert state.round_result.value == "pass"

    window.step_engine.start_round()
    window._update_step_display(window.step_engine.get_state(), live_action="")
    assert [card.isHidden() for card in window._step_cards] == [True, True]


def test_later_step_stays_hidden_until_previous_passed(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["散热片", "导热垫", "", "", "", ""]
    config.category_counts = [1, 3, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine.start_round()

    state = window.step_engine.update([_detection("导热垫"), _detection("导热垫")])
    window._update_step_display(state)

    assert window._step_cards[0].isHidden() is True
    assert window._step_cards[1].isHidden() is True


def test_roi_waiting_items_stay_hidden_before_locator(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.roi_inspection_enabled = True
    config.roi_template_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window._sync_roi_cards()

    assert [card.isHidden() for card in window._step_cards] == [True, True]


def test_kpi_capacity_shows_inspected_total(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    window = _window(config)

    assert window._kpi_row._cells["output"].text() == "0"
    window._stats_manager.record_pass()
    window._update_kpi()
    assert window._kpi_row._cells["output"].text() == "1"


def test_config_round_trip_keeps_production_fields_off_legacy_form(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config)
    page._model_path_input.setText("site.onnx")
    page._step_inputs[0].setText("脚垫")
    page._step_count_inputs[0].setValue(4)
    page._action_pass_frames_input.setValue(3)
    page._cooldown_input.setValue(1.5)
    page._switch_confirm_input.setValue(0.8)
    page._pass_sound_cb.setChecked(True)
    page._action_order_cb.setChecked(False)
    page._on_save()

    reloaded = ConfigManager()
    reloaded.load(tmp_path / "config.json")
    again = ConfigPage(reloaded)
    assert again._model_path_input.text() == "site.onnx"
    assert again._step_inputs[0].text() == "脚垫"
    assert again._step_count_inputs[0].value() == 4
    assert again._action_pass_frames_input.value() == 3
    assert again._cooldown_input.value() == 1.5
    assert again._switch_confirm_input.value() == pytest.approx(0.8)
    assert again._pass_sound_cb.isChecked() is True

    labels = " ".join(label.text() for label in again.findChildren(QLabel))
    for banned in ("扭力枪", "标定", "洞口", "融合", "关节追踪"):
        assert banned not in labels


def test_full_page_save_does_not_overwrite_independent_roi_rects(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    saved_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.roi_inspection_rect = saved_rect
    page = ConfigPage(config)
    page._roi_editor.set_rect(RoiKind.INSPECTION, NormalizedRect(0.1, 0.1, 0.2, 0.2))
    page._action_order_cb.setChecked(False)
    page._on_save()

    assert config.roi_inspection_rect == saved_rect


def test_overlay_events_advance_judgement_without_rendering(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.action_order_constraint_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine.start_round()

    judged: list[list] = []
    original_update = window.step_engine.update

    def _spy(detections, **kwargs):
        state = original_update(detections, **kwargs)
        judged.append(list(detections))
        return state

    window.step_engine.update = _spy

    rendered: list[int] = []
    window._show_frame = lambda frame: rendered.append(1)

    for frame_id in range(1, 6):
        window._on_overlay_updated(_overlay_payload(frame_id))

    assert len(judged) == 5
    assert rendered == []


def test_display_renders_latest_state_after_backlog(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine.start_round()

    for frame_id in range(1, 6):
        window._on_overlay_updated(_overlay_payload(frame_id))

    assert window._latest_runtime_overlay.source_frame_id == 5

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    window.worker.get_display_frame = lambda: (frame.copy(), 5)
    window.worker.get_display_frame_id = lambda: 5

    rendered: list[int] = []
    window._show_frame = lambda image: rendered.append(1)
    window._update_display()

    assert rendered == [1]


def _detecting_window(config: ConfigManager, frame_id: int = 7):
    window = _window(config)
    window.step_engine.start_round()
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    window.worker.get_display_frame = lambda: (frame.copy(), window._fake_frame_id)
    window.worker.get_display_frame_id = lambda: window._fake_frame_id
    window._fake_frame_id = frame_id
    return window


def test_detecting_display_skips_repeat_render_of_same_result(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _detecting_window(config, frame_id=7)
    window._on_overlay_updated(_overlay_payload(7))

    rendered: list[int] = []
    window._show_frame = lambda image: rendered.append(1)

    for _ in range(5):
        window._update_display()

    assert rendered == [1]

    window._fake_frame_id = 8
    window._on_overlay_updated(_overlay_payload(8))
    window._update_display()

    assert rendered == [1, 1]


def test_preview_display_skips_repeat_render_of_same_camera_frame(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _window(config)
    window.state.set_camera_on(True)

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    window._fake_preview_id = 3
    window.worker.get_latest_preview_frame = lambda: frame.copy()
    window.worker.get_latest_preview_frame_id = lambda: window._fake_preview_id

    rendered: list[int] = []
    window._show_frame = lambda image: rendered.append(1)

    for _ in range(4):
        window._update_display()

    assert rendered == [1]

    window._fake_preview_id = 4
    window._update_display()

    assert rendered == [1, 1]


def test_display_redraws_when_video_label_resized(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _detecting_window(config, frame_id=7)
    window._on_overlay_updated(_overlay_payload(7))

    rendered: list[int] = []
    window._show_frame = lambda image: rendered.append(1)

    window._update_display()
    window._update_display()
    assert rendered == [1]

    window._video_label.setFixedSize(320, 240)
    window._update_display()
    assert rendered == [1, 1]


def test_stopped_detection_draws_no_boxes_even_within_freshness_window(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.display_result_max_age_ms = 5000
    window = _detecting_window(config, frame_id=7)
    window._on_overlay_updated(_overlay_payload(7))

    assert window._should_draw_runtime_overlay(window._latest_runtime_overlay, 7) is True

    window.state.set_inference_on(False)

    assert window._should_draw_runtime_overlay(window._latest_runtime_overlay, 7) is False


def test_preview_keeps_roi_drawing_after_detection_stops(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _detecting_window(config, frame_id=7)
    window._on_overlay_updated(_overlay_payload(7))

    calls = {"overlay": 0, "roi": 0}

    def _fake_overlay(frame, overlay):
        calls["overlay"] += 1
        return frame

    def _fake_roi(frame):
        calls["roi"] += 1
        return frame

    window._draw_overlay = _fake_overlay
    window._draw_roi_overlay = _fake_roi
    window._show_frame = lambda image: None

    window._update_display()
    assert calls == {"overlay": 1, "roi": 0}

    window.state.set_inference_on(False)
    window._roi_engine = object()
    preview = np.zeros((40, 60, 3), dtype=np.uint8)
    window.worker.get_latest_preview_frame = lambda: preview.copy()
    window.worker.get_latest_preview_frame_id = lambda: 9
    window._update_display()

    assert calls == {"overlay": 1, "roi": 1}


def test_frame_is_scaled_before_color_conversion(qapp, monkeypatch):
    import ui.main_window as main_window_module

    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _window(config)
    window._video_label.setFixedSize(160, 120)

    converted: list[tuple] = []
    original_cvt = main_window_module.cv2.cvtColor

    def _spy(src, code, *args, **kwargs):
        converted.append(src.shape[:2])
        return original_cvt(src, code, *args, **kwargs)

    monkeypatch.setattr(main_window_module.cv2, "cvtColor", _spy)

    window._show_frame(np.zeros((1200, 1600, 3), dtype=np.uint8))

    assert converted == [(120, 160)]
    pixmap = window._video_label.pixmap()
    assert (pixmap.width(), pixmap.height()) == (160, 120)


def test_frame_scaling_keeps_aspect_ratio_and_never_upscales_beyond_label(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    window = _window(config)
    window._video_label.setFixedSize(200, 200)

    scaled = window._scale_frame_to_label(np.zeros((300, 600, 3), dtype=np.uint8))

    assert scaled.shape[:2] == (100, 200)


def test_return_without_save_keeps_last_saved_config(qapp):
    config = ConfigManager()
    config.yolo_model_path = "saved.onnx"
    window = _window(config)
    window._on_config_clicked()
    window._config_page._model_path_input.setText("unsaved.onnx")
    window._on_config_back()

    assert config.yolo_model_path == "saved.onnx"
    assert window._stacked.currentWidget() is window._main_page


def test_sequential_ng_paints_only_failed_class_red(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["A", "B", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine.start_round(require_first_step_rearm=False)
    window.step_engine._mark_quantity_ng(window.step_engine._steps[1])

    assert window._overlay_box_color("B") == (0, 0, 255)
    assert window._overlay_box_color("A") == (0, 255, 128)

    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    overlay = DetectionOverlayState(
        source_frame_id=1,
        detections=[
            ObbDetection(
                class_id=0,
                label="A",
                conf=0.9,
                track_id=None,
                polygon=[(10, 10), (40, 10), (40, 40), (10, 40)],
                box=(10, 10, 40, 40),
                center=(25, 25),
            ),
            ObbDetection(
                class_id=1,
                label="B",
                conf=0.9,
                track_id=None,
                polygon=[(90, 10), (130, 10), (130, 40), (90, 40)],
                box=(90, 10, 130, 40),
                center=(110, 25),
            ),
        ],
    )
    painted = window._draw_overlay(frame, overlay)
    assert tuple(int(v) for v in painted[10, 20]) == (0, 255, 128)
    assert tuple(int(v) for v in painted[10, 110]) == (0, 0, 255)

    window.step_engine.start_round(require_first_step_rearm=False)
    assert window._overlay_box_color("B") == (0, 255, 128)
