from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from pcb_inspection.models import PcbInspectionResult, PcbResult
from ui.config_page import ConfigPage
from ui.main_window import MainWindow
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


class DummyProcessor:
    def __init__(self, name: str, class_names: dict[int, str] | None = None) -> None:
        self.name = name
        self._class_names = class_names or {0: "模型类别A", 1: "模型类别B"}

    def get_class_names(self) -> dict[int, str]:
        return dict(self._class_names)


class SoundSpy:
    def __init__(self, *, raise_on_play: bool = False) -> None:
        self.enabled = True
        self.raise_on_play = raise_on_play
        self.pass_calls = 0
        self.fail_calls = 0

    def play_pass(self) -> None:
        self.pass_calls += 1
        if self.raise_on_play:
            raise RuntimeError("pass sound failed")

    def play_fail(self) -> None:
        self.fail_calls += 1
        if self.raise_on_play:
            raise RuntimeError("fail sound failed")


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _config(tmp_path: Path) -> ConfigManager:
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.yolo_model_path = "old.onnx"
    config.yolo_conf_threshold = 0.3
    config.category_names = ["手动1", "手动2", "", "", "", ""]
    return config


def _window(
    qapp,
    config: ConfigManager,
    processor_factory,
    processor=None,
    sound_feedback=None,
) -> MainWindow:
    state = AppState()
    step_engine = StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=getattr(config, "category_counts", None),
    )
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=step_engine,
        processor_factory=processor_factory,
        app_base_dir=Path(config.get_config_path()).parent,
        sound_feedback=sound_feedback,
    )
    if processor is not None:
        window.worker.set_frame_processor(processor)
    return window


def test_config_page_save_persists_manual_categories_without_model_autofill(qapp, tmp_path, monkeypatch):
    config = _config(tmp_path)
    page = ConfigPage(config)
    emitted = []
    page.config_saved.connect(lambda: emitted.append(True))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    page._model_path_input.setText("new.onnx")
    page._step_inputs[0].setText("人工步骤A")
    page._step_inputs[1].setText("人工步骤B")

    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert emitted == [True]
    assert saved["yolo_model_path"] == "new.onnx"
    assert saved["category_names"][:2] == ["人工步骤A", "人工步骤B"]
    assert config.category_names[:2] == ["人工步骤A", "人工步骤B"]


def test_config_page_loads_and_saves_independent_result_sound_switches(qapp, tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.pass_sound_enabled = True
    config.fail_sound_enabled = False
    page = ConfigPage(config)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    assert page._pass_sound_cb.isChecked() is True
    assert page._fail_sound_cb.isChecked() is False

    page._pass_sound_cb.setChecked(False)
    page._fail_sound_cb.setChecked(True)
    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["pass_sound_enabled"] is False
    assert saved["fail_sound_enabled"] is True
    assert config.pass_sound_enabled is False
    assert config.fail_sound_enabled is True


def test_runtime_uses_independent_result_sound_switches(qapp, tmp_path):
    config = _config(tmp_path)
    config.pass_sound_enabled = False
    config.fail_sound_enabled = False
    config.sound_feedback_enabled = True
    config.fail_evidence_enabled = False
    sound = SoundSpy()

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(
        qapp,
        config,
        factory,
        processor=DummyProcessor("old"),
        sound_feedback=sound,
    )
    ng_state = SimpleNamespace(round_id=1, action_ng_step=-1)

    window._play_pass_feedback()
    window._handle_action_ng_feedback(ng_state)
    assert sound.pass_calls == 0
    assert sound.fail_calls == 0

    config.pass_sound_enabled = True
    config.fail_sound_enabled = True
    window._play_pass_feedback()
    window._handle_action_ng_feedback(SimpleNamespace(round_id=2, action_ng_step=-1))
    assert sound.pass_calls == 1
    assert sound.fail_calls == 1


def test_normal_mode_result_sound_plays_once_per_round(qapp, tmp_path):
    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    def detection(label):
        return ObbDetection(
            class_id=0,
            label=label,
            conf=0.9,
            track_id=1,
            polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
            box=(10, 10, 30, 30),
            center=(20, 20),
        )

    pass_config = _config(tmp_path / "pass")
    pass_config.category_names = ["A", "B", "C", "", "", ""]
    pass_config.pass_sound_enabled = True
    pass_config.fail_sound_enabled = False
    pass_config.round_cooldown_seconds = 60.0
    pass_sound = SoundSpy()
    pass_window = _window(
        qapp,
        pass_config,
        factory,
        processor=DummyProcessor("old"),
        sound_feedback=pass_sound,
    )
    pass_window.step_engine = StepSequenceEngine(
        step_class_names=pass_config.category_names,
        enter_stable_frames=1,
        out_of_order_frames=1,
    )
    pass_window._refresh_steps()
    pass_window.step_engine.start_round(require_first_step_rearm=False)
    pass_window.step_engine.update([detection("A")])
    pass_window.step_engine.update([detection("B")])
    pass_state = pass_window.step_engine.update([detection("C")])

    pass_window._update_step_display(pass_state)
    pass_window._update_step_display(pass_state)
    assert pass_sound.pass_calls == 1
    assert pass_sound.fail_calls == 0
    pass_window._round_pass_timer.stop()

    fail_config = _config(tmp_path / "fail")
    fail_config.category_names = ["A", "B", "C", "", "", ""]
    fail_config.pass_sound_enabled = False
    fail_config.fail_sound_enabled = True
    fail_config.round_cooldown_seconds = 60.0
    fail_sound = SoundSpy()
    fail_window = _window(
        qapp,
        fail_config,
        factory,
        processor=DummyProcessor("old"),
        sound_feedback=fail_sound,
    )
    fail_window.step_engine = StepSequenceEngine(
        step_class_names=fail_config.category_names,
        enter_stable_frames=1,
        out_of_order_frames=1,
    )
    fail_window._refresh_steps()
    fail_window.step_engine.start_round(require_first_step_rearm=False)
    fail_window.step_engine.update([detection("A")])
    fail_state = fail_window.step_engine.update([detection("C")])

    fail_window._update_step_display(fail_state)
    fail_window._update_step_display(fail_state)
    assert fail_sound.pass_calls == 0
    assert fail_sound.fail_calls == 1
    fail_window._round_pass_timer.stop()


def test_region_fail_sound_is_deduplicated_by_attempt_not_signature(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["parent", "child", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.pass_sound_enabled = False
    config.fail_sound_enabled = True
    config.fail_evidence_enabled = False
    sound = SoundSpy()

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(
        qapp,
        config,
        factory,
        processor=DummyProcessor("old"),
        sound_feedback=sound,
    )
    base_ng = window._stats_manager.batch.ng

    first_attempt = PcbInspectionResult(
        track_id=1,
        attempt_id=10,
        result=PcbResult.FAIL,
        is_new_fail_signature=True,
        missing_classes=["child"],
        observed_counts={"child": 3},
        required_counts={"child": 4},
    )
    same_signature_new_attempt = PcbInspectionResult(
        track_id=1,
        attempt_id=11,
        result=PcbResult.FAIL,
        is_new_fail_signature=False,
        missing_classes=["child"],
        observed_counts={"child": 3},
        required_counts={"child": 4},
    )
    second_parent = PcbInspectionResult(
        track_id=2,
        attempt_id=12,
        result=PcbResult.FAIL,
        is_new_fail_signature=True,
        missing_classes=["child"],
        observed_counts={"child": 0},
        required_counts={"child": 4},
    )

    window._handle_pcb_results([first_attempt], frame_id=1)
    window._handle_pcb_results([first_attempt], frame_id=1)
    window._handle_pcb_results([same_signature_new_attempt], frame_id=2)
    window._handle_pcb_results([second_parent], frame_id=3)

    assert sound.fail_calls == 3
    assert window._stats_manager.batch.ng == base_ng + 2

    config.fail_sound_enabled = False
    window._handle_pcb_results(
        [
            PcbInspectionResult(
                track_id=3,
                attempt_id=13,
                result=PcbResult.FAIL,
                is_new_fail_signature=True,
                missing_classes=["child"],
            )
        ],
        frame_id=4,
    )
    assert sound.fail_calls == 3


def test_sound_player_failure_does_not_interrupt_region_result_flow(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["parent", "child", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 1
    config.round_cooldown_seconds = 0.0
    config.pass_sound_enabled = False
    config.fail_sound_enabled = True
    config.fail_evidence_enabled = False
    sound = SoundSpy(raise_on_play=True)

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(
        qapp,
        config,
        factory,
        processor=DummyProcessor("old"),
        sound_feedback=sound,
    )
    base_pass = window._stats_manager.batch.ok
    base_ng = window._stats_manager.batch.ng

    def parent(track_id, x):
        return ObbDetection(
            class_id=0,
            label="parent",
            conf=0.9,
            track_id=track_id,
            polygon=[(x, 100), (x + 200, 100), (x + 200, 300), (x, 300)],
            box=(x, 100, x + 200, 300),
            center=(x + 100, 200),
        )

    def children(x, count):
        return [
            ObbDetection(
                class_id=1,
                label="child",
                conf=0.9,
                track_id=None,
                polygon=[
                    (x + 20 + index * 30, 130),
                    (x + 40 + index * 30, 130),
                    (x + 40 + index * 30, 150),
                    (x + 20 + index * 30, 150),
                ],
                box=(x + 20 + index * 30, 130, x + 40 + index * 30, 150),
                center=(x + 30 + index * 30, 140),
            )
            for index in range(count)
        ]

    detections = [parent(1, 100)] + children(100, 3) + [parent(2, 400)] + children(400, 4)
    serialized = []
    for detection in detections:
        item = detection.to_dict()
        item["center"] = item["center_px"]
        serialized.append(item)

    window._on_overlay_updated(
        {
            "source_frame_id": 1,
            "timestamp": 0.0,
            "model_path": "model.onnx",
            "task_type": "onnx_obb",
            "detections": serialized,
            "image_size": (1000, 1000),
        }
    )

    assert sound.fail_calls == 1
    assert window._stats_manager.batch.ng == base_ng + 1
    assert window._stats_manager.batch.ok == base_pass + 1
    assert window._pcb_engine.pcb_states[1].result == PcbResult.FAIL
    assert window._pcb_engine.pcb_states[2].result == PcbResult.PASS
    assert window._pcb_engine.current_round_id == 1

    overlay = DetectionOverlayState(detections=window._pcb_engine.last_resolved_detections)
    rendered = window._draw_overlay(np.zeros((1000, 1000, 3), dtype=np.uint8), overlay)
    assert tuple(int(value) for value in rendered[100, 100]) == (0, 0, 255)


def test_config_save_reloads_processor_and_updates_worker(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")
    created = []

    def factory(runtime_config, app_base_dir=None):
        created.append((runtime_config.yolo_model_path, Path(app_base_dir)))
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=old_processor)
    config.yolo_model_path = "new.onnx"

    window._on_config_saved()

    assert created == [("new.onnx", tmp_path)]
    assert window.processor is window.worker.frame_processor
    assert window.processor is not old_processor
    assert window.processor.name == "new.onnx"


def test_failed_model_reload_preserves_previous_processor(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")

    def factory(runtime_config, app_base_dir=None):
        raise RuntimeError("bad model")

    window = _window(qapp, config, factory, processor=old_processor)
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    config.yolo_model_path = "broken.onnx"

    window._on_config_saved()

    assert window.processor is old_processor
    assert window.worker.frame_processor is old_processor
    assert window.state.inference_on is True
    assert "模型加载失败" in window._status_label.text()


def test_manual_categories_drive_rebuilt_step_engine(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path, class_names={0: "模型自动A"})

    window = _window(qapp, config, factory, processor=old_processor)
    config.yolo_model_path = "new.onnx"
    config.category_names = ["人工A", "人工B", "", "", "", ""]

    window._on_config_saved()

    state = window.step_engine.get_state()
    assert [step.class_name for step in state.steps[:2]] == ["人工A", "人工B"]
    assert config.category_names[:2] == ["人工A", "人工B"]


def test_runtime_header_uses_centered_brand_and_compact_actions(qapp, tmp_path):
    config = _config(tmp_path)

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    window.resize(1200, 760)
    window.show()
    qapp.processEvents()

    assert window._title_label.text() == "双翼科技AI系统"
    assert window._title_label.alignment() == Qt.AlignmentFlag.AlignCenter
    assert abs(window._title_label.geometry().center().x() - window._header.rect().center().x()) <= 2
    assert window._status_label.text() == "READY"
    assert window._btn_config.height() == 36
    assert window._btn_close.height() == 36
    assert window._status_label.geometry().center().y() == window._btn_config.geometry().center().y()
    window.close()


def test_configured_steps_are_hidden_until_detected(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))

    assert [item._name_label.text() for item in window._step_cards] == ["pcb", "sanreidian"]
    assert all(not item.isVisibleTo(window._steps_container) for item in window._step_cards)
    assert [item.quantity_text() for item in window._step_cards] == ["0 / 1", "0 / 4"]
    assert window._recognized_count_label.text() == "0 / 2"


def test_steps_are_revealed_in_detection_order(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    window.step_engine.start_round()
    pcb = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=1,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )

    state = window.step_engine.update([pcb])
    window._update_step_display(state)

    assert window._step_cards[0].isVisibleTo(window._steps_container)
    assert not window._step_cards[1].isVisibleTo(window._steps_container)


def test_region_steps_reveal_parent_then_detected_child(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 3

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )
    child = ObbDetection(
        class_id=0,
        label="sanreidian",
        conf=0.9,
        track_id=None,
        polygon=[(120, 130), (140, 130), (140, 150), (120, 150)],
        box=(120, 130, 140, 150),
        center=(130, 140),
    )

    assert window._pcb_engine.update([parent], image_size=(1000, 1000)) == []
    window._sync_region_observation_cards()
    assert window._step_cards[0].isVisibleTo(window._steps_container)
    assert window._step_cards[1].isVisibleTo(window._steps_container)
    assert window._step_cards[1].quantity_text() == "0 / 4"
    assert window._step_cards[1].state_name() == "active"

    assert window._pcb_engine.update([parent, child], image_size=(1000, 1000)) == []
    window._sync_region_observation_cards()
    assert window._step_cards[1].isVisibleTo(window._steps_container)
    assert window._step_cards[1].quantity_text() == "1 / 4"


def test_region_steps_hide_when_current_frame_has_no_parent(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )
    child = ObbDetection(
        class_id=0,
        label="sanreidian",
        conf=0.9,
        track_id=None,
        polygon=[(120, 130), (140, 130), (140, 150), (120, 150)],
        box=(120, 130, 140, 150),
        center=(130, 140),
    )

    window._pcb_engine.update([parent, child], image_size=(1000, 1000))
    window._sync_region_observation_cards()
    assert window._step_cards[0].isVisibleTo(window._steps_container)
    assert window._step_cards[1].isVisibleTo(window._steps_container)

    window._pcb_engine.update([], image_size=(1000, 1000))
    window._sync_region_observation_cards()

    assert not window._step_cards[0].isVisibleTo(window._steps_container)
    assert not window._step_cards[1].isVisibleTo(window._steps_container)
    assert window._recognized_count_label.text() == "0 / 2"


def test_region_detail_binds_one_parent_id_without_merging_counts(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    left_parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
        box=(50, 50, 250, 250),
        center=(150, 150),
    )
    right_parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(500, 50), (700, 50), (700, 250), (500, 250)],
        box=(500, 50, 700, 250),
        center=(600, 150),
    )

    def child(x):
        return ObbDetection(
            class_id=0,
            label="sanreidian",
            conf=0.9,
            track_id=None,
            polygon=[(x, 100), (x + 20, 100), (x + 20, 120), (x, 120)],
            box=(x, 100, x + 20, 120),
            center=(x + 10, 110),
        )

    window._pcb_engine.update(
        [left_parent, right_parent, child(80), child(530), child(570)],
        image_size=(1000, 1000),
    )
    window._sync_region_observation_cards()

    assert window._step_cards[0].status_text() == "区域 #-1 检查中"
    assert window._step_cards[1].quantity_text() == "1 / 4"


def test_region_detail_uses_current_round_id_after_old_parent_failed(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 1
    config.round_cooldown_seconds = 0.0
    config.action_pass_stable_frames = 2

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))

    def parent(x):
        return ObbDetection(
            class_id=1,
            label="pcb",
            conf=0.9,
            track_id=None,
            polygon=[(x, 100), (x + 200, 100), (x + 200, 300), (x, 300)],
            box=(x, 100, x + 200, 300),
            center=(x + 100, 200),
        )

    def child(x):
        return ObbDetection(
            class_id=0,
            label="sanreidian",
            conf=0.9,
            track_id=None,
            polygon=[(x, 130), (x + 20, 130), (x + 20, 150), (x, 150)],
            box=(x, 130, x + 20, 150),
            center=(x + 10, 140),
        )

    failed = window._pcb_engine.update(
        [parent(100)] + [child(120 + index * 30) for index in range(3)],
        image_size=(1000, 1000),
    )
    assert len(failed) == 1 and failed[0].track_id == -1 and failed[0].result == PcbResult.FAIL

    assert window._pcb_engine.update(
        [parent(100), parent(500)] + [child(520 + index * 30) for index in range(4)],
        image_size=(1000, 1000),
    ) == []
    window._sync_region_observation_cards()

    assert window._step_cards[0].status_text() == "区域 #-2 检查中"
    assert window._step_cards[1].quantity_text() == "4 / 4"
    assert window._step_cards[1].state_name() == "active"


def test_region_overlay_records_synthetic_parent_track_id(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 3

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )
    parent_data = parent.to_dict()
    parent_data["center"] = parent_data["center_px"]

    window._on_overlay_updated({
        "source_frame_id": 1,
        "timestamp": 0.0,
        "model_path": "model.onnx",
        "task_type": "onnx_obb",
        "detections": [parent_data],
        "image_size": (1000, 1000),
    })

    assigned = window._latest_runtime_overlay.detections[0].track_id
    assert assigned is not None and assigned < 0
    assert window._last_overlay_data["detections"][0]["track_id"] == assigned


def test_region_quantity_ng_keeps_parent_overlay_red(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 1
    config.round_cooldown_seconds = 0.0

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )

    def children(count):
        return [
            ObbDetection(
                class_id=0,
                label="sanreidian",
                conf=0.9,
                track_id=None,
                polygon=[
                    (120 + index * 30, 130),
                    (140 + index * 30, 130),
                    (140 + index * 30, 150),
                    (120 + index * 30, 150),
                ],
                box=(120 + index * 30, 130, 140 + index * 30, 150),
                center=(130 + index * 30, 140),
            )
            for index in range(count)
        ]

    results = window._pcb_engine.update([parent] + children(3), image_size=(400, 400))
    assert len(results) == 1 and results[0].result == PcbResult.FAIL
    overlay = DetectionOverlayState(detections=window._pcb_engine.last_resolved_detections)
    rendered = window._draw_overlay(np.zeros((400, 400, 3), dtype=np.uint8), overlay)
    assert tuple(int(value) for value in rendered[100, 100]) == (0, 0, 255)

    second_parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=None,
        polygon=[(500, 100), (700, 100), (700, 300), (500, 300)],
        box=(500, 100, 700, 300),
        center=(600, 200),
    )
    second_children = [
        ObbDetection(
            class_id=0,
            label="sanreidian",
            conf=0.9,
            track_id=None,
            polygon=[
                (520 + index * 30, 130),
                (540 + index * 30, 130),
                (540 + index * 30, 150),
                (520 + index * 30, 150),
            ],
            box=(520 + index * 30, 130, 540 + index * 30, 150),
            center=(530 + index * 30, 140),
        )
        for index in range(4)
    ]
    window._pcb_engine.update(
        [parent, second_parent] + children(4) + second_children,
        image_size=(800, 400),
    )
    overlay = DetectionOverlayState(detections=window._pcb_engine.last_resolved_detections)
    rendered = window._draw_overlay(np.zeros((400, 800, 3), dtype=np.uint8), overlay)
    assert tuple(int(value) for value in rendered[100, 100]) == (0, 0, 255)
    assert tuple(int(value) for value in rendered[100, 500]) == (0, 200, 0)

    recovered = window._pcb_engine.update([parent] + children(4), image_size=(400, 400))
    assert len(recovered) == 1 and recovered[0].result == PcbResult.PASS
    overlay = DetectionOverlayState(detections=window._pcb_engine.last_resolved_detections)
    rendered = window._draw_overlay(np.zeros((400, 400, 3), dtype=np.uint8), overlay)
    assert tuple(int(value) for value in rendered[100, 100]) == (0, 200, 0)


def test_region_terminal_ng_updates_production_stats_once(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 1
    config.sound_feedback_enabled = False
    config.fail_evidence_enabled = False

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    detections = [
        ObbDetection(
            class_id=1,
            label="pcb",
            conf=0.9,
            track_id=None,
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
            box=(100, 100, 300, 300),
            center=(200, 200),
        )
    ] + [
        ObbDetection(
            class_id=0,
            label="sanreidian",
            conf=0.9,
            track_id=None,
            polygon=[
                (120 + index * 30, 130),
                (140 + index * 30, 130),
                (140 + index * 30, 150),
                (120 + index * 30, 150),
            ],
            box=(120 + index * 30, 130, 140 + index * 30, 150),
            center=(130 + index * 30, 140),
        )
        for index in range(3)
    ]
    detection_data = []
    for detection in detections:
        serialized = detection.to_dict()
        serialized["center"] = serialized["center_px"]
        detection_data.append(serialized)

    payload = {
        "source_frame_id": 1,
        "timestamp": 0.0,
        "model_path": "model.onnx",
        "task_type": "onnx_obb",
        "detections": detection_data,
        "image_size": (1000, 1000),
    }
    window._on_overlay_updated(payload)
    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ng == 1

    payload["source_frame_id"] = 2
    window._on_overlay_updated(payload)
    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ng == 1


def test_startup_step_engine_uses_configured_counts():
    from main import create_step_engine

    config = ConfigManager()
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]

    engine = create_step_engine(config)

    assert [(step.class_name, step.required_count) for step in engine.get_state().steps[:2]] == [
        ("pcb", 1),
        ("sanreidian", 4),
    ]


def test_region_result_updates_child_quantity_progress(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    result = PcbInspectionResult(
        track_id=1,
        result=PcbResult.FAIL,
        slot_states={"sanreidian": False},
        missing_classes=["sanreidian"],
        observed_counts={"sanreidian": 3},
        required_counts={"sanreidian": 4},
    )

    window._sync_region_step_cards(result)

    assert window._step_cards[0].status_text() == "区域 #1 子控件数量不符"
    assert window._step_cards[1].quantity_text() == "3 / 4"
    assert window._step_cards[1].state_name() == "ng"


def test_region_observation_shows_live_same_frame_quantity(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    config.action_ng_stable_frames = 3

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    parent = ObbDetection(
        class_id=1,
        label="pcb",
        conf=0.9,
        track_id=7,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        box=(100, 100, 300, 300),
        center=(200, 200),
    )
    children = [
        ObbDetection(
            class_id=0,
            label="sanreidian",
            conf=0.9,
            track_id=None,
            polygon=[(120 + i * 30, 130), (140 + i * 30, 130), (140 + i * 30, 150), (120 + i * 30, 150)],
            box=(120 + i * 30, 130, 140 + i * 30, 150),
            center=(130 + i * 30, 140),
        )
        for i in range(3)
    ]

    assert window._pcb_engine.update([parent] + children, image_size=(1000, 1000)) == []
    window._sync_region_observation_cards()

    assert window._step_cards[1].quantity_text() == "3 / 4"
    assert window._step_cards[1].state_name() == "active"



def test_round_cooldown_delays_next_round_start(qapp, tmp_path):
    config = _config(tmp_path)
    config.round_cooldown_seconds = 10.0

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    window.step_engine.start_round()
    original_round_id = window.step_engine.get_state().round_id

    window._start_next_round_after_pass()

    assert window.step_engine.get_state().round_id == original_round_id
    assert window._round_pass_timer.isActive()
    assert window._round_pass_timer.interval() == 10000

    window._round_pass_timer.stop()
    window._on_round_pass_settled()

    assert window.step_engine.get_state().round_id == original_round_id + 1


def test_runtime_recognition_list_holds_pass_and_quantity_during_cooldown(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["脚垫", "", "", "", "", ""]
    config.category_counts = [4, 1, 1, 1, 1, 1]
    config.round_cooldown_seconds = 10.0

    def factory(runtime_config, app_base_dir=None):
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(qapp, config, factory, processor=DummyProcessor("old"))
    window.step_engine = StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.category_counts,
        enter_stable_frames=1,
    )
    window._refresh_steps()
    window.step_engine.start_round()

    def detection():
        return ObbDetection(
            class_id=0,
            label="脚垫",
            conf=0.9,
            track_id=1,
            polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
            box=(10, 10, 30, 30),
            center=(20, 20),
        )

    partial_state = window.step_engine.update([detection() for _ in range(3)])
    window._update_step_display(partial_state)
    item = window._step_cards[0]
    assert item.isVisibleTo(window._steps_container)
    assert item.quantity_text() == "3 / 4"

    pass_state = window.step_engine.update([detection() for _ in range(4)])
    pass_round_id = pass_state.round_id
    window._update_step_display(pass_state)

    assert item.state_name() == "pass"
    assert item.quantity_text() == "4 / 4"
    assert window._round_pass_timer.isActive()
    assert window.step_engine.get_state().round_id == pass_round_id
    window._round_pass_timer.stop()
