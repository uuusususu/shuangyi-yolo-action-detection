from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.config_page import ConfigPage
from ui.main_window import MainWindow
from yolo_runtime.yolo_result_models import ObbDetection


def _structured_frame(width=640, height=480, seed=7):
    "有结构的测试画面：形状匹配需要方向多样的轮廓，纯噪声不可匹配。"
    import cv2

    rng = np.random.RandomState(seed)
    frame = np.full((height, width, 3), 40, np.uint8)
    for row in range(0, height, 120):
        for col in range(0, width, 160):
            cv2.rectangle(frame, (col + 18, row + 14), (col + 140, row + 104),
                          (215, 215, 215), 2)
            cv2.line(frame, (col + 30, row + 24), (col + 30, row + 94), (235, 235, 235), 3)
            cv2.circle(frame, (col + 108, row + 46), 13, (120, 120, 120), 2)
            cv2.line(frame, (col + 40, row + 92), (col + 120, row + 30), (180, 180, 180), 2)
    return np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)


class DummyProcessor:
    def __init__(self, name: str, class_names: dict[int, str] | None = None) -> None:
        self.name = name
        self._class_names = class_names or {0: "模型类别A", 1: "模型类别B"}

    def get_class_names(self) -> dict[int, str]:
        return dict(self._class_names)


class SoundSpy:
    def __init__(self) -> None:
        self.enabled = True
        self.pass_calls = 0
        self.fail_calls = 0

    def play_pass(self) -> None:
        self.pass_calls += 1

    def play_fail(self) -> None:
        self.fail_calls += 1


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _config(tmp_path: Path) -> ConfigManager:
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.yolo_model_path = "old.onnx"
    config.yolo_conf_threshold = 0.3
    config.category_names = ["手动1", "手动2", "", "", "", ""]
    return config


def _window(
    config: ConfigManager,
    processor_factory,
    processor=None,
    sound_feedback=None,
) -> MainWindow:
    window = MainWindow(
        config,
        AppState(),
        processor=processor,
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.category_counts,
        ),
        processor_factory=processor_factory,
        app_base_dir=Path(config.get_config_path()).parent,
        sound_feedback=sound_feedback,
    )
    if processor is not None:
        window.worker.set_frame_processor(processor)
    return window


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


def test_config_page_save_persists_manual_categories_without_model_autofill(qapp, tmp_path, monkeypatch):
    config = _config(tmp_path)
    page = ConfigPage(config)
    emitted = []
    page.config_saved.connect(lambda: emitted.append(True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    page._model_path_input.setText("new.onnx")
    page._step_inputs[0].setText("人工步骤A")
    page._step_inputs[1].setText("人工步骤B")

    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert emitted == [True]
    assert saved["yolo_model_path"] == "new.onnx"
    assert saved["category_names"][:2] == ["人工步骤A", "人工步骤B"]


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


def test_runtime_uses_independent_result_sound_switches(qapp, tmp_path):
    config = _config(tmp_path)
    config.pass_sound_enabled = False
    config.fail_sound_enabled = False
    config.fail_evidence_enabled = False
    sound = SoundSpy()
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"), sound)

    window._play_pass_feedback()
    window._handle_action_ng_feedback(SimpleNamespace(round_id=1, action_ng_step=-1))
    assert sound.pass_calls == 0
    assert sound.fail_calls == 0

    config.pass_sound_enabled = True
    config.fail_sound_enabled = True
    window._play_pass_feedback()
    window._handle_action_ng_feedback(SimpleNamespace(round_id=2, action_ng_step=-1))
    assert sound.pass_calls == 1
    assert sound.fail_calls == 1


def test_normal_mode_result_sound_plays_once_per_round(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["A", "B", "", "", "", ""]
    config.pass_sound_enabled = True
    config.round_cooldown_seconds = 60.0
    sound = SoundSpy()
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"), sound)
    window.step_engine = StepSequenceEngine(
        step_class_names=config.category_names,
        enter_stable_frames=1,
    )
    window._refresh_steps()
    window.step_engine.start_round(require_first_step_rearm=False)
    window.step_engine.update([_detection("A")])
    state = window.step_engine.update([_detection("B")])

    window._update_step_display(state)
    window._update_step_display(state)

    assert sound.pass_calls == 1
    window._round_pass_timer.stop()


def test_config_save_reloads_processor_and_updates_worker(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")
    created = []

    def factory(runtime_config, app_base_dir=None):
        created.append((runtime_config.yolo_model_path, Path(app_base_dir)))
        return DummyProcessor(runtime_config.yolo_model_path)

    window = _window(config, factory, old_processor)
    window._set_notice("处理器未加载，请复制日志文件：runtime.log")
    config.yolo_model_path = "new.onnx"
    window._on_config_saved()

    assert created == [("new.onnx", tmp_path)]
    assert window.processor is window.worker.frame_processor
    assert window.processor is not old_processor
    assert window._notice_label.text() == ""


def test_template_config_change_reloads_processor_and_updates_worker(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")
    created = []

    def factory(runtime_config, app_base_dir=None):
        created.append(
            (
                bool(runtime_config.roi_template_enabled),
                runtime_config.roi_template_path,
                Path(app_base_dir),
            )
        )
        return DummyProcessor("template-enabled")

    window = _window(config, factory, old_processor)
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_template_search_rect = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    config.roi_template_search_rect = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}

    window._on_config_saved()

    assert created == [(True, "locator_template.png", tmp_path)]
    assert window.processor is window.worker.frame_processor
    assert window.processor.name == "template-enabled"


def test_template_path_is_relative_to_actual_config_file(tmp_path):
    from main import resolve_template_path

    source_config = ConfigManager()
    source_config._config_path = str(tmp_path / "source" / "config" / "config.json")
    source_config.roi_template_path = "locator_template.png"
    portable_config = ConfigManager()
    portable_config._config_path = str(tmp_path / "portable" / "config.json")
    portable_config.roi_template_path = "locator_template.png"

    assert resolve_template_path(source_config, tmp_path / "source") == (
        tmp_path / "source" / "config" / "locator_template.png"
    )
    assert resolve_template_path(portable_config, tmp_path / "portable") == (
        tmp_path / "portable" / "locator_template.png"
    )


def test_create_processor_fails_closed_when_enabled_template_is_missing(tmp_path, monkeypatch):
    import main as main_module

    class FakeOnnxProcessor(DummyProcessor):
        def __init__(self, **_kwargs):
            super().__init__("onnx", {0: "散热片"})

        def load(self):
            return None

    config = _config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_template_search_rect = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    monkeypatch.setattr(main_module, "OnnxObbProcessor", FakeOnnxProcessor)

    with pytest.raises(FileNotFoundError, match="定位模板"):
        main_module.create_yolo_processor(config, app_base_dir=tmp_path)


def test_create_processor_wraps_yolo_when_template_is_valid(tmp_path, monkeypatch):
    import main as main_module
    from yolo_runtime.template_gated_processor import TemplateGatedProcessor

    class FakeOnnxProcessor(DummyProcessor):
        def __init__(self, **_kwargs):
            super().__init__("onnx", {0: "散热片"})

        def load(self):
            return None

    config = _config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_template_search_rect = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    rng = np.random.RandomState(42)
    template = rng.randint(0, 256, size=(160, 220, 3), dtype=np.uint8)
    cv2.rectangle(template, (20, 20), (120, 100), (255, 255, 255), 3)
    cv2.circle(template, (170, 90), 35, (0, 0, 0), 4)
    ok, encoded = cv2.imencode(".png", template)
    assert ok is True
    (tmp_path / "locator_template.png").write_bytes(encoded.tobytes())
    monkeypatch.setattr(main_module, "OnnxObbProcessor", FakeOnnxProcessor)

    processor = main_module.create_yolo_processor(config, app_base_dir=tmp_path)

    assert isinstance(processor, TemplateGatedProcessor)
    assert processor.get_class_names() == {0: "散热片"}


def test_saving_template_roi_does_not_reload_unrelated_runtime_processor(
    qapp,
    tmp_path,
    monkeypatch,
):
    from roi_inspection.geometry import NormalizedRect
    from ui.roi_editor import RoiKind

    config = _config(tmp_path)
    old_processor = DummyProcessor("old")
    created = []

    def factory(runtime_config, app_base_dir=None):
        created.append(
            (
                bool(runtime_config.roi_template_enabled),
                Path(app_base_dir) / runtime_config.roi_template_path,
            )
        )
        return DummyProcessor("template-runtime", {0: "手动1", 1: "手动2"})

    window = _window(config, factory, old_processor)
    page = ConfigPage(
        config,
        roi_frame_provider=lambda: _structured_frame(),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    page._refresh_roi_frame()
    template_index = page._roi_kind_input.findData(RoiKind.LOCATOR.value)
    page._roi_kind_input.setCurrentIndex(template_index)
    page._roi_editor.set_rect(
        RoiKind.LOCATOR,
        NormalizedRect(0.1, 0.1, 0.2, 0.2),
    )
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.02, 0.02, 0.75, 0.85))
    page._roi_kind_input.setCurrentIndex(template_index)

    page._save_roi_btn.click()

    assert created == []
    assert window.processor is old_processor
    assert window.worker.frame_processor is window.processor
    assert config.roi_inspection_enabled is False
    assert config.roi_template_enabled is False
    assert config.roi_template_path == "locator_template.png"
    assert (tmp_path / "locator_template.png").is_file()


def test_template_roi_save_is_not_blocked_by_runtime_reload_failure(
    qapp,
    tmp_path,
    monkeypatch,
):
    from roi_inspection.geometry import NormalizedRect
    from ui.roi_editor import RoiKind

    config = _config(tmp_path)
    old_processor = DummyProcessor("old")

    def failing_factory(*_args, **_kwargs):
        raise RuntimeError("processor failed")

    window = _window(config, failing_factory, old_processor)
    page = ConfigPage(
        config,
        roi_frame_provider=lambda: _structured_frame(),
    )
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    page._refresh_roi_frame()
    template_index = page._roi_kind_input.findData(RoiKind.LOCATOR.value)
    page._roi_kind_input.setCurrentIndex(template_index)
    page._roi_editor.set_rect(
        RoiKind.LOCATOR,
        NormalizedRect(0.2, 0.2, 0.20, 0.20),
    )
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.02, 0.02, 0.80, 0.85))
    page._roi_kind_input.setCurrentIndex(template_index)

    page._save_roi_btn.click()

    assert warnings == []
    assert (tmp_path / "locator_template.png").is_file()
    assert config.roi_inspection_enabled is False
    assert config.roi_template_enabled is False
    assert window.processor is old_processor
    assert window.worker.frame_processor is old_processor


def test_start_detection_rejects_template_mode_without_gated_processor(qapp, tmp_path):
    config = _config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    config.roi_template_path = "locator_template.png"
    config.roi_inspection_rect = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    (tmp_path / "locator_template.png").write_bytes(b"template")
    base_processor = DummyProcessor("base", {0: "手动1", 1: "手动2"})
    window = _window(config, lambda *_args, **_kwargs: base_processor, base_processor)
    window.state.set_camera_on(True)

    window._on_toggle_detect()

    assert window.state.inference_on is False
    assert "模板定位处理器未加载" in window._notice_label.text()


def test_config_page_saves_manual_classes_then_reloads_the_selected_model(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    config.roi_inspection_enabled = True
    config.roi_template_enabled = False
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    config.category_names = ["散热片", "", "", "", "", ""]
    config.category_counts = [6, 1, 1, 1, 1, 1]
    old_processor = DummyProcessor("old", {0: "旧模型类别"})
    created = []

    def factory(runtime_config, app_base_dir=None):
        created.append(runtime_config.yolo_model_path)
        return DummyProcessor(
            runtime_config.yolo_model_path,
            {0: "散热片"},
        )

    window = _window(config, factory, old_processor)
    window._camera_devices = []
    window._camera_enumeration_error = ""
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    window._on_config_clicked()
    page = window._config_page
    page._model_path_input.setText("new-model.onnx")
    page._step_inputs[0].setText("散热片")

    page._on_save()

    assert warnings == []
    assert created == ["new-model.onnx"]
    assert window.processor.get_class_names() == {0: "散热片"}
    assert window.worker.frame_processor is window.processor


def test_failed_model_reload_preserves_previous_processor(qapp, tmp_path):
    config = _config(tmp_path)
    old_processor = DummyProcessor("old")

    def factory(*_args, **_kwargs):
        raise RuntimeError("bad model")

    window = _window(config, factory, old_processor)
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
    window = _window(
        config,
        lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path, {0: "模型自动A"}),
        DummyProcessor("old"),
    )
    config.yolo_model_path = "new.onnx"
    config.category_names = ["人工A", "人工B", "", "", "", ""]
    window._on_config_saved()

    assert [step.class_name for step in window.step_engine.get_state().steps[:2]] == ["人工A", "人工B"]


def test_runtime_header_uses_centered_brand_and_compact_actions(qapp, tmp_path):
    config = _config(tmp_path)
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"))
    window.resize(1200, 760)
    window.show()
    qapp.processEvents()

    assert window._title_label.text() == "双翼科技AI系统"
    assert window._title_label.alignment() == Qt.AlignmentFlag.AlignCenter
    assert abs(window._title_label.geometry().center().x() - window._header.rect().center().x()) <= 2
    assert window._btn_config.height() == 40
    window.close()


def test_configured_steps_are_hidden_until_detected(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    config.category_counts = [1, 4, 1, 1, 1, 1]
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"))

    assert [item._name_label.text() for item in window._step_cards] == ["pcb", "sanreidian"]
    assert all(not item.isVisibleTo(window._steps_container) for item in window._step_cards)
    assert [item.quantity_text() for item in window._step_cards] == ["0 / 1", "0 / 4"]


def test_steps_are_revealed_in_detection_order(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["pcb", "sanreidian", "", "", "", ""]
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"))
    window.step_engine.start_round()

    state = window.step_engine.update([_detection("pcb")])
    window._update_step_display(state)

    assert window._step_cards[0].isHidden() is False
    assert window._step_cards[1].isHidden() is True


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


def test_round_cooldown_delays_next_round_start(qapp, tmp_path):
    config = _config(tmp_path)
    config.round_cooldown_seconds = 10.0
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"))
    window.step_engine.start_round()
    original_round_id = window.step_engine.get_state().round_id

    window._start_next_round_after_pass()

    assert window.step_engine.get_state().round_id == original_round_id
    assert window._round_pass_timer.interval() == 10000
    window._round_pass_timer.stop()
    window._on_round_pass_settled()
    assert window.step_engine.get_state().round_id == original_round_id + 1


def test_runtime_recognition_list_holds_pass_and_quantity_during_cooldown(qapp, tmp_path):
    config = _config(tmp_path)
    config.category_names = ["脚垫", "", "", "", "", ""]
    config.category_counts = [4, 1, 1, 1, 1, 1]
    config.round_cooldown_seconds = 10.0
    window = _window(config, lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path), DummyProcessor("old"))
    window.step_engine = StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.category_counts,
        enter_stable_frames=1,
    )
    window._refresh_steps()
    window.step_engine.start_round()

    partial_state = window.step_engine.update([_detection("脚垫") for _ in range(3)])
    window._update_step_display(partial_state)
    item = window._step_cards[0]
    assert item.quantity_text() == "3 / 4"

    pass_state = window.step_engine.update([_detection("脚垫") for _ in range(4)])
    window._update_step_display(pass_state)
    assert item.state_name() == "pass"
    assert item.quantity_text() == "4 / 4"
    assert window.step_engine.get_state().round_id == pass_state.round_id
    window._round_pass_timer.stop()
