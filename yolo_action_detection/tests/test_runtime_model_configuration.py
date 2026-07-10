from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QMessageBox

from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.config_page import ConfigPage
from ui.main_window import MainWindow


class DummyProcessor:
    def __init__(self, name: str, class_names: dict[int, str] | None = None) -> None:
        self.name = name
        self._class_names = class_names or {0: "模型类别A", 1: "模型类别B"}

    def get_class_names(self) -> dict[int, str]:
        return dict(self._class_names)


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


def _window(qapp, config: ConfigManager, processor_factory, processor=None) -> MainWindow:
    state = AppState()
    step_engine = StepSequenceEngine(step_class_names=config.category_names)
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=step_engine,
        processor_factory=processor_factory,
        app_base_dir=Path(config.get_config_path()).parent,
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
