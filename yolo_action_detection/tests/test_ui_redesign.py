from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication

from config import ConfigManager
from ui.config_page import ConfigPage
from ui.widgets.native_panels import RecognitionListItem


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
