"""ONNX 会话参数的硬编码默认值与生效。

推理线程数 / 图优化级别已从配置页删除（提案 02 节删除项 3–4），运行时固定为
intra_op_threads=0（交给 ONNX Runtime）、graph_optimization="all"。
这里验证处理器默认值、SessionOptions 翻译，以及换模型类型时跟踪器开关的置灰。
"""
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
from yolo_runtime.onnx_obb_processor import OnnxObbProcessor


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_removed_session_fields_no_longer_exist_on_config():
    config = ConfigManager()

    assert not hasattr(config, "onnx_intra_op_threads")
    assert not hasattr(config, "onnx_graph_optimization")


def test_config_page_has_no_session_option_inputs(qapp):
    config = ConfigManager()
    config.yolo_model_path = "site.onnx"
    page = ConfigPage(config)

    assert not hasattr(page, "_onnx_threads_input")
    assert not hasattr(page, "_onnx_graph_opt_input")
    assert not hasattr(page, "_device_input")
    assert not hasattr(page, "_max_det_input")


def test_ultralytics_only_fields_are_disabled_for_onnx_models(qapp):
    config = ConfigManager()
    config.yolo_model_path = "site.onnx"
    page = ConfigPage(config)

    assert page._tracker_input.isEnabled() is False
    assert page._track_persist_cb.isEnabled() is False
    assert page._show_tid_cb.isEnabled() is False


def test_switching_to_pt_model_re_enables_ultralytics_fields(qapp):
    config = ConfigManager()
    config.yolo_model_path = "site.onnx"
    page = ConfigPage(config)

    page._model_path_input.setText("config/best.pt")

    assert page._tracker_input.isEnabled() is True
    assert page._track_persist_cb.isEnabled() is True
    assert page._show_tid_cb.isEnabled() is True


def test_processor_reports_configured_session_options():
    processor = OnnxObbProcessor(
        model_path="unused.onnx",
        intra_op_threads=3,
        graph_optimization="basic",
    )

    assert processor.describe_session_options() == {
        "intra_op_threads": 3,
        "graph_optimization": "basic",
        "providers": ["CPUExecutionProvider"],
    }


@pytest.mark.parametrize(
    "level,expected_attribute",
    [
        ("disabled", "ORT_DISABLE_ALL"),
        ("basic", "ORT_ENABLE_BASIC"),
        ("extended", "ORT_ENABLE_EXTENDED"),
        ("all", "ORT_ENABLE_ALL"),
        ("nonsense", "ORT_ENABLE_ALL"),
    ],
)
def test_session_options_translate_to_onnxruntime_levels(level, expected_attribute):
    ort = pytest.importorskip("onnxruntime")
    processor = OnnxObbProcessor(
        model_path="unused.onnx", intra_op_threads=4, graph_optimization=level
    )

    options = processor._build_session_options(ort)

    assert options.intra_op_num_threads == 4
    assert options.graph_optimization_level == getattr(
        ort.GraphOptimizationLevel, expected_attribute
    )


def test_automatic_thread_count_leaves_onnxruntime_default():
    ort = pytest.importorskip("onnxruntime")
    processor = OnnxObbProcessor(model_path="unused.onnx", intra_op_threads=0)

    options = processor._build_session_options(ort)

    assert options.intra_op_num_threads == ort.SessionOptions().intra_op_num_threads
