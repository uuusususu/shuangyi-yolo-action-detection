"""ROI 配置页和矩形编辑器测试。"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from config import CURRENT_CONFIG_SCHEMA_VERSION, ConfigManager
from roi_inspection.geometry import NormalizedRect
from ui.config_page import ConfigPage
from ui.roi_editor import RoiEditorWidget, RoiKind


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _textured_frame(width=640, height=480, seed=42):
    """有结构的测试画面。

    形状匹配靠梯度方向工作，纯随机噪声不构成可匹配的轮廓（自匹配也只有约 0.7），
    因此这里画出有方向多样性的刚体图形，再叠加轻噪声。
    """
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
    noise = rng.normal(0, 3, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def _select_roi(page, kind):
    index = page._roi_kind_input.findData(kind.value)
    assert index >= 0
    page._roi_kind_input.setCurrentIndex(index)


def test_roi_page_disables_creation_without_camera_frame(qapp):
    page = ConfigPage(ConfigManager(), roi_frame_provider=lambda: None)

    page._refresh_roi_frame()

    assert page._roi_editor.has_frame is False
    assert page._create_roi_btn.isEnabled() is False
    assert "没有可用画面" in page._roi_frame_status.text()


def test_invalid_numeric_configuration_can_still_open_for_repair(qapp):
    config = ConfigManager()
    config.category_counts[0] = 1.5

    page = ConfigPage(config)

    assert page._step_count_inputs[0].value() == 1


def test_roi_page_refreshes_a_static_copy_and_enables_creation(qapp):
    source = np.full((480, 640, 3), 127, dtype=np.uint8)
    page = ConfigPage(ConfigManager(), roi_frame_provider=lambda: source)

    page._refresh_roi_frame()
    source[:] = 0

    assert page._roi_editor.has_frame is True
    assert page._roi_editor.frame_shape == (480, 640, 3)
    assert int(page._roi_editor.frame_copy()[0, 0, 0]) == 127
    assert page._create_roi_btn.isEnabled() is True


def test_roi_editor_creates_moves_resizes_deletes_and_replaces(qapp):
    editor = RoiEditorWidget()
    editor.set_frame(np.zeros((100, 200, 3), dtype=np.uint8))

    first = NormalizedRect(0.1, 0.1, 0.3, 0.3)
    replacement = NormalizedRect(0.2, 0.2, 0.4, 0.4)
    editor.set_rect(RoiKind.LOCATOR, first)
    editor.set_rect(RoiKind.LOCATOR, replacement)
    assert editor.roi_rect(RoiKind.LOCATOR) == replacement
    assert len(editor.rects) == 1

    editor.move_rect(RoiKind.LOCATOR, 0.1, -0.1)
    assert editor.roi_rect(RoiKind.LOCATOR) == NormalizedRect(0.3, 0.1, 0.4, 0.4)

    editor.resize_rect(RoiKind.LOCATOR, 0.5, 0.6)
    assert editor.roi_rect(RoiKind.LOCATOR) == NormalizedRect(0.3, 0.1, 0.5, 0.6)

    editor.delete_rect(RoiKind.LOCATOR)
    assert editor.roi_rect(RoiKind.LOCATOR) is None

    editor.set_rect(RoiKind.INSPECTION, first)
    editor.set_rect(RoiKind.SEARCH, replacement)
    assert set(editor.rects) == {RoiKind.INSPECTION, RoiKind.SEARCH}


def test_roi_editor_can_render_without_overriding_qwidget_rect(qapp):
    editor = RoiEditorWidget()
    editor.resize(640, 480)
    editor.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    editor.set_rect(RoiKind.LOCATOR, NormalizedRect(0.1, 0.1, 0.4, 0.4))
    editor.show()
    qapp.processEvents()

    image = editor.grab().toImage()

    assert image.isNull() is False
    assert image.width() == 640
    assert image.height() == 480
    editor.close()


def test_roi_editor_edge_click_creates_an_in_bounds_minimum_rect(qapp):
    editor = RoiEditorWidget()
    editor.resize(640, 480)
    editor.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    editor.show()
    qapp.processEvents()

    QTest.mouseClick(
        editor,
        Qt.MouseButton.LeftButton,
        pos=QPoint(editor.width() - 1, editor.height() - 1),
    )

    rect = editor.roi_rect(RoiKind.LOCATOR)
    assert rect is not None
    assert rect.x + rect.width <= 1.0
    assert rect.y + rect.height <= 1.0
    editor.close()


def test_roi_editor_has_one_save_action_and_no_separate_template_controls(qapp):
    config = ConfigManager()
    page = ConfigPage(config)

    assert hasattr(page, "_roi_template_cb")
    assert hasattr(page, "_template_threshold_input")
    assert hasattr(page, "_template_angle_input")
    assert hasattr(page, "_template_angle_step_input")
    assert hasattr(page, "_roi_kind_input")
    assert hasattr(page, "_create_roi_btn")
    assert hasattr(page, "_delete_roi_btn")
    assert hasattr(page, "_save_roi_btn")
    assert not hasattr(page, "_create_template_btn")
    assert not hasattr(page, "_save_template_btn")
    assert not hasattr(page, "_template_preview")
    assert page._save_roi_btn.text() == "保存当前区域"
    assert page._roi_template_cb.isEnabled() is False
    assert page._save_roi_btn.isEnabled() is False


def test_template_parameters_enabled_when_roi_and_template_on(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    page = ConfigPage(config)

    assert page._roi_template_cb.isEnabled() is True
    assert page._save_roi_btn.isEnabled() is False
    assert page._template_threshold_input.isEnabled() is True
    assert page._template_angle_input.isEnabled() is True
    assert page._template_angle_step_input.isEnabled() is True


def test_template_threshold_defaults(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = True
    config.roi_template_enabled = True
    page = ConfigPage(config)

    assert page._template_threshold_input.value() == pytest.approx(0.7)
    assert page._template_angle_input.value() == pytest.approx(15.0)
    # 0 表示按模板尺寸自动推算角度步进
    assert page._template_angle_step_input.value() == pytest.approx(0.0)


def test_roi_type_dropdown_creates_and_deletes_selected_kind(qapp):
    page = ConfigPage(ConfigManager(), roi_frame_provider=lambda: _textured_frame())
    page._refresh_roi_frame()

    labels = [page._roi_kind_input.itemText(index) for index in range(page._roi_kind_input.count())]
    assert labels == ["识别区域", "定位模板", "搜索区域"]

    search_index = page._roi_kind_input.findData(RoiKind.SEARCH.value)
    page._roi_kind_input.setCurrentIndex(search_index)
    page._create_roi_btn.click()
    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is not None
    assert page._roi_editor.roi_rect(RoiKind.LOCATOR) is None
    assert page._save_roi_btn.isEnabled() is True

    page._delete_roi_btn.click()
    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is None


def test_fresh_roi_page_has_no_boxes_until_create(qapp):
    page = ConfigPage(ConfigManager(), roi_frame_provider=lambda: _textured_frame())
    page._refresh_roi_frame()

    assert page._roi_editor.roi_rect(RoiKind.LOCATOR) is None
    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is None
    assert page._roi_editor.roi_rect(RoiKind.INSPECTION) is None


def test_create_roi_does_not_persist_until_save(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.SEARCH)
    page._create_roi_btn.click()

    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is not None
    assert config.roi_template_search_rect is None
    assert (tmp_path / "config.json").exists() is False


def test_template_roi_saves_without_search_roi(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.LOCATOR)
    page._roi_editor.set_rect(RoiKind.LOCATOR, NormalizedRect(0.15, 0.20, 0.20, 0.20))
    page._save_roi_btn.click()

    assert warnings == []
    assert (tmp_path / "locator_template.png").is_file()
    assert (tmp_path / "locator_model.npz").is_file()
    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_template_rect"] == {
        "x": 0.15,
        "y": 0.20,
        "width": 0.20,
        "height": 0.20,
    }
    assert saved["roi_template_search_rect"] is None
    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is None


def test_template_roi_save_ignores_small_search_draft(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.LOCATOR)
    page._roi_editor.set_rect(RoiKind.LOCATOR, NormalizedRect(0.15, 0.20, 0.25, 0.25))
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.10, 0.15, 0.28, 0.28))
    _select_roi(page, RoiKind.LOCATOR)
    page._save_roi_btn.click()

    assert warnings == []
    assert (tmp_path / "locator_template.png").is_file()
    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_template_search_rect"] is None


def test_template_roi_saves_with_adequate_search_roi_and_shows_feedback(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "中文配置" / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    page._refresh_roi_frame()

    _select_roi(page, RoiKind.LOCATOR)
    page._create_roi_btn.click()
    page._roi_editor.set_rect(
        RoiKind.LOCATOR,
        NormalizedRect(0.15, 0.20, 0.20, 0.20),
    )
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.05, 0.05, 0.60, 0.70))
    _select_roi(page, RoiKind.LOCATOR)

    template_path = tmp_path / "中文配置" / "locator_template.png"
    model_path = tmp_path / "中文配置" / "locator_model.npz"
    assert template_path.exists() is False
    assert page._roi_editor.roi_rect(RoiKind.INSPECTION) is None
    assert page._save_roi_btn.isEnabled() is True

    page._save_roi_btn.click()

    assert warnings == []
    assert template_path.is_file()
    assert model_path.is_file()
    assert page._roi_editor.roi_rect(RoiKind.LOCATOR) == NormalizedRect(
        0.15,
        0.20,
        0.20,
        0.20,
    )
    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_inspection_enabled"] is False
    assert saved["roi_template_enabled"] is False
    assert saved["roi_template_path"] == "locator_template.png"
    assert saved["roi_template_model_path"] == "locator_model.npz"
    assert saved["roi_template_rect"] == {
        "x": 0.15,
        "y": 0.20,
        "width": 0.20,
        "height": 0.20,
    }
    assert saved["roi_template_search_rect"] is None
    assert saved["roi_inspection_rect"] is None
    feedback = page._roi_editor.template_feedback
    assert feedback is not None
    assert feedback["success"] is True
    assert feedback["score"] >= 0.95
    assert feedback["box"] and len(feedback["box"]) == 4


def test_invalid_replacement_keeps_previous_template_and_saved_rect(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.LOCATOR)
    first_rect = NormalizedRect(0.10, 0.10, 0.20, 0.20)
    page._roi_editor.set_rect(RoiKind.LOCATOR, first_rect)
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.02, 0.02, 0.70, 0.80))
    _select_roi(page, RoiKind.LOCATOR)
    page._save_roi_btn.click()
    template_path = tmp_path / "locator_template.png"
    original_bytes = template_path.read_bytes()

    page._roi_editor.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    page._roi_editor.set_rect(
        RoiKind.LOCATOR,
        NormalizedRect(0.30, 0.20, 0.20, 0.20),
    )
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.02, 0.02, 0.70, 0.80))
    _select_roi(page, RoiKind.LOCATOR)
    page._save_roi_btn.click()

    assert warnings
    assert "轮廓" in warnings[-1]
    assert template_path.read_bytes() == original_bytes
    assert config.roi_template_rect == first_rect.to_mapping()
    feedback = page._roi_editor.template_feedback
    assert feedback is not None and feedback["success"] is False


def test_template_roi_save_allows_disjoint_search_and_inspection_rois(
    qapp,
    tmp_path,
    monkeypatch,
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.LOCATOR)
    page._roi_editor.set_rect(RoiKind.LOCATOR, NormalizedRect(0.1, 0.1, 0.2, 0.2))
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.40, 0.40, 0.55, 0.55))
    page._roi_editor.set_rect(RoiKind.INSPECTION, NormalizedRect(0.6, 0.0, 0.3, 0.5))
    _select_roi(page, RoiKind.LOCATOR)

    page._save_roi_btn.click()

    assert warnings == []
    assert (tmp_path / "locator_template.png").is_file()
    feedback = page._roi_editor.template_feedback
    assert feedback is not None and feedback["success"] is True


def test_save_current_roi_only_persists_selected_kind(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    old_template = NormalizedRect(0.05, 0.05, 0.2, 0.2)
    old_search = NormalizedRect(0.1, 0.1, 0.3, 0.3)
    old_inspection = NormalizedRect(0.5, 0.1, 0.4, 0.8)
    config.roi_template_rect = old_template.to_mapping()
    config.roi_template_search_rect = old_search.to_mapping()
    config.roi_inspection_rect = old_inspection.to_mapping()
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    page._refresh_roi_frame()

    new_template = NormalizedRect(0.2, 0.2, 0.25, 0.25)
    new_search = NormalizedRect(0.65, 0.05, 0.25, 0.4)
    new_inspection = NormalizedRect(0.05, 0.55, 0.45, 0.35)
    page._roi_editor.set_rect(RoiKind.LOCATOR, new_template)
    page._roi_editor.set_rect(RoiKind.SEARCH, new_search)
    page._roi_editor.set_rect(RoiKind.INSPECTION, new_inspection)

    _select_roi(page, RoiKind.SEARCH)
    page._save_roi_btn.click()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_template_search_rect"] == new_search.to_mapping()
    assert saved["roi_template_rect"] == old_template.to_mapping()
    assert saved["roi_inspection_rect"] == old_inspection.to_mapping()

    _select_roi(page, RoiKind.INSPECTION)
    page._save_roi_btn.click()
    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_template_search_rect"] == new_search.to_mapping()
    assert saved["roi_template_rect"] == old_template.to_mapping()
    assert saved["roi_inspection_rect"] == new_inspection.to_mapping()


def test_delete_search_roi_clears_saved_rect_only(qapp, tmp_path):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3}
    config.roi_inspection_rect = {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    page = ConfigPage(config, roi_frame_provider=lambda: _textured_frame())
    page._refresh_roi_frame()
    _select_roi(page, RoiKind.SEARCH)

    page._delete_roi_btn.click()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert config.roi_template_search_rect is None
    assert saved["roi_template_search_rect"] is None
    assert saved["roi_inspection_rect"] == {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    assert saved["roi_template_rect"] == {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    assert page._roi_editor.roi_rect(RoiKind.SEARCH) is None
    assert "已删除" in page._roi_frame_status.text()


def test_roi_page_saves_new_schema_and_template_config(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.category_names = ["散热片", "", "", "", "", ""]
    config.category_counts = [6, 1, 1, 1, 1, 1]
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    page = ConfigPage(
        config,
        roi_frame_provider=lambda: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    page._refresh_roi_frame()
    page._roi_enabled_cb.setChecked(True)
    page._roi_template_cb.setChecked(True)
    page._template_threshold_input.setValue(0.5)
    page._template_angle_input.setValue(20.0)
    page._template_angle_step_input.setValue(2.0)
    page._roi_editor.set_rect(RoiKind.INSPECTION, NormalizedRect(0.5, 0.0, 0.5, 1.0))
    page._roi_editor.set_rect(RoiKind.SEARCH, NormalizedRect(0.1, 0.1, 0.5, 0.5))
    _select_roi(page, RoiKind.INSPECTION)
    page._save_roi_btn.click()
    _select_roi(page, RoiKind.SEARCH)
    page._save_roi_btn.click()

    emitted = []
    page.config_saved.connect(lambda: emitted.append(True))
    page._action_order_cb.setChecked(False)
    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert emitted == [True]
    assert saved["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    assert saved["roi_inspection_enabled"] is True
    assert saved["roi_template_enabled"] is True
    assert saved["roi_template_match_threshold"] == 0.5
    assert saved["roi_template_angle_range"] == 20.0
    assert saved["roi_template_angle_step"] == 2.0
    assert "roi_template_min_matches" not in saved
    assert saved["roi_template_search_rect"] == {
        "x": 0.1,
        "y": 0.1,
        "width": 0.5,
        "height": 0.5,
    }
    assert saved["roi_inspection_rect"] == {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0}
    assert "roi_locator_class_name" not in saved
    assert "roi_locator_target_count" not in saved
    assert "roi_locator_rect" not in saved


def test_roi_page_can_save_draft_before_rects_are_drawn(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.category_names = ["散热片", "", "", "", "", ""]
    config.category_counts = [6, 1, 1, 1, 1, 1]
    page = ConfigPage(config)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    page._roi_enabled_cb.setChecked(True)
    page._roi_template_cb.setChecked(False)
    page._template_threshold_input.setValue(0.6)
    page._template_angle_input.setValue(15.0)
    page._template_angle_step_input.setValue(1.0)

    emitted = []
    page.config_saved.connect(lambda: emitted.append(True))
    page._action_order_cb.setChecked(False)
    page._on_save()

    assert warnings == []
    assert emitted == [True]
    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["roi_inspection_enabled"] is True
    assert saved["roi_template_enabled"] is False
    assert saved["roi_inspection_rect"] is None
    assert saved["roi_template_rect"] is None
    assert saved["roi_template_search_rect"] is None
