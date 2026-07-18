from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.config_page import ConfigPage
from ui.main_window import MainWindow


class DummyProcessor:
    def get_class_names(self) -> dict[int, str]:
        return {0: "父类", 1: "配件"}


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def themed_app(qapp):
    from ui.theme import apply_application_theme

    previous_stylesheet = qapp.styleSheet()
    apply_application_theme(qapp)
    try:
        yield qapp
    finally:
        qapp.setStyleSheet(previous_stylesheet)


def _window(config: ConfigManager) -> MainWindow:
    return MainWindow(
        config,
        AppState(),
        processor=DummyProcessor(),
        step_engine=StepSequenceEngine(
            step_class_names=config.category_names,
            step_counts=config.category_counts,
        ),
        processor_factory=lambda *args, **kwargs: DummyProcessor(),
    )


def test_theme_initialization_uses_dark_mode_and_project_accent(qapp, monkeypatch):
    import ui.theme as theme

    calls = []

    class ThemeSpy:
        @staticmethod
        def setup_theme(**kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(theme, "import_module", lambda name: ThemeSpy)

    theme.apply_application_theme(qapp)

    assert calls == [
        {
            "theme": "dark",
            "corner_shape": "rounded",
            "custom_colors": {"primary": "#35D7FF"},
            "additional_qss": theme.SEMANTIC_CONTROL_QSS,
        }
    ]
    assert qapp.property("openSourceControlTheme") == "pyqtdarktheme-2.1.0"


def test_theme_dependency_failure_is_explicit(qapp, monkeypatch):
    import ui.theme as theme

    def missing_dependency(_name):
        raise ModuleNotFoundError("No module named 'qdarktheme'")

    monkeypatch.setattr(theme, "import_module", missing_dependency)

    with pytest.raises(RuntimeError, match="pyqtdarktheme==2.1.0"):
        theme.apply_application_theme(qapp)


def test_open_source_theme_defines_complete_standard_control_affordances(themed_app):
    stylesheet = themed_app.styleSheet()

    assert re.search(r"QPushButton\s*\{[^}]*border\s*:\s*1px solid", stylesheet)
    assert "QPushButton:hover" in stylesheet
    assert "QPushButton:pressed" in stylesheet
    assert "QPushButton:focus" in stylesheet
    assert "QPushButton:disabled" in stylesheet
    assert "QComboBox::down-arrow" in stylesheet
    assert "QAbstractSpinBox::up-arrow" in stylesheet
    assert "QAbstractSpinBox::down-arrow" in stylesheet
    assert "QCheckBox::indicator" in stylesheet


def test_config_page_uses_theme_controls_without_local_border_overrides(themed_app):
    page = ConfigPage(ConfigManager())

    buttons = [
        page._sidebar_back_btn,
        page._select_model_btn,
        page._camera_refresh_btn,
        page._back_btn,
        page._save_btn,
        page._reset_btn,
        *page._nav_buttons,
    ]
    assert all(isinstance(button, QPushButton) for button in buttons)
    assert all("border" not in button.styleSheet().lower() for button in buttons)
    assert "QLineEdit" not in page.styleSheet()
    assert "QComboBox" not in page.styleSheet()
    assert "QSpinBox" not in page.styleSheet()

    assert page._save_btn.property("buttonRole") == "primary"
    assert page._save_btn.isDefault() is True
    assert page._reset_btn.property("buttonRole") == "danger"
    assert page._reset_btn.isDefault() is False
    assert all(button.property("navigation") is True for button in page._nav_buttons)


def test_config_page_preserves_native_control_types_and_binary_api(themed_app):
    page = ConfigPage(ConfigManager())

    assert isinstance(page._model_path_input, QLineEdit)
    assert isinstance(page._tracker_input, QComboBox)
    assert isinstance(page._conf_input, QDoubleSpinBox)
    assert isinstance(page._max_det_input, QSpinBox)
    assert isinstance(page._first_category_region_cb, QCheckBox)
    assert page._first_category_region_cb.styleSheet() == ""

    states = []
    page._first_category_region_cb.stateChanged.connect(states.append)
    page._first_category_region_cb.setChecked(True)
    assert page._first_category_region_cb.isChecked() is True
    assert states


def test_main_window_uses_theme_button_roles_without_local_styles(themed_app):
    window = _window(ConfigManager())

    assert window._btn_config.styleSheet() == ""
    assert window._btn_close.styleSheet() == ""
    assert window._btn_camera.styleSheet() == ""
    assert window._btn_detect.styleSheet() == ""
    assert window._btn_close.property("buttonRole") == "danger"
    assert window._btn_close.isDefault() is False
    assert window._btn_camera.property("buttonRole") == "primary"
    assert window._btn_detect.property("buttonRole") == "primary"
