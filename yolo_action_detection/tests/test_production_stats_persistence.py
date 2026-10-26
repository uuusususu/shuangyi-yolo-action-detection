"""产能计数写入配置，关程序和重启保留，清零先归档。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class Processor:
    def get_class_names(self):
        return {0: "零件"}


def _window(config, tmp_path) -> MainWindow:
    config._config_path = str(tmp_path / "config.json")
    return MainWindow(
        config,
        AppState(),
        processor=Processor(),
        step_engine=StepSequenceEngine(step_class_names=config.category_names),
        processor_factory=lambda *args, **kwargs: Processor(),
        evidence_base_dir=tmp_path / "evidence",
    )


def test_config_button_opens_the_config_page_without_login(qapp, tmp_path):
    window = _window(ConfigManager(), tmp_path)
    window._on_config_clicked()
    assert window._stacked.currentWidget() is window._config_page
    window.close()


def test_counts_are_written_to_config_and_restored_after_restart(qapp, tmp_path):
    config = ConfigManager()
    config.production_started_at = 1_700_000_000
    config.production_ok = 2
    config.production_ng = 1
    config.validate()
    window = _window(config, tmp_path)
    assert window._stats_manager.batch.total == 3
    assert window._kpi_row._cells["ok"].text() == "2"
    assert window._kpi_row._cells["ng"].text() == "1"

    window._stats_manager.record_pass()
    # 计数先合并，不在出件的那一刻写盘。
    assert not (tmp_path / "config.json").exists()
    window.flush_production_stats()
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["production_ok"] == 3
    assert saved["production_ng"] == 1
    assert saved["production_total"] == 4
    assert saved["production_started_at"] == 1_700_000_000
    window.close()


def test_burst_of_counts_is_written_once_and_close_flushes(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config.validate()
    window = _window(config, tmp_path)
    saves = []
    original_save = ConfigManager.save

    def counting_save(self, path):
        saves.append(path)
        original_save(self, path)

    monkeypatch.setattr(ConfigManager, "save", counting_save)

    for _ in range(5):
        window._stats_manager.record_pass()
    window._stats_manager.record_ng()
    assert saves == []
    window.close()

    assert len(saves) == 1
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["production_ok"] == 5
    assert saved["production_ng"] == 1


def test_a_new_window_restores_counts_saved_in_config(qapp, tmp_path):
    config = ConfigManager()
    config.production_started_at = 1_700_000_000
    config.production_ok = 3
    config.production_ng = 1
    config.validate()
    config._config_path = str(tmp_path / "config.json")
    config.save(tmp_path / "config.json")

    restarted = ConfigManager()
    restarted.load(tmp_path / "config.json")
    window = _window(restarted, tmp_path)
    assert window._stats_manager.batch.ok == 3
    assert window._stats_manager.batch.ng == 1
    assert window._stats_manager.batch.total == 4
    assert window._kpi_row._cells["output"].text() == "4"
    window.close()


def test_clear_archives_then_writes_zero_into_config(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    config = ConfigManager()
    config.production_started_at = 1_700_000_000
    config.production_ok = 2
    config.production_ng = 1
    config.validate()
    window = _window(config, tmp_path)
    window._on_clear_production_stats()

    assert window._stats_manager.batch.total == 0
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["production_ok"] == 0
    assert saved["production_ng"] == 0
    assert saved["production_total"] == 0
    records = list((tmp_path / "production_records").glob("record_*.json"))
    assert len(records) == 1
    archived = json.loads(records[0].read_text(encoding="utf-8"))
    assert archived["ok"] == 2
    assert archived["ng"] == 1
    window.close()


def test_two_archives_in_the_same_second_keep_both_records(tmp_path, monkeypatch):
    from detection_logging import production_stats
    from detection_logging.production_stats import ProductionBatch, ProductionStatsManager

    monkeypatch.setattr(production_stats.time, "time", lambda: 1_700_000_100.0)
    manager = ProductionStatsManager(records_dir=tmp_path)
    for _ in range(2):
        manager.restore(ProductionBatch(started_at=1_700_000_000.0, total=1, ok=1, ng=0))
        assert manager.reset_and_archive().success
    assert len(list(tmp_path.glob("record_*.json"))) == 2


def test_config_form_save_keeps_the_latest_counts(qapp, tmp_path):
    config = ConfigManager()
    config.production_ok = 4
    config.production_ng = 1
    config.production_started_at = 1_700_000_000
    config.validate()
    window = _window(config, tmp_path)
    window._on_config_clicked()
    from copy import deepcopy

    candidate = deepcopy(window.config)
    window._config_page._write_form_into_config(candidate)
    assert candidate.production_ok == 4
    assert candidate.production_ng == 1
    window.close()


def test_model_load_runs_off_the_gui_thread_and_keeps_errors(qapp, tmp_path):
    import threading

    window = _window(ConfigManager(), tmp_path)
    gui_thread = threading.get_ident()
    seen = []

    def load():
        seen.append(threading.get_ident())
        return "loaded"

    assert window._run_model_load(load) == "loaded"
    assert seen and seen[0] != gui_thread

    def broken():
        raise ValueError("模型坏了")

    with pytest.raises(ValueError, match="模型坏了"):
        window._run_model_load(broken)
    assert window._model_loading is False
    window.close()
