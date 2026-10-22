from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QMessageBox
from app_state import AppState
from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine, RoundResult
from ui.config_page import ConfigPage
from ui.main_window import MainWindow
from detection_logging.fail_evidence import FailEvidenceSaver


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


class Processor:
    def get_class_names(self):
        return {0: "零件"}


def make_window(tmp_path):
    config = ConfigManager()
    config.action_order_constraint_enabled = False
    config.roi_inspection_enabled = False
    config.category_names = ["零件", "", "", "", "", ""]
    config._config_path = str(tmp_path / "config.json")
    config.save(config.get_config_path())
    processor = Processor()
    window = MainWindow(config, AppState(), processor=processor,
                        step_engine=StepSequenceEngine(step_class_names=config.category_names),
                        processor_factory=lambda *args, **kwargs: Processor(),
                        evidence_base_dir=tmp_path / "evidence")
    window.worker.set_frame_processor(processor)
    return window


def test_shutdown_keeps_the_model_folder_saved_from_the_config_page(tmp_path):
    from main import save_live_config

    path = tmp_path / "config.json"
    startup = ConfigManager()
    startup.lstm_bundle_path = "models/action"
    startup.camera_manual_exposure_us = 30001
    startup._config_path = str(path)
    startup.save(path)
    live = deepcopy(startup)
    live.lstm_bundle_path = r"E:\runs\20261020-174254"
    live.save(path)
    window = SimpleNamespace(config=live)

    save_live_config(window, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["lstm_bundle_path"] == r"E:\runs\20261020-174254"
    assert saved["camera_manual_exposure_us"] == 30001
    source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = source.read_text(encoding="utf-8")
    assert "save_live_config(window, config_path)" in text


def test_a_second_launch_does_not_own_the_running_instance():
    from main import ERROR_ALREADY_EXISTS, gui_instance_lock_required, this_process_owns_instance

    assert this_process_owns_instance(1, 0) is True
    assert this_process_owns_instance(1, ERROR_ALREADY_EXISTS) is False
    assert this_process_owns_instance(0, ERROR_ALREADY_EXISTS) is True
    assert gui_instance_lock_required(["YOLOActionDetection.exe"]) is True
    assert gui_instance_lock_required(["YOLOActionDetection.exe", "--portable-smoke-test"]) is False
    source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text(encoding="utf-8")
    assert source.index("_stop_if_another_instance()") < source.index("from ui.main_window import MainWindow")


def test_save_reloads_the_action_model_when_the_session_is_still_old(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    window.config.roi_inspection_enabled = False
    new_dir = tmp_path / "20261020-174254"
    window.config.lstm_bundle_path = str(new_dir)
    window.worker._action_session = SimpleNamespace(
        bundle=SimpleNamespace(directory=tmp_path / "models" / "action"),
        marker="old",
    )
    new_session = SimpleNamespace(bundle=SimpleNamespace(directory=new_dir), marker="new")

    def load(*_args):
        return new_session, ""

    monkeypatch.setattr(window, "_load_action_session", load)
    window.lstm_action.state.loaded = True
    window._reload_configured_action_model()
    assert window.worker.get_action_session() is new_session
    assert "20261020-174254" in window._lstm_status_label.text()
    assert "20261020-174254" in window._config_applied_text()


def test_form_candidate_failure_does_not_mutate_shared_config(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    seen = []
    def reject(candidate):
        seen.append((candidate is window.config, window.config.yolo_model_path))
        raise ValueError("真实模型无法加载")
    page = ConfigPage(window.config, config_committer=reject)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: None)
    old_bytes = Path(window.config.get_config_path()).read_bytes()
    old_path = window.config.yolo_model_path
    page._model_path_input.setText("bad.pt")
    page._on_save()
    assert seen == [(False, old_path)]
    assert window.config.yolo_model_path == old_path
    assert Path(window.config.get_config_path()).read_bytes() == old_bytes


@pytest.mark.parametrize("failure", ["processor", "session", "save", "swap"])
def test_commit_failure_preserves_config_disk_processor_session(qapp, tmp_path, monkeypatch, failure):
    window = make_window(tmp_path)
    old_config, old_processor = window.config, window.processor
    old_session = SimpleNamespace(marker="旧会话")
    window.worker._action_session = old_session
    old_bytes = Path(old_config.get_config_path()).read_bytes()
    candidate = deepcopy(old_config)
    candidate.yolo_model_path = "new.pt"
    if failure == "processor":
        monkeypatch.setattr(window, "_create_processor_from_config", lambda *a: (_ for _ in ()).throw(ValueError("坏 YOLO")))
    elif failure == "session":
        candidate.action_order_constraint_enabled = True
        monkeypatch.setattr(window, "_load_action_session", lambda *a: (None, "坏 LSTM"))
    elif failure == "save":
        monkeypatch.setattr(candidate, "save", lambda *a: (_ for _ in ()).throw(OSError("写入失败")))
    else:
        monkeypatch.setattr(window.worker, "replace_runtime", lambda *a: (_ for _ in ()).throw(RuntimeError("交换失败")))
    with pytest.raises((ValueError, OSError, RuntimeError)):
        window._commit_candidate_config(candidate)
    assert window.config is old_config
    assert window.worker.config is old_config
    assert window.processor is old_processor
    assert window.worker._action_session is old_session
    assert Path(old_config.get_config_path()).read_bytes() == old_bytes


def test_successful_commit_replaces_actual_runtime_and_backs_up_disk(qapp, tmp_path):
    window = make_window(tmp_path)
    old_processor = window.processor
    old_bytes = Path(window.config.get_config_path()).read_bytes()
    candidate = deepcopy(window.config)
    candidate.yolo_model_path = "new.pt"
    window._commit_candidate_config(candidate)
    assert window.config is candidate
    assert window.worker.config is candidate
    assert window.processor is not old_processor
    assert window.worker.frame_processor is window.processor
    assert json.loads(Path(candidate.get_config_path()).read_text(encoding="utf-8"))["yolo_model_path"] == "new.pt"
    assert Path(candidate.get_config_path() + ".bak").read_bytes() == old_bytes


def test_bundle_content_and_mode_are_in_both_runtime_signatures(qapp, tmp_path):
    window = make_window(tmp_path)
    candidate = deepcopy(window.config)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("best_lstm.pt", "contract.json", "runtime_manifest.json"):
        (bundle / name).write_text("a")
    candidate.lstm_bundle_path = str(bundle)
    before = window._model_runtime_signature(candidate), window._step_engine_config_signature(candidate)
    (bundle / "best_lstm.pt").write_text("b")
    after = window._model_runtime_signature(candidate), window._step_engine_config_signature(candidate)
    assert all(a != b for a, b in zip(before, after))
    candidate.roi_inspection_enabled = True
    assert all(a != b for a, b in zip(after, (window._model_runtime_signature(candidate), window._step_engine_config_signature(candidate))))


def test_lstm_ng_uses_real_boundary_frame_and_matching_detections(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    boundary_frame = np.full((2, 2, 3), 50, dtype=np.uint8)
    seen = []
    monkeypatch.setattr(window.worker, "get_action_boundary_evidence", lambda fid: (boundary_frame, [{"label": "边界零件"}], {"source_frame_id": 50, "timestamp": 12.0}))
    monkeypatch.setattr(window._fail_evidence_saver, "save_fail_event", lambda frame, context: seen.append((frame, context)))
    window._last_overlay_data = {"source_frame_id": 99, "detections": [{"label": "后继零件"}]}
    window._save_fail_evidence(SimpleNamespace(action_boundary_frame=50, action_decision_frame=99))
    assert np.array_equal(seen[0][0], boundary_frame)
    assert seen[0][1].source_frame_id == 50
    assert seen[0][1].detections == [{"label": "边界零件"}]
    assert seen[0][1].decision_frame == 99


def test_missing_boundary_evidence_does_not_save_decision_frame(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    seen = []
    monkeypatch.setattr(window.worker, "get_action_boundary_evidence", lambda fid: None)
    monkeypatch.setattr(window._fail_evidence_saver, "save_fail_event", lambda *a: seen.append(a))
    window._save_fail_evidence(SimpleNamespace(action_boundary_frame=50, action_decision_frame=99))
    assert not seen
    assert "证据不可用" in window._notice_label.text()


def test_new_round_sets_source_frame_fence(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    calls = []
    monkeypatch.setattr(window.worker, "get_latest_preview_frame_id", lambda: 42, raising=False)
    monkeypatch.setattr(window.step_engine, "start_round", lambda **kw: calls.append(kw))
    window._start_new_round_now()
    assert calls[0]["action_min_source_frame"] == 43


def test_captured_during_cooldown_action_cannot_pass_when_inference_lags(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    monkeypatch.setattr(window.worker, "get_display_frame_id", lambda: 40)
    monkeypatch.setattr(window.worker, "get_latest_preview_frame_id", lambda: 45, raising=False)
    window._start_new_round_now()
    late_event = {"session_id": 1, "event_id": 1, "class_id": "part_action", "name": "零件",
                  "start_frame": 43, "end_frame": 60, "decision_frame": 65, "parts": [], "missing_parts": []}
    window._on_overlay_updated({"source_frame_id": 65, "lstm_action": {"events": [late_event]}})
    assert window.step_engine.get_state().round_result == RoundResult.IN_PROGRESS
    assert window._stats_manager.batch.ok == 0
    new_event = dict(late_event, event_id=2, start_frame=46, end_frame=70, decision_frame=75)
    window._on_overlay_updated({"source_frame_id": 75, "lstm_action": {"events": [new_event]}})
    assert window.step_engine.get_state().round_result == RoundResult.PASS
    assert window._stats_manager.batch.ok == 1
    window._round_pass_timer.stop()


def test_changing_bundle_replaces_session_at_commit(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    old_session = SimpleNamespace(marker="旧动作模型")
    new_session = SimpleNamespace(marker="新动作模型")
    window.worker._action_session = old_session
    candidate = deepcopy(window.config)
    candidate.action_order_constraint_enabled = True
    candidate.lstm_bundle_path = "bundle-B"
    monkeypatch.setattr(window, "_load_action_session", lambda *a: (new_session, ""))
    window._commit_candidate_config(candidate)
    assert window.worker.get_action_session() is new_session
    assert window.config.lstm_bundle_path == "bundle-B"
    assert window.step_engine._detection_updates is False


def test_lstm_events_during_cooldown_are_kept_for_the_next_round(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    monkeypatch.setattr(window.worker, "get_latest_preview_frame_id", lambda: 10, raising=False)
    window._begin_step_round()
    part = {"session_id": 1, "event_id": 1, "class_id": "part_action", "name": "零件",
            "start_frame": 12, "end_frame": 40, "decision_frame": 40, "parts": [], "missing_parts": []}
    window._on_overlay_updated({"source_frame_id": 40, "lstm_action": {"events": [part]}})
    assert window._round_pass_timer.isActive()
    assert window._stats_manager.batch.ok == 1
    during = dict(part, event_id=2, start_frame=40, end_frame=70, decision_frame=70)
    window._on_overlay_updated({"source_frame_id": 70, "lstm_action": {"events": [during]}})
    monkeypatch.setattr(window.worker, "get_latest_preview_frame_id", lambda: 75, raising=False)
    window._round_pass_timer.stop()
    window._on_round_pass_settled()
    assert window.step_engine.get_state().round_result == RoundResult.PASS
    assert window._stats_manager.batch.ok == 2
    window._round_pass_timer.stop()


def test_lstm_events_are_dropped_while_inference_is_off(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    window.state.set_camera_on(True)
    window.state.set_inference_on(False)
    seen = []
    monkeypatch.setattr(window.step_engine, "apply_action_event", lambda event: seen.append(event))
    window._on_overlay_updated({"source_frame_id": 42, "lstm_action": {"events": [{"event_id": 1}]}})
    assert not seen


def test_model_fault_stops_detection_without_product_ng(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    window._on_overlay_updated({"source_frame_id": 42, "lstm_action": {"fault": "手部推理失败"}})
    assert window.state.inference_on is False
    assert "模型故障" in window._notice_label.text()
    assert window.step_engine.get_state().round_result.value != "ACTION_NG"
    assert window._stats_manager.batch.ng == 0


def test_old_generation_events_cannot_enter_new_runtime(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.worker.runtime_generation = 2
    window.config.action_order_constraint_enabled = True
    window.state.set_camera_on(True)
    window.state.set_inference_on(True)
    seen = []
    monkeypatch.setattr(window.step_engine, "apply_action_event", lambda event: seen.append(event))
    window._on_overlay_updated({"runtime_generation": 1, "lstm_action": {"events": [{"event_id": 1}]}})
    assert not seen


def test_action_evidence_uses_capture_wall_clock_for_archive_date(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    captured_wall_clock = time.time()
    monkeypatch.setattr(window.worker, "get_action_boundary_evidence", lambda fid: (
        np.zeros((4, 4, 3), dtype=np.uint8), [],
        {"source_frame_id": 50, "timestamp": captured_wall_clock, "capture_monotonic": 1234.5},
    ))
    window._fail_evidence_saver = FailEvidenceSaver(base_dir=tmp_path / "evidence")
    window._save_fail_evidence(SimpleNamespace(
        action_boundary_frame=50, action_decision_frame=99,
        action_weights_sha256="a" * 64,
        action_start_time=1200.0, action_end_time=1234.5, action_decision_time=1240.0,
    ))
    metadata_path = next((tmp_path / "evidence").rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["timestamp"] == captured_wall_clock
    assert metadata["model_sha256"] == "a" * 64
    assert metadata["action_start_time"] == 1200.0
    assert metadata["action_end_time"] == 1234.5
    assert metadata["action_decision_time"] == 1240.0
    assert metadata["action_time_clock"] == "monotonic"
    assert metadata_path.parent.parent.name == time.strftime("%Y%m%d", time.localtime(captured_wall_clock))


def test_camera_failure_after_model_commit_has_clear_saved_status(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    candidate = deepcopy(window.config)
    monkeypatch.setattr(window, "_apply_camera_selection", lambda: False)
    window._commit_candidate_config(candidate)
    assert window.config is candidate
    assert "配置和模型已保存" in window._notice_label.text()


def test_late_action_events_after_stop_cannot_pass(qapp, tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.config.action_order_constraint_enabled = True
    seen = []
    monkeypatch.setattr(window.step_engine, "apply_action_event", lambda event: seen.append(event))
    window.state.set_inference_on(False)
    window._on_overlay_updated({"source_frame_id": 42, "lstm_action": {"events": [{"event_id": 1}]}})
    assert not seen


def test_runtime_paths_and_bundle_hash_are_independent_of_cwd(qapp, tmp_path, monkeypatch):
    import lstm_action.bundle
    window = make_window(tmp_path)
    window._app_base_dir = tmp_path
    window.config.lstm_bundle_path = "bundle"
    window.config.yolo_model_path = "models/yolo.pt"
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/best_lstm.pt").write_bytes(b"a")
    before = window._action_bundle_signature(window.config)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    assert window._action_bundle_signature(window.config) == before
    calls = []
    def capture(path, **kwargs):
        calls.append((path, kwargs["yolo_path"]))
        raise ValueError("捕获路径")
    monkeypatch.setattr(lstm_action.bundle, "load_runtime_bundle", capture)
    window._load_action_session()
    assert calls == [(tmp_path / "bundle", str(tmp_path / "models/yolo.pt"))]
