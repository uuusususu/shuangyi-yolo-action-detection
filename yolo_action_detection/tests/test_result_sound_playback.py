"""PASS/FAIL 独立声音播放运行时行为测试（tasks 2.1-2.5）。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from PySide6.QtWidgets import QApplication

from config import ConfigManager
from app_state import AppState
from detection_logging.production_stats import ProductionStatsManager
from detection_logging.audio_feedback import SoundFeedback
from step_sequence.step_sequence_engine import RoundResult, StepStatus, StepSequenceState, StepState
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_window(qapp, tmp_path, *, pass_sound=False, fail_sound=True):
    config = ConfigManager()
    config.pass_sound_enabled = pass_sound
    config.fail_sound_enabled = fail_sound
    config.fail_evidence_enabled = False
    state = AppState()
    sound_spy = MagicMock(spec=SoundFeedback)
    window = MainWindow(
        config, state, processor=None, step_engine=None,
        sound_feedback=sound_spy,
        evidence_base_dir=str(tmp_path / "evidence"),
    )
    return window, sound_spy


def _make_ng_state(round_id=1, ng_step=1):
    return StepSequenceState(
        steps=[StepState(index=0, class_name="A", configured=True, status=StepStatus.PASS),
               StepState(index=1, class_name="B", configured=True, status=StepStatus.NG)],
        current_step_index=0,
        round_result=RoundResult.ACTION_NG,
        round_id=round_id,
        action_ng_step=ng_step,
    )


def _make_pass_state(round_id=1):
    return StepSequenceState(
        steps=[StepState(index=0, class_name="A", configured=True, status=StepStatus.PASS)],
        current_step_index=-1,
        round_result=RoundResult.PASS,
        round_id=round_id,
    )


# --- Task 2.1: 普通顺序模式 PASS/FAIL 独立开关 + 去重 ---

def test_pass_plays_when_enabled(qapp, tmp_path):
    """PASS 开关开启时播放 PASS 声音。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=True, fail_sound=False)
    state = _make_pass_state(round_id=1)
    w._update_step_display(state)
    spy.play_pass.assert_called_once()


def test_pass_silent_when_disabled(qapp, tmp_path):
    """PASS 开关关闭时不播放。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=False)
    state = _make_pass_state(round_id=1)
    w._update_step_display(state)
    spy.play_pass.assert_not_called()


def test_fail_plays_when_enabled(qapp, tmp_path):
    """FAIL 开关开启时播放 FAIL 声音。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=True)
    state = _make_ng_state(round_id=1)
    w._update_step_display(state)
    spy.play_fail.assert_called_once()


def test_fail_silent_when_disabled(qapp, tmp_path):
    """FAIL 开关关闭时不播放。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=False)
    state = _make_ng_state(round_id=1)
    w._update_step_display(state)
    spy.play_fail.assert_not_called()


def test_same_round_id_no_duplicate_sound(qapp, tmp_path):
    """同一 round_id 重复刷新只播放一次。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=True, fail_sound=True)
    state = _make_pass_state(round_id=1)
    w._update_step_display(state)
    w._update_step_display(state)
    w._update_step_display(state)
    assert spy.play_pass.call_count == 1


# --- Task 2.5: 声音播放异常安全 ---

def test_sound_exception_does_not_crash(qapp, tmp_path):
    """声音播放器抛异常时检测结果、统计和下一轮仍继续。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=True, fail_sound=True)
    spy.play_pass.side_effect = RuntimeError("audio device error")

    state = _make_pass_state(round_id=1)
    # 不应抛异常
    w._update_step_display(state)
    # 统计应已记录
    assert w._stats_manager.batch.total == 1
    assert w._stats_manager.batch.ok == 1


# --- Task 2.2-2.4: 首类别（PCB）模式声音测试 ---

def _make_pcb_result(track_id, attempt_id, result, missing=None, is_new_sig=True):
    """构造 PCB 检查结果。"""
    from pcb_inspection.models import PcbInspectionResult, PcbResult
    return PcbInspectionResult(
        track_id=track_id,
        result=result,
        slot_states={},
        missing_classes=missing or [],
        source_frame_id=1,
        timestamp=0.0,
        attempt_id=attempt_id,
        is_new_fail_signature=is_new_sig,
    )


def test_pcb_fail_plays_once_per_attempt(qapp, tmp_path):
    """同一父类同一轮 FAIL 重复结果只播放一次。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=True)
    result = _make_pcb_result(track_id=1, attempt_id=1, result=None)
    from pcb_inspection.models import PcbResult
    result.result = PcbResult.FAIL

    w._handle_pcb_results([result], frame_id=1)
    w._handle_pcb_results([result], frame_id=2)
    w._handle_pcb_results([result], frame_id=3)
    assert spy.play_fail.call_count == 1


def test_pcb_new_attempt_plays_again(qapp, tmp_path):
    """同一父类新轮次（新 attempt_id）再次 FAIL 时再播放一次。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=True)
    from pcb_inspection.models import PcbResult

    r1 = _make_pcb_result(track_id=1, attempt_id=1, result=PcbResult.FAIL)
    w._handle_pcb_results([r1], frame_id=1)

    r2 = _make_pcb_result(track_id=1, attempt_id=2, result=PcbResult.FAIL)
    w._handle_pcb_results([r2], frame_id=2)

    assert spy.play_fail.call_count == 2


def test_pcb_different_parents_each_play_once(qapp, tmp_path):
    """两个不同父类分别 FAIL 时各播放一次。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=True)
    from pcb_inspection.models import PcbResult

    r1 = _make_pcb_result(track_id=1, attempt_id=1, result=PcbResult.FAIL)
    r2 = _make_pcb_result(track_id=2, attempt_id=1, result=PcbResult.FAIL)
    w._handle_pcb_results([r1], frame_id=1)
    w._handle_pcb_results([r2], frame_id=2)
    assert spy.play_fail.call_count == 2


def test_pcb_fail_silent_when_disabled(qapp, tmp_path):
    """FAIL 开关关闭时 PCB 模式完全静音。"""
    w, spy = _make_window(qapp, tmp_path, pass_sound=False, fail_sound=False)
    from pcb_inspection.models import PcbResult

    r = _make_pcb_result(track_id=1, attempt_id=1, result=PcbResult.FAIL)
    w._handle_pcb_results([r], frame_id=1)
    spy.play_fail.assert_not_called()