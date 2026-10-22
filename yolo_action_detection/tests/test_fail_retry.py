"""顺序模式类别/数量失败重试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine, RoundResult, StepStatus
from yolo_runtime.yolo_result_models import ObbDetection


def _det(label: str) -> ObbDetection:
    return ObbDetection(
        class_id=0, label=label, conf=0.9, track_id=1,
        polygon=[(10, 10), (50, 10), (50, 50), (10, 50)],
        box=(10, 10, 50, 50), center=(30, 30),
    )


def _engine(**kwargs) -> StepSequenceEngine:
    params = dict(
        step_class_names=["散热片", "导热垫"],
        step_counts=[1, 3],
        enter_stable_frames=1,
        leave_stable_frames=4,
        out_of_order_frames=10,
        order_constraint_enabled=True,
        fail_retry_enabled=True,
        fail_retry_frames=10,
    )
    params.update(kwargs)
    eng = StepSequenceEngine(**params)
    eng.start_round()
    return eng


def test_config_defaults_disable_fail_retry():
    config = ConfigManager()
    assert config.action_fail_retry_enabled is False
    assert config.action_fail_retry_frames == 10


def test_engine_keeps_large_fail_retry_frames():
    eng = _engine(fail_retry_enabled=True, fail_retry_frames=200)
    assert eng._fail_retry_frames == 200


def test_config_validate_clamps_fail_retry_frames(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "config_schema_version": 4,
        "model_task": "ultralytics_obb",
        "action_fail_retry_enabled": True,
        "action_fail_retry_frames": 0,
    }), encoding="utf-8")
    config = ConfigManager()
    config.load(path)
    assert config.action_fail_retry_enabled is True
    assert config.action_fail_retry_frames == 1

    path.write_text(json.dumps({
        "config_schema_version": 4,
        "model_task": "ultralytics_obb",
        "action_fail_retry_frames": 999,
    }), encoding="utf-8")
    config = ConfigManager()
    config.load(path)
    assert config.action_fail_retry_frames == 100
    assert config.action_fail_retry_enabled is False


def test_disabled_wrong_quantity_does_not_ng():
    eng = _engine(fail_retry_enabled=False)
    for _ in range(20):
        state = eng.update([_det("导热垫"), _det("导热垫")])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.steps[0].status == StepStatus.WAITING
    assert state.steps[0].fail_retry_count == 0


def test_first_step_zero_does_not_count():
    eng = _engine()
    for _ in range(20):
        state = eng.update([])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.current_step_index == 0
    assert state.steps[0].fail_retry_count == 0
    assert state.fail_retry_count == 0


def test_second_step_missing_ten_frames_ng():
    eng = _engine()
    assert eng.update([_det("散热片")]).steps[0].status == StepStatus.PASS
    for i in range(9):
        state = eng.update([])
        assert state.round_result == RoundResult.IN_PROGRESS, i
        assert state.current_step_index == 1
        assert state.fail_retry_count == i + 1
    state = eng.update([])
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 1
    assert state.steps[1].status == StepStatus.NG
    assert "导热垫" in state.action_ng_reason
    assert "实际0/要求3" in state.action_ng_reason
    assert "连续10帧" in state.action_ng_reason


def test_ninth_wrong_frame_still_in_progress():
    eng = _engine()
    eng.update([_det("散热片")])
    for _ in range(9):
        state = eng.update([_det("导热垫"), _det("导热垫")])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.fail_retry_count == 9


def test_tenth_too_many_ng():
    eng = _engine()
    eng.update([_det("散热片")])
    four = [_det("导热垫") for _ in range(4)]
    for _ in range(9):
        eng.update(four)
    state = eng.update(four)
    assert state.round_result == RoundResult.ACTION_NG
    assert "实际4/要求3" in state.action_ng_reason


def test_correct_on_tenth_frame_passes_and_resets():
    eng = _engine()
    eng.update([_det("散热片")])
    two = [_det("导热垫"), _det("导热垫")]
    for _ in range(9):
        eng.update(two)
    state = eng.update([_det("导热垫") for _ in range(3)])
    assert state.round_result == RoundResult.PASS
    assert state.steps[1].status == StepStatus.PASS
    assert state.steps[1].fail_retry_count == 0


def test_first_step_wrong_quantity_ng_stays_on_first():
    eng = _engine()
    two = [_det("散热片"), _det("散热片")]
    for _ in range(9):
        state = eng.update(two)
        assert state.current_step_index == 0
        assert state.round_result == RoundResult.IN_PROGRESS
    state = eng.update(two)
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 0
    assert state.current_step_index == 0
    assert "实际2/要求1" in state.action_ng_reason


def test_quantity_ng_wins_over_order_ng_same_frame():
    eng = _engine(
        step_class_names=["散热片", "导热垫", "标签"],
        step_counts=[1, 3, 1],
        out_of_order_frames=1,
    )
    eng.update([_det("散热片")])
    for _ in range(9):
        eng.update([])
    state = eng.update([_det("标签")])
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 1
    assert "导热垫" in state.action_ng_reason
    assert "检测到步骤" not in state.action_ng_reason


def test_order_ng_still_works_before_retry_exhausted():
    eng = StepSequenceEngine(
        step_class_names=["A", "B", "C"],
        step_counts=[1, 1, 1],
        enter_stable_frames=1,
        out_of_order_frames=1,
        order_constraint_enabled=True,
        fail_retry_enabled=True,
        fail_retry_frames=10,
    )
    eng.start_round()
    eng.update([_det("A")])
    state = eng.update([_det("C")])
    assert state.round_result == RoundResult.ACTION_NG
    assert "检测到步骤3" in state.action_ng_reason
