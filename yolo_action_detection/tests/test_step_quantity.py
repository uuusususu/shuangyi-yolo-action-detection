"""步骤数量匹配测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from step_sequence.step_sequence_engine import StepSequenceEngine, RoundResult, StepStatus
from yolo_runtime.yolo_result_models import ObbDetection


def _make_det(label, conf=0.9):
    return ObbDetection(
        class_id=0, label=label, conf=conf, track_id=1,
        polygon=[(10, 10), (50, 10), (50, 50), (10, 50)],
        box=(10, 10, 50, 50), center=(30, 30),
    )


def test_config_defaults_quantity_one():
    """旧配置没有 category_counts 时默认数量为 1。"""
    eng = StepSequenceEngine(
        step_class_names=["A", "B", ""],
        enter_stable_frames=1,
    )
    assert eng._steps[0].required_count == 1
    assert eng._steps[1].required_count == 1


def test_quantity_four_pass():
    """当前帧检测到 4 个，目标 4 -> PASS。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=1,
    )
    eng.start_round()
    dets = [_make_det("脚垫") for _ in range(4)]
    state = eng.update(dets)
    assert state.steps[0].status == StepStatus.PASS


def test_quantity_three_not_pass():
    """当前帧 3 个，目标 4 -> 不 PASS。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=1,
    )
    eng.start_round()
    dets = [_make_det("脚垫") for _ in range(3)]
    state = eng.update(dets)
    assert state.steps[0].status == StepStatus.WAITING
    assert state.round_result == RoundResult.IN_PROGRESS


def test_quantity_five_not_pass():
    """当前帧 5 个，目标 4 -> 不 PASS（严格等于）。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=1,
    )
    eng.start_round()
    dets = [_make_det("脚垫") for _ in range(5)]
    state = eng.update(dets)
    assert state.steps[0].status == StepStatus.WAITING


def test_no_cross_frame_accumulation():
    """不同帧数量不累计：2+2 不等于 4。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=1,
    )
    eng.start_round()
    # 第一帧 2 个
    eng.update([_make_det("脚垫") for _ in range(2)])
    assert eng.get_state().steps[0].status == StepStatus.WAITING
    # 第二帧 2 个
    eng.update([_make_det("脚垫") for _ in range(2)])
    assert eng.get_state().steps[0].status == StepStatus.WAITING


def test_stable_frames_two_requires_consecutive():
    """稳定帧=2 需要连续两帧各自匹配数量。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=2,
    )
    eng.start_round()
    dets4 = [_make_det("脚垫") for _ in range(4)]
    # 第一帧 4 个 -> enter_count=1，不 PASS
    state1 = eng.update(dets4)
    assert state1.steps[0].status == StepStatus.WAITING
    # 第二帧 4 个 -> enter_count=2，PASS
    state2 = eng.update(dets4)
    assert state2.steps[0].status == StepStatus.PASS


def test_quantity_one_backward_compatible():
    """数量 1 等价于旧的类别命中。"""
    eng = StepSequenceEngine(
        step_class_names=["A", "B", ""],
        step_counts=[1, 1],
        enter_stable_frames=1,
    )
    eng.start_round()
    state = eng.update([_make_det("A")])
    assert state.steps[0].status == StepStatus.PASS


def test_current_count_displayed():
    """步骤状态包含 current_count 供 UI 显示。"""
    eng = StepSequenceEngine(
        step_class_names=["脚垫", ""],
        step_counts=[4],
        enter_stable_frames=1,
    )
    eng.start_round()
    dets = [_make_det("脚垫") for _ in range(3)]
    state = eng.update(dets)
    assert state.steps[0].current_count == 3
    assert state.steps[0].required_count == 4