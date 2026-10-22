"""连续作业：轮次归属、间隔期间的动作暂存、缺步后的轮次恢复。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from step_sequence.step_sequence_engine import RoundResult, StepSequenceEngine, StepStatus

PARTS = {
    "包装": ["充电器", "合盖", "纸盒", "网线", "路由器"],
    "封装": ["充电器", "合盖", "纸盒", "网线"],
}

# 现场 action_trace.log 2026-10-22 10:46–10:49 的 20 个事件：(编号, 动作, 开始帧, 结束帧)
FIELD_EVENTS = [
    (1, "包装", 24, 68), (2, "封装", 68, 103), (3, "包装", 103, 150), (4, "封装", 150, 170),
    (5, "包装", 170, 215), (6, "封装", 215, 250), (7, "包装", 250, 294), (8, "封装", 294, 328),
    (9, "包装", 328, 380), (10, "封装", 380, 397), (11, "包装", 397, 441), (12, "封装", 441, 481),
    (13, "包装", 521, 565), (14, "封装", 565, 606), (15, "包装", 633, 678), (16, "封装", 678, 717),
    (17, "包装", 726, 769), (18, "封装", 769, 810), (19, "包装", 810, 854), (20, "封装", 854, 895),
]


def _engine(names=("包装", "封装")):
    engine = StepSequenceEngine(step_class_names=list(names), step_counts=[1] * len(names), detection_updates=False)
    engine.start_round(action_min_source_frame=4)
    return engine


def _event(event_id, name, start, end, missing=()):
    return {
        "session_id": 1, "event_id": event_id, "class_id": f"c-{name}", "name": name,
        "start_frame": start, "end_frame": end, "decision_frame": end,
        "parts": PARTS.get(name, []), "missing_parts": list(missing),
    }


def test_field_sequence_passes_every_round_when_the_next_round_follows_the_previous_boundary():
    engine = _engine()
    results = []
    for event_id, name, start, end in FIELD_EVENTS:
        state = engine.apply_action_event(_event(event_id, name, start, end))
        if state.round_result != RoundResult.IN_PROGRESS:
            results.append(state.round_result)
            # 现场冷却约 1 秒，预览帧已比结束边界多走 6 帧。
            engine.start_round(action_min_source_frame=end + 6, continue_from_previous=True)
    assert results == [RoundResult.PASS] * 10


def test_field_sequence_survives_events_that_finish_during_the_cooldown():
    engine = _engine()
    results = []
    pending_restart = None
    for event_id, name, start, end in FIELD_EVENTS:
        state = engine.apply_action_event(_event(event_id, name, start, end))
        if pending_restart is not None:
            # 下一件的包装在冷却还没结束时已经完成，到这里才换轮。
            engine.start_round(action_min_source_frame=pending_restart, continue_from_previous=True)
            pending_restart = None
            state = engine.get_state()
        if state.round_result != RoundResult.IN_PROGRESS and state.round_id not in [r for r, _ in results]:
            results.append((state.round_id, state.round_result))
            pending_restart = end + 60
    assert [result for _, result in results] == [RoundResult.PASS] * 10


def test_the_next_round_uses_the_previous_closing_boundary_not_the_preview_frame():
    engine = _engine()
    engine.apply_action_event(_event(1, "包装", 24, 68))
    engine.apply_action_event(_event(2, "封装", 68, 103))
    engine.start_round(action_min_source_frame=109, continue_from_previous=True)
    state = engine.apply_action_event(_event(3, "包装", 103, 150))
    assert state.steps[0].status == StepStatus.PASS
    assert state.steps[1].status == StepStatus.WAITING


def test_starting_detection_uses_the_preview_frame_and_drops_held_events():
    engine = _engine()
    engine.apply_action_event(_event(1, "包装", 24, 68))
    engine.apply_action_event(_event(2, "封装", 68, 103))
    engine.apply_action_event(_event(3, "包装", 103, 150))
    engine.start_round(action_min_source_frame=160)
    state = engine.get_state()
    assert state.steps[0].status == StepStatus.WAITING
    state = engine.apply_action_event(_event(4, "包装", 150, 190))
    assert state.steps[0].status == StepStatus.WAITING


def test_a_held_event_is_not_consumed_twice():
    engine = _engine()
    engine.apply_action_event(_event(1, "包装", 24, 68))
    engine.apply_action_event(_event(2, "封装", 68, 103))
    held = _event(3, "包装", 103, 150)
    engine.apply_action_event(held)
    engine.apply_action_event(held)
    engine.start_round(action_min_source_frame=160, continue_from_previous=True)
    state = engine.get_state()
    assert state.steps[0].status == StepStatus.PASS
    assert state.round_result == RoundResult.IN_PROGRESS
    assert engine._pending_events == []


def test_the_hold_queue_is_bounded():
    engine = _engine()
    engine.apply_action_event(_event(1, "包装", 24, 68))
    engine.apply_action_event(_event(2, "封装", 68, 103))
    for event_id in range(3, 3 + StepSequenceEngine.PENDING_EVENT_LIMIT + 5):
        engine.apply_action_event(_event(event_id, "包装", 103 + event_id, 104 + event_id))
    assert len(engine._pending_events) == StepSequenceEngine.PENDING_EVENT_LIMIT


def test_seal_before_pack_ends_the_round_and_the_next_pack_starts_a_clean_round():
    engine = _engine()
    state = engine.apply_action_event(_event(1, "封装", 24, 60))
    assert state.round_result == RoundResult.ACTION_NG
    engine.start_round(action_min_source_frame=66, continue_from_previous=True)
    engine.apply_action_event(_event(2, "包装", 60, 100))
    state = engine.apply_action_event(_event(3, "封装", 100, 140))
    assert state.round_result == RoundResult.PASS


def test_an_earlier_action_after_later_steps_starts_the_next_product():
    engine = _engine(("A", "B", "C"))
    engine.apply_action_event(_event(1, "A", 10, 20))
    engine.apply_action_event(_event(2, "B", 20, 30))
    state = engine.apply_action_event(_event(3, "A", 30, 40))
    assert state.round_result == RoundResult.ACTION_NG
    assert "未检测到步骤3(C)" in state.action_ng_reason
    assert state.action_boundary_frame == 30
    engine.start_round(action_min_source_frame=50, continue_from_previous=True)
    state = engine.get_state()
    assert state.steps[0].status == StepStatus.PASS
    assert state.steps[1].status == StepStatus.WAITING


def test_the_same_action_split_by_a_short_pause_does_not_start_a_new_product():
    engine = _engine()
    engine.apply_action_event(_event(1, "包装", 24, 60))
    state = engine.apply_action_event(_event(2, "包装", 63, 80))
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.steps[0].status == StepStatus.PASS
    assert state.steps[1].status == StepStatus.WAITING
