"""产线连续时序的采样、结算与轮次隔离回归。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(ROOT), str(SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from lstm_action.stream import TimelineSession, TimeBaseSampler
from lstm_workbench.service import resample_indices
from step_sequence.step_sequence_engine import StepSequenceEngine, StepStatus


def _session(fps=10.0, min_seconds=1.0, confirm_seconds=None):
    return TimelineSession(SimpleNamespace(
        min_action_seconds=min_seconds, time_base_fps=fps,
        model=SimpleNamespace(eval=lambda: None), class_ids=["idle", "pack", "seal"],
        names_by_id={"idle": "空闲", "pack": "包装", "seal": "封装"},
        part_names=["零件"], required_parts={"pack": ["零件"], "seal": ["零件"]},
        yolo_conf_threshold=0.8, weights_sha256="test",
    ), confirm_seconds=confirm_seconds)


def _observe(session, frame, label, *, fps=10.0, parts=False):
    return session.observe_frame(
        source_frame=frame, timestamp=frame / fps,
        detections=[{"label": "零件", "conf": 0.9}] if parts else [],
        hands=[], frame_w=10, frame_h=10, class_index=label,
    )


@pytest.mark.parametrize("source_fps,target_fps,count", [(6.30469, 6.248084, 20), (24, 10, 48), (10, 5, 20)])
def test_causal_sampler_selects_the_workbench_floor_frame(source_fps, target_fps, count):
    sampler = TimeBaseSampler(target_fps)
    selected = []
    for frame in range(count):
        slot = sampler.push(frame, frame / source_fps)
        if isinstance(slot, int):
            selected.append(sampler.selected_frame)
            assert frame / source_fps - sampler.selected_time <= sampler.period + 1e-9
    expected = resample_indices(count, source_fps, target_fps)
    assert selected == expected[:len(selected)]
    assert len(selected) == len(set(selected))


def test_shorter_switch_confirm_settles_before_the_model_minimum():
    session = _session(confirm_seconds=0.5)
    labels = [0] * 10 + [1] * 10 + [2] * 5
    reports = [_observe(session, i, label) for i, label in enumerate(labels)]
    assert reports[-2]["events"] == []
    event = reports[-1]["events"][0]
    assert event["name"] == "包装"
    assert event["end_frame"] == 20
    assert event["end_time"] == pytest.approx(2.0)
    assert event["decision_frame"] == 24
    assert event["decision_time"] == pytest.approx(2.4)
    assert session._runs[-1].class_index == 2


def test_flicker_shorter_than_switch_confirm_does_not_finish_the_action():
    session = _session(confirm_seconds=0.5)
    labels = [0] * 10 + [1] * 10 + [2] * 2 + [1] * 10
    events = []
    for index, label in enumerate(labels):
        events.extend(_observe(session, index, label)["events"])
    assert events == []


def test_confirmed_switch_that_ends_before_the_model_minimum_is_not_an_action():
    session = _session(confirm_seconds=0.5)
    labels = [0] * 10 + [1] * 10 + [2] * 6 + [1] * 10 + [0] * 10
    events = []
    for index, label in enumerate(labels):
        events.extend(_observe(session, index, label)["events"])
    assert [event["class_id"] for event in events] == ["pack", "pack"]


def test_live_seen_parts_follow_the_required_list_for_that_action():
    session = _session()
    session.bundle.part_names = ["路由器", "合盖"]
    session.bundle.required_parts = {"pack": ["路由器", "网线"], "seal": ["合盖"]}
    report = session.observe_frame(
        source_frame=0, timestamp=0.0,
        detections=[{"label": "路由器", "conf": 0.9}, {"label": "合盖", "conf": 0.9}],
        hands=[], frame_w=10, frame_h=10, class_index=2,
    )
    assert report["action"] == "封装"
    assert report["seen_parts"] == ["合盖"]
    assert report["missing_parts"] == []


def test_decision_uses_the_actual_settlement_frame_and_half_open_boundary():
    session = _session()
    reports = [_observe(session, i, label) for i, label in enumerate([0] * 10 + [1] * 10 + [0] * 10)]
    event = reports[-1]["events"][0]
    assert event["end_frame"] == 20
    assert event["end_time"] == pytest.approx(2.0)
    assert event["decision_frame"] == 29
    assert event["decision_time"] == pytest.approx(2.9)


def test_finished_short_runs_are_bounded_and_match_offline_closed_prefix():
    from lstm_workbench.check_clip import absorb_short_segments

    session = _session()
    labels = [0] * 10 + [1] * 10 + [value for _ in range(1000) for value in (2, 1)] + [0] * 10
    events = []
    peak = 0
    for i, label in enumerate(labels):
        report = _observe(session, i, label, parts=(i == 301))
        events.extend(report["events"])
        peak = max(peak, len(session._runs))
        assert len(report["evidence_frames"]) <= 8
    assert peak <= 3
    raw = []
    for i, label in enumerate(labels):
        if raw and raw[-1]["class_id"] == str(label):
            raw[-1]["end_frame"] = i + 1
        else:
            raw.append({"class_id": str(label), "start_frame": i, "end_frame": i + 1})
    offline = absorb_short_segments(raw, 10, len(labels), 10.0)
    expected = [(item["start_frame"], item["end_frame"]) for item in offline[1:-1] if item["class_id"] != "0"]
    assert [(event["start_frame"], event["end_frame"]) for event in events] == expected
    assert events[0]["parts"] == ["零件"]


def test_pending_sampling_classifies_the_selected_frame_and_keeps_its_parts():
    session = _session(fps=10, min_seconds=0.2)
    seen = []

    def classify(detections, hands, width, height):
        seen.append(detections[0]["frame"])
        return 0

    session._classify = classify
    for i in range(48):
        session.observe_frame(source_frame=i, timestamp=i / 24.0,
                              detections=[{"frame": i}], hands=[], frame_w=10, frame_h=10)
    assert seen == resample_indices(48, 24, 10)[:len(seen)]


def test_downsampling_boundaries_include_unsampled_parts_without_successor_leakage():
    session = _session(fps=10, min_seconds=0.3)
    events = []
    for i in range(65):
        # 抽样的包装是源帧7、9、12，空闲从14开始。源帧13也属于包装。
        label = 1 if 7 <= i < 14 else 0
        report = _observe(session, i, label, fps=24.0, parts=(i == 13))
        events.extend(report["events"])
    assert len(events) == 1
    assert (events[0]["start_frame"], events[0]["end_frame"]) == (7, 14)
    assert events[0]["start_time"] == pytest.approx(7 / 24)
    assert events[0]["end_time"] == pytest.approx(14 / 24)
    assert events[0]["missing_parts"] == []

    second = _session(fps=10, min_seconds=0.3)
    events = []
    for i in range(65):
        events.extend(_observe(second, i, 1 if 7 <= i < 14 else 0, fps=24.0, parts=(i == 14))["events"])
    assert events[0]["missing_parts"] == ["零件"]


def test_model_runtime_error_is_a_locked_fault():
    session = _session()

    def classify(*args):
        raise RuntimeError("LSTM 推理异常")

    session._classify = classify
    report = session.observe_frame(source_frame=0, timestamp=0, detections=[], hands=[], frame_w=10, frame_h=10)
    assert "LSTM 推理异常" in report["fault"]
    assert report["events"] == []


def test_slow_processing_does_not_stop_the_line_or_invent_samples():
    session = _session(fps=10, min_seconds=0.2)
    events = []
    selected = []
    for i in range(60):
        report = _observe(session, i, 0 if i < 10 or i >= 30 else 1, fps=6)
        assert report["fault"] == ""
        events.extend(report["events"])
        if session.sampler.selected_frame is not None and (
            not selected or selected[-1] != session.sampler.selected_frame
        ):
            selected.append(session.sampler.selected_frame)
    assert selected == sorted(set(selected))
    assert len(selected) <= 60
    assert events == [] or all(event["start_frame"] in selected for event in events)


def test_jittered_timestamps_select_only_real_preceding_frames():
    times = [i / 24 + (0.003 if i % 2 else 0) for i in range(48)]
    sampler = TimeBaseSampler(10)
    selected = []
    for i, timestamp in enumerate(times):
        result = sampler.push(i, timestamp)
        assert result != "interrupt"
        if isinstance(result, int):
            selected.append(sampler.selected_frame)
            assert times[sampler.selected_frame] <= result * 0.1 + 1e-9
    expected = [max(i for i, time in enumerate(times) if time <= slot * 0.1 + 1e-9) for slot in range(len(selected))]
    assert selected == expected


def test_round_source_lower_bound_discards_cross_round_actions_and_deduplicates():
    engine = StepSequenceEngine(step_class_names=["包装"], detection_updates=False)
    engine.start_round(action_min_source_frame=45)
    event = {"session_id": 1, "event_id": 1, "class_id": "pack", "name": "包装",
             "start_frame": 30, "end_frame": 60, "missing_parts": []}
    state = engine.apply_action_event(event)
    assert state.steps[0].status == StepStatus.WAITING
    engine.start_round(action_min_source_frame=0)
    assert engine.apply_action_event(event).steps[0].status == StepStatus.WAITING
    assert engine.apply_action_event({**event, "event_id": 2, "start_frame": 60}).steps[0].status == StepStatus.PASS


def test_long_running_round_deduplication_uses_bounded_high_watermark():
    engine = StepSequenceEngine(step_class_names=["包装"], detection_updates=False)
    for index in range(1, 2001):
        engine.start_round(action_min_source_frame=index)
        engine.apply_action_event({"session_id": 1, "event_id": index, "class_id": "pack",
                                   "name": "包装", "start_frame": index, "missing_parts": []})
    assert engine._last_consumed_event == (1, 2000)
    engine.start_round()
    old = {"session_id": 1, "event_id": 1, "class_id": "pack", "name": "包装", "missing_parts": []}
    assert engine.apply_action_event(old).steps[0].status == StepStatus.WAITING
    assert engine.apply_action_event({**old, "session_id": 2}).steps[0].status == StepStatus.PASS
    engine.start_round()
    assert engine.apply_action_event({**old, "event_id": 2001}).steps[0].status == StepStatus.WAITING
    replacement = StepSequenceEngine(step_class_names=["包装"], detection_updates=False)
    assert replacement.apply_action_event(old).steps[0].status == StepStatus.PASS


def test_public_fault_latches_until_hard_reset():
    session = _session()
    session.set_fault("YOLO 运行失败")
    assert _observe(session, 0, 0)["fault"] == "YOLO 运行失败"
    session.reset()
    assert not _observe(session, 0, 0)["fault"]


def test_real_action_event_keeps_evidence_identity_and_capture_times_in_state():
    session = _session()
    events = []
    for index, label in enumerate([0] * 10 + [1] * 10 + [0] * 10):
        events.extend(_observe(session, index, label)["events"])
    event = events[0]
    engine = StepSequenceEngine(step_class_names=["包装"], detection_updates=False)
    state = engine.apply_action_event(event)
    assert state.action_weights_sha256 == event["weights_sha256"] == "test"
    assert state.action_start_time == event["start_time"] == 1.0
    assert state.action_end_time == event["end_time"] == 2.0
    assert state.action_decision_time == event["decision_time"] == 2.9
    engine.start_round(action_min_source_frame=30)
    fresh = engine.get_state()
    assert fresh.action_weights_sha256 == ""
    assert (fresh.action_start_time, fresh.action_end_time, fresh.action_decision_time) == (0.0, 0.0, 0.0)
