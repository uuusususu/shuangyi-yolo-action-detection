"""动作追溯日志：采样概率、确认计数、起止帧和拒收原因。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(ROOT), str(SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from detection_logging.action_trace import configure_action_trace
from lstm_action.stream import TimelineSession
from step_sequence.step_sequence_engine import StepSequenceEngine, StepStatus


def _session(confirm_seconds=0.5):
    return TimelineSession(SimpleNamespace(
        min_action_seconds=1.0, time_base_fps=10.0,
        model=SimpleNamespace(eval=lambda: None), class_ids=["idle", "pack", "seal"],
        names_by_id={"idle": "空闲", "pack": "包装", "seal": "封装"},
        part_names=["零件"], required_parts={"pack": ["零件"], "seal": ["零件"]},
        yolo_conf_threshold=0.8, weights_sha256="test",
    ), confirm_seconds=confirm_seconds)


def _feed(session, label, frame, *, parts=False):
    return session.observe_frame(
        source_frame=frame, timestamp=frame / 10.0,
        detections=[{"label": "零件", "conf": 0.9}] if parts else [],
        hands=[], frame_w=10, frame_h=10, class_index=label,
    )


def test_trace_records_samples_confirm_count_and_event_bounds(tmp_path):
    path = configure_action_trace(tmp_path)
    classified = _session()
    classified._classify = lambda *_args: (1, "空闲:0.100,包装:0.800,封装:0.100")
    classified.observe_frame(
        source_frame=0, timestamp=0.0, detections=[], hands=[], frame_w=10, frame_h=10,
    )
    session = _session()
    for frame, label in enumerate([0] * 10 + [1] * 10 + [2] * 5):
        _feed(session, label, frame, parts=(label == 1))

    text = path.read_text(encoding="utf-8")
    assert "session_open" in text
    assert "probs=空闲:0.100,包装:0.800,封装:0.100" in text
    assert "open_len=5 confirm_frames=5" in text
    assert "event_reject reason=startup_first_run" in text
    assert "name=包装 start_frame=10 end_frame=20" in text
    assert "missing=-" in text


def test_trace_records_round_threshold_and_reject_reason(tmp_path):
    path = configure_action_trace(tmp_path)
    engine = StepSequenceEngine(step_class_names=["包装", "封装"], detection_updates=False)
    engine.start_round(action_min_source_frame=100)
    early = {
        "session_id": 1, "event_id": 1, "class_id": "pack", "name": "包装",
        "start_frame": 20, "end_frame": 40, "missing_parts": [],
    }
    engine.apply_action_event(early)
    assert engine.get_state().steps[0].status == StepStatus.WAITING
    accepted = {**early, "event_id": 2, "start_frame": 120, "missing_parts": ["网线"]}
    engine.apply_action_event(accepted)
    assert engine.get_state().steps[0].status == StepStatus.NG

    text = path.read_text(encoding="utf-8")
    assert "round_start round_id=1 min_source_frame=100" in text
    assert "reason=before_round_start" in text
    assert "min_source_frame=100" in text
    assert "reason=missing_parts" in text
    assert "missing=网线" in text
