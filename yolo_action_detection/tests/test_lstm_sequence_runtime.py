"""连续 LSTM 接入顺序判定。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(ROOT), str(SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from lstm_workbench.model import ActionLSTM
from lstm_workbench.model_registry import file_sha256
from lstm_workbench.service import resample_indices
from lstm_action.bundle import (
    bundle_directory_for_model_file,
    load_runtime_bundle,
    prepare_runtime_bundle,
    uses_lstm_sequence,
)
from lstm_action.stream import TimeBaseSampler, TimelineSession
from step_sequence.step_sequence_engine import RoundResult, StepSequenceEngine, StepStatus


class _Config:
    def __init__(self, **kwargs):
        self.roi_inspection_enabled = kwargs.get("roi", False)
        self.action_order_constraint_enabled = kwargs.get("order", True)
        self.lstm_action_enabled = kwargs.get("lstm", False)


def _contract(ids, yolo_sha, **extra):
    payload = {
        "class_ids": ids,
        "part_class_names": ["路由器"],
        "input_size": 131,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.0,
        "feature_code_version": "1",
        "network": "lstm",
        "train_unit": "timeline",
        "decode_version": 1,
        "time_base_fps": 10,
        "min_action_seconds": 1.0,
        "feature_mean": [0.0] * 131,
        "feature_std": [1.0] * 131,
        "yolo_conf_threshold": 0.8,
        "yolo_sha256": yolo_sha,
        "hands_sha256": "hands",
        "hands_model_complexity": 0,
        "required_parts": {class_id: ["路由器"] for class_id in ids if class_id != "idle"},
    }
    payload.update(extra)
    return payload


def _bundle(tmp_path: Path, ids=None, mapping=None, contract_extra=None):
    ids = ids or ["idle", "c-pack", "c-seal"]
    source = tmp_path / "run"
    source.mkdir(parents=True)
    torch.save(ActionLSTM(131, len(ids), dropout=0.0).state_dict(), source / "best_lstm.pt")
    yolo = tmp_path / "yolo.onnx"
    yolo.write_bytes(b"yolo")
    payload = _contract(ids, file_sha256(yolo), **(contract_extra or {}))
    (source / "contract.json").write_text(json.dumps(payload), encoding="utf-8")
    names = mapping or {"idle": "空闲", "c-pack": "包装", "c-seal": "封装"}
    dest = tmp_path / "bundle"
    prepare_runtime_bundle(source, dest, names)
    return dest, yolo


def _load(dest, yolo, names, counts=None):
    return load_runtime_bundle(
        dest,
        category_names=names,
        yolo_path=str(yolo),
        yolo_conf=0.8,
        hands_sha="hands",
        hands_complexity=0,
        step_counts=counts,
    )


def _feed(session: TimelineSession, labels, parts_at=None):
    events = []
    reports = []
    for index, label in enumerate(labels):
        report = session.observe_frame(
            source_frame=index,
            timestamp=index / 10.0,
            detections=(parts_at or {}).get(index, []),
            hands=[],
            frame_w=10,
            frame_h=10,
            class_index=label,
        )
        reports.append(report)
        events.extend(report["events"])
    return events, reports


def test_mode_follows_roi_and_order_not_the_old_switch():
    assert uses_lstm_sequence(_Config(roi=False, order=True, lstm=False)) is True
    assert uses_lstm_sequence(_Config(roi=False, order=False, lstm=True)) is False
    assert uses_lstm_sequence(_Config(roi=True, order=True, lstm=True)) is False


def test_bundle_maps_internal_ids_without_reordering_the_model(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    bundle = _load(dest, yolo, ["包装", "封装"])
    assert bundle.ids_by_name["包装"] == "c-pack"
    assert bundle.class_ids == ["idle", "c-pack", "c-seal"]
    assert file_sha256(dest / "best_lstm.pt") == file_sha256(tmp_path / "run" / "best_lstm.pt")


def test_bundle_refuses_a_guessed_or_incomplete_mapping(tmp_path: Path):
    source = tmp_path / "run"
    source.mkdir(parents=True)
    torch.save(ActionLSTM(131, 2, dropout=0.0).state_dict(), source / "best_lstm.pt")
    yolo = tmp_path / "yolo.onnx"
    yolo.write_bytes(b"yolo")
    (source / "contract.json").write_text(
        json.dumps(_contract(["idle", "c-pack"], file_sha256(yolo))),
        encoding="utf-8",
    )
    try:
        prepare_runtime_bundle(source, tmp_path / "out", None)
    except ValueError as exc:
        assert "映射" in str(exc)
    else:
        raise AssertionError("缺少映射时不应导出")


def test_bundle_rejects_mstcn_empty_parts_and_a_foreign_name(tmp_path: Path):
    dest, yolo = _bundle(tmp_path, contract_extra={"network": "mstcn"})
    try:
        _load(dest, yolo, ["包装", "封装"])
    except ValueError as exc:
        assert "MS-TCN" in str(exc) or "不支持" in str(exc)
    else:
        raise AssertionError("MS-TCN 不应加载")

    dest2, yolo2 = _bundle(tmp_path / "empty", contract_extra={"required_parts": {"c-pack": [], "c-seal": ["路由器"]}})
    try:
        _load(dest2, yolo2, ["包装", "封装"])
    except ValueError as exc:
        assert "零件" in str(exc)
    else:
        raise AssertionError("空零件要求不应放行")

    dest3, yolo3 = _bundle(tmp_path / "name")
    try:
        _load(dest3, yolo3, ["包装", "封装", "螺丝"])
    except ValueError as exc:
        assert "螺丝" in str(exc)
    else:
        raise AssertionError("多余名称不应通过")


def test_reversed_config_order_keeps_the_model_index(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    bundle = _load(dest, yolo, ["封装", "包装"])
    assert bundle.class_ids[1:] == ["c-pack", "c-seal"]
    assert bundle.ids_by_name["封装"] == "c-seal"


def test_chunked_lstm_matches_the_full_forward():
    model = ActionLSTM(4, 3, hidden_size=8, num_layers=2, dropout=0.0)
    model.eval()
    data = torch.randn(1, 17, 4)
    with torch.no_grad():
        full = model(data, all_steps=True)
        hidden = None
        pieces = []
        cursor = 0
        for size in (1, 4, 5, 7):
            logits, hidden = model.step(data[:, cursor:cursor + size], hidden)
            pieces.append(logits)
            cursor += size
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)


def test_fixed_rate_samples_match_the_workbench_indices():
    sampler = TimeBaseSampler(5)
    chosen = []
    for frame in range(20):
        slot = sampler.push(frame, frame / 10.0)
        if isinstance(slot, int):
            chosen.append(frame)
    assert chosen == [index for index in resample_indices(20, 10.0, 5.0)]


def test_a_repeated_frame_stops_the_stream(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    session.observe_frame(
        source_frame=1, timestamp=0.0, detections=[], hands=[], frame_w=8, frame_h=8, class_index=0,
    )
    report = session.observe_frame(
        source_frame=1, timestamp=0.1, detections=[], hands=[], frame_w=8, frame_h=8, class_index=0,
    )
    assert report["fault"]
    assert report["events"] == []


def test_a_gap_longer_than_two_slots_keeps_running(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    session.observe_frame(
        source_frame=1, timestamp=0.0, detections=[], hands=[], frame_w=8, frame_h=8, class_index=0,
    )
    report = session.observe_frame(
        source_frame=2, timestamp=0.5, detections=[], hands=[], frame_w=8, frame_h=8, class_index=0,
    )
    assert report["fault"] == ""
    assert report["events"] == []
    assert session.sampler.selected_frame == 2


def test_complete_pack_waits_for_a_stable_successor_and_keeps_parts(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    router = [{"label": "路由器", "conf": 0.9, "box": (0, 0, 2, 2)}]
    late = [{"label": "路由器", "conf": 0.9, "box": (0, 0, 2, 2)}]
    labels = [0] * 10 + [1] * 10 + [0] * 10
    parts = {index: router for index in range(10, 16)}
    parts[22] = late
    events, reports = _feed(session, labels, parts)
    assert [item["name"] for item in events] == ["包装"]
    assert events[0]["parts"] == ["路由器"]
    assert events[0]["missing_parts"] == []
    assert events[0]["end_frame"] <= 20
    assert any(item["phase"] == "等待结束" for item in reports)


def test_direct_switch_emits_one_pack_and_a_blip_does_not(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    labels = [0] * 10 + [1] * 12 + [2] * 3 + [1] * 12 + [2] * 10 + [0] * 10
    events, _reports = _feed(session, labels)
    names = [item["name"] for item in events]
    assert names.count("包装") == 1
    assert "封装" in names


def test_startup_mid_action_is_not_a_pass(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    events, _reports = _feed(session, [1] * 10 + [0] * 10)
    assert events == []


def test_missing_depth_is_a_fault_not_a_silent_zero(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    report = session.observe_frame(
        source_frame=1,
        timestamp=0.0,
        detections=[],
        hands=[{"points": [[0, 0]] * 21}],
        frame_w=8,
        frame_h=8,
    )
    assert "深度" in report["fault"]


def test_yolo_hits_do_not_pass_a_step_while_detection_updates_are_off():
    engine = StepSequenceEngine(step_class_names=["包装", "封装"], step_counts=[1, 1], detection_updates=False)

    class _Det:
        def __init__(self, label):
            self.label = label

    state = engine.update([_Det("包装")])
    assert state.steps[0].status == StepStatus.WAITING


def test_action_events_pass_only_the_current_step_and_reject_duplicates():
    engine = StepSequenceEngine(step_class_names=["包装", "封装"], step_counts=[1, 1], detection_updates=False)
    event = {
        "session_id": 1, "event_id": 1, "class_id": "c-pack", "name": "包装",
        "end_frame": 20, "decision_frame": 30, "parts": ["路由器"], "missing_parts": [],
    }
    state = engine.apply_action_event(event)
    assert state.steps[0].status == StepStatus.PASS
    assert state.steps[1].status == StepStatus.WAITING
    again = engine.apply_action_event(event)
    assert again.steps[1].status == StepStatus.WAITING
    assert again.round_result == RoundResult.IN_PROGRESS


def test_missing_parts_marks_that_step_red_and_continues():
    engine = StepSequenceEngine(step_class_names=["包装", "封装"], step_counts=[1, 1], detection_updates=False)
    state = engine.apply_action_event({
        "session_id": 1, "event_id": 1, "class_id": "c-pack", "name": "包装",
        "end_frame": 100, "decision_frame": 112, "parts": [], "missing_parts": ["路由器"],
    })
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.steps[0].status == StepStatus.NG
    assert state.steps[1].status == StepStatus.WAITING
    assert state.action_boundary_frame == 100
    assert state.action_decision_frame == 112
    state = engine.apply_action_event({
        "session_id": 1, "event_id": 2, "class_id": "c-seal", "name": "封装",
        "parts": ["路由器"], "missing_parts": [],
    })
    assert state.steps[1].status == StepStatus.PASS
    assert state.round_result == RoundResult.ACTION_NG
    assert "路由器" in state.action_ng_reason


def test_a_future_action_before_the_first_pass_ends_the_round_without_waiting_for_it():
    engine = StepSequenceEngine(step_class_names=["包装", "封装"], step_counts=[1, 1], detection_updates=False)
    state = engine.apply_action_event({
        "session_id": 1, "event_id": 2, "class_id": "c-seal", "name": "封装",
        "end_frame": 40, "decision_frame": 50, "parts": ["路由器"], "missing_parts": [],
    })
    assert state.round_result == RoundResult.ACTION_NG
    assert state.steps[0].status == StepStatus.NG
    assert "未检测到步骤1(包装)" in state.steps[0].ng_reason
    assert state.steps[1].status == StepStatus.NG


def test_a_skipped_middle_step_is_not_detected_and_the_last_step_still_closes_the_round():
    engine = StepSequenceEngine(step_class_names=["包装", "封装", "检测"], step_counts=[1, 1, 1], detection_updates=False)
    engine.apply_action_event({
        "session_id": 1, "event_id": 1, "class_id": "c-pack", "name": "包装",
        "parts": ["路由器"], "missing_parts": [],
    })
    state = engine.apply_action_event({
        "session_id": 1, "event_id": 2, "class_id": "c-check", "name": "检测",
        "parts": ["路由器"], "missing_parts": [],
    })
    assert state.steps[0].status == StepStatus.PASS
    assert state.steps[1].status == StepStatus.NG
    assert state.steps[2].status == StepStatus.NG
    assert state.round_result == RoundResult.ACTION_NG
    assert "未检测到步骤2(封装)" in state.action_ng_reason
    assert "检测" in state.action_ng_reason


def test_action_quantity_must_be_one(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    try:
        _load(dest, yolo, ["包装", "封装"], counts=[1, 2])
    except ValueError as exc:
        assert "数量必须为 1" in str(exc)
    else:
        raise AssertionError("数量 2 不应通过")


def test_hand_depth_matches_the_workbench_row_and_empty_hands_stay_zero():
    from lstm_workbench.features import build_feature_row
    from lstm_action.stream import _hands_xyz

    points = [[float(index), float(20 - index)] for index in range(21)]
    depths = [0.25 + index * 0.01 for index in range(21)]
    xyz = [(points[index][0], points[index][1], depths[index]) for index in range(21)]
    direct = build_feature_row([], [xyz], 100, 80, ["路由器"], 0.8)
    wrapped = build_feature_row(
        [],
        _hands_xyz([{"points": points, "depths": depths}]),
        100,
        80,
        ["路由器"],
        0.8,
    )
    assert direct == wrapped
    assert wrapped[7] == 0.25
    zeros = build_feature_row([], [], 100, 80, ["路由器"], 0.8)
    assert zeros[5:] == [0.0] * 126


def test_low_confidence_and_the_next_action_do_not_change_this_segment(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    strong = [{"label": "路由器", "conf": 0.95, "box": (0, 0, 1, 1)}]
    weak = [{"label": "路由器", "conf": 0.2, "box": (0, 0, 1, 1)}]
    labels = [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    parts = {10: strong, 18: weak}
    events, _reports = _feed(session, labels, parts)
    assert [item["missing_parts"] for item in events] == [[], ["路由器"]]
    assert events[0]["parts"] == ["路由器"]
    assert "路由器" not in events[1]["parts"]


def test_stop_does_not_finish_an_open_action(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    _feed(session, [0] * 10 + [1] * 8)
    assert session.stop()["events"] == []


def test_reset_requires_a_new_boundary_before_the_next_pass(tmp_path: Path):
    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    finished, _reports = _feed(session, [0] * 10 + [1] * 10 + [0] * 10)
    assert [item["name"] for item in finished] == ["包装"]
    session.reset()
    truncated, _reports = _feed(session, [1] * 10 + [0] * 10)
    assert truncated == []


def test_closed_prefix_matches_the_workbench_segments(tmp_path: Path):
    from lstm_workbench.check_clip import absorb_short_segments

    dest, yolo = _bundle(tmp_path)
    session = TimelineSession(_load(dest, yolo, ["包装", "封装"]))
    labels = [0] * 10 + [1] * 15 + [2] * 3 + [1] * 12 + [0] * 10
    events, _reports = _feed(session, labels)
    raw = []
    start = 0
    for index, label in enumerate(labels + [-1]):
        if index == len(labels) or label != labels[start]:
            raw.append({
                "class_id": str(labels[start]),
                "start_frame": start,
                "end_frame": index,
            })
            start = index
    absorbed = absorb_short_segments(raw, session.min_frames, len(labels), 10.0)
    assert [(item["class_id"], item["start_frame"], item["end_frame"]) for item in absorbed] == [
        ("0", 0, 10),
        ("1", 10, 40),
        ("0", 40, 50),
    ]
    assert [(item["name"], item["start_frame"], item["end_frame"]) for item in events] == [
        ("包装", 10, 40),
    ]


def test_a_completed_round_drops_the_event_and_a_new_round_ignores_the_old_id():
    engine = StepSequenceEngine(
        step_class_names=["包装", "封装"], step_counts=[1, 1], detection_updates=False,
    )
    first = {
        "session_id": 1, "event_id": 1, "class_id": "c-pack", "name": "包装",
        "parts": ["路由器"], "missing_parts": [],
    }
    engine.apply_action_event(first)
    engine._round_result = RoundResult.PASS
    held = engine.apply_action_event({
        "session_id": 1, "event_id": 2, "class_id": "c-seal", "name": "封装",
        "parts": ["路由器"], "missing_parts": [],
    })
    assert held.round_result == RoundResult.PASS
    engine.start_round()
    replay = engine.apply_action_event(first)
    assert replay.steps[0].status == StepStatus.WAITING


def test_missing_part_evidence_records_the_boundary_and_the_decision_frame(tmp_path: Path):
    import numpy as np

    from detection_logging.fail_evidence import FailEvidenceContext, FailEvidenceSaver

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    saver = FailEvidenceSaver(base_dir=tmp_path)
    folder = saver.save_fail_event(frame, FailEvidenceContext(
        round_id=3,
        action_ng_step=0,
        step_name="包装",
        source_frame_id=100,
        boundary_frame=100,
        decision_frame=112,
        seen_parts=["纸盒"],
        missing_parts=["路由器"],
        ng_reason="动作包装缺少路由器；当前等待包装",
        model_sha256="abc",
    ))
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_frame_id"] == 100
    assert metadata["boundary_frame"] == 100
    assert metadata["decision_frame"] == 112
    assert metadata["missing_parts"] == ["路由器"]
    assert metadata["model_sha256"] == "abc"


def test_a_bundle_trained_without_hands_does_not_probe_hands(tmp_path, monkeypatch):
    dest, yolo = _bundle(tmp_path)
    contract_path = dest / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update({
        "hands_in_training": False,
        "hands_sha256": "",
        "hands_model_complexity": None,
        "input_size": 5,
        "feature_mean": [0.0] * 5,
        "feature_std": [1.0] * 5,
    })
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    torch.save(ActionLSTM(5, 3, dropout=0.0).state_dict(), dest / "best_lstm.pt")
    manifest = json.loads((dest / "runtime_manifest.json").read_text(encoding="utf-8"))
    from lstm_workbench.model_registry import file_sha256 as sha
    manifest["weights_sha256"] = sha(dest / "best_lstm.pt")
    manifest["contract_sha256"] = sha(contract_path)
    (dest / "runtime_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def boom():
        raise AssertionError("不用手部的模型包不该去加载手部")

    monkeypatch.setattr("lstm_workbench.model_registry.probe_hands", boom)
    bundle = _load(dest, yolo, ["包装", "封装"])
    assert bundle.hands_in_training is False


def test_selecting_the_weight_file_uses_its_bundle_directory(tmp_path: Path):
    dest, _yolo = _bundle(tmp_path)
    assert bundle_directory_for_model_file(dest / "best_lstm.pt") == dest
    assert bundle_directory_for_model_file(dest) == dest
    try:
        bundle_directory_for_model_file(tmp_path / "missing.pt")
    except ValueError as exc:
        assert "best_lstm.pt" in str(exc)
    else:
        raise AssertionError("不存在的模型文件应该拒绝")
