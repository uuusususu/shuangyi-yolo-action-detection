"""验证视频的分段：凑满窗口之前没有类别，之后连续同一类合成一段，并标出开始和结束。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lstm_workbench.check_clip import (
    build_segments,
    draft_annotations,
    ensure_long_enough,
    frame_overlay,
    grow_complete_actions,
    indices_for_model,
    save_check_video,
    slide_predictions,
)
from lstm_workbench.dataset import segment_spans
from lstm_workbench.timeline import ClipAnnotation, Segment
from lstm_workbench.model import ActionLSTM


def test_check_video_cannot_escape_the_run_directory(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    saved = save_check_video(run, "..\\evil.mp4", b"video", replace=True)
    assert saved.parent == run / "check" / "media"
    assert saved.name == "evil.mp4"
    assert not (tmp_path / "evil.mp4").exists()
    again = save_check_video(run, "next.mp4", b"next", replace=True)
    assert again.read_bytes() == b"next"
    assert not saved.exists()


def test_a_picture_is_not_a_check_video(tmp_path):
    with pytest.raises(ValueError, match="请上传视频"):
        save_check_video(tmp_path, "still.jpg", b"jpg", replace=True)


def test_frame_overlay_stores_the_box_corners_and_the_hand():
    overlay = frame_overlay(
        [{"label": "纸盒", "polygon": [(0, 0), (10, 0), (10, 20), (0, 20)]}],
        [[(float(i), float(i), 0.0) for i in range(21)]],
        100,
        50,
    )
    assert overlay["boxes"][0]["label"] == "纸盒"
    assert overlay["boxes"][0]["poly"][1] == [0.1, 0.0]
    assert len(overlay["hands"][0]) == 21
    assert frame_overlay([], [], 10, 10) is None


def test_a_labeled_span_is_one_sample_not_a_pile_of_windows():
    clip = ClipAnnotation(
        "c", 80, 10, 8, 8,
        [Segment("s", "pack", 10, 70)],
    )
    spans = segment_spans(clip, window=20)
    assert spans == [("pack", 10, 70)]


def test_a_short_flip_does_not_become_its_own_action():
    def classify(start, end):
        # 整段从 0 读到 12 仍是 0；从 12 起整段是 1。中间没有单独的几帧动作。
        if start < 12 and end <= 12:
            return 0, 0.9
        return 1, 0.8

    actions = grow_complete_actions(20, min_frames=4, classify=classify)
    assert [(item["index"], item["start"], item["end"]) for item in actions] == [
        (0, 0, 12),
        (1, 12, 20),
    ]


def test_prelabel_follows_the_model_clock_not_the_other_clips():
    """模型按训练时的帧率读帧。当前片子不同也不去问工程里其他片子。"""
    same = indices_for_model({"frame_count": 50, "fps": 10.277}, {"time_base_fps": 10.277})
    assert same == list(range(50))
    slower = indices_for_model({"frame_count": 100, "fps": 6.25}, {"time_base_fps": 10.277277})
    assert len(slower) == round(100 / 6.25 * 10.277277)
    assert slower[0] == 0
    assert slower[-1] <= 99


def test_a_short_flicker_is_merged_back_into_the_previous_action():
    from lstm_workbench.check_clip import absorb_short_segments

    segments = [
        {"class_id": "pack", "name": "包装", "start_frame": 0, "end_frame": 30, "prob": 0.9},
        {"class_id": "seal", "name": "封装", "start_frame": 30, "end_frame": 35, "prob": 0.6},
        {"class_id": "pack", "name": "包装", "start_frame": 35, "end_frame": 80, "prob": 0.8},
    ]
    got = absorb_short_segments(segments, min_frames=12, frame_count=80, fps=10)
    assert [(item["class_id"], item["start_frame"], item["end_frame"]) for item in got] == [
        ("pack", 0, 80),
    ]


def test_draft_drops_idle_and_cuts_out_the_excluded_span():
    segments = [
        {"class_id": "idle", "start_frame": 0, "end_frame": 10},
        {"class_id": "pack", "start_frame": 10, "end_frame": 40, "part_check": "fail", "missing_parts": ["网线"]},
        {"class_id": "gone", "start_frame": 40, "end_frame": 50},
    ]
    got = draft_annotations(segments, [{"start": 20, "end": 25}], {"pack"})
    assert [(item["class_id"], item["start"], item["end"], item.get("missing_parts")) for item in got] == [
        ("pack", 10, 20, ["网线"]),
        ("pack", 25, 40, ["网线"]),
    ]


def test_a_clip_shorter_than_the_window_is_refused():
    with pytest.raises(ValueError, match="凑不满"):
        ensure_long_enough(8, 10, 20, 1.92, 10)


def test_slide_waits_until_the_window_is_full():
    torch.manual_seed(0)
    model = ActionLSTM(2, 2, hidden_size=4, num_layers=1, dropout=0.0)
    rows = np.zeros((5, 2), dtype=np.float32)
    labels = slide_predictions(rows, model, ["idle", "act"], 3, torch.device("cpu"))
    assert labels[0] is None
    assert labels[1] is None
    assert labels[2]["class_id"] in {"idle", "act"}
    assert 0.0 <= labels[2]["prob"] <= 1.0
    assert labels[4]["class_id"] in {"idle", "act"}


def test_segments_record_where_each_action_starts_and_ends():
    source = [0, 3, 6, 9, 12]
    labels = [
        None,
        None,
        {"class_id": "pack", "prob": 0.9},
        {"class_id": "pack", "prob": 0.7},
        {"class_id": "idle", "prob": 0.8},
    ]
    names = {"pack": "包装", "idle": "空闲"}
    warming, segments = build_segments(source, labels, frame_count=15, fps=10, names=names)
    assert warming == {"start_frame": 0, "end_frame": 6, "start_sec": 0.0, "end_sec": 0.6}
    assert [(item["name"], item["start_frame"], item["end_frame"]) for item in segments] == [
        ("包装", 6, 12),
        ("空闲", 12, 15),
    ]
    assert segments[0]["start_sec"] == pytest.approx(0.6)
    assert segments[0]["end_sec"] == pytest.approx(1.2)
    assert segments[1]["end_sec"] == pytest.approx(1.5)


def test_absorbed_short_segments_drive_the_frame_index():
    import torch

    from lstm_workbench.check_clip import decode_timeline

    class Scripted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch, lengths=None, all_steps=False):
            steps = int(batch.shape[1])
            logits = torch.full((1, steps, 3), -8.0)
            for index in range(steps):
                choice = 2 if 20 <= index < 23 else 1
                logits[0, index, choice] = 8.0
            return logits

    contract = {
        "time_base_fps": 10.0,
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "min_action_seconds": 1.0,
        "class_ids": ["idle", "pack", "seal"],
        "required_parts": {},
    }
    model = Scripted()
    model.eval()
    segments, frame_index = decode_timeline(
        np.zeros((40, 2), dtype=np.float32), 10.0, model, contract, [],
        {"idle": "空闲", "pack": "包装", "seal": "封装"},
    )
    actions = [item for item in segments if item["class_id"] != "idle"]
    assert [item["class_id"] for item in actions] == ["pack"]
    assert set(frame_index.tolist()) == {1}


def test_a_tiny_contract_std_does_not_amplify_the_feature():
    import torch

    from lstm_workbench.check_clip import decode_timeline

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.seen = None

        def forward(self, batch, lengths=None, all_steps=False):
            self.seen = batch.detach().cpu().numpy()
            return torch.zeros(1, int(batch.shape[1]), 2)

    model = Capture()
    model.eval()
    decode_timeline(
        np.ones((4, 1), dtype=np.float32),
        10.0,
        model,
        {
            "time_base_fps": 10.0,
            "feature_mean": [0.0],
            "feature_std": [1e-4],
            "min_action_seconds": 1.0,
            "class_ids": ["idle", "pack"],
            "required_parts": {},
        },
        [],
        {"idle": "空闲", "pack": "包装"},
    )
    assert model.seen[0, 0, 0] == pytest.approx(1.0)


def test_source_segments_keep_the_part_gate():
    from lstm_workbench.check_clip import _segments_on_source

    shown = _segments_on_source(
        [{
            "class_id": "pack", "name": "包装", "start_frame": 0, "end_frame": 2,
            "prob": 0.9, "part_check": "fail", "missing_parts": ["网线"],
        }],
        [0, 1, 2],
        3,
        10.0,
        {"pack": "包装"},
    )
    assert shown[0]["part_check"] == "fail"
    assert shown[0]["missing_parts"] == ["网线"]


def test_a_saved_project_clip_is_returned_with_the_check(tmp_path):
    import json

    from lstm_workbench.check_clip import check_status

    target = tmp_path / "check" / "clips" / "c1"
    target.mkdir(parents=True)
    (target / "result.json").write_text(
        json.dumps({"ready": True, "project_clip": True, "clip_id": "c1", "checked_at": 10}),
        encoding="utf-8",
    )
    status = check_status(tmp_path)
    assert status["saved_clip"]["clip_id"] == "c1"


def test_a_timeline_contract_is_not_decoded_by_the_old_path():
    model = ActionLSTM(2, 1, hidden_size=4, num_layers=1, dropout=0.0)
    from lstm_workbench.check_clip import segments_for_contract

    with pytest.raises(ValueError, match="decode_timeline"):
        segments_for_contract(
            np.zeros((8, 2), dtype=np.float32),
            model,
            ["idle"],
            torch.device("cpu"),
            {"train_unit": "timeline", "label_scope": "frame", "window": 4},
            list(range(8)),
            8,
            10.0,
            {"idle": "空闲"},
        )
