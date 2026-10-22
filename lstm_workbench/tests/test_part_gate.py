"""标注里出现过的零件，验证时缺了就失败。"""
from __future__ import annotations

import numpy as np

from lstm_workbench.part_gate import apply_part_gate, required_parts_for_split
from lstm_workbench.timeline import ClipAnnotation, Segment


class _Split:
    def __init__(self, train, val):
        self.train = train
        self.val = val


def test_one_flash_frame_does_not_become_a_required_part():
    clip = ClipAnnotation("c", 30, 10, 8, 8, [Segment("pack", "pack", 0, 30)])
    features = np.zeros((30, 10), dtype=np.float32)
    features[4, 0] = 0.9
    got = required_parts_for_split(
        [clip],
        {"c": features},
        _Split([("c", "pack")], []),
        20,
        ["充电器", "网线"],
    )
    assert got == {"pack": []}


def test_twenty_percent_of_a_segment_counts_and_just_under_does_not():
    from lstm_workbench.part_gate import parts_stable_in_block

    exactly = np.zeros((5, 5), dtype=np.float32)
    exactly[0, 0] = 0.8
    assert parts_stable_in_block(exactly, ["充电器"]) == {"充电器"}
    under = np.zeros((6, 5), dtype=np.float32)
    under[0, 0] = 0.8
    assert parts_stable_in_block(under, ["充电器"]) == set()


def test_half_the_segments_keep_a_part_and_fewer_do_not():
    from lstm_workbench.part_gate import required_parts_from_labeled_rows

    def rows(present: bool, length: int = 10) -> np.ndarray:
        block = np.zeros((length, 5), dtype=np.float32)
        if present:
            block[:, 0] = 0.9
        return block

    labels = np.ones(10, dtype=np.int64)
    kept = required_parts_from_labeled_rows(
        [(rows(True), labels), (rows(False), labels)],
        ["idle", "pack"],
        ["充电器"],
    )
    assert kept == {"pack": ["充电器"]}
    dropped = required_parts_from_labeled_rows(
        [(rows(True), labels), (rows(False), labels), (rows(False), labels)],
        ["idle", "pack"],
        ["充电器"],
    )
    assert dropped == {"pack": []}


def test_idle_does_not_record_parts_and_a_class_with_nothing_stays_empty():
    clip = ClipAnnotation(
        "c", 80, 10, 8, 8,
        [Segment("seal", "seal", 0, 40), Segment("pack", "pack", 40, 80)],
    )
    features = np.zeros((80, 10), dtype=np.float32)
    features[40:80, 5] = 0.4
    got = required_parts_for_split(
        [clip],
        {"c": features},
        _Split([("c", "seal"), ("c", "pack")], []),
        20,
        ["充电器", "网线"],
    )
    assert got["seal"] == []
    assert got["pack"] == ["网线"]


def test_missing_part_fails_the_action_and_names_it():
    rows = np.zeros((10, 10), dtype=np.float32)
    rows[1, 0] = 0.8
    segments = [{"class_id": "pack", "start_frame": 0, "end_frame": 10}]
    apply_part_gate(segments, rows, ["充电器", "网线"], {"pack": ["充电器", "网线", "插头"]})
    assert segments[0]["part_check"] == "fail"
    assert segments[0]["missing_parts"] == ["网线", "插头"]


def test_a_part_seen_once_passes_and_idle_is_not_gated():
    rows = np.zeros((6, 5), dtype=np.float32)
    rows[3, 0] = 0.2
    segments = [
        {"class_id": "pack", "start_frame": 0, "end_frame": 6},
        {"class_id": "idle", "start_frame": 0, "end_frame": 6},
    ]
    apply_part_gate(segments, rows, ["充电器"], {"pack": ["充电器"]})
    assert segments[0]["part_check"] == "pass"
    assert segments[0]["missing_parts"] == []
    assert "part_check" not in segments[1]


def test_old_runs_without_required_parts_are_left_alone():
    segments = [{"class_id": "pack", "start_frame": 0, "end_frame": 2}]
    apply_part_gate(segments, np.zeros((2, 5), dtype=np.float32), ["充电器"], None)
    assert "part_check" not in segments[0]
    apply_part_gate(segments, np.zeros((2, 5), dtype=np.float32), ["充电器"], {"pack": []})
    assert "part_check" not in segments[0]
