"""首类别区域检查配置和引擎派生测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from config import ConfigManager
from pcb_inspection.models import PcbInspectionConfig, PcbResult, PcbStatus, SlotStatus
from pcb_inspection.engine import MultiPcbInspectionEngine
from yolo_runtime.yolo_result_models import ObbDetection


def _make_parent(track_id=1, x=100, y=100, w=200, h=200, label="parent"):
    return ObbDetection(
        class_id=0, label=label, conf=0.9, track_id=track_id,
        polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        box=(x, y, x + w, y + h), center=(x + w / 2, y + h / 2),
    )


def _make_child(label, x, y, w=20, h=20):
    return ObbDetection(
        class_id=1, label=label, conf=0.9, track_id=None,
        polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        box=(x, y, x + w, y + h), center=(x + w / 2, y + h / 2),
    )


def test_first_category_region_config_defaults_disabled():
    c = ConfigManager()
    assert c.first_category_region_check_enabled is False


def test_first_category_region_derives_parent_and_children():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    assert c.get_first_category_region_classes() == ("parent", ["pad", "screw"])


def test_first_category_region_requires_parent_and_child():
    c = ConfigManager()
    c.first_category_region_check_enabled = True
    c.category_names = ["parent", "", "", "", "", ""]
    with pytest.raises(ValueError, match="子类别"):
        c.validate()


def test_first_category_region_builds_pcb_config_from_category_sequence():
    c = ConfigManager()
    c.first_category_region_check_enabled = True
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.validate()
    derived = PcbInspectionConfig.from_first_category_config(c)
    assert derived.pcb_class_name == "parent"
    assert derived.component_class_names == ["pad", "screw"]


def test_first_category_region_derives_child_required_counts():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.category_counts = [1, 4, 2, 1, 1, 1]

    derived = PcbInspectionConfig.from_first_category_config(c)

    assert derived.component_required_counts == {"pad": 4, "screw": 2}


def test_first_category_region_passes_when_children_inside_parent():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.action_ng_stable_frames = 1
    derived = PcbInspectionConfig.from_first_category_config(c)
    engine = MultiPcbInspectionEngine(derived)
    results = engine.update([
        _make_parent(),
        _make_child("pad", 120, 120),
        _make_child("screw", 160, 120),
    ], image_size=(1000, 1000))
    assert len(results) == 1
    assert results[0].result == PcbResult.PASS


def test_first_category_region_child_outside_parent_stays_waiting():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.action_ng_stable_frames = 1
    derived = PcbInspectionConfig.from_first_category_config(c)
    engine = MultiPcbInspectionEngine(derived)
    results = engine.update([
        _make_parent(),
        _make_child("pad", 120, 120),
        _make_child("screw", 500, 500),
    ], image_size=(1000, 1000))
    assert len(results) == 1
    assert results[0].result == PcbResult.FAIL
    assert engine.pcb_states[1].status == PcbStatus.FAIL_RETRY_WAITING
    assert engine.pcb_states[1].last_slot_states["screw"].observed_count == 0


@pytest.mark.parametrize(
    ("observed_count", "expected_result"),
    [(3, PcbResult.FAIL), (4, PcbResult.PASS), (5, PcbResult.FAIL)],
)
def test_first_category_region_requires_exact_same_frame_quantity(observed_count, expected_result):
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 1
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))

    results = engine.update(
        [_make_parent()] + [_make_child("pad", 120 + index * 20, 120) for index in range(observed_count)],
        image_size=(1000, 1000),
    )

    assert len(results) == 1
    assert results[0].result == expected_result
    assert results[0].observed_counts == {"pad": observed_count}
    assert results[0].required_counts == {"pad": 4}


def test_onnx_parent_without_track_id_passes_with_four_same_frame_children():
    c = ConfigManager()
    c.category_names = ["pcb", "sanreidian", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 1
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    detections = [_make_parent(track_id=None, label="pcb")] + [
        _make_child("sanreidian", 120 + index * 30, 120)
        for index in range(4)
    ]

    results = engine.update(detections, image_size=(1000, 1000))

    assert len(results) == 1
    assert results[0].result == PcbResult.PASS
    assert results[0].track_id < 0
    assert results[0].observed_counts == {"sanreidian": 4}
    assert engine.update(detections, image_size=(1000, 1000)) == []


def test_first_category_region_quantity_does_not_accumulate_across_frames():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 3
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))

    two_children = [_make_child("pad", 120, 120), _make_child("pad", 160, 120)]
    assert engine.update([_make_parent()] + two_children, image_size=(1000, 1000)) == []
    assert engine.update([_make_parent()] + two_children, image_size=(1000, 1000)) == []
    results = engine.update(
        [_make_parent()] + [_make_child("pad", 120 + index * 20, 120) for index in range(4)],
        image_size=(1000, 1000),
    )

    assert len(results) == 1
    assert results[0].result == PcbResult.PASS
    assert results[0].observed_counts == {"pad": 4}


def test_first_category_region_zero_child_observation_fails_after_stability():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 3
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))

    assert engine.update([_make_parent()], image_size=(1000, 1000)) == []
    assert engine.update([_make_parent()], image_size=(1000, 1000)) == []
    results = engine.update([_make_parent()], image_size=(1000, 1000))

    assert len(results) == 1
    assert results[0].result == PcbResult.FAIL
    assert results[0].observed_counts == {"pad": 0}
    state = engine.pcb_states[1]
    assert state.status == PcbStatus.FAIL_RETRY_WAITING
    assert state.consecutive_fail == 0
    assert state.mismatch_streaks["pad"] == 0
    assert state.last_slot_states["pad"].observed_count == 0


def test_first_category_region_correct_quantity_requires_stability_and_latches():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 2, 1, 1, 1, 1]
    c.action_pass_stable_frames = 2
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    detections = [
        _make_parent(),
        _make_child("pad", 120, 120),
        _make_child("pad", 160, 120),
    ]

    assert engine.update(detections, image_size=(1000, 1000)) == []
    assert engine.pcb_states[1].last_slot_states["pad"].status == SlotStatus.MATCHING
    second = engine.update(detections, image_size=(1000, 1000))
    assert len(second) == 1
    assert second[0].result == PcbResult.PASS
    assert engine.update([_make_parent()], image_size=(1000, 1000)) == []
    assert engine.pcb_states[1].last_slot_states["pad"].status == SlotStatus.COMPLETED


def test_first_category_region_children_complete_across_different_frames():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.category_counts = [1, 1, 1, 1, 1, 1]
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))

    assert engine.update(
        [_make_parent(), _make_child("pad", 120, 120)],
        image_size=(1000, 1000),
    ) == []
    results = engine.update(
        [_make_parent(), _make_child("screw", 160, 120)],
        image_size=(1000, 1000),
    )

    assert len(results) == 1
    assert results[0].result == PcbResult.PASS
    assert results[0].slot_states == {"pad": True, "screw": True}


@pytest.mark.parametrize("observed_count", [3, 5])
def test_first_category_region_quantity_ng_is_retriable_and_can_recover(observed_count):
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 2
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    mismatch = [_make_parent()] + [
        _make_child("pad", 120 + index * 20, 120)
        for index in range(observed_count)
    ]

    assert engine.update(mismatch, image_size=(1000, 1000)) == []
    results = engine.update(mismatch, image_size=(1000, 1000))
    assert len(results) == 1
    assert results[0].result == PcbResult.FAIL
    assert engine.pcb_states[1].status == PcbStatus.FAIL_RETRY_WAITING

    corrected = [_make_parent()] + [
        _make_child("pad", 120 + index * 20, 120)
        for index in range(4)
    ]
    recovered = engine.update(corrected, image_size=(1000, 1000))
    assert len(recovered) == 1
    assert recovered[0].result == PcbResult.PASS
    assert engine.pcb_states[1].status == PcbStatus.PASS_LATCHED
    assert engine.pcb_states[1].result == PcbResult.PASS


def test_first_category_region_fail_preserves_completed_steps_and_resets_mismatch():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.category_counts = [1, 1, 2, 1, 1, 1]
    c.action_ng_stable_frames = 2
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    partial = [_make_parent(), _make_child("pad", 120, 120), _make_child("screw", 160, 120)]

    assert engine.update(partial, image_size=(1000, 1000)) == []
    failed = engine.update(partial, image_size=(1000, 1000))

    assert len(failed) == 1
    assert failed[0].result == PcbResult.FAIL
    state = engine.pcb_states[1]
    assert "pad" in state.completed_classes
    assert state.mismatch_streaks["screw"] == 0
    assert state.status == PcbStatus.FAIL_RETRY_WAITING

    corrected = [
        _make_parent(),
        _make_child("pad", 120, 120),
        _make_child("screw", 160, 120),
        _make_child("screw", 190, 120),
    ]
    recovered = engine.update(corrected, image_size=(1000, 1000))

    assert len(recovered) == 1
    assert recovered[0].result == PcbResult.PASS
    assert engine.pcb_states[1].status == PcbStatus.PASS_LATCHED


def test_first_category_region_same_fail_signature_emits_each_new_attempt():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 1
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    mismatch = [_make_parent()] + [
        _make_child("pad", 120 + index * 20, 120)
        for index in range(3)
    ]

    first = engine.update(mismatch, image_size=(1000, 1000))
    second = engine.update(mismatch, image_size=(1000, 1000))

    assert len(first) == 1
    assert first[0].result == PcbResult.FAIL
    assert len(second) == 1
    assert second[0].result == PcbResult.FAIL
    assert first[0].attempt_id != second[0].attempt_id
    assert first[0].is_new_fail_signature is True
    assert second[0].is_new_fail_signature is False


def test_first_category_region_failed_id_does_not_block_new_parent():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 1
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))

    failed = engine.update(
        [_make_parent(track_id=1, x=100)]
        + [_make_child("pad", 120 + index * 20, 120) for index in range(3)],
        image_size=(1000, 1000),
    )
    assert len(failed) == 1 and failed[0].track_id == 1

    next_round = engine.update(
        [_make_parent(track_id=1, x=100)]
        + [_make_parent(track_id=2, x=400)]
        + [_make_child("pad", 420 + index * 20, 120) for index in range(4)],
        image_size=(1000, 1000),
    )

    assert [result.track_id for result in next_round] == [2]
    assert next_round[0].result == PcbResult.PASS


def test_two_visible_parents_are_judged_as_two_separate_pass_rounds():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    detections = (
        [_make_parent(track_id=1, x=100)]
        + [_make_child("pad", 120 + index * 20, 120) for index in range(4)]
        + [_make_parent(track_id=2, x=400)]
        + [_make_child("pad", 420 + index * 20, 120) for index in range(4)]
    )

    results = engine.update(detections, image_size=(1000, 1000))

    assert [result.track_id for result in results] == [1, 2]
    assert [result.result for result in results] == [PcbResult.PASS, PcbResult.PASS]
    assert engine.update(detections, image_size=(1000, 1000)) == []


def test_waiting_parent_does_not_accumulate_zero_count_until_active():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.action_ng_stable_frames = 3
    c.round_cooldown_seconds = 0.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    first_pass_second_empty = (
        [_make_parent(track_id=1, x=100)]
        + [_make_child("pad", 120 + index * 20, 120) for index in range(4)]
        + [_make_parent(track_id=2, x=400)]
    )

    first = engine.update(first_pass_second_empty, image_size=(1000, 1000))

    assert len(first) == 1
    assert first[0].track_id == 1
    assert engine.pcb_states[2].status == PcbStatus.ACTIVE
    assert engine.pcb_states[2].consecutive_fail == 1

    assert engine.update(first_pass_second_empty, image_size=(1000, 1000)) == []
    failed = engine.update(first_pass_second_empty, image_size=(1000, 1000))
    assert len(failed) == 1
    assert failed[0].track_id == 2
    assert failed[0].result == PcbResult.FAIL
    assert failed[0].observed_counts == {"pad": 0}


def test_round_cooldown_preselects_next_parent_but_delays_counting():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.round_cooldown_seconds = 60.0
    engine = MultiPcbInspectionEngine(PcbInspectionConfig.from_first_category_config(c))
    detections = (
        [_make_parent(track_id=1, x=100)]
        + [_make_child("pad", 120 + index * 20, 120) for index in range(4)]
        + [_make_parent(track_id=2, x=400)]
        + [_make_child("pad", 420 + index * 20, 120) for index in range(4)]
    )

    first = engine.update(detections, image_size=(1000, 1000))

    assert len(first) == 1
    assert first[0].track_id == 1
    assert engine.current_round_id == 2
    assert engine.pcb_states[2].status == PcbStatus.COOLDOWN
    assert engine.pcb_states[2].last_slot_states == {}
    assert engine.update(detections, image_size=(1000, 1000)) == []
    assert engine.pcb_states[2].last_slot_states == {}



def test_first_category_region_uses_round_cooldown_seconds():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "screw", "", "", ""]
    c.round_cooldown_seconds = 5.0
    c.pcb_round_interval_seconds = 0.0
    derived = PcbInspectionConfig.from_first_category_config(c)
    assert derived.round_interval_seconds == 5.0


def test_round_cooldown_blocks_second_parent_counting_until_interval():
    c = ConfigManager()
    c.category_names = ["parent", "pad", "", "", "", ""]
    c.category_counts = [1, 4, 1, 1, 1, 1]
    c.round_cooldown_seconds = 60.0
    derived = PcbInspectionConfig.from_first_category_config(c)
    engine = MultiPcbInspectionEngine(derived)

    first = engine.update(
        [_make_parent(track_id=1, x=100)]
        + [_make_child("pad", 120 + index * 20, 120) for index in range(4)]
        + [_make_parent(track_id=2, x=400)]
        + [_make_child("pad", 420 + index * 20, 120) for index in range(4)],
        image_size=(1000, 1000),
    )
    assert len(first) == 1

    assert engine.current_round_id == 2
    assert engine.pcb_states[2].status == PcbStatus.COOLDOWN
    assert engine.pcb_states[2].result == PcbResult.NONE
