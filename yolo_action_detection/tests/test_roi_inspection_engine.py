"""通用 ROI 定位轮次引擎测试（模板匹配定位架构）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roi_inspection.engine import RoiInspectionEngine
from roi_inspection.geometry import NormalizedRect
from roi_inspection.models import InspectionOutcome, RoiInspectionConfig, RoundPhase
from yolo_runtime.yolo_result_models import ObbDetection


IMAGE_SIZE = (1000, 1000)


def _det(label: str, x: float, y: float) -> ObbDetection:
    return ObbDetection(
        class_id=0,
        label=label,
        conf=0.9,
        track_id=None,
        polygon=[(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5), (x - 5, y + 5)],
        box=(x - 5, y - 5, x + 5, y + 5),
        center=(x, y),
    )


def _config(**overrides) -> RoiInspectionConfig:
    """默认模板模式配置：inspection_roi 覆盖右半幅 (x>=500)。"""
    values = {
        "inspection_roi": NormalizedRect(0.5, 0.0, 0.5, 1.0),
        "required_counts": {"A": 2, "B": 1},
        "template_enabled": True,
        "pass_stable_frames": 1,
        "fail_stable_frames": 3,
        "round_interval_seconds": 2.0,
    }
    values.update(overrides)
    return RoiInspectionConfig(**values)


# --------------------------------------------------------------------------- #
# 1. 模板匹配触发本轮
# --------------------------------------------------------------------------- #
def test_template_matched_triggers_round_and_can_pass():
    engine = RoiInspectionEngine(_config())

    # matched=True 触发本轮；该帧检测到 A=2、B=1，pass_stable_frames=1 直接 PASS。
    detections = [_det("A", 600, 100), _det("A", 700, 100), _det("B", 800, 100)]
    results = engine.update(
        detections,
        IMAGE_SIZE,
        source_frame_id=7,
        timestamp=1.0,
        template_match={"matched": True, "score": 0.7},
    )

    assert len(results) == 1
    assert results[0].outcome == InspectionOutcome.PASS
    assert results[0].source_frame_id == 7
    assert results[0].actual_counts == {"A": 2, "B": 1}
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert engine.snapshot.template_matched is True
    assert engine.snapshot.template_score == 0.7


# --------------------------------------------------------------------------- #
# 2. 模板未匹配不触发，保持 WAITING
# --------------------------------------------------------------------------- #
def test_template_not_matched_keeps_waiting():
    engine = RoiInspectionEngine(_config())

    # matched=False 不触发；即便识别 ROI 内有目标也只停留在 WAITING。
    detections = [_det("A", 600, 100), _det("A", 700, 100), _det("B", 800, 100)]
    results = engine.update(
        detections,
        IMAGE_SIZE,
        source_frame_id=3,
        timestamp=1.0,
        template_match={"matched": False, "score": 0.3},
    )

    assert results == []
    assert engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert engine.snapshot.template_matched is False
    assert engine.snapshot.template_score == 0.3
    # WAITING 阶段不进入 _inspect，不累计稳定帧。
    assert all(state.match_streak == 0 for state in engine.snapshot.category_states.values())
    assert all(
        state.mismatch_streak == 0 for state in engine.snapshot.category_states.values()
    )


# --------------------------------------------------------------------------- #
# 3. 无模板模式：识别 ROI 内任意类别触发
# --------------------------------------------------------------------------- #
def test_no_template_any_category_in_inspection_roi_triggers_round():
    engine = RoiInspectionEngine(_config(template_enabled=False))

    # template_enabled=False：不依赖 template_match，单帧 A 进入识别 ROI 即触发。
    detections = [_det("A", 600, 100), _det("A", 700, 100), _det("B", 800, 100)]
    results = engine.update(detections, IMAGE_SIZE, source_frame_id=5, timestamp=1.0)

    assert len(results) == 1
    assert results[0].outcome == InspectionOutcome.PASS
    assert results[0].source_frame_id == 5
    assert results[0].actual_counts == {"A": 2, "B": 1}
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD


# --------------------------------------------------------------------------- #
# 4. 无模板模式空帧不触发不 FAIL
# --------------------------------------------------------------------------- #
def test_no_template_empty_frame_does_not_trigger_and_does_not_fail():
    engine = RoiInspectionEngine(_config(template_enabled=False))

    # 空帧：不触发，不累计 mismatch_streak。
    assert engine.update([], IMAGE_SIZE, timestamp=0.0) == []
    assert engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert all(
        state.mismatch_streak == 0 for state in engine.snapshot.category_states.values()
    )

    # 识别 ROI 外的目标也不触发。
    assert engine.update([_det("A", 100, 100)], IMAGE_SIZE, timestamp=0.1) == []
    assert engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert all(
        state.mismatch_streak == 0 for state in engine.snapshot.category_states.values()
    )


# --------------------------------------------------------------------------- #
# 5. 触发帧计入稳定帧累计
# --------------------------------------------------------------------------- #
def test_trigger_frame_counts_toward_stable_streak():
    engine = RoiInspectionEngine(_config(pass_stable_frames=2))

    # 触发帧 A=2、B=1 已匹配，但 pass_stable_frames=2，需再 1 帧才 PASS。
    passing = [_det("A", 600, 100), _det("A", 700, 100), _det("B", 800, 100)]
    assert (
        engine.update(
            passing,
            IMAGE_SIZE,
            timestamp=0.0,
            template_match={"matched": True, "score": 0.8},
        )
        == []
    )
    assert engine.snapshot.phase == RoundPhase.INSPECTING
    assert engine.snapshot.category_states["A"].match_streak == 1
    assert engine.snapshot.category_states["B"].match_streak == 1

    # 第二帧（已进入 INSPECTING，template_match 不再起作用）累计到 2 -> PASS。
    results = engine.update(
        passing,
        IMAGE_SIZE,
        timestamp=0.1,
        template_match={"matched": False, "score": 0.0},
    )
    assert len(results) == 1
    assert results[0].outcome == InspectionOutcome.PASS


# --------------------------------------------------------------------------- #
# 6. PASS / FAIL 稳定帧锁存
# --------------------------------------------------------------------------- #
def test_pass_fail_stable_streaks_latch_independently():
    engine = RoiInspectionEngine(_config(pass_stable_frames=2, fail_stable_frames=3))
    matched_tmpl = {"matched": True, "score": 0.8}

    # 帧 1：A=2 匹配 (streak 1)，B 缺失不匹配 (mismatch 1) -> 未完成。
    assert (
        engine.update(
            [_det("A", 600, 100), _det("A", 700, 100)],
            IMAGE_SIZE,
            timestamp=0.0,
            template_match=matched_tmpl,
        )
        == []
    )
    assert engine.snapshot.category_states["A"].match_streak == 1
    assert engine.snapshot.category_states["B"].mismatch_streak == 1

    # 帧 2：A 缺失 (match streak 归零，mismatch 1)，B=1 匹配 (match streak 1)。
    assert (
        engine.update(
            [_det("B", 800, 100)],
            IMAGE_SIZE,
            timestamp=0.1,
            template_match={"matched": False, "score": 0.0},
        )
        == []
    )
    assert engine.snapshot.category_states["A"].match_streak == 0
    assert engine.snapshot.category_states["A"].mismatch_streak == 1
    assert engine.snapshot.category_states["B"].match_streak == 1
    assert engine.snapshot.category_states["B"].mismatch_streak == 0

    # 帧 3：A=2、B=1 -> A streak 1，B streak 2 -> B 锁存 PASS。
    assert (
        engine.update(
            [_det("A", 600, 100), _det("A", 700, 100), _det("B", 800, 100)],
            IMAGE_SIZE,
            timestamp=0.2,
            template_match={"matched": False, "score": 0.0},
        )
        == []
    )
    assert engine.snapshot.category_states["B"].passed is True
    assert engine.snapshot.category_states["A"].match_streak == 1
    assert engine.snapshot.category_states["A"].passed is False

    # 帧 4：A=2 -> A streak 2 -> A 锁存 PASS，所有类别通过 -> 本轮 PASS。
    # 注意 B 缺失但已 latched，不再翻回未通过。
    results = engine.update(
        [_det("A", 600, 100), _det("A", 700, 100)],
        IMAGE_SIZE,
        timestamp=0.3,
        template_match={"matched": False, "score": 0.0},
    )
    assert len(results) == 1
    assert results[0].outcome == InspectionOutcome.PASS
    assert engine.snapshot.category_states["B"].passed is True
    assert engine.snapshot.category_states["A"].passed is True


def test_fail_latches_after_mismatch_streak_threshold():
    # 模板模式下：触发后连续 mismatch_stable_frames 帧数量不对 -> FAIL。
    engine = RoiInspectionEngine(_config(fail_stable_frames=2))
    matched_tmpl = {"matched": True, "score": 0.8}

    # 帧 1：触发并进入 INSPECTING；A=1（需 2）开始累计 mismatch。
    wrong = [_det("A", 600, 100), _det("B", 800, 100)]
    assert (
        engine.update(
            wrong,
            IMAGE_SIZE,
            timestamp=0.0,
            template_match=matched_tmpl,
        )
        == []
    )
    assert engine.snapshot.category_states["A"].mismatch_streak == 1

    # 帧 2：A 仍 1 个 -> mismatch 累计到 2 -> FAIL。
    results = engine.update(
        wrong,
        IMAGE_SIZE,
        timestamp=0.1,
        template_match={"matched": False, "score": 0.0},
    )
    assert len(results) == 1
    assert results[0].outcome == InspectionOutcome.FAIL
    assert "A" in results[0].failed_categories
    assert results[0].actual_counts["A"] == 1
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD


# --------------------------------------------------------------------------- #
# 7. cooldown 续轮（无模板模式：ROI 仍有目标即续轮）
# --------------------------------------------------------------------------- #
def test_no_template_cooldown_retriggers_when_category_still_in_roi():
    engine = RoiInspectionEngine(
        _config(required_counts={"A": 1}, template_enabled=False, round_interval_seconds=2.0)
    )
    passing = [_det("A", 700, 100)]

    first = engine.update(passing, IMAGE_SIZE, source_frame_id=1, timestamp=1.0)
    assert len(first) == 1
    assert first[0].feedback_key == "roi-round-1-pass"
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD

    # cooldown 未到期：仍持有，不触发。
    assert engine.update(passing, IMAGE_SIZE, source_frame_id=2, timestamp=2.9) == []
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert engine.snapshot.round_id == 1

    # cooldown 到期：识别 ROI 内仍有 A，立即触发下一轮（不要求离开）。
    second = engine.update(passing, IMAGE_SIZE, source_frame_id=3, timestamp=3.0)
    assert len(second) == 1
    assert second[0].feedback_key == "roi-round-2-pass"
    assert engine.snapshot.round_id == 2


def test_no_template_cooldown_returns_to_waiting_when_roi_empty():
    engine = RoiInspectionEngine(
        _config(required_counts={"A": 1}, template_enabled=False, round_interval_seconds=2.0)
    )

    first = engine.update([_det("A", 700, 100)], IMAGE_SIZE, source_frame_id=1, timestamp=1.0)
    assert len(first) == 1
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD

    # cooldown 到期但识别 ROI 为空：回到 WAITING，不触发、不判 FAIL。
    assert engine.update([], IMAGE_SIZE, source_frame_id=2, timestamp=3.0) == []
    assert engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert all(
        state.mismatch_streak == 0 for state in engine.snapshot.category_states.values()
    )


# --------------------------------------------------------------------------- #
# 8. 模板模式下 PASS 后 cooldown 到期续轮（需再次 template matched）
# --------------------------------------------------------------------------- #
def test_template_cooldown_retriggers_when_template_matched_again():
    engine = RoiInspectionEngine(
        _config(required_counts={"A": 1}, round_interval_seconds=2.0)
    )
    passing = [_det("A", 700, 100)]
    matched_tmpl = {"matched": True, "score": 0.8}

    first = engine.update(
        passing,
        IMAGE_SIZE,
        source_frame_id=1,
        timestamp=1.0,
        template_match=matched_tmpl,
    )
    assert len(first) == 1
    assert first[0].feedback_key == "roi-round-1-pass"
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD

    # cooldown 未到期：跳过。
    assert (
        engine.update(
            passing,
            IMAGE_SIZE,
            source_frame_id=2,
            timestamp=2.9,
            template_match=matched_tmpl,
        )
        == []
    )
    assert engine.snapshot.phase == RoundPhase.RESULT_HOLD
    assert engine.snapshot.round_id == 1

    # cooldown 到期但模板未匹配：回到 WAITING，不触发、不 FAIL。
    assert (
        engine.update(
            passing,
            IMAGE_SIZE,
            source_frame_id=3,
            timestamp=3.0,
            template_match={"matched": False, "score": 0.2},
        )
        == []
    )
    assert engine.snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR
    assert engine.snapshot.template_matched is False
    assert all(
        state.mismatch_streak == 0 for state in engine.snapshot.category_states.values()
    )

    # 再次模板匹配：触发下一轮并 PASS。
    second = engine.update(
        passing,
        IMAGE_SIZE,
        source_frame_id=4,
        timestamp=3.1,
        template_match=matched_tmpl,
    )
    assert len(second) == 1
    assert second[0].feedback_key == "roi-round-2-pass"
    assert engine.snapshot.round_id == 2
