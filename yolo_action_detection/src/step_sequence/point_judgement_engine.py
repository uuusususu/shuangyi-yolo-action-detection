"""点位状态机：枪头点到洞口中心距离判定，替代面积交集。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from calibration.models import (
    CalibrationTransform, HoleDefinition, HoleJudgement, HoleZone,
    PointJudgementState, TipPoint,
)
from calibration.calibrator import distance_mm, judge_hole_zone, find_nearest_hole, extract_tip_point
from step_sequence.step_sequence_engine import StepStatus, RoundResult, StepState, StepSequenceState


class PointJudgementEngine:
    """点位判定引擎。

    PASS: 枪头点进入当前步骤洞口内圈并连续稳定若干帧。
    ACTION_NG: 枪头点进入非当前步骤洞口内圈并连续稳定若干帧。
    """

    def __init__(
        self,
        tool_class_name: str = "扭力枪",
        step_class_names: Optional[List[str]] = None,
        holes: Optional[List[HoleDefinition]] = None,
        calibrator: Optional[CalibrationTransform] = None,
        tip_enter_stable_frames: int = 2,
        tip_wrong_stable_frames: int = 2,
        point_judgement_enabled: bool = True,
    ) -> None:
        self._tool_class = tool_class_name
        self._step_classes = step_class_names or []
        self._holes = holes or []
        self._calibrator = calibrator
        self._enter_stable = tip_enter_stable_frames
        self._wrong_stable = tip_wrong_stable_frames
        self._enabled = point_judgement_enabled

        self._steps: List[StepState] = []
        self._round_id = 0
        self._round_started_at = 0.0
        self._round_result = RoundResult.IN_PROGRESS
        self._current_step_idx = -1
        self._action_ng_step = -1
        self._action_ng_reason = ""
        self._round_completed_at = 0.0
        self._init_steps()

    def _init_steps(self) -> None:
        self._steps = []
        for i, name in enumerate(self._step_classes):
            hole = self._get_hole(i)
            configured = bool(name and name.strip()) and hole is not None and hole.enabled
            status = StepStatus.READY if configured else StepStatus.NOT_CONFIGURED
            self._steps.append(StepState(index=i, class_name=name, configured=configured, status=status))
        self._advance_to_next_waiting()

    def _get_hole(self, step_index: int) -> Optional[HoleDefinition]:
        for h in self._holes:
            if h.step_index == step_index:
                return h
        return None

    def _advance_to_next_waiting(self) -> None:
        self._current_step_idx = -1
        for step in self._steps:
            if step.configured and step.status in (StepStatus.READY, StepStatus.WAITING):
                step.status = StepStatus.WAITING
                self._current_step_idx = step.index
                return

    def start_round(self) -> None:
        self._round_id += 1
        self._round_started_at = time.time()
        self._round_result = RoundResult.IN_PROGRESS
        self._action_ng_step = -1
        self._action_ng_reason = ""
        self._round_completed_at = 0.0
        for step in self._steps:
            if step.configured:
                step.status = StepStatus.READY
                step.enter_count = 0
                step.wrong_order_count = 0
                step.leave_count = 0
                step.passed_at = 0.0
                step.ng_at = 0.0
                step.ng_reason = ""
        self._advance_to_next_waiting()

    def update(self, detections, frame_id: int = 0) -> StepSequenceState:
        if self._round_result != RoundResult.IN_PROGRESS:
            return self.get_state()

        if not self._enabled or not self._calibrator or not self._calibrator.is_valid:
            return self.get_state()

        tip, diagnostics = extract_tip_point(
            detections, self._tool_class, self._calibrator, frame_id
        )
        if tip is None:
            return self.get_state()

        expected_idx = self._current_step_idx
        if expected_idx < 0:
            return self.get_state()

        expected_hole = self._get_hole(expected_idx)
        if expected_hole is None or not expected_hole.enabled:
            return self.get_state()

        # 判断枪头相对当前步骤洞口
        judgement = judge_hole_zone((tip.mm_x, tip.mm_y), expected_hole)
        expected_step = self._steps[expected_idx]

        if judgement.zone == HoleZone.INSIDE:
            expected_step.enter_count += 1
            expected_step.leave_count = 0
            if expected_step.enter_count >= self._enter_stable:
                expected_step.status = StepStatus.PASS
                expected_step.passed_at = time.time()
                if self._all_configured_passed():
                    self._round_result = RoundResult.PASS
                    self._round_completed_at = time.time()
                else:
                    self._advance_to_next_waiting()
        else:
            expected_step.enter_count = 0

        # 检查错序 NG
        if self._round_result == RoundResult.IN_PROGRESS:
            for step in self._steps:
                if not step.configured or step.index == expected_idx or step.status == StepStatus.NG:
                    continue
                hole = self._get_hole(step.index)
                if hole is None or not hole.enabled:
                    continue
                j = judge_hole_zone((tip.mm_x, tip.mm_y), hole)
                if j.zone == HoleZone.INSIDE:
                    step.wrong_order_count += 1
                    if step.wrong_order_count >= self._wrong_stable:
                        self._round_result = RoundResult.ACTION_NG
                        self._action_ng_step = step.index
                        self._action_ng_reason = (
                            f"枪头进入步骤{step.index + 1}({step.class_name})洞口内圈，"
                            f"但当前等待步骤{expected_idx + 1}({expected_step.class_name})"
                        )
                        step.status = StepStatus.NG
                        step.ng_at = time.time()
                        step.ng_reason = self._action_ng_reason
                        self._round_completed_at = time.time()
                        break
                else:
                    step.wrong_order_count = 0

        return self.get_state()

    def get_state(self) -> StepSequenceState:
        return StepSequenceState(
            steps=list(self._steps),
            current_step_index=self._current_step_idx,
            round_result=self._round_result,
            round_id=self._round_id,
            round_started_at=self._round_started_at,
            round_completed_at=self._round_completed_at,
            action_ng_step=self._action_ng_step,
            action_ng_reason=self._action_ng_reason,
        )

    def _all_configured_passed(self) -> bool:
        for step in self._steps:
            if step.configured and step.status != StepStatus.PASS:
                return False
        return True
