"""步骤顺序引擎：直接类别顺序、错序 NG、离开保护帧。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from yolo_runtime.yolo_result_models import ObbDetection


class StepStatus(str, Enum):
    NOT_CONFIGURED = "not_configured"
    READY = "ready"
    WAITING = "waiting"
    PASS = "pass"
    NG = "ng"


class RoundResult(str, Enum):
    IN_PROGRESS = "in_progress"
    PASS = "pass"
    ACTION_NG = "action_ng"


@dataclass
class StepState:
    """单个步骤状态。"""
    index: int
    class_name: str
    configured: bool
    status: StepStatus = StepStatus.NOT_CONFIGURED
    enter_count: int = 0
    wrong_order_count: int = 0
    leave_count: int = 0
    passed_at: float = 0.0
    ng_at: float = 0.0
    ng_reason: str = ""
    required_count: int = 1
    current_count: int = 0


@dataclass
class StepSequenceState:
    """步骤序列整体状态。"""
    steps: List[StepState] = field(default_factory=list)
    current_step_index: int = -1
    round_result: RoundResult = RoundResult.IN_PROGRESS
    round_id: int = 0
    round_started_at: float = 0.0
    round_completed_at: float = 0.0
    action_ng_step: int = -1
    action_ng_reason: str = ""
    ng_rearm_pending: bool = False


class StepSequenceEngine:
    """步骤顺序引擎。

    严格按配置顺序完成步骤。模型输出的类别名就是动作类别：
    当前等待类别稳定出现后 PASS；当前未命中时，后续未完成类别稳定出现后 ACTION_NG。
    """

    def __init__(
        self,
        tool_class_name: str = "扭力枪",
        step_class_names: Optional[List[str]] = None,
        step_counts: Optional[List[int]] = None,
        enter_threshold: float = 0.18,
        enter_stable_frames: int = 1,
        leave_stable_frames: int = 4,
        out_of_order_frames: int = 2,
        order_constraint_enabled: bool = True,
        fusion_iou_threshold: float = 0.10,
        fusion_center_dist_threshold: int = 60,
        fusion_corner_inside_enabled: bool = True,
    ) -> None:
        # 以下 legacy 参数保留构造兼容，但生产判定不再依赖扭力枪或 OBB 接触。
        self._tool_class = tool_class_name
        self._step_classes = step_class_names or []
        self._step_counts = step_counts or []
        self._enter_stable = enter_stable_frames
        self._leave_stable = leave_stable_frames
        self._out_of_order_frames = out_of_order_frames
        self._order_constraint = order_constraint_enabled

        self._steps: List[StepState] = []
        self._round_id = 0
        self._round_started_at = 0.0
        self._round_result = RoundResult.IN_PROGRESS
        self._current_step_idx = -1
        self._action_ng_step = -1
        self._action_ng_reason = ""
        self._round_completed_at = 0.0
        self._ng_rearm_pending = False
        self._init_steps()

    def _init_steps(self) -> None:
        self._steps = []
        for i, name in enumerate(self._step_classes):
            configured = bool(name and name.strip())
            status = StepStatus.READY if configured else StepStatus.NOT_CONFIGURED
            required = 1
            if i < len(self._step_counts):
                required = max(1, int(self._step_counts[i]))
            self._steps.append(StepState(
                index=i,
                class_name=name,
                configured=configured,
                status=status,
                required_count=required,
            ))
        self._advance_to_next_waiting()

    def _advance_to_next_waiting(self) -> None:
        """将当前等待步骤推进到第一个 configured 且未 PASS 的步骤。"""
        self._current_step_idx = -1
        for step in self._steps:
            if step.configured and step.status in (StepStatus.READY, StepStatus.WAITING):
                step.status = StepStatus.WAITING
                self._current_step_idx = step.index
                return

    def _first_configured_step_index(self) -> int:
        for step in self._steps:
            if step.configured:
                return step.index
        return -1

    def start_round(self, *, require_first_step_rearm: bool = True) -> None:
        """开始新一轮。"""
        self._round_id += 1
        self._round_started_at = time.time()
        self._round_result = RoundResult.IN_PROGRESS
        self._action_ng_step = -1
        self._action_ng_reason = ""
        self._round_completed_at = 0.0
        self._ng_rearm_pending = bool(require_first_step_rearm and self._first_configured_step_index() >= 0)
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
            ng_rearm_pending=self._ng_rearm_pending,
        )

    def _first_configured_step_passed(self) -> bool:
        first_idx = self._first_configured_step_index()
        if first_idx < 0:
            return True
        return self._steps[first_idx].status == StepStatus.PASS

    def acknowledge_round_result(self) -> None:
        """兼容旧调用：确认本轮 NG 结果并恢复当前轮判定。

        当前生产默认路径不再调用该方法；主界面在 ACTION_NG 后会记录 FAIL
        并通过 ``start_round(require_first_step_rearm=True)`` 立即进入下一轮。
        该方法保留给需要人工确认后继续当前轮的调试/兼容场景。
        """
        if self._round_result != RoundResult.ACTION_NG:
            return
        if 0 <= self._action_ng_step < len(self._steps):
            step = self._steps[self._action_ng_step]
            if step.status == StepStatus.NG:
                step.status = StepStatus.WAITING
            step.wrong_order_count = 0
            step.ng_at = 0.0
            step.ng_reason = ""
        self._action_ng_step = -1
        self._action_ng_reason = ""
        self._round_result = RoundResult.IN_PROGRESS
        self._round_completed_at = 0.0
        # 若当前等待步骤已 PASS（理论上 NG 时不会，但防御性处理），推进到下一个
        if self._current_step_idx < 0 or (
            0 <= self._current_step_idx < len(self._steps)
            and self._steps[self._current_step_idx].status == StepStatus.PASS
        ):
            self._advance_to_next_waiting()

    def update(self, detections: List[ObbDetection]) -> StepSequenceState:
        """用当前帧检测结果更新步骤状态。"""
        if self._round_result != RoundResult.IN_PROGRESS:
            return self.get_state()

        expected_idx = self._current_step_idx
        if expected_idx < 0:
            return self.get_state()

        expected_step = self._steps[expected_idx]
        from collections import Counter
        label_counts = Counter(str(d.label) for d in detections if getattr(d, "label", ""))

        # 当前类别命中优先，不在同一帧判未来步骤 NG。
        current_count = label_counts.get(expected_step.class_name, 0)
        expected_step.current_count = current_count
        current_step_hit = current_count == expected_step.required_count
        if current_step_hit:
            expected_step.enter_count += 1
            expected_step.leave_count = 0
            for step in self._steps:
                if step.index > expected_idx:
                    step.wrong_order_count = 0
            if expected_step.enter_count >= self._enter_stable:
                expected_step.status = StepStatus.PASS
                expected_step.passed_at = time.time()
                if expected_step.index == self._first_configured_step_index():
                    self._ng_rearm_pending = False
                if self._all_configured_passed():
                    self._round_result = RoundResult.PASS
                    self._round_completed_at = time.time()
                else:
                    self._advance_to_next_waiting()
        else:
            expected_step.leave_count += 1
            if expected_step.leave_count > self._leave_stable:
                expected_step.enter_count = 0
                expected_step.leave_count = 0

        # 检查错序 NG
        first_step_passed = self._first_configured_step_passed()
        if not first_step_passed:
            for step in self._steps:
                if step.index > expected_idx:
                    step.wrong_order_count = 0

        if (
            self._order_constraint
            and self._round_result == RoundResult.IN_PROGRESS
            and not current_step_hit
            and not self._ng_rearm_pending
            and first_step_passed
        ):
            for step in self._steps:
                if not step.configured:
                    continue
                if step.index <= expected_idx:
                    continue
                if step.status in (StepStatus.PASS, StepStatus.NG):
                    continue
                step_count = label_counts.get(step.class_name, 0)
                step.current_count = step_count
                if step_count == step.required_count:
                    step.wrong_order_count += 1
                    if step.wrong_order_count >= self._out_of_order_frames:
                        self._round_result = RoundResult.ACTION_NG
                        self._action_ng_step = step.index
                        self._action_ng_reason = (
                            f"检测到步骤{step.index + 1}({step.class_name})，"
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

    def _all_configured_passed(self) -> bool:
        for step in self._steps:
            if step.configured and step.status != StepStatus.PASS:
                return False
        return True
