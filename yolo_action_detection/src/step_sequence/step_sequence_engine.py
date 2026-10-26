"""步骤顺序引擎：直接类别顺序、错序 NG、离开保护帧。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from detection_logging.action_trace import trace
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
    fail_retry_count: int = 0


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
    fail_retry_enabled: bool = False
    fail_retry_frames: int = 10
    fail_retry_count: int = 0
    action_boundary_frame: int = 0
    action_decision_frame: int = 0
    action_seen_parts: List[str] = field(default_factory=list)
    action_missing_parts: List[str] = field(default_factory=list)
    action_event_name: str = ""
    action_weights_sha256: str = ""
    action_start_time: float = 0.0
    action_end_time: float = 0.0
    action_decision_time: float = 0.0


class StepSequenceEngine:
    """步骤顺序引擎。

    严格按配置顺序完成步骤。模型输出的类别名就是动作类别：
    当前等待类别稳定出现后 PASS；当前未命中时，后续未完成类别稳定出现后 ACTION_NG。
    """

    PENDING_EVENT_LIMIT = 16

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
        fail_retry_enabled: bool = False,
        fail_retry_frames: int = 10,
        fusion_iou_threshold: float = 0.10,
        fusion_center_dist_threshold: int = 60,
        fusion_corner_inside_enabled: bool = True,
        detection_updates: bool = True,
    ) -> None:
        # 以下 legacy 参数保留构造兼容，但生产判定不再依赖扭力枪或 OBB 接触。
        self._tool_class = tool_class_name
        self._step_classes = step_class_names or []
        self._step_counts = step_counts or []
        self._enter_stable = enter_stable_frames
        self._leave_stable = leave_stable_frames
        self._out_of_order_frames = out_of_order_frames
        self._order_constraint = order_constraint_enabled
        self._fail_retry_enabled = bool(fail_retry_enabled)
        self._fail_retry_frames = max(1, int(fail_retry_frames))
        self._detection_updates = bool(detection_updates)
        # 单一相机会话串行产生单调编号；换运行实例时须一起新建引擎。
        # 普通换轮保留高水位，既拒绝重放也避免全天运行无限积累事件集合。
        self._last_consumed_event: tuple[int, int] | None = None
        self._action_min_source_frame: int | None = None
        # 本轮结束后到达的完整动作属于下一件，换轮时重放；上限防止长冷却积压。
        self._pending_events: List[dict] = []
        self._next_round_boundary: int | None = None
        self._boundary_frame = 0
        self._decision_frame = 0
        self._seen_parts: List[str] = []
        self._missing_parts: List[str] = []
        self._event_name = ""
        self._weights_sha256 = ""
        self._action_start_time = 0.0
        self._action_end_time = 0.0
        self._action_decision_time = 0.0

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

    def start_round(
        self,
        *,
        require_first_step_rearm: bool = True,
        action_min_source_frame: int | None = None,
        continue_from_previous: bool = False,
    ) -> None:
        """开始新一轮。

        连续作业时下一件动作从上一轮结束边界开始，门槛取该边界；
        开始检测或重载时取调用方给的预览帧门槛，并丢弃暂存的动作。
        """
        self._round_id += 1
        if continue_from_previous and self._next_round_boundary is not None:
            threshold: int | None = self._next_round_boundary
            boundary_source = "previous_round"
        else:
            threshold = int(action_min_source_frame) if action_min_source_frame is not None else None
            boundary_source = "preview"
            self._pending_events.clear()
        self._next_round_boundary = None
        self._action_min_source_frame = threshold
        self._boundary_frame = 0
        self._decision_frame = 0
        self._seen_parts = []
        self._missing_parts = []
        self._event_name = ""
        self._weights_sha256 = ""
        self._action_start_time = 0.0
        self._action_end_time = 0.0
        self._action_decision_time = 0.0
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
                step.current_count = 0
                step.fail_retry_count = 0
        self._advance_to_next_waiting()
        trace(
            "round_start",
            round_id=self._round_id,
            min_source_frame=self._action_min_source_frame if self._action_min_source_frame is not None else "-",
            rearm=int(self._ng_rearm_pending),
            boundary_source=boundary_source,
            queued=len(self._pending_events),
            note="新轮起始门槛",
        )
        while self._pending_events and self._round_result == RoundResult.IN_PROGRESS:
            event = self._pending_events.pop(0)
            self._trace_event(event, "replay", "replayed", "重放上一轮间隔中的动作")
            self._settle_action_event(event)

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
            fail_retry_enabled=self._fail_retry_enabled,
            fail_retry_frames=self._fail_retry_frames,
            fail_retry_count=(
                self._steps[self._current_step_idx].fail_retry_count
                if 0 <= self._current_step_idx < len(self._steps)
                else 0
            ),
            action_boundary_frame=self._boundary_frame,
            action_decision_frame=self._decision_frame,
            action_seen_parts=list(self._seen_parts),
            action_missing_parts=list(self._missing_parts),
            action_event_name=self._event_name,
            action_weights_sha256=self._weights_sha256,
            action_start_time=self._action_start_time,
            action_end_time=self._action_end_time,
            action_decision_time=self._action_decision_time,
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
        """用当前帧检测结果更新步骤状态。动作模式不走这条路径。"""
        if not self._detection_updates:
            return self.get_state()
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
            expected_step.fail_retry_count = 0
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
            if self._should_count_fail_retry(expected_step, current_count):
                expected_step.fail_retry_count += 1
                if expected_step.fail_retry_count >= self._fail_retry_frames:
                    self._mark_quantity_ng(expected_step)
                    return self.get_state()

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

    def apply_action_event(self, event: dict) -> StepSequenceState:
        """结算一个已经结束的动作。同一编号只消费一次；本轮已结束时暂存给下一轮。"""
        key = (int(event.get("session_id") or 0), int(event.get("event_id") or 0))
        if self._last_consumed_event is not None and key <= self._last_consumed_event:
            self._trace_event(event, "reject", "duplicate", "重复事件")
            return self.get_state()
        self._last_consumed_event = key
        if self._round_result != RoundResult.IN_PROGRESS:
            self._queue_event(event)
            return self.get_state()
        return self._settle_action_event(event)

    def _queue_event(self, event: dict) -> None:
        self._pending_events.append(event)
        self._trace_event(event, "queued", "queued", "本轮已结束，留给下一轮", queued=len(self._pending_events))
        if len(self._pending_events) > self.PENDING_EVENT_LIMIT:
            dropped = self._pending_events.pop(0)
            self._trace_event(dropped, "reject", "queue_overflow", "间隔中暂存的动作太多，丢弃最早一个")

    def _settle_action_event(self, event: dict) -> StepSequenceState:
        if self._action_min_source_frame is not None:
            start = event.get("start_frame")
            if start is None or int(start) < self._action_min_source_frame:
                self._trace_event(
                    event, "reject", "before_round_start", "动作开始帧早于新轮门槛",
                    min_source_frame=self._action_min_source_frame,
                )
                return self.get_state()
        if str(event.get("class_id") or "") == "idle":
            self._trace_event(event, "reject", "idle", "空闲不结算")
            return self.get_state()
        name = str(event.get("name") or "")
        if self._starts_next_product(name):
            self._roll_over_to_next_round(event)
            return self.get_state()
        self._boundary_frame = int(event.get("end_frame") or 0)
        self._decision_frame = int(event.get("decision_frame") or 0)
        self._seen_parts = [str(item) for item in event.get("parts") or []]
        self._missing_parts = [str(item) for item in event.get("missing_parts") or []]
        self._event_name = name
        self._weights_sha256 = str(event.get("weights_sha256") or "")
        self._action_start_time = float(event.get("start_time") or 0.0)
        self._action_end_time = float(event.get("end_time") or 0.0)
        self._action_decision_time = float(event.get("decision_time") or 0.0)
        expected = self._steps[self._current_step_idx] if 0 <= self._current_step_idx < len(self._steps) else None
        expected_name = expected.class_name if expected is not None else ""
        matched = next(
            (
                step for step in self._steps
                if step.configured
                and step.class_name == name
                and step.status not in (StepStatus.PASS, StepStatus.NG)
            ),
            None,
        )
        if self._missing_parts:
            target = matched if matched is not None else expected
            if target is None or target.status in (StepStatus.PASS, StepStatus.NG):
                self._trace_event(event, "reject", "missing_parts_no_open_step", "缺少零件但没有可标红的步骤")
                return self.get_state()
            self._mark_step_red(
                target,
                f"动作{name}缺少{'、'.join(self._missing_parts)}；当前等待{expected_name}",
            )
            self._trace_event(event, "ng", "missing_parts", "缺少零件", step=target.class_name)
            return self.get_state()
        if expected is not None and name == expected.class_name:
            self._pass_action_step(expected)
            self._trace_event(event, "pass", "expected_step", "与当前步骤一致", step=expected.class_name)
            return self.get_state()
        if matched is None:
            self._trace_event(
                event, "reject", "no_matching_step", "没有对应的未完成步骤",
                expected=expected_name or "-",
            )
            return self.get_state()
        self._mark_step_red(
            matched,
            f"检测到步骤{matched.index + 1}({name})，但当前等待步骤"
            f"{(expected.index + 1) if expected is not None else '?'}({expected_name})",
        )
        self._trace_event(
            event, "ng", "wrong_order", "顺序不对",
            step=matched.class_name, expected=expected_name or "-",
        )
        return self.get_state()

    def _trace_event(self, event: dict, result: str, reason: str, note: str, **extra) -> None:
        missing = ",".join(str(item) for item in event.get("missing_parts") or []) or "-"
        trace(
            "event_decision",
            result=result,
            reason=reason,
            note=note,
            round_id=self._round_id,
            name=event.get("name") or "-",
            start_frame=event.get("start_frame"),
            end_frame=event.get("end_frame"),
            session=event.get("session_id"),
            event=event.get("event_id"),
            missing=missing,
            **extra,
        )

    def _pass_action_step(self, step: StepState) -> None:
        step.status = StepStatus.PASS
        step.passed_at = time.time()
        step.current_count = 1
        if step.index == self._first_configured_step_index():
            self._ng_rearm_pending = False
        self._finish_action_round()

    def _mark_step_red(self, step: StepState, reason: str) -> None:
        """错序或少零件标红这一步；它之前没做的步骤判未检测到，不再回头等。"""
        step.status = StepStatus.NG
        step.ng_at = time.time()
        step.ng_reason = reason
        step.current_count = 0
        self._mark_open_steps_not_detected(before=step.index)
        self._finish_action_round()

    def _mark_open_steps_not_detected(self, before: int | None = None) -> None:
        for step in self._steps:
            if not step.configured or step.status in (StepStatus.PASS, StepStatus.NG):
                continue
            if before is not None and step.index >= before:
                continue
            step.status = StepStatus.NG
            step.ng_at = time.time()
            step.ng_reason = f"未检测到步骤{step.index + 1}({step.class_name})"
            step.current_count = 0

    def _starts_next_product(self, name: str) -> bool:
        """已越过的前序动作再次出现，说明下一件开始了。

        与最后出结果的步骤同名不算：同一动作被短暂停顿切成两段时会这样。
        """
        configured = [step for step in self._steps if step.configured]
        if any(step.class_name == name and step.status not in (StepStatus.PASS, StepStatus.NG) for step in configured):
            return False
        resolved = [step.index for step in configured if step.status in (StepStatus.PASS, StepStatus.NG)]
        if not resolved or self._steps[max(resolved)].class_name == name:
            return False
        return any(step.class_name == name and step.index < max(resolved) for step in configured)

    def _roll_over_to_next_round(self, event: dict) -> None:
        self._trace_event(event, "rollover", "round_rollover", "前序动作再次出现，本轮结束，交给下一轮")
        self._mark_open_steps_not_detected()
        self._finish_action_round()
        self._next_round_boundary = int(event.get("start_frame") or 0)
        self._pending_events.insert(0, event)

    def _finish_action_round(self) -> None:
        """每一步都有结果后才结束本轮。有红步时整轮只记一次 NG。"""
        configured = [step for step in self._steps if step.configured]
        if any(step.status not in (StepStatus.PASS, StepStatus.NG) for step in configured):
            self._advance_to_next_waiting()
            return
        self._next_round_boundary = self._boundary_frame
        failed = [step for step in configured if step.status == StepStatus.NG]
        if not failed:
            self._round_result = RoundResult.PASS
            self._round_completed_at = time.time()
            trace("round_result", round_id=self._round_id, result="pass", note="本轮通过")
            return
        self._round_result = RoundResult.ACTION_NG
        self._action_ng_step = failed[0].index
        self._action_ng_reason = "；".join(step.ng_reason for step in failed if step.ng_reason)
        self._round_completed_at = time.time()
        trace(
            "round_result",
            round_id=self._round_id,
            result="ng",
            note="本轮未通过",
            reason=self._action_ng_reason,
        )

    def _mark_action_ng(self, step_index: int, reason: str) -> None:
        self._round_result = RoundResult.ACTION_NG
        self._action_ng_step = step_index
        self._action_ng_reason = reason
        self._round_completed_at = time.time()
        if 0 <= step_index < len(self._steps):
            step = self._steps[step_index]
            step.status = StepStatus.NG
            step.ng_at = time.time()
            step.ng_reason = reason

    def _quantity_ng_reason(self, step: StepState) -> str:
        return (
            f"步骤{step.index + 1}({step.class_name})连续{self._fail_retry_frames}帧"
            f"类别未检出或数量不符（实际{step.current_count}/要求{step.required_count}）"
        )

    def _should_count_fail_retry(self, step: StepState, current_count: int) -> bool:
        if not self._fail_retry_enabled:
            return False
        first_idx = self._first_configured_step_index()
        if step.index == first_idx:
            return current_count != 0 and current_count != step.required_count
        return current_count != step.required_count

    def _mark_quantity_ng(self, step: StepState) -> None:
        reason = self._quantity_ng_reason(step)
        self._round_result = RoundResult.ACTION_NG
        self._action_ng_step = step.index
        self._action_ng_reason = reason
        step.status = StepStatus.NG
        step.ng_at = time.time()
        step.ng_reason = reason
        self._round_completed_at = time.time()

    def _all_configured_passed(self) -> bool:
        for step in self._steps:
            if step.configured and step.status != StepStatus.PASS:
                return False
        return True


def build_step_engine(config) -> StepSequenceEngine:
    """按配置创建步骤引擎，所有构造入口共用。

    顺序模式由 LSTM 动作结算步骤时，逐帧检测不能推进步骤。
    """
    from lstm_action.bundle import uses_lstm_sequence

    return StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.get_runtime_category_counts(),
        enter_stable_frames=config.action_pass_stable_frames,
        out_of_order_frames=config.action_ng_stable_frames,
        leave_stable_frames=getattr(config, "action_leave_stable_frames", 4),
        order_constraint_enabled=config.action_order_constraint_enabled,
        fail_retry_enabled=getattr(config, "action_fail_retry_enabled", False),
        fail_retry_frames=getattr(config, "action_ng_stable_frames", 10),
        detection_updates=not uses_lstm_sequence(config),
    )
