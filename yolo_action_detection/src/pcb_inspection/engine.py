"""多 PCB 检查引擎：并行判定、连续 FAIL、已处理去重、结果队列。"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from pcb_inspection.geometry import ComponentOwnershipResolver
from pcb_inspection.models import (
    PcbInspectionConfig,
    PcbInspectionResult,
    PcbState,
    PcbResult,
    PcbStatus,
    SlotObservation,
)
from yolo_runtime.yolo_result_models import ObbDetection


class MultiPcbInspectionEngine:
    """多 PCB 元器件检查引擎。

    - 每帧推理结果只处理一次，同时更新所有 PCB
    - PCB 用 track_id 做身份，元器件用空间归属
    - 四个槽位全齐 -> PASS，连续 N 帧缺件 -> FAIL
    - 已处理 PCB 不再重复判定
    """

    def __init__(self, config: PcbInspectionConfig) -> None:
        self._config = config
        self._resolver = ComponentOwnershipResolver(
            margin_ratio=config.assignment_margin_ratio,
        )
        self._pcb_states: Dict[int, PcbState] = {}
        self._result_queue: List[PcbInspectionResult] = []
        self._last_emit_time: float = 0.0
        self._frame_counter: int = 0

    @property
    def pcb_states(self) -> Dict[int, PcbState]:
        return self._pcb_states

    def update(self, detections: List[ObbDetection], image_size: Optional[Tuple[int, int]] = None) -> List[PcbInspectionResult]:
        """用一帧检测结果更新所有 PCB 状态，返回本轮可输出的结果列表。"""
        self._frame_counter += 1
        frame_id = self._frame_counter

        # 分离 PCB 和元器件
        pcb_dets = [
            d for d in detections
            if d.label == self._config.pcb_class_name and d.track_id is not None
        ]
        comp_dets = [
            d for d in detections
            if d.label in self._config.component_class_names
        ]

        # 为每个 PCB 建立状态
        for pcb_det in pcb_dets:
            tid = pcb_det.track_id
            if tid not in self._pcb_states:
                self._pcb_states[tid] = PcbState(track_id=tid, first_seen_frame=frame_id)
            self._pcb_states[tid].last_seen_frame = frame_id

        # 计算每个元器件的归属
        pcb_list = [
            {"track_id": d.track_id, "polygon": d.polygon, "center": d.center}
            for d in pcb_dets
        ]

        # 每块 PCB 的本帧槽位观测
        slot_observations: Dict[int, Dict[str, SlotObservation]] = {
            d.track_id: {
                name: SlotObservation(class_name=name, present=False, unreliable=False)
                for name in self._config.component_class_names
                if name
            }
            for d in pcb_dets
        }

        # 歧义标记集合
        ambiguous_pcb_ids: set = set()

        for comp_det in comp_dets:
            owner_id, ambiguous = self._resolver.resolve(
                component_center=comp_det.center,
                component_poly=comp_det.polygon,
                pcb_detections=pcb_list,
            )
            if ambiguous:
                ambiguous_pcb_ids.update(ambiguous)
            if owner_id is not None and owner_id in slot_observations:
                slots = slot_observations[owner_id]
                if comp_det.label in slots:
                    slots[comp_det.label].present = True

        # 标记歧义 PCB 的观测为不可靠
        for amb_id in ambiguous_pcb_ids:
            if amb_id in slot_observations:
                for slot in slot_observations[amb_id].values():
                    slot.unreliable = True

        # 更新每个 PCB 的判定
        for pcb_det in pcb_dets:
            tid = pcb_det.track_id
            state = self._pcb_states[tid]
            if not state.can_judge:
                continue

            # 检查 PCB 是否完整可见
            if not self._is_pcb_fully_visible(pcb_det, image_size):
                continue

            slots = slot_observations.get(tid, {})
            # 如果有不可靠观测，本帧不累计 FAIL
            has_unreliable = any(s.unreliable for s in slots.values())
            all_present = all(s.present for s in slots.values()) if slots else False

            state.last_slot_states = slots

            if all_present:
                # PASS
                state.status = PcbStatus.DECIDED
                state.result = PcbResult.PASS
                state.result_frame = frame_id
                state.result_timestamp = time.time()
                state.missing_classes = []
            elif has_unreliable:
                # 不可靠帧不累计
                pass
            else:
                # 缺件
                state.consecutive_fail += 1
                state.missing_classes = [
                    name for name, slot in slots.items() if not slot.present
                ]
                if state.consecutive_fail >= self._config.fail_stable_frames:
                    state.status = PcbStatus.DECIDED
                    state.result = PcbResult.FAIL
                    state.result_frame = frame_id
                    state.result_timestamp = time.time()

        # 收集可输出的结果
        emitted = self._collect_emittable_results()
        return emitted

    def _is_pcb_fully_visible(self, pcb_det: ObbDetection, image_size: Optional[Tuple[int, int]]) -> bool:
        """PCB 多边形完整在画面安全区域内。"""
        if image_size is None:
            return True  # 无法判断时默认可检查
        w, h = image_size
        margin = self._config.frame_margin_px
        for px, py in pcb_det.polygon:
            if px < margin or px > w - margin or py < margin or py > h - margin:
                return False
        return True

    def _collect_emittable_results(self) -> List[PcbInspectionResult]:
        """收集已决策但未输出的结果，按间隔控制输出。"""
        now = time.time()
        results: List[PcbInspectionResult] = []

        for tid, state in self._pcb_states.items():
            if state.status != PcbStatus.DECIDED:
                continue

            # 间隔控制
            if self._config.round_interval_seconds > 0:
                elapsed = now - self._last_emit_time
                if elapsed < self._config.round_interval_seconds and results:
                    # 已有结果在等间隔，后续的也排队
                    break

            slot_states = {name: slot.present for name, slot in state.last_slot_states.items()}
            result = PcbInspectionResult(
                track_id=tid,
                result=state.result,
                slot_states=slot_states,
                missing_classes=list(state.missing_classes),
                source_frame_id=state.result_frame,
                timestamp=state.result_timestamp,
            )
            results.append(result)
            state.status = PcbStatus.EMITTED
            self._last_emit_time = now

        return results

    def get_state_summary(self) -> Dict:
        """返回所有 PCB 状态摘要（用于 UI 显示）。"""
        summary = {}
        for tid, state in self._pcb_states.items():
            summary[tid] = {
                "status": state.status.value,
                "result": state.result.value,
                "consecutive_fail": state.consecutive_fail,
                "slots": {name: {"present": s.present, "unreliable": s.unreliable}
                          for name, s in state.last_slot_states.items()},
                "missing_classes": list(state.missing_classes),
            }
        return summary