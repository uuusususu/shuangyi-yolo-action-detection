"""多 PCB 检查引擎：父类并行追踪，单 active_id 串行判定。"""
from __future__ import annotations

import time
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from pcb_inspection.geometry import ComponentOwnershipResolver
from pcb_inspection.models import (
    PcbInspectionConfig,
    PcbInspectionResult,
    PcbState,
    PcbResult,
    PcbStatus,
    SlotObservation,
    SlotStatus,
)
from step_sequence.geometry import polygon_iou
from yolo_runtime.yolo_result_models import ObbDetection


@dataclass
class _ParentTrack:
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]
    last_seen_frame: int
    source_track_id: Optional[int] = None


class MultiPcbInspectionEngine:
    """多 PCB 元器件检查引擎。

    - 每帧推理结果只处理一次，同时追踪所有 PCB
    - 父区域优先使用原生 track_id，无原生 ID 时按位置分配短生命周期 ID
    - 同一时间只允许一个 active_id 推进子控件数量判定
    - 各子控件数量精确匹配 -> PASS，连续 N 帧数量异常 -> 本轮 FAIL
    - PASS 才是终局；FAIL 只报警并重新入队，允许后续补齐后 PASS
    """

    _PARENT_TRACK_MIN_IOU = 0.30
    _PARENT_TRACK_MAX_MISSED_FRAMES = 15

    def __init__(self, config: PcbInspectionConfig) -> None:
        self._config = config
        self._resolver = ComponentOwnershipResolver(
            margin_ratio=config.assignment_margin_ratio,
        )
        self._pcb_states: Dict[int, PcbState] = {}
        self._result_queue: List[PcbInspectionResult] = []
        self._last_emit_time: float = 0.0
        self._frame_counter: int = 0
        self._parent_tracks: Dict[int, _ParentTrack] = {}
        self._source_to_business_id: Dict[int, int] = {}
        self._next_business_track_id: int = -1
        self._last_resolved_detections: List[ObbDetection] = []
        self._current_parent_ids: set[int] = set()
        self._waiting_parent_ids: List[int] = []
        self._current_round_id: Optional[int] = None
        self._next_attempt_id: int = 0
        self._next_round_ready_time: float = 0.0

    @property
    def pcb_states(self) -> Dict[int, PcbState]:
        return self._pcb_states

    @property
    def last_resolved_detections(self) -> List[ObbDetection]:
        """返回当前帧检测，其中父区域已写入业务追踪 ID。"""
        return list(self._last_resolved_detections)

    @property
    def current_parent_ids(self) -> set[int]:
        """返回当前帧实际检测并解析出的父类业务 ID。"""
        return set(self._current_parent_ids)

    @property
    def current_round_id(self) -> Optional[int]:
        """返回当前正在冷却或判定的父类业务 ID。"""
        return self._current_round_id

    @property
    def current_step_states(self) -> Dict[str, SlotObservation]:
        """返回当前轮次 ID 的子步骤状态。"""
        if self._current_round_id is None:
            return {}
        state = self._pcb_states.get(self._current_round_id)
        if state is None:
            return {}
        return dict(state.last_slot_states)

    def update(self, detections: List[ObbDetection], image_size: Optional[Tuple[int, int]] = None) -> List[PcbInspectionResult]:
        """用一帧检测结果更新所有 PCB 状态，返回本轮可输出的结果列表。"""
        self._frame_counter += 1
        frame_id = self._frame_counter

        # 分离父区域和子控件；ONNX 结果没有原生 track_id 时在此补稳定身份。
        parent_dets = [
            d for d in detections
            if d.label == self._config.pcb_class_name
        ]
        pcb_dets = self._assign_parent_track_ids(parent_dets, frame_id)
        self._current_parent_ids = {d.track_id for d in pcb_dets if d.track_id is not None}
        resolved_parents = iter(pcb_dets)
        self._last_resolved_detections = [
            next(resolved_parents) if detection.label == self._config.pcb_class_name else detection
            for detection in detections
        ]
        comp_dets = [
            d for d in detections
            if d.label in self._config.component_class_names
        ]

        # 为每个 PCB 建立状态，并按稳定出现顺序进入等待队列。
        new_parents: List[Tuple[float, int]] = []
        for pcb_det in pcb_dets:
            tid = pcb_det.track_id
            if tid not in self._pcb_states:
                self._pcb_states[tid] = PcbState(track_id=tid, first_seen_frame=frame_id)
                self._pcb_states[tid].queued_frame = frame_id
                new_parents.append((float(pcb_det.center[0]), tid))
            self._pcb_states[tid].last_seen_frame = frame_id
        self._enqueue_new_parents(new_parents)
        now = time.time()
        self._sync_current_round(now)

        # 计算每个元器件的归属
        pcb_list = [
            {"track_id": d.track_id, "polygon": d.polygon, "center": d.center}
            for d in pcb_dets
        ]

        # 每块 PCB 的本帧槽位观测
        slot_observations: Dict[int, Dict[str, SlotObservation]] = {
            d.track_id: {
                name: SlotObservation(
                    class_name=name,
                    present=False,
                    unreliable=False,
                    required_count=self._config.component_required_counts.get(name, 1),
                )
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
                    slots[comp_det.label].observed_count += 1

        for slots in slot_observations.values():
            for slot in slots.values():
                slot.present = slot.observed_count == slot.required_count

        # 标记歧义 PCB 的观测为不可靠
        for amb_id in ambiguous_pcb_ids:
            if amb_id in slot_observations:
                for slot in slot_observations[amb_id].values():
                    slot.unreliable = True

        # 只更新当前 active_id 的判定；其他父类只追踪和绘制。
        for pcb_det in pcb_dets:
            tid = pcb_det.track_id
            if tid != self._current_round_id:
                continue
            state = self._pcb_states[tid]
            if not state.can_judge:
                continue

            # 检查 PCB 是否完整可见
            if not self._is_pcb_fully_visible(pcb_det, image_size):
                continue

            slots = slot_observations.get(tid, {})
            # 如果有不可靠观测，本帧不推进任何稳定计数。
            has_unreliable = any(s.unreliable for s in slots.values())

            state.last_slot_states = slots

            if not has_unreliable:
                self._update_slot_stability(state, slots)

            for name, slot in slots.items():
                slot.present = name in state.completed_classes

            terminal_mismatches = [
                name
                for name, streak in state.mismatch_streaks.items()
                if streak >= self._config.fail_stable_frames
            ]
            state.consecutive_fail = max(state.mismatch_streaks.values(), default=0)

            if terminal_mismatches:
                self._fail_current_round(
                    state=state,
                    slots=slots,
                    terminal_mismatches=terminal_mismatches,
                    frame_id=frame_id,
                )
            elif slots and all(name in state.completed_classes for name in slots):
                state.status = PcbStatus.PASS_LATCHED
                state.result = PcbResult.PASS
                state.result_frame = frame_id
                state.result_timestamp = time.time()
                state.missing_classes = []
                state.last_fail_signature = None
                self._queue_result_event(state)
                self._finish_current_round(state, retry=False)
            else:
                state.missing_classes = [
                    name
                    for name, slot in slots.items()
                    if name not in state.completed_classes
                    and slot.observed_count != slot.required_count
                ]

        # 收集可输出的结果
        emitted = self._collect_emittable_results()
        return emitted

    def _update_slot_stability(
        self,
        state: PcbState,
        slots: Dict[str, SlotObservation],
    ) -> None:
        """推进单个父类 ID 内各子步骤的正确/错误数量稳定计数。"""
        for name, slot in slots.items():
            if name in state.completed_classes:
                slot.status = SlotStatus.COMPLETED
                continue

            if slot.observed_count == slot.required_count:
                slot.status = SlotStatus.MATCHING
                state.match_streaks[name] = state.match_streaks.get(name, 0) + 1
                state.mismatch_streaks[name] = 0
                if state.match_streaks[name] >= self._config.pass_stable_frames:
                    state.completed_classes.add(name)
                    slot.status = SlotStatus.COMPLETED
                continue

            slot.status = SlotStatus.MISMATCHING
            state.match_streaks[name] = 0
            state.mismatch_streaks[name] = state.mismatch_streaks.get(name, 0) + 1

    def _enqueue_new_parents(self, new_parents: List[Tuple[float, int]]) -> None:
        """把新父类按同帧从左到右加入等待队列。"""
        for _, track_id in sorted(new_parents):
            state = self._pcb_states.get(track_id)
            if state is None or state.is_decided:
                continue
            if track_id == self._current_round_id or track_id in self._waiting_parent_ids:
                continue
            state.status = PcbStatus.WAITING
            self._waiting_parent_ids.append(track_id)

    def _sync_current_round(self, now: float) -> None:
        """维护当前轮次的冷却和激活状态。"""
        if self._current_round_id is not None:
            current = self._pcb_states.get(self._current_round_id)
            if current is None or current.status == PcbStatus.RETIRED:
                self._current_round_id = None
            elif current.status == PcbStatus.COOLDOWN and current.cooldown_until <= now:
                current.status = PcbStatus.ACTIVE
                current.cooldown_until = 0.0

        if self._current_round_id is None:
            self._select_next_round(now)

    def _select_next_round(self, now: float, skip_track_id: Optional[int] = None) -> None:
        """从等待队列选择下一个仍可用的父类 ID。"""
        candidate = self._take_next_round_candidate(
            skip_track_id=skip_track_id,
            prefer_non_retry=True,
        )
        if candidate is None:
            candidate = self._take_next_round_candidate(
                skip_track_id=skip_track_id,
                prefer_non_retry=False,
            )
        if candidate is None:
            return
        track_id, state = candidate
        self._current_round_id = track_id
        self._next_attempt_id += 1
        state.attempt_id = self._next_attempt_id
        if self._next_round_ready_time > now:
            state.status = PcbStatus.COOLDOWN
            state.cooldown_until = self._next_round_ready_time
        else:
            state.status = PcbStatus.ACTIVE
            state.cooldown_until = 0.0

    def _take_next_round_candidate(
        self,
        *,
        skip_track_id: Optional[int],
        prefer_non_retry: bool,
    ) -> Optional[Tuple[int, PcbState]]:
        attempts = len(self._waiting_parent_ids)
        for _ in range(attempts):
            track_id = self._waiting_parent_ids.pop(0)
            state = self._pcb_states.get(track_id)
            if state is None or state.is_decided or state.status == PcbStatus.RETIRED:
                continue
            if skip_track_id is not None and track_id == skip_track_id:
                self._waiting_parent_ids.append(track_id)
                continue
            if prefer_non_retry and state.status == PcbStatus.FAIL_RETRY_WAITING:
                self._waiting_parent_ids.append(track_id)
                continue
            return track_id, state
        return None

    def _finish_current_round(self, state: PcbState, *, retry: bool) -> None:
        """结束当前轮次，并立即预选下一轮。"""
        if self._current_round_id != state.track_id:
            return
        self._current_round_id = None
        self._next_round_ready_time = time.time() + self._config.round_interval_seconds
        if retry:
            self._enqueue_retry_parent(state.track_id)
            self._select_next_round(time.time(), skip_track_id=state.track_id)
        else:
            self._sync_current_round(time.time())

    def _fail_current_round(
        self,
        *,
        state: PcbState,
        slots: Dict[str, SlotObservation],
        terminal_mismatches: List[str],
        frame_id: int,
    ) -> None:
        """记录本轮 FAIL，但保留该父类 ID 的后续重试资格。"""
        terminal_mismatches = sorted(terminal_mismatches)
        for name in terminal_mismatches:
            slots[name].status = SlotStatus.NG_LATCHED
        state.status = PcbStatus.FAIL_RETRY_WAITING
        state.result = PcbResult.FAIL
        state.result_frame = frame_id
        state.result_timestamp = time.time()
        state.missing_classes = terminal_mismatches
        signature = tuple(
            (
                name,
                int(slots[name].observed_count),
                int(slots[name].required_count),
            )
            for name in terminal_mismatches
            if name in slots
        )
        state.last_fail_signature = signature
        is_new_fail_signature = bool(
            signature and signature not in state.emitted_fail_signatures
        )
        if is_new_fail_signature:
            state.emitted_fail_signatures.add(signature)
        self._queue_result_event(
            state,
            is_new_fail_signature=is_new_fail_signature,
        )

        for name in slots:
            if name not in state.completed_classes:
                state.match_streaks[name] = 0
                state.mismatch_streaks[name] = 0
        state.consecutive_fail = 0
        self._finish_current_round(state, retry=True)

    def _enqueue_retry_parent(self, track_id: int) -> None:
        """FAIL 后把该父类 ID 放回等待队尾，PASS/退休不再入队。"""
        state = self._pcb_states.get(track_id)
        if state is None or state.status in (PcbStatus.PASS_LATCHED, PcbStatus.RETIRED):
            return
        if track_id not in self._waiting_parent_ids:
            state.status = PcbStatus.FAIL_RETRY_WAITING
            self._waiting_parent_ids.append(track_id)

    def _assign_parent_track_ids(
        self,
        detections: List[ObbDetection],
        frame_id: int,
    ) -> List[ObbDetection]:
        """把原生 ID 或空间匹配结果解析为不可复用的业务 ID。"""
        self._expire_parent_tracks(frame_id)
        assigned: List[Optional[ObbDetection]] = [None] * len(detections)
        unresolved_indexes: List[int] = []
        used_track_ids: set[int] = set()

        for index, detection in enumerate(detections):
            if detection.track_id is None:
                unresolved_indexes.append(index)
                continue

            source_track_id = int(detection.track_id)
            track_id = self._source_to_business_id.get(source_track_id)
            if track_id is None or track_id not in self._parent_tracks or track_id in used_track_ids:
                track_id = self._allocate_business_track_id(preferred=source_track_id)
                self._source_to_business_id[source_track_id] = track_id
            assigned[index] = replace(detection, track_id=track_id)
            self._update_parent_track(track_id, detection, frame_id, source_track_id)
            used_track_ids.add(track_id)

        candidates = []
        for index in unresolved_indexes:
            detection = detections[index]
            for track_id, track in self._parent_tracks.items():
                if track_id in used_track_ids:
                    continue
                score = self._parent_match_score(detection, track)
                if score is not None:
                    candidates.append((score, index, track_id))

        used_indexes = set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if index in used_indexes or track_id in used_track_ids:
                continue
            detection = detections[index]
            assigned[index] = replace(detection, track_id=track_id)
            self._update_parent_track(
                track_id,
                detection,
                frame_id,
                self._parent_tracks[track_id].source_track_id,
            )
            used_indexes.add(index)
            used_track_ids.add(track_id)

        for index in unresolved_indexes:
            if index in used_indexes:
                continue
            detection = detections[index]
            track_id = self._allocate_business_track_id()
            assigned[index] = replace(detection, track_id=track_id)
            self._update_parent_track(track_id, detection, frame_id, None)

        return [detection for detection in assigned if detection is not None]

    def _allocate_business_track_id(self, preferred: Optional[int] = None) -> int:
        if preferred is not None and preferred not in self._pcb_states and preferred not in self._parent_tracks:
            return preferred
        while (
            self._next_business_track_id in self._pcb_states
            or self._next_business_track_id in self._parent_tracks
        ):
            self._next_business_track_id -= 1
        track_id = self._next_business_track_id
        self._next_business_track_id -= 1
        return track_id

    def _update_parent_track(
        self,
        track_id: int,
        detection: ObbDetection,
        frame_id: int,
        source_track_id: Optional[int],
    ) -> None:
        self._parent_tracks[track_id] = _ParentTrack(
            polygon=list(detection.polygon),
            center=detection.center,
            last_seen_frame=frame_id,
            source_track_id=source_track_id,
        )

    def _parent_match_score(
        self,
        detection: ObbDetection,
        track: _ParentTrack,
    ) -> Optional[float]:
        iou = polygon_iou(detection.polygon, track.polygon)
        distance = math.dist(detection.center, track.center)
        xs = [point[0] for point in track.polygon]
        ys = [point[1] for point in track.polygon]
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        max_distance = max(30.0, diagonal * 0.75)
        if iou < self._PARENT_TRACK_MIN_IOU and distance > max_distance:
            return None
        center_score = max(0.0, 1.0 - distance / max_distance)
        return iou + center_score * 0.1

    def _expire_parent_tracks(self, frame_id: int) -> None:
        expired_ids = [
            track_id
            for track_id, track in self._parent_tracks.items()
            if frame_id - track.last_seen_frame > self._PARENT_TRACK_MAX_MISSED_FRAMES
        ]
        for track_id in expired_ids:
            track = self._parent_tracks.pop(track_id)
            if (
                track.source_track_id is not None
                and self._source_to_business_id.get(track.source_track_id) == track_id
            ):
                del self._source_to_business_id[track.source_track_id]
            state = self._pcb_states.get(track_id)
            if state is not None:
                state.status = PcbStatus.RETIRED
            if track_id == self._current_round_id:
                self._current_round_id = None
            self._waiting_parent_ids = [
                queued_id for queued_id in self._waiting_parent_ids if queued_id != track_id
            ]

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
        """返回本帧新产生的 PASS/FAIL 事件。"""
        results = list(self._result_queue)
        self._result_queue.clear()
        if results:
            self._last_emit_time = time.time()
        return results

    def _queue_result_event(
        self,
        state: PcbState,
        *,
        is_new_fail_signature: bool = False,
    ) -> None:
        """把当前状态快照写入事件队列。"""
        slot_states = {name: slot.present for name, slot in state.last_slot_states.items()}
        observed_counts = {
            name: slot.observed_count for name, slot in state.last_slot_states.items()
        }
        required_counts = {
            name: slot.required_count for name, slot in state.last_slot_states.items()
        }
        self._result_queue.append(
            PcbInspectionResult(
                track_id=state.track_id,
                result=state.result,
                attempt_id=state.attempt_id,
                is_new_fail_signature=is_new_fail_signature,
                slot_states=slot_states,
                missing_classes=list(state.missing_classes),
                observed_counts=observed_counts,
                required_counts=required_counts,
                source_frame_id=state.result_frame,
                timestamp=state.result_timestamp,
            )
        )
        if state.result == PcbResult.PASS:
            state.result_emitted = True

    def get_state_summary(self) -> Dict:
        """返回所有 PCB 状态摘要（用于 UI 显示）。"""
        summary = {}
        for tid, state in self._pcb_states.items():
            summary[tid] = {
                "status": state.status.value,
                "result": state.result.value,
                "consecutive_fail": state.consecutive_fail,
                "slots": {name: {"present": s.present, "unreliable": s.unreliable, "status": s.status.value}
                          for name, s in state.last_slot_states.items()},
                "missing_classes": list(state.missing_classes),
            }
        return summary
