"""不依赖产品 ID 或追踪的 ROI 定位轮次引擎。"""
from __future__ import annotations

import logging
import time
from typing import Iterable, List

from roi_inspection.geometry import detection_center_in_rect
from roi_inspection.models import (
    CategoryInspectionState,
    InspectionOutcome,
    RoiInspectionConfig,
    RoiInspectionResult,
    RoiInspectionSnapshot,
    RoundPhase,
)
from yolo_runtime.yolo_result_models import ObbDetection

logger = logging.getLogger("yolo_action_detection.runtime")


class RoiInspectionEngine:
    """单帧定位触发、分类别稳定帧锁存的一轮判定引擎。"""

    def __init__(self, config: RoiInspectionConfig) -> None:
        self._config = config
        self._round_id = 1
        self._hold_until = 0.0
        self._snapshot = self._new_snapshot()
        logger.info(
            "roi_engine_init template_enabled=%s inspection_roi=%s required=%s "
            "pass_frames=%s fail_frames=%s interval=%s",
            self._config.template_enabled,
            self._config.inspection_roi,
            self._config.required_counts,
            self._config.pass_stable_frames,
            self._config.fail_stable_frames,
            self._config.round_interval_seconds,
        )

    @property
    def snapshot(self) -> RoiInspectionSnapshot:
        return self._snapshot

    def reset(self) -> None:
        """清空运行态；生产统计由调用方独立保留。"""
        self._round_id = 1
        self._hold_until = 0.0
        self._snapshot = self._new_snapshot()

    def update(
        self,
        detections: Iterable[ObbDetection],
        image_size: tuple[int, int],
        *,
        source_frame_id: int = 0,
        timestamp: float | None = None,
        template_match: dict | None = None,
    ) -> List[RoiInspectionResult]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        current = list(detections)
        det_summary = {}
        for det in current:
            det_summary[det.label] = det_summary.get(det.label, 0) + 1

        if self._snapshot.phase == RoundPhase.RESULT_HOLD:
            remaining = self._hold_until - now
            if now < self._hold_until:
                logger.debug(
                    "roi_update frame_id=%s phase=result_hold hold_remaining=%.3fs "
                    "det_count=%d labels=%s image_size=%s -> skipped",
                    source_frame_id,
                    remaining,
                    len(current),
                    det_summary,
                    tuple(image_size),
                )
                return []
            logger.info(
                "roi_update frame_id=%s phase=result_hold hold_expired -> start_next_round",
                source_frame_id,
            )
            self._round_id += 1
            self._snapshot = self._new_snapshot()

        if self._snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR:
            triggered = self._check_round_trigger(
                current, image_size, source_frame_id, det_summary, template_match
            )
            if not triggered:
                return []
            self._snapshot.phase = RoundPhase.INSPECTING
            logger.info(
                "roi_update frame_id=%s phase=inspecting round_id=%d -> round_triggered",
                source_frame_id,
                self._round_id,
            )

        return self._inspect(
            current,
            image_size,
            source_frame_id=int(source_frame_id),
            timestamp=now,
        )

    def _check_round_trigger(
        self,
        current: List[ObbDetection],
        image_size: tuple[int, int],
        source_frame_id: int,
        det_summary: dict,
        template_match: dict | None,
    ) -> bool:
        """WAITING_FOR_LOCATOR 阶段的触发判定。"""
        if self._config.has_template_locator:
            return self._check_template_trigger(current, image_size, source_frame_id, det_summary, template_match)
        return self._check_inspection_roi_trigger(current, image_size, source_frame_id, det_summary)

    def _check_template_trigger(
        self,
        current: List[ObbDetection],
        image_size: tuple[int, int],
        source_frame_id: int,
        det_summary: dict,
        template_match: dict | None,
    ) -> bool:
        """模板匹配定位：overlay 携带 matched=True 即触发本轮。"""
        matched = bool(template_match and template_match.get("matched"))
        score = float(template_match.get("score", 0.0)) if template_match else 0.0
        center = template_match.get("center") if template_match else None
        angle = template_match.get("angle") if template_match else None
        self._snapshot.template_matched = matched
        self._snapshot.template_score = score
        logger.info(
            "roi_update frame_id=%s phase=waiting_for_template matched=%s score=%.3f "
            "center=%s angle=%s det_count=%d labels=%s -> triggered=%s",
            source_frame_id,
            matched,
            score,
            center,
            angle,
            len(current),
            det_summary,
            matched,
        )
        return matched

    def _check_inspection_roi_trigger(
        self,
        current: List[ObbDetection],
        image_size: tuple[int, int],
        source_frame_id: int,
        det_summary: dict,
    ) -> bool:
        """无模板定位：识别 ROI 内出现任意配置类别中心即触发本轮。"""
        inspection_roi_pixel = self._config.inspection_roi.to_pixel_rect(image_size)
        in_roi_by_class: dict = {}
        in_roi_centers: List[tuple] = []
        for det in current:
            if det.label not in self._config.required_counts:
                continue
            if detection_center_in_rect(det, self._config.inspection_roi, image_size):
                in_roi_by_class[det.label] = in_roi_by_class.get(det.label, 0) + 1
                in_roi_centers.append((det.label, tuple(float(v) for v in det.center)))
        triggered = bool(in_roi_by_class)
        logger.info(
            "roi_update frame_id=%s phase=waiting_for_inspection_trigger "
            "inspection_roi_pixel=(%.1f,%.1f,%.1f,%.1f) image_size=%s det_count=%d labels=%s "
            "in_roi_classes=%s in_roi_centers=%s -> triggered=%s",
            source_frame_id,
            inspection_roi_pixel[0],
            inspection_roi_pixel[1],
            inspection_roi_pixel[2],
            inspection_roi_pixel[3],
            tuple(image_size),
            len(current),
            det_summary,
            in_roi_by_class,
            in_roi_centers,
            triggered,
        )
        return triggered

    def _inspect(
        self,
        detections: List[ObbDetection],
        image_size: tuple[int, int],
        *,
        source_frame_id: int,
        timestamp: float,
    ) -> List[RoiInspectionResult]:
        inspection_roi_pixel = self._config.inspection_roi.to_pixel_rect(image_size)
        failed_categories: List[str] = []
        per_category_log: List[str] = []
        for name, state in self._snapshot.category_states.items():
            if state.passed:
                per_category_log.append(
                    f"{name}:passed required={state.required_count}"
                )
                continue
            in_roi_centers: List[tuple] = []
            for det in detections:
                if det.label != name:
                    continue
                if detection_center_in_rect(det, self._config.inspection_roi, image_size):
                    in_roi_centers.append(tuple(float(v) for v in det.center))
            observed = len(in_roi_centers)
            state.observed_count = observed
            if observed == state.required_count:
                state.match_streak += 1
                state.mismatch_streak = 0
                if state.match_streak >= self._config.pass_stable_frames:
                    state.passed = True
            else:
                state.match_streak = 0
                state.mismatch_streak += 1
                if state.mismatch_streak >= self._config.fail_stable_frames:
                    failed_categories.append(name)
            per_category_log.append(
                f"{name}:observed={observed}/required={state.required_count} "
                f"match_streak={state.match_streak}/{self._config.pass_stable_frames} "
                f"mismatch_streak={state.mismatch_streak}/{self._config.fail_stable_frames} "
                f"passed={state.passed} centers={in_roi_centers}"
            )

        logger.info(
            "roi_inspect frame_id=%s round_id=%d roi_pixel=(%.1f,%.1f,%.1f,%.1f) "
            "image_size=%s categories=[%s] failed=%s",
            source_frame_id,
            self._round_id,
            inspection_roi_pixel[0],
            inspection_roi_pixel[1],
            inspection_roi_pixel[2],
            inspection_roi_pixel[3],
            tuple(image_size),
            " | ".join(per_category_log),
            failed_categories,
        )

        if failed_categories:
            return [self._finish(InspectionOutcome.FAIL, failed_categories, source_frame_id, timestamp)]
        if all(state.passed for state in self._snapshot.category_states.values()):
            return [self._finish(InspectionOutcome.PASS, [], source_frame_id, timestamp)]
        return []

    def _finish(
        self,
        outcome: InspectionOutcome,
        failed_categories: List[str],
        source_frame_id: int,
        timestamp: float,
    ) -> RoiInspectionResult:
        result = RoiInspectionResult(
            round_id=self._round_id,
            outcome=outcome,
            actual_counts={
                name: state.observed_count for name, state in self._snapshot.category_states.items()
            },
            required_counts=dict(self._config.required_counts),
            failed_categories=list(failed_categories),
            source_frame_id=source_frame_id,
            timestamp=timestamp,
            feedback_key=f"roi-round-{self._round_id}-{outcome.value}",
        )
        self._snapshot.phase = RoundPhase.RESULT_HOLD
        self._snapshot.result = result
        self._hold_until = timestamp + self._config.round_interval_seconds
        logger.info(
            "roi_round_finished frame_id=%s round_id=%d outcome=%s actual=%s required=%s "
            "failed=%s hold_until=%.3f interval=%.3fs",
            source_frame_id,
            self._round_id,
            outcome.value,
            result.actual_counts,
            result.required_counts,
            failed_categories,
            self._hold_until,
            self._config.round_interval_seconds,
        )
        return result

    def _new_snapshot(self) -> RoiInspectionSnapshot:
        return RoiInspectionSnapshot(
            phase=RoundPhase.WAITING_FOR_LOCATOR,
            round_id=self._round_id,
            template_matched=False,
            template_score=0.0,
            category_states={
                name: CategoryInspectionState(class_name=name, required_count=count)
                for name, count in self._config.required_counts.items()
            },
        )
