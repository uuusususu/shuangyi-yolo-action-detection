"""形状匹配门控 YOLO processor：定位命中后才运行 YOLO。

每帧先在搜索 ROI 内做形状匹配（Halcon 风格的定向边缘点集 + 梯度方向点积）：
- 命中 -> 调内部 processor 跑 YOLO，overlay 附带定位结果
- 未命中 -> 返回空检测 overlay，不跑 YOLO

形状匹配同时充当"到位信号"：命中即认为工件到位，可以开始本轮计数。
"""
from __future__ import annotations

import logging
import time

import numpy as np

from template_matching.match_diagnostics import get_recorder
from template_matching.shape_model import ShapeModel
from yolo_runtime.yolo_result_models import DetectionOverlayState

logger = logging.getLogger("yolo_action_detection.runtime")


class TemplateGatedProcessor:
    """用搜索 ROI 内的形状匹配门控 YOLO 推理。"""

    def __init__(
        self,
        inner_processor,
        template_model: ShapeModel,
        search_rect: object,
    ) -> None:
        self._inner = inner_processor
        self._template_model = template_model
        self._search_rect = search_rect

    @property
    def template_model(self) -> ShapeModel:
        return self._template_model

    @property
    def search_rect(self) -> object:
        return self._search_rect

    def get_class_names(self) -> dict:
        if hasattr(self._inner, "get_class_names"):
            return self._inner.get_class_names()
        return {}

    def process_frame(
        self,
        frame: np.ndarray,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        t0 = time.perf_counter()
        result = self._template_model.find(frame, self._search_rect)

        if not result.matched:
            overlay = DetectionOverlayState(
                source_frame_id=source_frame_id,
                timestamp=time.time(),
                task_type="template_gated",
                detections=[],
                status=(
                    f"template_not_matched:{result.failure}"
                    if result.failure else "template_not_matched"
                ),
                template_matched=False,
                template_score=result.score,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            overlay = self._inner.process_frame(
                frame, source_frame_id=source_frame_id, round_id=round_id
            )
            overlay.template_matched = True
            overlay.template_score = result.score
            overlay.template_center = result.center
            overlay.template_angle = result.angle

        self._record_diagnostics(frame, result, overlay, source_frame_id, round_id)
        return overlay

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def _record_diagnostics(self, frame, result, overlay, source_frame_id, round_id) -> None:
        recorder = get_recorder()
        if not recorder.active:
            return
        try:
            payload = self._build_payload(frame, result, overlay, source_frame_id, round_id)
            recorder.record(payload, sample_image=frame)
        except Exception as exc:  # 诊断绝不影响生产判定
            logger.debug("template_diagnostics=record_failed error=%s", exc)

    def _build_payload(self, frame, result, overlay, source_frame_id, round_id) -> dict:
        height, width = frame.shape[:2]
        model = self._template_model
        settings = model.settings
        threshold = float(settings.min_score)
        return {
            "frame_id": int(source_frame_id),
            "round_id": int(round_id),
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "matcher": "shape",
            "matched": bool(result.matched),
            "score": float(result.score),
            "threshold": threshold,
            "margin": float(result.score) - threshold,
            "angle": result.angle,
            "center": list(result.center) if result.center else None,
            "outside_ratio": float(result.outside_ratio),
            "failure": str(result.failure or ""),
            "angle_scores": [[float(a), float(s)] for a, s in result.angle_scores],
            "level_candidates": list(result.level_candidates),
            "levels_used": int(result.levels_used),
            "search_rect_px": list(result.search_roi_px) if result.search_roi_px else None,
            "center_range_px": list(result.center_range_px) if result.center_range_px else None,
            "center_travel_px": list(result.center_travel_px) if result.center_travel_px else None,
            "required_roi_px": list(result.required_roi_px) if result.required_roi_px else None,
            "rotated_bbox_px": list(result.rotated_bbox_px) if result.rotated_bbox_px else None,
            "frame_size": [int(width), int(height)],
            "template": {
                "size": list(model.template_size),
                "levels": int(model.level_count),
                "point_counts": list(model.point_counts),
                "base_angle_step": float(model.base_angle_step),
                "polarity": str(settings.polarity),
                "angle_start": float(settings.angle_start),
                "angle_extent": float(settings.angle_extent),
            },
            "search_stats": self._search_stats(frame, result),
            "coarse_ms": float(result.coarse_ms),
            "refine_ms": float(result.refine_ms),
            "total_ms": float(result.total_ms),
            "yolo": self._yolo_stats(result, overlay),
        }

    @staticmethod
    def _search_stats(frame: np.ndarray, result) -> dict:
        """搜索域的曝光与梯度统计，用于识别过曝/欠曝造成的无轮廓帧。"""
        rect = result.search_roi_px
        if not rect:
            return {}
        x0, y0, x1, y1 = (int(v) for v in rect)
        if x1 <= x0 or y1 <= y0:
            return {}
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return {}
        gray = crop if crop.ndim == 2 else crop[:, :, 0]
        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "p01": float(np.percentile(gray, 1)),
            "p99": float(np.percentile(gray, 99)),
            "gradient_pixels": int(result.search_gradient_pixels),
            "edge_pixels": int(result.search_gradient_pixels),
            "area_px": int((x1 - x0) * (y1 - y0)),
        }

    @staticmethod
    def _yolo_stats(result, overlay) -> dict:
        if not result.matched:
            return {"ran": False, "det_count": 0, "labels": {}}
        labels: dict[str, int] = {}
        for detection in overlay.detections or []:
            label = str(getattr(detection, "label", ""))
            if label:
                labels[label] = labels.get(label, 0) + 1
        return {
            "ran": True,
            "det_count": len(overlay.detections or []),
            "labels": labels,
            "latency_ms": float(overlay.latency_ms or 0.0),
            "error": str(overlay.error or ""),
        }
