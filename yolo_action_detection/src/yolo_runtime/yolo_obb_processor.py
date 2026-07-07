"""YOLO OBB 推理处理器。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from yolo_runtime.yolo_result_models import ObbDetection, DetectionOverlayState


class YoloObbProcessor:
    """加载 YOLO OBB .pt 模型，执行 track，解析 OBB 结果。"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.7,
        iou_threshold: float = 0.5,
        device: str = "",
        half: bool = False,
        tracker: str = "bytetrack.yaml",
        max_det: int = 300,
        track_persist: bool = True,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._device = device or None
        self._half = bool(half)
        self._tracker = tracker
        self._max_det = max_det
        self._track_persist = track_persist
        self._model = None
        self._class_names: dict = {}

    def load(self) -> None:
        """加载 YOLO 模型。"""
        from ultralytics import YOLO
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")
        self._model = YOLO(str(path))
        self._class_names = self._model.names

    def get_class_names(self) -> dict:
        """返回模型类别名称映射。"""
        return dict(self._class_names)

    def process_frame(
        self,
        frame: np.ndarray,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        """对单帧执行 YOLO OBB track 并返回统一检测状态。"""
        if self._model is None:
            return DetectionOverlayState(
                source_frame_id=source_frame_id,
                round_id=round_id,
                error="模型未加载",
            )

        t0 = time.perf_counter()
        try:
            results = self._model.track(
                frame,
                persist=self._track_persist,
                tracker=self._tracker,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                device=self._device,
                half=self._half,
                max_det=self._max_det,
                verbose=False,
            )
        except Exception as exc:
            return DetectionOverlayState(
                source_frame_id=source_frame_id,
                round_id=round_id,
                error=f"推理失败: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        detections: List[ObbDetection] = []

        if results and len(results) > 0:
            result = results[0]
            if result.obb is not None:
                obb = result.obb
                n = len(obb.cls)
                for i in range(n):
                    cls_id = int(obb.cls[i])
                    conf = float(obb.conf[i])
                    track_id = int(obb.id[i]) if obb.id is not None else None
                    xyxyxyxy = obb.xyxyxyxy[i].cpu().numpy()
                    polygon = [(float(xyxyxyxy[j][0]), float(xyxyxyxy[j][1])) for j in range(4)]
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    box = (min(xs), min(ys), max(xs), max(ys))
                    center = (sum(xs) / 4, sum(ys) / 4)
                    label = self._class_names.get(cls_id, str(cls_id))
                    detections.append(ObbDetection(
                        class_id=cls_id,
                        label=label,
                        conf=conf,
                        track_id=track_id,
                        polygon=polygon,
                        box=box,
                        center=center,
                    ))

        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            timestamp=time.time(),
            model_path=self._model_path,
            detections=detections,
            round_id=round_id,
            latency_ms=latency_ms,
        )
