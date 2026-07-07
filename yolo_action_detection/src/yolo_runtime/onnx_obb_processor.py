"""Pure ONNX Runtime YOLO OBB processor."""
from __future__ import annotations

import ast
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


class OnnxObbProcessor:
    """Run YOLO OBB ONNX directly with ONNX Runtime and project-local postprocess."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.7,
        iou_threshold: float = 0.5,
        max_det: int = 300,
        providers: Optional[Sequence[str]] = None,
        session_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = float(conf_threshold)
        self._iou_threshold = float(iou_threshold)
        self._max_det = int(max_det)
        self._providers = list(providers) if providers else ["CPUExecutionProvider"]
        self._session_factory = session_factory

        self._session = None
        self._input_name = ""
        self._output_names: List[str] = []
        self._input_size: Tuple[int, int] = (640, 640)  # width, height
        self._class_names: Dict[int, str] = {}

    def load(self) -> None:
        """Load ONNX Runtime session and metadata."""
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")

        if self._session_factory is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(path), providers=self._providers)
        else:
            self._session = self._session_factory(str(path), providers=self._providers)

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs:
            raise ValueError("ONNX 模型没有输入节点")
        if not outputs:
            raise ValueError("ONNX 模型没有输出节点")

        self._input_name = inputs[0].name
        self._output_names = [out.name for out in outputs]
        self._input_size = self._resolve_input_size(inputs[0].shape)
        self._class_names = self._resolve_class_names()

    def get_class_names(self) -> dict:
        """Return model class name mapping."""
        return dict(self._class_names)

    def process_frame(
        self,
        frame: np.ndarray,
        source_frame_id: int = 0,
        round_id: int = 0,
    ) -> DetectionOverlayState:
        """Run ONNX OBB inference for one BGR frame."""
        if self._session is None:
            return DetectionOverlayState(
                source_frame_id=source_frame_id,
                round_id=round_id,
                model_path=self._model_path,
                error="模型未加载",
            )

        t0 = time.perf_counter()
        try:
            input_tensor, meta = self._preprocess(frame)
            outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
            detections = self._postprocess(outputs[0], meta)
        except Exception as exc:
            return DetectionOverlayState(
                source_frame_id=source_frame_id,
                round_id=round_id,
                model_path=self._model_path,
                task_type="onnx_obb",
                error=f"推理失败: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            timestamp=time.time(),
            model_path=self._model_path,
            task_type="onnx_obb",
            detections=detections,
            round_id=round_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _resolve_input_size(self, shape: Sequence[object]) -> Tuple[int, int]:
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            return int(shape[3]), int(shape[2])
        meta = self._metadata()
        imgsz = meta.get("imgsz", "")
        try:
            parsed = ast.literal_eval(imgsz)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return int(parsed[1]), int(parsed[0])
        except (SyntaxError, ValueError, TypeError):
            pass
        return 640, 640

    def _resolve_class_names(self) -> Dict[int, str]:
        meta = self._metadata()
        names_raw = meta.get("names", "")
        try:
            names = ast.literal_eval(names_raw)
        except (SyntaxError, ValueError):
            names = {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def _metadata(self) -> Dict[str, str]:
        if self._session is None:
            return {}
        try:
            return dict(self._session.get_modelmeta().custom_metadata_map)
        except Exception:
            return {}

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        if frame is None or frame.ndim != 3:
            raise ValueError("输入帧必须是 HxWx3 图像")
        target_w, target_h = self._input_size
        src_h, src_w = frame.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized_w = int(round(src_w * scale))
        resized_h = int(round(src_h * scale))
        pad_w = target_w - resized_w
        pad_h = target_h - resized_h
        left = int(round(pad_w / 2 - 0.1))
        right = int(round(pad_w / 2 + 0.1))
        top = int(round(pad_h / 2 - 0.1))
        bottom = int(round(pad_h / 2 + 0.1))

        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor[None])
        return tensor, {
            "scale": scale,
            "pad": (left, top),
            "source_shape": (src_h, src_w),
        }

    def _postprocess(self, output: np.ndarray, meta: Dict[str, object]) -> List[ObbDetection]:
        predictions = self._normalize_output(output)
        if predictions.size == 0:
            return []

        num_classes = len(self._class_names) or max(0, predictions.shape[1] - 5)
        if predictions.shape[1] < 5 or num_classes <= 0:
            return []

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4 : 4 + num_classes]
        angles = predictions[:, 4 + num_classes]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = scores >= self._conf_threshold
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        angles = angles[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        selected = self._rotated_nms(boxes, angles, scores, class_ids)
        detections: List[ObbDetection] = []
        for idx in selected[: self._max_det]:
            detections.append(
                self._candidate_to_detection(
                    boxes[idx],
                    float(angles[idx]),
                    float(scores[idx]),
                    int(class_ids[idx]),
                    meta,
                )
            )
        detections.sort(key=lambda det: det.conf, reverse=True)
        return detections

    def _normalize_output(self, output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)
        expected_cols = 5 + len(self._class_names) if self._class_names else 0
        if expected_cols:
            if arr.shape[0] == expected_cols and arr.shape[1] != expected_cols:
                arr = arr.T
            return arr.astype(np.float32, copy=False)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32, copy=False)

    def _rotated_nms(
        self,
        boxes: np.ndarray,
        angles: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[int]:
        kept: List[int] = []
        for cls_id in sorted(set(int(v) for v in class_ids.tolist())):
            idxs = np.where(class_ids == cls_id)[0]
            rotated_rects = [
                (
                    (float(boxes[i][0]), float(boxes[i][1])),
                    (max(float(boxes[i][2]), 1.0), max(float(boxes[i][3]), 1.0)),
                    math.degrees(float(angles[i])),
                )
                for i in idxs
            ]
            class_scores = [float(scores[i]) for i in idxs]
            nms = cv2.dnn.NMSBoxesRotated(
                rotated_rects,
                class_scores,
                self._conf_threshold,
                self._iou_threshold,
                top_k=self._max_det,
            )
            for local_i in _flatten_indices(nms):
                kept.append(int(idxs[local_i]))
        kept.sort(key=lambda i: float(scores[i]), reverse=True)
        return kept

    def _candidate_to_detection(
        self,
        box_xywh: np.ndarray,
        angle: float,
        score: float,
        class_id: int,
        meta: Dict[str, object],
    ) -> ObbDetection:
        rect = (
            (float(box_xywh[0]), float(box_xywh[1])),
            (max(float(box_xywh[2]), 1.0), max(float(box_xywh[3]), 1.0)),
            math.degrees(float(angle)),
        )
        points = cv2.boxPoints(rect)
        polygon = [self._unletterbox_point(float(x), float(y), meta) for x, y in points]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        center = (sum(xs) / 4.0, sum(ys) / 4.0)
        label = self._class_names.get(class_id, str(class_id))
        return ObbDetection(
            class_id=class_id,
            label=label,
            conf=score,
            track_id=None,
            polygon=polygon,
            box=(min(xs), min(ys), max(xs), max(ys)),
            center=center,
            task_type="onnx_obb",
        )

    def _unletterbox_point(
        self,
        x: float,
        y: float,
        meta: Dict[str, object],
    ) -> Tuple[float, float]:
        scale = float(meta["scale"])
        pad_x, pad_y = meta["pad"]
        src_h, src_w = meta["source_shape"]
        ox = (x - float(pad_x)) / scale
        oy = (y - float(pad_y)) / scale
        ox = min(max(ox, 0.0), float(src_w - 1))
        oy = min(max(oy, 0.0), float(src_h - 1))
        return ox, oy


def _flatten_indices(indices) -> List[int]:
    """Normalize OpenCV NMS index return variants."""
    if indices is None:
        return []
    arr = np.asarray(indices)
    if arr.size == 0:
        return []
    return [int(v) for v in arr.reshape(-1).tolist()]
