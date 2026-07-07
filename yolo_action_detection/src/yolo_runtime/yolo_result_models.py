"""YOLO OBB 检测结果数据模型。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ObbDetection:
    """单个 OBB 检测结果。"""
    class_id: int
    label: str
    conf: float
    track_id: Optional[int]
    polygon: List[Tuple[float, float]]
    box: Tuple[float, float, float, float]
    center: Tuple[float, float]
    task_type: str = "ultralytics_obb"

    def to_dict(self, center_mm: Optional[Tuple[float, float]] = None) -> Dict[str, object]:
        """返回稳定的日志/UI 序列化结构。"""
        return {
            "class_id": int(self.class_id),
            "label": str(self.label),
            "conf": float(self.conf),
            "track_id": self.track_id,
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "box": [float(v) for v in self.box],
            "center_px": [float(self.center[0]), float(self.center[1])],
            "center_mm": (
                [float(center_mm[0]), float(center_mm[1])]
                if center_mm is not None
                else None
            ),
            "task_type": str(self.task_type),
        }


@dataclass
class DetectionOverlayState:
    """结构化检测覆盖层状态。"""
    source_frame_id: int = 0
    timestamp: float = 0.0
    model_path: str = ""
    task_type: str = "ultralytics_obb"
    detections: List[ObbDetection] = field(default_factory=list)
    status: str = "idle"
    error: str = ""
    round_id: int = 0
    latency_ms: float = 0.0


@dataclass
class PipelineStats:
    """实时管线统计。"""
    capture_fps: float = 0.0
    preview_fps: float = 0.0
    infer_fps: float = 0.0
    dropped_for_infer: int = 0
    infer_latency_ms: float = 0.0
    task_type: str = "ultralytics_obb"
    model_path: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "capture_fps": float(self.capture_fps),
            "preview_fps": float(self.preview_fps),
            "infer_fps": float(self.infer_fps),
            "dropped_for_infer": int(self.dropped_for_infer),
            "infer_latency_ms": float(self.infer_latency_ms),
            "task_type": str(self.task_type),
            "model_path": str(self.model_path),
        }
