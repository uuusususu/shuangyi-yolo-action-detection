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
    # 分段耗时：latency_ms 的口径随处理器而变，定位慢还是推理慢要靠这两项区分。
    template_ms: float = 0.0
    yolo_ms: float = 0.0
    template_matched: bool = False
    template_score: float = 0.0
    template_center: Optional[Tuple[float, float]] = None
    template_angle: Optional[float] = None
    # 顺序模式手部骨架，只给显示用；ROI / 步骤引擎不要读这个字段。
    hand_landmarks: List[dict] = field(default_factory=list)


@dataclass
class PipelineStats:
    """实时管线统计。"""
    capture_fps: float = 0.0
    preview_fps: float = 0.0
    infer_fps: float = 0.0
    dropped_for_infer: int = 0
    infer_latency_ms: float = 0.0
    # 从取到一帧到画面更新的四段耗时，用来定位"框追不上目标"到底卡在哪一段。
    capture_ms: float = 0.0
    template_ms: float = 0.0
    yolo_ms: float = 0.0
    render_ms: float = 0.0
    task_type: str = "ultralytics_obb"
    model_path: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "capture_fps": float(self.capture_fps),
            "preview_fps": float(self.preview_fps),
            "infer_fps": float(self.infer_fps),
            "dropped_for_infer": int(self.dropped_for_infer),
            "infer_latency_ms": float(self.infer_latency_ms),
            "capture_ms": float(self.capture_ms),
            "template_ms": float(self.template_ms),
            "yolo_ms": float(self.yolo_ms),
            "render_ms": float(self.render_ms),
            "task_type": str(self.task_type),
            "model_path": str(self.model_path),
        }
