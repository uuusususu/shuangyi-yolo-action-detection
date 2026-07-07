"""数据模型模块。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HandResult:
    """手部检测结果
    
    Attributes:
        landmarks: 21 个关键点坐标列表，每个点为 (x, y, z)
        handedness: 左右手标识 ("Left" 或 "Right")
        confidence: 检测置信度 (0.0-1.0)
    """
    
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    confidence: float
    
    # 关键点索引常量
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    # 手指连接关系（用于绘制）
    CONNECTIONS = [
        # 拇指
        (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP),
        (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
        # 食指
        (WRIST, INDEX_FINGER_MCP), (INDEX_FINGER_MCP, INDEX_FINGER_PIP),
        (INDEX_FINGER_PIP, INDEX_FINGER_DIP), (INDEX_FINGER_DIP, INDEX_FINGER_TIP),
        # 中指
        (WRIST, MIDDLE_FINGER_MCP), (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP),
        (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP), (MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP),
        # 无名指
        (WRIST, RING_FINGER_MCP), (RING_FINGER_MCP, RING_FINGER_PIP),
        (RING_FINGER_PIP, RING_FINGER_DIP), (RING_FINGER_DIP, RING_FINGER_TIP),
        # 小指
        (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP),
        (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
        # 手掌连接
        (INDEX_FINGER_MCP, MIDDLE_FINGER_MCP),
        (MIDDLE_FINGER_MCP, RING_FINGER_MCP),
        (RING_FINGER_MCP, PINKY_MCP),
    ]
    
    def get_landmark(self, index: int) -> Optional[Tuple[float, float, float]]:
        """获取指定索引的关键点"""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def is_left_hand(self) -> bool:
        """是否为左手"""
        return self.handedness.lower() == "left"
    
    def is_right_hand(self) -> bool:
        """是否为右手"""
        return self.handedness.lower() == "right"


@dataclass
class FrameData:
    """帧数据（用于线程间传递）
    
    Attributes:
        timestamp: 帧时间戳
        fps: 当前帧率
        hands: 检测到的手部列表
        inference_active: 推理是否激活
    """
    
    timestamp: float
    fps: float
    hands: List[HandResult]
    inference_active: bool


@dataclass
class PreviewFrame:
    """最新预览帧元数据。"""

    frame_id: int
    timestamp: float
    image: object
    width: int
    height: int


@dataclass
class DetectionOverlayState:
    """结构化检测覆盖层状态。"""

    source_frame_id: int = 0
    timestamp: float = 0.0
    model_path: str = ""
    task_type: str = "mediapipe_gesture"
    detections: List[Dict] = field(default_factory=list)
    classification: Dict = field(default_factory=dict)
    matched_category_names: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    status: str = "idle"
    error: str = ""
    round_id: int = 0
    official_speed_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineStats:
    """实时管线统计。"""

    capture_fps: float = 0.0
    preview_fps: float = 0.0
    infer_fps: float = 0.0
    dropped_for_infer: int = 0
    preview_latency_ms: float = 0.0
    infer_latency_ms: float = 0.0
    display_result_age_ms: float = 0.0
    display_source_frame_id: int = 0
    display_strategy: str = "preview_overlay"
    task_type: str = ""
    model_path: str = ""
    config_path: str = ""
    result_count: int = 0
    has_boxes: bool = False
    has_masks: bool = False
    has_keypoints: bool = False
    has_probs: bool = False
    processor_latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "capture_fps": float(self.capture_fps),
            "preview_fps": float(self.preview_fps),
            "infer_fps": float(self.infer_fps),
            "dropped_for_infer": int(self.dropped_for_infer),
            "preview_latency_ms": float(self.preview_latency_ms),
            "infer_latency_ms": float(self.infer_latency_ms),
            "display_result_age_ms": float(self.display_result_age_ms),
            "display_source_frame_id": int(self.display_source_frame_id),
            "display_strategy": str(self.display_strategy),
            "task_type": str(self.task_type),
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "result_count": int(self.result_count),
            "has_boxes": bool(self.has_boxes),
            "has_masks": bool(self.has_masks),
            "has_keypoints": bool(self.has_keypoints),
            "has_probs": bool(self.has_probs),
            "processor_latency_ms": dict(self.processor_latency_ms),
        }


@dataclass
class RoundProgress:
    """当前轮次进度。"""

    round_id: int = 1
    seen_targets: List[str] = field(default_factory=list)
    target_count: int = 0
    completed_count: int = 0
    holding: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_id": int(self.round_id),
            "seen_targets": list(self.seen_targets),
            "target_count": int(self.target_count),
            "completed_count": int(self.completed_count),
            "holding": bool(self.holding),
        }


@dataclass
class RoundCompleted:
    """轮次完成事件。"""

    round_id: int
    completed_targets: List[str]
    completed_at: float
    duration_ms: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_id": int(self.round_id),
            "completed_targets": list(self.completed_targets),
            "completed_at": float(self.completed_at),
            "duration_ms": int(self.duration_ms),
        }
