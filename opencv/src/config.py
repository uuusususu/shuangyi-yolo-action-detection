"""配置管理器模块"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2

DEFAULT_CLASS_MAPPING: Dict[int, str] = {
    0: "hold",
    1: "peace",
    2: "tear",
    3: "left_press",
}
DEFAULT_MEDIAPIPE_TASK_PATH = "config/models/gesture_recognizer.task"
DEFAULT_OBJECT_MODEL_PATH = "config/models/box_detector.tflite"
DEFAULT_YOLO_MODEL_PATH = "../models/best.pt"
MEDIAPIPE_MODEL_TASK = "mediapipe_gesture"
MEDIAPIPE_OBJECT_TASK = "mediapipe_object_detection"
MEDIAPIPE_COMPOSITE_TASK = "mediapipe_object_detection+mediapipe_gesture"
SUPPORTED_MODEL_TASKS: Tuple[str, ...] = (MEDIAPIPE_MODEL_TASK, MEDIAPIPE_OBJECT_TASK, MEDIAPIPE_COMPOSITE_TASK)
MODEL_TASK_DISPLAY_NAMES: Dict[str, str] = {
    MEDIAPIPE_MODEL_TASK: "MediaPipe 手势",
    MEDIAPIPE_OBJECT_TASK: "MediaPipe 目标检测",
    MEDIAPIPE_COMPOSITE_TASK: "MediaPipe 目标检测 + 手势",
}
LEGACY_MODEL_CONFIG_KEYS: Tuple[str, ...] = (
    "onnx_model_path",
    "onnx_input_shape",
    "conf_threshold",
    "iou_threshold",
    "ultralytics_rect",
    "ultralytics_half",
    "ultralytics_device",
    "ultralytics_max_det",
    "ultralytics_track_persist",
    "ultralytics_tracker",
    "hand_temporal_enabled",
    "hand_track_max_miss_frames",
    "hand_kpt_smoothing_alpha",
    "hand_point_conf_threshold",
    "hand_point_hold_frames",
    "action_enter_frames",
    "action_exit_frames",
    "class_mapping",
)


@dataclass
class ConfigManager:
    """配置管理器，支持 JSON 持久化"""
    
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    camera_index: int = 0
    camera_backend: int = int(cv2.CAP_DSHOW)
    camera_max_index: int = 10
    mvsdk_friendly_name: str = ""
    
    # 手势检测阈值
    pinch_open_threshold: float = 1.2
    finger_extended_angle: float = 160.0
    finger_curled_angle: float = 120.0

    static_n_on: int = 3
    static_n_off: int = 6

    # 手势触发器配置
    gesture_n_on: int = 3
    gesture_hold_seconds: float = 2.0
    
    # 调试选项
    debug_gesture_overlay: bool = False
    show_confidence_overlay: bool = True
    round_cooldown_seconds: float = 2.0
    today_target_capacity: int = 300
    
    # 类别名称配置
    category_names: List[str] = field(default_factory=lambda: ["fang", "wo", "", "", "", ""])
    
    # 模型配置
    model_path: str = DEFAULT_MEDIAPIPE_TASK_PATH
    model_task: str = MEDIAPIPE_MODEL_TASK
    mediapipe_task_path: str = DEFAULT_MEDIAPIPE_TASK_PATH
    enable_gesture_detection: bool = True
    enable_object_detection: bool = False
    object_model_path: str = DEFAULT_OBJECT_MODEL_PATH
    object_score_threshold: float = 0.3
    object_max_results: int = 5
    object_result_hold_ms: int = 250
    yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH
    yolo_conf_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    mediapipe_num_hands: int = 2
    mediapipe_score_threshold: float = 0.65
    mediapipe_min_hand_detection_confidence: float = 0.5
    mediapipe_min_hand_presence_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    gesture_window_size: int = 5
    gesture_enter_frames: int = 3
    gesture_exit_frames: int = 5

    # Legacy model fields retained for loading older config files only.
    onnx_model_path: str = "../models/best.onnx"
    onnx_input_shape: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    ultralytics_rect: bool = True
    ultralytics_half: bool = False
    ultralytics_device: str = ""
    ultralytics_max_det: int = 300
    ultralytics_track_persist: bool = True
    ultralytics_tracker: str = "botsort.yaml"
    display_sync_to_inference: bool = True
    display_result_max_age_ms: float = 250.0
    hand_temporal_enabled: bool = True
    hand_track_max_miss_frames: int = 8
    hand_kpt_smoothing_alpha: float = 0.45
    hand_point_conf_threshold: float = 0.35
    hand_point_hold_frames: int = 4
    action_enter_frames: int = 4
    action_exit_frames: int = 6
    joint_tracking_enabled: bool = True
    tool_class_name: str = "扭力枪"
    track_seed_stable_frames: int = 3
    track_enter_stable_frames: int = 2
    track_leave_stable_frames: int = 4
    track_gun_lost_frames: int = 6
    track_hole_occlusion_grace_frames: int = 12
    track_enter_threshold: float = 0.18
    track_leave_threshold: float = 0.08
    track_min_score_gap: float = 0.05
    track_order_constraint_enabled: bool = True
    track_order_rule: str = "configured_order"
    track_out_of_order_frames: int = 2
    track_step_leave_frames: int = 4
    track_debug_csv_enabled: bool = False
    detection_log_enabled: bool = True
    detection_log_dir: str = "logs/detections"
    detection_log_every_n_frames: int = 1
    detection_log_flush_interval: int = 30
    class_mapping: Dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_MAPPING)
    )
    
    _config_path: Optional[Path] = None
    
    def __post_init__(self):
        """初始化后验证参数"""
        self.validate()
    
    def validate(self) -> bool:
        """验证配置参数范围"""
        errors = []
        
        if not 1 <= self.max_num_hands <= 2:
            errors.append(f"max_num_hands 必须在 1-2 范围内")
            self.max_num_hands = max(1, min(2, self.max_num_hands))
        
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            errors.append(f"min_detection_confidence 必须在 0.0-1.0 范围内")
            self.min_detection_confidence = max(0.0, min(1.0, self.min_detection_confidence))
        
        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            errors.append(f"min_tracking_confidence 必须在 0.0-1.0 范围内")
            self.min_tracking_confidence = max(0.0, min(1.0, self.min_tracking_confidence))
        
        if self.camera_index < 0:
            self.camera_index = 0

        if int(self.camera_max_index) < 0:
            self.camera_max_index = 0

        if int(self.camera_backend) < 0:
            self.camera_backend = int(cv2.CAP_DSHOW)

        if self.mvsdk_friendly_name is None:
            self.mvsdk_friendly_name = ""
        if not isinstance(self.mvsdk_friendly_name, str):
            self.mvsdk_friendly_name = str(self.mvsdk_friendly_name)
        
        for key in ("finger_extended_angle", "finger_curled_angle"):
            value = getattr(self, key)
            if not 0.0 <= float(value) <= 180.0:
                setattr(self, key, max(0.0, min(180.0, float(value))))

        if float(self.finger_extended_angle) < float(self.finger_curled_angle):
            self.finger_extended_angle, self.finger_curled_angle = (
                float(self.finger_curled_angle),
                float(self.finger_extended_angle),
            )

        if self.static_n_on < 1:
            self.static_n_on = 1

        if self.static_n_off < 1:
            self.static_n_off = 1

        self.show_confidence_overlay = bool(getattr(self, "show_confidence_overlay", True))

        try:
            self.round_cooldown_seconds = max(
                0.1, float(getattr(self, "round_cooldown_seconds", 2.0))
            )
        except (TypeError, ValueError):
            self.round_cooldown_seconds = 2.0

        try:
            self.today_target_capacity = max(
                1, int(getattr(self, "today_target_capacity", 300))
            )
        except (TypeError, ValueError):
            self.today_target_capacity = 300

        try:
            self.conf_threshold = max(0.0, min(1.0, float(self.conf_threshold)))
        except (TypeError, ValueError):
            self.conf_threshold = 0.5

        try:
            self.iou_threshold = max(0.0, min(1.0, float(self.iou_threshold)))
        except (TypeError, ValueError):
            self.iou_threshold = 0.45

        def _clamp_float(field_name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> None:
            try:
                value = float(getattr(self, field_name, default))
            except (TypeError, ValueError):
                value = default
            setattr(self, field_name, max(minimum, min(maximum, value)))

        requested_task_path = str(getattr(self, "mediapipe_task_path", "") or "").strip()
        legacy_model_path = str(getattr(self, "model_path", "") or "").strip()
        if not requested_task_path and legacy_model_path.lower().endswith(".task"):
            requested_task_path = legacy_model_path
        if not requested_task_path or not requested_task_path.lower().endswith(".task"):
            requested_task_path = DEFAULT_MEDIAPIPE_TASK_PATH
        self.mediapipe_task_path = requested_task_path
        self.model_path = requested_task_path

        self.enable_gesture_detection = bool(getattr(self, "enable_gesture_detection", True))
        self.enable_object_detection = bool(getattr(self, "enable_object_detection", False))
        raw_object_model_path = str(getattr(self, "object_model_path", "") or "").strip()
        self.object_model_path = raw_object_model_path or DEFAULT_OBJECT_MODEL_PATH
        _clamp_float("object_score_threshold", 0.3)
        try:
            self.object_max_results = max(1, int(getattr(self, "object_max_results", 5)))
        except (TypeError, ValueError):
            self.object_max_results = 5
        try:
            self.object_result_hold_ms = max(0, int(getattr(self, "object_result_hold_ms", 250)))
        except (TypeError, ValueError):
            self.object_result_hold_ms = 250
        self.yolo_model_path = str(getattr(self, "yolo_model_path", DEFAULT_YOLO_MODEL_PATH) or DEFAULT_YOLO_MODEL_PATH).strip()
        _clamp_float("yolo_conf_threshold", 0.5)
        _clamp_float("yolo_iou_threshold", 0.45)

        legacy_onnx_model_path = str(getattr(self, "onnx_model_path", "") or "").strip()
        self.onnx_model_path = legacy_onnx_model_path or "../models/best.onnx"
        self.model_task = MEDIAPIPE_MODEL_TASK

        try:
            self.mediapipe_num_hands = max(1, min(2, int(getattr(self, "mediapipe_num_hands", 2))))
        except (TypeError, ValueError):
            self.mediapipe_num_hands = 2
        _clamp_float("mediapipe_score_threshold", 0.65)
        _clamp_float("mediapipe_min_hand_detection_confidence", 0.5)
        _clamp_float("mediapipe_min_hand_presence_confidence", 0.5)
        _clamp_float("mediapipe_min_tracking_confidence", 0.5)
        try:
            self.gesture_window_size = max(1, int(getattr(self, "gesture_window_size", 5)))
        except (TypeError, ValueError):
            self.gesture_window_size = 5
        try:
            self.gesture_enter_frames = max(1, int(getattr(self, "gesture_enter_frames", 3)))
        except (TypeError, ValueError):
            self.gesture_enter_frames = 3
        try:
            self.gesture_exit_frames = max(1, int(getattr(self, "gesture_exit_frames", 5)))
        except (TypeError, ValueError):
            self.gesture_exit_frames = 5

        self.ultralytics_rect = bool(getattr(self, "ultralytics_rect", True))
        self.ultralytics_half = bool(getattr(self, "ultralytics_half", False))
        self.ultralytics_device = str(getattr(self, "ultralytics_device", "") or "").strip()

        try:
            self.ultralytics_max_det = max(1, int(getattr(self, "ultralytics_max_det", 300)))
        except (TypeError, ValueError):
            self.ultralytics_max_det = 300
        self.ultralytics_track_persist = bool(getattr(self, "ultralytics_track_persist", True))
        self.ultralytics_tracker = str(getattr(self, "ultralytics_tracker", "botsort.yaml") or "botsort.yaml").strip() or "botsort.yaml"

        self.display_sync_to_inference = bool(getattr(self, "display_sync_to_inference", True))
        try:
            self.display_result_max_age_ms = max(
                1.0, float(getattr(self, "display_result_max_age_ms", 250.0))
            )
        except (TypeError, ValueError):
            self.display_result_max_age_ms = 250.0
        self.hand_temporal_enabled = bool(getattr(self, "hand_temporal_enabled", True))
        try:
            self.hand_track_max_miss_frames = max(1, int(getattr(self, "hand_track_max_miss_frames", 8)))
        except (TypeError, ValueError):
            self.hand_track_max_miss_frames = 8
        try:
            self.hand_kpt_smoothing_alpha = min(1.0, max(0.0, float(getattr(self, "hand_kpt_smoothing_alpha", 0.45))))
        except (TypeError, ValueError):
            self.hand_kpt_smoothing_alpha = 0.45
        try:
            self.hand_point_conf_threshold = min(1.0, max(0.0, float(getattr(self, "hand_point_conf_threshold", 0.35))))
        except (TypeError, ValueError):
            self.hand_point_conf_threshold = 0.35
        try:
            self.hand_point_hold_frames = max(1, int(getattr(self, "hand_point_hold_frames", 4)))
        except (TypeError, ValueError):
            self.hand_point_hold_frames = 4
        try:
            self.action_enter_frames = max(1, int(getattr(self, "action_enter_frames", 4)))
        except (TypeError, ValueError):
            self.action_enter_frames = 4
        try:
            self.action_exit_frames = max(1, int(getattr(self, "action_exit_frames", 6)))
        except (TypeError, ValueError):
            self.action_exit_frames = 6

        self.joint_tracking_enabled = bool(getattr(self, "joint_tracking_enabled", True))
        self.tool_class_name = str(getattr(self, "tool_class_name", "扭力枪") or "扭力枪").strip() or "扭力枪"

        def _clamp_int(field_name: str, default: int, minimum: int = 1, maximum: int = 999999) -> None:
            try:
                value = int(getattr(self, field_name, default))
            except (TypeError, ValueError):
                value = default
            setattr(self, field_name, max(minimum, min(maximum, value)))

        _clamp_int("track_seed_stable_frames", 3)
        _clamp_int("track_enter_stable_frames", 2)
        _clamp_int("track_leave_stable_frames", 4)
        _clamp_int("track_gun_lost_frames", 6)
        _clamp_int("track_hole_occlusion_grace_frames", 12)
        _clamp_float("track_enter_threshold", 0.18)
        _clamp_float("track_leave_threshold", 0.08)
        _clamp_float("track_min_score_gap", 0.05)
        self.track_order_constraint_enabled = bool(getattr(self, "track_order_constraint_enabled", True))
        self.track_order_rule = (
            str(getattr(self, "track_order_rule", "configured_order") or "configured_order").strip()
            or "configured_order"
        )
        _clamp_int("track_out_of_order_frames", 2)
        _clamp_int("track_step_leave_frames", 4)
        self.track_debug_csv_enabled = bool(getattr(self, "track_debug_csv_enabled", False))

        self.detection_log_enabled = bool(getattr(self, "detection_log_enabled", True))

        raw_log_dir = str(getattr(self, "detection_log_dir", "logs/detections")).strip()
        self.detection_log_dir = raw_log_dir or "logs/detections"

        try:
            self.detection_log_every_n_frames = max(
                1, int(getattr(self, "detection_log_every_n_frames", 1))
            )
        except (TypeError, ValueError):
            self.detection_log_every_n_frames = 1

        try:
            self.detection_log_flush_interval = max(
                1, int(getattr(self, "detection_log_flush_interval", 30))
            )
        except (TypeError, ValueError):
            self.detection_log_flush_interval = 30

        raw_mapping = getattr(self, "class_mapping", {})
        normalized_mapping: Dict[int, str] = {}
        if isinstance(raw_mapping, dict):
            for raw_key, raw_value in raw_mapping.items():
                try:
                    key = int(raw_key)
                except (TypeError, ValueError):
                    continue
                value = str(raw_value).strip()
                if value:
                    normalized_mapping[key] = value
        for key, value in DEFAULT_CLASS_MAPPING.items():
            normalized_mapping.setdefault(key, value)
        self.class_mapping = normalized_mapping

        return len(errors) == 0
    
    def to_dict(self) -> dict:
        """转换为字典（排除私有字段）"""
        return {
            "max_num_hands": self.max_num_hands,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "camera_index": self.camera_index,
            "camera_backend": int(self.camera_backend),
            "camera_max_index": int(self.camera_max_index),
            "mvsdk_friendly_name": str(self.mvsdk_friendly_name),
            "pinch_open_threshold": self.pinch_open_threshold,
            "finger_extended_angle": self.finger_extended_angle,
            "finger_curled_angle": self.finger_curled_angle,
            "static_n_on": self.static_n_on,
            "static_n_off": self.static_n_off,
            "gesture_n_on": self.gesture_n_on,
            "gesture_hold_seconds": self.gesture_hold_seconds,
            "debug_gesture_overlay": self.debug_gesture_overlay,
            "show_confidence_overlay": bool(self.show_confidence_overlay),
            "round_cooldown_seconds": float(self.round_cooldown_seconds),
            "today_target_capacity": int(self.today_target_capacity),
            "category_names": list(self.category_names),
            # 模型配置
            "model_path": self.model_path,
            "model_task": self.model_task,
            "mediapipe_task_path": self.mediapipe_task_path,
            "enable_gesture_detection": bool(self.enable_gesture_detection),
            "enable_object_detection": bool(self.enable_object_detection),
            "object_model_path": str(self.object_model_path),
            "object_score_threshold": float(self.object_score_threshold),
            "object_max_results": int(self.object_max_results),
            "object_result_hold_ms": int(self.object_result_hold_ms),
            "yolo_model_path": str(self.yolo_model_path),
            "yolo_conf_threshold": float(self.yolo_conf_threshold),
            "yolo_iou_threshold": float(self.yolo_iou_threshold),
            "mediapipe_num_hands": int(self.mediapipe_num_hands),
            "mediapipe_score_threshold": float(self.mediapipe_score_threshold),
            "mediapipe_min_hand_detection_confidence": float(self.mediapipe_min_hand_detection_confidence),
            "mediapipe_min_hand_presence_confidence": float(self.mediapipe_min_hand_presence_confidence),
            "mediapipe_min_tracking_confidence": float(self.mediapipe_min_tracking_confidence),
            "gesture_window_size": int(self.gesture_window_size),
            "gesture_enter_frames": int(self.gesture_enter_frames),
            "gesture_exit_frames": int(self.gesture_exit_frames),
            "ultralytics_device": str(self.ultralytics_device),
            "ultralytics_max_det": int(self.ultralytics_max_det),
            "ultralytics_track_persist": bool(self.ultralytics_track_persist),
            "ultralytics_tracker": str(self.ultralytics_tracker),
            "joint_tracking_enabled": bool(self.joint_tracking_enabled),
            "tool_class_name": str(self.tool_class_name),
            "track_seed_stable_frames": int(self.track_seed_stable_frames),
            "track_enter_stable_frames": int(self.track_enter_stable_frames),
            "track_leave_stable_frames": int(self.track_leave_stable_frames),
            "track_gun_lost_frames": int(self.track_gun_lost_frames),
            "track_hole_occlusion_grace_frames": int(self.track_hole_occlusion_grace_frames),
            "track_enter_threshold": float(self.track_enter_threshold),
            "track_leave_threshold": float(self.track_leave_threshold),
            "track_min_score_gap": float(self.track_min_score_gap),
            "track_order_constraint_enabled": bool(self.track_order_constraint_enabled),
            "track_order_rule": str(self.track_order_rule),
            "track_out_of_order_frames": int(self.track_out_of_order_frames),
            "track_step_leave_frames": int(self.track_step_leave_frames),
            "track_debug_csv_enabled": bool(self.track_debug_csv_enabled),
            "display_sync_to_inference": bool(self.display_sync_to_inference),
            "display_result_max_age_ms": float(self.display_result_max_age_ms),
            "detection_log_enabled": bool(self.detection_log_enabled),
            "detection_log_dir": str(self.detection_log_dir),
            "detection_log_every_n_frames": int(self.detection_log_every_n_frames),
            "detection_log_flush_interval": int(self.detection_log_flush_interval),
        }
    
    def update(self, **kwargs) -> None:
        """更新配置项"""
        for key, value in kwargs.items():
            if hasattr(self, key) and not key.startswith('_'):
                setattr(self, key, value)
        self.validate()
    
    def load(self, path: Optional[Path] = None) -> None:
        """从文件加载配置"""
        config_path = Path(path) if path is not None else (self._config_path or Path("config.json"))
        self._config_path = config_path.resolve()
        
        if not self._config_path.exists():
            return
        
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if hasattr(self, key) and not key.startswith('_'):
                    setattr(self, key, value)

            name = getattr(self, "mvsdk_friendly_name", "")
            if name is None:
                self.mvsdk_friendly_name = ""
            elif not isinstance(name, str):
                self.mvsdk_friendly_name = str(name)
            
            self.validate()
        except (json.JSONDecodeError, IOError) as e:
            print(f"加载配置文件失败，使用默认配置: {e}")
    
    def save(self, path: Optional[Path] = None) -> None:
        """保存配置到文件"""
        config_path = Path(path) if path is not None else (self._config_path or Path("config.json"))
        self._config_path = config_path.resolve()
        
        try:
            self.validate()
            merged = {}
            if self._config_path.exists():
                try:
                    with open(self._config_path, 'r', encoding='utf-8') as rf:
                        existing = json.load(rf)
                    if isinstance(existing, dict):
                        merged.update(existing)
                except (json.JSONDecodeError, IOError):
                    pass

            for legacy_key in LEGACY_MODEL_CONFIG_KEYS:
                merged.pop(legacy_key, None)
            merged.update(self.to_dict())
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"保存配置文件失败: {e}")
    
    @classmethod
    def from_file(cls, path: Path) -> "ConfigManager":
        """从文件创建配置管理器"""
        config = cls()
        config.load(path)
        return config

    def get_model_path(self) -> str:
        """返回当前实际使用的模型路径。"""
        task_path = str(getattr(self, "mediapipe_task_path", "") or "").strip()
        if task_path and task_path.lower().endswith(".task"):
            return task_path
        model_path = str(getattr(self, "model_path", "") or "").strip()
        if model_path and model_path.lower().endswith(".task"):
            return model_path
        return DEFAULT_MEDIAPIPE_TASK_PATH

    def get_model_task(self) -> str:
        return MEDIAPIPE_MODEL_TASK

    def has_enabled_detection_mode(self) -> bool:
        return bool(getattr(self, "enable_gesture_detection", True)) or bool(
            getattr(self, "enable_object_detection", False)
        )

    def get_enabled_detection_modes(self) -> List[str]:
        modes: List[str] = []
        if bool(getattr(self, "enable_object_detection", False)):
            modes.append("object")
        if bool(getattr(self, "enable_gesture_detection", True)):
            modes.append("gesture")
        return modes

    def get_detection_mode_task_type(self) -> str:
        modes = self.get_enabled_detection_modes()
        if modes == ["object", "gesture"]:
            return MEDIAPIPE_COMPOSITE_TASK
        if modes == ["object"]:
            return MEDIAPIPE_OBJECT_TASK
        if modes == ["gesture"]:
            return MEDIAPIPE_MODEL_TASK
        return "disabled"

    def get_detection_mode_display_name(self) -> str:
        modes = self.get_enabled_detection_modes()
        if modes == ["object", "gesture"]:
            return "目标检测 + 手势检测"
        if modes == ["object"]:
            return "目标检测"
        if modes == ["gesture"]:
            return "手势检测"
        return "未启用检测"

    def is_supported_model_task(self) -> bool:
        return self.get_model_task() in SUPPORTED_MODEL_TASKS

    def get_model_task_display_name(self) -> str:
        task = self.get_detection_mode_task_type()
        return MODEL_TASK_DISPLAY_NAMES.get(task, task.upper() or "UNKNOWN")

    def get_config_path(self) -> Optional[Path]:
        if self._config_path is None:
            return None
        return Path(self._config_path)

    def is_supported_model_task(self, task: Optional[str] = None) -> bool:
        current_task = str(task or self.get_model_task()).strip().lower()
        return current_task in SUPPORTED_MODEL_TASKS

    def get_model_task_display_name(self, task: Optional[str] = None) -> str:
        current_task = str(task or self.get_detection_mode_task_type()).strip().lower()
        return MODEL_TASK_DISPLAY_NAMES.get(current_task, current_task.upper() or "UNKNOWN")
