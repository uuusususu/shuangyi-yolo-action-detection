"""YOLO OBB 配置管理器。"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ULTRALYTICS_OBB_TASK = "ultralytics_obb"
SUPPORTED_MODEL_TASKS = (ULTRALYTICS_OBB_TASK,)
DEFAULT_YOLO_MODEL_PATH = "config/best.pt"
SUPPORTED_YOLO_MODEL_SUFFIXES = (".pt", ".onnx")


@dataclass
class ConfigManager:
    """配置管理器，只支持 YOLO OBB。"""

    model_task: str = ULTRALYTICS_OBB_TASK
    yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH
    yolo_conf_threshold: float = 0.7
    yolo_iou_threshold: float = 0.5
    ultralytics_device: str = ""
    ultralytics_half: bool = False
    ultralytics_tracker: str = "bytetrack.yaml"
    ultralytics_max_det: int = 300
    ultralytics_track_persist: bool = True

    category_names: List[str] = field(default_factory=lambda: ["1号", "2号", "3号", "4号", "5号", ""])
    category_counts: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    # 旧版扭力枪/交集判定字段仅用于兼容历史配置和坐标诊断日志，不参与生产步骤判定。
    tool_class_name: str = "扭力枪"

    camera_index: int = 0
    mvsdk_friendly_name: str = ""
    mvsdk_camera_sn: str = ""

    # 相机参数模式: preserve / load_group / load_file / manual
    camera_parameter_mode: str = "preserve"
    camera_parameter_group: int = 0
    camera_parameter_file: str = ""
    camera_manual_ae: bool = False
    camera_manual_exposure_us: int = 30000
    camera_manual_gain: int = 0
    camera_manual_gamma: int = 100
    camera_manual_contrast: int = 0
    camera_manual_brightness: int = 0

    # 标定配置
    calibration_points: List[Dict] = field(default_factory=list)

    # 洞口配置
    holes: List[Dict] = field(default_factory=list)

    # 点位判定
    tip_enter_stable_frames: int = 2
    tip_wrong_stable_frames: int = 2
    point_judgement_enabled: bool = True

    # 动作判定：模型输出类别名直接匹配步骤类别序列。
    action_overlap_threshold: float = 0.20
    action_pass_stable_frames: int = 1
    action_ng_stable_frames: int = 10
    action_leave_stable_frames: int = 4
    action_order_constraint_enabled: bool = True

    # 旧版融合判定字段保留加载/保存兼容，生产路径不再使用。
    fusion_iou_threshold: float = 0.10
    fusion_center_dist_threshold: int = 60
    fusion_corner_inside_enabled: bool = True

    # 坐标诊断日志
    coordinate_logging_dir: str = "logs/coordinate_sessions"
    coordinate_logging_max_queue: int = 1000
    coordinate_logging_summary_enabled: bool = True
    coordinate_logging_max_file_mb: int = 0

    joint_tracking_enabled: bool = True
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

    round_cooldown_seconds: float = 2.0
    today_target_capacity: int = 300
    show_confidence_overlay: bool = True
    show_track_id_overlay: bool = True
    display_result_max_age_ms: int = 250

    # 结果反馈
    pass_sound_enabled: bool = False
    fail_sound_enabled: bool = True
    # 旧统一开关仅用于配置兼容，运行时以两个独立开关为准。
    sound_feedback_enabled: bool = True
    fail_evidence_enabled: bool = True

    # 首类别区域检查模式：category_names[0] 为父区域，后续类别为子控件。
    first_category_region_check_enabled: bool = False

    # PCB 多板元器件检查模式（保留历史配置兼容）。
    pcb_inspection_enabled: bool = False
    pcb_class_name: str = "pcb"
    pcb_component_class_names: List[str] = field(default_factory=lambda: ["", "", "", ""])
    pcb_fail_stable_frames: int = 10
    pcb_round_interval_seconds: float = 0.0
    pcb_assignment_margin_ratio: float = 0.15

    _config_path: Optional[str] = field(default=None, repr=False)

    def validate(self) -> None:
        """验证配置。"""
        if self.model_task != ULTRALYTICS_OBB_TASK:
            raise ValueError(
                f"不支持的 model_task: {self.model_task}，"
                f"只支持 {ULTRALYTICS_OBB_TASK}"
            )
        suffix = Path(str(self.yolo_model_path)).suffix.lower()
        if suffix and suffix not in SUPPORTED_YOLO_MODEL_SUFFIXES:
            raise ValueError("模型文件只支持 .pt 或 .onnx")
        self.yolo_conf_threshold = max(0.0, min(1.0, self.yolo_conf_threshold))
        self.yolo_iou_threshold = max(0.0, min(1.0, self.yolo_iou_threshold))
        self.action_overlap_threshold = max(0.0, min(1.0, self.action_overlap_threshold))
        self.action_pass_stable_frames = max(1, int(self.action_pass_stable_frames))
        self.action_ng_stable_frames = max(1, int(self.action_ng_stable_frames))
        self.fusion_iou_threshold = max(0.01, min(1.0, self.fusion_iou_threshold))
        self.fusion_center_dist_threshold = max(10, min(500, int(self.fusion_center_dist_threshold)))
        self.display_result_max_age_ms = max(0, int(self.display_result_max_age_ms))
        self.round_cooldown_seconds = max(0.0, float(self.round_cooldown_seconds))
        self.pass_sound_enabled = _coerce_bool(self.pass_sound_enabled)
        self.fail_sound_enabled = _coerce_bool(self.fail_sound_enabled)
        self.sound_feedback_enabled = self.fail_sound_enabled
        self.fail_evidence_enabled = _coerce_bool(self.fail_evidence_enabled)
        self.first_category_region_check_enabled = _coerce_bool(self.first_category_region_check_enabled)
        self.pcb_inspection_enabled = _coerce_bool(self.pcb_inspection_enabled)
        self.pcb_fail_stable_frames = max(1, int(self.pcb_fail_stable_frames))
        self.pcb_round_interval_seconds = max(0.0, float(self.pcb_round_interval_seconds))
        self.pcb_assignment_margin_ratio = max(0.0, min(1.0, float(self.pcb_assignment_margin_ratio)))
        if self.first_category_region_check_enabled:
            self._validate_first_category_region_config()
        self._normalize_category_counts()

    def _normalize_category_counts(self) -> None:
        """确保 category_counts 与 category_names 下标对齐，且每个值 >= 1。"""
        names_len = len(self.category_names)
        counts = list(self.category_counts)
        # 补齐到与 names 相同长度
        while len(counts) < names_len:
            counts.append(1)
        # 裁剪到相同长度
        counts = counts[:names_len]
        # 每个值至少为 1
        counts = [max(1, int(c)) for c in counts]
        self.category_counts = counts

    def load(self, path: Path | str) -> None:
        """从 JSON 文件加载配置。"""
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 检查 model_task
        mt = data.get("model_task", "")
        if mt and mt != ULTRALYTICS_OBB_TASK:
            raise ValueError(
                f"配置文件的 model_task 为 '{mt}'，"
                f"不支持该类型，只支持 '{ULTRALYTICS_OBB_TASK}'"
            )
        has_new_sound_setting = (
            "pass_sound_enabled" in data or "fail_sound_enabled" in data
        )
        if has_new_sound_setting:
            data.setdefault("pass_sound_enabled", False)
            data.setdefault("fail_sound_enabled", True)
        else:
            data["pass_sound_enabled"] = False
            data["fail_sound_enabled"] = data.get("sound_feedback_enabled", True)
        for key, value in data.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)
        self.validate()
        self._config_path = str(path)

    def save(self, path: Path | str) -> None:
        """保存配置到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pass_sound_enabled = _coerce_bool(self.pass_sound_enabled)
        self.fail_sound_enabled = _coerce_bool(self.fail_sound_enabled)
        self.sound_feedback_enabled = self.fail_sound_enabled
        data = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            data[k] = v
        data["model_task"] = ULTRALYTICS_OBB_TASK
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_config_path(self) -> str:
        return self._config_path or ""

    def get_model_path(self) -> str:
        return self.yolo_model_path

    def get_step_class_names(self) -> List[str]:
        """返回有效步骤类别（非空）。"""
        return [c for c in self.category_names if c and c.strip()]

    def get_first_category_region_classes(self) -> tuple[str, List[str]]:
        """返回首类别区域检查的父类别和子类别。"""
        names = [c.strip() for c in self.category_names if c and c.strip()]
        if not names:
            return "", []
        return names[0], names[1:]

    def _validate_first_category_region_config(self) -> None:
        """校验首类别区域检查模式配置。"""
        parent, children = self.get_first_category_region_classes()
        if not parent:
            raise ValueError("首类别区域检查已启用，但类别1为空")
        if not children:
            raise ValueError("首类别区域检查至少需要 1 个子类别")
        if parent in children:
            raise ValueError(f"父区域类别 '{parent}' 不能与子类别相同")
        if len(set(children)) != len(children):
            raise ValueError("首类别区域检查的子类别存在重复")

    def _validate_pcb_config(self) -> None:
        """校验 PCB 检查模式配置。"""
        if not self.pcb_class_name or not self.pcb_class_name.strip():
            raise ValueError("PCB 检查模式已启用，但 PCB 类别名称为空")
        pcb_class = self.pcb_class_name.strip()
        comps = [c.strip() for c in self.pcb_component_class_names if c and c.strip()]
        if len(comps) != 4:
            raise ValueError(
                f"PCB 检查模式需要恰好 4 个非空元器件类别，当前有 {len(comps)} 个"
            )
        if len(set(comps)) != 4:
            raise ValueError("PCB 元器件类别存在重复，必须 4 个互不相同")
        if pcb_class in comps:
            raise ValueError(f"PCB 类别 '{pcb_class}' 不能与元器件类别相同")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)
