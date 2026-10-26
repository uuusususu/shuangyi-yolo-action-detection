"""YOLO OBB 配置管理器。"""
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from roi_inspection.geometry import NormalizedRect

ULTRALYTICS_OBB_TASK = "ultralytics_obb"
SUPPORTED_MODEL_TASKS = (ULTRALYTICS_OBB_TASK,)
DEFAULT_YOLO_MODEL_PATH = "config/best.pt"
SUPPORTED_YOLO_MODEL_SUFFIXES = (".pt", ".onnx")
CURRENT_CONFIG_SCHEMA_VERSION = 4


@dataclass
class ConfigManager:
    """配置管理器，只支持 YOLO OBB。"""

    config_schema_version: int = CURRENT_CONFIG_SCHEMA_VERSION
    model_task: str = ULTRALYTICS_OBB_TASK
    yolo_model_path: str = DEFAULT_YOLO_MODEL_PATH
    yolo_conf_threshold: float = 0.7
    yolo_iou_threshold: float = 0.5
    ultralytics_half: bool = False
    ultralytics_tracker: str = "bytetrack.yaml"
    ultralytics_track_persist: bool = True

    category_names: List[str] = field(default_factory=lambda: ["1号", "2号", "3号", "4号", "5号", ""])
    category_counts: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    # 旧版扭力枪/交集判定字段仅用于兼容历史配置和坐标诊断日志，不参与生产步骤判定。
    tool_class_name: str = "扭力枪"

    camera_index: int = 0
    mvsdk_friendly_name: str = ""
    mvsdk_camera_sn: str = ""

    # 相机手动曝光（us）；参数加载统一走 preserve（不覆盖相机端已有参数）。
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
    lstm_action_enabled: bool = False
    lstm_contract_path: str = ""
    lstm_bundle_path: str = ""
    action_fail_retry_enabled: bool = False
    action_fail_retry_frames: int = 10

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
    # 切到下一个动作后，新动作连续保持这么久，上一个动作才结算。换掉模型里的最短动作时长。
    action_switch_confirm_seconds: float = 0.5
    show_confidence_overlay: bool = True
    show_track_id_overlay: bool = True
    display_result_max_age_ms: int = 250

    # 结果反馈
    pass_sound_enabled: bool = False
    fail_sound_enabled: bool = True
    # 旧统一开关仅用于配置兼容，运行时以两个独立开关为准。
    sound_feedback_enabled: bool = True
    fail_evidence_enabled: bool = True

    # 固定工位 ROI 定位检测。
    roi_inspection_enabled: bool = False
    roi_inspection_rect: Optional[Dict[str, float]] = None
    # 搜索 ROI 内的边缘模板匹配定位。
    roi_template_enabled: bool = False
    roi_template_path: str = "locator_template.png"
    # 形状模型（分层定向边缘点集）落盘文件，与模板同目录。
    roi_template_model_path: str = "locator_model.npz"
    # 形状匹配得分近似"方向一致且可见的轮廓比例"：0.7 即允许 30% 轮廓缺失。
    roi_template_match_threshold: float = 0.7
    roi_template_angle_range: float = 15.0
    # 0 表示按模板尺寸自动推算角度步进（金字塔下不需要手动调）。
    roi_template_angle_step: float = 0.0
    # use = 区分明暗方向（判别力强）；ignore_global = 允许工件与背景对比度反转。
    roi_template_polarity: str = "use"
    # 搜索框相对模板必须留出的最小平移余量，抗工位停位波动。
    roi_template_min_travel_px: int = 40
    roi_template_min_travel_ratio: float = 0.30
    roi_template_rect: Optional[Dict[str, float]] = None
    roi_template_search_rect: Optional[Dict[str, float]] = None

    # 模板匹配现场诊断（默认关闭；现场测试时打开）。
    template_diagnostics_enabled: bool = False
    template_diagnostics_dir: str = "logs/template_sessions"
    template_diagnostics_shadow_enabled: bool = True
    template_diagnostics_sample_frames: int = 120
    template_diagnostics_max_queue: int = 600

    # 当前产能批次。关程序和重启都保留，只有清零才归档后归零。
    production_started_at: float = 0.0
    production_total: int = 0
    production_ok: int = 0
    production_ng: int = 0

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
        self.action_fail_retry_enabled = _coerce_bool(self.action_fail_retry_enabled)
        self.action_fail_retry_frames = max(1, min(100, int(self.action_fail_retry_frames)))
        self.fusion_iou_threshold = max(0.01, min(1.0, self.fusion_iou_threshold))
        self.fusion_center_dist_threshold = max(10, min(500, int(self.fusion_center_dist_threshold)))
        self.display_result_max_age_ms = max(0, int(self.display_result_max_age_ms))
        self.round_cooldown_seconds = max(0.0, float(self.round_cooldown_seconds))
        self.action_switch_confirm_seconds = max(
            0.1, min(5.0, float(self.action_switch_confirm_seconds))
        )
        self.pass_sound_enabled = _coerce_bool(self.pass_sound_enabled)
        self.fail_sound_enabled = _coerce_bool(self.fail_sound_enabled)
        self.sound_feedback_enabled = self.fail_sound_enabled
        self.fail_evidence_enabled = _coerce_bool(self.fail_evidence_enabled)
        self.roi_inspection_enabled = _coerce_bool(self.roi_inspection_enabled)
        self.template_diagnostics_enabled = _coerce_bool(self.template_diagnostics_enabled)
        self.template_diagnostics_shadow_enabled = _coerce_bool(
            self.template_diagnostics_shadow_enabled
        )
        self.template_diagnostics_sample_frames = max(
            0, min(2000, int(self.template_diagnostics_sample_frames))
        )
        self.template_diagnostics_max_queue = max(
            50, min(20000, int(self.template_diagnostics_max_queue))
        )
        self.production_started_at = max(0.0, float(self.production_started_at or 0))
        self.production_ok = max(0, int(self.production_ok or 0))
        self.production_ng = max(0, int(self.production_ng or 0))
        self.production_total = self.production_ok + self.production_ng
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
        self.category_counts = counts

    def load(self, path: Path | str) -> None:
        """从 JSON 文件加载配置。"""
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # v3 只窄迁移边缘模板新增字段，其他现场参数原样保留。
        data["config_schema_version"] = data.get("config_schema_version", 0)
        if data["config_schema_version"] == 3:
            data["config_schema_version"] = CURRENT_CONFIG_SCHEMA_VERSION
            data.setdefault("roi_template_search_rect", None)
            data.setdefault("roi_template_angle_step", 1.0)
            data.pop("roi_template_min_matches", None)
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
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as f:
                temporary_path = Path(f.name)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            if path.exists():
                shutil.copy2(path, Path(str(path) + ".bak"))
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_config_path(self) -> str:
        return self._config_path or ""

    def get_model_path(self) -> str:
        return self.yolo_model_path

    def get_step_class_names(self) -> List[str]:
        """返回有效步骤类别（非空）。"""
        return [c for c in self.category_names if c and c.strip()]

    def get_runtime_category_counts(self) -> List[int]:
        """返回不会阻止配置界面启动的临时运行数量；有效性仍由校验器决定。"""
        counts = list(self.category_counts or [])
        result: List[int] = []
        for index in range(len(self.category_names or [])):
            value = counts[index] if index < len(counts) else 1
            result.append(int(value) if _is_positive_int(value) else 1)
        return result

    def inspection_validation_errors(
        self,
        model_class_names: Optional[Iterable[str]] = None,
        *,
        require_roi_rects: bool = True,
    ) -> List[str]:
        """返回配置错误；保存草稿时可允许尚未创建的 ROI。"""
        errors: List[str] = []
        if (
            type(self.config_schema_version) is not int
            or self.config_schema_version != CURRENT_CONFIG_SCHEMA_VERSION
        ):
            errors.append(
                f"配置版本无效：需要 {CURRENT_CONFIG_SCHEMA_VERSION}，"
                f"当前为 {self.config_schema_version}；请重新配置并保存"
            )

        available = None
        if model_class_names is not None:
            available = {str(name).strip() for name in model_class_names if str(name).strip()}

        names = [str(name).strip() for name in (self.category_names or [])]
        counts = list(self.category_counts or [])
        valid_names = [name for name in names if name]
        if not valid_names:
            errors.append("至少配置一个有效待检类别")
        if len(valid_names) != len(set(valid_names)):
            errors.append("待检类别不能重复")
        for index, name in enumerate(names):
            if not name:
                continue
            count = counts[index] if index < len(counts) else 0
            if not _is_positive_int(count):
                errors.append(f"待检数量无效：类别 '{name}' 必须为正整数")
                continue
            action_mode = (
                not bool(getattr(self, "roi_inspection_enabled", False))
                and bool(getattr(self, "action_order_constraint_enabled", False))
            )
            if action_mode and int(count) != 1:
                errors.append(f"动作步骤「{name}」的数量必须为 1")
                continue
            if action_mode:
                continue
            if available is not None and name not in available:
                errors.append(f"待检类别 '{name}' 不属于当前模型")

        if not getattr(self, "roi_template_enabled", False):
            return errors

        threshold = _as_float(getattr(self, "roi_template_match_threshold", 0.6))
        if threshold is None or not 0.0 < threshold <= 1.0:
            errors.append("匹配分数阈值必须在 0 到 1 之间")
        angle_range = _as_float(getattr(self, "roi_template_angle_range", 15.0))
        if angle_range is None or not 0.0 <= angle_range <= 180.0:
            errors.append("旋转角度范围必须在 0 到 180 度之间")
        # 0 表示按模板尺寸自动推算步进（金字塔下不需要手动指定）。
        angle_step = _as_float(getattr(self, "roi_template_angle_step", 0.0))
        if angle_step is None or not 0.0 <= angle_step <= 180.0:
            errors.append("模板角度步进必须在 0 到 180 度之间（0 表示自动）")
        polarity = str(getattr(self, "roi_template_polarity", "use") or "use")
        if polarity not in ("use", "ignore_global"):
            errors.append("模板极性只支持 use 或 ignore_global")
        return errors

    def require_valid_inspection_config(
        self,
        model_class_names: Optional[Iterable[str]] = None,
        *,
        require_roi_rects: bool = True,
    ) -> None:
        errors = self.inspection_validation_errors(
            model_class_names,
            require_roi_rects=require_roi_rects,
        )
        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def _append_roi_error(
        errors: List[str],
        label: str,
        value: Optional[Dict[str, float]],
    ) -> None:
        if value is None:
            errors.append(f"{label} 未配置")
            return
        try:
            NormalizedRect.from_mapping(value)
        except ValueError as exc:
            errors.append(f"{label} 无效：{exc}")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _is_positive_int(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        text = value.strip()
        return text.isdigit() and int(text) > 0
    return False


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
