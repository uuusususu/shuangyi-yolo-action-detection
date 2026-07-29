"""ROI 定位轮次的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from roi_inspection.geometry import NormalizedRect


class RoundPhase(str, Enum):
    WAITING_FOR_LOCATOR = "waiting_for_locator"
    INSPECTING = "inspecting"
    RESULT_HOLD = "result_hold"


class InspectionOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class CategoryInspectionState:
    class_name: str
    required_count: int
    observed_count: int = 0
    match_streak: int = 0
    mismatch_streak: int = 0
    passed: bool = False


@dataclass(frozen=True)
class RoiInspectionConfig:
    inspection_roi: NormalizedRect
    required_counts: Dict[str, int]
    template_enabled: bool = False
    pass_stable_frames: int = 1
    fail_stable_frames: int = 10
    round_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.required_counts:
            raise ValueError("至少需要一个待检类别")
        if any(not name.strip() or int(count) <= 0 for name, count in self.required_counts.items()):
            raise ValueError("待检类别名称不能为空且数量必须为正整数")
        object.__setattr__(
            self,
            "required_counts",
            {str(name).strip(): int(count) for name, count in self.required_counts.items()},
        )
        object.__setattr__(self, "template_enabled", bool(self.template_enabled))
        object.__setattr__(self, "pass_stable_frames", max(1, int(self.pass_stable_frames)))
        object.__setattr__(self, "fail_stable_frames", max(1, int(self.fail_stable_frames)))
        object.__setattr__(self, "round_interval_seconds", max(0.0, float(self.round_interval_seconds)))

    @property
    def has_template_locator(self) -> bool:
        """是否启用模板定位；False 即识别 ROI 内任意类别触发。"""
        return bool(self.template_enabled)

    @classmethod
    def from_config(cls, config) -> "RoiInspectionConfig":
        required_counts: Dict[str, int] = {}
        names = list(getattr(config, "category_names", []) or [])
        counts = list(getattr(config, "category_counts", []) or [])
        for index, raw_name in enumerate(names):
            name = str(raw_name).strip()
            if not name:
                continue
            required_counts[name] = int(counts[index] if index < len(counts) else 1)
        return cls(
            inspection_roi=NormalizedRect.from_mapping(config.roi_inspection_rect),
            required_counts=required_counts,
            template_enabled=bool(getattr(config, "roi_template_enabled", False)),
            pass_stable_frames=getattr(config, "action_pass_stable_frames", 1),
            fail_stable_frames=getattr(config, "action_ng_stable_frames", 10),
            round_interval_seconds=getattr(config, "round_cooldown_seconds", 0.0),
        )


@dataclass(frozen=True)
class RoiInspectionResult:
    round_id: int
    outcome: InspectionOutcome
    actual_counts: Dict[str, int]
    required_counts: Dict[str, int]
    failed_categories: List[str]
    source_frame_id: int
    timestamp: float
    feedback_key: str


@dataclass
class RoiInspectionSnapshot:
    phase: RoundPhase
    round_id: int
    template_matched: bool = False
    template_score: float = 0.0
    category_states: Dict[str, CategoryInspectionState] = field(default_factory=dict)
    result: Optional[RoiInspectionResult] = None
