"""PCB 检查数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PcbStatus(str, Enum):
    WAITING = "waiting"
    COOLDOWN = "cooldown"
    ACTIVE = "active"
    # Backward-compatible alias for older tests/callers; new scheduling uses ACTIVE.
    OBSERVING = "active"
    FAIL_RETRY_WAITING = "fail_retry_waiting"
    PASS_LATCHED = "pass_latched"
    NG_LATCHED = "ng_latched"
    RETIRED = "retired"


class PcbResult(str, Enum):
    NONE = "none"
    PASS = "pass"
    FAIL = "fail"


class SlotStatus(str, Enum):
    WAITING = "waiting"
    MATCHING = "matching"
    COMPLETED = "completed"
    MISMATCHING = "mismatching"
    NG_LATCHED = "ng_latched"


@dataclass
class SlotObservation:
    """单个逻辑槽位在本帧的观测结果。"""
    class_name: str
    present: bool = False
    unreliable: bool = False
    observed_count: int = 0
    required_count: int = 1
    status: SlotStatus = SlotStatus.WAITING


@dataclass
class PcbState:
    """单块 PCB 的检查状态。"""
    track_id: int
    first_seen_frame: int = 0
    status: PcbStatus = PcbStatus.WAITING
    result: PcbResult = PcbResult.NONE
    consecutive_fail: int = 0
    last_slot_states: Dict[str, SlotObservation] = field(default_factory=dict)
    result_frame: int = 0
    result_timestamp: float = 0.0
    missing_classes: List[str] = field(default_factory=list)
    last_seen_frame: int = 0
    completed_classes: set[str] = field(default_factory=set)
    match_streaks: Dict[str, int] = field(default_factory=dict)
    mismatch_streaks: Dict[str, int] = field(default_factory=dict)
    result_emitted: bool = False
    queued_frame: int = 0
    cooldown_until: float = 0.0
    attempt_id: int = 0
    last_fail_signature: Optional[Tuple[Tuple[str, int, int], ...]] = None
    emitted_fail_signatures: set[Tuple[Tuple[str, int, int], ...]] = field(default_factory=set)

    @property
    def is_decided(self) -> bool:
        return self.status in (PcbStatus.PASS_LATCHED, PcbStatus.RETIRED)

    @property
    def can_judge(self) -> bool:
        return self.status == PcbStatus.ACTIVE


@dataclass
class PcbInspectionResult:
    """一次 PCB 检查的最终结果。"""
    track_id: int
    result: PcbResult
    attempt_id: int = 0
    is_new_fail_signature: bool = False
    slot_states: Dict[str, bool] = field(default_factory=dict)
    missing_classes: List[str] = field(default_factory=list)
    observed_counts: Dict[str, int] = field(default_factory=dict)
    required_counts: Dict[str, int] = field(default_factory=dict)
    source_frame_id: int = 0
    timestamp: float = 0.0


@dataclass
class PcbInspectionConfig:
    """PCB 检查配置快照。"""
    pcb_class_name: str = "pcb"
    component_class_names: List[str] = field(default_factory=lambda: ["", "", "", ""])
    component_required_counts: Dict[str, int] = field(default_factory=dict)
    pass_stable_frames: int = 1
    fail_stable_frames: int = 10
    round_interval_seconds: float = 0.0
    assignment_margin_ratio: float = 0.15
    # 画面安全边距（像素），PCB 完整可见才算有效检查帧
    frame_margin_px: int = 10

    def __post_init__(self) -> None:
        self.pass_stable_frames = max(1, int(self.pass_stable_frames))
        self.fail_stable_frames = max(1, int(self.fail_stable_frames))
        self.component_required_counts = {
            name: max(1, int(self.component_required_counts.get(name, 1)))
            for name in self.component_class_names
            if name
        }

    @classmethod
    def from_config(cls, config) -> "PcbInspectionConfig":
        return cls(
            pcb_class_name=getattr(config, "pcb_class_name", "pcb"),
            component_class_names=list(getattr(config, "pcb_component_class_names", ["", "", "", ""])),
            pass_stable_frames=getattr(config, "action_pass_stable_frames", 1),
            fail_stable_frames=getattr(
                config,
                "action_ng_stable_frames",
                getattr(config, "pcb_fail_stable_frames", 10),
            ),
            round_interval_seconds=getattr(config, "pcb_round_interval_seconds", 0.0),
            assignment_margin_ratio=getattr(config, "pcb_assignment_margin_ratio", 0.15),
        )

    @classmethod
    def from_first_category_config(cls, config) -> "PcbInspectionConfig":
        parent, children = config.get_first_category_region_classes()
        category_names = list(getattr(config, "category_names", []) or [])
        category_counts = list(getattr(config, "category_counts", []) or [])
        required_counts = {}
        for index, name in enumerate(category_names):
            normalized = str(name).strip()
            if normalized in children:
                count = category_counts[index] if index < len(category_counts) else 1
                required_counts[normalized] = max(1, int(count))
        return cls(
            pcb_class_name=parent,
            component_class_names=list(children),
            component_required_counts=required_counts,
            pass_stable_frames=getattr(config, "action_pass_stable_frames", 1),
            fail_stable_frames=getattr(
                config,
                "action_ng_stable_frames",
                getattr(config, "pcb_fail_stable_frames", 10),
            ),
            round_interval_seconds=getattr(config, "round_cooldown_seconds", 0.0),
            assignment_margin_ratio=getattr(config, "pcb_assignment_margin_ratio", 0.15),
        )
