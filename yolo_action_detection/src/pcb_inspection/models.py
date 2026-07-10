"""PCB 检查数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PcbStatus(str, Enum):
    OBSERVING = "observing"
    DECIDED = "decided"
    EMITTED = "emitted"
    RETIRED = "retired"


class PcbResult(str, Enum):
    NONE = "none"
    PASS = "pass"
    FAIL = "fail"


@dataclass
class SlotObservation:
    """单个逻辑槽位在本帧的观测结果。"""
    class_name: str
    present: bool = False
    unreliable: bool = False


@dataclass
class PcbState:
    """单块 PCB 的检查状态。"""
    track_id: int
    first_seen_frame: int = 0
    status: PcbStatus = PcbStatus.OBSERVING
    result: PcbResult = PcbResult.NONE
    consecutive_fail: int = 0
    last_slot_states: Dict[str, SlotObservation] = field(default_factory=dict)
    result_frame: int = 0
    result_timestamp: float = 0.0
    missing_classes: List[str] = field(default_factory=list)
    last_seen_frame: int = 0

    @property
    def is_decided(self) -> bool:
        return self.status in (PcbStatus.DECIDED, PcbStatus.EMITTED, PcbStatus.RETIRED)

    @property
    def can_judge(self) -> bool:
        return self.status == PcbStatus.OBSERVING


@dataclass
class PcbInspectionResult:
    """一次 PCB 检查的最终结果。"""
    track_id: int
    result: PcbResult
    slot_states: Dict[str, bool] = field(default_factory=dict)
    missing_classes: List[str] = field(default_factory=list)
    source_frame_id: int = 0
    timestamp: float = 0.0


@dataclass
class PcbInspectionConfig:
    """PCB 检查配置快照。"""
    pcb_class_name: str = "pcb"
    component_class_names: List[str] = field(default_factory=lambda: ["", "", "", ""])
    fail_stable_frames: int = 3
    round_interval_seconds: float = 0.0
    assignment_margin_ratio: float = 0.15
    # 画面安全边距（像素），PCB 完整可见才算有效检查帧
    frame_margin_px: int = 10

    @classmethod
    def from_config(cls, config) -> "PcbInspectionConfig":
        return cls(
            pcb_class_name=getattr(config, "pcb_class_name", "pcb"),
            component_class_names=list(getattr(config, "pcb_component_class_names", ["", "", "", ""])),
            fail_stable_frames=getattr(config, "pcb_fail_stable_frames", 3),
            round_interval_seconds=getattr(config, "pcb_round_interval_seconds", 0.0),
            assignment_margin_ratio=getattr(config, "pcb_assignment_margin_ratio", 0.15),
        )