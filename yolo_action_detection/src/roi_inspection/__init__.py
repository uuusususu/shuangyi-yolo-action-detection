"""通用 ROI 定位检测。"""

from roi_inspection.engine import RoiInspectionEngine
from roi_inspection.geometry import NormalizedRect
from roi_inspection.models import (
    InspectionOutcome,
    RoiInspectionConfig,
    RoiInspectionResult,
    RoiInspectionSnapshot,
    RoundPhase,
)

__all__ = [
    "InspectionOutcome",
    "NormalizedRect",
    "RoiInspectionConfig",
    "RoiInspectionEngine",
    "RoiInspectionResult",
    "RoiInspectionSnapshot",
    "RoundPhase",
]
