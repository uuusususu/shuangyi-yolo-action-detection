"""标定与洞口数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class HoleZone(str, Enum):
    OUTSIDE = "outside"
    NEAR = "near"
    INSIDE = "inside"


@dataclass
class CalibrationPoint:
    px: float
    py: float
    mm_x: float
    mm_y: float


@dataclass
class CalibrationTransform:
    points: List[CalibrationPoint] = field(default_factory=list)
    _H: object = field(default=None, repr=False)
    _H_inv: object = field(default=None, repr=False)
    _valid: bool = field(default=False, repr=False)

    @property
    def is_valid(self) -> bool:
        return self._valid and self._H is not None

    def compute(self) -> bool:
        import cv2, numpy as np
        if len(self.points) < 4:
            self._valid = False
            return False
        src = np.array([[p.px, p.py] for p in self.points], dtype=np.float64)
        dst = np.array([[p.mm_x, p.mm_y] for p in self.points], dtype=np.float64)
        try:
            H, _ = cv2.findHomography(src, dst)
            if H is None:
                self._valid = False
                return False
            self._H = H
            H_inv, _ = cv2.findHomography(dst, src)
            self._H_inv = H_inv
            self._valid = True
            return True
        except Exception:
            self._valid = False
            return False

    def pixel_to_mm(self, px: float, py: float) -> Tuple[float, float]:
        import cv2, numpy as np
        if not self.is_valid:
            return (0.0, 0.0)
        pt = np.array([[[px, py]]], dtype=np.float64)
        mm = cv2.perspectiveTransform(pt, self._H)
        return (float(mm[0][0][0]), float(mm[0][0][1]))

    def mm_to_pixel(self, mm_x: float, mm_y: float) -> Tuple[float, float]:
        import cv2, numpy as np
        if not self.is_valid or self._H_inv is None:
            return (0.0, 0.0)
        pt = np.array([[[mm_x, mm_y]]], dtype=np.float64)
        px = cv2.perspectiveTransform(pt, self._H_inv)
        return (float(px[0][0][0]), float(px[0][0][1]))

    def to_dict(self) -> dict:
        return {"points": [{"px": p.px, "py": p.py, "mm_x": p.mm_x, "mm_y": p.mm_y} for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationTransform:
        points = [CalibrationPoint(**p) for p in data.get("points", [])]
        t = cls(points=points)
        t.compute()
        return t


@dataclass
class HoleDefinition:
    step_index: int
    name: str = ""
    center_px: Tuple[float, float] = (0.0, 0.0)
    center_mm: Tuple[float, float] = (0.0, 0.0)
    inner_radius_mm: float = 10.0
    outer_radius_mm: float = 20.0
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index, "name": self.name,
            "center_px": list(self.center_px), "center_mm": list(self.center_mm),
            "inner_radius_mm": self.inner_radius_mm, "outer_radius_mm": self.outer_radius_mm,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HoleDefinition:
        return cls(
            step_index=data.get("step_index", 0), name=data.get("name", ""),
            center_px=tuple(data.get("center_px", (0, 0))),
            center_mm=tuple(data.get("center_mm", (0, 0))),
            inner_radius_mm=data.get("inner_radius_mm", 10.0),
            outer_radius_mm=data.get("outer_radius_mm", 20.0),
            enabled=data.get("enabled", True),
        )


@dataclass
class TipPoint:
    px: float
    py: float
    mm_x: float = 0.0
    mm_y: float = 0.0
    conf: float = 0.0
    track_id: Optional[int] = None
    frame_id: int = 0
    label: str = ""

    def to_dict(self) -> dict:
        return {"px": self.px, "py": self.py, "mm_x": self.mm_x, "mm_y": self.mm_y,
                "conf": self.conf, "track_id": self.track_id, "frame_id": self.frame_id, "label": self.label}


@dataclass
class HoleJudgement:
    zone: HoleZone = HoleZone.OUTSIDE
    distance_mm: float = 0.0
    hole: Optional[HoleDefinition] = None
    reason: str = ""


@dataclass
class PointJudgementState:
    calibrated: bool = False
    tip: Optional[TipPoint] = None
    expected_hole: Optional[HoleDefinition] = None
    nearest_hole: Optional[HoleDefinition] = None
    judgement: Optional[HoleJudgement] = None
    diagnostics: List[str] = field(default_factory=list)
