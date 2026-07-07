from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class _TrackState:
    track_id: int
    points: np.ndarray
    conf: np.ndarray
    last_action: str = ""
    stable_action: str = ""
    candidate_action: str = ""
    candidate_count: int = 0
    exit_count: int = 0
    miss_frames: int = 0


class HandTemporalTracker:
    """Minimal temporal layer for a single hand-pose stream."""

    def __init__(
        self,
        *,
        smoothing_alpha: float = 0.45,
        point_conf_threshold: float = 0.35,
        point_hold_frames: int = 4,
        track_max_miss_frames: int = 8,
        action_enter_frames: int = 4,
        action_exit_frames: int = 6,
    ) -> None:
        self.smoothing_alpha = float(smoothing_alpha)
        self.point_conf_threshold = float(point_conf_threshold)
        self.point_hold_frames = int(point_hold_frames)
        self.track_max_miss_frames = int(track_max_miss_frames)
        self.action_enter_frames = int(action_enter_frames)
        self.action_exit_frames = int(action_exit_frames)
        self._tracks: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def _stabilize_points(
        self,
        *,
        state: Optional[_TrackState],
        points: np.ndarray,
        conf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if state is None:
            return points.copy(), conf.copy()

        smoothed = points.copy()
        smoothed_conf = conf.copy()
        for idx in range(len(points)):
            prev_point = state.points[idx]
            prev_conf = float(state.conf[idx]) if idx < len(state.conf) else 0.0
            cur_conf = float(conf[idx]) if idx < len(conf) else 0.0
            if cur_conf >= self.point_conf_threshold:
                smoothed[idx] = (
                    self.smoothing_alpha * points[idx]
                    + (1.0 - self.smoothing_alpha) * prev_point
                )
            elif state.miss_frames < self.point_hold_frames and prev_conf > 0.0:
                smoothed[idx] = prev_point
                smoothed_conf[idx] = prev_conf
        return smoothed, smoothed_conf

    def _stabilize_action(self, state: _TrackState, action: str) -> None:
        current_action = str(action or "").strip()
        state.last_action = current_action

        if not current_action:
            if state.stable_action:
                state.exit_count += 1
                if state.exit_count >= self.action_exit_frames:
                    state.stable_action = ""
                    state.candidate_action = ""
                    state.candidate_count = 0
                    state.exit_count = 0
            return

        if current_action == state.stable_action:
            state.exit_count = 0
            state.candidate_action = current_action
            state.candidate_count = self.action_enter_frames
            return

        if current_action == state.candidate_action:
            state.candidate_count += 1
        else:
            state.candidate_action = current_action
            state.candidate_count = 1

        if state.candidate_count >= self.action_enter_frames:
            state.stable_action = current_action
            state.exit_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids: set[int] = set()
        stabilized: List[Dict[str, Any]] = []

        for det in detections:
            keypoints = det.get("keypoints")
            if not isinstance(keypoints, dict):
                stabilized.append(det)
                continue

            points = np.asarray(keypoints.get("points", []), dtype=np.float32)
            conf = np.asarray(keypoints.get("conf", []), dtype=np.float32)
            if points.size == 0 or conf.size == 0:
                stabilized.append(det)
                continue

            track_id = int(det.get("track_id", -1))
            if track_id < 0:
                track_id = 0
            seen_ids.add(track_id)

            state = self._tracks.get(track_id)
            smoothed_points, smoothed_conf = self._stabilize_points(
                state=state,
                points=points,
                conf=conf,
            )

            if state is None:
                state = _TrackState(
                    track_id=track_id,
                    points=smoothed_points.copy(),
                    conf=smoothed_conf.copy(),
                )
                self._tracks[track_id] = state
            else:
                state.points = smoothed_points.copy()
                state.conf = smoothed_conf.copy()
                state.miss_frames = 0

            self._stabilize_action(state, str(det.get("action", "") or ""))

            stabilized_det = dict(det)
            stabilized_det["track_id"] = track_id
            stabilized_det["keypoints"] = {
                "points": smoothed_points.tolist(),
                "conf": smoothed_conf.tolist(),
                "connections": list(HAND_CONNECTIONS),
            }
            if state.stable_action:
                stabilized_det["stable_action"] = state.stable_action
            stabilized.append(stabilized_det)

        stale_ids = [track_id for track_id in self._tracks.keys() if track_id not in seen_ids]
        for track_id in stale_ids:
            state = self._tracks[track_id]
            state.miss_frames += 1
            if state.miss_frames > self.track_max_miss_frames:
                self._tracks.pop(track_id, None)

        return stabilized
