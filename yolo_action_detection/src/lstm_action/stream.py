"""连续 LSTM 的在线切段。隐藏状态一直留着，完整动作只在边界确认后结算一次。"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from detection_logging.action_trace import trace
from lstm_workbench.features import build_feature_row

from .bundle import RuntimeBundle


@dataclass
class _Run:
    class_index: int
    start: int
    end: int
    source_start: int
    source_end: int
    time_start: float
    time_end: float
    parts: set[str] = field(default_factory=set)

    def copy(self) -> "_Run":
        return _Run(
            self.class_index,
            self.start,
            self.end,
            self.source_start,
            self.source_end,
            self.time_start,
            self.time_end,
            set(self.parts),
        )


class TimeBaseSampler:
    """按时间格向前选真实帧，最多延迟一个周期；缺格不重复帧。"""

    def __init__(self, time_base_fps: float) -> None:
        if not math.isfinite(float(time_base_fps)) or float(time_base_fps) <= 0:
            raise ValueError("时间基必须是正有限值")
        self.period = 1.0 / float(time_base_fps)
        self.origin: float | None = None
        self.next_slot = 0
        self.last_frame: int | None = None
        self.last_time: float | None = None
        self.hold = False
        self.selected_frame: int | None = None
        self.selected_time: float | None = None

    def push(self, frame_id: int, timestamp: float) -> str | int | None:
        frame_id, timestamp = int(frame_id), float(timestamp)
        if not math.isfinite(timestamp):
            return "interrupt"
        previous_frame, previous_time = self.last_frame, self.last_time
        if self.last_frame is not None and self.last_time is not None:
            if int(frame_id) <= self.last_frame or float(timestamp) + 1e-9 < self.last_time:
                return "interrupt"
        self.last_frame = int(frame_id)
        self.last_time = float(timestamp)
        if self.origin is None:
            self.origin = float(timestamp)
        if timestamp + 1e-9 < self.origin + self.next_slot * self.period:
            return None
        # 推理慢会隔过好几格。只取当前时刻对应的最后一格，用还没用过的真实帧，
        # 不重复补帧，也不把产线判成断流。
        slot = int((timestamp - self.origin) / self.period + 1e-9)
        if slot < self.next_slot:
            slot = self.next_slot
        target = self.origin + slot * self.period
        if abs(timestamp - target) <= 1e-9 or previous_frame is None:
            selected_frame, selected_time = frame_id, timestamp
        elif previous_time <= target + 1e-9:
            selected_frame, selected_time = previous_frame, previous_time
        else:
            selected_frame, selected_time = frame_id, timestamp
        self.next_slot = slot + 1
        self.hold = False
        if selected_frame is None or selected_frame == self.selected_frame:
            if frame_id == self.selected_frame:
                return None
            selected_frame, selected_time = frame_id, timestamp
        self.selected_frame, self.selected_time = selected_frame, selected_time
        return slot

    def reset(self) -> None:
        self.origin = None
        self.next_slot = 0
        self.last_frame = None
        self.last_time = None
        self.hold = False
        self.selected_frame = None
        self.selected_time = None


class TimelineSession:
    """相机推理线程里的一次连续会话。"""

    def __init__(self, bundle: RuntimeBundle, confirm_seconds: float | None = None) -> None:
        self.bundle = bundle
        # 归一化参数随模型包固定，首帧转成数组后复用，不在每帧重建。
        self._feature_norm: tuple[np.ndarray, np.ndarray] | None = None
        fps = float(bundle.time_base_fps)
        self.min_frames = max(1, int(round(float(bundle.min_action_seconds) * fps)))
        # 未单独配置时沿用模型最短动作，避免旧回放提前结算。
        seconds = bundle.min_action_seconds if confirm_seconds is None else confirm_seconds
        self.confirm_frames = max(1, int(round(float(seconds) * fps)))
        self.sampler = TimeBaseSampler(bundle.time_base_fps)
        self.session_id = 1
        self._hidden = None
        self._runs: list[_Run] = []
        self._sample_index = 0
        self._event_serial = 0
        self._business_ready = False
        self._fault = ""
        self._phase = "待就绪"
        self._active_name = ""
        self._seen: list[str] = []
        self._missing: list[str] = []
        self.model = bundle.model
        self._pending: dict | None = None
        self.model.eval()
        trace(
            "session_open",
            session=self.session_id,
            min_frames=self.min_frames,
            confirm_frames=self.confirm_frames,
            fps=f"{fps:.3f}",
            classes=",".join(self._class_name(index) for index in range(len(bundle.class_ids))),
        )

    def reset(self) -> None:
        """硬重置：清掉记忆、采样和零件证据，重新等待动作边界。"""
        self.sampler.reset()
        self._hidden = None
        self._runs = []
        self._sample_index = 0
        self._business_ready = False
        self._fault = ""
        self._phase = "待就绪"
        self._active_name = ""
        self._seen = []
        self._missing = []
        self._pending = None
        self.session_id += 1
        trace("session_reset", session=self.session_id)

    def stop(self) -> dict:
        """停止时不把还没结束的尾段算成完成。"""
        return self._status([])

    def observe_frame(
        self,
        *,
        source_frame: int,
        timestamp: float,
        detections: list[dict],
        hands: list,
        frame_w: int,
        frame_h: int,
        hand_fault: str = "",
        class_index: int | None = None,
    ) -> dict:
        if self._fault:
            return self._status([])
        if hand_fault:
            self._fail(hand_fault)
            return self._status([])
        decision = self.sampler.push(int(source_frame), float(timestamp))
        if decision == "interrupt":
            self._fail("采集流中断，动作判定已停止")
            return self._status([])
        current = {
            "source_frame": int(source_frame), "timestamp": float(timestamp),
            "detections": detections, "hands": hands, "frame_w": frame_w, "frame_h": frame_h,
            "class_index": class_index, "parts": self._parts_in_frame(detections),
        }
        selected = None
        if isinstance(decision, int):
            selected = current if self.sampler.selected_frame == source_frame else self._pending
        if self._pending is not None and self._pending is not selected:
            self._add_parts(self._pending["parts"])
        self._pending = None if selected is current else current
        if selected is None:
            trace(
                "frame",
                frame=int(source_frame),
                selected=0,
                decision="hold" if self.sampler.hold else "skip",
            )
            self._refresh_phase()
            return self._status([])
        class_index = selected["class_index"]
        probs = None
        if class_index is None:
            try:
                classified = self._classify(selected["detections"], selected["hands"], selected["frame_w"], selected["frame_h"])
            except Exception as exc:
                self._fail(str(exc))
                return self._status([])
            if isinstance(classified, tuple):
                class_index, probs = classified
            else:
                class_index = classified
        self._append(int(class_index), selected["source_frame"], selected["timestamp"], selected["parts"])
        self._trace_sample(selected, int(class_index), probs)
        events = [] if self.sampler.hold else self._close_confirmed(int(source_frame), float(timestamp))
        self._refresh_phase()
        return self._status(events)

    def set_fault(self, reason: str) -> dict:
        """推理线程的依赖异常统一锁定会话，只有硬重置才能恢复。"""
        self._fail(str(reason))
        return self._status([])

    def _fail(self, reason: str) -> None:
        self._fault = reason
        self.sampler.reset()
        self._hidden = None
        self._runs = []
        self._business_ready = False
        self._phase = "故障"
        self._active_name = ""
        self._seen = []
        self._missing = []
        self._pending = None
        trace("session_fault", session=self.session_id, reason=self._fault)

    def _parts_in_frame(self, detections: list[dict]) -> set[str]:
        found: set[str] = set()
        threshold = self.bundle.yolo_conf_threshold
        known = set(self.bundle.part_names)
        for item in detections or []:
            name = str(item.get("label") or "")
            if name not in known:
                continue
            if float(item.get("conf") or 0.0) + 1e-9 < threshold:
                continue
            found.add(name)
        return found

    def _add_parts(self, parts: set[str]) -> None:
        if self._runs and parts:
            self._runs[-1].parts.update(parts)

    def _classify(self, detections, hands, frame_w: int, frame_h: int) -> int:
        import torch

        row = build_feature_row(
            detections,
            _hands_xyz(hands),
            frame_w,
            frame_h,
            self.bundle.part_names,
            self.bundle.yolo_conf_threshold,
            include_hands=bool(getattr(self.bundle, "hands_in_training", True)),
        )
        if self._feature_norm is None:
            self._feature_norm = (
                np.asarray(self.bundle.feature_mean, dtype=np.float32),
                np.asarray(self.bundle.feature_std, dtype=np.float32),
            )
        mean, std = self._feature_norm
        values = (np.asarray(row, dtype=np.float32) - mean) / std
        batch = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            logits, self._hidden = self.model.step(batch, self._hidden)
        values = torch.softmax(logits[0, -1], dim=0)
        probs = ",".join(
            f"{self._class_name(index)}:{float(value):.3f}"
            for index, value in enumerate(values.tolist())
        )
        return int(torch.argmax(values).item()), probs

    def _append(self, class_index: int, source_frame: int, timestamp: float, parts: set[str]) -> None:
        if self._runs and self._runs[-1].class_index != class_index:
            # 后继真实样本确立前一段的半开终点，不能用最后样本时间冒充终点。
            self._runs[-1].source_end = source_frame
            self._runs[-1].time_end = timestamp
            # 已结束的短段不再可能变长，立即并入前段，避免抖动永久堆积。
            if len(self._runs) > 1 and _length(self._runs[-1]) < self.min_frames:
                short = self._runs[-1]
                previous = self._runs[-2]
                trace(
                    "absorb_short",
                    note="短段并入上一段",
                    name=self._class_name(short.class_index),
                    into=self._class_name(previous.class_index),
                    length=_length(short),
                    start_frame=short.source_start,
                    end_frame=short.source_end,
                )
                self._runs.pop()
                previous = self._runs[-1]
                previous.end = short.end
                previous.source_end = short.source_end
                previous.time_end = short.time_end
                previous.parts.update(short.parts)
        if self._runs and self._runs[-1].class_index == class_index:
            run = self._runs[-1]
            run.end = self._sample_index + 1
            run.source_end = source_frame + 1
            run.time_end = timestamp + self.sampler.period
            run.parts.update(parts)
        else:
            self._runs.append(_Run(
                class_index,
                self._sample_index,
                self._sample_index + 1,
                source_frame,
                source_frame + 1,
                timestamp,
                timestamp + self.sampler.period,
                set(parts),
            ))
        self._sample_index += 1

    def _close_confirmed(self, decision_frame: int, decision_time: float) -> list[dict]:
        if len(self._runs) < 2 or _length(self._runs[-1]) < self.confirm_frames:
            return []
        open_run = self._runs[-1]
        # 确认时间短于动作最短长度时，后继段还没长到可被吸收的门槛。
        # 这里必须留下后继段，否则它会被并回上一个动作，切换又变回原来那么慢。
        if _length(open_run) < self.min_frames:
            events = []
            for run in _absorb(self._runs[:-1], self.min_frames):
                # 前一段已经结算过，短段没有可并入的目标；不够最短动作长度的直接丢掉，不能当成完成。
                if _length(run) >= self.min_frames:
                    events.extend(self._settle(run, decision_frame, decision_time))
                else:
                    trace(
                        "event_reject",
                        reason="shorter_than_min_action",
                        note="不够最短动作长度",
                        name=self._class_name(run.class_index),
                        start_frame=run.source_start,
                        end_frame=run.source_end,
                        length=_length(run),
                        min_frames=self.min_frames,
                    )
            self._runs = [open_run]
            return events
        closed = _absorb(self._runs, self.min_frames)
        if not closed:
            return []
        open_run = closed[-1]
        events = []
        for run in closed[:-1]:
            events.extend(self._settle(run, decision_frame, decision_time))
        self._runs = [open_run]
        return events

    def _settle(self, run: _Run, decision_frame: int, decision_time: float) -> list[dict]:
        class_id = self.bundle.class_ids[run.class_index]
        name = self.bundle.names_by_id.get(class_id, class_id)
        if not self._business_ready:
            self._business_ready = True
            trace(
                "event_reject",
                reason="startup_first_run",
                note="启动后的第一段不结算",
                name=name,
                start_frame=run.source_start,
                end_frame=run.source_end,
                length=_length(run),
            )
            return []
        if class_id == "idle":
            trace(
                "event_reject",
                reason="idle",
                note="空闲不结算",
                name=name,
                start_frame=run.source_start,
                end_frame=run.source_end,
            )
            return []
        required = self.bundle.required_parts.get(class_id) or []
        missing = [part for part in required if part not in run.parts]
        self._event_serial += 1
        trace(
            "event_emit",
            session=self.session_id,
            event=self._event_serial,
            name=name,
            start_frame=run.source_start,
            end_frame=run.source_end,
            decision_frame=int(decision_frame),
            parts=",".join(sorted(run.parts)) or "-",
            missing=",".join(missing) or "-",
        )
        return [{
            "session_id": self.session_id,
            "event_id": self._event_serial,
            "class_id": class_id,
            "name": name,
            "start_frame": run.source_start,
            "end_frame": run.source_end,
            "start_time": run.time_start,
            "end_time": run.time_end,
            "decision_frame": int(decision_frame),
            "decision_time": float(decision_time),
            "parts": sorted(run.parts),
            "missing_parts": missing,
            "weights_sha256": self.bundle.weights_sha256,
        }]

    def _refresh_phase(self) -> None:
        if self._fault:
            self._phase = "故障"
            return
        if not self._runs:
            self._phase = "待就绪"
            self._active_name = ""
            self._seen = []
            self._missing = []
            return
        run = self._runs[-1]
        class_id = self.bundle.class_ids[run.class_index]
        name = self.bundle.names_by_id.get(class_id, class_id)
        if class_id == "idle" or not self._business_ready:
            self._phase = "待就绪"
            self._active_name = "" if class_id == "idle" else name
        elif _length(run) >= self.min_frames:
            self._phase = "等待结束"
            self._active_name = name
        else:
            self._phase = "进行中"
            self._active_name = name
        required = self.bundle.required_parts.get(class_id) or []
        self._seen = sorted(part for part in run.parts if part in required)
        self._missing = [part for part in required if part not in run.parts] if class_id != "idle" else []

    def _class_name(self, class_index: int) -> str:
        try:
            class_id = self.bundle.class_ids[int(class_index)]
        except (IndexError, TypeError):
            return str(class_index)
        return self.bundle.names_by_id.get(class_id, class_id)

    def _trace_sample(self, selected: dict, class_index: int, probs: str | None) -> None:
        run = self._runs[-1]
        class_id = self.bundle.class_ids[run.class_index]
        required = self.bundle.required_parts.get(class_id) or []
        missing = [part for part in required if part not in run.parts] if class_id != "idle" else []
        trace(
            "sample",
            frame=selected["source_frame"],
            time=f"{float(selected['timestamp']):.3f}",
            class_name=self._class_name(class_index),
            probs=probs or "-",
            open=self._class_name(run.class_index),
            open_len=_length(run),
            confirm_frames=self.confirm_frames,
            min_frames=self.min_frames,
            detected=",".join(sorted(run.parts)) or "-",
            missing=",".join(missing) or "-",
        )

    def _status(self, events: list[dict]) -> dict:
        return {
            "events": events,
            "fault": self._fault,
            "phase": self._phase,
            "action": self._active_name,
            "seen_parts": list(self._seen),
            "missing_parts": list(self._missing),
            "session_id": self.session_id,
            "evidence_frames": sorted({
                frame for run in self._runs for frame in (run.source_start, run.source_end)
            } | ({self._pending["source_frame"]} if self._pending else set())),
        }


def _length(run: _Run) -> int:
    return int(run.end) - int(run.start)


def _absorb(runs: list[_Run], min_frames: int) -> list[_Run]:
    """和离线 absorb_short_segments 同一套：短段并进前一段，开头短段并进后一段，相邻同类再合并。"""
    if not runs:
        return []
    merged = [runs[0].copy()]
    for run in runs[1:]:
        if _length(run) < min_frames:
            merged[-1].end = run.end
            merged[-1].source_end = run.source_end
            merged[-1].time_end = run.time_end
            merged[-1].parts.update(run.parts)
            continue
        merged.append(run.copy())
    if len(merged) > 1 and _length(merged[0]) < min_frames:
        merged[1].start = merged[0].start
        merged[1].source_start = merged[0].source_start
        merged[1].time_start = merged[0].time_start
        merged[1].parts.update(merged[0].parts)
        merged.pop(0)
    coalesced = [merged[0]]
    for run in merged[1:]:
        if run.class_index == coalesced[-1].class_index:
            coalesced[-1].end = run.end
            coalesced[-1].source_end = run.source_end
            coalesced[-1].time_end = run.time_end
            coalesced[-1].parts.update(run.parts)
            continue
        coalesced.append(run)
    return coalesced


def _hands_xyz(hands) -> list:
    converted = []
    for hand in hands or []:
        if isinstance(hand, dict):
            points = hand.get("points") or []
            depths = list(hand.get("depths") or [])
            if len(points) >= 21 and len(depths) < 21:
                raise ValueError("手部点缺少深度，不能当成没检出手")
            xyz = []
            for index, point in enumerate(points[:21]):
                xyz.append((float(point[0]), float(point[1]), float(depths[index]) if index < len(depths) else 0.0))
            converted.append(xyz)
            continue
        converted.append(hand)
    return converted
