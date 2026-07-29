"""模板匹配现场诊断记录器。

产出三样东西，供现场测试后离线分析：

1. ``<session>.ndjson``          每帧一条记录，含完整的逐角度得分曲线
2. ``<session>.summary.json``    会话汇总：得分分布、最佳角度直方图、
                                 逐角度平均得分曲线、旋转掉分比、耗时分位数、失败原因统计
3. ``<session>_frames/*.png``    按得分分层抽样的搜索域原始裁剪（无损 PNG），
                                 用于离线重放任意算法

设计约束：
- ``record()`` 在推理线程调用，必须非阻塞且绝不抛出异常
- 未启用时所有调用都是廉价的空操作
- 队列有界、抽样帧数与字节数有上限，不会写爆现场磁盘
"""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("yolo_action_detection.runtime")

RECORD_SCHEMA_VERSION = "template_match_log.v1"
SUMMARY_SCHEMA_VERSION = "template_match_summary.v1"

_SCORE_BIN = 0.05
_SAMPLE_PER_BAND = 6
# 抽样裁剪在搜索 ROI 四周各留一倍余量，便于离线验证“放大搜索 ROI 后”的效果。
_SAMPLE_MARGIN_RATIO = 1.0
_MAX_SAMPLE_BYTES = 256 * 1024 * 1024


def resolve_diagnostics_dir(log_dir: str | Path, base_dir: str | Path | None = None) -> Path:
    """相对目录优先落在应用基准目录下，便携包与源码运行都可写。"""
    path = Path(log_dir)
    if path.is_absolute():
        return path
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    return root / path


class TemplateMatchDiagnostics:
    """模板匹配会话记录器。"""

    def __init__(self) -> None:
        self._enabled = False
        self._shadow_enabled = False
        self._log_dir = resolve_diagnostics_dir("logs/template_sessions")
        self._sample_limit = 0
        self._max_queue = 600
        self._threaded = True

        self._lock = threading.Lock()
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._file = None
        self._active = False
        self._session_id = ""
        self._log_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        self._last_error = ""
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._sampled = 0
        self._sample_bytes = 0
        self._band_counts: Dict[int, int] = {}
        self._stats = _new_stats()

    # ------------------------------------------------------------------
    # 配置与生命周期
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        enabled: bool,
        log_dir: str | Path = "logs/template_sessions",
        base_dir: str | Path | None = None,
        shadow_enabled: bool = True,
        sample_frames: int = 120,
        max_queue: int = 600,
        threaded: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._shadow_enabled = bool(shadow_enabled)
        self._log_dir = resolve_diagnostics_dir(log_dir, base_dir)
        self._sample_limit = max(0, int(sample_frames or 0))
        self._max_queue = max(1, int(max_queue or 1))
        self._threaded = bool(threaded)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def shadow_enabled(self) -> bool:
        return self._enabled and self._shadow_enabled

    @property
    def active(self) -> bool:
        return self._active

    @property
    def log_path(self) -> str:
        return str(self._log_path or "")

    @property
    def summary_path(self) -> str:
        return str(self._summary_path or "")

    @property
    def last_error(self) -> str:
        return self._last_error

    def start_session(
        self,
        context: Optional[Dict[str, object]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """开始一次诊断会话；未启用或创建失败时返回空字符串。"""
        if not self._enabled or self._active:
            return self._session_id if self._active else ""
        self._last_error = ""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._session_id = self._unique_session_id(session_id)
            self._log_path = self._log_dir / f"{self._session_id}.ndjson"
            self._summary_path = self._log_dir / f"{self._session_id}.summary.json"
            self._frames_dir = self._log_dir / f"{self._session_id}_frames"
            if self._sample_limit > 0:
                self._frames_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self._log_path, "w", encoding="utf-8")
            self._queue = queue.Queue(maxsize=self._max_queue)
            self._stop_event.clear()
            self._submitted = 0
            self._written = 0
            self._dropped = 0
            self._sampled = 0
            self._sample_bytes = 0
            self._band_counts = {}
            self._stats = _new_stats()
            self._stats["session_id"] = self._session_id
            self._stats["started_at"] = time.time()
            self._stats["context"] = dict(context or {})
            self._active = True
            if self._threaded:
                self._thread = threading.Thread(
                    target=self._writer_loop,
                    name="TemplateDiagWriter",
                    daemon=True,
                )
                self._thread.start()
            logger.info(
                "template_diagnostics=started session=%s log=%s shadow=%s sample_limit=%d",
                self._session_id,
                self._log_path,
                self._shadow_enabled,
                self._sample_limit,
            )
            return self._session_id
        except Exception as exc:
            self._last_error = f"模板诊断日志创建失败: {exc}"
            logger.exception("template_diagnostics=start_failed error=%s", exc)
            self._active = False
            self._close_file()
            return ""

    def stop_session(self) -> Dict[str, object]:
        if not self._active:
            return {}
        self._active = False
        self._stop_event.set()
        if self._threaded and self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._last_error = "模板诊断日志关闭超时，可能有少量记录未落盘"
        else:
            self._drain_pending()
        self._thread = None
        self._close_file()
        summary = self.build_summary()
        if self._summary_path is not None:
            try:
                self._summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self._last_error = f"模板诊断 summary 写入失败: {exc}"
                logger.exception("template_diagnostics=summary_failed error=%s", exc)
        logger.info(
            "template_diagnostics=stopped session=%s frames=%s matched_rate=%.3f summary=%s",
            self._session_id,
            summary.get("frames"),
            float(summary.get("matched_rate") or 0.0),
            self._summary_path,
        )
        return summary

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def record(
        self,
        payload: Dict[str, object],
        sample_image: Optional[np.ndarray] = None,
    ) -> bool:
        """提交单帧记录。推理线程调用，绝不抛出。"""
        if not self._active or self._queue is None:
            return False
        try:
            record = dict(payload)
            record["schema_version"] = RECORD_SCHEMA_VERSION
            crop = None
            if sample_image is not None:
                crop = self._maybe_take_sample(record, sample_image)
            self._queue.put_nowait((record, crop))
            with self._lock:
                self._submitted += 1
            return True
        except queue.Full:
            self._increment_dropped("queue_full")
            return False
        except Exception as exc:
            self._increment_dropped(f"record_failed({exc})")
            return False

    def _maybe_take_sample(
        self,
        record: Dict[str, object],
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """按得分分层抽样搜索域裁剪，用于离线重放。"""
        if self._sample_limit <= 0 or self._frames_dir is None:
            return None
        with self._lock:
            if self._sampled >= self._sample_limit or self._sample_bytes >= _MAX_SAMPLE_BYTES:
                return None
            score = float(record.get("score") or 0.0)
            band = int(min(0.999999, max(0.0, score)) / _SCORE_BIN)
            interesting = bool(record.get("shadow_disagrees")) or bool(record.get("failure"))
            quota = _SAMPLE_PER_BAND * (2 if interesting else 1)
            if self._band_counts.get(band, 0) >= quota and self._submitted > 5:
                return None
            self._band_counts[band] = self._band_counts.get(band, 0) + 1
            self._sampled += 1
            index = self._sampled

        rect = record.get("search_rect_px")
        height, width = frame.shape[:2]
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            x0, y0, x1, y1 = (int(v) for v in rect)
            margin_x = int((x1 - x0) * _SAMPLE_MARGIN_RATIO)
            margin_y = int((y1 - y0) * _SAMPLE_MARGIN_RATIO)
            x0 = max(0, x0 - margin_x)
            y0 = max(0, y0 - margin_y)
            x1 = min(width, x1 + margin_x)
            y1 = min(height, y1 + margin_y)
        else:
            x0, y0, x1, y1 = 0, 0, width, height
        if x1 <= x0 or y1 <= y0:
            return None
        name = f"f{index:04d}_fid{int(record.get('frame_id') or 0)}_s{int(round(float(record.get('score') or 0.0) * 1000)):04d}.png"
        record["sample_frame"] = f"{self._frames_dir.name}/{name}"
        record["sample_offset_px"] = [x0, y0]
        return frame[y0:y1, x0:x1].copy()

    # ------------------------------------------------------------------
    # 写入线程
    # ------------------------------------------------------------------
    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or (self._queue is not None and not self._queue.empty()):
            try:
                item = self._queue.get(timeout=0.1) if self._queue is not None else None
            except queue.Empty:
                continue
            if item is None:
                continue
            try:
                self._write(item[0], item[1])
            finally:
                if self._queue is not None:
                    self._queue.task_done()

    def _drain_pending(self) -> None:
        if self._queue is None:
            return
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write(item[0], item[1])
            self._queue.task_done()

    def _write(self, record: Dict[str, object], crop: Optional[np.ndarray]) -> None:
        if self._file is None:
            self._increment_dropped("file_closed")
            return
        try:
            self._file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            self._file.flush()
            self._accumulate(record)
            with self._lock:
                self._written += 1
        except Exception as exc:
            self._increment_dropped(f"write_failed({exc})")
            self._last_error = f"模板诊断写入失败: {exc}"
        if crop is None:
            return
        try:
            import cv2

            name = str(record.get("sample_frame") or "").split("/")[-1]
            if not name or self._frames_dir is None:
                return
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                return
            payload = encoded.tobytes()
            (self._frames_dir / name).write_bytes(payload)
            with self._lock:
                self._sample_bytes += len(payload)
        except Exception as exc:
            self._last_error = f"模板诊断抽样帧写入失败: {exc}"

    def _increment_dropped(self, reason: str) -> None:
        with self._lock:
            self._dropped += 1
            self._last_error = reason

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def _accumulate(self, record: Dict[str, object]) -> None:
        with self._lock:
            stats = self._stats
            stats["frames"] += 1
            score = _as_float(record.get("score"))
            matched = bool(record.get("matched"))
            if matched:
                stats["matched_frames"] += 1
            if score is not None:
                stats["_scores"].append(score)
                band = round(math.floor(score / _SCORE_BIN) * _SCORE_BIN, 2)
                stats["score_histogram"][f"{band:.2f}"] = (
                    stats["score_histogram"].get(f"{band:.2f}", 0) + 1
                )

            failure = str(record.get("failure") or "")
            if failure:
                stats["failures"][failure] = stats["failures"].get(failure, 0) + 1
            if record.get("skipped_angles"):
                stats["frames_with_skipped_angles"] += 1
                for angle in record["skipped_angles"]:
                    key = f"{_as_float(angle) or 0.0:.1f}"
                    stats["skipped_angle_histogram"][key] = (
                        stats["skipped_angle_histogram"].get(key, 0) + 1
                    )

            best_angle = _as_float(record.get("angle"))
            if best_angle is not None:
                key = f"{best_angle:.1f}"
                stats["best_angle_histogram"][key] = stats["best_angle_histogram"].get(key, 0) + 1
                if score is not None:
                    if abs(best_angle) < 1e-6:
                        stats["_zero_angle_scores"].append(score)
                    else:
                        stats["_nonzero_angle_scores"].append(score)

            for entry in record.get("angle_scores") or []:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    continue
                angle = _as_float(entry[0])
                value = _as_float(entry[1])
                if angle is None or value is None:
                    continue
                key = f"{angle:.1f}"
                bucket = stats["_angle_curve"].setdefault(key, [0, 0.0, -9.9])
                bucket[0] += 1
                bucket[1] += value
                bucket[2] = max(bucket[2], value)

            center = record.get("center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                cx, cy = _as_float(center[0]), _as_float(center[1])
                if cx is not None and cy is not None:
                    stats["_center_x"].append(cx)
                    stats["_center_y"].append(cy)

            total_ms = _as_float(record.get("total_ms"))
            if total_ms is not None:
                stats["_total_ms"].append(total_ms)
            search = record.get("search_stats") or {}
            for name in ("mean", "std", "edge_pixels"):
                value = _as_float(search.get(name))
                if value is not None:
                    stats["_search"][name].append(value)

            shadow = record.get("shadow") or {}
            shadow_score = _as_float(shadow.get("score"))
            if shadow_score is not None:
                stats["_shadow_scores"].append(shadow_score)
                shadow_angle = _as_float(shadow.get("angle"))
                if shadow_angle is not None:
                    key = f"{shadow_angle:.1f}"
                    stats["shadow_best_angle_histogram"][key] = (
                        stats["shadow_best_angle_histogram"].get(key, 0) + 1
                    )
                    if abs(shadow_angle) < 1e-6:
                        stats["_shadow_zero_angle_scores"].append(shadow_score)
                    else:
                        stats["_shadow_nonzero_angle_scores"].append(shadow_score)
                elapsed = _as_float(shadow.get("elapsed_ms"))
                if elapsed is not None:
                    stats["_shadow_ms"].append(elapsed)
            yolo = record.get("yolo") or {}
            if yolo.get("ran"):
                stats["yolo_frames"] += 1
                for label, count in (yolo.get("labels") or {}).items():
                    label_stats = stats["yolo_label_counts"].setdefault(str(label), [0, 0])
                    label_stats[0] += 1
                    label_stats[1] += int(count or 0)

    def build_summary(self) -> Dict[str, object]:
        with self._lock:
            stats = self._stats
            scores = sorted(stats["_scores"])
            shadow_scores = sorted(stats["_shadow_scores"])
            frames = int(stats["frames"])
            summary: Dict[str, object] = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "session_id": stats["session_id"],
                "started_at": stats["started_at"],
                "ended_at": time.time(),
                "context": stats["context"],
                "log_path": str(self._log_path or ""),
                "frames": frames,
                "matched_frames": int(stats["matched_frames"]),
                "matched_rate": (stats["matched_frames"] / frames) if frames else 0.0,
                "yolo_frames": int(stats["yolo_frames"]),
                "submitted": self._submitted,
                "written": self._written,
                "dropped": self._dropped,
                "sampled_frames": self._sampled,
                "sample_bytes": self._sample_bytes,
                "last_error": self._last_error,
                "score": _distribution(scores),
                "score_histogram": dict(sorted(stats["score_histogram"].items())),
                "best_angle_histogram": _sorted_numeric(stats["best_angle_histogram"]),
                "angle_score_curve": {
                    key: {
                        "count": bucket[0],
                        "mean": bucket[1] / bucket[0] if bucket[0] else 0.0,
                        "max": bucket[2] if bucket[0] else 0.0,
                    }
                    for key, bucket in sorted(
                        stats["_angle_curve"].items(), key=lambda kv: float(kv[0])
                    )
                },
                "rotation": _rotation_stats(
                    stats["_zero_angle_scores"], stats["_nonzero_angle_scores"]
                ),
                "center_jitter_px": {
                    "x_std": _std(stats["_center_x"]),
                    "y_std": _std(stats["_center_y"]),
                    "x_range": _value_range(stats["_center_x"]),
                    "y_range": _value_range(stats["_center_y"]),
                },
                "total_ms": _distribution(sorted(stats["_total_ms"])),
                "failures": dict(sorted(stats["failures"].items())),
                "frames_with_skipped_angles": int(stats["frames_with_skipped_angles"]),
                "skipped_angle_histogram": _sorted_numeric(stats["skipped_angle_histogram"]),
                "search_stats": {
                    name: _distribution(sorted(values))
                    for name, values in stats["_search"].items()
                },
                "yolo_label_counts": {
                    label: {"frames": value[0], "total": value[1]}
                    for label, value in sorted(stats["yolo_label_counts"].items())
                },
            }
            if shadow_scores:
                summary["shadow"] = {
                    "frames": len(shadow_scores),
                    "score": _distribution(shadow_scores),
                    "best_angle_histogram": _sorted_numeric(stats["shadow_best_angle_histogram"]),
                    "rotation": _rotation_stats(
                        stats["_shadow_zero_angle_scores"],
                        stats["_shadow_nonzero_angle_scores"],
                    ),
                    "elapsed_ms": _distribution(sorted(stats["_shadow_ms"])),
                }
            return summary

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "active": self._active,
                "session_id": self._session_id,
                "log_path": str(self._log_path or ""),
                "summary_path": str(self._summary_path or ""),
                "submitted": self._submitted,
                "written": self._written,
                "dropped": self._dropped,
                "sampled_frames": self._sampled,
                "last_error": self._last_error,
            }

    # ------------------------------------------------------------------
    def _unique_session_id(self, requested: Optional[str]) -> str:
        base = requested or f"tpl_{time.strftime('%Y%m%d_%H%M%S')}"
        candidate = base
        index = 2
        while (self._log_dir / f"{candidate}.ndjson").exists():
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception as exc:
                self._last_error = f"模板诊断日志关闭失败: {exc}"
            self._file = None


def _new_stats() -> Dict[str, object]:
    return {
        "session_id": "",
        "started_at": 0.0,
        "context": {},
        "frames": 0,
        "matched_frames": 0,
        "yolo_frames": 0,
        "frames_with_skipped_angles": 0,
        "score_histogram": {},
        "best_angle_histogram": {},
        "shadow_best_angle_histogram": {},
        "skipped_angle_histogram": {},
        "failures": {},
        "yolo_label_counts": {},
        "_scores": [],
        "_shadow_scores": [],
        "_zero_angle_scores": [],
        "_nonzero_angle_scores": [],
        "_shadow_zero_angle_scores": [],
        "_shadow_nonzero_angle_scores": [],
        "_angle_curve": {},
        "_center_x": [],
        "_center_y": [],
        "_total_ms": [],
        "_shadow_ms": [],
        "_search": {"mean": [], "std": [], "edge_pixels": []},
    }


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _distribution(sorted_values: List[float]) -> Dict[str, float]:
    if not sorted_values:
        return {}
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p05": _percentile(sorted_values, 5),
        "p25": _percentile(sorted_values, 25),
        "p50": _percentile(sorted_values, 50),
        "p75": _percentile(sorted_values, 75),
        "p95": _percentile(sorted_values, 95),
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / len(sorted_values),
    }


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _value_range(values: List[float]) -> Optional[List[float]]:
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def _rotation_stats(zero_scores: List[float], nonzero_scores: List[float]) -> Dict[str, object]:
    """旋转掉分比：非零最佳角度帧的平均得分 / 零度帧的平均得分。

    显著小于 1 说明“旋转候选本身在退化”，而不是工件真的不像模板。
    """
    zero_mean = sum(zero_scores) / len(zero_scores) if zero_scores else None
    nonzero_mean = sum(nonzero_scores) / len(nonzero_scores) if nonzero_scores else None
    ratio = None
    if zero_mean and nonzero_mean is not None and zero_mean > 1e-9:
        ratio = nonzero_mean / zero_mean
    return {
        "zero_angle_frames": len(zero_scores),
        "nonzero_angle_frames": len(nonzero_scores),
        "zero_angle_mean_score": zero_mean,
        "nonzero_angle_mean_score": nonzero_mean,
        "penalty_ratio": ratio,
    }


def _sorted_numeric(histogram: Dict[str, int]) -> Dict[str, int]:
    try:
        return dict(sorted(histogram.items(), key=lambda kv: float(kv[0])))
    except ValueError:
        return dict(sorted(histogram.items()))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


_RECORDER = TemplateMatchDiagnostics()


def get_recorder() -> TemplateMatchDiagnostics:
    """全局记录器。未启用时所有调用都是空操作。"""
    return _RECORDER
