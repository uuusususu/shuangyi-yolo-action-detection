"""Realtime pipeline diagnostics for the YOLO OBB runtime."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from app_state import AppState
from camera.camera_worker import CameraWorker
from config import ConfigManager
from yolo_runtime.yolo_obb_processor import YoloObbProcessor


def create_processor(config: ConfigManager) -> YoloObbProcessor:
    """Create the same YOLO processor settings used by the desktop runtime."""
    processor = YoloObbProcessor(
        model_path=config.yolo_model_path,
        conf_threshold=config.yolo_conf_threshold,
        iou_threshold=config.yolo_iou_threshold,
        device=config.ultralytics_device,
        half=config.ultralytics_half,
        tracker=config.ultralytics_tracker,
        max_det=config.ultralytics_max_det,
        track_persist=config.ultralytics_track_persist,
    )
    processor.load()
    return processor


def summarize_worker(worker: CameraWorker, *, started_at: float, ended_at: float) -> dict:
    """Return JSON-serializable realtime pipeline metrics."""
    stats = worker.get_pipeline_stats()
    latest_overlay = worker.get_latest_overlay()
    latest_preview_id = worker.get_latest_preview_frame_id()
    latest_display_frame, latest_display_id = worker.get_display_frame()
    result_age_ms: Optional[float] = None
    if latest_overlay.timestamp:
        result_age_ms = max(0.0, (ended_at - latest_overlay.timestamp) * 1000)

    return {
        "duration_s": round(max(0.0, ended_at - started_at), 3),
        "capture_fps": stats.get("capture_fps", 0.0),
        "infer_fps": stats.get("infer_fps", 0.0),
        "dropped_for_infer": stats.get("dropped_for_infer", 0),
        "infer_latency_ms": stats.get("infer_latency_ms", 0.0),
        "latest_preview_frame_id": latest_preview_id,
        "latest_inferred_frame_id": latest_overlay.source_frame_id,
        "latest_display_frame_id": latest_display_id if latest_display_frame is not None else 0,
        "display_lag_frames": max(0, latest_preview_id - int(latest_overlay.source_frame_id or 0)),
        "result_age_ms": result_age_ms,
        "overlay_detection_count": len(latest_overlay.detections),
        "overlay_error": latest_overlay.error,
    }


def run_realtime_diagnostics(
    *,
    config_path: Path,
    duration_s: float,
    poll_interval_s: float = 0.05,
    open_timeout_s: float = 3.0,
    processor_factory: Callable[[ConfigManager], object] = create_processor,
) -> dict:
    """Run the real camera/YOLO worker for a short interval and return metrics."""
    config = ConfigManager()
    config.load(config_path)
    state = AppState()
    worker = CameraWorker(config, state)

    started_at = time.time()
    try:
        open_result = _open_camera_with_timeout(worker, timeout_s=open_timeout_s)
        if open_result == "timeout":
            return {
                "ok": False,
                "error": "camera_open_timeout",
                "duration_s": round(max(0.0, time.time() - started_at), 3),
            }
        if open_result is not True:
            return {
                "ok": False,
                "error": state.last_error or "camera_open_failed",
                "duration_s": round(max(0.0, time.time() - started_at), 3),
            }
        processor = processor_factory(config)
        worker.set_frame_processor(processor)
        worker.start_inference()
        deadline = time.time() + max(0.1, float(duration_s))
        while time.time() < deadline:
            time.sleep(max(0.001, float(poll_interval_s)))
        ended_at = time.time()
        summary = summarize_worker(worker, started_at=started_at, ended_at=ended_at)
        summary["ok"] = True
        return summary
    finally:
        worker.close_camera()


def _open_camera_with_timeout(worker: CameraWorker, *, timeout_s: float):
    """Open camera with timeout protection for SDK/OpenCV calls that can block."""
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put(worker.open_camera())
        except Exception as exc:
            result_queue.put(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        result = result_queue.get(timeout=max(0.001, float(timeout_s)))
    except queue.Empty:
        return "timeout"
    if isinstance(result, Exception):
        raise result
    return result


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "config.json"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run realtime YOLO pipeline diagnostics.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to config.json")
    parser.add_argument("--duration", type=float, default=5.0, help="Diagnostic duration in seconds")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Polling interval in seconds")
    parser.add_argument("--open-timeout", type=float, default=3.0, help="Camera open timeout in seconds")
    args = parser.parse_args(argv)

    summary = run_realtime_diagnostics(
        config_path=args.config,
        duration_s=args.duration,
        poll_interval_s=args.poll_interval,
        open_timeout_s=args.open_timeout,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
