"""真实视频独立回放：一次实际抽特征，对照离线和在线的数值、边界与零件门禁。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "yolo_action_detection" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lstm_action.bundle import load_runtime_bundle
from lstm_action.stream import TimelineSession
from lstm_workbench.check_clip import decode_timeline
from lstm_workbench.features import build_feature_row
from lstm_workbench.model_registry import file_sha256, live_backends
from lstm_workbench.service import resample_indices


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.logits = []
        self.inputs = []

    def eval(self):
        self.model.eval()

    def step(self, values, hidden):
        logits, hidden = self.model.step(values, hidden)
        self.inputs.append(values.detach().cpu().numpy()[0, 0].copy())
        self.logits.append(logits.detach().cpu().numpy()[0, -1].copy())
        return logits, hidden


class RecordingSession(TimelineSession):
    def __init__(self, bundle):
        super().__init__(bundle)
        self.selected_sources = []

    def _classify(self, detections, hands, frame_w, frame_h):
        marker = next(item for item in detections if "_replay_source" in item)
        self.selected_sources.append(marker["_replay_source"])
        return super()._classify(detections, hands, frame_w, frame_h)


def verify(bundle_path: Path, video_path: Path, yolo_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("输出目录已存在，不覆盖已有验收记录")
    output.mkdir(parents=True)
    contract = json.loads((bundle_path / "contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle_path / "runtime_manifest.json").read_text(encoding="utf-8"))
    names = {item["id"]: item["name"] for item in manifest["action_classes"]}
    bundle = load_runtime_bundle(
        bundle_path, category_names=[name for key, name in names.items() if key != "idle"],
        yolo_path=str(yolo_path), yolo_conf=float(contract["yolo_conf_threshold"]),
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"打不开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps > 0:
        raise ValueError("视频没有真实帧率")
    settings = {"yolo_path": str(yolo_path), "yolo_conf_threshold": bundle.yolo_conf_threshold}
    detect, hands_backend = live_backends(settings)
    recorder = RecordingModel(bundle.model)
    session = RecordingSession(replace(bundle, model=recorder))
    rows, events, frames = [], [], []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source = len(rows)
            height, width = frame.shape[:2]
            detections = detect(frame)
            hands = hands_backend(frame)
            rows.append(build_feature_row(detections, hands, width, height, bundle.part_names, bundle.yolo_conf_threshold))
            frames.append({"source_frame": source, "timestamp": source / fps, "detections": detections, "hands": hands})
            # 标记只供独立校验，不属于模型类别，不改变特征或零件门禁。
            tagged = detections + [{"_replay_source": source}]
            report = session.observe_frame(source_frame=source, timestamp=source / fps,
                                           detections=tagged, hands=hands, frame_w=width, frame_h=height)
            if report.get("fault"):
                raise RuntimeError(f"第 {source} 帧动作故障：{report['fault']}")
            events.extend(report["events"])
            if source % 80 == 0:
                print(f"已回放 {source} 帧", flush=True)
    finally:
        capture.release()
        close = getattr(hands_backend, "close", None)
        if close:
            close()
    elapsed = time.perf_counter() - started
    rows = np.asarray(rows, dtype=np.float32)
    np.save(output / "features.npy", rows)
    (output / "frame-results.json").write_text(json.dumps(frames, ensure_ascii=False), encoding="utf-8")
    indices = resample_indices(len(rows), fps, bundle.time_base_fps)
    normalized = (rows[np.asarray(indices)] - np.asarray(bundle.feature_mean, dtype=np.float32)) / np.asarray(bundle.feature_std, dtype=np.float32)
    with torch.no_grad():
        offline_logits = bundle.model(torch.tensor(normalized[None], dtype=torch.float32), all_steps=True)[0].cpu().numpy()
    online_logits = np.asarray(recorder.logits, dtype=np.float32)
    count = len(online_logits)
    sampling_ok = session.selected_sources == indices[:count]
    input_ok = bool(np.allclose(np.asarray(recorder.inputs), normalized[:count], atol=1e-6, rtol=1e-6))
    logits_ok = bool(np.allclose(online_logits, offline_logits[:count], atol=1e-5, rtol=1e-5))
    offline_segments, _labels = decode_timeline(rows, fps, bundle.model, contract, bundle.part_names, names)
    mismatches = []
    for event in events:
        match = next((segment for segment in offline_segments
                      if str(segment["class_id"]) == str(event["class_id"])
                      and int(segment["start_frame"]) == int(event["start_frame"])
                      and int(segment["end_frame"]) == int(event["end_frame"])), None)
        if match is None or sorted(match.get("missing_parts") or []) != sorted(event.get("missing_parts") or []):
            mismatches.append({"event": event, "offline": match})
    # 除头段和未结束尾段外，应结算的稳定段不能凭“没有事件”空通过。
    expected = [segment for segment in offline_segments[1:-1] if segment["class_id"] != "idle"]
    event_keys = {(event["class_id"], event["start_frame"], event["end_frame"]) for event in events}
    missing_events = [segment for segment in expected if
                      (segment["class_id"], segment["start_frame"], segment["end_frame"]) not in event_keys]
    before_stop = len(events)
    stop_report = session.stop()
    stop_ok = not stop_report.get("events") and len(events) == before_stop
    report = {
        "ok": sampling_ok and input_ok and logits_ok and not mismatches and not missing_events and stop_ok and count > 0,
        "scope": "真实视频及实际三模型的回放一致性；不是现场产线验收或模型准确率结论",
        "video": str(video_path.resolve()), "video_sha256": file_sha256(video_path),
        "bundle": str(bundle_path.resolve()), "weights_sha256": bundle.weights_sha256,
        "contract_sha256": bundle.contract_sha256, "yolo_sha256": file_sha256(yolo_path),
        "hands_sha256": contract["hands_sha256"], "fps": fps,
        "time_base_fps": bundle.time_base_fps, "frames": len(rows), "samples": count,
        "elapsed_seconds": elapsed, "processed_fps": len(rows) / elapsed,
        "sampling_ok": sampling_ok, "feature_inputs_ok": input_ok, "logits_ok": logits_ok,
        "max_logit_error": float(np.max(np.abs(online_logits - offline_logits[:count]))) if count else None,
        "selected_sources": session.selected_sources, "offline_indices": indices,
        "offline_segments": offline_segments, "events": events, "mismatches": mismatches,
        "expected_stable_events": len(expected), "missing_events": missing_events,
        "stop_does_not_settle_tail": stop_ok,
        "head_tail_policy": "启动首个非空闲动作丢弃；未确认结束的尾段不补事件；因果采样未消费的最后源帧不补造样本",
    }
    (output / "replay-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.bundle, args.video, args.yolo, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"events", "offline_segments", "selected_sources", "offline_indices"}}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
