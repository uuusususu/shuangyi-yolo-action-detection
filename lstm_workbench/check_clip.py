"""用某一轮的权重看一条新视频。视频只放在这一轮的 check 目录，不进训练素材。"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .features import FEATURE_CODE_VERSION, build_feature_row, feature_size, hands_in_training, safe_feature_std
from .feature_store import conf_threshold_of
from .part_gate import apply_part_gate
from .service import resample_indices


def ActionLSTM(*args, **kwargs):  # noqa: N802
    from .model import ActionLSTM as _ActionLSTM

    return _ActionLSTM(*args, **kwargs)


def resolve_device(*args, **kwargs):
    from .trainer import resolve_device as _resolve_device

    return _resolve_device(*args, **kwargs)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class Stopped(RuntimeError):
    """验证被人停下。"""


def check_dir(run_dir: Path) -> Path:
    return run_dir / "check"


def clear_check(run_dir: Path) -> None:
    """去掉这一轮的验证视频和识别结果。权重和训练素材不动。"""
    folder = check_dir(run_dir)
    if folder.exists():
        shutil.rmtree(folder)


def reset_check(run_dir: Path) -> Path:
    """换一条验证视频时清掉上一条的文件和结果。训练素材不动。"""
    folder = check_dir(run_dir)
    if folder.exists():
        shutil.rmtree(folder)
    media = folder / "media"
    media.mkdir(parents=True)
    return folder


def save_check_video(run_dir: Path, filename: str, data: bytes, *, replace: bool) -> Path:
    """视频只落在 runs/<轮次>/check/media/。文件名里的路径丢掉，避免写出这一轮目录。"""
    name = Path(filename.replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("文件名不合法")
    if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("请上传视频。这一步要看一整段画面，一张图凑不满窗口。")
    if not data:
        raise ValueError("没有文件内容")
    if replace or not (check_dir(run_dir) / "media").is_dir():
        reset_check(run_dir)
    media = check_dir(run_dir) / "media"
    media.mkdir(parents=True, exist_ok=True)
    target = (media / name).resolve()
    if media.resolve() not in target.parents:
        raise ValueError("文件名不合法")
    target.write_bytes(data)
    return target


def locate_check_video(run_dir: Path) -> Path:
    media = check_dir(run_dir) / "media"
    if not media.is_dir():
        raise FileNotFoundError("还没有上传视频")
    videos = [path for path in media.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise FileNotFoundError("还没有上传视频")
    if len(videos) > 1:
        raise ValueError("一次只看一个视频")
    return videos[0]


def load_result(run_dir: Path) -> dict | None:
    path = check_dir(run_dir) / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def overlay_path(run_dir: Path) -> Path:
    return check_dir(run_dir) / "overlay.json"


def latest_project_clip(run_dir: Path) -> dict | None:
    """结果页刷新后还能看到上次用工程片子验证的结果。"""
    root = check_dir(run_dir) / "clips"
    if not root.is_dir():
        return None
    files = [path / "result.json" for path in root.iterdir() if path.is_dir() and (path / "result.json").is_file()]
    if not files:
        return None
    newest = max(files, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("clip_id", newest.parent.name)
    payload.setdefault("checked_at", newest.stat().st_mtime)
    payload.setdefault("project_clip", True)
    return payload


def _attach_saved_clip(run_dir: Path, result: dict) -> dict:
    saved = latest_project_clip(run_dir)
    if saved:
        result["saved_clip"] = saved
    return result


def check_status(run_dir: Path) -> dict:
    """有结果就返回分段。只有视频时也把文件信息交出去，页面可以先播放、拖动。"""
    result = load_result(run_dir)
    if result:
        result = dict(result)
        result["overlay"] = overlay_path(run_dir).is_file()
        result.setdefault("checked_at", (check_dir(run_dir) / "result.json").stat().st_mtime)
        return _attach_saved_clip(run_dir, result)
    try:
        video = locate_check_video(run_dir)
    except (FileNotFoundError, ValueError):
        return _attach_saved_clip(run_dir, {"ready": False, "overlay": False})
    media = {"kind": "video", "file": video.name, "fps": 0.0, "frame_count": 0, "width": 0, "height": 0}
    try:
        from .extract import probe_media

        probe = probe_media(video)
        media = {
            "kind": "video",
            "file": video.name,
            "fps": float(probe.get("fps") or 0),
            "frame_count": int(probe.get("frame_count") or 0),
            "width": int(probe.get("width") or 0),
            "height": int(probe.get("height") or 0),
        }
    except ValueError:
        pass
    return _attach_saved_clip(run_dir, {"ready": False, "overlay": False, "media": media})


def _sampled_indices(frame_count: int, fps: float, time_base_fps: float) -> list[int]:
    if fps <= 0:
        raise ValueError("这段视频读不到帧率")
    if time_base_fps <= 0:
        raise ValueError("这一轮没有时间基，不能按训练时的窗口来看")
    return resample_indices(frame_count, fps, time_base_fps)


def ensure_long_enough(frame_count: int, fps: float, window: int, window_seconds: float, time_base_fps: float) -> list[int]:
    """窗口是按这一轮的时间基数的。短于一个窗口就还没开始出类别。"""
    indices = _sampled_indices(frame_count, fps, time_base_fps)
    if len(indices) < window:
        seconds = frame_count / fps if fps else 0
        raise ValueError(
            f"这段只有 {seconds:.2f} 秒，比这一轮的窗口 {window_seconds:g} 秒短，凑不满。"
        )
    return indices


def require_same_models(settings: dict, contract: dict) -> None:
    """特征必须和这一轮训练时是同一套，否则权重吃进去的数对不上。"""
    if str(contract.get("feature_code_version") or "") != FEATURE_CODE_VERSION:
        raise ValueError("这一轮的特征版本和现在的代码不一致")
    parts = list(contract.get("part_class_names") or [])
    if list(settings.get("yolo_classes") or []) != parts:
        raise ValueError("现在的零件类别和这一轮不一致，验出来对不上")
    use_hands = hands_in_training(contract)
    if feature_size(len(parts), hands=use_hands) != int(contract.get("input_size") or 0):
        raise ValueError("特征长度和这一轮的权重对不上")
    yolo_sha = str(settings.get("yolo_sha256") or "")
    recorded_yolo = str(contract.get("yolo_sha256") or "")
    if yolo_sha and recorded_yolo and yolo_sha != recorded_yolo:
        raise ValueError("现在的零件模型和这一轮不是同一个文件")
    if use_hands and hands_in_training(settings) is not use_hands:
        raise ValueError("这一轮训练用了手部，当前开关是关的")
    hands_sha = str(settings.get("hands_sha256") or "")
    recorded_hands = str(contract.get("hands_sha256") or "")
    if use_hands and hands_sha and recorded_hands and hands_sha != recorded_hands:
        raise ValueError("现在的手部模型和这一轮不是同一个文件")
    current_conf = conf_threshold_of(settings)
    trained_conf = conf_threshold_of(contract)
    if abs(current_conf - trained_conf) > 1e-6:
        raise ValueError("现在的置信度阈值和这一轮不同")


def grow_complete_actions(length: int, min_frames: int, classify) -> list[dict]:
    """一个动作是从它的起点一直读到结束的整段。

    classify(start, end) 看的是这一整段，返回 (类别下标, 概率)。
    整段的类别变了，上一段就收到变之前的那一帧。短于最短样本的尾巴并进上一段，
    不单独算一个动作。
    """
    actions: list[dict] = []
    start = 0
    while start < length:
        if length - start < min_frames:
            if actions:
                actions[-1]["end"] = length
            break
        committed = None
        score = 0.0
        last = start
        end = start + min_frames
        while end <= length:
            index, prob = classify(start, end)
            if committed is None:
                committed = index
                score = prob
                last = end
            elif index == committed:
                last = end
                score = prob
            else:
                break
            end += 1
        if committed is None or last <= start:
            break
        actions.append({"index": int(committed), "start": start, "end": last, "prob": float(score)})
        start = last
    return actions


def _segments_from_complete_actions(
    gathered: np.ndarray,
    model,
    class_ids: list[str],
    device,
    min_frames: int,
    source_index: list[int],
    frame_count: int,
    fps: float,
    names: dict[str, str],
) -> tuple[None, list[dict]]:
    import torch

    def classify(start: int, end: int) -> tuple[int, float]:
        batch = torch.tensor(gathered[start:end][None], dtype=torch.float32, device=device)
        with torch.no_grad():
            probability = torch.softmax(model(batch), dim=1)[0]
        index = int(torch.argmax(probability).item())
        return index, float(probability[index].item())

    actions = grow_complete_actions(int(gathered.shape[0]), min_frames, classify)
    segments = []
    cursor = 0
    for action in actions:
        start_frame = max(int(source_index[action["start"]]), cursor)
        end_at = action["end"]
        end_frame = int(source_index[end_at]) if end_at < len(source_index) else frame_count
        if end_frame <= start_frame:
            end_frame = min(frame_count, start_frame + 1)
        class_id = class_ids[action["index"]]
        segments.append({
            "class_id": class_id,
            "name": names.get(class_id, class_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": start_frame / fps if fps else 0.0,
            "end_sec": end_frame / fps if fps else 0.0,
            "prob": action["prob"],
        })
        cursor = end_frame
    return None, segments


def slide_predictions(rows: np.ndarray, model, class_ids: list[str], window: int, device) -> list[dict | None]:
    """前 window-1 帧还在攒窗口，没有类别。凑满之后每一帧给一个类别，和工位逐帧滑动一致。"""
    count = int(rows.shape[0])
    labels: list[dict | None] = [None] * count
    if count < window or window <= 0:
        return labels
    import torch

    model.eval()
    chunk = 32
    with torch.no_grad():
        for offset in range(0, count - window + 1, chunk):
            stop = min(offset + chunk, count - window + 1)
            batch_np = np.stack([rows[index : index + window] for index in range(offset, stop)])
            batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            probabilities = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
            for row_index, probability in enumerate(probabilities):
                choice = int(np.argmax(probability))
                labels[offset + row_index + window - 1] = {
                    "class_id": class_ids[choice],
                    "prob": float(probability[choice]),
                }
    return labels


def build_segments(
    source_index: list[int],
    labels: list[dict | None],
    frame_count: int,
    fps: float,
    names: dict[str, str],
) -> tuple[dict | None, list[dict]]:
    """连续同一类合成一段。结束帧是下一段的开始，和标注里的区间一样，右端不算进去。"""
    warming_end = 0
    while warming_end < len(labels) and labels[warming_end] is None:
        warming_end += 1
    warming = None
    if warming_end:
        end_frame = int(source_index[warming_end]) if warming_end < len(source_index) else frame_count
        warming = {"start_frame": int(source_index[0]), "end_frame": end_frame}

    segments: list[dict] = []
    index = warming_end
    while index < len(labels):
        label = labels[index]
        if label is None:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] is not None and labels[end]["class_id"] == label["class_id"]:
            end += 1
        start_frame = int(source_index[index])
        end_frame = int(source_index[end]) if end < len(source_index) else frame_count
        probabilities = [float(labels[cursor]["prob"]) for cursor in range(index, end) if labels[cursor] is not None]
        class_id = str(label["class_id"])
        segments.append({
            "class_id": class_id,
            "name": names.get(class_id, class_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "prob": (sum(probabilities) / len(probabilities)) if probabilities else 0.0,
        })
        index = end

    cursor = warming["end_frame"] if warming else 0
    kept = []
    for segment in segments:
        start_frame = max(segment["start_frame"], cursor)
        end_frame = segment["end_frame"]
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        segment["start_frame"] = start_frame
        segment["end_frame"] = min(end_frame, frame_count) if frame_count else end_frame
        if segment["end_frame"] <= segment["start_frame"]:
            continue
        segment["start_sec"] = segment["start_frame"] / fps if fps else 0.0
        segment["end_sec"] = segment["end_frame"] / fps if fps else 0.0
        kept.append(segment)
        cursor = segment["end_frame"]
    if warming and frame_count:
        warming["end_frame"] = min(warming["end_frame"], frame_count)
        warming["start_sec"] = warming["start_frame"] / fps if fps else 0.0
        warming["end_sec"] = warming["end_frame"] / fps if fps else 0.0
    return warming, kept


def _class_names(annotations: dict, class_ids: list[str]) -> dict[str, str]:
    found = {str(item.get("id")): str(item.get("name") or item.get("id")) for item in annotations.get("classes") or []}
    found["idle"] = "空闲"
    return {class_id: found.get(class_id, class_id) for class_id in class_ids}


def _iter_frames(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"打不开视频 {path.name}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def _norm_poly(points, width: int, height: int) -> list[list[float]]:
    normalized = []
    for point in points:
        normalized.append([round(float(point[0]) / width, 4), round(float(point[1]) / height, 4)])
    return normalized


def frame_overlay(detections: list, hand_sets: list, width: int, height: int) -> dict | None:
    """一帧上的零件框和手部点，坐标收成 0 到 1，播放时再按画面放大。"""
    width = max(int(width), 1)
    height = max(int(height), 1)
    boxes = []
    for item in detections:
        poly = list(item.get("polygon") or [])
        if len(poly) < 3:
            box = item.get("box") or (0, 0, 0, 0)
            x1, y1, x2, y2 = [float(value) for value in box]
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        boxes.append({"label": str(item.get("label") or ""), "poly": _norm_poly(poly, width, height)})
    hands = []
    for hand in hand_sets:
        if len(hand) < 21:
            continue
        hands.append(_norm_poly(hand[:21], width, height))
    if not boxes and not hands:
        return None
    return {"boxes": boxes, "hands": hands}


def save_overlay(run_dir: Path, frames: list) -> None:
    import time

    target = overlay_path(run_dir)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"frames": frames}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    last_error = None
    for _ in range(30):
        try:
            temporary.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error


def _backend_settings_for(contract: dict, settings: dict, conf: float) -> dict:
    """验证视频时抽特征用的设置：手部开关跟着这一轮的契约走，不跟当前工程设置。

    以前只传 yolo_path 和阈值，hands_in_training 判断恒为假：带手部训练的一轮，
    验证时抽出来的特征少了手部那 126 维，送进模型直接报形状错误，叠加图上也没有手。
    """
    return {
        "yolo_path": settings.get("yolo_path") or "",
        "yolo_conf_threshold": conf,
        "hands_in_training": hands_in_training(contract),
    }


def _require_feature_width(rows: np.ndarray, contract: dict) -> None:
    expected = int(contract.get("input_size") or 0)
    actual = int(rows.shape[-1]) if rows.ndim == 2 else 0
    if expected and actual != expected:
        raise ValueError(
            f"抽出的特征是 {actual} 维，这一轮模型要 {expected} 维。"
            "多半是零件类别或手部开关和训练时不一样，请用同一套模型设置。"
        )


def _extract_rows(path: Path, settings: dict, part_names: list[str], conf: float, on_frame, should_stop, on_overlay=None) -> tuple[np.ndarray, list]:
    from .model_registry import live_backends, yolo_backend

    use_hands = hands_in_training(settings)
    if use_hands:
        detect, hands = live_backends(settings)
    else:
        detect = yolo_backend(settings)
        def hands(_frame):
            return []
        hands.close = lambda: None
    rows = []
    overlays = []
    frames = _iter_frames(path)
    try:
        for frame in frames:
            if should_stop is not None and should_stop():
                raise Stopped("已停止验证")
            height, width = frame.shape[:2]
            detected = detect(frame)
            pointed = hands(frame) if use_hands else []
            rows.append(build_feature_row(
                detected, pointed, width, height, part_names, conf, include_hands=use_hands,
            ))
            overlays.append(frame_overlay(detected, pointed, width, height))
            if on_overlay is not None and (len(rows) % 24 == 0):
                on_overlay(overlays)
            if on_frame is not None:
                on_frame(len(rows))
    finally:
        close_frames = getattr(frames, "close", None)
        if close_frames is not None:
            close_frames()
        close_hands = getattr(hands, "close", None)
        if close_hands is not None:
            close_hands()
    if not rows:
        raise ValueError(f"视频没有帧 {path.name}")
    return np.asarray(rows, dtype=np.float32), overlays


def frame_predictions(rows: np.ndarray, model, class_ids: list[str], device) -> list[dict]:
    """整段一次前向，每一帧一个类别。因果网络，这一帧只用到它和它之前的帧。"""
    import torch

    batch = torch.tensor(rows[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(batch, all_steps=True)[0]
        probability = torch.softmax(logits, dim=1)
        choice = torch.argmax(probability, dim=1)
    labels = []
    for index in range(int(choice.shape[0])):
        picked = int(choice[index])
        labels.append({"class_id": class_ids[picked], "prob": float(probability[index, picked])})
    return labels


def absorb_short_segments(segments: list[dict], min_frames: int, frame_count: int, fps: float) -> list[dict]:
    """短于一个窗口的碎段并回上一段，相邻的同一类再合成一段。"""
    if not segments:
        return []
    merged = [dict(segments[0])]
    for segment in segments[1:]:
        length = int(segment["end_frame"]) - int(segment["start_frame"])
        if length < min_frames:
            merged[-1]["end_frame"] = int(segment["end_frame"])
            continue
        merged.append(dict(segment))
    if len(merged) > 1 and int(merged[0]["end_frame"]) - int(merged[0]["start_frame"]) < min_frames:
        merged[1]["start_frame"] = int(merged[0]["start_frame"])
        merged.pop(0)
    coalesced = [merged[0]]
    for segment in merged[1:]:
        if segment["class_id"] == coalesced[-1]["class_id"]:
            coalesced[-1]["end_frame"] = int(segment["end_frame"])
            continue
        coalesced.append(segment)
    for segment in coalesced:
        segment["start_frame"] = max(0, int(segment["start_frame"]))
        segment["end_frame"] = min(frame_count, int(segment["end_frame"])) if frame_count else int(segment["end_frame"])
        segment["start_sec"] = segment["start_frame"] / fps if fps else 0.0
        segment["end_sec"] = segment["end_frame"] / fps if fps else 0.0
    return [segment for segment in coalesced if segment["end_frame"] > segment["start_frame"]]


def require_timeline_contract(contract: dict, settings: dict | None = None) -> None:
    """新契约缺字段、或和现在的模型对不上，就不能切段。"""
    required = (
        "train_unit", "decode_version", "network", "context_frames", "time_base_fps",
        "min_action_seconds", "feature_mean", "feature_std", "class_ids",
        "part_class_names", "required_parts", "input_size", "yolo_conf_threshold",
    )
    for key in required:
        if key not in contract or contract.get(key) in (None, ""):
            if key == "required_parts" and isinstance(contract.get(key), dict):
                continue
            raise ValueError(f"这份契约缺 {key}，不能按整条切段")
    if contract.get("train_unit") != "timeline":
        raise ValueError("这份契约不是按整条连续推理训练的")
    if not contract.get("feature_std"):
        raise ValueError("这份契约缺 feature_std，不能按整条切段")
    if not contract.get("feature_mean"):
        raise ValueError("这份契约缺 feature_mean，不能按整条切段")
    mean = contract.get("feature_mean") or []
    std = contract.get("feature_std") or []
    if len(mean) != len(std) or len(mean) != int(contract.get("input_size") or 0):
        raise ValueError("feature_mean 的长度和特征长度对不上")
    if settings is not None:
        require_same_models(settings, contract)


def decode_timeline(rows: np.ndarray, fps: float, model, contract: dict, part_names: list[str], names: dict[str, str]):
    """整条片子一次因果前向，再收成段。训练时验证、结果页、预标注都走这里。"""
    from .service import resample_indices

    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("没有可以切段的特征")
    base = float(contract.get("time_base_fps") or 0.0)
    indices = resample_indices(int(array.shape[0]), float(fps), base) if fps > 0 and base > 0 else list(range(array.shape[0]))
    gathered = array[np.asarray(indices, dtype=np.int64)]
    mean = np.asarray(contract.get("feature_mean") or [], dtype=np.float32)
    std = np.asarray(contract.get("feature_std") or [], dtype=np.float32)
    if mean.shape[0] == gathered.shape[1] and std.shape[0] == gathered.shape[1]:
        gathered = (gathered - mean) / safe_feature_std(std)
    class_ids = [str(item) for item in contract.get("class_ids") or []]
    device = next(model.parameters()).device
    labels = frame_predictions(gathered, model, class_ids, device)
    for label in labels:
        label["name"] = names.get(label["class_id"], label["class_id"])
    _warming, segments = build_segments(indices, labels, int(array.shape[0]), float(fps), names)
    min_frames = max(1, int(round(float(contract.get("min_action_seconds") or contract.get("window_seconds") or 1.92) * float(fps)))) if fps else 1
    segments = absorb_short_segments(segments, min_frames, int(array.shape[0]), float(fps))
    apply_part_gate(segments, array, list(part_names), contract.get("required_parts"))
    frame_index = frame_index_from_segments(len(labels), indices, segments, class_ids)
    return segments, frame_index


def frame_index_from_segments(
    length: int,
    indices: list[int],
    segments: list[dict],
    class_ids: list[str],
) -> np.ndarray:
    """逐帧分数跟最终切出来的段用同一套类别。短段被并掉之后，原来的碎片不再参与打分。"""
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    idle = index.get("idle", 0)
    frame_index = np.full(int(length), idle, dtype=np.int64)
    for step, source_frame in enumerate(indices):
        source_frame = int(source_frame)
        for segment in segments:
            if int(segment["start_frame"]) <= source_frame < int(segment["end_frame"]):
                frame_index[step] = index.get(str(segment["class_id"]), idle)
                break
    return frame_index


def segments_for_contract(
    gathered: np.ndarray,
    model,
    class_ids: list[str],
    device,
    contract: dict,
    indices: list[int],
    frame_count: int,
    fps: float,
    names: dict[str, str],
) -> tuple[dict | None, list[dict]]:
    """旧模型按窗口或逐帧收段。整条连续训练的契约必须走 decode_timeline，这里不再换算第二次。"""
    if contract.get("train_unit") == "timeline":
        raise ValueError("整条连续推理要走 decode_timeline")
    window = int(contract.get("window") or 1)
    if contract.get("label_scope") == "frame":
        labels = frame_predictions(gathered, model, class_ids, device)
        for label in labels:
            label["name"] = names.get(label["class_id"], label["class_id"])
        _warming, segments = build_segments(indices, labels, frame_count, fps, names)
        min_frames = max(1, int(round(float(contract.get("window_seconds") or 1.92) * fps))) if fps else window
        return None, absorb_short_segments(segments, min_frames, frame_count, fps)
    if contract.get("sample_unit") == "segment":
        return _segments_from_complete_actions(
            gathered, model, class_ids, device, window, indices, frame_count, fps, names
        )
    labels = slide_predictions(gathered, model, class_ids, window, device)
    for label in labels:
        if label is not None:
            label["name"] = names.get(label["class_id"], label["class_id"])
    return build_segments(indices, labels, frame_count, fps, names)


def _load_model(contract: dict, weights: Path, device):
    import torch

    class_ids = [str(item) for item in contract.get("class_ids") or []]
    if not class_ids:
        raise ValueError("这一轮没有类别")
    network = str(contract.get("network") or "lstm")
    if network != "lstm":
        # MS-TCN 已从工作台移除（产线也加载不了）。旧轮次给出明确原因，不再尝试加载。
        raise ValueError("这一轮是用 MS-TCN 训练的，工作台已不再支持，请用 LSTM 重新训练")
    model = ActionLSTM(
        int(contract["input_size"]),
        len(class_ids),
        hidden_size=int(contract.get("hidden_size") or 128),
        num_layers=int(contract.get("num_layers") or 2),
        dropout=float(contract.get("dropout") or 0.0),
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, class_ids


def run_check(run_dir: Path, settings: dict, annotations: dict, on_progress=None, should_stop=None) -> dict:
    """抽特征后按工位的方式逐帧滑动，再把连续同一类收成一段。"""
    contract_path = run_dir / "contract.json"
    weights = run_dir / "best_lstm.pt"
    if not contract_path.is_file() or not weights.is_file():
        raise FileNotFoundError("这一轮没有可用的权重")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require_same_models(settings, contract)
    video = locate_check_video(run_dir)
    from .extract import probe_media

    probe = probe_media(video)
    if probe.get("kind") != "video":
        raise ValueError("请上传视频。这一步要看一整段画面，一张图凑不满窗口。")
    window = int(contract.get("window") or 0)
    window_seconds = float(contract.get("window_seconds") or 0)
    time_base_fps = float(contract.get("time_base_fps") or 0)
    fps = float(probe.get("fps") or 0)
    ensure_long_enough(int(probe["frame_count"]), fps, window, window_seconds, time_base_fps)

    part_names = list(contract.get("part_class_names") or [])
    conf = float(contract.get("yolo_conf_threshold") or 0)
    backend_settings = _backend_settings_for(contract, settings, conf)
    total_hint = int(probe["frame_count"])

    def publish(done: int) -> None:
        if on_progress is None:
            return
        if done == total_hint or done % 8 == 0:
            on_progress(done, total_hint)

    rows, overlays = _extract_rows(
        video, backend_settings, part_names, conf, publish, should_stop,
        on_overlay=lambda frames: save_overlay(run_dir, frames),
    )
    save_overlay(run_dir, overlays)
    _require_feature_width(rows, contract)
    device, _info = resolve_device()
    model, class_ids = _load_model(contract, weights, device)
    names = _class_names(annotations, class_ids)
    if contract.get("train_unit") == "timeline":
        require_timeline_contract(contract, settings)
        warming, segments = None, decode_timeline(rows, fps, model, contract, part_names, names)[0]
    else:
        indices = ensure_long_enough(len(rows), fps, window, window_seconds, time_base_fps)
        gathered = rows[np.asarray(indices, dtype=np.int64)]
        warming, segments = segments_for_contract(
            gathered, model, class_ids, device, contract, indices, len(rows), fps, names
        )
        apply_part_gate(segments, rows, list(contract.get("part_class_names") or []), contract.get("required_parts"))
    result = {
        "ready": True,
        "media": {
            "kind": "video",
            "file": video.name,
            "fps": fps,
            "frame_count": len(rows),
            "width": int(probe.get("width") or 0),
            "height": int(probe.get("height") or 0),
        },
        "window": window,
        "window_seconds": window_seconds,
        "time_base_fps": time_base_fps,
        "classes": [{"id": class_id, "name": names.get(class_id, class_id)} for class_id in class_ids],
        "warming": warming,
        "segments": segments,
        "checked_at": time.time(),
    }
    target = check_dir(run_dir) / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if on_progress is not None:
        on_progress(total_hint, total_hint)
    return result


def fill_overlay(run_dir: Path, settings: dict, on_progress=None, should_stop=None) -> dict:
    """已经有验证结果、但没留下框和手的，补画这一层。不再重跑 LSTM。"""
    contract_path = run_dir / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError("这一轮没有可用的权重")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require_same_models(settings, contract)
    video = locate_check_video(run_dir)
    from .extract import probe_media

    probe = probe_media(video)
    part_names = list(contract.get("part_class_names") or [])
    conf = float(contract.get("yolo_conf_threshold") or 0)
    total_hint = int(probe.get("frame_count") or 0)

    def publish(done: int) -> None:
        if on_progress is None:
            return
        if done == total_hint or done % 8 == 0:
            on_progress(done, total_hint)

    _rows, overlays = _extract_rows(
        video,
        _backend_settings_for(contract, settings, conf),
        part_names,
        conf,
        publish,
        should_stop,
        on_overlay=lambda frames: save_overlay(run_dir, frames),
    )
    save_overlay(run_dir, overlays)
    if on_progress is not None:
        on_progress(total_hint, total_hint)
    return {"frames": len(overlays)}


def _subtract_cuts(spans: list[tuple[int, int]], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = list(spans)
    for cut_start, cut_end in cuts:
        nxt: list[tuple[int, int]] = []
        for start, end in pieces:
            if end <= cut_start or start >= cut_end:
                nxt.append((start, end))
                continue
            if start < cut_start:
                nxt.append((start, cut_start))
            if cut_end < end:
                nxt.append((cut_end, end))
        pieces = nxt
    return [(start, end) for start, end in pieces if end - start >= 1]


def draft_annotations(segments: list[dict], excluded: list[dict], known_ids: set[str]) -> list[dict]:
    """只留下当前工程还存在的动作段。空闲不写。和「不训练」重叠的部分挖掉。"""
    cuts = [(int(item["start"]), int(item["end"])) for item in excluded]
    drafted = []
    for index, segment in enumerate(segments):
        class_id = str(segment.get("class_id") or "")
        if class_id == "idle" or class_id not in known_ids:
            continue
        pieces = _subtract_cuts([(int(segment["start_frame"]), int(segment["end_frame"]))], cuts)
        for start, end in pieces:
            item = {"id": f"p{index}-{start}", "class_id": class_id, "start": start, "end": end}
            if segment.get("part_check") == "fail":
                item["missing_parts"] = list(segment.get("missing_parts") or [])
            drafted.append(item)
    return drafted


def indices_for_model(raw: dict, contract: dict) -> list[int]:
    """当前片子对齐到这一轮训练时的时间基。工程里其他片子的帧率不参与。"""
    from .service import resample_indices

    frame_count = int(raw.get("frame_count") or 0)
    if frame_count <= 0:
        return []
    fps = float(raw.get("fps") or 0.0)
    base = float(contract.get("time_base_fps") or 0.0)
    if fps <= 0 or base <= 0:
        return list(range(frame_count))
    return resample_indices(frame_count, fps, base)


def prelabel_clip(
    folder: Path,
    clip_id: str,
    run_dir: Path,
    settings: dict,
    annotations: dict,
    on_progress=None,
    should_stop=None,
) -> dict:
    """用某一轮的权重给当前片子画段。段在页面上，保存之前不写标注文件。"""
    from . import feature_store

    contract_path = run_dir / "contract.json"
    weights = run_dir / "best_lstm.pt"
    if not contract_path.is_file() or not weights.is_file():
        raise FileNotFoundError("这一轮没有可用的权重")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require_same_models(settings, contract)
    raw = next((item for item in annotations.get("clips") or [] if item.get("id") == clip_id), None)
    if raw is None:
        raise FileNotFoundError("没有这条片子")
    summary = feature_store.summary(folder, annotations, settings)
    info = next((item for item in summary["clips"] if item["clip_id"] == clip_id), None)
    if info is None:
        raise FileNotFoundError("没有这条片子")
    if info.get("note"):
        raise FileNotFoundError(str(info["note"]))
    extracted = not info.get("fresh")
    if extracted:
        def on_frame(done: int, total: int) -> None:
            if on_progress is not None:
                on_progress(done, total)

        feature_store.extract_project(
            folder,
            annotations,
            settings,
            should_stop=should_stop,
            clip_ids=[clip_id],
            on_frame=on_frame,
        )
    rows = np.load(folder / "features" / f"{clip_id}.npy")
    if int(rows.shape[1]) != int(contract.get("input_size") or 0):
        raise ValueError("特征长度和这一轮的权重对不上")
    device, _info = resolve_device()
    model, class_ids = _load_model(contract, weights, device)
    names = _class_names(annotations, class_ids)
    window = int(contract.get("window") or 1)
    fps = float(raw.get("fps") or 0)
    if contract.get("train_unit") == "timeline":
        require_timeline_contract(contract, settings)
        _warming, segments = None, decode_timeline(rows, fps, model, contract, list(contract.get("part_class_names") or []), names)[0]
    else:
        indices = indices_for_model(raw, contract)
        if not indices:
            raise ValueError("这条片子没有可用的帧")
        gathered = rows[np.asarray(indices, dtype=np.int64)]
        _warming, segments = segments_for_contract(
            gathered, model, class_ids, device, contract, indices, len(rows), fps, names
        )
        apply_part_gate(segments, rows, list(contract.get("part_class_names") or []), contract.get("required_parts"))
    known = {str(item.get("id")) for item in annotations.get("classes") or []}
    drafted = draft_annotations(segments, list(raw.get("excluded") or []), known)
    if on_progress is not None:
        on_progress(len(rows), len(rows))
    return {"segments": drafted, "extracted": extracted}


def _clip_ids(entries) -> set[str]:
    found: set[str] = set()
    for item in entries or []:
        if isinstance(item, (list, tuple)) and item:
            found.add(str(item[0]))
        elif item:
            found.add(str(item))
    return found


def _role_text(run_dir: Path, clip_id: str) -> str:
    split_path = run_dir / "split.json"
    if not split_path.is_file():
        return "这条片子没参与训练，分数可信"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = _clip_ids(split.get("train_clips") or split.get("train"))
    val_ids = _clip_ids(split.get("val_clips") or split.get("val"))
    if clip_id in val_ids:
        return "这条片子是这一轮的验证片"
    if clip_id in train_ids:
        return "这条片子参与过训练，分数偏乐观"
    return "这条片子没参与训练，分数可信"


def check_project_clip(
    folder: Path,
    clip_id: str,
    run_dir: Path,
    settings: dict,
    annotations: dict,
    on_progress=None,
    should_stop=None,
) -> dict:
    """用工程里标过的片子验证。不复制视频，不改标注。"""
    from .dataset import frame_labels
    from .metrics import edit_score, score_timeline, segmental_f1, segments_from_labels
    from .service import annotations_to_clips, scale_clip_to_time_base
    from . import feature_store

    contract_path = run_dir / "contract.json"
    weights = run_dir / "best_lstm.pt"
    if not contract_path.is_file() or not weights.is_file():
        raise FileNotFoundError("这一轮没有可用的权重")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require_same_models(settings, contract) if contract.get("train_unit") != "timeline" else require_timeline_contract(contract, settings)
    raw = next((item for item in annotations.get("clips") or [] if item.get("id") == clip_id), None)
    if raw is None:
        raise FileNotFoundError("没有这条片子")
    summary = feature_store.summary(folder, annotations, settings)
    info = next((item for item in summary["clips"] if item["clip_id"] == clip_id), None)
    if info is None:
        raise FileNotFoundError("没有这条片子")
    if info.get("note"):
        raise FileNotFoundError(str(info["note"]))
    if not info.get("fresh"):
        feature_store.extract_project(
            folder, annotations, settings, should_stop=should_stop, clip_ids=[clip_id], on_frame=on_progress,
        )
    rows = np.load(folder / "features" / f"{clip_id}.npy")
    if int(rows.shape[1]) != int(contract.get("input_size") or 0):
        raise ValueError("特征长度和这一轮的权重对不上")
    originals = annotations_to_clips({"classes": annotations.get("classes") or [], "clips": [raw]}, settings)
    if not originals:
        raise ValueError("这条片子没有帧")
    clip = originals[0]
    base = float(contract.get("time_base_fps") or clip.fps or 0.0)
    if base <= 0:
        raise ValueError("这一轮没有时间基")
    scaled, indices = scale_clip_to_time_base(clip, base)
    rows = np.asarray(rows, dtype=np.float32)
    # 实际解码帧数可能比标注里的估计值少几帧，越界的索引落到最后一帧，不报 IndexError。
    safe = np.minimum(np.asarray(indices, dtype=np.int64), max(rows.shape[0] - 1, 0))
    gathered = rows[safe]
    class_ids = [str(item) for item in contract.get("class_ids") or []]
    names = _class_names(annotations, class_ids)
    device, _info = resolve_device()
    model, class_ids = _load_model(contract, weights, device)
    part_names = list(contract.get("part_class_names") or [])
    if contract.get("train_unit") == "timeline":
        # gathered 已经按时间基取过帧，这里必须传时间基帧率。scale_clip_to_time_base 在帧数没变时
        # 原样返回片子（fps 仍是源帧率，比如 25.02 对 25），传 scaled.fps 会让 decode_timeline 再重采一遍。
        segments, frame_index = decode_timeline(gathered, base, model, contract, part_names, names)
    else:
        segments, frame_index = _legacy_scaled_decode(
            gathered, indices, scaled, model, class_ids, device, contract, names
        )
    labels = frame_labels(scaled, class_ids)
    if int(frame_index.shape[0]) != int(labels.shape[0]):
        raise ValueError("切段帧数和时间基对不上")
    metrics = score_timeline(labels, frame_index, class_ids)
    truth_scaled = segments_from_labels(labels, class_ids)
    pred_scaled = [(segment["class_id"], int(segment["start_frame"]), int(segment["end_frame"])) for segment in segments]
    metrics["f1_10"] = segmental_f1(truth_scaled, pred_scaled, 0.10)
    metrics["f1_25"] = segmental_f1(truth_scaled, pred_scaled, 0.25)
    metrics["f1_50"] = segmental_f1(truth_scaled, pred_scaled, 0.50)
    metrics["edit"] = edit_score(truth_scaled, pred_scaled)
    metrics["pred_segments"] = sum(item[0] != "idle" for item in pred_scaled)
    metrics["true_segments"] = sum(item[0] != "idle" for item in truth_scaled)
    shown = _segments_on_source(segments, indices, clip.frame_count, clip.fps, names)
    truth_shown = _segments_on_source(
        [{"class_id": class_id, "name": names.get(class_id, class_id), "start_frame": start, "end_frame": end, "prob": 1.0}
         for class_id, start, end in truth_scaled],
        indices, clip.frame_count, clip.fps, names,
    )
    result = {
        "ready": True,
        "project_clip": True,
        "clip_id": clip_id,
        "checked_at": time.time(),
        "role": _role_text(run_dir, clip_id),
        "media": {
            "kind": "video",
            "file": str(raw.get("file") or clip_id),
            "fps": float(clip.fps or 0),
            "frame_count": int(clip.frame_count),
            "project_file": True,
        },
        "window": int(contract.get("window") or 0),
        "window_seconds": float(contract.get("window_seconds") or 0),
        "time_base_fps": base,
        "classes": [{"id": class_id, "name": names.get(class_id, class_id)} for class_id in class_ids],
        "segments": shown,
        "truth_segments": truth_shown,
        "scores": metrics,
        "legacy": contract.get("train_unit") != "timeline",
    }
    target = check_dir(run_dir) / "clips" / clip_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _legacy_scaled_decode(gathered, indices, scaled, model, class_ids, device, contract, names):
    _warming, segments = segments_for_contract(
        gathered, model, class_ids, device, contract, list(range(len(indices))),
        scaled.frame_count, float(scaled.fps or 0), names,
    )
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    frame_index = np.full(scaled.frame_count, index.get("idle", 0), dtype=np.int64)
    for segment in segments:
        class_index = index.get(segment["class_id"], 0)
        frame_index[int(segment["start_frame"]):int(segment["end_frame"])] = class_index
    return segments, frame_index


def _segments_on_source(segments, indices, frame_count, fps, names):
    shown = []
    total = len(indices)
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        src_start = int(indices[min(max(start, 0), total - 1)]) if total else 0
        src_end = int(indices[end]) if total and end < total else frame_count
        if src_end <= src_start:
            src_end = min(frame_count, src_start + 1)
        class_id = str(segment["class_id"])
        item = {
            "class_id": class_id,
            "name": names.get(class_id, segment.get("name") or class_id),
            "start_frame": src_start,
            "end_frame": src_end,
            "start_sec": src_start / fps if fps else 0.0,
            "end_sec": src_end / fps if fps else 0.0,
            "prob": float(segment.get("prob") or 0.0),
        }
        if segment.get("part_check"):
            item["part_check"] = segment["part_check"]
            item["missing_parts"] = list(segment.get("missing_parts") or [])
        shown.append(item)
    return shown

