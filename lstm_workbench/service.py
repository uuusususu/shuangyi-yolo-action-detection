"""从标注生成窗口，并调用训练写入 run 目录。

时间基从片子帧率推导，不是某个写死的常量。窗口按秒定义，帧数由时间基换算 ——
同一台相机采集与推理时，重采样是恒等操作，一帧不丢也不重复。
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import feature_store, settings as settings_module
from .dataset import (
    check_split,
    frame_labels,
    holdout_split,
    propose_split,
    split_stats,
)
from .project_store import ProjectStore
from .timeline import ClipAnnotation, Segment, WINDOW_SECONDS, find_overlaps, window_frames
from .model import mstcn_layers_for, receptive_field
from .part_gate import required_parts_from_labeled_rows
from .trainer import train_timeline

SAMPLE_SHORTFALL = 10


def write_progress(folder: Path, payload: dict) -> None:
    target = folder / "progress.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _replace_retry(temporary, target)


def _replace_retry(temporary: Path, target: Path) -> None:
    """Windows 上进度文件正被页面读取时，替换会报拒绝访问。多试几次。"""
    last_error: PermissionError | None = None
    for _ in range(30):
        try:
            temporary.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error


def resample_indices(frame_count: int, source_fps: float, target_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("帧率必须大于 0")
    if abs(source_fps - target_fps) < 1e-3:
        return list(range(frame_count))
    duration = frame_count / source_fps
    target_count = max(1, int(round(duration * target_fps)))
    return [min(frame_count - 1, int(index * source_fps / target_fps)) for index in range(target_count)]


def clip_signature(clips: list[ClipAnnotation]) -> str:
    """片子的身份与帧率签名。片子增删、换素材或改了裁剪之后，确认过的分割就作废。"""
    parts = []
    for clip in sorted(clips, key=lambda item: item.clip_id):
        piece = f"{clip.clip_id}:{clip.frame_count}:{clip.fps:g}"
        cuts = sorted(clip.excluded, key=lambda item: (item.start, item.end, item.id))
        if cuts:
            piece += ":cut=" + ",".join(f"{item.start}-{item.end}" for item in cuts)
        parts.append(piece)
    return "|".join(parts)


def _scale_spans(segments: list[Segment], ratio: float) -> list[Segment]:
    scaled = []
    for segment in segments:
        start = int(round(segment.start * ratio))
        end = int(round(segment.end * ratio))
        if end > start:
            scaled.append(Segment(segment.id, segment.class_id, start, end))
    return scaled


def time_base(clips: list[ClipAnnotation], override: float = 0.0) -> float:
    """参加训练的片子共用一个时间基。默认取最慢的帧率，快的片子按秒抽帧。

    同一路相机读出来的帧率差零点零几，取中位。手动设的时间基不能比任何一条片子快，
    否则只能靠重复帧去补，动作的时间会被拉长。
    """
    rated = [clip for clip in clips if clip.fps > 0]
    if not rated:
        raise ValueError("所有片子都读不到帧率（图片序列没有帧率），请在模型页显式指定时间基帧率")
    if override and override > 0:
        slow = min(rated, key=lambda clip: (clip.fps, clip.clip_id))
        if float(override) - slow.fps > 0.05:
            raise ValueError(
                f"{slow.clip_id} 是 {slow.fps:g} fps，时间基设成了 {float(override):g} fps，比它快，不能用重复帧补齐"
            )
        return float(override)
    values = sorted({round(clip.fps, 6) for clip in rated})
    if max(values) - min(values) <= 0.05:
        return float(values[len(values) // 2])
    return float(min(values))


def scale_clip_to_time_base(clip: ClipAnnotation, time_base_fps: float) -> tuple[ClipAnnotation, list[int]]:
    """换算到训练时间基。帧率相同时 indices 就是 range(n)，等价于什么都没做。

    帧率为 0 的片子（图片序列）没有可换算的时间，按已经落在时间基上处理。
    """
    if clip.fps <= 0:
        return clip, list(range(clip.frame_count))
    indices = resample_indices(clip.frame_count, clip.fps, time_base_fps)
    if len(indices) == clip.frame_count:
        return clip, indices
    ratio = len(indices) / max(clip.frame_count, 1)
    scaled = _scale_spans(clip.segments, ratio)
    excluded = _scale_spans(clip.excluded, ratio)
    return (
        ClipAnnotation(clip.clip_id, len(indices), time_base_fps, clip.width, clip.height, scaled, excluded),
        indices,
    )


# 标注里没写分辨率时用它兜底。特征是按每一帧自己的宽高归一化的，这个值不会影响
# 任何特征数值，只在素材元数据缺失时让管线还能跑完。
FALLBACK_FRAME = (1920, 1080)


@dataclass
class PreparedClips:
    """标注换算到训练时间基之后的结果。划分、体检、训练都吃这一份，口径只此一处。

    窗口帧数是按时间基算的，所以「段够不够长」「哪些是空闲段」这些判断也必须在
    时间基上做。以前确认划分时拿没换算的片子、训练时拿换算后的片子，两边口径不同：
    时间基覆盖成和片子帧率不同的值时，确认过的划分永远对不上，训练被拒。

    signature 永远取自原始片子 —— 它回答的是「素材变没变」，不是「换算成了什么」。
    """

    base: float
    window: int
    window_seconds: float
    originals: list[ClipAnnotation]
    clips: list[ClipAnnotation]
    indices: dict[str, list[int]]
    signature: str


def effective_window_seconds(settings: dict) -> float:
    """窗口秒数。设置里的值才算数，缺了才用默认。"""
    value = settings.get("window_seconds")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return WINDOW_SECONDS
    return seconds if seconds > 0 else WINDOW_SECONDS


def clips_for_training(payload: dict, settings: dict) -> dict:
    """没勾上的片子不进划分、时间基和训练。"""
    excluded = {str(item) for item in (settings.get("train_excluded") or [])}
    if not excluded:
        return payload
    clips = [clip for clip in (payload.get("clips") or []) if str(clip.get("id")) not in excluded]
    return {**payload, "clips": clips}


def prepare_clips(payload: dict, settings: dict) -> PreparedClips:
    full = list(payload.get("clips") or [])
    payload = clips_for_training(payload, settings)
    if full and not payload.get("clips"):
        raise ValueError("还没有勾选参加训练的片子")
    originals = annotations_to_clips(payload, settings)
    base = time_base(originals, float(settings.get("time_base_fps") or 0.0))
    seconds = effective_window_seconds(settings)
    scaled: list[ClipAnnotation] = []
    indices: dict[str, list[int]] = {}
    for clip in originals:
        converted, picked = scale_clip_to_time_base(clip, base)
        scaled.append(converted)
        indices[clip.clip_id] = picked
    return PreparedClips(
        base=base,
        window=window_frames(base, seconds),
        window_seconds=seconds,
        originals=originals,
        clips=scaled,
        indices=indices,
        signature=clip_signature(originals),
    )


def annotations_to_clips(payload: dict, settings: dict) -> list[ClipAnnotation]:
    clips = []
    for raw in payload.get("clips", []):
        segments = [
            Segment(item["id"], item["class_id"], int(item["start"]), int(item["end"]))
            for item in raw.get("segments", [])
        ]
        excluded = [
            Segment(item["id"], "cut", int(item["start"]), int(item["end"]))
            for item in raw.get("excluded") or []
        ]
        clips.append(
            ClipAnnotation(
                clip_id=raw["id"],
                frame_count=int(raw["frame_count"]),
                fps=float(raw.get("fps") or 0.0),
                width=int(raw.get("width") or FALLBACK_FRAME[0]),
                height=int(raw.get("height") or FALLBACK_FRAME[1]),
                segments=segments,
                excluded=excluded,
            )
        )
    return clips


def resolve_split(
    clips: list[ClipAnnotation],
    window: int,
    settings: dict,
    signature: str | None = None,
):
    """优先用确认过的划分；确认结果对不上当前片子集合就作废并重新提案。

    signature 是确认时记下的那种签名（取自原始片子）。clips 已经换算过时必须显式传，
    否则拿换算后的帧数和帧率去比，永远对不上。不传就按 clips 自己算（没换算的老调用）。
    """
    if str(settings.get("split_mode") or "segment") == "clip":
        split = holdout_split(clips, window, list(settings.get("val_clip_ids") or []))
        return split, split.ok
    seed = int(settings.get("split_seed") or 7)
    val_windows = int(settings.get("split_val_count") or 0) or None
    train = [tuple(item) for item in (settings.get("split_train") or [])]
    val = [tuple(item) for item in (settings.get("split_val") or [])]
    confirmed_for = str(settings.get("split_for") or "")
    current_signature = signature if signature is not None else clip_signature(clips)
    if train and val and confirmed_for == current_signature:
        try:
            checked = check_split(clips, window, train, val)
            if checked.ok:
                return checked, True
        except ValueError:
            pass
    return propose_split(clips, window, seed, val_windows), False


def annotation_blockers(payload: dict, settings: dict) -> list[str]:
    scoped = clips_for_training(payload, settings)
    originals = annotations_to_clips(scoped, settings)
    reasons: list[str] = []
    if payload.get("clips") and not originals:
        return ["还没有勾选参加训练的片子"]
    if not originals:
        return ["还没有片子，先放入素材"]
    for clip in originals:
        if find_overlaps(clip.segments):
            reasons.append(f"{clip.clip_id} 有交叠，不能训练")
    if reasons:
        return reasons
    try:
        prepared = prepare_clips(payload, settings)
    except ValueError as exc:
        return [str(exc)]
    split, _confirmed = resolve_split(prepared.clips, prepared.window, settings, prepared.signature)
    if not split.ok:
        reasons.append(split.blocked_reason)
    return reasons


def annotation_warnings(payload: dict, settings: dict) -> list[str]:
    """样本不足只警告，不挡训练。"""
    try:
        prepared = prepare_clips(payload, settings)
    except ValueError:
        return []
    clips, window = prepared.clips, prepared.window
    counts: dict[str, int] = {}
    for clip in clips:
        for segment in clip.trainable_segments(window):
            if segment.class_id == "idle":
                continue
            counts[segment.class_id] = counts.get(segment.class_id, 0) + 1
    names = {"idle": "空闲"}
    for item in payload.get("classes", []):
        names[item["id"]] = item["name"]
    return [
        f"{names.get(class_id, class_id)} 只有 {count} 段，少于 {SAMPLE_SHORTFALL} 段，样本不足"
        for class_id, count in sorted(counts.items())
        if count < SAMPLE_SHORTFALL
    ]


def _unlabeled_tails(clips: list[ClipAnnotation]) -> list[dict]:
    """片尾超过 30 秒既没涂也没剪掉，提醒会当空闲学。不拦训练。"""
    from .dataset import unlabeled_tail_frames

    tails = []
    for clip in clips:
        if clip.fps <= 0:
            continue
        seconds = unlabeled_tail_frames(clip) / clip.fps
        if seconds > 30:
            tails.append({"clip_id": clip.clip_id, "seconds": round(seconds, 1)})
    return tails


def split_overview(store: ProjectStore, project_id: str, settings: dict) -> dict:
    """给界面用的分割视图：确认过的（或提案）、两边统计、是否同源。"""
    payload = store.load_annotations(project_id)
    mode = str(settings.get("split_mode") or "segment")
    if not annotations_to_clips(payload, settings):
        return {"ok": False, "reason": "还没有片子", "same_source": False, "mode": mode}
    try:
        prepared = prepare_clips(payload, settings)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc), "same_source": False, "mode": mode}
    clips, window = prepared.clips, prepared.window
    split, confirmed = resolve_split(clips, window, settings, prepared.signature)
    out = {
        "ok": split.ok,
        "reason": split.blocked_reason,
        "confirmed": confirmed,
        "time_base_fps": prepared.base,
        "window": window,
        "window_seconds": prepared.window_seconds,
        "mode": "clip",
        "tails": _unlabeled_tails(prepared.originals),
    }
    if split.ok:
        out.update(split_stats(clips, window, split))
        out["train_entries"] = [list(item) for item in split.train]
        out["val_entries"] = [list(item) for item in split.val]
    return out


def start_training(
    store: ProjectStore,
    project_id: str,
    feature_arrays: dict[str, np.ndarray],
    should_stop=None,
    display_name: str = "",
) -> dict:
    folder = store.open_dir(project_id)
    settings = settings_module.clean({}, store.load_settings(project_id))
    payload = store.load_annotations(project_id)
    blockers = annotation_blockers(payload, settings)
    if blockers:
        raise ValueError(blockers[0])
    prepared = prepare_clips(payload, settings)
    base, window, clips = prepared.base, prepared.window, prepared.clips
    for clip in prepared.originals:
        if find_overlaps(clip.segments):
            raise ValueError(f"{clip.clip_id} 有交叠，不能训练")
        source = feature_arrays[clip.clip_id]
        feature_arrays[clip.clip_id] = np.asarray(source)[prepared.indices[clip.clip_id]]
    split, confirmed = resolve_split(clips, window, settings, prepared.signature)
    if not split.ok:
        raise ValueError(split.blocked_reason)
    if not confirmed:
        raise ValueError("验证集划分还没确认 —— 训练页先把一条片子设成只做验证")

    class_ids = ["idle"] + [item["id"] for item in payload.get("classes", [])]
    part_names = list(settings.get("yolo_classes") or [])
    val_ids = list(split.val_clip_ids)
    train_ids = [clip.clip_id for clip in clips if clip.clip_id not in set(val_ids)]
    by_id = {clip.clip_id: clip for clip in clips}

    def rows_and_labels(clip_id: str):
        clip = by_id[clip_id]
        rows = np.asarray(feature_arrays[clip_id], dtype=np.float32)
        if rows.shape[0] < clip.frame_count:
            raise ValueError(f"{clip_id} 特征帧数不足")
        return rows[: clip.frame_count], frame_labels(clip, class_ids)

    train_rows, train_label_rows = [], []
    for clip_id in train_ids:
        rows, labels = rows_and_labels(clip_id)
        train_rows.append(rows)
        train_label_rows.append(labels)
    val_rows, val_label_rows = [], []
    for clip_id in val_ids:
        rows, labels = rows_and_labels(clip_id)
        val_rows.append(rows)
        val_label_rows.append(labels)
    if not train_rows or not val_rows:
        raise ValueError("没有可用的训练或验证片子")

    def present(label_rows: list[np.ndarray]) -> set[str]:
        found: set[str] = set()
        for labels in label_rows:
            for value in np.unique(labels):
                if int(value) < 0:
                    continue
                name = class_ids[int(value)]
                if name != "idle":
                    found.add(name)
        return found

    for side, found in (("训练", present(train_label_rows)), ("验证", present(val_label_rows))):
        missing = [class_id for class_id in class_ids if class_id != "idle" and class_id not in found]
        if missing:
            raise ValueError(f"类别 {missing[0]} 没有进入{side}集，不能同时进入训练和验证")
    required_parts_out = required_parts_from_labeled_rows(
        list(zip(train_rows, train_label_rows)),
        class_ids,
        part_names,
    )
    names = {"idle": "空闲"}
    for item in payload.get("classes", []):
        names[item["id"]] = item["name"]
    network = str(settings.get("network") or "lstm")
    budget = max(1, int(float(settings.get("context_seconds") or 30.0) * base))
    if network == "mstcn":
        layers = mstcn_layers_for(budget)
        context_frames = receptive_field(layers, 2)
    else:
        layers = 8
        context_frames = window_frames(base, float(settings.get("context_seconds") or 30.0))
    target_frames = window_frames(base, float(settings.get("target_seconds") or 30.0))
    run = store.create_run(project_id)
    store.write_run_files(run, {"split.json": {
        "same_source": False,
        "train": train_ids,
        "val": val_ids,
        "train_clips": train_ids,
        "val_clips": val_ids,
    }})

    def publish(snapshot: dict) -> None:
        write_progress(folder, {**snapshot, "names": names})

    feature_length = int(train_rows[0].shape[-1])
    context = {
        "project": project_id,
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clips": [clip.clip_id for clip in clips],
        "classes": [names[class_id] for class_id in class_ids],
        "train_clips": train_ids,
        "val_clips": val_ids,
        "feature_length": feature_length,
        "yolo_path": str(settings.get("yolo_path") or ""),
        "yolo_sha256": str(settings.get("yolo_sha256") or ""),
        "hands_sha256": str(settings.get("hands_sha256") or ""),
        "hands_in_training": bool(settings.get("hands_in_training")),
    }
    report = train_timeline(
        train_rows,
        train_label_rows,
        val_rows,
        val_label_rows,
        val_ids,
        class_ids,
        run,
        epochs=int(settings.get("epochs") or 30),
        learning_rate=float(settings.get("learning_rate") or 1e-3),
        dropout=float(settings.get("dropout") or 0.2),
        network=network,
        context_frames=context_frames,
        target_frames=target_frames,
        batch_size=int(settings.get("batch_size") or 8),
        time_base_fps=base,
        window_seconds=prepared.window_seconds,
        mstcn_layers=layers,
        same_source_validation=False,
        part_class_names=part_names,
        required_parts=required_parts_out,
        yolo_path=str(settings.get("yolo_path") or ""),
        yolo_sha256=str(settings.get("yolo_sha256") or ""),
        hands_sha256=str(settings.get("hands_sha256") or "") if settings.get("hands_in_training") else "",
        hands_in_training=bool(settings.get("hands_in_training")),
        conf_threshold=float(settings.get("yolo_conf_threshold") or 0.8),
        station_debounce_frames=int(settings.get("station_debounce_frames") or 3),
        station_min_confidence=float(settings.get("station_min_confidence") or 0.5),
        should_stop=should_stop,
        context=context,
        on_epoch=publish,
        display_name=display_name,
        action_names=names,
    )
    publish({
        "status": "stopped" if report.get("stopped") else "done",
        "epoch": len(report["history"]),
        "epochs": int(settings.get("epochs") or 30),
        "history": report["history"],
        "metrics": report["metrics"],
        "mistakes": report["mistakes"],
        "usable": report["usable"],
        "reason": report["reason"],
        "warnings": annotation_warnings(payload, settings),
        "recipe": report.get("recipe") or {},
    })
    return {"run": str(run), "report": report}


def clip_identity(relative: str) -> tuple[str, str]:
    """单文件用主文件名当片子 id；连续图目录用目录名。"""
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("文件名不合法")
    if len(path.parts) == 1 and path.suffix:
        return path.stem, path.name
    if path.parts:
        return path.parts[0], path.parts[0]
    raise ValueError("文件名不合法")


def clip_named(annotations: dict, relative: str) -> dict | None:
    """按文件名或片子 id 找已经在工程里的那一条。"""
    clip_id, file_field = clip_identity(relative)
    for clip in annotations.get("clips") or []:
        if clip.get("id") == clip_id or clip.get("file") == file_field:
            return clip
    return None


def clear_confirmed_split(store: ProjectStore, project_id: str) -> None:
    """片子增删之后，确认过的划分不再对应当前素材。"""
    current = settings_module.clean({}, store.load_settings(project_id))
    current["split_train"] = []
    current["split_val"] = []
    current["split_for"] = ""
    store.save_settings(project_id, current)


def forget_clip(store: ProjectStore, project_id: str, clip_id: str) -> dict:
    """删掉一条片子的视频、标注和特征。已经训完的轮次留在 runs/。"""
    folder = store.open_dir(project_id)
    annotations = store.load_annotations(project_id)
    clips = list(annotations.get("clips") or [])
    found = next((clip for clip in clips if clip.get("id") == clip_id), None)
    if found is None:
        raise FileNotFoundError(clip_id)
    _remove_media(folder, found)
    feature_store.drop_clip_cache(folder, clip_id)
    annotations["clips"] = [clip for clip in clips if clip.get("id") != clip_id]
    store.save_annotations(project_id, annotations)
    clear_confirmed_split(store, project_id)
    return {"id": clip_id, "file": found.get("file") or clip_id}


def adopt_clip(store: ProjectStore, project_id: str, relative: str, probe: dict, *, replace: bool) -> dict:
    """把刚放进 media/ 的文件收成一条片子。同名且未确认替换时拒绝，不改原标注。"""
    annotations = store.load_annotations(project_id)
    existing = clip_named(annotations, relative)
    if existing and not replace:
        raise FileExistsError(str(existing.get("file") or existing.get("id")))
    if existing and replace:
        forget_clip(store, project_id, str(existing["id"]))
        annotations = store.load_annotations(project_id)
    clip_id, file_field = clip_identity(relative)
    clips = list(annotations.get("clips") or [])
    clips.append({
        "id": clip_id,
        "file": file_field,
        "frame_count": int(probe["frame_count"]),
        "fps": float(probe.get("fps") or 0.0),
        "width": int(probe["width"]),
        "height": int(probe["height"]),
        "segments": [],
    })
    annotations["clips"] = clips
    store.save_annotations(project_id, annotations)
    clear_confirmed_split(store, project_id)
    return {"id": clip_id, "file": file_field, "replaced": bool(existing)}


def _remove_media(folder: Path, clip: dict) -> None:
    name = str(clip.get("file") or clip.get("id") or "")
    if not name:
        return
    root = (folder / "media").resolve()
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        return
    if target.is_dir():
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()
