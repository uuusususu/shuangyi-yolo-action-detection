"""特征：缓存清单、失效判定，以及抽出时的比例统计。"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from .extract import iter_frames, rows_and_stats_from_frames
from .features import FEATURE_CODE_VERSION, feature_size, hands_in_training
from .model_registry import hands_cache_identity, live_backends, yolo_backend

MANIFEST = "manifest.json"
NO_MODEL_REASON = "还没有能用的模型类别，抽不了特征。先到「模型」页选一个 ONNX，点「检查并保存」。"


def conf_threshold_of(settings: dict, default: float = 0.8) -> float:
    """取置信度阈值。0 是合法设置（全部检测都保留），不能用 `or` 兜底成 0.8：
    以前抽特征按 0.8、缓存键和契约又各按各的，之后验证一律报"置信度阈值不同"。"""
    value = settings.get("yolo_conf_threshold")
    return default if value is None or value == "" else float(value)


class Stopped(RuntimeError):
    """抽特征被人停下。"""


def find_media(media_dir: Path, clip_id: str) -> Path:
    if not media_dir.is_dir():
        raise FileNotFoundError(f"缺少素材 {clip_id}")
    for path in media_dir.iterdir():
        if path.stem == clip_id or path.name == clip_id:
            return path
    raise FileNotFoundError(f"缺少素材 {clip_id}")


def media_signature(path: Path) -> str:
    """素材指纹。视频看大小和修改时间，连续图目录看文件名清单和总大小。"""
    path = Path(path)
    if path.is_dir():
        names = sorted(item.name for item in path.iterdir() if item.is_file())
        total = sum((path / name).stat().st_size for name in names)
        payload = f"{len(names)}\n{total}\n" + "\n".join(names)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    stat = path.stat()
    return hashlib.sha256(f"{path.name}\n{stat.st_size}\n{int(stat.st_mtime)}".encode("utf-8")).hexdigest()


def _yolo_file_identity(settings: dict) -> str:
    """磁盘上 YOLO 文件此刻的大小和修改时间。

    yolo_sha256 只在模型页点"检查并保存"时更新。直接用同名文件覆盖库里的 ONNX、不再点检查，
    旧缓存会被当成最新的继续用，而新抽的片子用的是新模型，两批特征混在一起训练。
    """
    from .model_registry import resolve_yolo_path

    path = Path(resolve_yolo_path(str(settings.get("yolo_path") or "")))
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def cache_key(settings: dict, signature: str, frame_count: int) -> dict:
    """这几项任意一项变了，旧特征就不能再用。"""
    use_hands = hands_in_training(settings)
    key = {
        "feature_code_version": FEATURE_CODE_VERSION,
        "hands_in_training": use_hands,
        "yolo_sha256": str(settings.get("yolo_sha256") or ""),
        "yolo_file": _yolo_file_identity(settings),
        "yolo_conf_threshold": conf_threshold_of(settings),
        "yolo_classes": list(settings.get("yolo_classes") or []),
        "media_signature": signature,
        "frame_count": int(frame_count),
    }
    if use_hands:
        key.update(hands_cache_identity())
    return key


def load_manifest(folder: Path) -> dict:
    path = folder / "features" / MANIFEST
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_manifest(folder: Path, manifest: dict) -> None:
    target = folder / "features" / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _totals(rows: list[dict]) -> dict:
    frames = sum(row["frames"] for row in rows)
    if not frames:
        return {"frames": 0, "no_hand_ratio": 0.0, "no_part_ratio": 0.0, "feature_length": 0}
    no_hand = sum(row["no_hand_frames"] for row in rows)
    no_part = sum(row["no_part_frames"] for row in rows)
    lengths = {row["feature_length"] for row in rows}
    if len(lengths) > 1:
        raise ValueError("同一批特征长度不一致：" + "、".join(str(item) for item in sorted(lengths)))
    return {
        "frames": frames,
        "no_hand_ratio": no_hand / frames,
        "no_part_ratio": no_part / frames,
        "feature_length": lengths.pop(),
    }


def summary(folder: Path, annotations: dict, settings: dict) -> dict:
    """只看清单，不抽特征。页面打开时用它显示每条片子是新的还是过期的。"""
    manifest = load_manifest(folder)
    part_names = list(settings.get("yolo_classes") or [])
    rows = []
    for clip in annotations.get("clips", []):
        clip_id = clip["id"]
        cached = manifest.get(clip_id) or {}
        note = ""
        fresh = False
        try:
            media = find_media(folder / "media", clip_id)
            key = cache_key(settings, media_signature(media), clip.get("frame_count") or 0)
            fresh = (
                bool(cached)
                and cached.get("key") == key
                and (folder / "features" / f"{clip_id}.npy").is_file()
                and _valid_array(folder / "features" / f"{clip_id}.npy", settings, cached)
            )
        except (FileNotFoundError, RuntimeError) as exc:
            note = str(exc)
        npy = folder / "features" / f"{clip_id}.npy"
        file_info = _file_info(npy, cached)
        rows.append({
            "clip_id": clip_id,
            "fresh": fresh,
            "note": note,
            **(cached.get("stats") or {}),
            **file_info,
        })
    return {
        "clips": rows,
        "fresh": sum(1 for row in rows if row["fresh"]),
        "stale": sum(1 for row in rows if not row["fresh"] and not row["note"]),
        "missing": sum(1 for row in rows if row["note"]),
        "length": feature_size(len(part_names), hands=hands_in_training(settings)) if part_names else 0,
        "blocked_reason": "" if part_names else NO_MODEL_REASON,
    }


def _valid_array(path: Path, settings: dict, cached: dict) -> bool:
    """只映射文件并校验形状，损坏或写到一半的缓存不能标成就绪。"""
    array = None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.ndim == 2 and array.shape[0] > 0
            and array.shape[1] == feature_size(len(settings.get("yolo_classes") or []), hands=hands_in_training(settings))
            and array.shape[0] == (cached.get("stats") or {}).get("frames")
            and array.dtype in (np.dtype('float32'), np.dtype('float64'))
        )
    except (OSError, ValueError, EOFError, TypeError, AttributeError, MemoryError):
        return False
    finally:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def drop_clip_cache(folder: Path, clip_id: str) -> None:
    """删片子时把这条的 npy 和清单项一起去掉。已经训完的轮次不动。"""
    npy = folder / "features" / f"{clip_id}.npy"
    if npy.is_file():
        npy.unlink()
    manifest = load_manifest(folder)
    if clip_id in manifest:
        manifest.pop(clip_id)
        save_manifest(folder, manifest)


def _file_info(npy: Path, cached: dict) -> dict:
    """页面要能看见抽出来的文件。没有文件就留空，不编一个路径。"""
    if not npy.is_file():
        return {"file": "", "bytes": None, "extracted_at": ""}
    when = str(cached.get("extracted_at") or "")
    if not when:
        when = datetime.fromtimestamp(npy.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "file": f"features/{npy.name}",
        "bytes": npy.stat().st_size,
        "extracted_at": when,
    }


def extract_project(
    folder: Path,
    annotations: dict,
    settings: dict,
    on_progress=None,
    should_stop=None,
    clip_ids: list[str] | None = None,
    on_frame=None,
    on_clip_frame=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """逐条片子抽特征。清单键没变就复用 npy，变了就重抽。"""
    clips = annotations.get("clips", [])
    if clip_ids is not None:
        wanted = [str(item) for item in clip_ids]
        if not wanted:
            raise ValueError("先勾选要抽取的片子")
        by_id = {clip["id"]: clip for clip in clips}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise ValueError(f"没有这条片子：{missing[0]}")
        clips = [by_id[item] for item in wanted]
    if not clips:
        raise FileNotFoundError("还没有片子，先放入素材")
    part_names = list(settings.get("yolo_classes") or [])
    if not part_names:
        raise ValueError(NO_MODEL_REASON)
    feature_dir = folder / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(folder)
    conf_threshold = conf_threshold_of(settings)
    arrays: dict[str, np.ndarray] = {}
    report: list[dict] = []

    for clip in clips:
        if should_stop is not None and should_stop():
            raise Stopped("已停止抽特征")
        clip_id = clip["id"]
        media = find_media(folder / "media", clip_id)
        key = cache_key(settings, media_signature(media), clip.get("frame_count") or 0)
        npy = feature_dir / f"{clip_id}.npy"
        cached = manifest.get(clip_id) or {}
        frame_count = int(clip.get("frame_count") or 0)
        if on_clip_frame is not None:
            on_clip_frame(clip_id, 0, frame_count)

        def tick(done: int, total: int, _clip_id: str = clip_id) -> None:
            if on_frame is not None:
                on_frame(done, total)
            if on_clip_frame is not None:
                on_clip_frame(_clip_id, done, total)

        if npy.is_file() and cached.get("key") == key and _valid_array(npy, settings, cached):
            arrays[clip_id] = np.load(npy)
            report.append({**cached["stats"], "clip_id": clip_id, "cached": True})
            tick(frame_count, frame_count)
        else:
            use_hands = hands_in_training(settings)
            if use_hands:
                detect, hands = live_backends(settings)
            else:
                detect = yolo_backend(settings)
                def hands(_frame):
                    return []
                hands.close = lambda: None
            frames = iter_frames(media)
            try:
                array, stats = rows_and_stats_from_frames(
                    frames, detect, hands, part_names, conf_threshold,
                    on_frame=tick, frame_count=frame_count,
                    include_hands=use_hands, should_stop=should_stop,
                )
            finally:
                close_frames = getattr(frames, "close", None)
                if close_frames is not None:
                    close_frames()
                close_hands = getattr(hands, "close", None)
                if close_hands is not None:
                    close_hands()
            if should_stop is not None and should_stop():
                raise Stopped("已停止抽特征")
            if cache_key(settings, media_signature(media), frame_count) != key:
                raise RuntimeError("视频或模型在准备期间发生变化，请重新准备")
            temporary = npy.with_name(f".{clip_id}-{uuid.uuid4().hex}.npy")
            try:
                np.save(temporary, array)
                for attempt in range(30):
                    try:
                        temporary.replace(npy)
                        break
                    except PermissionError:
                        if attempt == 29:
                            raise
                        time.sleep(.05)
            finally:
                temporary.unlink(missing_ok=True)
            manifest[clip_id] = {
                "key": key,
                "stats": stats,
                "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            save_manifest(folder, manifest)
            arrays[clip_id] = array
            report.append({**stats, "clip_id": clip_id, "cached": False})
        if on_progress is not None:
            on_progress(report)

    widths = {int(array.shape[1]) for array in arrays.values()}
    if len(widths) > 1:
        raise ValueError(f"特征长度不一致：{sorted(widths)}，不能放进同一次训练")
    summary = {
        **_totals(report),
        "clips": report,
        "reused": sum(1 for row in report if row["cached"]),
        "extracted": sum(1 for row in report if not row["cached"]),
    }
    return arrays, summary
