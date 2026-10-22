"""工程设置：默认值，以及边界校验。

时间基不在这里写死 —— 它由片子帧率推导，这个文件只留一个可选的显式覆盖。
"""
from __future__ import annotations

DEFAULT_SETTINGS = {
    "yolo_path": "",
    "yolo_conf_threshold": 0.8,
    "epochs": 30,
    "station_debounce_frames": 3,
    "station_min_confidence": 0.5,
    # 窗口按秒定义；帧数由时间基换算。
    "window_seconds": 1.92,
    # 0 = 由片子帧率推导。只有全是图片序列（读不到帧率）时才需要显式指定。
    "time_base_fps": 0.0,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "stride": 8,
    # 截段训练：上文和目标段都按秒，换算到时间基。每批多少段。
    "context_seconds": 30.0,
    "target_seconds": 30.0,
    "batch_size": 8,
    # 验证集划分。划分单位是段，split_train / split_val 是 [[clip_id, segment_id], ...]。
    "split_seed": 7,
    # 验证集目标窗口数。0 = 按 20% 自动定。按窗口数而不是段数，短段才不会白占名额。
    "split_val_count": 0,
    "split_train": [],
    "split_val": [],
    # 不参加这次训练的片子 id。空列表表示全部参加。
    "train_excluded": [],
    # lstm：现在这套。mstcn：因果 MS-TCN，训练时加相邻帧平滑。
    "network": "lstm",
    # 新训练只按整条片子留验证。读到旧的 segment 会转成 clip 并清掉按段划分。
    "split_mode": "clip",
    "val_clip_ids": [],
    # 由确认动作写入，页面传什么都不算。
    "split_for": "",
    "yolo_sha256": "",
    "yolo_classes": [],
    "hands_sha256": "",
    # 关掉时训练和特征都不跑手部模型。旧工程没有这一项时按关掉处理。
    "hands_in_training": False,
}

# 这些项由服务端写入，页面传什么都不算。
SERVER_OWNED_KEYS = ("yolo_sha256", "yolo_classes", "hands_sha256", "split_for")
EDITABLE_KEYS = tuple(key for key in DEFAULT_SETTINGS if key not in SERVER_OWNED_KEYS)


def _number(key: str, value: object, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是数字") from None
    if not low <= number <= high:
        raise ValueError(f"{key} 要在 {low} 到 {high} 之间")
    return number


def _positive_int(key: str, value: object, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是整数") from None
    if not 1 <= number <= high:
        raise ValueError(f"{key} 要在 1 到 {high} 之间")
    return number


def _non_negative_int(key: str, value: object, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是整数") from None
    if not 0 <= number <= high:
        raise ValueError(f"{key} 要在 0 到 {high} 之间")
    return number


def _entries(key: str, value: object) -> list[list[str]]:
    """[[clip_id, segment_id], ...]。空列表合法 —— 表示还没划分过。"""
    if not isinstance(value, list):
        raise ValueError(f"{key} 必须是列表")
    out: list[list[str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{key} 里每一项都要是 [片子, 段]")
        clip_id, segment_id = str(item[0]).strip(), str(item[1]).strip()
        if not clip_id or not segment_id:
            raise ValueError(f"{key} 里不能有空白的片子或段")
        out.append([clip_id, segment_id])
    return out


def _id_list(key: str, value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} 必须是列表")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        clip_id = str(item).strip()
        if not clip_id or "/" in clip_id or "\\" in clip_id or clip_id in {".", ".."}:
            raise ValueError(f"{key} 里有不合法的片子")
        if clip_id not in seen:
            seen.add(clip_id)
            out.append(clip_id)
    return out


def display_name(value) -> str:
    """一次训练的名字。空字符串表示没起名。不能当路径用。"""
    name = str(value or "").strip()
    if not name:
        return ""
    if "/" in name or "\\" in name or len(name) > 40:
        raise ValueError("模型名字不能带路径，最多 40 个字")
    return name


def clean(patch: object, current: dict | None = None) -> dict:
    """把页面传来的改动合并进现有设置，逐项校验。指纹与分割签名只能由服务端写入。"""
    if not isinstance(patch, dict):
        raise ValueError("设置必须是对象")
    merged = {**(current or DEFAULT_SETTINGS)}
    for key in patch:
        if key in SERVER_OWNED_KEYS:
            continue
        if key not in EDITABLE_KEYS:
            raise ValueError(f"不接受这一项：{key}")
    if "yolo_path" in patch:
        posted = str(patch["yolo_path"] or "").strip()
        # 下拉框没读到值时页面会交来空字符串。空字符串不能把已经选定的模型清掉。
        if posted or not str(merged.get("yolo_path") or "").strip():
            merged["yolo_path"] = posted
    if "yolo_conf_threshold" in patch:
        merged["yolo_conf_threshold"] = _number("yolo_conf_threshold", patch["yolo_conf_threshold"], 0.0, 1.0)
    if "epochs" in patch:
        merged["epochs"] = _positive_int("epochs", patch["epochs"], 500)
    if "station_debounce_frames" in patch:
        merged["station_debounce_frames"] = _positive_int("station_debounce_frames", patch["station_debounce_frames"], 120)
    if "station_min_confidence" in patch:
        merged["station_min_confidence"] = _number("station_min_confidence", patch["station_min_confidence"], 0.0, 1.0)
    if "window_seconds" in patch:
        new_seconds = _number("window_seconds", patch["window_seconds"], 0.04, 60.0)
        if new_seconds != float(merged.get("window_seconds") or 0.0):
            # 窗口变了，哪些段够长、哪些是空闲段都跟着变，确认过的划分不能再沿用。
            merged["split_train"] = []
            merged["split_val"] = []
            merged["split_for"] = ""
        merged["window_seconds"] = new_seconds
    if "time_base_fps" in patch:
        raw = float(patch["time_base_fps"] or 0.0)
        merged["time_base_fps"] = 0.0 if raw <= 0 else _number("time_base_fps", raw, 1e-3, 1000.0)
    if "learning_rate" in patch:
        merged["learning_rate"] = _number("learning_rate", patch["learning_rate"], 1e-6, 1.0)
    if "dropout" in patch:
        merged["dropout"] = _number("dropout", patch["dropout"], 0.0, 0.9)
    if "stride" in patch:
        merged["stride"] = _positive_int("stride", patch["stride"], 64)
    if "context_seconds" in patch:
        merged["context_seconds"] = _number("context_seconds", patch["context_seconds"], 2.0, 120.0)
    if "target_seconds" in patch:
        merged["target_seconds"] = _number("target_seconds", patch["target_seconds"], 2.0, 120.0)
    if "batch_size" in patch:
        merged["batch_size"] = _positive_int("batch_size", patch["batch_size"], 32)
    if "split_seed" in patch:
        merged["split_seed"] = _non_negative_int("split_seed", patch["split_seed"], 1000000)
        merged["split_train"] = []
        merged["split_val"] = []
        merged["split_for"] = ""
    if "split_val_count" in patch:
        merged["split_val_count"] = _non_negative_int("split_val_count", patch["split_val_count"], 100000)
        merged["split_train"] = []
        merged["split_val"] = []
        merged["split_for"] = ""
    if "split_train" in patch:
        merged["split_train"] = _entries("split_train", patch["split_train"])
    if "split_val" in patch:
        merged["split_val"] = _entries("split_val", patch["split_val"])
    if "train_excluded" in patch:
        excluded = _id_list("train_excluded", patch["train_excluded"])
        previous = list(merged.get("train_excluded") or [])
        if set(excluded) != set(previous):
            merged["split_train"] = []
            merged["split_val"] = []
            merged["split_for"] = ""
        merged["train_excluded"] = excluded
    if "hands_in_training" in patch:
        merged["hands_in_training"] = bool(patch["hands_in_training"])
    if "network" in patch:
        name = str(patch["network"] or "lstm").strip()
        if name not in {"lstm", "mstcn"}:
            raise ValueError("网络只能选 LSTM 或因果 MS-TCN")
        merged["network"] = name
    if "split_mode" in patch and str(patch["split_mode"] or "clip").strip() not in {"clip", "segment"}:
        raise ValueError("划分方式只能是整条片子")
    if "val_clip_ids" in patch:
        ids = _id_list("val_clip_ids", patch["val_clip_ids"])
        if set(ids) != set(merged.get("val_clip_ids") or []):
            merged["split_train"] = []
            merged["split_val"] = []
            merged["split_for"] = ""
        merged["val_clip_ids"] = ids
    held_out = set(merged.get("train_excluded") or [])
    merged["val_clip_ids"] = [item for item in (merged.get("val_clip_ids") or []) if item not in held_out]
    if str(merged.get("split_mode") or "clip") != "clip":
        merged["split_train"] = []
        merged["split_val"] = []
        merged["split_for"] = ""
    merged["split_mode"] = "clip"
    return {**DEFAULT_SETTINGS, **{key: value for key, value in merged.items() if key in DEFAULT_SETTINGS}}
