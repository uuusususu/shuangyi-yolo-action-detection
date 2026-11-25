"""训练循环。权重、契约、划分、结果和日志都写进 run 目录。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import safe_feature_std
from .metrics import gate_passed, predict_segment, score_segments, timeline_mistakes, val_loss_rose
from .model import ActionLSTM
from .model_registry import HANDS_MODEL_COMPLEXITY, file_sha256

SEED = 7

# 梯度裁剪和权重衰减参照 PyTorch 官方 LSTM 示例的做法。
# 在本项目真实数据上逐项对照（同一划分、跑满 200 轮）：
#   不裁剪           帧准确率 0.922  空闲误报 0.104
#   裁剪 0.25        帧准确率 0.931  空闲误报 0.046
#   裁剪 + 衰减 1e-4  帧准确率 0.929  空闲误报 0.046（第 87 轮就到最好）
# 早停和"验证不变好就降学习率"反而更差（空闲误报 0.118）：段 F1 早早到 1.0，
# 空闲识别要到几十轮之后才学稳，提前停或过早降学习率都会错过。所以不用。
GRAD_CLIP = 0.25
WEIGHT_DECAY = 1e-4


def resolve_device() -> tuple[torch.device, dict]:
    """有可用显卡就用它，否则用 CPU。探测失败也退回 CPU，不让训练因此中断。"""
    if not torch.cuda.is_available():
        return torch.device("cpu"), {
            "device": "CPU",
            "device_kind": "cpu",
            "device_note": "没有可用显卡",
        }
    try:
        torch.zeros(1, device="cuda")
        name = torch.cuda.get_device_name(0) or "CUDA"
        return torch.device("cuda"), {"device": name, "device_kind": "cuda", "device_note": ""}
    except Exception as exc:
        return torch.device("cpu"), {
            "device": "CPU",
            "device_kind": "cpu",
            "device_note": f"显卡不可用：{exc}",
        }


def _hash_or_recorded(path_text: str, recorded: str) -> str:
    """优先用模型页记下的指纹，没有才现算，避免每轮重读几百兆。"""
    if recorded:
        return recorded
    if path_text:
        from .model_registry import resolve_yolo_path

        path = Path(resolve_yolo_path(path_text))
        if path.is_file():
            return file_sha256(path)
    return ""


def _log_text(context: dict, history: list[dict], reason: str) -> str:
    lines = [
        f"工程：{context.get('project', '')}",
        f"开始：{context.get('started', '')}",
        f"片子：{'、'.join(context.get('clips') or [])}",
        f"类别：{'、'.join(context.get('classes') or [])}",
        f"划分：训练 {'、'.join(context.get('train_clips') or [])} / 验证 {'、'.join(context.get('val_clips') or [])}",
        f"窗口：{context.get('window', 0)} 帧，步长 {context.get('stride', 0)}",
        f"特征长度：{context.get('feature_length', 0)}",
        f"轮数：{context.get('epochs', 0)}",
        f"学习率：{context.get('learning_rate', '')}  Dropout：{context.get('dropout', '')}",
        f"设备：{context.get('device', '')}",
        f"YOLO：{context.get('yolo_path', '')}",
        f"YOLO 指纹：{context.get('yolo_sha256', '')}",
        f"手部模型指纹：{context.get('hands_sha256', '')}",
        "---",
    ]
    for row in history:
        lines.append(
            f"第 {row['epoch']} 轮  训练损失 {row['train_loss']:.4f}  验证损失 {row['val_loss']:.4f}  "
            f"宏平均 F1 {row['macro_f1']:.3f}  最低召回 {row['min_recall']:.3f}  空闲误报 {row['idle_false_alarm']:.3f}"
        )
    lines.append("---")
    lines.append(f"结果：{reason}")
    return "\n".join(lines) + "\n"


def _write_run_files(run_dir: Path, report: dict, context: dict, contract: dict | None) -> None:
    import json

    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "log.txt").write_text(
        _log_text(context, report["history"], report["reason"]), encoding="utf-8"
    )
    if contract is not None:
        (run_dir / "contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _write_runtime_manifest(run_dir: Path, class_ids: list[str], names: dict) -> None:
    """训练目录本身就能被产线选中。名称来自标注，不猜测。"""
    import json

    from .model_registry import file_sha256

    missing = [class_id for class_id in class_ids if not str(names.get(class_id) or "").strip()]
    if missing:
        raise ValueError(f"类别 {missing[0]} 没有名称，不能写成可加载的模型包")
    manifest = {
        "manifest_version": 1,
        "action_classes": [
            {"id": class_id, "name": str(names[class_id]).strip()} for class_id in class_ids
        ],
        "contract_sha256": file_sha256(run_dir / "contract.json"),
        "weights_sha256": file_sha256(run_dir / "best_lstm.pt"),
        "mapping_source": "confirmed",
    }
    (run_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _frame_probabilities(model, sequences: list[np.ndarray], device) -> list[np.ndarray]:
    batch, lengths = _pack_sequences(sequences, device)
    logits = model(batch, lengths, all_steps=True)
    rows = []
    for index, length in enumerate(lengths):
        rows.extend(torch.softmax(logits[index, :length], dim=1).detach().cpu().numpy())
    return rows


def batch_logits(model, sequences: list[np.ndarray], device):
    """一段一个样本，长短不一样。补齐只为能放进一批，类别取每一段真实的最后一帧。"""
    packed = _pack_sequences(sequences, device)
    return model(packed[0], packed[1])


def frame_loss(model, sequences: list[np.ndarray], labels: list[int], device, loss_fn) -> torch.Tensor:
    """标注覆盖到的每一帧都按这段的类别学习，不只学最后一帧。"""
    batch, lengths = _pack_sequences(sequences, device)
    stages = model.stage_logits(batch)
    logits = stages[-1]
    pieces = [logits[index, : lengths[index]] for index in range(len(sequences))]
    flat = torch.cat(pieces, dim=0)
    targets = torch.cat([
        torch.full((piece.shape[0],), int(label), dtype=torch.long, device=device)
        for piece, label in zip(pieces, labels)
    ])
    return loss_fn(flat, targets)


def _pack_sequences(sequences: list[np.ndarray], device):
    lengths = [int(seq.shape[0]) for seq in sequences]
    width = int(sequences[0].shape[-1])
    if len(set(lengths)) == 1:
        batch = torch.tensor(np.stack(sequences), dtype=torch.float32, device=device)
        return batch, lengths
    longest = max(lengths)
    batch = torch.zeros((len(sequences), longest, width), dtype=torch.float32, device=device)
    for index, seq in enumerate(sequences):
        batch[index, : lengths[index]] = torch.as_tensor(seq, dtype=torch.float32, device=device)
    return batch, lengths


def train_lstm(
    train_windows: list[np.ndarray],
    train_labels: list[int],
    val_groups: list[tuple],
    class_ids: list[str],
    run_dir: Path,
    *,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    window: int,
    stride: int = 8,
    window_seconds: float = 0.0,
    time_base_fps: float = 0.0,
    same_source_validation: bool = False,
    part_class_names: list[str] | None = None,
    required_parts: dict[str, list[str]] | None = None,
    yolo_path: str = "",
    yolo_sha256: str = "",
    hands_path: str = "",
    hands_sha256: str = "",
    hands_in_training: bool = True,
    conf_threshold: float = 0.8,
    station_debounce_frames: int = 3,
    station_min_confidence: float = 0.5,
    feature_code_version: str = "1",
    display_name: str = "",
    network: str = "lstm",
    should_stop=None,
    context: dict | None = None,
    on_epoch=None,
) -> dict:
    if not train_windows:
        raise ValueError("没有训练窗口")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device, device_info = resolve_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    input_size = int(train_windows[0].shape[-1])
    chosen = "lstm"
    model = ActionLSTM(input_size, len(class_ids), dropout=dropout)
    try:
        model = model.to(device)
    except Exception as exc:
        device = torch.device("cpu")
        device_info = {"device": "CPU", "device_kind": "cpu", "device_note": f"显卡不可用：{exc}"}
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    recipe = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "network": chosen,
        "smoothing": False,
        "window_seconds": round(float(window_seconds), 6),
        "window": window,
        "stride": stride,
        **device_info,
    }

    best_f1 = -1.0
    best_state = None
    best_metrics = None
    best_mistakes: list[dict] = []
    best_train = 0.0
    best_val = 0.0
    best_previous_train = None
    best_previous_val = None
    best_rising = 0
    previous_train = None
    previous_val = None
    # 验证损失连续抬头的轮数。它只在「入选检查点」那一刻才影响结论：
    # 入选点之前早就过拟合过又恢复，或者入选点之后才开始抬头，部署的权重都不受影响。
    rising_epochs = 0
    history: list[dict] = []
    stopped = False
    if on_epoch is not None:
        on_epoch({"status": "training", "epoch": 0, "epochs": epochs, "history": history, "recipe": recipe})

    for epoch in range(epochs):
        if should_stop is not None and should_stop():
            stopped = True
            break
        model.train()
        optimizer.zero_grad()
        loss = frame_loss(model, train_windows, train_labels, device, loss_fn)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        model.eval()
        truths = []
        predictions = []
        mistakes = []
        val_losses = []
        with torch.no_grad():
            for group in val_groups:
                class_id, windows = group[0], group[1]
                class_index = class_ids.index(class_id)
                val_loss_value = frame_loss(
                    model, windows, [class_index] * len(windows), device, loss_fn
                )
                val_losses.append(float(val_loss_value.item()))
                frame_rows = _frame_probabilities(model, windows, device)
                truths.append(class_id)
                predictions.append(class_ids[predict_segment(frame_rows)])
                spans = group[2] if len(group) > 2 else []
                clip_id = group[3] if len(group) > 3 else ""
                pred_name = predictions[-1]
                if pred_name != class_id and spans:
                    start, end = spans[0]
                    mistakes.append({
                        "clip_id": clip_id, "truth": class_id, "predicted": pred_name,
                        "start": int(start), "end": int(end),
                    })
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        metrics = score_segments(truths, predictions, class_ids)
        if val_loss_rose(previous_train, train_loss, previous_val, val_loss):
            rising_epochs += 1
        else:
            rising_epochs = 0
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_f1": metrics["macro_f1"],
            "min_recall": metrics["min_recall"],
            "idle_false_alarm": metrics["idle_false_alarm"],
        })
        if on_epoch is not None:
            on_epoch({
                "status": "training",
                "epoch": epoch + 1,
                "epochs": epochs,
                "history": history,
                "metrics": metrics,
                "mistakes": mistakes,
                "recipe": recipe,
            })
        if metrics["macro_f1"] >= best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            best_mistakes = list(mistakes)
            best_train = train_loss
            best_val = val_loss
            best_previous_train = previous_train
            best_previous_val = previous_val
            best_rising = rising_epochs
        previous_train = train_loss
        previous_val = val_loss

    record = dict(context or {})
    record["epochs"] = epochs
    record["window"] = window
    record["stride"] = stride
    record["learning_rate"] = learning_rate
    record["dropout"] = dropout
    record["device"] = device_info["device"]

    if best_state is None or best_metrics is None:
        report = {
            "usable": False,
            "stopped": True,
            "reason": "已停止，还没跑完一轮，没有可用的检查点",
            "metrics": None,
            "mistakes": [],
            "history": history,
            "recipe": recipe,
        }
        _write_run_files(run_dir, report, record, None)
        return report

    cpu_state = {key: value.detach().cpu() for key, value in best_state.items()}
    torch.save(cpu_state, run_dir / "best_lstm.pt")
    passed, reason = gate_passed(
        best_metrics,
        train_loss=best_train,
        previous_train_loss=best_previous_train,
        val_loss=best_val,
        previous_val_loss=best_previous_val,
        missing_val_class=any(row["support"] == 0 for row in best_metrics["per_class"]),
        rising_epochs=best_rising,
    )
    if stopped:
        reason = f"已停止，保留当时最好的一轮。{reason}"

    contract = {
        "feature_code_version": feature_code_version,
        "input_size": input_size,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": dropout,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device_info["device"],
        "device_kind": device_info["device_kind"],
        "window": window,
        "stride": stride,
        "window_seconds": round(float(window_seconds), 6),
        "time_base_fps": round(float(time_base_fps), 6),
        "same_source_validation": bool(same_source_validation),
        "seed": SEED,
        "sample_unit": "segment",
        "label_scope": "frame",
        "network": chosen,
        "smoothing": False,
        "display_name": display_name,
        "class_ids": class_ids,
        "part_class_names": part_class_names or [],
        "required_parts": required_parts or {},
        "yolo_conf_threshold": conf_threshold,
        "yolo_sha256": _hash_or_recorded(yolo_path, yolo_sha256),
        "hands_in_training": bool(hands_in_training),
        "hands_sha256": _hash_or_recorded(hands_path, hands_sha256) if hands_in_training else "",
        "hands_model_complexity": HANDS_MODEL_COMPLEXITY if hands_in_training else None,
        "station_debounce_frames": station_debounce_frames,
        "station_min_confidence": station_min_confidence,
    }
    report = {
        "usable": passed,
        "stopped": stopped,
        "reason": reason,
        "metrics": best_metrics,
        "mistakes": best_mistakes,
        "history": history,
        "recipe": recipe,
    }
    _write_run_files(run_dir, report, record, contract)
    return report


def fit_feature_norm(rows: list[np.ndarray], labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """只用没被忽略的帧。标准差过小的维换成 1，避免除出无穷。"""
    pieces = []
    for array, label in zip(rows, labels):
        keep = np.asarray(label) != -100
        if bool(np.any(keep)):
            pieces.append(np.asarray(array, dtype=np.float32)[keep])
    width = int(np.asarray(rows[0]).shape[-1])
    if not pieces:
        return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    stacked = np.concatenate(pieces, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = safe_feature_std(stacked.std(axis=0))
    return mean, std


def class_frame_weights(labels: list[np.ndarray], class_count: int) -> np.ndarray:
    """帧数平方根的倒数，再归一化成平均为 1。空闲多的时候动作不会被淹没。"""
    counts = np.zeros(class_count, dtype=np.float64)
    for label in labels:
        valid = np.asarray(label)
        valid = valid[valid >= 0]
        if valid.size == 0:
            continue
        values, amount = np.unique(valid, return_counts=True)
        for value, count in zip(values, amount):
            counts[int(value)] += float(count)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / float(weights.mean())
    return weights.astype(np.float32)


def train_timeline(
    train_rows: list[np.ndarray],
    train_labels: list[np.ndarray],
    val_rows: list[np.ndarray],
    val_labels: list[np.ndarray],
    val_ids: list[str],
    class_ids: list[str],
    run_dir: Path,
    *,
    epochs: int,
    learning_rate: float,
    dropout: float,
    network: str,
    context_frames: int,
    target_frames: int,
    batch_size: int,
    time_base_fps: float,
    window_seconds: float,
    same_source_validation: bool = False,
    part_class_names: list[str] | None = None,
    required_parts: dict | None = None,
    yolo_path: str = "",
    yolo_sha256: str = "",
    hands_path: str = "",
    hands_sha256: str = "",
    hands_in_training: bool = True,
    conf_threshold: float = 0.8,
    station_debounce_frames: int = 3,
    station_min_confidence: float = 0.5,
    feature_code_version: str = "1",
    display_name: str = "",
    action_names: dict | None = None,
    should_stop=None,
    context: dict | None = None,
    on_epoch=None,
) -> dict:
    """按整条片子的连续截段训练。每一轮把验证片子整段跑一遍再打分。"""
    from .dataset import draw_crops
    from .metrics import pick_best, score_timeline, segments_from_labels, shift_segments

    if not train_rows or not val_rows:
        raise ValueError("没有可用的训练或验证片子")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device, device_info = resolve_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    mean, std = fit_feature_norm(train_rows, train_labels)
    chosen = "lstm"
    input_size = int(np.asarray(train_rows[0]).shape[-1])
    model = ActionLSTM(input_size, len(class_ids), dropout=dropout)
    try:
        model = model.to(device)
    except Exception as exc:
        device = torch.device("cpu")
        device_info = {"device": "CPU", "device_kind": "cpu", "device_note": f"显卡不可用：{exc}"}
        model = model.to(device)
    weights = torch.tensor(class_frame_weights(train_labels, len(class_ids)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    # 验证损失和训练用同一个带类别权重的口径，两条曲线才能直接比较是否过拟合。
    plain_loss = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    names = {class_id: class_id for class_id in class_ids}
    names["idle"] = "空闲"
    decode_contract = {
        "train_unit": "timeline",
        "time_base_fps": float(time_base_fps),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "min_action_seconds": float(window_seconds),
        "window_seconds": float(window_seconds),
        "class_ids": class_ids,
        "part_class_names": part_class_names or [],
        "required_parts": required_parts or {},
    }

    def normalize(array: np.ndarray) -> np.ndarray:
        return ((np.asarray(array, dtype=np.float32) - mean) / std).astype(np.float32)

    def batch_loss(batch: list[tuple[np.ndarray, np.ndarray, int]]) -> torch.Tensor:
        sequences = [item[0] for item in batch]
        packed, lengths = _pack_sequences(sequences, device)
        label_rows = []
        width = int(max(lengths))
        label_batch = torch.full((len(batch), width), -100, dtype=torch.long, device=device)
        for index, (_rows, labels, _offset) in enumerate(batch):
            label_batch[index, : lengths[index]] = torch.as_tensor(labels, dtype=torch.long, device=device)
            label_rows.append(labels)
        stages = model.stage_logits(packed)
        logits = stages[-1]
        pieces = []
        targets = []
        for index, length in enumerate(lengths):
            offset = int(batch[index][2])
            pieces.append(logits[index, offset:length])
            targets.append(label_batch[index, offset:length])
        return loss_fn(torch.cat(pieces, dim=0), torch.cat(targets, dim=0))

    def validate() -> tuple[dict, float, list[dict]]:
        from .check_clip import decode_timeline

        truths = []
        predictions = []
        truth_segments = []
        pred_segments = []
        losses = []
        mistakes = []
        offset = 0
        model.eval()
        with torch.no_grad():
            for clip_id, rows, labels in zip(val_ids, val_rows, val_labels):
                normed = normalize(rows)
                packed, lengths = _pack_sequences([normed], device)
                logits = model.stage_logits(packed)[-1][0, : lengths[0]]
                target = torch.as_tensor(labels, dtype=torch.long, device=device)
                losses.append(float(plain_loss(logits, target).item()))
                segments, _frame_index = decode_timeline(
                    rows, float(time_base_fps), model, decode_contract, part_class_names or [], names
                )
                pred_index = np.asarray(_frame_index, dtype=np.int64)
                if pred_index.shape[0] != labels.shape[0]:
                    raise ValueError("验证切段的帧数和时间基对不上")
                truths.append(labels)
                predictions.append(pred_index)
                truth_segments.extend(shift_segments(segments_from_labels(labels, class_ids), offset))
                pred_segments.extend(shift_segments(
                    [(segment["class_id"], int(segment["start_frame"]), int(segment["end_frame"])) for segment in segments],
                    offset,
                ))
                mistakes.extend(timeline_mistakes(clip_id, labels, segments, class_ids))
                offset += int(labels.shape[0])
        metrics = score_timeline(np.concatenate(truths), np.concatenate(predictions), class_ids)
        metrics["f1_50"] = _f1_from_segments(truth_segments, pred_segments, 0.50)
        metrics["f1_25"] = _f1_from_segments(truth_segments, pred_segments, 0.25)
        metrics["f1_10"] = _f1_from_segments(truth_segments, pred_segments, 0.10)
        metrics["edit"] = _edit_from_segments(truth_segments, pred_segments)
        metrics["pred_segments"] = sum(segment[0] != "idle" for segment in pred_segments)
        metrics["true_segments"] = sum(segment[0] != "idle" for segment in truth_segments)
        metrics["macro_f1"] = metrics["f1_50"]
        return metrics, float(np.mean(losses)) if losses else 0.0, mistakes

    recipe = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "network": chosen,
        "smoothing": False,
        "context_frames": int(context_frames),
        "target_frames": int(target_frames),
        "batch_size": int(batch_size),
        "window_seconds": round(float(window_seconds), 6),
        "train_unit": "timeline",
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        **device_info,
    }
    history: list[dict] = []
    best_rank: tuple | None = None
    best_epoch = 0
    best_state = None
    best_metrics = None
    best_mistakes: list[dict] = []
    stopped = False
    if on_epoch is not None:
        on_epoch({"status": "training", "epoch": 0, "epochs": epochs, "history": history, "recipe": recipe})

    lengths = [int(array.shape[0]) for array in train_rows]
    heartbeat = {"at": time.monotonic()}
    for epoch in range(epochs):
        if should_stop is not None and should_stop():
            stopped = True
            break
        crops = draw_crops(lengths, int(target_frames), int(context_frames), SEED, epoch)
        model.train()
        running = 0.0
        steps = 0
        for start in range(0, len(crops), max(1, int(batch_size))):
            if should_stop is not None and should_stop():
                stopped = True
                break
            # 进度原本每轮写一次。CPU 上一轮可能超过 45 秒，页面会误判"训练台已断开"。
            # 轮内每 10 秒把上一轮的进度原样再写一次，只为让页面知道还活着。
            now = time.monotonic()
            if on_epoch is not None and now - heartbeat["at"] > 10.0:
                heartbeat["at"] = now
                on_epoch(heartbeat.get("last") or {
                    "status": "training", "epoch": epoch, "epochs": epochs, "history": history, "recipe": recipe,
                })
            batch = []
            for clip_index, input_start, target_start, target_end in crops[start:start + int(batch_size)]:
                rows = normalize(train_rows[clip_index][input_start:target_end])
                labels = np.asarray(train_labels[clip_index][input_start:target_end])
                if not bool(np.any(labels[target_start - input_start:] != -100)):
                    continue
                batch.append((rows, labels, target_start - input_start))
            if not batch:
                continue
            optimizer.zero_grad()
            loss = batch_loss(batch)
            loss.backward()
            # 梯度裁剪：LSTM 长序列反传容易梯度爆炸，官方示例的标准做法。
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += float(loss.item())
            steps += 1
        if stopped and steps == 0 and not history:
            break
        train_loss = running / steps if steps else 0.0
        metrics, val_loss, mistakes = validate()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_50": metrics["f1_50"],
            "f1_25": metrics["f1_25"],
            "f1_10": metrics["f1_10"],
            "edit": metrics["edit"],
            "frame_accuracy": metrics["frame_accuracy"],
            "idle_false_alarm": metrics["idle_false_alarm"],
            "pred_segments": metrics["pred_segments"],
            "true_segments": metrics["true_segments"],
            "macro_f1": metrics["f1_50"],
            "min_recall": metrics["min_recall"],
        })
        if on_epoch is not None:
            heartbeat["last"] = {
                "status": "training", "epoch": epoch + 1, "epochs": epochs,
                "history": history, "metrics": metrics, "mistakes": mistakes, "recipe": recipe,
            }
            heartbeat["at"] = time.monotonic()
            on_epoch(heartbeat["last"])
        # 挑最好的一轮按业务指标：段 F1 → 编辑分 → 帧准确率，并列时验证损失小者优先。
        # 实测只按带类别权重的验证损失挑，会偏向多报动作，空闲误报从 0.04 涨到 0.12；
        # 把损失放在最后打破并列，避免 F1 早早到 1.0 后一直停在最早那一轮。
        rank = (metrics["f1_50"], metrics["edit"], metrics["frame_accuracy"], -val_loss)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            best_mistakes = list(mistakes)
        if stopped:
            break

    record = dict(context or {})
    record["epochs"] = epochs
    record["device"] = device_info["device"]
    if best_state is None or best_metrics is None:
        report = {
            "usable": False, "stopped": True, "reason": "已停止，还没跑完一轮，没有可用的检查点",
            "metrics": None, "mistakes": [], "history": history, "recipe": recipe,
        }
        _write_run_files(run_dir, report, record, None)
        return report
    torch.save(best_state, run_dir / "best_lstm.pt")
    reason = "已停止，保留当时最好的一轮。已保存" if stopped else "已保存"
    contract = {
        "feature_code_version": feature_code_version,
        "input_size": input_size,
        "dropout": dropout,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device_info["device"],
        "device_kind": device_info["device_kind"],
        "window": int(round(float(window_seconds) * float(time_base_fps))),
        "stride": 0,
        "window_seconds": round(float(window_seconds), 6),
        "time_base_fps": round(float(time_base_fps), 6),
        "same_source_validation": bool(same_source_validation),
        "seed": SEED,
        "sample_unit": "timeline",
        "label_scope": "frame",
        "train_unit": "timeline",
        "decode_version": 1,
        "network": chosen,
        "smoothing": False,
        "context_frames": int(context_frames),
        "target_frames": int(target_frames),
        "min_action_seconds": round(float(window_seconds), 6),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "display_name": display_name,
        "class_ids": class_ids,
        "part_class_names": part_class_names or [],
        "required_parts": required_parts or {},
        "yolo_conf_threshold": conf_threshold,
        "yolo_sha256": _hash_or_recorded(yolo_path, yolo_sha256),
        "hands_in_training": bool(hands_in_training),
        "hands_sha256": _hash_or_recorded(hands_path, hands_sha256) if hands_in_training else "",
        "hands_model_complexity": HANDS_MODEL_COMPLEXITY if hands_in_training else None,
        "station_debounce_frames": station_debounce_frames,
        "station_min_confidence": station_min_confidence,
    }
    contract["hidden_size"] = 128
    contract["num_layers"] = 2
    report = {
        "usable": True,
        "stopped": stopped,
        "best_epoch": best_epoch,
        "reason": reason,
        "metrics": best_metrics,
        "mistakes": best_mistakes,
        "history": history,
        "recipe": recipe,
    }
    _write_run_files(run_dir, report, record, contract)
    if action_names and (run_dir / "best_lstm.pt").is_file():
        _write_runtime_manifest(run_dir, class_ids, action_names)
    return report


def _f1_from_segments(truth, predicted, iou: float) -> float:
    from .metrics import segmental_f1

    return segmental_f1(truth, predicted, iou)


def _edit_from_segments(truth, predicted) -> float:
    from .metrics import edit_score

    return edit_score(truth, predicted)
