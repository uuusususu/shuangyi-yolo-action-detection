"""训练循环。权重、契约、划分、结果和日志都写进 run 目录。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import safe_feature_std
from .metrics import gate_passed, predict_segment, score_segments, timeline_mistakes, val_loss_rose
from .model import ActionLSTM, CausalMSTCN
from .model_registry import HANDS_MODEL_COMPLEXITY, file_sha256

SEED = 7


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
    if path_text and Path(path_text).is_file():
        return file_sha256(Path(path_text))
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


SMOOTH_LAMBDA = 0.15
SMOOTH_TAU = 4.0


def _smoothing(logits: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """相邻帧 log-softmax 之差，截断后再平方。用来压类别来回跳。"""
    log_probs = torch.log_softmax(logits, dim=-1)
    pieces = []
    for index, length in enumerate(lengths):
        if length < 2:
            continue
        delta = log_probs[index, 1:length] - log_probs[index, : length - 1]
        pieces.append(torch.clamp(delta, min=-SMOOTH_TAU, max=SMOOTH_TAU).pow(2).mean())
    if not pieces:
        return logits.new_zeros(())
    return torch.stack(pieces).mean()


def frame_loss(model, sequences: list[np.ndarray], labels: list[int], device, loss_fn, smooth: bool = False) -> torch.Tensor:
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
    loss = loss_fn(flat, targets)
    if not smooth:
        return loss
    extra = torch.stack([_smoothing(stage, lengths) for stage in stages]).mean()
    return loss + SMOOTH_LAMBDA * extra


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
    chosen = "mstcn" if network == "mstcn" else "lstm"
    if chosen == "mstcn":
        model = CausalMSTCN(input_size, len(class_ids), dropout=dropout)
    else:
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
        "smoothing": chosen == "mstcn",
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
        loss = frame_loss(model, train_windows, train_labels, device, loss_fn, smooth=(chosen == "mstcn"))
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
        "smoothing": chosen == "mstcn",
        "mstcn_stages": 2,
        "mstcn_layers": 8,
        "mstcn_channels": 64,
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


def _smooth_targets(logits: torch.Tensor, labels: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if end - start < 2:
        return logits.new_zeros(())
    log_probs = torch.log_softmax(logits[start:end], dim=-1)
    kept = labels[start:end]
    delta = log_probs[1:] - log_probs[:-1]
    valid = (kept[1:] != -100) & (kept[:-1] != -100)
    if not bool(valid.any()):
        return logits.new_zeros(())
    clipped = torch.clamp(delta, min=-SMOOTH_TAU, max=SMOOTH_TAU).pow(2)
    return clipped[valid].mean()


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
    mstcn_layers: int = 8,
    mstcn_stages: int = 2,
    mstcn_channels: int = 64,
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
    from .model import CausalMSTCN

    if not train_rows or not val_rows:
        raise ValueError("没有可用的训练或验证片子")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device, device_info = resolve_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    mean, std = fit_feature_norm(train_rows, train_labels)
    chosen = "mstcn" if network == "mstcn" else "lstm"
    input_size = int(np.asarray(train_rows[0]).shape[-1])
    if chosen == "mstcn":
        model = CausalMSTCN(
            input_size, len(class_ids), channels=mstcn_channels,
            layers=mstcn_layers, stages=mstcn_stages, dropout=dropout,
        )
    else:
        model = ActionLSTM(input_size, len(class_ids), dropout=dropout)
    try:
        model = model.to(device)
    except Exception as exc:
        device = torch.device("cpu")
        device_info = {"device": "CPU", "device_kind": "cpu", "device_note": f"显卡不可用：{exc}"}
        model = model.to(device)
    weights = torch.tensor(class_frame_weights(train_labels, len(class_ids)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    plain_loss = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

    def batch_loss(batch: list[tuple[np.ndarray, np.ndarray, int]], smooth: bool) -> torch.Tensor:
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
        loss = loss_fn(torch.cat(pieces, dim=0), torch.cat(targets, dim=0))
        if not smooth:
            return loss
        extras = []
        for stage in stages:
            for index, length in enumerate(lengths):
                extras.append(_smooth_targets(stage[index], label_batch[index], int(batch[index][2]), length))
        return loss + SMOOTH_LAMBDA * torch.stack(extras).mean()

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
        "smoothing": chosen == "mstcn",
        "context_frames": int(context_frames),
        "target_frames": int(target_frames),
        "batch_size": int(batch_size),
        "window_seconds": round(float(window_seconds), 6),
        "train_unit": "timeline",
        **device_info,
    }
    history: list[dict] = []
    best_key = None
    best_state = None
    best_metrics = None
    best_mistakes: list[dict] = []
    stopped = False
    if on_epoch is not None:
        on_epoch({"status": "training", "epoch": 0, "epochs": epochs, "history": history, "recipe": recipe})

    lengths = [int(array.shape[0]) for array in train_rows]
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
            loss = batch_loss(batch, smooth=(chosen == "mstcn"))
            loss.backward()
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
            on_epoch({
                "status": "training", "epoch": epoch + 1, "epochs": epochs,
                "history": history, "metrics": metrics, "mistakes": mistakes, "recipe": recipe,
            })
        rank = (metrics["f1_50"], metrics["edit"], metrics["frame_accuracy"])
        if best_key is None or rank > best_key:
            best_key = rank
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
        "smoothing": chosen == "mstcn",
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
    if chosen == "lstm":
        contract["hidden_size"] = 128
        contract["num_layers"] = 2
    else:
        contract["mstcn_stages"] = int(mstcn_stages)
        contract["mstcn_layers"] = int(mstcn_layers)
        contract["mstcn_channels"] = int(mstcn_channels)
    report = {
        "usable": True,
        "stopped": stopped,
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
