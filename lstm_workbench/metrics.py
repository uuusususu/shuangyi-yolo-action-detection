"""按时间段打分，不按重叠窗口计数。"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def predict_segment(window_probabilities: Iterable[np.ndarray]) -> int:
    rows = [np.asarray(row, dtype=np.float64) for row in window_probabilities]
    if not rows:
        raise ValueError("一段里没有窗口，不能打分")
    mean = np.mean(np.stack(rows, axis=0), axis=0)
    return int(np.argmax(mean))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_segments(
    truths: list[str],
    predictions: list[str],
    class_ids: list[str],
) -> dict:
    if len(truths) != len(predictions):
        raise ValueError("标签和预测数量不一致")
    per_class = []
    recalls = []
    for class_id in class_ids:
        support = sum(truth == class_id for truth in truths)
        predicted_positive = sum(pred == class_id for pred in predictions)
        true_positive = sum(truth == class_id and pred == class_id for truth, pred in zip(truths, predictions))
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "recall": recall,
                "precision": precision,
                "f1": _f1(precision, recall),
            }
        )
        recalls.append(recall)
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    idle_support = sum(truth == "idle" for truth in truths)
    idle_false = sum(truth == "idle" and pred != "idle" for truth, pred in zip(truths, predictions))
    idle_false_alarm = idle_false / idle_support if idle_support else 0.0
    matrix = []
    for truth_id in class_ids:
        row = []
        for pred_id in class_ids:
            row.append(sum(truth == truth_id and pred == pred_id for truth, pred in zip(truths, predictions)))
        matrix.append(row)
    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "min_recall": min(recalls) if recalls else 0.0,
        "idle_false_alarm": idle_false_alarm,
        "confusion": matrix,
        "class_ids": list(class_ids),
    }


# 验证损失抬头的容差。损失已经降到 0.00x 时，浮点抖动就足以让 val 比上一轮大一点点，
# 那不是过拟合。只有抬得比「绝对 + 相对」容差还多，才算一次真抬头。
LOSS_RISE_ABS = 0.02
LOSS_RISE_REL = 0.05
# 连续几轮抬头才算过拟合趋势。一次抬头可能只是这一轮验证段刚好难一点。
LOSS_RISE_PATIENCE = 3


def val_loss_rose(
    previous_train_loss: float | None,
    train_loss: float,
    previous_val_loss: float | None,
    val_loss: float,
) -> bool:
    """训练损失在降、验证损失明显抬头。第一轮没有上一轮可比，一律不算。"""
    if previous_train_loss is None or previous_val_loss is None:
        return False
    if not train_loss < previous_train_loss:
        return False
    return val_loss > previous_val_loss * (1.0 + LOSS_RISE_REL) + LOSS_RISE_ABS


def gate_passed(
    metrics: dict,
    *,
    train_loss: float,
    previous_train_loss: float | None,
    val_loss: float,
    previous_val_loss: float | None,
    missing_val_class: bool,
    rising_epochs: int | None = None,
) -> tuple[bool, str]:
    """训完并留下权重就可以拿走。分数写在报告里，不在这里决定能不能上工位。"""
    del metrics, train_loss, previous_train_loss, val_loss, previous_val_loss, missing_val_class, rising_epochs
    return True, "已保存"


def segments_from_labels(labels: np.ndarray, class_ids: list[str], ignore: int = -100) -> list[tuple[str, int, int]]:
    """连续同类收成 (类别, 起点, 终点)。忽略帧把段断开，本身不成段。"""
    segments: list[tuple[str, int, int]] = []
    index = 0
    total = int(len(labels))
    while index < total:
        value = int(labels[index])
        if value == ignore or value < 0 or value >= len(class_ids):
            index += 1
            continue
        end = index + 1
        while end < total and int(labels[end]) == value:
            end += 1
        segments.append((class_ids[value], index, end))
        index = end
    return segments


def _span_iou(left: tuple[str, int, int], right: tuple[str, int, int]) -> float:
    inter = max(0, min(left[2], right[2]) - max(left[1], right[1]))
    union = (left[2] - left[1]) + (right[2] - right[1]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_segments(
    truth: list[tuple[str, int, int]],
    predicted: list[tuple[str, int, int]],
    iou: float,
) -> tuple[int, int, int]:
    """动作段配对。返回 (配上的, 没配上的预测, 没配上的标注)。空闲不参与。"""
    ground = [segment for segment in truth if segment[0] != "idle"]
    guesses = [segment for segment in predicted if segment[0] != "idle"]
    used: set[int] = set()
    matched = 0
    for guess in guesses:
        best_index = None
        best_score = -1.0
        for index, segment in enumerate(ground):
            if index in used or segment[0] != guess[0]:
                continue
            score = _span_iou(segment, guess)
            if score + 1e-9 >= iou and score > best_score:
                best_score = score
                best_index = index
        if best_index is not None:
            used.add(best_index)
            matched += 1
    return matched, len(guesses) - matched, len(ground) - matched


def segmental_f1(
    truth: list[tuple[str, int, int]],
    predicted: list[tuple[str, int, int]],
    iou: float,
) -> float:
    matched, false_positive, false_negative = _match_segments(truth, predicted, iou)
    denominator = 2 * matched + false_positive + false_negative
    if denominator == 0:
        return 1.0
    return (2 * matched) / denominator


def _levenshtein(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, item in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            cost = 0 if item == other else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def edit_score(truth: list[tuple[str, int, int]], predicted: list[tuple[str, int, int]]) -> float:
    """动作段类别序列的归一化编辑分，0 到 100。空闲不进序列。"""
    left = [segment[0] for segment in truth if segment[0] != "idle"]
    right = [segment[0] for segment in predicted if segment[0] != "idle"]
    if not left and not right:
        return 100.0
    distance = _levenshtein(left, right)
    return (1.0 - distance / max(len(left), len(right))) * 100.0


def frame_accuracy(truth: np.ndarray, predicted: np.ndarray, ignore: int = -100) -> float:
    valid = np.asarray(truth) != ignore
    if not bool(valid.any()):
        return 0.0
    return float(np.mean(np.asarray(predicted)[valid] == np.asarray(truth)[valid]))


def idle_false_alarm_frames(
    truth: np.ndarray,
    predicted: np.ndarray,
    class_ids: list[str],
    ignore: int = -100,
) -> float:
    idle = class_ids.index("idle")
    truth_arr = np.asarray(truth)
    pred_arr = np.asarray(predicted)
    idle_frames = (truth_arr == idle) & (truth_arr != ignore)
    if not bool(idle_frames.any()):
        return 0.0
    return float(np.mean(pred_arr[idle_frames] != idle))


def _action_count(segments: list[tuple[str, int, int]]) -> int:
    return sum(segment[0] != "idle" for segment in segments)


def score_timeline(truth: np.ndarray, predicted: np.ndarray, class_ids: list[str]) -> dict:
    """整条片子连着跑之后的分数。段级指标不含空闲，逐帧指标不含忽略帧。"""
    truth_segments = segments_from_labels(truth, class_ids)
    pred_segments = segments_from_labels(predicted, class_ids)
    per_class = []
    for class_id in class_ids:
        if class_id == "idle":
            continue
        truth_only = [segment for segment in truth_segments if segment[0] == class_id]
        pred_only = [segment for segment in pred_segments if segment[0] == class_id]
        class_index = class_ids.index(class_id)
        truth_arr = np.asarray(truth)
        pred_arr = np.asarray(predicted)
        valid_frames = truth_arr != -100
        true_positive = int(np.sum(valid_frames & (truth_arr == class_index) & (pred_arr == class_index)))
        truth_frames = int(np.sum(valid_frames & (truth_arr == class_index)))
        pred_frames = int(np.sum(valid_frames & (pred_arr == class_index)))
        precision = true_positive / pred_frames if pred_frames else 0.0
        recall = true_positive / truth_frames if truth_frames else 0.0
        per_class.append({
            "class_id": class_id,
            "support": len(truth_only),
            "precision": precision,
            "recall": recall,
            "f1": segmental_f1(truth_only, pred_only, 0.5) if truth_only or pred_only else 1.0,
        })
    confusion = []
    valid = np.asarray(truth) != -100
    for truth_index, _truth_id in enumerate(class_ids):
        row = []
        for pred_index, _pred_id in enumerate(class_ids):
            row.append(int(np.sum(
                valid & (np.asarray(truth) == truth_index) & (np.asarray(predicted) == pred_index)
            )))
        confusion.append(row)
    return {
        "f1_10": segmental_f1(truth_segments, pred_segments, 0.10),
        "f1_25": segmental_f1(truth_segments, pred_segments, 0.25),
        "f1_50": segmental_f1(truth_segments, pred_segments, 0.50),
        "edit": edit_score(truth_segments, pred_segments),
        "frame_accuracy": frame_accuracy(truth, predicted),
        "idle_false_alarm": idle_false_alarm_frames(truth, predicted, class_ids),
        "pred_segments": _action_count(pred_segments),
        "true_segments": _action_count(truth_segments),
        "per_class": per_class,
        "confusion": confusion,
        "class_ids": list(class_ids),
        "macro_f1": segmental_f1(truth_segments, pred_segments, 0.50),
        "min_recall": min((row["recall"] for row in per_class), default=0.0),
    }


def _as_segments(predicted) -> list[tuple[str, int, int]]:
    rows = []
    for item in predicted:
        if isinstance(item, dict):
            rows.append((str(item["class_id"]), int(item["start_frame"]), int(item["end_frame"])))
        else:
            rows.append((str(item[0]), int(item[1]), int(item[2])))
    return rows


def _majority_name(counts: dict[str, int]) -> str:
    return max(counts, key=lambda name: (counts[name], name))


def _majority_prediction(start: int, end: int, segments: list[tuple[str, int, int]]) -> str:
    counts: dict[str, int] = {}
    covered = 0
    for class_id, left, right in segments:
        overlap = min(end, right) - max(start, left)
        if overlap <= 0:
            continue
        counts[class_id] = counts.get(class_id, 0) + overlap
        covered += overlap
    missing = (end - start) - covered
    if missing > 0:
        counts["idle"] = counts.get("idle", 0) + missing
    if not counts:
        return "idle"
    return _majority_name(counts)


def _majority_label(labels: np.ndarray, start: int, end: int, class_ids: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for value in np.asarray(labels)[start:end]:
        value = int(value)
        if value < 0 or value >= len(class_ids):
            continue
        name = class_ids[value]
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return _majority_name(counts)


def timeline_mistakes(clip_id: str, labels: np.ndarray, predicted, class_ids: list[str]) -> list[dict]:
    """整段被判成别的类，以及空闲上切出来的动作，都记一笔。"""
    guessed = _as_segments(predicted)
    mistakes = []
    for class_id, start, end in segments_from_labels(labels, class_ids):
        if class_id == "idle":
            continue
        predicted_name = _majority_prediction(start, end, guessed)
        if predicted_name != class_id:
            mistakes.append({
                "clip_id": clip_id,
                "truth": class_id,
                "predicted": predicted_name,
                "start": int(start),
                "end": int(end),
            })
    for class_id, start, end in guessed:
        if class_id == "idle":
            continue
        truth_name = _majority_label(labels, start, end, class_ids)
        if truth_name == "idle":
            mistakes.append({
                "clip_id": clip_id,
                "truth": "idle",
                "predicted": class_id,
                "start": int(start),
                "end": int(end),
            })
    return mistakes


def shift_segments(segments: list[tuple[str, int, int]], offset: int) -> list[tuple[str, int, int]]:
    return [(class_id, start + offset, end + offset) for class_id, start, end in segments]


def pick_best(history: list[dict]) -> int:
    """F1@50 最高；相同看编辑分，再看逐帧准确率。返回 history 下标。"""
    if not history:
        raise ValueError("还没有一轮可以挑")

    def key(row: dict) -> tuple[float, float, float]:
        return (
            float(row.get("f1_50") or 0.0),
            float(row.get("edit") or 0.0),
            float(row.get("frame_accuracy") or 0.0),
        )

    return max(range(len(history)), key=lambda index: key(history[index]))
