"""模板匹配现场日志离线分析。

用法：
    python tools/analyze_template_log.py logs/template_sessions/tpl_20260728_101530.ndjson
    python tools/analyze_template_log.py logs/template_sessions            # 分析目录下全部会话

回答四个问题：
1. 当前算法在现场能不能用？（命中率、得分分布、与阈值的余量）
2. 工件实际的旋转范围是多少？（最佳角度直方图）
3. 旋转是否在系统性地拉低得分？（逐角度平均得分曲线 + 旋转掉分比）
4. 换成候选算法会不会更好？（并行记录的 shadow 得分对照）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [warn] {path.name}:{line_no} 不是合法 JSON，已跳过")
    return records


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(data) - 1)
    weight = rank - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def mean(values: Iterable[float]) -> Optional[float]:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def bar(value: float, maximum: float, width: int = 42) -> str:
    if maximum <= 0:
        return ""
    return "#" * max(0, int(round(value / maximum * width)))


def otsu_split(scores: List[float], bins: int = 50) -> Optional[dict]:
    """一维 Otsu 双簇分离。

    现场会话里必然混着真阳（工件在位）和真负（工件不在/未到位）两簇，
    直接拿整体分位数推荐阈值会被真负样本拖偏。这里找类间方差最大的切分点，
    并给出两簇的均值和间隔，间隔太小说明没有任何阈值可用。
    """
    if len(scores) < 8:
        return None
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-6:
        return None
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in scores:
        index = min(bins - 1, int((value - lo) / width))
        counts[index] += 1
    total = len(scores)
    best = None
    for cut in range(1, bins):
        n_low = sum(counts[:cut])
        n_high = total - n_low
        if n_low == 0 or n_high == 0:
            continue
        low_values = [v for v in scores if v < lo + cut * width]
        high_values = [v for v in scores if v >= lo + cut * width]
        mean_low = sum(low_values) / len(low_values)
        mean_high = sum(high_values) / len(high_values)
        between = (n_low / total) * (n_high / total) * (mean_high - mean_low) ** 2
        if best is None or between > best["between_variance"]:
            best = {
                "threshold": lo + cut * width,
                "between_variance": between,
                "low_mean": mean_low,
                "high_mean": mean_high,
                "low_count": n_low,
                "high_count": n_high,
                "low_max": max(low_values),
                "high_min": min(high_values),
            }
    if best is None:
        return None
    best["gap"] = best["high_min"] - best["low_max"]
    best["separation"] = best["high_mean"] - best["low_mean"]
    return best


def report_threshold(label: str, scores: List[float], current: Optional[float] = None) -> None:
    split = otsu_split(scores)
    if split is None:
        print(f"    {label}：样本不足或得分无区分度，无法推荐阈值")
        return
    suggested = (split["low_max"] + split["high_min"]) / 2.0
    print(f"    {label}：双簇分离 低簇 {split['low_count']}帧 均值 {split['low_mean']:.3f} "
          f"(max {split['low_max']:.3f}) | 高簇 {split['high_count']}帧 均值 "
          f"{split['high_mean']:.3f} (min {split['high_min']:.3f})")
    if split["gap"] > 0.08:
        print(f"      两簇有 {split['gap']:.3f} 的干净间隔，建议阈值 {suggested:.2f}")
    elif split["gap"] > 0.0:
        print(f"      两簇间隔仅 {split['gap']:.3f}，阈值 {suggested:.2f} 可用但余量很薄，"
              f"光照或工件批次一变就会误判")
    else:
        print(f"      !! 两簇重叠 {abs(split['gap']):.3f}，没有任何阈值能同时保证不漏检和不误触发。"
              f"必须改算法，不是调阈值。")
    if current is not None:
        side = "偏高" if current > suggested + 0.05 else ("偏低" if current < suggested - 0.05 else "合适")
        print(f"      当前阈值 {current:.3f} 相对建议值{side}")


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def report_search_roi_headroom(records: List[dict], context: dict) -> None:
    """检查搜索 ROI 给模板中心留了多少平移余量，并看命中位置是否顶到边界。

    matchTemplate 只输出模板与搜索域完全重叠的位置，所以模板中心的可搜索范围
    等于 (搜索窗 - 旋转包围盒)。这个余量小于工位停位波动时，模板永远对不准，
    表现就是“得分一直很低、根本不触发”。
    """
    sample = next((r for r in records if r.get("search_rect_px") and r.get("template")), None)
    if sample is None:
        return
    tw, th = (int(v) for v in sample["template"]["size"])
    x0, y0, x1, y1 = (int(v) for v in sample["search_rect_px"])
    sw, sh = x1 - x0, y1 - y0
    angle_range = float(context.get("angle_range") or sample["template"].get("angle_range") or 0.0)

    print("\n[1b] 搜索 ROI 平移余量")
    print(f"    模板 {tw}x{th}   搜索窗 {sw}x{sh} @ ({x0},{y0})-({x1},{y1})")

    # 形状匹配会直接把余量记进日志，优先用它，比事后按角度反推准确
    travel = sample.get("center_travel_px")
    if travel:
        required = sample.get("required_roi_px")
        bbox = sample.get("rotated_bbox_px")
        print(f"    运行时报告：旋转包围盒 {bbox}，平移余量 {travel[0]}x{travel[1]}px，"
              f"最小搜索框 {required}")
        starved = [r for r in records
                   if (r.get("center_travel_px") or [1, 1])[0] < 40
                   or (r.get("center_travel_px") or [1, 1])[1] < 40]
        if starved:
            print(f"    !! {len(starved)}/{len(records)} 帧的平移余量不足 40px，"
                  f"工位停位稍有波动就会对不准，请把搜索框画大。")
    print(f"    {'角度':>6}{'旋转包围盒':>14}{'中心x范围':>20}{'中心y范围':>20}{'可平移':>12}")
    ranges = {}
    for angle in sorted({0.0, angle_range / 2.0, angle_range}):
        rad = math.radians(angle)
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        ow = math.ceil(th * sin_a + tw * cos_a)
        oh = math.ceil(th * cos_a + tw * sin_a)
        if ow > sw or oh > sh:
            print(f"    {angle:5.1f}° {ow:6d}x{oh:<6d}  超出搜索窗，该角度被丢弃")
            continue
        cx0, cx1 = x0 + ow / 2, x1 - ow / 2
        cy0, cy1 = y0 + oh / 2, y1 - oh / 2
        ranges[angle] = (cx0, cx1, cy0, cy1)
        print(f"    {angle:5.1f}° {ow:6d}x{oh:<6d}  [{cx0:.0f}, {cx1:.0f}]{'':>6}"
              f"[{cy0:.0f}, {cy1:.0f}]{'':>6}{sw - ow:.0f}x{sh - oh:.0f}")

    # 用命中位置（优先取候选度量的高分帧，否则取旧算法得分最高的帧）判断是否顶边界
    hits = [r for r in records
            if isinstance(r.get("shadow"), dict) and float(r["shadow"].get("score") or 0) > 0.6
            and r["shadow"].get("center")]
    source = "候选度量>0.6"
    if not hits:
        ranked = sorted((r for r in records if r.get("center")),
                        key=lambda r: -float(r.get("score") or 0))
        hits = ranked[:max(1, len(ranked) // 10)]
        source = "旧算法得分最高的 10%"
        for r in hits:
            r.setdefault("shadow", {})["center"] = r["center"]
    if not hits or 0.0 not in ranges:
        return
    cx0, cx1, cy0, cy1 = ranges[0.0]
    xs = [float(r["shadow"]["center"][0]) for r in hits]
    ys = [float(r["shadow"]["center"][1]) for r in hits]
    at_edge = sum(1 for x, y in zip(xs, ys)
                  if x - cx0 < 10 or cx1 - x < 10 or y - cy0 < 10 or cy1 - y < 10)
    print(f"\n    命中中心分布（依据 {source}，{len(hits)} 帧）："
          f"x [{min(xs):.0f}, {max(xs):.0f}] 跨度 {max(xs)-min(xs):.0f}px，"
          f"y [{min(ys):.0f}, {max(ys):.0f}] 跨度 {max(ys)-min(ys):.0f}px")
    print(f"    其中贴到 0° 可搜索边界（<10px）的帧数：{at_edge}/{len(hits)}")
    if at_edge > 0:
        need_w = math.ceil(tw + (max(xs) - min(xs)) + 200)
        need_h = math.ceil(th + (max(ys) - min(ys)) + 200)
        rad = math.radians(angle_range)
        need_w = max(need_w, math.ceil(th * abs(math.sin(rad)) + tw * abs(math.cos(rad))) + 200)
        need_h = max(need_h, math.ceil(th * abs(math.cos(rad)) + tw * abs(math.sin(rad))) + 200)
        ccx, ccy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        frame = sample.get("frame_size") or [0, 0]
        print("    !! 命中位置顶到搜索边界 —— 模板根本没机会对准，这就是“不触发”的直接原因。")
        print(f"       建议搜索 ROI 至少 {need_w}x{need_h} px，中心对准 ({ccx:.0f}, {ccy:.0f})")
        if frame and frame[0] and frame[1]:
            nx = max(0.0, (ccx - need_w / 2) / frame[0])
            ny = max(0.0, (ccy - need_h / 2) / frame[1])
            print(f"       归一化: x={nx:.4f} y={ny:.4f} "
                  f"width={need_w / frame[0]:.4f} height={need_h / frame[1]:.4f}")


def analyze(path: Path) -> dict:
    records = load_records(path)
    print("=" * 92)
    print(f"会话 {path.name}    帧数 {len(records)}")
    print("=" * 92)
    if not records:
        print("  没有可分析的记录")
        return {}

    summary_path = path.with_suffix("").with_suffix(".summary.json")
    if not summary_path.exists():
        summary_path = path.parent / f"{path.stem}.summary.json"
    context = {}
    if summary_path.exists():
        try:
            context = json.loads(summary_path.read_text(encoding="utf-8")).get("context", {})
        except Exception:
            context = {}
    if not context:
        context = records[0].get("context", {}) or {}

    threshold = float(records[0].get("threshold") or 0.6)
    print(f"\n[配置] 阈值={threshold:.3f} "
          f"角度范围=±{context.get('angle_range', '?')}° 步进={context.get('angle_step', '?')}° "
          f"模板={context.get('template_path', '?')}")
    if context.get("search_rect"):
        print(f"       搜索ROI={context.get('search_rect')}")
        print(f"       识别ROI={context.get('inspection_rect')}")

    scores = [float(r.get("score") or 0.0) for r in records]
    matched = [bool(r.get("matched")) for r in records]
    matched_count = sum(1 for m in matched if m)

    # ---- 1 当前算法可用性 ----
    print("\n[1] 当前算法可用性")
    print(f"    命中率        {matched_count}/{len(records)} = {matched_count / len(records) * 100:.1f}%")
    print(f"    得分 min/p05/p50/p95/max  "
          f"{min(scores):.3f} / {percentile(scores, 5):.3f} / {percentile(scores, 50):.3f} / "
          f"{percentile(scores, 95):.3f} / {max(scores):.3f}")
    margin = percentile(scores, 50) - threshold
    print(f"    中位得分与阈值余量  {margin:+.3f}")
    hist = Counter(round(math.floor(s / 0.05) * 0.05, 2) for s in scores)
    peak = max(hist.values()) if hist else 1
    print("    得分分布：")
    for bucket in sorted(hist):
        flag = "<-阈值" if bucket <= threshold < bucket + 0.05 else ""
        print(f"      {bucket:.2f}-{bucket + 0.05:.2f} {hist[bucket]:6d} "
              f"{bar(hist[bucket], peak)} {flag}")

    print("    阈值可分性：")
    report_threshold("当前算法", scores, threshold)

    failures = Counter(str(r.get("failure") or "") for r in records if r.get("failure"))
    if failures:
        print("    失败原因：")
        for reason, count in failures.most_common():
            print(f"      {reason:26s} {count}")

    matcher = {str(r.get("matcher") or "edge_ncc") for r in records}
    if matcher:
        print(f"    匹配器：{'/'.join(sorted(matcher))}")

    skipped_frames = [r for r in records if r.get("skipped_angles")]
    if skipped_frames:
        skipped = Counter()
        for record in skipped_frames:
            for angle in record["skipped_angles"]:
                skipped[float(angle)] += 1
        print(f"    !! {len(skipped_frames)} 帧存在被静默丢弃的角度候选："
              f"{sorted(skipped)[:6]}{' ...' if len(skipped) > 6 else ''}")
        print("       说明搜索 ROI 装不下旋转后的模板，有效角度范围被悄悄裁剪。")

    report_search_roi_headroom(records, context)

    # ---- 2 工件实际旋转范围 ----
    print("\n[2] 工件实际旋转范围（最佳角度分布）")
    angles = [float(r["angle"]) for r in records if r.get("angle") is not None]
    if angles:
        angle_hist = Counter(angles)
        peak = max(angle_hist.values())
        for angle in sorted(angle_hist):
            print(f"      {angle:+6.1f}° {angle_hist[angle]:6d} {bar(angle_hist[angle], peak)}")
        nonzero = [a for a in angles if abs(a) > 1e-6]
        print(f"    非零角度帧占比 {len(nonzero) / len(angles) * 100:.1f}%，"
              f"角度绝对值 p50={percentile([abs(a) for a in angles], 50):.1f}° "
              f"p95={percentile([abs(a) for a in angles], 95):.1f}°")
        at_limit = [a for a in angles if abs(abs(a) - float(context.get("angle_range") or 0)) < 1e-6]
        if at_limit and len(at_limit) / len(angles) > 0.05:
            print(f"    !! {len(at_limit) / len(angles) * 100:.1f}% 的帧最佳角度落在范围边界，"
                  f"真实旋转可能超出 ±{context.get('angle_range')}°，建议放宽角度范围。")
    else:
        print("      没有成功的角度估计")

    # ---- 3 旋转是否在拉低得分 ----
    print("\n[3] 逐角度平均得分曲线（判断旋转是否在系统性掉分）")
    curve: Dict[float, List[float]] = {}
    for record in records:
        for entry in record.get("angle_scores") or []:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                curve.setdefault(float(entry[0]), []).append(float(entry[1]))
    if curve:
        peak = max(mean(v) or 0.0 for v in curve.values())
        for angle in sorted(curve):
            avg = mean(curve[angle]) or 0.0
            print(f"      {angle:+6.1f}°  mean={avg:.3f}  n={len(curve[angle]):5d} "
                  f"{bar(avg, peak)}")
        zero_mean = mean(curve.get(0.0, [])) or 0.0
        nonzero = [v for a, vs in curve.items() if abs(a) > 1e-6 for v in vs]
        nonzero_mean = mean(nonzero)
        if zero_mean > 1e-9 and nonzero_mean is not None:
            ratio = nonzero_mean / zero_mean
            print(f"\n    0° 平均 {zero_mean:.3f} ；非 0° 平均 {fmt(nonzero_mean)} ；比值 {ratio:.3f}")
            if ratio < 0.85:
                print("    !! 判定：旋转候选本身在退化。比值远小于 1 说明不是工件不像模板，")
                print("       而是“旋转模板图再重跑 Canny”引入了插值模糊和黑边假边缘。")
                print("       修法：旋转边缘点集而不是图像；搜索图改为不二值化的梯度方向匹配。")
            else:
                print("    判定：旋转候选没有明显退化，掉分主要来自工件本身或光照。")

    zero_scores = [float(r["score"]) for r in records
                   if r.get("angle") is not None and abs(float(r["angle"])) < 1e-6]
    tilt_scores = [float(r["score"]) for r in records
                   if r.get("angle") is not None and abs(float(r["angle"])) > 1e-6]
    if zero_scores and tilt_scores:
        print(f"    实际帧对比：最佳角=0° 的帧平均得分 {mean(zero_scores):.3f} (n={len(zero_scores)})，"
              f"最佳角≠0° 的帧平均得分 {mean(tilt_scores):.3f} (n={len(tilt_scores)})")
        zero_rate = sum(1 for r in records
                        if r.get("angle") is not None and abs(float(r["angle"])) < 1e-6
                        and r.get("matched")) / len(zero_scores)
        tilt_rate = sum(1 for r in records
                        if r.get("angle") is not None and abs(float(r["angle"])) > 1e-6
                        and r.get("matched")) / len(tilt_scores)
        print(f"    命中率：0° 帧 {zero_rate * 100:.1f}%  vs  倾斜帧 {tilt_rate * 100:.1f}%")
        if zero_rate - tilt_rate > 0.15:
            print("    !! 倾斜工件的漏检明显高于正放工件 —— 这就是现场正确率上不去的直接原因。")

    # ---- 4 候选算法对照 ----
    shadow = [r for r in records if isinstance(r.get("shadow"), dict)
              and r["shadow"].get("score") is not None]
    print("\n[4] 候选算法（梯度方向点积）对照")
    if not shadow:
        print("      本次会话未启用并行对照")
    else:
        old = [float(r["score"]) for r in shadow]
        new = [float(r["shadow"]["score"]) for r in shadow]
        print(f"    样本 {len(shadow)} 帧")
        print(f"    旧算法 p05/p50/p95  {percentile(old, 5):.3f} / {percentile(old, 50):.3f} / {percentile(old, 95):.3f}")
        print(f"    新度量 p05/p50/p95  {percentile(new, 5):.3f} / {percentile(new, 50):.3f} / {percentile(new, 95):.3f}")
        s_ms = [float(r["shadow"].get("elapsed_ms") or 0.0) for r in shadow]
        o_ms = [float(r.get("total_ms") or 0.0) for r in shadow]
        print(f"    耗时 p50  旧 {percentile(o_ms, 50):.1f}ms  新 {percentile(s_ms, 50):.1f}ms")
        s_tilt = [r for r in shadow if r["shadow"].get("angle") is not None
                  and abs(float(r["shadow"]["angle"])) > 1e-6]
        s_zero = [r for r in shadow if r["shadow"].get("angle") is not None
                  and abs(float(r["shadow"]["angle"])) < 1e-6]
        if s_tilt and s_zero:
            zm = mean([float(r["shadow"]["score"]) for r in s_zero]) or 0.0
            tm = mean([float(r["shadow"]["score"]) for r in s_tilt]) or 0.0
            print(f"    新度量旋转掉分比 {tm / zm:.3f} (0°={zm:.3f}, 非0°={tm:.3f})"
                  if zm > 1e-9 else "")
        disagree = [r for r in shadow if r.get("shadow_disagrees")]
        print(f"    两者判定不一致 {len(disagree)} 帧 "
              f"({len(disagree) / len(shadow) * 100:.1f}%)")
        print("\n    阈值可分性（真阳/真负双簇分离）：")
        report_threshold("旧算法", old, threshold)
        report_threshold("新度量", new)

    # ---- 定位抖动与耗时 ----
    print("\n[5] 定位稳定性与耗时")
    cx = [float(r["center"][0]) for r in records if r.get("center")]
    cy = [float(r["center"][1]) for r in records if r.get("center")]
    if cx and cy:
        print(f"    命中中心 x 标准差 {std(cx):.2f}px  y 标准差 {std(cy):.2f}px")
        print(f"    x 范围 [{min(cx):.0f}, {max(cx):.0f}]  y 范围 [{min(cy):.0f}, {max(cy):.0f}]")
        if max(cx) - min(cx) > 40 or max(cy) - min(cy) > 40:
            print("    !! 命中位置跨度较大。识别 ROI 是固定的，位置漂移会直接导致计数错误，")
            print("       必须加位姿容差门控，或让识别 ROI 跟随模板位姿。")
    total_ms = [float(r.get("total_ms") or 0.0) for r in records]
    print(f"    模板匹配耗时 p50/p95/max  {percentile(total_ms, 50):.1f} / "
          f"{percentile(total_ms, 95):.1f} / {max(total_ms):.1f} ms")

    search_mean = [float((r.get("search_stats") or {}).get("mean") or 0.0) for r in records]
    search_mean = [v for v in search_mean if v > 0]
    if search_mean:
        print(f"    搜索域灰度均值 p05/p50/p95  {percentile(search_mean, 5):.1f} / "
              f"{percentile(search_mean, 50):.1f} / {percentile(search_mean, 95):.1f}")
        if max(search_mean) - min(search_mean) > 25:
            print("    !! 曝光/环境光在会话内有明显漂移，会影响自适应 Canny 阈值的稳定性。")

    # ---- YOLO 计数 ----
    yolo = [r for r in records if (r.get("yolo") or {}).get("ran")]
    if yolo:
        print(f"\n[6] 门控通过后的 YOLO 计数（{len(yolo)} 帧）")
        labels = Counter()
        for record in yolo:
            for label, count in (record["yolo"].get("labels") or {}).items():
                labels[label] += int(count or 0)
        for label, total in labels.most_common():
            print(f"      {label:20s} 累计 {total}  平均每帧 {total / len(yolo):.2f}")

    print()
    return {"path": str(path), "frames": len(records),
            "matched_rate": matched_count / len(records)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分析模板匹配现场诊断日志")
    parser.add_argument("target", help="ndjson 文件或包含会话的目录")
    args = parser.parse_args(argv)

    target = Path(args.target)
    if target.is_dir():
        paths = sorted(target.glob("*.ndjson"))
        if not paths:
            print(f"目录内没有 .ndjson 会话：{target}")
            return 1
    elif target.is_file():
        paths = [target]
    else:
        print(f"路径不存在：{target}")
        return 1

    results = [analyze(path) for path in paths]
    if len(results) > 1:
        print("=" * 92)
        print("全部会话汇总")
        print("=" * 92)
        for item in results:
            if item:
                print(f"  {Path(item['path']).name:44s} 帧数 {item['frames']:6d} "
                      f"命中率 {item['matched_rate'] * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
