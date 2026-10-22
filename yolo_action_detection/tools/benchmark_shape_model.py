"""对比优化前后 ShapeModel.find() 在高像素帧上的耗时。

用法（在 yolo_action_detection 目录下）：
    python tools/benchmark_shape_model.py
    python tools/benchmark_shape_model.py --frame 5472 3648 --runs 20

基线取自 git HEAD 的 shape_model.py，与工作区当前实现跑同一组输入，
既比耗时也比结果，确保提速没有改变匹配数值。
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_RELATIVE = "yolo_action_detection/src/template_matching/shape_model.py"
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_baseline_module(revision: str):
    """把指定 git 版本的 shape_model.py 作为独立模块加载。"""
    repo_root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ["git", "show", f"{revision}:{REPO_RELATIVE}"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    temp_dir = Path(tempfile.mkdtemp(prefix="shape_baseline_"))
    module_path = temp_dir / "shape_model_baseline_impl.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    # dataclass 装饰器要能通过 __module__ 找回模块，所以先注册再执行。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_target(width: int, height: int, cx: float, cy: float, angle: float) -> np.ndarray:
    """在大画幅上放一个占比很小的目标，贴近现场：搜索窗远小于整幅。"""
    image = np.full((height, width, 3), 40, np.uint8)
    box = cv2.boxPoints(((cx, cy), (170, 110), angle)).astype(np.int32)
    cv2.fillPoly(image, [box], (200, 200, 200))
    cv2.polylines(image, [box], True, (255, 255, 255), 2)
    rad = math.radians(angle)

    def local(dx, dy):
        return (int(cx + dx * math.cos(rad) - dy * math.sin(rad)),
                int(cy + dx * math.sin(rad) + dy * math.cos(rad)))

    cv2.circle(image, local(-45, 0), 13, (30, 30, 30), -1)
    cv2.circle(image, local(45, 0), 13, (30, 30, 30), -1)
    cv2.line(image, local(-70, -40), local(20, 40), (60, 60, 60), 3)
    return image


def build_case(module, width: int, height: int):
    cx, cy = width / 2.0, height / 2.0
    template_scene = make_target(width, height, cx, cy, 0.0)
    top = int(cy) - 65
    left = int(cx) - 95
    template = template_scene[top:top + 130, left:left + 190].copy()
    model = module.ShapeModel(
        template,
        module.ShapeModelSettings.from_legacy(match_threshold=0.6, angle_range=15.0),
    )
    need_w, need_h = model.min_search_size(60)
    rect = {
        "x": (cx - need_w / 2) / width,
        "y": (cy - need_h / 2) / height,
        "width": need_w / width,
        "height": need_h / height,
    }
    scene = make_target(width, height, cx, cy, 3.0)
    return model, scene, rect


def measure(model, scene, rect, runs: int, **find_kwargs):
    model.find(scene, rect, **find_kwargs)  # 预热核缓存
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        result = model.find(scene, rect, **find_kwargs)
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), min(samples), result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", nargs=2, type=int, default=(5472, 3648),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args(argv)
    width, height = args.frame

    print(f"帧尺寸 {width}x{height}，每项测 {args.runs} 次取中位数\n")

    import template_matching.shape_model as current

    baseline = load_baseline_module(args.revision)

    base_model, base_scene, base_rect = build_case(baseline, width, height)
    base_ms, base_min, base_result = measure(base_model, base_scene, base_rect, args.runs)

    cur_model, cur_scene, cur_rect = build_case(current, width, height)
    cur_ms, cur_min, cur_result = measure(cur_model, cur_scene, cur_rect, args.runs)
    lean_ms, lean_min, lean_result = measure(
        cur_model, cur_scene, cur_rect, args.runs, collect_angle_curve=False
    )

    print(f"{'':<28}{'中位 ms':>10}{'最快 ms':>10}{'得分':>10}")
    print(f"{'基线 (' + args.revision + ')':<28}{base_ms:>10.2f}{base_min:>10.2f}{base_result.score:>10.4f}")
    print(f"{'优化后 (含角度曲线)':<26}{cur_ms:>10.2f}{cur_min:>10.2f}{cur_result.score:>10.4f}")
    print(f"{'优化后 (诊断关闭)':<27}{lean_ms:>10.2f}{lean_min:>10.2f}{lean_result.score:>10.4f}")
    print()
    print(f"提速 (含曲线): {base_ms / cur_ms:.2f}x，省 {base_ms - cur_ms:.1f} ms")
    print(f"提速 (诊断关闭): {base_ms / lean_ms:.2f}x，省 {base_ms - lean_ms:.1f} ms")

    center_delta = math.hypot(
        cur_result.center[0] - base_result.center[0],
        cur_result.center[1] - base_result.center[1],
    )
    print()
    print(f"结果一致性: 得分差 {abs(cur_result.score - base_result.score):.2e}，"
          f"中心差 {center_delta:.2e} px，"
          f"角度差 {abs(cur_result.angle - base_result.angle):.2e} 度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
