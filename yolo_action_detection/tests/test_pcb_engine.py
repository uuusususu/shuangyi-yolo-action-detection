"""多 PCB 检查引擎测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from pcb_inspection.models import PcbInspectionConfig, PcbResult, PcbStatus
from pcb_inspection.engine import MultiPcbInspectionEngine
from yolo_runtime.yolo_result_models import ObbDetection


COMP_NAMES = ["R1", "C2", "U3", "Q4"]


def _make_pcb_det(track_id, x, y, w, h, conf=0.9):
    poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ObbDetection(
        class_id=0, label="pcb", conf=conf, track_id=track_id,
        polygon=poly, box=(x, y, x + w, y + h), center=(x + w / 2, y + h / 2),
    )


def _make_comp_det(label, x, y, w=20, h=20, conf=0.9):
    poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ObbDetection(
        class_id=1, label=label, conf=conf, track_id=None,
        polygon=poly, box=(x, y, x + w, y + h), center=(x + w / 2, y + h / 2),
    )


def _config(fail_stable=3, interval=0.0, margin=0.15):
    return PcbInspectionConfig(
        pcb_class_name="pcb",
        component_class_names=list(COMP_NAMES),
        fail_stable_frames=fail_stable,
        round_interval_seconds=interval,
        assignment_margin_ratio=margin,
    )


def test_first_frame_pass():
    """首帧四个元器件全在 -> 立即 PASS。"""
    eng = MultiPcbInspectionEngine(_config())
    dets = [
        _make_pcb_det(1, 100, 100, 200, 200),
        _make_comp_det("R1", 120, 120),
        _make_comp_det("C2", 160, 120),
        _make_comp_det("U3", 120, 160),
        _make_comp_det("Q4", 160, 160),
    ]
    results = eng.update(dets, image_size=(1000, 1000))
    assert len(results) == 1
    assert results[0].result == PcbResult.PASS
    assert results[0].track_id == 1


def test_consecutive_fail():
    """连续三帧缺件 -> 第三帧 FAIL。"""
    eng = MultiPcbInspectionEngine(_config(fail_stable=3))
    dets = [
        _make_pcb_det(1, 100, 100, 200, 200),
        _make_comp_det("R1", 120, 120),
        _make_comp_det("C2", 160, 120),
        _make_comp_det("U3", 120, 160),
        # 缺 Q4
    ]
    # 前两帧不应 FAIL
    r1 = eng.update(dets, image_size=(1000, 1000))
    assert len(r1) == 0
    r2 = eng.update(dets, image_size=(1000, 1000))
    assert len(r2) == 0
    # 第三帧 FAIL
    r3 = eng.update(dets, image_size=(1000, 1000))
    assert len(r3) == 1
    assert r3[0].result == PcbResult.FAIL
    assert "Q4" in r3[0].missing_classes


def test_third_frame_recovers_pass():
    """前两帧缺件，第三帧补齐 -> PASS 不 FAIL。"""
    eng = MultiPcbInspectionEngine(_config(fail_stable=3))
    partial = [
        _make_pcb_det(1, 100, 100, 200, 200),
        _make_comp_det("R1", 120, 120),
        _make_comp_det("C2", 160, 120),
        _make_comp_det("U3", 120, 160),
    ]
    eng.update(partial, image_size=(1000, 1000))
    eng.update(partial, image_size=(1000, 1000))

    complete = partial + [_make_comp_det("Q4", 160, 160)]
    r3 = eng.update(complete, image_size=(1000, 1000))
    assert len(r3) == 1
    assert r3[0].result == PcbResult.PASS


def test_no_pcb_id_no_judgment():
    """PCB 没有 track_id -> 不判定。"""
    eng = MultiPcbInspectionEngine(_config())
    # PCB 检测没有 track_id
    dets = [
        ObbDetection(
            class_id=0, label="pcb", conf=0.9, track_id=None,
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
            box=(100, 100, 300, 300), center=(200, 200),
        ),
        _make_comp_det("R1", 120, 120),
        _make_comp_det("C2", 160, 120),
        _make_comp_det("U3", 120, 160),
        _make_comp_det("Q4", 160, 160),
    ]
    results = eng.update(dets, image_size=(1000, 1000))
    assert len(results) == 0


def test_decided_pcb_not_recounted():
    """已完成 PCB 继续在画面中不重复计数。"""
    eng = MultiPcbInspectionEngine(_config())
    dets = [
        _make_pcb_det(1, 100, 100, 200, 200),
        _make_comp_det("R1", 120, 120),
        _make_comp_det("C2", 160, 120),
        _make_comp_det("U3", 120, 160),
        _make_comp_det("Q4", 160, 160),
    ]
    r1 = eng.update(dets, image_size=(1000, 1000))
    assert len(r1) == 1

    # 后续帧不应再产生结果
    r2 = eng.update(dets, image_size=(1000, 1000))
    r3 = eng.update(dets, image_size=(1000, 1000))
    assert len(r2) == 0
    assert len(r3) == 0


def test_two_pcb_parallel_update():
    """两块 PCB 同时在画面，一块 PASS 一块缺件。"""
    eng = MultiPcbInspectionEngine(_config(fail_stable=3))
    dets = [
        _make_pcb_det(1, 50, 50, 200, 200),
        _make_pcb_det(2, 500, 50, 200, 200),
        # PCB1 四个都在
        _make_comp_det("R1", 70, 70),
        _make_comp_det("C2", 110, 70),
        _make_comp_det("U3", 70, 110),
        _make_comp_det("Q4", 110, 110),
        # PCB2 缺 Q4
        _make_comp_det("R1", 520, 70),
        _make_comp_det("C2", 560, 70),
        _make_comp_det("U3", 520, 110),
    ]
    r = eng.update(dets, image_size=(1000, 1000))
    # PCB1 PASS
    pass_results = [x for x in r if x.result == PcbResult.PASS]
    assert len(pass_results) == 1
    assert pass_results[0].track_id == 1
    # PCB2 未决策（仅 1 帧缺件）
    assert eng.pcb_states[2].consecutive_fail == 1
    assert eng.pcb_states[2].status == PcbStatus.OBSERVING


def test_partial_pcb_no_fail():
    """PCB 正在进入画面（接触边界）-> 不计 FAIL。"""
    eng = MultiPcbInspectionEngine(_config(fail_stable=3))
    dets = [
        _make_pcb_det(1, 0, 100, 200, 200),  # 接触左边界
        _make_comp_det("R1", 20, 120),
    ]
    r = eng.update(dets, image_size=(1000, 1000))
    assert len(r) == 0
    assert eng.pcb_states[1].consecutive_fail == 0


def test_zero_interval_continuous_output():
    """间隔为 0 时多个结果连续输出。"""
    eng = MultiPcbInspectionEngine(_config(interval=0.0))
    dets1 = [
        _make_pcb_det(1, 50, 50, 200, 200),
        _make_comp_det("R1", 70, 70),
        _make_comp_det("C2", 110, 70),
        _make_comp_det("U3", 70, 110),
        _make_comp_det("Q4", 110, 110),
    ]
    dets2 = [
        _make_pcb_det(2, 500, 50, 200, 200),
        _make_comp_det("R1", 520, 70),
        _make_comp_det("C2", 560, 70),
        _make_comp_det("U3", 520, 110),
        _make_comp_det("Q4", 560, 110),
    ]
    r1 = eng.update(dets1, image_size=(1000, 1000))
    assert len(r1) == 1
    r2 = eng.update(dets2, image_size=(1000, 1000))
    assert len(r2) == 1
    assert r2[0].track_id == 2