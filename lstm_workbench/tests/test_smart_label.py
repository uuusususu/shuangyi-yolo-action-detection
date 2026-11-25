"""智能标注（Meta SAM 3 / SAM 2.1 / YOLOE）、安全 OBB 拟合、工程视频抽帧与 8GB 显存训练互斥测试。"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from lstm_workbench import app, model_registry, smart_obb, yolo_jobs
from lstm_workbench.parts_store import PartsStore, _boxes, _label_text
from lstm_workbench.smart_label import TrainingLockedError
from lstm_workbench.workbench import Workbench


# ---------- 1. Mask -> OBB 边界安全拟合 ----------


def test_mask_to_obb_edge_tilted_stays_within_unit_square():
    h, w = 200, 300
    mask = np.zeros((h, w), dtype=np.uint8)
    # 画一个紧贴左上角的倾斜长条，原始 minAreaRect 必有负坐标顶点
    pts = np.array([[0, 25], [20, 0], [95, 55], [75, 80]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    raw_rect = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
    assert float(raw_rect.min()) < 0.0, "测试用例本身应触发原始 minAreaRect 越界"

    fitted = smart_obb.mask_to_obb(mask, w, h, fit_mode="obb")
    assert fitted is not None
    assert smart_obb.is_valid_obb(fitted["points"])
    # 必须 100% 通过 PartsStore 严格校验
    cleaned = _boxes([{"class_id": 0, "points": fitted["points"], "source": "sam3", "confidence": 0.94}], ["零件A"])
    assert cleaned[0]["source"] == "sam3"
    assert cleaned[0]["confidence"] == 0.94
    assert len(fitted["polygon"]) >= 3


def test_mask_to_obb_hbb_and_degenerate_cases():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 20:70] = 255
    res = smart_obb.mask_to_obb(mask, 100, 100, fit_mode="hbb")
    assert res is not None and smart_obb.is_valid_obb(res["points"])
    # 空掩码与极小噪点返回 None
    assert smart_obb.mask_to_obb(np.zeros((50, 50), dtype=np.uint8), 50, 50) is None
    tiny = np.zeros((50, 50), dtype=np.uint8)
    tiny[5, 5] = 255
    assert smart_obb.mask_to_obb(tiny, 50, 50) is None


# ---------- 2. PartsStore 元信息、自动标注门禁与工程视频抽帧 ----------


def _make_sample_video(path: Path, frames: int = 8) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    assert writer.isOpened()
    for i in range(frames):
        img = np.full((64, 96, 3), (i * 25) % 255, dtype=np.uint8)
        cv2.rectangle(img, (5 + i * 6, 10), (25 + i * 6, 40), (0, 200, 255), -1)
        writer.write(img)
    writer.release()


def test_parts_store_metadata_stripping_auto_gate_and_clip_import(tmp_path: Path):
    proj = tmp_path / "proj"
    media = proj / "media"
    media.mkdir(parents=True)
    _make_sample_video(media / "demo.avi", frames=6)

    store = PartsStore(proj)
    clips = store.list_project_media_clips()
    assert len(clips) == 1 and clips[0]["file"] == "demo.avi"

    # 拒绝路径穿越
    with pytest.raises(ValueError):
        store.import_project_clip("../escape.avi")

    snap = store.import_project_clip("demo.avi", interval=2, max_frames=10, dedupe=False)
    assert len(snap["items"]) == 3
    store.set_classes(["散热片"], snap["revision"])

    item0, item1 = snap["items"][0], snap["items"][1]
    box = {"class_id": 0, "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]], "source": "sam3", "confidence": 0.91}

    # apply_auto_boxes 写入 pending，保留元信息
    r0 = store.apply_auto_boxes(item0["id"], [box])
    assert r0["applied"] is True
    assert r0["item"]["status"] == "pending"
    assert r0["item"]["boxes"][0]["source"] == "sam3"

    # 导出标准 YOLO OBB .txt 时必须剥离 source / confidence
    txt = _label_text(r0["item"]["boxes"]).strip()
    assert len(txt.split()) == 9
    assert "sam3" not in txt and "0.91" not in txt

    # 已审核素材必须跳过不被自动标注覆盖
    store.save_item(item0["id"], [box], "reviewed", r0["item"]["revision"])
    skipped_rev = store.apply_auto_boxes(item0["id"], [box])
    assert skipped_rev["applied"] is False

    # 确认无目标素材同样必须跳过
    store.save_item(item1["id"], [], "negative", item1["revision"])
    skipped_neg = store.apply_auto_boxes(item1["id"], [box])
    assert skipped_neg["applied"] is False

    # 非法 source 或越界 confidence 被拒绝
    with pytest.raises(ValueError):
        _boxes([{**box, "source": "evil"}], ["散热片"])
    with pytest.raises(ValueError):
        _boxes([{**box, "confidence": 1.5}], ["散热片"])


# ---------- 3. 模型库注册、默认 SAM3 不可用降级面板、训练强互斥与 HTTP 路由 ----------


@pytest.fixture()
def smart_server(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(model_registry, "MODEL_LIBRARY", tmp_path / "models")
    monkeypatch.setattr(model_registry, "YOLO_LIBRARY", tmp_path / "models" / "yolo")
    monkeypatch.setattr(model_registry, "SAM_LIBRARY", tmp_path / "models" / "sam")
    wb = Workbench(tmp_path / "projects")
    wb.store.create("demo", {})
    # 模拟探测环境：有 YOLO Python 与 8GB GPU，但未装 clip / mobileclip，且无 sam3.pt
    monkeypatch.setattr(
        wb.smart_label,
        "probe_env",
        lambda refresh=False: {
            "ok": True,
            "python_path": r"E:\fake\python.exe",
            "cuda": True,
            "gpu": "RTX 5060",
            "total_mb": 8151,
            "free_mb": 6000,
            "ultralytics": "8.4.102",
            "clip": False,
            "mobileclip": False,
            "clip_install_cmd": '"E:\\fake\\python.exe" -m pip install git+https://github.com/ultralytics/CLIP.git',
            "mobileclip_install_cmd": '"E:\\fake\\python.exe" -m pip install git+https://github.com/ultralytics/mobileclip.git',
        },
    )
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(wb))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield wb, f"http://127.0.0.1:{port}"
    finally:
        wb.close_async_jobs()
        server.shutdown()
        server.server_close()


def _req(base: str, method: str, path: str, payload: dict | None = None, headers: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = dict(headers or {})
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(base + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_sam3_missing_shows_available_models_panel_and_clip_cmd(smart_server):
    wb, base = smart_server
    code, res = _req(base, "POST", "/api/projects/demo/parts/smart/activate", {"model": "sam3.pt", "auto_default": True})
    assert code == 200
    assert res["state"] == "unavailable"
    assert res["reason_code"] == "missing_weight"
    assert res["show_model_panel"] is True
    assert res["active_model"] == ""  # 绝不静默切换其他模型
    names = [m["name"] for m in res["available_models"]]
    assert "sam2.1_b.pt" in names and "sam2.1_t.pt" in names and "yoloe-26s-seg.pt" in names
    assert "CLIP.git" in res["env"]["clip_install_cmd"]
    assert "mobileclip.git" in res["env"]["mobileclip_install_cmd"]


def test_training_locks_all_smart_labeling_and_preempts_before_start(smart_server, monkeypatch):
    wb, base = smart_server
    preempt_calls: list[str] = []
    orig_stop = wb.smart_label.stop_and_release

    def tracked_stop(reason: str = "", timeout: float = 5.0):
        preempt_calls.append(reason)
        orig_stop(reason=reason, timeout=timeout)

    monkeypatch.setattr(wb.smart_label, "stop_and_release", tracked_stop)

    # 1. 前台任务 begin_job 必须先同步抢占停止智能标注
    assert wb.begin_job("demo") is True
    assert len(preempt_calls) == 1
    assert wb.is_training_locked() is True

    # 训练期间所有智能标注接口（含 activate / encode / predict / batch）全部 409 硬拒绝
    for path, body in (
        ("/api/projects/demo/parts/smart/activate", {"model": "mobile_sam.pt"}),
        ("/api/projects/demo/parts/smart/encode", {"item_id": "0123456789abcdef"}),
        ("/api/projects/demo/parts/smart/predict", {"item_id": "0123456789abcdef", "points": [[0.5, 0.5]]}),
        ("/api/projects/demo/parts/smart/batch-zeroshot", {"engine": "yoloe_vp"}),
        ("/api/projects/demo/parts/smart/batch-track", {"seed_item_id": "0123456789abcdef"}),
    ):
        code, payload = _req(base, "POST", path, body)
        assert code == 409, f"{path} 应在训练期返回 409，实际 {code}: {payload}"
        assert payload.get("training_locked") is True

    code, st = _req(base, "GET", "/api/projects/demo/parts/smart/status")
    assert code == 200 and st["state"] == "blocked_by_training" and st["training_locked"] is True

    # 结束后立即解除锁定
    wb.end_job("demo")
    assert wb.is_training_locked() is False

    # 2. YOLO 训练启动接口也必须先调用抢占再拉起 start_training
    events: list[str] = []
    monkeypatch.setattr(wb, "preempt_smart_label", lambda reason="", timeout=5.0: events.append("preempt"))
    monkeypatch.setattr(yolo_jobs, "start_training", lambda folder, body: (events.append("yolo_train"), {"name": "r1", "status": "running"})[1])
    code, _ = _req(base, "POST", "/api/projects/demo/yolo/train", {"epochs": 1})
    assert code == 201
    assert events == ["preempt", "yolo_train"]


def test_background_preparation_yields_to_active_smart_label(smart_server, monkeypatch):
    wb, _base = smart_server
    monkeypatch.setattr(wb.smart_label, "is_active", lambda: True)
    assert wb.begin_preparation("demo") is False
    monkeypatch.setattr(wb.smart_label, "is_active", lambda: False)
    assert wb.begin_preparation("demo") is True
    wb.end_preparation("demo")


def test_lan_readonly_and_pin_protect_smart_routes(smart_server, monkeypatch):
    wb, base = smart_server
    wb.lan_enabled = True
    wb.lan_auth.configure(pin="1234", readonly=True)
    monkeypatch.setattr(app, "is_loopback_client", lambda ip: False)

    # 未解锁 PIN -> 401
    code, body = _req(base, "GET", "/api/projects/demo/parts/smart/status")
    assert code == 401 and body.get("auth_required") is True

    # 解锁 PIN 但处于只读模式 -> GET 允许，POST 写/推理拦截 403
    token = wb.lan_auth.issue_token()
    cookie_pair = wb.lan_auth.cookie_header(token).split(";", 1)[0]
    cookie = {"Cookie": cookie_pair}
    code, _ = _req(base, "GET", "/api/projects/demo/parts/smart/status", headers=cookie)
    assert code == 200
    code, body = _req(base, "POST", "/api/projects/demo/parts/smart/activate", {"model": "sam3.pt"}, headers=cookie)
    assert code == 403 and body.get("readonly") is True


def test_sam_folder_auto_created_upload_activate_and_stop(smart_server, monkeypatch):
    wb, base = smart_server
    sam_dir = model_registry.sam_library_dir()
    # 1. 工作台初始化时应自动创建 models/sam 文件夹和说明文件
    assert sam_dir.is_dir()
    assert (sam_dir / "README.txt").is_file()

    # 2. 通过 /api/smart-models/upload 上传本地 Meta 模型到 models/sam 文件夹
    raw_bytes = b"fake-sam2-weights-bytes"
    up_req = urllib.request.Request(
        base + "/api/smart-models/upload",
        data=raw_bytes,
        method="POST",
        headers={"X-Filename": "sam2.1_b.pt"},
    )
    with urllib.request.urlopen(up_req, timeout=10) as resp:
        assert resp.status == 201
        up_payload = json.loads(resp.read().decode("utf-8"))
    assert up_payload["uploaded_model"] == "sam2.1_b.pt"
    assert (sam_dir / "sam2.1_b.pt").read_bytes() == raw_bytes
    m_item = next(m for m in up_payload["available_models"] if m["name"] == "sam2.1_b.pt")
    assert m_item["local"] is True and m_item["bytes"] == len(raw_bytes)

    # 非法后缀应被拒绝
    bad_req = urllib.request.Request(
        base + "/api/smart-models/upload",
        data=b"evil",
        method="POST",
        headers={"X-Filename": "evil.exe"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_req, timeout=10)
    assert exc_info.value.code == 400

    # 3. 模拟子进程加载与停止：全局 /api/smart-models/activate 可选择加载指定模型，/unload 可随时停止
    loaded_calls: list[dict] = []

    def fake_rpc(payload: dict) -> dict:
        loaded_calls.append(payload)
        return {
            "ok": True,
            "model_name": payload["model_name"],
            "family": "sam2",
            "slot": payload["slot"],
            "device": "0",
            "half": True,
        }

    class _AliveProc:
        pid = 0
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(wb.smart_label, "_call_rpc", fake_rpc)
    wb.smart_label._proc = _AliveProc()

    code, act = _req(base, "POST", "/api/smart-models/activate", {"model": "sam2.1_b.pt", "slot": "auto"})
    assert code == 200
    assert act["state"] == "ready"
    assert act["active_model"] == "sam2.1_b.pt"
    assert loaded_calls and loaded_calls[0]["slot"] == "interactive"

    code, stopped = _req(base, "POST", "/api/smart-models/unload", {"reason": "手动停止测试"})
    assert code == 200
    assert stopped["state"] == "idle"
    assert stopped["active_model"] == ""
    assert stopped["reason_code"] == "stopped"

    # 4. 远端访问 open-dir 应被拒绝（仅限本机打开资源管理器）
    monkeypatch.setattr(app, "is_loopback_client", lambda ip: False)
    code, body = _req(base, "POST", "/api/smart-models/open-dir", {})
    assert code == 403 and "dir" in body
