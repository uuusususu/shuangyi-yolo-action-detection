"""验证本轮升级能力：
1. ul:// 平台命令拆解、鉴权解析、NDJSON 并发拉取、自动打包生成 .zip 及下载；
2. 内网模式（LAN）Host/Origin 安全放行与跨机分块流式上传（图片/ZIP）；
3. YOLO 官方全套图表与验证集真值/预测配对枚举；
4. 验证/预标注子进程逐张进度回写。
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import io
import json
from pathlib import Path
import threading
import time
import zipfile

import cv2
import numpy as np

from lstm_workbench import dataset_transfer, model_registry, yolo_jobs, yolo_validation, yolo_validation_worker
from lstm_workbench.app import _is_allowed_host, detect_lan_ips, serve


def test_parse_ultralytics_command_extracts_key_uri_model_and_params():
    snippet = (
        'export ULTRALYTICS_API_KEY="ul_demo1234567890abcdef123456"\n'
        'yolo train model=ul://ultralytics/yolo26/yolo26n-obb '
        'data=ul://unicorn-kingfisher/datasets/sanreipian '
        'task=obb project=unicorn-kingfisher/sanreidian batch=70'
    )
    parsed = dataset_transfer.parse_ultralytics_input(snippet)
    assert parsed["kind"] == "ul_dataset"
    assert parsed["api_key"] == "ul_demo1234567890abcdef123456"
    assert "••••••••" in parsed["api_key_masked"]
    assert parsed["data"] == "ul://unicorn-kingfisher/datasets/sanreipian"
    assert parsed["model"] == "ul://ultralytics/yolo26/yolo26n-obb"
    assert parsed["model_filename"] == "yolo26n-obb.pt"
    assert parsed["params"]["task"] == "obb"
    assert parsed["params"]["batch"] == 70


def test_lan_host_guard_allows_private_ips_and_blocks_dns_rebinding(tmp_path):
    assert _is_allowed_host("127.0.0.1", lan_enabled=False) is True
    assert _is_allowed_host("192.168.1.88", lan_enabled=False) is False
    assert _is_allowed_host("192.168.1.88", lan_enabled=True) is True
    assert _is_allowed_host("10.0.0.15", lan_enabled=True) is True
    assert _is_allowed_host("attacker.example", lan_enabled=True) is False
    assert isinstance(detect_lan_ips(), list)

    server = serve(tmp_path, port=0, open_browser=False, host="0.0.0.0")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        def request(method, path, body=b"", headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            raw = resp.read()
            conn.close()
            return resp.status, raw

        # 模拟局域网浏览器带私网 Host 与 Origin 访问并新建工程
        lan_headers = {
            "Host": f"192.168.1.50:{port}",
            "Origin": f"http://192.168.1.50:{port}",
            "Content-Type": "application/json",
        }
        code, raw = request("POST", "/api/projects", json.dumps({"id": "lan_proj"}).encode("utf-8"), lan_headers)
        assert code == 201

        code, raw = request("GET", "/api/server-info", headers={"Host": f"192.168.1.50:{port}"})
        assert code == 200
        info = json.loads(raw.decode("utf-8"))
        assert info["lan_enabled"] is True

        # 即使开启内网，恶意外部域名与跨站 Origin 仍必须返回 403
        code, _ = request("GET", "/api/projects", headers={"Host": f"evil.example:{port}"})
        assert code == 403
        code, _ = request(
            "POST",
            "/api/projects",
            json.dumps({"id": "bad"}).encode("utf-8"),
            {"Host": f"192.168.1.50:{port}", "Origin": "https://evil.example", "Content-Type": "application/json"},
        )
        assert code == 403

        # 测试内网客户端直接流式推送图片至零件工作区
        ok, buf = cv2.imencode(".png", np.full((64, 64, 3), 180, dtype=np.uint8))
        assert ok
        code, raw = request(
            "POST",
            "/api/projects/lan_proj/parts/upload",
            buf.tobytes(),
            {"Host": f"192.168.1.50:{port}", "Origin": f"http://192.168.1.50:{port}", "X-Filename": "snap01.png"},
        )
        assert code == 201
        snap = json.loads(raw.decode("utf-8"))
        assert len(snap["items"]) == 1
    finally:
        server.shutdown()


@contextmanager
def _fake_ul_platform(image_bytes: bytes):
    state = {"port": 0, "auth_headers": []}

    class PlatformHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_HEAD(self):  # noqa: N802
            state["auth_headers"].append(self.headers.get("Authorization"))
            if self.headers.get("Authorization") != "Bearer ul_valid_test_key_12345":
                self.send_response(401)
                self.end_headers()
                return
            if self.path == "/api/webhooks/datasets/unicorn-kingfisher/sanreipian/export":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{state['port']}/cdn/sanreipian.ndjson")
                self.end_headers()
                return
            if self.path == "/api/webhooks/models/ultralytics/yolo26/yolo26n-obb/download":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{state['port']}/cdn/yolo26n-obb.pt")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == "/cdn/yolo26n-obb.pt":
                body = b"FAKE_PT_WEIGHTS"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/cdn/sanreipian.ndjson":
                lines = [
                    json.dumps({"type": "dataset", "task": "obb", "name": "sanreipian", "class_names": {"0": "散热片A", "1": "散热片B"}}, ensure_ascii=False),
                    json.dumps({
                        "type": "image", "file": "a.jpg", "url": f"http://127.0.0.1:{state['port']}/cdn/img1.jpg",
                        "width": 100, "height": 80, "split": "train",
                        "annotations": {"obb": [[0, 0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]]},
                    }),
                    json.dumps({
                        "type": "image", "file": "b.jpg", "url": f"http://127.0.0.1:{state['port']}/cdn/img2.jpg",
                        "width": 100, "height": 80, "split": "val",
                        "annotations": {"obb": [[1, 0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6]]},
                    }),
                ]
                body = ("\n".join(lines) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in {"/cdn/img1.jpg", "/cdn/img2.jpg"}:
                payload = image_bytes + (b"\x00" if self.path.endswith("1.jpg") else b"")
                # 保证两张图内容不同以免去重合并
                img = np.full((80, 100, 3), 60 if self.path.endswith("1.jpg") else 190, dtype=np.uint8)
                encoded = cv2.imencode(".jpg", img)[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), PlatformHandler)
    state["port"] = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{state['port']}/api/webhooks", state
    finally:
        srv.shutdown()


def test_ul_transfer_downloads_dataset_generates_zip_and_imports(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    (lib / "yolo_base").mkdir(parents=True)
    monkeypatch.setattr(model_registry, "MODEL_LIBRARY", lib)
    monkeypatch.setattr(model_registry, "YOLO_LIBRARY", lib / "yolo")

    dummy_jpg = cv2.imencode(".jpg", np.zeros((40, 40, 3), dtype=np.uint8))[1].tobytes()
    with _fake_ul_platform(dummy_jpg) as (api_base, _st):
        monkeypatch.setattr(dataset_transfer, "PLATFORM_API_URL", api_base)
        projects_dir = tmp_path / "projects"
        server = serve(projects_dir, port=0, open_browser=False)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def call(method, path, payload=None):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
                body = json.dumps(payload).encode("utf-8") if payload is not None else None
                hdrs = {"Host": f"127.0.0.1:{port}"}
                if body is not None:
                    hdrs["Content-Type"] = "application/json"
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                raw = resp.read()
                conn.close()
                return resp.status, raw

            assert call("POST", "/api/projects", {"id": "p1"})[0] == 201
            cmd = (
                'export ULTRALYTICS_API_KEY="ul_valid_test_key_12345"\n'
                'yolo train model=ul://ultralytics/yolo26/yolo26n-obb '
                'data=ul://unicorn-kingfisher/datasets/sanreipian task=obb batch=70'
            )
            code, raw = call("POST", "/api/projects/p1/transfer", {"source": cmd, "target": "both", "download_model": True})
            assert code == 202

            deadline = time.time() + 15
            status_obj = {}
            while time.time() < deadline:
                _, raw = call("GET", "/api/projects/p1/transfer")
                status_obj = json.loads(raw.decode("utf-8"))
                if status_obj.get("status") != "running":
                    break
                time.sleep(0.1)

            assert status_obj.get("status") == "done", status_obj
            assert status_obj["zip_file"] == "sanreipian.zip"
            assert status_obj["downloaded_model"] == "yolo26n-obb.pt"
            assert (lib / "yolo_base" / "yolo26n-obb.pt").is_file()

            # 检查生成的 ZIP 可正常通过 HTTP 下载且包含标准 YOLO OBB 目录结构
            code, zip_bytes = call("GET", "/api/projects/p1/yolo/downloads/sanreipian.zip")
            assert code == 200
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                names = set(zf.namelist())
                assert "data.yaml" in names
                assert any(n.startswith("images/train/") for n in names)
                assert any(n.startswith("labels/val/") for n in names)

            # 检查工程 YOLO 数据集与原生 PartsStore 都已同步入库
            _, yolo_raw = call("GET", "/api/projects/p1/yolo")
            yolo_info = json.loads(yolo_raw.decode("utf-8"))
            assert yolo_info["dataset"]["exists"] is True
            assert yolo_info["dataset"]["names"] == ["散热片A", "散热片B"]

            _, parts_raw = call("GET", "/api/projects/p1/parts")
            parts_info = json.loads(parts_raw.decode("utf-8"))
            assert len(parts_info["items"]) == 2
            assert all(len(it["boxes"]) == 1 for it in parts_info["items"])

            # 检查凭据查询只返回掩码，不泄露明文 API Key
            _, cred_raw = call("GET", "/api/ultralytics/credentials")
            cred = json.loads(cred_raw.decode("utf-8"))
            assert cred["has_key"] is True
            assert "ul_valid_test_key_12345" not in cred_raw.decode("utf-8")
        finally:
            server.shutdown()


def test_yolo_run_detail_enumerates_official_artifacts_and_val_pairs(tmp_path):
    project = tmp_path / "proj"
    run = project / "yolo" / "runs" / "20261118-120000"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({"name": run.name, "status": "done", "params": {"epochs": 10}, "started_at": 100.0, "finished_at": 160.0}), encoding="utf-8")
    (run / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.85,0.80,0.88,0.62\n",
        encoding="utf-8",
    )
    for fname in ("results.png", "confusion_matrix_normalized.png", "OBBPR_curve.png", "labels.jpg", "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"):
        (run / fname).write_bytes(b"PNGDATA")

    detail = yolo_jobs.run_detail(run)
    assert detail["precision"] == 0.85
    assert detail["recall"] == 0.80
    assert detail["elapsed_seconds"] == 60.0
    artifact_names = [a["name"] for a in detail["artifacts"]]
    assert "results.png" in artifact_names
    assert "confusion_matrix_normalized.png" in artifact_names
    assert "OBBPR_curve.png" in artifact_names
    assert "labels.jpg" in artifact_names
    assert "train_batch0.jpg" in artifact_names
    assert len(detail["val_pairs"]) == 1
    assert detail["val_pairs"][0]["labels"] == "val_batch0_labels.jpg"
    assert detail["val_pairs"][0]["pred"] == "val_batch0_pred.jpg"


def test_validation_progress_reports_done_count(tmp_path):
    folder = tmp_path / "proj"
    eval_dir = folder / "parts" / "evaluations" / "eval01"
    eval_dir.mkdir(parents=True)
    record = {
        "id": "eval01",
        "status": "running",
        "mode": "validate",
        "models": [{"identifier": "best.pt"}],
        "items": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    }
    (eval_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")
    yolo_validation_worker._write_progress(eval_dir / "progress.json", 2, 3)

    detail = yolo_validation.validation_detail(folder, "eval01")
    assert detail["done"] == 2
    assert detail["total"] == 3
