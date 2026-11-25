"""零件模型（YOLO）训练接入：数据集导入、开训、进度、互斥、导出入库。

不依赖真的 ultralytics：用一个假脚本代替训练进程，写出和 Ultralytics 一样的
results.csv、weights/best.pt；导出时假装生成 .onnx。真训练在部署机器上跑。
"""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from lstm_workbench import model_registry, yolo_jobs
from lstm_workbench.app import serve
from lstm_workbench.project_store import ProjectStore

FAKE = '''
import sys, time, pathlib
args = sys.argv[1:]
if args and args[0] == "-c" and "entrypoint" in args[1]:
    args = args[2:]          # 训练：python -c "<调用 entrypoint>" <task> train key=value ...
elif args and args[0] == "-c":
    code = args[1]
    if "torch.cuda" in code:
        print('{"python": "3.11", "ultralytics": "8.4.0", "torch": "2.8.0", "cuda": false, "gpu": ""}')
        sys.exit(0)
    best = pathlib.Path(args[2])
    out = best.with_suffix(".onnx")
    out.write_bytes(b"onnx")
    print("EXPORTED=" + str(out))
    sys.exit(0)
kv = dict(item.split("=", 1) for item in args if "=" in item)
run = pathlib.Path(kv["project"]) / kv["name"]
(run / "weights").mkdir(parents=True, exist_ok=True)
epochs = int(kv["epochs"])
header = "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss"
lines = [header]
for e in range(1, epochs + 1):
    lines.append(f"{e},{1.0/e:.3f},{0.5/e:.3f},{0.5+0.1*e:.3f},{0.3+0.1*e:.3f},{1.2/e:.3f},{0.6/e:.3f}")
    (run / "results.csv").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"epoch {e}/{epochs}", flush=True)
    time.sleep(float(kv.get("sleep", "0")))
(run / "weights" / "best.pt").write_bytes(b"pt")
'''


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    """假 YOLO 环境：把"YOLO 的 Python"指到一个标记路径，起子进程时换成"当前 Python + 假脚本"。

    不用 .cmd 包装：Windows 的批处理会在第一个换行处截断 -c 后面的多行脚本。
    返回模型库目录。
    """
    script = tmp_path / "fake_yolo.py"
    script.write_text(FAKE, encoding="utf-8")
    marker = tmp_path / "yolo_env" / "python.exe"
    marker.parent.mkdir()
    marker.write_bytes(b"")
    real_run, real_popen = yolo_jobs.subprocess.run, yolo_jobs.subprocess.Popen

    def swap(args):
        if args and str(args[0]) == str(marker):
            return [sys.executable, str(script)] + list(args[1:])
        return args

    monkeypatch.setattr(yolo_jobs.subprocess, "run", lambda args, *a, **k: real_run(swap(args), *a, **k))
    monkeypatch.setattr(yolo_jobs.subprocess, "Popen", lambda args, *a, **k: real_popen(swap(args), *a, **k))
    monkeypatch.setattr(yolo_jobs, "_candidate_pythons", lambda: [marker])
    monkeypatch.setattr(yolo_jobs, "base_models_dir", lambda: tmp_path / "yolo_base")
    monkeypatch.setattr(yolo_jobs, "_ENV_CACHE", None)
    library = tmp_path / "library"
    monkeypatch.setattr(model_registry, "YOLO_LIBRARY", library)
    return library


def _dataset(root: Path, with_val: bool = True) -> Path:
    for split in ("train", "val") if with_val else ("train",):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(6):
            (root / "images" / split / f"{index}.jpg").write_bytes(b"jpg")
            (root / "labels" / split / f"{index}.txt").write_text(
                "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "task: obb\nnames:\n  0: 导热垫\n  1: 散热片\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    return root


@contextmanager
def _server(root):
    server = serve(root, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, json.loads(payload or b"{}")

    try:
        yield call
    finally:
        server.shutdown()


def _wait(call, project, predicate, timeout=60):
    deadline = time.time() + timeout
    info = None
    while time.time() < deadline:
        _status, info = call("GET", f"/api/projects/{project}/yolo")
        if predicate(info):
            return info
        time.sleep(0.3)
    raise AssertionError(f"等待超时：{info}")


def test_import_counts_classes_and_splits_validation_when_missing(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    folder = store.create("p", {})
    source = _dataset(tmp_path / "src", with_val=False)
    info = yolo_jobs.import_dataset(folder, str(source))
    assert info["names"] == ["导热垫", "散热片"]
    assert info["train_images"] + info["val_images"] == 6
    assert info["val_images"] >= 1
    assert info["class_counts"] == [6, 6]
    assert (source / "images" / "train" / "0.jpg").is_file(), "原数据集不能被改动"


def test_train_progress_export_and_use_end_to_end(tmp_path, fake_env, monkeypatch):
    monkeypatch.setattr(model_registry, "probe_yolo", lambda path, conf: {
        "ok": True, "path": path, "sha256": "abc", "classes": ["导热垫", "散热片"],
    })
    with _server(tmp_path / "projects") as call:
        assert call("POST", "/api/projects", {"id": "p"})[0] == 201
        assert call("POST", "/api/projects/p/yolo/dataset", {"source": str(_dataset(tmp_path / "src"))})[0] == 200
        status, body = call("POST", "/api/projects/p/yolo/train", {"task": "obb", "model": "x.pt", "epochs": 3, "device": "cpu"})
        assert status == 201, body
        run = body["run"]["name"]
        info = _wait(call, "p", lambda i: not i["active"] and i["runs"] and i["runs"][0]["status"] != "training")
        summary = info["runs"][0]
        assert summary["status"] == "done", summary
        assert summary["epoch"] == 3 and summary["best_epoch"] == 3
        assert summary["map50"] == pytest.approx(0.8)

        status, detail = call("GET", f"/api/projects/p/yolo/runs/{run}")
        assert status == 200 and "metrics/mAP50(B)" in detail["series"]
        status, log = call("GET", f"/api/projects/p/yolo/runs/{run}/log?after=0")
        assert any("epoch 3/3" in line for line in log["lines"])

        status, out = call("POST", f"/api/projects/p/yolo/runs/{run}/export", {})
        assert status == 200, out
        assert (fake_env / out["export"]["file"]).is_file()
        status, used = call("POST", f"/api/projects/p/yolo/runs/{run}/use", {})
        assert status == 200, used
        assert used["settings"]["yolo_path"] == out["export"]["file"]
        assert used["settings"]["yolo_classes"] == ["导热垫", "散热片"]

        status, library = call("GET", "/api/models")
        exported = [item for item in library["yolo"]["files"] if item["name"] == out["export"]["file"]]
        assert exported and exported[0]["used_by"] == ["p"] and exported[0]["origin"]["run"] == run


def test_gpu_is_exclusive_and_stop_ends_yolo_training(tmp_path, fake_env):
    with _server(tmp_path / "projects") as call:
        for name in ("p", "q"):
            assert call("POST", "/api/projects", {"id": name})[0] == 201
            source = _dataset(tmp_path / f"src-{name}")
            assert call("POST", f"/api/projects/{name}/yolo/dataset", {"source": str(source)})[0] == 200
        status, body = call("POST", "/api/projects/p/yolo/train", {"epochs": 50, "device": "cpu", "extra": "sleep=0.2"})
        assert status == 201, body
        _wait(call, "p", lambda i: bool(i["active"]))
        # 另一个工程也不能同时占显卡
        assert call("POST", "/api/projects/q/yolo/train", {"epochs": 1})[0] == 409
        assert call("POST", "/api/projects/q/features", {"clips": []})[0] == 409
        assert call("POST", "/api/projects/p/stop", {})[0] == 200
        info = _wait(call, "p", lambda i: not i["active"])
        assert info["runs"][0]["status"] == "stopped"


def test_extra_args_cannot_override_managed_paths(tmp_path):
    with pytest.raises(ValueError):
        yolo_jobs.build_command("py", tmp_path / "d.yaml", tmp_path, "r", {"extra": "project=C:/elsewhere"})
    with pytest.raises(ValueError):
        yolo_jobs.build_command("py", tmp_path / "d.yaml", tmp_path, "r", {"extra": "notakeyvalue"})
