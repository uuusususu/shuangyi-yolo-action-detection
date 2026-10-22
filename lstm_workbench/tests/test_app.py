"""本机网页只监听 127.0.0.1，接口按文档收口。"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from lstm_workbench.app import serve
from lstm_workbench.project_store import ProjectStore


@contextmanager
def running(tmp_path):
    server = serve(tmp_path, port=0, open_browser=False)
    host, port = server.server_address
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def call(method, path, body=None, headers=None, raw=None):
        if raw is not None:
            data = raw
            hdrs = dict(headers or {})
        else:
            data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            hdrs = {"Content-Type": "application/json", **(headers or {})}
        request = Request(base + path, data=data, method=method, headers=hdrs)
        try:
            with urlopen(request) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    try:
        yield call
    finally:
        server.shutdown()


def test_server_binds_loopback_and_creates_project(tmp_path):
    with running(tmp_path) as call:
        status, payload = call("POST", "/api/projects", {"id": "demo", "settings": {"window_seconds": 2.5}})
        assert status == 201
        assert payload["id"] == "demo"
        assert (tmp_path / "demo" / "annotations.json").is_file()

        status, payload = call("GET", "/api/projects/demo")
        assert status == 200
        assert payload["settings"]["window_seconds"] == 2.5
        assert payload["settings"]["epochs"] == 30, "老文件缺的新项要补上默认值"
        assert payload["blockers"], "没有片子时不能开训"
        assert payload["features"]["clips"] == []


def test_project_name_cannot_escape_the_projects_folder(tmp_path):
    with running(tmp_path) as call:
        for bad in ("../evil", "a/b", ".."):
            status, _payload = call("POST", "/api/projects", {"id": bad})
            assert status == 400
        assert not (tmp_path.parent / "evil").exists()


def test_model_endpoint_refuses_bad_settings_and_keeps_them_out(tmp_path):
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        status, payload = call("POST", "/api/projects/demo/model", {"epochs": 0})
        assert status == 400
        assert "epochs" in payload["error"]
        status, payload = call("GET", "/api/projects/demo")
        assert payload["settings"]["epochs"] == 30, "校验失败不能写进文件"

        status, payload = call("POST", "/api/projects/demo/model", {"yolo_path": "没有这个文件.onnx"})
        assert status == 400
        assert payload["yolo"]["ok"] is False
        status, payload = call("GET", "/api/projects/demo")
        assert payload["settings"]["yolo_sha256"] == "", "加载不了就不记指纹"


def test_annotations_must_carry_two_lists(tmp_path):
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        status, payload = call("POST", "/api/projects/demo/annotations", {"clips": []})
        assert status == 400
        assert "classes" in payload["error"]
        status, _payload = call("POST", "/api/projects/demo/annotations", {"classes": [], "clips": []})
        assert status == 200


def test_stop_is_reachable_while_a_job_holds_the_project(tmp_path):
    """停止不走工程锁，任务还在跑的时候也要能收到。"""
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        status, payload = call("POST", "/api/projects/demo/stop")
        assert status == 200
        assert payload["stopping"] is True


def test_run_detail_returns_report_contract_and_split(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("demo", {})
    run = store.create_run("demo")
    store.write_run_files(run, {
        "metrics.json": {"usable": False, "reason": "验证宏平均 F1 低于 0.90", "metrics": {"macro_f1": 0.4}, "mistakes": []},
        "contract.json": {"class_ids": ["idle", "step1"], "input_size": 131, "window": 48, "stride": 8},
        "split.json": {"train": ["a"], "val": ["b"]},
    })
    (run / "best_lstm.pt").write_bytes(b"weights")
    with running(tmp_path) as call:
        status, payload = call("GET", f"/api/projects/demo/runs/{run.name}")
        assert status == 200
        assert payload["report"]["reason"] == "验证宏平均 F1 低于 0.90"
        assert payload["contract"]["input_size"] == 131
        assert payload["split"]["val"] == ["b"]
        assert "best_lstm.pt" in payload["files"]

        status, _payload = call("GET", "/api/projects/demo/runs/" + quote("没有这一轮"))
        assert status == 404


def test_progress_is_idle_before_any_job(tmp_path):
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        status, payload = call("GET", "/api/projects/demo/progress")
        assert status == 200
        assert payload["status"] == "idle"


def test_train_request_keeps_the_epochs_the_user_typed(tmp_path):
    """训练页填的轮数要先落盘。后面训练失败也不能把这个数丢了。"""
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        status, payload = call("POST", "/api/projects/demo/train", {"epochs": 12, "learning_rate": 0.02})
        assert status == 400
        assert "epochs" not in payload["error"]
        _status, project = call("GET", "/api/projects/demo")
        assert project["settings"]["epochs"] == 12
        assert project["settings"]["learning_rate"] == 0.02


def test_delete_clip_removes_media_features_and_confirmed_split(tmp_path):
    with running(tmp_path) as call:
        call("POST", "/api/projects", {"id": "demo"})
        store = ProjectStore(tmp_path)
        (tmp_path / "demo" / "media" / "1.mp4").write_bytes(b"video")
        (tmp_path / "demo" / "features" / "1.npy").write_bytes(b"npy")
        store.save_annotations("demo", {"classes": [], "clips": [{
            "id": "1", "file": "1.mp4", "frame_count": 10, "fps": 10,
            "width": 8, "height": 8,
            "segments": [{"id": "s", "class_id": "a", "start": 0, "end": 10}],
        }]})
        settings = store.load_settings("demo")
        settings["split_train"] = [["1", "s"]]
        settings["split_val"] = [["1", "idle-0"]]
        settings["split_for"] = "1:10:10"
        store.save_settings("demo", settings)
        status, payload = call("DELETE", "/api/projects/demo/clips/1")
        assert status == 200
        assert payload["deleted"]["file"] == "1.mp4"
        assert not (tmp_path / "demo" / "media" / "1.mp4").exists()
        assert not (tmp_path / "demo" / "features" / "1.npy").exists()
        assert store.load_annotations("demo")["clips"] == []
        saved = store.load_settings("demo")
        assert saved["split_train"] == []
        assert saved["split_val"] == []
        assert saved["split_for"] == ""
        assert (tmp_path / "demo" / "runs").is_dir()


def test_check_video_stays_out_of_the_training_set(tmp_path):
    with running(tmp_path) as call:
        assert call("POST", "/api/projects", {"id": "demo"})[0] == 201
        run = tmp_path / "demo" / "runs" / "r1"
        run.mkdir(parents=True)
        (run / "contract.json").write_text("{}", encoding="utf-8")
        status, payload = call(
            "POST", "/api/projects/demo/runs/r1/check/media",
            raw=b"abcdef",
            headers={"X-Filename": "../out.mp4", "X-Check-Reset": "1"},
        )
        assert status == 201
        assert (run / "check" / "media" / "out.mp4").read_bytes() == b"abcdef"
        assert not (tmp_path / "out.mp4").exists()
        annotations = json.loads((tmp_path / "demo" / "annotations.json").read_text(encoding="utf-8"))
        assert annotations["clips"] == []
        status, payload = call("GET", "/api/projects/demo/runs/r1/check")
        assert status == 200
        assert payload["ready"] is False
        status, payload = call(
            "POST", "/api/projects/demo/runs/r1/check/media",
            raw=b"jpg",
            headers={"X-Filename": "still.jpg", "X-Check-Reset": "1"},
        )
        assert status == 400
        assert "视频" in payload["error"]
