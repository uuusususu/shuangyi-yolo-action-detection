"""新建工程、中文名、上传、标注保存这条主流程上找到的问题，每条一个回归用例。"""
from __future__ import annotations

import http.client
import json
import threading
import urllib.parse
from contextlib import contextmanager

import numpy as np

from lstm_workbench import service
from lstm_workbench.app import serve
from lstm_workbench.extract import read_image
from lstm_workbench.project_store import ProjectStore, project_id_problem


@contextmanager
def _server(root):
    server = serve(root, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, body=b"", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        try:
            return response.status, json.loads(data)
        except ValueError:
            return response.status, data

    try:
        yield call
    finally:
        server.shutdown()


JSON = {"Content-Type": "application/json"}


def _create(call, name):
    return call("POST", "/api/projects", json.dumps({"id": name}, ensure_ascii=False).encode("utf-8"), JSON)


def test_chinese_and_spaced_project_names_can_be_opened_after_creation(tmp_path):
    with _server(tmp_path) as call:
        for name in ["散热片工程", "包装线 02"]:
            assert _create(call, name)[0] == 201
            quoted = urllib.parse.quote(name, safe="")
            status, payload = call("GET", f"/api/projects/{quoted}")
            assert status == 200, payload
            assert payload["id"] == name
            assert call("GET", f"/api/projects/{quoted}/progress")[0] == 200


def test_windows_illegal_and_case_duplicate_names_are_refused_with_a_reason(tmp_path):
    with _server(tmp_path) as call:
        assert _create(call, "Demo")[0] == 201
        assert _create(call, "demo")[0] == 409
        for name in ["a?b", "CON", "abc.", "a|b"]:
            status, payload = _create(call, name)
            assert status == 400 and payload["error"], name
        assert call("GET", "/api/projects")[1]["projects"] == ["Demo"]


def test_listing_skips_folders_that_are_not_projects(tmp_path):
    (tmp_path / "随手放的文件夹").mkdir()
    with _server(tmp_path) as call:
        assert _create(call, "p")[0] == 201
        assert call("GET", "/api/projects")[1]["projects"] == ["p"]


def test_bad_json_bodies_get_a_400_instead_of_a_dropped_connection(tmp_path):
    with _server(tmp_path) as call:
        assert call("POST", "/api/projects", b"{bad", JSON)[0] == 400
        assert call("POST", "/api/projects", b"[]", JSON)[0] == 400


def test_invalid_annotations_are_refused_and_the_project_still_opens(tmp_path):
    with _server(tmp_path) as call:
        assert _create(call, "p")[0] == 201
        status, payload = call("POST", "/api/projects/p/annotations", json.dumps({"classes": [{}], "clips": []}).encode(), JSON)
        assert status == 400 and payload["error"]
        assert call("GET", "/api/projects/p")[0] == 200


def test_upload_with_chinese_file_name_is_decoded(tmp_path, monkeypatch):
    import lstm_workbench.app as app_module

    monkeypatch.setattr(app_module, "_probe_or_empty", lambda path: {
        "kind": "video", "frame_count": 10, "fps": 10.0, "width": 64, "height": 48,
    })
    with _server(tmp_path) as call:
        assert _create(call, "p")[0] == 201
        status, payload = call("POST", "/api/projects/p/media", b"video", {
            "X-Filename": urllib.parse.quote("测试视频.mp4"),
            "Content-Type": "application/octet-stream",
        })
        assert status == 201, payload
        assert payload["saved"] == "测试视频.mp4"
        assert (tmp_path / "p" / "media" / "测试视频.mp4").read_bytes() == b"video"


def test_failed_replacement_keeps_the_original_clip(tmp_path, monkeypatch):
    import lstm_workbench.app as app_module

    monkeypatch.setattr(app_module, "_probe_or_empty", lambda path: {
        "kind": "video", "frame_count": 10, "fps": 10.0, "width": 64, "height": 48,
    })
    headers = {"X-Filename": "a.mp4", "Content-Type": "application/octet-stream"}
    with _server(tmp_path) as call:
        assert _create(call, "p")[0] == 201
        assert call("POST", "/api/projects/p/media", b"old", headers)[0] == 201

        def locked(*_args, **_kwargs):
            raise PermissionError("文件被占用")

        monkeypatch.setattr(ProjectStore, "move_upload", locked)
        status, _payload = call("POST", "/api/projects/p/media", b"new", {**headers, "X-Replace": "1"})
        assert status == 409
        assert (tmp_path / "p" / "media" / "a.mp4").read_bytes() == b"old"
        clips = json.loads((tmp_path / "p" / "annotations.json").read_text(encoding="utf-8"))["clips"]
        assert [clip["file"] for clip in clips] == ["a.mp4"]


def test_interrupted_job_is_not_shown_as_running_after_restart(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("p", {})
    (tmp_path / "p" / "progress.json").write_text(json.dumps({"status": "training", "history": []}), encoding="utf-8")
    with _server(tmp_path) as call:
        assert call("GET", "/api/projects/p/progress")[1]["status"] == "error"


def test_project_id_rules():
    assert project_id_problem("散热片 02") == ""
    assert project_id_problem("lpt1.txt")
    assert project_id_problem("a|b")


def test_read_image_handles_chinese_paths(tmp_path):
    import cv2

    folder = tmp_path / "中文目录"
    folder.mkdir()
    ok, encoded = cv2.imencode(".png", np.full((4, 5, 3), 7, dtype=np.uint8))
    assert ok
    (folder / "图.png").write_bytes(encoded.tobytes())
    image = read_image(folder / "图.png")
    assert image is not None and image.shape == (4, 5, 3)


def test_image_only_project_uses_the_configured_time_base():
    clip = service.ClipAnnotation(clip_id="a", frame_count=10, fps=0.0, width=64, height=48, segments=[], excluded=[])
    assert service.time_base([clip], 12.5) == 12.5


def test_zero_confidence_threshold_is_kept():
    from lstm_workbench.feature_store import conf_threshold_of

    assert conf_threshold_of({"yolo_conf_threshold": 0.0}) == 0.0
    assert conf_threshold_of({}) == 0.8
