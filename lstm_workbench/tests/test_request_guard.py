"""训练台只接受本机页面的请求：跨站、DNS 重绑定、非 JSON 写操作一律拒绝。"""
from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager

from lstm_workbench.app import _parse_range, serve


@contextmanager
def _server(tmp_path):
    server = serve(tmp_path, port=0, open_browser=False)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, body=b"", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
        conn.request(method, path, body=body, headers=hdrs)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data, dict(response.getheaders())

    try:
        yield call, port
    finally:
        server.shutdown()


def _json(payload):
    return json.dumps(payload).encode("utf-8")


def test_cross_site_text_plain_post_is_refused(tmp_path):
    with _server(tmp_path) as (call, _port):
        status, _, _ = call("POST", "/api/projects", _json({"id": "evil"}), {"Content-Type": "text/plain"})
        assert status == 403
        assert not (tmp_path / "evil").exists()


def test_foreign_origin_is_refused_even_with_json(tmp_path):
    with _server(tmp_path) as (call, _port):
        status, _, _ = call(
            "POST", "/api/projects", _json({"id": "evil"}),
            {"Content-Type": "application/json", "Origin": "https://attacker.example"},
        )
        assert status == 403
        assert not (tmp_path / "evil").exists()


def test_dns_rebinding_host_is_refused(tmp_path):
    with _server(tmp_path) as (call, port):
        status, _, _ = call("GET", "/api/projects", headers={"Host": f"attacker.example:{port}"})
        assert status == 403


def test_local_page_requests_still_work(tmp_path):
    with _server(tmp_path) as (call, port):
        status, _, _ = call(
            "POST", "/api/projects", _json({"id": "local"}),
            {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"},
        )
        assert status == 201
        status, data, _ = call("GET", "/api/projects", headers={"Host": f"localhost:{port}"})
        assert status == 200
        assert json.loads(data)["projects"] == ["local"]


def test_project_id_with_path_is_not_resolved(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "annotations.json").write_text("{}", encoding="utf-8")
    with _server(tmp_path / "projects") as (call, _port):
        status, _, _ = call("GET", "/api/projects/..")
        assert status in {403, 404}
        status, _, _ = call("GET", "/api/projects/..%2Foutside")
        assert status == 404


def test_media_range_is_streamed_and_bad_range_is_416(tmp_path):
    with _server(tmp_path) as (call, _port):
        assert call("POST", "/api/projects", _json({"id": "p"}), {"Content-Type": "application/json"})[0] == 201
        media = tmp_path / "p" / "media"
        (media / "clip.bin").write_bytes(bytes(range(256)) * 4)

        status, data, headers = call("GET", "/api/projects/p/file/clip.bin", headers={"Range": "bytes=10-19"})
        assert status == 206
        assert data == bytes(range(10, 20))
        assert headers["Content-Range"] == "bytes 10-19/1024"

        status, data, _ = call("GET", "/api/projects/p/file/clip.bin", headers={"Range": "bytes=-4"})
        assert status == 206
        assert data == bytes(range(252, 256))

        status, _, headers = call("GET", "/api/projects/p/file/clip.bin", headers={"Range": "bytes=abc-"})
        assert status == 416
        assert headers["Content-Range"] == "bytes */1024"


def test_parse_range_rejects_malformed_and_out_of_bounds():
    assert _parse_range("bytes=0-9", 100) == (0, 9)
    assert _parse_range("bytes=90-", 100) == (90, 99)
    assert _parse_range("bytes=95-500", 100) == (95, 99)
    assert _parse_range("bytes=-10", 100) == (90, 99)
    assert _parse_range("bytes=x-1", 100) is None
    assert _parse_range("bytes=100-", 100) is None
    assert _parse_range("bytes=5", 100) is None
