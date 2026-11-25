"""内网轻量保护（PIN 口令、HMAC Cookie、只读模式）与模块化路由集成测试。"""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from lstm_workbench import app as app_module
from lstm_workbench.app import serve
from lstm_workbench.lan_auth import LanAuthManager


def test_lan_auth_manager_pin_hashing_tokens_and_rate_limit(tmp_path):
    cfg = tmp_path / ".lan_auth.json"
    auth = LanAuthManager(config_path=cfg)
    assert auth.pin_enabled is False
    assert auth.readonly is False

    with pytest.raises(ValueError, match="4 到 32 位"):
        auth.configure(pin="123")

    st = auth.configure(pin="246810", readonly=True)
    assert st["pin_enabled"] is True
    assert st["readonly"] is True
    raw_file = cfg.read_text(encoding="utf-8")
    assert "246810" not in raw_file, "磁盘不得存明文 PIN"

    # 重建实例应从磁盘恢复同一套策略
    reloaded = LanAuthManager(config_path=cfg)
    assert reloaded.pin_enabled is True
    assert reloaded.readonly is True
    assert reloaded.verify_pin("246810", "192.168.1.20") is True

    token = reloaded.issue_token()
    assert reloaded.verify_token(token) is True
    assert reloaded.verify_token(token + "x") is False

    # 更换口令后旧会话令牌立即失效
    reloaded.configure(pin="654321")
    assert reloaded.verify_token(token) is False

    # 连续错试触发频控
    for _ in range(8):
        assert reloaded.verify_pin("000000", "192.168.1.99") is False
    with pytest.raises(ValueError, match="频繁"):
        reloaded.verify_pin("654321", "192.168.1.99")

    # 关闭口令后恢复免密
    reloaded.configure(clear_pin=True, readonly=False)
    assert reloaded.pin_enabled is False
    assert reloaded.readonly is False


def test_lan_pin_and_readonly_http_enforcement(tmp_path, monkeypatch):
    auth_file = tmp_path / ".lan_auth.json"
    server = serve(
        tmp_path / "projects",
        port=0,
        open_browser=False,
        host="0.0.0.0",
        auth_config_path=auth_file,
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    simulated_local = {"value": True}
    monkeypatch.setattr(app_module, "is_loopback_client", lambda _ip: simulated_local["value"])

    def call(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        hdrs = {"Host": f"192.168.1.50:{port}", **(headers or {})}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        resp_headers = dict(resp.getheaders())
        conn.close()
        return resp.status, raw, resp_headers

    try:
        # 1. 本机配置开启 PIN 口令
        simulated_local["value"] = True
        code, raw, _ = call("POST", "/api/auth/config", {"pin": "889900", "readonly": False})
        assert code == 200
        assert json.loads(raw)["pin_enabled"] is True

        # 本机不受 PIN 拦截
        assert call("GET", "/api/projects")[0] == 200

        # 2. 切换为远端内网客户端：首页与 server-info 可访问，业务接口被 401 拦下
        simulated_local["value"] = False
        assert call("GET", "/")[0] == 200
        code, info_raw, _ = call("GET", "/api/server-info")
        assert code == 200
        info = json.loads(info_raw)
        assert info["pin_enabled"] is True
        assert info["auth_required"] is True

        code, blocked_raw, _ = call("GET", "/api/projects")
        assert code == 401
        assert json.loads(blocked_raw)["auth_required"] is True

        # 远端不能直接改安全配置（未解锁前 401，解锁后也必须 403）
        assert call("POST", "/api/auth/config", {"clear_pin": True})[0] == 401

        # 错口令返回 401
        assert call("POST", "/api/auth/unlock", {"pin": "111111"})[0] == 401

        # 正确口令下发 HttpOnly Cookie
        code, _, hdrs = call("POST", "/api/auth/unlock", {"pin": "889900"})
        assert code == 200
        set_cookie = hdrs.get("Set-Cookie") or ""
        assert "wb_lan_session=" in set_cookie and "HttpOnly" in set_cookie
        cookie_pair = set_cookie.split(";", 1)[0]

        # 携带 Cookie 访问业务接口成功，且仍不能修改主机安全配置
        assert call("GET", "/api/projects", headers={"Cookie": cookie_pair})[0] == 200
        assert call("POST", "/api/auth/config", {"clear_pin": True}, headers={"Cookie": cookie_pair})[0] == 403
        assert call("POST", "/api/projects", {"id": "from_lan"}, headers={"Cookie": cookie_pair})[0] == 201

        # 3. 本机开启内网只读模式：远端读 200，写 403，本机写依然 201
        simulated_local["value"] = True
        assert call("POST", "/api/auth/config", {"readonly": True})[0] == 200

        simulated_local["value"] = False
        assert call("GET", "/api/projects", headers={"Cookie": cookie_pair})[0] == 200
        code, ro_raw, _ = call("POST", "/api/projects", {"id": "blocked_write"}, headers={"Cookie": cookie_pair})
        assert code == 403
        assert json.loads(ro_raw)["readonly"] is True

        simulated_local["value"] = True
        assert call("POST", "/api/projects", {"id": "local_write"})[0] == 201
    finally:
        server.shutdown()
