"""内网轻量安全保护：可选访问口令（PIN）、HMAC 会话 Cookie 与内网只读模式。

原则：
- 本机回环访问（127.0.0.1 / ::1）始终免口令且拥有完整控制权，并独占安全策略配置权；
- 局域网远端设备访问时，若开启了 PIN 保护，则未解锁前仅可读取静态页面与鉴权接口；
- 若开启了内网只读模式，远端设备可浏览页面、看训练曲线与下载 ZIP，但禁止写操作/删数据/停训练。
"""
from __future__ import annotations

import hashlib
import hmac
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import secrets
import threading
import time

from . import model_registry
from .project_store import ProjectStore

SESSION_COOKIE = "wb_lan_session"
_MAX_FAILS = 8
_WINDOW_SECONDS = 60.0
_SESSION_TTL = 12 * 3600


def _hash_pin(pin: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt.encode("ascii"), 100_000)
    return digest.hex()


class LanAuthManager:
    """管理本机持久化的内网访问策略（PIN 哈希 + 只读开关 + 会话签发与防暴力尝试）。"""

    def __init__(self, config_path: Path | None = None, *, initial_pin: str = "", initial_readonly: bool | None = None):
        self._path = config_path
        self._lock = threading.RLock()
        self._secret = secrets.token_hex(32)
        self._salt = ""
        self._pin_hash = ""
        self._readonly = False
        self._session_epoch = 1
        self._failures: dict[str, list[float]] = {}
        self._load_persisted()

        env_pin = (initial_pin or os.environ.get("LSTM_WORKBENCH_LAN_PIN") or "").strip()
        if env_pin:
            self._apply_pin_in_memory(env_pin)
        if initial_readonly is not None:
            self._readonly = bool(initial_readonly)
        elif os.environ.get("LSTM_WORKBENCH_LAN_READONLY", "").strip().lower() in {"1", "true", "yes"}:
            self._readonly = True

    def _storage_path(self) -> Path:
        if self._path is not None:
            return self._path
        return model_registry.MODEL_LIBRARY / ".lan_auth.json"

    def _load_persisted(self) -> None:
        path = self._storage_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        salt = str(data.get("salt") or "")
        pin_hash = str(data.get("pin_hash") or "")
        if salt and pin_hash:
            self._salt = salt
            self._pin_hash = pin_hash
        self._readonly = bool(data.get("readonly", False))
        secret = str(data.get("secret") or "")
        if len(secret) >= 32:
            self._secret = secret

    def _save_persisted(self) -> None:
        path = self._storage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ProjectStore.write_json(path, {
            "salt": self._salt,
            "pin_hash": self._pin_hash,
            "readonly": self._readonly,
            "secret": self._secret,
            "updated_at": time.time(),
        })

    def _apply_pin_in_memory(self, pin: str) -> None:
        cleaned = str(pin or "").strip()
        if not cleaned:
            self._salt = ""
            self._pin_hash = ""
            self._session_epoch += 1
            return
        if len(cleaned) < 4 or len(cleaned) > 32 or any(ch in cleaned for ch in "\r\n\t\x00"):
            raise ValueError("内网访问口令（PIN）长度应为 4 到 32 位，且不能包含换行符")
        salt = secrets.token_hex(16)
        self._salt = salt
        self._pin_hash = _hash_pin(cleaned, salt)
        self._session_epoch += 1

    @property
    def pin_enabled(self) -> bool:
        with self._lock:
            return bool(self._pin_hash and self._salt)

    @property
    def readonly(self) -> bool:
        with self._lock:
            return self._readonly

    def status(self, *, is_local: bool, headers=None) -> dict:
        with self._lock:
            pin_on = bool(self._pin_hash and self._salt)
            unlocked = True if (is_local or not pin_on) else self.is_request_unlocked(headers)
            return {
                "pin_enabled": pin_on,
                "readonly": self._readonly,
                "is_local": is_local,
                "unlocked": unlocked,
                "auth_required": (not is_local) and pin_on and (not unlocked),
            }

    def configure(self, *, pin: str | None = None, clear_pin: bool = False, readonly: bool | None = None) -> dict:
        with self._lock:
            if clear_pin:
                self._apply_pin_in_memory("")
            elif pin is not None and str(pin).strip() != "":
                self._apply_pin_in_memory(str(pin))
            if readonly is not None:
                if not isinstance(readonly, bool):
                    raise ValueError("只读开关必须是布尔值")
                self._readonly = readonly
            self._save_persisted()
            return self.status(is_local=True)

    def _check_rate_limit(self, client_ip: str) -> None:
        now = time.monotonic()
        bucket = [t for t in self._failures.get(client_ip, []) if now - t < _WINDOW_SECONDS]
        self._failures[client_ip] = bucket
        if len(bucket) >= _MAX_FAILS:
            raise ValueError("口令尝试过于频繁，请 1 分钟后再试")

    def _record_failure(self, client_ip: str) -> None:
        now = time.monotonic()
        bucket = [t for t in self._failures.get(client_ip, []) if now - t < _WINDOW_SECONDS]
        bucket.append(now)
        self._failures[client_ip] = bucket

    def verify_pin(self, candidate: str, client_ip: str = "") -> bool:
        with self._lock:
            if not self._pin_hash or not self._salt:
                return True
            self._check_rate_limit(client_ip)
            cleaned = str(candidate or "").strip()
            expected = _hash_pin(cleaned, self._salt)
            ok = hmac.compare_digest(expected, self._pin_hash)
            if ok:
                self._failures.pop(client_ip, None)
            else:
                self._record_failure(client_ip)
            return ok

    def issue_token(self) -> str:
        with self._lock:
            expires = int(time.time()) + _SESSION_TTL
            payload = f"{self._session_epoch}:{expires}"
            sig = hmac.new(self._secret.encode("ascii"), payload.encode("ascii"), hashlib.sha256).hexdigest()
            return f"{payload}:{sig}"

    def verify_token(self, token: str) -> bool:
        with self._lock:
            if not self._pin_hash:
                return True
            parts = str(token or "").strip().split(":")
            if len(parts) != 3:
                return False
            epoch_text, exp_text, sig = parts
            try:
                epoch = int(epoch_text)
                expires = int(exp_text)
            except ValueError:
                return False
            if epoch != self._session_epoch or expires < int(time.time()):
                return False
            payload = f"{epoch}:{expires}"
            expected = hmac.new(self._secret.encode("ascii"), payload.encode("ascii"), hashlib.sha256).hexdigest()
            return hmac.compare_digest(sig, expected)

    def _extract_cookie_token(self, headers) -> str:
        if headers is None:
            return ""
        raw_cookie = headers.get("Cookie") or ""
        if not raw_cookie:
            return ""
        jar = SimpleCookie()
        try:
            jar.load(raw_cookie)
        except Exception:
            return ""
        morsel = jar.get(SESSION_COOKIE)
        return morsel.value if morsel is not None else ""

    def is_request_unlocked(self, headers, client_ip: str = "") -> bool:
        with self._lock:
            if not self._pin_hash:
                return True
            token = self._extract_cookie_token(headers)
            if token and self.verify_token(token):
                return True
            header_pin = (headers.get("X-Workbench-Pin") or "").strip() if headers is not None else ""
            if header_pin:
                try:
                    return self.verify_pin(header_pin, client_ip=client_ip)
                except ValueError:
                    return False
            return False

    def cookie_header(self, token: str | None) -> str:
        if not token:
            return f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
        return f"{SESSION_COOKIE}={token}; Path=/; Max-Age={_SESSION_TTL}; HttpOnly; SameSite=Strict"
