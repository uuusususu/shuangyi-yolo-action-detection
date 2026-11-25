"""HTTP 协议辅助、本机/局域网地址探测与安全边界工具。"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import unquote, urlparse

from . import settings as settings_module
from .project_store import ProjectStore

LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}
UPLOAD_CHUNK = 1024 * 1024


class BadRequest(Exception):
    """请求本身有问题（JSON 坏了、字段不对），回 400 给页面。"""


def _host_name(netloc: str) -> str:
    """Host 头或 Origin 的 netloc 去掉端口，只留主机名。"""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def _valid_name(name: str) -> bool:
    """轮次名、片子 id 这类单层名字：不能空、不能是 . / ..，不能带路径分隔符。"""
    return bool(name) and name not in {".", ".."} and not any(ch in name for ch in "/\\:\0")


def _path_parts(raw_path: str) -> list[str]:
    """把请求路径拆成段，并逐段解码。

    必须先拆再解码：段里的 %2F 解码后是名字里的 "/"，不能变成新的路径层级。
    """
    return [unquote(part) for part in urlparse(raw_path).path.strip("/").split("/")]


def _parse_range(header: str, file_size: int) -> tuple[int, int] | None:
    """解析单段 bytes=start-end。格式不对或越界返回 None，由调用方回 416。"""
    try:
        start_text, end_text = header.removeprefix("bytes=").split("-", 1)
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                return None
            return max(0, file_size - suffix), file_size - 1
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    end = min(end, file_size - 1)
    if start < 0 or start > end:
        return None
    return start, end


def detect_lan_ips() -> list[str]:
    """探测本机可用的局域网 IPv4 地址（优先物理网卡常用的 192.168 / 10 网段）。"""
    found: list[str] = []

    def add_candidate(raw: str) -> None:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return
        if ip.version != 4 or ip.is_loopback or ip.is_unspecified or ip.is_multicast:
            return
        if not (ip.is_private or ip.is_link_local):
            return
        text = str(ip)
        if text not in found:
            found.append(text)

    for target in (("192.168.1.1", 80), ("10.255.255.255", 1), ("8.8.8.8", 80)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(0.2)
            sock.connect(target)
            add_candidate(sock.getsockname()[0])
        except OSError:
            pass
        finally:
            sock.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            add_candidate(info[4][0])
    except OSError:
        pass

    def priority(addr: str) -> tuple[int, str]:
        if addr.startswith("192.168."):
            return (0, addr)
        if addr.startswith("10."):
            return (1, addr)
        if addr.startswith("172."):
            return (2, addr)
        return (3, addr)

    return sorted(found, key=priority)


def _is_allowed_host(host_only: str, lan_enabled: bool) -> bool:
    """校验 Host / Origin 主机名：始终拦截外部公网域名以防止 DNS 重绑定与跨站伪造。"""
    if not host_only:
        return False
    if host_only in LOOPBACK_NAMES:
        return True
    if not lan_enabled:
        return False
    try:
        ip = ipaddress.ip_address(host_only)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        pass
    try:
        local_hostname = socket.gethostname().strip().lower()
        if local_hostname and host_only in {local_hostname, f"{local_hostname}.local"}:
            return True
    except OSError:
        pass
    return False


def is_loopback_client(client_ip: str) -> bool:
    """判断客户端连接是否来自本机回环地址。"""
    raw = str(client_ip or "").strip()
    if raw in {"127.0.0.1", "::1"}:
        return True
    if raw.startswith("::ffff:"):
        raw = raw[7:]
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def read_settings(store: ProjectStore, project_id: str) -> dict:
    """文件里可能缺新加的项，读出来一律补全。"""
    return settings_module.clean({}, store.load_settings(project_id))
