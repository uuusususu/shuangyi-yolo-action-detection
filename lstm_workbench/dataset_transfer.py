"""数据集与底模远程拉取（ul://、HTTP/HTTPS ZIP/NDJSON）、命令智能解析、ZIP 打包与异步导入进度管理。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import uuid
import zipfile

import cv2
import numpy as np

from . import model_registry, yolo_jobs
from .parts_store import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PartsStore
from .project_store import ProjectStore

PLATFORM_URL = os.environ.get("ULTRALYTICS_PLATFORM_URL", "https://platform.ultralytics.com").rstrip("/")
PLATFORM_API_URL = os.environ.get("PLATFORM_API_URL", f"{PLATFORM_URL}/api/webhooks").rstrip("/")
GITHUB_ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

_SAFE_FILE = re.compile(r"^[\w.\-\u4e00-\u9fff]+$")
_ENV_KEY_PATTERNS = (
    re.compile(r"""(?:export|set)\s+ULTRALYTICS_API_KEY\s*=\s*['"]?([^\s'";]+)['"]?""", re.I),
    re.compile(r"""\$env:ULTRALYTICS_API_KEY\s*=\s*['"]?([^\s'";]+)['"]?""", re.I),
    re.compile(r"""(?:yolo|ul)\s+login\s+['"]?([A-Za-z0-9_\-]{16,128})['"]?""", re.I),
    re.compile(r"""\bULTRALYTICS_API_KEY\s*=\s*['"]?([^\s'";]+)['"]?""", re.I),
)


class TransferCancelled(RuntimeError):
    """用户主动取消下载或导入任务。"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def mask_api_key(key: str) -> str:
    text = (key or "").strip()
    if not text:
        return ""
    if len(text) <= 10:
        return text[:2] + "••••" + text[-2:]
    return f"{text[:6]}••••••••{text[-4:]}"


def _credentials_path() -> Path:
    return model_registry.MODEL_LIBRARY / ".ultralytics_credentials.json"


def get_credentials_info() -> dict:
    env_key = os.environ.get("ULTRALYTICS_API_KEY", "").strip()
    saved_key = _read_saved_key()
    effective = saved_key or env_key
    source = "saved" if saved_key else ("env" if env_key else "none")
    return {
        "has_key": bool(effective),
        "masked": mask_api_key(effective),
        "source": source,
    }


def _read_saved_key() -> str:
    path = _credentials_path()
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = payload.get("ultralytics_api_key") if isinstance(payload, dict) else ""
    return value.strip() if isinstance(value, str) else ""


def save_api_key(api_key: str) -> dict:
    cleaned = str(api_key or "").strip().strip("'\"")
    if not cleaned:
        path = _credentials_path()
        path.unlink(missing_ok=True)
        return get_credentials_info()
    if len(cleaned) < 8 or len(cleaned) > 256 or any(ch in cleaned for ch in "\r\n\t\x00 "):
        raise ValueError("API Key 格式不合法，请粘贴完整的 ULTRALYTICS_API_KEY")
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ProjectStore.write_json(path, {"ultralytics_api_key": cleaned, "updated_at": time.time()})
    return get_credentials_info()


def resolve_effective_api_key(explicit: str = "", remember: bool = True) -> str:
    cleaned = str(explicit or "").strip().strip("'\"")
    if cleaned:
        if remember:
            save_api_key(cleaned)
        return cleaned
    return _read_saved_key() or os.environ.get("ULTRALYTICS_API_KEY", "").strip()


def normalize_platform_uri(uri: str) -> str:
    text = str(uri or "").strip().strip("'\"")
    prefix = PLATFORM_URL + "/"
    if text.lower().startswith(prefix.lower()):
        rest = text[len(prefix):].strip("/")
        if rest:
            return f"ul://{rest}"
    return text


def _model_base_filename(model_uri: str) -> str:
    text = normalize_platform_uri(model_uri)
    if text.lower().startswith("ul://"):
        leaf = text[5:].strip("/").split("/")[-1]
    else:
        leaf = Path(text).name
    if not leaf:
        return ""
    if not leaf.lower().endswith((".pt", ".onnx")):
        leaf = f"{leaf}.pt"
    return leaf


def parse_ultralytics_input(raw_text: str) -> dict:
    """解析整段 Ultralytics 命令、ul:// 链接、HTTP 下载链接或本机路径。"""
    text = str(raw_text or "").strip()
    if not text:
        return {
            "kind": "empty",
            "api_key": "",
            "api_key_masked": "",
            "data": "",
            "model": "",
            "model_filename": "",
            "params": {},
        }

    api_key = ""
    for pattern in _ENV_KEY_PATTERNS:
        match = pattern.search(text)
        if match:
            api_key = match.group(1).strip().strip("'\"")
            break

    # 将多行反斜杠续行合并，并拆分单行中的 && / ;，便于解析命令行参数
    joined = re.sub(r"\\\r?\n", " ", text)
    lines = [
        segment.strip()
        for raw_line in joined.splitlines()
        for segment in re.split(r"&&|;", raw_line)
        if segment.strip()
    ]

    data_value = ""
    model_value = ""
    params: dict[str, object] = {}

    kv_pairs: dict[str, str] = {}
    for line in lines:
        if re.match(r"^(?:export|set|\$env:)\b", line, re.I):
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            tokens = line.split()
        for idx, tok in enumerate(tokens):
            if "=" in tok and not tok.startswith(("http://", "https://", "ul://")):
                k, v = tok.split("=", 1)
                k_low = k.strip().lower()
                v_clean = v.strip().strip("'\"")
                if k_low and v_clean:
                    kv_pairs[k_low] = v_clean
            elif idx in {1, 2} and tok.lower() in {"obb", "detect", "segment"} and "task" not in kv_pairs:
                kv_pairs["task"] = tok.lower()

    if "data" in kv_pairs:
        data_value = normalize_platform_uri(kv_pairs["data"])
    if "model" in kv_pairs:
        model_value = normalize_platform_uri(kv_pairs["model"])
    for key in ("task", "project", "name", "device"):
        if key in kv_pairs:
            params[key] = kv_pairs[key]
    for int_key in ("batch", "epochs", "imgsz", "workers"):
        if int_key in kv_pairs:
            try:
                params[int_key] = int(kv_pairs[int_key])
            except ValueError:
                pass

    # 如果不是 key=value 形式，则查找独立的链接或本机路径
    if not data_value:
        ul_match = re.search(r"(ul://[^\s'\";]+|https?://[^\s'\";]+)", text, re.I)
        if ul_match:
            candidate = normalize_platform_uri(ul_match.group(1))
            if candidate.lower().startswith("ul://") and "/datasets/" not in candidate.lower() and len(candidate[5:].split("/")) == 3:
                model_value = model_value or candidate
            else:
                data_value = candidate
        else:
            # 去掉环境变量行后剩余的第一行作为本地路径
            non_env = [
                ln for ln in lines
                if not re.match(r"^(?:export|set|\$env:|yolo\s+login|ul\s+login)\b", ln, re.I)
            ]
            if non_env:
                data_value = non_env[-1].strip().strip("'\"")

    # 若 model 名里带 -obb 且未显式写 task，自动推断 task=obb
    if "task" not in params and "-obb" in (model_value or "").lower():
        params["task"] = "obb"

    kind = _classify_source(data_value)
    return {
        "kind": kind,
        "api_key": api_key,
        "api_key_masked": mask_api_key(api_key),
        "data": data_value,
        "model": model_value,
        "model_filename": _model_base_filename(model_value) if model_value else "",
        "params": params,
    }


def _classify_source(source: str) -> str:
    s = (source or "").strip()
    low = s.lower()
    if not s:
        return "empty"
    if low.startswith("ul://"):
        return "ul_dataset" if "/datasets/" in low else "ul_model"
    if low.startswith(("http://", "https://")):
        clean = low.split("?", 1)[0]
        if clean.endswith(".ndjson"):
            return "http_ndjson"
        return "http_zip"
    if low.endswith(".ndjson"):
        return "local_ndjson"
    if low.endswith(".zip"):
        return "local_zip"
    return "local_path"


def resolve_platform_signed_url(uri: str, api_key: str, *, base_url: str | None = None) -> str:
    """将 ul:// URI 解析为平台签名的临时 HTTPS 下载地址。"""
    normalized = normalize_platform_uri(uri)
    if not normalized.lower().startswith("ul://"):
        raise ValueError(f"不是合法的 ul:// 地址：{uri}")
    parts = [p for p in normalized[5:].strip("/").split("/") if p]
    if len(parts) != 3 or any(p in {".", ".."} for p in parts):
        raise ValueError(f"不支持的 ul:// 格式：{uri}（应为 ul://用户名/datasets/数据集名 或 ul://用户名/项目/模型名）")

    api_base = (base_url or PLATFORM_API_URL).rstrip("/")
    if parts[1] == "datasets":
        username, _, slug = parts
        endpoint = f"{api_base}/datasets/{username}/{slug}/export"
    else:
        username, project, model = parts
        endpoint = f"{api_base}/models/{username}/{project}/{model}/download"

    if not api_key:
        # 官方公共底模（如 ul://ultralytics/yolo26/yolo26n-obb）允许无 Key 回退到官方 release
        if parts[0].lower() == "ultralytics" and parts[1] != "datasets":
            return f"{GITHUB_ASSETS_URL}/{_model_base_filename(normalized)}"
        raise ValueError(f"拉取 {normalized} 需要填写 ULTRALYTICS_API_KEY")

    opener = build_opener(_NoRedirect)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Ultralytics-Workbench/1.0",
    }
    timeout = 90 if parts[1] == "datasets" else 45
    last_exc: Exception | None = None
    for attempt in range(3):
        req = Request(endpoint, method="HEAD", headers=headers)
        try:
            resp = opener.open(req, timeout=timeout)
            status = getattr(resp, "status", resp.getcode())
            resp_headers = resp.headers
            resp.close()
        except HTTPError as exc:
            status = exc.code
            resp_headers = exc.headers
            exc.close()
            if status in {301, 302, 303, 307, 308}:
                location = resp_headers.get("Location") or resp_headers.get("location")
                if location:
                    return location
            if status == 401:
                raise ValueError("ULTRALYTICS_API_KEY 无效或已过期（HTTP 401）") from None
            if status == 403:
                raise ValueError(f"无权访问平台资源 {normalized}（HTTP 403），请核对账号权限或私有设置") from None
            if status == 404:
                if parts[0].lower() == "ultralytics" and parts[1] != "datasets":
                    return f"{GITHUB_ASSETS_URL}/{_model_base_filename(normalized)}"
                raise FileNotFoundError(f"平台上找不到资源：{normalized}（HTTP 404）") from None
            if status == 409:
                raise RuntimeError(f"平台数据集正在后台生成导出包，请稍后重试：{normalized}（HTTP 409）") from None
            last_exc = exc
            if status not in {408, 429} and status < 500:
                raise RuntimeError(f"平台接口返回异常 HTTP {status}：{normalized}") from None
        except (URLError, TimeoutError, OSError) as exc:
            last_exc = exc
        else:
            if 300 <= status < 400:
                location = resp_headers.get("Location") or resp_headers.get("location")
                if location:
                    return location
            break
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    if parts[0].lower() == "ultralytics" and parts[1] != "datasets":
        return f"{GITHUB_ASSETS_URL}/{_model_base_filename(normalized)}"
    raise RuntimeError(f"无法连接 Ultralytics 平台解析 {normalized}：{last_exc or '未知响应'}")


def downloads_dir(project_folder: Path) -> Path:
    target = (Path(project_folder).resolve() / "downloads").resolve()
    if Path(project_folder).resolve() not in target.parents:
        raise ValueError("下载目录路径越界")
    return target


def list_downloads(project_folder: Path) -> list[dict]:
    folder = downloads_dir(project_folder)
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".zip" or path.name.startswith("."):
            continue
        stat = path.stat()
        items.append({
            "name": path.name,
            "bytes": int(stat.st_size),
            "mtime": int(stat.st_mtime),
        })
    return items


def download_archive_path(project_folder: Path, filename: str) -> Path:
    name = str(filename or "").strip()
    if not name or not _SAFE_FILE.fullmatch(name) or not name.lower().endswith(".zip") or name.startswith("."):
        raise ValueError("压缩包文件名不合法")
    folder = downloads_dir(project_folder)
    target = (folder / name).resolve()
    if folder not in target.parents or not target.is_file() or target.is_symlink():
        raise FileNotFoundError("找不到这个 ZIP 数据集文件")
    return target


def _stream_url_to_file(
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    should_stop=None,
    on_bytes=None,
    timeout: int = 60,
) -> int:
    req_headers = {"User-Agent": "Ultralytics-Workbench/1.0", **(headers or {})}
    req = Request(url, headers=req_headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
    written = 0
    try:
        with urlopen(req, timeout=timeout) as resp:
            total_header = resp.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            with tmp.open("wb") as out:
                while True:
                    if should_stop is not None and should_stop():
                        raise TransferCancelled("已取消下载")
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if on_bytes is not None:
                        on_bytes(written, total)
        os.replace(tmp, dest)
        return written
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _safe_slug(text: str, fallback: str = "dataset") -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", str(text or "").strip()).strip("-_")
    return cleaned[:80] or fallback


def _pack_directory_to_zip(source_dir: Path, zip_target: Path, *, should_stop=None, on_progress=None) -> int:
    files = [p for p in sorted(source_dir.rglob("*")) if p.is_file() and not p.is_symlink()]
    zip_target.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_target.with_name(f".{zip_target.name}.{uuid.uuid4().hex[:8]}.tmp")
    total = max(len(files), 1)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for idx, file_path in enumerate(files, 1):
                if should_stop is not None and should_stop():
                    raise TransferCancelled("已取消打包 ZIP")
                arcname = file_path.relative_to(source_dir).as_posix()
                zf.write(file_path, arcname=arcname)
                if on_progress is not None and (idx == len(files) or idx % 10 == 0):
                    on_progress(idx, total)
        os.replace(tmp, zip_target)
        return int(zip_target.stat().st_size)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _safe_extract_zip(zip_path: Path, extract_to: Path, *, should_stop=None, on_progress=None) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"文件不是有效的 ZIP 压缩包：{zip_path.name}")
    extract_root = extract_to.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) > 50000:
            raise ValueError("压缩包内文件数量超过安全上限（50000）")
        total = max(len(members), 1)
        for idx, info in enumerate(members, 1):
            if should_stop is not None and should_stop():
                raise TransferCancelled("已取消解压")
            raw_name = info.filename.replace("\\", "/")
            if not raw_name or raw_name.startswith("/") or ":" in raw_name:
                raise ValueError(f"压缩包包含越界路径：{info.filename}")
            rel = Path(raw_name)
            if any(part in {"", ".", ".."} for part in rel.parts):
                raise ValueError(f"压缩包包含不安全路径：{info.filename}")
            # 检查 Unix symlink 标志位
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise ValueError(f"压缩包包含符号链接，拒绝解压：{info.filename}")
            target = (extract_root / rel).resolve()
            if extract_root not in target.parents:
                raise ValueError(f"压缩包解压路径越界：{info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=256 * 1024)
            if on_progress is not None and (idx == len(members) or idx % 15 == 0):
                on_progress(idx, total)


def _find_yaml_in_extracted(root: Path) -> Path | None:
    for candidate in (root / "data.yaml", root / "data.yml"):
        if candidate.is_file():
            return candidate
    yamls = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml"} and "__MACOSX" not in p.parts
    ]
    for p in yamls:
        if p.name.lower() in {"data.yaml", "data.yml"}:
            return p
    return yamls[0] if len(yamls) == 1 else None


def convert_ndjson_file_to_yolo_dir(
    ndjson_path: Path,
    output_dir: Path,
    *,
    should_stop=None,
    on_image_progress=None,
) -> dict:
    """解析 Ultralytics NDJSON 数据集，并发下载图片并生成标准 YOLO OBB/Detect 目录结构。"""
    lines = [
        json.loads(line.strip())
        for line in ndjson_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not lines or not isinstance(lines[0], dict):
        raise ValueError("NDJSON 数据集为空或格式不正确")
    dataset_record = lines[0]
    image_records = [r for r in lines[1:] if isinstance(r, dict) and r.get("type", "image") == "image"]
    if not image_records:
        raise ValueError("NDJSON 数据集中没有任何图片记录")

    task = str(dataset_record.get("task") or "obb").strip().lower()
    if task not in yolo_jobs.TASKS:
        raise ValueError(f"当前工作台仅支持 obb / detect / segment 数据集，收到：{task}")

    raw_names = dataset_record.get("class_names") or dataset_record.get("names") or {}
    class_map: dict[int, str] = {}
    if isinstance(raw_names, dict):
        for k, v in raw_names.items():
            class_map[int(k)] = str(v).strip()
    elif isinstance(raw_names, list):
        for i, v in enumerate(raw_names):
            class_map[i] = str(v).strip()

    # 扫描所有标注中的最大类别编号，补齐缺省名
    max_cls = max(class_map.keys(), default=-1)
    for rec in image_records:
        ann = rec.get("annotations") or {}
        if isinstance(ann, dict):
            for items in ann.values():
                if isinstance(items, list):
                    for row in items:
                        if isinstance(row, list) and row:
                            max_cls = max(max_cls, int(row[0]))
    if max_cls < 0:
        raise ValueError("NDJSON 数据集缺少类别定义（class_names）")
    names = [class_map.get(i, f"class_{i}") for i in range(max_cls + 1)]

    # 校验并规范 split；若缺少 val，则从 train 中稳定抽取 15%
    normalized_records = []
    for idx, rec in enumerate(image_records, 1):
        split = str(rec.get("split") or "train").strip().lower()
        if split not in {"train", "val", "test"}:
            split = "train"
        raw_file = Path(str(rec.get("file") or f"image_{idx:05d}.jpg")).name
        stem = _safe_slug(Path(raw_file).stem, f"img_{idx:05d}")
        suffix = Path(raw_file).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        safe_name = f"{idx:05d}_{stem}{suffix}"
        normalized_records.append({
            "index": idx,
            "split": split,
            "file": safe_name,
            "orig_file": raw_file,
            "url": str(rec.get("url") or "").strip(),
            "annotations": rec.get("annotations") or {},
        })

    splits = {r["split"] for r in normalized_records}
    if "train" not in splits:
        for r in normalized_records:
            r["split"] = "train"
        splits = {"train"}
    if "val" not in splits:
        train_only = [r for r in normalized_records if r["split"] == "train"]
        if len(train_only) >= 2:
            rng = random.Random(0)
            shuffled = list(train_only)
            rng.shuffle(shuffled)
            val_n = max(1, round(len(shuffled) * 0.15))
            for r in shuffled[:val_n]:
                r["split"] = "val"
            splits.add("val")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(splits):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    ndjson_parent = ndjson_path.parent
    total_images = len(normalized_records)
    lock = threading.Lock()
    progress_state = {"done": 0, "bytes": 0}

    def fetch_one(rec: dict) -> int:
        if should_stop is not None and should_stop():
            raise TransferCancelled("已取消图片下载")
        split = rec["split"]
        img_dest = output_dir / "images" / split / rec["file"]
        lbl_dest = output_dir / "labels" / split / f"{Path(rec['file']).stem}.txt"

        # 写标签文件（优先取 obb，其次 boxes / segments）
        ann = rec["annotations"] if isinstance(rec["annotations"], dict) else {}
        rows = None
        for preferred in ("obb", "boxes", "segments"):
            if preferred in ann and isinstance(ann[preferred], list):
                rows = ann[preferred]
                break
        if rows is None:
            for val in ann.values():
                if isinstance(val, list):
                    rows = val
                    break
        label_lines = [" ".join(str(x) for x in item) for item in (rows or []) if isinstance(item, list) and len(item) >= 5]
        lbl_dest.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")

        url = rec["url"]
        size = 0
        if url.startswith(("http://", "https://")):
            last_err: Exception | None = None
            for attempt in range(3):
                if should_stop is not None and should_stop():
                    raise TransferCancelled("已取消图片下载")
                try:
                    req = Request(url, headers={"User-Agent": "Ultralytics-Workbench/1.0"})
                    with urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    if not data:
                        raise ValueError("下载到的图片为空")
                    img_dest.write_bytes(data)
                    size = len(data)
                    last_err = None
                    break
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                    last_err = exc
                    if isinstance(exc, HTTPError) and exc.code not in {408, 429} and exc.code < 500:
                        break
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
            if last_err is not None:
                raise RuntimeError(f"图片 {rec['orig_file']} 下载失败：{last_err}")
        else:
            # 兼容离线本地 NDJSON（同目录或 images 子目录下的本地图片）
            candidates = [
                ndjson_parent / rec["orig_file"],
                ndjson_parent / "images" / split / rec["orig_file"],
                ndjson_parent / "images" / rec["orig_file"],
            ]
            local_img = next((c for c in candidates if c.is_file()), None)
            if local_img is None:
                raise FileNotFoundError(f"NDJSON 记录缺少下载链接且找不到本地图片：{rec['orig_file']}")
            shutil.copyfile(local_img, img_dest)
            size = int(img_dest.stat().st_size)

        with lock:
            progress_state["done"] += 1
            progress_state["bytes"] += size
            done_now = progress_state["done"]
            bytes_now = progress_state["bytes"]
        if on_image_progress is not None:
            on_image_progress(done_now, total_images, bytes_now, rec["orig_file"])
        return size

    workers = max(1, min(6, total_images))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, rec) for rec in normalized_records]
        for fut in as_completed(futures):
            fut.result()

    yaml_lines = [
        f"task: {task}",
        "path: .",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {json.dumps(name, ensure_ascii=False)}")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {
        "data_yaml": data_yaml,
        "name": _safe_slug(str(dataset_record.get("name") or ndjson_path.stem), "dataset"),
        "task": task,
        "names": names,
        "images": total_images,
        "bytes": progress_state["bytes"],
    }


def _yolo_label_line_to_obb_box(line: str, class_count: int) -> dict | None:
    tokens = line.strip().split()
    if len(tokens) not in {5, 9}:
        return None
    try:
        cls = int(tokens[0])
        coords = [float(x) for x in tokens[1:]]
    except ValueError:
        return None
    if not (0 <= cls < class_count) or any(not math.isfinite(v) for v in coords):
        return None
    if len(coords) == 4:
        cx, cy, bw, bh = coords
        x1, y1 = max(0.0, min(1.0, cx - bw / 2.0)), max(0.0, min(1.0, cy - bh / 2.0))
        x2, y2 = max(0.0, min(1.0, cx + bw / 2.0)), max(0.0, min(1.0, cy + bh / 2.0))
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        points = [
            [max(0.0, min(1.0, coords[i])), max(0.0, min(1.0, coords[i + 1]))]
            for i in range(0, 8, 2)
        ]
    poly = np.asarray(points, dtype=np.float32)
    if not cv2.isContourConvex(poly) or abs(cv2.contourArea(poly)) <= 1e-8:
        return None
    return {"class_id": cls, "points": points}


def sync_yolo_dir_to_parts_store(
    project_folder: Path,
    dataset_root: Path,
    names: list[str],
    dataset_slug: str,
) -> dict:
    """将标准 YOLO 数据集同步导入到工程原生 PartsStore 工作区（状态设为 pending 供人工复核）。"""
    store = PartsStore(project_folder)
    snap = store.snapshot()
    existing_classes = list(snap.get("classes") or [])
    if not existing_classes:
        store.set_classes(names, snap["revision"])
        target_classes = names
    elif existing_classes == names[:len(existing_classes)]:
        if len(names) > len(existing_classes):
            store.set_classes(names, snap["revision"])
        target_classes = names
    else:
        # 已有不同顺序的类别时不强行改写类别表，跳过工作区同步，仅保留外部 YOLO 数据集导入
        return {"synced": 0, "skipped_reason": "当前零件标注工作区已有不同顺序的类别，已仅导入至 YOLO 训练数据集"}

    synced = 0
    for split in ("train", "val", "test"):
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.is_dir():
            continue
        before_ids = {item["id"] for item in store.snapshot()["items"]}
        after_snap = store.import_source(str(img_dir), source_group=f"{dataset_slug}:{split}", dedupe=True)
        for item in after_snap["items"]:
            if item["id"] in before_ids:
                continue
            stem = Path(item["name"]).stem
            lbl_file = lbl_dir / f"{stem}.txt"
            boxes = []
            if lbl_file.is_file():
                for line in lbl_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    parsed = _yolo_label_line_to_obb_box(line, len(target_classes))
                    if parsed is not None:
                        boxes.append(parsed)
            if boxes:
                store.save_item(item["id"], boxes, "pending", item["revision"])
            synced += 1
    return {"synced": synced, "skipped_reason": ""}


class TransferManager:
    """管理每个工程的数据集下载、解压打包与异步导入任务进度。"""

    def __init__(self, workbench):
        self.wb = workbench
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._cancels: dict[str, threading.Event] = {}

    def _key(self, project_id: str) -> str:
        return self.wb._key(project_id)

    def status(self, project_id: str) -> dict:
        key = self._key(project_id)
        with self._lock:
            job = self._jobs.get(key)
            return dict(job) if job else {"status": "idle"}

    def cancel(self, project_id: str) -> dict:
        key = self._key(project_id)
        with self._lock:
            ev = self._cancels.get(key)
            if ev is not None:
                ev.set()
            job = self._jobs.get(key)
            return dict(job) if job else {"status": "idle"}

    def _update(self, key: str, **patch) -> None:
        with self._lock:
            current = self._jobs.get(key)
            if current is None:
                return
            current.update(patch)
            started = float(current.get("started_at") or time.time())
            elapsed = max(0.001, time.time() - started)
            current["elapsed"] = round(elapsed, 1)
            b_done = int(current.get("bytes_done") or 0)
            if b_done > 0:
                current["speed_bps"] = int(b_done / elapsed)
            done = float(current.get("done") or 0)
            total = float(current.get("total") or 0)
            if 0 < done < total:
                current["eta_seconds"] = max(1, int(round((total - done) * (elapsed / done))))
            else:
                current["eta_seconds"] = None

    def start_remote_or_local_transfer(self, project_id: str, payload: dict) -> dict:
        folder = self.wb.store.open_dir(project_id)
        raw_input = str(payload.get("source") or payload.get("command") or "").strip()
        if not raw_input:
            raise ValueError("请粘贴 ul:// 命令、下载链接或填写本地路径")

        parsed = parse_ultralytics_input(raw_input)
        explicit_key = str(payload.get("api_key") or parsed.get("api_key") or "").strip()
        remember_key = bool(payload.get("remember_key", True))
        target_mode = str(payload.get("target") or "both").strip()  # 'yolo' | 'parts' | 'both'
        download_model = bool(payload.get("download_model", True))
        source_group = str(payload.get("source_group") or "").strip()
        interval = int(payload.get("interval") or 30)
        max_frames = int(payload.get("max_frames") or 300)
        dedupe = bool(payload.get("dedupe", True))

        key = self._key(project_id)
        with self._lock:
            existing = self._jobs.get(key)
            if existing and existing.get("status") == "running":
                raise RuntimeError("当前工程已有一个下载或导入任务正在执行，请等待完成或先取消")
            cancel_ev = threading.Event()
            job_id = uuid.uuid4().hex[:12]
            record = {
                "id": job_id,
                "project": folder.name,
                "status": "running",
                "phase": "starting",
                "message": "正在解析输入与准备任务…",
                "done": 0,
                "total": 100,
                "unit": "%",
                "bytes_done": 0,
                "bytes_total": 0,
                "speed_bps": 0,
                "eta_seconds": None,
                "started_at": time.time(),
                "finished_at": None,
                "zip_file": "",
                "zip_bytes": 0,
                "downloaded_model": "",
                "parsed": {
                    "kind": parsed["kind"],
                    "data": parsed["data"],
                    "model": parsed["model"],
                    "model_filename": parsed["model_filename"],
                    "params": parsed["params"],
                },
                "result": None,
                "error": "",
            }
            self._cancels[key] = cancel_ev
            self._jobs[key] = record

        thread = threading.Thread(
            target=self._run_transfer,
            name=f"transfer-{folder.name}-{job_id}",
            daemon=True,
            args=(project_id, folder, key, cancel_ev, parsed, explicit_key, remember_key, target_mode, download_model, source_group, interval, max_frames, dedupe),
        )
        thread.start()
        return dict(record)

    def _run_transfer(
        self,
        project_id: str,
        folder: Path,
        key: str,
        cancel_ev: threading.Event,
        parsed: dict,
        explicit_key: str,
        remember_key: bool,
        target_mode: str,
        download_model: bool,
        source_group: str,
        interval: int,
        max_frames: int,
        dedupe: bool,
    ) -> None:
        staging_root = folder / f".transfer-{uuid.uuid4().hex[:10]}"
        try:
            api_key = resolve_effective_api_key(explicit_key, remember=remember_key) if (explicit_key or parsed["kind"].startswith("ul_") or (parsed.get("model") or "").lower().startswith("ul://")) else ""
            downloaded_model_name = ""

            # 1. 如果命令里带有底模（如 model=ul://ultralytics/yolo26/yolo26n-obb），优先检查并下载底模进 models/yolo_base
            model_uri = str(parsed.get("model") or "").strip()
            if download_model and model_uri and (model_uri.lower().startswith(("ul://", "http://", "https://"))):
                model_fname = _model_base_filename(model_uri)
                base_dir = yolo_jobs.base_models_dir()
                base_dir.mkdir(parents=True, exist_ok=True)
                local_model = base_dir / model_fname
                if local_model.is_file():
                    downloaded_model_name = model_fname
                else:
                    self._update(key, phase="model", message=f"正在拉取底模 {model_fname}…", done=0, total=100, unit="%")
                    signed_model_url = resolve_platform_signed_url(model_uri, api_key) if model_uri.lower().startswith("ul://") else model_uri

                    def on_model_bytes(done_b: int, total_b: int) -> None:
                        pct = min(99, int(done_b * 100 / total_b)) if total_b > 0 else 50
                        self._update(
                            key,
                            phase="model",
                            message=f"正在下载底模 {model_fname}（{done_b / 1048576:.1f} MB）",
                            done=pct,
                            total=100,
                            unit="%",
                            bytes_done=done_b,
                            bytes_total=total_b,
                        )

                    _stream_url_to_file(signed_model_url, local_model, should_stop=cancel_ev.is_set, on_bytes=on_model_bytes, timeout=90)
                    downloaded_model_name = model_fname
                    self._update(key, downloaded_model=downloaded_model_name)

            kind = parsed["kind"]
            data_source = str(parsed.get("data") or "").strip()

            # 如果只传了模型链接而没有 data
            if not data_source and downloaded_model_name:
                self._update(
                    key,
                    status="done",
                    phase="done",
                    done=100,
                    total=100,
                    unit="%",
                    message=f"底模 {downloaded_model_name} 已就绪",
                    downloaded_model=downloaded_model_name,
                    finished_at=time.time(),
                )
                return

            if kind in {"ul_dataset", "http_ndjson", "local_ndjson"}:
                staging_root.mkdir(parents=True, exist_ok=True)
                if kind == "ul_dataset":
                    self._update(key, phase="auth", message=f"正在向 Ultralytics 平台鉴权并请求导出 {data_source}…", done=2, total=100, unit="%")
                    signed_url = resolve_platform_signed_url(data_source, api_key)
                    slug = _safe_slug(data_source.rstrip("/").split("/")[-1], "dataset")
                    ndjson_file = staging_root / f"{slug}.ndjson"
                    self._update(key, phase="manifest", message="正在下载数据集标注清单（NDJSON）…")
                    _stream_url_to_file(signed_url, ndjson_file, should_stop=cancel_ev.is_set, timeout=120)
                elif kind == "http_ndjson":
                    slug = _safe_slug(Path(urlparse(data_source).path).stem, "dataset")
                    ndjson_file = staging_root / f"{slug}.ndjson"
                    self._update(key, phase="manifest", message="正在下载 NDJSON 清单…", done=5, total=100, unit="%")
                    _stream_url_to_file(data_source, ndjson_file, should_stop=cancel_ev.is_set, timeout=120)
                else:
                    src_ndjson = Path(data_source).resolve()
                    if not src_ndjson.is_file():
                        raise FileNotFoundError(f"找不到本地 NDJSON 文件：{data_source}")
                    slug = _safe_slug(src_ndjson.stem, "dataset")
                    ndjson_file = src_ndjson

                dataset_out = staging_root / slug

                def on_img(done_i: int, total_i: int, bytes_i: int, cur_name: str) -> None:
                    self._update(
                        key,
                        phase="images",
                        message=f"正在拉取图片与生成标注（{done_i}/{total_i} 张）· {cur_name}",
                        done=done_i,
                        total=total_i,
                        unit="张",
                        bytes_done=bytes_i,
                    )

                self._update(key, phase="images", message="正在解析标注并并发下载数据集图片…", done=0, total=1, unit="张")
                converted = convert_ndjson_file_to_yolo_dir(
                    ndjson_file, dataset_out, should_stop=cancel_ev.is_set, on_image_progress=on_img
                )
                zip_name = f"{converted['name']}.zip"
                zip_dest = downloads_dir(folder) / zip_name
                self._update(key, phase="zip", message=f"正在打包生成标准数据集压缩包 {zip_name}…", done=0, total=1, unit="文件")
                zip_bytes = _pack_directory_to_zip(
                    dataset_out,
                    zip_dest,
                    should_stop=cancel_ev.is_set,
                    on_progress=lambda d, t: self._update(key, phase="zip", done=d, total=t, unit="文件"),
                )
                self._update(key, phase="import", message="正在将数据集导入当前工程…", zip_file=zip_name, zip_bytes=zip_bytes)
                result_info = self._commit_yolo_and_parts(
                    project_id, folder, dataset_out, converted["names"], converted["name"], target_mode
                )
                self._update(
                    key,
                    status="done",
                    phase="done",
                    done=converted["images"],
                    total=converted["images"],
                    unit="张",
                    zip_file=zip_name,
                    zip_bytes=zip_bytes,
                    downloaded_model=downloaded_model_name,
                    result=result_info,
                    message=f"已完成拉取（共 {converted['images']} 张），已生成 {zip_name} 并导入工程",
                    finished_at=time.time(),
                )
                return

            if kind in {"http_zip", "local_zip"}:
                staging_root.mkdir(parents=True, exist_ok=True)
                if kind == "http_zip":
                    raw_leaf = unquote(Path(urlparse(data_source).path).name) or "dataset.zip"
                    slug = _safe_slug(Path(raw_leaf).stem, "dataset")
                    zip_name = f"{slug}.zip"
                    zip_dest = downloads_dir(folder) / zip_name

                    def on_zip_bytes(done_b: int, total_b: int) -> None:
                        pct = min(99, int(done_b * 100 / total_b)) if total_b > 0 else 50
                        self._update(
                            key,
                            phase="downloading",
                            message=f"正在下载压缩包 {zip_name}（{done_b / 1048576:.1f}"
                                    + (f" / {total_b / 1048576:.1f} MB）" if total_b else " MB）"),
                            done=pct,
                            total=100,
                            unit="%",
                            bytes_done=done_b,
                            bytes_total=total_b,
                        )

                    self._update(key, phase="downloading", message=f"正在连接并下载 {zip_name}…", done=0, total=100, unit="%")
                    zip_bytes = _stream_url_to_file(
                        data_source, zip_dest, should_stop=cancel_ev.is_set, on_bytes=on_zip_bytes, timeout=120
                    )
                else:
                    src_zip = Path(data_source).resolve()
                    if not src_zip.is_file():
                        raise FileNotFoundError(f"找不到本地 ZIP 压缩包：{data_source}")
                    slug = _safe_slug(src_zip.stem, "dataset")
                    zip_name = f"{slug}.zip"
                    zip_dest = downloads_dir(folder) / zip_name
                    zip_dest.parent.mkdir(parents=True, exist_ok=True)
                    if src_zip != zip_dest.resolve():
                        shutil.copy2(src_zip, zip_dest)
                    zip_bytes = int(zip_dest.stat().st_size)

                extracted = staging_root / "extracted"
                self._update(key, phase="extracting", message=f"正在安全校验并解压 {zip_name}…", zip_file=zip_name, zip_bytes=zip_bytes)
                _safe_extract_zip(
                    zip_dest,
                    extracted,
                    should_stop=cancel_ev.is_set,
                    on_progress=lambda d, t: self._update(key, phase="extracting", done=d, total=t, unit="文件"),
                )
                result_info = self._import_extracted_folder(
                    project_id, folder, extracted, slug, target_mode, source_group
                )
                self._update(
                    key,
                    status="done",
                    phase="done",
                    done=100,
                    total=100,
                    unit="%",
                    zip_file=zip_name,
                    zip_bytes=zip_bytes,
                    downloaded_model=downloaded_model_name,
                    result=result_info,
                    message=f"压缩包 {zip_name} 解压并导入完成",
                    finished_at=time.time(),
                )
                return

            # 本地目录 / 图片 / 视频 / data.yaml
            src_path = Path(data_source).expanduser()
            if src_path.is_file() and src_path.suffix.lower() in {".yaml", ".yml"} or (src_path.is_dir() and (src_path / "data.yaml").is_file()):
                self._update(key, phase="import", message="正在导入本地 YOLO 数据集并生成归档 ZIP…", done=20, total=100, unit="%")
                yaml_file = src_path if src_path.is_file() else (src_path / "data.yaml")
                base_dir = yaml_file.parent
                slug = _safe_slug(base_dir.name, "dataset")
                zip_name = f"{slug}.zip"
                zip_dest = downloads_dir(folder) / zip_name
                zip_bytes = _pack_directory_to_zip(base_dir, zip_dest, should_stop=cancel_ev.is_set)
                names = yolo_jobs._read_yaml_names(yaml_file.read_text(encoding="utf-8", errors="replace"))
                result_info = self._commit_yolo_and_parts(project_id, folder, base_dir, names, slug, target_mode)
                self._update(
                    key,
                    status="done",
                    phase="done",
                    done=100,
                    total=100,
                    unit="%",
                    zip_file=zip_name,
                    zip_bytes=zip_bytes,
                    result=result_info,
                    message=f"本地 YOLO 数据集已导入，并生成归档 {zip_name}",
                    finished_at=time.time(),
                )
                return

            # 本地图片、图片目录或视频 -> 导入到 PartsStore（带逐帧进度）
            self._update(key, phase="import", message="正在导入本地零件素材…", done=0, total=1, unit="张")

            def on_parts_prog(done_n: int, total_n: int, added_n: int, skipped_n: int) -> None:
                self._update(
                    key,
                    phase="import",
                    done=done_n,
                    total=max(total_n, 1),
                    unit="帧",
                    message=f"正在处理素材（{done_n}/{max(total_n, done_n)}）· 新增 {added_n} 张，去重跳过 {skipped_n} 张",
                )

            with self.wb.data_lock(project_id):
                snap = PartsStore(folder).import_source(
                    data_source,
                    source_group=source_group,
                    interval=interval,
                    max_frames=max_frames,
                    dedupe=dedupe,
                    on_progress=on_parts_prog,
                    should_stop=cancel_ev.is_set,
                )
            self._update(
                key,
                status="done",
                phase="done",
                done=len(snap.get("items") or []),
                total=max(len(snap.get("items") or []), 1),
                unit="张",
                result={"parts_items": len(snap.get("items") or [])},
                message=f"本地素材导入完成，工作区当前共 {len(snap.get('items') or [])} 张图片",
                finished_at=time.time(),
            )
        except TransferCancelled as exc:
            self._update(key, status="stopped", phase="stopped", error=str(exc), message=str(exc), finished_at=time.time())
        except Exception as exc:
            self._update(key, status="failed", phase="failed", error=str(exc), message=f"任务失败：{exc}", finished_at=time.time())
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _commit_yolo_and_parts(
        self,
        project_id: str,
        folder: Path,
        dataset_root: Path,
        names: list[str],
        slug: str,
        target_mode: str,
    ) -> dict:
        if self.wb.yolo_training():
            raise RuntimeError("当前工程正在训练零件模型，请等待结束或停止后再导入数据集")
        out: dict[str, object] = {}
        with self.wb.data_lock(project_id):
            if target_mode in {"yolo", "both"}:
                out["dataset"] = yolo_jobs.import_dataset(folder, str(dataset_root / "data.yaml"))
            if target_mode in {"parts", "both"} and names:
                out["parts_sync"] = sync_yolo_dir_to_parts_store(folder, dataset_root, names, slug)
        return out

    def _import_extracted_folder(
        self,
        project_id: str,
        folder: Path,
        extracted: Path,
        slug: str,
        target_mode: str,
        source_group: str,
    ) -> dict:
        yaml_file = _find_yaml_in_extracted(extracted)
        if yaml_file is not None and target_mode in {"yolo", "both"}:
            names = yolo_jobs._read_yaml_names(yaml_file.read_text(encoding="utf-8", errors="replace"))
            return self._commit_yolo_and_parts(project_id, folder, yaml_file.parent, names, slug, target_mode)
        # 没有 data.yaml 时作为原始图片集导入到零件标注工作区
        with self.wb.data_lock(project_id):
            snap = PartsStore(folder).import_source(
                str(extracted), source_group=source_group or slug, dedupe=True
            )
        return {"parts_items": len(snap.get("items") or [])}
