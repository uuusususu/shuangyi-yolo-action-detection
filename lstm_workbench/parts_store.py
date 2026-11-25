"""零件 OBB 素材、人工审核及不可变数据集版本。

工作区位于工程的 parts/，与可被重新导入覆盖的 yolo/ 分开。
所有框使用图像归一化四点坐标；发布只接受已审核框或明确确认的负样本。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .extract import read_image
from .project_store import ProjectStore

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.m4v', '.wmv'}
STATUSES = {'unlabeled', 'pending', 'reviewed', 'negative'}
BOX_SOURCES = {'manual', 'sam3', 'sam2', 'sam', 'yoloe', 'yolo', 'video_track', 'copy'}
_LOCKS: dict[str, threading.RLock] = {}
_LOCK_GUARD = threading.Lock()
_ID = re.compile(r'^[a-f0-9]{16,32}$')


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _classes(value: object) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 1000:
        raise ValueError('类别必须是包含 1 到 1000 个名称的列表')
    if any(not isinstance(name, str) or not name.strip() or len(name.strip()) > 120 or any(c in name for c in '\x00\r\n') for name in value):
        raise ValueError('类别名称不能为空，不能包含换行，最长 120 字符')
    names = [name.strip() for name in value]
    if len(set(names)) != len(names):
        raise ValueError('类别名称不能重复')
    return names


def _integer(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'{name} 必须是 {low} 到 {high} 的整数')
    return value


def _group(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 256 or any(c in value for c in '\x00\r\n'):
        raise ValueError('来源组不能为空，不能包含换行，最长 256 字符')
    return value.strip()


def _boxes(value: object, classes: list[str]) -> list[dict]:
    if not isinstance(value, list) or len(value) > 10000:
        raise ValueError('标注框必须是列表，每张图最多 10000 个框')
    cleaned = []
    for box in value:
        if not isinstance(box, dict):
            raise ValueError('每个框必须包含 class_id 和 points')
        cls = _integer(box.get('class_id'), '类别编号', 0, len(classes) - 1)
        points = box.get('points')
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError('OBB 标注必须恰好包含四个顶点')
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('顶点必须是 [x, y]')
            if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or not 0 <= n <= 1 for n in point):
                raise ValueError('顶点坐标必须是 0 到 1 的有限数值')
            normalized.append([float(n) for n in point])
        polygon = np.asarray(normalized, dtype=np.float32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) <= 1e-8:
            raise ValueError('OBB 四点必须依次围成不交叉、非退化的凸四边形')
        entry: dict = {'class_id': cls, 'points': normalized}
        raw_source = box.get('source')
        if raw_source is not None:
            if not isinstance(raw_source, str) or raw_source.strip() not in BOX_SOURCES:
                raise ValueError('标注框来源标记不合法')
            entry['source'] = raw_source.strip()
        raw_conf = box.get('confidence')
        if raw_conf is not None:
            if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)) or not math.isfinite(raw_conf) or not 0 <= raw_conf <= 1:
                raise ValueError('标注框置信度必须是 0 到 1 的有限数值')
            entry['confidence'] = round(float(raw_conf), 4)
        cleaned.append(entry)
    return cleaned


def _label_text(boxes: list[dict]) -> str:
    return ''.join(str(box['class_id']) + ' ' + ' '.join(format(n, '.10g') for point in box['points'] for n in point) + '\n' for box in boxes)


def _yaml_text(classes: list[str]) -> str:
    lines = ['task: obb', 'path: .', 'train: images/train', 'val: images/val', 'test: images/test', 'names:']
    lines += [f'  {n}: {json.dumps(name, ensure_ascii=False)}' for n, name in enumerate(classes)]
    return '\n'.join(lines) + '\n'


class PartsStore:
    """单工程零件数据。多个实例共用路径锁，HTTP 层另有工程数据锁。"""

    def __init__(self, project_folder: Path):
        self.project = Path(project_folder).resolve()
        self.project.mkdir(parents=True, exist_ok=True)
        self.root = self.project / 'parts'
        self._ensure_root()
        key = os.path.normcase(str(self.root))
        with _LOCK_GUARD:
            self._lock = _LOCKS.setdefault(key, threading.RLock())

    def _ensure_root(self) -> None:
        if self.root.is_symlink() or self.root.resolve().parent != self.project:
            raise ValueError('零件工作区路径越界或包含符号链接')
        self.root.mkdir(exist_ok=True)

    def _path(self, relative: str | Path, base: Path | None = None) -> Path:
        self._ensure_root()
        raw = str(relative)
        if not raw or '\\' in raw or ':' in raw or '\x00' in raw:
            raise ValueError('文件路径不合法')
        name = Path(raw)
        if name.is_absolute() or any(part in {'.', '..'} for part in name.parts):
            raise ValueError('文件路径越界')
        boundary = self.root if base is None else base
        target = boundary / name
        if not target.resolve().is_relative_to(boundary.resolve()) or not boundary.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('文件路径越界')
        current = target
        while current != self.root:
            if current.is_symlink():
                raise ValueError('零件文件不能使用符号链接')
            current = current.parent
        return target

    @staticmethod
    def _id(name: object) -> str:
        if not isinstance(name, str) or not _ID.fullmatch(name):
            raise ValueError('素材或版本编号不合法')
        return name

    def _load(self) -> dict:
        path = self._path('manifest.json')
        if not path.exists():
            return {'schema': 1, 'revision': 0, 'classes': [], 'items': []}
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(result, dict) or not isinstance(result.get('items'), list) or not isinstance(result.get('classes'), list):
                raise ValueError()
            _integer(result.get('revision'), '清单修订号', 0, 2**53 - 1)
            identities = set()
            for item in result['items']:
                if not isinstance(item, dict):
                    raise ValueError()
                item_id = self._id(item.get('id'))
                _integer(item.get('revision'), '素材修订号', 0, 2**53 - 1)
                if item_id in identities:
                    raise ValueError()
                identities.add(item_id)
            return result
        except (ValueError, TypeError) as exc:
            raise ValueError('零件素材清单损坏，请从备份恢复') from exc

    def _write(self, manifest: dict) -> None:
        manifest['revision'] += 1
        ProjectStore.write_json(self._path('manifest.json'), manifest)

    def snapshot(self) -> dict:
        with self._lock:
            result = self._load()
            result['versions'] = []
            versions = self._path('versions')
            if versions.is_dir():
                for path in sorted(versions.iterdir(), reverse=True):
                    if _ID.fullmatch(path.name) and path.is_dir():
                        data = self._version_manifest(path.name)
                        counts = {split: sum(i['split'] == split for i in data['items']) for split in ('train', 'val', 'test')}
                        result['versions'].append({key: data[key] for key in ('id', 'sha256', 'created_at', 'classes') } | {'counts': counts})
            return result

    def set_classes(self, classes: list[str], revision: int | None = None) -> dict:
        names = _classes(classes)
        with self._lock:
            data = self._load()
            if revision is not None and _integer(revision, '修订号', 0, 2**53 - 1) != data['revision']:
                raise ValueError('修订冲突：类别已经变化，请刷新后再保存')
            if any(item.get('boxes') for item in data['items']) and names[:len(data['classes'])] != data['classes']:
                raise ValueError('已有框标注，不能删除或重排已有类别；可以在末尾追加类别')
            if data['classes'] != names:
                data['classes'] = names
                self._write(data)
            return self.snapshot()

    @staticmethod
    def _default_group(path: Path, video: bool) -> str:
        source = path if video else path.parent
        digest = _sha(os.path.normcase(str(source)).encode('utf-8'))[:12]
        return f'{source.name or "来源"}:{digest}'

    def import_source(self, source: str, source_group: str = '', interval: int = 30,
                      max_frames: int = 300, dedupe: bool = True, *,
                      on_progress=None, should_stop=None) -> dict:
        interval = _integer(interval, '抽帧间隔', 1, 100000)
        max_frames = _integer(max_frames, '最多采样数', 1, 100000)
        if not isinstance(dedupe, bool):
            raise ValueError('去重选项必须是布尔值')
        if not isinstance(source, str) or not source.strip() or '\x00' in source:
            raise ValueError('请提供素材文件或目录路径')
        original = Path(source.strip().strip('"'))
        if original.is_symlink():
            raise ValueError('不能导入符号链接素材')
        path = original.resolve()
        if not path.exists():
            raise ValueError('找不到素材文件或目录')
        if path.is_relative_to(self.root) or (path.is_dir() and self.root.is_relative_to(path)):
            raise ValueError('不能把零件工作区及其父目录递归导入自身')
        if source_group:
            source_group = _group(source_group)
        elif not isinstance(source_group, str):
            raise ValueError('来源组必须是文本')
        directory = path.is_dir()
        if directory:
            files = []
            for candidate in sorted(path.rglob('*')):
                if candidate.suffix.lower() not in IMAGE_EXTENSIONS or not candidate.is_file():
                    continue
                if candidate.is_symlink() or not candidate.resolve().is_relative_to(path):
                    continue
                if any(parent.is_symlink() for parent in candidate.parents if parent != path and path in parent.parents):
                    continue
                files.append(candidate)
            if not files:
                raise ValueError('目录中没有可导入的图片')
            group = source_group or self._default_group(path / '_', False)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            files = [path]
            group = source_group or self._default_group(path, False)
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            files = []
            group = source_group or self._default_group(path, True)
        else:
            raise ValueError('不支持此素材格式，请选择图片、图片目录或视频')

        skipped_count = 0
        added_count = 0

        def check_stop():
            if should_stop is not None and should_stop():
                from .dataset_transfer import TransferCancelled
                raise TransferCancelled('已取消素材导入')

        def images():
            nonlocal skipped_count
            if files:
                total_files = len(files)
                for idx, file in enumerate(files, 1):
                    check_stop()
                    frame = read_image(file)
                    if frame is None:
                        raise ValueError(f'图片无法解码：{file.name}')
                    yield frame, file.read_bytes(), file.suffix.lower(), file.name, str(file.resolve()), None, idx, total_files
                return
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f'视频无法打开：{path.name}')
            raw_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            est_samples = min(max_frames, max(1, (raw_total + interval - 1) // interval)) if raw_total > 0 else max_frames
            frame_index = 0
            sampled = 0
            previous = None
            try:
                while sampled < max_frames:
                    check_stop()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % interval == 0:
                        sampled += 1
                        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
                        similar = dedupe and previous is not None and float(np.mean(np.abs(tiny - previous))) < 2.0
                        if not similar:
                            encoded, data = cv2.imencode('.jpg', frame)
                            if not encoded:
                                raise ValueError('视频帧编码失败')
                            yield frame, data.tobytes(), '.jpg', f'{path.stem}-{frame_index:08d}.jpg', str(path), frame_index, sampled, est_samples
                            previous = tiny
                        else:
                            skipped_count += 1
                            if on_progress is not None:
                                on_progress(sampled, est_samples, added_count, skipped_count)
                    frame_index += 1
            finally:
                capture.release()
            if frame_index == 0:
                raise ValueError('视频中没有可以解码的帧')

        with self._lock:
            data = self._load()
            known = {item.get('pixel_sha256', item.get('sha256')) for item in data['items']}
            added = []
            paths = []
            try:
                for frame, content, suffix, name, origin, index, done_n, total_n in images():
                    check_stop()
                    height, width = frame.shape[:2]
                    pixels = _sha(str(frame.shape).encode('ascii') + frame.tobytes())
                    if dedupe and pixels in known:
                        skipped_count += 1
                        if on_progress is not None:
                            on_progress(done_n, total_n, added_count, skipped_count)
                        continue
                    item_id = uuid.uuid4().hex[:20]
                    relative = f'images/{item_id}{suffix}'
                    target = self._path(relative)
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(content)
                    paths.append(target)
                    added.append({'id': item_id, 'name': name, 'image': relative,
                                  'width': int(width), 'height': int(height), 'source': origin,
                                  'source_group': group, 'frame_index': index, 'sha256': _sha(content),
                                  'pixel_sha256': pixels, 'revision': 0, 'status': 'unlabeled', 'boxes': []})
                    known.add(pixels)
                    added_count = len(added)
                    if on_progress is not None:
                        on_progress(done_n, total_n, added_count, skipped_count)
                if added:
                    data['items'].extend(added)
                    self._write(data)
            except BaseException:
                for target in paths:
                    target.unlink(missing_ok=True)
                raise
            return self.snapshot()

    def _item(self, data: dict, item_id: str) -> dict:
        self._id(item_id)
        for item in data['items']:
            if item.get('id') == item_id:
                return item
        raise FileNotFoundError('找不到这张零件素材')

    def save_item(self, item_id: str, boxes: list[dict], status: str, revision: int,
                  source_group: str | None = None, *, expected_classes: list[str] | None = None) -> dict:
        revision = _integer(revision, '素材修订号', 0, 2**53 - 1)
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError('标注状态不合法')
        with self._lock:
            data = self._load()
            # 自动预测的编号含义取决于启动时的类别顺序；核对和写入必须共用锁。
            if expected_classes is not None and data['classes'] != expected_classes:
                raise ValueError('类别表已变化，自动标注未写入，请使用当前类别重新发起预标注')
            item = self._item(data, item_id)
            if revision != item['revision']:
                raise ValueError('修订冲突：这张素材已被修改，请刷新后再保存')
            cleaned = _boxes(boxes, data['classes'])
            if status == 'reviewed' and not cleaned:
                raise ValueError('没有框时请明确选择“确认无目标”，不能直接标成已审核')
            if status == 'negative' and cleaned:
                raise ValueError('确认无目标的图片不能包含框')
            group = item['source_group'] if source_group is None else _group(source_group)
            item.update(boxes=cleaned, status=status, source_group=group, revision=revision + 1)
            self._write(data)
            return json.loads(json.dumps(item))

    def apply_auto_boxes(
        self,
        item_id: str,
        boxes: list[dict],
        *,
        expected_classes: list[str] | None = None,
        expected_revision: int | None = None,
        replace: bool = True,
    ) -> dict:
        """写入自动标注框，强制保持 pending 审核态，默认跳过已审核与确认无目标素材。"""
        with self._lock:
            data = self._load()
            if expected_classes is not None and data['classes'] != expected_classes:
                return {'applied': False, 'reason': '类别表已变化，自动标注未写入', 'item': None}
            item = self._item(data, item_id)
            if item['status'] in {'reviewed', 'negative'}:
                return {'applied': False, 'reason': '已人工审核或确认无目标，已跳过', 'item': None}
            if expected_revision is not None and item['revision'] != expected_revision:
                return {'applied': False, 'reason': '素材在推理期间已被人工修改，已跳过', 'item': None}
            cleaned = _boxes(boxes, data['classes'])
            if not cleaned:
                return {'applied': False, 'reason': '未检测到有效目标', 'item': None}
            merged = cleaned if replace else list(item.get('boxes') or []) + cleaned
            if len(merged) > 10000:
                merged = merged[:10000]
            item.update(boxes=merged, status='pending', revision=item['revision'] + 1)
            self._write(data)
            return {'applied': True, 'reason': '', 'item': json.loads(json.dumps(item))}

    def list_project_media_clips(self) -> list[dict]:
        """列出当前工程 media/ 下可直接抽帧到零件工作区的视频文件。"""
        media_dir = (self.project / 'media').resolve()
        if not media_dir.is_dir() or media_dir.is_symlink():
            return []
        clips: list[dict] = []
        for entry in sorted(media_dir.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_file() or entry.is_symlink():
                continue
            if entry.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = entry.resolve()
            if resolved.parent != media_dir:
                continue
            clips.append({'file': entry.name, 'stem': entry.stem, 'bytes': entry.stat().st_size})
        return clips

    def import_project_clip(
        self,
        clip_file: str,
        *,
        source_group: str = '',
        interval: int = 30,
        max_frames: int = 300,
        dedupe: bool = True,
    ) -> dict:
        """从当前工程 media/ 已有动作视频直接抽帧导入到零件工作区。"""
        raw = str(clip_file or '').strip()
        if not raw or '/' in raw or '\\' in raw or ':' in raw or '\x00' in raw or raw in {'.', '..'}:
            raise ValueError('工程视频文件名不合法')
        media_dir = (self.project / 'media').resolve()
        if not media_dir.is_dir():
            raise FileNotFoundError('当前工程还没有动作视频目录 media/')
        target = (media_dir / raw).resolve()
        if target.parent != media_dir or target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f'找不到工程视频：{raw}')
        if target.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError('仅支持从工程视频文件抽帧')
        group = source_group.strip() if isinstance(source_group, str) else ''
        return self.import_source(
            str(target),
            source_group=group or f'clip:{target.stem}',
            interval=interval,
            max_frames=max_frames,
            dedupe=dedupe,
        )

    def image_path(self, item_id: str) -> Path:
        with self._lock:
            item = self._item(self._load(), item_id)
            path = self._path(item['image'])
            if not path.is_file():
                raise FileNotFoundError('零件素材文件缺失')
            return path

    def delete_item(self, item_id: str, revision: int) -> dict:
        """移除工作区素材；清单写入失败时恢复原图，已发布版本始终保留。

        缺失或损坏的图片也可以从清单移除。先将图片停放到工作区内的
        .trash/，再原子保存清单，避免保存失败后留下指向已删除图片的记录。
        """
        revision = _integer(revision, '素材修订号', 0, 2**53 - 1)
        with self._lock:
            data = self._load()
            item = self._item(data, item_id)
            if revision != item['revision']:
                raise ValueError('修订冲突：这张素材已被修改，请刷新后再删除')
            relative = Path(item['image'])
            if (relative.as_posix() != f'images/{item_id}{relative.suffix}'
                    or relative.suffix.lower() not in IMAGE_EXTENSIONS):
                raise ValueError('只能删除工作区 images/ 中与素材编号对应的原图')
            target = self._path(item['image'])
            if target.exists() and not target.is_file():
                raise ValueError('工作区素材路径不是普通文件，拒绝删除')
            parked = None
            if target.is_file():
                parked = self._path(f'.trash/{item_id}-{uuid.uuid4().hex}{relative.suffix}')
                parked.parent.mkdir(exist_ok=True)
                os.replace(target, parked)
            data['items'] = [entry for entry in data['items'] if entry['id'] != item_id]
            try:
                self._write(data)
            except Exception:
                if parked is not None:
                    os.replace(parked, target)
                raise
            if parked is not None:
                try:
                    # 只清理本次停放的单个文件，绝不递归删除目录或版本文件。
                    parked.unlink()
                except OSError:
                    # 清单已提交，保留停放文件供后续人工整理，不回滚成功的删除。
                    pass
            return self.snapshot()

    def quality(self) -> dict:
        with self._lock:
            data = self._load()
            issues = []

            def issue(item_id: str, code: str, message: str):
                issues.append({'item_id': item_id, 'code': code, 'message': message})

            try:
                classes = _classes(data['classes'])
            except ValueError as exc:
                classes = []
                issue('', 'classes', str(exc))
            if not data['items']:
                issue('', 'empty', '还没有零件素材')
            hashes = {}
            video_groups: dict[str, str] = {}
            class_counts = [0] * len(classes)
            for item in data['items']:
                item_id = item.get('id', '')
                try:
                    _group(item.get('source_group'))
                except ValueError as exc:
                    issue(item_id, 'source-group', str(exc))
                if item.get('frame_index') is not None:
                    source = os.path.normcase(str(item.get('source') or ''))
                    group = item.get('source_group')
                    if source in video_groups and video_groups[source] != group:
                        issue(item_id, 'source-leakage', '同一原视频被分到不同来源组，请统一来源组，避免训练验证泄漏')
                    video_groups[source] = group
                if item.get('status') not in {'reviewed', 'negative'}:
                    issue(item_id, 'review', '素材尚未审核，请确认框或明确确认无目标')
                try:
                    path = self._path(item['image'])
                    frame = read_image(path) if path.is_file() else None
                    if frame is None:
                        issue(item_id, 'image-decode', '素材缺失或图片无法解码')
                    else:
                        h, w = frame.shape[:2]
                        if (w, h) != (item.get('width'), item.get('height')):
                            issue(item_id, 'image-size', '图片尺寸已变化，请重新导入')
                        if _file_sha(path) != item.get('sha256'):
                            issue(item_id, 'image-fingerprint', '图片内容指纹已变化，请重新导入')
                        pixels = _sha(str(frame.shape).encode('ascii') + frame.tobytes())
                        if pixels in hashes:
                            issue(item_id, 'duplicate-image', f'图片与素材 {hashes[pixels]} 重复，请勿重复发布')
                        hashes[pixels] = item_id
                except (ValueError, OSError, KeyError) as exc:
                    issue(item_id, 'image-path', str(exc))
                try:
                    boxes = _boxes(item.get('boxes'), classes)
                    if item.get('status') == 'negative' and boxes:
                        issue(item_id, 'negative-box', '确认无目标的素材仍含标注框')
                    if item.get('status') == 'reviewed' and not boxes:
                        issue(item_id, 'empty-reviewed', '已审核素材没有框，请明确确认无目标')
                    seen = set()
                    for box in boxes:
                        class_counts[box['class_id']] += 1
                        key = (box['class_id'], tuple(sorted(tuple(round(n, 6) for n in p) for p in box['points'])))
                        if key in seen:
                            issue(item_id, 'duplicate-box', '同一位置和类别存在重复框')
                        seen.add(key)
                        area = abs(cv2.contourArea(np.asarray(box['points'], dtype=np.float32)))
                        if area * item.get('width', 0) * item.get('height', 0) < 1:
                            issue(item_id, 'tiny-box', '标注框面积不足一个像素，请检查')
                except (ValueError, TypeError, KeyError) as exc:
                    issue(item_id, 'boxes', str(exc))
            groups = sorted({str(item.get('source_group') or '') for item in data['items']})
            return {'ok': not issues, 'issues': issues, 'items': len(data['items']),
                    'groups': groups, 'classes': classes, 'class_counts': class_counts,
                    'reviewed': sum(item.get('status') in {'reviewed', 'negative'} for item in data['items'])}

    def _splits(self, groups: list[str], splits: dict[str, str] | None, val_ratio: float, test_ratio: float) -> dict[str, str]:
        for number in (val_ratio, test_ratio):
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError('验证/测试比例必须是有限数值')
        if not 0 < val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
            raise ValueError('验证比例须大于零，测试比例不得小于零，两者之和必须小于一')
        if len(groups) < 2:
            raise ValueError('至少需要两个独立来源组，不能把同一视频的相邻帧拆入训练和验证')
        if splits is not None:
            if not isinstance(splits, dict) or set(splits) != set(groups):
                raise ValueError('来源划分必须包含每个来源组，且不能包含未知组')
            if any(value not in {'train', 'val', 'test'} for value in splits.values()):
                raise ValueError('来源组用途只能是 train、val 或 test')
            if not {'train', 'val'} <= set(splits.values()):
                raise ValueError('来源划分必须同时包含独立训练和验证组')
            return dict(splits)
        ordered = sorted(groups)
        random.Random(20261108).shuffle(ordered)
        test_count = min(len(groups) - 2, max(1, round(len(groups) * test_ratio))) if test_ratio > 0 else 0
        val_count = min(len(groups) - test_count - 1, max(1, round(len(groups) * val_ratio)))
        return {group: 'test' if index < test_count else 'val' if index < test_count + val_count else 'train'
                for index, group in enumerate(ordered)}

    def publish(self, splits: dict[str, str] | None = None, val_ratio: float = .2, test_ratio: float = .1) -> dict:
        with self._lock:
            data = self._load()
            report = self.quality()
            if not report['ok']:
                raise ValueError('数据质检或审核未通过：' + '; '.join(issue['message'] for issue in report['issues'][:5]))
            mapping = self._splits(report['groups'], splits, val_ratio, test_ratio)
            if not any(item['boxes'] and mapping[item['source_group']] == 'train' for item in data['items']):
                raise ValueError('训练来源组至少需要一个正样本框，不能全部为无目标图片')
            frozen = []
            for item in data['items']:
                split = mapping[item['source_group']]
                suffix = Path(item['image']).suffix.lower()
                label = _label_text(item['boxes']).encode('utf-8')
                frozen.append({key: item[key] for key in ('id', 'name', 'width', 'height', 'source', 'source_group', 'frame_index', 'sha256', 'status', 'boxes')} |
                              {'split': split, 'image': f'images/{split}/{item["id"]}{suffix}',
                               'label': f'labels/{split}/{item["id"]}.txt', 'label_sha256': _sha(label)})
            content = {'schema': 1, 'classes': data['classes'], 'splits': mapping, 'items': frozen}
            digest = _sha(_canonical(content))
            version_id = digest[:16]
            target = self._path(f'versions/{version_id}')
            if target.exists():
                return self.version_info(version_id)
            staging = self._path(f'versions/.publishing-{uuid.uuid4().hex}')
            staging.mkdir(parents=True)
            try:
                for split in ('train', 'val', 'test'):
                    self._path(f'images/{split}', staging).mkdir(parents=True)
                    self._path(f'labels/{split}', staging).mkdir(parents=True)
                for item, original in zip(frozen, data['items']):
                    copied = self._path(item['image'], staging)
                    shutil.copyfile(self._path(original['image']), copied)
                    if _file_sha(copied) != item['sha256']:
                        raise ValueError('发布时素材内容发生变化，图片指纹不符，请重新质检')
                    self._path(item['label'], staging).write_text(_label_text(item['boxes']), encoding='utf-8', newline='\n')
                self._path('data.yaml', staging).write_text(_yaml_text(data['classes']), encoding='utf-8', newline='\n')
                manifest = content | {'id': version_id, 'sha256': digest, 'created_at': time.time()}
                ProjectStore.write_json(self._path('manifest.json', staging), manifest)
                os.replace(staging, target)
            finally:
                # staging 由本次创建，路径始终限制在 parts/versions 内。
                if staging.exists():
                    if not staging.resolve().is_relative_to(self.root.resolve()):
                        raise ValueError('发布暂存目录越界，拒绝清理')
                    shutil.rmtree(staging)
            return self.version_info(version_id)

    def _version_manifest(self, version_id: str) -> dict:
        self._id(version_id)
        root = self._path(f'versions/{version_id}')
        path = self._path('manifest.json', root)
        if not path.is_file():
            raise FileNotFoundError('找不到这个数据集版本')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            content = {key: data[key] for key in ('schema', 'classes', 'splits', 'items')}
            digest = _sha(_canonical(content))
            if data['id'] != version_id or data['sha256'] != digest or digest[:16] != version_id:
                raise ValueError()
            _classes(data['classes'])
            return data
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError('数据集版本清单指纹不符或已损坏') from exc

    def version_info(self, version_id: str) -> dict:
        with self._lock:
            data = self._version_manifest(version_id)
            root = self._path(f'versions/{version_id}')
            yaml = self._path('data.yaml', root)
            if not yaml.is_file() or yaml.read_text(encoding='utf-8') != _yaml_text(data['classes']):
                raise ValueError('数据集版本配置已损坏')
            for item in data['items']:
                image = self._path(item['image'], root)
                label = self._path(item['label'], root)
                if not image.is_file() or _file_sha(image) != item['sha256']:
                    raise ValueError('数据集版本图片缺失或指纹不符')
                if not label.is_file() or _file_sha(label) != item['label_sha256']:
                    raise ValueError('数据集版本标签缺失或指纹不符')
                item['image'] = str(image.resolve())
                item['label'] = str(label.resolve())
            data['root'] = str(root.resolve())
            data['data_yaml'] = str(yaml.resolve())
            data['counts'] = {split: sum(item['split'] == split for item in data['items']) for split in ('train', 'val', 'test')}
            return data
