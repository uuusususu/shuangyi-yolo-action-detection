"""零件模型（YOLO）训练：数据集、训练进程、进度、导出 ONNX 入库。

YOLO 训练跑在独立的 Python 环境里（ultralytics 和这边的 torch 版本不同，不合成一个环境）。
这里只负责：拼命令、起子进程、读 results.csv、停止、导出。

一个工程下的零件数据：
    <工程>/yolo/data.yaml           Ultralytics 数据集配置（train/val 指向下面的 images）
    <工程>/yolo/images, labels      导入时整份复制进来
    <工程>/yolo/runs/<轮次>/         Ultralytics 的输出目录：results.csv、weights/best.pt
    <工程>/yolo/runs/<轮次>/job.json 训练参数、进程号、状态
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import model_registry
from .project_store import ProjectStore

ROOT = Path(__file__).resolve().parent

TASKS = {"obb", "detect", "segment"}
# 底模名。先在 models/yolo_base 里找同名文件，找不到才由 Ultralytics 去下（需要联网）。
DEFAULT_BASE = {"obb": "yolo11n-obb.pt", "detect": "yolo11n.pt", "segment": "yolo11n-seg.pt"}


def base_models_dir() -> Path:
    """底模放这里。训练进程的工作目录也设在这里：Ultralytics 的 AMP 自检会按文件名
    加载 yolo26n.pt，工作目录里有这个文件就不用联网下载（产线电脑多半连不上 GitHub）。"""
    return model_registry.MODEL_LIBRARY / "yolo_base"


def list_base_models() -> list[str]:
    folder = base_models_dir()
    if not folder.is_dir():
        return []
    return sorted(item.name for item in folder.iterdir() if item.suffix.lower() == ".pt")


def _resolve_base(name: str) -> str:
    """底模：完整路径原样用；只给文件名时，底模库里有就换成库里的绝对路径。"""
    text = str(name or "").strip()
    if not text:
        return text
    path = Path(text)
    if path.is_absolute():
        return text
    local = base_models_dir() / path.name
    return str(local) if local.is_file() else text
EXPORT_OPSET = 12
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_RUNNING = {"done", "failed", "stopped"}


class YoloEnvMissing(RuntimeError):
    """没找到装了 ultralytics 的 Python。"""


# ---------- 环境 ----------

def _candidate_pythons() -> list[Path]:
    """YOLO 训练用的 Python，按优先级。

    1. 环境变量 YOLO_PYTHON（部署到别的电脑时最直接）
    2. 工作台文件夹里的 yolo_env\\（便携放法）
    3. 和动作检测同级的 ultralytics-main\\.venv（开发机现在的放法）
    4. 当前运行工作台的 Python（自己也装了 ultralytics 时）
    """
    found: list[Path] = []
    explicit = os.environ.get("YOLO_PYTHON", "").strip()
    if explicit:
        found.append(Path(explicit))
    # 离线便携包：动作训练和零件训练共用文件夹里的 runtime\python.exe。
    for base in (model_registry.HOME / "runtime", ROOT / "runtime"):
        found.append(base / "python.exe")
    home = model_registry.HOME
    for base in (home / "yolo_env", ROOT / "yolo_env"):
        found.append(base / "Scripts" / "python.exe")
        found.append(base / "python.exe")
    for up in (ROOT.parent.parent, ROOT.parent.parent.parent):
        found.append(up / "ultralytics-main" / ".venv" / "Scripts" / "python.exe")
    found.append(Path(sys.executable))
    unique: list[Path] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


_ENV_CACHE: dict | None = None
_ENV_LOCK = threading.Lock()


def environment(refresh: bool = False) -> dict:
    """找到能 import ultralytics 的 Python，并报告版本和显卡。结果缓存，探测一次要好几秒。"""
    global _ENV_CACHE
    with _ENV_LOCK:
        if _ENV_CACHE is not None and not refresh:
            return dict(_ENV_CACHE)
        probe = (
            "import json,sys\n"
            "import ultralytics, torch\n"
            "cuda = torch.cuda.is_available()\n"
            "print(json.dumps({'python': sys.version.split()[0], 'ultralytics': ultralytics.__version__,"
            " 'torch': torch.__version__, 'cuda': cuda,"
            " 'gpu': torch.cuda.get_device_name(0) if cuda else ''}))\n"
        )
        tried = []
        for python in _candidate_pythons():
            if not python.is_file():
                continue
            try:
                done = subprocess.run(
                    [str(python), "-c", probe], capture_output=True, text=True, timeout=120,
                    encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                tried.append(f"{python}：{exc}")
                continue
            line = (done.stdout or "").strip().splitlines()[-1:] or [""]
            try:
                info = json.loads(line[0])
            except ValueError:
                tried.append(f"{python}：没装 ultralytics")
                continue
            _ENV_CACHE = {"ok": True, "python_path": str(python), **info}
            return dict(_ENV_CACHE)
        _ENV_CACHE = {
            "ok": False,
            "error": "没找到装了 ultralytics 的 Python。设置环境变量 YOLO_PYTHON，"
                     "或把 YOLO 环境放到工作台目录下的 yolo_env\\",
            "tried": tried,
        }
        return dict(_ENV_CACHE)


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _child_env(project_folder: Path) -> dict:
    """YOLO 子进程的环境变量。

    YOLO_OFFLINE：现场电脑不联网，Ultralytics 默认会联网检查更新、下载字体和自检模型，
    每一项都要等到超时。离线模式直接跳过。
    YOLO_CONFIG_DIR：设置和缓存放在工程里，不碰用户目录下的全局设置；随包带的字体也复制进去。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["YOLO_OFFLINE"] = "1"
    # 依赖由部署环境预装；导出和验证不能隐式启动 pip 安装 GPU 运行库。
    env["YOLO_AUTOINSTALL"] = "false"
    config = (yolo_dir(project_folder) / ".ultralytics").resolve()
    _seed_fonts(config / "Ultralytics")
    env["YOLO_CONFIG_DIR"] = str(config)
    return env


def _seed_fonts(target: Path) -> None:
    """画训练曲线、混淆矩阵要 Arial 字体，没有就去网上下。随包带的字体放在 models/yolo_base/fonts。"""
    source = base_models_dir() / "fonts"
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for font in source.glob("*.ttf"):
        if not (target / font.name).is_file():
            shutil.copy2(font, target / font.name)


def _python() -> str:
    env = environment()
    if not env.get("ok"):
        raise YoloEnvMissing(env.get("error") or "YOLO 环境不可用")
    return env["python_path"]


# ---------- 数据集 ----------

def yolo_dir(project_folder: Path) -> Path:
    target = (project_folder / 'yolo').resolve()
    if project_folder.resolve() not in target.parents:
        raise ValueError('零件数据目录越界')
    return target


def _read_yaml_names(text: str) -> list[str]:
    """data.yaml 里的 names。只认两种常见写法，不引入 yaml 依赖：
        names: [a, b]        或        names:\\n  0: a\\n  1: b
    """
    inline = re.search(r"^names:\s*\[(.*)\]\s*$", text, re.M)
    if inline:
        return [item.strip().strip("'\"") for item in inline.group(1).split(",") if item.strip()]
    block = re.search(r"^names:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
    if not block:
        return []
    names: dict[int, str] = {}
    for line in block.group(1).splitlines():
        match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip("'\"")
        else:
            item = re.match(r"\s*-\s*(.+?)\s*$", line)
            if item:
                names[len(names)] = item.group(1).strip("'\"")
    return [names[key] for key in sorted(names)]


def _read_yaml_value(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
    return match.group(1).strip("'\"") if match else ""


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for item in folder.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def dataset_info(project_folder: Path) -> dict:
    """本工程零件数据集的概况。没有就返回 exists=False。"""
    root = yolo_dir(project_folder)
    data = root / "data.yaml"
    if not data.is_file():
        return {"exists": False}
    text = data.read_text(encoding="utf-8", errors="replace")
    names = _read_yaml_names(text)
    counts = [0] * len(names)
    labels_root = root / "labels"
    boxes = 0
    if labels_root.is_dir():
        for label in labels_root.rglob("*.txt"):
            try:
                for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                    head = line.split(" ", 1)[0]
                    if head.isdigit() and int(head) < len(counts):
                        counts[int(head)] += 1
                        boxes += 1
            except OSError:
                continue
    return {
        "exists": True,
        "task": _read_yaml_value(text, "task") or "obb",
        "names": names,
        "class_counts": counts,
        "boxes": boxes,
        "train_images": _count_images(root / "images" / "train"),
        "val_images": _count_images(root / "images" / "val"),
        "source": _read_yaml_value(text, "imported_from"),
        "source_warning": "外部数据集缺少来源审核记录，不能据此证明训练与验证来源独立。请优先使用标注工作区发布的版本。",
    }


def _check_import_tree(path: Path, project: Path) -> None:
    """改名或清理前逐项确认边界，拒绝符号链接及 Windows Junction。"""
    project = project.resolve()
    if path.resolve().parent != project:
        raise ValueError('数据集事务目录越界')
    pending = [path]
    while pending:
        current = pending.pop()
        if current.exists() or current.is_symlink():
            attributes = current.lstat()
            if (current.is_symlink()
                    or getattr(attributes, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    or not current.resolve().is_relative_to(project)):
                raise ValueError('数据集目录含符号链接或 Junction，拒绝改名或清理')
            if current.is_dir():
                pending.extend(current.iterdir())


def _clean_import_tree(path: Path, project: Path) -> None:
    if path.exists():
        _check_import_tree(path, project)
        shutil.rmtree(path)


def import_dataset(project_folder: Path, source_text: str) -> dict:
    """把一份 Ultralytics 格式的数据集整份复制进工程。

    source 可以是 data.yaml 的路径，也可以是包含 data.yaml 的目录（训练包、平台下载的数据集都是这样）。
    要求 images/train 和 images/val 都在；只有 train 时从里面拨 15% 当验证。
    """
    source = Path(str(source_text or "").strip().strip('"'))
    if source.is_dir():
        source = source / "data.yaml"
    if not source.is_file() or source.name.lower() not in {"data.yaml", "data.yml"} and source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"找不到数据集配置：{source}")
    base = source.parent
    text = source.read_text(encoding="utf-8", errors="replace")
    names = _read_yaml_names(text)
    if not names:
        raise ValueError("data.yaml 里没有 names（类别名）")
    task = _read_yaml_value(text, "task") or "obb"
    if task not in TASKS:
        raise ValueError(f"不支持的任务类型：{task}")
    train_images = base / "images" / "train"
    if not train_images.is_dir():
        raise ValueError(f"{base} 下没有 images/train")

    project = Path(project_folder).resolve()
    project.mkdir(parents=True, exist_ok=True)
    target = yolo_dir(project)
    _check_import_tree(target, project)
    token = uuid.uuid4().hex
    staging = project / f'.yolo-import-{token}'
    backup = project / f'.yolo-backup-{token}'
    staging.mkdir()
    backed_up = moved_runs = committed = False
    try:
        for sub in ("images", "labels"):
            if (base / sub).is_dir():
                shutil.copytree(base / sub, staging / sub, ignore=shutil.ignore_patterns("*.cache"))
        if not (staging / "images" / "val").is_dir():
            _split_validation(staging)
        lines = [
            f"task: {task}",
            "path: .",
            "train: images/train",
            "val: images/val",
            f"imported_from: '{base}'",
            "names:",
        ] + [f"  {index}: {name}" for index, name in enumerate(names)]
        (staging / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _check_import_tree(staging, project)
        if target.exists():
            _check_import_tree(target, project)
            os.replace(target, backup)
            backed_up = True
            old_runs = backup / 'runs'
            if old_runs.is_dir():
                os.replace(old_runs, staging / 'runs')
                moved_runs = True
        os.replace(staging, target)
        committed = True
    except BaseException as original:
        if backed_up:
            try:
                _check_import_tree(backup, project)
                _check_import_tree(staging, project)
                if moved_runs:
                    os.replace(staging / 'runs', backup / 'runs')
                os.replace(backup, target)
            except (OSError, ValueError) as recovery:
                # 两处都保留：暂存区可能包含唯一的旧权重，不能用 finally 清理。
                raise OSError(f'导入失败且自动恢复失败；请保留并恢复备份 {backup} 和暂存 {staging}：{recovery}') from original
        _clean_import_tree(staging, project)
        raise
    if committed and backup.exists():
        try:
            _clean_import_tree(backup, project)
        except OSError:
            # 新数据已提交；残余备份可保留，不把成功导入误报为失败。
            pass
    return dataset_info(project_folder)


def _split_validation(root: Path, ratio: float = 0.15) -> None:
    train = root / "images" / "train"
    images = sorted(item for item in train.iterdir() if item.is_file())
    if len(images) < 2:
        raise ValueError("训练图片太少，拨不出验证集")
    step = max(2, round(1 / ratio))
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for image in images[::step]:
        shutil.move(str(image), str(root / "images" / "val" / image.name))
        label = root / "labels" / "train" / (image.stem + ".txt")
        if label.is_file():
            shutil.move(str(label), str(root / "labels" / "val" / label.name))


def _data_yaml_for_training(project_folder: Path) -> Path:
    """训练用的 data.yaml。path 写成绝对路径：Ultralytics 对相对 path 的解析基准不是 yaml 所在目录。"""
    root = yolo_dir(project_folder)
    source = root / "data.yaml"
    if not source.is_file():
        raise FileNotFoundError("还没有零件数据，先在「零件数据」页导入")
    text = source.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^path:.*$", f"path: '{root.resolve().as_posix()}'", text, flags=re.M)
    resolved = root / ".train_data.yaml"
    resolved.write_text(text, encoding="utf-8")
    return resolved


# ---------- 训练 ----------

def runs_dir(project_folder: Path) -> Path:
    target = (yolo_dir(project_folder) / 'runs').resolve()
    if project_folder.resolve() not in target.parents:
        raise ValueError('零件轮次目录越界')
    return target


def _run_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _integer(params: dict, key: str, default: int, low: int, high: int) -> int:
    value = params.get(key, default)
    if value in (None, ''):
        value = default
    try:
        number = int(value)
        if isinstance(value, bool) or float(value) != number or not low <= number <= high:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} 必须是 {low} 到 {high} 的整数") from None
    return number


def build_command(python: str, data_yaml: Path, run_root: Path, name: str, params: dict) -> list[str]:
    """表单参数 → yolo 命令。extra 里可以追加任意 key=value，给懂行的人用。"""
    task = str(params.get("task") or "obb")
    if task not in TASKS:
        raise ValueError(f"不支持的任务类型：{task}")
    base = _resolve_base(str(params.get("model") or DEFAULT_BASE[task]))
    epochs = _integer(params, 'epochs', 100, 1, 100000)
    imgsz = _integer(params, 'imgsz', 640, 32, 4096)
    workers = _integer(params, 'workers', 4, 0, 64)
    batch = _integer(params, 'batch', 16, -1, 4096)
    if batch == 0:
        raise ValueError('batch 不能为 0')
    # ultralytics 包没有 __main__，不能 python -m ultralytics；用它的命令行入口函数。
    # 不用 Scripts\yolo.exe：便携复制的 venv 里那个 exe 写死了原来的 Python 路径。
    args = [
        python, "-c", "import sys; from ultralytics.cfg import entrypoint; entrypoint()", task, "train",
        f"data={data_yaml.as_posix()}",
        f"model={base}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={params.get('device') if params.get('device') not in (None, '') else 0}",
        f"project={run_root.as_posix()}",
        f"name={name}",
        "exist_ok=True",
        f"workers={workers}",
        "plots=True",
    ]
    extra = str(params.get("extra") or "").strip()
    for token in extra.split():
        managed = {'data','project','name','model','task','mode','resume','exist_ok','callbacks',
                   'epochs','imgsz','batch','device','workers','save_dir','cfg'}
        if "=" not in token or token.split('=', 1)[0].lower() in managed:
            raise ValueError(f"高级参数不能覆盖工作台托管的路径、模型、任务或训练表单参数：{token}")
        args.append(token)
    return args


def _write_json(path: Path, payload: object) -> None:
    ProjectStore.write_json(path, payload)


def _process_key(run: Path) -> str:
    return os.path.normcase(str(run.resolve()))


def _snapshot_metadata(root: Path, kind: str, version: str = '') -> dict:
    files = []
    resolved = root.resolve()
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix != '.cache':
            if resolved not in path.resolve().parents:
                raise ValueError('数据目录包含越界链接')
            files.append({'file':path.relative_to(root).as_posix(),'sha256':model_registry.file_sha256(path)})
    content = json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode('utf-8')
    data = root/'data.yaml'
    return {'kind':kind,'version':version,'sha256':hashlib.sha256(content).hexdigest(),
            'classes':_read_yaml_names(data.read_text(encoding='utf-8')),
            'data_yaml':str(data.resolve()),'root':str(root.resolve()),'files':files}


def _freeze_dataset(project_folder: Path, run: Path, version: str) -> dict:
    if version:
        from .parts_store import PartsStore
        info = PartsStore(project_folder).version_info(version)
        data = Path(info['data_yaml'])
        metadata = _snapshot_metadata(data.parent, 'published', version)
        metadata['classes'] = list(info['classes'])
        metadata['content_sha256'] = info['sha256']
        text = data.read_text(encoding='utf-8')
        text = re.sub(r'^path:.*$', 'path: ' + json.dumps(data.parent.resolve().as_posix(), ensure_ascii=False), text, flags=re.M)
        resolved = run/'dataset.yaml'
        resolved.write_text(text,encoding='utf-8')
        metadata['data_yaml'] = str(resolved.resolve())
        metadata['training_yaml_sha256'] = model_registry.file_sha256(resolved)
    else:
        source = yolo_dir(project_folder)
        if not (source/'data.yaml').is_file():
            raise FileNotFoundError('还没有零件数据，请先导入或发布数据集')
        target = run/'dataset'
        target.mkdir()
        for sub in ('images','labels'):
            if (source/sub).is_dir():
                for path in (source/sub).rglob('*'):
                    if source.resolve() not in path.resolve().parents:
                        raise ValueError('数据集包含越界链接')
                shutil.copytree(source/sub, target/sub, ignore=shutil.ignore_patterns('*.cache'))
        text = (source/'data.yaml').read_text(encoding='utf-8')
        text = re.sub(r'^path:.*$', 'path: ' + json.dumps(target.resolve().as_posix(), ensure_ascii=False), text, flags=re.M)
        (target/'data.yaml').write_text(text, encoding='utf-8')
        metadata = _snapshot_metadata(target, 'legacy')
        metadata['warning'] = '外部导入数据没有来源审核记录；指标可能受同源泄漏影响。'
    _write_json(run/'dataset.json', metadata)
    return metadata


def _verify_snapshot(metadata: dict) -> None:
    root = Path(metadata.get('root') or Path(metadata.get('data_yaml') or '').parent)
    if not root.is_dir() or not metadata.get('files'):
        raise ValueError('这一轮缺少可验证的冻结数据，不能安全续训')
    for record in metadata['files']:
        path = (root/record['file']).resolve()
        if root.resolve() not in path.parents or not path.is_file() or model_registry.file_sha256(path) != record['sha256']:
            raise ValueError('冻结数据已缺失或变化，拒绝续训')
    yaml_path = Path(metadata.get('data_yaml') or '')
    if metadata.get('training_yaml_sha256') and (not yaml_path.is_file() or model_registry.file_sha256(yaml_path) != metadata['training_yaml_sha256']):
        raise ValueError('训练数据配置已变化，拒绝续训')


def start_training(project_folder: Path, params: dict) -> dict:
    from .parts_store import PartsStore
    version = str(params.get('dataset_version') or '')
    if not version and PartsStore(project_folder).snapshot().get('versions'):
        raise ValueError('请选择一个已发布的数据集版本')
    # 在环境探测、复制数据和起进程之前校验表单。
    build_command('python', Path('data.yaml'), Path('.'), 'validate', params)
    python = _python()
    root = runs_dir(project_folder)
    root.mkdir(parents=True, exist_ok=True)
    name = _run_name()
    while (root/name).exists():
        name = _run_name() + '-' + uuid.uuid4().hex[:6]
    if version and str(params.get('task') or 'obb') != 'obb':
        raise ValueError('已发布零件标注版本仅支持 OBB 训练')
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    try:
        (yolo_dir(project_folder) / ".ultralytics").mkdir(parents=True, exist_ok=True)
        dataset = _freeze_dataset(project_folder, run, version)
        command = build_command(python, Path(dataset['data_yaml']), root.resolve(), name, params)
        env = _child_env(project_folder)
        workdir = base_models_dir()
        workdir.mkdir(parents=True, exist_ok=True)
    except BaseException:
        # 还没起进程就失败：这一轮目录里只有冻结的数据集副本，没有 job.json，列表里看不见，
        # 留着只会白占一整份数据集的磁盘。
        shutil.rmtree(run, ignore_errors=True)
        raise
    job = {
        "name": name,
        "status": "training",
        "started_at": time.time(),
        "params": {**{'task':'obb','epochs':100,'imgsz':640,'batch':16,'device':0,'workers':4,'extra':''}, **params},
        "command": command[1:],
        "dataset": dataset,
        "environment": environment(),
        "model": {'identifier':command[6].split('=',1)[1],
                  'sha256':model_registry.file_sha256(Path(command[6].split('=',1)[1])) if Path(command[6].split('=',1)[1]).is_file() else None},
    }
    try:
        with (run / 'train.log').open('w', encoding='utf-8', errors='replace') as log:
            process = subprocess.Popen(command, cwd=str(workdir), stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=_NO_WINDOW | getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
    except BaseException:
        shutil.rmtree(run, ignore_errors=True)
        raise
    with _STATUS_LOCK:
        try:
            job['pid'] = process.pid
            job['pid_created'] = _pid_created(process.pid)
            _write_job(run, job)
        except BaseException:
            _abort_process(process)
            raise
        _PROCESSES[_process_key(run)] = process
    return job


_PROCESSES: dict[str, subprocess.Popen] = {}

# 本机 Ultralytics engine/model.py 对不可续训权重会把 resume=True 降为新训练。
# 必须在调用前检查状态和原始路径，避免界面声称续训却重新训练或写进旧工程。
RESUME_SCRIPT = '''
import json, os, sys
from pathlib import Path
from ultralytics import YOLO
checkpoint, expected_data, expected_run, expected_classes = sys.argv[1:5]
model = YOLO(checkpoint)
ckpt = model.ckpt
if not isinstance(ckpt, dict) or not isinstance(ckpt.get('epoch'), int) or ckpt['epoch'] < 0 or ckpt.get('optimizer') is None:
    raise ValueError('last.pt 缺少有效 epoch 或 optimizer 状态，拒绝降级为新训练')
args = ckpt.get('train_args') or {}
def same_path(value, expected):
    return isinstance(value, str) and bool(value) and os.path.normcase(str(Path(value).resolve())) == os.path.normcase(str(Path(expected).resolve()))
if not same_path(args.get('data'), expected_data) or not same_path(args.get('save_dir'), expected_run):
    raise ValueError('检查点的数据或输出目录与当前轮次不一致；移动或替换过的检查点不能直接续训')
if not isinstance(args.get('epochs'), int) or ckpt['epoch'] + 1 >= args['epochs']:
    raise ValueError('检查点已达到原定训练轮数，没有可续训的剩余轮次')
names = model.names
classes = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
if classes != json.loads(expected_classes):
    raise ValueError('检查点类别顺序与本轮冻结数据不一致，拒绝续训')
model.train(resume=True)
'''


def _abort_process(process: subprocess.Popen) -> None:
    """登记任务失败时收回进程树，不能让界面看不到的训练继续占用显卡。"""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=_NO_WINDOW)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _write_job(run: Path, job: dict) -> None:
    _write_json(run / 'job.json', job)


def _read_job(run: Path) -> dict:
    path = run / "job.json"
    # 另一个线程正在替换 job.json 时读会拒绝访问；读成空字典会让"正在训练"闪成"空闲"，
    # 显卡互斥也跟着失效，所以只在文件真不存在时才返回空。
    for _ in range(40):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except PermissionError:
            time.sleep(0.05)
        except (OSError, ValueError):
            return {}
    return {}


def resume_training(project_folder: Path, run_name: str, params: dict | None = None) -> dict:
    """只接续本工程已停止/失败轮次的 last.pt，并先验证原数据未变化。"""
    if not isinstance(run_name, str) or not run_name or run_name in {'.','..'} or any(c in run_name for c in '/\\:\0'):
        raise ValueError('轮次名不合法')
    run = (runs_dir(project_folder)/run_name).resolve()
    if runs_dir(project_folder).resolve() not in run.parents:
        raise ValueError('轮次路径越界')
    job = refresh_status(run)
    if job.get('status') not in {'stopped','failed'}:
        raise ValueError('只能续训已停止或失败的轮次，已完成轮次请新建训练')
    last = run/'weights'/'last.pt'
    if not last.is_file() or run not in last.resolve().parents:
        raise ValueError('没有可用的 last.pt，至少完成一轮后才能续训')
    overrides = params or {}
    if overrides:
        raise ValueError('续训沿用原参数与数据；修改参数请新建训练轮次')
    metadata = job.get('dataset') or {}
    _verify_snapshot(metadata)
    command = [_python(), '-c', RESUME_SCRIPT, str(last), str(metadata['data_yaml']), str(run),
               json.dumps(metadata.get('classes') or [],ensure_ascii=False)]
    old_job = dict(job)
    # 原记录先归档，失败不丢失上次的状态。
    history = list(job.get('resume_history') or [])
    history.append({key:old_job.get(key) for key in ('status','started_at','finished_at','pid','error')})
    with (run/'train.log').open('a', encoding='utf-8', errors='replace') as log:
        process = subprocess.Popen(command, cwd=str(base_models_dir()), stdout=log, stderr=subprocess.STDOUT,
                                   env=_child_env(project_folder),
                                   creationflags=_NO_WINDOW | getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
    with _STATUS_LOCK:
        try:
            (run/'stop.request').unlink(missing_ok=True)
            job.update({'status':'training','pid':process.pid,'pid_created':_pid_created(process.pid),
                        'started_at':time.time(),'finished_at':None,
                        'error':'','command':command[1:],'resume_count':int(job.get('resume_count') or 0)+1,
                        'resume_history':history})
            _write_job(run,job)
        except BaseException:
            _abort_process(process)
            raise
        _PROCESSES[_process_key(run)] = process
    return job


def _pid_created(pid: int) -> float | None:
    """进程的创建时刻；进程不在或读不到时为 None。"""
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _pid_alive(pid: int, created: float | None = None) -> bool:
    """记录的训练进程是否还在。

    Windows 会复用 PID：训练台重启后，旧训练早已退出，同号可能已经是别的程序。
    job.json 里存了启动时的创建时刻，对不上就当旧训练已经结束，免得一个无关进程
    把所有工程的显卡任务一直挡住，停止时也不会误杀它的进程树。
    """
    if pid <= 0:
        return False
    actual = _pid_created(pid)
    if created is not None:
        return actual is not None and abs(actual - float(created)) < 1.0
    if actual is not None:
        return True
    if os.name == "nt":
        done = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
            creationflags=_NO_WINDOW,
        )
        return str(pid) in (done.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _job_process_alive(job: dict) -> bool:
    created = job.get("pid_created")
    return _pid_alive(int(job.get("pid") or 0), float(created) if created is not None else None)


_STATUS_LOCK = threading.RLock()


def refresh_status(run: Path) -> dict:
    """根据进程是否还活着、有没有 best.pt，把 job.json 的状态改成最终值。

    几乎每个请求和后台准备线程都会调到这里。读、判、写必须整段串行：
    否则一个线程刚按退出码 0 写成 done 并移除进程句柄，另一个线程拿着旧的读数、
    却找不到句柄，会把它改写成"中断/failed"。"""
    job = _read_job(run)
    if not job or job.get("status") != "training":
        return job
    with _STATUS_LOCK:
        return _refresh_status_locked(run)


def _refresh_status_locked(run: Path) -> dict:
    job = _read_job(run)
    if not job or job.get("status") != "training":
        return job
    process = _PROCESSES.get(_process_key(run))
    if process is not None:
        code = process.poll()
        alive = code is None
    else:
        code = None
        alive = _job_process_alive(job)
    if alive:
        return job
    best = run / "weights" / "best.pt"
    if (run / "stop.request").exists():
        job["status"] = "stopped"
    elif best.is_file() and code == 0:
        job["status"] = "done"
    else:
        job["status"] = "failed"
        job["error"] = ('训练进程已中断，无法确认正常完成；如有有效 last.pt 可尝试续训'
                        if process is None else
                        _last_error_line(run / "train.log") or f"训练进程退出（代码 {code}）")
    job["finished_at"] = time.time()
    _write_job(run, job)
    _PROCESSES.pop(_process_key(run), None)
    return job


def _last_error_line(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-80:]):
        if any(word in line for word in ("Error", "error", "Exception", "Traceback")):
            return line.strip()[:300]
    return ""


def stop_training(run: Path) -> None:
    job = _read_job(run)
    (run / "stop.request").write_text("stop", encoding="utf-8")
    pid = int(job.get("pid") or 0)
    if pid <= 0:
        return
    # PID 已被别的程序复用时不能按号杀进程树。
    if _PROCESSES.get(_process_key(run)) is None and not _job_process_alive(job):
        return
    # Ultralytics 用 DataLoader 起了一堆子进程，必须连进程树一起结束。
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass


def active_run(project_folder: Path) -> Path | None:
    root = runs_dir(project_folder)
    if not root.is_dir():
        return None
    for run in sorted(root.iterdir(), reverse=True):
        if run.is_dir() and refresh_status(run).get("status") == "training":
            return run
    return None


def read_results(run: Path) -> dict:
    """results.csv → 按列的数组，曲线直接画。列名去掉空格，(B)/(OBB) 这类后缀原样保留。"""
    path = run / "results.csv"
    if not path.is_file():
        return {"columns": [], "rows": 0, "series": {}}
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return {"columns": [], "rows": 0, "series": {}}
    if len(rows) < 2:
        return {"columns": [], "rows": 0, "series": {}}
    header = [item.strip() for item in rows[0]]
    series: dict[str, list[float | None]] = {name: [] for name in header}
    for row in rows[1:]:
        for index, name in enumerate(header):
            value = row[index].strip() if index < len(row) else ""
            try:
                series[name].append(float(value))
            except ValueError:
                series[name].append(None)
    return {"columns": header, "rows": len(rows) - 1, "series": series}


def _metric(series: dict, *keys: str) -> list:
    for key in series:
        if any(key.startswith(prefix) for prefix in keys):
            return series[key]
    return []


_NAMED_PLOTS = (
    ("results.png", "eval", "训练总览曲线 (results.png)"),
    ("confusion_matrix_normalized.png", "eval", "归一化混淆矩阵"),
    ("confusion_matrix.png", "eval", "混淆矩阵 (样本计数)"),
    ("OBBPR_curve.png", "eval", "PR 精确率-召回率曲线 (OBB)"),
    ("BoxPR_curve.png", "eval", "PR 精确率-召回率曲线"),
    ("PR_curve.png", "eval", "PR 精确率-召回率曲线"),
    ("OBBF1_curve.png", "eval", "F1-置信度曲线 (OBB)"),
    ("BoxF1_curve.png", "eval", "F1-置信度曲线"),
    ("F1_curve.png", "eval", "F1-置信度曲线"),
    ("OBBP_curve.png", "eval", "精确率 P-置信度曲线 (OBB)"),
    ("BoxP_curve.png", "eval", "精确率 P-置信度曲线"),
    ("P_curve.png", "eval", "精确率 P-置信度曲线"),
    ("OBBR_curve.png", "eval", "召回率 R-置信度曲线 (OBB)"),
    ("BoxR_curve.png", "eval", "召回率 R-置信度曲线"),
    ("R_curve.png", "eval", "召回率 R-置信度曲线"),
    ("labels.jpg", "data", "训练集类别、框尺寸与角度分布"),
    ("labels.png", "data", "训练集类别、框尺寸与角度分布"),
    ("labels_correlogram.jpg", "data", "标注几何相关性分布图"),
    ("labels_correlogram.png", "data", "标注几何相关性分布图"),
)


def list_run_artifacts(run: Path) -> tuple[list[dict], list[dict]]:
    """枚举 Ultralytics 训练在轮次目录下生成的图表与批次预览图，并配对验证集真值/预测图。"""
    if not run.is_dir():
        return [], []
    artifacts: list[dict] = []
    seen: set[str] = set()

    def add_file(path: Path, category: str, label: str) -> None:
        if not path.is_file() or path.is_symlink() or path.name in seen:
            return
        try:
            st = path.stat()
        except OSError:
            return
        seen.add(path.name)
        artifacts.append({
            "name": path.name,
            "category": category,
            "label": label,
            "mtime": int(st.st_mtime),
            "bytes": int(st.st_size),
        })

    for fname, category, label in _NAMED_PLOTS:
        add_file(run / fname, category, label)

    train_batches = sorted(
        (p for p in run.iterdir() if re.fullmatch(r"train_batch\d+\.(?:jpg|jpeg|png)", p.name, re.I)),
        key=lambda p: int(re.search(r"\d+", p.name).group(0)),
    )
    for path in train_batches:
        idx = int(re.search(r"\d+", path.name).group(0))
        add_file(path, "train_batch", f"训练增强批次 #{idx} ({path.name})")

    val_batches: dict[int, dict[str, str]] = {}
    for p in run.iterdir():
        m = re.fullmatch(r"val_batch(\d+)_(labels|pred)\.(?:jpg|jpeg|png)", p.name, re.I)
        if not m or not p.is_file() or p.is_symlink():
            continue
        b_idx = int(m.group(1))
        kind = m.group(2).lower()
        val_batches.setdefault(b_idx, {})[kind] = p.name

    val_pairs: list[dict] = []
    for b_idx in sorted(val_batches):
        pair = val_batches[b_idx]
        lbl_name = pair.get("labels", "")
        pred_name = pair.get("pred", "")
        if lbl_name:
            add_file(run / lbl_name, "val_batch", f"验证批次 #{b_idx} 人工真值")
        if pred_name:
            add_file(run / pred_name, "val_batch", f"验证批次 #{b_idx} 模型预测")
        mtime = 0
        for fn in (lbl_name, pred_name):
            if fn and (run / fn).is_file():
                try:
                    mtime = max(mtime, int((run / fn).stat().st_mtime))
                except OSError:
                    pass
        val_pairs.append({
            "batch": b_idx,
            "labels": lbl_name,
            "pred": pred_name,
            "mtime": mtime,
        })

    return artifacts, val_pairs


def run_summary(run: Path) -> dict:
    job = refresh_status(run)
    results = read_results(run)
    series = results["series"]
    epochs_total = int((job.get("params") or {}).get("epochs") or 0)
    map50 = _metric(series, "metrics/mAP50(")
    map5095 = _metric(series, "metrics/mAP50-95(")
    precision = _metric(series, "metrics/precision(")
    recall = _metric(series, "metrics/recall(")
    best_index = None
    if map5095:
        valid = [(value, index) for index, value in enumerate(map5095) if value is not None]
        if valid:
            best_index = max(valid)[1]
    exported = _read_export(run)
    started_at = job.get("started_at")
    finished_at = job.get("finished_at")
    epoch_done = results["rows"]
    elapsed = None
    eta_seconds = None
    if started_at:
        end_ref = finished_at if finished_at else time.time()
        elapsed = max(0.0, float(end_ref) - float(started_at))
        if job.get("status") == "training" and 0 < epoch_done < epochs_total:
            eta_seconds = max(1, int(round((epochs_total - epoch_done) * (elapsed / epoch_done))))
    return {
        "name": run.name,
        "status": job.get("status") or "unknown",
        "error": job.get("error") or "",
        "params": job.get("params") or {},
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
        "eta_seconds": eta_seconds,
        "epoch": epoch_done,
        "epochs": epochs_total,
        "best_epoch": (best_index + 1) if best_index is not None else None,
        "map50": map50[best_index] if best_index is not None and best_index < len(map50) else None,
        "map5095": map5095[best_index] if best_index is not None else None,
        "precision": precision[best_index] if best_index is not None and best_index < len(precision) else None,
        "recall": recall[best_index] if best_index is not None and best_index < len(recall) else None,
        "has_best": (run / "weights" / "best.pt").is_file(),
        "has_last": (run / "weights" / "last.pt").is_file(),
        "resumable": job.get('status') in {'stopped','failed'} and (run/'weights'/'last.pt').is_file() and bool(job.get('dataset')),
        "dataset": job.get('dataset'),
        "model": job.get('model'),
        "environment": job.get('environment'),
        "resume_count": job.get('resume_count', 0),
        "exported": exported,
    }


def run_detail(run: Path) -> dict:
    summary = run_summary(run)
    results = read_results(run)
    artifacts, val_pairs = list_run_artifacts(run)
    summary["series"] = results["series"]
    summary["columns"] = results["columns"]
    summary["command"] = (_read_job(run).get("command") or [])
    summary["artifacts"] = artifacts
    summary["val_pairs"] = val_pairs
    return summary


def read_log(run: Path, after: int = 0, limit: int = 400) -> dict:
    """日志增量读取。Ultralytics 的进度条用 \\r 覆盖同一行，这里按 \\r 也切行，只留最后一段。"""
    path = run / "train.log"
    if not path.is_file():
        return {"lines": [], "next": 0}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"lines": [], "next": after}
    # 去掉终端控制符（颜色、清行 \x1b[K），页面上不显示乱码。
    text = _ANSI.sub("", text)
    lines = [segment.split("\r")[-1] for segment in text.split("\n")]
    lines = [line for line in lines if line.strip()]
    start = max(0, int(after))
    chunk = lines[start:start + limit]
    return {"lines": chunk, "next": start + len(chunk)}


def list_runs(project_folder: Path) -> list[dict]:
    root = runs_dir(project_folder)
    if not root.is_dir():
        return []
    return [run_summary(run) for run in sorted(root.iterdir(), reverse=True) if run.is_dir() and (run / "job.json").is_file()]


# ---------- 导出 ----------

def _read_export(run: Path) -> dict | None:
    try:
        return json.loads((run / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def export_onnx(project_id: str, project_folder: Path, run: Path, imgsz: int | None = None) -> dict:
    """best.pt → ONNX，放进全局模型库 models/yolo，文件名带工程和轮次，避免同名覆盖。

    导出固定用 CPU、opset 12，和产线 ONNX Runtime 的使用方式一致（参照 models/export_onnx.py）。
    """
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError("这一轮还没有 best.pt，训练没跑完或失败了")
    job = _read_job(run)
    size = int(imgsz or (job.get("params") or {}).get("imgsz") or 640)
    script = (
        "import sys\n"
        "from ultralytics import YOLO\n"
        f"path = YOLO(sys.argv[1]).export(format='onnx', imgsz={size}, opset={EXPORT_OPSET},"
        " simplify=True, dynamic=False, half=False, device='cpu')\n"
        "print('EXPORTED=' + str(path))\n"
    )
    env = _child_env(project_folder)
    done = subprocess.run(
        [_python(), "-c", script, str(best)], capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace", env=env, creationflags=_NO_WINDOW,
    )
    produced = None
    for line in (done.stdout or "").splitlines():
        if line.startswith("EXPORTED="):
            produced = Path(line.split("=", 1)[1].strip())
    if done.returncode != 0 or produced is None or not produced.is_file():
        tail = ((done.stderr or "") + (done.stdout or "")).strip().splitlines()[-3:]
        raise RuntimeError("导出 ONNX 失败：" + " ".join(tail)[:400])
    safe_project = re.sub(r"[^\w\-]+", "_", project_id).strip("_") or "project"
    library = model_registry.YOLO_LIBRARY
    library.mkdir(parents=True, exist_ok=True)
    target = library / f"{safe_project}-{run.name}.onnx"
    produced_sha = model_registry.file_sha256(produced)
    # 别的工程按 sha256 钉住了库里的文件。同名文件内容不同时（比如换了 imgsz 重新导出）
    # 不能原地覆盖，否则那些工程会因为指纹对不上而拒绝运行；改用带尺寸的新文件名。
    if target.is_file() and model_registry.file_sha256(target) != produced_sha:
        target = library / f"{safe_project}-{run.name}-{size}.onnx"
        suffix = 2
        while target.is_file() and model_registry.file_sha256(target) != produced_sha:
            target = library / f"{safe_project}-{run.name}-{size}-{suffix}.onnx"
            suffix += 1
    if not (target.is_file() and model_registry.file_sha256(target) == produced_sha):
        # 先复制到同目录临时文件再改名：复制到一半失败时库里不会出现半截模型。
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            shutil.copy2(produced, temporary)
            os.replace(temporary, target)
        except OSError as exc:
            raise RuntimeError(f"模型写不进模型库，文件可能正被占用：{exc}") from None
        finally:
            temporary.unlink(missing_ok=True)
    info = {
        "file": target.name,
        "sha256": produced_sha,
        "source_sha256": model_registry.file_sha256(best),
        "source_file": "best.pt",
        "imgsz": size,
        "exported_at": time.time(),
        "project": project_id,
        "run": run.name,
    }
    _write_json(run / "export.json", info)
    # 模型库卡片要显示来源，旁边放一份同名说明。
    _write_json(target.with_suffix(".json"), info)
    return info
