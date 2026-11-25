"""第三轮复审确认的问题：状态刷新竞争、锁泄漏、后台线程退出、同名上传、PID 复用、导出覆盖、残留目录、盘符文件名。"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from lstm_workbench import model_registry, preparation, yolo_jobs, yolo_validation
from lstm_workbench.app import Workbench
from lstm_workbench.project_store import ProjectStore


class _ExitedProcess:
    """一个已经正常退出的训练进程句柄。"""
    pid = 4242

    def poll(self):
        return 0


def _training_run(tmp_path: Path) -> Path:
    run = tmp_path / "p" / "yolo" / "runs" / "one"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    yolo_jobs._write_job(run, {"name": "one", "status": "training", "pid": 4242, "params": {"epochs": 1}})
    return run


def test_concurrent_refresh_keeps_finished_run_done(tmp_path, monkeypatch):
    """两个线程同时发现进程退出：先写 done 的那个移除了句柄，后一个不能拿旧读数改成 failed。

    固定交错顺序：B 先读到 training 后停住，A 完整跑完（写 done、移除句柄），B 再继续。"""
    run = _training_run(tmp_path)
    key = yolo_jobs._process_key(run)
    yolo_jobs._PROCESSES[key] = _ExitedProcess()
    monkeypatch.setattr(yolo_jobs, "_pid_alive", lambda pid, created=None: False)
    original = yolo_jobs._read_job
    b_has_read = threading.Event()
    a_finished = threading.Event()

    def staged_read(path):
        job = original(path)
        if threading.current_thread().name == "B" and job.get("status") == "training" and not b_has_read.is_set():
            b_has_read.set()
            a_finished.wait(5)
        return job

    monkeypatch.setattr(yolo_jobs, "_read_job", staged_read)
    results = {}

    def run_b():
        results["b"] = yolo_jobs.refresh_status(run)["status"]

    def run_a():
        b_has_read.wait(5)
        results["a"] = yolo_jobs.refresh_status(run)["status"]
        a_finished.set()

    tb = threading.Thread(target=run_b, name="B")
    ta = threading.Thread(target=run_a, name="A")
    tb.start()
    ta.start()
    ta.join(10)
    a_finished.set()
    tb.join(10)
    monkeypatch.setattr(yolo_jobs, "_read_job", original)
    yolo_jobs._PROCESSES.pop(key, None)
    assert yolo_jobs._read_job(run)["status"] == "done"
    assert results == {"a": "done", "b": "done"}


def test_begin_job_releases_locks_when_status_check_raises(tmp_path, monkeypatch):
    wb = Workbench(tmp_path)
    ProjectStore(tmp_path).create("p", {})
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("job.json 被占用")
        return ""

    monkeypatch.setattr(wb, "yolo_training", flaky)
    with pytest.raises(OSError):
        wb.begin_job("p")
    assert wb.try_begin_job("p") is True, "锁必须已经放掉"
    wb.end_job("p")
    wb.preparation.close()


def test_preparation_thread_survives_status_check_error(tmp_path, monkeypatch):
    wb = Workbench(tmp_path)
    ProjectStore(tmp_path).create("p", {})
    calls = {"n": 0}

    def flaky(project_id):
        calls["n"] += 1
        raise ValueError("yolo 目录越界")

    monkeypatch.setattr(wb, "begin_preparation", flaky)
    manager = wb.preparation
    manager.start()
    with manager._condition:
        manager._states["p"] = {"paused": False, "pending": ["c1"], "errors": {}}
        manager._condition.notify_all()
    deadline = time.time() + 5
    while calls["n"] < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert calls["n"] >= 2, "抛错后后台线程应继续重试"
    assert manager._thread.is_alive()
    manager.close()


def test_write_json_retries_while_target_is_briefly_locked(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    real = os.replace
    attempts = {"n": 0}

    def flaky(src, dst):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError(5, "拒绝访问")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    ProjectStore.write_json(target, {"ok": 1})
    assert target.read_text(encoding="utf-8").strip().startswith("{")
    assert not list(tmp_path.glob(".a.json.*.tmp"))


def test_upload_name_with_drive_letter_is_refused(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("p", {})
    for name in ("c:evil.mp4", "C:\\evil.mp4", "a/c:x.mp4"):
        with pytest.raises(ValueError):
            store._upload_target("p", name)
    assert store._upload_target("p", "ok.mp4").name == "ok.mp4"


def test_validation_record_with_reused_pid_is_marked_interrupted(tmp_path, monkeypatch):
    folder = tmp_path / "p"
    root = yolo_validation._root(folder)
    record_dir = root / "v1"
    record_dir.mkdir(parents=True)
    me = os.getpid()
    other = me + 1
    monkeypatch.setattr(yolo_jobs, "_pid_created", lambda pid: 1000.0)
    yolo_validation._write(record_dir / "result.json", {
        "id": "v1", "status": "running", "owner_pid": other, "owner_created": 5.0, "created_at": 1,
        "items": [], "classes": [], "models": [], "mode": "validate", "params": {},
    })
    assert yolo_validation.mark_interrupted(folder) == 1
    assert yolo_validation.validation_detail(folder, "v1")["status"] == "failed"


def test_reexport_with_different_content_does_not_overwrite_pinned_file(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    monkeypatch.setattr(model_registry, "YOLO_LIBRARY", library)
    library.mkdir()
    pinned = library / "p-run1.onnx"
    pinned.write_bytes(b"old-model")
    pinned_sha = model_registry.file_sha256(pinned)
    run = tmp_path / "p" / "yolo" / "runs" / "run1"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    produced = tmp_path / "out.onnx"
    produced.write_bytes(b"new-model")

    class Done:
        returncode = 0
        stdout = f"EXPORTED={produced}\n"
        stderr = ""

    monkeypatch.setattr(yolo_jobs.subprocess, "run", lambda *a, **k: Done())
    monkeypatch.setattr(yolo_jobs, "_child_env", lambda folder: {})
    monkeypatch.setattr(yolo_jobs, "_python", lambda: "python")
    info = yolo_jobs.export_onnx("p", tmp_path / "p", run, 960)
    assert model_registry.file_sha256(pinned) == pinned_sha, "被别的工程钉住的文件不能被改写"
    assert info["file"] != "p-run1.onnx"
    assert (library / info["file"]).read_bytes() == b"new-model"


def test_failed_start_removes_orphan_run_folder(tmp_path, monkeypatch):
    folder = tmp_path / "p"
    (folder / "yolo").mkdir(parents=True)
    monkeypatch.setattr(yolo_jobs, "_python", lambda: "python")
    monkeypatch.setattr(yolo_jobs, "_freeze_dataset", lambda f, run, version: {"data_yaml": str(run / "data.yaml")})

    real_build = yolo_jobs.build_command
    calls = {"n": 0}

    def broken(*a, **k):
        # 第一次是建目录前的表单预检，放过；第二次在复制数据集之后，模拟失败。
        calls["n"] += 1
        if calls["n"] == 1:
            return real_build(*a, **k)
        raise ValueError("参数不合法")

    monkeypatch.setattr(yolo_jobs, "build_command", broken)
    with pytest.raises(ValueError):
        yolo_jobs.start_training(folder, {"task": "obb"})
    runs = yolo_jobs.runs_dir(folder)
    assert not runs.exists() or not any(runs.iterdir())
