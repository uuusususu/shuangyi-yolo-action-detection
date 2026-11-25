"""工作台全局协调器：管理工程存储、单卡 GPU 互斥锁、工程数据锁、后台特征准备与远程传输。"""
from __future__ import annotations

import os
from pathlib import Path
import threading

from . import dataset_transfer, model_registry, yolo_jobs
from .lan_auth import LanAuthManager
from .preparation import PreparationManager
from .project_store import ProjectStore
from .smart_label import SmartLabelManager


class Workbench:
    def __init__(self, projects_root: Path, *, lan_auth: LanAuthManager | None = None):
        self.store = ProjectStore(projects_root)
        try:
            model_registry.ensure_sam_library()
        except OSError:
            pass
        self._locks: dict[str, threading.Lock] = {}
        self._data_locks: dict[str, threading.RLock] = {}
        self._stops: dict[str, threading.Event] = {}
        self._guards = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._foreground_projects: set[str] = set()
        self._background_project: str | None = None
        self._async_jobs: dict[str, threading.Thread] = {}
        self.lan_enabled: bool = False
        self.listen_port: int = 8771
        self.lan_auth = lan_auth or LanAuthManager()
        self.smart_label = SmartLabelManager(self)
        self.preparation = PreparationManager(self)
        self.transfer = dataset_transfer.TransferManager(self)

    def is_training_locked(self) -> bool:
        """是否有任意 YOLO 训练或前台 GPU/训练任务正在运行。运行期间彻底禁用智能标注。"""
        if self.yolo_training():
            return True
        with self._guards:
            return bool(self._foreground_projects)

    def preempt_smart_label(self, reason: str = "训练或重型 GPU 任务即将启动，已自动停止标注模型释放显存", timeout: float = 5.0) -> None:
        """训练或重型 GPU 任务启动前，同步停止常驻智能标注模型并确认显存归还。"""
        self.smart_label.stop_and_release(reason=reason, timeout=timeout)

    def begin_preparation(self, project_id: str) -> bool:
        lock, key = self.lock_for(project_id), self._key(project_id)
        if self.yolo_training() or self.smart_label.is_active():
            return False
        with self._guards:
            if self._foreground_projects or not lock.acquire(blocking=False):
                return False
            if not self._gpu_lock.acquire(blocking=False):
                lock.release()
                return False
            self._background_project = key
            return True

    def end_preparation(self, project_id: str) -> None:
        lock = self.lock_for(project_id)
        with self._guards:
            self._background_project = None
            self._gpu_lock.release()
            lock.release()

    def begin_job(self, project_id: str) -> bool:
        lock, key = self.lock_for(project_id), self._key(project_id)
        stop = self.stop_event(project_id)
        with self._guards:
            if self._foreground_projects or (lock.locked() and self._background_project != key):
                return False
            self._foreground_projects.add(key)
            stop.clear()
        lock.acquire()
        self._gpu_lock.acquire()
        try:
            busy = self.yolo_training()
            if not busy:
                self.preempt_smart_label("前台训练或特征任务即将启动，已停止智能标注模型释放显存")
        except BaseException:
            self.end_job(project_id)
            raise
        if busy:
            self.end_job(project_id)
            return False
        return True

    def try_begin_job(self, project_id: str) -> bool:
        """交互式后台任务快速获取全局资源，忙碌时由网页重试，不阻塞 HTTP。"""
        lock, key = self.lock_for(project_id), self._key(project_id)
        with self._guards:
            if self._foreground_projects or self._background_project is not None or not lock.acquire(blocking=False):
                return False
            if not self._gpu_lock.acquire(blocking=False):
                lock.release()
                return False
            self._foreground_projects.add(key)
            self.stop_event_unlocked(project_id).clear()
        try:
            self.preempt_smart_label("零件验证或导出任务即将启动，已停止智能标注模型释放显存")
        except BaseException:
            self.end_job(project_id)
            raise
        return True

    def stop_event_unlocked(self, project_id: str) -> threading.Event:
        """调用方已经持有 _guards。"""
        return self._stops.setdefault(self._key(project_id), threading.Event())

    def end_job(self, project_id: str) -> None:
        lock, key = self.lock_for(project_id), self._key(project_id)
        with self._guards:
            self._gpu_lock.release()
            lock.release()
            self._foreground_projects.discard(key)

    def _key(self, project_id: str) -> str:
        try:
            return os.path.normcase(str(self.store.project_dir(project_id).resolve()))
        except (ValueError, OSError):
            return project_id

    def lock_for(self, project_id: str) -> threading.Lock:
        """任务锁：抽特征、训练、检查这类长任务同一工程同时只跑一个。"""
        with self._guards:
            return self._locks.setdefault(self._key(project_id), threading.Lock())

    def data_lock(self, project_id: str) -> threading.RLock:
        """数据锁：读改写 annotations.json / settings.json 时持有，与长任务锁分离。"""
        with self._guards:
            return self._data_locks.setdefault(self._key(project_id), threading.RLock())

    def stop_event(self, project_id: str) -> threading.Event:
        """停止不走工程锁，训练还在跑的时候也要能收到。"""
        with self._guards:
            return self._stops.setdefault(self._key(project_id), threading.Event())

    def any_job_running(self) -> bool:
        """任意工程的抽特征 / 动作训练 / 检查是否在跑。"""
        with self._guards:
            locks = list(self._locks.values())
        return any(lock.locked() for lock in locks)

    def yolo_training(self) -> str:
        """正在训练零件模型的"工程/轮次"，没有就是空串。显卡只有一张，全局互斥。"""
        for project_id in self.store.list_projects():
            run = yolo_jobs.active_run(self.store.root / project_id)
            if run is not None:
                return f"{project_id}/{run.name}"
        return ""

    def close_async_jobs(self) -> None:
        """关闭服务时通知推理子进程退出，等待其保存终态并释放资源。"""
        self.smart_label.stop_and_release(reason="训练台服务正在关闭", timeout=3.0)
        with self._guards:
            stops = list(self._stops.values())
            threads = list(self._async_jobs.values())
        for stop in stops:
            stop.set()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=6)
