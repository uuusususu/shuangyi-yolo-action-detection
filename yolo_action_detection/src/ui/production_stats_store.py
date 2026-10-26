"""产能计数和配置之间的读写：启动时从配置恢复，计数变化后合并写盘。

从 MainWindow 拆出来，主窗口只负责把计数显示出来。
"""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer

from detection_logging.production_stats import ProductionBatch, ProductionStatsManager

# 产能计数合并写盘的间隔。断电最多丢这段时间内的计数。
STATS_PERSIST_DELAY_MS = 2000


def batch_from_config(config) -> ProductionBatch:
    """从配置里的 production_* 字段恢复当前批次。"""
    started = float(getattr(config, "production_started_at", 0) or 0)
    if started <= 0:
        started = time.time()
    ok = int(getattr(config, "production_ok", 0) or 0)
    ng = int(getattr(config, "production_ng", 0) or 0)
    return ProductionBatch(started_at=started, total=ok + ng, ok=ok, ng=ng)


class ProductionStatsStore(QObject):
    """把 ProductionStatsManager 的计数合并后写进当前配置。

    每个产品都整份 fsync 配置会卡住界面线程，所以计数变化只启动定时器，
    到点或调用 flush() 时再写。config_provider 每次取当前配置，
    配置页保存后换了实例也能写到新配置里。
    """

    def __init__(
        self,
        manager: ProductionStatsManager,
        config_provider: Callable[[], object],
        on_error: Callable[[str], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._config_provider = config_provider
        self._on_error = on_error
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(STATS_PERSIST_DELAY_MS)
        self._timer.timeout.connect(self.persist)
        manager.set_on_changed(self.schedule)

    @property
    def pending(self) -> bool:
        return self._timer.isActive()

    def schedule(self) -> None:
        # 已在计时就不重启：连续出件时也保证每个间隔至少落盘一次。
        if not self._timer.isActive():
            self._timer.start()

    def flush(self) -> None:
        """立即写入当前计数（关窗时调用）。

        不只看定时器：配置页提交时会用自己的候选配置整份覆盖文件，
        这之后定时器可能已空闲，但盘上的计数未必是最新的。
        """
        self.persist()

    def apply_to(self, config) -> None:
        """把内存里的最新计数写进 config 对象（不落盘）。

        配置页保存时候选配置是点击那一刻深拷贝的，之后还可能出件；
        保存前调用它，避免候选配置把较新的计数覆盖回旧值。
        """
        batch = self._manager.batch
        config.production_started_at = float(batch.started_at)
        config.production_ok = int(batch.ok)
        config.production_ng = int(batch.ng)
        config.production_total = int(batch.ok) + int(batch.ng)

    def persist(self) -> None:
        """把当前批次写进正在使用的配置。没有配置路径时只留在内存里。"""
        self._timer.stop()
        config = self._config_provider()
        self.apply_to(config)
        path = config.get_config_path()
        if not path:
            return
        try:
            config.save(path)
        except Exception as exc:
            self._on_error(f"产能计数未能写入配置：{exc}")
