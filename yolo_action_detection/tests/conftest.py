from __future__ import annotations

import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _reclaim_qt_widgets():
    """回收用例遗留的顶层控件。

    主窗口和配置页都靠信号连接构成引用环，分代 GC 不会及时回收它们。
    存活控件堆积后，QApplication.setStyleSheet 这类全局操作要按控件数量
    重新 polish，整套用例的耗时会随文件数量不断累积。

    这里只触发 GC 让 Python 自己释放所有权，不主动 deleteLater：顶层控件由
    Python 持有，手动删除会和析构重复释放同一块内存。
    """
    yield

    gc.collect()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
