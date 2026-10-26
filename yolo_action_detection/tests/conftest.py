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

    不用 deleteLater：顶层控件由 Python 持有，C++ 侧删除后 Python 析构会重复释放。
    改用 shiboken6.delete 先按 Qt 父子顺序同步销毁顶层控件，shiboken 会同时让
    Python 包装失效，不会二次释放。若只靠 GC 回收"主窗口 + 子对话框"的引用环，
    回收顺序随内存布局变化，曾在 Windows 上触发堆损坏（0xc0000374）。
    """
    yield

    try:
        import shiboken6
        from PySide6.QtWidgets import QApplication
    except ImportError:
        gc.collect()
        return

    app = QApplication.instance()
    if app is not None:
        from PySide6.QtWidgets import QDialog, QMainWindow

        for widget in list(app.topLevelWidgets()):
            # 只删用例自己建的窗口/对话框，不碰 Qt 内部的顶层控件（如提示气泡）。
            # 不调 close()：那会走 closeEvent 业务收尾，用例里的替身未必支持。
            if (
                isinstance(widget, (QMainWindow, QDialog))
                and shiboken6.isValid(widget)
                and widget.parent() is None
            ):
                shiboken6.delete(widget)
    gc.collect()
    if app is not None:
        app.processEvents()
