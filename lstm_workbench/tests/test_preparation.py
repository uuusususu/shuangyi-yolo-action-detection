"""后台任务的去重、恢复、排除与前台接续。"""
import threading
import time

import numpy as np

from lstm_workbench.app import Workbench
from lstm_workbench.preparation import PreparationManager


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('后台任务未在期限内完成')


def setup_project(tmp_path, monkeypatch):
    from lstm_workbench import feature_store
    wb = Workbench(tmp_path)
    wb.store.create('demo', {'yolo_path': 'fake.onnx', 'yolo_classes': ['part'], 'yolo_sha256': 'abc', 'hands_in_training': False})
    annotations = {'classes': [{'id': 'wrap', 'name': '包装'}], 'clips': []}
    for name in ['a', 'b']:
        (tmp_path / 'demo' / 'media' / f'{name}.mp4').write_bytes(b'video')
        annotations['clips'].append({'id': name, 'frame_count': 2, 'fps': 10, 'width': 16, 'height': 8, 'segments': [], 'annotation_complete': True})
    wb.store.save_annotations('demo', annotations)
    calls = []
    frame = np.zeros((8, 16, 3), np.uint8)
    monkeypatch.setattr(feature_store, 'iter_frames', lambda path: iter([frame, frame]))
    def detect(_frame):
        calls.append(1)
        return []
    monkeypatch.setattr(feature_store, 'yolo_backend', lambda settings: detect)
    return wb, calls


def test_deduplication_reuse_and_exclusion(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    settings = wb.store.load_settings('demo')
    settings['train_excluded'] = ['b']
    wb.store.save_settings('demo', settings)
    manager = wb.preparation
    try:
        manager.request('demo')
        manager.request('demo')
        manager.start()
        wait_for(lambda: manager.status('demo')['ready'] == 1)
        manager.request('demo')
        assert len(calls) == 2
        assert manager.status('demo')['total'] == 1
    finally:
        manager.close()


def test_pause_survives_restart_and_resume_skips_ready(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    first = wb.preparation
    first.request('demo', ['a'])
    first.pause('demo')
    first.close()
    second = PreparationManager(wb)
    try:
        second.start()
        assert second.status('demo')['paused']
        second.request('demo', resume=True)
        wait_for(lambda: second.status('demo')['ready'] == 2)
        assert len(calls) == 4
    finally:
        second.close()


def test_foreground_waits_for_background_and_prevents_duplicate_work(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    from lstm_workbench import feature_store
    original = feature_store.extract_project
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(feature_store, 'extract_project', slow)
    manager = wb.preparation
    acquired = threading.Event()
    try:
        manager.request('demo', ['a'])
        manager.start()
        assert entered.wait(5)
        def foreground():
            assert wb.begin_job('demo')
            acquired.set()
            wb.end_job('demo')
        thread = threading.Thread(target=foreground)
        thread.start()
        assert not acquired.wait(.05)
        release.set()
        thread.join(5)
        assert acquired.is_set()
        assert len(calls) == 2
        assert manager.status('demo')['ready'] == 1
    finally:
        release.set()
        manager.close()


def test_failure_is_visible_and_explicit_retry_repairs_it(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    from lstm_workbench import feature_store
    original = feature_store.extract_project
    monkeypatch.setattr(feature_store, 'extract_project', lambda *a, **kw: (_ for _ in ()).throw(OSError('测试读取失败')))
    manager = wb.preparation
    try:
        manager.request('demo', ['a'])
        manager.start()
        wait_for(lambda: manager.status('demo')['failed'] == 1)
        monkeypatch.setattr(feature_store, 'extract_project', original)
        manager.request('demo', ['a'], resume=True)
        wait_for(lambda: manager.status('demo')['ready'] == 1)
        assert len(calls) == 2
    finally:
        manager.close()


def test_restart_reuses_completed_files(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    manager = wb.preparation
    manager.request('demo', ['a'])
    manager.start()
    wait_for(lambda: manager.status('demo')['ready'] == 1)
    manager.close()
    restored = PreparationManager(wb)
    try:
        restored.request('demo')
        restored.start()
        wait_for(lambda: restored.status('demo')['ready'] == 2)
        assert len(calls) == 4
    finally:
        restored.close()


def test_yolo_occupancy_keeps_background_queued(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    monkeypatch.setattr(wb, 'yolo_training', lambda: 'demo/run')
    manager = wb.preparation
    try:
        manager.request('demo', ['a'])
        manager.start()
        assert manager.status('demo')['queued'] == 1
        assert not calls
        monkeypatch.setattr(wb, 'yolo_training', lambda: '')
        wait_for(lambda: manager.status('demo')['ready'] == 1)
    finally:
        manager.close()


def test_invalid_clip_ids_cannot_enter_the_queue(tmp_path, monkeypatch):
    import pytest
    wb, _ = setup_project(tmp_path, monkeypatch)
    for selected in [['../outside'], ['missing'], 'a', [None]]:
        with pytest.raises(ValueError):
            wb.preparation.request('demo', selected)
