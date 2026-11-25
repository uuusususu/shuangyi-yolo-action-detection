"""后台准备依赖的缓存复用与安全写入契约。"""
from copy import deepcopy

import numpy as np
import pytest

from lstm_workbench import feature_store
from lstm_workbench.tests.test_feature_store import _fake, _project, _settings


def test_editing_labels_and_training_settings_reuses_features(tmp_path, monkeypatch):
    settings = _settings()
    _, folder, annotations = _project(tmp_path, settings)
    calls = []
    _fake(monkeypatch, calls)
    feature_store.extract_project(folder, annotations, settings)
    changed = deepcopy(annotations)
    changed['classes'] = [{'id': 'wrap', 'name': '包装'}]
    changed['clips'][0]['segments'] = [{'id': 's', 'class_id': 'wrap', 'start': 0, 'end': 2}]
    before = len(calls)
    _, report = feature_store.extract_project(folder, changed, {**settings, 'epochs': 99, 'learning_rate': .02})
    assert report['reused'] == 1
    assert len(calls) == before


def test_corrupt_cache_is_reported_stale_and_repaired(tmp_path, monkeypatch):
    settings = _settings()
    _, folder, annotations = _project(tmp_path, settings)
    calls = []
    _fake(monkeypatch, calls)
    feature_store.extract_project(folder, annotations, settings)
    (folder / 'features/a.npy').write_bytes(b'interrupted numpy file')
    assert feature_store.summary(folder, annotations, settings)['fresh'] == 0
    before = len(calls)
    arrays, report = feature_store.extract_project(folder, annotations, settings)
    assert report['extracted'] == 1 and len(calls) > before
    assert arrays['a'].shape[0] == 2


def test_failed_write_keeps_previous_cache_and_manifest(tmp_path, monkeypatch):
    settings = _settings()
    _, folder, annotations = _project(tmp_path, settings)
    _fake(monkeypatch, [])
    feature_store.extract_project(folder, annotations, settings)
    npy = folder / 'features/a.npy'
    previous = npy.read_bytes()
    manifest = (folder / 'features/manifest.json').read_bytes()

    def interrupted(path, array):
        from pathlib import Path
        Path(path).write_bytes(b'partial')
        raise OSError('模拟磁盘写入中断')

    monkeypatch.setattr(np, 'save', interrupted)
    with pytest.raises(OSError):
        feature_store.extract_project(folder, annotations, {**settings, 'yolo_sha256': 'new'})
    assert npy.read_bytes() == previous
    assert (folder / 'features/manifest.json').read_bytes() == manifest
