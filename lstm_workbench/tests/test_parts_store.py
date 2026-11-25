"""零件素材、四点标注及不可变数据版本。全部使用临时工程。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lstm_workbench.parts_store import PartsStore


def image(path: Path, color: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((48, 64, 3), color, dtype=np.uint8)
    array[8:24, 8:24, 1] = 255
    ok, data = cv2.imencode('.png', array)
    assert ok
    path.write_bytes(data.tobytes())
    return path


def box(class_id=0):
    return {'class_id': class_id, 'points': [[.1, .1], [.7, .1], [.7, .7], [.1, .7]]}


def store_with_items(tmp_path, count=3):
    store = PartsStore(tmp_path / '工程')
    store.set_classes(['零件甲', '零件乙'], 0)
    for n in range(count):
        store.import_source(str(image(tmp_path / f'来源{n}' / '图片.png', n * 50)))
    for item in store.snapshot()['items']:
        store.save_item(item['id'], [box()], 'reviewed', item['revision'])
    return store


def test_import_and_revision_and_classes(tmp_path):
    store = PartsStore(tmp_path / '工程')
    assert store.snapshot()['items'] == []
    assert store.snapshot()['revision'] == 0
    snap = store.set_classes(['垫片', '螺丝'], 0)
    assert snap['revision'] == 1
    source = image(tmp_path / '原始' / '中文图.png')
    snap = store.import_source(str(source), '批次甲')
    item = snap['items'][0]
    assert item['source_group'] == '批次甲'
    assert item['name'] == '中文图.png'
    assert item['width'] == 64 and item['height'] == 48
    assert item['status'] == 'unlabeled' and item['boxes'] == []
    assert store.image_path(item['id']).read_bytes() == source.read_bytes()
    assert store.import_source(str(source))['items'] == snap['items']
    saved = store.save_item(item['id'], [box()], 'pending', item['revision'])
    assert saved['revision'] > item['revision']
    with pytest.raises(ValueError, match='修订|冲突'):
        store.save_item(item['id'], [], 'negative', item['revision'])
    with pytest.raises(ValueError, match='修订|冲突'):
        store.set_classes(['其他'], 0)
    with pytest.raises(ValueError):
        store.set_classes(['螺丝', '垫片'], store.snapshot()['revision'])
    assert store.set_classes(['垫片', '螺丝', '外壳'], store.snapshot()['revision'])['classes'][-1] == '外壳'


@pytest.mark.parametrize('classes', [[], ['a', 'a'], [''], [' a ', 'a'], [1], 'abc'])
def test_bad_classes(tmp_path, classes):
    with pytest.raises(ValueError):
        PartsStore(tmp_path).set_classes(classes, 0)


@pytest.mark.parametrize('bad_box', [
    {'class_id': 2, 'points': [[.1,.1],[.7,.1],[.7,.7],[.1,.7]]},
    {'class_id': True, 'points': [[.1,.1],[.7,.1],[.7,.7],[.1,.7]]},
    {'class_id': 0, 'points': [[-.1,.1],[.7,.1],[.7,.7],[.1,.7]]},
    {'class_id': 0, 'points': [[float('nan'),.1],[.7,.1],[.7,.7],[.1,.7]]},
    {'class_id': 0, 'points': [[.1,.1],[.7,.7],[.7,.1],[.1,.7]]},
    {'class_id': 0, 'points': [[.1,.1],[.1,.1],[.1,.1],[.1,.1]]},
    {'class_id': 0, 'points': [[.1,.1],[.7,.1],[.2,.2],[.1,.7]]},
    {'class_id': 0, 'points': [[.1,.1],[.7,.1],[.7,.7]]},
])
def test_invalid_boxes_do_not_change_manifest(tmp_path, bad_box):
    store = store_with_items(tmp_path, 1)
    before = store.snapshot()
    item = before['items'][0]
    with pytest.raises(ValueError):
        store.save_item(item['id'], [bad_box], 'reviewed', item['revision'])
    assert store.snapshot() == before


def test_negative_and_review_are_explicit(tmp_path):
    store = store_with_items(tmp_path, 2)
    item = store.snapshot()['items'][0]
    with pytest.raises(ValueError):
        store.save_item(item['id'], [], 'reviewed', item['revision'])
    with pytest.raises(ValueError):
        store.save_item(item['id'], [box()], 'negative', item['revision'])
    with pytest.raises(ValueError):
        store.save_item(item['id'], [], 'unknown', item['revision'])
    changed = store.save_item(item['id'], [], 'pending', item['revision'])
    assert not store.quality()['ok']
    with pytest.raises(ValueError, match='审核|待审核'):
        store.publish()
    store.save_item(item['id'], [], 'negative', changed['revision'])
    assert store.quality()['ok']
    groups = store.snapshot()['items']
    version = store.publish({groups[0]['source_group']: 'val', groups[1]['source_group']: 'train'})
    assert any(Path(i['label']).read_text() == '' for i in version['items'])


def test_publish_by_source_freezes_content(tmp_path):
    store = store_with_items(tmp_path)
    groups = [item['source_group'] for item in store.snapshot()['items']]
    version = store.publish({groups[0]: 'train', groups[1]: 'val', groups[2]: 'test'})
    assert set(i['split'] for i in version['items']) == {'train', 'val', 'test'}
    assert Path(version['data_yaml']).is_file()
    assert len(version['sha256']) == 64
    assert version['classes'] == ['零件甲', '零件乙']
    before = {i['id']: (Path(i['image']).read_bytes(), Path(i['label']).read_bytes()) for i in version['items']}
    assert store.publish({groups[0]: 'train', groups[1]: 'val', groups[2]: 'test'})['id'] == version['id']
    item = store.snapshot()['items'][0]
    store.save_item(item['id'], [box(1)], 'reviewed', item['revision'])
    frozen = store.version_info(version['id'])
    assert before == {i['id']: (Path(i['image']).read_bytes(), Path(i['label']).read_bytes()) for i in frozen['items']}
    assert store.publish({groups[0]: 'train', groups[1]: 'val', groups[2]: 'test'})['id'] != version['id']
    assert len(store.snapshot()['versions']) == 2


def test_default_split_and_source_wholeness(tmp_path):
    store = store_with_items(tmp_path, 4)
    first = store.snapshot()['items'][0]
    store.import_source(str(image(tmp_path / '更多' / '额外.png', 245)), first['source_group'])
    item = store.snapshot()['items'][-1]
    store.save_item(item['id'], [box()], 'reviewed', item['revision'])
    version = store.publish()
    group_splits = {}
    for item in version['items']:
        group_splits.setdefault(item['source_group'], set()).add(item['split'])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {'train', 'val'} <= {i['split'] for i in version['items']}


def test_single_source_cannot_be_split_or_unknown_mapping(tmp_path):
    store = store_with_items(tmp_path, 1)
    with pytest.raises(ValueError, match='来源|分组'):
        store.publish()
    with pytest.raises(ValueError):
        store.publish({'other': 'val'})


def test_duplicate_boxes_and_broken_media_are_located(tmp_path):
    store = store_with_items(tmp_path, 2)
    item = store.snapshot()['items'][0]
    store.save_item(item['id'], [box(), box()], 'reviewed', item['revision'])
    issues = store.quality()['issues']
    assert any(i['item_id'] == item['id'] and i['code'] == 'duplicate-box' for i in issues)
    with pytest.raises(ValueError):
        store.publish()
    store.image_path(item['id']).write_bytes(b'broken')
    assert any(i['item_id'] == item['id'] and i['code'] == 'image-decode' for i in store.quality()['issues'])


def test_directory_import_is_recursive_and_skips_unsafe_links(tmp_path):
    root = tmp_path / '素材'
    image(root / 'a.png', 0)
    image(root / '子目录' / 'b.png', 100)
    (root / 'readme.txt').write_text('略')
    store = PartsStore(tmp_path / '工程')
    assert len(store.import_source(str(root))['items']) == 2
    assert len({i['source_group'] for i in store.snapshot()['items']}) == 1


def test_video_sampling_and_near_duplicate_filter(tmp_path):
    video = tmp_path / '来源.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,48))
    assert writer.isOpened()
    for n in range(9):
        frame = np.full((48,64,3), n // 3 * 80, np.uint8)
        writer.write(frame)
    writer.release()
    store = PartsStore(tmp_path / '工程')
    snap = store.import_source(str(video), interval=2, max_frames=5, dedupe=False)
    assert [i['frame_index'] for i in snap['items']] == [0,2,4,6,8]
    assert all(i['source'] == str(video.resolve()) for i in snap['items'])
    other = PartsStore(tmp_path / '去重工程')
    assert len(other.import_source(str(video), interval=1, dedupe=True)['items']) < 9


@pytest.mark.parametrize('kw', [{'interval':0}, {'max_frames':0}, {'interval':True}, {'max_frames':100001}, {'dedupe':'yes'}])
def test_import_parameters_are_bounded(tmp_path, kw):
    store = PartsStore(tmp_path / '工程')
    with pytest.raises(ValueError):
        store.import_source(str(image(tmp_path / 'a.png')), **kw)


def test_paths_and_missing_files_are_refused(tmp_path):
    store = PartsStore(tmp_path / '工程')
    for name in ['../x', '..', '/absolute', 'a/b']:
        with pytest.raises((ValueError, FileNotFoundError)):
            store.image_path(name)
        with pytest.raises((ValueError, FileNotFoundError)):
            store.version_info(name)
    with pytest.raises(ValueError):
        store.import_source(str(tmp_path / 'missing.png'))
    invalid = tmp_path / 'bad.png'
    invalid.write_bytes(b'png')
    with pytest.raises(ValueError):
        store.import_source(str(invalid))
    with pytest.raises(ValueError):
        store.import_source(str(tmp_path / '工程'))


def test_manifest_tampered_image_path_never_escapes(tmp_path):
    store = store_with_items(tmp_path, 1)
    path = tmp_path / '工程' / 'parts' / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['items'][0]['image'] = '../../outside.png'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        store.image_path(manifest['items'][0]['id'])


def test_frozen_version_tamper_is_detected(tmp_path):
    store = store_with_items(tmp_path, 2)
    version = store.publish()
    Path(version['items'][0]['label']).write_text('0 0 0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='指纹|损坏|标签'):
        store.version_info(version['id'])


def test_moved_project_resolves_frozen_images_from_new_location(tmp_path):
    import shutil

    store = store_with_items(tmp_path, 2)
    version = store.publish()
    copied = tmp_path / '便携副本'
    shutil.copytree(tmp_path / '工程', copied)
    moved = PartsStore(copied).version_info(version['id'])
    assert all(Path(item['image']).is_relative_to(copied) for item in moved['items'])
    assert moved['sha256'] == version['sha256']


def test_import_failure_rolls_back_partial_batch(tmp_path):
    folder = tmp_path / '批次'
    image(folder / 'a.png')
    (folder / 'b.png').write_bytes(b'broken')
    store = PartsStore(tmp_path / '工程')
    with pytest.raises(ValueError, match='解码'):
        store.import_source(str(folder))
    assert store.snapshot()['items'] == []
    assert list((tmp_path / '工程' / 'parts' / 'images').iterdir()) == []


def test_same_video_cannot_be_reassigned_to_different_groups(tmp_path):
    store = store_with_items(tmp_path, 3)
    path = tmp_path / '工程' / 'parts' / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for item in manifest['items'][:2]:
        item['source'] = '原视频.mp4'
        item['frame_index'] = 10
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert any(issue['code'] == 'source-leakage' for issue in store.quality()['issues'])
    with pytest.raises(ValueError, match='来源'):
        store.publish()


def test_all_negative_training_split_is_rejected(tmp_path):
    store = store_with_items(tmp_path, 2)
    items = store.snapshot()['items']
    store.save_item(items[0]['id'], [], 'negative', items[0]['revision'])
    with pytest.raises(ValueError, match='训练.*正样本'):
        store.publish({items[0]['source_group']: 'train', items[1]['source_group']: 'val'})


def test_workspace_symlink_guard_is_refused(tmp_path, monkeypatch):
    project = tmp_path / '工程'
    outside = tmp_path / '外部'
    project.mkdir()
    outside.mkdir()
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: self == project / 'parts' or original(self))
    with pytest.raises(ValueError, match='越界|符号链接'):
        PartsStore(project)
    assert list(outside.iterdir()) == []


def test_duplicate_images_in_different_groups_do_not_publish(tmp_path):
    store = store_with_items(tmp_path, 2)
    source = tmp_path / '来源0' / '图片.png'
    store.import_source(str(source), '另一个来源', dedupe=False)
    item = store.snapshot()['items'][-1]
    store.save_item(item['id'], [box()], 'reviewed', item['revision'])
    assert any(i['code'] == 'duplicate-image' for i in store.quality()['issues'])


def test_remove_duplicate_and_broken_image_repairs_quality(tmp_path):
    store = store_with_items(tmp_path, 3)
    store.import_source(str(tmp_path / '来源0' / '图片.png'), '重复来源', dedupe=False)
    duplicate = store.snapshot()['items'][-1]
    duplicate = store.save_item(duplicate['id'], [box()], 'reviewed', duplicate['revision'])
    broken = store.snapshot()['items'][2]
    store.image_path(broken['id']).write_bytes(b'broken')
    assert not store.quality()['ok']
    duplicate_path = store.image_path(duplicate['id'])
    broken_path = store.image_path(broken['id'])
    store.delete_item(duplicate['id'], duplicate['revision'])
    result = store.delete_item(broken['id'], broken['revision'])
    assert len(result['items']) == 2
    assert not duplicate_path.exists() and not broken_path.exists()
    assert store.quality()['ok']
    assert len(store.publish()['items']) == 2


def test_delete_workspace_item_keeps_published_version(tmp_path):
    store = store_with_items(tmp_path, 2)
    version = store.publish()
    item = store.snapshot()['items'][0]
    original = store.image_path(item['id'])
    frozen = {i['id']: Path(i['image']).read_bytes() for i in version['items']}
    result = store.delete_item(item['id'], item['revision'])
    assert not original.exists()
    assert len(result['items']) == 1
    assert result['versions'][0]['id'] == version['id']
    assert frozen == {i['id']: Path(i['image']).read_bytes() for i in store.version_info(version['id'])['items']}


def test_delete_requires_current_revision_and_bounded_id(tmp_path):
    store = store_with_items(tmp_path, 1)
    before = store.snapshot()
    item = before['items'][0]
    for revision in (None, True, -1, item['revision'] + 1):
        with pytest.raises(ValueError):
            store.delete_item(item['id'], revision)
    for bad in ('../outside', '..', 'images/x', '/outside'):
        with pytest.raises(ValueError):
            store.delete_item(bad, item['revision'])
    with pytest.raises(FileNotFoundError):
        store.delete_item('f' * 20, 0)
    assert store.snapshot() == before
    assert store.image_path(item['id']).is_file()


def test_delete_rolls_back_media_when_manifest_write_fails(tmp_path, monkeypatch):
    from lstm_workbench.project_store import ProjectStore

    store = store_with_items(tmp_path, 1)
    before = store.snapshot()
    item = before['items'][0]
    path = store.image_path(item['id'])
    original = path.read_bytes()
    def fail(*args):
        raise OSError('模拟磁盘写入失败')
    monkeypatch.setattr(ProjectStore, 'write_json', fail)
    with pytest.raises(OSError, match='写入失败'):
        store.delete_item(item['id'], item['revision'])
    assert store.snapshot() == before
    assert path.read_bytes() == original
    assert list((tmp_path / '工程' / 'parts' / '.trash').iterdir()) == []


def test_delete_missing_image_and_refuse_version_path(tmp_path):
    store = store_with_items(tmp_path, 2)
    version = store.publish()
    item = store.snapshot()['items'][0]
    store.image_path(item['id']).unlink()
    assert len(store.delete_item(item['id'], item['revision'])['items']) == 1
    manifest_path = tmp_path / '工程' / 'parts' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    remaining = manifest['items'][0]
    frozen = version['items'][0]
    remaining['image'] = Path(frozen['image']).relative_to(tmp_path / '工程' / 'parts').as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='工作区|路径'):
        store.delete_item(remaining['id'], remaining['revision'])
    assert Path(frozen['image']).is_file()


@pytest.mark.parametrize('ratios', [(0,.1), (.9,.9), (float('nan'),.1), (-.1,.1)])
def test_split_ratios_are_validated(tmp_path, ratios):
    store = store_with_items(tmp_path, 2)
    with pytest.raises(ValueError):
        store.publish(val_ratio=ratios[0], test_ratio=ratios[1])
