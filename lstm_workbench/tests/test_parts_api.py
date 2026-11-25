"""零件标注的本机 HTTP 闭环与请求边界。"""
from pathlib import Path

import cv2
import numpy as np

from lstm_workbench.tests.test_app import running


def _image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', np.full((32, 48, 3), value, dtype=np.uint8))
    assert ok
    encoded.tofile(path)
    return path


def test_parts_http_import_review_quality_and_publish(tmp_path):
    with running(tmp_path / 'projects') as call:
        assert call('POST', '/api/projects', {'id': 'demo'})[0] == 201
        status, snapshot = call('GET', '/api/projects/demo/parts')
        assert status == 200, snapshot
        status, snapshot = call('POST', '/api/projects/demo/parts/classes', {
            'classes': ['零件'], 'revision': snapshot['revision'],
        })
        assert status == 200, snapshot
        for index in range(2):
            source = _image(tmp_path / f'batch{index}' / 'sample.png', index * 90)
            status, snapshot = call('POST', '/api/projects/demo/parts/import', {
                'source': str(source), 'source_group': f'batch{index}',
            })
            assert status == 200, snapshot
        assert len(snapshot['items']) == 2
        assert call('POST', '/api/projects/demo/parts/publish', {})[0] == 400
        for item in snapshot['items']:
            status, saved = call('POST', f"/api/projects/demo/parts/items/{item['id']}", {
                'boxes': [{'class_id': 0, 'points': [[.1, .1], [.7, .1], [.7, .7], [.1, .7]]}],
                'status': 'reviewed', 'revision': item['revision'],
            })
            assert status == 200, saved
            assert saved['item']['status'] == 'reviewed'
            assert saved['revision'] > snapshot['revision']
            assert call('POST', f"/api/projects/demo/parts/items/{item['id']}", {
                'boxes': [], 'status': 'negative', 'revision': item['revision'],
            })[0] == 409
        status, quality = call('GET', '/api/projects/demo/parts/quality')
        assert status == 200 and quality['ok'], quality
        status, version = call('POST', '/api/projects/demo/parts/publish', {
            'splits': {'batch0': 'train', 'batch1': 'val'},
        })
        assert status == 200, version
        assert version['id']
        assert Path(version['data_yaml']).is_file()
        assert call('GET', '/api/projects/demo/parts')[1]['versions']
        first = snapshot['items'][0]
        assert call('POST', f"/api/projects/demo/parts/items/{first['id']}/delete", {'revision': first['revision']})[0] == 409
        status, remaining = call('POST', f"/api/projects/demo/parts/items/{first['id']}/delete", {'revision': first['revision'] + 1})
        assert status == 200 and len(remaining['items']) == 1, remaining
        assert len(version['items']) == 2
        assert all(Path(item['image']).is_file() for item in version['items']), '删除工作区素材不能破坏已发布快照'


def test_parts_http_rejects_cross_origin_and_bad_inputs(tmp_path):
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        assert call('POST', '/api/projects/demo/parts/classes', {'classes': ['零件'], 'revision': 0},
                    headers={'Origin': 'https://foreign.example'})[0] == 403
        for payload in ({'classes': '零件'}, {'classes': ['零件']}, {'classes': ['零件'], 'revision': []}):
            assert call('POST', '/api/projects/demo/parts/classes', payload)[0] == 400
        assert call('POST', '/api/projects/demo/parts/import', {'source': str(tmp_path / 'missing.png')})[0] in (400, 404)
        assert call('GET', '/api/projects/demo/parts/items/%2e%2e/image')[0] == 400
        assert call('GET', '/api/projects/missing/parts')[0] == 404
