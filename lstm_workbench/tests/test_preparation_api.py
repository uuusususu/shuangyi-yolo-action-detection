"""保存标注到后台准备、训练接续的 HTTP 集成验证。"""
import threading

from lstm_workbench import app, feature_store
from lstm_workbench.tests.test_app import running
from lstm_workbench.tests.test_preparation import setup_project, wait_for


def test_save_prepares_current_clip_and_repeated_training_reuses_it(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    annotations = wb.store.load_annotations('demo')
    monkeypatch.setattr(app, 'start_training', lambda store, project, arrays, **kw: {'frames': sum(len(x) for x in arrays.values())})
    with running(tmp_path) as call:
        status, saved = call('POST', '/api/projects/demo/annotations?clip_id=a', annotations)
        assert status == 200 and saved['saved']
        wait_for(lambda: call('GET', '/api/projects/demo/preparation')[1]['ready'] == 1)
        assert len(calls) == 2
        status, result = call('POST', '/api/projects/demo/train', {'epochs': 2})
        assert status == 200 and result['frames'] == 4
        assert len(calls) == 4
        status, result = call('POST', '/api/projects/demo/train', {'epochs': 7})
        assert status == 200 and len(calls) == 4
        assert call('GET', '/api/projects/demo/progress')[1]['status'] == 'loading-features'
        status, _ = call('POST', '/api/projects/demo/annotations?clip_id=missing', annotations)
        assert status == 400
        assert call('POST', '/api/projects/demo/preparation', {'clips': ['../outside']})[0] == 400
        assert call('POST', '/api/projects/demo/preparation', {'action': 'unknown'})[0] == 400


def test_train_request_joins_inflight_background_preparation(tmp_path, monkeypatch):
    wb, calls = setup_project(tmp_path, monkeypatch)
    original = feature_store.extract_project
    entered, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(feature_store, 'extract_project', slow)
    monkeypatch.setattr(app, 'start_training', lambda *a, **kw: {'done': True})
    with running(tmp_path) as call:
        annotations = wb.store.load_annotations('demo')
        assert call('POST', '/api/projects/demo/annotations?clip_id=a', annotations)[0] == 200
        assert entered.wait(5)
        result = []
        thread = threading.Thread(target=lambda: result.append(call('POST', '/api/projects/demo/train', {})))
        thread.start()
        try:
            wait_for(lambda: call('GET', '/api/projects/demo/progress')[1]['status'] == 'waiting-features')
            assert thread.is_alive()
        finally:
            release.set()
        thread.join(5)
        assert result and result[0][0] == 200
        assert len(calls) == 4
