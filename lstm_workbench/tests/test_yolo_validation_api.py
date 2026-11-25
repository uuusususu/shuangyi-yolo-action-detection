"""验证与预标注通过后台任务运行，锁及停止仍由工作台统一管理。"""
import threading
import time

from lstm_workbench import app
from lstm_workbench.tests.test_app import running


def test_dataset_import_excludes_train_resume_validation_and_other_import(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def importing(folder, source):
        entered.set()
        assert release.wait(10)
        return {'exists': True}
    monkeypatch.setattr(app.yolo_jobs, 'import_dataset', importing)
    monkeypatch.setattr(app.yolo_jobs, 'start_training', lambda *a: {'name': 'unexpected'})
    with running(tmp_path / 'projects') as call:
        for project in ('demo', 'other'):
            call('POST', '/api/projects', {'id': project})
        result = []
        thread = threading.Thread(target=lambda: result.append(call('POST', '/api/projects/demo/yolo/dataset', {'source': 'x'})))
        thread.start()
        try:
            assert entered.wait(5)
            assert call('POST', '/api/projects/demo/yolo/train', {})[0] == 409
            assert call('POST', '/api/projects/other/yolo/train', {})[0] == 409
            assert call('POST', '/api/projects/other/train', {})[0] == 409
            assert call('POST', '/api/projects/other/yolo/validate', {})[0] == 409
            assert call('POST', '/api/projects/other/yolo/runs/old/resume', {})[0] == 409
            assert call('POST', '/api/projects/other/yolo/dataset', {'source': 'y'})[0] == 409
        finally:
            release.set()
            thread.join(10)
        assert result[0][0] == 200
        assert call('POST', '/api/projects/other/yolo/train', {})[0] == 201


def test_dataset_import_failure_releases_global_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(app.yolo_jobs, 'import_dataset', lambda *args: (_ for _ in ()).throw(OSError('import failed')))
    monkeypatch.setattr(app.yolo_jobs, 'start_training', lambda *args: {'name': 'next'})
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        assert call('POST', '/api/projects/demo/yolo/dataset', {'source': 'x'})[0] == 400
        assert call('POST', '/api/projects/demo/yolo/train', {})[0] == 201


def test_validation_async_progress_stop_and_global_exclusion(tmp_path, monkeypatch):
    from lstm_workbench import yolo_validation
    entered, release = threading.Event(), threading.Event()
    records = {}

    def prepare(folder, params):
        identifier = f"evaluation-{len(records)}"
        records[identifier] = {'id': identifier, 'status': 'running', 'mode': params.get('mode', 'validate')}
        return dict(records[identifier])

    def execute(folder, identifier, stop_event=None):
        entered.set()
        assert release.wait(5)
        records[identifier]['status'] = 'stopped' if stop_event.is_set() else 'done'
        return dict(records[identifier])

    monkeypatch.setattr(yolo_validation, 'prepare_validation', prepare)
    monkeypatch.setattr(yolo_validation, 'execute_validation', execute)
    monkeypatch.setattr(yolo_validation, 'list_validations', lambda folder: list(records.values()))
    monkeypatch.setattr(yolo_validation, 'validation_detail', lambda folder, identifier: records[identifier])
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        call('POST', '/api/projects', {'id': 'other'})
        status, record = call('POST', '/api/projects/demo/yolo/validate', {})
        assert status == 202, record
        assert record['status'] == 'running' and entered.wait(2)
        assert call('GET', '/api/projects/demo/yolo/evaluations')[1]['evaluations'][0]['id'] == record['id']
        assert call('GET', '/api/projects/demo/yolo/evaluations/' + record['id'])[1]['status'] == 'running'
        assert call('POST', '/api/projects/demo/yolo/preannotate', {})[0] == 409
        assert call('POST', '/api/projects/other/yolo/validate', {})[0] == 409
        assert call('POST', '/api/projects/demo/stop', {})[0] == 200
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = call('GET', '/api/projects/demo/yolo/evaluations/' + record['id'])[1]
            if result['status'] == 'stopped':
                break
            time.sleep(.02)
        assert result['status'] == 'stopped'
        time.sleep(.05)
        assert call('POST', '/api/projects/other/yolo/preannotate', {})[0] == 202


def test_validation_failure_releases_job_lock(tmp_path, monkeypatch):
    from lstm_workbench import yolo_validation
    monkeypatch.setattr(yolo_validation, 'prepare_validation', lambda folder, params: (_ for _ in ()).throw(ValueError('参数有误')))
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        assert call('POST', '/api/projects/demo/yolo/validate', {})[0] == 400
        assert call('POST', '/api/projects/demo/yolo/preannotate', {})[0] == 400
        assert call('GET', '/api/projects/demo/yolo/evaluations/%2e%2e')[0] == 400


def test_unexpected_background_failure_is_visible_and_unlocks(tmp_path, monkeypatch):
    from lstm_workbench import yolo_validation
    from lstm_workbench.project_store import ProjectStore

    def prepare(folder, params):
        record = {'id': 'broken-task', 'status': 'running', 'mode': params['mode']}
        target = folder / 'parts' / 'evaluations' / record['id'] / 'result.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        ProjectStore.write_json(target, record)
        return record

    def broken(*args, **kwargs):
        raise RuntimeError('意外执行失败')

    monkeypatch.setattr(yolo_validation, 'prepare_validation', prepare)
    monkeypatch.setattr(yolo_validation, 'execute_validation', broken)
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        status, record = call('POST', '/api/projects/demo/yolo/validate', {})
        assert status == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result_status, result = call('GET', '/api/projects/demo/yolo/evaluations/' + record['id'])
            assert result_status == 200, result
            if result['status'] != 'running':
                break
            time.sleep(.02)
        assert result['status'] == 'failed'
        assert '意外执行失败' in result['error']
        assert call('POST', '/api/projects/demo/yolo/preannotate', {})[0] == 202


def test_resume_http_rejects_bad_run_and_releases_gpu_after_error(tmp_path, monkeypatch):
    from lstm_workbench import yolo_jobs
    calls = []

    def resume(folder, name, params):
        calls.append((folder.name, name, params))
        if params:
            raise ValueError('续训不能更换原参数')
        return {'name': name, 'status': 'training', 'resume_count': 1}

    monkeypatch.setattr(yolo_jobs, 'resume_training', resume)
    with running(tmp_path / 'projects') as call:
        call('POST', '/api/projects', {'id': 'demo'})
        assert call('POST', '/api/projects/demo/yolo/runs/%2e%2e/resume', {})[0] == 400
        assert call('POST', '/api/projects/demo/yolo/runs/old/resume', {'model': 'other.pt'})[0] == 400
        status, result = call('POST', '/api/projects/demo/yolo/runs/old/resume', {})
        assert status == 201 and result['run']['resume_count'] == 1
        assert calls == [('demo', 'old', {'model': 'other.pt'}), ('demo', 'old', {})]
