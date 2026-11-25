"""单机后台特征准备：逐视频持久化请求，与前台任务共用显卡。"""
from __future__ import annotations

import json
import threading
from . import feature_store, settings as settings_module


class PreparationManager:
    def __init__(self, workbench):
        self.wb = workbench
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[str, dict] = {}
        self._active: dict | None = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def _snapshot(self, project_id):
        with self.wb.data_lock(project_id):
            folder = self.wb.store.open_dir(project_id)
            annotations = self.wb.store.load_annotations(project_id)
            settings = settings_module.clean({}, self.wb.store.load_settings(project_id))
        return folder, annotations, settings

    def _state(self, project_id):
        project_id = self.wb.store.open_dir(project_id).name
        if project_id not in self._states:
            path = self.wb.store.open_dir(project_id) / 'preparation.json'
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                saved = {}
            if not isinstance(saved, dict):
                saved = {}
            pending = saved.get('pending')
            self._states[project_id] = {
                'pending': list(dict.fromkeys(x for x in pending if isinstance(x, str))) if isinstance(pending, list) else [],
                'paused': bool(saved.get('paused')),
                'errors': saved.get('errors', {}) if isinstance(saved.get('errors'), dict) else {},
            }
        return project_id, self._states[project_id]

    def _save(self, project_id, state):
        from .service import _replace_retry
        path = self.wb.store.open_dir(project_id) / 'preparation.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        _replace_retry(temporary, path)

    def start(self):
        with self._condition:
            if self._thread is not None:
                return
            for project_id in self.wb.store.list_projects():
                if (self.wb.store.open_dir(project_id) / 'preparation.json').is_file():
                    self._state(project_id)
            self._thread = threading.Thread(target=self._run, name='特征准备', daemon=True)
            self._thread.start()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def request(self, project_id, clip_ids=None, resume=False):
        folder, annotations, settings = self._snapshot(project_id)
        project_id = folder.name
        known = {str(c['id']) for c in annotations.get('clips', [])}
        if clip_ids is not None and (not isinstance(clip_ids, list) or any(not isinstance(x, str) or x not in known for x in clip_ids)):
            raise ValueError('准备列表中有不存在的视频')
        selected = known - set(settings.get('train_excluded') or [])
        wanted = selected if clip_ids is None else selected.intersection(clip_ids)
        summary = feature_store.summary(folder, annotations, settings)
        fresh = {r['clip_id'] for r in summary['clips'] if r['fresh']}
        with self._condition:
            _, state = self._state(project_id)
            state['pending'] = [x for x in state['pending'] if x in selected and x not in fresh]
            if not summary['blocked_reason']:
                for clip_id in sorted(wanted - fresh):
                    if clip_id not in state['pending']:
                        state['pending'].append(clip_id)
                    state['errors'].pop(clip_id, None)
            state['errors'] = {k: v for k, v in state['errors'].items() if k in state['pending']}
            if resume:
                state['paused'] = False
            self._save(project_id, state)
            self._condition.notify_all()
        return self.status(project_id)

    def pause(self, project_id):
        with self._condition:
            project_id, state = self._state(project_id)
            state['paused'] = True
            self._save(project_id, state)
            self._condition.notify_all()
        return self.status(project_id)

    def status(self, project_id):
        folder, annotations, settings = self._snapshot(project_id)
        summary = feature_store.summary(folder, annotations, settings)
        excluded = set(settings.get('train_excluded') or [])
        with self._condition:
            project_id, state = self._state(folder.name)
            pending, errors, paused = set(state['pending']), dict(state['errors']), state['paused']
            active = dict(self._active) if self._active and self._active['project'] == project_id else None
        rows = []
        for row in summary['clips']:
            clip_id = row['clip_id']
            if clip_id in excluded:
                kind = 'excluded'
            elif row['fresh']:
                kind = 'ready'
            elif paused and clip_id in pending:
                kind = 'paused'
            elif active and active['clip_id'] == clip_id:
                kind = 'preparing'
            elif clip_id in errors:
                kind = 'failed'
            elif clip_id in pending:
                kind = 'queued'
            else:
                kind = 'pending'
            rows.append({'clip_id': clip_id, 'state': kind, 'error': errors.get(clip_id) or row.get('note') or ''})
        chosen = [r for r in rows if r['state'] != 'excluded']
        return {
            'clips': rows, 'total': len(chosen),
            'ready': sum(r['state'] == 'ready' for r in chosen),
            'queued': sum(r['state'] in {'queued', 'paused'} for r in chosen),
            'failed': sum(r['state'] == 'failed' for r in chosen),
            'paused': paused, 'active': active, 'blocked_reason': summary['blocked_reason'],
        }

    def _cancelled(self, project_id, clip_id):
        with self._condition:
            state = self._states[project_id]
            return self._closed or state['paused'] or clip_id not in state['pending']

    def _tick(self, project_id, clip_id, done, total):
        with self._condition:
            if self._active and self._active['project'] == project_id and self._active['clip_id'] == clip_id:
                self._active.update(done=done, total=total)

    def _run(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                item = next(((pid, clip_id) for pid, state in self._states.items()
                             if not state['paused'] for clip_id in state['pending']
                             if clip_id not in state['errors']), None)
                if item is None:
                    self._condition.wait(.25)
                    continue
            project_id, clip_id = item
            # 这里抛错会直接结束后台线程，之后所有工程的片子都永远停在"排队"；记下来稍后重试。
            try:
                acquired = self.wb.begin_preparation(project_id)
            except Exception:
                import traceback
                traceback.print_exc()
                acquired = False
            if not acquired:
                with self._condition:
                    self._condition.wait(.2)
                continue
            try:
                folder, annotations, settings = self._snapshot(project_id)
                if clip_id in set(settings.get('train_excluded') or []) or not any(c['id'] == clip_id for c in annotations.get('clips', [])):
                    with self._condition:
                        state = self._states[project_id]
                        state['pending'] = [x for x in state['pending'] if x != clip_id]
                        self._save(project_id, state)
                    continue
                with self._condition:
                    self._active = {'project': project_id, 'clip_id': clip_id, 'done': 0, 'total': 0}
                feature_store.extract_project(
                    folder, annotations, settings, clip_ids=[clip_id],
                    should_stop=lambda: self._cancelled(project_id, clip_id),
                    on_clip_frame=lambda cid, done, total: self._tick(project_id, cid, done, total),
                )
                latest_folder, latest_annotations, latest_settings = self._snapshot(project_id)
                latest = feature_store.summary(latest_folder, latest_annotations, latest_settings)
                ready = any(row['clip_id'] == clip_id and row['fresh'] for row in latest['clips'])
                with self._condition:
                    state = self._states[project_id]
                    if ready:
                        state['pending'] = [x for x in state['pending'] if x != clip_id]
                    state['errors'].pop(clip_id, None)
                    self._save(project_id, state)
            except feature_store.Stopped:
                pass
            except Exception as exc:
                with self._condition:
                    state = self._states[project_id]
                    state['errors'][clip_id] = str(exc)
                    try:
                        self._save(project_id, state)
                    except OSError:
                        pass
            finally:
                with self._condition:
                    self._active = None
                self.wb.end_preparation(project_id)
