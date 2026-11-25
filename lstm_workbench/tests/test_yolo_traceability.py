"""零件训练必须能追到冻结数据，续训不能悄悄换数据或模型。"""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


from lstm_workbench import yolo_jobs
from lstm_workbench.tests.test_yolo_jobs import _dataset, fake_env


def test_child_environment_disables_implicit_dependency_install(tmp_path):
    assert yolo_jobs._child_env(tmp_path)['YOLO_AUTOINSTALL'] == 'false'


@pytest.mark.parametrize('extra', ['model=x.pt', 'task=detect', 'resume=True', 'mode=val', 'callbacks=x', 'exist_ok=False', 'DATA=x'])
def test_managed_arguments_cannot_be_overridden(tmp_path, extra):
    with pytest.raises(ValueError):
        yolo_jobs.build_command('py', tmp_path/'d.yaml', tmp_path, 'r', {'extra': extra})


@pytest.mark.parametrize('params', [{'epochs': 0}, {'epochs': -1}, {'imgsz': 0}, {'batch': 0}, {'workers': -1}])
def test_invalid_training_numbers_are_rejected(tmp_path, params):
    with pytest.raises(ValueError):
        yolo_jobs.build_command('py', tmp_path/'d.yaml', tmp_path, 'r', params)


def test_training_freezes_legacy_data_and_records_provenance(tmp_path, fake_env):
    folder = tmp_path/'p'
    folder.mkdir()
    yolo_jobs.import_dataset(folder, str(_dataset(tmp_path/'src')))
    job = yolo_jobs.start_training(folder, {'epochs': 1, 'device': 'cpu'})
    run = yolo_jobs.runs_dir(folder)/job['name']
    frozen = json.loads((run/'dataset.json').read_text(encoding='utf-8'))
    assert frozen['sha256'] and frozen['classes'] == ['导热垫', '散热片']
    assert frozen['kind'] == 'legacy'
    assert Path(frozen['data_yaml']).parent == run/'dataset'
    assert job['environment']['ultralytics']
    assert job['model']['identifier']
    assert job['dataset']['sha256'] == frozen['sha256']
    before = (run/'dataset'/'images'/'train'/'0.jpg').read_bytes()
    (folder/'yolo'/'images'/'train'/'0.jpg').write_bytes(b'changed')
    assert (run/'dataset'/'images'/'train'/'0.jpg').read_bytes() == before
    yolo_jobs._PROCESSES[yolo_jobs._process_key(run)].wait(timeout=10)


def test_native_dataset_requires_explicit_version(tmp_path, monkeypatch):
    from lstm_workbench.parts_store import PartsStore
    monkeypatch.setattr(PartsStore, 'snapshot', lambda self: {'versions': [{'id':'one'}]})
    with pytest.raises(ValueError, match='版本'):
        yolo_jobs.start_training(tmp_path, {})


def test_resume_uses_own_last_checkpoint_and_verifies_frozen_data(tmp_path, monkeypatch):
    folder = tmp_path/'p'
    run = yolo_jobs.runs_dir(folder)/'one'
    (run/'weights').mkdir(parents=True)
    last = run/'weights'/'last.pt'
    last.write_bytes(b'checkpoint')
    data = run/'dataset'
    data.mkdir()
    (data/'data.yaml').write_text('names: [a]\n', encoding='utf-8')
    metadata = yolo_jobs._snapshot_metadata(data, kind='legacy')
    yolo_jobs._write_json(run/'dataset.json', metadata)
    yolo_jobs._write_job(run, {'name':'one','status':'stopped','params':{'epochs':5},'dataset':metadata})
    monkeypatch.setattr(yolo_jobs, '_python', lambda:'py')
    monkeypatch.setattr(yolo_jobs, '_child_env', lambda folder:{})
    class Process:
        pid = 12345
        def poll(self): return None
    seen = []
    monkeypatch.setattr(yolo_jobs.subprocess, 'Popen', lambda command, **kw: seen.append(command) or Process())
    resumed = yolo_jobs.resume_training(folder, 'one')
    assert resumed['status'] == 'training'
    assert resumed['resume_count'] == 1
    assert str(last) in seen[0]
    assert 'resume=True' in seen[0][2]
    assert yolo_jobs.run_summary(run)['has_last']
    yolo_jobs._write_job(run, {**resumed,'status':'stopped'})
    (data/'data.yaml').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='数据'):
        yolo_jobs.resume_training(folder, 'one')
    with pytest.raises(ValueError):
        yolo_jobs.resume_training(folder, '../other')


def test_process_keys_do_not_collide_between_projects(tmp_path):
    assert yolo_jobs._process_key(tmp_path/'p'/'same') != yolo_jobs._process_key(tmp_path/'q'/'same')


def test_published_training_uses_fixed_version_and_absolute_yaml(tmp_path,fake_env):
    from lstm_workbench.tests.test_parts_store import store_with_items
    store=store_with_items(tmp_path)
    version=store.publish()
    folder=store.root.parent
    job=yolo_jobs.start_training(folder,{'dataset_version':version['id'],'epochs':1,'device':'cpu'})
    run=yolo_jobs.runs_dir(folder)/job['name']
    assert job['dataset']['version']==version['id']
    assert job['dataset']['content_sha256']==version['sha256']
    assert Path(version['root']).as_posix() in Path(job['dataset']['data_yaml']).read_text(encoding='utf-8')
    yolo_jobs._verify_snapshot(job['dataset'])
    yolo_jobs._PROCESSES[yolo_jobs._process_key(run)].wait(timeout=10)
    first=store.snapshot()['items'][0]
    store.save_item(first['id'],[],'negative',first['revision'])
    yolo_jobs._verify_snapshot(job['dataset'])
    Path(job['dataset']['data_yaml']).write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='配置'):
        yolo_jobs._verify_snapshot(job['dataset'])


def test_same_second_training_creates_distinct_runs(tmp_path,fake_env,monkeypatch):
    folder=tmp_path/'p'
    folder.mkdir()
    yolo_jobs.import_dataset(folder,str(_dataset(tmp_path/'src')))
    monkeypatch.setattr(yolo_jobs,'_run_name',lambda:'fixed-second')
    first=yolo_jobs.start_training(folder,{'epochs':1,'device':'cpu'})
    second=yolo_jobs.start_training(folder,{'epochs':1,'device':'cpu'})
    assert first['name']!=second['name']
    for job in (first,second):
        yolo_jobs._PROCESSES[yolo_jobs._process_key(yolo_jobs.runs_dir(folder)/job['name'])].wait(timeout=10)


def test_resume_refuses_missing_last_completed_runs_and_parameter_changes(tmp_path):
    folder=tmp_path/'p'
    run=yolo_jobs.runs_dir(folder)/'one'
    (run/'weights').mkdir(parents=True)
    yolo_jobs._write_job(run,{'name':'one','status':'done'})
    with pytest.raises(ValueError,match='只能续训'):
        yolo_jobs.resume_training(folder,'one')
    yolo_jobs._write_job(run,{'name':'one','status':'stopped'})
    with pytest.raises(ValueError,match='last.pt'):
        yolo_jobs.resume_training(folder,'one')
    (run/'weights'/'last.pt').write_bytes(b'last')
    with pytest.raises(ValueError,match='原参数'):
        yolo_jobs.resume_training(folder,'one',{'epochs':999})
    with pytest.raises(ValueError,match='冻结数据'):
        yolo_jobs.resume_training(folder,'one')


def test_failed_job_registration_terminates_the_untracked_process(tmp_path,fake_env,monkeypatch):
    folder=tmp_path/'p'
    folder.mkdir()
    yolo_jobs.import_dataset(folder,str(_dataset(tmp_path/'src')))
    processes=[]
    real_popen=yolo_jobs.subprocess.Popen
    def remember(*args,**kwargs):
        process=real_popen(*args,**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(yolo_jobs.subprocess,'Popen',remember)
    monkeypatch.setattr(yolo_jobs,'_write_job',lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError,match='disk full'):
        yolo_jobs.start_training(folder,{'epochs':100,'extra':'sleep=1','device':'cpu'})
    assert processes[-1].poll() is not None


@pytest.mark.parametrize('broken',['stripped','no_optimizer','wrong_data','wrong_output','complete','wrong_classes',None])
def test_resume_checkpoint_guard_never_downgrades_to_new_training(tmp_path,monkeypatch,broken):
    data,run=tmp_path/'dataset.yaml',tmp_path/'run'
    checkpoint={'epoch':1,'optimizer':{'state':{}},'train_args':{'epochs':5,'data':str(data),'save_dir':str(run)}}
    names={0:'a'}
    if broken=='stripped': checkpoint['epoch']=-1
    if broken=='no_optimizer': checkpoint['optimizer']=None
    if broken=='wrong_data': checkpoint['train_args']['data']=str(tmp_path/'other.yaml')
    if broken=='wrong_output': checkpoint['train_args']['save_dir']=str(tmp_path/'other-run')
    if broken=='complete': checkpoint['epoch']=4
    if broken=='wrong_classes': names={0:'wrong'}
    called=[]
    model=SimpleNamespace(ckpt=checkpoint,names=names,train=lambda **kwargs:called.append(kwargs))
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda path:model))
    monkeypatch.setattr(sys,'argv',['resume',str(run/'weights'/'last.pt'),str(data),str(run),json.dumps(['a'])])
    if broken is not None:
        with pytest.raises(ValueError):
            exec(yolo_jobs.RESUME_SCRIPT,{})
        assert called==[]
    else:
        exec(yolo_jobs.RESUME_SCRIPT,{})
        assert called==[{'resume':True}]


def _old_import_project(tmp_path):
    folder=tmp_path/'p'
    folder.mkdir()
    source=_dataset(tmp_path/'src')
    yolo_jobs.import_dataset(folder,str(source))
    run=folder/'yolo'/'runs'/'old'/'weights'
    run.mkdir(parents=True)
    (run/'best.pt').write_bytes(b'valuable-old-weights')
    return folder,source


@pytest.mark.parametrize('failure',['commit','move-runs','rollback'])
def test_import_failure_preserves_old_weights(tmp_path,monkeypatch,failure):
    folder,source=_old_import_project(tmp_path)
    replace=yolo_jobs.os.replace
    def fail(src,dst):
        src,dst=Path(src),Path(dst)
        commit=dst.name=='yolo' and not src.name.startswith('.yolo-backup-')
        move_runs=src.name=='runs' and dst.parent.name.startswith('.yolo-import-')
        restore=src.name.startswith('.yolo-backup-') and dst.name=='yolo'
        if (commit and failure in {'commit','rollback'}) or (move_runs and failure=='move-runs') or (restore and failure=='rollback'):
            raise OSError('injected '+failure)
        return replace(src,dst)
    monkeypatch.setattr(yolo_jobs.os,'replace',fail)
    with pytest.raises(OSError):
        yolo_jobs.import_dataset(folder,str(source))
    candidates=list(folder.rglob('best.pt'))
    assert candidates and any(p.read_bytes()==b'valuable-old-weights' for p in candidates)
    if failure!='rollback':
        assert (folder/'yolo'/'runs'/'old'/'weights'/'best.pt').read_bytes()==b'valuable-old-weights'
    else:
        assert list(folder.glob('.yolo-backup-*')), '恢复失败必须保留可找回的备份'


def test_import_does_not_remove_another_staging_directory(tmp_path):
    folder,source=_old_import_project(tmp_path)
    other=folder/'yolo.importing';other.mkdir();(other/'keep').write_bytes(b'other-job')
    yolo_jobs.import_dataset(folder,str(source))
    assert (other/'keep').read_bytes()==b'other-job'
    assert (folder/'yolo'/'runs'/'old'/'weights'/'best.pt').read_bytes()==b'valuable-old-weights'


def test_restarted_partial_training_is_not_done_and_can_resume(tmp_path,monkeypatch):
    folder=tmp_path/'p';run=folder/'yolo'/'runs'/'one';(run/'weights').mkdir(parents=True)
    for name in ('best.pt','last.pt'):(run/'weights'/name).write_bytes(b'checkpoint')
    yolo_jobs._write_job(run,{'name':'one','status':'training','pid':99999999,'params':{'epochs':100},'dataset':{'sha256':'frozen'}})
    (run/'results.csv').write_text('epoch,metrics/mAP50(B)\n1,0.1\n')
    monkeypatch.setattr(yolo_jobs,'_pid_alive',lambda pid,created=None:False)
    summary=yolo_jobs.run_summary(run)
    assert summary['status']=='failed'
    assert summary['resumable'] is True
    assert '中断' in summary['error']


def test_reused_pid_does_not_keep_dead_training_alive(tmp_path,monkeypatch):
    """重启后旧 PID 被别的程序占用：创建时刻对不上，旧训练按中断处理，也不去杀那个进程。"""
    import os
    folder=tmp_path/'p';run=folder/'yolo'/'runs'/'one';(run/'weights').mkdir(parents=True)
    me=os.getpid()
    yolo_jobs._write_job(run,{'name':'one','status':'training','pid':me,
                              'pid_created':yolo_jobs._pid_created(me)-3600,'params':{'epochs':100}})
    killed=[]
    monkeypatch.setattr(yolo_jobs.subprocess,'run',lambda *a,**k:killed.append(a))
    assert yolo_jobs._job_process_alive(yolo_jobs._read_job(run)) is False
    yolo_jobs.stop_training(run)
    assert killed==[]
    assert yolo_jobs.refresh_status(run)['status'] in ('stopped','failed')


def test_recorded_creation_time_keeps_live_training(tmp_path):
    import os
    me=os.getpid()
    assert yolo_jobs._pid_alive(me,yolo_jobs._pid_created(me)) is True


def test_import_refuses_escaped_target_before_touching_old_data(tmp_path,monkeypatch):
    folder,source=_old_import_project(tmp_path)
    target=folder/'yolo';outside=tmp_path/'outside';outside.mkdir()
    original=Path.resolve
    def resolved(path,*args,**kwargs):
        return outside if path==target else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',resolved)
    with pytest.raises(ValueError,match='越界'):
        yolo_jobs.import_dataset(folder,str(source))
    assert (target/'runs'/'old'/'weights'/'best.pt').read_bytes()==b'valuable-old-weights'


def test_import_refuses_reparse_point_in_old_directory(tmp_path,monkeypatch):
    folder,source=_old_import_project(tmp_path)
    target=folder/'yolo';original=Path.lstat
    def lstat(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        return SimpleNamespace(st_mode=value.st_mode,st_file_attributes=0x400) if path==target else value
    monkeypatch.setattr(Path,'lstat',lstat)
    with pytest.raises(ValueError,match='Junction'):
        yolo_jobs.import_dataset(folder,str(source))
    assert (target/'runs'/'old'/'weights'/'best.pt').read_bytes()==b'valuable-old-weights'
