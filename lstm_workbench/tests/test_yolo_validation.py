"""独立验证的指标、类别门禁、预标注审核和错例回流。"""
import json
import threading
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from lstm_workbench import model_registry, yolo_validation


BOX = {'class_id':0,'points':[[.1,.1],[.4,.1],[.4,.4],[.1,.4]]}


def test_preannotation_rejects_classes_changed_while_predicting(tmp_path, monkeypatch):
    from lstm_workbench.parts_store import PartsStore
    folder = tmp_path / 'project'
    store = PartsStore(folder)
    store.set_classes(['零件'], 0)
    source = tmp_path / 'input.png'
    cv2.imencode('.png', np.zeros((80, 100, 3), np.uint8))[1].tofile(source)
    item = store.import_source(str(source))['items'][0]
    library = tmp_path / 'models'
    library.mkdir()
    (library / 'part.onnx').write_bytes(b'model')
    monkeypatch.setattr(model_registry, 'YOLO_LIBRARY', library)
    def predicting(request, **kwargs):
        store.set_classes(['另一种零件'], store.snapshot()['revision'])
        return fake_result(request)
    monkeypatch.setattr(yolo_validation, '_run_worker', predicting)
    result = yolo_validation.preannotate(folder, {'model': 'part.onnx', 'items': [item['id']]})
    assert result['status'] == 'failed' and '类别' in result['error']
    assert result['applied'] == []
    assert store.snapshot()['items'][0]['boxes'] == []


def test_prediction_class_guard_is_atomic_with_item_write(tmp_path):
    from lstm_workbench.parts_store import PartsStore
    store = PartsStore(tmp_path / 'project')
    store.set_classes(['旧类别'], 0)
    source = tmp_path / 'input.png'
    cv2.imencode('.png', np.zeros((80, 100, 3), np.uint8))[1].tofile(source)
    item = store.import_source(str(source))['items'][0]
    store.set_classes(['新类别'], store.snapshot()['revision'])
    with pytest.raises(ValueError, match='类别'):
        store.save_item(item['id'], [BOX], 'pending', item['revision'], expected_classes=['旧类别'])
    assert store.snapshot()['items'][0]['boxes'] == []


def test_preannotation_reports_classes_changed_between_snapshot_and_save(tmp_path, monkeypatch):
    from lstm_workbench.parts_store import PartsStore
    folder = tmp_path / 'project'
    store = PartsStore(folder)
    store.set_classes(['零件'], 0)
    source = tmp_path / 'input.png'
    cv2.imencode('.png', np.zeros((80, 100, 3), np.uint8))[1].tofile(source)
    item = store.import_source(str(source))['items'][0]
    library = tmp_path / 'models'
    library.mkdir()
    (library / 'part.onnx').write_bytes(b'model')
    monkeypatch.setattr(model_registry, 'YOLO_LIBRARY', library)
    monkeypatch.setattr(yolo_validation, '_run_worker', fake_result)
    original = PartsStore.save_item
    def changed_before_save(self, *args, **kwargs):
        self.set_classes(['新类别'], self.snapshot()['revision'])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PartsStore, 'save_item', changed_before_save)
    result = yolo_validation.preannotate(folder, {'model': 'part.onnx', 'items': [item['id']]})
    assert result['applied'] == [] and result['skipped'] == [item['id']]
    assert '类别' in result['skip_reasons'][item['id']]
    assert store.snapshot()['items'][0]['boxes'] == []


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    folder = tmp_path/'p'
    folder.mkdir()
    image = tmp_path/'a.png'
    cv2.imencode('.png',np.zeros((80,100,3),np.uint8))[1].tofile(image)
    snapshot = {'classes':['零件'],'items':[{'id':'one','name':'a.png','status':'reviewed',
                'revision':1,'boxes':[BOX],'source_group':'camera-a','image':str(image)}]}
    class Store:
        def __init__(self, folder): pass
        def snapshot(self): return snapshot
        def image_path(self, item): return image
        def save_item(self, item, boxes, status, revision, *, expected_classes=None):
            assert revision == snapshot['items'][0]['revision']
            assert expected_classes is None or expected_classes == snapshot['classes']
            snapshot['items'][0].update({'boxes':boxes,'status':status,'revision':revision+1})
            return snapshot['items'][0]
    monkeypatch.setattr(yolo_validation, 'PartsStore', Store)
    library = tmp_path/'models'
    library.mkdir()
    (library/'part.onnx').write_bytes(b'model')
    monkeypatch.setattr(model_registry, 'YOLO_LIBRARY', library)
    return folder, snapshot


def fake_result(request, **kwargs):
    return {'models':[{'classes':['零件'],'items':[{'id':item['id'],'predictions':[dict(BOX,confidence=.8)],'elapsed_ms':2.5}
                                               for item in request['items']]}]}


def test_polygon_metrics_distinguish_duplicates_and_wrong_classes():
    pred = [dict(BOX,confidence=.9),dict(BOX,confidence=.8),dict(BOX,class_id=1,confidence=.7)]
    result = yolo_validation.score_predictions([BOX],pred,['a','b'],.5)
    assert (result['tp'],result['fp'],result['fn']) == (1,2,0)
    assert result['precision'] == pytest.approx(1/3)
    assert result['per_class'][1]['fp'] == 1


def test_validation_persists_frozen_inputs_metrics_and_feedback(prepared, monkeypatch):
    folder,snapshot = prepared
    monkeypatch.setattr(yolo_validation, '_run_worker', fake_result)
    result = yolo_validation.start_validation(folder, {'model':'part.onnx','items':['one'],'conf':.2,'iou':.5})
    assert result['status'] == 'done'
    assert result['metrics']['tp'] == 1
    assert result['metrics']['inference_ms'] == 2.5
    assert result['models'][0]['sha256']
    assert yolo_validation.validation_detail(folder,result['id'])['items'][0]['predictions'][0]['confidence'] == .8
    assert yolo_validation.image_file(folder,result['id'],'one').is_file()
    feedback = yolo_validation.feedback(folder,result['id'],['one'])
    assert feedback['updated'] == ['one']
    assert snapshot['items'][0]['status'] == 'pending'
    assert snapshot['items'][0]['boxes'] == [BOX]


def test_preannotation_skips_reviewed_and_never_reviews_predictions(prepared, monkeypatch):
    folder,snapshot = prepared
    monkeypatch.setattr(yolo_validation, '_run_worker', fake_result)
    reviewed = yolo_validation.preannotate(folder, {'model':'part.onnx','items':['one']})
    assert reviewed['applied'] == [] and reviewed['skipped'] == ['one']
    snapshot['items'][0]['status'] = 'unlabeled'
    result = yolo_validation.preannotate(folder, {'model':'part.onnx','items':['one']})
    assert result['applied'] == ['one']
    assert snapshot['items'][0]['status'] == 'pending'
    assert result['items'][0]['metrics'] is None


def test_model_classes_must_match_dataset(prepared, monkeypatch):
    folder,_ = prepared
    monkeypatch.setattr(yolo_validation,'_run_worker',lambda request,**kw:{'models':[{'classes':['另一类'],'items':[]}]})
    result = yolo_validation.start_validation(folder,{'model':'part.onnx','items':['one']})
    assert result['status'] == 'failed' and '类别' in result['error']


@pytest.mark.parametrize('params',[{'model':'../outside.onnx','items':['one']},
                                 {'model':'part.onnx','items':['one'],'conf':float('nan')},
                                 {'model':'part.onnx','items':['one'],'iou':1.1},
                                 {'model':'part.onnx','items':['missing']},
                                 {'model':'part.onnx','items':[]}])
def test_validation_rejects_unsafe_or_invalid_requests(prepared,params):
    with pytest.raises((ValueError,FileNotFoundError)):
        yolo_validation.prepare_validation(prepared[0],params)


def test_stop_and_model_changes_are_recorded(prepared,monkeypatch):
    folder,_ = prepared
    record = yolo_validation.prepare_validation(folder,{'model':'part.onnx','items':['one']})
    stop = threading.Event()
    stop.set()
    result = yolo_validation.execute_validation(folder,record['id'],stop)
    assert result['status'] == 'stopped'
    record = yolo_validation.prepare_validation(folder,{'model':'part.onnx','items':['one']})
    Path(record['models'][0]['path']).write_bytes(b'changed')
    result = yolo_validation.execute_validation(folder,record['id'])
    assert result['status'] == 'failed' and '变化' in result['error']


def test_path_cannot_escape_evaluation_folder(prepared):
    with pytest.raises(ValueError):
        yolo_validation.validation_detail(prepared[0],'../other')


def test_restarted_service_marks_only_running_jobs_failed(prepared,monkeypatch):
    folder,_=prepared
    record=yolo_validation.prepare_validation(folder,{'model':'part.onnx','items':['one']})
    assert yolo_validation.mark_interrupted(folder)==0, '另一个同进程服务不能覆盖当前任务'
    path=folder/'parts'/'evaluations'/record['id']/'result.json'
    record.pop('owner_pid')
    path.write_text(json.dumps(record),encoding='utf-8')
    assert yolo_validation.mark_interrupted(folder)==1
    assert yolo_validation.validation_detail(folder,record['id'])['status']=='failed'
    assert yolo_validation.mark_interrupted(folder)==0
    assert yolo_validation.list_validations(folder)[0]['id']==record['id']


def test_published_split_and_onnx_comparison(prepared,monkeypatch):
    folder,snapshot=prepared
    image=Path(snapshot['items'][0]['image'])
    class Store(yolo_validation.PartsStore):
        def version_info(self,version):
            return {'id':version,'sha256':'version-hash','classes':['零件'],
                    'items':[dict(snapshot['items'][0],image=str(image),split='val')]}
    monkeypatch.setattr(yolo_validation,'PartsStore',Store)
    run=folder/'yolo'/'runs'/'one'
    (run/'weights').mkdir(parents=True)
    (run/'weights'/'best.pt').write_bytes(b'pt')
    (run/'export.json').write_text(json.dumps({'file':'part.onnx','source_sha256':model_registry.file_sha256(run/'weights'/'best.pt'),
                                             'sha256':model_registry.file_sha256(model_registry.YOLO_LIBRARY/'part.onnx')}))
    def compare(request,**kwargs):
        return {'models':[fake_result(request)['models'][0],fake_result(request)['models'][0]]}
    monkeypatch.setattr(yolo_validation,'_run_worker',compare)
    result=yolo_validation.start_validation(folder,{'run':'one','model':'best.pt','dataset_version':'v1','compare_onnx':True})
    assert result['status']=='done',result
    assert result['dataset']['sha256']=='version-hash'
    assert result['comparison']['metrics']['tp']==1
    assert result['comparison']['agreement']['f1']==1
    (run/'weights'/'last.pt').write_bytes(b'last')
    with pytest.raises(ValueError,match='best.pt'):
        yolo_validation.prepare_validation(folder,{'run':'one','model':'last.pt','items':['one'],'compare_onnx':True})
    (run/'weights'/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='重新导出'):
        yolo_validation.prepare_validation(folder,{'run':'one','model':'best.pt','items':['one'],'compare_onnx':True})
    for params in ({'dataset_version':'v1','split':'train'}, {'dataset_version':'v1','split':'test'},
                   {'dataset_version':'v1','mode':'preannotate'}, {'model':'part.onnx','items':['one'],'compare_onnx':True}):
        with pytest.raises(ValueError):
            yolo_validation.prepare_validation(folder,params)


def test_running_errors_and_changed_annotations_are_not_overwritten(prepared,monkeypatch):
    folder,snapshot=prepared
    monkeypatch.setattr(yolo_validation,'_run_worker',lambda request,**kw:{'models':[]})
    result=yolo_validation.start_validation(folder,{'model':'part.onnx','items':['one']})
    assert result['status']=='failed'
    assert yolo_validation.execute_validation(folder,result['id'])==result
    snapshot['items'][0]['status']='unlabeled'
    record=yolo_validation.prepare_validation(folder,{'model':'part.onnx','items':['one'],'mode':'preannotate'})
    snapshot['items'][0]['revision']+=1
    monkeypatch.setattr(yolo_validation,'_run_worker',fake_result)
    result=yolo_validation.execute_validation(folder,record['id'])
    assert result['skipped']==['one'] and result['applied']==[]
    with pytest.raises(ValueError):
        yolo_validation.feedback(folder,result['id'],['elsewhere'])
    with pytest.raises(ValueError):
        yolo_validation.feedback(folder,result['id'],[])


@pytest.mark.parametrize('box',[dict(BOX,class_id=100),dict(BOX,class_id=True),dict(BOX,points=[[0,0]]),dict(BOX,confidence=float('nan'))])
def test_malformed_predictions_are_rejected(box):
    with pytest.raises(ValueError):
        yolo_validation._normalize_predictions([box],['a'])


def test_degenerate_predicted_box_is_ignored():
    assert yolo_validation._normalize_predictions([dict(BOX,points=[[0,0]]*4)],['a'])==[]
    assert yolo_validation._polygon_iou([[0,0]],BOX['points'])==0
    assert yolo_validation._polygon_iou([[0,0]]*4,BOX['points'])==0


def test_subprocess_transport_success_failure_stop_and_timeout(prepared,tmp_path,monkeypatch):
    folder,_=prepared
    # 真起当前 Python 子进程，只将 GPU 推理脚本替换成确定性的 JSON 回显脚本。
    script=tmp_path/'yolo_validation_worker.py'
    script.write_text("import json,sys; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps({'models':[]}),encoding='utf-8')")
    monkeypatch.setattr(yolo_validation,'__file__',str(tmp_path/'yolo_validation.py'))
    monkeypatch.setattr(yolo_validation.yolo_jobs,'_python',lambda:sys.executable)
    root=tmp_path/'job'
    root.mkdir()
    assert yolo_validation._run_worker({},folder=folder,root=root)=={'models':[]}
    script.write_text("raise ValueError('expected failure')")
    with pytest.raises(RuntimeError,match='expected failure'):
        yolo_validation._run_worker({},folder=folder,root=root)
    script.write_text('import time; time.sleep(10)')
    stop=threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        yolo_validation._run_worker({},folder=folder,root=root,stop_event=stop)
    monkeypatch.setattr(yolo_validation,'WORKER_TIMEOUT',0)
    with pytest.raises(TimeoutError):
        yolo_validation._run_worker({},folder=folder,root=root)


def test_worker_uses_actual_obb_api_and_writes_result(prepared,tmp_path,monkeypatch):
    from lstm_workbench import yolo_validation_worker as worker
    folder,snapshot=prepared
    class Array:
        def __init__(self,data): self.data=np.asarray(data)
        def cpu(self): return self
        def numpy(self): return self.data
        def int(self): return self
        def tolist(self): return self.data.tolist()
    class Model:
        def __init__(self,path,task):
            assert task=='obb'
            self.ready=False
        @property
        def names(self):
            assert self.ready, '必须先按指定设备初始化推理器'
            return {0:'零件'}
        def predict(self,image,**kwargs):
            assert kwargs['save'] is False
            assert kwargs['device']=='cpu'
            self.ready=True
            obb=SimpleNamespace(xyxyxyxyn=Array([BOX['points']]),cls=Array([0]),conf=Array([.9]))
            return [SimpleNamespace(obb=obb)]
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Model))
    request={'models':[{'path':'test.onnx'}], 'items':[{'id':'one','image':snapshot['items'][0]['image']}],
             'params':{'conf':.2,'imgsz':64,'device':'cpu'}}
    source,target=tmp_path/'request.json',tmp_path/'result.json'
    source.write_text(json.dumps(request),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['worker',str(source),str(target)])
    worker.main()
    result=json.loads(target.read_text(encoding='utf-8'))
    assert result['models'][0]['classes']==['零件']
    assert result['models'][0]['items'][0]['predictions'][0]['points']==BOX['points']


def test_resolved_evaluation_and_weight_roots_must_stay_inside_project(prepared,tmp_path,monkeypatch):
    folder,_=prepared
    real_resolve=Path.resolve
    outside=tmp_path/'outside'
    outside.mkdir()
    def resolve(path,*args,**kwargs):
        result=real_resolve(path,*args,**kwargs)
        if path.name=='evaluations' or path.name=='weights':
            return outside
        return result
    # 模拟 Windows Junction / 符号链接解析结果，不依赖管理员权限。
    monkeypatch.setattr(Path,'resolve',resolve)
    with pytest.raises(ValueError,match='越界'):
        yolo_validation.list_validations(folder)
    with pytest.raises(ValueError,match='越界'):
        yolo_validation.prepare_validation(folder,{'run':'one','model':'best.pt','items':['one']})
