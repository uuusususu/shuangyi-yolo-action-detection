"""独立 OBB 验证、模型预标注及错例回流；只接受本工程模型或模型库模型。"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import model_registry, yolo_jobs
from .parts_store import PartsStore
from .project_store import ProjectStore

_LOCK = threading.RLock()
_NAME = re.compile(r'^[\w-]+$')
WORKER_TIMEOUT = 1800


def _named(root: Path, name: str) -> Path:
    if not isinstance(name,str) or not _NAME.fullmatch(name):
        raise ValueError('记录名称不合法')
    path = (root/name).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('记录路径越界')
    return path


def _project_path(folder: Path, path: Path) -> Path:
    project=folder.resolve()
    resolved=path.resolve()
    if project not in resolved.parents:
        raise ValueError('工程记录路径越界')
    return resolved


def _root(folder: Path) -> Path:
    return _project_path(folder,folder/'parts'/'evaluations')


def _read(path: Path) -> dict:
    # Windows 上短暂读句柄也可能阻止原子替换；任务轮询与写终态共用锁。
    with _LOCK:
        return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, record: dict) -> None:
    with _LOCK:
        ProjectStore.write_json(path, record)


def fail_validation(folder: Path, evaluation_id: str, error: str, record: dict) -> None:
    target = _named(_root(folder), evaluation_id) / 'result.json'
    _write(target, dict(record, status='failed', error=error, finished_at=time.time()))


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(params.get(key,default))
        if isinstance(params.get(key),bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError
    except (ValueError,TypeError,OverflowError):
        raise ValueError(f'{key} 必须在 {low} 到 {high} 之间') from None
    return value


def _model(folder: Path, params: dict, compare: bool = False) -> dict:
    name = str(params.get('model') or 'best.pt')
    run_name = str(params.get('run') or '')
    if compare:
        run = _project_path(folder,_named(yolo_jobs.runs_dir(folder),run_name))
        exported = yolo_jobs._read_export(run) or {}
        name = str(exported.get('file') or '')
        if not name:
            raise ValueError('这一轮还没有导出 ONNX，不能对照')
        best = _project_path(folder,run/'weights'/'best.pt')
        if not best.is_file() or not exported.get('source_sha256') or exported['source_sha256']!=model_registry.file_sha256(best):
            raise ValueError('导出记录缺少对应 best.pt 指纹或权重已变化，请重新导出 ONNX 后对照')
    if name in {'best.pt','last.pt'}:
        run = _project_path(folder,_named(yolo_jobs.runs_dir(folder),run_name))
        root = _project_path(folder,run/'weights')
        path = root/name
    else:
        if Path(name).name != name or '/' in name or '\\' in name or ':' in name or not name.lower().endswith('.onnx'):
            raise ValueError('模型只能选本工程 best.pt/last.pt 或模型库 ONNX 文件名')
        root = model_registry.YOLO_LIBRARY
        path = root/name
    if root.resolve() not in path.resolve().parents or not path.is_file():
        raise FileNotFoundError(f'模型不存在或路径越界：{name}')
    if compare and exported.get('sha256') != model_registry.file_sha256(path):
        raise ValueError('ONNX 已变化或导出指纹不完整，请重新导出后对照')
    return {'identifier':name,'path':str(path.resolve()),'sha256':model_registry.file_sha256(path),
            'format':path.suffix.lower().lstrip('.')}


def _collect(folder: Path, params: dict) -> tuple[list[str],list[dict],dict | None]:
    store = PartsStore(folder)
    version_id = str(params.get('dataset_version') or '')
    if version_id:
        if params['mode'] == 'preannotate':
            raise ValueError('预标注只处理工作区素材，不能修改已发布版本')
        version = store.version_info(version_id)
        split = str(params.get('split') or 'val')
        if split not in {'val','test'}:
            raise ValueError('独立验证只使用验证集或测试集')
        items = [dict(item,ground_truth=copy.deepcopy(item.get('boxes') or []))
                 for item in version['items'] if item.get('split') == split]
        if not items:
            raise ValueError('所选集合没有素材')
        return version['classes'],items,{'id':version['id'],'sha256':version['sha256'],'split':split}
    snapshot = store.snapshot()
    selected = params.get('items')
    if not isinstance(selected,list) or not selected or len(selected)>2000 or any(not isinstance(i,str) for i in selected):
        raise ValueError('请选择 1 到 2000 条素材')
    by_id = {item['id']:item for item in snapshot['items']}
    if any(item not in by_id for item in selected):
        raise ValueError('所选素材不存在')
    items = []
    for item_id in dict.fromkeys(selected):
        item = copy.deepcopy(by_id[item_id])
        item['image'] = str(store.image_path(item_id))
        item['ground_truth'] = copy.deepcopy(item.get('boxes') or []) if item.get('status') in {'reviewed','negative'} else None
        items.append(item)
    return list(snapshot['classes']),items,None


def prepare_validation(folder: Path, params: dict) -> dict:
    """快速冻结输入，返回可立即轮询的记录；不在 HTTP 接收线程里加载模型。"""
    params = dict(params)
    mode = params.get('mode') or 'validate'
    if mode not in {'validate','preannotate'}:
        raise ValueError('不支持的验证任务类型')
    params['mode'] = mode
    params['conf'] = _number(params,'conf',.25,0,1)
    params['iou'] = _number(params,'iou',.5,.01,1)
    params['imgsz'] = yolo_jobs._integer(params,'imgsz',640,32,4096)
    if not isinstance(params.get('compare_onnx',False),bool):
        raise ValueError('compare_onnx 必须是布尔值')
    classes,items,version = _collect(folder,params)
    if not classes:
        raise ValueError('请先设置零件类别')
    models = [_model(folder,params)]
    if params.get('compare_onnx'):
        if models[0]['identifier'] != 'best.pt':
            raise ValueError('PT 与 ONNX 对照必须选择导出所对应的 best.pt，不能使用 last.pt')
        models.append(_model(folder,params,compare=True))
    evaluation_id = uuid.uuid4().hex
    root = _named(_root(folder),evaluation_id)
    (root/'images').mkdir(parents=True)
    try:
        for item in items:
            image = Path(item['image'])
            target = root/'images'/f"{item['id']}{image.suffix.lower()}"
            if not _NAME.fullmatch(str(item['id'])):
                raise ValueError('素材编号不合法')
            shutil.copyfile(image,target)
            item['image'] = target.relative_to(root).as_posix()
            item['predictions'] = []
            item['metrics'] = None
        record = {'id':evaluation_id,'status':'running','mode':mode,'params':params,'created_at':time.time(),
                  'owner_pid':os.getpid(),'owner_created':yolo_jobs._pid_created(os.getpid()),
                  'classes':classes,'dataset':version,'models':models,'items':items,
                  'metrics':None,'comparison':None,'applied':[],'skipped':[],'error':''}
        _write(root/'result.json',record)
        return record
    except Exception:
        # root 是本函数创建的 UUID 子目录，清理仅限这一项。
        shutil.rmtree(root)
        raise


def _with_progress(root: Path, record: dict) -> dict:
    total = len(record.get('items') or []) * max(1, len(record.get('models') or []))
    item_total = len(record.get('items') or [])
    record['total'] = item_total
    if record.get('status') == 'done':
        record['done'] = item_total
        return record
    prog_path = root / 'progress.json'
    if prog_path.is_file():
        try:
            prog = json.loads(prog_path.read_text(encoding='utf-8'))
            raw_done = int(prog.get('done') or 0)
            model_count = max(1, len(record.get('models') or []))
            record['done'] = min(item_total, raw_done // model_count if model_count > 1 else raw_done)
            return record
        except (OSError, ValueError, TypeError):
            pass
    record.setdefault('done', 0)
    return record


def validation_detail(folder: Path, evaluation_id: str) -> dict:
    root = _named(_root(folder), evaluation_id)
    return _with_progress(root, _read(root / 'result.json'))


def list_validations(folder: Path) -> list[dict]:
    root = _root(folder)
    if not root.is_dir():
        return []
    summaries=[]
    for path in root.iterdir():
        if path.is_dir() and _NAME.fullmatch(path.name) and root.resolve() in path.resolve().parents:
            try:
                record = _with_progress(path, _read(path/'result.json'))
            except (OSError,ValueError):
                continue
            summaries.append({key:record.get(key) for key in ('id','status','mode','done','total','created_at','finished_at','params','metrics','comparison','error','applied','skipped')})
    return sorted(summaries,key=lambda item:item.get('created_at') or 0,reverse=True)


def mark_interrupted(folder: Path) -> int:
    """仅供服务启动时调用：上次进程退出后，不能让旧任务永久显示运行中。"""
    count=0
    for summary in list_validations(folder):
        if summary.get('status')=='running':
            record=validation_detail(folder,summary['id'])
            owner=int(record.get('owner_pid') or 0)
            created=record.get('owner_created')
            # 和训练一样核对创建时刻：PID 被别的程序复用时，这条记录不能一直挂在"运行中"。
            if owner>0 and (owner==os.getpid() or yolo_jobs._pid_alive(owner,float(created) if created is not None else None)):
                continue
            record.update(status='failed',error='上次任务中断，请重新发起验证或预标注',finished_at=time.time())
            _write(_named(_root(folder),summary['id'])/'result.json',record)
            count+=1
    return count


def image_file(folder: Path, evaluation_id: str, item_id: str) -> Path:
    root = _named(_root(folder),evaluation_id)
    record = validation_detail(folder,evaluation_id)
    item = next((item for item in record['items'] if item['id']==item_id),None)
    if item is None:
        raise FileNotFoundError('验证素材不存在')
    path = (root/item['image']).resolve()
    if (root/'images').resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError('验证素材不存在或路径越界')
    return path


def _polygon_iou(left: list, right: list) -> float:
    a,b = np.asarray(left,dtype=np.float32),np.asarray(right,dtype=np.float32)
    if a.shape!=(4,2) or b.shape!=(4,2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.
    area_a,area_b = abs(cv2.contourArea(a)),abs(cv2.contourArea(b))
    if area_a<=0 or area_b<=0 or not cv2.isContourConvex(a) or not cv2.isContourConvex(b):
        return 0.
    intersection,_ = cv2.intersectConvexConvex(a,b)
    union=area_a+area_b-intersection
    return max(0.,min(1.,float(intersection/union))) if union>0 else 0.


def _measures(tp: int, fp: int, fn: int) -> dict:
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}


def score_predictions(truth: list[dict], predictions: list[dict], classes: list[str], iou: float = .5) -> dict:
    per_class=[]
    for class_id,name in enumerate(classes):
        gt=[box for box in truth if box['class_id']==class_id]
        pred=sorted((box for box in predictions if box['class_id']==class_id),key=lambda box:box.get('confidence',0),reverse=True)
        matched=set()
        tp=0
        for box in pred:
            candidates=[(_polygon_iou(box['points'],target['points']),i) for i,target in enumerate(gt) if i not in matched]
            if candidates:
                overlap,index=max(candidates)
                if overlap>=iou:
                    matched.add(index)
                    tp+=1
        per_class.append({'class_id':class_id,'name':name,**_measures(tp,len(pred)-tp,len(gt)-tp)})
    return {**_measures(*(sum(row[key] for row in per_class) for key in ('tp','fp','fn'))),'per_class':per_class}


def _run_worker(request: dict, *, folder: Path, root: Path, stop_event: threading.Event | None = None) -> dict:
    request_path=root/'request.json'
    response_path=root/'predictions.json'
    _write(request_path,request)
    command=[yolo_jobs._python(),str(Path(__file__).with_name('yolo_validation_worker.py')),str(request_path),str(response_path)]
    with (root/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=yolo_jobs._child_env(folder),
                                 creationflags=yolo_jobs._NO_WINDOW)
        deadline=time.monotonic()+WORKER_TIMEOUT
        try:
            while process.poll() is None:
                if stop_event is not None and stop_event.wait(.1):
                    raise InterruptedError('验证已停止')
                if stop_event is None:
                    time.sleep(.1)
                if time.monotonic()>deadline:
                    raise TimeoutError('模型验证超时')
            if process.returncode!=0 or not response_path.is_file():
                tail=(root/'worker.log').read_text(encoding='utf-8',errors='replace')[-1200:]
                raise RuntimeError('模型推理失败：'+tail)
            return _read(response_path)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _normalize_predictions(boxes: list, classes: list[str]) -> list[dict]:
    result=[]
    for box in boxes:
        class_id=box.get('class_id')
        if isinstance(class_id,bool) or not isinstance(class_id,int) or not 0<=class_id<len(classes):
            raise ValueError('模型预测类别索引越界')
        points=np.asarray(box.get('points'),dtype=np.float64)
        confidence=float(box.get('confidence',0))
        if points.shape!=(4,2) or not np.isfinite(points).all() or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('模型预测坐标或置信度无效')
        points=np.clip(points,0,1)
        if not cv2.isContourConvex(points.astype(np.float32)) or abs(cv2.contourArea(points.astype(np.float32)))<1e-8:
            continue
        result.append({'class_id':class_id,'points':points.tolist(),'confidence':confidence})
    return result


def _aggregate(items: list[dict], classes: list[str], comparison: bool = False) -> dict | None:
    key='comparison_metrics' if comparison else 'metrics'
    scored=[item[key] for item in items if item.get(key) is not None]
    if not scored:
        return None
    rows=[]
    for class_id,name in enumerate(classes):
        rows.append({'class_id':class_id,'name':name,**_measures(*(sum(score['per_class'][class_id][k] for score in scored) for k in ('tp','fp','fn')))})
    return {**_measures(*(sum(score[k] for score in scored) for k in ('tp','fp','fn'))),'per_class':rows,
            'scored_images':len(scored),'definition':'固定置信度与旋转框 IoU 阈值下的匹配指标，不是 mAP'}


def execute_validation(folder: Path, evaluation_id: str, stop_event: threading.Event | None = None) -> dict:
    root=_named(_root(folder),evaluation_id)
    record=validation_detail(folder,evaluation_id)
    if record['status']!='running':
        return record
    try:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('验证已停止')
        for model in record['models']:
            path=Path(model['path'])
            if not path.is_file() or model_registry.file_sha256(path)!=model['sha256']:
                raise ValueError('模型文件已变化，请重新发起验证')
        request={'models':record['models'],'items':[{'id':item['id'],'image':str((root/item['image']).resolve())} for item in record['items']],
                 'params':record['params']}
        output=_run_worker(request,folder=folder,root=root,stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('验证已停止')
        outputs=output.get('models') or []
        if len(outputs)!=len(record['models']):
            raise ValueError('模型推理返回的结果数量不完整')
        for model_index,prediction_set in enumerate(outputs):
            if prediction_set.get('classes')!=record['classes']:
                raise ValueError('模型类别顺序与数据集不一致，请选择正确的模型或版本')
            by_id={item['id']:item for item in prediction_set.get('items') or []}
            elapsed=[]
            for item in record['items']:
                if item['id'] not in by_id:
                    raise ValueError('模型推理缺少素材结果')
                predicted=by_id[item['id']]
                boxes=_normalize_predictions(predicted.get('predictions') or [],record['classes'])
                prediction_key='predictions' if model_index==0 else 'comparison_predictions'
                metric_key='metrics' if model_index==0 else 'comparison_metrics'
                item[prediction_key]=boxes
                item[metric_key]=score_predictions(item['ground_truth'],boxes,record['classes'],record['params']['iou']) if item['ground_truth'] is not None else None
                milliseconds=float(predicted.get('elapsed_ms',0))
                if not math.isfinite(milliseconds) or milliseconds<0:
                    raise ValueError('推理时间无效')
                elapsed.append(milliseconds)
            metrics=_aggregate(record['items'],record['classes'],model_index==1)
            summary={'metrics':metrics,'inference_ms':sum(elapsed)/max(1,len(elapsed)),
                     'images':len(elapsed),'model':record['models'][model_index]}
            if metrics is not None:
                metrics['inference_ms']=summary['inference_ms']
            if model_index==0:
                record['metrics']=metrics
                record['inference_ms']=summary['inference_ms']
            else:
                record['comparison']=summary
        if record['comparison'] is not None:
            # 同一张图、同一阈值下将 PT 预测作为参照，单独报告导出前后的预测一致程度。
            reference=[dict(item,metrics=score_predictions(item['predictions'],item['comparison_predictions'],
                                                        record['classes'],record['params']['iou']))
                       for item in record['items']]
            record['comparison']['agreement']=_aggregate(reference,record['classes'])
            record['comparison']['agreement']['definition']='以 PT 预测为参照的 ONNX 预测匹配程度，不是真值精度'
        if record['mode']=='preannotate':
            store=PartsStore(folder)
            with _LOCK:
                snapshot=store.snapshot()
                if snapshot['classes'] != record['classes']:
                    raise ValueError('预标注期间类别表已变化，预测未应用，请重新发起预标注')
                current={item['id']:item for item in snapshot['items']}
                for item in record['items']:
                    latest=current.get(item['id'])
                    if latest is None or latest.get('status') in {'reviewed','negative'} or latest['revision']!=item['revision']:
                        record['skipped'].append(item['id'])
                        continue
                    # 自动结果永远是待审核；不把模型的置信度冒充人工标注字段。
                    boxes=[{k:box[k] for k in ('class_id','points')} for box in item['predictions']]
                    try:
                        store.save_item(item['id'],boxes,'pending',latest['revision'],
                                        expected_classes=record['classes'])
                    except ValueError as exc:
                        record['skipped'].append(item['id'])
                        record.setdefault('skip_reasons', {})[item['id']] = str(exc)
                    else:
                        record['applied'].append(item['id'])
        record['status']='done'
    except InterruptedError as exc:
        record.update(status='stopped',error=str(exc))
    except Exception as exc:
        record.update(status='failed',error=str(exc))
    record['finished_at']=time.time()
    _write(root/'result.json',record)
    return record


def start_validation(folder: Path, params: dict, stop_event: threading.Event | None = None) -> dict:
    record=prepare_validation(folder,dict(params,mode='validate'))
    return execute_validation(folder,record['id'],stop_event)


def preannotate(folder: Path, params: dict, stop_event: threading.Event | None = None) -> dict:
    record=prepare_validation(folder,dict(params,mode='preannotate'))
    return execute_validation(folder,record['id'],stop_event)


def feedback(folder: Path, evaluation_id: str, item_ids: list[str]) -> dict:
    record=validation_detail(folder,evaluation_id)
    if record['status']!='done':
        raise ValueError('只能回流已完成的验证结果')
    if not isinstance(item_ids,list) or not item_ids or any(not isinstance(i,str) for i in item_ids):
        raise ValueError('请选择要回流的错例')
    eligible={item['id'] for item in record['items']}
    if any(i not in eligible for i in item_ids):
        raise ValueError('回流素材不属于本轮验证')
    updated,skipped=[],[]
    store=PartsStore(folder)
    with _LOCK:
        current={item['id']:item for item in store.snapshot()['items']}
        for item_id in dict.fromkeys(item_ids):
            item=current.get(item_id)
            if item is None:
                skipped.append(item_id)
                continue
            try:
                store.save_item(item_id,item.get('boxes') or [],'pending',item['revision'])
            except ValueError:
                skipped.append(item_id)
            else:
                updated.append(item_id)
        record['feedback']={'updated':updated,'skipped':skipped,'at':time.time()}
        _write(_named(_root(folder),evaluation_id)/'result.json',record)
    return {'updated':updated,'skipped':skipped}
