"""在现有 YOLO Python 环境内执行预测，JSON 文件传递输入输出。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _write_progress(progress_path: Path | None, done: int, total: int) -> None:
    if progress_path is None:
        return
    try:
        tmp = progress_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'done': done, 'total': total}, ensure_ascii=False), encoding='utf-8')
        tmp.replace(progress_path)
    except OSError:
        pass


def predict(request: dict, progress_path: Path | None = None) -> dict:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    params=request['params']
    outputs=[]
    total_steps = len(request.get('models') or []) * len(request.get('items') or [])
    done_steps = 0
    _write_progress(progress_path, 0, total_steps)
    for descriptor in request['models']:
        model=YOLO(descriptor['path'],task='obb')
        classes=[]
        rows=[]
        for item in request['items']:
            image=cv2.imdecode(np.fromfile(item['image'],dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('无法解码验证图片：'+item['id'])
            started=time.perf_counter()
            # API 来源：Ultralytics docs/en/tasks/obb.md；坐标属性也与安装包 results.OBB 一致。
            result=model.predict(image,conf=params['conf'],iou=.7,imgsz=params['imgsz'],
                                 device=params.get('device') or 'cpu',verbose=False,save=False)[0]
            elapsed=(time.perf_counter()-started)*1000
            # ONNX 的 names 属性会懒加载推理器。必须先按用户指定设备 predict，
            # 否则属性读取会用默认 GPU 初始化，CPU 环境也可能因此失败。
            names=model.names
            classes=[names[i] for i in range(len(names))] if isinstance(names,dict) else list(names)
            boxes=[]
            if result.obb is None:
                raise ValueError('所选模型不是 OBB 旋转框模型')
            points=result.obb.xyxyxyxyn.cpu().numpy()
            ids=result.obb.cls.int().cpu().tolist()
            confidences=result.obb.conf.cpu().tolist()
            for class_id,vertices,confidence in zip(ids,points,confidences):
                boxes.append({'class_id':class_id,'points':vertices.tolist(),'confidence':confidence})
            rows.append({'id':item['id'],'predictions':boxes,'elapsed_ms':elapsed})
            done_steps += 1
            _write_progress(progress_path, done_steps, total_steps)
        outputs.append({'classes':classes,'items':rows})
    return {'models':outputs}


def main() -> None:
    source,target=map(Path,sys.argv[1:3])
    result=predict(json.loads(source.read_text(encoding='utf-8')), progress_path=source.parent / 'progress.json')
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(target)


if __name__=='__main__':
    main()
