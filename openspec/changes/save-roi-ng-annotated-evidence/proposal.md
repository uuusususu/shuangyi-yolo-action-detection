## Why

当前 ROI 数量检测已经能够把最终 NG 的同帧绿色 YOLO OBB 图用于弹窗和证据保存，但图片仍是整张相机画面。现场真正需要员工检查的是配置的识别 ROI；整图会缩小关键区域，并把其他 PCB 或无关目标带入 NG 画面。

## What Changes

- 仅在 ROI 数量检测产生一次最终 NG 结果且“保存 NG 证据”已启用时，保存一张该轮证据图片。
- 直接使用产生 NG 的 overlay 事件中的 `source_frame`、`source_frame_id` 和 detections，禁止回退读取 worker 最新帧。
- 严格按当前识别 ROI 裁剪 NG 弹窗和保存的 `frame.jpg`，不增加边距；正常主页面仍保持整图显示。
- 仅保留中心点位于识别 ROI 内的 detections，并在裁剪图上绘制绿色闭合旋转框；越过 ROI 边界的框线自然裁切，不绘制类别、置信度、跟踪 ID、ROI 或其他文字。
- `metadata.json` 仅记录识别 ROI 内 detections，把 polygon、box、center/center_px 转换为裁剪图局部坐标，并记录识别 ROI 在原图中的像素位置和原图尺寸。
- 继续只保存一个 `frame.jpg`，不另外保存无框原图；保留现有日期/轮次/帧号目录结构和 `metadata.json`。
- 识别 ROI 内没有 detections 时仍显示并保存同帧 ROI 无框图片；同帧图像缺失、帧编号不一致或识别 ROI 无效时不回退整图并写入错误日志。
- 保存失败只记录日志，不阻塞检测、不改变 NG 统计、弹窗、报警或下一轮节奏。
- 不修改旧动作检测的 NG 证据保存行为，也不自动修改现有现场配置。

## Capabilities

### New Capabilities

- `roi-ng-annotated-evidence`: 约束 ROI NG 证据保存的启用条件、同帧绑定、绿色 OBB 绘制、单轮去重、文件结构和失败隔离行为。

### Modified Capabilities

无。

## Impact

- 主要影响 `yolo_action_detection/src/ui/main_window.py` 中 ROI 结果、裁剪、弹窗和证据保存链路，以及 `detection_logging/fail_evidence.py` 的可选 ROI 元数据。
- 复用现有 `NormalizedRect`、`FailEvidenceSaver`、`FailEvidenceContext` 和纯绿色 OBB 绘制函数，不新增第三方依赖或配置字段。
- `frame.jpg` 在 ROI NG 模式下变为同帧识别 ROI 裁剪图；现有目录和文件命名保持兼容，`metadata.json` 仅为 ROI NG 增加 `source_roi`。
