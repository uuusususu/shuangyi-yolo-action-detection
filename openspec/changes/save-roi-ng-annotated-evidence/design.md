## Context

ROI 数量检测在 `_handle_roi_results()` 中对每个最终结果按 `feedback_key` 去重。当前 NG 分支先调用 `_save_roi_fail_evidence(result, frame_id)`，该函数通过 `_get_feedback_frame()` 读取 worker 的可变最新帧，再把 `_last_overlay_data` 中的 detections 写入元数据；因此保存的 `frame.jpg` 没有检测框，且在界面队列积压时可能与本轮 NG 的 `source_frame_id` 不一致。

同一分支随后调用 `_show_roi_fail_alert(result, overlay_data)`。弹窗与保存目前共用 `_prepare_roi_ng_annotated_frame()`，会校验同帧后绘制整图全部绿色 YOLO OBB。现场确认最终 NG 只需要展示配置的识别 ROI，因此共用入口需要进一步负责严格裁剪、目标筛选和坐标平移，同时保持主页面整图、报警和轮次控制不变。

## Goals / Non-Goals

**Goals:**

- 当且仅当 ROI 数量检测产生一次最终 NG 且保存开关启用时，保存一张与该 NG 推理结果同帧的识别 ROI 裁剪图。
- 让保存图与 NG 弹窗使用相同的帧校验、严格 ROI 裁剪、中心点筛选和纯绿色 OBB 绘制规则。
- 把保存 detections 转为裁剪图局部坐标，并在 metadata 中记录原图 `source_roi`。
- 保持一轮一次保存和现有目录命名；对同帧缺失、帧编号不一致、无效 ROI 和磁盘写入失败提供可诊断日志，并保持检测流程继续运行。

**Non-Goals:**

- 不保存第二张无框原图，不增加图片格式、JPEG 质量或保存目录配置。
- 不改变旧动作检测的 NG 证据保存路径。
- 不改变 YOLO 推理、ROI 判定、稳定帧、完成间隔、统计、弹窗、声音或证据开关默认值。
- 不改变正常主页面的整图显示和现有 ROI 框。
- 不在保存图中增加类别、置信度、跟踪 ID、ROI、数量摘要或理论缺失位置，也不为裁剪增加边距。

## Decisions

### 1. ROI 证据保存显式消费当前 overlay 事件

将 ROI NG 保存接口从只接收 `frame_id` 调整为接收当前 `overlay_data`。保存逻辑从 payload 读取 `source_frame_id`、`source_frame`、detections、模型路径和时间戳，并用 `result.source_frame_id` 交叉校验。

不再调用 `_get_feedback_frame()` 或依赖 `_last_overlay_data`。这避免 worker 在 UI 处理前推进到下一帧时保存错产品，也使图片、检测框和元数据共享同一事件来源。

### 2. 保存与弹窗共用同帧 ROI 裁剪构图入口

提取或复用一个内部构图入口，负责：

1. 校验结果帧编号与事件帧编号一致；
2. 校验 `source_frame` 是非空 NumPy 图像；
3. 从 `config.roi_inspection_rect` 创建 `NormalizedRect`，按原图尺寸得到严格裁剪边界并拒绝无效或空 ROI；
4. 使用与 ROI 数量判定相同的中心点规则筛选 detections；
5. 深复制并把 polygon、box、center/center_px 平移为裁剪图坐标；
6. 裁剪原图后调用现有 `_draw_roi_ng_detection_outlines()`，返回带框 ROI 图、局部 detections 和 `source_roi`。

保存和弹窗都使用该入口，确保二者不会出现不同的帧选择、裁剪范围或覆盖层规则。构图始终复制裁剪区和 detections，不修改 payload、相机缓冲或后续证据数据。中心位于 ROI 内但 polygon 越界的目标仍保留，OpenCV 在裁剪图边界自然截断框线。

### 3. 继续让结果去重控制保存次数

保存调用保留在 `_remember_result_key()` 判定为新最终结果之后。因此 NG 稳定帧、重复 overlay 或已处理的 feedback key 不会重复落盘；每个最终 NG 轮次最多调用一次保存。

不在 `FailEvidenceSaver` 中增加第二套轮次去重状态，避免生命周期和清理策略重复。

### 4. `FailEvidenceSaver` 保持通用写入职责

ROI 路径把已经绘制完成的 BGR ROI 帧、局部 detections 和原图 `source_roi` 传给 `save_fail_event()`。写入器继续生成现有日期/事件目录、单个 `frame.jpg` 和 `metadata.json`，不感知裁剪或 OBB 绘制规则。

`FailEvidenceContext` 增加可选 `source_roi`；仅 ROI NG 填充并写入 metadata，旧动作 NG 不出现该字段。识别 ROI 内 detections 为空时，构图入口仍返回同帧 ROI 副本，写入器保存无框裁剪图。

### 5. 所有证据错误均为非阻塞错误

帧编号不一致、同帧图像缺失或识别 ROI 无效时不调用写入器、不回退整图，并记录包含 result/event frame ID 或 ROI 原因的运行日志。`save_fail_event()` 返回 `None` 时记录保存失败日志，但不抛出到 ROI 结果主链路。

统计、NG 标签和下一轮控制继续执行。识别 ROI 无效而无法创建图片弹窗时只播放一次 FAIL 音，避免无窗口可关闭却持续循环；正常 ROI NG 仍使用现有弹窗循环报警。旧动作检测继续使用原 `_save_fail_evidence()`。

## Risks / Trade-offs

- [裁剪图失去 ROI 外上下文] -> 这是现场明确选择；主页面仍显示整图，metadata 的 `source_roi` 能定位裁剪区在原图中的位置。
- [平移后越界 polygon 会包含负数或超过裁剪尺寸的坐标] -> 保留旋转框原始几何并让图像边界自然裁线，避免把 OBB 错改为不规则裁剪多边形；目标中心始终位于裁剪区内。
- [保存与弹窗分别构图可能产生重复图像复制] -> 只在最终 NG 时发生，不在每个推理帧发生；优先保证两个消费者互不修改同一数组。
- [JPEG 压缩会让绿色边缘出现轻微色差] -> 沿用现有 JPEG 写入方式，不把像素精确值作为落盘验收条件；自动化测试在编码前验证绘制颜色和内容。
- [写入器吞掉底层异常只能得到通用失败原因] -> 调用方至少记录事件帧号、轮次和返回失败；不为本次小改动扩大写入器异常 API。

## Migration Plan

1. 先把 ROI NG 测试改为识别 ROI 严格裁剪，覆盖 ROI 内外目标、局部坐标、越界框、空 detections、无效 ROI、帧错配、开关和一轮一次。
2. 调整共用构图入口和保存上下文，加入 ROI 裁剪、筛选、坐标平移与可选 `source_roi`；保持旧动作 NG 方法不变。
3. 运行相关测试、完整测试、编译、diff 和 OpenSpec 严格校验。
4. 构建便携包并确认现有配置、模型和模板哈希不变；使用临时启用开关的测试配置验证落盘 `frame.jpg`，不得覆盖现场配置。

回滚时恢复 ROI 保存接口和调用点即可；没有配置 schema、目录迁移或持久化数据回写。

## Open Questions

无。保存范围、图片数量、框范围、异常策略、频率、开关、目录、空检测和非阻塞行为均已确认。
