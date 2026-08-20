# roi-ng-visual-alert Specification

## Purpose
TBD - created by archiving change show-yolo-boxes-in-roi-ng-alert. Update Purpose after archive.
## Requirements
### Requirement: NG 告警必须使用判定同帧证据
系统 MUST 在 ROI 轮次产生 NG 时冻结触发该结果的推理帧，并 MUST 使用与该帧相同 `source_frame_id` 的 YOLO 检测结果生成告警图像。系统 MUST NOT 使用弹窗打开时刻的最新预览帧或其他帧替代该 NG 帧。

#### Scenario: NG 结果产生前预览已推进
- **WHEN** ROI 引擎使用 `source_frame_id=42` 的推理结果产生 NG，且相机最新预览已经推进到更大的帧编号
- **THEN** 告警弹窗显示第 42 帧及第 42 帧的检测框，不显示较新的预览帧

#### Scenario: 后续推理在界面处理前完成
- **WHEN** 第 42 帧的 overlay 事件仍在界面队列中，而 worker 已完成第 43 帧推理
- **THEN** 第 42 帧产生的 NG 告警仍使用随第 42 帧事件传递的图像和 detections

#### Scenario: 同帧图像不可用
- **WHEN** NG 事件不包含有效的同帧图像
- **THEN** 系统不得把其他帧与当前 detections 组合，并记录可诊断的同帧证据缺失错误

### Requirement: 告警图像必须绘制全部 YOLO 旋转框
系统 MUST 在冻结帧上绘制该推理事件中全部有效 YOLO OBB polygon，无论目标是否位于识别 ROI 内。每个检测 MUST 使用绿色闭合旋转四边形轮廓，且 MUST NOT绘制类别名、置信度、跟踪 ID 或其他检测文字。

#### Scenario: 检测目标同时位于识别 ROI 内外
- **WHEN** 触发 NG 的推理帧包含识别 ROI 内和识别 ROI 外的有效 YOLO detections
- **THEN** 告警图像为两组 detections 全部绘制绿色旋转轮廓

#### Scenario: 检测结果带有类别与置信度
- **WHEN** detection 数据同时包含 polygon、label、confidence 和 track ID
- **THEN** 告警图像只绘制 polygon 轮廓，不绘制任何 label、confidence 或 track ID 文字

#### Scenario: 单个检测 polygon 无效
- **WHEN** 同一帧中一个 detection 的 polygon 少于 4 个点而其他 detections 有效
- **THEN** 系统跳过无效 polygon，并继续绘制该帧其余全部有效旋转框

#### Scenario: 当前帧没有检测结果
- **WHEN** ROI 数量判定产生 NG 且该帧 detections 为空
- **THEN** 弹窗显示无检测框的同一 NG 冻结帧，不生成虚拟框或位置标记

### Requirement: 告警弹窗内容必须仅显示图像
ROI NG 告警弹窗的 Qt 内容区 MUST 仅包含带检测框的冻结画面。系统 MUST NOT 在内容区显示数量摘要、类别缺失文字、置信度、操作提示、ROI 框或其他说明文字。

#### Scenario: 首次 NG 打开弹窗
- **WHEN** 一轮 ROI 检测首次产生 NG
- **THEN** 弹窗内容区只显示该轮带绿色检测框的冻结画面，不创建摘要滚动区或提示标签

#### Scenario: 弹窗打开期间发生新一轮 NG
- **WHEN** 告警弹窗仍打开且后续轮次产生新的 NG
- **THEN** 系统使用新一轮同帧带框图替换当前图片，不追加任何文字记录

### Requirement: 现有告警生命周期必须保持不变
系统 MUST 保持现有 ROI NG 循环报警、弹窗复用和回车关闭行为。首次打开 NG 弹窗时 MUST 启动一次循环报警；弹窗已打开时刷新新 NG 图片 MUST NOT 重启循环报警；按回车关闭弹窗时 MUST 停止循环报警。

#### Scenario: 首次 NG 启动报警
- **WHEN** 当前没有打开的 ROI NG 弹窗且一轮检测产生 NG
- **THEN** 系统打开纯图像弹窗并启动一次循环报警

#### Scenario: 已打开弹窗刷新图片
- **WHEN** ROI NG 弹窗已打开且下一轮再次产生 NG
- **THEN** 系统替换告警图片并保持原循环报警，不重复启动声音

#### Scenario: 员工按回车确认
- **WHEN** 员工在 ROI NG 告警弹窗中按下回车键
- **THEN** 系统关闭弹窗、清空当前图片并停止循环报警

#### Scenario: ROI 轮次通过
- **WHEN** ROI 轮次产生 PASS
- **THEN** 系统不打开 NG 告警弹窗，也不启动 NG 循环报警

