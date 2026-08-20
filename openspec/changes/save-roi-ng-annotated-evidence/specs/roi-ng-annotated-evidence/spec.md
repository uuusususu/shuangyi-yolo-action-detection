## ADDED Requirements

### Requirement: ROI NG 证据保存必须服从现有开关和轮次去重
系统 MUST 仅在 ROI 数量检测产生新的最终 NG 结果且现有“保存 NG 证据”配置已启用时保存证据图片。每个完成的 NG 轮次 MUST 最多保存一次，系统 MUST NOT 为 NG 稳定帧或重复结果保存多张图片，也 MUST NOT 自动修改现场保存开关。

#### Scenario: 保存开关已启用并产生最终 NG
- **WHEN** ROI 数量检测完成一轮并首次产生最终 NG，且“保存 NG 证据”已启用
- **THEN** 系统为该轮调用一次证据保存并继续执行现有 NG 反馈链路

#### Scenario: 保存开关未启用
- **WHEN** ROI 数量检测产生最终 NG，但“保存 NG 证据”未启用
- **THEN** 系统不保存图片且不改变该配置值

#### Scenario: NG 稳定帧和重复结果到达
- **WHEN** 同一轮 NG 的稳定帧或相同 feedback key 再次到达界面
- **THEN** 系统不为该轮重复保存证据图片

#### Scenario: ROI 轮次通过
- **WHEN** ROI 数量检测产生 PASS
- **THEN** 系统不保存 NG 证据图片

### Requirement: 保存图片必须与 NG 判定和检测结果同帧
系统 MUST 使用产生当前 ROI NG 结果的 overlay 事件中的 `source_frame`、`source_frame_id` 和 detections 构造证据图片，并 MUST 校验结果帧编号与事件帧编号一致。系统 MUST NOT 从 worker 最新显示帧、最新预览帧或历史全局 overlay 中取得替代图片或 detections。

#### Scenario: worker 最新帧已经推进
- **WHEN** 第 42 帧产生 ROI NG，而 worker 最新显示帧已经推进到第 43 帧或更大编号
- **THEN** 保存图片和 metadata detections 均来自第 42 帧的同一 overlay 事件

#### Scenario: 结果帧编号与事件帧编号不一致
- **WHEN** ROI NG 结果帧编号与传入 overlay 事件帧编号不一致
- **THEN** 系统不保存图片、不回退使用其他帧，并记录包含两个帧编号的错误日志

#### Scenario: 同帧图像缺失
- **WHEN** 当前 overlay 事件没有有效的非空 `source_frame`
- **THEN** 系统不保存图片、不回退使用其他帧，并记录同帧证据缺失日志

### Requirement: ROI NG 弹窗和证据图片必须严格裁剪为识别 ROI
系统 MUST 严格按当前识别 ROI 裁剪最终 NG 弹窗和保存的 `frame.jpg`，MUST NOT 增加边距或回退整图。正常主页面 MUST 继续显示整张相机画面和现有 ROI 框，弹窗 MUST 保持裁剪图宽高比并自适应现有可用区域。

#### Scenario: 最终 NG 产生有效识别 ROI 裁剪图
- **WHEN** 当前识别 ROI 在 NG 同帧原图内具有有效正面积
- **THEN** 弹窗和 `frame.jpg` 只包含该识别 ROI 的像素区域且尺寸等于裁剪尺寸

#### Scenario: 正常检测继续显示整图
- **WHEN** 系统尚未产生最终 NG 或 NG 弹窗已关闭
- **THEN** 主页面画面保持现有整图显示行为

#### Scenario: 识别 ROI 无效
- **WHEN** 识别 ROI 配置无效、越界或计算后的裁剪区域为空
- **THEN** 系统不创建图片弹窗、不保存证据、不回退整图并记录错误，同时继续 NG 统计、声音反馈和轮次控制

### Requirement: ROI NG 裁剪图必须只绘制 ROI 内绿色 YOLO OBB
系统 MUST 仅保留中心点位于识别 ROI 内的 detections，并使用与 ROI 数量判定相同的中心点规则。每个有效 polygon MUST 平移为裁剪图坐标并使用绿色闭合旋转轮廓；图片 MUST NOT 绘制类别名、置信度、跟踪 ID、ROI、数量摘要、操作提示或其他文字。

#### Scenario: ROI 内外均有检测目标
- **WHEN** 触发 NG 的同帧 detections 同时包含识别 ROI 内和识别 ROI 外的有效 OBB polygon
- **THEN** 弹窗、保存图片和 metadata 只包含中心点位于识别 ROI 内的目标

#### Scenario: ROI 内目标的旋转框越过边界
- **WHEN** detection 中心点位于识别 ROI 内但 polygon 一部分越过 ROI 边界
- **THEN** 系统保留该目标并在裁剪图边界自然裁掉越界框线

#### Scenario: detection 包含附加字段
- **WHEN** detection 同时包含 polygon、类别、置信度和跟踪 ID
- **THEN** 保存图片只绘制绿色 polygon，不绘制任何附加字段文字

#### Scenario: 部分 polygon 无效
- **WHEN** 当前帧包含少于 4 个点的无效 polygon 和其他有效 polygon
- **THEN** 系统跳过无效 polygon 并继续绘制其余全部有效 polygon

#### Scenario: 识别 ROI 内没有 detections
- **WHEN** ROI 数量检测产生 NG 且识别 ROI 内 detections 为空
- **THEN** 系统仍弹出并保存该 NG 同帧 ROI 裁剪画面，但不生成虚拟框、理论缺失位置或其他标记

### Requirement: ROI NG 证据文件结构必须保持兼容
系统 MUST 继续使用现有日期、时间、轮次和帧号事件目录，并 MUST 在每个 ROI NG 事件目录中只保存一个 ROI 裁剪 `frame.jpg` 和 `metadata.json`。系统 MUST NOT 额外保存整图；metadata MUST 记录同一 overlay 事件的帧编号、模型路径、时间戳、仅 ROI 内 detections 和 `source_roi`。polygon、box、center/center_px MUST 转换为裁剪图局部坐标；`source_roi` MUST 记录裁剪区在原图中的 x、y、width、height 和原图尺寸。

#### Scenario: ROI NG 证据保存成功
- **WHEN** 同帧带框画面成功写入证据目录
- **THEN** 事件目录包含 ROI 裁剪 `frame.jpg` 和 `metadata.json`，且 metadata 的帧编号、局部 detections 和 `source_roi` 与图片一致

#### Scenario: 检查保存目录
- **WHEN** 员工按日期、轮次或帧号查找 NG 证据
- **THEN** 现有目录命名结构保持不变且无需查找第二张图片

### Requirement: ROI NG 证据失败不得阻塞生产检测
系统 MUST 将同帧校验失败和文件写入失败记录到运行日志，但 MUST 继续完成当前 NG 的统计、主页面反馈、弹窗、循环报警和轮次控制。旧动作检测的 NG 证据保存行为 MUST 保持不变。

#### Scenario: JPEG 或目录写入失败
- **WHEN** `FailEvidenceSaver` 无法创建目录或写入 `frame.jpg`、`metadata.json`
- **THEN** 系统记录包含轮次和帧编号的保存失败日志，并继续执行当前 ROI NG 反馈和后续检测

#### Scenario: 旧动作检测产生 NG
- **WHEN** 非 ROI 的旧动作检测产生 NG
- **THEN** 系统继续使用原有动作 NG 证据保存逻辑，不应用本规格的 ROI 同帧带框保存改动
