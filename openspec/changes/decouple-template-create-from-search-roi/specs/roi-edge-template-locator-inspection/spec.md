## ADDED Requirements

### Requirement: Uncreated ROI boxes stay empty
The configuration editor SHALL show no ROI rectangle for a kind until the operator creates that kind on the current static frame. The system MUST NOT persist a rectangle for a kind until the operator clicks save-current-ROI for that kind.

#### Scenario: Fresh page has no boxes
- **WHEN** the operator opens ROI configuration and the saved template, search, and inspection rectangles are empty
- **THEN** the editor SHALL show no ROI boxes

#### Scenario: Create does not persist
- **WHEN** the operator clicks create-or-replace for one ROI kind
- **THEN** that kind MAY appear as an unsaved draft
- **AND** `ConfigManager` SHALL keep the last saved rectangle for that kind, including `None`

#### Scenario: Only explicit ROI save writes the box
- **WHEN** the operator has an unsaved draft and clicks save-current-ROI
- **THEN** only that kind SHALL be written
- **AND** the other two kinds SHALL remain unchanged, including remaining absent

## MODIFIED Requirements

### Requirement: 配置页必须统一创建三种 ROI
系统 SHALL 在 ROI 配置页提供 ROI 类型下拉框，选项为“识别 ROI”“模板 ROI”“搜索 ROI”，并提供统一的创建或替换命令以及位于编辑器下方的“保存当前 ROI”按钮。三种 ROI MUST 独立创建和保存。系统 MUST NOT 要求搜索 ROI 包含模板 ROI，MUST NOT 要求先创建搜索 ROI 才能保存模板，也 MUST NOT 在保存当前 ROI 时覆盖另外两种未提交草稿。

#### Scenario: 创建搜索 ROI
- **WHEN** 操作员在下拉框选择“搜索 ROI”并执行创建命令
- **THEN** 操作员可以在当前静态原图上拖拽搜索区域
- **AND** 点击“保存当前 ROI”后只将该区域以归一化坐标保存在配置中

#### Scenario: 三种 ROI 没有包含关系
- **WHEN** 搜索 ROI 与模板 ROI 分离或部分重叠
- **THEN** 系统仍允许分别保存两种 ROI
- **AND** 运行时只使用搜索 ROI 限定查找范围

#### Scenario: 单独保存所选 ROI
- **WHEN** 操作员已同时编辑多种 ROI 草稿
- **AND** 下拉框当前选择其中一种 ROI
- **THEN** “保存当前 ROI”只持久化所选 ROI
- **AND** 另外两种 ROI 的已保存配置不得被草稿覆盖

### Requirement: 保存模板后必须在 ROI 原图显示轮廓和创建得分
系统 SHALL 在编辑阶段显示 ROI 草稿。系统 MUST 在操作员选择模板 ROI 并点击“保存当前 ROI”后才创建并验证正式模板；验证成功后 MUST 在当前 ROI 编辑原图上显示绿色真实边缘轮廓、绿色外框和创建得分，并 MUST NOT 显示独立模板裁剪预览。验证 MUST 只使用模板 ROI 自身范围（必要时向四周扩展到算法所需最小窗，或退回整张静态图），MUST NOT 读取搜索 ROI 草稿或已保存搜索 ROI。

#### Scenario: 保存并回查成功
- **WHEN** 模板 ROI 有效
- **AND** 搜索 ROI 和识别 ROI 尚未配置或位于其他位置
- **AND** 临时边缘模板在当前静态原图的模板 ROI 自身范围回查得分达到阈值
- **THEN** 系统原子保存模板文件和配置
- **AND** 系统显示绿色真实轮廓、绿色外框和创建得分
- **AND** 系统不得因搜索 ROI、识别 ROI、模型路径或运行处理器尚未完成而将模板保存判为失败

#### Scenario: 没有搜索 ROI 也可以保存模板
- **WHEN** 操作员只框选并保存模板 ROI
- **AND** 搜索 ROI 未创建
- **THEN** 系统仍创建并保存模板文件和 `roi_template_rect`
- **AND** `roi_template_search_rect` SHALL remain unset

#### Scenario: 保存验证失败
- **WHEN** 模板轮廓质量不足或回查得分未达到阈值
- **THEN** 系统显示红色失败结果和具体原因
- **AND** 系统不得覆盖旧模板、旧配置或旧处理器

#### Scenario: 不提供持续测试模式
- **WHEN** 模板保存成功
- **THEN** 系统只显示本次静态原图回查的创建得分
- **AND** 系统不增加“测试模板”按钮或持续实时测试
