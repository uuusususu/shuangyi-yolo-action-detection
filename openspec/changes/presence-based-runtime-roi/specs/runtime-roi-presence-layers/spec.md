## ADDED Requirements

### Requirement: Detection assembles optional ROI layers from presence

系统 SHALL 在开始检测以及验证并保存后的处理器重建时，按已保存配置组装可选层。有效识别 ROI MUST 在判定前过滤目标中心。模板开关打开、搜索 ROI 有效、且定位模板可加载时，系统 MUST 挂上模板门控。任一条件缺失时，系统 MUST 跳过对应层，且 MUST NOT 因该层缺失拒绝开始检测。`roi_inspection_enabled` MUST 只选择顺序认法或不按顺序认法，MUST NOT 阻止已保存框被加载。

#### Scenario: Sequential mode uses a saved inspection ROI

- **WHEN** `roi_inspection_enabled` 为 false
- **AND** 已保存有效识别 ROI
- **AND** 一帧里同时有中心在框内和框外的待检目标
- **THEN** 步骤引擎 MUST 只收到中心在框内的目标
- **AND** overlay 仍可保留框外目标供画框

#### Scenario: Empty inspection ROI does not block sequential start

- **WHEN** `roi_inspection_enabled` 为 false
- **AND** 识别 ROI 为空或无法通过归一化矩形校验
- **THEN** 系统 MUST 允许开始检测
- **AND** 步骤引擎 MUST 使用该帧全部待检目标

#### Scenario: Sequential mode can load template gate

- **WHEN** `roi_inspection_enabled` 为 false
- **AND** `roi_template_enabled` 为 true
- **AND** 搜索 ROI 有效且定位模板可加载
- **THEN** 运行处理器 MUST 是 `TemplateGatedProcessor`

#### Scenario: Incomplete template layer is skipped

- **WHEN** 模板开关打开
- **AND** 仅有搜索 ROI、定位模板文件不存在
- **THEN** 系统 MUST 使用未门控的 YOLO 处理器开始检测
- **AND** MUST NOT 提示「搜索 ROI 未配置」或「识别 ROI 未配置」

#### Scenario: Corrupt template file blocks start

- **WHEN** 模板开关打开
- **AND** 搜索 ROI 有效
- **AND** 定位模板文件存在但加载失败
- **THEN** 系统 MUST 拒绝开始检测
- **AND** 错误文案 MUST 说明是定位模板不可用

### Requirement: Both recognition styles count configured quantities

系统 SHALL 在两种认法下都按 `category_counts` 判定数量。顺序认法 MUST 逐步要求当前类别数量刚好等于该步配置。不按顺序认法 MUST 在同一轮对全部已配类别同时按各自配置数量精确计数。系统 MUST NOT 把认法开关解释成「只有一种认法才计数」。

#### Scenario: Sequential still requires exact step counts

- **WHEN** 顺序认法正在等待某一步
- **AND** 过滤后该类别数量刚好等于该步配置并达到通过稳定帧
- **THEN** 该步 MUST PASS 并进入下一步

#### Scenario: Unordered mode counts all categories after optional locate

- **WHEN** 不按顺序认法已进入本轮检查
- **THEN** 每个已配类别的实际数量 MUST 刚好等于其 `category_counts` 才算该类通过

### Requirement: Unordered mode without a live gate must not wait for locate

当模板门控未实际挂上时，不按顺序认法 MUST 按无模板触发：识别 ROI（或整图）内出现待检类别即开轮。系统 MUST NOT 在门控未挂上时等待 `template_match`。

#### Scenario: Template switch on but layer not assembled

- **WHEN** `roi_inspection_enabled` 为 true
- **AND** `roi_template_enabled` 为 true
- **AND** 搜索 ROI 或定位模板缺失导致未挂门控
- **AND** 识别 ROI 或整图内出现待检类别
- **THEN** 系统 MUST 开轮并按配置数量计数
- **AND** MUST NOT 停在等待定位

#### Scenario: Empty inspection ROI counts the full frame in unordered mode

- **WHEN** `roi_inspection_enabled` 为 true
- **AND** 识别 ROI 为空或非法
- **THEN** 系统 MUST 允许开始检测
- **AND** 本轮 MUST 按整图待检目标计数
