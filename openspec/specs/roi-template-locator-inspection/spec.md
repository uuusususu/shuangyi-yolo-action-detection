# roi-template-locator-inspection Specification

## Purpose
TBD - created by archiving change replace-yolo-locator-with-sift-template-matching. Update Purpose after archive.
## Requirements
### Requirement: ROI 定位检测必须支持 SIFT 模板匹配定位
系统 SHALL 在 ROI 定位检测模式下提供基于 SIFT 特征点的模板匹配定位方式。启用模板定位时，系统 MUST NOT 依赖 YOLO 模型识别定位类别，MUST NOT 使用定位类别名称或定位目标数量配置，并 SHALL 用 SIFT 模板匹配结果决定是否触发本轮。关闭模板定位时，系统 SHALL 退化为识别 ROI 内出现任意配置类别即触发本轮。

#### Scenario: 启用模板匹配定位
- **WHEN** 操作员启用 ROI 定位检测并启用模板匹配定位，且已创建定位模板
- **THEN** 系统每帧先执行 SIFT 模板匹配
- **AND** 匹配分数达到配置阈值且旋转角度在配置范围内时触发本轮
- **AND** 匹配成功后才执行 YOLO 推理

#### Scenario: 关闭模板匹配定位
- **WHEN** 操作员启用 ROI 定位检测但关闭模板匹配定位
- **THEN** 系统每帧执行 YOLO 推理
- **AND** 识别 ROI 内出现任意配置类别时触发本轮

#### Scenario: 模板匹配未命中时跳过 YOLO 推理
- **WHEN** 模板匹配定位已启用且当前帧匹配分数低于阈值或角度超出范围
- **THEN** 系统跳过 YOLO 推理
- **AND** 系统不触发本轮
- **AND** 系统继续等待下一帧的模板匹配

### Requirement: 操作员必须能从静态帧框选创建定位模板
系统 SHALL 在 ROI 配置页面提供"创建模板"和"保存模板"功能。操作员点击创建后在刷新后的静态帧上框选矩形区域；系统仅在操作员点击保存后截取、验证并保存模板图片，并 MUST 用该图片初始化 SIFT 特征点模板。模板图片 MUST 保存在实际配置文件所在目录，文件名固定为 `locator_template.png`。框选区域的归一化坐标 MUST 持久化，用于保存后显示蓝色轮廓和重新截取。

#### Scenario: 框选创建模板
- **WHEN** 操作员在 ROI 配置页面刷新静态帧后点击"创建定位模板"
- **THEN** 系统在静态帧上进入框选模式
- **AND** 操作员拖拽矩形时系统只显示蓝色选框，不显示实时裁剪预览
- **AND** 操作员点击"保存模板"后系统才截取该区域
- **AND** 系统验证临时图片可写、可重新读取且能够提取足够的 SIFT 特征
- **AND** 验证成功后系统将截取图片保存为 `locator_template.png`
- **AND** 系统用该图片创建 SIFT 特征点模板
- **AND** 系统保留蓝色轮廓并显示从已保存文件读取的模板静态预览
- **AND** 系统立即启用 ROI 检测和模板定位、持久化模板相关配置并重载运行处理器
- **AND** 操作员不需要再次点击"保存配置"

#### Scenario: 重新框选替换模板
- **WHEN** 已存在定位模板且操作员再次框选创建
- **THEN** 系统先在临时文件上验证新模板
- **AND** 验证成功后原子替换旧模板文件并重新创建运行处理器
- **AND** 验证失败时保留旧模板和旧运行处理器
- **AND** 系统明确提示新模板未生效

#### Scenario: 模板预览仅在保存后显示
- **WHEN** 操作员正在创建或调整模板 ROI 但尚未点击"保存模板"
- **THEN** 系统不显示裁剪模板预览
- **WHEN** 模板保存并验证成功
- **THEN** 系统显示已保存模板静态预览
- **AND** 主检测画面不显示模板四边形轮廓

#### Scenario: 模板图片丢失
- **WHEN** 模板定位已启用但模板图片文件不存在
- **THEN** 系统在配置校验时报错"定位模板文件不存在"
- **AND** 系统阻止开始检测

### Requirement: 模板路径、保存和运行时重载必须保持一致
系统 MUST 以实际 `config.json` 所在目录为基准解析相对模板路径。配置页保存、启动加载、保存后重载、配置校验和便携 smoke test MUST 使用同一个模板路径解析规则。模板保存 MUST 使用同目录临时文件验证后原子替换，MUST NOT 向运行线程暴露未完整写入的模板文件。

#### Scenario: 便携版保存并加载模板
- **WHEN** 便携版配置文件位于 EXE 同级且模板路径为 `locator_template.png`
- **THEN** 保存路径和运行时加载路径都必须是 EXE 同级的 `locator_template.png`

#### Scenario: 保存后立即重载成功
- **WHEN** 新模板通过图片读取和 SIFT 初始化验证
- **THEN** 系统保存模板相关配置并立即重建模板门控处理器
- **AND** 相机线程使用新处理器恢复检测

#### Scenario: 模板门控处理器不可用
- **WHEN** 模板定位已启用但模板缺失、不可读、特征不足或处理器创建失败
- **THEN** 系统 MUST NOT 静默退回普通 YOLO
- **AND** 系统显示包含失败原因和模板实际路径的错误
- **AND** 没有有效旧模板时系统阻止开始检测

### Requirement: 模板匹配参数必须可配置
系统 SHALL 提供以下模板匹配参数的配置项：匹配分数阈值（0 到 1 之间的浮点数，默认 0.6）、允许旋转角度范围（度数，默认 15.0）、最少匹配特征点数（正整数，默认 10）。这些参数 MUST 持久化到配置文件。

#### Scenario: 调整匹配分数阈值
- **WHEN** 操作员将匹配分数阈值设为 0.5 并保存
- **THEN** 模板匹配分数达到 0.5 及以上时判定为匹配成功

#### Scenario: 调整旋转角度范围
- **WHEN** 操作员将旋转角度范围设为 30 度并保存
- **THEN** 模板匹配估算的旋转角在 ±30 度范围内判定为匹配成功

#### Scenario: 调整最少匹配点数
- **WHEN** 操作员将最少匹配点数设为 5 并保存
- **THEN** good matches 数量达到 5 及以上时才尝试解算模板位置

### Requirement: 模板匹配成功后必须门控 YOLO 推理并触发识别阶段
系统 SHALL 用 `TemplateGatedProcessor` 包装现有 YOLO processor。每帧先执行模板匹配，匹配成功时 SHALL 调用内部 YOLO processor 执行推理并在 overlay 中附带模板匹配结果（matched=true、score、center、angle）；匹配失败时 SHALL 返回空检测结果 overlay（matched=false、score），MUST NOT 执行 YOLO 推理。

#### Scenario: 模板匹配成功触发 YOLO 推理
- **WHEN** 模板匹配分数达到阈值且角度在范围内
- **THEN** 系统执行 YOLO 推理
- **AND** overlay 携带 matched=true、score、center、angle
- **AND** ROI 引擎从 overlay 读取匹配结果触发本轮识别阶段

#### Scenario: 模板匹配失败跳过 YOLO 推理
- **WHEN** 模板匹配分数低于阈值或角度超出范围
- **THEN** 系统返回空检测 overlay
- **AND** overlay 携带 matched=false、score
- **AND** 系统 MUST NOT 执行 YOLO 推理
- **AND** ROI 引擎保持等待定位状态

### Requirement: ROI 引擎必须从 overlay 模板匹配结果判定触发
系统 SHALL 在 ROI 引擎的 WAITING 阶段从 overlay 携带的模板匹配结果判定是否触发本轮。模板定位启用时，matched=true 即触发；模板定位关闭时， SHALL 从 overlay 的 YOLO 检测结果中判断识别 ROI 内是否出现任意配置类别来触发。触发后进入 INSPECTING 阶段，识别 ROI 内类别数量判定、稳定帧锁存、PASS/FAIL、cooldown 续轮等行为 MUST 与现有定义保持一致。

#### Scenario: 模板定位触发本轮
- **WHEN** 引擎处于 WAITING 阶段且 overlay 携带 matched=true
- **THEN** 引擎切换到 INSPECTING 阶段
- **AND** 引擎使用同一帧的 YOLO 检测结果开始识别 ROI 内类别数量判定

#### Scenario: 模板未匹配保持等待
- **WHEN** 引擎处于 WAITING 阶段且 overlay 携带 matched=false
- **THEN** 引擎保持 WAITING 阶段
- **AND** 引擎不推进任何类别的 PASS 或 FAIL 帧

#### Scenario: 模板定位关闭时识别 ROI 内任意类别触发
- **WHEN** 模板定位关闭且引擎处于 WAITING 阶段且识别 ROI 内出现任意配置类别
- **THEN** 引擎切换到 INSPECTING 阶段

### Requirement: 主画面必须显示模板匹配结果
系统 SHALL 在模板匹配定位启用时，在主画面上绘制模板匹配位置的十字标记和匹配分数。系统 MUST NOT 绘制定位 ROI 框。系统 SHALL 继续绘制识别 ROI 框和各类别数量状态。

#### Scenario: 显示模板匹配位置和分数
- **WHEN** 模板匹配定位已启用且当前帧匹配成功
- **THEN** 主画面在模板匹配中心位置绘制十字标记
- **AND** 主画面显示匹配分数数值

#### Scenario: 匹配失败时画面显示
- **WHEN** 模板匹配定位已启用且当前帧匹配失败
- **THEN** 主画面不绘制十字标记
- **AND** 主画面显示当前匹配分数

### Requirement: 系统必须删除旧定位配置字段并升级配置版本
系统 MUST 删除 `roi_locator_class_name`、`roi_locator_target_count`、`roi_locator_rect` 配置字段。系统 MUST 新增 `roi_template_enabled`、`roi_template_path`、`roi_template_match_threshold`、`roi_template_angle_range`、`roi_template_min_matches`、`roi_template_rect` 配置字段。配置版本 MUST 从 2 升级到 3。旧版本配置 MUST 被拒绝开始检测并提示重新配置，但 MUST 允许打开配置页修复。

#### Scenario: 旧版本配置被拒绝
- **WHEN** 配置版本小于 3 且操作员尝试开始检测
- **THEN** 系统显示"配置版本无效，请重新配置"
- **AND** 系统阻止开始检测
- **AND** 系统允许打开配置页

#### Scenario: 旧字段不保留
- **WHEN** 旧配置包含 `roi_locator_class_name` 或 `roi_locator_rect` 字段
- **THEN** 系统在加载时忽略这些字段
- **AND** 保存后的新配置不包含这些字段

### Requirement: 配置校验必须覆盖模板匹配参数
系统 SHALL 在 ROI 定位检测启用且模板匹配启用时校验：模板图片文件存在且可读、能提取足够的 SIFT 特征、匹配分数阈值在 0 到 1 之间、角度范围为正数、最少匹配点数为正整数。识别 ROI 在 ROI 模式启用时 MUST 始终有效。校验失败时 MUST 列出具体无效项并阻止开始检测；创建新模板失败时 MUST 保留旧模板，不得显示保存成功。

#### Scenario: 模板未创建时校验失败
- **WHEN** 模板匹配已启用但未创建模板
- **THEN** 校验报错"定位模板文件不存在"
- **AND** 系统阻止保存和开始检测

#### Scenario: 阈值越界时校验失败
- **WHEN** 匹配分数阈值设为 1.5
- **THEN** 校验报错"匹配分数阈值必须在 0 到 1 之间"

#### Scenario: 有效模板配置通过校验
- **WHEN** 模板匹配已启用、模板图片存在、阈值 0.6、角度 15、最少匹配点 10、识别 ROI 有效
- **THEN** 校验通过
- **AND** 系统允许保存和开始检测

