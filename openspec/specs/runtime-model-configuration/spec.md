# runtime-model-configuration Specification

## Purpose
TBD - created by archiving change reload-model-on-config-save. Update Purpose after archive.
## Requirements
### Requirement: Save configuration applies runtime model changes
系统 SHALL 在配置页保存成功后立即应用运行时配置。若模型路径或推理相关参数发生变化，系统 SHALL 重新创建并加载推理处理器，并将新处理器设置为相机推理线程使用的处理器。

#### Scenario: Model path changes and reload succeeds
- **WHEN** 操作员选择新的 `.pt` 或 `.onnx` 模型并保存配置
- **THEN** 系统重新加载新模型并替换运行中的推理处理器
- **AND** 后续推理使用新模型输出结果

#### Scenario: Inference parameters change and reload succeeds
- **WHEN** 操作员修改置信度、IoU、设备、tracker 或最大目标数等推理相关配置并保存
- **THEN** 系统使用新参数重新创建推理处理器
- **AND** 相机推理线程使用新处理器处理后续帧

### Requirement: Manual category configuration remains authoritative
系统 MUST 将操作员手动填写的步骤类别配置作为业务判断的唯一配置来源。系统 MUST NOT 从模型类别列表自动填充、覆盖或重排 `category_names`。

#### Scenario: New model exposes class names
- **WHEN** 新模型加载成功并可返回模型类别名称
- **THEN** 系统不修改操作员手动填写的步骤类别配置
- **AND** `category_names` 保持保存前表单中的手动输入值

#### Scenario: Manual steps differ from model class list
- **WHEN** 手动配置的步骤类别与模型类别列表不一致
- **THEN** 系统不自动改写步骤类别
- **AND** 业务判断继续按手动配置类别执行

### Requirement: Business judgement uses detection labels against manual steps
系统 SHALL 使用模型推理返回的检测类别名称与手动配置的当前步骤类别名称进行业务判断。匹配当前步骤时推进 PASS；匹配未来未完成步骤且错序规则启用时触发 FAIL；不属于配置步骤的类别不推进当前步骤。

#### Scenario: Detection label matches current manual step
- **WHEN** 当前等待步骤类别为 `2号`，模型推理返回的 `detection.label` 为 `2号`
- **THEN** 系统按现有 PASS 稳定帧规则推进当前步骤

#### Scenario: Detection label matches a future manual step
- **WHEN** 当前等待步骤类别为 `2号`，模型推理返回的 `detection.label` 为 `4号`
- **THEN** 系统按现有 NG 稳定帧和错序规则判定 FAIL

#### Scenario: Detection label is not configured
- **WHEN** 模型推理返回的 `detection.label` 不存在于手动配置步骤中
- **THEN** 系统不推进任何步骤
- **AND** 系统不自动新增或修改步骤类别

### Requirement: Failed model reload preserves previous processor
系统 SHALL 在新模型加载失败时保留当前可用推理处理器，并向操作员显示加载失败原因。系统 MUST NOT 在新模型失败时清空旧处理器导致检测链路无模型可用。

#### Scenario: Selected model file is invalid
- **WHEN** 操作员选择无法加载的模型文件并保存配置
- **THEN** 系统提示模型加载失败
- **AND** 运行中的推理处理器仍保持为保存前的可用处理器

### Requirement: Saved configuration persists immediately
系统 SHALL 在配置页保存成功后立即写入当前运行目录的配置文件，确保重启后继续使用最后一次成功保存的配置。

#### Scenario: Save configuration succeeds
- **WHEN** 操作员修改模型路径或手动步骤类别并保存配置
- **THEN** 系统立即写入当前配置文件
- **AND** 程序重启后加载该配置文件中的模型路径和手动步骤类别

