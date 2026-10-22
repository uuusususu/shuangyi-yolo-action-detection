## ADDED Requirements

### Requirement: 仅对特定推理后端有效的参数不得在其他后端下可编辑

系统 MUST 依据当前模型格式决定推理参数在配置页是否可编辑。仅在 Ultralytics 路径生效的参数（推理设备、半精度、tracker、跟踪持久化）在使用 ONNX 模型时 MUST 不可编辑，并 MUST 给出不可编辑的原因。当检测结果不含跟踪信息时，跟踪 ID 叠加开关 MUST 同样不可编辑。系统 MUST 保留这些参数已保存的值，MUST NOT 因不可编辑而清空，以便切换回对应后端时恢复原设置。

#### Scenario: 使用 ONNX 模型

- **WHEN** 当前模型为 `.onnx`
- **THEN** 推理设备、半精度、tracker 与跟踪持久化在配置页不可编辑并显示原因
- **AND** 这些参数已保存的值保持不变

#### Scenario: 切换回 Ultralytics 模型

- **WHEN** 操作员把模型改回 `.pt`
- **THEN** 这些参数恢复可编辑
- **AND** 显示的仍是此前保存的值

#### Scenario: 检测结果不含跟踪信息

- **WHEN** 当前推理后端产出的检测结果不含跟踪 ID
- **THEN** 跟踪 ID 叠加开关不可编辑

## MODIFIED Requirements

### Requirement: Save configuration applies runtime model changes

系统 SHALL 在配置页保存成功后立即应用运行时配置。若模型路径或推理相关参数发生变化，系统 SHALL 重新创建并加载推理处理器，并将新处理器设置为相机推理线程使用的处理器。推理相关参数 SHALL 包含 ONNX intra-op 线程数与 ONNX 图优化级别。系统 MUST 在创建 ONNX 推理会话时应用这两项配置，MUST NOT 忽略配置而固定使用默认会话选项。

#### Scenario: Model path changes and reload succeeds

- **WHEN** 操作员选择新的 `.pt` 或 `.onnx` 模型并保存配置
- **THEN** 系统重新加载新模型并替换运行中的推理处理器
- **AND** 后续推理使用新模型输出结果

#### Scenario: Inference parameters change and reload succeeds

- **WHEN** 操作员修改置信度、IoU、设备、tracker 或最大目标数等推理相关配置并保存
- **THEN** 系统使用新参数重新创建推理处理器
- **AND** 相机推理线程使用新处理器处理后续帧

#### Scenario: ONNX session options change and reload succeeds

- **WHEN** 操作员修改 ONNX intra-op 线程数或图优化级别并保存
- **THEN** 系统使用新的会话选项重新创建 ONNX 推理会话
- **AND** 相机推理线程使用新处理器处理后续帧

#### Scenario: ONNX threading left at automatic

- **WHEN** ONNX intra-op 线程数配置为 `0`
- **THEN** 系统不显式设置线程数，交由 ONNX 运行时自行决定
