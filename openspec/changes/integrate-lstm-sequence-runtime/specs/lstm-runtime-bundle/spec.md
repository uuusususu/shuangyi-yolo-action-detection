## ADDED Requirements

### Requirement: 部署模型包绑定权重契约与动作名称

系统 SHALL 从同一部署目录读取 `best_lstm.pt`、`contract.json` 和 `runtime_manifest.json`。清单 SHALL 包含版本 1、有序动作 ID/名称映射及契约和权重 SHA256。映射 SHALL 完整覆盖契约 `class_ids`，ID 和名称 MUST 唯一；`idle` MUST 保持空闲语义。系统 MUST NOT 通过配置步骤位置猜测模型输出名称。

#### Scenario: 内部 ID 映射到包装

- **WHEN** 模型输出内部 ID `c1790157821705`，清单将其映射为“包装”
- **THEN** 运行时以“包装”匹配业务步骤，并保持模型原始输出索引不变

#### Scenario: 文件被替换或映射缺失

- **WHEN** 文件指纹不匹配，或映射少于契约中的一个动作 ID
- **THEN** 模型包检查失败，系统显示具体原因并禁止以该模型启动动作判定

### Requirement: 已有模型补充部署元数据不需要重训

准备工具 SHALL 写入新的部署目录，原样复制权重和契约，并生成部署清单。没有绑定于该轮的名字元数据时，工具 MUST 要求显式指定经人工确认的 ID/名称映射，MUST NOT 自动按当前配置、日志排列或最新标注猜测。准备工具 MUST NOT 覆盖原 run 或修改训练工程。

#### Scenario: 老契约只有内部 ID

- **WHEN** 操作员选定旧连续 LSTM run 并提供完整、已确认的 ID/名称映射
- **THEN** 工具生成可校验的部署目录，复制后权重和契约 SHA256 与源文件一致，且不启动训练

#### Scenario: 未提供可确认的名称映射

- **WHEN** 源 run 没有动作名字元数据，操作员也未指定映射
- **THEN** 工具报告缺少映射并终止导出，不用配置类别的先后位置补齐

### Requirement: 动作模式严格校验连续 LSTM 契约

系统 SHALL 仅接受 `network: lstm`、`train_unit: timeline` 和受支持解码版本，校验网络结构、特征版本和长度、标准化向量长度及有限值、正时间基和最短段时长。系统 SHALL 校验实际 YOLO / 手部模型指纹、手部复杂度、零件类别顺序和 YOLO 阈值与契约一致。权重 SHALL 以仅权重方式加载，MUST NOT 通过任意对象反序列化加载模型。

#### Scenario: 加载 MS-TCN 或旧滑窗模型

- **WHEN** 所选模型是 MS-TCN 或未声明连续训练的旧 LSTM
- **THEN** 动作模式拒绝加载并显示不支持原因，不自动选择另一种推理算法

#### Scenario: 现场阈值或手部模型不同

- **WHEN** 当前 YOLO 阈值或实际手部文件指纹与契约不同
- **THEN** 动作模式校验失败，不能用不同特征启动生产判定

### Requirement: 每个动作必须携带有效零件要求

每个非空闲动作 SHALL 在 `required_parts` 中有非空集合，且集合成员 SHALL 属于契约的零件名单。系统 SHALL 原样使用该集合，MUST NOT 从现场当前画面或下一动作重新推导。

#### Scenario: 契约缺少包装的零件要求

- **WHEN** “包装”动作的 `required_parts` 缺失或为空
- **THEN** 动作模式拒绝启动，不能按纯 LSTM 直接 PASS

#### Scenario: 契约包含不存在的零件

- **WHEN** 某动作要求的零件名不在 `part_class_names` 中
- **THEN** 模型包检查显示动作和非法零件名称并失败
