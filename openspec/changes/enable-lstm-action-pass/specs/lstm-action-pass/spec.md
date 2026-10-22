## ADDED Requirements

### Requirement: 时序动作默认关闭

工位配置 SHALL 提供 `lstm_action_enabled`，缺省为关闭，并 SHALL 提供 `lstm_contract_path`，缺省为空。关闭或路径为空时，系统 MUST NOT 加载 LSTM，MUST NOT 产生时序 Pass。顺序检测和零件检测 SHALL 保持各自现有判定。

#### Scenario: 未勾选使能

- **WHEN** `lstm_action_enabled` 为关闭
- **THEN** 运行中不加载 `best_lstm.pt`，且顺序检测仍按单帧类别数量更新

#### Scenario: 旧配置没有新键

- **WHEN** 已有配置文件不含 `lstm_action_enabled` 和 `lstm_contract_path`
- **THEN** 两者按关闭和空路径处理，且不要求提高配置 schema 版本

### Requirement: 类别和置信度沿用现有配置

时序 Pass SHALL 只使用 `category_names` 中非空类别，并 SHALL 使用 `yolo_conf_threshold` 作为概率下限。系统 MUST NOT 为时序动作另存一套类别名单或置信度。一次合格输出 SHALL 立即 Pass，MUST NOT 等待 `action_pass_stable_frames`。

#### Scenario: 输出类别不在已配置类别中

- **WHEN** 使能打开且 LSTM 最高概率类别不是任何一个非空 `category_names`
- **THEN** 该帧不产生时序 Pass

#### Scenario: 概率低于现有置信度

- **WHEN** 使能打开且最高类别属于已配置类别，但概率低于 `yolo_conf_threshold`
- **THEN** 该帧不产生时序 Pass，且命中计数清零

### Requirement: 只有契约匹配时才判过

系统 SHALL 从 `lstm_contract_path` 读取 `contract.json`，并从同一目录加载 `best_lstm.pt`。动作类别顺序必须为 `idle` 加非空 `category_names`。特征长度、网络结构和契约中的 YOLO 文件指纹任一不一致时，系统 MUST NOT 用该模型判过，并 MUST 继续运行顺序检测和零件检测。

#### Scenario: 类别顺序不一致

- **WHEN** 使能打开且契约类别顺序与 `idle` 加当前非空类别名不一致
- **THEN** 时序状态为未加载，且本帧顺序检测仍执行更新

#### Scenario: 契约匹配后连续命中

- **WHEN** 使能打开、契约匹配、队列已满 48 帧，且某已配置类别的概率不低于 `yolo_conf_threshold`
- **THEN** 该类别在这一次输出上记为 Pass，且顺序检测的步骤状态不被这次 Pass 修改

### Requirement: 特征与训练台共用

运行中的特征行 SHALL 调用 `lstm_workbench.features.build_feature_row`，使用同一帧的检测框和手部关键点。框的置信度过滤 SHALL 使用 `yolo_conf_threshold`。窗口长度 SHALL 为契约中的 48 帧。

#### Scenario: 使能关闭时不抽时序特征

- **WHEN** `lstm_action_enabled` 为关闭
- **THEN** 该帧不为时序模型追加特征行

#### Scenario: 手部缺失

- **WHEN** 使能打开且该帧没有手部关键点
- **THEN** 手部槽位按全 0 写入特征行，且顺序检测不受影响
