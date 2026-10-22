## ADDED Requirements

### Requirement: One decode function turns a clip into segments
契约为 `train_unit: "timeline"` 的模型 SHALL 只有一个切段函数。给它一条片子的原始特征行和帧率，它 SHALL 依次：换算到契约的时间基；按 `feature_mean`、`feature_std` 标准化；整条片子一次因果前向；每一帧取概率最高的类别；连续同类收成一段；短于 `min_action_seconds` 的段并进前一段（开头那段并进后一段）；相邻同类再合成一段；帧号换回片子自己的帧号；最后过零件门禁。每轮验证、结果页验证、标注页预标注 SHALL 都调用这个函数。

#### Scenario: Same input gives the same segments everywhere
- **WHEN** 同一条片子的特征分别由最好那一轮的每轮验证、结果页验证、预标注来切段，用的是同一轮模型
- **THEN** 三处切出的段边界和类别完全一样

#### Scenario: A short flicker is absorbed
- **WHEN** 逐帧类别是包装 3 秒、封装 0.5 秒、包装 4 秒，`min_action_seconds` 是 1.92
- **THEN** 切段结果是一段包装，覆盖整整 7.5 秒

#### Scenario: Frames map back to the source clip
- **WHEN** 一条 10 fps 的片子按 5 fps 时间基切段
- **THEN** 返回的帧号和秒数都是这条片子自己 10 fps 下的帧号和秒数

### Requirement: The contract fully describes how to decode
新训练的契约 SHALL 写全：`train_unit: "timeline"`、`decode_version`、`network`、`context_frames`、`time_base_fps`、`min_action_seconds`、`feature_mean`、`feature_std`、`class_ids`、`part_class_names`、`required_parts`、`input_size`、YOLO 和手部模型指纹、置信度阈值；网络是 MS-TCN 时还有它的尺寸字段。缺任何一项，或和现在的模型对不上，切段函数 SHALL 拒绝运行，并写明哪一项不对。

#### Scenario: Missing normalization
- **WHEN** 契约是 `train_unit: "timeline"`，但没有 `feature_std`
- **THEN** 拿它验证或预标注都会失败，提示写明 `feature_std`

#### Scenario: Different YOLO file
- **WHEN** 现在的 YOLO 文件指纹和契约里记的不一样
- **THEN** 验证被拒，提示「现在的零件模型和这一轮不是同一个文件」

### Requirement: Old runs keep their own decode and are labeled
契约里没有 `train_unit: "timeline"` 的旧训练 SHALL 继续用它原来的解码方式。结果页和标注页的模型列表 SHALL 给它们标上「旧训练方式」。它们 MUST NOT 被自动重训或改写。

#### Scenario: Picking an old model for prelabel
- **WHEN** 用户在标注页选模型3
- **THEN** 列表里它的名字带「旧训练方式」，预标注仍用原来按完整动作生长的解码

### Requirement: The station refuses contracts it cannot run
工位的 LSTM 动作引擎遇到 `network` 不是 `lstm`、或 `train_unit` 是 `timeline` 的契约时，SHALL 拒绝加载，原因写「未加载：这份模型要按整条连续推理，工位还没接入」。它 MUST NOT 把这类权重塞进固定窗口的 LSTM。工位的其他行为和配置 MUST NOT 改。

#### Scenario: MS-TCN contract on the station
- **WHEN** 工位配置了一份 `network` 为 `mstcn` 的契约
- **THEN** 引擎显示未加载和上面的原因，不抛异常

#### Scenario: Old LSTM contract on the station
- **WHEN** 工位配置了一份没有 `train_unit` 的旧 LSTM 契约
- **THEN** 照原来一样加载
