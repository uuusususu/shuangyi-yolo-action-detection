## Context

顺序检测在 `StepSequenceEngine.update` 里按单帧类别数量和 `action_pass_stable_frames` 推进步骤。零件检测在 ROI 引擎里按区域和数量判定。两者都读取同一份 `category_names` 和 `yolo_conf_threshold`。LSTM 训练在 `lstm_workbench/`，特征由 `build_feature_row` 生成，权重和 `contract.json` 写在训练工程的 `runs/<时间>/`。工位还没有加载这条模型的入口。

## Goals / Non-Goals

**Goals:**

- 增加默认关闭的使能。关闭时运行路径与现在相同。
- 打开后，用现有类别、现有置信度、现有 Pass 稳定帧，由 LSTM 在独立状态里判过。
- 契约或权重不可用时，只停时序判过。

**Non-Goals:**

- 不修改 `StepSequenceEngine` 的判定、顺序错和 NG。
- 不修改 ROI 零件检测的数量和区域判定。
- 不新增类别名单或置信度字段。
- 不把训练台的工程目录写进工位，也不在工位里训练。

## Decisions

1. 新配置只有两项：`lstm_action_enabled`（默认 false）和 `lstm_contract_path`（默认空字符串）。类别继续用 `category_names` 里非空名称，顺序与界面一致。置信度继续用 `yolo_conf_threshold`。时序 Pass 不累计稳定帧。不提高配置 schema 版本，旧文件缺这两个键时按默认值。

2. 时序运行放在新模块，由 `main_window` 在顺序检测更新之后调用。它读取同一帧已经算出的检测框和手部关键点，调用 `lstm_workbench.features.build_feature_row`，不复制特征公式。`step_engine.update` 的入参和返回值不变。

3. 加载时读取契约和同目录的 `best_lstm.pt`。契约里的动作类别顺序必须等于非空 `category_names` 再前面加上 `idle`。特征长度、网络结构（双层、隐层 128、dropout 0.2）和 YOLO 文件指纹必须与契约一致。任一不符则本帧不起模型，状态为未加载，本帧不产生时序 Pass。

4. 每帧把特征行追加进长度 48 的队列。队列满后取最后 48 帧做一次前向，对最后时刻 logits 做 softmax。最高概率的类别属于已配置类别，且概率不低于 `yolo_conf_threshold`，该类别当帧在时序状态上 Pass。输出为 `idle`、空类别，或概率不足，这一帧不 Pass。`action_pass_stable_frames` 只留给顺序检测。

5. 时序 Pass 只写自己的状态并在界面单独显示。它不调用顺序引擎，不改变步骤卡片，不改变 ROI 结果。

备选是把 LSTM 输出送进 `StepSequenceEngine` 代替数框。这样会使能开关改变现有顺序检测，已明确排除。

## Risks / Trade-offs

- [契约类别顺序和工位 `category_names` 不一致] → 拒绝加载，顺序检测照常。需要用同一列类别重新训练或改工位类别名后再启用。
- [特征里的零件类顺序来自契约 `part_class_names`] → 与当前 YOLO 标签对不上时拒绝加载，避免用错槽位仍判过。
- [每帧做 LSTM 前向] → 只在使能且模型已加载、队列满 48 帧后计算。使能关闭时不建队列、不建网络。
- [手部关键点缺失] → 按训练约定填 0，不因此停掉顺序检测。

## Migration Plan

旧 `config.json` 不写新键也能打开，使能为关闭。回退时取消勾选并保存，即停止加载模型。不迁移已有步骤状态。

## Open Questions

无。类别和置信度沿用现有配置。时序 Pass 不等稳定帧。
