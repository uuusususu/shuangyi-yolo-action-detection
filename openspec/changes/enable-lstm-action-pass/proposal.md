## Why

工位现在只能靠单帧类别数量把步骤判过。时序模型已经能在训练台里认出一段动作，但工位还没有开关去用它。顺序检测和零件检测必须保持原样，时序判过不能插进这两条路。

## What Changes

- 新增一个使能 `lstm_action_enabled`，默认关闭。关闭时不加载 LSTM，不产生时序 Pass，顺序检测和零件检测与现在一致。
- 勾选后，用现有 `category_names` 里已填写的类别，以及现有 `yolo_conf_threshold`，由 LSTM 在单独的状态上把对上的类别判为 Pass。
- 不新增第二套类别名单，不新增第二套置信度。时序输出一旦对上已配置类别且达到置信度，该类别立即 Pass，不使用稳定帧数。顺序检测自己的稳定帧数不变。
- 契约路径 `lstm_contract_path` 只用来找到 `contract.json` 和同目录的 `best_lstm.pt`。类别名顺序、特征长度或文件指纹对不上时，时序动作不加载、不判过；顺序检测和零件检测继续跑。
- 特征沿用 `lstm_workbench.features.build_feature_row`。YOLO 框和手只提供特征，YOLO 类别名不直接当成时序 Pass。

## Capabilities

### New Capabilities

- `lstm-action-pass`: 使能打开时，用现有类别和置信度，由 LSTM 单独判过这些类别；关闭或契约不匹配时不影响顺序检测和零件检测。

### Modified Capabilities

## Impact

- `yolo_action_detection/src/config.py`：增加使能和契约路径，缺省为关闭和空路径。不提高 `config_schema_version`。
- `yolo_action_detection/src/ui/config_page.py`：增加使能勾选和契约路径。置信度输入和类别名称输入保持原控件。
- 新增时序动作运行模块，在 `main_window` 的叠加帧回调里于顺序检测之后单独调用。不修改 `step_sequence_engine.py` 的判定，不修改 ROI 零件检测。
- 运行时依赖已训练的 `best_lstm.pt` 与 `contract.json`，以及现有的 YOLO 检测和手部关键点。不写回训练台工程，不改 `config.json` 里已有字段的含义。
