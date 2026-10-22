## Why

动作模型只会把一段时间判成包装或封装。标注里出现过的零件如果在验证这段里一次都没出现，它仍然把这段当成完成。缺件必须在验证时卡住，并且不用再训一个模型。

## What Changes

- 开训时扫这一轮用到的标注段。某一类动作里，一件零件只要有一帧置信度大于 0，就记到这一类上。空闲不记。记录写进这一轮的 `contract.json`，不写进权重。
- 验证时先用这一轮的动作模型把视频分成段。判成某一类动作之后，再看契约里这一类记过的零件，在这段特征里是否都出现过。
- 都出现过，这段完成。少一件，这段失败，并写上缺的零件名。不看先后。后面被挡住没有关系，这段里前面出现过就算。
- 动作模型没判成需要卡零件的类别，不查零件，按动作结果本身失败。
- 没有 `required_parts` 的旧轮次不卡，验证结果与现在相同。

## Capabilities

### New Capabilities

- `labeled-part-gate`: 从标注段的特征里记下每类动作出现过的零件，验证时缺件则这段动作失败。

### Modified Capabilities

## Impact

- `lstm_workbench/service.py`：开训时根据已经抽好的特征计算 `required_parts`，交给训练写入契约。
- `lstm_workbench/trainer.py`：契约增加 `required_parts`。不改变网络结构，不重训零件模型。
- `lstm_workbench/check_clip.py`：动作段生成之后，用同一段特征做缺件判断，写入每段的完成或失败。
- `lstm_workbench/static/app/view-runs.js`：当前动作那一行和段表显示完成，或失败以及缺的零件。
- 不修改工位顺序检测，不修改 ROI，不写工位 `config.json`。零件模型仍只做推理。
