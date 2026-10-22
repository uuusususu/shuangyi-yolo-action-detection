## Why

实施状态补充（2026-10-15）：本提案已获批准并实施；用户随后授权修复实现审核发现的问题。当前处于本地验证和隔离交付准备，真实产线验收及用户确认尚未完成，下文“本次仅完成提案”的文字保留为提案创建时的历史范围。

工作台已经完成连续视频动作训练，但产线仍按 YOLO 单帧类别和数量推进步骤；旧 LSTM 入口仅维护独立 PASS 列表，还明确拒绝 `train_unit: timeline` 的新模型。需要把工作台训练的 LSTM 接入实际顺序判定，使“完整动作被识别，并且该动作段内要求的 YOLO 类别都出现过”成为步骤 PASS 的条件。

## What Changes

- **BREAKING：以现有模式开关确定唯一判定入口。** `roi_inspection_enabled == false` 且 `action_order_constraint_enabled == true` 时使用 LSTM 动作顺序判定；关闭顺序约束时沿用原 YOLO 步骤判定，开启 ROI 时沿用原 ROI / YOLO 判定。旧 `lstm_action_enabled` 不再构成第二个使能条件，也不能绕过动作检查。
- **BREAKING：顺序模式必须加载合格的连续 LSTM 模型包。** 配置类别与模型动作类别不一致时报错并阻止启动，不混用 YOLO 步骤，不在模型缺失或失败时自动退回 YOLO。这里只支持 `network: lstm`、`train_unit: timeline`，不接入 MS-TCN 和旧滑窗动作模型。
- 增加部署清单，保存动作内部 ID 与中文名称的映射，并绑定契约和权重指纹。保留现有权重和 `required_parts`，不重训，不改变工作台的标注、训练和验证算法。
- 同帧运行 YOLO 与 MediaPipe，使用训练一致的完整手部 x/y/z、特征顺序、时间基及标准化；LSTM 连续保留隐藏状态，动作结果在推理线程产生，界面只消费结果。
- 一个动作必须在起点之后被完整观测，并在结束边界确认后结算。同一段内累计出现的 YOLO 类别可分布在不同帧，不要求同时出现，也不校验它们的出现顺序或数量。
- 已结束动作类别匹配当前步骤且 `required_parts` 全部出现，才推进一次 PASS；缺少类别时立即 NG、结束本轮，并记录缺项。首次步骤 PASS 后执行未来动作越序 NG；首步前延续现有不触发错序 NG 的规则。
- 保留现有结果提示、声音、NG 证据和生产统计入口；动作事件按唯一编号去重，轮次之间不复用零件证据。
- 本次仅完成提案、设计、规格和任务清单；实现、真实产线验收与归档均需用户分别批准。

## Capabilities

### New Capabilities

- `lstm-runtime-bundle`：部署清单、动作 ID/名称映射、文件和特征契约校验，以及不重训的现有模型打包。
- `streaming-lstm-actions`：同帧特征、训练时间基上的连续 LSTM 推理、完整动作段结算、事件去重及中断恢复。
- `action-sequence-part-gate`：按模式路由，动作与段内 YOLO 共同推进顺序，缺件 / 错序 NG 和轮次隔离。

### Modified Capabilities

- `runtime-model-configuration`：顺序模式改为依据 LSTM 动作名称校验和判定；纯 YOLO 模式维持原类别判断，配置保存时原子校验模型与模式，不自动改写手工类别。

## Impact

- 产线：`src/lstm_action/engine.py`、`src/camera/camera_worker.py`、`src/hand_tracking/mediapipe_hands.py`、`src/step_sequence/step_sequence_engine.py`、`src/ui/main_window.py`、`src/ui/config_page.py`、`src/config.py`、`src/main.py`，以及 NG 证据元数据和便携诊断。
- 共用算法：复用 `lstm_workbench/features.py`、`model.py`、`check_clip.py` 的既有数值规则；如需抽取增量切段辅助模块，必须证明工作台原有输出不变。不把训练器或网页服务带进产线执行路径。
- 部署：新增部署模型包准备工具、模型依赖和 PyInstaller 收集检查；Torch 与 MediaPipe 必须在实际便携包中验证，不以源码可导入代替交付验收。
- 文档与测试：新增流式推理 / 段内零件 / 模式回归测试，更新中文操作说明；保留所有既有训练结果及工位数据。
- 该提案在本次范围内取代 `enable-lstm-action-pass` 中“独立 PASS、不影响顺序、失败可继续 YOLO”的旧设计，并解除 `whole-video-temporal-training` 对新 LSTM 契约的产线拒绝；不改写这些历史变更，不顺带归档。
