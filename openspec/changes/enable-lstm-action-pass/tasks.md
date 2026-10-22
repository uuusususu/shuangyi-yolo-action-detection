## 1. 配置

- [x] 1.1 在 `ConfigManager` 增加 `lstm_action_enabled=False` 和 `lstm_contract_path=""`，旧文件缺键时保持这两个默认值，不提高 schema 版本
- [x] 1.2 在配置页增加时序使能勾选和契约路径，保存时写回这两项；置信度和类别名称仍只写入 `yolo_conf_threshold` 与 `category_names`

## 2. 时序判过

- [x] 2.1 新增运行模块：使能关闭或路径为空时不建网络、不追加特征
- [x] 2.2 加载同目录的 `contract.json` 与 `best_lstm.pt`；`idle` 加非空类别顺序、特征长度、网络结构或 YOLO 指纹不一致时标记未加载且不判过
- [x] 2.3 用 `build_feature_row` 和 `yolo_conf_threshold` 攒满 48 帧后前向；概率达到该置信度时当帧只在时序状态上 Pass 该类别，不等稳定帧

## 3. 接入而不改原判定

- [x] 3.1 在 `main_window` 顺序检测更新之后调用时序模块，不修改 `StepSequenceEngine.update` 的入参、返回值或步骤卡片
- [x] 3.2 单独显示时序 Pass 或未加载；ROI 零件检测结果保持原样

## 4. 验证

- [x] 4.1 测试：使能关闭时顺序引擎仍按数框更新，且不加载权重
- [x] 4.2 测试：类别顺序不一致时不判过，顺序引擎仍被调用
- [x] 4.3 测试：契约匹配且连续命中达到稳定帧时，只有时序状态 Pass，步骤状态不变
