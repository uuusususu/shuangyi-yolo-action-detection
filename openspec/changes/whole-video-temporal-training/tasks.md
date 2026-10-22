## 1. 逐帧标签与时间基

- [x] 1.1 在 `timeline.py` / `dataset.py` 加 `frame_labels(clip, class_ids)`：色块是类别下标，没涂是空闲，「不训练」是 -100；短段和短空档都保留。写单元测试，覆盖规格里四个场景（没涂是空闲、短空档、相邻无空档、剪掉的忽略）
- [x] 1.2 改 `service.time_base`：默认取最慢的帧率（相差 ≤ 0.05 取中位数）；手动时间基快过任一片子时拦住并写明片子和两个帧率；去掉用重复帧补慢片子。改 `test_service.py` 里原来按最快帧率的测试
- [x] 1.3 在 `settings.py` 加 `context_seconds`（默认 30）、`target_seconds`（默认 30）、`batch_size`（默认 8），校验上下限；`split_mode` 只接受 `clip`，旧值读入时转成 `clip`，并清掉旧的段划分
- [x] 1.4 训练页对超过 30 秒的未涂片尾给出提醒（`service` 算出秒数，`view-train.js` 显示），不拦训练

## 2. 指标

- [x] 2.1 在 `metrics.py` 加 `segments_from_labels`、`segmental_f1(truth, pred, iou)`、`edit_score`、`frame_accuracy`、`idle_false_alarm_frames`、逐帧混淆矩阵和每类段级 F1@50；动作段不含空闲，忽略帧不计
- [x] 2.2 单元测试：完全对上为 1；只切出 1 段、标注 12 段时 F1@50 = 2/13；过度切碎时编辑分下降；忽略帧不影响分数
- [x] 2.3 加 `pick_best(history)`：F1@50 → 编辑分 → 逐帧准确率，写测试覆盖规格里第 40 轮和第 60 轮的例子

## 3. 共用切段函数

- [x] 3.1 在 `check_clip.py` 加 `decode_timeline(rows, fps, model, contract, part_names)`：换算时间基、标准化、一次因果前向、逐帧取最大概率、`build_segments`、`absorb_short_segments`、帧号换回、零件门禁
- [x] 3.2 加 `require_timeline_contract(contract)`：缺字段或和现在的 YOLO / 手部 / 置信度对不上就拒绝，并写明哪一项
- [x] 3.3 `segments_for_contract` 遇到 `train_unit == "timeline"` 走 `decode_timeline`，其他契约保持原来的分支；`run_check`、`prelabel_clip` 不再自己写切段
- [x] 3.4 `_load_model` 按 `network`、`mstcn_layers`、`mstcn_stages`、`mstcn_channels` 还原；没有 `mstcn_layers` 的旧 MS-TCN 按 8 层还原。写测试：新旧 MS-TCN 契约和旧 LSTM 契约都能加载

## 4. 网络

- [x] 4.1 `model.py` 的 `CausalMSTCN` 接受层数参数；加 `mstcn_layers_for(context_frames)` 和 `receptive_field(layers, stages)`，层数取 4 到 10 之间不超过上文长度的最大值
- [x] 4.2 测试：两种网络改动第 t 帧之后的输入，第 t 帧及之前的输出都不变；6.25 fps、30 秒时层数和往回看长度符合规格

## 5. 截段训练

- [x] 5.1 在 `dataset.py` 加 `draw_crops(lengths, target, context, seed, epoch)`：目标起点随机（含 0），输入从 `max(0, 起点 − context)` 开始，每轮目标帧总数不少于训练帧数；测试可复现，并且每一帧都能被抽到
- [x] 5.2 `trainer.py` 新增 `train_timeline(...)`：小批（`batch_size`）、每批一步、Adam、类别权重为帧数的平方根倒数、只在目标帧算损失；MS-TCN 加平滑（τ = 4，λ = 0.15，各阶段相加，只算目标帧）
- [x] 5.3 用训练片子的有效帧拟合 `feature_mean`、`feature_std`（标准差下限 1e-3 换成 1），训练、验证都套用；测试验证片子的帧不进统计
- [x] 5.4 每轮结束把验证片子整条跑过 `decode_timeline`，算第 2 节的全部指标，写进 history 和 progress；按 `pick_best` 保留最好一轮；停止时保留目前最好的
- [x] 5.5 契约写入 `train_unit: "timeline"`、`decode_version: 1`、`network`、`smoothing`、`context_frames`、`target_frames`、`min_action_seconds`、`time_base_fps`、`feature_mean`、`feature_std`、`mstcn_*`、`same_source_validation`；只有 LSTM 契约写 `hidden_size` 和 `num_layers`
- [x] 5.6 `service.start_training` 改走 `train_timeline`：整条留出划分是唯一入口，没有验证片子、缺类时拦住并写明缺哪一类、缺在哪一边；`split.json` 按片子写

## 6. 训练页

- [x] 6.1 划分卡改成整条片子：每条片子「参加训练 / 只做验证 / 不参加」；显示两边的片子、每类帧数和动作段数；去掉「同一批片子里留几段」和「确认这个划分」
- [x] 6.2 配方加「上文秒数」「目标段秒数」「每批段数」，并写明它们的含义
- [x] 6.3 分数卡换成段级 F1@50、编辑分、逐帧准确率、空闲误报，并加「切出 N 段 / 标注 M 段」；曲线换成这四项加两条损失；训练结束后显示每类柱状图和逐帧混淆矩阵
- [x] 6.4 旧训练按契约显示原来的三张卡，并注明「旧训练方式，分数不能和新模型比」

## 7. 结果页与标注页

- [x] 7.1 验证卡加一个来源「工程里的片子」：后端加接口，用缓存特征（过期先抽），走 `decode_timeline`，结果存到这一轮的 `check/<clip_id>/`，不复制视频、不动标注
- [x] 7.2 标过的工程片子：同一时间轴上画出标注轨和预测轨，给出和训练页一样的分数；点没对上的标注段，视频跳到它的开头
- [x] 7.3 按 `split.json` 写明「没参与训练」「是这一轮的验证片」「参与过训练，分数偏乐观」
- [x] 7.4 结果页和标注页的模型列表给旧训练标「旧训练方式」；标注页预标注走 `segments_for_contract`

## 8. 工位拒绝

- [x] 8.1 `yolo_action_detection/src/lstm_action/engine.py`：`train_unit == "timeline"` 或 `network` 不是 `lstm` 时，状态写「未加载：这份模型要按整条连续推理，工位还没接入」，不加载、不抛异常
- [x] 8.2 在 `yolo_action_detection/tests/test_lstm_action_pass.py` 加测试：MS-TCN 契约和 timeline 契约被拒，旧 LSTM 契约照常加载

## 9. 端到端验收

- [x] 9.1 合成数据端到端测试：三条合成片子，动作交替、空档长短不一、有无空档的相邻动作都有；两条训练、一条整条留出；LSTM 和 MS-TCN 各训一次，留出片子的 F1@50 ≥ 0.9；同一组特征经过训练时验证、`run_check`、`prelabel_clip` 切出的段完全一样
- [x] 9.2 跑 `lstm_workbench/tests` 全部测试，以及 `test_lstm_action_pass.py`
- [x] 9.3 重启训练台（用 venv 里的 python），确认页面能打开，训练页显示新划分卡和新分数卡，旧训练仍能打开、能验证
- [ ] 9.4 真实数据验收，由用户在页面上操作：1、2、3、7、8 号片子参加训练，6 号片子只做验证，LSTM 和因果 MS-TCN 各训一次；结果页拿 6 号工程片子验证，记下两套网络的「切出 N 段 / 标注 12 段」、F1@50、编辑分，并核对结果页的分数等于训练页最好一轮的分数。这里的数据写回本变更，不当成上线验收
