## 1. ROI NG 证据回归测试

- [x] 1.1 在 `tests/test_roi_runtime_integration.py` 增加保存开关启用/禁用和 PASS 不保存测试，证明系统不自动改变配置。
- [x] 1.2 增加一轮最终 NG 只保存一次的测试，覆盖稳定帧、重复 overlay 或相同 feedback key 不重复落盘。
- [x] 1.3 更新 worker 最新帧已推进测试，证明证据仍使用 NG overlay 同帧识别 ROI 裁剪图、帧号和 ROI 内 detections。
- [x] 1.4 增加严格 ROI 尺寸、中心点筛选、局部坐标、越界框自然裁切测试，并证明图片不含 ROI 外目标或其他覆盖层。
- [x] 1.5 增加 ROI 内空 detections 仍保存无框裁剪图、无效 polygon 跳过单框，以及无效/空 ROI 禁止回退整图的测试。
- [x] 1.6 增加结果帧编号不一致和同帧图像缺失时不调用写入器、禁止回退最新帧并记录错误的测试。
- [x] 1.7 增加写入器返回失败时记录日志但 NG 统计、弹窗、报警和轮次处理继续执行的测试。
- [x] 1.8 保留或补充旧动作 NG 证据测试，证明旧 `_save_fail_evidence()` 行为未改变。

## 2. 同帧带框保存实现

- [x] 2.1 提取供 ROI NG 保存和弹窗共用的同帧校验与带框构图入口，校验 result/event frame ID 和非空 `source_frame`。
- [x] 2.2 调整 `_handle_roi_results()` 和 `_save_roi_fail_evidence()`，显式传入当前 overlay payload，不再读取 `_get_feedback_frame()` 或 `_last_overlay_data`。
- [x] 2.3 在共用构图入口中严格裁剪识别 ROI，按中心点筛选 detections，深复制并平移 polygon、box、center/center_px 后绘制绿色 OBB。
- [x] 2.4 使用 ROI 局部 detections 和原图 `source_roi` 构造 `FailEvidenceContext`；仅 ROI NG metadata 增加 `source_roi`，保持单个 `frame.jpg` 和目录命名不变。
- [x] 2.5 对帧错配、同帧缺失、识别 ROI 无效和 `save_fail_event()` 返回 `None` 写入运行日志；无效 ROI 不回退整图且不得阻塞统计、声音和轮次。
- [x] 2.6 确认保存仍位于结果 feedback key 去重之后，并保持旧动作检测保存方法、配置默认值和其他业务逻辑不变。

## 3. 文档与验收

- [x] 3.1 更新 `yolo_action_detection/README.md` 和简易操作手册源文档，说明 NG 弹窗和证据图严格裁剪为识别 ROI，并重新生成 DOCX。
- [x] 3.2 运行 ROI 运行时、证据保存和弹窗相关测试，确认新增裁剪回归用例全部通过。
- [x] 3.3 运行 `python -m pytest yolo_action_detection/tests -q`、`python -m compileall yolo_action_detection/src` 与 `git diff --check`。
- [x] 3.4 运行 `openspec validate save-roi-ng-annotated-evidence --strict`，逐项对照 proposal、design 和 spec 自检。
- [x] 3.5 构建便携包并运行 portable smoke test，构建前后校验现场配置、模型和定位模板哈希不变。
- [x] 3.6 使用临时启用保存开关的配置触发一次多 OBB ROI NG，检查实际事件目录只有 ROI 裁剪 `frame.jpg` 与 `metadata.json`，图片尺寸、绿色框、局部坐标和 `source_roi` 一致；不得覆盖现场配置。
