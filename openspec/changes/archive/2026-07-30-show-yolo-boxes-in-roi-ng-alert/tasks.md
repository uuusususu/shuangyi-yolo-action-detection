## 1. 同帧证据回归测试

- [x] 1.1 在 `tests/test_camera_selection.py` 或新的 worker 测试中增加用不同像素值连续推理两帧的用例，证明 overlay 事件携带的图像与 `source_frame_id` 来自同一次推理。
- [x] 1.2 在 `tests/test_roi_runtime_integration.py` 增加“最新预览和 worker 最新显示帧均已推进，但 NG 弹窗仍使用触发帧”的失败测试。
- [x] 1.3 增加 NG 轮廓绘制测试，覆盖识别 ROI 内外的全部有效 OBB polygon、绿色像素、旋转形状以及无效 polygon 跳过。
- [x] 1.4 增加禁止项测试，证明 NG 图像不绘制类别名、置信度、跟踪 ID、搜索 ROI、识别 ROI 或其他覆盖层。
- [x] 1.5 更新弹窗控件测试，证明内容区只有图像控件，不再存在摘要滚动区、数量文字或操作提示。
- [x] 1.6 保留并调整声音与键盘回归测试，覆盖首次 NG 只启动一次循环报警、后续 NG 只替换图片、回车关闭并停止报警、PASS 不弹窗。

## 2. 同帧数据链路实现

- [x] 2.1 调整 `CameraWorker` 的 overlay 事件载荷，把当前推理帧与 `source_frame_id`、detections 作为同一事件数据传给界面，并保持现有公开统计字段不变。
- [x] 2.2 保证事件中的帧对象不会被后续采集或绘制原地修改，且相机关闭和会话重置后不保留历史帧引用。
- [x] 2.3 调整 `MainWindow._on_overlay_updated()` 和 ROI 结果处理接口，使 NG 分支显式消费本次事件载荷，不再从 worker 读取可变的最新反馈帧。
- [x] 2.4 对同帧图像缺失或帧编号不一致增加运行日志和保护逻辑，禁止把另一帧与当前 detections 组合。

## 3. 纯检测框告警界面

- [x] 3.1 实现独立的 NG 检测轮廓绘制函数：复制冻结帧并为全部有效 YOLO OBB polygon 绘制固定线宽的绿色闭合轮廓。
- [x] 3.2 确保 NG 绘制路径不调用会追加文字、枪头、孔位或 ROI 的主画面完整覆盖层函数。
- [x] 3.3 简化 `RoiFailAlertDialog` 为单一图像内容，删除摘要列表、数量文字、提示标签及相关累积状态。
- [x] 3.4 将弹窗刷新接口改为只替换带框冻结图；弹窗已打开时新 NG 替换图片但不重建窗口、不重启声音。
- [x] 3.5 保留回车关闭、`alert_closed` 信号、循环声音停止和窗口清理行为。

## 4. 文档与验收

- [x] 4.1 更新 `yolo_action_detection/README.md` 和简易操作手册源文档，说明 NG 弹窗显示冻结检测帧、绿色旋转框以及回车关闭操作，并重新生成 DOCX。
- [x] 4.2 运行 ROI 弹窗和相机 worker 相关测试，确认新增回归用例全部通过。
- [x] 4.3 运行 `python -m pytest yolo_action_detection/tests -q`、`python -m compileall yolo_action_detection/src` 与 `git diff --check`。
- [x] 4.4 运行 `openspec validate show-yolo-boxes-in-roi-ng-alert --strict`，逐项对照 proposal、design 和 spec 自检。
- [x] 4.5 构建便携包并运行 portable smoke test，确认不新增依赖、原配置和模型文件保持不变。
- [x] 4.6 在 100% Windows DPI 下触发一次带多个旋转检测框的模拟 NG，保存并人工检查弹窗截图：框图同帧、全部框为绿色、内容区无文字且无重叠裁切。
