## Why

当前 ROI 检测进入 NG 后，告警弹窗显示的是接近原图的最新预览画面，员工无法从画面直接确认 YOLO 实际框出了哪些目标；同时，预览帧可能已经晚于触发 NG 的推理帧，存在画面与检测结果不对应的风险。需要把告警证据收敛为触发 NG 的同一帧图像及其检测框，让现场人员无需阅读数量文字即可查看识别结果。

## What Changes

- ROI 轮次触发 NG 时，冻结与该 NG 结果对应的推理帧，而不是重新读取弹窗时刻的最新预览帧。
- 在冻结帧上绘制该帧全部 YOLO OBB 检测结果，使用绿色旋转四边形轮廓。
- 告警图片不显示类别名、置信度、跟踪 ID、ROI 框或其他文字标记。
- 移除告警弹窗中的数量摘要和操作提示，仅保留带检测框的冻结画面。
- 保持现有 NG 循环报警、弹窗去重以及回车关闭并停止报警的行为不变。

## Capabilities

### New Capabilities

- `roi-ng-visual-alert`: 定义 ROI 轮次 NG 告警的同帧冻结证据、YOLO OBB 轮廓绘制和纯图像弹窗行为。

### Modified Capabilities

无。

## Impact

- 主要影响 `yolo_action_detection/src/ui/main_window.py` 与 `yolo_action_detection/src/ui/roi_fail_alert_dialog.py`。
- 需要在相机推理结果链路中保留或按 `source_frame_id` 取得与检测结果严格对应的帧，可能涉及 `yolo_action_detection/src/camera/camera_worker.py` 的有限接口调整。
- 需要更新 ROI 运行集成测试，覆盖同帧绑定、全部检测框、纯图像布局、声音和回车关闭回归。
- 不新增第三方依赖，不修改 YOLO 推理、ROI 数量判定、生产统计、FAIL 证据保存或配置文件格式。
