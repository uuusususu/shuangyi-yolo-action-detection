# 开源 Qt 控件主题验收记录

## 交付产物

- 程序：`E:\双翼科技项目\动作检测\yolo_action_detection\dist\YOLOActionDetection\YOLOActionDetection.exe`
- 构建时间：`2026-07-18 00:37:50`（Asia/Shanghai）
- 文件大小：`5,234,867` 字节
- SHA256：`18EEEDE33AF6C574C9D56982CF0EF0E07793718B927794F75B23B3FCB5DD6FA6`
- 控件主题：`pyqtdarktheme==2.1.0`，MIT 许可证

## 自动化验证

- 主题专项测试：`6 passed`
- UI 与配置回归：`52 passed`
- 联合行为回归：`92 passed`
- 最终全量测试：`131 passed in 28.28s`
- Python 编译检查：`compileall` 通过
- 源码便携冒烟：通过，输出包含 `mvsdk=available`、`theme=pyqtdarktheme-2.1.0` 和 `ok`
- 打包程序便携冒烟：`PACKAGED_SMOKE_EXITCODE=0`
- OpenSpec 校验：`openspec validate adopt-open-source-qt-control-theme --strict` 通过

## 视觉验收

- Windows 当前系统 DPI 为 `96`，在真实 100% 缩放下启动打包 EXE，逐页检查“模型与步骤、动作判定、工业相机、显示与反馈、区域检查、生产统计”六个配置分区。
- 使用同一打包 EXE 和进程级 `QT_SCALE_FACTOR=1.25` 完成 125% Qt 渲染复核，窗口截图尺寸为 `1503x960`。
- 按钮静止边界、主次操作层级、危险按钮、下拉箭头、数字框步进按钮、标准复选框选中/未选中状态和中文文字均清晰。
- 固定底栏、配置导航、表单卡片和主检测页信息架构没有裁切、重叠或溢出；深蓝背景、青色强调和 PASS/FAIL 业务色保持不变。
- README 展示截图已替换为本次打包程序的 100% 实际运行截图。
- 100% 截图证据：`yolo_action_detection/outputs/ui_acceptance/packaged-scale-1_0/`
- 125% 截图证据：`yolo_action_detection/outputs/ui_acceptance/packaged-scale-1_25/`

## 未消除风险

- 125% 验收通过进程级 Qt 缩放完成，没有修改 Windows 全局显示设置；这能覆盖 Qt 控件尺寸与布局，但目标电脑若使用特殊多屏 DPI 组合，仍建议现场做一次窗口拖屏复核。
- 本次主题变更没有修改迈德 SDK 包。最终 UI 验收时没有在线迈德相机，设备枚举和保存后切换由自动化回归覆盖，真实硬件采集不属于本次主题替换的验收范围。
- 主题版本已固定为 `2.1.0`；后续升级依赖时必须重新执行控件状态截图、全量测试和打包冒烟。

## 结论

本变更的实现、打包和验收任务已经完成，可交付用户确认。未得到用户确认前不归档 OpenSpec 变更。
