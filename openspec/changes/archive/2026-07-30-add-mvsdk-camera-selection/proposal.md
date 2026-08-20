## Why

当前程序虽然能通过迈德 MvSDK 枚举相机，但未提供设备选择界面；配置未指定名称时会默认打开第一台相机，按 FriendlyName 查找失败时也会回退第一台。在多相机现场，这可能连接到错误设备，且关闭或切换后旧图像与检测状态仍留在界面。

## What Changes

- 在配置界面的“工业相机”分区增加迈德相机下拉框和刷新按钮，展示当前 SDK 枚举到的全部在线设备，显示内容包含 FriendlyName、接口类型和 SN。
- 每次程序启动重新调用 `CameraEnumerateDevice()` 获取最新设备列表；设备列表仅保存在内存中，不写入配置文件。
- 配置只持久化操作员选中的相机 SN；启动重新枚举后按 SN 恢复选择，SN 不在线时提示重新选择，不自动改选第一台。
- 点击配置页“保存”后才应用相机变更。若当前相机已打开且 SN 发生变化，系统停止检测、关闭旧相机、清理旧画面及本轮检测状态，再按新 SN 打开相机并开始预览。
- 按 SDK 枚举结果中的 `tSdkCameraDevInfo` 调用 `CameraInit()`，不使用枚举索引作为持久身份，也不再在匹配失败时回退第一台相机。
- 相机关闭、切换失败或设备离线后清空最后一帧、检测框、步骤状态和 PASS/FAIL 文案，并显示明确状态；切换失败时保持相机关闭。
- 保留旧 `mvsdk_friendly_name` 配置的读取兼容，但新的选择与打开流程以 SN 为准。
- **BREAKING**：未明确选择相机、已选 SN 不在线或设备匹配失败时不再自动打开枚举列表中的第一台相机，操作员必须在配置页选择并保存有效设备。

## Capabilities

### New Capabilities

- `mvsdk-camera-selection`: 定义迈德相机实时枚举、按 SN 选择与持久化、保存后安全切换、错误处理及关闭清理行为。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/camera/mvsdk_camera.py`：设备身份、按 SN 精确匹配和 SDK 错误传播。
- `yolo_action_detection/src/camera/camera_worker.py`：设备枚举接口、指定设备打开、线程停止及帧缓存清理。
- `yolo_action_detection/src/ui/config_page.py`：工业相机设备下拉框、刷新、缺失设备提示和保存字段。
- `yolo_action_detection/src/ui/main_window.py`：保存后切换编排、视觉会话清理和失败状态反馈。
- `yolo_action_detection/src/config.py`、配置示例：新增所选相机 SN 字段并兼容旧 FriendlyName 字段。
- 测试覆盖 SDK 枚举映射、SN 精确选择、无错误回退、配置持久化、运行时切换和清屏行为。
