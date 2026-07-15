## 1. 配置与 SDK 设备身份

- [x] 1.1 先增加配置回归测试，覆盖 `mvsdk_camera_sn` 默认值、JSON 往返持久化以及设备列表和枚举索引不落盘。
- [x] 1.2 先用伪造 MvSDK 增加多设备测试，覆盖 FriendlyName 重名、枚举顺序变化、按 SN 精确匹配、目标缺失不回退第一台和 SDK 错误信息。
- [x] 1.3 在 `ConfigManager`、默认配置和示例配置中增加 `mvsdk_camera_sn`，保留旧 `mvsdk_friendly_name` 兼容字段。
- [x] 1.4 调整 `MvSdkDevice` 显示信息和 `MvSdkCamera.open()`，从最新 `CameraEnumerateDevice()` 结果中按 SN 选择 `tSdkCameraDevInfo` 并把具体失败原因提供给调用方。

## 2. CameraWorker 生命周期与缓存清理

- [x] 2.1 先增加 worker 测试，覆盖当前设备 SN、未选择设备拒绝打开、采集线程退出后再释放 SDK 资源以及关闭后所有帧/overlay 缓存归零。
- [x] 2.2 为 `CameraWorker` 增加设备枚举和当前活动设备查询接口，并移除配置为空时自动选择第一台的路径。
- [x] 2.3 实现有界停止采集/推理线程后再关闭 SDK 句柄的顺序，避免 `CameraGetImageBuffer()` 与 `CameraUnInit()`/`CameraAlignFree()` 并发。
- [x] 2.4 实现锁保护的 preview、infer、display、overlay、帧 ID 和 pipeline stats 会话缓存清理，并确保重新打开只产生新相机帧。

## 3. 工业相机配置界面

- [x] 3.1 先增加 `ConfigPage` 测试，覆盖多设备下拉项显示、SN item data、无设备、空 SN、保存 SN、未保存不生效和手动刷新信号。
- [x] 3.2 增加启动设备目录并在配置页创建时注入当前枚举结果，保证设备目录仅驻留内存。
- [x] 3.3 在“工业相机”分区增加设备下拉框、刷新按钮、发现数量和状态提示，显示 `FriendlyName | PortType | SN`。
- [x] 3.4 实现已保存 SN 在线时预选、离线时警告、旧 FriendlyName 唯一匹配时预选以及重名时要求人工选择。
- [x] 3.5 把有效下拉选择接入现有保存流程；空 SN 设备不得保存，点击返回或切换分区不得改变运行中的相机。

## 4. 保存后切换与统一清屏

- [x] 4.1 先增加主窗口测试，覆盖相机关闭时只保存、相同 SN 不重启、不同 SN 停止检测并切换、目标离线保持关闭以及生产统计不清零。
- [x] 4.2 先增加视觉清理测试，覆盖视频 pixmap、运行时 overlay、最后数据、步骤/父区域状态、异常提示和 PASS/FAIL 文案全部清空。
- [x] 4.3 在 `MainWindow._on_config_saved()` 中比较目标 SN 和活动 SN，并按“停止检测 -> 停线程 -> 释放旧相机 -> 清理会话 -> 精确打开新相机”的顺序编排切换。
- [x] 4.4 抽取并复用统一的相机会话清理路径，让关闭相机、保存后切换和窗口退出使用一致的线程停止与 SDK 释放逻辑。
- [x] 4.5 新相机打开成功后仅恢复预览，打开失败时保持空白和相机关闭，并显示目标 SN 及 SDK 错误码或原因，禁止回退其他设备。

## 5. 文档与验证

- [x] 5.1 更新中文 README 和关键配置说明，记录多相机选择位置、SN 持久化、刷新、保存后切换和首次升级需明确选相机的行为。
- [x] 5.2 运行相机、配置页和主窗口相关定向测试，确认所有新增失败场景通过。
- [x] 5.3 运行 `pytest yolo_action_detection/tests -q`，确认完整测试集无回归。
- [x] 5.4 运行源码 `--portable-smoke-test` 和 `openspec validate add-mvsdk-camera-selection --strict`。
- [x] 5.5 执行 `yolo_action_detection/packaging/build_yolo.bat` 并运行打包程序冒烟测试，确认内置 `python_demo/mvsdk.py` 和 SDK DLL 仍可加载。
