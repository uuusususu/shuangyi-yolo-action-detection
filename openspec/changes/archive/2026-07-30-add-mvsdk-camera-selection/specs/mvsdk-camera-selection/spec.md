## ADDED Requirements

### Requirement: Application enumerates current MvSDK devices
系统 SHALL 在每次程序启动时调用迈德 SDK `CameraEnumerateDevice()` 获取当前在线相机，并 SHALL 将枚举结果仅保存在当前进程内存中。系统 MUST NOT 把设备列表或枚举索引写入配置文件。

#### Scenario: Startup finds multiple cameras
- **WHEN** 程序启动时 SDK 枚举到多台迈德相机
- **THEN** 系统保存本次会话的全部在线设备信息
- **AND** 每台设备信息包含 FriendlyName、PortType 和 SN

#### Scenario: Startup finds no camera
- **WHEN** 程序启动时 SDK 未枚举到相机
- **THEN** 系统保持可运行并显示没有在线迈德相机
- **AND** 系统不使用历史设备列表填充下拉框

#### Scenario: MvSDK cannot be loaded
- **WHEN** 程序启动时无法加载 `mvsdk.py` 或底层 SDK
- **THEN** 系统显示 SDK 不可用的具体原因
- **AND** 程序其他配置功能仍可使用

### Requirement: Camera configuration lists online devices
系统 SHALL 在配置界面的“工业相机”分区提供设备下拉框和刷新设备操作。下拉项 SHALL 显示足以区分设备的 FriendlyName、PortType 和 SN，并 SHALL 使用 SN 作为选择值。

#### Scenario: Operator opens camera configuration
- **WHEN** 操作员打开“工业相机”配置分区且当前会话枚举到设备
- **THEN** 下拉框显示全部在线设备
- **AND** 每一项的显示格式包含 FriendlyName、接口类型和 SN

#### Scenario: Operator refreshes devices
- **WHEN** 操作员点击“刷新设备”
- **THEN** 系统重新调用 SDK 枚举接口
- **AND** 下拉框替换为本次枚举得到的最新设备列表

#### Scenario: Enumerated device has no SN
- **WHEN** SDK 返回一台 SN 为空的设备
- **THEN** 系统显示该设备的诊断信息
- **AND** 系统阻止将该设备保存为稳定选择

### Requirement: Selected camera identity persists by SN
系统 SHALL 只持久化操作员选中的相机 SN 作为运行时设备身份。系统 MUST NOT 使用 FriendlyName 或枚举索引作为新配置的唯一选择依据。

#### Scenario: Saved SN is online after restart
- **WHEN** 程序启动后枚举结果包含已保存的 `mvsdk_camera_sn`
- **THEN** 配置页默认选中该 SN 对应的当前在线设备

#### Scenario: Saved SN is offline after restart
- **WHEN** 程序启动后枚举结果不包含已保存的 `mvsdk_camera_sn`
- **THEN** 系统不改写已保存 SN
- **AND** 系统提示已保存相机不在线并要求操作员重新选择
- **AND** 系统不自动选择第一台在线相机

#### Scenario: Legacy FriendlyName has one match
- **WHEN** 新 SN 字段为空且旧 `mvsdk_friendly_name` 恰好匹配一台在线设备
- **THEN** 配置页预选该设备
- **AND** 操作员下一次保存时系统写入该设备 SN

#### Scenario: Legacy FriendlyName is ambiguous
- **WHEN** 新 SN 字段为空且旧 `mvsdk_friendly_name` 匹配多台在线设备
- **THEN** 系统不自动选择任何一台
- **AND** 系统要求操作员根据 SN 明确选择

### Requirement: Camera selection applies only after configuration save
系统 SHALL 在配置保存成功后才应用相机选择变化。只改变下拉框但未保存时，系统 MUST 保持当前相机连接不变。

#### Scenario: Operator changes selection without saving
- **WHEN** 操作员在下拉框选择另一台相机但未点击保存
- **THEN** 当前相机继续采集
- **AND** 配置文件中的相机 SN 保持不变

#### Scenario: Saved selection changes while camera is closed
- **WHEN** 相机未打开且操作员选择有效设备并保存配置
- **THEN** 系统持久化新 SN
- **AND** 系统不自动打开相机

#### Scenario: Saved selection matches active camera
- **WHEN** 相机已打开且保存的目标 SN 与当前设备 SN 相同
- **THEN** 系统保持当前采集连接
- **AND** 系统不重启相机

### Requirement: Active camera switches safely after save
当相机已打开且保存的目标 SN 发生变化时，系统 SHALL 停止检测和旧相机采集，释放旧 SDK 资源，清理旧视觉会话，再按新 SN 打开目标设备并开始预览。系统 MUST NOT 自动恢复检测。

#### Scenario: Switching from one online camera to another succeeds
- **WHEN** 当前相机已打开且操作员选择另一台在线相机并保存
- **THEN** 系统停止当前检测和采集线程
- **AND** 系统在采集线程退出后调用 `CameraUnInit()` 并释放对齐帧缓冲
- **AND** 系统按目标 SN 打开新相机并显示新相机预览
- **AND** 检测保持停止直到操作员再次点击开始检测

#### Scenario: Device order changes before switch
- **WHEN** 设备枚举顺序发生变化但目标 SN 仍在线
- **THEN** 系统从最新枚举结果中按 SN 找到目标 `tSdkCameraDevInfo`
- **AND** 系统把该设备结构传给 `CameraInit()`
- **AND** 系统不依赖先前枚举索引

### Requirement: Device mismatch and switch failure fail closed
系统 MUST 在未选择设备、目标 SN 不在线或 SDK 初始化失败时保持相机关闭，并 SHALL 显示可操作的失败原因。系统 MUST NOT 回退到第一台相机、旧相机或 OpenCV 相机。

#### Scenario: Target SN disappears before save is applied
- **WHEN** 下拉框中的目标相机在保存应用前离线
- **THEN** 系统关闭旧相机并保持空白画面
- **AND** 系统提示目标 SN 不在线
- **AND** 系统不打开其他在线相机

#### Scenario: CameraInit reports device occupied
- **WHEN** `CameraInit()` 因设备被占用返回 SDK 错误
- **THEN** 系统保持相机关闭
- **AND** 系统显示目标设备与 SDK 错误码或错误原因

#### Scenario: No camera has been explicitly selected
- **WHEN** 相机 SN 为空且操作员点击打开相机
- **THEN** 系统提示先到配置页选择并保存相机
- **AND** 系统不打开枚举列表第一台设备

### Requirement: Closing or switching clears the visual session
系统 SHALL 在关闭相机或切换相机时清除属于旧相机会话的帧和检测状态，但 MUST 保留生产统计与批次累计值。

#### Scenario: Operator closes camera
- **WHEN** 操作员点击关闭相机
- **THEN** 视频区域不再显示最后一帧
- **AND** 系统清空检测框、步骤状态、父区域跟踪、异常提示和 PASS/FAIL 文案
- **AND** 状态显示相机未打开

#### Scenario: Camera switch starts
- **WHEN** 系统开始从旧相机切换到新相机
- **THEN** 系统立即停止显示旧相机缓存帧和 overlay
- **AND** 新预览只能显示新相机打开后采集的帧

#### Scenario: Camera session is cleared
- **WHEN** 相机会话因关闭、切换或打开失败被清理
- **THEN** 总数、OK、NG、良率和当前批次记录保持不变
