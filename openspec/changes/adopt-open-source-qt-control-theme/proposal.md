## Why

当前 PySide6 主界面和配置页虽然已经采用深色科技风布局，但按钮、下拉框、数字框和开关仍依赖分散的自定义 QSS，部分控件在默认状态下边界、箭头或选中反馈不明显，容易让操作员把可点击控件误认为普通文字。需要用成熟的开源 Qt 控件主题统一交互状态，在保持现有布局和工业科技风的前提下提升专业性与可操作性。

## What Changes

- 引入 MIT 许可的 `PyQtDarkTheme`，使用其 PySide6 原生控件主题统一按钮、输入框、下拉框、数字框、复选框及其默认、悬停、按下、焦点、选中和禁用状态。
- 保留当前深蓝背景、青色强调色、卡片分组、左侧配置导航、顶部标题、底部固定保存栏以及主检测页布局，不切换为库的默认灰黑视觉。
- 移除配置页和公共组件中会覆盖开源主题的自绘控件 QSS；容器、状态卡片、PASS/FAIL 颜色和检测叠加样式继续使用现有科技风 token。
- 让所有可执行操作在静止状态下仍具有明确的可操作边界；主操作、次级操作和危险操作保持不同的视觉层级。
- 使用主题库提供的标准复选框状态替换当前缺少滑块反馈的自绘胶囊开关，不新增自绘控件。
- 保持配置字段、保存校验、相机 SN 枚举与切换、检测状态机、统计、声音和证据保存行为不变。
- 增加控件类型、交互状态、配置行为和屏幕尺寸的回归验收，并完成源码运行、全量测试、便携版打包和打包 EXE 冒烟验证。

## Capabilities

### New Capabilities

- `professional-qt-control-appearance`: 定义开源 Qt 控件主题的使用边界、控件可操作性、科技风保持、行为兼容和视觉/打包验收标准。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/requirements.txt`：增加开源主题依赖并固定经过验证的版本范围。
- `yolo_action_detection/src/main.py`：在创建窗口前初始化 Qt 控件主题和项目强调色。
- `yolo_action_detection/src/ui/runtime_ui_tokens.py`：保留布局、颜色和状态 token，移除与主题库重复或冲突的通用控件绘制规则。
- `yolo_action_detection/src/ui/config_page.py`：让配置页输入与操作控件使用统一主题，保留现有页面结构和信号语义。
- `yolo_action_detection/src/ui/widgets/native_panels.py`：调整导航、开关行和底部操作栏的控件样式边界，不改变组件职责。
- `yolo_action_detection/src/ui/main_window.py`：统一主界面配置、关闭、相机和检测按钮的控件状态，不改变检测流程。
- `yolo_action_detection/tests/`：增加主题初始化、控件边界、导航、保存、相机选择和状态保持回归测试。
- `yolo_action_detection/packaging/`：确认主题库及其资源被 PyInstaller 收集，并重新验证便携版程序。
