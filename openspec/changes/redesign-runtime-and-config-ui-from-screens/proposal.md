## Why

当前 PySide6 主界面和配置页已经具备检测、步骤、统计和配置能力，但信息架构与新设计稿不一致：主界面右侧仍是传统步骤卡片，配置页仍是长滚动表单。项目中已有 `yolo_action_detection/main-screen.html` 和 `yolo_action_detection/config-screen.html` 两个目标原型，需要把它们转成原生 PySide6 界面，提升现场操作清晰度和配置效率。

## What Changes

- 主检测页按 `main-screen.html` 重构为：顶部状态栏、左侧大相机视窗、右侧 KPI/识别列表/异常提示/操作按钮。
- 右侧步骤区从大卡片列表改为“识别列表”风格，支持按顺序显示识别项和数量进度，例如 `脚垫 1/4 → 4/4`。
- 已配置步骤在程序启动和保存配置后立即显示，未识别步骤以等待状态呈现，不再因尚未命中而隐藏。
- 普通顺序模式和首类别区域检查都加载 `category_counts`；区域检查按每个父区域内的同帧精确数量判定子控件。
- 配置页按 `config-screen.html` 重构为：左侧分类导航、右侧分区表单、底部固定保存栏。
- 配置页分类包括：模型与步骤、动作判定、工业相机、显示与反馈、区域检查、生产统计。
- 区域检查配置只保留首类别区域检查开关、失败稳定帧和归属边距；完成一轮间隔只保留在动作判定分区。
- 保持现有业务逻辑、配置字段、模型加载、统计、证据保存和测试入口不变；本次主要改变视觉结构和交互组织。

## Capabilities

### New Capabilities

- `runtime-config-ui-redesign`: 定义基于 `main-screen.html` 与 `config-screen.html` 的主检测页和配置页信息架构、视觉层级、交互行为和验收标准。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/ui/main_window.py`：调整主检测页布局、右侧识别列表和操作区结构。
- `yolo_action_detection/src/ui/config_page.py`：重构为左侧导航 + 右侧 `QStackedWidget` 分区表单。
- `yolo_action_detection/src/ui/widgets/native_panels.py`：新增或调整识别列表项、配置导航按钮、表单分组、开关行等复用组件。
- `yolo_action_detection/src/ui/runtime_ui_tokens.py`：统一颜色、间距、边框、字体和状态 token，使 PySide6 与 HTML 原型一致。
- 测试覆盖保存配置、运行时重载、步骤/数量显示、完成一轮间隔保持颜色和配置页导航切换。
