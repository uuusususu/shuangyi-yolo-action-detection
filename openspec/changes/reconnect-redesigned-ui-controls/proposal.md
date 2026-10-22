## Why

主检测页和配置页已经按新信息架构重做成 PySide6 原生界面，并叠了 PyQtDarkTheme，但对接并不完整：HTML 原型已从仓库消失、旧控件与已删除的 PCB/首类别模块还留着痕迹、识别列表在普通动作模式下会藏起未命中项、部分生产配置有字段无控件。现场现在需要的是把新控件重新接到现有检测、配置和统计链路上，而不是再做一轮视觉重设计。

## What Changes

- 以当前 PySide6 页面和仓库截图为唯一运行时 UI 真源，不再依赖已删除的 `main-screen.html` / `config-screen.html`。
- 建立新控件清单：哪些必须接到业务、哪些只是容器主题、哪些是已废弃旧控件。
- 把识别列表重新接到步骤引擎和 ROI 引擎：未识别到的步骤不显示，识别到后才依次出现在步骤框里。
- 核对配置页六个分区的加载/保存/返回应用边界，保证验证并保存后相机、模型、步骤、ROI、声音和统计立即生效。
- 把配置字段分成「界面可改」「仅文件保留」「禁止恢复」三类，不把已删除的 PCB / 首类别区域检查 / 标定打孔 UI 接回来。
- 清理或隔离未使用的旧控件（`StepCard`、`PanelSection`、`MetricCard`、`DefectChip`），避免后续误接到生产路径。
- 修正 KPI「今日产能」与 `today_target_capacity` / 实际产量的对应关系，避免操作员看错产能。

## Capabilities

### New Capabilities

- `redesigned-ui-control-wiring`: 定义新 UI 控件与运行时/配置业务的对接清单、保存应用契约、已删除模块边界，以及现场必须保持可用的操作路径。

### Modified Capabilities

- `runtime-config-ui-redesign`: 将原型依据从丢失的 HTML 改为现有 PySide6 与截图；配置第五分区以 ROI 配置为准；识别列表按识别结果依次出现，未识别项保持隐藏。

## Impact

- `yolo_action_detection/src/ui/main_window.py`：识别列表显隐、KPI 含义、配置保存后的引擎应用。
- `yolo_action_detection/src/ui/config_page.py`：分区控件与 `ConfigManager` 字段对照、保存/返回边界。
- `yolo_action_detection/src/ui/widgets/native_panels.py`：新控件职责与废弃控件处理。
- `yolo_action_detection/src/ui/theme.py`、`runtime_ui_tokens.py`：主题库画标准控件，token 只负责面板和业务色。
- `yolo_action_detection/src/config.py`：字段分类，不恢复 PCB/首类别运行开关。
- `yolo_action_detection/tests/`：识别列表可见性、配置保存应用、废弃控件不进入主路径。
- `openspec/specs/runtime-config-ui-redesign/spec.md`：与当前 ROI 配置页对齐。
