## Context

仓库里已经完成两轮前端改动：

1. `redesign-runtime-and-config-ui-from-screens`：把主界面改成顶栏 + 左相机 + 右 KPI/识别列表/操作，把配置页改成左导航 + 右分区 + 底栏。
2. `adopt-open-source-qt-control-theme`：标准按钮、输入框、下拉框、数字框、复选框交给 PyQtDarkTheme 画状态。

之后业务继续演进：PCB / 首类别区域检查模块已删除，配置第五分区改成 ROI 定位，主界面又加了同帧显示和 ROI NG 弹窗。设计稿 `main-screen.html`、`config-screen.html` 已不在仓库；Git 历史里也找不到这两个文件，只能从现有 PySide6、`assets/screenshots/` 和归档 spec 还原意图。

当前对接缺口：

- `RecognitionListItem` 已替换 `StepCard`。现场要求未识别步骤不进列表，识别到才依次出现；对接时不要改成「已配置即可见」。
- `native_panels.py` 仍保留 `StepCard`、`PanelSection`、`MetricCard`、`DefectChip`，主路径未使用。
- `ConfigManager` 仍有大量 legacy 字段（标定、洞口、扭力枪、融合、关节追踪）；配置页不再暴露它们，这是正确边界，需要写死以免被「找回来」误接到新控件。
- KPI「今日产能」显示的是本批 `total`，配置里的 `today_target_capacity` 只进配置页，两边语义不一致。
- 主题库与自绘 token 边界已经存在，但旧控件若被重新启用会再次覆盖主题。

## Goals / Non-Goals

**Goals:**

- 把新 UI 控件和现有检测、配置、统计链路逐项对上，保证打开相机、开始检测、改配置、看进度、归零归档仍可用。
- 识别列表在两种模式下都只显示已经识别到的类别，并按识别顺序依次增加。
- 明确控件、配置字段、已删除模块三类边界，避免把 PCB/标定 UI 接回生产路径。
- 用现有截图和 PySide6 代码替代丢失的 HTML 原型作为验收依据。

**Non-Goals:**

- 不再做新的视觉重设计，不引入 WebView 或恢复 HTML 运行时。
- 不恢复 `pcb_inspection`、首类别区域检查、标定/打孔判定作为生产功能。
- 不把 legacy 配置字段全部铺回界面。
- 不改 YOLO、形状匹配、ROI 轮次和顺序状态机的判定规则。
- 不改已经提交的同帧显示和 ROI NG 证据裁剪行为。

## Decisions

### 1. 运行时 UI 真源是 PySide6，不是丢失的 HTML

HTML 原型只存在于归档文档引用中，仓库和 Git 历史都没有文件。继续把它们当实现依据会让对接无法验收。

替代方案是以 `src/ui/main_window.py`、`src/ui/config_page.py` 和 `assets/screenshots/main-window.png`、`config-page.png` 为结构真源，spec 中删除对不存在 HTML 的硬依赖。

### 2. 新控件按「必须接线 / 主题容器 / 废弃」三类对接

必须接到业务的控件：

| 控件 | 接到 |
| --- | --- |
| 顶栏「配置」「关闭」 | 切到 `ConfigPage` / 退出 |
| 「打开相机」「开始检测」 | `CameraWorker` + `AppState` |
| `KpiRow` | `ProductionStatsManager`；产能含义见决策 5 |
| `RecognitionListItem` | `StepSequenceEngine` 或 `RoiInspectionEngine` |
| 异常提示条 | 错序 NG / ROI 数量 NG |
| `ConfigNavButton` + `QStackedWidget` | 六个配置分区，切换不丢未保存值 |
| `FormGroup` / `FormField` / `SwitchRow` | 对应 `ConfigManager` 生产字段 |
| `RoiEditorWidget` | 三种 ROI 草稿；只有「保存当前 ROI」才落盘 |
| `FooterActionBar` | 验证并保存整页生产字段，不覆盖三种 ROI 草稿 |
| ROI NG 弹窗 | 同帧识别 ROI 裁剪图 |

主题容器：页面底色、卡片边框、识别项 PASS/NG 色继续用 `runtime_ui_tokens`；`QPushButton`、`QLineEdit`、`QComboBox`、`QSpinBox`、`QCheckBox` 的边界和交互态继续由 PyQtDarkTheme 负责。

废弃、禁止重新接到主路径：`StepCard`、`PanelSection`、`MetricCard`、`DefectChip`。实现时删除或移出生产 import，并加测试保证主窗和配置页不实例化它们。

### 3. 配置字段三分类，防止「找回来」接错

界面可改（保存必须读写）：

- 模型：路径、conf、iou、device、tracker、max_det、track_persist
- 步骤：`category_names`、`category_counts`
- 动作：PASS/NG 稳定帧、顺序约束、`round_cooldown_seconds`
- 相机：SN、参数模式/组/文件、手动曝光
- 反馈：置信度/Track ID 叠加、PASS/FAIL 声音、失败证据
- ROI：定位开关、模板开关、阈值/角度/步进、三种 ROI 矩形、模板诊断开关
- 统计：`today_target_capacity`、归零归档

仅文件保留、不接新控件：标定、洞口、扭力枪、融合、关节追踪、坐标日志、`ultralytics_half`、`display_result_max_age_ms`、模板 polarity / 最小平移（后两项由保存模板时的内部校验使用）。

禁止恢复到界面或运行时：`pcb_inspection_*`、`first_category_region_check_*`。打包脚本已剥离这些键，保持剥离。

### 4. 识别列表按识别结果依次出现

`_refresh_steps()` 创建卡片后先 `setVisible(False)`。普通模式 `_update_step_display()` 只在当前数量大于 0、PASS 或 NG 时显示该项。ROI 模式同样只在本轮已观察到数量、已锁存 PASS，或该类别已 NG 时显示。未识别项保持隐藏，识别到才依次进入步骤框。空配置仍显示「请到配置页设置」空态。

### 5. KPI 产能含义固定为「本批已检 / 今日目标」

`today_target_capacity` 已经能在配置页保存，但主界面「今日产能」只显示 `batch.total`。对接后显示 `已检 / 目标`，例如 `12 / 300`。良率、OK、NG 仍来自当前批次，不把目标当成已产量。

### 6. 保存和返回的应用边界保持现有契约

- 「验证并保存」写生产字段并 `emit config_saved`，主窗立即重载模型、切换相机 SN、重建步骤/ROI 引擎。
- 三种 ROI 仍只由「保存当前 ROI」单独落盘，整页保存不得用草稿覆盖已存矩形。
- 「返回检测」不把未点保存的表单写进 `ConfigManager`；只按已保存配置刷新引擎和 KPI。
- 未保存切换六个分区时，输入值必须保留。这是现有行为，对接时加回归，不改语义。

## Risks / Trade-offs

- [现场误以为未识别项也该提前列出] → 保持隐藏直到识别到；配置页才展示全部已配置步骤。
- [删除旧控件可能有隐藏引用] → 先用测试和全局搜索确认无生产 import，再删除。
- [KPI 改成已检/目标可能让习惯看总数的人困惑] → 标签保持「今日产能」，值格式改为 `已检 / 目标`，统计页仍单独显示总数。
- [找不到 HTML 原型] → 不阻塞实现；用截图和现有页面验收结构，不再等待恢复 HTML。

## Migration Plan

1. 先恢复识别列表按识别结果依次出现，并修正 KPI 格式。
2. 再加控件/字段清单测试，锁住保存应用边界。
3. 最后删除或隔离废弃控件，并更新 `runtime-config-ui-redesign` spec。
4. 回滚：只改 UI 显隐和文案，不改配置 schema；出问题可只回退主界面显示逻辑。

## Open Questions

- 无。模板 polarity 和搜索余量继续走内部校验，不在本 change 做成新表单项。
