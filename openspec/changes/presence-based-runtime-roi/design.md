## Context

三种 ROI 已经能单独保存，但 `_wrap_with_template_gate`、`_init_roi_engine`、`_template_runtime_validation_error` 都先看 `roi_inspection_enabled`。开关关掉时，已存框不进检测。开检还走 `inspection_validation_errors(require_roi_rects=True)`，缺识别 ROI 会拒绝不按顺序认法。

判定本身已经按数量工作：顺序用 `StepSequenceEngine` 逐步数 `category_counts`，不按顺序用 `RoiInspectionEngine` 同时数。缺的是检测入口把已存框用起来。

相邻变更 `decouple-template-create-from-search-roi` 只改创建/回查。本设计不碰配置页存框路径。

## Goals / Non-Goals

**Goals:**

- 开始检测、保存后重建时，按有效框和可加载模板组装可选层。
- 顺序识别也能过滤识别 ROI、也能挂模板门控。
- 空层跳过，不拦启动；损坏的模板文件才拦。
- 认法开关只分顺序 / 不按顺序，两种都按配置数量计数。

**Non-Goals:**

- 不改步骤引擎、ROI 引擎的数量公式、稳定帧、错序、失败重试。
- 不改形状匹配算法、YOLO 后处理、Hands、相机、打包、配置版本。
- 不把模板 ROI 当运行过滤框。
- 不恢复 PCB / 首类别 / 标定打孔。
- 不改「保存当前 ROI」与整页保存不覆盖框草稿。

## Decisions

### 1. 认法开关与层开关拆开

`roi_inspection_enabled` 只选引擎：关 → 步骤引擎，开 → ROI 引擎。  
层是否挂上另算：识别 ROI 有效就过滤；`roi_template_enabled` 开且搜索 ROI 有效且模板能加载才挂 `TemplateGatedProcessor`。

备选：取消认法开关、完全看有没有框。否决，因为「按顺序」和「定位后同时计数」仍要入口。

### 2. 共用「矩形是否可用」

在 `roi_inspection/geometry.py` 增加函数：能通过 `NormalizedRect.from_mapping` 才算有效，否则 `None`。校验、门控、过滤、画框都走它。非法旧框当空并打警告，不拦启动。

### 3. 过滤放在判定入口，不改 overlay

`_on_overlay_updated` 在喂引擎前过滤。`overlay.detections` 保持整图，框外目标仍可画框。步骤引擎源码不改。

### 4. 空识别 ROI 在 ROI 引擎里当整图

`RoiInspectionConfig.from_config` 在框不可用时用 `{x:0,y:0,width:1,height:1}`，避免 `from_mapping(None)` 把引擎建失败。测试按「空 = 整图」断言。

### 5. 引擎 `template_enabled` 跟门控是否挂上

不能只读开关字面。开关开但搜索或模板缺一样时，不按顺序认法按无模板触发（框内或整图出现待检类别即开轮）。否则「缺了不拦启动」会变成永远等 `template_match`。

`_wrap_with_template_gate` 同样：不再要求认法开关；缺一层就返回未门控 processor，不抛「搜索 ROI 未配置」。文件存在但 `load_shape_model` 失败继续抛错。

### 6. 开检校验收窄

`inspection_validation_errors` 不再把缺框/非法框当错误。`_template_runtime_validation_error` 只在门控层会实际启用时检查：processor 已门控、模型能加载、搜索余量够。余量不够仍用现有「搜索框太小」文案。

### 7. 主画面画有效识别框和搜索框

现规格「只画搜索 ROI、不画识别 ROI」改为：有效的识别 ROI、搜索 ROI 都画；模板轮廓、得分、模板 ROI 仍不画。否则顺序识别里框了识别 ROI，现场看不到过滤范围。

## Risks / Trade-offs

- [现场配置里留着以前的合法识别框，顺序识别会突然只数那一块] → 主画面画出该框；不对就删除或重框。非法框当空。
- [模板开关开、文件缺失时若仍把 `template_enabled=True` 传给引擎，会永远等定位] → 决策 5，以门控是否挂上为准。
- [框外目标仍画框，操作员可能以为已经计数] → 判定列表只反映过滤后数量；不在本变更里改画框策略。
- [与 `decouple-template-create-from-search-roi` 抢 `main_window` 开检校验] → 本变更只收窄「何时检查余量」，创建/回查仍归那个变更。

## Migration Plan

1. 先加「矩形是否可用」和校验放宽，再改门控组装与判定过滤，最后改画框。
2. 不升 schema，旧配置直接生效。
3. 回滚：还原上述文件即可，配置字段未变。

## Open Questions

- 无。开关说明文字是否改配置页文案，实现时按现场可读性决定，不挡行为。
