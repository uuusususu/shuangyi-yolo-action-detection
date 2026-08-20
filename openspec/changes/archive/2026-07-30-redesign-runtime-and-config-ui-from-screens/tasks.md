## 1. UI Tokens and Components

- [x] 1.1 Align PySide6 color, border, spacing, typography, and button tokens with `main-screen.html` and `config-screen.html`.
- [x] 1.2 Add reusable config navigation button, form group, form field, switch row, and footer action components.
- [x] 1.3 Add a compact recognition list item component for runtime progress display.

## 2. Runtime Main Screen

- [x] 2.1 Rework `MainWindow` layout to match `main-screen.html`: top bar, camera viewport, right-side panel, and bottom controls.
- [x] 2.2 Replace large step-card display with compact recognition progress list while preserving status semantics.
- [x] 2.3 Show quantity progress such as `3 / 4` in the recognition list.
- [x] 2.4 Keep PASS/NG colors visible during `round_cooldown_seconds` interval.
- [x] 2.5 Preserve camera, detection, configuration, close, KPI, audio, evidence, and stats behavior.

## 3. Config Screen

- [x] 3.1 Rework `ConfigPage` into left navigation plus right `QStackedWidget` sections.
- [x] 3.2 Implement sections: model/steps, action judgement, camera, display/feedback, region check, production statistics.
- [x] 3.3 Move completion interval only to action judgement and bind it to `round_cooldown_seconds`.
- [x] 3.4 Keep region check section generic and free of PCB-specific user-facing fields.
- [x] 3.5 Add fixed footer with cancel/back and validate/save actions plus saved feedback.

## 4. Tests and Verification

- [x] 4.1 Update config page save tests to work with the new navigation layout.
- [x] 4.2 Add or update tests for section navigation preserving unsaved values.
- [x] 4.3 Add or update tests for recognition list PASS/NG and quantity progress state.
- [x] 4.4 Run `pytest yolo_action_detection/tests -q`.
- [x] 4.5 Run `openspec validate redesign-runtime-and-config-ui-from-screens --strict`.

## 5. 步骤加载与区域数量回归修复

- [x] 5.1 增加已配置步骤启动即显示的回归测试。
- [x] 5.2 修复程序首次启动漏传 `category_counts` 的问题。
- [x] 5.3 为首类别区域检查派生子控件目标数量。
- [x] 5.4 按每个父区域内的同帧精确数量判定并同步数量进度到主界面。
- [x] 5.5 统一本地打包测试配置并重新生成可执行版本。
- [x] 5.6 运行全量测试、便携版冒烟测试和 OpenSpec 严格校验。
