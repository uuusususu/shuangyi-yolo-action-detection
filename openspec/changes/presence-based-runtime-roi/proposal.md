## Why

顺序识别时，已保存的识别 ROI / 搜索 ROI / 模板被「启用 ROI 定位检测」挡住，检测根本不加载。现场框了识别 ROI 却等于没配。空框本不该拦启动，现在缺识别 ROI 会拒绝不按顺序认法。

## What Changes

- 开始检测时按「有效框 / 可加载模板」组装可选层：有识别 ROI 就过滤判定目标；模板开关开且搜索 ROI + 模板文件齐全就先定位再 YOLO。空的跳过，不报「未配置」。
- 「启用 ROI 定位检测」只切换认法（顺序 / 不按顺序），不再决定框加不加载。两种认法都按 `category_counts` 计数。
- 模板开关只做手动关闭。开关开但层没挂上时，不按顺序认法按无模板触发，不能一直等定位。
- 主画面：有效识别 ROI、搜索 ROI 都画；模板 ROI 检测时不画。框外目标仍可画检测框，但不进判定。
- **BREAKING（行为）**：配置里已有合法识别 ROI 时，顺序识别会突然只数框内。非法旧框当空。缺识别 ROI 不再拦启动。
- 不升 `config_schema_version`，不加新字段。

## Capabilities

### New Capabilities

- `runtime-roi-presence-layers`: 检测时按有无组装识别过滤层和模板门控层；顺序识别也能用已保存框；空层不拦启动。

### Modified Capabilities

- `roi-edge-template-locator-inspection`: 开检不再因缺识别 ROI 失败；主画面允许同时画识别 ROI 与搜索 ROI；模板门控不再绑定认法开关；引擎的 `template_enabled` 以门控是否挂上为准。

## Impact

已定设计：`docs/superpowers/specs/2026-09-09-presence-based-runtime-roi-design.md`。  
相邻未归档变更 `decouple-template-create-from-search-roi` 只管创建/回查，不管检测加载；本变更不改配置页保存契约。

### 会改

| 文件 | 边界 |
| --- | --- |
| `src/main.py` | 只改 `_wrap_with_template_gate` 启用条件 |
| `src/config.py` | 只改 `inspection_validation_errors`：缺框/非法框不再当启动错误 |
| `src/roi_inspection/geometry.py` | 新增「矩形是否可用」 |
| `src/roi_inspection/models.py` | 空识别 ROI 当整图；`template_enabled` 跟实际门控 |
| `src/ui/main_window.py` | 开检校验、判定入口过滤、画有效识别/搜索框、模板运行时校验 |
| `src/ui/config_page.py` | **仅**开关说明文字（可选）；保存/创建/单独存框逻辑不动 |
| `tests/test_roi_config_and_geometry.py` | 「缺识别 ROI 必须失败」改为「空框整图」 |
| `tests/test_roi_runtime_integration.py` | 顺序识别过滤、主画面画框 |
| `tests/test_runtime_model_configuration.py` | 顺序识别也能挂门控 |
| 新测试 `tests/test_runtime_roi_presence.py` | 空/非法/损坏/只配搜索 等启动与过滤用例 |
| `openspec/specs/roi-edge-template-locator-inspection/spec.md` | 随本变更 delta 归档后同步 |

### 不改

- `step_sequence/step_sequence_engine.py`：数量公式、稳定帧、错序、失败重试
- `roi_inspection/engine.py`：轮次数量公式；只通过 config 入参改变「有没有框 / 有没有门控」
- `template_matching/`、`yolo_runtime/template_gated_processor.py` 算法与匹配语义
- `hand_tracking/`、Hands 启用条件（仍是顺序认法 + 检测中）
- `camera/`、ONNX 推理、打包、`config_schema_version`
- 配置页「保存当前 ROI」与整页保存不覆盖框草稿
- 不恢复 PCB / 首类别区域检查 / 标定打孔
