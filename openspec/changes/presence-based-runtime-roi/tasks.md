## 1. 矩形可用性与校验放宽

- [x] 1.1 在 `roi_inspection/geometry.py` 增加「矩形是否可用」，非法/`None` 返回空；补单测。
- [x] 1.2 改 `inspection_validation_errors`：缺框、非法框不再当错误；类别和数量仍校验。
- [x] 1.3 改现有「缺识别 ROI 必须失败」测试为「空框允许开检 / 整图计数」。

## 2. 门控组装与引擎入参

- [x] 2.1 改 `_wrap_with_template_gate`：不看认法开关；开关开 + 搜索有效 + 模板可加载才门控；缺一层返回原 processor。
- [x] 2.2 `RoiInspectionConfig.from_config`：识别 ROI 空则整图；`template_enabled` 以门控是否挂上为准。
- [x] 2.3 改 `_template_runtime_validation_error`：仅门控层会启用时检查 processor、模型、搜索余量。
- [x] 2.4 测试：顺序认法齐全配置挂 `TemplateGatedProcessor`；只有搜索没有模板则不门控且开检成功；模板损坏拒绝开检。

## 3. 判定入口过滤

- [x] 3.1 `_on_overlay_updated` 喂步骤引擎前按有效识别 ROI 过滤；overlay 仍保留框外目标。
- [x] 3.2 不按顺序且未挂门控时，按无模板触发，不得空等 `template_match`。
- [x] 3.3 测试：框外目标不推进步骤；非法旧框当空；不按顺序 + 空识别 ROI 整图计数；开关开但层未挂上仍能开轮。

## 4. 主画面

- [x] 4.1 两种认法下画出有效识别 ROI 与搜索 ROI；不画模板 ROI、轮廓和得分。
- [x] 4.2 测试：顺序认法 + 有效识别 ROI 时主画面含该矩形。

## 5. 回归

- [x] 5.1 配置页单独存框、整页保存不覆盖框的测试仍通过。
- [x] 5.2 Hands 仍只在顺序认法检测中启用。
- [x] 5.3 运行 `python -m pytest yolo_action_detection/tests -q`。
- [x] 5.4 运行 `openspec validate presence-based-runtime-roi --strict`。
