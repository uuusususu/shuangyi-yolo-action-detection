## Why

现场创建定位模板时被搜索 ROI 绑住：没有搜索框就不能保存模板，回查也必须在搜索框里做，默认框还没确认就可能进配置。说明书和 `roi-edge-template-locator-inspection` 已经要求三种 ROI 独立，代码后来偏离了。

## What Changes

- 保存模板 ROI 只裁模板框、建模，并在模板 ROI 自身范围回查；不再要求搜索 ROI 已创建、已保存或包住模板。
- 搜索框平移余量只在开始检测 / 启用模板定位时检查，保存搜索 ROI 或模板 ROI 时不再因此拒绝。
- 未创建的 ROI 不出现在编辑器里，也不写入配置。点「创建/替换」只产生未保存草稿；只有「保存当前 ROI」才落盘对应一种框。
- 整页「验证并保存」继续不覆盖三种 ROI 草稿。

## Capabilities

### New Capabilities

- 无。本次只纠正现有 ROI 模板定位配置契约。

### Modified Capabilities

- `roi-edge-template-locator-inspection`: 明确模板创建与搜索 ROI 解耦，以及未创建前不得持久化或显示 ROI 框。

## Impact

- `yolo_action_detection/src/ui/config_page.py`：模板保存回查、搜索框保存校验、创建草稿不落盘。
- `yolo_action_detection/src/ui/main_window.py`：开检时继续检查已保存搜索 ROI 是否够大。
- `yolo_action_detection/tests/test_roi_config_page.py`：去掉「必须先有搜索框才能存模板」的断言，补独立创建与空框测试。
- `openspec/specs/roi-edge-template-locator-inspection/spec.md`：与现场确认的独立创建行为对齐。
