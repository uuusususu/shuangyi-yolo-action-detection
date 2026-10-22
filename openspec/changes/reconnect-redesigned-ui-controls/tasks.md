## 1. Inventory and tests first

- [x] 1.1 Add a control-inventory test that main window and config page construct `KpiRow`, `RecognitionListItem`, `ConfigNavButton`, `FormGroup`, `SwitchRow`, `FooterActionBar`, and `RoiEditorWidget`, and never construct `StepCard`, `PanelSection`, `MetricCard`, or `DefectChip`.
- [x] 1.2 Add a recognition-list test for normal sequence mode: configured waiting items stay hidden with `0 / N` before any detection, then appear in detection order.
- [x] 1.3 Add a KPI test that capacity shows `inspected / today_target_capacity` after save.
- [x] 1.4 Add or extend a config round-trip test for UI-editable production fields and assert legacy torque-gun, hole, calibration, fusion, and joint-tracking fields are absent from the form.

## 2. Reconnect runtime list and KPI

- [x] 2.1 Change `_refresh_steps()` / `_update_step_display()` so unrecognized categories stay hidden; engines reveal an item only after quantity, PASS, or NG.
- [x] 2.2 Keep ROI mode cards hidden while waiting for locator or inspection trigger, and reveal only recognized or NG categories.
- [x] 2.3 Update `KpiRow` capacity to `batch.total / today_target_capacity` and refresh it after config save and stats reset.

## 3. Lock save/apply and field boundaries

- [x] 3.1 Verify validate-and-save still writes only UI-editable production fields and does not overwrite independently saved ROI rectangles.
- [x] 3.2 Verify return-to-detection without save keeps the last saved `ConfigManager` values and only reapplies saved engines.
- [x] 3.3 Confirm portable config preparation still strips `pcb_inspection_*` and `first_category_region_check_*`.

## 4. Retire leftover widgets

- [x] 4.1 Remove unused production imports/usages of `StepCard`, `PanelSection`, `MetricCard`, and `DefectChip`, or delete the classes if nothing else references them.
- [x] 4.2 Confirm PyQtDarkTheme still owns standard control chrome and token QSS is limited to panels and PASS/NG colors.

## 5. Spec alignment and verification

- [x] 5.1 Run `python -m pytest yolo_action_detection/tests -q` and `python -m compileall yolo_action_detection/src`.
- [x] 5.2 Run `openspec validate reconnect-redesigned-ui-controls --strict`.
- [x] 5.3 Manually confirm the six config nav labels still match 模型与步骤 / 动作判定 / 工业相机 / 显示与反馈 / ROI 配置 / 生产统计, and that camera, detect, save, and reset still work on the source-run path.
