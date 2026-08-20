## Why

当前 ROI 定位检测的"定位触发"依赖 YOLO 模型识别定位类别（第一类别）并在定位 ROI 内做精确数量匹配。这要求现场必须为"定位"单独训练或标注一个类别，增加了模型准备成本。同时，之前为简化定位新增的"无定位模式"在空场景下仍需靠 YOLO 推理发现目标，无法真正省去推理算力。

现场固定工位相机的产品外观相对固定，适合用模板匹配做定位：从静态帧框选目标区域创建模板，运行时用 SIFT 特征点匹配定位，匹配成功才触发 YOLO 识别。这样定位不需要训练模型，且定位未命中时可以跳过 YOLO 推理，节省算力。

## What Changes

- **BREAKING**：删除"YOLO 定位类别"和"无定位模式"两种定位方式，以及对应的配置字段 `roi_locator_class_name`、`roi_locator_target_count`、`roi_locator_rect`。
- 新增"SIFT 模板匹配定位"作为 ROI 检测的唯一可选定位方式。关闭时 ROI 模式退化为"识别 ROI 内任意类别即触发"（原无定位模式的触发语义）。
- 新增模板匹配模块：用 OpenCV SIFT 提取模板特征点，FLANN KNN 匹配，Lowe 比率测试筛选 good matches，单应性矩阵解算模板位置和旋转角度。
- 新增可配置参数：匹配分数阈值、允许旋转角度范围、最少匹配点数。模板图片和参数均保存在配置目录。
- 新增"创建定位模板"交互：点击创建后在配置页静态帧上拖拽蓝色 ROI；点击保存后才显示已保存模板预览和蓝色轮廓，并立即启用模板定位。再次创建并保存时原子覆盖旧模板。
- 模板文件路径统一相对于实际 `config.json` 所在目录解析，源码版和便携版使用同一规则；模板保存、读取或 SIFT 初始化失败时保留旧模板，没有可用旧模板时禁止开始检测。
- 新增 `TemplateGatedProcessor`：包装现有 YOLO processor，每帧先跑模板匹配，匹配成功才跑 YOLO 推理；匹配结果通过 overlay 传递给 ROI 引擎。
- ROI 引擎 WAITING 阶段改为：读 overlay 中的模板匹配结果，分数达标且角度在范围内即触发本轮，进入识别 ROI 类别数量判定。
- 配置版本从 2 升级到 3；旧配置提示"配置版本无效，请重新配置"。
- 主画面在模板匹配模式下显示匹配位置十字标记和匹配分数，不画定位 ROI 框。
- 识别 ROI 内的类别数量判定、稳定帧锁存、PASS/FAIL、FAIL 告警弹窗、cooldown 续轮等行为保持不变。

## Capabilities

### New Capabilities

- `roi-template-locator-inspection`：定义 SIFT 模板匹配定位触发、模板创建与持久化、可配置匹配参数、模板门控 YOLO 推理、识别 ROI 类别数量判定与稳定帧锁存、整轮 PASS/FAIL/cooldown 续轮。

### Modified Capabilities

无。本提案取代未归档变更 `replace-first-category-with-roi-localized-inspection` 中的 YOLO 定位类别触发方案，识别 ROI 类别数量判定部分保持该提案定义的语义。

## Impact

- `yolo_action_detection/src/template_matching/`（新增）：SIFT 模板匹配定位器。
- `yolo_action_detection/src/yolo_runtime/template_gated_processor.py`（新增）：模板门控 YOLO processor 包装器。
- `yolo_action_detection/src/config.py`：删除旧定位字段，新增模板匹配配置字段，配置版本升至 3，更新校验。
- `yolo_action_detection/src/roi_inspection/models.py`：删除 locator 旧字段，新增模板定位配置；`from_config` 适配。
- `yolo_action_detection/src/roi_inspection/engine.py`：WAITING 阶段改为读 overlay 模板匹配结果触发；删除旧定位/无定位分支。
- `yolo_action_detection/src/main.py`：ROI 模式 + 模板启用时用 `TemplateGatedProcessor` 包装 processor。
- `yolo_action_detection/src/ui/config_page.py`：删除定位类别/数量/定位 ROI 控件；新增模板创建、保存后预览、即时启用和匹配参数控件。
- `yolo_action_detection/src/ui/main_window.py`：overlay 处理和画面绘制适配模板匹配。
- `yolo_action_detection/src/ui/roi_editor.py`：定位 ROI 语义改为模板框选区域。
- `yolo_action_detection/tests/`：删除旧定位测试，新增模板匹配单元测试和集成测试。
- `yolo_action_detection/config/config.json` 和 `config.example.json`：更新字段。
- 中文 README 和 DOCX 操作手册：更新定位配置流程。
- 打包脚本：模板图片纳入 datas。
