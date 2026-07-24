## Why

当前 ROI 定位使用 SIFT 特征点匹配。现场创建的模板可能只有很少的特征点，但仍能通过最低校验；配置页保存后主要显示裁剪图片，操作员看不到算法实际采用的轮廓，也无法通过创建得分判断模板是否生成成功。现场模板曾只有 11 个 SIFT 特征点，运行得分长期约为 0.09，说明当前创建反馈不足以支持可靠调试。

参考项目采用成熟的形状模板工作流：模板 ROI 用于创建边缘模型，搜索 ROI 限制运行时查找范围，创建后在原图显示模型轮廓和得分。OpenCV 没有 HALCON 的单个完整形状模板算子，因此本变更在项目内封装一个完整的边缘 NCC 模板模型接口，统一负责创建、验证、保存、加载、查找和结果输出。

## What Changes

- **BREAKING**：用“固定比例、多角度、搜索 ROI 内的边缘 NCC 模板匹配”替换 SIFT + FLANN + Homography 定位，不再使用“最少 SIFT 匹配点数”。
- 新增独立的“搜索 ROI”。模板 ROI 必须完全位于搜索 ROI 内；运行时只在搜索 ROI 中查找模板。
- ROI 配置页改为统一的 ROI 类型下拉框，支持选择“识别 ROI / 模板 ROI / 搜索 ROI”后创建或替换。
- 点击“保存模板”后才创建正式模板。保存时从模板 ROI 提取真实边缘轮廓，并在同一静态原图的搜索 ROI 内执行一次回查。
- 保存验证成功后，在 ROI 编辑原图上显示绿色模板轮廓、绿色外框和创建得分；不再仅依赖裁剪图片说明模板内容。
- 保存验证失败时显示红色结果和具体原因，不覆盖旧模板、旧配置或旧运行处理器。
- 运行时每帧先在搜索 ROI 内做模板匹配；匹配成功才调用现有整图 YOLO 处理器，匹配失败跳过该帧 YOLO。
- 模板匹配只作为定位触发，不改变识别 ROI、类别数量判断、PASS/FAIL 稳定帧、完成一轮间隔、统计、声音和证据保存业务。
- 主检测页面只显示搜索 ROI 框；不显示模板轮廓、模板得分、角度、定位中心或识别 ROI 框。
- 配置仅开放“匹配阈值、允许角度范围、角度步进”；Canny 和轮廓质量参数使用程序内成熟默认值。
- 配置版本升级并自动迁移现有版本 3 配置，保留模型、类别、相机、识别 ROI、声音、PASS/FAIL 帧数和完成间隔；新增搜索 ROI 初始为空，要求操作员创建后才能启用模板定位。
- 修复配置已有效并成功开始检测后仍残留旧“识别 ROI 未配置”提示的问题。

## Capabilities

### New Capabilities

- `roi-edge-template-locator-inspection`：定义搜索 ROI 内的边缘 NCC 模板创建、保存后轮廓/得分确认、多角度定位触发、主页面搜索 ROI 显示，以及对现有 ROI 数量判定业务的无侵入门控。

### Modified Capabilities

无。该能力取代尚未归档变更 `replace-yolo-locator-with-sift-template-matching` 中的 SIFT 定位实现，但不修改现有 ROI 类别数量判定语义。

## Impact

- `yolo_action_detection/src/template_matching/`：用完整边缘 NCC 模板模型替换 SIFT 定位器，提供创建、查找、轮廓和得分结果。
- `yolo_action_detection/src/yolo_runtime/template_gated_processor.py`：改为在搜索 ROI 中调用边缘模板模型，命中后继续调用原有 YOLO。
- `yolo_action_detection/src/config.py`：新增搜索 ROI、角度步进和版本迁移；删除 SIFT 最少匹配点配置。
- `yolo_action_detection/src/ui/roi_editor.py`：新增搜索 ROI 类型和保存后模板轮廓/得分覆盖层。
- `yolo_action_detection/src/ui/config_page.py`：新增统一 ROI 类型下拉框及保存后创建验证反馈。
- `yolo_action_detection/src/ui/main_window.py`：主画面仅显示搜索 ROI 框，并清理成功启动后的旧错误提示。
- `yolo_action_detection/config/*.json`、README、DOCX 手册、测试和打包验证同步更新。
