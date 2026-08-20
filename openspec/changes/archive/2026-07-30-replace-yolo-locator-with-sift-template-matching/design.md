## Context

当前 ROI 定位检测引擎（`roi_inspection/engine.py`）通过 `WAITING_FOR_LOCATOR -> INSPECTING -> RESULT_HOLD` 三阶段状态机驱动一轮判定。定位触发依赖 YOLO 模型识别"定位类别"并在"定位 ROI"内做精确数量匹配。这套机制要求现场为定位单独准备模型类别，且每帧都要跑 YOLO 推理。

参考项目 `E:/双翼科技项目/参考项目` 使用 HALCON 形状模板/NCC 模板匹配做定位：从静态帧框选区域创建模板，运行时 `find_shape_model` 返回行列坐标和角度，匹配成功后用固定 ROI 做后续检测。本提案用 OpenCV SIFT 实现等效流程，不引入商业库依赖。

### 现有管线

```
camera_worker._inference_loop:
  frame = 最新帧
  overlay = frame_processor.process_frame(frame)   # YOLO 推理，每帧都跑
  emit overlay_state_updated(overlay_dict)

main_window._on_overlay_updated:
  obb_dets = 从 overlay_dict 重建 ObbDetection 列表
  roi_engine.update(obb_dets, image_size)          # 引擎判定
  _handle_roi_results(results)                      # 统计/声音/证据/弹窗
```

### SIFT 模板匹配原理

SIFT（Scale-Invariant Feature Transform）提取图像局部特征点（关键点 + 描述子），对旋转、缩放、光照变化具有不变性。匹配流程：

1. 模板图和当前帧各提取 SIFT 特征点
2. FLANN KNN 匹配描述子（k=2）
3. Lowe 比率测试（ratio < 0.7）筛选 good matches
4. good matches 数量 ≥ min_matches 时，用 `cv2.findHomography` 解算单应性矩阵
5. 用单应性矩阵把模板四角变换到当前帧坐标，得到模板位置
6. 通过匹配点对的角度差统计估算旋转角，过滤超角度范围的匹配
7. score = good_matches / 模板特征点数

## Goals / Non-Goals

### Goals

- 用 SIFT 模板匹配替换 YOLO 定位类别和无定位模式
- 定位未命中时跳过 YOLO 推理，节省算力
- 模板从静态帧框选创建，不需要训练模型
- 匹配阈值、角度范围、最少匹配点数可配置
- 识别 ROI 类别数量判定、稳定帧锁存、PASS/FAIL、告警弹窗等行为不变

### Non-Goals

- 不引入 HALCON 或其他商业视觉库
- 不支持模板的涂抹/擦除轮廓编辑（参考项目 C# 工具的功能，后续可扩展）
- 不改变相机采集、生产统计、证据保存等无关模块
- 不支持多模板多目标同时匹配（当前只需单目标定位触发）

## Decisions

### D1: 模板匹配在 processor 层做，不在 camera_worker 改

`camera_worker._inference_loop` 每帧调 `frame_processor.process_frame(frame)`。新增 `TemplateGatedProcessor` 包装现有 YOLO processor：

```
TemplateGatedProcessor.process_frame(frame):
  result = sift_locator.locate(frame)
  if result.matched:
    overlay = inner_processor.process_frame(frame)   # 跑 YOLO
    overlay.template_matched = True
    overlay.template_score = result.score
    overlay.template_center = result.center
    overlay.template_angle = result.angle
  else:
    overlay = DetectionOverlayState(detections=[], error=None)
    overlay.template_matched = False
    overlay.template_score = result.score
  return overlay
```

camera_worker 完全不用改，只换 processor。匹配结果通过 overlay 的额外字段传递给引擎。

### D2: 引擎从 overlay 读模板匹配结果触发

引擎 `update` 方法新增 `template_match` 参数（可选 dict，含 matched/score/center/angle）。WAITING 阶段：

- 有模板定位配置：`template_match.matched == True` 即触发本轮
- 无模板定位（模板功能关闭）：识别 ROI 内有任意类别即触发（保留原无定位语义）

引擎本身不跑 SIFT（不持有帧），只读 processor 传来的匹配结果。

### D3: 模板创建复用 ROI 编辑器，并以保存动作即时生效

配置页 ROI 编辑器已支持鼠标框选矩形。模板创建采用明确的草稿与已保存状态：

1. 用户点击"创建模板"后进入模板框选状态。
2. 用户在静态帧上拖拽蓝色矩形 ROI；框选期间只显示蓝色选框，不显示实时裁剪预览。
3. 用户点击"保存模板"后，系统才截取该区域、验证图片和 SIFT 特征并保存模板。
4. 保存成功后保留蓝色轮廓，并显示实际从磁盘重新读取的已保存模板静态预览，让操作员确认模板内容。
5. 保存动作同时持久化 `roi_template_rect`、当前模板参数，并启用 ROI 检测和模板定位；不需要再次点击"保存配置"。
6. 再次点击"创建模板"可进入新草稿；新模板验证成功后覆盖旧模板，失败时旧模板和旧运行处理器保持不变。

模板轮廓和模板图片预览只用于配置页创建确认。主检测页面不增加模板四边形轮廓，避免干扰生产画面。

### D4: 删除旧定位字段，配置版本升至 3

删除：`roi_locator_class_name`、`roi_locator_target_count`、`roi_locator_rect`
新增：`roi_template_enabled`、`roi_template_path`、`roi_template_match_threshold`、`roi_template_angle_range`、`roi_template_min_matches`、`roi_template_rect`

`CURRENT_CONFIG_SCHEMA_VERSION` 从 2 升到 3。旧配置（版本 0/1/2）提示重新配置，不自动迁移。

### D5: SIFT 参数与性能

- SIFT 特征点提取：`cv2.SIFT_create(nfeatures=1000)`，限制特征点数量控制耗时
- FLANN 匹配：`cv2.FlannBasedMatcher`（KD-tree），KNN k=2
- Lowe 比率：0.7（标准值，不暴露为配置项）
- 单应性矩阵：`cv2.findHomography` + RANSAC
- 旋转角估算：从 good matches 的关键点角度差取中位数
- 预期性能：SIFT 在 1080p 帧上约 20-50ms，FLANN 匹配约 5-10ms，总耗时 30-60ms/帧，可接受

### D6: overlay_dict 扩展

`_overlay_to_dict` 和 `_on_overlay_updated` 增加 `template_match` 字段：
```json
{
  "template_matched": true,
  "template_score": 0.72,
  "template_center": [640.0, 512.0],
  "template_angle": 3.5
}
```
引擎和主画面绘制均从此字段读取。

### D7: 模板保存必须原子化，路径和运行处理器必须统一

模板固定命名为 `locator_template.png`，但不再由不同模块自行拼接路径。配置页、启动加载、运行时重载、校验和便携 smoke test 必须使用同一个解析函数：模板相对路径一律以实际 `config.json` 所在目录为基准。这样源码版的 `config/config.json` 与便携版 EXE 同级 `config.json` 都能找到各自模板。

保存采用以下事务顺序：

1. 将裁剪图编码到配置目录内的临时 PNG，检查写入结果。
2. 从临时文件重新读取图片，并用当前阈值、角度和最少匹配点创建 `SiftLocator`，确认模板可用。
3. 停止当前推理，原子替换正式模板文件并原子保存模板相关配置。
4. 通过唯一的运行处理器工厂重新创建模板门控处理器，设置给相机线程后恢复原检测状态。
5. 任一步失败都删除临时文件、保留旧模板和旧处理器，并向操作员显示具体失败步骤。

模板启用后禁止静默退回普通 YOLO。启动、保存后重载或开始检测时，如果模板缺失、不可读、SIFT 特征不足或门控处理器创建失败，系统必须显示明确错误并禁止开始检测。首次创建成功后立即生效；替换失败且旧模板仍有效时继续使用旧模板，并明确提示新模板未生效。

## Risks

- **SIFT 专利**：SIFT 专利已于 2020 年到期，OpenCV 4.4+ 已将 SIFT 移入主模块（非 contrib），当前 OpenCV 4.11.0 确认可用 `cv2.SIFT_create()`。
- **纹理稀疏目标**：SIFT 依赖纹理特征，若定位目标表面光滑无纹理（如纯色金属件），特征点不足可能导致匹配失败。缓解：配置 `min_matches` 可调低；极端情况可后续加 NCC 备选。
- **性能波动**：SIFT 耗时随帧分辨率和特征点数量波动。缓解：限制 nfeatures；FLANN 近似匹配速度快。
- **模板泛化**：同一产品不同批次外观差异大时，单一模板可能不够。当前方案只支持单模板，后续可扩展多模板。
- **便携版路径差异**：源码配置位于 `config/config.json`，便携配置位于 EXE 同级。缓解：所有模板路径都相对于实际配置文件目录解析，并用源码版、便携版路径测试覆盖。
- **替换期间状态不一致**：直接覆盖模板可能让推理线程读取半写文件。缓解：临时文件验证通过后原子替换，重载期间暂停推理，成功后再恢复。

## Migration

旧配置（版本 0/1/2）在启动时检测到版本不匹配，显示"配置版本无效，请重新配置"，阻止开始检测，但允许打开配置页修复。用户需要：
1. 重新选择模型、类别、识别 ROI（这些字段保留）
2. 框选创建定位模板
3. 保存后版本自动写为 3
