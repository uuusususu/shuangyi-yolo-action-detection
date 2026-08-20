## Context

当前定位链路由 `TemplateGatedProcessor` 每帧调用 `SiftLocator`，SIFT 通过特征点、FLANN、Lowe 比率和单应矩阵判断是否命中。配置页保存 `locator_template.png`，但只显示模板裁剪图和矩形，无法像参考项目一样把实际模型轮廓和创建得分反馈给操作员。

本变更保留 `TemplateGatedProcessor -> 原有 YOLO processor -> RoiInspectionEngine` 的业务边界，只替换前置定位器，并增加独立搜索 ROI。定位结果仍只提供布尔门控；识别 ROI 不跟随模板变换，后续业务完全沿用现有实现。

## Goals / Non-Goals

### Goals

- 在项目内形成类似 HALCON 形状模板算子族的完整 OpenCV 封装，而不是继续散落拼接匹配逻辑。
- 用真实边缘轮廓让操作员在保存后确认模板内容和创建得分。
- 通过独立搜索 ROI 限制查找范围，降低相似器件误匹配和计算量。
- 支持固定比例、可配置角度范围的模板定位。
- 模板匹配仅作定位触发，保持现有 YOLO 和轮次业务不变。
- 保留用户已有配置，避免升级或重新打包删除现场参数。

### Non-Goals

- 不支持缩放搜索。
- 不支持多模板、多匹配目标、定位目标数量或跟踪 ID。
- 不让识别 ROI 跟随模板平移或旋转。
- 不裁剪识别 ROI 后再运行 YOLO；仍使用整图 YOLO。
- 不在主检测页面显示模板轮廓、得分、角度或中心。
- 不开放 Canny、轮廓长度、平滑核等专业参数。
- 不引入 HALCON 商业依赖。

## Decisions

### D1: 封装完整的 `EdgeTemplateModel`

新增项目级模板模型，外部只使用稳定接口：

```python
model = EdgeTemplateModel.create(template_image, settings)
result = model.find(frame, search_rect)
model.save(template_path)
model = EdgeTemplateModel.load(template_path, settings)
```

`TemplateMatchResult` 至少包含：

- `matched: bool`
- `score: float`
- `center: tuple[float, float] | None`
- `angle: float | None`
- `box: tuple[tuple[float, float], ...] | None`
- `contours: tuple[np.ndarray, ...]`

UI、运行处理器和验证代码不得自行重复旋转、评分或轮廓变换逻辑。

### D2: 使用边缘图上的多角度归一化互相关

模板创建流程：

1. 模板图转灰度并做轻量高斯平滑。
2. 使用基于灰度中位数推导的自动 Canny 阈值生成边缘图。
3. 用 `findContours` 提取用于显示的真实边缘轮廓。
4. 检查有效边缘像素和轮廓长度，拒绝几乎无轮廓的模板。
5. 按配置角度范围和角度步进生成旋转后的边缘模板及有效区域 mask，并缓存于内存。

查找流程：

1. 将归一化搜索 ROI 转为当前帧像素区域并裁剪。
2. 对搜索区域使用相同预处理生成边缘图。
3. 对每个候选角度调用 OpenCV `matchTemplate(..., TM_CCORR_NORMED, mask=...)`。
4. 选择全角度最高分，达到配置阈值即命中。
5. 由最佳位置和角度生成中心、外框及变换后的真实轮廓。

比例固定为 1.0，不执行多尺度循环。

### D3: 模板 ROI、搜索 ROI 和识别 ROI 职责分离

- 模板 ROI：只定义创建模板时截取的目标。
- 搜索 ROI：只定义运行时允许查找模板的位置，与创建模板时的模板 ROI 没有坐标包含关系。
- 识别 ROI：继续用于现有 YOLO 类别数量过滤和判定，保持固定，不参与模板位姿变换。

配置页使用一个 ROI 类型下拉框、一组“创建/替换 ROI”“删除 ROI”命令和编辑器下方的“保存当前 ROI”按钮。操作员选中哪种 ROI，就只编辑并保存哪种 ROI；保存不得覆盖另外两种尚未提交的草稿。删除某个 ROI 时只删除当前下拉框所选类型。

### D4: 保存后才显示绿色模型轮廓和创建得分

拖拽编辑阶段只显示当前 ROI 草稿。下拉框选中模板 ROI 并点击“保存当前 ROI”后执行以下事务：

1. 只校验模板 ROI 和当前静态原图，不校验搜索 ROI 或识别 ROI。
2. 从当前静态原图裁剪模板，创建临时 `EdgeTemplateModel`。
3. 用临时模型在同一原图的模板 ROI 自身范围回查，使创建验证与运行时搜索 ROI 完全解耦。
4. 回查得分达到配置阈值且轮廓质量合格后，原子替换 `locator_template.png` 并保存配置。
5. 在 ROI 编辑原图上显示绿色变换轮廓、绿色外框和“创建得分”。配置页不再显示独立裁剪预览。

任一步失败时，新草稿用红色结果提示，旧模板文件、旧配置和旧运行处理器均保持不变。单独保存 ROI 不强制重载运行处理器，避免模型路径或其他尚未完成的配置反过来导致 ROI 保存失败；完成三种 ROI 和开关设置后仍通过页面底部“验证并保存”统一应用运行配置。

创建得分只表示当前静态原图上的保存验证，不宣称代表跨批次或现场长期准确率；本变更不增加独立测试按钮或持续测试模式。

### D5: 运行时仅增加前置定位门控

```text
当前帧
  -> 搜索 ROI 内 EdgeTemplateModel.find
  -> 未命中：空检测结果，跳过本帧 YOLO
  -> 命中：调用原有整图 YOLO processor
  -> 原有固定识别 ROI 类别数量判断
  -> 原有 PASS/FAIL 稳定帧、完成间隔、统计、声音和证据
```

不新增跟踪、锁定位姿、多目标分配或新的轮次状态。搜索 ROI 中物理上只存在一个定位目标，因此只返回最高分结果。

### D6: 主检测页面只显示搜索 ROI 框

模板定位启用且搜索 ROI 有效时，主画面绘制搜索 ROI 矩形。主画面不绘制模板轮廓、得分、角度、中心、模板 ROI、识别 ROI 或 ROI 内类别计数文字；类别状态继续由右侧步骤卡片显示。

配置应用成功或检测成功启动时，必须清除之前由无效 ROI 产生的旧错误提示，避免出现“画面有有效 ROI 但仍提示未配置”的矛盾状态。

### D7: 配置升级必须保留现场参数

配置版本升级到 4：

- 新增 `roi_template_search_rect: Optional[Dict[str, float]] = None`
- 新增 `roi_template_angle_step: float = 1.0`
- 保留 `roi_template_match_threshold` 和 `roi_template_angle_range`
- 删除输出中的 `roi_template_min_matches`

加载版本 3 时执行窄迁移：保留全部已有字段，仅补充搜索 ROI 和角度步进；旧的最少匹配点字段忽略且下次保存不再输出。因为无法可靠推断搜索范围，搜索 ROI 不自动生成，模板定位在用户创建搜索 ROI 前 fail closed。不得清空模型、类别、相机、识别 ROI、声音、稳定帧或完成间隔。

模板路径仍相对于实际 `config.json` 目录解析。重打包继续保留用户 `config.json` 和 `locator_template.png`。

## Risks / Trade-offs

- 边缘 NCC 对严重遮挡或轮廓变化敏感；通过搜索 ROI、角度范围和模板质量校验降低风险。
- 多角度循环会增加 CPU 开销；通过固定比例、受限搜索 ROI、角度缓存和可配置步进控制。
- 同一静态图回查得分可能偏高；界面明确称为“创建得分”，不将其解释为长期准确率。
- 自动 Canny 对极端低对比度目标可能提取不足；保存时拒绝无有效轮廓模板，要求重新框选包含清晰边缘的区域。
- 配置升级后必须创建搜索 ROI；通过自动迁移保留其他全部现场设置，避免整体重配。

## Migration Plan

1. 启动时识别版本 3 配置并迁移到内存中的版本 4，不覆盖原文件直到用户保存。
2. 配置页加载现有模板 ROI 和识别 ROI，搜索 ROI 为空并提示创建。
3. 用户依次选择三种 ROI，通过“保存当前 ROI”分别保存；三者不要求相互包含。
4. 保存模板 ROI 时，系统创建边缘模型、在模板 ROI 自身回查并显示绿色轮廓和创建得分。
5. 三种 ROI 完成后，用户启用对应开关并点击页面底部“验证并保存”，系统统一应用运行定位器。

## Open Questions

无。搜索 ROI 内只有一个定位目标；匹配固定比例并允许角度变化；模板匹配只作现有业务的前置触发；主页面只显示搜索 ROI 框。
