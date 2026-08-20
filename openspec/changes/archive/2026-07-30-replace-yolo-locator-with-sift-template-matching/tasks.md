# 实施任务

## 1. SIFT 模板匹配核心模块

- [x] 1.1 新建 `src/template_matching/__init__.py`，导出 `SiftLocator`、`LocatorResult`
- [x] 1.2 新建 `src/template_matching/sift_locator.py`：
  - `LocatorResult` dataclass：`matched: bool`、`score: float`、`center: tuple[float, float] | None`、`angle: float | None`
  - `SiftLocator` 类：
    - `__init__(template_path, match_threshold=0.6, angle_range=15.0, min_matches=10)`
    - 加载模板图（BGR -> 灰度），`cv2.SIFT_create(nfeatures=1000)` 提取模板特征点和描述子
    - `locate(frame) -> LocatorResult`：当前帧灰度化 -> SIFT 提取 -> FLANN KNN 匹配 -> Lowe 比率 0.7 筛选 -> good matches ≥ min_matches 时 `findHomography`(RANSAC) 解算 -> 计算模板中心 -> 估算旋转角 -> 角度在 ±angle_range 内且 score ≥ threshold 返回 matched=True
- [x] 1.3 单元测试 `tests/test_sift_locator.py`：合成模板图验证匹配成功/失败/旋转/分数计算

## 2. 配置字段与校验

- [x] 2.1 `config.py`：`CURRENT_CONFIG_SCHEMA_VERSION` 改为 3
- [x] 2.2 `config.py`：删除 `roi_locator_class_name`、`roi_locator_target_count`、`roi_locator_rect` 字段
- [x] 2.3 `config.py`：新增 `roi_template_enabled: bool = False`、`roi_template_path: str = ""`、`roi_template_match_threshold: float = 0.6`、`roi_template_angle_range: float = 15.0`、`roi_template_min_matches: int = 10`、`roi_template_rect: Optional[Dict] = None`
- [x] 2.4 更新配置和运行时校验：模板启用时校验路径、可读性、SIFT 特征、阈值、角度和 min_matches；删除旧定位类别/数量/ROI 校验
- [x] 2.5 `config/config.json` 和 `config.example.json`：更新字段，删除旧字段
- [x] 2.6 测试：更新 `tests/test_roi_config_and_geometry.py` 适配新校验

## 3. 引擎配置模型适配

- [x] 3.1 `roi_inspection/models.py`：删除 `locator_class_name`、`locator_target_count`、`locator_roi`、`has_locator`
- [x] 3.2 `roi_inspection/models.py`：新增 `template_enabled: bool`、`template_match_threshold: float`、`template_angle_range: float`、`template_min_matches: int` 字段到 `RoiInspectionConfig`
- [x] 3.3 `roi_inspection/models.py`：`from_config` 适配新字段
- [x] 3.4 `roi_inspection/models.py`：`RoiInspectionSnapshot` 删除 `locator_observed_count`/`locator_required_count`，新增 `template_matched: bool`、`template_score: float`

## 4. TemplateGatedProcessor 包装器

- [x] 4.1 新建 `src/yolo_runtime/template_gated_processor.py`：
  - `TemplateGatedProcessor` 类：持有 `SiftLocator` + 内部 YOLO processor
  - `process_frame(frame, source_frame_id)`：先 `locator.locate(frame)`，匹配成功调内部 processor 并在 overlay 附加 `template_matched/score/center/angle`；失败返回空 overlay 附带 `template_matched=False`
  - `get_class_names()` 透传
- [x] 4.2 `yolo_runtime/yolo_result_models.py`：`DetectionOverlayState` 新增 `template_matched: bool = False`、`template_score: float = 0.0`、`template_center: Optional[Tuple] = None`、`template_angle: Optional[float] = None` 字段
- [x] 4.3 `camera/camera_worker.py`：`_overlay_to_dict` 序列化模板匹配字段
- [x] 4.4 测试 `tests/test_template_gated_processor.py`：mock SiftLocator 和内部 processor，验证匹配成功跑 YOLO、失败跳过

## 5. 引擎触发逻辑适配

- [x] 5.1 `roi_inspection/engine.py`：`update` 方法新增 `template_match: Optional[dict]` 参数
- [x] 5.2 `roi_inspection/engine.py`：WAITING 阶段改为：
  - 模板启用：`template_match.matched == True` 触发
  - 模板关闭：识别 ROI 内任意类别触发（保留原无定位语义）
  - 删除 `_check_locator_trigger`、`_check_any_category_trigger`，新增 `_check_template_trigger` 和 `_check_inspection_roi_trigger`
- [x] 5.3 `roi_inspection/engine.py`：日志适配模板匹配字段
- [x] 5.4 测试：更新 `tests/test_roi_inspection_engine.py`，删除旧定位测试，新增模板触发/无模板触发测试

## 6. main.py 接入

- [x] 6.1 `main.py`：统一运行处理器工厂；ROI 模式 + 模板启用时必须返回 `TemplateGatedProcessor`，不得在启动入口额外二次包装
- [x] 6.2 `main.py`：新增统一模板路径解析（相对于实际 `config.json` 所在目录），供 UI、启动、重载和 smoke test 复用
- [x] 6.3 `main.py`：`run_portable_smoke_check` 适配模板配置校验
- [x] 6.4 模板启用但缺失或加载失败时 fail closed，修复启动包装日志变量错误，不得静默退回普通 YOLO
- [x] 6.5 测试源码版与便携版模板路径、启动模板包装和失败阻断

## 7. 主窗口适配

- [x] 7.1 `ui/main_window.py`：`_on_overlay_updated` 读 overlay 的 `template_match` 字段传给引擎
- [x] 7.2 `ui/main_window.py`：`_draw_roi_overlay` 模板启用时画十字标记和分数，不画定位 ROI 框
- [x] 7.3 `ui/main_window.py`：`_init_roi_engine` 用新配置创建引擎
- [x] 7.4 `ui/main_window.py`：`_roi_engine_config_signature` 适配新字段
- [x] 7.5 `ui/main_window.py`：删除 `_roi_engine_snapshot_has_locator` 等旧方法
- [x] 7.6 `ui/main_window.py`：`_sync_roi_cards` 适配 snapshot 新字段
- [x] 7.7 `ui/main_window.py`：保存模板后暂停推理、重建统一运行处理器、切换成功后恢复原检测状态；失败保留旧处理器

## 8. 配置页 UI

- [x] 8.1 `ui/config_page.py`：删除定位类别输入框、定位数量输入框、定位 ROI 创建/删除按钮
- [x] 8.2 `ui/config_page.py`：新增"启用模板定位"开关
- [x] 8.3 `ui/config_page.py`：新增"创建模板"和"保存模板"两阶段交互；创建后只框选，不显示实时预览
- [x] 8.4 `ui/config_page.py`：新增匹配阈值、角度范围、最少匹配点数输入框
- [x] 8.5 `ui/config_page.py`：保存并验证成功后才显示磁盘模板静态预览并保留蓝色轮廓
- [x] 8.6 `ui/config_page.py`：`_load_values` 和 `_apply` 适配新字段
- [x] 8.7 `ui/config_page.py`：模板启用时联动禁用/启用匹配参数输入框
- [x] 8.8 测试：更新 `tests/test_roi_config_page.py`
- [x] 8.9 模板写入使用同目录临时 PNG、读回和 SIFT 初始化验证后原子覆盖；失败保留旧模板
- [x] 8.10 保存模板时立即启用 ROI/模板定位、持久化模板相关配置并发出运行时重载信号，不依赖"保存配置"

## 9. ROI 编辑器适配

- [x] 9.1 `ui/roi_editor.py`：`RoiKind.LOCATOR` 语义改为"模板框选区域"（用于创建模板时框选）
- [x] 9.2 `ui/roi_editor.py`：新增 `capture_rect_frame(rect_kind) -> np.ndarray` 方法，截取当前静态帧指定区域返回像素图

## 10. 集成测试

- [x] 10.1 `tests/test_roi_runtime_integration.py`：适配模板匹配定位流程
- [x] 10.2 新增模板创建 -> 定位触发 -> YOLO 识别 -> PASS/FAIL 全链路测试
- [x] 10.3 模板未匹配时跳过 YOLO、不触发的测试
- [x] 10.4 新增创建 -> 拖拽 -> 保存后预览 -> 即时重载 -> 覆盖成功/失败回滚的集成测试

## 11. 文档与打包

- [x] 11.1 `README.md`：删除 YOLO 定位类别和无定位模式说明，加 SIFT 模板匹配定位说明
- [x] 11.2 `docs/build_simple_user_guide.py`：第六章改为模板匹配配置流程
- [x] 11.3 `packaging/yolo_action_detection.spec`：无默认模板；用户模板由打包脚本保留和恢复，不写入 datas
- [x] 11.4 重新生成 docx 手册
- [x] 11.5 重新打包验证
