# 顺序模式：MediaPipe Hands 骨架叠加

日期：2026-09-02  
状态：已实现  
分支：`feature/mediapipe-hands-overlay`  
范围：仅顺序模式（`roi_inspection_enabled == false`）下，开始检测后叠加手部 21 点骨架。不改步骤判定、ROI、形状匹配。

## 背景

顺序检测已用 YOLO OBB 做零件类别与数量判定。现场还希望在**同一检测画面**上看到手部关节，便于观察操作姿态。本阶段只做识别与显示，不做任何手势业务。

## 目标

- 关 ROI（顺序模式）且用户点击「开始检测」后，自动启用 MediaPipe Hands。
- 对与 YOLO **同一帧**跑手部关键点，在主画面画 21 点 + 手指连线。
- 手部结果**不进入** `StepSequenceEngine`，不触发 PASS/NG/统计/声音/证据。
- Hands 初始化或推理失败时：只打日志，**不阻断**顺序检测与 YOLO。

## 非目标

- 不识别具体手势（抓、放、指向等）。
- 不用手部结果约束步骤顺序或数量。
- 不在 ROI 定位检测模式下启用 Hands。
- 本阶段配置页不加开关（启用条件写死为顺序模式 + 检测中）。
- 不升 `config_schema_version`。

## 启用条件

| 条件 | 行为 |
| --- | --- |
| `roi_inspection_enabled == false` 且 `inference_on == true` | 启 Hands（懒加载） |
| 停止检测 | 停用 / 可释放 Hands 会话 |
| 切换到 ROI 模式或保存后变为 ROI | 不跑 Hands；若已加载可释放 |

## 架构（方案 A）

```
相机帧
  → CameraWorker 推理线程
      → 现有 YOLO（或 TemplateGated 仅 ROI 路径；顺序模式无门控）
      → 同帧 MediaPipe Hands（仅顺序模式）
      → overlay 字典增加 hand_landmarks（可选）
  → MainWindow._draw_overlay
      → 画 YOLO 框/中文标签
      → 若有 hand_landmarks：画关节点与骨架线
  → _on_overlay_updated
      → 仍只用 detections 喂 StepSequenceEngine
```

手部数据只服务显示层。

## 模块

新建 `yolo_action_detection/src/hand_tracking/`：

| 文件 | 职责 |
| --- | --- |
| `mediapipe_hands.py` | 封装 Hands：init / process(bgr) / close |
| `__init__.py` | 导出公共类型 |

建议接口（实现时可微调命名，语义不变）：

```python
@dataclass(frozen=True)
class HandLandmarks:
    """一只手的 21 个像素坐标点（相对当前帧宽高）。"""
    points: tuple[tuple[float, float], ...]  # 长度 21
    handedness: str  # "Left" / "Right" / ""

class MediaPipeHandsTracker:
    def __init__(self, *, max_num_hands: int = 2, ...): ...
    def process(self, frame_bgr) -> list[HandLandmarks]: ...
    def close(self) -> None: ...
```

依赖：`requirements.txt` 增加 `mediapipe`。模型资源随 mediapipe 包或官方默认路径加载；失败则 tracker 标记不可用。

## 接线

### CameraWorker

在 YOLO 产出 overlay 字典、emit `overlay_state_updated` 之前：

1. 若当前不应启用 Hands → 不调用、不附字段。
2. 若应启用且 tracker 未创建 → 懒加载；失败则记日志，本会话不再重试或按退避重试（实现计划选定一种，默认：失败后本会话禁用并打一次 error）。
3. 成功则 `overlay["hand_landmarks"] = [...]`（可序列化为 list[dict]，便于信号跨线程）。

Worker 需要知道「是否顺序模式」。任选其一（实现计划锁定）：

- 由 MainWindow 在开始/停止检测或配置应用时调用 `worker.set_hand_overlay_enabled(bool)`；或
- Worker 持有对 config 的只读引用，读 `roi_inspection_enabled` 与 `inference_on`。

推荐显式 `set_hand_overlay_enabled`，避免 worker 推断错误。

### MainWindow

- `_on_toggle_detect` 开始检测且非 ROI：`set_hand_overlay_enabled(True)`。
- 停止检测或切 ROI：`set_hand_overlay_enabled(False)`。
- `_draw_overlay`：读取 `hand_landmarks`（若 overlay 对象/旁路缓存带该字段），用 OpenCV 画圆点与 MediaPipe 标准手指连线；颜色固定（如青色），线宽与点半径跟分辨率可简单固定。
- `_on_overlay_updated`：**禁止**把 hand 结果传入 `step_engine.update`。

同帧显示：Hands 必须使用与本次 YOLO 相同的 `frame` / `source_frame_id`，不得另取「最新预览帧」。

## Overlay 载荷约定

```json
{
  "hand_landmarks": [
    {
      "handedness": "Right",
      "points": [[x0, y0], ..., [x20, y20]]
    }
  ]
}
```

- 坐标：像素，浮点或整型均可，绘制时转 int。
- 无手：`[]` 或省略字段。
- 旧客户端忽略未知字段。

## 打包

- `requirements.txt` / 构建环境安装 `mediapipe`。
- `packaging/yolo_action_detection.spec`：确认 mediapipe 及其二进制/数据被收集；若 PyInstaller 漏文件，在 spec 的 `datas`/`hiddenimports` 补齐（实现时按实际告警修）。
- Hands 缺失时：源码与便携包均应降级为「无骨架」，检测仍可用。

## 测试

1. 单元：给定假帧或夹具图，tracker 返回结构长度为 0 或 21 点（可用 mock mediapipe）。
2. Worker：顺序模式 enabled 时 overlay 含 `hand_landmarks` 键；ROI/未检测时不含或为空且不调用 process。
3. UI：`_draw_overlay` 在提供 landmarks 时调用画线逻辑（可对 draw helper 单测，或对 frame 非零像素断言）。
4. 回归：现有 `test_fail_retry` / `test_step_quantity` / 顺序卡片可见性不被破坏。
5. 失败注入：mediapipe import 失败 → 检测仍启动。

## 实现顺序

1. 建分支（已建：`feature/mediapipe-hands-overlay`）、依赖、`MediaPipeHandsTracker`。
2. Worker 接线 + enable 开关。
3. MainWindow 启停与绘制。
4. 打包验证与失败降级。
5. 测试与 README 一句说明。

## 决策记录

| 项 | 结论 |
| --- | --- |
| 模型 | MediaPipe Hands（21 点） |
| 挂载 | YOLO 同帧之后（方案 A） |
| 业务 | 无 |
| 开关 | 顺序模式 + 开始检测自动开 |
| ROI | 不启用 |
| 失败 | 不阻断 YOLO/顺序判定 |
