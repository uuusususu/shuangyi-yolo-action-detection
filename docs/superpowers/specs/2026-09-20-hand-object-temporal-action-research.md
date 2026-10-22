# 手部动作 + 目标 + 起止连贯判定：完整调研

日期：2026-09-20  
范围：对照抖音视频 `7681180931898453221` 的效果，分析本仓库现有判定为何更粗，给出可落地实现、训练与开源库选型。  
视频：https://www.douyin.com/video/7681180931898453221  
原分享链：https://www.douyin.com/user/MS4wLjABAAAAV3vBzmtMIny-dbnYRMpI_bbvAiWJ6ySAA5pCWECTD0nZfwrD7HTDW0_EWW1WiqAi?modal_id=7681180931898453221

## 1. 证据边界

已在本机用系统默认浏览器打开上述视频，供直接观看。

未能做到的部分：Cursor 内置浏览器 MCP 本轮未注册成功；无登录抓取抖音网页/接口只得到登录墙，无法在本报告里贴逐帧截图或作者文案。因此「视频里画面长什么样」以你的现场描述为准，不编造未看到的 UI 细节。

本报告已核验的部分：

- 你描述的效果：用手部动作和目标一起判，从起始到结束，目标出现不够，还要动作连贯才算完成。
- 本仓库当前判定：只认 YOLO 类别数量是否连续命中，MediaPipe 手部骨架只画不判。
- 工业界做成这种效果的主流技术路线、训练方法和 GitHub 实现库（仓库页已抓取核对）。

## 2. 对方视频在判什么

你看到的「更准」，不是框画得更漂亮，而是**完成定义不同**。

对方把一个动作拆成一段时间，而不是一张图：

```
空闲
  → 手/工具靠近目标（起始）
  → 接触或进入作业区（过程开始）
  → 持续操作：压、拧、插、对齐、停留（连贯）
  → 手离开，目标状态已变（结束）
  → 整段合法才 PASS
```

单帧「看到零件」或「手在附近晃一下」都不够。漏步骤、顺序反了、只把零件放进画面、手没有真正作业，都应失败。

这和学术上的三件事对应：

| 名称 | 它回答的问题 | 对方视频用到的部分 |
| --- | --- | --- |
| 手-物交互 HOI / 接触检测 | 手有没有在操作这个目标 | 手 + 目标同时成立 |
| 时序动作分割 / 定位 TAL | 动作从哪一帧开始、哪一帧结束 | 起始到结束 |
| 工序步骤识别 PSR | 这一步是否真正做完，而不只是做了个样子 | 连贯完成后才报完成 |

工业里常叫 SOP 防错：YOLO 看零件/工具，手部关键点看操作，时序模型或状态机看过程，最后才出 PASS/NG。

## 3. 我们现在为什么更粗

当前生产路径写在 `yolo_action_detection`：

1. YOLO OBB 检出类别（配置里当前是「散热片」等）。
2. `StepSequenceEngine.update()` 只吃 `detections`：当前步骤类别数量恰好等于配置数量，并连续 `enter_stable_frames` 帧，就 PASS。
3. MediaPipe Hands 已接到同一帧，但设计写明**不进入判定**：

```text
手部结果不进入 StepSequenceEngine，不触发 PASS/NG/统计/声音/证据。
```

见 `docs/superpowers/specs/2026-09-02-mediapipe-hands-overlay-design.md` 与 `src/hand_tracking/__init__.py`。

ROI 模式更是「识别框里数量对了就过」，同样没有手、没有过程。

仓库里其实留过更接近「接触」的旧路，但生产已停用：

- `fusion_iou_threshold`、`joint_tracking_*`：注释写明不参与生产步骤判定。
- `PointJudgementEngine`：枪头点进洞口内圈才过。这是「工具尖点 vs 目标」几何接触，仍不是「手部轨迹 + 目标状态变化」的整段动作。

对照表：

| 判定条件 | 本仓库现在 | 对方视频效果 |
| --- | --- | --- |
| 目标出现且数量对 | 足够 PASS | 只是必要条件 |
| 手部 21 点 | 只叠加显示 | 参与接近、接触、操作 |
| 起始 | 类别第一次稳定出现 | 手/工具开始靠近目标 |
| 过程连贯 | 无 | 距离、接触、停留、轨迹要连续 |
| 结束 | 稳定帧满了就过 | 手离开且目标状态已变 |
| 错序 | 后续类别先稳定出现 → NG | 子动作顺序也可检 |
| 零件摆在视野里不动 | 会 PASS | 应 NG |
| 手在旁边比划、没作业 | 只要目标在就会 PASS | 应 NG |
| 戴手套 / 手被挡住 | 不影响，因为手不判 | 必须换手部模型或改用工具框 |

所以精度差不是 YOLO 框差一点，而是**完成判据少了「过程」这一维**。把置信度从 0.8 调到 0.9 补不上这个缺口。

## 4. 怎么实现：四条路线

没有一个现成模型能直接替代你们整套现场程序。要叠四层，而不是换一个更大的 YOLO。

```
相机帧
  ├─ 目标检测（已有 YOLO OBB）：零件 / 工具 / 可选「安装前/安装后」状态
  ├─ 手部关键点（已有 MediaPipe，未接入判定）
  ├─ 帧特征：手-目标距离、接触、速度、停留、数量
  └─ 时序层：状态机 和/或 LSTM/TCN/视频模型
        └─ 只有完整相位走完才 PASS
```

### 方案 A：几何 + 相位状态机（先上，不训练新模型）

用现有 YOLO 框和现有 21 点，每帧算：

- 掌心 / 食指尖到目标 OBB 中心的距离
- 手包围盒与目标 OBB 的 IoU 或「指尖是否落在目标内」
- 手腕速度、距离是否单调变小（接近）
- 接触后停留帧数、离开后目标是否还在

相位：

```
IDLE → APPROACH → CONTACT → OPERATE → RELEASE → COMPLETE
```

规则示例（散热片安装）：

- APPROACH：手-目标距离连续 N 帧下降，且低于接近阈值
- CONTACT：指尖或掌心进入目标外扩区，速度变低
- OPERATE：接触保持 M 帧（压装）；若是拧螺丝，再加周期性小幅运动
- RELEASE：手离开，目标仍在安装位
- COMPLETE：上述相位按序走完，缺一段就 NG 或超时 NG

工作量：大约 1–2 周接到 `StepSequenceEngine` 前面或并列。不新训模型。现场可调阈值。

限制：拧、插、旋转这类细微动作，纯几何容易和「手停在旁边」混淆。规则多了难维护。

### 方案 B：YOLO + 手点 + 轻量时序分类（最贴近对方效果）

和开源项目 [AI-SOP](https://github.com/xiabie666/AI-SOP)（82 stars）同一条路，也是工业论文里常见的混合系统（YOLO 找起点，LSTM 分类手部轨迹）。

每帧拼一个特征向量，例如：

- 每个待检目标：中心、宽高、角度、置信度、是否在识别 ROI
- 每只手：21 点相对目标中心的坐标、指尖开合、手腕速度
- 手-目标最小距离、IoU、接触标记

用 16/32/48 帧滑窗送 LSTM 或 TCN，输出当前原子动作：`idle / approach / grasp / align / press / release / other`。

外层仍用状态机：模型说「当前是 press」只能推进 OPERATE，不能直接 PASS。连续命中 + 相位完整才完成。

工作量：2–4 周采集标注 + 训练 + 接入。CPU/小 GPU 都能跑。和现有 ONNX YOLO 可并行。

这是**推荐主方案**：改动面落在你们已经有的两路感知上，补的是「时间」。

### 方案 C：目标状态检测 + 工序步骤识别 PSR

不直接认「动作像不像」，而认「零件状态变没变」。

YOLO 类别从「散热片」拆成：

- `heatsink_in_hand`
- `heatsink_unmounted`
- `heatsink_mounted`
- 可选：工具、手

完成 = 状态从 unmounted 变成 mounted，并且中间有手/工具参与，之后手离开。

这就是 [IndustReal](https://github.com/TimSchoonbeek/IndustReal) 提出的 Procedure Step Recognition：普通动作识别只说「做了安装动作」，PSR 还要求「安装结果是对的」。后续工作 [STORM-PSR](https://github.com/shaohsuanhung/STORM-PSR) 用「目标状态流 + 时序流」降低完成时刻的延迟。

适合：零件装上/没装上肉眼可分（散热片坐到位、螺丝没露头）。  
不适合：状态外观几乎不变、只靠手法对错的工序。

建议和方案 B 一起用：B 管过程，C 管结果。对方视频「更准」，通常是这两件事同时成立。

### 方案 D：端到端视频动作模型

[MMAction2](https://github.com/open-mmlab/mmaction2)、[SlowFast](https://github.com/facebookresearch/SlowFast)、VideoMAE、[ActionFormer](https://github.com/happyharrycn/actionformer_release) 这类：输入一段视频，输出动作类别或起止时间。

精度上限高，但：

- 每个工位要上百段起止标注
- 换产品/换相机要重训
- 延迟和算力比滑窗 LSTM 大
- 和现有 OBB + 数量统计不好拼

只在动作种类很多、几何规则写不清、且有 GPU 训练条件时再上。不要作为第一刀。

## 5. 推荐怎么接到本仓库

按投入产出，不要一上来训 VideoMAE。

**第一刀（不训练）**

1. 把 `overlay.hand_landmarks` 送进新的 `ActionPhaseEngine`，不要再只喂 `detections`。
2. 现有 `StepSequenceEngine` 的「数量命中」改成必要条件，不再是充分条件。
3. PASS 改为：数量对 **且** 相位 `IDLE→…→COMPLETE` 走完。
4. 超时、只有目标没有接近、接触后立刻消失、手没离开就报完成，都 NG。

**第二刀（小数据训练）**

1. 若手套/遮挡导致 MediaPipe 丢手：加 RTMPose-Hand 或 YOLO-Pose，见 [MMPose RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)。
2. 若「装上/没装上」看得出：YOLO 增加状态类（方案 C）。
3. 采集 30–80 个完整循环（含错序、只摆零件、手比划），标原子动作，训 LSTM/TCN（方案 B）。

**第三刀（可选）**

动作种类 > 15，或细微手法必须看 RGB 纹理，再上 MMAction2 / ActionFormer。

和现有模块的关系：

| 现有模块 | 保留 | 改什么 |
| --- | --- | --- |
| YOLO OBB ONNX | 保留 | 可加状态类、工具类 |
| 识别 / 搜索 / 模板 ROI | 保留 | 相位计算仍可限制在识别 ROI |
| MediaPipe Hands | 保留 | 从「只显示」改为判定输入 |
| `StepSequenceEngine` | 保留顺序与错序 | PASS 增加相位完成条件 |
| `PointJudgementEngine` | 可作工具尖点补充 | 不要单独当整段动作 |
| 稳定帧 / 失败重试 | 保留 | 用在相位上，不用在「看见类别」上 |

## 6. 怎么训练

分三块，不要混成「再训一个大模型」。

### 6.1 目标检测（你们已经会）

- 数据：本工位多光照、多人员、含遮挡。
- 类别建议：零件、工具、安装位；能拆状态就拆 `*_unmounted` / `*_mounted` / `*_in_hand`。
- 标注：OBB 继续用现有流程。
- 训练：Ultralytics OBB，导出 ONNX，走现有 `onnx_obb_processor`。
- 手不必强行让 YOLO 检，除非 MediaPipe 现场不可用。

### 6.2 手部关键点

- 默认：**不训**，继续 MediaPipe Hands（已集成）。
- 戴手套、油污、强反光、俯视太远：用 RTMPose-Hand（Hand5 预训练）或 Ultralytics YOLO-Pose，再在本工位 200–500 张图上微调。
- 接触状态若规则不够：可试用 [100DOH hand_object_detector](https://github.com/ddshan/hand_object_detector) 的接触头（手在不在摸东西）。工业俯视要微调，不能直接当现场模型。

### 6.3 时序动作（方案 B 的核心）

**先定义原子动作，不要直接标「这一步完成」。**  
HA-ViD 的标法值得抄：主语 + 动词 + 被操作物 + 目标物 + 工具。例如：

- 手-拿起-散热片
- 手-对齐-安装位-散热片
- 手-压入-安装位-散热片
- 手-离开-散热片

**采集量（单工位、单动作经验值）**

| 数据 | 数量 | 作用 |
| --- | --- | --- |
| 标准完成循环 | 40–80 | 正样本 |
| 只把零件放进视野 | 15–20 | 防「看见就过」 |
| 手比划不作业 | 15–20 | 防空动作 |
| 错序 / 做到一半放下 | 15–20 | NG |
| 不同人、不同速度 | 每人至少 10 段 | 防过拟合某只手 |

**标注**

- 工具：CVAT 或 Label Studio 的视频时间轴，标 `start_frame, end_frame, action_name`。
- 同时保留 YOLO 框（可半自动：先推理再改）。
- 导出两种监督：
  1. 帧级动作类（给 LSTM/TCN）
  2. 完成时刻（手离开且状态已变的那一帧，给完成检测）

**模型**

- 入门：2 层 LSTM，隐层 128，窗长 32 帧，交叉熵。
- 边界更稳：MS-TCN / ASFormer（动作分割，每帧一个类，起止自然出现）。
- 输入用融合特征，不要一上来喂整图 RGB，数据少时整图会过拟合背景。
- 增强：时间拉伸 0.7–1.3×、丢点、轻微旋转、左右手对换。

**训练完成后怎么接到判定**

```
滑窗 LSTM 输出每帧动作类
  → 当前期望相位的类连续命中 K 帧
  → 推进相位
  → 全部相位完成 且（可选）目标状态已变
  → 步骤 PASS
```

模型负责「现在在做哪一段」，状态机负责「能不能算做完」。不要让 softmax 最大类直接打 PASS。

### 6.4 方案 D 若真要训

- 格式：ActivityNet / THUMOS 风格 `{video, start, end, label}`。
- 骨干：Kinetics 预训练 VideoMAE / SlowFast，再在本工位微调。
- 工具箱：[MMAction2](https://github.com/open-mmlab/mmaction2)、[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)。
- 定位起止：[ActionFormer](https://github.com/happyharrycn/actionformer_release)。
- 单动作至少 100+ 段，否则不如方案 B。

### 6.5 公开数据只能用来预训练，不能当现场模型

| 数据集 | 库 | 用途 |
| --- | --- | --- |
| HA-ViD | [iai-hrc/ha-vid](https://github.com/iai-hrc/ha-vid) | 工业装配、动词+物体+工具标注 |
| IndustReal | [TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal) | 含错误的工序完成 |
| Assembly101 | [assembly-101](https://github.com/assembly-101) | 大规模手-物、细粒度分段 |
| 100DOH | [ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector) | 手接触状态 |
| MECCANO | 论文配套 | 自我中心装配 |

现场散热片/扭力枪必须用自己的相机再标一轮。公开数据只适合作特征或预训练。

## 7. GitHub / 开源库（已核对）

按「能不能直接帮上这个功能」分组。星数为抓取仓库页时的公开数字。

### 最接近你们要的完整工程

| 库 | Stars | 做什么 | 和本项目的关系 |
| --- | --- | --- | --- |
| [xiabie666/AI-SOP](https://github.com/xiabie666/AI-SOP) | 82 | YOLOv8 + MediaPipe Hands + LSTM + SOP 顺序 + 连续命中 | **最该先读的完整闭环**，和推荐方案 B 同构 |
| [grgzpp/human-action-recognition-and-anticipation](https://github.com/grgzpp/human-action-recognition-and-anticipation) | 6 | 装配协作里用关键点 + YOLO 认动作 | 论文配套，体量小，可看输入怎么拼 |
| [hzhengrd/A-human-robot-collaborative-assembly-framework-with-quality-checking](https://github.com/hzhengrd/A-human-robot-collaborative-assembly-framework-with-quality-checking) | 5 | DuHa-v2 双手动作分割 + 步骤后质检 | 「过程分割 + 做完再看结果」 |

### 手部 / 手-物（感知层）

| 库 | Stars | 做什么 | 建议 |
| --- | --- | --- | --- |
| [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | 37006 | Hands 21 点、Gesture Recognizer | 已在仓库里；Gesture 是张开/点赞这类，**不够 SOP** |
| [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | 7902 | RTMPose 手部 21 点，ONNX/TRT | MediaPipe 现场不稳时换它 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 61799 | 检测 / OBB / Pose | 已在用；Pose 可当手套方案 |
| [ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector) | 297 | 100DOH：手框 + 接触状态 + 接触物体框 | 补「有没有在摸目标」 |
| [fhx020826/hand-gesture-detect](https://github.com/fhx020826/hand-gesture-detect) | — | MediaPipe + 轨迹认挥手等 | 只说明「关键点 + 历史轨迹」能做动态手势，不是工业 SOP |

### 时序动作（判定层）

| 库 | Stars | 做什么 | 建议 |
| --- | --- | --- | --- |
| [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) | 5159 | 识别 / 定位 / 骨架动作（TSM、SlowFast、VideoMAE、ST-GCN、PoseC3D） | 方案 D 或骨架动作的工具箱 |
| [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) | 7421 | 视频理解；IndustReal 基线用它 | 研究级训练 |
| [PaddlePaddle/PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) | 1701 | 中文视频工具箱，含标注与轻量模型 | 国内部署文档全 |
| [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | 576 | 时序动作定位，出起止时刻 | 要「从第几帧到第几帧」时用 |
| [yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) | 262 | 多阶段 TCN 动作分割 | 数据中等时比 LSTM 边界更稳 |
| [ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer) | 140 | Transformer 动作分割 | MS-TCN 的后续 |
| [FarnooshHeidari/CompletionDetection](https://github.com/FarnooshHeidari/CompletionDetection) | — | 完成时刻标注与论文模型 | 教你怎么标「做完的那一帧」 |

### 工序完成 / 数据集

| 库 | Stars | 做什么 | 建议 |
| --- | --- | --- | --- |
| [TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal) | 31 | PSR 任务、含错误、SlowFast/MViT 权重 | 先读任务定义再抄网络 |
| [shaohsuanhung/STORM-PSR](https://github.com/shaohsuanhung/STORM-PSR) | 6 | 状态流 + 时序流，完成更准时 | 方案 C 的实现参考 |
| [iai-hrc/ha-vid](https://github.com/iai-hrc/ha-vid) | 11 | 3222 工业装配视频，动词+物体+工具 | 标注规范直接抄 |
| [assembly-101/assembly101-download-scripts](https://github.com/assembly-101/assembly101-download-scripts) | 79 | 大规模手-物装配 | 预训练，不是现场模型 |

没有找到「克隆下来改个配置就能替代本仓库」的库。AI-SOP 是最接近的工程模板；其余是感知、时序或数据组件。

## 8. 对方更准的原因（压到本项目）

| 本仓库会误过 | 对方 / 方案 B+C 怎么挡住 |
| --- | --- |
| 散热片放在视野里不动 | 没有 APPROACH→CONTACT 轨迹 |
| 手从旁边扫过 | 接触停留不够 |
| 类别闪一下连续 1 帧 | 相位要走完，稳定帧加在相位上 |
| 装歪了但还能检出「散热片」 | 状态类仍是 unmounted，或压入相位不完整 |
| 先装后面的零件 | 顺序约束仍在，且子动作也不能跳 |

## 9. 风险

- MediaPipe 对手套、反光、远距离俯视会丢点。丢点时不要让系统「当没手、退回只看目标」，否则精度回到现在。
- 只训 LSTM、不拆相位，模型会在边界帧抖动，现场仍会误过。
- 端到端视频模型换工位就要重标，总拥有成本高于几何 + 小 LSTM。
- 公开 SOP 演示项目（含 AI-SOP）是演示闭环，不是产线认证软件；规则、超时、证据、ROI 仍要用本仓库这套。

## 10. 结论

对方视频的效果 = **目标在不在 + 手有没有按顺序做完这一段**。  
本仓库已经有目标和手，缺的是把两路合成「起始 → 连贯 → 结束」。

落地顺序：

1. 方案 A：手-目标几何相位，先把「看见就过」改掉。  
2. 方案 C：能看出来的安装状态，用 YOLO 多类锁结果。  
3. 方案 B：本工位小数据 LSTM/TCN，锁细微手法。  
4. 方案 D：只在前三步不够时再上。

第一份该打开的代码：[AI-SOP](https://github.com/xiabie666/AI-SOP)。第一份该读的任务定义：[IndustReal PSR](https://github.com/TimSchoonbeek/IndustReal)。手部升级备用：[MMPose RTMPose Hand](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)。
