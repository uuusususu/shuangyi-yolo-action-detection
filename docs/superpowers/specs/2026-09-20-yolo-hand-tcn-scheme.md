# YOLO + 手部特征 + 小时序模型：推荐实现方案

日期：2026-09-20  
问题：用手部模型和已训练 YOLO 抽连续帧特征，再训小模型，做完才 PASS。用什么模型最好。

## 结论（先用这套）

| 层 | 模型 | 训不训 | 原因 |
| --- | --- | --- | --- |
| 目标 | **现有 Ultralytics YOLO OBB ONNX** | 不训（已有） | 零件/袋/工件已经会检 |
| 手 | **MediaPipe Hands**（已接入） | 不训 | 21 点现成；现场丢手再换下一行 |
| 手（备选） | **RTMPose-m Hand ONNX** | 仅现场手套/反光时微调 | 工业俯视比 MediaPipe 稳 |
| 时序（主选） | **因果 MS-TCN++ 精简版** | 只训这个 | GitHub 上 TAS 采用最多的骨干；骨架特征上优于 Transformer；工位必须只看过去帧 |
| 时序（对照） | 2 层 LSTM-128 | 可先做通 | 实现快，相位会抖 |
| 完成 | 顺序状态机 + `done` 连续命中 | 规则，不训 | 小模型不准直接 PASS |
| 不要用 | VideoMAE / SlowFast / ASFormer（在本特征上） | — | 太重，或骨架特征上反而差 |

对外仍一个名称（套袋 / 散热片安装）。对内小模型输出相位，走完才过。

## 为什么时序头选 MS-TCN++，不选 LSTM / Transformer

要判的是**动作分割**：每一帧属于 idle / pickup / insert / done 的哪一段，起始和结束要分开。这和「整段视频一个类名」不是同一个任务。

2025 年制造场景评测（HA4M，装配工业零件，MS-TCN / MS-TCN++ / C2F-TCN / ASFormer / LTContext）：

- 输入是 **I3D 整图特征** 时，ASFormer 最好。
- 输入是 **骨架坐标** 时，**MS-TCN++ 最好**（UpSkel 上 Acc 约 94.9%，F1@60 约 92.6%）。
- **ASFormer 在骨架特征上最差。** LTContext 同样差。

我们抽的正是「手 21 点 + YOLO 框几何」，和骨架/关节是同一类输入，**不是** I3D 图像特征。因此主选 MS-TCN++，不要上 ASFormer、VideoMAE。

LSTM（AI-SOP 用的）能做，但帧级相位容易来回跳，刚拿起/刚套入的边界糊。MS-TCN++ 多阶段膨胀卷积就是为了把边界修干净。

数据只有几十条循环时：用 **精简 MS-TCN++**（2 个 stage、每 stage 8～10 层、通道 64），不要上论文里的全尺寸。参数大约 10^5 量级，CPU + ONNX 可实时。

## 整条数据流

```
相机帧
  ├─ YOLO OBB（已有 ONNX）→ 目标中心 / 框 / 类 / 置信度
  └─ MediaPipe Hands（已有）→ 最多 2 手 × 21 点
        ↓
  冻住，只当特征器
        ↓
  本帧向量 F_t（约 100 维，相对目标归一化）
        ↓
  缓冲最近 T=32 帧  →  X ∈ R^{32×F}
        ↓
  MS-TCN++ lite → 每帧相位  +  可选 done 概率
        ↓
  状态机：pickup → insert → done
  done 连续 ≥K 帧，且前面相位都到过
  且目标仍在、手已离开
        ↓
  对外：该步骤 PASS（名称仍叫「套袋」）
```

YOLO、Hands 不回传梯度。换动作只重训 TCN。

## 特征（小模型的输入，不是图片）

以当前步骤目标（袋、工件、散热片）为原点：

1. YOLO：在场 0/1、conf、cx、cy、w、h、angle（相对识别 ROI 归一化）
2. 手：2×21×(x,y)，减去目标中心，除以目标对角线；缺手填 0，另加左右手可见位
3. 关系：指尖最小距离、掌心距离、接触、手框∩目标 IoU
4. 一阶差分：Δ距离、掌心位移（靠近还是离开）

不要把 RGB 或 I3D 拼进这个小模型：制造评测里换工位时图像特征掉得比骨架狠；你们工位少样本，图像头更容易过拟合台面。

## 四个模型怎么选（只改这一层）

| 时序模型 | 输入 | 少样本 | 起止边界 | CPU | 对本项目 |
| --- | --- | --- | --- | --- | --- |
| **因果 MS-TCN++ lite** | 手+YOLO 向量序列 | 好 | 最好 | 好 | **主选（官方仓库要改成只看过去帧）** |
| 2 层 LSTM-128 | 同上 | 好 | 一般，会抖 | 好 | 第一周打通对照 |
| 普通 TCN（单 stage） | 同上 | 好 | 中 | 最好 | 数据极少时的简化 |
| ASFormer | 同上 | 差 | 骨架上差 | 一般 | 不要用 |
| ActionFormer | 视频/I3D | 要更多段 | 出起止时刻 | 重 | 以后动作种类很多再说 |
| VideoMAE / SlowFast | 原始视频 | 要很多 | 整段分类 | 重 | 不要当第一方案 |
| ST-GCN / MS-GCN | 只骨架图 | 中 | 中 | 中 | 你们还有 YOLO 框，拼成向量用 TCN 更直接 |
| MediaPipe Gesture / 单帧 MLP | 单帧手势 | — | 无时间 | 最好 | 会刚拿起就响，排除 |

手部检测本身：

| 模型 | 何时用 |
| --- | --- |
| MediaPipe Hands | 默认，已在 `camera_worker` |
| RTMPose-m Hand（MMPose，导 ONNX） | 手套、油污、远俯视丢点 |
| YOLO-Pose | 只要手框+少量点，精度不如 RTMPose |
| 100DOH 接触头 | 可选附加「有没有摸到东西」，不替代时序头 |

## 训练

标时间轴，不标检测框：

```text
video,start,end,phase
bag_001,0,30,idle
bag_001,31,70,pickup
bag_001,71,140,insert
bag_001,141,200,done
```

- 用现成 YOLO+Hands 跑录像，存 `T×F` 特征，不要存整段图去训 TCN。
- 窗长 32，步长 2；监督是窗内每一帧的相位（分割）或最后一帧相位（分类）。MS-TCN++ 用前者。
- 只拿起就放下：标 pickup→idle，不准出现 done。
- 损失：帧级交叉熵 + MS-TCN 的平滑损失（减少相位闪烁）。可选第二个头：`done` 二分类，只在完成段为正。
- 增强：时间拉伸 0.7～1.3、随机丢帧、点坐标小抖动。不要随机裁切整图。
- 导出 ONNX，和现有 `onnxruntime` 一起跑。

单工位一个动作：40～80 次完整循环 + 15～20 次半截。CPU/一张普通 GPU 数小时内可训完。

## 接到本仓库

1. 特征：`detections`（`ObbDetection.center/box/label/conf`）+ `hand_landmarks`（已有 21 点）。
2. `ActionPhaseEngine.update(detections, hands)`：写特征、推 32 帧、跑 TCN。
3. `StepSequenceEngine`：数量命中改为必要条件；PASS = 数量对 **且** 相位走到 done。
4. 手点从「只显示」改为判定输入；丢手时保持当前相位，超时 NG，禁止退回「只看目标」。

实现顺序：先 LSTM 对照把数据管线跑通 → 换成**因果** MS-TCN++ lite → 现场手套不行再换 RTMPose。

官方 [sj-li/MS-TCN2](https://github.com/sj-li/MS-TCN2) 默认卷积是双向的，会看未来帧，不能直接用于工位实时。实现时改成 causal padding（只看过去），这是 [OnlineTAS](https://github.com/QingZhong1996/OnlineTAS)（NeurIPS 2024）对 MS-TCN 的标准改法。

## GitHub 全网检索：有没有更好的「最佳」

检索时间：2026-09-20。工具：`gh search repos` + 仓库页星数。任务过滤：动作分割 / 工序时序 / 手+目标序列。星数为当时公开数字。

### 动作分割代码（按星数）

| 库 | Stars | 年份 | 输入默认 | 实时工位 | 对本项目 |
| --- | --- | --- | --- | --- | --- |
| [yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) | 262 | 2019 | I3D 特征 | 官方离线 | TAS 采用最多的骨干 |
| [nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation) | 253 | 清单 | — | — | 领域索引，2024 仍列 FACT/DiffAct |
| [sj-li/MS-TCN2](https://github.com/sj-li/MS-TCN2) | 193 | 2020 | I3D 特征 | 官方离线 | **MS-TCN++ 官方实现** |
| [cmhungsteve/SSTDA](https://github.com/cmhungsteve/SSTDA) | 158 | 2020 | I3D | 离线 | 域适应，不是工位小模型 |
| [ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer) | 140 | 2021 | I3D | 离线、更重 | 图像特征上强，**骨架上差** |
| [ZijiaLewisLu/CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT) | 112 | 2024 | I3D/长视频 | 离线 | 论文 SOTA 之一，数据与算力不对口 |
| [MCG-NJU/BCN](https://github.com/MCG-NJU/BCN) | 89 | 2020 | I3D | 离线 | 边界级联，叠在 TCN 上 |
| [xiabie666/AI-SOP](https://github.com/xiabie666/AI-SOP) | 82 | 工程 | YOLO+手点 | **实时** | 和本仓库最像，时序是 LSTM |
| [Finspire13/DiffAct](https://github.com/Finspire13/DiffAct) | 81 | 2023 | I3D + 扩散迭代 | 慢 | 多步去噪，不适合逐帧工位 |
| [LTContext/LTContext](https://github.com/LTContext/LTContext) | 50 | 2023 | I3D | 离线 | HA4M 骨架上差 |
| [Yuhan-Shen/ProTAS](https://github.com/Yuhan-Shen/ProTAS) | 40 | 2024 | 第一人称视频 | 在线工序 | 任务像，特征不是手+YOLO |
| [BenjaminFiltjens/MS-GCN](https://github.com/BenjaminFiltjens/MS-GCN) | 40 | 骨架图 | 仅关节图 | 中 | 你们还有目标框，拼向量即可 |
| [dipika-singhania/C2F-TCN](https://github.com/dipika-singhania/C2F-TCN) | 21 | 2023 | I3D | 离线 | Assembly101 基线，更重 |
| [QingZhong1996/OnlineTAS](https://github.com/QingZhong1996/OnlineTAS) | 5 | 2024 | 因果 MS-TCN | **在线** | 明确选 TCN 而不是 Transformer/扩散 |

没有搜到「YOLO + 手点 + MS-TCN」现成工位库。最接近的工程是 AI-SOP（LSTM）。最接近的在线论文实现是 OnlineTAS（因果 MS-TCN，星少）。

### 论文榜上的「更好」为什么不是本项目的更好

FACT、DiffAct、ASFormer 在 Breakfast / 50Salads / GTEA 上往往超过 MS-TCN++，awesome-list 2024 节也把 FACT 放在前面。它们赢的前提是：

- 输入是 **I3D 2048 维整图特征**，不是 100 维手+框；
- 视频很长、类别多、标注是整段做饭/做沙拉；
- 推理可以看**未来帧**（离线切段）；
- DiffAct 还要迭代十几步。

2025 制造装配评测 HA4M 把同一批模型换到**骨架坐标**上：MS-TCN++ 最好，ASFormer / LTContext 最差。我们的特征属于骨架/几何，不属于 I3D。

NeurIPS 2024 OnlineTAS 写明：做在线分割时以 MS-TCN 为底座，因为算力低于 attention / diffusion，把普通卷积换成 **causal convolution** 即可只看过去。

### 结论：GitHub 没有另一个「必须换」的最佳库

- **星数最高、被引用最多的时序骨干**仍是 MS-TCN / MS-TCN++，不是 FACT。
- **论文图像 SOTA（FACT/DiffAct）** 对本仓库特征、样本量和逐帧时延都不合适。
- **工位工程上被用得最多的同类**是 AI-SOP 的 LSTM，更简单但边界更抖；适合当对照，不适合当最终头。
- **真正该改的一点**：不要用官方离线 MS-TCN++，要用 **因果 MS-TCN++ lite**（同一结构，只看已发生的帧）。这是 GitHub 上在线 TAS 对这个家族的标准改法。

所以选这个时序，不是因为它 2026 年在所有榜上第一，而是：在「手点+YOLO、几十条循环、工位 CPU、做完才 PASS」这个约束下，GitHub 上没有更合适的现成替代；更新的高星模型赢在另一种输入和离线场景。

## 依据

- `gh search repos "action segmentation"`（2026-09-20）：ms-tcn 262、MS-TCN2 193、ASFormer 140、FACT 112、DiffAct 81。
- HA4M 制造装配分割（EAAI 2025）：骨架特征上 MS-TCN++ 优于 ASFormer / LTContext。
- OnlineTAS（NeurIPS 2024）：在线场景选因果 MS-TCN，不用 Transformer/扩散。
- AI-SOP：YOLO + MediaPipe + LSTM 工程闭环，边界弱于 TCN。
- 本仓库：YOLO OBB ONNX + MediaPipe 已同帧产出，缺的是特征序列 + 因果时序头。
