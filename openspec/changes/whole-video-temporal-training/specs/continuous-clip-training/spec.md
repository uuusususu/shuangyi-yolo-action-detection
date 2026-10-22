## ADDED Requirements

### Requirement: Training samples are continuous crops of whole clips with full left context
每一步训练 SHALL 从训练片子里抽一小批连续截段。每个截段 SHALL 由目标段和它前面的上文组成：上文长 `context_frames` 帧，片子开头不够长时 SHALL 从片子第一帧开始。损失 SHALL 只算在目标段里没被忽略的帧上。目标段的起点 SHALL 随机选，包括从片子第一帧开始的情况。每条训练片子的每一帧 SHALL 都有机会成为目标帧。

#### Scenario: A target frame sees the same left context as in whole-clip inference
- **WHEN** 一个截段的目标段从第 500 帧开始，`context_frames` 是 180
- **THEN** 送进网络的输入从第 320 帧开始，损失只算第 500 帧及之后的帧

#### Scenario: Crops near the clip start
- **WHEN** 目标段从第 40 帧开始，`context_frames` 是 180
- **THEN** 输入从第 0 帧开始，前面不补任何虚构的帧

#### Scenario: Crops contain action changes
- **WHEN** 一条训练片子每隔几秒就在封装、空闲、包装之间切换
- **THEN** 默认长度的截段里有多次类别切换，切换两侧的帧都进损失

### Requirement: Each epoch performs many optimizer steps
一轮 SHALL 定义为：抽到的截段目标帧总数不少于所有训练片子里没被忽略的帧数。截段 SHALL 组成小批（默认每批 8 段），每一小批 SHALL 更新一次权重。训练页上的「训练轮数」SHALL 仍是跑多少轮。随机种子 SHALL 固定，同样的数据和设置抽到同样的截段。

#### Scenario: More than one step per epoch
- **WHEN** 训练片子一共 6000 个有效帧，目标段长 300 帧，每批 8 段
- **THEN** 一轮至少抽 20 个截段、至少更新 3 次权重

#### Scenario: Reproducible crops
- **WHEN** 同一个工程用同样的设置和种子训练两次
- **THEN** 两次抽到的截段位置和顺序完全一样

### Requirement: Loss is class-weighted frame cross-entropy, with smoothing for causal MS-TCN
训练损失 SHALL 是目标帧上的逐帧交叉熵，按类别加权：权重是该类在训练片子里帧数的平方根倒数，再归一化成平均为 1。网络是因果 MS-TCN 时，SHALL 再加上相邻帧 log 概率的截断均方平滑项（τ = 4，λ = 0.15），各阶段相加，只算目标帧。LSTM MUST NOT 加平滑项。报给页面的验证损失 SHALL 是验证片子上不加权、不加平滑的逐帧交叉熵。

#### Scenario: Idle does not drown out actions
- **WHEN** 训练片子里空闲帧是包装帧的 4 倍
- **THEN** 一帧包装的交叉熵权重大约是一帧空闲的 2 倍

#### Scenario: Smoothing only on MS-TCN
- **WHEN** 选的网络是 LSTM
- **THEN** 训练损失里没有平滑项，契约记下 `smoothing: false`

### Requirement: Both networks are causal and sized in seconds
训练页 SHALL 可以选 LSTM 或因果 MS-TCN，两者都 SHALL 只用当前帧和之前的帧。因果 MS-TCN 每阶段的空洞层数 SHALL 取 4 到 10 之间最大的那个值，使两阶段合起来往回看的长度不超过 `context_seconds`（默认 30 秒）换算到时间基的帧数；`context_frames` SHALL 等于这个往回看的长度。LSTM 的 `context_frames` SHALL 等于 `context_seconds` 换算到时间基的帧数。契约 SHALL 记下 `network`、`context_frames`，MS-TCN 还 SHALL 记下 `mstcn_stages`、`mstcn_layers`、`mstcn_channels`，让每个加载的地方都能还原出同一个网络。

#### Scenario: MS-TCN depth follows the time base
- **WHEN** 时间基是 6.25 fps，`context_seconds` 是 30
- **THEN** 每阶段取两阶段合起来往回看不超过 187 帧的最大层数，这个长度写进 `context_frames`

#### Scenario: Future frames do not change past outputs
- **WHEN** 改动第 t 帧之后的输入，再把同一条片子跑一遍
- **THEN** 两种网络第 t 帧及之前每一帧的类别概率都不变

#### Scenario: Old MS-TCN runs still load
- **WHEN** 契约是 `network: "mstcn"`，但没有 `mstcn_layers`
- **THEN** 加载时按当时训练的 8 层还原，不报错

### Requirement: Training extracts stale features first and reports progress
点开始训练 SHALL 先给特征过期的参加训练的片子抽特征，显示帧进度，再训练，显示轮进度。点停止 SHALL 保留到目前为止最好的一轮。这一轮用的设置、划分和网络 SHALL 写进这一轮的文件夹。

#### Scenario: Train after annotating new clips
- **WHEN** 新标了两条片子、还没抽过特征，用户点开始训练
- **THEN** 任务条先显示「抽取特征」和帧数，再显示「训练」和轮数，这一轮的文件夹里有 `best_lstm.pt`、`contract.json`、`split.json`、`metrics.json`、`log.txt`
