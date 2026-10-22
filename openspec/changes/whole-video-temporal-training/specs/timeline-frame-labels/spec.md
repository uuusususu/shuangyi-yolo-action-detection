## ADDED Requirements

### Requirement: Every frame of a participating clip has a training label
训练台 SHALL 把每条参加训练的片子的标注换算到训练时间基上，变成逐帧标签。涂了色块的帧 SHALL 是该段的类别；没被任何色块覆盖的帧 SHALL 是空闲；「不训练」剪掉的帧 SHALL 标为忽略，不进任何损失和分数。短于窗口的动作段和空档 SHALL 照样打标签，MUST NOT 丢掉。

#### Scenario: Unpainted frames are idle
- **WHEN** 一条 120 帧的片子在第 10–50 帧涂了包装、第 60–100 帧涂了封装
- **THEN** 第 0–9、50–59、100–119 帧是空闲，第 10–49 帧是包装，第 60–99 帧是封装

#### Scenario: A short gap between two actions is kept as idle
- **WHEN** 两段动作之间的空档短于 1.92 秒
- **THEN** 这个空档的每一帧都标成空闲，并参加训练

#### Scenario: Adjacent actions with no gap keep their boundary
- **WHEN** 包装在某一帧结束，封装正好从这一帧开始
- **THEN** 标签在这一帧直接换类别，中间没有空闲帧

#### Scenario: Excluded frames are ignored
- **WHEN** 一条片子第 200–260 帧被剪成「不训练」
- **THEN** 这些帧标为忽略，不进损失、不进验证分数、不计入类别频率

### Requirement: The time base never fabricates frames
用户没有手动设时间基时，训练台 SHALL 取参加训练片子里最慢的帧率作为时间基；帧率相差不超过 0.05 的算同一个时钟，取中位数。快的片子 SHALL 按时间抽帧。若有片子比时间基慢 0.05 fps 以上，训练 SHALL 被拦住，提示里写明片子名和两个帧率。时间基 SHALL 写进契约，验证、结果页、预标注都 SHALL 对齐到它。

#### Scenario: Mixed fps clips share the slowest clock
- **WHEN** 10.277 fps 和 6.248 fps 的片子一起训练
- **THEN** 时间基是 6.248 fps，10.277 fps 的片子按时间抽帧，不用重复帧去补慢的片子

#### Scenario: An explicit time base faster than a clip is refused
- **WHEN** 用户把时间基设成 10 fps，而有一条参加训练的片子是 6.3 fps
- **THEN** 训练被拦住，提示写明这条片子、6.3 fps 和 10 fps

#### Scenario: Segment boundaries follow the time base
- **WHEN** 一条 10 fps 的片子换算到 5 fps 的时间基
- **THEN** 原来第 20–60 帧的段变成时间基上的第 10–30 帧

### Requirement: Feature standardization is fitted on training clips and stored in the contract
训练台 SHALL 只用训练片子里没被忽略的帧（换算到时间基之后）算出每一维的均值和标准差；标准差小于 1e-3 的 SHALL 换成 1。均值和标准差 SHALL 以 `feature_mean`、`feature_std` 写进契约。训练、每轮验证、结果页验证、预标注 SHALL 在送进网络之前都按这组数标准化。原始特征行的格式和长度 MUST NOT 改。

#### Scenario: Validation clip does not leak into the statistics
- **WHEN** 6 号片子设成「只做验证」
- **THEN** 6 号片子的任何一帧都不参与 `feature_mean` 和 `feature_std` 的计算

#### Scenario: Contract carries the statistics
- **WHEN** 一轮新训练结束
- **THEN** `contract.json` 里有 `feature_mean` 和 `feature_std`，长度都等于 `input_size`

#### Scenario: Existing feature caches stay valid
- **WHEN** 特征已经用同一个 YOLO、手部模型和置信度阈值抽过
- **THEN** 训练直接用缓存的 `.npy`，不重新抽

### Requirement: The training page warns about unlabeled tails
训练页 SHALL 对每条参加训练的片子检查：最后一段色块之后，既没涂也没剪掉的帧连续超过 30 秒时给出提醒，说明这些帧会当空闲学，没标完请用「不训练」剪掉。这条提醒 MUST NOT 拦住训练。

#### Scenario: Long unpainted tail
- **WHEN** 一条片子最后一段色块结束后还有 45 秒，并且这段没有被剪掉
- **THEN** 训练页显示「这条片子最后 45 秒没涂，会当空闲学；没标完请用「不训练」剪掉」，开始训练按钮仍可点
