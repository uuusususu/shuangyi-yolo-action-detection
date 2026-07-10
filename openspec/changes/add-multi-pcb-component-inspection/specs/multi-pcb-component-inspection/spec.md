## ADDED Requirements

### Requirement: 可配置的 PCB 检查模式
系统 MUST 允许操作员启用 PCB 检查模式，并配置一个 PCB 类别名称、恰好四个非空且互不重复的元器件类别名称、FAIL 连续帧数和非负的新一轮间隔时间。PCB 类别不得与任一元器件类别相同；配置无效时，系统 MUST 阻止开始 PCB 生产判定并显示具体错误。

#### Scenario: 保存有效的 PCB 检查配置
- **WHEN** 操作员填写一个 PCB 类别、四个不同的元器件类别、正整数 FAIL 帧数和非负轮次间隔并保存
- **THEN** 系统持久化配置并允许 PCB 检查模式启动

#### Scenario: 拒绝重复或缺失类别
- **WHEN** PCB 类别为空、元器件类别不足四个、存在空值、存在重复值或与 PCB 类别冲突
- **THEN** 系统拒绝启动 PCB 检查并明确指出无效字段

### Requirement: 单模型单帧推理
PCB 检查模式 MUST 对每个选中的完整相机帧至多调用一次当前 YOLO OBB 模型，并 MUST 使用该次输出同时处理画面中的全部 PCB 和配置元器件。系统 MUST NOT 为单块 PCB 启动第二模型或额外裁剪推理。

#### Scenario: 同帧出现多块 PCB
- **WHEN** 一个推理帧中检测到多块保持间隔的 PCB 及其元器件
- **THEN** 系统使用同一份推理结果更新全部 PCB 的独立检查状态

### Requirement: PCB 是唯一的持续产品身份
系统 MUST 仅使用 PCB 检测的有效跟踪 ID 创建和维护产品检查状态。元器件检测的 tracker ID MUST NOT 参与业务归属或产品去重；元器件的业务身份 MUST 由所属 PCB ID 与配置类别共同确定。

#### Scenario: 元器件 ID 在连续帧发生变化
- **WHEN** 同一物理元器件在连续帧中获得不同 tracker ID，但类别和所属 PCB 保持一致
- **THEN** 系统继续把它作为同一 PCB 的同一逻辑槽位观测，不重置 PCB 状态

#### Scenario: PCB 没有有效跟踪 ID
- **WHEN** 检测到 PCB 但该检测没有有效跟踪 ID
- **THEN** 系统不为该检测创建可判定产品状态，也不产生 PASS 或 FAIL

### Requirement: 元器件必须唯一归属到 PCB
系统 MUST 根据元器件中心点、元器件与 PCB 归属区域的交叠比例以及候选得分差，把每个配置元器件分配给至多一块 PCB。未达到归属阈值或候选不唯一的元器件 MUST 保持未归属，并 MUST NOT 为任何 PCB 提供槽位存在证据；若歧义候选包含某块 PCB，系统 MUST 将该 PCB 的本帧观测视为不可靠而不累计 FAIL。

#### Scenario: 元器件明确位于一块 PCB 内
- **WHEN** 元器件中心和主要检测区域只落在一块 PCB 的允许归属区域内
- **THEN** 系统把该元器件类别记录为该 PCB 对应逻辑槽位的本帧观测

#### Scenario: 元器件位于所有 PCB 之外
- **WHEN** 元器件没有达到任何 PCB 的归属阈值
- **THEN** 系统忽略该元器件的产品完整性贡献

#### Scenario: 归属候选存在歧义
- **WHEN** 一个元器件对两块 PCB 的归属得分均合格且差值不足以唯一选择
- **THEN** 系统不分配该元器件、记录归属歧义诊断，并且不为两块候选 PCB 累计本帧 FAIL

### Requirement: 仅评估完整可见的 PCB
系统 MUST 仅在 PCB 具有有效跟踪 ID 且 PCB 多边形完整处于画面安全区域内时，把该帧计为有效检查帧。部分进入、部分离开或无法可靠确定归属区域的 PCB MUST 保持等待，且 MUST NOT 增加 FAIL 连续帧数。

#### Scenario: PCB 正在进入画面
- **WHEN** PCB 多边形仍接触画面安全边界或未完整可见
- **THEN** 系统保持该 PCB 为等待状态，不因暂时看不到元器件而判 FAIL

#### Scenario: PCB 完整进入可检查区域
- **WHEN** PCB 具有有效跟踪 ID 且完整位于画面安全区域内
- **THEN** 系统使用该帧的元器件归属结果更新该 PCB 的完整性状态

### Requirement: 按四个逻辑槽位判断 PASS 和 FAIL
系统 MUST 将四个配置元器件类别视为四个逻辑槽位。在任一有效检查帧中四个槽位全部存在时，系统 MUST 立即为该 PCB 生成 PASS 决策；任意槽位缺失时，系统 MUST 增加该 PCB 的连续 FAIL 计数，并在计数达到配置阈值时生成 FAIL 决策。默认 FAIL 阈值 MUST 为三帧。

#### Scenario: 正常 PCB 在首个有效帧完整
- **WHEN** 一块 PCB 在首个有效检查帧中同时归属到四个配置元器件类别
- **THEN** 系统立即生成一次 PASS 决策，不再等待额外帧

#### Scenario: PCB 连续三个有效帧缺件
- **WHEN** FAIL 阈值配置为 3，且一块 PCB 连续三个有效检查帧均缺少至少一个配置类别
- **THEN** 系统在第三个有效检查帧生成一次 FAIL 决策，并记录最终缺失类别

#### Scenario: 第三个有效帧恢复完整
- **WHEN** PCB 前两个有效检查帧不完整，但第三个有效检查帧四个槽位全部存在
- **THEN** 系统生成 PASS 决策且不得生成 FAIL

### Requirement: 每块 PCB 只产生一次最终结果
系统 MUST 在 PCB 生成 PASS 或 FAIL 决策时立即将其标记为已处理。在该 PCB 跟踪生命周期内，后续检测 MUST NOT 再改变结果、重复反馈或重复写入生产统计。

#### Scenario: 已完成 PCB 继续停留在画面
- **WHEN** 已产生最终结果的 PCB 在后续多个推理帧中继续被检测到
- **THEN** 系统仅维持结果显示和离场状态，不再次判断或计数

#### Scenario: 短暂 ID 切换后重新匹配已完成 PCB
- **WHEN** 已完成 PCB 在离场宽限期内被唯一匹配到相邻位置的新跟踪 ID
- **THEN** 系统迁移已处理状态并禁止重复产生结果

### Requirement: 多 PCB 并行判定
系统 MUST 在每次推理后并行更新所有未完成 PCB 的检查状态，而 MUST NOT 因当前 PCB 正在累计 FAIL 帧就停止其他 PCB 的观测。

#### Scenario: 两块 PCB 同时处于有效区域
- **WHEN** 两块未处理 PCB 同时完整可见，其中一块完整而另一块缺件
- **THEN** 系统在同一次推理结果中对第一块生成 PASS，并为第二块增加自己的 FAIL 连续计数

### Requirement: 新一轮间隔不得阻塞检测
系统 MUST 使用配置的轮次间隔控制最终结果事件之间的最小时间。间隔期间系统 MUST 继续相机采集、单模型推理、PCB 跟踪和未完成 PCB 的状态更新；已生成决策的结果 MUST 按 PCB 首次出现顺序排队，且不得因间隔而丢失。

#### Scenario: 间隔期间后续 PCB 达到判定条件
- **WHEN** 上一块 PCB 的结果已输出且轮次间隔尚未结束，后续 PCB 已达到 PASS 或 FAIL 条件
- **THEN** 系统立即把后续 PCB 标记为已处理并缓存结果，在间隔结束后按顺序输出

#### Scenario: 轮次间隔配置为零
- **WHEN** 轮次间隔为 0 且多个 PCB 结果已经就绪
- **THEN** 系统无需额外等待即可依次输出所有就绪结果

### Requirement: 结果必须复用生产反馈链路
每个 PCB 的最终结果 MUST 复用现有 PASS/FAIL 声音、FAIL 证据和生产统计入口，并 MUST 携带 PCB 跟踪 ID、四个槽位状态、缺失类别、源帧编号和时间戳。未获得足够有效检查帧便离场的 PCB MUST NOT 写入 PASS/FAIL 统计。

#### Scenario: 缺件 PCB 完成 FAIL
- **WHEN** 一块 PCB 达到 FAIL 连续帧阈值
- **THEN** 系统只写入一次 FAIL 统计和证据，并在记录中包含最终缺失类别

#### Scenario: PCB 在获得足够证据前离场
- **WHEN** 一块不完整 PCB 在达到 FAIL 阈值前离开并超过状态宽限期
- **THEN** 系统释放其未完成状态、记录诊断原因，但不增加 PASS 或 FAIL 数量
