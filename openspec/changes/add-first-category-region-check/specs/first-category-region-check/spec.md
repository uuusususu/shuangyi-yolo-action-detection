## ADDED Requirements

### Requirement: 首类别模式开关与类别派生
系统 SHALL 仅在启用首类别区域检查时，把第一个有效配置类别作为父类，并把后续有效类别及其配置数量作为父类内的子步骤。

#### Scenario: 开关关闭保留普通顺序模式
- **WHEN** `first_category_region_check_enabled` 为关闭状态
- **THEN** 系统 SHALL 继续使用现有动作顺序检测状态机
- **AND** 系统 SHALL NOT 创建父类轮次队列

#### Scenario: 开关开启派生父类和子步骤
- **WHEN** 首类别区域检查已启用，`category_names` 为 [`产品主体`, `脚垫`, `螺丝`]，对应数量为 [1, 4, 2]
- **THEN** 系统 SHALL 使用 `产品主体` 作为父类
- **AND** 系统 SHALL 创建数量分别为 4 和 2 的 `脚垫`、`螺丝` 子步骤

#### Scenario: 父类或子步骤缺失时阻止启用
- **WHEN** 首类别区域检查已启用，但没有有效父类或没有至少一个有效子步骤
- **THEN** 系统 MUST 在检测启动前拒绝该配置并显示明确错误

### Requirement: 父类业务 ID 与等待队列
系统 SHALL 为每个稳定出现的父类实例建立不可复用业务 ID，并 SHALL 按稳定出现顺序加入等待队列。

#### Scenario: 多个父类获得不同 ID
- **WHEN** 同一帧出现两个空间可区分的父类实例
- **THEN** 系统 SHALL 为它们分配不同业务 ID
- **AND** 系统 SHALL 按画面从左到右顺序加入等待队列

#### Scenario: 无原生追踪 ID 时创建业务 ID
- **WHEN** 父类检测结果没有原生 `track_id`
- **THEN** 系统 MUST 根据跨帧空间匹配为父类分配稳定业务 ID

#### Scenario: 短暂漏检保持原 ID
- **WHEN** 一个活动、等待或可重试父类短暂漏检且在离场宽限帧数内重新出现，并满足唯一空间匹配条件
- **THEN** 系统 SHALL 恢复该父类原业务 ID、已完成步骤和最近 FAIL 显示状态

#### Scenario: 真正离场后新实例使用新 ID
- **WHEN** 父类连续缺失超过离场宽限并已退休，随后相同类别再次进入画面
- **THEN** 系统 MUST 分配从未用于上一生命周期的新业务 ID

### Requirement: 单当前轮次 ID 判定
系统 SHALL 同一时间只允许一个父类 ID 推进子类别数量判定，其他父类 ID 只追踪和显示。

#### Scenario: 等待 ID 不累计数量
- **WHEN** 父类 ID 处于 `WAITING` 状态
- **THEN** 系统 SHALL 维护其追踪和画面显示
- **AND** 系统 SHALL NOT 使用该 ID 区域内的子类别推进 PASS 或 FAIL

#### Scenario: 冷却 ID 不累计数量
- **WHEN** 父类 ID 已被选为下一轮但处于 `COOLDOWN` 状态
- **THEN** 系统 SHALL 显示该 ID 为当前轮次候选
- **AND** 系统 SHALL NOT 在冷却结束前统计其子类别数量

#### Scenario: 可重试 FAIL ID 等待时不累计数量
- **WHEN** 父类 ID 处于 `FAIL_RETRY_WAITING` 状态
- **THEN** 系统 SHALL 当前可见时绘制红框
- **AND** 系统 SHALL NOT 在该 ID 再次进入 `ACTIVE` 前统计其子类别数量

#### Scenario: 只有 ACTIVE ID 统计子类别
- **WHEN** 父类 ID 进入 `ACTIVE` 状态
- **THEN** 系统 SHALL 仅对该 ID 的父类区域统计后续类别数量
- **AND** 其他父类 ID 的子类别观测 SHALL NOT 修改当前步骤卡或结果状态

### Requirement: 子类别空间归属
系统 SHALL 只把当前可靠帧中唯一归属于当前 `ACTIVE` 父类区域的子类别检测计入该父类的数量。

#### Scenario: ACTIVE 父类区域内的子类别计入数量
- **WHEN** 子类别中心或 OBB 满足当前 `ACTIVE` 父类区域归属规则，且不存在其他父类归属歧义
- **THEN** 该检测 SHALL 计入当前 `ACTIVE` 父类对应子步骤的当前数量

#### Scenario: 非 ACTIVE 父类区域内的子类别不参与判定
- **WHEN** 子类别位于非当前轮次父类区域内
- **THEN** 该检测 SHALL NOT 推进任何子步骤的 PASS 或 FAIL

#### Scenario: 归属歧义帧冻结状态推进
- **WHEN** 子类别无法唯一归属到当前 `ACTIVE` 父类，或父类尚未完整进入安全画面区域
- **THEN** 受影响子步骤的正确数量连续帧和错误数量连续帧 SHALL 均不增加

### Requirement: 子步骤正确数量完成锁存
系统 SHALL 为每个父类 ID 分别维护子步骤状态，并只在该 ID 作为当前 `ACTIVE` 父类且正确数量连续稳定后锁存完成。

#### Scenario: 正确数量达到稳定阈值后完成
- **WHEN** 某个子步骤的归属数量连续 `action_pass_stable_frames` 帧等于配置数量
- **THEN** 系统 SHALL 将当前父类 ID 的该子步骤锁存为已完成

#### Scenario: 完成后漏检不撤销
- **WHEN** 某个子步骤已经锁存完成，后续帧暂时未检测到该子类别
- **THEN** 该步骤 MUST 保持已完成状态

#### Scenario: 完成步骤不再重新判错
- **WHEN** 某个子步骤已经锁存完成，后续帧出现不同数量的同类别检测
- **THEN** 系统 SHALL NOT 撤销完成状态
- **AND** 系统 SHALL NOT 使用该步骤触发新的 FAIL

### Requirement: 数量不对触发可重试 FAIL
系统 SHALL 把当前 `ACTIVE` 父类 ID 的未完成子步骤连续数量不对判定为本轮 FAIL/NG 报警，但 SHALL NOT 将该父类 ID 终局报废。

#### Scenario: 零数量也累计 FAIL
- **WHEN** 未完成子步骤配置数量为 4，当前 `ACTIVE` 父类区域内连续可靠帧的归属数量为 0
- **THEN** 系统 SHALL 显示该步骤为 `0/4`
- **AND** 系统 SHALL 累计错误数量连续帧

#### Scenario: 数量少于配置值触发本轮 FAIL
- **WHEN** 未完成子步骤的归属数量少于配置数量，并连续达到 `action_ng_stable_frames`
- **THEN** 对应父类 ID SHALL 产生本轮 FAIL/NG 报警
- **AND** 系统 SHALL 记录实际数量和配置数量
- **AND** 该父类 ID SHALL 进入可重试等待状态

#### Scenario: 数量多于配置值触发本轮 FAIL
- **WHEN** 未完成子步骤的归属数量多于配置数量，并连续达到 `action_ng_stable_frames`
- **THEN** 对应父类 ID SHALL 产生本轮 FAIL/NG 报警
- **AND** 系统 SHALL 记录实际数量和配置数量
- **AND** 该父类 ID SHALL 进入可重试等待状态

#### Scenario: 配置帧数稳定错误后红色
- **WHEN** `action_ng_stable_frames` 配置为 10，当前 `ACTIVE` 父类 ID 连续 10 个可靠帧数量不对
- **THEN** 系统 SHALL 将该父类 ID 本轮判定为 FAIL/NG
- **AND** 对应父类框和异常步骤 SHALL 变为红色

#### Scenario: 中间恢复正确数量时清零错误连续帧
- **WHEN** 数量不对尚未达到 `action_ng_stable_frames`，中间出现一帧正确配置数量
- **THEN** 系统 SHALL 清零错误数量连续帧
- **AND** 系统 SHALL 重新累计正确数量连续帧

#### Scenario: FAIL 后清零本轮错误连续帧
- **WHEN** 当前父类 ID 因未完成子步骤数量不对触发本轮 FAIL
- **THEN** 系统 SHALL 保留该 ID 已完成的子步骤
- **AND** 系统 SHALL 清零未完成步骤的本轮错误数量连续帧
- **AND** 该 ID 后续再次进入 `ACTIVE` 时 MUST 重新累计 `action_ng_stable_frames` 后才可再次 FAIL

#### Scenario: FAIL 后补齐仍可 PASS
- **WHEN** 父类 ID 曾经触发本轮 FAIL，后续重新进入 `ACTIVE`，且所有未完成子步骤连续达到正确配置数量
- **THEN** 系统 SHALL 允许该父类 ID 锁存剩余步骤完成
- **AND** 当全部子步骤完成时 SHALL 将该父类 ID 转为 PASS

### Requirement: PASS 终局与 FAIL 事件去重
系统 SHALL 将 PASS 作为父类 ID 的终局完成状态，并 SHALL 对可重试 FAIL 事件按异常签名去重/节流。

#### Scenario: 全部子步骤完成后输出一次 PASS
- **WHEN** 同一父类 ID 的全部有效子步骤均已锁存完成
- **THEN** 系统 SHALL 将父类置为 `PASS_LATCHED`
- **AND** 系统 SHALL 只输出一次 PASS
- **AND** 系统 SHALL NOT 再将该父类 ID 重新加入等待队列

#### Scenario: FAIL 不终局
- **WHEN** 同一父类 ID 首次触发本轮 FAIL/NG
- **THEN** 系统 SHALL 输出一次 FAIL/NG 报警
- **AND** 系统 SHALL NOT 将该父类 ID 视为最终结束
- **AND** 系统 SHALL 在该 ID 未退休且未 PASS 时允许它重新进入等待队列

#### Scenario: 同一异常签名不重复刷事件
- **WHEN** 同一父类 ID、同一子步骤、同一实际数量和同一配置数量的异常已经上报过
- **THEN** 后续重试中再次出现相同异常时，系统 SHALL 保持红框和状态显示
- **AND** 系统 SHALL NOT 重复增加统计、播放声音或保存证据

#### Scenario: 异常签名变化允许新 FAIL 事件
- **WHEN** 同一父类 ID 后续出现不同子步骤、不同实际数量或不同配置数量的异常
- **THEN** 系统 MAY 产生新的 FAIL/NG 事件

### Requirement: FAIL 后调度下一轮与重试
系统 SHALL 在当前父类 ID FAIL 后立即释放当前轮次，调度下一个等待 ID，并 SHALL 让 FAIL 的旧 ID 保持可重试。

#### Scenario: FAIL 后立即预选下一个 ID
- **WHEN** 当前 `ACTIVE` 父类 ID 触发本轮 FAIL
- **THEN** 系统 SHALL 立即从等待队列选择下一个可用父类 ID
- **AND** 已 FAIL 的旧 ID SHALL NOT 阻塞新 ID

#### Scenario: FAIL ID 进入等待队尾
- **WHEN** 父类 ID 触发本轮 FAIL 且仍未退休、未 PASS
- **THEN** 系统 SHALL 将该 ID 放入可重试等待队列尾部
- **AND** 系统 SHOULD 在存在其他等待 ID 时优先处理其他 ID

#### Scenario: 只有 FAIL ID 可用时按间隔重试自己
- **WHEN** 当前没有其他可用等待 ID，但 FAIL 的父类 ID 仍可见或仍在漏检宽限内
- **THEN** 系统 SHALL 按 `round_cooldown_seconds` 等待后允许该 ID 重新进入 `ACTIVE`

#### Scenario: PASS 后立即预选下一个 ID
- **WHEN** 当前 `ACTIVE` 父类 ID 进入 `PASS_LATCHED`
- **THEN** 系统 SHALL 立即从等待队列选择下一个可用父类 ID

#### Scenario: 按间隔开始下一轮
- **WHEN** 下一个父类 ID 已被预选，且 `round_cooldown_seconds` 大于 0
- **THEN** 系统 SHALL 等待该间隔结束后才让该 ID 进入 `ACTIVE` 并开始统计子类别数量

#### Scenario: 间隔为零时立即开始下一轮
- **WHEN** 下一个父类 ID 已被预选，且 `round_cooldown_seconds` 为 0
- **THEN** 系统 SHALL 立即让该 ID 进入 `ACTIVE`

### Requirement: FAIL 红框可恢复
系统 SHALL 在数量异常报警后对当前仍在画面中的父类使用红色绘制，但 SHALL 在该父类后续 PASS、不可见或退休时按状态停止或改变红框显示。

#### Scenario: FAIL 父类当前可见时持续绘制红框
- **WHEN** 父类 ID 已触发本轮 FAIL 且当前帧检测到该父类
- **THEN** 系统 SHALL 使用红色绘制该父类 OBB
- **AND** 系统 SHALL 标注父类业务 ID 和数量异常状态

#### Scenario: 父类当前帧识别不到时不绘制陈旧红框
- **WHEN** 已 FAIL 的父类当前帧未检测到且未可靠恢复
- **THEN** UI SHALL NOT 绘制上一帧残留的父类多边形

#### Scenario: 短暂漏检后恢复仍为红色
- **WHEN** 已 FAIL 的父类在离场宽限内短暂漏检后以同一 ID 恢复
- **THEN** 恢复后的父类框 MUST 继续为红色

#### Scenario: 补齐 PASS 后红框恢复为完成颜色
- **WHEN** 已 FAIL 的父类 ID 后续补齐配件并进入 `PASS_LATCHED`
- **THEN** UI SHALL 停止使用 FAIL 红色绘制该父类
- **AND** UI SHALL 使用 PASS/完成颜色绘制该父类

#### Scenario: 父类退休后不绘制陈旧红框
- **WHEN** 已 FAIL 的父类连续离场超过宽限并退休
- **THEN** UI SHALL 停止绘制该父类的历史多边形

### Requirement: UI 绑定当前轮次 ID
系统 SHALL 使用引擎解析业务 ID 后的当前帧结果绘制父类状态，并 SHALL 让顶部状态、步骤卡和数量绑定同一个当前轮次 ID。

#### Scenario: 多个父类同时显示独立状态
- **WHEN** 当前帧包含多个活动、可重试 FAIL 或 PASS 父类 ID
- **THEN** 相机画面 SHALL 同时绘制每个父类的 ID 和各自状态颜色
- **AND** 不同父类 SHALL NOT 共享步骤结果

#### Scenario: 详情步骤列表明确绑定当前轮次 ID
- **WHEN** 右侧详情区展示步骤
- **THEN** UI MUST 明确绑定 `current_round_id`
- **AND** UI SHALL 只读取该 ID 的步骤状态

#### Scenario: 旧 FAIL 红框不覆盖当前步骤卡
- **WHEN** 旧 FAIL 父类仍在画面中红框显示，且另一个父类 ID 是当前轮次
- **THEN** 右侧步骤卡和顶部状态 SHALL 显示当前轮次 ID
- **AND** SHALL NOT 显示旧 FAIL ID 的步骤状态

#### Scenario: 当前帧没有父类时不显示旧识别
- **WHEN** 当前帧没有可绘制父类，且没有在当前帧可靠恢复的活动轨迹
- **THEN** UI SHALL NOT 把上一帧 `last_slot_states` 显示为当前识别结果

#### Scenario: 禁止按最大 ID 选择步骤对象
- **WHEN** 同时存在多个父类业务 ID
- **THEN** UI SHALL NOT 使用 `max(track_id)`、原始检测顺序或上一帧缓存选择步骤卡对象
- **AND** UI MUST 使用引擎提供的 `current_round_id`

### Requirement: 操作员文案与兼容
系统 SHALL 使用通用的父类区域检查文案，并 SHALL 保持旧配置文件可加载。

#### Scenario: 配置页说明首类别语义
- **WHEN** 操作员查看首类别区域检查配置
- **THEN** 配置页 SHALL 明确说明“类别 1 为父类，后续类别在父类区域内按数量完成”

#### Scenario: 文档说明 FAIL 可重试与 PASS 终局
- **WHEN** 操作员阅读首类别模式说明
- **THEN** 文档 SHALL 说明 `0/配置数量` 也会累计 FAIL、稳定 FAIL 后父类红框、FAIL 后该 ID 可重试、补齐后可 PASS、PASS 后才终局
