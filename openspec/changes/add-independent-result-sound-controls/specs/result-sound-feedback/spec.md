## ADDED Requirements

### Requirement: PASS 与 FAIL 声音独立配置
系统 SHALL 提供相互独立的 PASS 提示音开关和 FAIL 提示音开关，任一开关的状态 SHALL NOT 改变另一类结果的声音行为。

#### Scenario: 只开启 FAIL 提示音
- **WHEN** 操作员关闭 PASS 提示音并开启 FAIL 提示音后保存配置
- **THEN** 后续 PASS 结果 SHALL 保持静音
- **AND** 后续符合播放条件的 FAIL 轮次 SHALL 播放 FAIL 提示音

#### Scenario: 只开启 PASS 提示音
- **WHEN** 操作员开启 PASS 提示音并关闭 FAIL 提示音后保存配置
- **THEN** 后续符合播放条件的 PASS 结果 SHALL 播放 PASS 提示音
- **AND** 后续 FAIL 轮次 SHALL 保持静音

#### Scenario: 两个提示音均关闭
- **WHEN** 操作员关闭 PASS 提示音和 FAIL 提示音后保存配置
- **THEN** 后续 PASS 与 FAIL 结果 SHALL 均保持静音

### Requirement: 新配置使用现场安全默认值
系统 MUST 对没有已保存声音配置的新配置使用 PASS 提示音关闭、FAIL 提示音开启的默认值。

#### Scenario: 首次创建配置
- **WHEN** 程序使用默认配置启动，且没有可加载的声音配置字段
- **THEN** PASS 提示音开关 SHALL 默认关闭
- **AND** FAIL 提示音开关 SHALL 默认开启

### Requirement: 旧声音配置兼容迁移
系统 SHALL 兼容只包含旧字段 `sound_feedback_enabled` 的配置文件，并 SHALL 将旧字段值迁移为 FAIL 提示音开关值，同时保持 PASS 提示音关闭。

#### Scenario: 旧声音开关为开启
- **WHEN** 程序加载只包含 `sound_feedback_enabled: true`、不包含新声音字段的配置
- **THEN** FAIL 提示音 SHALL 设为开启
- **AND** PASS 提示音 SHALL 设为关闭

#### Scenario: 旧声音开关为关闭
- **WHEN** 程序加载只包含 `sound_feedback_enabled: false`、不包含新声音字段的配置
- **THEN** FAIL 提示音 SHALL 设为关闭
- **AND** PASS 提示音 SHALL 设为关闭

#### Scenario: 保存迁移后的配置
- **WHEN** 操作员加载旧配置后保存配置
- **THEN** 系统 SHALL 持久化 PASS 与 FAIL 两个独立声音配置字段
- **AND** 程序重启后 SHALL 恢复保存的两个独立开关状态

### Requirement: PASS 声音按完成结果播放一次
当 PASS 提示音开启时，系统 SHALL 对每个新完成的 PASS 结果播放一次 PASS 提示音，并 MUST NOT 因同一结果的重复刷新或重复回调再次播放。

#### Scenario: 普通顺序模式完成 PASS
- **WHEN** 普通顺序模式中的一个轮次首次进入 PASS，且 PASS 提示音已开启
- **THEN** 系统 SHALL 播放一次 PASS 提示音
- **AND** 同一 PASS 状态后续继续刷新时 SHALL NOT 再次播放

#### Scenario: 首类别父类完成 PASS
- **WHEN** 首类别模式中的一个父类 ID 首次进入终局 PASS，且 PASS 提示音已开启
- **THEN** 系统 SHALL 播放一次 PASS 提示音
- **AND** 该父类 ID 后续继续可见时 SHALL NOT 再次播放

#### Scenario: PASS 提示音关闭
- **WHEN** 任一模式产生 PASS 结果，且 PASS 提示音已关闭
- **THEN** 系统 SHALL NOT 播放 PASS 提示音

### Requirement: FAIL 声音按识别轮次播放一次
当 FAIL 提示音开启时，系统 SHALL 在某一识别轮次达到配置的连续 FAIL 稳定帧并首次判定为 FAIL 时播放一次 FAIL 提示音。同一轮次的重复检测、重复帧或重复结果处理 MUST NOT 再次播放。

#### Scenario: 普通顺序模式一轮 FAIL
- **WHEN** 普通顺序模式中的一个轮次首次达到 FAIL 判定条件，且 FAIL 提示音已开启
- **THEN** 系统 SHALL 播放一次 FAIL 提示音
- **AND** 同一轮次后续重复处理 FAIL 状态时 SHALL NOT 再次播放

#### Scenario: 首类别模式一轮 FAIL
- **WHEN** 当前父类 ID 在本轮首次达到配置的 FAIL 稳定帧，且 FAIL 提示音已开启
- **THEN** 系统 SHALL 播放一次 FAIL 提示音
- **AND** 本轮剩余重复检测或重复结果处理 SHALL NOT 再次播放

#### Scenario: 同一父类新轮次再次出现相同异常
- **WHEN** 某父类 ID 在上一轮 FAIL 后重新入队并进入新的识别轮次，且新轮次再次达到 FAIL 判定条件
- **THEN** 系统 SHALL 将其视为新的声音播放轮次
- **AND** 即使异常子类别、实际数量和配置数量与上一轮相同，也 SHALL 再播放一次 FAIL 提示音

#### Scenario: 不同父类分别 FAIL
- **WHEN** 两个不同父类 ID 在各自识别轮次中分别达到 FAIL 判定条件
- **THEN** 系统 SHALL 为每个 FAIL 轮次分别播放一次 FAIL 提示音

#### Scenario: FAIL 提示音关闭
- **WHEN** 任一模式产生 FAIL 结果，且 FAIL 提示音已关闭
- **THEN** 系统 SHALL NOT 播放 FAIL 提示音

### Requirement: 声音轮次去重不改变其他反馈语义
系统 SHALL 将声音播放去重与异常统计、证据保存和红框显示分离。按新轮次再次播放 FAIL 声音 MUST NOT 自动改变既有异常签名去重、证据保存或红框持续规则。

#### Scenario: 相同异常在新轮次再次 FAIL
- **WHEN** 同一父类 ID 在新轮次再次出现已上报过的相同异常签名
- **THEN** 系统 SHALL 按新轮次规则播放一次 FAIL 声音
- **AND** 统计与证据保存 SHALL 继续遵守各自既有的异常签名去重规则
- **AND** 红框 SHALL 继续遵守当前可见性和恢复规则

### Requirement: 声音配置保存后立即生效
系统 SHALL 在配置保存成功后立即应用 PASS 与 FAIL 声音开关，无需重启程序或重新打开相机。

#### Scenario: 检测运行中关闭 FAIL 提示音
- **WHEN** 检测运行期间保存配置并关闭 FAIL 提示音
- **THEN** 保存后产生的 FAIL 轮次 SHALL 保持静音
- **AND** 相机采集与识别 SHALL 继续运行

#### Scenario: 检测运行中开启 PASS 提示音
- **WHEN** 检测运行期间保存配置并开启 PASS 提示音
- **THEN** 保存后新产生的 PASS 结果 SHALL 按一次性规则播放提示音

### Requirement: 声音播放失败不阻塞检测
系统 MUST 在声音资源缺失、播放器不可用或播放发生异常时继续执行检测、状态更新、统计和画面显示。

#### Scenario: FAIL 声音播放异常
- **WHEN** 某一 FAIL 轮次需要播放声音但播放器抛出异常
- **THEN** 系统 SHALL 安全忽略该播放异常
- **AND** 本轮 FAIL 状态、红框、统计和后续轮次调度 SHALL 继续执行

