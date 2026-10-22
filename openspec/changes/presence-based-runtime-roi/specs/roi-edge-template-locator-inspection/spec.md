## MODIFIED Requirements

### Requirement: 模板匹配必须只作为现有业务的前置触发

系统 SHALL 在模板门控实际挂上时，对每个待推理帧先执行模板匹配。匹配成功时系统 MUST 调用现有整图 YOLO processor；匹配失败时系统 MUST 跳过该帧 YOLO。系统 MUST 保持类别精确数量、PASS/FAIL 稳定帧、完成一轮间隔、统计、声音和证据业务不变。识别 ROI 有效时 MUST 只把中心落在框内的目标交给判定；识别 ROI 空时 MUST 按整图判定。模板门控 MUST 在顺序认法和不按顺序认法下都可挂上，且 MUST NOT 要求 `roi_inspection_enabled` 为 true。

#### Scenario: 模板未匹配

- **WHEN** 当前帧模板匹配失败
- **THEN** 系统跳过该帧 YOLO
- **AND** 当前轮次不因新增逻辑产生额外 PASS 或 FAIL

#### Scenario: 模板匹配成功

- **WHEN** 当前帧模板匹配成功
- **THEN** 系统对整张当前帧执行原有 YOLO 推理
- **AND** 系统继续用有效识别 ROI（或整图）和现有类别数量逻辑处理结果

#### Scenario: 现有节奏配置保持有效

- **WHEN** 用户配置 PASS 帧数、FAIL 帧数和完成一轮间隔
- **THEN** 模板定位不得修改这些值或改变其现有业务语义

#### Scenario: 顺序认法也可以挂模板门控

- **WHEN** `roi_inspection_enabled` 为 false
- **AND** 模板开关打开且搜索 ROI 与定位模板都可用
- **THEN** 系统 MUST 先在搜索 ROI 内匹配，命中后才对整图跑 YOLO

### Requirement: 主检测页面必须只显示搜索 ROI 框

系统 SHALL 在主检测画面绘制当前有效的运行 ROI：有效搜索 ROI MUST 显示，有效识别 ROI MUST 显示。系统 MUST NOT 在主画面显示模板轮廓、模板得分、角度、定位中心、模板 ROI 或 ROI 类别数量覆盖文字。

#### Scenario: 主页面正在检测

- **WHEN** 搜索 ROI 已配置且相机画面可用
- **THEN** 主画面显示搜索 ROI 矩形
- **AND** 模板匹配详细结果只在内部用于触发

#### Scenario: 顺序认法显示已保存的识别 ROI

- **WHEN** 顺序认法正在检测
- **AND** 识别 ROI 有效
- **THEN** 主画面 MUST 画出该识别 ROI 矩形

### Requirement: 配置升级不得清空现场参数

系统 SHALL 将配置版本 3 窄迁移到版本 4，新增搜索 ROI 和角度步进，并删除持久化输出中的 SIFT 最少匹配点字段。迁移 MUST 保留现有模型、类别、相机、识别 ROI、声音、PASS/FAIL 帧数和完成间隔。

#### Scenario: 加载版本 3 配置

- **WHEN** 程序加载有效的版本 3 配置
- **THEN** 系统保留所有现有现场参数
- **AND** 搜索 ROI 初始化为空
- **AND** 在用户创建搜索 ROI 前系统 MUST NOT 挂模板门控，且 MUST NOT 因此拒绝开始检测

#### Scenario: 重新打包已有便携版本

- **WHEN** 便携目录已有用户配置和模板文件
- **THEN** 构建脚本在重建前保留这些文件并在重建后恢复
- **AND** 构建不得用默认配置覆盖用户配置

### Requirement: 成功状态必须清理旧配置错误提示

系统 SHALL 在配置成功应用或检测成功启动后清除之前由无效 ROI 或过期配置产生的错误提示。缺识别 ROI MUST NOT 再作为开始检测的拒绝原因。

#### Scenario: 修复 ROI 后成功开始检测

- **WHEN** 主页面仍显示旧的 ROI 错误提示
- **AND** 用户完成有效配置并成功开始检测
- **THEN** 主页面不再显示该旧提示

#### Scenario: 缺识别 ROI 不再拒绝开检

- **WHEN** 识别 ROI 未配置
- **AND** 待检类别与数量有效
- **THEN** 系统 MUST 允许开始检测
- **AND** MUST NOT 提示「识别 ROI 未配置」
