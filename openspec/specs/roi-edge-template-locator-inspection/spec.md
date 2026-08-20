# roi-edge-template-locator-inspection Specification

## Purpose
TBD - created by archiving change replace-sift-with-roi-edge-template-matching. Update Purpose after archive.
## Requirements
### Requirement: 系统必须提供完整的边缘模板模型接口

系统 SHALL 通过单一项目级接口完成模板创建、轮廓提取、保存、加载、搜索和匹配结果生成。运行处理器、配置页和测试代码 MUST NOT 各自重复实现角度循环、得分选择或轮廓坐标变换。

#### Scenario: 创建有效边缘模板

- **WHEN** 操作员提供包含清晰边缘的模板图像
- **THEN** 系统创建固定比例的边缘模板模型
- **AND** 模型包含用于显示的真实边缘轮廓
- **AND** 模型可以返回匹配得分、位置、角度、外框和变换后轮廓

#### Scenario: 模板缺少有效边缘

- **WHEN** 模板区域几乎没有可用边缘或有效轮廓长度不足
- **THEN** 系统拒绝创建模板
- **AND** 系统提示重新选择包含清晰边缘的区域

### Requirement: 运行时必须只在搜索 ROI 内查找模板

系统 SHALL 将搜索 ROI 转换为当前帧的像素区域，并仅在该区域内执行固定比例、多角度的边缘 NCC 匹配。系统 MUST 使用配置的匹配阈值、允许角度范围和角度步进，并 MUST 返回最高得分的单个结果。

#### Scenario: 搜索 ROI 内匹配成功

- **WHEN** 搜索 ROI 内存在模板目标
- **AND** 最佳边缘 NCC 得分达到配置阈值
- **THEN** 匹配结果为成功
- **AND** 结果包含目标中心、角度、外框和绿色显示轮廓所需坐标

#### Scenario: 搜索 ROI 外存在相似目标

- **WHEN** 相似目标只存在于搜索 ROI 外
- **THEN** 系统不得将该目标作为匹配结果

#### Scenario: 目标比例变化

- **WHEN** 目标只通过改变缩放比例才能匹配
- **THEN** 系统不得执行多尺度搜索
- **AND** 操作员需要重新调整相机或重新创建模板

### Requirement: 配置页必须统一创建三种 ROI

系统 SHALL 在 ROI 配置页提供 ROI 类型下拉框，选项为“识别 ROI”“模板 ROI”“搜索 ROI”，并提供统一的创建或替换命令以及位于编辑器下方的“保存当前 ROI”按钮。三种 ROI MUST 独立创建和保存，系统 MUST NOT 要求搜索 ROI 包含模板 ROI，也 MUST NOT 在保存当前 ROI 时覆盖另外两种未提交草稿。

#### Scenario: 创建搜索 ROI

- **WHEN** 操作员在下拉框选择“搜索 ROI”并执行创建命令
- **THEN** 操作员可以在当前静态原图上拖拽搜索区域
- **AND** 点击“保存当前 ROI”后只将该区域以归一化坐标保存在配置中

#### Scenario: 三种 ROI 没有包含关系

- **WHEN** 搜索 ROI 与模板 ROI 分离或部分重叠
- **THEN** 系统仍允许分别保存两种 ROI
- **AND** 运行时只使用搜索 ROI 限定查找范围

#### Scenario: 单独保存所选 ROI

- **WHEN** 操作员已同时编辑多种 ROI 草稿
- **AND** 下拉框当前选择其中一种 ROI
- **THEN** “保存当前 ROI”只持久化所选 ROI
- **AND** 另外两种 ROI 的已保存配置不得被草稿覆盖

### Requirement: 保存模板后必须在 ROI 原图显示轮廓和创建得分

系统 SHALL 在编辑阶段显示 ROI 草稿。系统 MUST 在操作员选择模板 ROI 并点击“保存当前 ROI”后才创建并验证正式模板；验证成功后 MUST 在当前 ROI 编辑原图上显示绿色真实边缘轮廓、绿色外框和创建得分，并 MUST NOT 显示独立模板裁剪预览。

#### Scenario: 保存并回查成功

- **WHEN** 模板 ROI 有效
- **AND** 搜索 ROI 和识别 ROI 尚未配置或位于其他位置
- **AND** 临时边缘模板在当前静态原图的模板 ROI 自身范围回查得分达到阈值
- **THEN** 系统原子保存模板文件和配置
- **AND** 系统显示绿色真实轮廓、绿色外框和创建得分
- **AND** 系统不得因搜索 ROI、识别 ROI、模型路径或运行处理器尚未完成而将模板保存判为失败

#### Scenario: 保存验证失败

- **WHEN** 模板轮廓质量不足或回查得分未达到阈值
- **THEN** 系统显示红色失败结果和具体原因
- **AND** 系统不得覆盖旧模板、旧配置或旧处理器

#### Scenario: 不提供持续测试模式

- **WHEN** 模板保存成功
- **THEN** 系统只显示本次静态原图回查的创建得分
- **AND** 系统不增加“测试模板”按钮或持续实时测试

### Requirement: 模板匹配必须只作为现有业务的前置触发

系统 SHALL 在每个待推理帧先执行模板匹配。匹配成功时系统 MUST 调用现有整图 YOLO processor；匹配失败时系统 MUST 跳过该帧 YOLO。系统 MUST 保持固定识别 ROI、类别精确数量、PASS/FAIL 稳定帧、完成一轮间隔、统计、声音和证据业务不变。

#### Scenario: 模板未匹配

- **WHEN** 当前帧模板匹配失败
- **THEN** 系统跳过该帧 YOLO
- **AND** 当前轮次不因新增逻辑产生额外 PASS 或 FAIL

#### Scenario: 模板匹配成功

- **WHEN** 当前帧模板匹配成功
- **THEN** 系统对整张当前帧执行原有 YOLO 推理
- **AND** 系统继续用固定识别 ROI 和现有类别数量逻辑处理结果

#### Scenario: 现有节奏配置保持有效

- **WHEN** 用户配置 PASS 帧数、FAIL 帧数和完成一轮间隔
- **THEN** 模板定位不得修改这些值或改变其现有业务语义

### Requirement: 主检测页面必须只显示搜索 ROI 框

系统 SHALL 在模板定位启用时在主检测画面显示搜索 ROI 框。系统 MUST NOT 在主画面显示模板轮廓、模板得分、角度、定位中心、模板 ROI、识别 ROI 或 ROI 类别数量覆盖文字。

#### Scenario: 主页面正在检测

- **WHEN** 搜索 ROI 已配置且相机画面可用
- **THEN** 主画面显示搜索 ROI 矩形
- **AND** 模板匹配详细结果只在内部用于触发

### Requirement: 配置升级不得清空现场参数

系统 SHALL 将配置版本 3 窄迁移到版本 4，新增搜索 ROI 和角度步进，并删除持久化输出中的 SIFT 最少匹配点字段。迁移 MUST 保留现有模型、类别、相机、识别 ROI、声音、PASS/FAIL 帧数和完成间隔。

#### Scenario: 加载版本 3 配置

- **WHEN** 程序加载有效的版本 3 配置
- **THEN** 系统保留所有现有现场参数
- **AND** 搜索 ROI 初始化为空
- **AND** 在用户创建搜索 ROI 前模板定位 fail closed

#### Scenario: 重新打包已有便携版本

- **WHEN** 便携目录已有用户配置和模板文件
- **THEN** 构建脚本在重建前保留这些文件并在重建后恢复
- **AND** 构建不得用默认配置覆盖用户配置

### Requirement: 成功状态必须清理旧配置错误提示

系统 SHALL 在配置成功应用或检测成功启动后清除之前由无效 ROI 产生的错误提示。

#### Scenario: 修复 ROI 后成功开始检测

- **WHEN** 之前因“识别 ROI 未配置”被拒绝
- **AND** 用户完成有效配置并成功开始检测
- **THEN** 主页面不再显示旧“识别 ROI 未配置”提示

