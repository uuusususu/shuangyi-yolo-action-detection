## ADDED Requirements

### Requirement: Standard controls use an open-source Qt theme
系统 SHALL 使用兼容 PySide6 且具有可接受开源许可证的主题库统一标准交互控件，并 SHALL 在创建任何主窗口或配置页之前完成主题初始化。标准按钮、输入框、下拉框、数字框和复选框 MUST NOT 通过自定义绘制代码重新实现。

#### Scenario: Application initializes the selected theme
- **WHEN** 应用程序创建 `QApplication` 并准备创建主窗口
- **THEN** 系统 SHALL 以深色模式和项目青色强调色初始化选定的开源 Qt 主题

#### Scenario: Theme dependency is unavailable
- **WHEN** 源码环境或打包环境缺少必需的主题依赖
- **THEN** 构建或启动验证 SHALL 明确失败并报告缺失依赖，而不是静默交付无主题或部分主题的程序

#### Scenario: Standard controls remain native Qt widgets
- **WHEN** 配置页创建按钮、输入框、下拉框、数字框或二元选项
- **THEN** 这些控件 SHALL 保持为主题库支持的 PySide6 标准控件并保留其既有信号和取值 API

### Requirement: Interactive controls have visible resting boundaries
系统 SHALL 让所有可执行操作在未悬停、未按下和未获得焦点的静止状态下仍可与普通文字或容器背景区分。除明确的非操作标签外，交互控件 MUST NOT 同时使用透明边界、透明背景和与周围文字相同的呈现方式。

#### Scenario: Secondary button is visible without hover
- **WHEN** 操作员打开主界面或任意配置分区且没有移动鼠标
- **THEN** “配置”“关闭”“浏览”“刷新设备”“返回检测”等次级操作 SHALL 具有可识别的边界或填充区域

#### Scenario: Primary action remains prominent
- **WHEN** 配置页显示“验证并保存”或主界面显示当前主要检测操作
- **THEN** 主要操作 SHALL 使用比次级操作更明确的强调填充或默认按钮状态，且文字与背景可读

#### Scenario: Dangerous action remains distinct
- **WHEN** “关闭”或“归零并归档”等危险操作可用
- **THEN** 该操作 SHALL 保留危险语义并 SHALL NOT 被设置为页面默认操作

#### Scenario: Navigation items remain identifiable
- **WHEN** 配置页显示六个分区导航项
- **THEN** 未选中的导航项 SHALL 仍可识别为可点击项，选中项 SHALL 具有不同的强调背景、边界或选中状态

### Requirement: Control states and affordances are complete
系统 SHALL 为标准控件提供默认、悬停、按下、键盘焦点、选中和禁用状态，并 SHALL 显示与控件功能相符的箭头或指示器。

#### Scenario: Button interaction states differ
- **WHEN** 同一按钮依次处于默认、悬停、按下、键盘焦点和禁用状态
- **THEN** 每种状态 SHALL 具有可辨认的视觉变化，且禁用状态 SHALL 不被误认为可执行状态

#### Scenario: Combo box arrow is visible
- **WHEN** 操作员查看模型跟踪器、相机设备或参数模式下拉框
- **THEN** 下拉箭头 SHALL 在默认、焦点和禁用状态下保持可见且不与文字重叠

#### Scenario: Spin box controls are visible
- **WHEN** 操作员查看置信度、帧数、数量、曝光或间隔数字控件
- **THEN** 上下步进指示 SHALL 可见，数值文字 SHALL 不被步进区域遮挡

#### Scenario: Binary setting states are visible
- **WHEN** 操作员切换持续跟踪、显示叠加、声音、证据保存或首类别区域检查选项
- **THEN** 选中与未选中状态 SHALL 同时通过标准指示器形态和颜色区分，且禁用状态 SHALL 清晰可辨

### Requirement: Existing industrial technology visual language is preserved
系统 SHALL 在采用开源控件主题后保留现有主界面和配置页布局、深蓝背景、青色强调色、卡片层级、文字层级以及业务状态颜色，不得整体切换为主题库默认灰黑外观。

#### Scenario: Configuration layout is unchanged
- **WHEN** 操作员在变更前后打开配置页
- **THEN** 左侧导航、顶部标题、右侧分区表单和底部固定操作栏 SHALL 保持原有位置关系和六个分区结构

#### Scenario: Runtime layout is unchanged
- **WHEN** 操作员在变更前后打开主检测页
- **THEN** 顶部状态栏、相机视窗、右侧 KPI/识别列表/异常提示和主要操作区 SHALL 保持原有信息架构

#### Scenario: Detection state colors remain semantic
- **WHEN** 识别项进入等待、检测中、PASS、FAIL 或锁定状态
- **THEN** 系统 SHALL 保持现有状态语义和红绿颜色区分，主题初始化 MUST NOT 覆盖检测框或识别步骤状态色

#### Scenario: Technology palette remains recognizable
- **WHEN** 主界面和配置页完成真实渲染
- **THEN** 页面 SHALL 继续以深蓝容器和青色强调为主，不得呈现为主题库未经项目配色覆盖的默认灰黑界面

### Requirement: Theme migration preserves application behavior
控件主题迁移 SHALL NOT 改变配置数据、相机选择、检测状态、声音、证据保存、统计或页面导航的业务语义。

#### Scenario: Navigation preserves unsaved values
- **WHEN** 操作员编辑某个配置值、切换到其他配置分区再返回
- **THEN** 未保存值 SHALL 保留，导航选择行为 SHALL 与迁移前一致

#### Scenario: Configuration save remains compatible
- **WHEN** 操作员修改配置并执行“验证并保存”
- **THEN** 现有校验、持久化、运行时配置应用和保存反馈 SHALL 正常工作，字段名称和值语义 SHALL 不变

#### Scenario: Camera selector preserves device identity behavior
- **WHEN** 配置页枚举多台迈德相机或显示无有效 SN 的设备
- **THEN** 下拉框 SHALL 继续按 SN 保存选择，并 SHALL 继续禁止选择无稳定 SN 的无效设备

#### Scenario: Theme changes do not reset runtime state
- **WHEN** 应用主题或保存仅影响界面反馈的配置
- **THEN** 系统 SHALL NOT 因视觉样式刷新而关闭相机、重建检测引擎、清空父类 ID、统计或当前识别轮次

### Requirement: Visual and packaged acceptance is reproducible
系统 SHALL 通过自动测试、真实 Qt 渲染截图、源码冒烟测试和打包 EXE 验证共同验收主题迁移，源码测试通过但打包程序未验证不得视为完成。

#### Scenario: Supported viewport screenshots pass
- **WHEN** 在 1366×920 和 1920×1080 视口渲染主界面及配置页六个分区
- **THEN** 所有文字和控件 SHALL 不裁切、不重叠，操作边界、下拉箭头、步进按钮和二元状态 SHALL 清晰可见

#### Scenario: Windows scaling screenshots pass
- **WHEN** 在 Windows 100% 和 125% 显示缩放下打开打包程序
- **THEN** 控件边界和文字 SHALL 保持清晰，布局 SHALL 不产生遮挡或溢出

#### Scenario: Source runtime passes
- **WHEN** 运行 UI 定向测试、全量测试和源码 `--portable-smoke-test`
- **THEN** 所有命令 SHALL 成功且现有配置、相机、检测和反馈回归测试 SHALL 通过

#### Scenario: Packaged runtime contains theme resources
- **WHEN** 使用现有构建脚本生成便携版并运行打包 EXE 冒烟测试
- **THEN** EXE SHALL 返回成功状态，主题模块和资源 SHALL 可用，配置页 SHALL 能打开并渲染完整控件状态
