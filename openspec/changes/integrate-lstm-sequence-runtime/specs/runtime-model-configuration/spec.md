## MODIFIED Requirements

### Requirement: Save configuration applies runtime model changes

系统 SHALL 在配置页保存成功后立即应用运行时配置。若模型路径或推理相关参数发生变化，系统 SHALL 重新创建并加载推理处理器，并将新处理器设置为相机推理线程使用的处理器。动作模式 SHALL 将 YOLO、手部、LSTM 模型包、类别和模式作为整体候选校验，校验通过后才替换当前配置与运行态；替换前 SHALL 备份当前配置并停止旧会话业务判定，替换后 SHALL 清理时序状态重新就绪。

#### Scenario: Model path changes and reload succeeds

- **WHEN** 操作员选择新的 `.pt` 或 `.onnx` 模型并保存配置
- **THEN** 系统重新加载新模型并替换运行中的推理处理器
- **AND** 后续推理使用新模型输出结果；动作模式额外要求其与动作模型包指纹一致

#### Scenario: Inference parameters change and reload succeeds

- **WHEN** 操作员修改置信度、IoU、设备、tracker 或最大目标数等推理相关配置并保存
- **THEN** 系统使用新参数重新创建推理处理器
- **AND** 相机推理线程使用新处理器处理后续帧；动作模式必须先通过完整契约校验

#### Scenario: 动作模式切换或模型包变更

- **WHEN** 新模式及模型包校验通过并保存
- **THEN** 系统备份旧配置、整体应用候选并清理旧动作会话，后续仅由新模式判定，旧事件不能进入新流程

### Requirement: Manual category configuration remains authoritative

系统 MUST 将操作员手动填写的步骤类别配置作为业务顺序的唯一配置来源。系统 MUST NOT 从模型类别列表自动填充、覆盖或重排 `category_names`。纯 YOLO / ROI 模式沿用 YOLO 名称校验；动作模式 SHALL 要求非空配置名称集合与部署清单的非空闲动作名称集合完全一致，不重复，并保持模型分类索引和配置顺序各自独立。不一致 SHALL 报错而非自动修改配置或退回 YOLO。

#### Scenario: New model exposes class names

- **WHEN** 新模型加载成功并可返回模型类别名称
- **THEN** 系统不修改操作员手动填写的步骤类别配置
- **AND** `category_names` 保持保存前表单中的手动输入值

#### Scenario: Manual steps differ from model class list

- **WHEN** 手动配置的步骤类别与当前判定模式对应的模型类别不一致
- **THEN** 系统不自动改写步骤类别
- **AND** 动作模式明确报告缺失和多余名称并阻止保存候选或启动；纯 YOLO / ROI 使用既有配置校验

#### Scenario: 动作类别不属于 YOLO 零件类别

- **WHEN** 动作模式配置包装、封装，而 YOLO 类别为纸盒、路由器等零件
- **THEN** 系统依据动作清单校验包装和封装，依据零件清单单独校验所需零件，不要求动作名属于 YOLO

### Requirement: Business judgement uses detection labels against manual steps

纯 YOLO 模式 SHALL 使用检测类别名称与手动当前步骤名称进行既有判断；ROI 模式 SHALL 使用既有 ROI 规则。ROI 关闭且顺序约束开启时，系统 SHALL 仅使用完整 LSTM 动作事件的名称与配置步骤匹配，并检查该动作段的全部必需 YOLO 类别后推进 PASS；该模式的原 YOLO 单帧步骤更新 MUST NOT 生效。未配置类别 MUST NOT 自动成为新步骤。

#### Scenario: Detection label matches current manual step

- **WHEN** 纯 YOLO 路径当前等待 `2号`，`detection.label` 为 `2号`
- **THEN** 系统按既有 PASS 稳定帧和同帧数量规则推进当前步骤

#### Scenario: Detection label matches a future manual step

- **WHEN** 调用既有 YOLO 判断路径，当前等待 `2号`、检测标签为 `4号` 且该路径错序规则启用
- **THEN** 原 YOLO 引擎按既有 NG 稳定帧和错序规则处理；新动作模式不向该入口投递检测结果

#### Scenario: Detection label is not configured

- **WHEN** 模型返回的检测标签不属于当前纯 YOLO / ROI 配置类别
- **THEN** 系统不因此推进步骤，也不自动新增或修改类别

#### Scenario: 动作事件驱动顺序步骤

- **WHEN** 动作模式当前等待包装，完整包装事件结束且本段必需零件齐全
- **THEN** 包装 PASS 并推进下一步，YOLO 单帧是否存在同名“包装”不参与该次判定

### Requirement: Failed model reload preserves previous processor

系统 SHALL 在候选模型加载或模式契约校验失败时保留当前可用处理器和完整旧配置，向操作员显示保存失败原因。系统 MUST NOT 清空旧处理器或让旧处理器配合失败候选的类别运行。此规则只允许继续运行完整的旧配置，MUST NOT 在新动作模式发生运行时故障后自动降级为 YOLO。

#### Scenario: Selected model file is invalid

- **WHEN** 操作员选择无法加载的模型文件并保存配置
- **THEN** 系统提示模型加载失败
- **AND** 旧处理器和旧配置均保持不变，失败候选不写入有效配置

#### Scenario: 动作模式已启动后模型故障

- **WHEN** 已提交的动作模型在运行中发生异常
- **THEN** 系统停止业务放行并显示故障，不调用纯 YOLO 路径替代动作判定
