## Why

当前配置页选择新模型并保存后，运行中的推理处理器不会立即重新加载，现场会出现“已经选择新模型但检测仍无变化或不识别”的问题。

同时，业务步骤类别应由操作员手动配置，不能从模型类别自动写入配置。系统需要明确以模型推理返回的类别名称和手动配置的步骤类别名称进行业务判断。

## What Changes

- 保存配置后立即持久化到当前运行目录的 `config.json`，不再等到程序退出时才保存。
- 当模型路径或推理参数变化时，运行时重新创建并加载 YOLO/ONNX 推理处理器。
- 新模型加载成功后，替换相机推理线程使用的 `frame_processor`。
- 模型重载失败时保留旧处理器，不让现场运行态直接变成无模型可用。
- 配置里的步骤类别继续由操作员手动填写，不从模型类别自动填充或覆盖。
- 业务判断继续使用模型推理返回的 `detection.label` 与手动配置的步骤类别名称进行匹配。
- 保存配置后重建步骤状态机，让手动配置的类别、PASS/NG 稳定帧和错序规则立即生效。

## Capabilities

### New Capabilities
- `runtime-model-configuration`: 运行中保存配置、重载模型，并以手动配置类别作为业务顺序判断来源。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/ui/config_page.py`: 保存配置时发出配置已保存事件，不自动写入模型类别。
- `yolo_action_detection/src/ui/main_window.py`: 监听配置保存，重载推理处理器、重建步骤状态机、替换 worker 处理器。
- `yolo_action_detection/src/main.py`: 复用现有 `create_yolo_processor` 路径，避免重复实现模型加载逻辑。
- `yolo_action_detection/src/step_sequence/step_sequence_engine.py`: 业务逻辑保持不变，继续基于检测类别名称和手动配置类别名称判断。
- 测试范围：覆盖模型路径变化后的处理器重载、配置类别不被模型覆盖、步骤状态机使用手动配置类别。
