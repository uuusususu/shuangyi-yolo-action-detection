# 双翼科技视觉行为引导系统

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySide6](https://img.shields.io/badge/UI-PySide6-2ea44f)
![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX_Runtime-orange)
![YOLO OBB](https://img.shields.io/badge/Model-YOLO_OBB-purple)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

基于 YOLO OBB、ONNX Runtime 和 PySide6 的工业动作防呆检测桌面系统。项目面向产线快速装配场景，通过工业相机实时采集画面，识别 1号 到 5号 动作类别，并按配置顺序完成 PASS / NG 判定、声音提示、证据保存和生产统计归档。

这个仓库是一个求职展示项目版本：只保留运行所需源码、模型、配置、资源和打包脚本，不包含开发期测试、OpenSpec 和草稿文档。

## 项目展示

### 主检测界面

![主检测界面](yolo_action_detection/assets/screenshots/main-window.png)

### 配置与生产统计

![配置界面](yolo_action_detection/assets/screenshots/config-page.png)

## 解决的问题

在产线装配中，动作节拍快、错误步骤不容易被人工及时发现。该系统将模型识别结果直接映射到业务步骤，形成实时防呆闭环：

- 正确动作顺序：按 1号 到 5号 依次 PASS。
- 错误顺序：第 1 步已完成后，后续步骤错序达到 NG 稳定帧立即 FAIL。
- 现场追溯：NG 时保存证据图片和 metadata。
- 生产统计：统计总数、OK、NG、良率，并支持手动归零归档。
- 交付部署：PyInstaller 打包后可复制到无 Python 的目标电脑运行。

## 核心能力

| 模块 | 能力 |
| --- | --- |
| 实时采集 | 支持工业相机 SDK 和 OpenCV 采集链路 |
| 模型推理 | 使用 ONNX YOLO OBB 模型进行实时动作类别识别 |
| 顺序判定 | 使用状态机管理 PASS、ACTION_NG、重新武装和去重 |
| 结果反馈 | PASS / FAIL 声音提示，FAIL 证据图片保存 |
| 生产统计 | 总数、OK、NG、良率展示，归零时生成统计记录 |
| 配置中心 | 模型路径、阈值、稳定帧、声音、证据、相机参数可配置 |
| 便携交付 | PyInstaller 打包，目标电脑无需安装 Python |

## 系统架构

    工业相机 / OpenCV
          |
          v
    CameraWorker 实时采集
          |
          v
    ONNX OBB Processor
          |
          v
    StepSequenceEngine 顺序状态机
          |
          +--> PySide6 主界面状态刷新
          +--> PASS / FAIL 声音反馈
          +--> FAIL 证据保存
          +--> ProductionStats 生产统计归档

## 技术栈

| 分类 | 技术 |
| --- | --- |
| 语言 | Python 3.11 |
| 桌面 UI | PySide6 |
| 视觉处理 | OpenCV |
| 推理引擎 | ONNX Runtime |
| 检测模型 | YOLO OBB |
| 工业相机 | MvSDK 封装 |
| 打包 | PyInstaller |
| 配置与记录 | JSON |

## 工程亮点

### 1. 模型类别直接驱动业务步骤

模型不再依赖“扭力枪与孔位交集”的二次业务判断，而是直接输出 1号 到 5号 类别。状态机只关注当前配置步骤是否出现，减少现场误差来源，让业务逻辑更接近产线真实动作。

### 2. 适合快节拍产线的 PASS / NG 稳定策略

PASS 和 NG 使用独立稳定帧配置。默认 PASS 1 帧、NG 2 帧，减少快速动作下的漏判，同时避免单帧抖动导致误报警。

### 3. 第 1 步 PASS 后才允许 NG

新一轮必须先完成第 1 步，才会允许后续步骤触发错序 NG。这样可以过滤上一件产品残影、模型短暂误输出和画面噪声。

### 4. 终止式 NG 和自动下一轮

检测到错序 NG 后，当前产品立即结束并记录 FAIL，系统自动进入下一轮第 1 步等待。重新检测到第 1 步后才允许下一次 NG，避免同一错误动作连续刷多次 FAIL。

### 5. 生产统计批次归档

开始检测、停止检测和关闭相机都不会清零统计。只有配置页的“归零并保存记录”会先写入生产记录，再清零并开启新批次，符合真实生产场景。

### 6. 便携部署

打包目录包含模型、配置、声音和运行依赖。目标电脑只需要相机驱动，复制目录后即可双击运行。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| yolo_action_detection/src/main.py | 程序入口和便携路径初始化 |
| yolo_action_detection/src/ui | PySide6 主界面、配置页和科技风组件 |
| yolo_action_detection/src/camera | 相机采集、SDK 封装和推理线程 |
| yolo_action_detection/src/yolo_runtime | YOLO OBB / ONNX Runtime 推理 |
| yolo_action_detection/src/step_sequence | 动作顺序状态机 |
| yolo_action_detection/src/detection_logging | 声音反馈、FAIL 证据、生产统计记录 |
| yolo_action_detection/config | 默认配置和 ONNX 模型 |
| yolo_action_detection/assets | PASS / FAIL 声音与展示截图 |
| yolo_action_detection/packaging | PyInstaller 打包脚本 |

## 快速启动

源码运行：

    cd yolo_action_detection
    ..\.venv\Scripts\python.exe src\main.py

便携打包：

    cd yolo_action_detection
    .\packaging\build_yolo.bat

打包输出：

    yolo_action_detection/dist/YOLOActionDetection

目标电脑需要安装相机驱动。复制打包目录后，双击 YOLOActionDetection.exe 或 start.bat 启动。

## 关键配置

| 配置项 | 说明 |
| --- | --- |
| yolo_model_path | ONNX / YOLO 模型路径 |
| category_names | 动作类别顺序，默认 1号 到 5号 |
| action_pass_stable_frames | PASS 稳定帧 |
| action_ng_stable_frames | NG 稳定帧 |
| action_order_constraint_enabled | 是否启用错序 NG |
| sound_feedback_enabled | 是否启用 PASS / FAIL 声音 |
| fail_evidence_enabled | 是否保存 FAIL 证据图片 |

## 输出与记录

| 输出 | 路径 |
| --- | --- |
| FAIL 证据图片 | yolo_action_detection/outputs/evidence |
| 生产统计记录 | yolo_action_detection/outputs/production_records |
| 便携 smoke 日志 | dist/YOLOActionDetection/logs |

## 验证

发布前已完成以下验证：

- 源码 portable smoke：通过
- 模型路径、声音资源、证据目录：通过
- MvSDK 可用性检查：通过
- main 分支只保留运行文件：46 个 tracked files
- google 分支只保留 legacy 运行文件：37 个 tracked files

## 分支说明

| 分支 | 说明 |
| --- | --- |
| main | YOLO OBB 动作检测正式版本 |
| google | Google / MediaPipe legacy 版本 |

## 求职展示重点

这个项目展示的不是单纯的模型调用，而是一个完整的工业视觉桌面系统：

- 视觉模型接入
- 实时推理链路
- 业务状态机
- 工业相机适配
- 桌面 UI 设计
- 现场配置能力
- 证据追溯
- 生产统计
- 便携部署

适合用于展示计算机视觉、桌面客户端、生产现场工具和工程交付能力。
