# YOLO OBB 动作检测程序

这是主程序目录。完整项目介绍请查看仓库根目录 README。

## 启动

    cd yolo_action_detection
    ..\.venv\Scripts\python.exe src\main.py

## 打包

    cd yolo_action_detection
    .\packaging\build_yolo.bat

## 关键目录

| 路径 | 说明 |
| --- | --- |
| src/ui | PySide6 主界面和配置页 |
| src/yolo_runtime | ONNX OBB 推理 |
| src/step_sequence | 动作顺序状态机 |
| src/camera | 相机采集与 SDK 封装 |
| src/detection_logging | 声音、证据和生产统计记录 |
| config | 模型和配置 |
| assets | 声音与 README 截图 |
| packaging | 打包脚本 |
