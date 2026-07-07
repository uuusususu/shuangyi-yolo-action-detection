# 双翼科技视觉行为引导系统

这是 YOLO OBB 动作检测桌面程序的发布分支。

## 内容

- PySide6 桌面运行程序源码
- ONNX 模型和默认配置
- PASS/FAIL 声音资源
- 相机、推理、动作顺序判定、证据保存、生产统计记录功能
- PyInstaller 打包脚本

## 启动

在本机开发环境中：

    cd yolo_action_detection
    ..\.venv\Scripts\python.exe src\main.py

打包后在目标电脑中双击 YOLOActionDetection.exe 或 start.bat。

## 分支

- main: YOLO OBB 动作检测正式程序
- google: Google/MediaPipe legacy 程序
