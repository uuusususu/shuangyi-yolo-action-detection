# YOLO OBB 动作检测程序

## 功能

- 使用 ONNX YOLO OBB 模型直接识别动作类别。
- 按配置的 1号 到 5号 类别顺序进行 PASS/NG 判定。
- 错序 NG 时播放 FAIL 声音，并可保存证据图片。
- 整轮 PASS 时播放 PASS 声音。
- 主界面显示产能、OK、NG、良率。
- 配置页支持生产统计归零并保存记录。
- 支持源码运行和 PyInstaller 便携打包。

## 源码启动

    cd yolo_action_detection
    ..\.venv\Scripts\python.exe src\main.py

## 便携打包

    cd yolo_action_detection
    .\packaging\build_yolo.bat

打包输出目录：

    yolo_action_detection/dist/YOLOActionDetection

目标电脑需要相机驱动。复制打包目录后可双击 YOLOActionDetection.exe 或 start.bat 启动。
