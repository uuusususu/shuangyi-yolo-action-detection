# Google/MediaPipe Legacy 程序

## 功能

- 使用 Google MediaPipe Gesture Recognizer task 模型识别手势。
- 可选加载 MediaPipe Object Detector tflite 模型。
- 支持工业相机采集、OpenCV 预览、PySide6 桌面 UI。
- 支持 JSONL 检测日志和 PyInstaller 打包。

## 源码启动

    cd opencv
    python src\main.py

## 主要模型

    opencv/config/models/gesture_recognizer.task
    opencv/config/models/box_detector.tflite

## 打包

    cd opencv
    build.bat
