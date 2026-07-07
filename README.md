# Google/MediaPipe Legacy 动作检测程序

这是旧版 Google MediaPipe 动作检测桌面程序的发布分支。

## 内容

- PySide6 桌面程序源码
- MediaPipe 手势模型和目标检测模型
- 工业相机与 OpenCV 采集代码
- 配置文件、声音资源和 PyInstaller 打包脚本

## 启动

    cd opencv
    python src\main.py

## 打包

    cd opencv
    build.bat

本分支只保留 legacy 程序运行所需文件，不包含测试、OpenSpec 和开发文档。
