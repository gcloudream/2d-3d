# Point Cloud Panorama Unified Viewer

本地桌面版点云/全景同空间查看器。

## 功能

- 一个 OpenGL 视窗同时显示全景影像和点云。
- 当前全景图作为 360 背景，点云先按 `算法例子/projectToPanoramic.py` 同一套球面公式投到全景图上，再一起显示。
- 鼠标拖动旋转空间时，显示的是“带点云的全景球面”，全景影像和点云共用同一个球面坐标关系。
- 图片列表切换全景相机位置。
- 支持显示/隐藏全景、显示/隐藏点云、调整点大小、重置视角。

## 运行

```bash
cd /Users/gengchen/Desktop/3dtiqu
uv pip install --python .venv/bin/python -r desktop_viewer/requirements.txt
.venv/bin/python desktop_viewer/app.py
```

默认读取：

```text
/Users/gengchen/Desktop/3dtiqu/20260129135824/CAM/camera_pos.cam
/Users/gengchen/Desktop/3dtiqu/20260129135824/CAM/
/Users/gengchen/Desktop/3dtiqu/20260129135824/LAS_Rgb/20260129135824_rgb_0.las
```

完整点云超过 5300 万点，界面会自动抽样到约 26 万点用于实时显示。

## 操作

- 鼠标左键拖动：旋转当前全景空间。
- 鼠标滚轮：调整视场角。
- 右侧图片列表：切换当前全景相机位置。
- `显示全景影像`：开关 360 背景。
- `显示点云`：开关叠加点云。
