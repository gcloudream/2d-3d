# 3d_viewer — 点云 + 全景统一查看器

PySide6 + ModernGL 实现的桌面查看器：

- 点云和全景共用同一个相机
- 全景作为天空盒，永远以相机为球心（背景策略 B）
- Keyframe 跳跃（在该点上转视角，不能自由飞行）
- 鼠标 hover 时吸附最近点云点，显示 (X, Y, Z) + RGB
- 当前 keyframe 门窗检测、检测框显示、点云区域高亮
- 内置平铺全景框编辑器，可手动画/删 door/window 框并回写到 3D 视图

## 安装

```bash
cd /Users/gengchen/Desktop/3dtiqu
.venv/bin/pip install -r 3d_viewer/requirements.txt
```

## 运行

macOS 推荐直接双击或执行根目录启动脚本：

```bash
cd /Users/gengchen/Desktop/3dtiqu
./run_3d_viewer.command
```

也可以手动启动：

```bash
cd /Users/gengchen/Desktop/3dtiqu
.venv/bin/python 3d_viewer/main.py
```

默认从工作区里查找第一个含 `CAM/camera_pos.cam` 的目录。

## 操作

| 操作 | 功能 |
|---|---|
| 鼠标左键拖动 | 转视角（yaw / pitch） |
| 滚轮 | 缩放 (FoV 20–110°) |
| ←/→ | 上/下一个 keyframe |
| 鼠标移动（不点击） | 吸附最近点云点，右侧显示 xyz + rgb |
| 1 | 显隐全景 |
| 2 | 显隐点云 |
| R | 重置视角 |

## 门窗检测与手工框

右侧面板提供门窗相关操作：

- `检测模式`：`精准模式` 用于减少误检，`召回模式` 用于尽量找全门窗。
- `检测当前帧`：对当前 keyframe 全景图运行 OWLv2 检测，结果保存到 `out/door_window_detections/`。
- `显示检测框`：在 3D 全景视图中显示/隐藏当前帧检测框。
- `门窗选择`：开启后点击点云点，如果该点投影落入 door/window 框，先得到 bbox 粗候选点，再按点击点附近的 3D 连通性和深度一致性精修高亮区域。
- `编辑当前全景框`：在主窗口左侧切换到平铺全景编辑器，不打开浏览器。

主窗口左侧普通模式为上下双视图：上方是全景影像、点云和检测框重叠的操作视图；下方是纯 3D 点云观察视图。门窗选择在上方完成，精修后的点云会同步高亮到下方纯 3D 视图，方便旋转查看空间位置。

平铺全景编辑器操作：

- 拖拽画框；
- 点击已有框选中；
- `Delete` 删除选中框；
- 通过类别下拉框设置 `window` 或 `door`；
- 点击 `编辑结束` 保存并返回 3D 视图；
- 点击 `取消编辑` 放弃本次编辑并返回。

手工框保存到 `out/door_window_annotations/<image_stem>.json`。3D Viewer 会优先读取手工框，其次读取模型检测结果，因此手工修正可以覆盖漏检和误检。

门窗点云提取分两层：全景 bbox 只提供粗范围，最终高亮会用点击点所在的 3D 连通区域和深度一致性做精修。玻璃、屏幕、黑色反光物体附近仍可能因为 LiDAR 点稀疏而低置信度，右侧状态栏会显示 coarse/refined 点数和原因。

## 目录结构

```
3d_viewer/
├── main.py                    入口
├── core/
│   ├── annotations.py         手工门窗框 JSON 保存
│   ├── dataset.py             camera_pos.cam 解析 + LAS 抽样加载
│   ├── detection_cache.py     检测/手工框缓存路径查找
│   ├── detection_runner.py    当前帧 OWLv2 检测调用
│   └── projection.py          rotation_from_angle (与算法例子保持一致)
├── render/
│   ├── camera.py              单一 Camera 对象
│   ├── pano_sphere.py         全景球 shader (在 fragment 里复刻 coordinate_to_pixel)
│   ├── point_cloud.py         点云 VBO + 高亮 shader
│   ├── picking.py             屏幕空间最近点查询
│   └── scene_view.py          ModernGL widget，组合渲染
└── ui/
    ├── main_window.py         Qt 主窗口
    └── pano_annotation_editor.py  内置平铺全景框编辑器
```

## 性能 & 限制

- 点云抽样到 30 万点（默认 max_points=300_000，原始点数 5300 万）
- hover 节流到 30ms
- 30 万点 hover 查询约 20–30 ms（和机器、视角内可见点数量有关）
- M3 + 24GB 单帧渲染 < 5ms，FoV 拖动流畅
- 仅 keyframe 模式，相机 position 不能自由移动

## 与已有代码的关系

- 数据解析 / LAS 抽样思路沿用 `desktop_viewer/data.py`
- `rotation_from_angle` 复刻 `算法例子/projectToPanoramic.py`，确保全景贴图和原始算法的坐标系一致
- 旧的 `desktop_viewer/` 是 OpenGL 2.1 + 全景 overlay 范式（点云投到全景图上当像素），与本目录不重叠

## 坐标校准说明

当前数据的点云和全景影像在水平平面存在稳定 90° 基准轴差异，UI 默认使用 `-90°` 全景水平校准。

详见 [`PANORAMA_YAW_ALIGNMENT.md`](PANORAMA_YAW_ALIGNMENT.md)。
