# 3d_viewer — 点云 + 全景统一查看器

PySide6 + ModernGL 实现的桌面查看器：

- 点云和全景共用同一个相机
- 全景作为天空盒，永远以相机为球心（背景策略 B）
- Keyframe 跳跃（在该点上转视角，不能自由飞行）
- 鼠标 hover 时吸附最近点云点，显示 (X, Y, Z) + RGB

## 安装

```bash
cd /Users/gengchen/Desktop/3dtiqu
.venv/bin/pip install -r 3d_viewer/requirements.txt
```

## 运行

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

## 目录结构

```
3d_viewer/
├── main.py                    入口
├── core/
│   ├── dataset.py             camera_pos.cam 解析 + LAS 抽样加载
│   └── projection.py          rotation_from_angle (与算法例子保持一致)
├── render/
│   ├── camera.py              单一 Camera 对象
│   ├── pano_sphere.py         全景球 shader (在 fragment 里复刻 coordinate_to_pixel)
│   ├── point_cloud.py         点云 VBO + 高亮 shader
│   ├── picking.py             屏幕空间最近点查询
│   └── scene_view.py          ModernGL widget，组合渲染
└── ui/
    └── main_window.py         Qt 主窗口
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
