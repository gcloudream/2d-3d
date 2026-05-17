# 3D 门窗点云提取功能 — 可行性分析

> 状态：方案评估（未实现）
> 目的：基于现有"点云 + 全景影像同步"链路，做一个"点击点云的门/窗点 → 用 2D 模型在全景图上识别 → 反投影出 3D 门/窗点云区域"的交互式工具。
> 整体结论：**技术可行，但有 2 个 hard problem 决定最终效果**——OWLv2 漏检率、玻璃在 LiDAR 中信号弱。MVP 实现成本约 2-3 个工作日。

---

## 1. 需求拆解

完整的交互链路：

```
用户在 3D 视窗里点一个点（位于某扇门/窗上）
        │
        ▼  ① 点云屏幕拾取（已有：picking.find_nearest_to_mouse）
得到世界坐标 P = (X, Y, Z)
        │
        ▼  ② 点云 → 全景图像素（用现有 coordinate_to_pixel 公式）
得到全景像素 (u, v)
        │
        ▼  ③ 2D 模型在全景图上检测所有门/窗 bbox
若已对该 keyframe 缓存，直接用；否则跑一次 OWLv2
        │
        ▼  ④ 找出 (u, v) 命中的 bbox
得到目标门/窗的全景区域 [u₁, v₁, u₂, v₂]
        │
        ▼  ⑤ 反向投影：所有点云点 → 全景像素 → 落在 bbox 内则保留
得到候选点子集 S
        │
        ▼  ⑥ 深度/区域生长过滤
去掉 bbox 内但在墙后/前方的点（深度遮挡）
        │
        ▼  ⑦ 输出：高亮 + 可导出 .ply
```

---

## 2. 现有能力盘点

| 模块 | 已有 | 状态 |
|---|---|---|
| 3D 屏幕拾取（最近点） | `render/picking.py:find_nearest_to_mouse` | ✓ 可直接用 |
| Hover → 屏幕坐标 → 世界点 | `render/scene_view.py:_do_hover` | ✓ 可改造成 click |
| 点云 → 全景像素映射 | `算法例子/projectToPanoramic.py:coordinate_to_pixel` | ✓ 公式现成，需 numpy 矢量化 |
| 全景门窗检测 | `2d/detect_owlvit.py:detect_owlvit()` | ✓ 已是 Python 函数，能 import 调 |
| 全景切 6 透视 / NMS / 反投影 | `2d/detect_doorwindow.py:equirect_to_perspective` 等 | ✓ 可复用 |
| 同步坐标系（R_pano） | `render/camera.py` / `pano_sphere.py` | ✓ 已用同一套公式 |

可以说**80% 的底层零件都已经有**，缺的是把它们串起来的 orchestration 层。

---

## 3. 技术可行性 — 逐步分析

### ① 点击拾取（无风险）

`picking.find_nearest_to_mouse` 已经在做这件事，只是当前只在 hover 时触发、节流 30ms、距离 14px 内吸附。改造成 click handler 一行代码的事：

```python
def mousePressEvent(self, e):
    if e.button() == Qt.LeftButton and not self._dragging:
        # 用同样的 picking 拿到 idx
        idx = find_nearest_to_mouse(...)
        if idx >= 0:
            self._emit_door_window_pick(idx)
```

**风险**：低。点云抽样到 30 万点，picking 屏幕距离 < 14px 已经能稳定锁到正确点。

**注意**：抽样导致原始 5300 万点里的细节会被错过——门窗边缘那一圈点不一定在抽样里。但作为"种子点"够用了。

### ② 世界点 → 全景像素（无风险）

`coordinate_to_pixel(R_pano @ (P - cam_pos))`。直接拿用就行。

**已验证结论**：用 `3d_viewer/validate_projection.py` 生成 raw `0°` 与 offset `-90°` 的落点对比图后，选择 **offset `-90°`（青色点）** 作为后续点云 → 全景 bbox 命中的坐标规则。

也就是说，虽然 OWLv2 看到的是磁盘上的原始全景图，但点云点投到该全景图像素时，需要和当前全景球 shader 的视觉校准保持一致：

```python
u_pix = (u_pix_raw + img_w * yaw_offset_deg / 360.0) % img_w
# 当前数据集使用 yaw_offset_deg = -90.0
```

后续实现中应把该 offset 作为数据集/会话级配置传入投影函数，而不是在门窗提取逻辑里重新手写一套偏移。

### ③ 全景门窗检测（**主要风险点**）

调 `detect_owlvit(model, processor, panorama_image, "mps", classes=["door","window"], score_thr=0.15)`，按现有 pipeline 加上 `--auto-pano` 切 6 视图 + NMS。

**风险高的两件事**：

**(a) 推理速度**

| 配置 | 单图延迟（M3 + 24G MPS）|
|---|---|
| OWLv2-base + 6 视图 + 1024 输入（默认） | ~30s |
| OWLv2-base + 4 视图 + 768 输入 | ~10s |
| 不切全景（直接喂 5760×2880）| ~3s，但精度差很多 |

**30 秒一次点击不能接受**。两个对策：

- **缓存**：第一次进 keyframe 时跑一次后台检测，结果缓存到 `panorama_name -> bbox列表`。之后该 keyframe 上的点击都是 < 50ms。
- **局部检测**：把点击方向附近 ±60° 切出一张 1024×1024 透视图，单图推理 ~3s。Bbox 直接在透视图上做，最后用 `persp_bbox_to_pano` 反投回去。

MVP 用缓存方案最简单：用户切到 keyframe 时后台开始检测（Qt 的 QThread），右下角显示 "检测中…"，完成后点击响应是即时的。

**(b) OWLv2 检测质量**

来自 `2d/README.md` 的实测数据：

> OWLv2 base + 6 视图 + 1024 输入 → 4 个有意义的 window 框（753 张里）
> "OWLv2 对 base 版仍有漏检，特别是被遮挡或姿态特殊的门"
> "玻璃门 vs 落地窗、橱柜门 vs 真实通行门 在视觉上接近，开放词汇模型本身分得不彻底"

也就是说**这个模型本身就不是高精度门窗检测器**。预计：
- 真实门：70-85% 召回。
- 真实窗：50-70% 召回（玻璃容易和墙混淆）。
- Bbox 紧致度：通常会带一圈墙边，可能多 10-20% 的面积。

升级到 `owlv2-large-patch14-ensemble`（2.7GB）能再提 5-10 个点，M3+24G 跑得动但速度更慢。

**用户体验影响**：
- 点击之后报"未检测到对应门窗"会比较常见，需要明确的失败提示。
- 检测到但 bbox 偏大，会导致提取出来的点云带着一圈墙——需要后续步骤过滤。

### ④ 点击点 → 命中的 bbox（低风险）

简单的点在矩形内判断。需要处理：

- **接缝跨越**：bbox 可能是 `[5500, ..., 6200, ...]`（图宽 5760，跨过左右接缝）。`detect_owlvit.py` 已用 `bbox_iou_wrap` 处理过，沿用即可。
- **多 bbox 命中**：点击点同时落在两个 bbox 里（门里面的窗），按面积小者优先 或 按 score 高者优先。

### ⑤ 反向投影（中等风险，主要是性能）

矢量化版 coordinate_to_pixel：

```python
def project_points_to_panorama(P: (N,3), R: (3,3), cam_pos: (3,), img_w, img_h) -> (N,2):
    rel = P - cam_pos[None, :]               # (N, 3)
    p = rel @ R.T                            # (N, 3) in pano local
    horiz = np.hypot(p[:,0], p[:,1])
    # 水平角
    tm_h = np.where(p[:,1] != 0, np.arctan(-p[:,0] / np.where(p[:,1]!=0, p[:,1], 1)), 0)
    left = p[:,0] < 0
    front = p[:,1] > 0
    # 四象限分支 → 向量化（用 np.where 嵌套或一次性 mask）
    # ... 详见 coordinate_to_pixel 原 Python 代码，转 numpy
    return np.column_stack([u_pix, v_pix])
```

30 万点矢量化大约 10-20ms，落在 bbox 里的过滤再 5ms。**总成本可接受**。

**风险**：bbox 接缝跨越时的点过滤逻辑要仔细写：

```python
if bbox_x2 < bbox_x1:  # 跨接缝
    mask = (u >= bbox_x1) | (u <= bbox_x2)
else:
    mask = (u >= bbox_x1) & (u <= bbox_x2)
```

### ⑥ 深度 / 遮挡过滤（**第二个主要风险点**）

Bbox 投影回 3D 后会拉出一根从相机出发的"视锥",bbox 里的点可能：
- 在门/窗本体上 ✓
- 在门/窗后面（穿过玻璃看见的）✗
- 在门/窗前面（前景遮挡）✗
- 同一视线方向上其他平面（罕见）

**方案 A：用户点击点 + 欧氏距离**

```python
seed = clicked_world_point
keep = np.linalg.norm(candidate_points - seed, axis=1) < 2.0  # 2 米半径
```

**优点**：实现一行。**缺点**：2 米半径阈值不通用——大窗户可能跨度 3 米，狭长玻璃门也是。

**方案 B：法线一致性 + 区域生长（推荐）**

1. 从种子点开始
2. 对邻居（kd-tree 半径 0.2m 内）求法线
3. 法线夹角 < 15° 且属于 bbox 候选集的点纳入
4. 迭代直到不再增长

3D 平面区域生长，OpenCV / PCL / Open3D 里都是标准操作。Open3D 用 `o3d.geometry.PointCloud.cluster_dbscan` + `estimate_normals` 一二十行能写完。

**方案 C：RANSAC 平面拟合**

把候选点喂给 RANSAC 平面模型，inlier 阈值 0.05m。

- 真实窗户：通常 90%+ 在一个平面上 → 收敛干净
- 半开门 / 弧形门面：失败
- 玻璃门：LiDAR 点很少 → RANSAC 不收敛

**最稳的实现**：方案 B 兜底，方案 C 作为后处理可选——平面拟合成功就吐 inliers，失败就直接吐区域生长结果。

### ⑦ UI（低风险）

需要新加的 UI：
- 鼠标左键点击模式切换（"导航" vs "门窗提取"）
- 检测中的 spinner / 进度条
- 检测结果列表（左侧/右侧）
- 高亮显示提取结果（已有 `point_cloud.py:highlight_idx`，扩成 `highlight_mask`）
- 导出按钮（→ .ply / .las）

---

## 4. 主要挑战汇总

| # | 挑战 | 严重程度 | 缓解方案 |
|---|---|---|---|
| 1 | OWLv2 漏检（特别是窗户） | 🔴 高 | 多 prompt + 升级 large 版 + UI 报失败 + 允许"无 2D 模型，纯距离阈值"兜底 |
| 2 | 推理 30s 卡顿 | 🔴 高 | 切 keyframe 时后台预跑 + 缓存 |
| 3 | Bbox 偏大带墙边 | 🟡 中 | 区域生长 / 平面 RANSAC 后处理 |
| 4 | 玻璃 LiDAR 信号弱 | 🟡 中 | 接受现实：用相邻窗框作种子，不直接用玻璃中心 |
| 5 | 抽样点云的边缘缺失 | 🟡 中 | 提取阶段加载完整 LAS 流式过滤（53M 点 numpy 矢量化也就 1-2s） |
| 6 | 深度遮挡误判 | 🟡 中 | 区域生长 + 法线一致性 |
| 7 | 接缝跨越的 bbox | 🟢 低 | 用现成 wrap 函数 |
| 8 | 多门窗叠加（门里嵌窗） | 🟢 低 | 面积小者优先 |

---

## 5. 实施建议（分阶段）

### Phase 1 — MVP（1-2 天）

**目标**：端到端跑通，验证可行性。允许失败率高、UI 粗糙。

- [ ] `core/projection.py` 加 `project_points_to_panorama(points, R, cam_pos, w, h)` 矢量化版
- [ ] `core/door_window.py` 新文件：封装 detection + bbox matching + bbox-to-points 过滤（无区域生长）
- [ ] `render/scene_view.py` 加 click 模式（用 Qt action 切换）
- [ ] `render/point_cloud.py` 把 `highlight_idx: int` 扩成 `highlight_mask: np.ndarray` (uint8 buffer)
- [ ] `ui/main_window.py` 加按钮 / 状态指示
- [ ] 每个 keyframe 切换时后台跑一次 OWLv2，结果缓存到 `dataset.panorama_name -> List[bbox]`

**预期效果**：~60% 的门窗一次点击能拿到一个**大致正确但带墙边**的区域。

### Phase 2 — 提取质量（再 1 天）

- [ ] 引入 Open3D，加 KD-Tree + 法线估计 + 区域生长
- [ ] Bbox 候选集进入区域生长，作为约束
- [ ] 加载完整 LAS（而非抽样）用于提取
- [ ] 导出 .ply

**预期效果**：~75% 的门窗能拿到**形状大致正确的 3D 区域**，可作为标注 / 测量基础。

### Phase 3 — 鲁棒性（视需求）

- [ ] 升级 OWLv2 large
- [ ] 允许用户手动调 bbox（在 2D 全景视图上拖动）
- [ ] 允许"纯 3D 区域生长"模式（不依赖 OWLv2，纯从种子点 + 法线 + 平面拟合）
- [ ] 多视角融合：同一扇门在多个 keyframe 上都能看到，融合多次检测的 bbox 提升精度

---

## 6. 预期效果（量化估计）

基于 OWLv2 已知性能 + 区域生长收敛率：

| 场景 | 一次点击成功率 | 提取边缘精度 |
|---|---|---|
| 室内木门，正面对齐 | 85-90% | < 5cm |
| 室内玻璃门 | 60-75% | 5-15cm（玻璃点稀疏） |
| 室内常规窗 | 70-80% | < 10cm |
| 落地窗 / 大面玻璃 | 50-65% | 10-20cm |
| 部分遮挡（柜子前的门） | 40-55% | 难以预估 |
| 极端姿态（斜视） | 30-45% | 差 |

**单次完整流程延迟**：

| 阶段 | 时间 |
|---|---|
| Picking + 投影 + bbox 匹配 | < 50ms |
| 反向投影 + bbox 过滤（30 万抽样） | ~20ms |
| 反向投影 + bbox 过滤（5300 万原始） | ~1-2s |
| 区域生长（候选点 ~10K） | ~200ms |
| **首次切 keyframe 时的后台检测** | ~30s（一次性） |

进入一个 keyframe 时**忍 30 秒**，之后每次点击 **2 秒内**反馈。

---

## 7. 风险评估 — 最坏情况

**如果 OWLv2 完全失效**（漏检 / 误检率太高）：

退路是 **Phase 3 的"纯 3D 区域生长"模式**——只要用户能精确点到门/窗上，就能用法线 + 平面拟合提取，不依赖 2D 模型。这种模式：
- 不能区分门和窗（用户得手动 label）
- 但能保证只要点对了就能提出一个连通的、共面的区域
- 把 2D 检测降级为"语义标签"而非"区域定义"

这个 fallback 让整个特性的下界变成"3D 标注辅助工具"，不会完全做不出来。

---

## 8. 综合判断

| 维度 | 评分 |
|---|---|
| 技术可行性 | 8/10（公式现成、数据现成、模型现成） |
| 实现成本 | 中等（MVP 2-3 天，完整 1 周） |
| 预期效果（普通门窗） | 7/10 |
| 预期效果（玻璃 / 极端姿态） | 4/10 |
| **能否端到端 demo** | ✓ |
| **能否生产使用** | 取决于业务对精度的容忍度，建议先做 MVP 实测 |

**推荐做法**：

1. 先做 **Phase 1 MVP + 后台缓存**，跑一遍 753 张 keyframe 的统计：每张图能检出多少门/窗，bbox 形状如何
2. 如果统计可接受 → 上 Phase 2
3. 如果 OWLv2 召回低于 50% → 评估升级 large 或切换到 GroundingDINO / SAM2 等替代方案

---

## 9. 相关文件

- `算法例子/projectToPanoramic.py` — coordinate_to_pixel 公式
- `2d/detect_owlvit.py` — OWLv2 检测主流程
- `2d/detect_doorwindow.py` — 全景透视切片 / NMS / 反投影工具
- `3d_viewer/render/picking.py` — 屏幕拾取
- `3d_viewer/render/camera.py` — 同步相机
- `3d_viewer/render/pano_sphere.py` — 全景球（含 R_pano 应用方式）
- `3d_viewer/PANORAMA_YAW_ALIGNMENT.md` — yaw offset 由来（实施时要避免混淆"渲染 offset"和"投影 offset"）
