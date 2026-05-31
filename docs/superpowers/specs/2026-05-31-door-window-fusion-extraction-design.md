# 门窗检测框 × 纯点云融合提取设计

## 目的

在"下方 3D 点云视图点击 → 高亮该门/窗所有点"的纯点云提取基础上，引入上方全景影像的 2D 检测框作为约束，把两套独立判据融合，提升门窗提取的精度与语义正确性。

本设计在已实现的 `core/pointcloud_extract.py`（纯点云提取）和 `core/door_window_refine.py`（几何原语）之上新增一个编排层，不重写几何算法。

## 背景：两套判据各自的盲区

| 方法 | 擅长 | 盲区 |
|---|---|---|
| 2D 检测框反投影（视锥） | 角度范围准、**带语义**（门/窗可区分） | 反投影是一条数米深的视锥，分不清门和它身后的墙 |
| 纯点云提取（共面区域生长） | 深度准、平面/尺寸/矩形度几何准 | **无语义**；门与墙严格共面时会顺着整面墙蔓延 |

二者盲区正好互补，取交集即可互相弥补：

```text
2D 视锥   → 限定角度范围，区域生长不会蔓延到整面墙
3D 几何   → 解决视锥的深度歧义，只保留门那一层
2D 标签   → 提供几何无法推断的门/窗语义
```

## 目标

- 用当前选中 keyframe 的检测框（策略 A）作为候选范围与语义来源。
- 把纯点云区域生长约束在检测框对应的视锥候选集内。
- 用检测框 label 做门/窗尺寸校验，而非靠尺寸猜测。
- 输出一致性置信度：两路一致才判 high。
- 种子不落在任何检测框视锥内时，回退到纯点云路径，保持现有交互不变。

## 非目标

- 不做跨 keyframe 的自动选帧（策略 B），首版固定用当前帧。
- 不替换或重训 2D 检测器。
- 不解决"门与墙严格共面"这一几何固有歧义，仅用视锥角度约束缓解。
- 不做完整 LAS 的导出级提取。

## 当前流程（融合前）

下方点云点击 `MainWindow._on_cloud_point_clicked`：

1. 屏幕拾取得到种子点 index。
2. `extract_planar_region_from_seed` 做纯点云提取（局部平面 → 有界球内共面候选 → 连通域生长 → 法向剪枝 → 几何打分 → 平面薄带收紧）。
3. 高亮 mask 同步给上下两个视图。

问题：纯点云无语义，且齐平门会蔓延，远处门窗常被错标为 object。

## 融合流程（本设计）

下方点云点击时优先尝试融合，失败回退纯点云：

1. **视锥归属**：用 `project_points_to_panorama` 把全部点投到当前全景，`match_points_to_detections` 判断种子落在哪个检测框内（嵌套框取面积最小者，与 2D 引导路径一致）。无命中 → 返回 `source="none"`，调用方回退纯点云。
2. **视锥候选集 + 深度裁剪**：取该检测框内的点，再按"种子相机距离 ±depth_window"裁掉视锥里堆叠的远处墙面（`frustum_candidate_mask`）。
3. **约束式区域生长**：调 `extract_planar_region_from_seed(..., candidate_mask=锥体候选, label_hint=检测框label)`，生长被限制在视锥内，齐平门不再蔓延到整面墙。
4. **几何打分**：`score_door_window_geometry` 用检测框 label 的尺寸限制（而非 "object"）做竖直平面 / 尺寸 / 矩形度校验。
5. **置信度融合**：
   - 几何 high → `high`，reason `fused_frustum_and_planar_geometry`（两路一致）
   - 有点但几何未确认 → `medium`，reason `frustum_only_<原因>`
   - 无点 → `low`

核心原则：`2D 框 = 候选生成器 + 语义来源`，`3D 几何 = 最终验证器 + 边界精修`。

## 组件

### `core/door_window_fusion.py`（新增）

```python
@dataclass(frozen=True)
class FusedSelection:
    mask: np.ndarray
    point_count: int
    confidence: str          # high | medium | low | none
    reason: str
    label: str               # door | window | object
    source: str              # fused | pointcloud_only | none
    detection_index: int
    score: float | None
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None

def frustum_candidate_mask(points, cam_pos, R_pano, img_w, img_h, detection,
                           yaw_offset_deg=0.0, seed_idx=None,
                           depth_window=DEFAULT_FRUSTUM_DEPTH_WINDOW) -> np.ndarray: ...

def fuse_detection_and_pointcloud(points, seed_idx, detections, cam_pos, R_pano,
                                  img_w, img_h, yaw_offset_deg=0.0,
                                  frustum_depth_window=...) -> FusedSelection: ...

def should_highlight_fused(selection) -> bool: ...
```

该模块不含新几何运算，仅复用 `door_window`（视锥归属）、`projection`（点→像素）、`pointcloud_extract`（约束式生长 + 打分）。

### `core/pointcloud_extract.py`（扩展，向后兼容）

`extract_planar_region_from_seed` 新增两个可选参数：

- `candidate_mask`：把局部平面拟合、共面候选、连通域生长都限制在该 mask 内。
- `label_hint`：用指定 label 的尺寸限制做几何校验，并作为输出 label。

### `ui/main_window.py`（接线）

- `_on_cloud_point_clicked` 先调 `_try_fused_extraction(idx)`；若 `source=="fused"` 用融合结果（面板显示"融合提取 #框号 + label + 置信度"），否则回退 `extract_planar_region_from_seed`（面板显示"纯点云"）。
- `_try_fused_extraction` 用当前 keyframe 的 pose、检测框、yaw_offset 调融合函数。

## 默认参数

- 视锥深度窗口 `DEFAULT_FRUSTUM_DEPTH_WINDOW = 0.6 m`：裁掉视锥内堆叠在种子后方的远墙。
- 其余生长/平面/法向/薄带参数沿用 `pointcloud_extract` 既有默认值。

## 错误处理

- 种子越界 → `seed_out_of_range`。
- 无检测框 / 图像尺寸非法 → `no_detections`，回退纯点云。
- 种子不在任何视锥内 → `seed_not_in_any_frustum`，`source="none"`，回退纯点云。
- 种子在视锥内但 `candidate_mask` 不含种子（极端裁剪）→ 由 `extract_planar_region_from_seed` 返回 `seed_outside_candidate_mask`。

## 测试策略

单元测试（`tests/test_door_window_fusion.py`，合成点云）：

- 视锥 mask 覆盖门面投影像素。
- 融合把生长限制在视锥内、门点占多数、继承 door 标签。
- 种子在所有视锥外 → `source="none"`。
- 无检测 / 种子越界 → 清晰原因。

`tests/test_pointcloud_extract.py` 补充 `candidate_mask` / `label_hint` 行为。

真实数据验证（keyframe `608.675997_IMG`，逐检测框对比纯点云 vs 融合）：

- 标签：纯点云常错（door 标成 window/object），融合全部正确。
- 宽度/深度：融合显著更紧（如 0.71×2.07×3.02m 深 1.5m → 0.35×0.76×1.64m 深 0.75m）。

## 实测效果（5 个检测框）

| 维度 | 纯点云 | 融合 |
|---|---|---|
| 语义标签 | 经常错 | 全部继承正确 2D label |
| 选区宽/深 | 易蔓延、深度大 | 受视锥框定、更紧更薄 |
| 一致性置信度 | 无 | 两路一致才 high |

## 已知限制与后续

- 远处仅见局部的门窗，几何尺寸不达 door/window 下限 → 融合判 `medium`（仍按正确 label 高亮）。可按需放宽尺寸下限或将 medium 视作接受。
- 策略 A 依赖"目标门在当前帧视野内且检测框命中"，否则回退纯点云。后续可加策略 B：自动遍历能看到该点的 keyframe 选最佳视锥。
- 门窗与墙严格共面仍无法纯几何分离，靠视锥角度约束缓解。

## 建议

先用策略 A 跑实景评估。若回退纯点云的比例偏高（很多门不在当前帧），再实现策略 B 的多帧选取与多视锥融合。
