"""无图形界面冒烟测试：跑通数据加载 + Camera 矩阵 + picking 项目 + KDTree 查询。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = HERE.parent

import numpy as np

from core.dataset import find_default_dataset, load_dataset
from core.projection import rotation_from_angle
from render.camera import Camera
from render.picking import find_nearest_to_mouse


def main():
    cfg = find_default_dataset(WORKSPACE)
    assert cfg is not None, "no dataset"
    print(f"[cfg] {cfg.data_root.name}")
    print(f"      cam={cfg.camera_file.relative_to(WORKSPACE)}")
    print(f"      las={cfg.pointcloud_file.relative_to(WORKSPACE)}")

    d = load_dataset(cfg, max_points=80_000)  # 测试用小一点
    print(f"[load] poses={len(d.poses)} points={d.points.shape} colors={d.colors.shape} "
          f"total={d.total_points:,} step={d.sample_step}")
    print(f"[bbox] xyz min={d.points.min(0)} max={d.points.max(0)}")

    p0 = d.poses[0]
    print(f"[pose0] {p0.image_name}  pos=({p0.x:.2f},{p0.y:.2f},{p0.z:.2f})  "
          f"rpy=({np.degrees(p0.roll):.1f},{np.degrees(p0.pitch):.1f},{np.degrees(p0.yaw):.1f})")

    R = rotation_from_angle(p0.roll, p0.pitch, p0.yaw)
    print(f"[R] det={np.linalg.det(R):.4f}  (~1.0)")

    cam = Camera()
    cam.set_keyframe(p0.position, p0.roll, p0.pitch, p0.yaw)
    cam.fov_deg = 75.0
    proj = cam.proj_matrix(16/9)
    view = cam.view_matrix()
    mvp = (proj @ view).astype(np.float32)
    print(f"[cam] pos={cam.position}  yaw={cam.yaw_deg}  pitch={cam.pitch_deg}")

    # 中心点附近的 picking
    cx, cy = 800, 450
    idx = find_nearest_to_mouse(d.points, mvp, 1600, 900, cx, cy, max_dist_px=200)
    from render.picking import project_points_to_screen
    screen_xy, visible = project_points_to_screen(d.points, mvp, 1600, 900)
    if idx >= 0:
        assert visible[idx], "picked point is outside the view frustum"
        xyz = d.points[idx]
        print(f"[pick] idx={idx}  xyz={xyz}  screen={screen_xy[idx]}")
    else:
        print("[pick] none")

    # 验证投影：随机选 1000 点投到屏幕，多少在视锥内
    sub = d.points[::max(1, len(d.points)//1000)]
    sxy, vis = project_points_to_screen(sub, mvp, 1600, 900)
    print(f"[proj] {vis.sum()}/{len(sub)} in frustum")
    if vis.sum() == 0:
        print("[warn] 0 points in frustum - camera or projection may be wrong")

    print("\n[OK] smoke test passed")


if __name__ == "__main__":
    main()
