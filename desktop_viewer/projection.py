from __future__ import annotations

import numpy as np


def rotation_from_angle(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Match the rotation matrix used by the original projection script."""
    roll = -roll
    pitch = -pitch

    rot = np.zeros((3, 3), dtype=np.float64)
    rot[0, 0] = np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw)
    rot[0, 1] = -np.cos(roll) * np.sin(yaw) + np.sin(pitch) * np.sin(roll) * np.cos(yaw)
    rot[0, 2] = np.cos(pitch) * np.sin(roll)

    rot[1, 0] = np.cos(pitch) * np.sin(yaw)
    rot[1, 1] = np.cos(pitch) * np.cos(yaw)
    rot[1, 2] = -np.sin(pitch)

    rot[2, 0] = -np.sin(roll) * np.cos(yaw) + np.sin(pitch) * np.cos(roll) * np.sin(yaw)
    rot[2, 1] = np.sin(roll) * np.sin(yaw) + np.sin(pitch) * np.cos(roll) * np.cos(yaw)
    rot[2, 2] = np.cos(roll) * np.cos(pitch)
    return rot


def project_points_to_pano(
    points: np.ndarray,
    camera_position: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to equirectangular image pixels.

    Returns:
        uv: Nx2 pixel coordinates.
        valid: N boolean mask for finite projected pixels.
    """
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool)

    rot = rotation_from_angle(roll, pitch, yaw)
    local = (rot @ (points.astype(np.float64) - camera_position.reshape(1, 3)).T).T
    x = local[:, 0]
    y = local[:, 1]
    z = local[:, 2]

    tm_h = np.zeros_like(x)
    nonzero_y = y != 0
    tm_h[nonzero_y] = np.arctan(-x[nonzero_y] / y[nonzero_y])

    u = np.empty_like(x, dtype=np.int64)
    left = x < 0
    front = y > 0

    mask = left & front
    u[mask] = np.rint(image_height - image_height * tm_h[mask] / np.pi).astype(np.int64)
    mask = left & ~front
    u[mask] = np.rint(image_height * (-tm_h[mask]) / np.pi).astype(np.int64)
    mask = ~left & front
    u[mask] = np.rint(image_height - image_height * tm_h[mask] / np.pi).astype(np.int64)
    mask = ~left & ~front
    u[mask] = np.rint(image_width - image_height * tm_h[mask] / np.pi).astype(np.int64)
    u %= image_width

    horizontal_dist = np.sqrt((x * x) + (y * y))
    tm_v = np.empty_like(z)
    nonzero_z = z != 0
    tm_v[nonzero_z] = np.arctan(horizontal_dist[nonzero_z] / z[nonzero_z])
    tm_v[~nonzero_z] = np.pi / 2
    tm_v[tm_v < 0] += np.pi

    v = np.rint(image_height * tm_v / np.pi).astype(np.int64) % image_height

    uv = np.column_stack((u, v)).astype(np.int32)
    valid = np.isfinite(local).all(axis=1)
    return uv, valid


def rasterize_points_to_overlay(
    uv: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    image_width: int,
    image_height: int,
    point_size: int,
    candidate_budget: int = 2_000_000,
) -> np.ndarray:
    """Rasterize projected points into an RGBA overlay with depth tests.

    The first point in input order wins ties at the same depth, matching the
    original script's strict depth comparison.
    """
    overlay = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    if uv.size == 0 or depths.size == 0 or colors.size == 0:
        return overlay

    uv = np.asarray(uv, dtype=np.int64)
    depths = np.asarray(depths, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if uv.shape[0] != depths.shape[0] or uv.shape[0] != colors.shape[0]:
        raise ValueError("uv, depths and colors must have the same length")

    radius = max(0, int(point_size))
    offsets: np.ndarray
    if radius <= 1:
        offsets = np.array([[0, 0]], dtype=np.int32)
    else:
        offset_list = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        offsets = np.asarray(offset_list, dtype=np.int32)
        if offsets.size == 0:
            offsets = np.array([[0, 0]], dtype=np.int32)

    footprint_size = int(offsets.shape[0])
    chunk_points = max(1, int(candidate_budget // max(1, footprint_size)))

    depth_flat = np.full(image_width * image_height, np.inf, dtype=np.float32)
    overlay_flat = overlay.reshape(-1, 4)
    offset_x = offsets[:, 0].astype(np.int64, copy=False)
    offset_y = offsets[:, 1].astype(np.int64, copy=False)

    for start in range(0, uv.shape[0], chunk_points):
        end = min(uv.shape[0], start + chunk_points)
        uv_chunk = uv[start:end]
        depth_chunk = depths[start:end]
        color_chunk = colors[start:end]
        px = uv_chunk[:, 0:1] + offset_x[None, :]
        py = uv_chunk[:, 1:2] + offset_y[None, :]
        px %= image_width
        py = np.clip(py, 0, image_height - 1)

        candidate_pixels = (py.astype(np.int64) * image_width + px.astype(np.int64)).reshape(-1)
        candidate_depths = np.repeat(depth_chunk, footprint_size)
        candidate_points = np.repeat(np.arange(start, end, dtype=np.int64), footprint_size)
        candidate_order = np.arange(candidate_pixels.size, dtype=np.int64)

        sort_idx = np.lexsort((candidate_order, candidate_depths, candidate_pixels))
        sorted_pixels = candidate_pixels[sort_idx]
        if sorted_pixels.size == 0:
            continue

        first_mask = np.empty(sorted_pixels.size, dtype=bool)
        first_mask[0] = True
        first_mask[1:] = sorted_pixels[1:] != sorted_pixels[:-1]

        selected_idx = sort_idx[first_mask]
        selected_pixels = sorted_pixels[first_mask]
        selected_depths = candidate_depths[selected_idx]
        selected_points = candidate_points[selected_idx]

        current_depths = depth_flat[selected_pixels]
        update_mask = selected_depths < current_depths
        if not np.any(update_mask):
            continue

        update_pixels = selected_pixels[update_mask]
        update_points = selected_points[update_mask]
        depth_flat[update_pixels] = selected_depths[update_mask]
        overlay_flat[update_pixels, :3] = color_chunk[update_points - start]
        overlay_flat[update_pixels, 3] = 255

    return overlay


def camera_forward_vector(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Approximate camera look direction in world coordinates for the 3D view."""
    rot = rotation_from_angle(roll, pitch, yaw)
    local_forward = np.array([0.0, 1.0, 0.0])
    world_forward = rot.T @ local_forward
    norm = np.linalg.norm(world_forward)
    if norm == 0:
        return local_forward
    return world_forward / norm
