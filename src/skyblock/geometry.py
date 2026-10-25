"""Point-cloud filtering, RANSAC plane segmentation and face-pose classification."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .types import Plane, PoseResult

CAMERA_AXIS = np.array([0.0, 0.0, 1.0])


def preprocess_points(points: np.ndarray, voxel: float = 0.0025,
                      remove_outliers: bool = True) -> np.ndarray:
    """Downsample (and optionally denoise) a point cloud with Open3D."""
    import open3d as o3d

    if len(points) > 60000:
        idx = np.random.default_rng(0).choice(len(points), 60000, replace=False)
        points = points[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    if remove_outliers and len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points)


def extract_planes(points: np.ndarray, distance_threshold: float = 0.008,
                   ransac_n: int = 3, num_iterations: int = 800,
                   max_planes: int = 3, min_plane_points: int = 80,
                   voxel: float = 0.0025) -> list[Plane]:
    """Iterative RANSAC plane segmentation on a masked point cloud."""
    import open3d as o3d

    planes: list[Plane] = []
    remaining = preprocess_points(points, voxel=voxel, remove_outliers=True)
    total = len(remaining)
    if total < min_plane_points:
        return planes

    for _ in range(max_planes):
        if len(remaining) < min_plane_points:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining)
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) < min_plane_points:
            break

        a, b, c, _ = model
        normal = np.array([a, b, c], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            break
        normal /= norm
        inlier_points = remaining[inliers]
        centroid = inlier_points.mean(axis=0)
        if np.dot(normal, centroid) > 0:      # point back towards the camera origin
            normal = -normal

        planes.append(Plane(normal, centroid, len(inliers), len(inliers) / max(total, 1)))
        remaining = np.delete(remaining, inliers, axis=0)
        if len(remaining) > 0 and len(inliers) / total < 0.06:
            break

    return planes


def plane_kind(normal: np.ndarray, top_angle_deg: float = 35.0) -> str:
    """Top face: normal close to the optical axis. Otherwise a side face."""
    cos_axis = abs(float(np.dot(normal, CAMERA_AXIS)))
    return "顶面" if cos_axis >= math.cos(math.radians(top_angle_deg)) else "单侧面"


def intersection_line(p1: Plane, p2: Plane) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Point and direction of the intersection line of two planes."""
    direction = np.cross(p1.normal, p2.normal)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None, None
    direction /= norm

    matrix = np.vstack((p1.normal, p2.normal, direction))
    rhs = np.array([
        np.dot(p1.normal, p1.centroid),
        np.dot(p2.normal, p2.centroid),
        0.0,
    ])
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        point = (p1.centroid + p2.centroid) / 2.0
    return point, direction


def classify_pose(planes: list[Plane], top_angle_deg: float = 35.0,
                  min_dihedral_deg: float = 15.0) -> PoseResult:
    """Classify 1 plane (top/side) or 2 planes (folded two faces)."""
    if not planes:
        return PoseResult("未知", "无有效平面")

    p1 = planes[0]
    kind1 = plane_kind(p1.normal, top_angle_deg)

    if len(planes) < 2:
        detail = (f"平面n=({p1.normal[0]:+.2f},{p1.normal[1]:+.2f},{p1.normal[2]:+.2f}) "
                  f"点数={p1.n_points}")
        return PoseResult(kind1, detail, planes=planes)

    p2 = planes[1]
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(p1.normal, p2.normal)), 0.0, 1.0))))
    if angle < min_dihedral_deg:
        detail = f"两面近似共面(法向夹角{angle:.1f}°)，按单面处理"
        return PoseResult(kind1, detail, planes=planes, dihedral_deg=angle)

    kind2 = plane_kind(p2.normal, top_angle_deg)
    point, direction = intersection_line(p1, p2)
    line_txt = ""
    if direction is not None:
        line_txt = f" 交线方向=({direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f})"
    detail = (f"{kind1}+{kind2} 法向夹角={angle:.1f}°{line_txt} "
              f"点数={p1.n_points}/{p2.n_points}")
    return PoseResult("折角双面", detail, planes=planes, dihedral_deg=angle,
                      intersection_point=point, intersection_direction=direction)


def estimate_center(points: np.ndarray) -> Optional[np.ndarray]:
    """Robust 3D centre: median of valid masked points."""
    if points is None or len(points) < 10:
        return None
    return np.median(points, axis=0)
