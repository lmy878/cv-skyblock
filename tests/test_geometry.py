import json

import numpy as np

from skyblock.geometry import classify_pose, estimate_center, intersection_line
from skyblock.types import Intrinsics, Plane


def _plane(normal, centroid=(0, 0, 0.5), n=1000, ratio=0.7):
    return Plane(np.asarray(normal, dtype=float), np.asarray(centroid, dtype=float), n, ratio)


def test_single_top_plane():
    result = classify_pose([_plane([0, 0, -1])])
    assert result.label == "顶面"
    assert result.dihedral_deg is None


def test_single_side_plane():
    result = classify_pose([_plane([-1, 0, 0])])
    assert result.label == "单侧面"


def test_two_fold_planes():
    result = classify_pose([_plane([0, 0, -1]), _plane([-1, 0, 0])])
    assert result.label == "折角双面"
    assert result.dihedral_deg == 90.0
    assert result.intersection_direction is not None
    assert abs(abs(result.intersection_direction[1]) - 1.0) < 1e-6


def test_coplanar_planes_are_single():
    result = classify_pose([_plane([0, 0, -1]), _plane([0, 0, -1], ratio=0.2)])
    assert result.label == "顶面"
    assert result.dihedral_deg == 0.0


def test_no_planes():
    result = classify_pose([])
    assert result.label == "未知"


def test_intersection_line_is_perpendicular_to_both_normals():
    p1 = _plane([0, 0, -1], centroid=(0, 0, 0.5))
    p2 = _plane([-1, 0, 0], centroid=(0.1, 0, 0.5))
    point, direction = intersection_line(p1, p2)
    assert point is not None and direction is not None
    assert abs(np.dot(direction, p1.normal)) < 1e-6
    assert abs(np.dot(direction, p2.normal)) < 1e-6


def test_estimate_center_is_median():
    points = np.vstack([
        np.tile([0.0, 0.0, 0.5], (20, 1)),
        np.array([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]),
    ])
    center = estimate_center(points)
    assert center is not None
    assert np.allclose(center, [0.0, 0.0, 0.5])


def test_intrinsics_deproject_project_roundtrip(tmp_path):
    intr = Intrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0, width=5, height=5)
    mask = np.ones((5, 5), dtype=bool)
    depth = np.ones((5, 5), dtype=np.float32) * 0.5
    points = intr.deproject(mask, depth)
    assert points.shape == (25, 3)
    u, v = intr.project(points[12])
    assert 0 <= u < 5 and 0 <= v < 5

    path = tmp_path / "intr.json"
    path.write_text(json.dumps({"fx": 100, "fy": 100, "cx": 2, "cy": 2,
                                "width": 5, "height": 5}), encoding="utf-8")
    loaded = Intrinsics.from_json(path)
    assert loaded.fx == 100.0 and loaded.cy == 2.0


def test_center_requires_enough_points():
    assert estimate_center(np.zeros((3, 3))) is None
