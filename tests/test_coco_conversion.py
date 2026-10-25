import numpy as np

from scripts.coco_to_yolo_seg import (
    bbox_to_polygon,
    clean_polygon,
    polygon_list,
    polygon_to_yolo,
)


def test_polygon_list_shapes():
    assert polygon_list([[0, 0, 1, 0, 1, 1]]) == [[0, 0, 1, 0, 1, 1]]
    assert polygon_list([0, 0, 1, 0, 1, 1]) == [[0, 0, 1, 0, 1, 1]]
    assert polygon_list({"counts": [1, 2, 3]}) == []
    assert polygon_list([]) == []


def test_clean_polygon_clips_and_drops_degenerate():
    pts = clean_polygon([-10, -10, 20, -10, 20, 20, -10, 20], width=10, height=10)
    assert pts is not None
    assert pts[:, 0].min() == 0 and pts[:, 0].max() == 9
    assert pts[:, 1].min() == 0 and pts[:, 1].max() == 9

    assert clean_polygon([0, 0, 1, 0, 2, 0], width=10, height=10) is None
    assert clean_polygon([0, 0, 1, 0], width=10, height=10) is None


def test_clean_polygon_removes_duplicate_points():
    pts = clean_polygon([0, 0, 0, 0, 10, 0, 10, 10, 0, 10, 0, 0], width=10, height=10)
    assert pts is not None
    assert len(pts) == 4


def test_polygon_to_yolo_is_normalized():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    parts = polygon_to_yolo(pts, 10, 10).split()
    values = np.array([float(v) for v in parts])
    assert len(values) == 8
    assert values.min() == 0.0
    assert values.max() == 1.0


def test_bbox_to_polygon():
    assert bbox_to_polygon([1, 2, 3, 4]) == [1, 2, 4, 2, 4, 6, 1, 6]
