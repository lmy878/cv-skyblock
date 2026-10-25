import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from skyblock.color import analyze_color  # noqa: E402


def _split_image(left_bgr, right_bgr, size=40):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, : size // 2] = left_bgr
    img[:, size // 2:] = right_bgr
    return img


def test_red_left_blue_right():
    img = _split_image((0, 0, 255), (255, 0, 0))
    mask = np.ones(img.shape[:2], dtype=bool)
    result = analyze_color(img, mask)
    assert result.state == "红左蓝右"
    assert result.layout == "horizontal"
    assert 0.4 < result.red_ratio < 0.6


def test_blue_left_red_right():
    img = _split_image((255, 0, 0), (0, 0, 255))
    mask = np.ones(img.shape[:2], dtype=bool)
    assert analyze_color(img, mask).state == "红右蓝左"


def test_pure_red():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    result = analyze_color(img, np.ones((40, 40), dtype=bool))
    assert result.state == "纯红"


def test_highlights_are_ignored():
    img = _split_image((0, 0, 255), (255, 0, 0))
    img[10:20, 10:20] = (255, 255, 255)          # specular blob
    mask = np.ones(img.shape[:2], dtype=bool)
    result = analyze_color(img, mask)
    assert result.state in {"红左蓝右", "红右蓝左"}
    assert result.red_px + result.blue_px > 0


def test_empty_mask_is_unknown():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    result = analyze_color(img, np.zeros((20, 20), dtype=bool))
    assert result.state == "颜色未知"
