"""HSV red/blue state estimation inside a segmentation mask."""

from __future__ import annotations

import numpy as np

from .types import ColorResult


def analyze_color(bgr: np.ndarray, mask: np.ndarray, *,
                  highlight_v: int = 245, min_sat: int = 70,
                  min_value: int = 30, min_pixels: int = 60) -> ColorResult:
    """Red/blue ratio + spatial layout, ignoring specular highlights.

    Hue ranges are OpenCV's H in [0, 180]:
      red : [0, 10] and [170, 180]
      blue: [95, 135]

    A pixel is treated as a specular highlight when it is very bright
    (V >= highlight_v) *and* desaturated (S < min_sat), so saturated pure red /
    blue with V=255 is still counted.
    """
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Erode so anti-aliased / background pixels do not leak in.
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if m.sum() < min_pixels:
        m = mask

    # Specular highlights are bright *and* desaturated.  Do not drop bright
    # saturated red/blue (V can legitimately be 255).
    highlight = (v >= highlight_v) & (s < min_sat)
    valid = m & (v >= min_value) & ~highlight
    red = valid & (((h <= 10) | (h >= 170)) & (s >= min_sat))
    blue = valid & ((h >= 95) & (h <= 135) & (s >= min_sat))

    n_red = int(red.sum())
    n_blue = int(blue.sum())
    total = n_red + n_blue
    result = ColorResult(red_px=n_red, blue_px=n_blue)
    if total < min_pixels:
        return result

    r_ratio = n_red / total
    b_ratio = n_blue / total
    result.red_ratio = r_ratio
    result.blue_ratio = b_ratio

    def centroid(m: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(m)
        return float(xs.mean()), float(ys.mean())

    if r_ratio >= 0.85:
        result.state = "纯红"
    elif b_ratio >= 0.85:
        result.state = "纯蓝"
    elif r_ratio >= 0.15 and b_ratio >= 0.15:
        rx, ry = centroid(red)
        bx, by = centroid(blue)
        dx, dy = rx - bx, ry - by
        if abs(dx) >= abs(dy):
            result.state = "红右蓝左" if dx > 0 else "红左蓝右"
            result.layout = "horizontal"
        else:
            result.state = "红下蓝上" if dy > 0 else "红上蓝下"
            result.layout = "vertical"
    else:
        result.state = "红为主" if r_ratio > b_ratio else "蓝为主"

    return result
