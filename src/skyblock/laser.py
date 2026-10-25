"""Single-point laser fallback interface."""

from __future__ import annotations

from typing import Optional


def laser_fallback(d_left: Optional[float], d_right: Optional[float],
                   diff_threshold: float = 0.03) -> Optional[str]:
    """Classify from two single-point laser distances (metres).

    * |D_left - D_right| > threshold -> 折角 (two faces)
    * otherwise                      -> 单平面
    * missing reading                -> None
    """
    if d_left is None or d_right is None:
        return None
    return "折角" if abs(float(d_left) - float(d_right)) > diff_threshold else "单平面"


class LaserProbe:
    """Holds the latest laser readings; override read() to plug in real hardware."""

    def __init__(self, diff_threshold: float = 0.03):
        self.diff_threshold = diff_threshold
        self.d_left: Optional[float] = None
        self.d_right: Optional[float] = None

    def update(self, d_left: Optional[float], d_right: Optional[float]) -> None:
        self.d_left, self.d_right = d_left, d_right

    def read(self) -> tuple[Optional[float], Optional[float]]:
        """TODO: replace with the real serial/socket/ROS read for your laser."""
        return self.d_left, self.d_right

    def classify(self) -> Optional[str]:
        d_left, d_right = self.read()
        return laser_fallback(d_left, d_right, self.diff_threshold)
