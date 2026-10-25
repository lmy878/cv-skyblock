"""Final "which face is this?" decision.

The final label is deliberately **geometric**:

    TOP     顶面
    SIDE    侧面（单侧面 和 折角双面 都属于这一类）
    UNKNOWN 证据不足

Colour is not baked into the face label.  The JSON already carries the colour
state separately (``color.state``, ``red_ratio``, ``blue_ratio``), so different
red/blue percentages can be analysed later without creating a combinatorial
label set.  ``pose`` still distinguishes 单侧面 from 折角双面.

If you later need finer faces (e.g. ``SIDE_RED_LEFT_BLUE_RIGHT``), add them to
:class:`FaceLabel` and :func:`decide_face`; nothing else needs to change.

Decision rules
--------------
3D pose 顶面                              -> TOP
3D pose 单侧面 or 折角双面                -> SIDE
3D pose 未知 + 激光折角                   -> SIDE (low confidence: fold is still a side)
3D pose 未知 + 激光单平面                 -> UNKNOWN (cannot tell top from side)
otherwise                                 -> UNKNOWN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FaceLabel(str, Enum):
    TOP = "TOP"
    SIDE = "SIDE"
    UNKNOWN = "UNKNOWN"


LABELS: tuple[str, ...] = tuple(label.value for label in FaceLabel)


@dataclass
class DecisionResult:
    label: FaceLabel
    confidence: float
    reasons: list[str] = field(default_factory=list)
    color_state: str = "颜色未知"

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "confidence": float(self.confidence),
            "color_state": self.color_state,
            "reasons": list(self.reasons),
        }


def _result(label: FaceLabel, confidence: float, color_state: str,
            *reasons: str) -> DecisionResult:
    return DecisionResult(label, confidence, [r for r in reasons if r], color_state)


def decide_face(pose: str, color_state: str = "颜色未知",
                laser_state: Optional[str] = None) -> DecisionResult:
    """Map (3D pose, colour state, laser fallback) to TOP / SIDE / UNKNOWN."""
    pose = pose or "未知"
    color_state = color_state or "颜色未知"
    colour_reason = f"颜色={color_state}"

    if pose == "顶面":
        if laser_state == "折角":
            return _result(FaceLabel.TOP, 0.75, color_state,
                           "3D=顶面", colour_reason, "激光提示折角(与3D冲突)")
        return _result(FaceLabel.TOP, 0.95, color_state, "3D=顶面", colour_reason)

    if pose in ("单侧面", "折角双面"):
        # Both are side-category; the exact geometry stays in `pose`.
        reasons = ["3D=单侧面" if pose == "单侧面" else "3D=折角双面",
                   colour_reason]
        if pose == "折角双面" and laser_state == "单平面":
            return _result(FaceLabel.SIDE, 0.80, color_state,
                           *reasons, "激光提示单平面(与3D冲突)")
        if pose == "单侧面" and laser_state == "折角":
            reasons.append("激光提示折角(同属侧面)")
        return _result(FaceLabel.SIDE, 0.95, color_state, *reasons)

    # 3D pose unknown.
    if laser_state == "折角":
        # A fold is still a side face, even without reliable 3D planes.
        return _result(FaceLabel.SIDE, 0.5, color_state,
                       "3D=未知", "激光=折角(属于侧面)", colour_reason)
    if laser_state == "单平面":
        return _result(FaceLabel.UNKNOWN, 0.1, color_state,
                       "3D=未知", "激光=单平面(无法区分顶面/侧面)", colour_reason)

    return _result(FaceLabel.UNKNOWN, 0.0, color_state, "无有效证据")
