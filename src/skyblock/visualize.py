"""OpenCV drawing helpers (ASCII labels only: cv2 cannot render Chinese)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import Detection, Intrinsics

POSE_EN = {
    "顶面": "TOP",
    "单侧面": "SIDE",
    "折角双面": "FOLD",
    "未知": "UNKNOWN",
}
COLOR_EN = {
    "纯红": "PURE_R",
    "纯蓝": "PURE_B",
    "红左蓝右": "R|B",
    "红右蓝左": "B|R",
    "红上蓝下": "R/B",
    "红下蓝上": "B/R",
    "红为主": "MOSTLY_R",
    "蓝为主": "MOSTLY_B",
    "颜色未知": "UNKNOWN",
}


def draw_detection(canvas: np.ndarray, det: Detection,
                   intr: Optional[Intrinsics] = None) -> None:
    import cv2

    mask = det.mask
    overlay = canvas.copy()
    overlay[mask] = (0, 200, 0)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if contours:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(canvas, [box], 0, (0, 255, 255), 2)

    if det.center_xyz_m is not None:
        if intr is not None:
            u, v = intr.project(det.center_xyz_m)
            if u >= 0:
                cv2.circle(canvas, (u, v), 5, (0, 0, 255), -1)
        z = det.center_xyz_m[2]
        txt = (f"#{det.index} {det.conf:.2f} {det.face_label} "
               f"{POSE_EN.get(det.pose, det.pose)} {COLOR_EN.get(det.color.state, det.color.state)} "
               f"Z={z:.3f}m")
    else:
        txt = (f"#{det.index} {det.conf:.2f} {det.face_label} "
               f"{POSE_EN.get(det.pose, det.pose)} {COLOR_EN.get(det.color.state, det.color.state)}")

    ty = max(y1 - 8, 16)
    cv2.putText(canvas, txt, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
