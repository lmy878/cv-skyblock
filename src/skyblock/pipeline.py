"""One-frame analysis: segmentation -> 3D pose -> colour -> final face decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .color import analyze_color
from .decision import decide_face
from .geometry import classify_pose, estimate_center, extract_planes
from .types import Detection, Frame, PoseResult


@dataclass
class AnalysisConfig:
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5

    zmin: float = 0.15
    zmax: float = 2.0
    plane_dist: float = 0.008
    max_planes: int = 3
    min_plane_points: int = 80
    voxel: float = 0.0025
    top_angle: float = 35.0
    min_dihedral: float = 15.0

    highlight_v: int = 245
    min_sat: int = 70
    min_color_pixels: int = 60

    extrinsic: Optional[np.ndarray] = field(default=None, repr=False)


def analyze_frame(model, frame: Frame, cfg: AnalysisConfig,
                  laser_state: Optional[str] = None) -> list[Detection]:
    """Run YOLO-seg + geometry + colour on one RGB-D frame."""
    import cv2

    results = model.predict(
        frame.color, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
        task="segment", retina_masks=True, verbose=False,
    )
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None

    detections: list[Detection] = []
    for i in range(len(boxes)):
        if masks is not None and i < len(masks):
            mask = masks[i] > 0.5
            if mask.shape != frame.color.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.color.shape[1], frame.color.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        else:  # detection model fallback: bbox rectangle as mask
            mask = np.zeros(frame.color.shape[:2], dtype=bool)
            x1, y1, x2, y2 = boxes[i].astype(int)
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True

        pose_result, center = _estimate_pose(frame, mask, cfg)

        color = analyze_color(
            frame.color, mask,
            highlight_v=cfg.highlight_v,
            min_sat=cfg.min_sat,
            min_pixels=cfg.min_color_pixels,
        )
        decision = decide_face(pose_result.label, color.state, laser_state)
        conf = float(confs[i]) if i < len(confs) else 0.0

        detections.append(Detection(
            index=i,
            conf=conf,
            bbox_xyxy=boxes[i],
            mask=mask,
            pose=pose_result.label,
            pose_detail=pose_result.detail,
            planes=pose_result.planes,
            dihedral_deg=pose_result.dihedral_deg,
            intersection_direction=pose_result.intersection_direction,
            color=color,
            center_xyz_m=center,
            laser=laser_state,
            face_label=decision.label.value,
            face_confidence=decision.confidence,
            face_reasons=decision.reasons,
        ))

    return detections


def _estimate_pose(frame: Frame, mask: np.ndarray,
                   cfg: AnalysisConfig) -> tuple[PoseResult, Optional[np.ndarray]]:
    if frame.depth is None or frame.intr is None or not mask.any():
        return PoseResult("未知", "无深度或空掩码"), None

    points = frame.intr.deproject(mask, frame.depth)
    if len(points):
        keep = (points[:, 2] >= cfg.zmin) & (points[:, 2] <= cfg.zmax)
        points = points[keep]
    if len(points) < cfg.min_plane_points:
        return PoseResult("未知", f"点云不足({len(points)})"), None

    planes = extract_planes(
        points,
        distance_threshold=cfg.plane_dist,
        max_planes=cfg.max_planes,
        min_plane_points=cfg.min_plane_points,
        voxel=cfg.voxel,
    )
    pose = classify_pose(planes, cfg.top_angle, cfg.min_dihedral)
    center = estimate_center(points)
    if center is not None and cfg.extrinsic is not None:
        center = (cfg.extrinsic[:3, :3] @ center) + cfg.extrinsic[:3, 3]
    return pose, center
