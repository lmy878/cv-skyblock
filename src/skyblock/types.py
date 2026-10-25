"""Shared dataclasses. Deliberately dependency-light (numpy only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @classmethod
    def from_realsense(cls, intr) -> "Intrinsics":
        return cls(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height)

    @classmethod
    def from_json(cls, path: Path) -> "Intrinsics":
        with Path(path).open(encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d.get("cx", d.get("ppx"))),
            cy=float(d.get("cy", d.get("ppy"))),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    def deproject(self, mask: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Masked depth (metres) -> Nx3 points in the camera optical frame."""
        vs, us = np.nonzero(mask)
        z = depth_m[vs, us].astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        us, vs, z = us[valid], vs[valid], z[valid]
        x = (us - self.cx) * z / self.fx
        y = (vs - self.cy) * z / self.fy
        return np.column_stack((x, y, z))

    def project(self, point_xyz: Iterable[float]) -> tuple[int, int]:
        x, y, z = point_xyz
        if z is None or z <= 1e-6:
            return -1, -1
        return int(round(x * self.fx / z + self.cx)), int(round(y * self.fy / z + self.cy))


@dataclass
class Frame:
    color: np.ndarray                    # BGR uint8 (H, W, 3)
    depth: Optional[np.ndarray]          # metres float32 (H, W) or None
    intr: Optional[Intrinsics]
    timestamp: float = 0.0


@dataclass
class Plane:
    normal: np.ndarray                   # unit normal, pointing towards the camera
    centroid: np.ndarray                 # plane centroid (camera frame, metres)
    n_points: int
    inlier_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "normal": [float(v) for v in self.normal],
            "centroid": [float(v) for v in self.centroid],
            "n_points": int(self.n_points),
            "inlier_ratio": float(self.inlier_ratio),
        }


@dataclass
class PoseResult:
    label: str
    detail: str = ""
    planes: list[Plane] = field(default_factory=list)
    dihedral_deg: Optional[float] = None
    intersection_point: Optional[np.ndarray] = None
    intersection_direction: Optional[np.ndarray] = None


@dataclass
class ColorResult:
    red_px: int = 0
    blue_px: int = 0
    red_ratio: float = 0.0
    blue_ratio: float = 0.0
    state: str = "颜色未知"
    layout: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "red_px": int(self.red_px),
            "blue_px": int(self.blue_px),
            "red_ratio": float(self.red_ratio),
            "blue_ratio": float(self.blue_ratio),
            "state": self.state,
            "layout": self.layout,
        }


@dataclass
class Detection:
    index: int
    conf: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray                     # HxW bool; not serialised
    pose: str
    pose_detail: str
    planes: list[Plane]
    dihedral_deg: Optional[float]
    intersection_direction: Optional[np.ndarray]
    color: ColorResult
    center_xyz_m: Optional[np.ndarray]
    laser: Optional[str] = None
    face_label: str = "UNKNOWN"
    face_confidence: float = 0.0
    face_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "conf": float(self.conf),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "pose": self.pose,
            "pose_detail": self.pose_detail,
            "planes": [p.to_dict() for p in self.planes],
            "dihedral_deg": None if self.dihedral_deg is None else float(self.dihedral_deg),
            "intersection_direction": (
                None if self.intersection_direction is None
                else [float(v) for v in self.intersection_direction]
            ),
            "color": self.color.to_dict(),
            "center_xyz_m": (
                None if self.center_xyz_m is None
                else [float(v) for v in self.center_xyz_m]
            ),
            "laser": self.laser,
            "face": {
                "label": self.face_label,
                "confidence": float(self.face_confidence),
                "color_state": self.color.state,
                "reasons": list(self.face_reasons),
            },
        }
