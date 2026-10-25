#!/usr/bin/env python3
"""Real-time sky-block inspection: 2D segmentation + 3D pose + red/blue state.

Thin CLI around :mod:`skyblock`.  See README.md for the full description.

Examples
--------
    python scripts/infer_realsense.py --weights weights/best.pt
    python scripts/infer_realsense.py --weights weights/best.pt --source 0
    python scripts/infer_realsense.py --weights weights/best.pt \
        --image test.jpg --depth test_depth.npy --intrinsics realsense_640x480.json
    python scripts/infer_realsense.py --weights weights/best.pt \
        --laser-left 0.42 --laser-right 0.51 --json-out predictions.jsonl
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from skyblock.laser import LaserProbe, laser_fallback  # noqa: F401  (re-exported for scripts)
from skyblock.pipeline import AnalysisConfig, analyze_frame
from skyblock.sources import ImageSource, RealSenseSource, VideoSource
from skyblock.types import Detection, Frame
from skyblock.visualize import draw_detection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", default="realsense",
                   help="'realsense', a webcam index ('0'), or a video path")
    p.add_argument("--image", type=Path, default=None, help="Offline RGB image")
    p.add_argument("--depth", type=Path, default=None, help="Offline depth: .npy / 16-bit png")
    p.add_argument("--intrinsics", type=Path, default=None, help="JSON fx,fy,cx,cy[,width,height]")
    p.add_argument("--depth-scale", type=float, default=None,
                   help="Metres per depth unit for offline depth (default: .npy->auto, png->0.001)")
    p.add_argument("--sample-id", default=None,
                   help="Sample id for --image, or prefix for streams (used by evaluate.py)")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)

    p.add_argument("--zmin", type=float, default=0.15, help="Metres, depth clipping")
    p.add_argument("--zmax", type=float, default=2.0)
    p.add_argument("--plane-dist", type=float, default=0.008, help="RANSAC inlier distance (m)")
    p.add_argument("--max-planes", type=int, default=3)
    p.add_argument("--min-plane-points", type=int, default=80)
    p.add_argument("--voxel", type=float, default=0.0025, help="Voxel downsample size (m)")
    p.add_argument("--top-angle", type=float, default=35.0,
                   help="Single plane normal within this angle of the optical axis = TOP face")
    p.add_argument("--min-dihedral", type=float, default=15.0,
                   help="Two planes whose normals differ less than this are treated as one face")

    p.add_argument("--highlight-v", type=int, default=245)
    p.add_argument("--min-sat", type=int, default=70)
    p.add_argument("--min-color-pixels", type=int, default=60)

    p.add_argument("--laser-threshold", type=float, default=0.03, help="Metres")
    p.add_argument("--laser-left", type=float, default=None, help="D_left for testing")
    p.add_argument("--laser-right", type=float, default=None, help="D_right for testing")

    p.add_argument("--extrinsic", type=Path, default=None,
                   help="4x4 camera->robot transform (.npy or .json), applied to the 3D centre")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--print-every", type=int, default=5, help="Print every N frames (0 = only on change)")
    p.add_argument("--json-out", type=Path, default=None, help="Write per-frame results as JSONL")
    return p.parse_args()


def load_extrinsic(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        transform = np.load(path)
    else:
        transform = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    if transform.shape != (4, 4):
        raise SystemExit(f"--extrinsic must be 4x4, got {transform.shape}")
    return transform


def make_source(args: argparse.Namespace):
    if args.image is not None:
        return ImageSource(args.image, args.depth, args.intrinsics, args.depth_scale,
                           repeat=not args.no_show)
    if args.source == "realsense":
        return RealSenseSource()
    return VideoSource(args.source)


def sample_id_for(args: argparse.Namespace, frame_idx: int) -> str:
    if args.image is not None:
        return args.sample_id or args.image.stem
    prefix = args.sample_id or "frame"
    return f"{prefix}_{frame_idx:06d}"


def format_detection(det: Detection) -> str:
    parts = [f"det{det.index}", f"conf={det.conf:.2f}", f"面={det.face_label}({det.face_confidence:.2f})",
             f"姿态={det.pose}"]
    if det.pose_detail:
        parts.append(f"({det.pose_detail})")
    if det.color.red_px + det.color.blue_px > 0:
        parts.append(f"颜色={det.color.state} R{det.color.red_ratio * 100:.0f}%/B{det.color.blue_ratio * 100:.0f}%")
    else:
        parts.append(f"颜色={det.color.state}")
    if det.center_xyz_m is not None:
        parts.append(f"中心=(X={det.center_xyz_m[0]:+.3f}, Y={det.center_xyz_m[1]:+.3f}, "
                     f"Z={det.center_xyz_m[2]:.3f})m")
    else:
        parts.append("中心=无(无深度)")
    if det.laser is not None:
        parts.append(f"激光={det.laser}")
    return " | ".join(parts)


def main() -> int:
    args = parse_args()

    if not args.weights.is_file():
        print(f"[ERROR] weights not found: {args.weights}\n"
              f"        Train first: python scripts/train_yolov8_seg.py", file=sys.stderr)
        return 2

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERROR] {exc}\n        pip install -r requirements.txt", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    source = make_source(args)
    laser = LaserProbe(args.laser_threshold)
    laser.update(args.laser_left, args.laser_right)

    cfg = AnalysisConfig(
        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        zmin=args.zmin, zmax=args.zmax,
        plane_dist=args.plane_dist, max_planes=args.max_planes,
        min_plane_points=args.min_plane_points, voxel=args.voxel,
        top_angle=args.top_angle, min_dihedral=args.min_dihedral,
        highlight_v=args.highlight_v, min_sat=args.min_sat,
        min_color_pixels=args.min_color_pixels,
        extrinsic=load_extrinsic(args.extrinsic),
    )

    json_f = None
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        json_f = args.json_out.open("w", encoding="utf-8")

    frame_idx = 0
    single_shot = isinstance(source, ImageSource)
    last_state = None
    fps_t0, fps_n = time.time(), 0
    try:
        while True:
            frame: Optional[Frame] = source.get()
            if frame is None:
                break

            t0 = time.perf_counter()
            laser_state = laser.classify()
            detections = analyze_frame(model, frame, cfg, laser_state=laser_state)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            canvas = frame.color.copy()
            for det in detections:
                draw_detection(canvas, det, frame.intr)

            sid = sample_id_for(args, frame_idx)
            frame_idx += 1
            fps_n += 1
            if fps_n >= 30:
                print(f"[INFO] ~{fps_n / max(time.time() - fps_t0, 1e-6):.1f} FPS")
                fps_t0, fps_n = time.time(), 0

            state_key = tuple((d.face_label, d.pose, d.color.state) for d in detections)
            if detections and (
                single_shot
                or args.print_every <= 0 and state_key != last_state
                or args.print_every > 0 and frame_idx % args.print_every == 0
            ):
                for det in detections:
                    print(format_detection(det))
                last_state = state_key
            if not detections and args.print_every > 0 and frame_idx % max(args.print_every, 1) == 0:
                print("[INFO] 未检测到天块")

            if json_f is not None:
                json_f.write(json.dumps({
                    "sample_id": sid,
                    "frame": frame_idx,
                    "timestamp": frame.timestamp,
                    "inference_ms": inference_ms,
                    "detections": [d.to_dict() for d in detections],
                }, ensure_ascii=False) + "\n")

            if not args.no_show:
                cv2.imshow("sky block", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if not cv2.getWindowProperty("sky block", cv2.WND_PROP_VISIBLE):
                    break
    finally:
        source.close()
        if json_f is not None:
            json_f.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
