#!/usr/bin/env python3
"""Print and save the RealSense colour intrinsics as JSON.

The saved file can be passed to infer_realsense.py for offline RGB-D runs::

    python scripts/dump_intrinsics.py --out realsense_640x480.json
    python scripts/infer_realsense.py --image rgb.png --depth depth.npy \
        --intrinsics realsense_640x480.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("realsense_640x480.json"))
    args = p.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed: pip install pyrealsense2")
        return 2

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(cfg)
    try:
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        data = {
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
        print(json.dumps(data, indent=2))
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {args.out.resolve()}")
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
