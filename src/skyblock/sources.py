"""Frame sources: RealSense, webcam/video, offline image + depth."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Frame, Intrinsics


class RealSenseSource:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise SystemExit(
                "pyrealsense2 is required for --source realsense.\n"
                "  pip install 'skyblock[realtime]'\n"
                "  (or use --source 0 for a normal webcam)"
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = Intrinsics.from_realsense(color_intr)
        try:
            name = self.profile.get_device().get_info(rs.camera_info.name)
        except Exception:
            name = "RealSense"
        print(f"[INFO] {name}  depth_scale={self.depth_scale:.6f} m/unit  "
              f"intr fx={self.intr.fx:.1f} fy={self.intr.fy:.1f}")

    def get(self) -> Optional[Frame]:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return Frame(color, depth_m, self.intr, time.time())

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class VideoSource:
    """Webcam / video file. No depth (3D disabled)."""

    def __init__(self, source):
        import cv2

        cam_id = int(source) if str(source).isdigit() else source
        if isinstance(cam_id, int):
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(str(cam_id))
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open video source: {source}")
        print(f"[INFO] video source: {source} (no depth -> 3D disabled)")

    def get(self) -> Optional[Frame]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return Frame(frame, None, None, time.time())

    def close(self) -> None:
        self.cap.release()


class ImageSource:
    def __init__(self, image_path: Path, depth_path: Optional[Path] = None,
                 intr_path: Optional[Path] = None, depth_scale: Optional[float] = None,
                 repeat: bool = False):
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Cannot read image: {image_path}")

        depth = None
        if depth_path is not None:
            depth = _load_depth(Path(depth_path), depth_scale, wanted_shape=img.shape[:2])
            if depth is None:
                raise SystemExit(f"Cannot read depth: {depth_path}")

        intr = Intrinsics.from_json(intr_path) if intr_path is not None else None
        if depth is not None and intr is None:
            print("[WARN] depth given without --intrinsics -> 3D disabled")
            depth = None

        self.frame = Frame(img, depth, intr, time.time())
        self.done = False
        self.repeat = repeat
        print(f"[INFO] image: {image_path}  depth: {'yes' if depth is not None else 'no'}  "
              f"3D: {'yes' if intr is not None else 'no'}")

    def get(self) -> Optional[Frame]:
        if self.done and not self.repeat:
            return None
        self.done = True
        return self.frame

    def close(self) -> None:
        pass


def _load_depth(path: Path, depth_scale: Optional[float],
                wanted_shape: tuple[int, int]) -> Optional[np.ndarray]:
    import cv2

    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
        if depth_scale is not None:
            depth *= depth_scale
        elif np.nanmax(depth) > 50:       # looks like millimetres
            depth *= 0.001
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        depth = raw.astype(np.float32) * (0.001 if depth_scale is None else depth_scale)

    if depth.shape[:2] != tuple(wanted_shape):
        depth = cv2.resize(depth, (wanted_shape[1], wanted_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return depth
