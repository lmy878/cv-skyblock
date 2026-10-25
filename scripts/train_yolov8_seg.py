#!/usr/bin/env python3
"""Train / validate a YOLOv8-seg model on the converted sky-block dataset.

Example
-------
    python scripts/train_yolov8_seg.py
    python scripts/train_yolov8_seg.py --model yolov8s-seg.pt --epochs 300 --imgsz 640
    python scripts/train_yolov8_seg.py --device cpu --batch 2      # no GPU

The best checkpoint is copied to:
    runs/segment/<name>/weights/best.pt
    weights/best.pt          (project root, easy to pass to inference)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sky_block" / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8n-seg.pt",
                   help="Pretrained weights: yolov8n/s/m/l/x-seg.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="Use -1 for auto batch (GPU only)")
    p.add_argument("--device", default=None, help="'0', '0,1', 'cpu' ... (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "segment")
    p.add_argument("--name", default="sky_block_yolov8n_seg")
    p.add_argument("--resume", action="store_true", help="Resume the last run in --project/--name")
    p.add_argument("--no-val", action="store_true", help="Skip the final validation pass")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        print(f"[ERROR] data.yaml not found: {args.data}\n"
              f"        Run scripts/coco_to_yolo_seg.py first.", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed.\n"
              "        pip install -r requirements.txt", file=sys.stderr)
        return 2

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        task="segment",
        exist_ok=True,
        plots=True,
        # Light, safe augmentation for a small real dataset:
        mosaic=0.5,
        close_mosaic=20,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.0,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[ERROR] best.pt not found under {save_dir}/weights", file=sys.stderr)
        return 1

    export = PROJECT_ROOT / "weights"
    export.mkdir(exist_ok=True)
    shutil.copy2(best, export / "best.pt")
    print(f"\n[OK] best checkpoint : {best}")
    print(f"[OK] copied to       : {export / 'best.pt'}")

    if not args.no_val:
        print("\n[INFO] running final validation on the val split ...")
        metrics = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz,
                                      split="val", plots=True,
                                      project=str(args.project), name=f"{args.name}_val")
        print(f"[OK] val results     : {metrics.save_dir}")
        try:
            box, seg = metrics.box, metrics.seg
            print(f"     box  mAP50={box.map50:.4f} mAP50-95={box.map:.4f}")
            print(f"     mask mAP50={seg.map50:.4f} mAP50-95={seg.map:.4f}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
