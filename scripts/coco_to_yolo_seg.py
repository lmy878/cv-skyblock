#!/usr/bin/env python3
"""Convert the provided COCO instance-segmentation dataset to YOLOv8-seg format.

Input layout (the dataset you downloaded)::

    recommended/
      train/annotations.json + train/images/*.jpg
      val/annotations.json   + val/images/*.jpg

Output layout::

    datasets/sky_block/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt      # class x1 y1 x2 y2 ... (normalized polygon)
      labels/val/*.txt
      data.yaml
      dataset_stats.json
      preview/                # optional, only with --preview

Every line of a label file is one instance::

    <class_index> <x1> <y1> <x2> <y2> ...   # 0..1 normalized

Images without any annotation (the 5 explicit negative samples) get an
*empty* .txt file, which is exactly what Ultralytics expects for negatives.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_COCO_ROOT = Path(
    "/home/liumengying/Downloads/sky_block_labeled_datasets_20261025 (2)/recommended"
)
DEFAULT_OUT = Path("/home/liumengying/sky_block_pipeline/datasets/sky_block")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT,
                   help="Folder containing train/ and val/ (each with annotations.json + images/)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output YOLO dataset folder")
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images instead of symlinking (default: symlink to save disk space)")
    p.add_argument("--preview", action="store_true", help="Render label overlays to out/preview/")
    p.add_argument("--overwrite", action="store_true", help="Delete the output folder first")
    return p.parse_args()


def polygon_list(segmentation) -> list[list[float]]:
    """Return a list of flat [x1,y1,x2,y2,...] polygons.

    Handles the three shapes COCO uses in practice:
      * [[x1,y1,...]]            -> already a list of polygons
      * [x1,y1,x2,y2,...]        -> single polygon
      * {"counts": ...} / None   -> not supported here (caller falls back to bbox)
    """
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if isinstance(segmentation[0], list):
        return [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
    if isinstance(segmentation[0], (int, float)):
        return [segmentation] if len(segmentation) >= 6 else []
    return []


def clean_polygon(flat: list[float], width: int, height: int) -> np.ndarray | None:
    """Clip to the image, drop duplicate/adjacent points, reject degenerate polys."""
    pts = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None

    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-6, axis=1)
    pts = pts[keep]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 1.0:  # less than one pixel
        return None
    return pts


def bbox_to_polygon(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def polygon_to_yolo(pts: np.ndarray, width: int, height: int) -> str:
    norm = pts.copy()
    norm[:, 0] /= max(width, 1)
    norm[:, 1] /= max(height, 1)
    norm = np.clip(norm, 0.0, 1.0)
    return " ".join(f"{v:.6f}" for v in norm.reshape(-1))


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert_split(split_dir: Path, out_dir: Path, split: str, copy_images: bool,
                  preview: bool) -> dict:
    ann_path = split_dir / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", [{"id": 1, "name": "sky_block"}]), key=lambda c: c["id"])
    cat_id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    category_names = [c["name"] for c in categories]

    images = {im["id"]: im for im in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview" / split
    if preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    n_instances = 0
    n_negatives = 0
    n_fallback_bbox = 0
    n_dropped = 0
    per_class = Counter()

    for image_id, info in images.items():
        file_name = Path(info["file_name"]).name
        src = split_dir / "images" / file_name
        if not src.is_file():
            raise FileNotFoundError(f"Image listed in JSON is missing: {src}")

        width, height = int(info["width"]), int(info["height"])
        dst = img_out / file_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_images:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

        label_lines: list[str] = []
        draw_polys: list[np.ndarray] = []

        for ann in anns_by_image.get(image_id, []):
            cls_idx = cat_id_to_index.get(ann["category_id"])
            if cls_idx is None:
                n_dropped += 1
                continue

            polys = polygon_list(ann.get("segmentation"))
            used_fallback = False
            if not polys and ann.get("bbox"):
                polys = [bbox_to_polygon(ann["bbox"])]
                used_fallback = True

            image_had_instance = False
            for flat in polys:
                pts = clean_polygon(flat, width, height)
                if pts is None:
                    n_dropped += 1
                    continue
                label_lines.append(f"{cls_idx} {polygon_to_yolo(pts, width, height)}")
                draw_polys.append(pts)
                per_class[categories[cls_idx]["name"]] += 1
                n_instances += 1
                image_had_instance = True
                if used_fallback:
                    n_fallback_bbox += 1

            if not image_had_instance and used_fallback:
                n_dropped += 1

        if not label_lines:
            n_negatives += 1

        stem = Path(file_name).stem
        (lbl_out / f"{stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""),
                                             encoding="utf-8")

        if preview:
            from PIL import Image, ImageDraw

            im = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(im)
            for pts in draw_polys:
                draw.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=(255, 0, 0), width=3)
            im.save(preview_dir / f"{stem}.jpg", quality=85)

    return {
        "split": split,
        "images": len(images),
        "instances": n_instances,
        "negative_images": n_negatives,
        "bbox_fallbacks": n_fallback_bbox,
        "dropped_annotations": n_dropped,
        "per_class": dict(per_class),
        "categories": category_names,
    }


def write_data_yaml(out_dir: Path, categories: list[str]) -> None:
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(categories):
        lines.append(f"  {i}: {name}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    coco_root = args.coco_root.expanduser()
    out_dir = args.out.expanduser()

    if not coco_root.is_dir():
        print(f"[ERROR] COCO root not found: {coco_root}", file=sys.stderr)
        return 2

    if out_dir.exists():
        if not args.overwrite:
            print(f"[ERROR] Output already exists: {out_dir}\n        use --overwrite to rebuild", file=sys.stderr)
            return 2
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    categories: list[str] = []
    for split in args.splits:
        split_dir = coco_root / split
        if not split_dir.is_dir():
            print(f"[WARN] skipping missing split: {split_dir}")
            continue
        print(f"[INFO] converting {split_dir} -> {out_dir}/{{images,labels}}/{split}")
        info = convert_split(split_dir, out_dir, split, args.copy_images, args.preview)
        if not categories:
            categories = info["categories"]
        stats.append(info)
        print("       " + json.dumps({k: v for k, v in info.items() if k != "categories"}, ensure_ascii=False))

    if not categories:
        print("[ERROR] Nothing converted", file=sys.stderr)
        return 1

    write_data_yaml(out_dir, categories)
    (out_dir / "dataset_stats.json").write_text(
        json.dumps({"source": str(coco_root), "splits": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print(f"  data.yaml : {out_dir / 'data.yaml'}")
    for s in stats:
        print(f"  {s['split']:<5} images={s['images']} instances={s['instances']} "
              f"negatives={s['negative_images']} bbox_fallback={s['bbox_fallbacks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
