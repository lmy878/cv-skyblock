#!/usr/bin/env python3
"""Evaluate predictions JSONL against a ground-truth JSON file.

    python scripts/evaluate.py --pred predictions.jsonl --gt gt.json --out reports/run1
    python scripts/evaluate.py --pred predictions.jsonl --gt gt.json --pick first

Writes: metrics.json, report.md, confusion_matrix.csv, per_sample.csv.
See docs/evaluation.md for the schemas and metric definitions.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

from skyblock.evaluation import EvalConfig, evaluate, load_gt, load_predictions, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True, help="predictions .jsonl from infer_realsense.py")
    p.add_argument("--gt", type=Path, required=True, help="ground-truth .json")
    p.add_argument("--out", type=Path, default=Path("reports/latest"), help="Report folder")
    p.add_argument("--pick", choices=["best", "first"], default="best",
                   help="Which detection to score when an image has several")
    p.add_argument("--center-tolerance-mm", type=float, nargs="+", default=[5.0, 10.0, 20.0],
                   help="Success thresholds for the 3D centre distance")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pred.is_file():
        print(f"[ERROR] predictions not found: {args.pred}", file=sys.stderr)
        return 2
    if not args.gt.is_file():
        print(f"[ERROR] ground truth not found: {args.gt}", file=sys.stderr)
        return 2

    pred_records = load_predictions(args.pred)
    gt_samples = load_gt(args.gt)
    cfg = EvalConfig(center_tolerances_mm=tuple(args.center_tolerance_mm), pick=args.pick)
    metrics, rows = evaluate(pred_records, gt_samples, cfg)
    write_report(metrics, rows, args.out)

    print(f"[OK] samples={metrics['dataset']['n_gt']} matched={metrics['dataset']['n_matched']}")
    print(f"     face accuracy   = {metrics['face'].get('accuracy')}")
    print(f"     face macro-F1   = {metrics['face'].get('macro_f1')}")
    print(f"     pose accuracy   = {metrics['pose'].get('accuracy')}")
    print(f"     colour accuracy = {metrics['color'].get('accuracy')}")
    print(f"     centre median   = {(metrics['center'].get('distance_mm') or {}).get('median')} mm")
    print(f"[OK] report: {(args.out / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
