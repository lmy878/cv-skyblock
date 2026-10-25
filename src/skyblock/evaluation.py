"""Offline evaluation: predictions (JSONL) vs ground truth (JSON).

Metrics are grouped by the part of the pipeline they test:

* face      : final label accuracy / macro-F1 / confusion matrix
* pose      : 3D pose label accuracy (顶面 / 单侧面 / 折角双面)
* colour    : colour-state accuracy
* centre    : 3D centre MAE/RMSE per axis, Euclidean mean/median/p95, success rates
* normal    : plane-normal angular error (degrees)
* dihedral  : angle between two planes error (degrees)
* laser     : laser fallback accuracy + agreement with the 3D pose
* latency   : per-frame inference time

Ground-truth and prediction schemas are documented in ``docs/evaluation.md``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .decision import LABELS


@dataclass
class EvalConfig:
    center_tolerances_mm: tuple[float, ...] = (5.0, 10.0, 20.0)
    pick: str = "best"                       # "best" (highest conf) or "first"


def load_gt(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if not isinstance(samples, list):
        raise ValueError("GT must be a list or {'samples': [...]}")
    for i, sample in enumerate(samples):
        sample.setdefault("sample_id", f"sample_{i:06d}")
    return samples


def load_predictions(path: Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _record_detections(record: dict) -> list[dict]:
    if "detections" in record:
        return list(record["detections"] or [])
    if "face" in record or "bbox_xyxy" in record:
        return [record]
    return []


def select_detection(record: dict, pick: str = "best") -> Optional[dict]:
    detections = _record_detections(record)
    if not detections:
        return None
    if pick == "first":
        return detections[0]
    return max(detections, key=lambda d: d.get("conf") or 0.0)


def _face_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("label")
    if value is None:
        return "UNKNOWN"
    return str(value)


def _pred_face(det: Optional[dict]) -> str:
    if not det:
        return "UNKNOWN"
    return _face_value(det.get("face", det.get("face_label")))


def _gt_face(sample: dict) -> str:
    return _face_value(sample.get("face"))


def _pred_center(det: Optional[dict]) -> Optional[np.ndarray]:
    if not det:
        return None
    value = det.get("center_xyz_m")
    return None if value is None else np.asarray(value, dtype=np.float64)


def _gt_center(sample: dict) -> Optional[np.ndarray]:
    value = sample.get("center_xyz_m", sample.get("center"))
    return None if value is None else np.asarray(value, dtype=np.float64)


def _pred_normal(det: Optional[dict]) -> Optional[np.ndarray]:
    if not det:
        return None
    planes = det.get("planes") or []
    if not planes:
        return None
    value = planes[0].get("normal")
    return None if value is None else np.asarray(value, dtype=np.float64)


def _gt_normal(sample: dict) -> Optional[np.ndarray]:
    value = sample.get("normal")
    return None if value is None else np.asarray(value, dtype=np.float64)


def _angle_between(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = float(np.clip(abs(np.dot(a, b)) / (na * nb), 0.0, 1.0))
    return math.degrees(math.acos(cos))


def _stats(values: Iterable[float]) -> Optional[dict]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _match_records(pred_records: list[dict], gt_samples: list[dict]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for idx, record in enumerate(pred_records):
        sid = str(record.get("sample_id", record.get("frame", idx)))
        if sid not in by_id or (select_detection(record) or {}).get("conf", 0) > (
                select_detection(by_id[sid]) or {}).get("conf", 0):
            by_id[sid] = record

    gt_ids = [str(s["sample_id"]) for s in gt_samples]
    if any(gid in by_id for gid in gt_ids):
        return {gid: by_id[gid] for gid in gt_ids if gid in by_id}

    # Fallback: positional matching (useful for per-frame JSONL without sample ids).
    if len(pred_records) == len(gt_samples):
        return {gid: pred_records[i] for i, gid in enumerate(gt_ids)}
    return {}


def evaluate(pred_records: list[dict], gt_samples: list[dict],
             cfg: EvalConfig | None = None) -> tuple[dict, list[dict]]:
    cfg = cfg or EvalConfig()
    matched = _match_records(pred_records, gt_samples)

    rows: list[dict] = []
    gt_faces: list[str] = []
    pred_faces: list[str] = []
    pose_pairs: list[tuple[str, str]] = []
    color_pairs: list[tuple[str, str]] = []
    center_errors: list[np.ndarray] = []
    center_distances: list[float] = []
    normal_errors: list[float] = []
    dihedral_errors: list[float] = []
    laser_pairs: list[tuple[str, str]] = []
    laser_pose_agree = 0
    laser_pose_total = 0
    latencies: list[float] = []
    n_no_detection = 0
    n_missing = 0

    for sample in gt_samples:
        sid = str(sample["sample_id"])
        record = matched.get(sid)
        det = select_detection(record, cfg.pick) if record is not None else None

        if record is None:
            n_missing += 1
        if det is None:
            n_no_detection += 1

        gt_face = _gt_face(sample)
        pred_face = _pred_face(det)
        gt_faces.append(gt_face)
        pred_faces.append(pred_face)

        gt_pose = sample.get("pose")
        pred_pose = None if det is None else det.get("pose")
        if gt_pose is not None:
            pose_pairs.append((str(gt_pose), str(pred_pose) if pred_pose is not None else "未知"))

        gt_color = sample.get("color_state", sample.get("color"))
        pred_color = None if det is None else (det.get("color") or {}).get("state")
        if gt_color is not None:
            color_pairs.append((str(gt_color), str(pred_color) if pred_color is not None else "颜色未知"))

        gt_center = _gt_center(sample)
        pred_center = _pred_center(det)
        center_error_mm = None
        if gt_center is not None and pred_center is not None and gt_center.shape == pred_center.shape:
            err = (pred_center - gt_center) * 1000.0
            center_errors.append(err)
            dist = float(np.linalg.norm(err))
            center_distances.append(dist)
            center_error_mm = dist

        gt_normal = _gt_normal(sample)
        pred_normal = _pred_normal(det)
        normal_error = _angle_between(pred_normal, gt_normal)
        if normal_error is not None:
            normal_errors.append(normal_error)

        gt_dihedral = sample.get("dihedral_deg")
        pred_dihedral = None if det is None else det.get("dihedral_deg")
        dihedral_error = None
        if gt_dihedral is not None and pred_dihedral is not None:
            dihedral_error = abs(float(pred_dihedral) - float(gt_dihedral))
            dihedral_errors.append(dihedral_error)

        gt_laser = sample.get("laser")
        pred_laser = None if det is None else det.get("laser")
        if gt_laser is not None:
            laser_pairs.append((str(gt_laser), str(pred_laser) if pred_laser is not None else "未知"))
        if pred_laser is not None and pred_pose is not None:
            laser_pose_total += 1
            pose_is_fold = "折角" in str(pred_pose)
            if pose_is_fold == (pred_laser == "折角"):
                laser_pose_agree += 1

        if record is not None and record.get("inference_ms") is not None:
            latencies.append(float(record["inference_ms"]))

        rows.append({
            "sample_id": sid,
            "detected": det is not None,
            "gt_face": gt_face,
            "pred_face": pred_face,
            "face_correct": int(gt_face == pred_face),
            "gt_pose": gt_pose,
            "pred_pose": pred_pose,
            "gt_color": gt_color,
            "pred_color": pred_color,
            "gt_center_xyz_m": None if gt_center is None else [float(v) for v in gt_center],
            "pred_center_xyz_m": None if pred_center is None else [float(v) for v in pred_center],
            "center_error_mm": center_error_mm,
            "normal_error_deg": normal_error,
            "dihedral_error_deg": dihedral_error,
            "gt_laser": gt_laser,
            "pred_laser": pred_laser,
            "pred_face_confidence": None if det is None else (det.get("face") or {}).get("confidence"),
            "inference_ms": None if record is None else record.get("inference_ms"),
        })

    metrics: dict = {
        "dataset": {
            "n_gt": len(gt_samples),
            "n_pred_records": len(pred_records),
            "n_matched": len(matched),
            "n_no_detection": n_no_detection,
            "n_missing_prediction": n_missing,
            "detection_rate": _safe_div(len(gt_samples) - n_no_detection, len(gt_samples)),
        },
        "face": _classification_metrics(gt_faces, pred_faces),
        "pose": _pair_accuracy(pose_pairs),
        "color": _pair_accuracy(color_pairs),
    }

    metrics["center"] = _center_metrics(center_errors, center_distances, cfg, len(gt_samples))
    metrics["normal"] = {"error_deg": _stats(normal_errors)}
    metrics["dihedral"] = {"error_deg": _stats(dihedral_errors)}
    metrics["laser"] = {
        "accuracy": _pair_accuracy(laser_pairs),
        "agreement_with_3d_pose": _safe_div(laser_pose_agree, laser_pose_total),
        "n_agreement": laser_pose_total,
    }
    metrics["latency"] = {"inference_ms": _stats(latencies)}

    return metrics, rows


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _pair_accuracy(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0, "accuracy": None}
    correct = sum(1 for gt, pred in pairs if gt == pred)
    return {"n": len(pairs), "accuracy": correct / len(pairs)}


def _classification_metrics(gt: list[str], pred: list[str]) -> dict:
    if not gt:
        return {"n": 0, "accuracy": None, "macro_f1": None, "labels": [], "confusion_matrix": []}

    labels = list(LABELS) + sorted((set(gt) | set(pred)) - set(LABELS))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for g, p in zip(gt, pred):
        matrix[index[g], index[p]] += 1

    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)

    per_class = {}
    f1s = []
    for i, label in enumerate(labels):
        precision = _safe_div(tp[i], predicted[i])
        recall = _safe_div(tp[i], support[i])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": int(support[i]),
            "predicted": int(predicted[i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support[i] > 0:
            f1s.append(f1)

    accuracy = _safe_div(float(np.trace(matrix)), len(gt))
    return {
        "n": len(gt),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def _center_metrics(center_errors: list[np.ndarray], distances: list[float],
                    cfg: EvalConfig, n_gt: int) -> dict:
    out: dict = {"n": len(distances), "coverage": _safe_div(len(distances), n_gt)}
    if not center_errors:
        out["x"] = out["y"] = out["z"] = None
        out["distance_mm"] = None
        out["success_rate"] = {}
        return out

    errors = np.vstack(center_errors)
    for axis, name in enumerate("xyz"):
        abs_err = np.abs(errors[:, axis])
        out[name] = {
            "mae_mm": float(abs_err.mean()),
            "rmse_mm": float(np.sqrt(np.mean(errors[:, axis] ** 2))),
        }
    dist = np.asarray(distances, dtype=np.float64)
    out["distance_mm"] = {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(dist.max()),
    }
    out["success_rate"] = {
        f"within_{tol:g}mm": float((dist <= tol).mean()) for tol in cfg.center_tolerances_mm
    }
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_report(metrics: dict, rows: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    face = metrics["face"]
    if face.get("labels"):
        with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gt\\pred", *face["labels"]])
            for label, row in zip(face["labels"], face["confusion_matrix"]):
                writer.writerow([label, *row])

    if rows:
        fieldnames = list(rows[0].keys())
        with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()
                })

    (out_dir / "report.md").write_text(_markdown_report(metrics), encoding="utf-8")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_report(metrics: dict) -> str:
    face = metrics["face"]
    center = metrics["center"]
    lines = [
        "# Sky-block evaluation report",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| samples | {metrics['dataset']['n_gt']} |",
        f"| matched predictions | {metrics['dataset']['n_matched']} |",
        f"| detection rate | {_fmt(metrics['dataset']['detection_rate'])} |",
        f"| **face accuracy** | **{_fmt(face.get('accuracy'))}** |",
        f"| face macro-F1 | {_fmt(face.get('macro_f1'))} |",
        f"| pose accuracy | {_fmt(metrics['pose'].get('accuracy'))} |",
        f"| colour accuracy | {_fmt(metrics['color'].get('accuracy'))} |",
        f"| centre MAE per axis (mm) | "
        f"{_fmt((center.get('x') or {}).get('mae_mm'), 2)} / "
        f"{_fmt((center.get('y') or {}).get('mae_mm'), 2)} / "
        f"{_fmt((center.get('z') or {}).get('mae_mm'), 2)} |",
        f"| centre distance median / p95 (mm) | "
        f"{_fmt((center.get('distance_mm') or {}).get('median'), 2)} / "
        f"{_fmt((center.get('distance_mm') or {}).get('p95'), 2)} |",
        f"| normal error mean (deg) | {_fmt((metrics['normal'].get('error_deg') or {}).get('mean'), 2)} |",
        f"| laser accuracy | {_fmt(metrics['laser']['accuracy'].get('accuracy'))} |",
        f"| laser vs 3D agreement | {_fmt(metrics['laser'].get('agreement_with_3d_pose'))} |",
        f"| inference p95 (ms) | {_fmt((metrics['latency'].get('inference_ms') or {}).get('p95'), 2)} |",
        "",
    ]

    if face.get("labels"):
        labels = face["labels"]
        lines += ["## Face confusion matrix (rows = GT, columns = prediction)", ""]
        lines.append("| GT \\ Pred | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for label, row in zip(labels, face["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

        lines += ["## Per-class face metrics", "",
                  "| label | support | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for label, stats in face["per_class"].items():
            lines.append(
                f"| {label} | {stats['support']} | {_fmt(stats['precision'])} | "
                f"{_fmt(stats['recall'])} | {_fmt(stats['f1'])} |")
        lines.append("")

    if center.get("success_rate"):
        lines += ["## Centre success rate", "", "| tolerance | rate |", "|---|---|"]
        for key, value in center["success_rate"].items():
            lines.append(f"| {key.replace('within_', '≤ ')} | {_fmt(value)} |")
        lines.append("")

    return "\n".join(lines)
