import numpy as np

from skyblock.evaluation import EvalConfig, evaluate


def _det(face, pose, color, center, normal=(0, 0, -1), dihedral=None, laser=None,
         conf=0.9, inference_ms=20.0):
    return {
        "conf": conf,
        "pose": pose,
        "face": {"label": face, "confidence": 0.9, "reasons": []},
        "color": {"state": color, "red_ratio": 0.5, "blue_ratio": 0.5},
        "center_xyz_m": list(center),
        "planes": [{"normal": list(normal), "centroid": list(center), "n_points": 500,
                    "inlier_ratio": 0.5}],
        "dihedral_deg": dihedral,
        "laser": laser,
        "inference_ms": inference_ms,
    }


def _gt(sample_id, face, pose, color, center, normal=(0, 0, -1), dihedral=None, laser=None):
    return {
        "sample_id": sample_id,
        "face": face,
        "pose": pose,
        "color_state": color,
        "center_xyz_m": list(center),
        "normal": list(normal),
        "dihedral_deg": dihedral,
        "laser": laser,
    }


def _dataset():
    gt = [
        _gt("s1", "TOP", "顶面", "纯红", [0, 0, 0], normal=(0, 0, -1), laser="单平面"),
        _gt("s2", "SIDE", "折角双面", "红左蓝右", [0.1, 0, 0.6],
            normal=(0, -1, 0), dihedral=90, laser="折角"),
        _gt("s3", "TOP", "顶面", "纯蓝", [0, 0, 0.5], normal=(0, 0, -1)),
    ]
    preds = [
        {"sample_id": "s1", "inference_ms": 20.0,
         "detections": [_det("TOP", "顶面", "纯红", [0, 0, 0.005], laser="单平面")]},
        {"sample_id": "s2", "inference_ms": 30.0,
         "detections": [_det("SIDE", "折角双面", "红左蓝右", [0.1, 0.001, 0.6],
                             normal=(0, -1, 0), dihedral=88, laser="折角")]},
        {"sample_id": "s3", "inference_ms": 20.0, "detections": []},
    ]
    return gt, preds


def test_end_to_end_metrics():
    gt, preds = _dataset()
    metrics, rows = evaluate(preds, gt, EvalConfig(center_tolerances_mm=(5.0, 10.0)))

    assert metrics["dataset"]["n_gt"] == 3
    assert metrics["dataset"]["n_matched"] == 3
    assert metrics["dataset"]["n_no_detection"] == 1

    assert abs(metrics["face"]["accuracy"] - 2 / 3) < 1e-9
    assert metrics["pose"]["accuracy"] == 2 / 3
    assert metrics["color"]["accuracy"] == 2 / 3

    assert metrics["center"]["n"] == 2
    assert metrics["center"]["coverage"] == 2 / 3
    assert abs(metrics["center"]["distance_mm"]["median"] - 3.0) < 1e-6
    assert metrics["center"]["success_rate"]["within_5mm"] == 1.0

    assert metrics["normal"]["error_deg"]["mean"] == 0.0
    assert metrics["dihedral"]["error_deg"]["mean"] == 2.0
    assert metrics["laser"]["accuracy"]["accuracy"] == 1.0
    assert metrics["laser"]["agreement_with_3d_pose"] == 1.0

    assert len(rows) == 3
    s2 = next(row for row in rows if row["sample_id"] == "s2")
    assert abs(s2["center_error_mm"] - 1.0) < 1e-6


def test_confusion_matrix_and_per_class():
    gt, preds = _dataset()
    metrics, _ = evaluate(preds, gt)
    face = metrics["face"]
    labels = face["labels"]
    matrix = np.asarray(face["confusion_matrix"])
    assert matrix.shape == (len(labels), len(labels))
    assert matrix[labels.index("TOP"), labels.index("TOP")] == 1
    assert matrix[labels.index("SIDE"), labels.index("SIDE")] == 1
    assert matrix[labels.index("TOP"), labels.index("UNKNOWN")] == 1
    assert face["per_class"]["TOP"]["recall"] == 0.5
    assert face["per_class"]["SIDE"]["recall"] == 1.0


def test_picks_highest_confidence_detection():
    gt = [_gt("s1", "TOP", "顶面", "纯红", [0, 0, 0])]
    preds = [{"sample_id": "s1", "detections": [
        _det("UNKNOWN", "未知", "颜色未知", [0, 0, 0], conf=0.2),
        _det("TOP", "顶面", "纯红", [0, 0, 0], conf=0.8),
    ]}]
    metrics, _ = evaluate(preds, gt)
    assert metrics["face"]["accuracy"] == 1.0


def test_positional_matching_without_sample_ids():
    gt = [_gt("sample_000000", "TOP", "顶面", "纯红", [0, 0, 0])]
    preds = [{"detections": [_det("TOP", "顶面", "纯红", [0, 0, 0])]}]
    metrics, _ = evaluate(preds, gt)
    assert metrics["dataset"]["n_matched"] == 1
    assert metrics["face"]["accuracy"] == 1.0
