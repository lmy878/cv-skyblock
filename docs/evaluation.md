# Evaluation guide

The evaluator compares the JSONL written by `scripts/infer_realsense.py` with a
small ground-truth JSON file and writes a report + a per-sample CSV.

```bash
python scripts/evaluate.py \
    --pred predictions.jsonl \
    --gt gt_test.json \
    --out reports/test_2026_11_01 \
    --pick best \
    --center-tolerance-mm 5 10 20
```

Outputs:

| file | content |
|---|---|
| `metrics.json` | all metrics, machine-readable |
| `report.md` | summary tables + confusion matrix |
| `confusion_matrix.csv` | rows = GT face, columns = predicted face |
| `per_sample.csv` | every sample with GT/pred and centre error |

---

## 1. Final face labels

The final label is **geometric only** and lives in
`src/skyblock/decision.py::FaceLabel`:

```
TOP, SIDE, UNKNOWN
```

`SIDE` covers both `单侧面` and `折角双面` (a fold is still a side face).  Colour
is kept as a separate axis (`color.state`, `red_ratio`, `blue_ratio`), so
different red/blue percentages do not create extra face classes.

| 3D pose | laser | final face |
|---|---|---|
| 顶面 | – | `TOP` |
| 单侧面 | – | `SIDE` |
| 折角双面 | – | `SIDE` |
| 折角双面 | 单平面 | `SIDE` (lower confidence + conflict reason) |
| 未知 | 折角 | `SIDE` (low confidence, laser-only; fold belongs to side) |
| 未知 | 单平面 | `UNKNOWN` (cannot tell top from side) |
| anything else | | `UNKNOWN` |

The colour state is still written into `face.color_state` and the sibling
`color` object for later analysis.  `pose` keeps the exact geometry
(`顶面 / 单侧面 / 折角双面`).

If you later need finer faces (for example `SIDE_RED_LEFT_BLUE_RIGHT`), add them
to `FaceLabel` and the corresponding rule in `decide_face()`. The evaluator and
the reports read `LABELS` from that module, so no other change is needed.

---

## 2. Prediction schema (JSONL)

One JSON object per frame, one frame per line:

```json
{
  "sample_id": "sessionA_000123",
  "frame": 123,
  "timestamp": 1761500000.0,
  "inference_ms": 41.2,
  "detections": [
    {
      "index": 0,
      "conf": 0.93,
      "bbox_xyxy": [120.0, 80.0, 360.0, 330.0],
      "pose": "折角双面",
      "pose_detail": "顶面+单侧面 法向夹角=90.0° ...",
      "planes": [
        {"normal": [0.0, 0.0, -1.0], "centroid": [0.0, 0.0, 0.52],
         "n_points": 812, "inlier_ratio": 0.41}
      ],
      "dihedral_deg": 90.0,
      "intersection_direction": [0.0, 1.0, 0.0],
      "color": {
        "red_px": 4210, "blue_px": 3950,
        "red_ratio": 0.516, "blue_ratio": 0.484,
        "state": "红左蓝右", "layout": "horizontal"
      },
      "center_xyz_m": [0.012, -0.034, 0.512],
      "laser": "折角",
      "face": {"label": "SIDE", "confidence": 0.95, "color_state": "红左蓝右", "reasons": ["..."]}
    }
  ]
}
```

Notes:

- `sample_id` is the RGB image stem when `--image` is used, or
  `<prefix>_<frame:06d>` for streams (`frame_000123` by default). Use
  `--sample-id` to control it.
- If `detections` is empty, the sample counts as "not detected" and receives
  `UNKNOWN` in the face metrics (it is not silently skipped).
- If an image contains several detections, `--pick best` chooses the highest
  confidence; `--pick first` takes the first.
- `planes[0].normal` is used for the normal error. It points towards the camera.

---

## 3. Ground-truth schema

```json
{
  "samples": [
    {
      "sample_id": "sessionA_000123",
      "pose": "折角双面",
      "color_state": "红左蓝右",
      "face": "SIDE",
      "center_xyz_m": [0.010, -0.030, 0.515],
      "normal": [0.0, 0.0, -1.0],
      "dihedral_deg": 90.0,
      "laser": "折角"
    }
  ]
}
```

- Only `sample_id` and `face` are strictly required for the face metrics.
- `pose` and `color_state` enable per-module accuracy.
- `center_xyz_m` is optional but strongly recommended; without it, the centre
  section is skipped.
- `normal` / `dihedral_deg` must be in the same coordinate frame as the
  predictions (RealSense optical frame unless `--extrinsic` is used).
- `laser` is the reference laser state (`折角` / `单平面`).

Matching:

1. By `sample_id` if any GT id is found in the predictions.
2. Otherwise, if the number of predictions and GT samples is equal, by order.
3. Otherwise, unmatched GT samples count as missing predictions.

A ready-to-copy file is at `examples/gt_example.json`.

### How to get 3D centre ground truth

- Best: touch the block centre with the robot TCP and record the base-frame
  position, then transform into the camera frame with the calibrated extrinsic.
- Alternative: place the block on a calibrated grid / measure with a caliper and
  compute the centre from the known geometry.
- Without absolute GT, report **repeatability** instead: run the same static
  scene N times and compute the standard deviation of the predicted centre.
  Do not call the mean error "accuracy" in that case.

---

## 4. Metrics and formulas

| group | metric | definition |
|---|---|---|
| face | accuracy | correct final labels / GT samples |
| face | macro-F1 | mean F1 over classes that have GT support |
| face | confusion matrix | rows = GT, columns = prediction |
| pose | accuracy | `pred pose == gt pose` (string equality) |
| colour | accuracy | `pred color_state == gt color_state` |
| centre | MAE per axis | mean(abs(pred - gt)) in mm |
| centre | RMSE per axis | sqrt(mean((pred - gt)^2)) in mm |
| centre | distance | Euclidean error in mm: mean, median, p95, max |
| centre | success rate | fraction of samples with distance ≤ threshold |
| normal | angular error | `acos(abs(dot(n_pred, n_gt)))` in degrees; sign is ignored |
| dihedral | angular error | `abs(pred_dihedral - gt_dihedral)` in degrees |
| laser | accuracy | `pred laser == gt laser` |
| laser | agreement with 3D | predicted laser says fold ⇔ predicted pose is `折角双面` |
| latency | inference_ms | mean / median / p95 / max |
| dataset | detection rate | samples with at least one detection / GT samples |

Face accuracy is the headline number; keep the per-module metrics to explain
*why* it failed (segmentation, pose, colour or decision rule).

---

## 5. Suggested acceptance thresholds

These are starting points; tighten them once you have a real test set.

| metric | target |
|---|---|
| face accuracy | ≥ 0.95 |
| face macro-F1 | ≥ 0.90 |
| pose accuracy | ≥ 0.95 |
| colour accuracy | ≥ 0.95 |
| centre median error | ≤ 5 mm |
| centre p95 error | ≤ 10 mm |
| normal mean error | ≤ 10° |
| dihedral mean error | ≤ 10° |
| laser accuracy | ≥ 0.95 |
| inference p95 | ≤ 100 ms (depends on hardware) |

Also report results **per condition** (distance, camera tilt, lighting, face
type) — a single aggregate number hides the failure modes. Add those columns to
the GT (e.g. `condition`) and split `per_sample.csv` accordingly.

---

## 6. 2D segmentation mAP

`evaluate.py` does not replace Ultralytics' validation. Run both:

```bash
python scripts/train_yolov8_seg.py --resume      # writes val plots + mAP
# or directly:
yolo segment val model=weights/best.pt data=datasets/sky_block/data.yaml imgsz=640
```

Use mAP50-95 (mask) for the segmenter, and `evaluate.py` for the downstream
3D/colour/decision metrics.
