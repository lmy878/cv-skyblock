from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from skyblock import pipeline  # noqa: E402
from skyblock.pipeline import AnalysisConfig, analyze_frame  # noqa: E402
from skyblock.types import Frame, Intrinsics, Plane  # noqa: E402


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _FakeModel:
    def __init__(self, mask):
        self.mask = mask

    def predict(self, *args, **kwargs):
        boxes = SimpleNamespace(
            xyxy=_Tensor([[5.0, 5.0, 35.0, 35.0]]),
            conf=_Tensor([0.9]),
        )
        masks = SimpleNamespace(data=_Tensor(self.mask[None, ...]))
        return [SimpleNamespace(boxes=boxes, masks=masks)]


def _frame(depth=True):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)                                  # pure red (BGR)
    intr = Intrinsics(fx=50.0, fy=50.0, cx=20.0, cy=20.0, width=40, height=40)
    depth_m = np.full((40, 40), 0.5, dtype=np.float32) if depth else None
    return Frame(img, depth_m, intr if depth else None, timestamp=0.0)


def _fake_planes(*args, **kwargs):
    return [Plane(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.5]), 500, 0.5)]


def test_analyze_frame_full_path(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_planes", _fake_planes)
    mask = np.ones((40, 40), dtype=bool)
    detections = analyze_frame(_FakeModel(mask), _frame(), AnalysisConfig(min_plane_points=10),
                               laser_state="单平面")

    assert len(detections) == 1
    det = detections[0]
    assert det.pose == "顶面"
    assert det.color.state == "纯红"
    assert det.face_label == "TOP"
    assert det.center_xyz_m is not None
    assert np.allclose(det.center_xyz_m[2], 0.5, atol=1e-6)

    payload = det.to_dict()
    assert payload["face"]["label"] == "TOP"
    assert payload["face"]["color_state"] == "纯红"
    assert payload["planes"][0]["normal"] == [0.0, 0.0, -1.0]
    assert "mask" not in payload


def test_analyze_frame_without_depth(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_planes", _fake_planes)
    mask = np.ones((40, 40), dtype=bool)
    detections = analyze_frame(_FakeModel(mask), _frame(depth=False), AnalysisConfig())
    assert len(detections) == 1
    assert detections[0].pose == "未知"
    assert detections[0].center_xyz_m is None
    assert detections[0].face_label == "UNKNOWN"


def test_bbox_fallback_when_model_has_no_masks(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_planes", _fake_planes)

    class _BoxOnlyModel:
        def predict(self, *args, **kwargs):
            boxes = SimpleNamespace(
                xyxy=_Tensor([[5.0, 5.0, 35.0, 35.0]]),
                conf=_Tensor([0.8]),
            )
            return [SimpleNamespace(boxes=boxes, masks=None)]

    detections = analyze_frame(_BoxOnlyModel(), _frame(), AnalysisConfig())
    assert len(detections) == 1
    assert detections[0].mask.sum() == 30 * 30
