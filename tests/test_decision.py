from skyblock.decision import FaceLabel, decide_face
from skyblock.laser import LaserProbe, laser_fallback


def test_top_is_geometric_and_colour_agnostic():
    for color in ("纯红", "纯蓝", "红左蓝右", "红上蓝下", "颜色未知"):
        result = decide_face("顶面", color)
        assert result.label is FaceLabel.TOP
        assert result.color_state == color


def test_single_side():
    assert decide_face("单侧面", "红左蓝右").label is FaceLabel.SIDE
    assert decide_face("单侧面", "纯蓝").label is FaceLabel.SIDE


def test_fold_is_still_a_side():
    for color in ("红左蓝右", "红右蓝左", "红上蓝下", "纯红", "颜色未知"):
        assert decide_face("折角双面", color).label is FaceLabel.SIDE
    assert decide_face("折角双面", "红左蓝右", laser_state="折角").label is FaceLabel.SIDE


def test_side_conflicts_lower_confidence():
    normal = decide_face("折角双面", "红左蓝右")
    conflict = decide_face("折角双面", "红左蓝右", laser_state="单平面")
    assert conflict.label is FaceLabel.SIDE
    assert conflict.confidence < normal.confidence
    assert any("冲突" in reason for reason in conflict.reasons)

    top_conflict = decide_face("顶面", "纯红", laser_state="折角")
    assert top_conflict.label is FaceLabel.TOP
    assert top_conflict.confidence < normal.confidence


def test_laser_fallback_rules():
    # A laser fold is a side face, even when 3D failed.
    laser_fold = decide_face("未知", "红左蓝右", laser_state="折角")
    assert laser_fold.label is FaceLabel.SIDE
    assert laser_fold.confidence < decide_face("单侧面", "红左蓝右").confidence

    # A single-plane laser cannot separate top from side.
    assert decide_face("未知", "颜色未知", laser_state="单平面").label is FaceLabel.UNKNOWN
    assert decide_face("未知", "颜色未知").label is FaceLabel.UNKNOWN


def test_laser_fallback():
    assert laser_fallback(0.40, 0.41, 0.03) == "单平面"
    assert laser_fallback(0.40, 0.46, 0.03) == "折角"
    assert laser_fallback(None, 0.46) is None
    assert laser_fallback(0.46, None) is None


def test_laser_probe_read_override():
    probe = LaserProbe(diff_threshold=0.03)
    probe.update(0.30, 0.36)
    assert probe.classify() == "折角"
    probe.update(0.30, 0.31)
    assert probe.classify() == "单平面"
