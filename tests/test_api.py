import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import COCO17, Calibration, Calibrator, project, to_centered, triangulate
from posecalib.sim import DEFAULT_CFG, intrinsics, make_seed_data


def _true_calibration(cams):
    """Ground-truth calibration of simulated cameras, in the world frame."""
    R = np.stack([c['R'] for c in cams])
    t = np.stack([c['t'] for c in cams])
    return Calibration(R, t, intrinsics(cams), np.zeros((0, 17, 3)), None, 'true',
                       np.zeros(len(cams)), np.ones(len(cams)))


@pytest.fixture(scope='module')
def scene():
    return make_seed_data(0, DEFAULT_CFG)


def test_to_centered_handles_principal_point_and_aspect():
    K = np.array([[1000.0, 0, 900], [0, 800.0, 500], [0, 0, 1]])
    uv, f = to_centered(np.array([[[1000.0, 600.0]]]), K)
    assert np.allclose(uv, [[[100.0, 125.0]]])
    assert np.allclose(f, [1000.0])


def test_project_then_triangulate_roundtrip(scene):
    calib = _true_calibration(scene['cams'])
    X = scene['walk']['J'][:30]                                  # (F,17,3)
    kp = project(calib, X)                                       # (C,F,17,2)
    conf = np.ones(kp.shape[:3])
    X_e, inliers = triangulate(calib, kp, conf)
    assert np.nanmax(np.abs(X_e - X)) < 1e-6
    assert inliers.all()


def test_transformed_keeps_projections(scene):
    calib = _true_calibration(scene['cams'])
    X = scene['walk']['J'][:5]
    Rw = Rotation.from_rotvec([0.1, -0.3, 0.7]).as_matrix()
    moved = calib.transformed(2.0, Rw, [0.5, -1.0, 2.0])
    assert np.allclose(project(moved, 2.0 * X @ Rw.T + [0.5, -1.0, 2.0]), project(calib, X), atol=1e-8)


def test_json_roundtrip(tmp_path, scene):
    calib = _true_calibration(scene['cams'])
    calib.save_json(tmp_path / 'rig.json')
    back = Calibration.load_json(tmp_path / 'rig.json')
    assert np.allclose(back.R, calib.R) and np.allclose(back.t, calib.t) and np.allclose(back.K, calib.K)


def test_input_validation():
    K = np.eye(3)
    with pytest.raises(ValueError):
        Calibrator(np.zeros((2, 5, 17, 3)), np.zeros((2, 5, 17)), K)
    with pytest.raises(ValueError):
        Calibrator(np.zeros((2, 5, 17, 2)), np.zeros((2, 5, 16)), K)
    with pytest.raises(ValueError):
        Calibrator(np.zeros((2, 5, 12, 2)), np.zeros((2, 5, 12)), K, skeleton=COCO17)
