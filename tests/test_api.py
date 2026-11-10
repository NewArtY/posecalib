import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import posecalib.api as api
from posecalib import (COCO17, Calibration, Calibrator, project, reprojection_errors,
                       to_centered, triangulate)
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


@pytest.mark.parametrize('refine_focal', [False, True])
def test_refinement_reports_image_pixel_errors_on_optimizer_observations(
        refine_focal, monkeypatch, tmp_path):
    rng = np.random.default_rng(20261110)
    R = Rotation.from_rotvec([[0., 0., 0.], [.02, .13, -.04], [-.16, .04, .06]]).as_matrix()
    t = np.array([[0., 0., 0.], [-1., 0., 0.], [0., -.8, 0.]])
    K = np.tile(np.eye(3), (3, 1, 1))
    K[:, 0, 0] = [1000., 900., 800.]
    K[:, 1, 1] = [500., 1200., 800.]
    K[:, :2, 2] = [[540., 380.], [600., 400.], [620., 420.]]
    points = rng.uniform([-.6, -.8, 3.5], [.6, .8, 5.5], (10, 17, 3))
    truth = Calibration(R, t, K, points, None, 'test', np.zeros(3), np.ones(3))
    keypoints = project(truth, points) + rng.normal(0., .5, (3, 10, 17, 2))
    confidence = rng.uniform(.7, 1., keypoints.shape[:-1])
    confidence[0, 2, 5] = 0.
    keypoints[1, 4, 6] += [20., -60.]
    cal = Calibrator(keypoints, confidence, K, camera_distance=(0, 1, 2.))
    cal.initialize(local_refine=False)

    optimizer = api._refine
    internal = {}

    def capture_optimizer(*args, **kwargs):
        out = optimizer(*args, **kwargs)
        internal.update(weights=out['w'].copy(), sigma=out['sig_hat'], residuals=out['r_px'].copy())
        return out

    monkeypatch.setattr(api, '_refine', capture_optimizer)
    result = cal.refine(use_segments=False, refine_focal=refine_focal, max_nfev=30, irls_max_iter=5)
    if refine_focal:
        assert np.max(np.abs(result.K[:, 0, 0] - K[:, 0, 0])) > .01
    else:
        np.testing.assert_array_equal(result.K, K)
    np.testing.assert_array_equal(result.info['weights'], internal['weights'])
    assert result.info['sigma_hat'] == internal['sigma']
    inliers = internal['weights'] > .5
    assert inliers.any() and not inliers.all()

    obs = cal.init['obs']
    cameras = obs['ci']
    tracks = obs['track_of_point'][obs['pi']]
    errors = reprojection_errors(result, keypoints, result.points).reshape(3, -1)[cameras, tracks]
    assert not np.allclose(errors[cameras == 0], internal['residuals'][cameras == 0])
    np.testing.assert_allclose(result.info['residuals_px'], errors, atol=1e-12)
    rmse = [np.sqrt(np.mean(errors[(cameras == k) & inliers] ** 2)) for k in range(3)]
    np.testing.assert_allclose(result.reprojection_rmse, rmse, atol=1e-12)
    np.testing.assert_allclose(result.inlier_fraction, [inliers[cameras == k].mean() for k in range(3)])
    np.testing.assert_allclose(result.to_dict()['reprojection_rmse_px'], rmse, atol=1e-12)
    path = tmp_path / 'rig.json'
    result.save_json(path)
    np.testing.assert_allclose(Calibration.load_json(path).reprojection_rmse, rmse, atol=1e-12)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('imaginary', [0., 100.])
def test_project_and_reprojection_errors_reject_complex_observations(imaginary):
    rig = Calibration(np.eye(3)[None], np.zeros((1, 3)), np.eye(3),
                      np.empty((0, 1, 3)), None, 'test', np.zeros(1), np.ones(1))
    points = np.array([[[1., 2., 5.]]])
    keypoints = project(rig, points)
    complex_points = points.astype(complex)
    complex_points.flat[0] += imaginary * 1j
    complex_keypoints = keypoints.astype(complex)
    complex_keypoints.flat[0] += imaginary * 1j
    with pytest.raises(ValueError, match='points must contain real'):
        project(rig, complex_points)
    with pytest.raises(ValueError, match='points must contain real'):
        reprojection_errors(rig, keypoints, complex_points)
    with pytest.raises(ValueError, match='keypoints must contain real'):
        reprojection_errors(rig, complex_keypoints, points)


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
