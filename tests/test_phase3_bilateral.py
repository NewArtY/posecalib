"""A6: default bilateral anatomy must never impute an unobserved side.

Uses the already fixed phase-3 exact-data tolerances: 2e-6 m/px for iterative
geometry, 1e-12 m for the directly applied measured baseline scale.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibrator
from posecalib.skeleton import BLAZEPOSE33


@pytest.mark.parametrize('loss', ['huber', 'cauchy'])
@pytest.mark.parametrize('weighted', [False, True])
def test_default_bilateral_unseen_side_stays_unknown(loss, weighted):
    rng = np.random.default_rng(211)
    R = Rotation.from_rotvec([[0., 0., 0.], [.03, -.07, .02], [-.08, .04, -.02]]).as_matrix()
    centers = np.array([[0., 0., 0.], [1., .2, .1], [-.8, -.4, .2]])
    t = -np.einsum('cij,cj->ci', R, centers)
    f = np.array([900., 1000., 1100.])
    X = rng.uniform([-.6, -.7, 4.], [.6, .7, 6.], (8, 33, 3))
    # All true limb lengths are constant; only right elbow observations vanish.
    # The left upper arm (11,13) is the observed mirror of right (12,14).
    X[:, 13] = X[:, 11] + [.4, 0., 0.]
    X[:, 15] = X[:, 13] + [0., -.3, 0.]
    X[:, 14] = X[:, 12] + [-.4, 0., 0.]
    X[:, 16] = X[:, 14] + [0., -.3, 0.]
    X[:, 24] = X[:, 23] + [.3, 0., 0.]
    X[:, 25] = X[:, 23] + [0., -.45, 0.]
    X[:, 27] = X[:, 25] + [0., -.4, 0.]
    X[:, 26] = X[:, 24] + [0., -.45, 0.]
    X[:, 28] = X[:, 26] + [0., -.4, 0.]
    xc = np.einsum('cij,fpj->cfpi', R, X) + t[:, None, None]
    kp = f[:, None, None, None] * xc[..., :2] / xc[..., 2:]
    confidence = rng.uniform(.75, 1., kp.shape[:-1])
    kp[:, :, 14] = np.nan
    confidence[:, :, 14] = 0.
    K = np.array([np.diag([ff, ff, 1.]) for ff in f])
    measured = 1.75 * np.linalg.norm(centers[1])
    # Infer the real default layout, including its bilateral metadata.
    cal = Calibrator(kp, confidence, K, camera_distance=(0, 1, measured)).initialize()
    assert cal.skeleton is BLAZEPOSE33
    assert (11, 12) in cal.skeleton.mirror_pairs and (13, 14) in cal.skeleton.mirror_pairs
    initial = cal.init['L'].copy()
    assert np.isfinite(initial[[0, 1]]).all()
    assert np.isnan(initial[[2, 3]]).all()
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        out = cal.refine(loss=loss, weighted=weighted)
    assert np.isnan(out.segment_lengths[[2, 3]]).all()
    np.testing.assert_allclose(out.segment_lengths[[0, 1]], [.7, .525], atol=2e-6, rtol=0.)
    np.testing.assert_array_equal(cal.init['L'], initial)
    assert out.scale_source == 'camera_distance'
    assert np.linalg.norm(out.centers[1] - out.centers[0]) == pytest.approx(measured, abs=1e-12)
    visible = np.delete(np.arange(33), 14)
    assert np.isnan(out.points[:, 14]).all()
    np.testing.assert_allclose(out.points[:, visible], 1.75 * X[:, visible], atol=2e-6, rtol=0.)
    assert np.max(out.reprojection_rmse) < 2e-6
