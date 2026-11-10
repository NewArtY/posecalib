import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibration, project, triangulate
from posecalib.geometry import (eight_point, homography_median, pnp_dlt, reproject,
                               skew, triangulate_dlt, triangulate_robust)


def _rig(rotations, centers):
    R = Rotation.from_rotvec(rotations).as_matrix()
    t = -np.einsum('mij,mj->mi', R, np.asarray(centers, float))
    K = np.tile([[1000., 0., 640.], [0., 1000., 360.], [0., 0., 1.]], (len(R), 1, 1))
    return Calibration(R, t, K, np.empty((0, 17, 3)), None, 'true',
                       np.zeros(len(R)), np.ones(len(R)))


def _points():
    return np.linspace([.2, .1, 4.], [.4, -.2, 5.], 34).reshape(2, 17, 3)


@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_clean_pair_survives_confident_behind_outlier(order):
    rig = _rig([[0, 0, 0], [0, 0, 0], [0, np.pi, 0]],
               [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    X = _points()
    kp = project(rig, X)
    kp[2] = [900., 200.]
    order = list(order)
    rig.R, rig.t, rig.K = rig.R[order], rig.t[order], rig.K[order]
    kp = kp[order]
    conf = np.ones(kp.shape[:3])
    reconstructed, inliers = triangulate(rig, kp, conf)
    expected = np.broadcast_to((np.array(order) != 2)[:, None, None], conf.shape)
    assert np.isfinite(reconstructed).all()
    np.testing.assert_allclose(reconstructed, X, atol=1e-8)
    np.testing.assert_array_equal(inliers, expected)

    # Turning off an already rejected view must not change reconstruction.
    conf[np.array(order) == 2] = 0.
    clean, _ = triangulate(rig, kp, conf)
    np.testing.assert_allclose(reconstructed, clean, atol=1e-8)


@pytest.mark.parametrize('n_irls', [0, 6])
@pytest.mark.parametrize('front_views', [0, 1, 2])
def test_only_positive_depth_consistent_views_are_inliers(front_views, n_irls):
    rotations = [[0, 0 if i < front_views else np.pi, 0] for i in range(3)]
    rig = _rig(rotations, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    X = _points().reshape(-1, 3)
    uv, depth, _ = reproject(rig.R, rig.t, rig.K[:, 0, 0], X)
    c = np.ones(uv.shape[:2])
    reconstructed, ok, _, inliers, _ = triangulate_robust(
        rig.R, rig.t, rig.K[:, 0, 0], uv, c, np.ones_like(c, bool),
        sig_hat=1., n_irls=n_irls)
    assert np.all((depth > 0).sum(1) == front_views)
    assert not inliers[depth <= 0].any()
    np.testing.assert_array_equal(ok, inliers.sum(1) >= 2)
    assert np.all(ok == (front_views >= 2))
    if front_views >= 2:
        np.testing.assert_allclose(reconstructed, X, atol=1e-8)

    public_points, public_inliers = triangulate(rig, project(rig, _points()),
                                               np.ones((3, 2, 17)))
    if front_views < 2:
        assert np.isnan(public_points).all()
        assert np.all(public_inliers.sum(0) < 2)
    else:
        np.testing.assert_allclose(public_points, _points(), atol=1e-8)


def test_only_one_usable_view_cannot_triangulate():
    rig = _rig([[0, 0, 0]] * 3, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    conf = np.zeros((3, 2, 17))
    conf[0] = 1.
    reconstructed, inliers = triangulate(rig, project(rig, _points()), conf)
    assert np.isnan(reconstructed).all()
    assert not inliers.any()


def test_mixed_orientations_keep_each_visible_pair_in_partial_overlap():
    rig = _rig([[0, 0, 0], [0, 0, 0], [0, np.pi, 0], [0, np.pi, 0]],
               [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    X = _points()
    X[1, :, 2] *= -1
    kp = project(rig, X)
    reconstructed, inliers = triangulate(rig, kp, np.ones(kp.shape[:3]))
    np.testing.assert_allclose(reconstructed, X, atol=1e-8)
    expected = np.zeros((4, 2, 17), bool)
    expected[:2, 0] = True
    expected[2:, 1] = True
    np.testing.assert_array_equal(inliers, expected)


@pytest.mark.parametrize('angle', [0., .3])
@pytest.mark.parametrize('rejected_third_view', [False, True])
def test_coincident_camera_centers_have_no_observable_depth(angle, rejected_third_view):
    # A translated origin catches arbitrary positive-depth DLT solutions that
    # happen to be hidden by a zero-depth result for cameras at the world origin.
    rotations = [[0, 0, 0], [0, angle, 0]]
    centers = [[.3, -.4, 1.]] * 2
    if rejected_third_view:
        rotations.append([0, np.pi, 0])
        centers.append([2., 0., 0.])
    rig = _rig(rotations, centers)
    kp = project(rig, _points())
    reconstructed, _ = triangulate(rig, kp, np.ones(kp.shape[:3]))
    assert np.isnan(reconstructed).all()


def test_small_nonzero_baseline_still_triangulates():
    rig = _rig([[0, 0, 0]] * 2, [[.3, -.4, 1.], [.3001, -.4, 1.]])
    X = _points()
    kp = project(rig, X)
    reconstructed, inliers = triangulate(rig, kp, np.ones(kp.shape[:3]))
    np.testing.assert_allclose(reconstructed, X, atol=1e-6)
    assert inliers.all()


def test_cameras_and_point_on_same_line_have_no_observable_depth():
    rig = _rig([[0, 0, 0]] * 2, [[0, 0, 0], [0, 0, 1]])
    X = np.broadcast_to([0., 0., 4.], (2, 17, 3))
    kp = project(rig, X)
    reconstructed, _ = triangulate(rig, kp, np.ones(kp.shape[:3]))
    assert np.isnan(reconstructed).all()


def _check_svd_shapes(monkeypatch):
    original = np.linalg.svd
    calls = []

    def checked(a, *args, **kwargs):
        # Check before calling LAPACK, so a regression cannot allocate a tall
        # matrix's quadratic-size U even while running this test.
        wide = a.shape[-2] < a.shape[-1]
        assert kwargs.get('full_matrices', True) == wide
        U, s, Vt = original(a, *args, **kwargs)
        assert U.shape[-1] == min(a.shape[-2:])
        assert Vt.shape[-2:] == (a.shape[-1], a.shape[-1])
        calls.append(a.shape[-2:])
        return U, s, Vt

    monkeypatch.setattr(np.linalg, 'svd', checked)
    return calls


@pytest.mark.parametrize('n', [6, 40, 1000])
def test_pnp_svd_uses_linear_memory_and_preserves_pose(monkeypatch, n):
    rng = np.random.default_rng(2)
    X = rng.uniform(-1., 1., (2, n, 3))
    R = Rotation.from_rotvec([[.1, .2, .1], [-.2, .1, -.1]]).as_matrix()
    t = np.array([[.3, -.4, 4.], [-.1, .2, 5.]])
    xc = np.einsum('bij,bnj->bni', R, X) + t[:, None]
    xn = xc[..., :2] / xc[..., 2:]
    # Compare to the former full-SVD path on small systems only.
    reference = None
    if n <= 40:
        original = np.linalg.svd

        def full(a, *args, **kwargs):
            kwargs['full_matrices'] = True
            return original(a, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, 'svd', full)
            reference = pnp_dlt(X, xn)
    calls = _check_svd_shapes(monkeypatch)
    estimated_R, estimated_t = pnp_dlt(X, xn)
    assert calls == [(2 * n, 12), (3, 3)]
    np.testing.assert_allclose(estimated_R, R, atol=1e-10)
    np.testing.assert_allclose(estimated_t, t, atol=1e-10)
    if reference is not None:
        np.testing.assert_allclose(estimated_R, reference[0], atol=1e-12)
        np.testing.assert_allclose(estimated_t, reference[1], atol=1e-12)


@pytest.mark.parametrize('n', [8, 9, 40])
def test_eight_point_svd_preserves_minimal_nullspace(monkeypatch, n):
    rng = np.random.default_rng(7)
    X = rng.uniform([-1., -1., 4.], [1., 1., 6.], (n, 3))
    R = Rotation.from_rotvec([.1, .3, -.2]).as_matrix()
    t = np.array([-1., .1, .2])
    xc = X @ R.T + t
    calls = _check_svd_shapes(monkeypatch)
    E = eight_point((X[:, :2] / X[:, 2:])[None],
                    (xc[:, :2] / xc[:, 2:])[None])[0]
    expected = skew(t) @ R
    E /= np.linalg.norm(E)
    expected /= np.linalg.norm(expected)
    if np.sum(E * expected) < 0:
        E = -E
    np.testing.assert_allclose(E, expected, atol=1e-8)
    assert calls == [(n, 9), (3, 3)]


def test_four_point_homography_svd_keeps_ninth_null_vector(monkeypatch):
    rng = np.random.default_rng(8)
    x1 = rng.uniform(-.5, .5, (30, 2))
    H = np.array([[1.1, .1, .2], [-.1, .9, -.1], [.05, -.03, 1.]])
    homogeneous = np.column_stack([x1, np.ones(len(x1))]) @ H.T
    x2 = homogeneous[:, :2] / homogeneous[:, 2:]
    calls = _check_svd_shapes(monkeypatch)
    error = homography_median(x1, x2, rng, n_iter=20)
    assert error < 1e-20
    assert calls == [(8, 9)]


@pytest.mark.parametrize('n_cameras', [2, 3, 8])
def test_triangulation_dlt_uses_reduced_svd(monkeypatch, n_cameras):
    centers = np.zeros((n_cameras, 3))
    centers[:, 0] = np.arange(n_cameras)
    rig = _rig(np.zeros((n_cameras, 3)), centers)
    X = _points().reshape(-1, 3)
    uv, _, _ = reproject(rig.R, rig.t, np.ones(n_cameras), X)
    calls = _check_svd_shapes(monkeypatch)
    estimated = triangulate_dlt(rig.R, rig.t, uv, np.ones(uv.shape[:2]))
    np.testing.assert_allclose(estimated, X, atol=1e-10)
    assert calls == [(2 * n_cameras, 4)]
