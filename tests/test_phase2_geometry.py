"""A2/A9: analytic camera truth, bounded SVD and minimal right nullspaces."""
import math

import numpy as np
import pytest

from posecalib import Calibration, triangulate
from posecalib.geometry import eight_point, homography_median, pnp_dlt, triangulate_robust


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@pytest.mark.parametrize('n_irls', [0, 6])
@pytest.mark.parametrize('case', ['false_behind', 'one_usable', 'all_behind',
                                 'nearly_parallel', 'small_baseline'])
def test_stereo_depth_against_analytic_disparity(case, n_irls):
    # Explicit pinhole equations, independent of project/reproject/triangulate_dlt.
    X = np.random.default_rng(20261103).uniform([-.4, -.3, 3.], [.6, .4, 6.], (34, 3))
    R = np.tile(np.eye(3), (3, 1, 1))
    R[2] = np.diag([-1., 1., -1.])
    baseline = {'nearly_parallel': 1e-10, 'small_baseline': 1e-4}.get(case, 1.)
    centers = np.array([[0., 0., 0.], [baseline, 0., 0.], [2., 0., 0.]])
    if case == 'all_behind':
        R[:] = np.diag([-1., 1., -1.])
    t = -np.einsum('cij,cj->ci', R, centers)
    camera_X = np.einsum('cij,tcj->tci', R, X[:, None] - centers)
    depth = camera_X[..., 2]
    uv = 1000. * camera_X[..., :2] / depth[..., None]
    valid = np.ones((34, 3), bool)
    if case == 'false_behind':
        uv[:, 2] = [260., -160.]  # original false detection, centred pixels
    elif case == 'one_usable':
        valid[:, 1:] = False
    else:
        valid[:, 2] = False
    confidence = valid.astype(float)
    result, ok, _, inliers, _ = triangulate_robust(
        R, t, np.full(3, 1000.), uv, confidence, valid, sig_hat=1., n_irls=n_irls)
    assert not inliers[~valid | (depth <= 0)].any()
    successful = case in ('false_behind', 'small_baseline')
    assert np.all(ok == successful)
    if successful:
        np.testing.assert_array_equal(inliers, np.tile([True, True, False], (34, 1)))
        # Closed-form stereo depth from disparity provides a second truth check.
        stereo_z = 1000. * baseline / (uv[:, 0, 0] - uv[:, 1, 0])
        np.testing.assert_allclose(stereo_z, X[:, 2], rtol=0, atol=1e-8)
        np.testing.assert_allclose(result, X, rtol=0, atol=1e-6 if baseline < 1. else 1e-8)
    K = np.array([[1000., 0., 640.], [0., 1000., 360.], [0., 0., 1.]])
    rig = Calibration(R, t, K, np.empty((0, 17, 3)), None, 'true', np.zeros(3), np.ones(3))
    pixels = (uv + [640., 360.]).transpose(1, 0, 2).reshape(3, 2, 17, 2)
    public, public_inliers = triangulate(rig, pixels, confidence.T.reshape(3, 2, 17))
    if successful:
        np.testing.assert_allclose(public.reshape(-1, 3), X, rtol=0,
                                   atol=1e-6 if baseline < 1. else 1e-8)
        np.testing.assert_array_equal(public_inliers[:, 0, 0], [True, True, False])
    else:
        assert np.isnan(public).all()


@pytest.mark.parametrize('seed', [13, 37, 91])
def test_minimal_eight_point_null_vector_against_linear_solve(monkeypatch, seed):
    X = np.random.default_rng(seed).uniform([-1., -1., 3.], [1., 1., 7.], (2, 8, 3))
    c, s = math.cos(.31), math.sin(.31)
    R = _rotation(-.17) @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    t = np.array([-.9, .17, .23])
    xc = X @ R.T + t
    first, second = X[..., :2] / X[..., 2:], xc[..., :2] / xc[..., 2:]
    calls = []
    original = np.linalg.svd

    def inspect(a, *args, **kwargs):
        result = original(a, *args, **kwargs)
        if a.shape[-2:] == (8, 9):
            U, singular, vt = result
            assert U.shape == (2, 8, 8) and vt.shape == (2, 9, 9)
            assert np.all(singular[:, -1] > 0.)
            # Solve a square inhomogeneous system; no SVD reference implementation.
            coefficients = np.linalg.solve(a[..., :8], -a[..., 8, None])[..., 0]
            reference = np.concatenate([coefficients, np.ones((2, 1))], axis=-1)
            reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
            direction = vt[:, -1]
            sign = np.where(np.sum(reference * direction, axis=-1) < 0., -1., 1.)
            np.testing.assert_allclose(direction * sign[:, None], reference, rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.einsum('bij,bj->bi', a, direction), 0., rtol=0, atol=1e-12)
            calls.append(a.shape)
        return result

    monkeypatch.setattr(np.linalg, 'svd', inspect)
    estimated = eight_point(first, second)
    # Column j of [t]x R is t cross R[:, j].
    expected = np.cross(t, R.T).T
    expected /= np.linalg.norm(expected)
    estimated /= np.linalg.norm(estimated, axis=(1, 2), keepdims=True)
    sign = np.where(np.sum(estimated * expected, axis=(1, 2)) < 0., -1., 1.)
    np.testing.assert_allclose(estimated * sign[:, None, None],
                               np.broadcast_to(expected, (2, 3, 3)), rtol=0, atol=1e-9)
    assert calls == [(2, 8, 9)]


@pytest.mark.parametrize('seed', [19, 53, 101])
def test_four_point_homography_null_vector_against_known_map(monkeypatch, seed):
    points = np.random.default_rng(seed).uniform(-.6, .6, (30, 2))
    H = np.array([[1.13, .12, .21], [-.07, .93, -.11], [.06, -.04, 1.]])
    mapped = np.column_stack([points, np.ones(len(points))]) @ H.T
    mapped = mapped[:, :2] / mapped[:, 2:]
    original = np.linalg.svd
    calls = []

    def inspect(a, *args, **kwargs):
        U, singular, vt = original(a, *args, **kwargs)
        assert a.shape == (24, 8, 9) and U.shape == (24, 8, 8) and vt.shape == (24, 9, 9)
        reference = np.concatenate([np.linalg.solve(a[..., :8], -a[..., 8, None])[..., 0],
                                    np.ones((24, 1))], axis=-1)
        vector = vt[:, -1] / vt[:, -1, -1:]
        np.testing.assert_allclose(vector, reference, rtol=0, atol=1e-9)
        np.testing.assert_allclose(vector.reshape(-1, 3, 3), np.broadcast_to(H, (24, 3, 3)),
                                   rtol=0, atol=1e-9)
        calls.append(a.shape)
        return U, singular, vt

    monkeypatch.setattr(np.linalg, 'svd', inspect)
    error = homography_median(points, mapped, np.random.default_rng(seed + 1), n_iter=24)
    assert error < 1e-20
    assert calls == [(24, 8, 9)]


@pytest.mark.parametrize('n', [6, 300, 1000, 2000])
def test_pnp_u_is_bounded_before_allocation(monkeypatch, n):
    X = np.random.default_rng(443).uniform(-1., 1., (2, n, 3))
    R = np.stack([_rotation(.37), _rotation(-.51)])
    t = np.array([[.31, -.27, 4.3], [-.2, .4, 5.1]])
    xc = X @ R.transpose(0, 2, 1) + t[:, None]
    original = np.linalg.svd
    calls = []

    def bounded(a, *args, **kwargs):
        # Assert before LAPACK: even the failing baseline never allocates large U.
        assert kwargs.get('full_matrices', True) is False
        U, singular, vt = original(a, *args, **kwargs)
        assert U.shape[-2:] == (a.shape[-2], min(a.shape[-2:]))
        assert U.size <= 2 * max(2 * n, 3) * 12
        calls.append(a.shape[-2:])
        return U, singular, vt

    monkeypatch.setattr(np.linalg, 'svd', bounded)
    Re, te = pnp_dlt(X, xc[..., :2] / xc[..., 2:])
    np.testing.assert_allclose(Re, R, rtol=0, atol=1e-10)
    np.testing.assert_allclose(te, t, rtol=0, atol=1e-10)
    assert calls == [(2 * n, 12), (3, 3)]


@pytest.mark.parametrize('n', [6, 17, 40])
@pytest.mark.parametrize('noise', [0., 1e-4])
def test_pnp_truth_and_former_small_full_svd(monkeypatch, n, noise):
    rng = np.random.default_rng(751)
    X = rng.uniform(-1., 1., (1, n, 3))
    R = _rotation(.37)
    t = np.array([.31, -.27, 4.3])
    xc = X @ R.T + t
    exact = xc[..., :2] / xc[..., 2:]
    xn = exact + rng.normal(0., noise, exact.shape)
    Re, te = pnp_dlt(X, xn)
    original = np.linalg.svd

    def former(a, *args, **kwargs):
        kwargs['full_matrices'] = True
        return original(a, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, 'svd', former)
        previous_R, previous_t = pnp_dlt(X, xn)
    np.testing.assert_allclose(Re, previous_R, rtol=0, atol=1e-12)
    np.testing.assert_allclose(te, previous_t, rtol=0, atol=1e-12)
    if noise == 0.:
        np.testing.assert_allclose(Re[0], R, rtol=0, atol=1e-10)
        np.testing.assert_allclose(te[0], t, rtol=0, atol=1e-10)
        reconstructed = X @ Re[0].T + te[0]
        np.testing.assert_allclose(reconstructed[..., :2] / reconstructed[..., 2:], exact,
                                   rtol=0, atol=1e-10)
