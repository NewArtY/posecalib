"""Astra A6/A7: partial skeletons and vanishing robust residual scales."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibrator
from posecalib import bundle
from posecalib.geometry import reproject
from posecalib.initialization import local_bundle_adjust
from posecalib.robust import robust_sigma
from posecalib.skeleton import BLAZEPOSE33, Skeleton


def _exact_scene():
    R = np.tile(np.eye(3), (3, 1, 1))
    t = np.array([[0., 0, 0], [-1., 0, 0], [0., -2., 0]])
    X = np.array([[x, y, z] for z in (4., 8., 16.)
                  for y in (0., 1.) for x in (0., 1.)])
    f = np.full(3, 1024.)
    pi, ci = np.indices((len(X), len(R)))
    pi, ci = pi.ravel(), ci.ravel()
    obs = dict(pi=pi, ci=ci, uv=reproject(R, t, f, X)[0][pi, ci],
               c=np.resize([.25, .5, 1.], len(pi)),
               bones=(np.arange(0, len(X), 2), np.arange(1, len(X), 2),
                      np.full(len(X) // 2, 2)))
    return R, t, X, f, obs, np.array([np.nan, np.nan, 1., np.nan])


@pytest.mark.parametrize('loss', ['cauchy', 'huber'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('use_segments', [False, True])
@pytest.mark.parametrize('noise', [0., 1e-9])
def test_exact_and_nearly_exact_refinement(loss, weighted, use_segments, noise):
    R, t, X, f, obs, L = _exact_scene()
    obs['uv'] += np.random.default_rng(32).normal(0, noise, obs['uv'].shape)
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        out = bundle.refine(R, t, X, f, obs, L=L, loss=loss, weighted=weighted,
                            use_segments=use_segments)
    for key in ('Rs', 'ts', 'X', 'f', 'r_px', 'r_w', 'w', 'cost'):
        assert np.isfinite(out[key]).all(), key
    assert 0 < out['sig_hat'] < 1e-4
    assert np.max(out['r_px']) < 1e-6
    assert np.all((out['w'] > .99) & (out['w'] <= 1))
    if use_segments:
        np.testing.assert_allclose(out['L'][2], 1., atol=1e-8)
        assert np.isnan(out['L'][[0, 1, 3]]).all()
    else:
        assert out['L'] is None


@pytest.mark.parametrize('jac_mode', ['analytic', 'fd'])
@pytest.mark.parametrize('empty_pairs', [False, True])
def test_bundle_excludes_unobserved_lengths(jac_mode, empty_pairs):
    R, t, X, f, obs, L = _exact_scene()
    if empty_pairs:
        obs['bones'] = tuple(np.empty(0, dtype=int) for _ in range(3))
        L[:] = np.nan
    out = bundle.bundle_adjust(R, t, X, f, obs['ci'], obs['pi'], obs['uv'], obs['c'],
                               bones=obs['bones'], bone_w=.7, L0=L, jac_mode=jac_mode)
    assert np.isfinite(out['X']).all()
    assert out['L'].shape == L.shape
    assert np.isnan(out['L']).all() if empty_pairs else np.isnan(out['L'][[0, 1, 3]]).all()
    if not empty_pairs:
        assert out['L'][2] == pytest.approx(1.)
    assert np.isnan(L[[0, 1, 3]]).all()  # input indexing/unknown sentinel survives


def test_masked_analytic_jacobian_matches_central_differences(monkeypatch):
    R, t, X, f, obs, _ = _exact_scene()
    R[1:] = Rotation.from_rotvec([[.1, -.2, .03], [-.2, .1, .04]]).as_matrix()
    obs['uv'] = reproject(R, t, f, X)[0][obs['pi'], obs['ci']]
    bones = (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7]), np.array([4, 2, 4, 1]))
    L = np.array([np.nan, np.nan, 1., np.nan, 1., np.nan])
    # Last row has an unknown length and must be omitted, including its weight.
    bw = np.array([.2, .7, 1.3, 2.])
    solve = bundle.least_squares
    checked = []

    def checked_solve(fun, x0, *, jac, **kw):
        assert np.isfinite(x0).all()
        J = jac(x0).toarray()
        assert J.shape == (2 * len(obs['ci']) + 3, 11 + 3 * len(X) + 2 + len(R))
        assert np.all(np.any(J != 0, axis=0))
        h = 1e-6
        eye = np.eye(len(x0))
        fd = np.column_stack([(fun(x0 + h * e) - fun(x0 - h * e)) / (2 * h) for e in eye])
        np.testing.assert_allclose(J, fd, rtol=2e-6, atol=2e-6)
        # Sorted original ids [2, 4] map to exactly two columns before focals.
        np.testing.assert_allclose(J[-3:, -5:-3], [[0, -.2], [-.7, 0], [0, -1.3]])
        checked.append(True)
        return solve(fun, x0, jac=jac, **kw)

    monkeypatch.setattr(bundle, 'least_squares', checked_solve)
    out = bundle.bundle_adjust(R, t, X, f, obs['ci'], obs['pi'], obs['uv'], obs['c'],
                               bones=bones, bone_w=bw, L0=L, refine_focal=True)
    assert checked == [True]
    np.testing.assert_allclose(out['L'][[2, 4]], [1, 1])
    assert np.isnan(out['L'][[0, 1, 3, 5]]).all()


@pytest.mark.parametrize('custom', [False, True])
def test_public_refinement_of_partial_skeleton_preserves_unknown_lengths(custom):
    rng = np.random.default_rng(62)
    R = Rotation.from_rotvec([[0, 0, 0], [.01, .05, 0], [-.1, .1, -.03]]).as_matrix()
    C = np.array([[0., 0, 0], [1., 0, 0], [-.8, -.3, .3]])
    t = -np.einsum('cij,cj->ci', R, C)
    f = np.full(3, 1000.)
    X = rng.uniform([-.6, -.8, 3.5], [.6, .8, 5.5], (10, 33, 3))
    X[:, 1] = X[:, 0] + [.4, 0, 0]
    uv = reproject(R, t, f, X.reshape(-1, 3))[0]
    kp = uv.transpose(1, 0, 2).reshape(3, 10, 33, 2)
    confidence = np.ones(kp.shape[:-1])
    skeleton = (Skeleton(names=BLAZEPOSE33.names, segments=np.array([[0, 1], [2, 3]]),
                         segment_names=('seen', 'missing')) if custom else BLAZEPOSE33)
    missing = [2, 3] if custom else list(range(11, 33))
    kp[:, :, missing] = np.nan
    confidence[:, :, missing] = 0
    K = np.tile(np.diag([1000., 1000., 1.]), (3, 1, 1))
    cal = Calibrator(kp, confidence, K, skeleton=skeleton, camera_distance=(0, 1, 1.)).initialize()
    L0 = cal.init['L'].copy()
    out = cal.refine()
    assert out.scale_source == 'camera_distance'
    assert np.max(out.reprojection_rmse) < 1e-6
    assert np.isnan(out.points[:, missing]).all()
    observed = np.ones(33, dtype=bool)
    observed[missing] = False
    np.testing.assert_allclose(out.points[:, observed], X[:, observed], atol=1e-6)
    if custom:
        assert np.isnan(L0[1]) and np.isnan(out.segment_lengths[1])
        assert out.segment_lengths[0] == pytest.approx(.4, abs=1e-6)
    else:
        assert np.isnan(L0).all() and np.isnan(out.segment_lengths).all()
    np.testing.assert_equal(cal.init['L'], L0)


def test_exact_local_initialization_bundle_is_finite():
    rng = np.random.default_rng(5)
    X = rng.integers(-2, 3, (60, 3)).astype(float)
    X[:, 2] = 8.
    R = np.tile(np.eye(3), (3, 1, 1))
    t = np.array([[0., 0, 0], [-1., 0, 0], [0., -2., 0]])
    f = np.full(3, 1024.)
    tracks = dict(uv=reproject(R, t, f, X)[0], c=np.full((60, 3), .5),
                  valid=np.ones((60, 3), dtype=bool))
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        Rout, tout = local_bundle_adjust(R, t, f, tracks, X, np.ones(3, bool), 0, np.ones(60, bool))
    np.testing.assert_allclose(Rout, R, atol=1e-12)
    np.testing.assert_allclose(tout, t, atol=1e-12)


@pytest.mark.parametrize('loss', ['cauchy', 'huber'])
@pytest.mark.parametrize('weighted', [False, True])
def test_noisy_fit_still_rejects_large_outliers(loss, weighted):
    rng = np.random.default_rng(78)
    R, t, _, f, _, _ = _exact_scene()
    X = rng.uniform([-1, -1, 4], [1, 1, 8], (60, 3))
    pi, ci = np.indices((len(X), 3))
    pi, ci = pi.ravel(), ci.ravel()
    uv = reproject(R, t, f, X)[0][pi, ci] + rng.normal(0, .2, (len(pi), 2))
    bad = np.array([5, 37, 91, 117])
    uv[bad] += [50, -70]
    obs = dict(pi=pi, ci=ci, uv=uv, c=rng.uniform(.4, 1, len(pi)), bones=None)
    out = bundle.refine(R, t, X, f, obs, loss=loss, weighted=weighted, use_segments=False)
    good = np.ones(len(pi), bool)
    good[bad] = False
    assert .01 < out['sig_hat'] < 1.
    assert np.max(out['w'][bad]) < .05
    assert np.median(out['r_px'][good]) < .5
    assert np.median(out['w'][good]) > .5


@pytest.mark.parametrize('noise', [0., .2])
def test_confidence_rescaling_preserves_noise_units_and_solution(noise):
    R, t, X, f, obs, L = _exact_scene()
    obs['uv'] += np.random.default_rng(9).normal(0, noise, obs['uv'].shape)
    a = bundle.refine(R, t, X, f, obs, L=L, max_nfev=100)
    obs['c'] *= .25
    b = bundle.refine(R, t, X, f, obs, L=L, max_nfev=100)
    assert b['sig_hat'] == pytest.approx(.25 * a['sig_hat'], rel=2e-3, abs=1e-12)
    np.testing.assert_allclose(b['w'], a['w'], atol=2e-3)
    np.testing.assert_allclose(b['r_px'], a['r_px'], atol=2e-3)
    np.testing.assert_allclose(b['L'], a['L'], atol=2e-3)


def test_zero_mad_keeps_exact_inliers_and_downweights_nonzero_outliers():
    residual = np.array([0., 0., 0., 0., 10.])
    sig = robust_sigma(residual)
    assert 0 < sig < 1e-6
    for loss in ('cauchy', 'huber'):
        w = bundle.robust_weight(residual / (2.5 * sig), loss)
        np.testing.assert_equal(w[:4], np.ones(4))
        assert w[-1] < 1e-6
