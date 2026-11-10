"""Independent phase-3 contracts; tolerances fixed before the first run.

Synthetic pixel truth uses explicit pinhole equations. Exact public solutions
allow 2e-6 m / 2e-6 px for iterative SfM and BA roundoff (no detector noise).
Central differences use relative steps 1e-6 and atol/rtol 2e-6: the truncation
error is O(h**2), while pixel subtraction loses about eps*1000/h ~ 2e-7.
These are numerical checks, not a physical camera accuracy claim.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibrator, bundle
from posecalib.initialization import local_bundle_adjust
from posecalib.skeleton import BLAZEPOSE33, Skeleton


def _pixels(R, t, f, X):
    xc = np.einsum('cij,pj->pci', R, X) + t[None]
    return xc[..., :2] / xc[..., 2:] * f[None, :, None]


def _scene(seed=503, n=24):
    rng = np.random.default_rng(seed)
    R = Rotation.from_rotvec([[0., 0., 0.], [.07, -.12, .04], [-.1, .06, -.03]]).as_matrix()
    t = np.array([[0., 0., 0.], [-1., .1, .04], [.2, -1.6, .1]])
    f = np.array([800., 1000., 1200.])
    X = rng.uniform([-.7, -.6, 4.], [.7, .6, 7.], (n, 3))
    pi, ci = np.indices((n, 3))
    pi, ci = pi.ravel(), ci.ravel()
    obs = dict(pi=pi, ci=ci, uv=_pixels(R, t, f, X)[pi, ci],
               c=rng.uniform(.4, 1., len(pi)), bones=None)
    return R, t, f, X, obs


@pytest.mark.parametrize('angle', [0., 1e-8, 1e-5, .7, np.pi - 1e-5])
@pytest.mark.parametrize('focal', [False, True])
def test_compact_jacobian_against_actual_objective(monkeypatch, angle, focal):
    R, t, f, X, obs = _scene(n=8)
    R[1] = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
    # Exercise all three choices of frozen translation component.
    axis = [0., 1e-8, 1e-5, .7, np.pi - 1e-5].index(angle) % 3
    t[2] = [.2, .1, .15]
    t[2, axis] = -2.
    obs['uv'] = _pixels(R, t, f, X)[obs['pi'], obs['ci']]
    # Finite but absent id 0, NaN id 1, infinite id 2, repeated id 6.
    L = np.array([.8, np.nan, np.inf, .45, np.nan, np.nan, .7])
    bones = (np.array([0, 2, 4, 6, 1]), np.array([1, 3, 5, 7, 2]), np.array([6, 1, 3, 6, 2]))
    weights = np.array([.3, .8, 1.2, .6, 1.7]) if focal else .7
    row_weights = np.broadcast_to(weights, (5,))[[0, 2, 3]]
    original = bundle.least_squares
    metrics = []

    def inspect(fun, x0, *, jac, **kw):
        # Independently calculate the initial objective in original segment ids.
        expected = np.r_[np.zeros(2 * len(obs['ci'])), row_weights * (
            np.linalg.norm(X[[0, 4, 6]] - X[[1, 5, 7]], axis=1) - L[[6, 3, 6]])]
        np.testing.assert_allclose(fun(x0), expected, atol=1e-10, rtol=1e-12)
        assert len(x0) == 11 + 3 * len(X) + 2 + (3 if focal else 0)
        assert np.isfinite(x0).all()
        for state, v in enumerate((x0, x0 + np.random.default_rng(409).normal(0, 1e-4, len(x0)))):
            J = jac(v).toarray()
            assert J.shape == (2 * len(obs['ci']) + 3, len(x0))
            # Differentiate the actual solver residual, without a copied Jacobian.
            fd = np.empty_like(J)
            for col in range(len(v)):
                h = 1e-6 * max(1., abs(v[col]))
                delta = np.zeros(len(v))
                delta[col] = h
                fd[:, col] = (fun(v + delta) - fun(v - delta)) / (2 * h)
            np.testing.assert_allclose(J, fd, atol=2e-6, rtol=2e-6)
            split = [0, 11, 11 + 3 * len(X), 13 + 3 * len(X), len(v)]
            record = dict(angle=float(angle), focal=focal, state=state, gauge_axis=axis)
            for name, a, b in zip(('poses', 'points', 'compact_lengths', 'focal'), split, split[1:]):
                if b > a:
                    error = np.abs(J[:, a:b] - fd[:, a:b])
                    record[name] = dict(max_absolute=float(error.max()),
                                       max_scaled=float((error / (1 + np.abs(fd[:, a:b]))).max()))
            metrics.append(record)
        return original(fun, x0, jac=jac, **kw)

    monkeypatch.setattr(bundle, 'least_squares', inspect)
    out = bundle.bundle_adjust(R, t, X, f, obs['ci'], obs['pi'], obs['uv'], obs['c'],
                               bones=bones, L0=L, bone_w=weights, refine_focal=focal, max_nfev=1)
    assert len(metrics) == 2
    np.testing.assert_allclose(out['Rs'][0], R[0], atol=1e-15)
    np.testing.assert_array_equal(out['ts'][0], t[0])
    assert out['ts'][2, axis] == t[2, axis]
    np.testing.assert_allclose(out['L'][[3, 6]], L[[3, 6]], atol=0.)
    assert np.isnan(out['L'][[0, 1, 2, 4, 5]]).all()
    # The harness records metrics as raw events alongside the ordinary test log.
    if os.environ.get('PHASE3_METRICS'):
        with Path(os.environ['PHASE3_METRICS']).open('a', encoding='utf-8') as stream:
            for record in metrics:
                stream.write(json.dumps(record) + '\n')


@pytest.mark.parametrize('loss', ['huber', 'cauchy'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('mode', ['no_pairs', 'unknown_pairs'])
def test_no_active_segments_is_geometry_only(loss, weighted, mode):
    R, t, f, X, obs = _scene()
    obs['uv'] += np.random.default_rng(17).normal(0, .03, obs['uv'].shape)
    if mode == 'no_pairs':
        obs['bones'] = tuple(np.empty(0, int) for _ in range(3))
        lengths = np.array([.3, np.nan, np.inf])
    else:
        obs['bones'] = (np.array([0, 2]), np.array([1, 3]), np.array([0, 2]))
        lengths = np.array([np.nan, .4, np.inf])
    before = lengths.copy()
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        segmented = bundle.refine(R, t, X, f, obs, L=lengths, loss=loss, weighted=weighted)
        geometry = bundle.refine(R, t, X, f, obs, loss=loss, weighted=weighted, use_segments=False)
    assert np.isnan(segmented['L']).all()
    assert geometry['L'] is None
    for key in ('Rs', 'ts', 'X', 'f', 'r_px', 'w', 'r_w', 'cost'):
        # Identical geometry objectives must take the same numerical path.
        np.testing.assert_array_equal(segmented[key], geometry[key])
        assert np.isfinite(segmented[key]).all()
    np.testing.assert_array_equal(lengths, before)


@pytest.mark.parametrize('kind', ['face33', 'custom_missing', 'custom_all_missing'])
@pytest.mark.parametrize('loss', ['huber', 'cauchy'])
@pytest.mark.parametrize('weighted', [False, True])
def test_public_partial_skeleton_measured_scale(kind, loss, weighted):
    R, t, f, _, _ = _scene()
    rng = np.random.default_rng(801)
    X = rng.uniform([-.7, -.8, 4.], [.7, .8, 6.], (8, 33, 3))
    X[:, 1] = X[:, 0] + [.4, 0., 0.]
    centers = -np.einsum('cji,cj->ci', R, t)
    baseline = float(np.linalg.norm(centers[1] - centers[0]))
    # Nontrivial measured scale tests independence from any anatomical prior.
    target_baseline = 2.75 * baseline
    kp = _pixels(R, t, f, X.reshape(-1, 3)).transpose(1, 0, 2).reshape(3, 8, 33, 2)
    confidence = rng.uniform(.75, 1., kp.shape[:-1])
    if kind == 'face33':
        skeleton = BLAZEPOSE33
        missing = np.arange(11, 33)
    else:
        segments = np.array([[0, 1], [2, 3]]) if kind == 'custom_missing' else np.array([[11, 13], [23, 25]])
        skeleton = Skeleton(BLAZEPOSE33.names, segments, ('first', 'second'))
        missing = np.array([2, 3]) if kind == 'custom_missing' else np.arange(11, 33)
    kp[:, :, missing] = np.nan
    confidence[:, :, missing] = 0.
    K = np.array([np.diag([ff, ff, 1.]) for ff in f])
    cal = Calibrator(kp, confidence, K, skeleton=skeleton, camera_distance=(0, 1, target_baseline)).initialize()
    initial_lengths = cal.init['L'].copy()
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        out = cal.refine(loss=loss, weighted=weighted)
    assert out.scale_source == 'camera_distance'
    assert np.linalg.norm(out.centers[1] - out.centers[0]) == pytest.approx(target_baseline, abs=1e-12)
    visible = np.setdiff1d(np.arange(33), missing)
    np.testing.assert_allclose(out.points[:, visible], 2.75 * X[:, visible], atol=2e-6, rtol=0.)
    assert np.isnan(out.points[:, missing]).all()
    assert np.max(out.reprojection_rmse) < 2e-6
    if kind == 'custom_missing':
        assert np.isnan(initial_lengths[1]) and np.isnan(out.segment_lengths[1])
        assert out.segment_lengths[0] == pytest.approx(1.1, abs=2e-6)
    else:
        assert np.isnan(initial_lengths).all() and np.isnan(out.segment_lengths).all()
    np.testing.assert_array_equal(cal.init['L'], initial_lengths)


@pytest.mark.parametrize('loss', ['huber', 'cauchy'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('noise', [0., 1e-10])
def test_majority_exact_fit_still_suppresses_nonzero_outliers(loss, weighted, noise):
    R, t, f, X, obs = _scene(n=40)
    obs['uv'] += np.random.default_rng(319).normal(0, noise, obs['uv'].shape)
    bad = np.array([4, 29, 57, 100])
    obs['uv'][bad] += [30., -40.]
    good = np.ones(len(obs['ci']), bool)
    good[bad] = False
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        out = bundle.refine(R, t, X, f, obs, loss=loss, weighted=weighted, use_segments=False)
    assert 0 < out['sig_hat'] < 1e-4
    assert np.max(out['w'][bad]) < 1e-5
    assert np.median(out['w'][good]) > .99
    assert np.median(out['r_px'][good]) < 1e-5
    for key in ('Rs', 'ts', 'X', 'f', 'r_px', 'r_w', 'w', 'cost'):
        assert np.isfinite(out[key]).all()


@pytest.mark.parametrize('noise', [0., 1e-10])
def test_local_initializer_near_zero_scale_with_outliers(noise):
    R, t, f, X, obs = _scene(n=60)
    uv = _pixels(R, t, f, X)
    uv += np.random.default_rng(607).normal(0, noise, uv.shape)
    uv[4, 1] += [30., -40.]
    uv[20, 2] += [-40., 30.]
    tracks = dict(uv=uv, c=np.ones((len(X), 3)), valid=np.ones((len(X), 3), bool))
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        Rout, tout = local_bundle_adjust(R, t, f, tracks, X, np.ones(3, bool), 0, np.ones(len(X), bool))
    assert np.isfinite(Rout).all() and np.isfinite(tout).all()
    np.testing.assert_allclose(Rout, R, atol=1e-6, rtol=0.)
    np.testing.assert_allclose(tout, t, atol=1e-5, rtol=0.)
