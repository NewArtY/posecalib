"""Bundle adjustment over keypoint observations.

:func:`bundle_adjust` solves one weighted least-squares problem with
``scipy.optimize.least_squares`` (trust-region reflective, LSMR, analytic sparse
Jacobian). :func:`refine` wraps it in iteratively reweighted least squares (IRLS)
to obtain a Huber or Cauchy M-estimate, optionally with a constant-segment-length
term and focal-length refinement.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .geometry import left_jacobian_so3, reproject, rotation_angle_deg, skew
from .robust import robust_sigma, robust_weight


def _observed_segments(bones, L, bone_w=0.0):
    """Compact finite, observed length parameters; retain full output indexing.

    No length is inferred for an unobserved segment. Rows without a finite
    initial length are omitted together with that nuisance parameter.
    """
    if bones is None:
        return None, None, bone_w, None, np.empty(0, dtype=int)
    L = np.asarray(L, float)
    bi, bj, bb = (np.asarray(v, dtype=int) for v in bones)
    keep = np.isfinite(L[bb])
    active, remapped = np.unique(bb[keep], return_inverse=True)
    full_L = np.full(L.shape, np.nan)
    bw = np.asarray(bone_w, float)
    if bw.ndim:
        bw = np.broadcast_to(bw, bb.shape)[keep]
    if not len(active):
        return None, None, bw, full_L, active
    return (bi[keep], bj[keep], remapped), L[active].copy(), bw, full_L, active


def bundle_adjust(Rs0, ts0, X0, f0, ci, pi, uv, sw, *, bones=None, bone_w=0.0, L0=None,
                  refine_focal=False, max_nfev=200, jac_mode='analytic'):
    """Weighted bundle adjustment.

    Parameters are the rotation vectors and translations of cameras ``1..M-1``, the
    3D points, optionally the segment lengths ``L`` and the focal lengths. Camera 0
    is fixed (``R = I, t = 0``); the scale is fixed by freezing the largest component
    of the translation of the camera with the longest baseline (gauge fixing).

    Args:
        Rs0, ts0, X0, f0: initial cameras ``(M,3,3)``, ``(M,3)``, points ``(P,3)``, focals ``(M,)``.
        ci, pi: camera and point index of each observation ``(N,)``.
        uv: ``(N,2)`` centred pixel observations.
        sw: ``(N,)`` square roots of observation weights.
        bones: optional ``(idx_i, idx_j, segment_id)`` arrays of point pairs.
        bone_w: scalar or per-pair weight of the segment term ``bone_w * (|Xi - Xj| - L)``.
        L0: initial segment lengths. Only finite lengths with observation pairs
            enter the solve; omitted segments are returned as NaN in the full array.
        refine_focal: also estimate one focal length per camera.
        jac_mode: ``'analytic'`` or ``'fd'`` (finite differences with the same sparsity; slower).
    """
    bones, L0, bone_w, full_L, active = _observed_segments(bones, L0, bone_w)
    M = len(Rs0)
    P = len(X0)
    cam_full = np.concatenate([Rotation.from_matrix(Rs0).as_rotvec(), ts0], axis=1)
    free = np.ones((M, 6), bool)
    free[0] = False
    kf = 1 + int(np.argmax(np.linalg.norm(ts0[1:], axis=1)))
    mf = int(np.argmax(np.abs(ts0[kf])))
    free[kf, 3 + mf] = False
    n_cam = int(free.sum())
    colmap = -np.ones((M, 6), int)
    colmap[free] = np.arange(n_cam)
    nb = len(L0) if bones is not None else 0
    nf = M if refine_focal else 0
    c_pts, c_L, c_f = n_cam, n_cam + 3 * P, n_cam + 3 * P + nb
    n_par = c_f + nf
    x0 = np.concatenate([cam_full[free], X0.ravel()] + ([L0] if nb else []) + ([f0] if nf else []))
    N = len(ci)
    order = np.argsort(ci, kind='stable')
    ci, pi, uv, sw = ci[order], pi[order], uv[order], sw[order]
    cam_slices = [np.flatnonzero(ci == k) for k in range(M)]

    def unpack(x):
        cam = cam_full.copy()
        cam[free] = x[:n_cam]
        X = x[c_pts:c_L].reshape(P, 3)
        L = x[c_L:c_f] if nb else None
        f = x[c_f:c_f + nf] if nf else f0
        return cam, X, L, f

    def fun(x):
        cam, X, L, f = unpack(x)
        R = Rotation.from_rotvec(cam[:, :3]).as_matrix()
        out = np.empty((N, 2))
        for k in range(M):
            s = cam_slices[k]
            Xc = X[pi[s]] @ R[k].T + cam[k, 3:]
            out[s] = f[k] * Xc[:, :2] / Xc[:, 2:3]
        r = ((out - uv) * sw[:, None]).ravel()
        if nb:
            d = np.linalg.norm(X[bones[0]] - X[bones[1]], axis=1)
            r = np.concatenate([r, bone_w * (d - L[bones[2]])])
        return r

    # Jacobian sparsity structure
    cols = [colmap[ci], c_pts + 3 * pi[:, None] + np.arange(3)]
    if nf:
        cols.append(c_f + ci[:, None])
    cols = np.concatenate(cols, axis=1)
    rows = np.broadcast_to((2 * np.arange(N))[:, None, None] + np.array([[0], [1]]), (N, 2, cols.shape[1]))
    cc = np.broadcast_to(cols[:, None, :], rows.shape)
    keep = cc >= 0
    rr_all, cc_all = [rows[keep]], [cc[keep]]
    n_res = 2 * N
    if nb:
        Nb = len(bones[0])
        rb = n_res + np.arange(Nb)
        cb = np.concatenate([c_pts + 3 * bones[0][:, None] + np.arange(3),
                             c_pts + 3 * bones[1][:, None] + np.arange(3),
                             (c_L + bones[2])[:, None]], axis=1)
        rr_all.append(np.repeat(rb, cb.shape[1]))
        cc_all.append(cb.ravel())
        n_res += Nb
    rr_all, cc_all = np.concatenate(rr_all), np.concatenate(cc_all)
    S = coo_matrix((np.ones(len(rr_all)), (rr_all, cc_all)), shape=(n_res, n_par)).tocsr()
    swN = sw[:, None, None]

    def jac(x):
        cam, X, L, f = unpack(x)
        rv = cam[:, :3]
        R = Rotation.from_rotvec(rv).as_matrix()
        Jl = left_jacobian_so3(rv)
        RX = np.einsum('nij,nj->ni', R[ci], X[pi])
        Xc = RX + cam[ci, 3:]
        z = Xc[:, 2]
        fz = f[ci] / z
        A = np.zeros((N, 2, 3))
        A[:, 0, 0] = fz
        A[:, 1, 1] = fz
        A[:, 0, 2] = -fz * Xc[:, 0] / z
        A[:, 1, 2] = -fz * Xc[:, 1] / z
        A *= swN
        Jr = A @ (-skew(RX) @ Jl[ci])                  # d(RX)/dr = -[RX]x J_l(r)
        blocks = [Jr, A, A @ R[ci]]
        if nf:
            blocks.append((sw[:, None] * Xc[:, :2] / z[:, None])[..., None])
        vals = np.concatenate(blocks, axis=2)[keep]
        data = [vals]
        if nb:
            dv = X[bones[0]] - X[bones[1]]
            u = dv / np.linalg.norm(dv, axis=1, keepdims=True)
            bwv = np.broadcast_to(np.asarray(bone_w, float), (len(u),))[:, None]
            data.append(np.concatenate([bwv * u, -bwv * u, -bwv], axis=1).ravel())
        return coo_matrix((np.concatenate(data), (rr_all, cc_all)), shape=(n_res, n_par)).tocsr()

    t0 = time.perf_counter()
    kw = dict(method='trf', x_scale='jac', tr_solver='lsmr', max_nfev=max_nfev,
              ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=0)
    if jac_mode == 'fd':
        res = least_squares(fun, x0, jac_sparsity=S, **kw)
    else:
        res = least_squares(fun, x0, jac=jac, **kw)
    dt = time.perf_counter() - t0
    cam, X, L, f = unpack(res.x)
    Rs = Rotation.from_rotvec(cam[:, :3]).as_matrix()
    r = fun(res.x)[:2 * N].reshape(N, 2)
    r_px = np.empty(N)
    r_px[order] = np.linalg.norm(r / sw[:, None], axis=1)        # unweighted error, px
    rw = np.empty((N, 2))
    rw[order] = r
    if full_L is not None:
        full_L[active] = L if L is not None else []
    return dict(Rs=Rs, ts=cam[:, 3:].copy(), X=X, L=full_L, f=np.array(f, float), nfev=res.nfev,
                status=res.status, time=dt, r_px=r_px, r_w=rw, cost=res.cost)


def refine(Rs, ts, X, f, obs, *, L=None, weighted=True, loss='cauchy', use_segments=True,
           refine_focal=False, robust_k=2.5, segment_sigma=0.01, max_nfev=60,
           irls_max_iter=20, irls_scale_iters=3, tol_deg=1e-3, tol_m=1e-4):
    """M-estimate of cameras and points by IRLS around :func:`bundle_adjust`.

    The residual scale is ``sigma = 1.4826 * MAD`` of the weighted residuals, updated
    during the first ``irls_scale_iters`` iterations and then frozen; the loss
    threshold is ``k = robust_k * sigma``. A numerical floor of sqrt(machine
    epsilon) times the median confidence-weighted coordinate magnitude (at least
    one pixel before weighting) keeps exact fits finite without adding detector
    noise or changing confidence units. With ``loss='linear'`` a single weighted
    least-squares solve is performed.

    The segment term ``(sigma / segment_sigma) * (|Xi - Xj| - L_b)`` equates a
    segment-length deviation of ``segment_sigma`` metres to ``sigma`` pixels, so the
    reconstruction must be roughly metric beforehand.

    Args:
        Rs, ts, X, f: initial cameras, points ``(P,3)`` and focal lengths.
        obs: dict with ``ci, pi, uv, c`` (observation arrays) and ``bones``
            (``(idx_i, idx_j, segment_id)``).
        L: initial segment lengths (required if ``use_segments``).

    Returns:
        dict with ``Rs, ts, X, L, f``, residuals ``r_px`` (unweighted, px), robust
        weights ``w`` and the scale ``sig_hat``.
    """
    sw0 = obs['c'] if weighted else np.ones(len(obs['c']))
    bones, L, _, full_L, active = _observed_segments(obs['bones'] if use_segments else None, L)
    f = np.array(f, float)
    sig_floor = (np.sqrt(np.finfo(float).eps)
                 * np.median(np.maximum(1.0, np.abs(obs['uv'])) * sw0[:, None]))

    def resid(Rs, ts, X, f, L):
        uvp, _, _ = reproject(Rs, ts, f, X)
        r = (uvp[obs['pi'], obs['ci']] - obs['uv']) * sw0[:, None]
        rb = None if bones is None else np.linalg.norm(X[bones[0]] - X[bones[1]], axis=1) - L[bones[2]]
        return r, rb

    r, rb = resid(Rs, ts, X, f, L)
    sig = robust_sigma(r, floor=sig_floor)
    n_outer = 1 if loss == 'linear' else irls_max_iter
    total_t, total_nfev, it = 0.0, 0, 0
    w_obs = np.ones(len(r))
    out = None
    for it in range(n_outer):
        k = robust_k * sig
        w_obs = robust_weight(np.linalg.norm(r, axis=1) / k, loss)
        bw = 0.0
        if bones is not None:
            rbn = np.abs(rb) * sig / segment_sigma
            bw = (sig / segment_sigma) * np.sqrt(robust_weight(rbn / k, loss))
        out = bundle_adjust(Rs, ts, X, f, obs['ci'], obs['pi'], obs['uv'], sw0 * np.sqrt(w_obs),
                            bones=bones, bone_w=bw, L0=L, refine_focal=refine_focal, max_nfev=max_nfev)
        dR = max(rotation_angle_deg(out['Rs'][m], Rs[m]) for m in range(len(Rs)))
        dt_ = float(np.max(np.abs(out['ts'] - ts)))
        Rs, ts, X, L, f = out['Rs'], out['ts'], out['X'], out['L'], out['f']
        total_t += out['time']
        total_nfev += out['nfev']
        r, rb = resid(Rs, ts, X, f, L)
        if it < irls_scale_iters:
            sig = robust_sigma(r, floor=sig_floor)
        if it > 0 and dR < tol_deg and dt_ < tol_m:
            break
    out['time'] = total_t
    out['nfev'] = total_nfev
    out['n_irls'] = it + 1
    out['sig_hat'] = sig
    out['r_px'] = np.linalg.norm(r, axis=1) / sw0
    out['w'] = robust_weight(np.linalg.norm(r, axis=1) / (robust_k * sig), loss)
    if full_L is not None:
        full_L[active] = L if L is not None else []
    out['L'] = full_L
    return out
