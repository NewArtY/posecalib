"""Multi-view geometry used by the calibrator.

Conventions
-----------
* A camera maps a world point ``X`` to camera coordinates ``Xc = R @ X + t``;
  the camera looks along ``+z``, ``x`` points right, ``y`` points down.
* Image coordinates are **pixels relative to the principal point** (``u - cx``,
  ``v - cy``) with square pixels; a camera is described by a single focal length
  ``f`` in pixels. Normalised coordinates are ``uv / f``.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def skew(v):
    """Batched cross-product matrix ``[v]x``."""
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2], S[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    S[..., 1, 0], S[..., 2, 0], S[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return S


def left_jacobian_so3(r):
    """Batched left Jacobian of SO(3): ``Exp(r + d) ~ Exp(J_l(r) d) Exp(r)``."""
    th = np.linalg.norm(r, axis=-1)[..., None, None]
    K = skew(r)
    K2 = K @ K
    small = th < 1e-6
    ths = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(ths)) / ths ** 2)
    b = np.where(small, 1.0 / 6.0, (ths - np.sin(ths)) / ths ** 3)
    return np.eye(3) + a * K + b * K2


def rotation_angle_deg(Ra, Rb) -> float:
    """Geodesic angle between two rotation matrices, degrees."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def project_to_so3(M):
    """Closest rotation matrix to ``M`` (Frobenius norm)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def umeyama(src, dst, with_scale=True):
    """Similarity ``dst ~ s R src + t`` in the least-squares sense (Umeyama, 1991)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    S = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(S)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(D) @ E) / ((xs ** 2).sum() / len(src)) if with_scale else 1.0
    t = md - s * R @ ms
    return s, R, t


def camera_centers(Rs, ts):
    """Camera centres ``C = -R^T t`` for stacked extrinsics."""
    return -np.einsum('mji,mj->mi', Rs, ts)


def reproject(Rs, ts, fs, X):
    """Project points ``X (T,3)`` into all cameras.

    Returns ``uv (T,M,2)`` (principal-point-centred pixels), depth ``z (T,M)`` and
    camera-frame points ``Xc (T,M,3)``.
    """
    Xc = np.einsum('mij,tj->tmi', Rs, X) + ts[None]
    z = Xc[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = fs[None, :, None] * Xc[..., :2] / z[..., None]
    return uv, z, Xc


# -----------------------------------------------------------------------------
# two-view geometry
# -----------------------------------------------------------------------------
def eight_point(x1, x2):
    """Normalised 8-point algorithm, batched: ``x1, x2 (B,n,2)`` normalised -> ``E (B,3,3)``."""
    B = x1.shape[0]

    def hartley(x):
        m = x.mean(axis=1, keepdims=True)
        d = np.linalg.norm(x - m, axis=-1).mean(axis=1)
        s = math.sqrt(2) / np.maximum(d, 1e-12)
        T = np.zeros((B, 3, 3))
        T[:, 0, 0] = s
        T[:, 1, 1] = s
        T[:, 0, 2] = -s * m[:, 0, 0]
        T[:, 1, 2] = -s * m[:, 0, 1]
        T[:, 2, 2] = 1.0
        return (x - m) * s[:, None, None], T

    a, T1 = hartley(x1)
    b, T2 = hartley(x2)
    o = np.ones_like(a[..., 0])
    A = np.stack([b[..., 0] * a[..., 0], b[..., 0] * a[..., 1], b[..., 0],
                  b[..., 1] * a[..., 0], b[..., 1] * a[..., 1], b[..., 1],
                  a[..., 0], a[..., 1], o], -1)
    _, _, Vt = np.linalg.svd(A)
    Fn = Vt[:, -1, :].reshape(B, 3, 3)
    E = T2.transpose(0, 2, 1) @ Fn @ T1
    U, _, Vt = np.linalg.svd(E)
    return (U * np.array([1.0, 1.0, 0.0])) @ Vt


def sampson_sq(E, x1, x2):
    """Squared Sampson distance (normalised units): ``E (B,3,3)``, ``x (n,2)`` -> ``(B,n)``."""
    X1 = np.column_stack([x1, np.ones(len(x1))])
    X2 = np.column_stack([x2, np.ones(len(x2))])
    Ex1 = np.einsum('bij,nj->bni', E, X1)
    Etx2 = np.einsum('bji,nj->bni', E, X2)
    num = np.einsum('ni,bni->bn', X2, Ex1)
    den = Ex1[..., 0] ** 2 + Ex1[..., 1] ** 2 + Etx2[..., 0] ** 2 + Etx2[..., 1] ** 2
    return num ** 2 / np.maximum(den, 1e-30)


def dedup_mask(feats, cell):
    """Keep one representative per grid cell.

    Joints that stay still produce clusters of nearly identical correspondences on
    which median-based estimators degenerate; thinning removes them.
    """
    qz = np.floor(feats / cell).astype(np.int64)
    _, idx = np.unique(qz, axis=0, return_index=True)
    m = np.zeros(len(feats), bool)
    m[idx] = True
    return m


def homography_median(x1, x2, rng, fpx=1200.0, n_iter=400):
    """Degeneracy test for a camera pair.

    Fits a homography by LMedS (4-point DLT on thinned correspondences) and returns the
    median squared transfer error ``|x2 - H x1|^2`` (normalised units). If a homography
    explains the correspondences about as well as an essential matrix, the points are
    close to a plane and the pair is a poor starting point.
    """
    i_d = np.flatnonzero(dedup_mask(np.hstack([x1, x2]), 8.0 / fpx))
    a, b = x1[i_d], x2[i_d]
    idx = np.argpartition(rng.random((n_iter, len(i_d))), 4, axis=1)[:, :4]
    p, q = a[idx], b[idx]
    o, z = np.ones(p.shape[:2]), np.zeros(p.shape[:2])
    A = np.empty((n_iter, 8, 9))
    A[:, 0::2] = np.stack([p[..., 0], p[..., 1], o, z, z, z,
                           -q[..., 0] * p[..., 0], -q[..., 0] * p[..., 1], -q[..., 0]], -1)
    A[:, 1::2] = np.stack([z, z, z, p[..., 0], p[..., 1], o,
                           -q[..., 1] * p[..., 0], -q[..., 1] * p[..., 1], -q[..., 1]], -1)
    H = np.linalg.svd(A)[2][:, -1].reshape(n_iter, 3, 3)
    ah = np.column_stack([a, np.ones(len(a))])
    y = np.einsum('bij,nj->bni', H, ah)
    with np.errstate(divide='ignore', invalid='ignore'):
        e2 = np.sum((y[..., :2] / y[..., 2:3] - b) ** 2, -1)
    e2[~np.isfinite(e2)] = np.inf
    return float(np.min(np.median(e2, axis=1)))


def triangulate_dlt(Rs, ts, xn, w):
    """Weighted linear triangulation, batched: ``xn (T,M,2)`` normalised, ``w (T,M)`` (0 = unused view)."""
    P = np.concatenate([Rs, ts[:, :, None]], axis=2)
    T, M = w.shape
    A = np.empty((T, 2 * M, 4))
    A[:, 0::2] = w[..., None] * (xn[..., 0, None] * P[None, :, 2, :] - P[None, :, 0, :])
    A[:, 1::2] = w[..., None] * (xn[..., 1, None] * P[None, :, 2, :] - P[None, :, 1, :])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        return Xh[:, :3] / Xh[:, 3:4]


def pose_from_essential(E, x1, x2):
    """Decompose ``E`` into ``(R, t)`` choosing the solution with most points in front of both cameras.

    Returns ``(n_front, R, t, X, front_mask)``.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    xn = np.stack([x1, x2], axis=1)
    w = np.ones((len(x1), 2))
    best = None
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            X = triangulate_dlt(np.stack([np.eye(3), R]), np.stack([np.zeros(3), t]), xn, w)
            front = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
            if best is None or front.sum() > best[0]:
                best = (int(front.sum()), R, t, X, front)
    return best


def refine_relative_pose(R, t, x1, x2, sig, max_nfev=100):
    """Non-linear refinement of a relative pose on Sampson distances with a Cauchy loss of scale ``sig``."""
    def fun(p):
        Rm = Rotation.from_rotvec(p[:3]).as_matrix()
        tv = p[3:] / np.linalg.norm(p[3:])
        return np.sqrt(sampson_sq((skew(tv) @ Rm)[None], x1, x2)[0])

    p0 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), t / np.linalg.norm(t)])
    res = least_squares(fun, p0, loss='cauchy', f_scale=sig, method='trf', x_scale='jac', max_nfev=max_nfev)
    return Rotation.from_rotvec(res.x[:3]).as_matrix(), res.x[3:] / np.linalg.norm(res.x[3:])


def estimate_relative_pose(x1, x2, rng, fpx=1200.0, n_iter=1000, n_lo=8):
    """Robust relative pose of a camera pair (``|t| = 1``).

    LMedS over 8-point samples drawn from thinned correspondences, then local
    optimisation of the ``n_lo`` best hypotheses: decomposition of E (cheirality) and a
    two-pass refinement on Sampson distances with a Cauchy loss, re-estimating the
    noise scale. The hypothesis with the smallest median error wins.

    Returns ``R, t, inlier_mask, sigma`` (sigma in normalised units).
    """
    i_d = np.flatnonzero(dedup_mask(np.hstack([x1, x2]), 8.0 / fpx))
    if len(i_d) < 16:
        raise RuntimeError('too few distinct correspondences to estimate E')
    idx = i_d[np.argpartition(rng.random((n_iter, len(i_d))), 8, axis=1)[:, :8]]
    E = eight_point(x1[idx], x2[idx])
    xd1, xd2 = x1[i_d], x2[i_d]
    med = np.concatenate([np.median(sampson_sq(E[i:i + 100], xd1, xd2), axis=1) for i in range(0, n_iter, 100)])
    med[~np.isfinite(med)] = np.inf
    floor = 0.3 / fpx
    cand = []
    for b in np.argsort(med)[:n_lo]:               # step 1: short refinement of every candidate
        if not np.isfinite(med[b]):
            continue
        sig = max(1.4826 * math.sqrt(med[b]), floor)
        inl = sampson_sq(E[b:b + 1], x1, x2)[0] < (2.5 * sig) ** 2
        if inl.sum() < 12:
            continue
        _, R, t, _, _ = pose_from_essential(E[b], x1[inl], x2[inl])
        R, t = refine_relative_pose(R, t, x1, x2, sig, max_nfev=25)
        mb = float(np.median(sampson_sq((skew(t) @ R)[None], x1, x2)[0]))
        cand.append((mb, R, t))
    if not cand:
        raise RuntimeError('LMedS found no essential-matrix model')
    best = None
    for mb, R, t in sorted(cand, key=lambda z: z[0])[:2]:     # step 2: full refinement of the two best
        sig = max(1.4826 * math.sqrt(mb), floor)
        for _ in range(2):
            R, t = refine_relative_pose(R, t, x1, x2, sig)
            mb = float(np.median(sampson_sq((skew(t) @ R)[None], x1, x2)[0]))
            sig = max(1.4826 * math.sqrt(mb), floor)
        if best is None or mb < best[0]:
            best = (mb, R, t, sig)
    mb, R, t, sig = best
    inl = sampson_sq((skew(t) @ R)[None], x1, x2)[0] < (2.5 * sig) ** 2
    return R, t, inl, sig


# -----------------------------------------------------------------------------
# robust multi-view triangulation
# -----------------------------------------------------------------------------
def triangulate_robust(Rs, ts, fs, uv, c, valid, cam_mask=None, sig_hat=None, k_trunc=3.0, n_irls=6):
    """Robust triangulation of keypoint tracks.

    For each track (one keypoint in one frame) every subset of 2..M views is
    triangulated and scored with a truncated quadratic cost over all views; the best
    start is then refined by IRLS with a Cauchy loss. View weights are proportional
    to the detector confidence.

    Args:
        Rs, ts, fs: camera rotations ``(M,3,3)``, translations ``(M,3)``, focal lengths ``(M,)``.
        uv: ``(T,M,2)`` principal-point-centred pixel coordinates.
        c: ``(T,M)`` confidences; ``valid``: ``(T,M)`` mask of usable observations.
        cam_mask: optional ``(M,)`` mask of cameras to use.
        sig_hat: residual scale in pixels; estimated from the data if ``None``.

    Returns:
        ``X (T,3)``, ``ok (T,)``, weighted residuals ``(T,M)``, inlier mask ``(T,M)``, ``sig_hat``.
    """
    T_all, M = c.shape
    if cam_mask is None:
        cam_mask = np.ones(M, bool)
    use_all = valid & cam_mask[None]
    nv_all = use_all.sum(1)
    sel = np.flatnonzero(nv_all >= 2)                  # only tracks seen in >= 2 views
    X_out = np.full((T_all, 3), np.nan)
    ok_out = np.zeros(T_all, bool)
    r_out = np.full((T_all, M), np.nan)
    inl_out = np.zeros((T_all, M), bool)
    if len(sel) == 0:
        return X_out, ok_out, r_out, inl_out, (3.0 if sig_hat is None else sig_hat)
    use, c_, uv_ = use_all[sel], np.where(use_all[sel], c[sel], 0.0), uv[sel]
    uv_ = np.where(use[..., None], uv_, 0.0)          # placeholders for unused views
    nv = nv_all[sel]
    T = len(sel)
    xn = uv_ / fs[None, :, None]
    cams_on = np.flatnonzero(cam_mask)

    def dlt2(wmask):
        w1 = c_ * wmask
        X1 = triangulate_dlt(Rs, ts, xn, w1)
        X1 = np.where(np.isfinite(X1), X1, 0.0)
        _, z, _ = reproject(Rs, ts, fs, X1)
        w2 = w1 * fs[None] / np.maximum(np.abs(np.nan_to_num(z)), 0.1)
        X2 = triangulate_dlt(Rs, ts, xn, w2)
        bad = ~np.isfinite(X2).all(1)
        X2[bad] = X1[bad]
        return X2

    def wres(X):
        with np.errstate(all='ignore'):
            uvp, z, _ = reproject(Rs, ts, fs, X)
            r = c_ * np.linalg.norm(uvp - uv_, axis=-1)
        r = np.where(use, r, 0.0)
        return r, z

    X_all = dlt2(use.astype(float))
    r_all, _ = wres(X_all)
    if sig_hat is None:
        rr = r_all[use & (nv[:, None] >= 3)]
        rr = rr[np.isfinite(rr)]
        if len(rr) > 30:
            sig_hat = float(np.median(rr) / 1.1774 * 1.3)
        else:                                          # two-view tracks only: residuals shrink ~2x
            rr = r_all[use]
            rr = rr[np.isfinite(rr)]
            sig_hat = float(np.median(rr) / 1.1774 * 2.0) if len(rr) > 10 else 3.0
        sig_hat = max(sig_hat, 0.3)
    Tt = k_trunc * sig_hat
    best_cost = np.full(T, np.inf)
    best_X = X_all.copy()
    for r_ in range(len(cams_on), 1, -1):
        for S in combinations(cams_on, r_):
            S = list(S)
            feas = use[:, S].all(1)
            if not feas.any():
                continue
            sm = np.zeros(M)
            sm[S] = 1.0
            X = dlt2(use * sm)
            r, z = wres(X)
            cost = np.where(use, np.minimum(r ** 2, Tt ** 2), 0.0).sum(1)
            neg = ((~(z > 0)) & use & (sm[None] > 0)).any(1)
            cost[neg | ~feas | ~np.isfinite(cost)] = np.inf
            better = cost < best_cost - 1e-9
            best_cost[better] = cost[better]
            best_X[better] = X[better]
    X = best_X.copy()
    s = 2.5 * sig_hat
    eye = np.eye(3)[None]
    for _ in range(n_irls):
        with np.errstate(all='ignore'):
            uvp, z, Xc = reproject(Rs, ts, fs, X)
            e = uvp - uv_
            r = c_ * np.linalg.norm(e, axis=-1)
            good = use & (z > 0.05) & np.isfinite(r)
            wr = np.where(good, c_ ** 2 / (1 + (r / s) ** 2), 0.0)
            zz = np.where(good, z, 1.0)
            Jm = fs[None, :, None, None] / zz[..., None, None] * (
                Rs[None, :, :2, :] - (Xc[..., :2, None] / zz[..., None, None]) * Rs[None, :, 2:3, :])
        Jm = np.where(good[..., None, None], Jm, 0.0)
        e = np.where(good[..., None], e, 0.0)
        H = np.einsum('tm,tmki,tmkj->tij', wr, Jm, Jm) + 1e-6 * eye
        g = np.einsum('tm,tmki,tmk->ti', wr, Jm, e)
        dX = np.linalg.solve(H, g[..., None])[..., 0]
        dX = np.where(np.isfinite(dX), dX, 0.0)
        X = X - dX
    r, z = wres(X)
    inl = use & (r < Tt) & (z > 0)
    ok = np.isfinite(X).all(1) & (inl.sum(1) >= 2) & ~((~(z > 0)) & use).any(1)
    X_out[sel], ok_out[sel], r_out[sel], inl_out[sel] = X, ok, np.where(use, r, np.nan), inl
    return X_out, ok_out, r_out, inl_out, sig_hat


# -----------------------------------------------------------------------------
# PnP
# -----------------------------------------------------------------------------
def pnp_dlt(X, xn):
    """Calibrated DLT-PnP, batched: ``X (B,n,3)``, normalised ``xn (B,n,2)`` -> ``R (B,3,3)``, ``t (B,3)``."""
    B, n = X.shape[:2]
    m = X.mean(1, keepdims=True)
    s = math.sqrt(3) / np.maximum(np.linalg.norm(X - m, axis=-1).mean(1), 1e-12)
    Xh = np.concatenate([(X - m) * s[:, None, None], np.ones((B, n, 1))], -1)
    Z = np.zeros_like(Xh)
    Au = np.concatenate([-Xh, Z, xn[..., 0:1] * Xh], -1)
    Av = np.concatenate([Z, -Xh, xn[..., 1:2] * Xh], -1)
    A = np.empty((B, 2 * n, 12))
    A[:, 0::2], A[:, 1::2] = Au, Av
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[:, -1].reshape(B, 3, 4)
    Tm = np.zeros((B, 4, 4))
    Tm[:, :3, :3] = s[:, None, None] * np.eye(3)
    Tm[:, :3, 3] = -s[:, None] * m[:, 0]
    Tm[:, 3, 3] = 1.0
    P = Pn @ Tm
    d = np.linalg.det(P[:, :, :3])
    P[d < 0] *= -1
    U, S, Vt = np.linalg.svd(P[:, :, :3])
    R = U @ Vt
    t = P[:, :, 3] / S.mean(1)[:, None]
    return R, t


def lmeds_pnp(X, uv, f, rng, n_iter=300):
    """Robust PnP: 6-point DLT hypotheses scored by LMedS on thinned 2D-3D matches,
    iterative inlier re-estimation, then non-linear refinement with a Cauchy loss.

    Args:
        X: ``(n,3)`` 3D points; uv: ``(n,2)`` centred pixels; f: focal length.

    Returns ``R, t, inlier_mask, sigma_px``.
    """
    xn = uv / f
    i_d = np.flatnonzero(dedup_mask(np.hstack([X / 0.03, uv / 8.0]), 1.0))
    if len(i_d) < 12:
        raise RuntimeError('too few distinct 2D-3D correspondences for PnP')
    idx = i_d[np.argpartition(rng.random((n_iter, len(i_d))), 6, axis=1)[:, :6]]
    R, t = pnp_dlt(X[idx], xn[idx])
    Xs6 = X[idx] - X[idx].mean(1, keepdims=True)
    sv = np.linalg.svd(Xs6, compute_uv=False)
    degen = sv[:, 2] < 0.1 * sv[:, 0]                  # DLT-PnP degenerates on coplanar samples

    def err2(R_, t_, Xs, uvs):
        Xc = np.einsum('...ij,nj->...ni', R_, Xs) + t_[..., None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            e2 = np.sum((f * Xc[..., :2] / Xc[..., 2:3] - uvs) ** 2, -1)
        e2[~(Xc[..., 2] > 0)] = np.inf
        e2[~np.isfinite(e2)] = np.inf
        return e2

    med = np.median(err2(R, t, X[i_d], uv[i_d]), axis=1)
    med[degen] = np.inf
    b = int(np.argmin(med))
    # median of a squared 2D Gaussian error = 2 ln2 sigma^2 -> sigma = sqrt(med)/1.1774 (+ small-sample factor)
    sig = max((1 + 5.0 / (len(i_d) - 6)) * math.sqrt(med[b]) / 1.1774, 0.3)
    Rb, tb = R[b], t[b]
    best_med = float(np.median(err2(Rb, tb, X, uv)))
    for _ in range(3):                       # accept a re-fit only while the median decreases
        inl = err2(Rb, tb, X, uv) < (3.0 * sig) ** 2
        if inl.sum() < 8:
            break
        R1, t1 = pnp_dlt(X[inl][None], xn[inl][None])
        mn = float(np.median(err2(R1[0], t1[0], X, uv)))
        if not np.isfinite(mn) or mn >= best_med:
            break
        Rb, tb, best_med = R1[0], t1[0], mn
        sig = max(math.sqrt(mn) / 1.1774, 0.3)
    x0 = np.concatenate([Rotation.from_matrix(Rb).as_rotvec(), tb])

    def fun(p):
        Rm = Rotation.from_rotvec(p[:3]).as_matrix()
        Xc = X @ Rm.T + p[3:]
        return (f * Xc[:, :2] / Xc[:, 2:3] - uv).ravel()

    res = least_squares(fun, x0, loss='cauchy', f_scale=max(2.0 * sig, 1.0), method='trf')
    Rr = Rotation.from_rotvec(res.x[:3]).as_matrix()
    inl = err2(Rr, res.x[3:], X, uv) < (3.0 * sig) ** 2
    return Rr, res.x[3:], inl, sig
