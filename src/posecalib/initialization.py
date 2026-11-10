"""Initial estimate of the extrinsics (incremental structure-from-motion).

Keypoint semantics give the cross-view correspondences directly: keypoint ``j`` in
frame ``f`` is the same 3D point in every camera. The pipeline is

1. relative pose of every camera pair from confident correspondences
   (8-point + LMedS + local optimisation, homography degeneracy test);
2. starting pair = maximum of ``n_inliers * sin(median triangulation angle)``;
3. robust triangulation, then PnP for the camera with most 2D-3D matches,
   re-triangulation and a local bundle adjustment, until all cameras are placed.

Tracks are the flattened ``(frame, keypoint)`` pairs: arrays of shape ``(F*J, M, ...)``.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .bundle import bundle_adjust
from .geometry import (estimate_relative_pose, homography_median, lmeds_pnp, reproject, sampson_sq,
                       skew, triangulate_dlt, triangulate_robust)
from .robust import robust_sigma, robust_weight

DEGEN_RATIO = 25.0
"""A pair is treated as near-degenerate (almost planar points) if the median homography
transfer error is less than ``DEGEN_RATIO`` times the median Sampson error of E."""


def local_bundle_adjust(Rs, ts, f, tracks, X, registered, ref, ok, n_outer=3):
    """Bundle adjustment of the already registered cameras (as in incremental SfM).

    Confidence weights, Huber loss via IRLS; camera ``ref`` is the gauge (``R = I, t = 0``).
    """
    cams = [ref] + [k for k in np.flatnonzero(registered) if k != ref]
    use = tracks['valid'][:, cams] & ok[:, None]
    pts = np.flatnonzero(use.sum(1) >= 2)
    if len(pts) < 50:
        return Rs, ts
    pidx = -np.ones(len(ok), int)
    pidx[pts] = np.arange(len(pts))
    ti, cj = np.nonzero(use[pts])
    ti = pts[ti]
    ci, pi = cj, pidx[ti]
    uv = tracks['uv'][ti, np.array(cams)[cj]]
    sw0 = tracks['c'][ti, np.array(cams)[cj]]
    X0 = X[pts]
    Rl, tl = Rs[cams].copy(), ts[cams].copy()
    uvp, _, _ = reproject(Rl, tl, f[cams], X0)
    r = (uvp[pi, ci] - uv) * sw0[:, None]
    sig_floor = (np.sqrt(np.finfo(float).eps)
                 * np.median(np.maximum(1.0, np.abs(uv)) * sw0[:, None]))
    sig = robust_sigma(r, floor=sig_floor)
    Xl = X0
    for it in range(n_outer):
        w = robust_weight(np.linalg.norm(r, axis=1) / (2.5 * sig), 'huber')
        out = bundle_adjust(Rl, tl, Xl, f[cams], ci, pi, uv, sw0 * np.sqrt(w), max_nfev=30)
        Rl, tl, Xl = out['Rs'], out['ts'], out['X']
        uvp, _, _ = reproject(Rl, tl, f[cams], Xl)
        r = (uvp[pi, ci] - uv) * sw0[:, None]
        if it < 2:
            sig = robust_sigma(r, floor=sig_floor)
    Rs, ts = Rs.copy(), ts.copy()
    Rs[cams], ts[cams] = Rl, tl
    return Rs, ts


def initialize(tracks, f, rng, init_confidence=0.6, min_matches=50, local_refine=True, max_start_pairs=3):
    """Initial extrinsics and points from keypoint tracks.

    Args:
        tracks: dict with ``uv (T,M,2)`` centred pixels, ``c (T,M)`` confidences and
            ``valid (T,M)``.
        f: ``(M,)`` focal lengths in pixels.
        rng: ``numpy.random.Generator`` for the robust estimators.
        init_confidence: only observations at least this confident are used for the
            pairwise relative poses.

    Returns:
        ``Rs, ts, X, ok, info`` in the frame of camera 0 (``R0 = I, t0 = 0``), arbitrary scale.
    """
    uv, c, valid = tracks['uv'], tracks['c'], tracks['valid']
    T, M = c.shape
    xn = uv / f[None, :, None]
    good = valid & (c >= init_confidence)
    pairs = {}
    for a, b in combinations(range(M), 2):
        m = good[:, a] & good[:, b]
        if m.sum() < min_matches:
            continue
        try:
            R, t, inl, sig = estimate_relative_pose(xn[m, a], xn[m, b], rng, fpx=0.5 * (f[a] + f[b]))
        except RuntimeError:
            continue
        if inl.sum() < 30:
            continue
        medE = float(np.median(sampson_sq((skew(t) @ R)[None], xn[m, a], xn[m, b])[0]))
        medH = homography_median(xn[m, a], xn[m, b], rng, fpx=0.5 * (f[a] + f[b]))
        degen_ratio = medH / max(medE, 1e-30)
        X = triangulate_dlt(np.stack([np.eye(3), R]), np.stack([np.zeros(3), t]),
                            np.stack([xn[m, a][inl], xn[m, b][inl]], axis=1), np.ones((int(inl.sum()), 2)))
        front = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
        if front.sum() < 10:                           # hypothesis puts the points behind a camera
            continue
        C2 = -R.T @ t
        v1, v2 = X[front], X[front] - C2
        cosang = np.sum(v1 * v2, 1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        ang = float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1)))))
        score = inl.sum() * math.sin(math.radians(ang))
        if degen_ratio < DEGEN_RATIO:
            score *= 1e-3                              # near-planar pair: try last
        pairs[(a, b)] = dict(R=R, t=t, n_match=int(m.sum()), n_inliers=int(inl.sum()), angle_deg=ang,
                             score=score, sigma_px=sig * 0.5 * (f[a] + f[b]), degeneracy_ratio=degen_ratio)
    if not pairs:
        raise RuntimeError('no camera pair has enough confident correspondences')

    def grow(a, b):
        Rs = np.tile(np.eye(3), (M, 1, 1))
        ts = np.zeros((M, 3))
        reg = np.zeros(M, bool)
        Rs[b], ts[b] = pairs[(a, b)]['R'], pairs[(a, b)]['t']
        reg[[a, b]] = True
        X, ok, *_ = triangulate_robust(Rs, ts, f, uv, c, valid, cam_mask=reg)
        order = [a, b]
        while not reg.all():
            cand = [(k, int((ok & good[:, k]).sum())) for k in range(M) if not reg[k]]
            k, nk = max(cand, key=lambda z: z[1])
            sel = ok & good[:, k]
            if nk < 30:                                # fall back to all accepted observations
                sel = ok & valid[:, k]
            if sel.sum() < 12:
                raise RuntimeError(f'too few 2D-3D correspondences to place camera {k}')
            Rk, tk, _, _ = lmeds_pnp(X[sel], uv[sel, k], f[k], rng)
            Rs[k], ts[k] = Rk, tk
            reg[k] = True
            order.append(k)
            X, ok, *_ = triangulate_robust(Rs, ts, f, uv, c, valid, cam_mask=reg)
            if local_refine:
                Rs, ts = local_bundle_adjust(Rs, ts, f, tracks, X, reg, a, ok)
                X, ok, *_ = triangulate_robust(Rs, ts, f, uv, c, valid, cam_mask=reg)
        return Rs, ts, X, ok, order

    err = None
    for a, b in sorted(pairs, key=lambda k: -pairs[k]['score'])[:max_start_pairs]:
        try:
            Rs, ts, X, ok, order = grow(a, b)
            break
        except (RuntimeError, np.linalg.LinAlgError) as e:
            err = e
    else:
        raise RuntimeError(f'initialisation failed: {err}')
    # express everything in the frame of camera 0
    R0, t0 = Rs[0].copy(), ts[0].copy()
    Rs_n = Rs @ R0.T
    ts_n = ts - np.einsum('mij,j->mi', Rs_n, t0)
    X_n = X @ R0.T + t0
    return Rs_n, ts_n, X_n, ok, dict(pairs=pairs, start_pair=(a, b), order=order)


def segment_lengths(X, ok, n_frames, n_joints, segments):
    """Median (over frames) length of each segment; NaN if never reconstructed."""
    Xf = X.reshape(n_frames, n_joints, 3)
    okf = ok.reshape(n_frames, n_joints)
    L = np.full(len(segments), np.nan)
    for i, (p, q) in enumerate(segments):
        m = okf[:, p] & okf[:, q]
        if m.sum() > 0:
            L[i] = np.median(np.linalg.norm(Xf[m, p] - Xf[m, q], axis=1))
    return L


def build_observations(tracks, ok, n_frames, n_joints, segments):
    """Observation set for bundle adjustment: all accepted observations of tracks
    triangulated from at least two views, plus segment point pairs."""
    pt_of_track = -np.ones(len(ok), int)
    pt_of_track[ok] = np.arange(ok.sum())
    use = tracks['valid'] & ok[:, None]
    ti, ci = np.nonzero(use)
    obs = dict(ci=ci, pi=pt_of_track[ti], uv=tracks['uv'][ti, ci], c=tracks['c'][ti, ci],
               track_of_point=np.flatnonzero(ok))
    bi, bj, bb = [], [], []
    for b, (p, q) in enumerate(segments):
        tp = np.arange(n_frames) * n_joints + p
        tq = np.arange(n_frames) * n_joints + q
        m = ok[tp] & ok[tq]
        bi.append(pt_of_track[tp[m]])
        bj.append(pt_of_track[tq[m]])
        bb.append(np.full(m.sum(), b))
    obs['bones'] = tuple(np.concatenate(items).astype(int) if items else np.empty(0,dtype=int)
                         for items in (bi,bj,bb))
    return obs
