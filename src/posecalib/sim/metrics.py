"""Accuracy metrics against simulated ground truth."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from ..geometry import camera_centers, rotation_angle_deg, triangulate_robust, umeyama
from .scene import NJ


def crlb_rotation(cams, uv_true_sig, valid, G):
    """Cramer-Rao bound on the camera rotations.

    Uses only observations without gross errors and the true detector noise. Camera
    parameters are a local rotation and the centre; camera 0 is fixed and the scale is
    fixed by one centre coordinate of the camera with the longest baseline.

    Args:
        cams: simulated cameras; uv_true_sig: ``(T,M)`` true noise std (px);
        valid: ``(T,M)`` mask of clean observations; G: ``(T,3)`` true points.

    Returns:
        RMS rotation error bound ``sqrt(tr Sigma_omega)`` in degrees for cameras ``1..M-1``.
    """
    M = len(cams)
    okp = valid.sum(1) >= 2
    ti, ci = np.nonzero(valid & okp[:, None])
    w = 1.0 / uv_true_sig[ti, ci] ** 2
    P = int(okp.sum())
    pidx = -np.ones(len(okp), int)
    pidx[okp] = np.arange(P)
    nc = 6 * M
    Hcc = np.zeros((nc, nc))
    Hcp = np.zeros((nc, P, 3))
    Hpp = np.zeros((P, 3, 3))
    for k in range(M):
        m = ci == k
        R, C, f = cams[k]['R'], cams[k]['C'], cams[k]['f']
        Xc = (G[ti[m]] - C) @ R.T
        z = Xc[:, 2]
        A = np.zeros((m.sum(), 2, 3))
        A[:, 0, 0] = A[:, 1, 1] = f / z
        A[:, 0, 2] = -f * Xc[:, 0] / z ** 2
        A[:, 1, 2] = -f * Xc[:, 1] / z ** 2
        Xs = np.zeros((m.sum(), 3, 3))                 # [Xc]x
        Xs[:, 0, 1], Xs[:, 0, 2], Xs[:, 1, 2] = -Xc[:, 2], Xc[:, 1], -Xc[:, 0]
        Xs[:, 1, 0], Xs[:, 2, 0], Xs[:, 2, 1] = Xc[:, 2], -Xc[:, 1], Xc[:, 0]
        Jc = np.concatenate([A @ (-Xs), A @ (-R)[None]], axis=2)
        JX = A @ R[None]
        ww = w[m]
        Hcc[6 * k:6 * k + 6, 6 * k:6 * k + 6] += np.einsum('n,nai,naj->ij', ww, Jc, Jc)
        pp = pidx[ti[m]]
        hcp = np.einsum('n,nai,naj->nij', ww, Jc, JX)
        for q in range(6):
            np.add.at(Hcp[6 * k + q], pp, hcp[:, q, :])
        np.add.at(Hpp, pp, np.einsum('n,nai,naj->nij', ww, JX, JX))
    Sc = Hcc - np.einsum('ipa,pab,jpb->ij', Hcp, np.linalg.inv(Hpp), Hcp)
    Ck = np.stack([c['C'] for c in cams])
    kf = 1 + int(np.argmax(np.linalg.norm(Ck[1:] - Ck[0], axis=1)))
    mf = int(np.argmax(np.abs(Ck[kf] - Ck[0])))
    keep = [i for i in range(6, nc) if i != 6 * kf + 3 + mf]
    full = np.zeros((nc, nc))
    full[np.ix_(keep, keep)] = np.linalg.inv(Sc[np.ix_(keep, keep)])
    return [math.degrees(math.sqrt(np.trace(full[6 * k:6 * k + 3, 6 * k:6 * k + 3]))) for k in range(1, M)]


def _root_relative(X):
    """Subtract the mid-hip point of every frame: ``(F,17,3)``."""
    return X - 0.5 * (X[:, 11:12] + X[:, 12:13])


def evaluate(Rs_e, ts_e, f, cams, test, scales):
    """Accuracy of an estimated calibration (frame of camera 0, metric-ish scale).

    Args:
        Rs_e, ts_e: estimated extrinsics in the frame of camera 0.
        f: focal lengths used for triangulation.
        cams: true cameras.
        test: dict from :func:`posecalib.sim.trial.prepare_test`.
        scales: ``{name: multiplier}`` applied to the estimate before comparison
            (``1.0`` = keep the estimate's own scale). ``'ideal'`` (scale from the true
            camera distances) is always added.

    Metrics: mean rotation error of the non-reference cameras (deg); centre errors (mm);
    ``disp_<name>_mm`` = mean root-relative displacement of test keypoints between
    triangulation with estimated and true cameras from the same 2D observations;
    ``mpjpe_true_mm`` = root-relative error of triangulation with the true cameras.
    """
    M = len(cams)
    R0, t0, C0 = cams[0]['R'], cams[0]['t'], cams[0]['C']
    Ct0 = (np.stack([c['C'] for c in cams]) - C0) @ R0.T           # true centres in camera-0 frame
    Ce = camera_centers(Rs_e, ts_e)
    pd_t = [np.linalg.norm(Ct0[i] - Ct0[j]) for i, j in combinations(range(M), 2)]
    pd_e = [np.linalg.norm(Ce[i] - Ce[j]) for i, j in combinations(range(M), 2)]
    s_true = float(np.sum(pd_t) / np.sum(pd_e))
    out = dict(rot_err_deg=float(np.mean([rotation_angle_deg(Rs_e[k], cams[k]['R'] @ R0.T)
                                          for k in range(1, M)])))
    su, Ru, tu = umeyama(Ce, Ct0)
    out['center_err_shape_mm'] = 1000 * float(np.linalg.norm(su * Ce @ Ru.T + tu - Ct0, axis=1).mean())
    tr = test['tracks']
    Xu, oku, *_ = triangulate_robust(Rs_e, ts_e, f, tr['uv'], tr['c'], tr['valid'], sig_hat=test['sig'])
    ok = (oku & test['ok']).reshape(-1, NJ)
    G = test['G']
    F = G.shape[0]
    Xt = test['X'].reshape(F, NJ, 3)
    root_ok = ok[:, 11] & ok[:, 12]
    m = ok & root_ok[:, None]
    out['mpjpe_true_mm'] = 1000 * float(np.linalg.norm(_root_relative(Xt) - _root_relative(G), axis=-1)[m].mean())
    for name, s in list(scales.items()) + [('ideal', s_true)]:
        out[f'center_err_{name}_mm'] = 1000 * float(np.linalg.norm(s * Ce[1:] - Ct0[1:], axis=1).mean())
        out[f'scale_err_{name}_pct'] = 100 * (s / s_true - 1.0)
        Xe = ((s * Xu - t0) @ R0).reshape(F, NJ, 3)                # to world via the true pose of camera 0
        out[f'disp_{name}_mm'] = 1000 * float(np.linalg.norm(_root_relative(Xe) - _root_relative(Xt),
                                                             axis=-1)[m].mean())
    return out
