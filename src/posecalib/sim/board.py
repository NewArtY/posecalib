"""Baseline: calibration with a planar ChArUco-like board (7 x 5 squares of 6 cm).

The simulation is deliberately favourable to the board: every visible corner is
detected with 0.3 px noise and there is no systematic error.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from ..geometry import project_to_so3, unit
from .scene import IMG_H, IMG_W, PP, project_px

BOARD_SQ = 0.06
BOARD_NX, BOARD_NY = 7, 5


def board_corners():
    """Inner corners of a 7 x 5 board (6 x 4 = 24) in the board frame, centred at 0."""
    xs = (np.arange(1, BOARD_NX) - BOARD_NX / 2.0) * BOARD_SQ
    ys = (np.arange(1, BOARD_NY) - BOARD_NY / 2.0) * BOARD_SQ
    return np.array([[x, y, 0.0] for y in ys for x in xs])


def simulate_board_session(rng, cams, cfg):
    """Board poses in the central zone (|x|,|y| <= 0.7 m, z 0.8-1.6 m), facing the bisector
    of two neighbouring cameras +-15 deg, rolled +-45 deg. A corner is detected if the
    board is seen at less than ``board_max_angle`` from its normal and the corner is in frame."""
    G = board_corners()
    M = len(cams)
    order = np.argsort([c['az'] for c in cams])
    poses, det = [], []
    d2r = np.deg2rad
    for p in range(cfg['board_poses']):
        a, b = order[p % M], order[(p + 1) % M]
        ctr = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(0.8, 1.6)])
        n0 = unit(unit(cams[a]['C'] - ctr) + unit(cams[b]['C'] - ctr))
        az = math.atan2(n0[1], n0[0]) + rng.normal(0, d2r(15))
        el = math.asin(n0[2]) + rng.normal(0, d2r(15))
        nrm = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        xb = unit(np.cross([0.0, 0.0, 1.0], nrm))
        yb = np.cross(nrm, xb)
        roll = rng.uniform(-d2r(45), d2r(45))
        xr = math.cos(roll) * xb + math.sin(roll) * yb
        yr = np.cross(nrm, xr)
        Rb = np.column_stack([xr, yr, nrm])
        poses.append((Rb, ctr))
        Xw = ctr + G @ Rb.T
        for k, cam in enumerate(cams):
            r = unit(cam['C'] - Xw)
            ang = np.degrees(np.arccos(np.clip(r @ nrm, -1, 1)))
            uv, z = project_px(cam['R'], cam['t'], cam['f'], Xw)
            vis = (ang < cfg['board_max_angle']) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & \
                  (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
            if vis.sum() >= cfg['board_min_corners']:
                det.append(dict(cam=k, pose=p, idx=np.flatnonzero(vis),
                                uv=uv[vis] + cfg['board_noise'] * rng.standard_normal((int(vis.sum()), 2))))
    return poses, det


def planar_pose(Gxy, uv, f):
    """Pose of a planar target from centred pixels: homography (DLT) -> [r1 r2 t] -> refinement."""
    n = len(Gxy)
    xn = uv / f
    A = np.zeros((2 * n, 9))
    X, Y = Gxy[:, 0], Gxy[:, 1]
    u, v = xn[:, 0], xn[:, 1]
    A[0::2, 0:3] = np.column_stack([X, Y, np.ones(n)])
    A[0::2, 6:9] = -u[:, None] * np.column_stack([X, Y, np.ones(n)])
    A[1::2, 3:6] = np.column_stack([X, Y, np.ones(n)])
    A[1::2, 6:9] = -v[:, None] * np.column_stack([X, Y, np.ones(n)])
    H = np.linalg.svd(A)[2][-1].reshape(3, 3)
    lam = 2.0 / (np.linalg.norm(H[:, 0]) + np.linalg.norm(H[:, 1]))
    if H[2, 2] * lam < 0:
        lam = -lam
    r1, r2, t = lam * H[:, 0], lam * H[:, 1], lam * H[:, 2]
    R = project_to_so3(np.column_stack([r1, r2, np.cross(r1, r2)]))
    G3 = np.column_stack([Gxy, np.zeros(n)])

    def fun(p):
        Rm = Rotation.from_rotvec(p[:3]).as_matrix()
        Xc = G3 @ Rm.T + p[3:]
        return (f * Xc[:, :2] / Xc[:, 2:3] - uv).ravel()

    res = least_squares(fun, np.concatenate([Rotation.from_matrix(R).as_rotvec(), t]), method='lm')
    return Rotation.from_rotvec(res.x[:3]).as_matrix(), res.x[3:]


def calibrate_board(f, det, M):
    """Board calibration: board pose in every view -> chain from camera 0 -> bundle adjustment on corners.

    ``det`` holds pixel detections (full image coordinates of the simulated cameras).
    Returns extrinsics in the frame of camera 0 with metric scale.
    """
    G = board_corners()
    views = {}
    for d in det:
        uv = d['uv'] - PP
        views[(d['cam'], d['pose'])] = (*planar_pose(G[d['idx'], :2], uv, f[d['cam']]), dict(d, uv=uv))
    reg = {0: (np.eye(3), np.zeros(3))}
    while len(reg) < M:
        best = None
        for k in range(M):
            if k in reg:
                continue
            sh = [(m, p) for (m, p) in views if m in reg and (k, p) in views]
            if sh and (best is None or len(sh) > len(best[1])):
                best = (k, sh)
        if best is None:
            raise RuntimeError('cameras are not connected by shared board poses')
        k, sh = best
        Rl, tl = [], []
        for m, p in sh:
            Rmb, tmb, _ = views[(m, p)]
            Rkb, tkb, _ = views[(k, p)]
            Rm, tm = reg[m]
            A = Rkb @ Rmb.T
            Rl.append(A @ Rm)
            tl.append(A @ (tm - tmb) + tkb)
        reg[k] = (project_to_so3(np.sum(Rl, axis=0)), np.mean(tl, axis=0))
    seen = {}
    for (k, p) in views:
        seen.setdefault(p, []).append(k)
    boards = [p for p in sorted(seen) if len(seen[p]) >= 2]
    if not boards:
        raise RuntimeError('no board pose is shared by two cameras')
    bidx = {p: i for i, p in enumerate(boards)}
    Bp = np.zeros((len(boards), 6))
    for p in boards:
        k = min(seen[p])
        Rkb, tkb, _ = views[(k, p)]
        Rk, tk = reg[k]
        Bp[bidx[p]] = np.concatenate([Rotation.from_matrix(Rk.T @ Rkb).as_rotvec(), Rk.T @ (tkb - tk)])
    obs_c, obs_b, obs_g, obs_uv = [], [], [], []
    for (k, p), (_, _, d) in views.items():
        if p in bidx:
            obs_c.append(np.full(len(d['idx']), k))
            obs_b.append(np.full(len(d['idx']), bidx[p]))
            obs_g.append(d['idx'])
            obs_uv.append(d['uv'])
    oc, ob, og, ouv = map(np.concatenate, (obs_c, obs_b, obs_g, obs_uv))
    N, nB = len(oc), len(boards)
    cam0 = np.zeros((M, 6))
    for k in range(M):
        cam0[k] = np.concatenate([Rotation.from_matrix(reg[k][0]).as_rotvec(), reg[k][1]])
    x0 = np.concatenate([cam0[1:].ravel(), Bp.ravel()])

    def unpack(x):
        cam = np.vstack([np.zeros((1, 6)), x[:6 * (M - 1)].reshape(M - 1, 6)])
        return cam, x[6 * (M - 1):].reshape(nB, 6)

    def fun(x):
        cam, B = unpack(x)
        Rc = Rotation.from_rotvec(cam[:, :3]).as_matrix()
        Rb = Rotation.from_rotvec(B[:, :3]).as_matrix()
        Xw = np.einsum('nij,nj->ni', Rb[ob], G[og]) + B[ob, 3:]
        Xc = np.einsum('nij,nj->ni', Rc[oc], Xw) + cam[oc, 3:]
        return (f[oc, None] * Xc[:, :2] / Xc[:, 2:3] - ouv).ravel()

    cols = np.concatenate([np.where(oc[:, None] > 0, 6 * (oc[:, None] - 1) + np.arange(6), -1),
                           6 * (M - 1) + 6 * ob[:, None] + np.arange(6)], axis=1)
    rows = np.broadcast_to((2 * np.arange(N))[:, None, None] + np.array([[0], [1]]), (N, 2, 12))
    cc = np.broadcast_to(cols[:, None, :], rows.shape)
    keep = cc >= 0
    S = coo_matrix((np.ones(keep.sum()), (rows[keep], cc[keep])), shape=(2 * N, len(x0))).tocsr()
    t0 = time.perf_counter()
    res = least_squares(fun, x0, jac_sparsity=S, method='trf', x_scale='jac', tr_solver='lsmr',
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
    cam, _ = unpack(res.x)
    r = res.fun.reshape(N, 2)
    return dict(Rs=Rotation.from_rotvec(cam[:, :3]).as_matrix(), ts=cam[:, 3:].copy(),
                n_boards=nB, n_obs=N, reproj_rmse_px=float(np.sqrt(np.mean(np.sum(r ** 2, 1)))),
                time=time.perf_counter() - t0)
