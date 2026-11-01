"""Synthetic scene: a ring of static cameras and a walking person (COCO-17 kinematics).

World frame: ``z`` up, a 4 x 4 m capture area centred at the origin on the plane ``z = 0``.
"""
from __future__ import annotations

import math

import numpy as np

from ..geometry import unit
from ..skeleton import COCO17

IMG_W, IMG_H = 1920, 1080
PP = np.array([IMG_W / 2.0, IMG_H / 2.0])       # principal point of the simulated cameras
FPS = 30
NJ = COCO17.num_keypoints

# anthropometry for a 1.75 m adult (Drillis-Contini proportions; widths between joint centres)
HEIGHT = 1.75
SEG = dict(thigh=0.245 * HEIGHT, shank=0.246 * HEIGHT,
           upper_arm=0.186 * HEIGHT, forearm=0.146 * HEIGHT,
           hip_w=0.18, sho_w=0.38, torso=0.50, neck=0.19)
ANKLE_Z = 0.08                                  # ankle-centre height above the floor in stance
HEAD_PTS = np.array([[0.095, 0.0, -0.03],       # nose        (head frame: x fwd, y left, z up)
                     [0.075, 0.032, 0.0],       # left eye
                     [0.075, -0.032, 0.0],      # right eye
                     [-0.005, 0.072, 0.0],      # left ear
                     [-0.005, -0.072, 0.0]])    # right ear
SEGMENT_LENGTHS = np.array([SEG['upper_arm'], SEG['forearm'], SEG['upper_arm'], SEG['forearm'],
                            SEG['thigh'], SEG['shank'], SEG['thigh'], SEG['shank'], SEG['hip_w']])

UPPER = np.zeros(NJ, bool)
UPPER[:11] = True                               # keypoints attached to the torso (else to the pelvis)

# body-frame direction of the systematic keypoint offset (x forward, y left, z up); see detector.py
BIAS_DIR_BODY = np.zeros((NJ, 3))
BIAS_DIR_BODY[11] = [1 / math.sqrt(2), 1 / math.sqrt(2), 0]    # left hip: lateral-forward
BIAS_DIR_BODY[12] = [1 / math.sqrt(2), -1 / math.sqrt(2), 0]   # right hip
BIAS_DIR_BODY[5] = [0, 1, 0]                                   # shoulders: lateral
BIAS_DIR_BODY[6] = [0, -1, 0]
BIAS_DIR_BODY[13] = BIAS_DIR_BODY[14] = [1, 0, 0]              # knees: forward (patella)
BIAS_DIR_BODY[7] = BIAS_DIR_BODY[8] = [-1, 0, 0]               # elbows: backward (olecranon)
BIAS_DIR_BODY[9] = [0, 1, 0]                                   # wrists: lateral
BIAS_DIR_BODY[10] = [0, -1, 0]
BIAS_DIR_BODY[15] = [0, 1, 0]                                  # ankles: lateral (malleoli)
BIAS_DIR_BODY[16] = [0, -1, 0]


# -----------------------------------------------------------------------------
# rotations
# -----------------------------------------------------------------------------
def rot_x(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def rot_y(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def rot_z(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _mv(R, v):
    return np.einsum('nij,nj->ni', R, v)


def look_at(C, target, up=(0.0, 0.0, 1.0)):
    """World-to-camera rotation of a camera at ``C`` looking at ``target`` (image y down)."""
    z = unit(np.asarray(target, float) - C)
    x = unit(np.cross(z, up))
    y = np.cross(z, x)
    return np.stack([x, y, z])


# -----------------------------------------------------------------------------
# cameras
# -----------------------------------------------------------------------------
def make_cameras(rng, cfg):
    """``n_cam`` cameras on a circle: azimuths ``360/n*k`` +- jitter, distinct heights,
    looking at about ``cam_target``. Returns a list of dicts ``R, t, C, f, az``."""
    n = cfg['n_cam']
    jit = cfg['az_jitter_deg']
    az = np.deg2rad(np.arange(n) * 360.0 / n + rng.uniform(-jit, jit, n))
    rad = rng.uniform(*cfg['cam_radius'], n)
    lo, hi = cfg['cam_height']
    edges = np.linspace(lo, hi, n + 1)
    hts = rng.permutation(rng.uniform(edges[:-1], edges[1:]))   # stratified -> distinct heights
    cams = []
    for k in range(n):
        C = np.array([rad[k] * math.cos(az[k]), rad[k] * math.sin(az[k]), hts[k]])
        tgt = np.asarray(cfg['cam_target']) + rng.uniform(-1, 1, 3) * np.asarray(cfg['cam_target_jitter'])
        R = look_at(C, tgt)
        f = rng.uniform(*cfg['focal'])
        cams.append(dict(R=R, t=-R @ C, C=C, f=f, az=float(az[k])))
    return cams


def intrinsics(cams):
    """``(C,3,3)`` intrinsic matrices of simulated cameras."""
    K = np.zeros((len(cams), 3, 3))
    for k, c in enumerate(cams):
        K[k] = [[c['f'], 0.0, PP[0]], [0.0, c['f'], PP[1]], [0.0, 0.0, 1.0]]
    return K


def cams_arrays(cams):
    return (np.stack([c['R'] for c in cams]), np.stack([c['t'] for c in cams]),
            np.array([c['f'] for c in cams]))


def project_px(R, t, f, X):
    """Project points ``X (...,3)`` to full pixel coordinates; returns ``uv (...,2)`` and depth."""
    Xc = X @ R.T + t
    z = Xc[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = f * Xc[..., :2] / z[..., None] + PP
    return uv, z


# -----------------------------------------------------------------------------
# person
# -----------------------------------------------------------------------------
def _ou(rng, n, dt, sigma, tau):
    """Ornstein-Uhlenbeck process (stationary std ``sigma``, correlation time ``tau``)."""
    a = math.exp(-dt / tau)
    b = sigma * math.sqrt(1.0 - a * a)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = sigma * e[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * e[i]
    return x


def _walk_path(rng, n, dt, half):
    """Walking towards random waypoints with a bounded turn rate."""
    v = np.clip(0.85 + _ou(rng, n, dt, 0.2, 3.0), 0.5, 1.2)
    om_noise = _ou(rng, n, dt, 0.5, 1.0)
    p = rng.uniform(-1.0, 1.0, 2)
    th = rng.uniform(-math.pi, math.pi)

    def new_wp(p):
        while True:
            w = rng.uniform(-half, half, 2)
            if np.linalg.norm(w - p) > 1.2:
                return w

    wp = new_wp(p)
    P = np.empty((n, 2))
    TH = np.empty(n)
    for i in range(n):
        d = wp - p
        if d @ d < 0.4 ** 2:
            wp = new_wp(p)
            d = wp - p
        err = (math.atan2(d[1], d[0]) - th + math.pi) % (2 * math.pi) - math.pi
        om = min(max(2.0 * err + om_noise[i], -1.8), 1.8)
        th += om * dt
        p = np.clip(p + v[i] * dt * np.array([math.cos(th), math.sin(th)]), -1.8, 1.8)
        P[i] = p
        TH[i] = th
    return P, TH, v


def _arm_program(rng, n, dt, gap_mean):
    """Arm episodes (raise, abduct, reach forward, wave): blend weight and target angles."""
    b = np.zeros(n)
    bf = np.zeros(n)
    ba = np.zeros(n)
    el = np.zeros(n)
    t = rng.exponential(gap_mean)
    T = n * dt
    while t < T:
        D = rng.uniform(1.2, 2.5)
        typ = rng.integers(4)
        i0, i1 = int(t / dt), min(n, int((t + D) / dt) + 1)
        if i1 > i0:
            tt = np.arange(i0, i1) * dt
            w = np.sin(np.pi * np.clip((tt - t) / D, 0.0, 1.0)) ** 2
            if typ == 0:      # overhead
                f_, a_, e_ = rng.uniform(140, 170), rng.uniform(0, 25), np.full(len(tt), rng.uniform(0, 20))
            elif typ == 1:    # sideways
                f_, a_, e_ = rng.uniform(-10, 20), rng.uniform(70, 100), np.full(len(tt), rng.uniform(0, 30))
            elif typ == 2:    # forward reach / punch
                f_, a_, e_ = rng.uniform(70, 100), rng.uniform(0, 20), np.full(len(tt), rng.uniform(0, 15))
            else:             # wave with a raised arm
                f_, a_ = rng.uniform(20, 50), rng.uniform(100, 140)
                e_ = 50 + 35 * np.sin(2 * np.pi * 2.0 * (tt - t))
            b[i0:i1] = w
            bf[i0:i1] = f_
            ba[i0:i1] = a_
            el[i0:i1] = e_
        t += D + rng.exponential(gap_mean)
    return b, np.deg2rad(bf), np.deg2rad(ba), np.deg2rad(el)


def simulate_person(rng, duration, mode='walk', fps=FPS, walk_half=1.55):
    """Motion of one person.

    ``mode='walk'``: walking around the area at 0.5-1.2 m/s with turns, gait cycle,
    arm swing and arm-raising episodes. ``mode='static'``: standing in place, turning
    the torso and moving the arms frequently.

    Returns a dict with ``J (n,17,3)`` keypoints and auxiliary axes for the detector model.
    """
    n = int(round(duration * fps))
    dt = 1.0 / fps
    d2r = np.deg2rad
    if mode == 'walk':
        P, heading, v = _walk_path(rng, n, dt, walk_half)
        psi = heading + d2r(12) * _ou(rng, n, dt, 1.0, 2.0)
        phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum((0.55 + 0.35 * v) * dt)
        sp = np.clip((v - 0.5) / 0.7, 0, 1)
        amp_h = d2r(12 + 14 * sp)
        th_L = amp_h * np.sin(phase) + d2r(3)
        th_R = amp_h * np.sin(phase + np.pi) + d2r(3)
        kn_L = d2r(5) + d2r(55) * np.maximum(0, np.cos(phase)) ** 2
        kn_R = d2r(5) + d2r(55) * np.maximum(0, np.cos(phase + np.pi)) ** 2
        lean = d2r(4) + d2r(3) * _ou(rng, n, dt, 1.0, 2.0)
        twist = -d2r(6) * np.sin(phase) * (v / 1.2) + d2r(8) * _ou(rng, n, dt, 1.0, 2.0)
        amp_a = d2r(10 + 15 * sp)
        bfL0, bfR0 = amp_a * np.sin(phase + np.pi), amp_a * np.sin(phase)
        elL0 = d2r(15) + d2r(10) * (1 + np.sin(phase + np.pi)) / 2
        elR0 = d2r(15) + d2r(10) * (1 + np.sin(phase)) / 2
        gap = 2.0
    elif mode == 'static':
        p0 = rng.uniform(-0.3, 0.3, 2)
        P = np.tile(p0, (n, 1))
        v = np.zeros(n)
        psi = rng.uniform(-np.pi, np.pi) + d2r(25) * _ou(rng, n, dt, 1.0, 4.0)
        th_L = d2r(2) + d2r(2) * _ou(rng, n, dt, 1.0, 2.0)
        th_R = d2r(2) + d2r(2) * _ou(rng, n, dt, 1.0, 2.0)
        kn_L = d2r(5) + d2r(3) * np.abs(_ou(rng, n, dt, 1.0, 2.0))
        kn_R = d2r(5) + d2r(3) * np.abs(_ou(rng, n, dt, 1.0, 2.0))
        lean = d2r(2) + d2r(4) * _ou(rng, n, dt, 1.0, 2.0)
        twist = d2r(15) * _ou(rng, n, dt, 1.0, 3.0)
        bfL0 = d2r(5) * _ou(rng, n, dt, 1.0, 2.0)
        bfR0 = d2r(5) * _ou(rng, n, dt, 1.0, 2.0)
        elL0 = np.full(n, d2r(15))
        elR0 = np.full(n, d2r(15))
        gap = 0.7
    else:
        raise ValueError(mode)
    roll = d2r(2) * _ou(rng, n, dt, 1.0, 2.0)
    hyaw = np.clip(d2r(20) * _ou(rng, n, dt, 1.0, 1.5), -d2r(60), d2r(60))
    hpitch = d2r(5) + d2r(8) * _ou(rng, n, dt, 1.0, 2.0)
    ba0 = d2r(8)
    bL, tfL, taL, teL = _arm_program(rng, n, dt, gap)
    bR, tfR, taR, teR = _arm_program(rng, n, dt, gap)
    bfL, baL, elL = (1 - bL) * bfL0 + bL * tfL, (1 - bL) * ba0 + bL * taL, (1 - bL) * elL0 + bL * teL
    bfR, baR, elR = (1 - bR) * bfR0 + bR * tfR, (1 - bR) * ba0 + bR * taR, (1 - bR) * elR0 + bR * teR

    # forward kinematics
    S = SEG
    Rp = rot_z(psi)                                             # pelvis (yaw only)
    Rt = rot_z(psi + twist) @ rot_y(lean) @ rot_x(roll)         # torso
    Rh = Rt @ rot_z(hyaw) @ rot_y(hpitch)                       # head

    def leg(th, kn):
        dth = np.stack([np.sin(th), 0 * th, -np.cos(th)], -1)
        dsh = np.stack([np.sin(th - kn), 0 * th, -np.cos(th - kn)], -1)
        return _mv(Rp, dth), _mv(Rp, dsh)

    gthL, gshL = leg(th_L, kn_L)
    gthR, gshR = leg(th_R, kn_R)
    drop = np.maximum(S['thigh'] * np.cos(th_L) + S['shank'] * np.cos(th_L - kn_L),
                      S['thigh'] * np.cos(th_R) + S['shank'] * np.cos(th_R - kn_R))
    pel = np.column_stack([P, ANKLE_Z + drop])                  # the stance foot touches the floor
    lp = Rp[:, :, 1]
    hipL, hipR = pel + 0.5 * S['hip_w'] * lp, pel - 0.5 * S['hip_w'] * lp
    kneL, kneR = hipL + S['thigh'] * gthL, hipR + S['thigh'] * gthR
    ankL, ankR = kneL + S['shank'] * gshL, kneR + S['shank'] * gshR
    sc = pel + S['torso'] * Rt[:, :, 2]
    shoL, shoR = sc + 0.5 * S['sho_w'] * Rt[:, :, 1], sc - 0.5 * S['sho_w'] * Rt[:, :, 1]
    down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), (n, 3))

    def arm(sign, bf, ba, el):
        Rua = rot_x(sign * ba) @ rot_y(-bf)
        return _mv(Rt, _mv(Rua, down)), _mv(Rt, _mv(Rua @ rot_y(-el), down))

    duaL, dfaL = arm(+1, bfL, baL, elL)
    duaR, dfaR = arm(-1, bfR, baR, elR)
    elbL, elbR = shoL + S['upper_arm'] * duaL, shoR + S['upper_arm'] * duaR
    wriL, wriR = elbL + S['forearm'] * dfaL, elbR + S['forearm'] * dfaR
    hc = sc + S['neck'] * Rt[:, :, 2]
    J = np.empty((n, NJ, 3))
    J[:, 0:5] = hc[:, None, :] + np.einsum('nij,kj->nki', Rh, HEAD_PTS)
    for j, X in zip(range(5, 17), [shoL, shoR, elbL, elbR, wriL, wriR, hipL, hipR, kneL, kneR, ankL, ankR]):
        J[:, j] = X

    # visibility normals (detector confidence model)
    fh, lh, lt = Rh[:, :, 0], Rh[:, :, 1], Rt[:, :, 1]
    vis_n = np.empty((n, NJ, 3))
    vis_n[:, 0:3] = fh[:, None]
    vis_n[:, 3] = unit(0.8 * lh + 0.3 * fh)
    vis_n[:, 4] = unit(-0.8 * lh + 0.3 * fh)
    vis_n[:, [5, 7, 9]] = lt[:, None]
    vis_n[:, [6, 8, 10]] = -lt[:, None]
    vis_n[:, [11, 13, 15]] = lp[:, None]
    vis_n[:, [12, 14, 16]] = -lp[:, None]
    # body axes for the systematic-offset model: torso for the upper body, pelvis below
    Rb = np.where(UPPER[None, :, None, None], Rt[:, None], Rp[:, None])     # (n,17,3,3)
    bias_dir = np.einsum('njab,jb->nja', Rb, BIAS_DIR_BODY)
    body_fwd = Rb[..., 0]
    return dict(J=J, vis_n=vis_n, bias_dir=bias_dir, body_fwd=body_fwd, pelvis=pel,
                psi=psi, v=v, mode=mode, fps=fps)
