"""Model of a 2D keypoint detector.

* confidence drops for keypoints on the far side of the body and near the image border;
* Gaussian noise with ``sigma = sigma0 / max(c_true, 1/sigma_cap)`` px;
* gross errors: outliers shifted by 30-150 px with unchanged confidence, and
  left/right swaps of whole skeletons;
* optional systematic offset: the detector reports a "visible" point shifted
  perpendicular to the viewing ray in a body-fixed direction with a sign that
  depends on the viewing azimuth, so different cameras see inconsistent offsets.
"""
from __future__ import annotations

import numpy as np

from ..geometry import unit
from ..skeleton import COCO17
from .scene import IMG_H, IMG_W, NJ, project_px

SWAP_IDX = np.arange(NJ)
for _a, _b in COCO17.mirror_pairs:
    SWAP_IDX[_a], SWAP_IDX[_b] = _b, _a
IS_LR = SWAP_IDX != np.arange(NJ)

# visibility: c_vis = cmin + (1 - cmin) * sigmoid((s - s0) / w), s = normal . direction to camera
VIS_S0 = np.full(NJ, -0.35)
VIS_W = np.full(NJ, 0.15)
VIS_CMIN = np.full(NJ, 0.10)
VIS_S0[[0, 1, 2]] = -0.15                       # the face is not visible from behind
VIS_W[[0, 1, 2]] = 0.12
VIS_CMIN[[0, 1, 2]] = 0.05
VIS_S0[[3, 4]] = -0.30

BIAS_W = np.zeros(NJ)
BIAS_W[[5, 6, 7, 8, 11, 12, 13, 14]] = 1.0      # shoulders, elbows, hips, knees: delta
BIAS_W[[9, 10, 15, 16]] = 0.5                   # wrists, ankles: delta / 2; face: 0


def draw_detector_noise(rng, n_cam, n_frames):
    """Pre-drawn random variables of the detector (shared by all sigma0 and delta settings)."""
    return dict(nxy=rng.standard_normal((n_cam, n_frames, NJ, 2)),
                nc=rng.standard_normal((n_cam, n_frames, NJ)),
                u_out=rng.random((n_cam, n_frames, NJ)),
                out_mag=rng.random((n_cam, n_frames, NJ)),
                out_ang=rng.random((n_cam, n_frames, NJ)) * 2 * np.pi,
                u_swap=rng.random((n_cam, n_frames)))


def subset_noise(noise, n):
    return {k: (v[:, :n] if v.ndim >= 2 else v) for k, v in noise.items()}


def bias_offsets(person, frames, C, delta, mode):
    """3D offset of the "visible" keypoint for a camera at ``C``.

    ``body_norm``: ``b = delta_j cos(az) P(R d_j) / max(|P(R d_j)|, 0.2)``;
    ``body_phys``: ``b = delta_j cos(az) P(R d_j)``;
    ``ray``: shift along the viewing ray (sanity check: does not change the projection).
    ``P`` projects onto the plane perpendicular to the ray; ``az`` is the camera
    azimuth relative to the body's forward direction.
    """
    J = person['J'][frames]
    r = unit(C - J)
    if mode == 'ray':
        return delta * BIAS_W[None, :, None] * r
    d = person['bias_dir'][frames]
    pv = d - np.sum(d * r, -1, keepdims=True) * r
    if mode == 'body_norm':
        pv = pv / np.maximum(np.linalg.norm(pv, axis=-1, keepdims=True), 0.2)
    elif mode != 'body_phys':
        raise ValueError(mode)
    fwd = person['body_fwd'][frames].copy()
    fwd[..., 2] = 0.0
    rh = r.copy()
    rh[..., 2] = 0.0
    w = np.sum(unit(fwd) * unit(rh), -1, keepdims=True)          # cos(azimuth)
    return delta * BIAS_W[None, :, None] * w * pv


def observe(cams, person, frames, noise, cfg):
    """Detector output for ``frames``.

    Returns pixel keypoints ``uv (C,F,17,2)``, reported confidence ``c (C,F,17)``,
    ``valid`` mask, ``gross`` flags (outlier or swapped) and the true noise ``sig``.
    """
    J = person['J'][frames]
    F = len(frames)
    M = len(cams)
    uv_o = np.empty((M, F, NJ, 2))
    c_o = np.empty((M, F, NJ))
    val_o = np.empty((M, F, NJ), bool)
    gro_o = np.empty((M, F, NJ), bool)
    swp_o = np.empty((M, F), bool)
    sig_o = np.empty((M, F, NJ))
    bias_px = []
    s0, w, cmin = VIS_S0[None], VIS_W[None], VIS_CMIN[None]
    lo, hi = cfg['out_range']
    for k, cam in enumerate(cams):
        X = J
        if cfg['delta'] > 0:
            X = J + bias_offsets(person, frames, cam['C'], cfg['delta'], cfg['delta_mode'])
        uv, z = project_px(cam['R'], cam['t'], cam['f'], X)
        if cfg['delta'] > 0:
            uv0, _ = project_px(cam['R'], cam['t'], cam['f'], J)
            bias_px.append(np.linalg.norm(uv - uv0, axis=-1))
        r = unit(cam['C'] - J)
        s = np.sum(person['vis_n'][frames] * r, -1)
        c_vis = cmin + (1 - cmin) / (1 + np.exp(-(s - s0) / w))
        bd = np.minimum(np.minimum(uv[..., 0], IMG_W - uv[..., 0]), np.minimum(uv[..., 1], IMG_H - uv[..., 1]))
        infr = (bd > 0) & (z > 0.1)
        c_edge = np.clip(0.5 + 0.5 * bd / cfg['edge_px'], 0.5, 1.0)
        c_true = c_vis * c_edge
        c_rep = np.clip(c_true + cfg['conf_noise'] * noise['nc'][k], 0.01, 1.0)
        sig = cfg['sigma0'] / np.maximum(c_true, 1.0 / cfg['sigma_cap'])
        uvo = uv + sig[..., None] * noise['nxy'][k]
        out = noise['u_out'][k] < cfg['p_out']
        mag = lo + (hi - lo) * noise['out_mag'][k]
        ang = noise['out_ang'][k]
        uvo = uvo + (out * mag)[..., None] * np.stack([np.cos(ang), np.sin(ang)], -1)
        val = infr & (c_rep >= cfg['tau'])
        sw = noise['u_swap'][k] < cfg['p_swap']
        if sw.any():
            uvo[sw] = uvo[sw][:, SWAP_IDX]
            c_rep[sw] = c_rep[sw][:, SWAP_IDX]
            val[sw] = val[sw][:, SWAP_IDX]
            out[sw] = out[sw][:, SWAP_IDX]
            sig[sw] = sig[sw][:, SWAP_IDX]
        gross = out.copy()
        gross[sw] |= IS_LR[None, :]
        uv_o[k], c_o[k], val_o[k], gro_o[k], swp_o[k], sig_o[k] = uvo, c_rep, val, gross, sw, sig
    res = dict(uv=uv_o, c=c_o, valid=val_o, gross=gro_o, swapped=swp_o, sig=sig_o)
    if bias_px:
        res['bias_px'] = np.stack(bias_px)
    return res


def as_detections(obs):
    """Detector output in the public input format: keypoints with NaN for invalid entries, confidence."""
    kp = np.where(obs['valid'][..., None], obs['uv'], np.nan)
    conf = np.where(obs['valid'], obs['c'], 0.0)
    return kp, conf
