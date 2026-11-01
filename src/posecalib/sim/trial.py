"""One simulated calibration trial: scene, calibration walk, detector, calibration, metrics."""
from __future__ import annotations

import time

import numpy as np

from ..api import Calibrator
from ..geometry import camera_centers, triangulate_robust
from .board import calibrate_board, simulate_board_session
from .detector import as_detections, draw_detector_noise, observe, subset_noise
from .metrics import crlb_rotation, evaluate
from .scene import FPS, NJ, PP, SEGMENT_LENGTHS, cams_arrays, intrinsics, make_cameras, simulate_person

DEFAULT_CFG = dict(
    # scene
    n_cam=4, cam_radius=(3.5, 4.0), cam_height=(1.2, 2.6), focal=(1100.0, 1300.0),
    az_jitter_deg=15.0, cam_target=(0.0, 0.0, 1.0), cam_target_jitter=(0.2, 0.2, 0.1),
    walk_half=1.55,
    # sequences
    cal_max_duration=60.0, cal_duration=40.0, cal_step=5, test_duration=10.0, motion='walk',
    # detector
    sigma0=3.0, tau=0.3, conf_noise=0.08, sigma_cap=4.0, edge_px=80.0,
    p_out=0.03, out_range=(30.0, 150.0), p_swap=0.02,
    delta=0.0, delta_mode='body_norm',
    # scale measurements: tape on 8 limb segments, laser distance between cameras 0 and 1
    tape_sigma=0.01, range_sigma=0.005,
    # algorithm
    init_confidence=0.6, compute_crlb=False,
    focal_err=0.0,                  # relative error (+-, uniform) of the assumed focal lengths
    # board baseline
    board_poses=25, board_noise=0.3, board_max_angle=60.0, board_min_corners=8,
)

VARIANTS = {
    'ls': dict(weighted=False, loss='linear', use_segments=False),
    'weighted_ls': dict(weighted=True, loss='linear', use_segments=False),
    'huber': dict(weighted=False, loss='huber', use_segments=False),
    'weighted_huber': dict(weighted=True, loss='huber', use_segments=False),
    'weighted_cauchy': dict(weighted=True, loss='cauchy', use_segments=False),
    'weighted_cauchy_segments': dict(weighted=True, loss='cauchy', use_segments=True),
}
DEFAULT_VARIANT = 'weighted_cauchy_segments'


def to_tracks(arr):
    """``(C,F,J,...)`` -> ``(F*J, C, ...)``."""
    C, F, J = arr.shape[:3]
    axes = (1, 2, 0) + tuple(range(3, arr.ndim))
    return arr.transpose(axes).reshape((F * J, C) + arr.shape[3:])


def make_seed_data(seed, cfg):
    """Scene, motions, pre-drawn detector noise and scale measurements for one seed."""
    ss = np.random.SeedSequence([20261022, int(seed)])
    rs = [np.random.default_rng(s) for s in ss.spawn(10)]
    cams = make_cameras(rs[0], cfg)
    walk = simulate_person(rs[1], cfg['cal_max_duration'], 'walk', walk_half=cfg['walk_half'])
    static = simulate_person(rs[2], cfg['cal_max_duration'], 'static', walk_half=cfg['walk_half'])
    test = simulate_person(rs[3], cfg['test_duration'], 'walk', walk_half=cfg['walk_half'])
    cal_frames = np.arange(0, len(walk['J']), cfg['cal_step'])
    M = cfg['n_cam']
    noise_cal = draw_detector_noise(rs[4], M, len(cal_frames))
    noise_test = draw_detector_noise(rs[5], M, len(test['J']))
    tape = SEGMENT_LENGTHS[:8] + cfg['tape_sigma'] * rs[6].standard_normal(8)
    d01 = float(np.linalg.norm(cams[0]['C'] - cams[1]['C'])) + cfg['range_sigma'] * rs[6].standard_normal()
    focal_eps = rs[7].uniform(-1.0, 1.0, M)
    return dict(seed=seed, cams=cams, walk=walk, static=static, test=test, cal_frames=cal_frames,
                noise_cal=noise_cal, noise_test=noise_test, tape=tape, d01=d01, focal_eps=focal_eps,
                rng_board=rs[8], alg_seed=[int(seed), 77])


def prepare_test(data, cfg):
    """Test sequence: detector output and triangulation with the TRUE cameras."""
    cams = data['cams']
    frames = np.arange(len(data['test']['J']))
    obs = observe(cams, data['test'], frames, data['noise_test'], cfg)
    tracks = dict(uv=to_tracks(obs['uv'] - PP), c=to_tracks(obs['c']), valid=to_tracks(obs['valid']))
    Rt, tt, ft = cams_arrays(cams)
    X, ok, _, _, sig = triangulate_robust(Rt, tt, ft, tracks['uv'], tracks['c'], tracks['valid'])
    return dict(tracks=tracks, X=X, ok=ok, sig=sig, G=data['test']['J'], f=ft)


def calibration_observations(data, cfg):
    person = data['walk'] if cfg['motion'] == 'walk' else data['static']
    n_use = int(round(cfg['cal_duration'] * FPS / cfg['cal_step']))
    frames = data['cal_frames'][:n_use]
    obs = observe(data['cams'], person, frames, subset_noise(data['noise_cal'], n_use), cfg)
    return obs, person['J'][frames]


def run_pose_trial(seed, cfg, variants=(DEFAULT_VARIANT,), refine_focal_list=(False,), data=None, test=None):
    """Calibrate from the simulated calibration walk and evaluate on the test sequence.

    Returns one result row per variant x refine_focal setting.
    """
    t_start = time.perf_counter()
    if data is None:
        data = make_seed_data(seed, cfg)
    cams = data['cams']
    obs, G = calibration_observations(data, cfg)
    kp, conf = as_detections(obs)
    K = intrinsics(cams)
    f_true = K[:, 0, 0].copy()
    K[:, 0, 0] *= 1.0 + cfg['focal_err'] * data['focal_eps']
    K[:, 1, 1] = K[:, 0, 0]
    if test is None:
        test = prepare_test(data, cfg)
    base = dict(seed=seed, sigma0=cfg['sigma0'], delta_cm=100 * cfg['delta'], delta_mode=cfg['delta_mode'],
                duration_s=cfg['cal_duration'], n_frames=obs['uv'].shape[1], motion=cfg['motion'],
                focal_err=cfg['focal_err'])
    if 'bias_px' in obs:
        bp = obs['bias_px'][obs['valid']]
        base['bias_px_rms'] = float(np.sqrt(np.mean(bp ** 2)))
    if cfg.get('compute_crlb'):
        clean = to_tracks(obs['valid'] & ~obs['gross'])
        crl = crlb_rotation(cams, to_tracks(obs['sig']), clean, G.reshape(-1, 3))
        base['crlb_rot_deg'] = float(np.mean(crl))
    tape = np.concatenate([data['tape'], [np.nan]])
    rows = []
    try:
        cal = Calibrator(kp, conf, K, segment_lengths=tape, seed=data['alg_seed'])
        cal.initialize(init_confidence=cfg['init_confidence'])
    except Exception as e:  # noqa: BLE001 - a failed seed is a result, not a crash
        return [dict(base, variant=v, refine_focal=False, ok=False, error=f'init: {e}') for v in variants]
    base.update(time_init_s=cal.init['info']['time_s'], n_points=cal.init['info']['n_points'])
    for v in variants:
        for rf in refine_focal_list:
            try:
                res = cal.refine(refine_focal=rf, **VARIANTS[v])
                Ce = camera_centers(res.R, res.t)
                s_range = data['d01'] / float(np.linalg.norm(Ce[0] - Ce[1]))
                ev = evaluate(res.R, res.t, res.K[:, 0, 0], cams, test, dict(tape=1.0, range=s_range))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(base, variant=v, refine_focal=rf, ok=False, error=f'refine: {e}'))
                continue
            rows.append(dict(base, variant=v, refine_focal=rf, ok=True, time_refine_s=res.info['refine_time_s'],
                             focal_err_pct=float(100 * np.mean(np.abs(res.K[:, 0, 0] / f_true - 1))), **ev))
    for r in rows:
        r['time_trial_s'] = time.perf_counter() - t_start
    return rows


def run_board_trial(seed, cfg, deltas_cm=(0.0,), data=None):
    """Board calibration for a seed, evaluated on test observations with several offsets ``delta``."""
    if data is None:
        data = make_seed_data(seed, cfg)
    cams = data['cams']
    f = intrinsics(cams)[:, 0, 0]
    _, det = simulate_board_session(data['rng_board'], cams, cfg)
    base = dict(seed=seed, method='board', board_poses=cfg['board_poses'])
    try:
        bc = calibrate_board(f, det, len(cams))
    except Exception as e:  # noqa: BLE001
        return [dict(base, ok=False, error=str(e), delta_cm=d) for d in deltas_cm]
    rows = []
    for dcm in deltas_cm:
        test = prepare_test(data, dict(cfg, delta=dcm / 100.0))
        ev = evaluate(bc['Rs'], bc['ts'], f, cams, test, dict(board=1.0))
        rows.append(dict(base, ok=True, delta_cm=dcm, n_boards_shared=bc['n_boards'], **ev))
    return rows
