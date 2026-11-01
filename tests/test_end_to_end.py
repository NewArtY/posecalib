"""End-to-end calibration on a synthetic four-camera scene (about a minute)."""
import numpy as np
import pytest

from posecalib import align_to_floor, calibrate
from posecalib.geometry import rotation_angle_deg
from posecalib.sim import DEFAULT_CFG, as_detections, calibration_observations, intrinsics, make_seed_data


@pytest.fixture(scope='module')
def result():
    cfg = dict(DEFAULT_CFG, cal_duration=20.0, sigma0=3.0)
    data = make_seed_data(1, cfg)
    obs, _ = calibration_observations(data, cfg)
    kp, conf = as_detections(obs)
    cams = data['cams']
    d01 = float(np.linalg.norm(cams[0]['C'] - cams[1]['C']))
    calib = calibrate(kp, conf, intrinsics(cams), camera_distance=(0, 1, d01), seed=0)
    return calib, cams


@pytest.mark.slow
def test_extrinsics_match_ground_truth(result):
    calib, cams = result
    R0, C0 = cams[0]['R'], cams[0]['C']
    for k in range(1, len(cams)):
        assert rotation_angle_deg(calib.R[k], cams[k]['R'] @ R0.T) < 0.3
        assert np.linalg.norm(calib.centers[k] - (cams[k]['C'] - C0) @ R0.T) < 0.03
    assert calib.scale_source == 'camera_distance'
    assert np.all(calib.reprojection_rmse < 10)


@pytest.mark.slow
def test_floor_alignment(result):
    calib, cams = result
    floor = align_to_floor(calib)
    heights = floor.centers[:, 2]
    true_heights = np.array([c['C'][2] for c in cams])
    assert np.max(np.abs(heights - true_heights)) < 0.03
    # vertical axis of the aligned frame agrees with the true vertical
    up_true = np.stack([c['R'] for c in cams]) @ np.array([0.0, 0.0, 1.0])     # world z in each camera
    up_est = floor.R @ np.array([0.0, 0.0, 1.0])
    ang = np.degrees(np.arccos(np.clip(np.sum(up_true * up_est, 1), -1, 1)))
    assert np.max(ang) < 1.0
