"""Calibrate a simulated four-camera rig from a 30-second calibration walk.

    python examples/synthetic_quickstart.py

Runs in about half a minute and prints the errors against the ground truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))   # run without installation

from posecalib import align_to_floor, calibrate  # noqa: E402
from posecalib.geometry import rotation_angle_deg  # noqa: E402
from posecalib.sim import DEFAULT_CFG, as_detections, calibration_observations, intrinsics, make_seed_data  # noqa: E402

cfg = dict(DEFAULT_CFG, cal_duration=30.0, sigma0=3.0)     # 3 px detector noise, 3% outliers, 2% L/R swaps
data = make_seed_data(seed=0, cfg=cfg)
cams = data['cams']
obs, _ = calibration_observations(data, cfg)
keypoints, confidence = as_detections(obs)                   # (C,F,17,2) with NaN, (C,F,17)
K = intrinsics(cams)
measured = float(np.linalg.norm(cams[0]['C'] - cams[1]['C'])) + 0.005   # tape/laser, 5 mm error

calib = calibrate(keypoints, confidence, K, camera_distance=(0, 1, measured))

R0, C0 = cams[0]['R'], cams[0]['C']
print('camera  rotation error, deg  centre error, mm  reprojection RMSE, px')
for k in range(len(cams)):
    rot = rotation_angle_deg(calib.R[k], cams[k]['R'] @ R0.T)
    ctr = 1000 * np.linalg.norm(calib.centers[k] - (cams[k]['C'] - C0) @ R0.T)
    print(f'{k:6d}  {rot:20.3f}  {ctr:16.1f}  {calib.reprojection_rmse[k]:20.2f}')

floor = align_to_floor(calib)
print('\ncamera heights above the floor, m (estimated / true):')
for k in range(len(cams)):
    print(f'  {k}: {floor.centers[k, 2]:.3f} / {cams[k]["C"][2]:.3f}')
