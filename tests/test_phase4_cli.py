"""Positive, bounded actual subprocess chain with independent motion truth."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from posecalib.geometry import rotation_angle_deg
from posecalib import Calibration
from test_phase4_clocks import scene, metric

ROOT = Path(__file__).resolve().parents[1]


def call(script, args, folder, label):
    # The outer harness places this complete descendant tree under512MiB/60s.
    command = [sys.executable, '-B', '-X', 'pycache_prefix='+str(folder/'empty_pycache'),
               str(ROOT/'examples'/script), *map(str, args)]
    (folder/(label+'_command.json')).write_text(json.dumps(command), encoding='utf-8')
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
    (folder/(label+'_stdout.txt')).write_text(proc.stdout, encoding='utf-8')
    (folder/(label+'_stderr.txt')).write_text(proc.stderr, encoding='utf-8')
    assert proc.returncode == 0, proc.stdout + proc.stderr


def export(folder, name, recording):
    true, world, _, data = scene(103, origin=17.25 if not recording else 50., recording=recording)
    paths = []
    for c, n in enumerate(data['frame_counts']):
        lm = np.full((n, 33, 5), np.nan)
        # Default33 face-only: no segments may be manufactured by the CLI.
        lm[:, :11, :2] = data['keypoints'][c, :n, :11] / [1920, 1080]
        lm[:, :11, 2] = 0
        lm[:, :11, 3:] = 1
        path = folder/f'{name}_camera{c}.npz'
        np.savez(path, landmarks=lm, image_size=[1920, 1080], K=data['K'][c], timestamps=data['timestamps'][c, :n])
        paths.append(path)
    packed = folder/f'{name}_packed.npz'
    call('prepare_blazepose_npz.py', [*paths, '--out', packed], folder, name+'_pack')
    with np.load(packed) as d:
        np.testing.assert_array_equal(d['frame_counts'], data['frame_counts'])
        assert np.isnan(d['keypoints'][1, data['frame_counts'][1]:]).all()
        assert np.all(d['confidence'][1, data['frame_counts'][1]:] == 0)
    return true, world, data, packed


def test_real_pack_calibrate_save_load_new_triangulate_cli(tmp_path):
    true, _, data, packed = export(tmp_path, 'calibration', 0)
    distance = np.linalg.norm(true.centers[1])
    path = tmp_path/'rig.json'
    call('calibrate_npz.py', [packed, '--distance', 0, 1, distance, '--frame-step', 3,
                             '--no-floor', '--fps', -5, '--out', path], tmp_path, 'calibrate')
    loaded = Calibration.load_json(path)
    angles = [rotation_angle_deg(a, b) for a, b in zip(loaded.R, true.R)]
    assert max(angles) < .05
    assert np.max(np.linalg.norm(loaded.centers-true.centers, axis=1)) < .003
    assert loaded.scale_source == 'camera_distance' and np.isnan(loaded.segment_lengths).all()
    assert loaded.time_offsets_seconds is None
    grid = data['timestamps'][0]
    grid = grid[(grid >= data['timestamps'][:, 0].max()) &
                (grid <= min(data['timestamps'][c, n-1] for c, n in enumerate(data['frame_counts'])))][::3]
    np.testing.assert_array_equal(loaded.info['reference_times'], grid)
    _, world, new_data, new_path = export(tmp_path, 'new', 1)
    out = tmp_path/'new_points.npz'
    call('triangulate_npz.py', [path, new_path, '--out', out], tmp_path, 'new_triangulate')
    with np.load(out) as d:
        points, times = d['points'], d['timestamps']
        assert np.isnan(points[:, 11:]).all()
        assert points.shape == (len(times), 33, 3)
        assert d['inliers'].shape == (3, len(times), 33)
        error = float(np.nanmedian(np.linalg.norm(points-world(times-50.), axis=-1)))
        assert error < .003
    metric(test='real_cli', angles_deg=angles, median_new_error_m=error, artifacts=str(tmp_path))


def test_cli_explicit_camera_fps_schema_without_timestamps(tmp_path):
    # A positive independent CFR control for the explicit camera_fps field.
    # Legacy fps=[30,30] stays invalid; no RuntimeError is hidden or remapped.
    true, world, pixels, data = scene(7)
    rates = np.array([30., 24., 20.])
    kp = np.full(data['keypoints'].shape, np.nan)
    co = np.zeros(data['confidence'].shape)
    for c, n in enumerate(data['frame_counts']):
        kp[c, :n, :11] = pixels(c, np.arange(n)/rates[c])[:, :11]
        co[c, :n, :11] = 1
    packed = tmp_path/'cfr.npz'
    np.savez(packed, keypoints=kp, confidence=co, K=data['K'],
             camera_fps=rates, frame_counts=data['frame_counts'])
    path = tmp_path/'cfr_rig.json'
    call('calibrate_npz.py', [packed, '--distance', 0, 1, np.linalg.norm(true.centers[1]),
                             '--frame-step', 3, '--no-floor', '--out', path], tmp_path, 'cfr_calibrate')
    loaded = Calibration.load_json(path)
    assert max(rotation_angle_deg(a, b) for a, b in zip(loaded.R, true.R)) < .05
    assert np.max(np.linalg.norm(loaded.centers-true.centers, axis=1)) < .003
    times = np.arange(len(kp[0]))/30
    times = times[times <= min((n-1)/r for n, r in zip(data['frame_counts'], rates))][::3]
    np.testing.assert_array_equal(loaded.info['reference_times'], times)
    out = tmp_path/'cfr_points.npz'
    call('triangulate_npz.py', [path, packed, '--out', out], tmp_path, 'cfr_triangulate')
    with np.load(out) as d:
        error = float(np.nanmedian(np.linalg.norm(d['points']-world(d['timestamps']), axis=-1)))
        assert error < .003
    metric(test='cfr_cli_schema', median_error_m=error)
