"""A4 analytic temporal truth; budgets registered before any production edit.

The oracle is a continuous screen polynomial or independent pinhole projection,
never another call to the production resampler. No unknown-offset fitting here.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibration, calibrate, resample_keypoints, triangulate


ROOT = Path(__file__).resolve().parents[1]


def metric(**data):
    if os.environ.get('PHASE4_METRICS'):
        with open(os.environ['PHASE4_METRICS'], 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(data) + '\n')


def screen(t):
    t = np.asarray(t)
    return np.stack([400 + 31*t - 17*t*t + 7*t**3,
                     200 - 9*t + 22*t*t - 5*t**3], -1)


@pytest.mark.parametrize('origin', [0., 17.25, 1000000.])
def test_continuous_cubic_screen_golden(origin):
    # <1e-6 px is numerical interpolation truth, far below 0.5px detector noise.
    # Four independent samples determine a cubic even at boundary intervals.
    rates = np.array([30., 24., 20.])
    ts = origin + np.arange(60)[None] / rates[:, None] + np.array([0, .017, .041])[:, None]
    kp = screen(ts-origin)[:, :, None, :]
    out, co, target = resample_keypoints(kp, np.ones(kp.shape[:3]), [0, 0, 0], timestamps=ts)
    error = float(np.max(np.abs(out[:, :, 0] - screen(target-origin))))
    metric(test='screen_cubic', origin=origin, max_error_px=error)
    assert np.all(co == 1)
    assert error < 1e-6


def scene(seed=7, origin=17.25, recording=0, joints=33, rigid=False):
    # Independent world and pinhole equations. A new recording changes cloud,
    # motion phase, lengths, origins and sampling phases, while retaining rig.
    rng = np.random.default_rng(seed + 1000*recording)
    R = Rotation.from_rotvec([[0, 0, 0], [.01, .05, 0], [-.1, .1, -.03]]).as_matrix()
    centers = np.array([[0., 0, 0], [1., .2, .1], [-.8, -.3, .3]])
    tr = -np.einsum('cij,cj->ci', R, centers)
    K = np.broadcast_to([[1100., 0, 960], [0, 850., 540], [0, 0, 1]], (3, 3, 3)).copy()
    rig = Calibration(R, tr, K, np.empty((0, joints, 3)), None, 'truth', np.zeros(3), np.ones(3))
    cloud = rng.uniform([-.4, -.8, 3.5], [.4, .8, 5.5], (joints, 3))
    phases = rng.uniform(-np.pi, np.pi, joints)
    if rigid:
        phases[:] = .4 + recording
    def world(t):
        t = np.asarray(t)[:, None]
        return cloud + np.stack([.35*np.sin(2.3*t+phases), .25*np.sin(1.7*t+2*phases),
                                 .25*np.cos(2.7*t+phases)], -1)
    def pixels(c, t):
        x = world(t) @ R[c].T + tr[c]
        return x[..., :2] / x[..., 2:] * [K[c, 0, 0], K[c, 1, 1]] + K[c, :2, 2]
    rates = np.array([30., 24., 20.])
    counts = np.array([90, 69, 56]) if not recording else np.array([72, 61, 44])
    offsets = np.array([0., .017, .041]) if not recording else np.array([0., .011, .029])
    ts = origin + np.arange(counts.max())[None] / rates[:, None] + offsets[:, None]
    kp = np.full((3, counts.max(), joints, 2), np.nan)
    co = np.zeros(kp.shape[:3])
    for c, n in enumerate(counts):
        kp[c, :n] = pixels(c, ts[c, :n]-origin)
        co[c, :n] = 1
    return rig, world, pixels, dict(keypoints=kp, confidence=co, timestamps=ts, K=K, frame_counts=counts)


@pytest.mark.parametrize('seed', [7, 19, 66, 103])
def test_independent_continuous_pinhole_golden(seed):
    true, world, pixels, d = scene(seed)
    kp, co, times = resample_keypoints(d['keypoints'], d['confidence'], [0, 0, 0], timestamps=d['timestamps'])
    # Analytic projection golden, not the interpolation formula. This tests the
    # curvature error that can bias calibration even with noise-free detections.
    errors = []
    for c in range(3):
        good = co[c] > 0
        errors.extend(np.abs(kp[c] - pixels(c, times-17.25))[good].ravel())
    maximum = float(np.max(errors))
    metric(test='pinhole_golden', seed=seed, max_error_px=maximum)
    assert maximum < .002


def test_gaps_misses_exact_hits_and_local_support():
    ts = np.array([0., .1, .2, .5, .6, .7, .8, .9, 1., 1.1])
    kp = screen(ts)[None, :, None]
    co = np.ones(kp.shape[:3])
    kp[0, 6] = np.nan
    co[0, 6] = 0
    query = np.array([0., .05, .1, .2, .3, .5, .6, .65, .7, .8, .9, 1., 1.1])
    out, scores, _ = resample_keypoints(kp, co, [0], timestamps=ts, reference_times=query)
    assert not scores[0, [4, 9]].any()  # timestamp gap and exact missed detection
    assert np.isnan(out[0, [4, 9]]).all()
    assert scores[0, 7, 0] == 1  # valid interval before miss, no bridging
    for i, q in enumerate(query):
        if q in ts and q != .8:
            np.testing.assert_array_equal(out[0, i, 0], screen(q))
    # Interpolation support must not reach past the missing sample or gap.
    changed = kp.copy()
    changed[0, 7:] += 1e6
    changed[0, :3] -= 1e6
    again, _, _ = resample_keypoints(changed, co, [0], timestamps=ts, reference_times=query)
    np.testing.assert_array_equal(again[0, 7], out[0, 7])


def test_known_padding_counts_crop_and_output_grid():
    _, _, _, d = scene()
    kp, co, times = resample_keypoints(d['keypoints'], d['confidence'], [0, 0, 0],
                                      timestamps=d['timestamps'], frame_counts=d['frame_counts'])
    # Camera 2's REAL last frame: 17.25+.041+55/20, no artificial padding.
    expected = d['timestamps'][0]
    expected = expected[(expected >= 17.291) & (expected <= 17.25+.041+55/20)]
    np.testing.assert_array_equal(times, expected)
    assert kp.shape[1] == len(times) and co.shape[1] == len(times)


@pytest.mark.parametrize('bad', [[90, 69, 0], [90, 69, 91], [90., 69., 56.5], [True, True, True]])
def test_invalid_frame_counts_fail(bad):
    _, _, _, d = scene()
    with pytest.raises(ValueError, match='frame_counts'):
        resample_keypoints(d['keypoints'], d['confidence'], [0, 0, 0],
                           timestamps=d['timestamps'], frame_counts=bad)


@pytest.mark.parametrize('rates', [[30., 24., 20.], 30.])
def test_fps_axes_in_seconds(rates):
    from posecalib.synchronization import time_axes
    expected_rates = np.broadcast_to(rates, (3,))
    np.testing.assert_allclose(time_axes(3, 9, fps=rates), np.arange(9)[None] / expected_rates[:, None])


def test_saved_reference_grid(tmp_path):
    rig, _, _, _ = scene()
    rig.info['reference_times'] = np.array([17.3, 17.4, 17.5])
    path = tmp_path / 'rig.json'
    rig.save_json(path)
    loaded = Calibration.load_json(path)
    np.testing.assert_array_equal(loaded.info['reference_times'], rig.info['reference_times'])
    np.testing.assert_array_equal(loaded.transformed(2.).info['reference_times'], rig.info['reference_times'])


@pytest.mark.parametrize('layout', ['face', 'bilateral'])
def test_calibrated_saved_rig_new_recording_with_occlusion(layout, tmp_path, monkeypatch):
    import posecalib.synchronization as sync
    def forbidden(*a, **k):
        pytest.fail('known clocks must not run the unknown-offset estimator')
    monkeypatch.setattr(sync, 'estimate_time_offsets', forbidden)
    true, world, _, d = scene(19, rigid=True)
    absent = np.arange(11, 33) if layout == 'face' else [14]
    d['keypoints'][:, :, absent] = np.nan
    d['confidence'][:, :, absent] = 0
    measured = np.linalg.norm(true.centers[1])
    cal = calibrate(d['keypoints'], d['confidence'], d['K'], timestamps=d['timestamps'],
                    frame_counts=d['frame_counts'], frame_step=3, estimate_sync=False,
                    camera_distance=(0, 1, measured))
    assert cal.info['skeleton'] == 'blazepose33' and cal.scale_source == 'camera_distance'
    assert np.linalg.norm(cal.centers[1]) == pytest.approx(measured, abs=1e-12)
    missing_lengths = np.arange(len(cal.segment_lengths)) if layout == 'face' else [2, 3]
    assert np.isnan(cal.segment_lengths[missing_lengths]).all()
    if layout == 'bilateral':
        assert np.isfinite(cal.segment_lengths[[0, 1]]).all()
    assert np.isnan(cal.points[:, absent]).all()
    valid = np.isfinite(cal.points).all(-1)
    assert np.median(np.linalg.norm(cal.points-world(cal.info['reference_times']-17.25), axis=-1)[valid]) < .003
    path = tmp_path / 'calibrated.json'
    cal.save_json(path)
    loaded = Calibration.load_json(path)
    np.testing.assert_array_equal(loaded.info['reference_times'], cal.info['reference_times'])
    # Different person/cloud and motion phase, lengths and origin, same static rig.
    _, new_world, _, new = scene(19, origin=50., recording=1, rigid=True)
    new['keypoints'][:, :, absent] = np.nan
    new['confidence'][:, :, absent] = 0
    points, inliers, times = triangulate(loaded, new['keypoints'], new['confidence'],
                                        timestamps=new['timestamps'], frame_counts=new['frame_counts'],
                                        return_times=True)
    assert np.isnan(points[:, absent]).all()
    finite = np.isfinite(points).all(-1)
    assert np.count_nonzero(finite) > len(times)*8
    error = float(np.median(np.linalg.norm(points-new_world(times-50.), axis=-1)[finite]))
    metric(test='saved_new_recording', layout=layout, median_error_m=error)
    assert error < .003


def test_stored_offsets_new_recording_ignore_and_aligned_bypass(tmp_path):
    true, world, _, d = scene(7, origin=50., recording=1)
    offsets = np.array([0., .027, -.019])
    true.time_offsets_seconds = offsets
    path = tmp_path / 'rig.json'
    true.save_json(path)
    loaded = Calibration.load_json(path)
    # Raw local clocks have these residual biases. Query source t+offset.
    raw_times = d['timestamps'] + offsets[:, None]
    points, _, times = triangulate(loaded, d['keypoints'], d['confidence'], timestamps=raw_times,
                                   frame_counts=d['frame_counts'], return_times=True)
    assert np.nanmedian(np.linalg.norm(points-world(times-50.), axis=-1)) < .003
    aligned, scores, target = resample_keypoints(d['keypoints'], d['confidence'], offsets,
                                                 timestamps=raw_times, frame_counts=d['frame_counts'])
    bypass, _, unused = triangulate(loaded, aligned, scores, apply_time_offsets=False,
                                    timestamps=np.zeros(2), return_times=True)
    assert unused is None
    np.testing.assert_array_equal(points, bypass)
    # New capture clocks are already corrected; ignore obsolete residual offsets
    # but STILL align unequal rates on the supplied known timestamps.
    ignored, _, grid = triangulate(loaded, d['keypoints'], d['confidence'], timestamps=d['timestamps'],
                                    frame_counts=d['frame_counts'], ignore_time_offsets=True, return_times=True)
    assert np.nanmedian(np.linalg.norm(ignored-world(grid-50.), axis=-1)) < .003
    metric(test='stored_offsets', applied_median_m=float(np.nanmedian(np.linalg.norm(points-world(times-50.), axis=-1))))


def test_explicit_clocks_override_fps_and_fps_array_matches_seconds():
    true, _, _, d = scene()
    by_times = triangulate(true, d['keypoints'], d['confidence'], timestamps=d['timestamps'],
                            fps=-5, frame_counts=d['frame_counts'], return_times=True)
    no_fps = triangulate(true, d['keypoints'], d['confidence'], timestamps=d['timestamps'],
                          frame_counts=d['frame_counts'], return_times=True)
    np.testing.assert_array_equal(by_times[0], no_fps[0])
    np.testing.assert_array_equal(by_times[2], no_fps[2])
    rates = np.array([30., 24., 20.])
    ts = np.arange(d['keypoints'].shape[1])[None] / rates[:, None]
    a = triangulate(true, d['keypoints'], d['confidence'], fps=rates, return_times=True)
    b = triangulate(true, d['keypoints'], d['confidence'], timestamps=ts, return_times=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])


@pytest.mark.parametrize('count', [1, 2, 3])
def test_short_valid_runs_and_exact_singletons(count):
    times = np.arange(6.)
    kp = screen(times)[None, :, None]
    co = np.ones(kp.shape[:3])
    kp[:, count:] = np.nan
    co[:, count:] = 0
    out, score, _ = resample_keypoints(kp, co, [0], timestamps=times)
    np.testing.assert_array_equal(out[:, :count], kp[:, :count])
    assert not score[:, count:].any()


@pytest.mark.parametrize('kind', ['nan_coordinate', 'nan_confidence', 'low_confidence'])
def test_detection_holes_split_spline_and_preserve_neighbor_hits(kind):
    times = np.arange(20)/30
    kp = screen(times)[None, :, None]
    co = np.ones(kp.shape[:3])
    if kind == 'nan_coordinate':
        kp[0, 9] = np.nan
    elif kind == 'nan_confidence':
        co[0, 9] = np.nan
    else:
        co[0, 9] = .2
    query = np.sort(np.r_[times, (times[8]+times[9])/2, (times[9]+times[10])/2])
    out, score, _ = resample_keypoints(kp, co, [0], timestamps=times, reference_times=query)
    holes = (query > times[8]) & (query < times[10])
    assert np.isnan(out[:, holes]).all() and not score[:, holes].any()
    for i in [8, 10]:
        np.testing.assert_array_equal(out[0, np.flatnonzero(query == times[i])[0]], kp[0, i])
    altered = kp.copy()
    altered[0, 10:] += 1e6
    again, _, _ = resample_keypoints(altered, co, [0], timestamps=times, reference_times=query)
    np.testing.assert_array_equal(out[:, query <= times[8]], again[:, query <= times[8]])


def test_padding_is_metadata_not_detection_inference():
    _, _, _, d = scene()
    a, ac, at = resample_keypoints(d['keypoints'], d['confidence'], [0, 0, 0],
                                  timestamps=d['timestamps'], frame_counts=d['frame_counts'])
    kp, co = d['keypoints'].copy(), d['confidence'].copy()
    for c, n in enumerate(d['frame_counts']):
        kp[c, n:] = 1e6
        co[c, n:] = 1
    b, bc, bt = resample_keypoints(kp, co, [0, 0, 0], timestamps=d['timestamps'], frame_counts=d['frame_counts'])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ac, bc)
    np.testing.assert_array_equal(at, bt)


@pytest.mark.parametrize('bad', [[1, 1], [1, float('nan')], [[1, 2]]])
def test_saved_reference_grid_rejects_invalid_metadata(bad, tmp_path):
    rig, _, _, _ = scene()
    payload = rig.to_dict()
    payload['reference_times'] = bad
    path = tmp_path/'bad_grid.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='reference_times'):
        Calibration.load_json(path)
