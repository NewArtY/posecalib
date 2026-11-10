"""Public regressions for canonical K, metric inputs, clocks and world frames."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import posecalib.api as api
import posecalib.synchronization as synchronization
from posecalib import (BLAZEPOSE33, COCO17, Calibration, Calibrator, align_to_floor,
                      calibrate, project, resample_keypoints, to_centered, triangulate)
from posecalib.geometry import rotation_angle_deg
from posecalib.mediapipe_io import load_blazepose_cameras, prepare_blazepose
from posecalib.validation import PINHOLE_ROW_ATOL, validate_intrinsics


K = np.array([[1100., 0, 960], [0, 850., 540], [0, 0, 1.]])


@pytest.fixture(scope='module')
def video():
    script = Path(__file__).resolve().parents[1] / 'examples' / 'prepare_blazepose_videos.py'
    spec = importlib.util.spec_from_file_location('validation_video', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rig(K=K, points=None):
    R = Rotation.from_rotvec([[0, 0, 0], [.01, .05, 0], [-.1, .1, -.03]]).as_matrix()
    centers = np.array([[0., 0, 0], [1., .2, .1], [-.8, -.3, .3]])
    t = -np.einsum('cij,cj->ci', R, centers)
    return Calibration(R, t, K, np.empty((0, 33, 3)) if points is None else points,
                       None, 'true', np.zeros(3), np.ones(3), {'skeleton': 'blazepose33'})


def tracks(joints=33):
    kp = np.zeros((3, 3, joints, 2))
    return kp, np.ones(kp.shape[:3])


BAD_K = [(0, 0, 0), (1, 1, 0), (0, 0, -1), (1, 1, -1),
         (0, 0, np.nan), (1, 1, np.inf), (0, 2, np.nan), (1, 2, -np.inf),
         (0, 1, 1), (0, 1, 1e-15), (1, 0, 250), (1, 0, -1e-15),
         (2, 0, .1), (2, 1, .1), (2, 2, 0), (2, 2, 2), (2, 2, 1 + 1e-8),
         (2, 0, np.nan), (2, 1, np.inf), (2, 2, np.nan)]


@pytest.mark.parametrize('i,j,value', BAD_K)
@pytest.mark.parametrize('per_camera', [False, True])
def test_invalid_K_rejected_by_core_adapter_and_loaders(i, j, value, per_camera, tmp_path, video):
    matrix = K.copy()
    matrix[i, j] = value
    supplied = np.stack([K, matrix, K]) if per_camera else matrix
    kp, co = tracks()
    landmarks = np.ones((3, 3, 33, 5))
    for call in [lambda: Calibrator(kp, co, supplied), lambda: to_centered(kp, supplied),
                 lambda: rig(supplied),
                 lambda: prepare_blazepose(landmarks, (1920, 1080), supplied)]:
        with pytest.raises(ValueError, match='K must'):
            call()
    path = tmp_path / 'intrinsics.npz'
    np.savez(path, K=supplied)
    with pytest.raises(ValueError, match='K must'):
        video.load_intrinsics(path, 3, True)
    path = tmp_path / 'camera.npz'
    np.savez(path, landmarks=landmarks[0], K=matrix, image_size=[1920, 1080])
    with pytest.raises(ValueError, match='K must'):
        load_blazepose_cameras([path])
    payload = rig().to_dict()
    payload['K'] = supplied.tolist()
    path = tmp_path / 'rig.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='K must'):
        Calibration.load_json(path)


@pytest.mark.parametrize('per_camera', [False, True])
def test_asymmetric_K_project_P_centered_and_JSON_agree(per_camera, tmp_path, video):
    supplied = np.stack([K, K * [1.1, .9, 1], K * [.8, 1.2, 1]]) if per_camera else K.copy()
    # Column multiplication above leaves the last row canonical, with unequal fx/fy.
    supplied[..., 2, :] += [PINHOLE_ROW_ATOL / 2, -PINHOLE_ROW_ATOL / 2, PINHOLE_ROW_ATOL / 2]
    original = supplied.copy()
    cal = rig(supplied)
    np.testing.assert_array_equal(supplied, original)
    np.testing.assert_array_equal(cal.K[:, 2], np.tile([0, 0, 1], (3, 1)))
    X = np.random.default_rng(7).uniform([-.3, -.5, 3], [.5, .3, 6], (4, 33, 3))
    pixels = project(cal, X)
    hom = np.einsum('cij,...j->c...i', cal.projection_matrices(),
                    np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1))
    np.testing.assert_allclose(pixels, hom[..., :2] / hom[..., 2:], atol=1e-10, rtol=0)
    uv, focal = to_centered(pixels, cal.K)
    xc = np.einsum('cij,...j->c...i', cal.R, X) + cal.t[:, None, None]
    np.testing.assert_allclose(uv / focal[:, None, None, None], xc[..., :2] / xc[..., 2:], atol=1e-15)
    path = tmp_path / 'rig.json'
    cal.save_json(path)
    np.testing.assert_array_equal(Calibration.load_json(path).K, cal.K)
    payload = cal.to_dict()
    payload['K'] = supplied.tolist()  # load accepts both shared and per-camera K
    path.write_text(json.dumps(payload), encoding='utf-8')
    np.testing.assert_array_equal(Calibration.load_json(path).K, cal.K)
    path = tmp_path / 'intrinsics.npz'
    np.savez(path, K=supplied)
    np.testing.assert_array_equal(video.load_intrinsics(path, 3, True)[0], cal.K)


def test_K_owned_copy_and_mutation_revalidated():
    supplied = np.tile(K, (3, 1, 1))
    cal = rig(supplied)
    supplied[0, 0, 0] = -1
    assert cal.K[0, 0, 0] == K[0, 0]
    cal.K[0, 1, 0] = 1
    kp, co = tracks()
    for call in [lambda: cal.projection_matrices(), lambda: project(cal, [[0, 0, 4]]),
                 lambda: cal.to_dict(), lambda: triangulate(cal, kp, co)]:
        with pytest.raises(ValueError, match='K must'):
            call()
    with pytest.raises(ValueError, match='K must'):
        validate_intrinsics(np.eye(4), 3)


BAD_DISTANCES = [(0, 0, 1), (-1, 1, 1), (0, 3, 1), (0., 1, 1), (0, 1.2, 1),
                 (True, 1, 1), (0, False, 1), ('0', 1, 1), (0, 1, 0), (0, 1, -1),
                 (0, 1, np.nan), (0, 1, np.inf), (0, 1, -np.inf), (0, 1, [1]),
                 (0, 1, True), (0, 1), 1]


@pytest.mark.parametrize('distance', BAD_DISTANCES)
def test_bad_distance_public_API_fails_before_geometry_or_sync(distance, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid measurement reached geometry or offset estimation')
    monkeypatch.setattr(api, 'initialize', forbidden)
    monkeypatch.setattr(synchronization, 'estimate_time_offsets', forbidden)
    kp, co = tracks()
    with pytest.raises(ValueError, match='camera_distance'):
        Calibrator(kp, co, K, camera_distance=distance)
    for estimate in (False, True):
        with pytest.raises(ValueError, match='camera_distance'):
            calibrate(kp, co, K, camera_distance=distance, estimate_sync=estimate)


@pytest.mark.parametrize('value', [0, -1, np.inf, -np.inf])
@pytest.mark.parametrize('mapping', [False, True])
def test_bad_segment_measurements_fail_early(value, mapping, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid measurement reached geometry')
    monkeypatch.setattr(api, 'initialize', forbidden)
    lengths = {COCO17.segment_names[0]: value} if mapping else np.full(COCO17.num_segments, value)
    kp, co = tracks(17)
    with pytest.raises(ValueError, match='segment_lengths'):
        calibrate(kp, co, K, segment_lengths=lengths)


@pytest.mark.parametrize('lengths', [{-1: .4}, {9: .4}, {True: .4}, {1.5: .4},
                                    {'typo': .4}, {'left_thigh': None},
                                    {'left_thigh': [.4]}, [None] * 9, np.ones(8)])
def test_bad_segment_indices_values_and_shapes(lengths):
    kp, co = tracks()
    with pytest.raises(ValueError, match='segment_lengths'):
        Calibrator(kp, co, K, segment_lengths=lengths)


def test_partial_and_unknown_segments_and_scale_failures():
    kp, co = tracks()
    lengths = np.full(9, np.nan)
    lengths[0] = .4
    for measurements in [lengths, {'left_upper_arm': .4, np.int64(1): np.nan}]:
        cal = Calibrator(kp, co, K, segment_lengths=measurements)
        estimated = np.full(9, np.nan)
        estimated[0] = .2
        assert cal._pre_scale(estimated) == 2
        assert cal._scale(rig().R, rig().t, estimated) == (2., 'segment_lengths')
        assert cal._pre_scale(np.full(9, np.nan)) == 1
        assert cal._scale(rig().R, rig().t, np.full(9, np.nan)) == (1., 'arbitrary')
    custom = replace(BLAZEPOSE33, typical_lengths=np.array([0, -1, np.inf, np.nan] + [-1] * 5))
    cal = Calibrator(kp, co, K, skeleton=custom)
    assert cal._pre_scale(np.ones(9)) == 1
    assert cal._scale(rig().R, rig().t, np.ones(9)) == (1., 'arbitrary')
    # Extreme finite numbers cannot overflow the length sums or yield a zero scale.
    assert cal._length_scale(np.full(9, 1e308), np.full(9, 1e308)) == 1
    assert cal._length_scale(np.full(9, 1e-308), np.full(9, 1e308)) is None
    cal.camera_distance = (np.int64(0), np.int64(1), 2.)
    assert cal._pre_scale(np.full(9, np.nan)) == 1
    scale, source = cal._scale(rig().R, rig().t, np.full(9, np.nan))
    assert scale > 0 and np.isfinite(scale) and source == 'camera_distance'
    for t in [np.zeros((3, 3)), np.full((3, 3), np.inf)]:
        with pytest.raises(ValueError, match='baseline'):
            cal._scale(rig().R, t, np.ones(9))
    cal.camera_distance = (0, 1, np.finfo(float).max)
    tiny_t = np.array([[0., 0, 0], [-1e-100, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match='scale'):
        cal._scale(np.tile(np.eye(3), (3, 1, 1)), tiny_t, np.ones(9))


@pytest.mark.parametrize('kwargs', [
    {'s': 0}, {'s': -1}, {'s': np.nan}, {'s': np.inf}, {'s': [1]},
    {'Rw': np.diag([2, 1, 1])}, {'Rw': np.diag([-1, 1, 1])},
    {'Rw': np.full((3, 3), np.nan)}, {'Rw': np.eye(4)},
    {'tw': [0, 0, np.inf]}, {'tw': [0, np.nan, 0]}, {'tw': [[0, 0, 0]]},
])
def test_invalid_world_transform_rejected(kwargs):
    with pytest.raises(ValueError):
        rig().transformed(**kwargs)


def test_positive_similarity_keeps_projections_all33_and_offsets(tmp_path):
    points = np.random.default_rng(3).uniform([-.5, -.5, 3], [.5, .5, 5], (8, 33, 3))
    cal = rig(points=points)
    cal.time_offsets_seconds = np.array([0, .02, -.01])
    cal.info['reference_times'] = 10 + np.arange(8) / 30
    Rw = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    moved = cal.transformed(2.3, Rw, [.1, .2, -.3])
    np.testing.assert_allclose(project(moved, moved.points), project(cal, points), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(moved.R), 1)
    np.testing.assert_array_equal(moved.time_offsets_seconds, cal.time_offsets_seconds)
    np.testing.assert_array_equal(moved.info['reference_times'], cal.info['reference_times'])
    path = tmp_path / 'rig.json'
    moved.save_json(path)
    np.testing.assert_array_equal(Calibration.load_json(path).time_offsets_seconds, cal.time_offsets_seconds)


def floor_rig(vertical=False):
    rng = np.random.default_rng(32)
    R = Rotation.from_rotvec([[np.pi / 2, 0, 0]] * 2).as_matrix()
    centers = np.array([[0., -4, 1.5], [0., -4, 2.5] if vertical else [2., -4, 1.5]])
    points = rng.uniform([-1, -1, .08], [1, 1, .08], (40, 33, 3))
    return Calibration(R, -np.einsum('cij,cj->ci', R, centers), K,
                       points, None, 'camera_distance', np.zeros(2), np.ones(2),
                       {'skeleton': 'blazepose33', 'foot_keypoints': [27, 28]})


@pytest.mark.parametrize('kwargs', [
    {'origin_camera': 0, 'x_camera': 0}, {'origin_camera': -1}, {'x_camera': 2},
    {'origin_camera': .5}, {'x_camera': True}, {'foot_height': -1},
    {'foot_height': np.inf}, {'foot_height': np.nan}, {'foot_height': [.08]},
])
def test_bad_floor_parameters(kwargs):
    with pytest.raises(ValueError):
        align_to_floor(floor_rig(), **kwargs)


def test_floor_rejects_vertical_baseline_and_preserves_valid_projections():
    with pytest.raises(ValueError, match='horizontal camera baseline'):
        align_to_floor(floor_rig(vertical=True))
    cal = floor_rig()
    floor = align_to_floor(cal)
    np.testing.assert_allclose(floor.centers[:, 2], 1.5, atol=1e-10)
    np.testing.assert_allclose(project(floor, floor.points), project(cal, cal.points), atol=1e-10)
    assert align_to_floor(cal, foot_height=0).points.shape == (40, 33, 3)


def capture(video, origins):
    """Independent world samples at 30/24/20 FPS, packed by the actual producer."""
    cal = rig()
    cloud = np.random.default_rng(66).uniform([-.4, -.8, 3.5], [.4, .8, 5.5], (33, 3))
    phases = np.linspace(0, 2 * np.pi, 33)
    def world(times):
        tt = np.asarray(times)[:, None]
        return cloud[None] + np.stack([.35 * np.sin(2.3 * tt + phases),
                                       .25 * np.sin(1.7 * tt + 2 * phases),
                                       .25 * np.cos(2.7 * tt + phases)], -1)
    rates = np.array([30., 24., 20.])
    rows = []
    for c, rate in enumerate(rates):
        time = np.arange(int(4 * rate)) / rate + origins[c]
        row = np.ones((len(time), 33, 5))
        row[..., :2] = project(cal, world(time - origins[0]))[c] / [1920, 1080]
        row[..., 2] = 0
        rows.append(row)
    arrays = video.pack_detections(rows, [(1920, 1080)] * 3, cal.K, rates)
    arrays['timestamps'] += np.asarray(origins)[:, None]
    return cal, world, arrays


@pytest.mark.parametrize('origins', [(0., 0., 0.), (17.25, 17.267, 17.291)])
def test_public_unequal_rate_known_clocks_pipeline(video, origins, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('known timestamps must not trigger unknown-offset estimation')
    monkeypatch.setattr(synchronization, 'estimate_time_offsets', forbidden)
    true, world, arrays = capture(video, origins)
    kp, co, ts = (arrays[key] for key in ('keypoints', 'confidence', 'timestamps'))
    aligned, scores, expected_times = resample_keypoints(kp, co, np.zeros(3), timestamps=ts)
    points, mask, times = triangulate(true, kp, co, timestamps=ts, return_times=True)
    np.testing.assert_array_equal(times, expected_times)
    assert times[0] >= max(origins) and times[0] >= origins[0]
    assert points.shape == (len(times), 33, 3) and mask.shape == (3, len(times), 33)
    assert np.nanmedian(np.linalg.norm(points - world(times - origins[0]), axis=-1)) < .0015
    direct, _ = triangulate(true, aligned, scores, apply_time_offsets=False)
    np.testing.assert_allclose(points, direct, equal_nan=True)
    distance = np.linalg.norm(true.centers[1] - true.centers[0])
    result = calibrate(kp, co, true.K, timestamps=ts, frame_step=3,
                       use_segments=False, camera_distance=(0, 1, distance))
    np.testing.assert_array_equal(result.info['reference_times'], times[::3])
    assert result.points.shape == (len(times[::3]), 33, 3)
    assert result.time_offsets_seconds is None
    assert max(rotation_angle_deg(a, b) for a, b in zip(result.R, true.R)) < .05
    assert np.max(np.linalg.norm(result.centers - true.centers, axis=1)) < .003
    # Held-out samples use the recovered rig and retain every landmark slot.
    held_out, _, held_times = triangulate(result, kp, co, timestamps=ts, return_times=True)
    assert np.nanmedian(np.linalg.norm(held_out - world(held_times - origins[0]), axis=-1)) < .003


def test_aligned_bypass_skips_timestamps_and_saved_offsets(video):
    true, _, arrays = capture(video, (10., 10.017, 10.04))
    aligned, co, times = resample_keypoints(arrays['keypoints'], arrays['confidence'], np.zeros(3),
                                           timestamps=arrays['timestamps'])
    expected, _ = triangulate(true, aligned, co, apply_time_offsets=False)
    true.time_offsets_seconds = np.array([0, .1, -.1])
    points, _, returned = triangulate(true, aligned, co, apply_time_offsets=False,
                                      timestamps=np.zeros(2), return_times=True)
    assert returned is None and points.shape[0] == len(times)
    np.testing.assert_allclose(points, expected, equal_nan=True)


def test_supplied_invalid_timestamps_are_not_silently_ignored():
    kp, co = tracks()
    for call in [lambda: calibrate(kp, co, K, timestamps=np.zeros((3, 3))),
                 lambda: triangulate(rig(), kp, co, timestamps=np.zeros((3, 3)))]:
        with pytest.raises(ValueError, match='strictly increasing'):
            call()


def test_initialize_keeps_all_unobserved_segment_lengths_unknown(video):
    true, _, arrays = capture(video, (0., 0., 0.))
    kp, co = arrays['keypoints'], arrays['confidence']
    kp[:, :, 11:] = np.nan
    co[:, :, 11:] = 0
    cal = Calibrator(kp[:, ::3], co[:, ::3], true.K, camera_distance=(0, 1, 1.)).initialize()
    assert np.isnan(cal.init['L']).all()
    assert np.isfinite(cal.init['X'][cal.init['ok']]).all()
    assert cal._pre_scale(cal.init['L']) == 1
    scale, source = cal._scale(cal.init['Rs'], cal.init['ts'], cal.init['L'])
    assert np.isfinite(scale) and scale > 0 and source == 'camera_distance'
