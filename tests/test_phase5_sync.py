"""Phase 5 sync acceptance cases with independently sampled pinhole tracks."""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibration, calibrate, estimate_time_offsets, project, resample_keypoints, triangulate


@dataclass(frozen=True)
class Case:
    mode: str
    seed: int
    rates: tuple
    noise: tuple
    bound: float
    offsets: tuple
    duration: float
    origin: float = 0.
    missing: bool = False
    alternate: bool = False
    period: float = .13
    uneven: bool = False
    support_frames: int = 0


# Independent seeds and a second motion family, with uneven capture starts
# for explicit clocks. Positives use interior shifts in the stated interval.
CASES = {
    'positive_24_narrow': Case('varied', 213, (24, 24), (.5, 2), .025, (0, .016), 4.5),
    'positive_30_reverse': Case('varied', 244, (30, 30), (2, .5), .025, (0, -.017), 5.),
    'positive_60': Case('varied', 279, (60, 60), (.5, 2), .1, (0, -.057), 3.4, 23.),
    'positive_mixed_3': Case('varied', 381, (30, 24, 60), (2, .5, 2), .1,
                             (0, -.063, .046), 5.6, 71., True, True),
    'positive_mixed_4': Case('varied', 497, (24, 60, 30, 24), (.5, 2, .5, 2), .15,
                             (0, .092, -.075, .054), 6.2, 105., True),
    'positive_long_4': Case('varied', 519, (30, 30, 30, 30), (2, .5, 2, .5), .15,
                             (0, -.091, .076, -.043), 8., 0., True, True),
    'translation_24_narrow': Case('translation', 557, (24, 24), (.5, 2), .025,
                                  (0, .014), 5., 0.),
    'translation_60_reverse': Case('translation', 602, (60, 60), (2, .5), .025,
                                   (0, -.017), 3.8, 57.),
    'translation_mixed_4': Case('translation', 628, (24, 30, 60, 30), (2, .5, 2, .5), .15,
                                 (0, -.069, .064, -.042), 4.5, 11., True),
    'static_noise': Case('static', 691, (30, 24), (2, .5), .1, (0, -.042), 5.1, 12.),
    'periodic_alias': Case('periodic', 720, (60, 60), (.5, 2), .15, (0, .012), 5.4),
    'periodic_alias_mixed': Case('periodic', 753, (30, 60, 24), (2, .5, 2), .15,
                                 (0, -.013, .017), 6., 37.),
    'short_support': Case('varied', 791, (30, 60), (.5, 2), .1, (0, .053), 4.,
                          missing=True, support_frames=22),
    # Fresh controls after diagnosis of the first periodic failure.
    'positive_uneven': Case('varied', 853, (24, 60), (2, .5), .1, (0, -.058), 4.9,
                            315., True, True, uneven=True),
    'periodic_second_harmonic': Case('periodic', 904, (60, 30), (2, .5), .15,
                                     (0, .061), 5.1, 24., alternate=True, period=.18),
    'periodic_short_cycle': Case('periodic', 981, (60, 60), (.5, 2), .15,
                                 (0, -.036), 4.6, 28., period=.101),
}


def capture(case, *, recording=0):
    rng = np.random.default_rng(case.seed)
    count = len(case.rates)
    intrinsics = np.array([[1100., 0, 960], [0, 1130., 540], [0, 0, 1]])
    rotations = Rotation.from_euler('yx', [[0, 0], [-9, 4], [12, -4], [-18, -6]][:count],
                                    degrees=True).as_matrix()
    centers = np.array([[0, 0, 0], [.95, .20, .13], [-.65, .35, .05], [1.25, -.43, .18]])[:count]
    centers[1:] += rng.uniform(-.14, .14, size=(count-1, 3))
    rig = Calibration(rotations, -np.einsum('cij,cj->ci', rotations, centers),
                      np.repeat(intrinsics[None], count, axis=0), np.empty((0, 17, 3)),
                      None, 'true', np.zeros(count), np.ones(count), {'skeleton': 'coco17'})
    cloud = rng.uniform([-.53, -.68, 3.4], [.53, .68, 4.7], (17, 3))
    phases = rng.uniform(-np.pi, np.pi, (17, 3)) + recording*.81
    amplitudes = rng.uniform(.10, .33, (17, 3))
    frequencies = rng.uniform(1.2, 3.7, (17, 3))

    def world(t):
        t = np.asarray(t)[:, None, None]
        if case.mode == 'static':
            return np.broadcast_to(cloud, (len(t), *cloud.shape))
        if case.mode == 'translation':
            return cloud + t*np.array([.34, -.19, .16])
        if case.mode in ('periodic', 'triangle'):
            phase = 2*np.pi*t/case.period
            if case.mode == 'triangle':
                return cloud + amplitudes*(2/np.pi)*np.arcsin(np.sin(phase + phases))
            result = cloud + amplitudes*np.sin(phase + phases)
            if case.alternate:
                result += .055*np.cos(2*phase + phases*.6)
            return result
        if case.alternate:
            return (cloud + amplitudes*np.sin(frequencies*t + .22*t**2 + phases)
                    + .12*np.sin(t*1.3 + phases*.45))
        return (cloud + amplitudes*np.sin(frequencies*t + phases)
                + np.array([.21, -.24, .16])*np.cos(t*1.56))

    starts = (np.arange(count)*.011 if case.origin else np.zeros(count))
    lengths = np.array([int((case.duration-.024*c)*r) for c, r in enumerate(case.rates)])
    frames = np.arange(lengths.max())
    stamps = case.origin + starts[:, None] + frames[None, :]/np.array(case.rates)[:, None]
    if case.uneven:
        stamps += rng.uniform(-.0015, .0015, stamps.shape)
        for c, n in enumerate(lengths):
            stamps[c, n//3:] += 2/case.rates[c]
    # Deliberately valid-looking padding must never enter either search or crop.
    keypoints = rng.uniform(0., 1900., (count, len(frames), 17, 2))
    confidence = np.ones(keypoints.shape[:3])
    for c, n in enumerate(lengths):
        t = stamps[c, :n]-case.origin-case.offsets[c]
        keypoints[c, :n] = project(rig, world(t))[c]
        keypoints[c, :n] += rng.normal(0., case.noise[c], (n, 17, 2))
        if case.missing:
            confidence[c, n//3:n//3+5, 2:7] = 0.
            confidence[c, n//2, :] = 0.
            if c == 1:
                confidence[c, :n, 9:14] = 0.  # unilateral/partial skeleton
        if case.support_frames:
            confidence[c, case.support_frames:n] = 0.
    return rig, world, keypoints, confidence, stamps, lengths


def _estimate(case):
    rig, _, kp, co, ts, counts = capture(case)
    return estimate_time_offsets(kp, co, rig.K, timestamps=ts, frame_counts=counts,
                                 max_offset_seconds=case.bound)


def test_raw_repeat_refuses_in_bound_triangle_alias():
    # A -71 ms lag admits a whole-cycle alternative at +122 ms.
    case = Case('triangle', 17241, (60, 60), (2., .5), .15, (0, -.071), 4.2,
                38., period=.193)
    with pytest.raises(ValueError, match='raw tracks repeat|multiple equivalent minima'):
        _estimate(case)


@pytest.mark.parametrize('name', CASES)
def test_held_out_matrix(name):
    case = CASES[name]
    if case.mode != 'varied' or case.support_frames:
        with pytest.raises(ValueError, match='motion|boundary|inconsistent') as refusal:
            sync = _estimate(case)
            print(dict(estimated=sync.offsets_seconds.tolist(), diagnostics=sync.diagnostics))
        print(dict(refused=str(refusal.value)))
        return
    sync = _estimate(case)
    error = float(np.max(np.abs(sync.offsets_seconds - case.offsets)))
    print(dict(estimated=sync.offsets_seconds.tolist(), error_ms=1000*error,
               consistency_ms=1000*sync.diagnostics['pair_consistency_seconds']))
    assert sync.offsets_seconds[0] == 0
    assert error < .012


@pytest.mark.parametrize('floor', [.75, 0.])
def test_confidence_is_explicit_mask_before_motion_and_search(floor):
    case = Case('varied', 81, (30, 30), (.5, 2), .1, (0, .041), 4.)
    rig, _, kp, co, ts, _ = capture(case)
    # Wrong-time moving tracks, below the public threshold but above the old
    # private 0.3 threshold in the high-floor case. Includes a whole absent limb.
    co[1, :, 5:14] = .5 if floor else 0.
    kp[1, :, 5:14] = np.roll(kp[1, :, 5:14], -3, axis=0)
    co[0, 31:36, 0:3] = .4 if floor else 0.
    kp[0, 31:36, 0:3] += 200.
    options = dict(timestamps=ts, max_offset_seconds=case.bound, min_confidence=floor)
    live = estimate_time_offsets(kp, co, rig.K, **options)
    missing = (co == 0) | (co < floor)
    explicit = estimate_time_offsets(np.where(missing[..., None], np.nan, kp),
                                     np.where(missing, 0., co), rig.K, **options)
    np.testing.assert_array_equal(live.offsets_seconds, explicit.offsets_seconds)
    np.testing.assert_array_equal(live.timestamps, explicit.timestamps)
    np.testing.assert_array_equal(live.keypoints, explicit.keypoints)
    np.testing.assert_array_equal(live.confidence, explicit.confidence)
    assert live.diagnostics == explicit.diagnostics
    assert np.max(np.abs(live.offsets_seconds-case.offsets)) < .012
    assert np.isnan(live.keypoints[1, :, 5:14]).all()


def test_estimated_offsets_save_and_apply_to_new_recording(tmp_path):
    case = Case('varied', 118, (30, 24, 60), (.25, .25, .25), .1, (0, .037, -.029),
                4.8, 23., True)
    true, _, kp, co, ts, counts = capture(case)
    step = 4
    rig = calibrate(kp, co, true.K, timestamps=ts, frame_counts=counts, estimate_sync=True,
                    max_offset_seconds=case.bound, use_segments=False, frame_step=step,
                    camera_distance=(0, 1, np.linalg.norm(true.centers[1])))
    assert np.max(np.abs(rig.time_offsets_seconds-case.offsets)) < .012
    _, _, cropped = resample_keypoints(kp, co, rig.time_offsets_seconds, timestamps=ts,
                                       frame_counts=counts)
    np.testing.assert_array_equal(rig.info['reference_times'], cropped[::step])
    assert len(rig.points) == len(cropped[::step])
    path = tmp_path/'rig.json'
    rig.save_json(path)
    loaded = Calibration.load_json(path)
    np.testing.assert_array_equal(loaded.time_offsets_seconds, rig.time_offsets_seconds)
    np.testing.assert_array_equal(loaded.info['reference_times'], cropped[::step])

    new_case = Case('varied', case.seed, (60, 30, 24), (.25, .25, .25), .1, case.offsets,
                    3.2, 65., True, True)
    _, world, kp, co, ts, counts = capture(new_case, recording=1)
    points, _, times = triangulate(loaded, kp, co, timestamps=ts, frame_counts=counts,
                                    return_times=True)
    _, _, expected = resample_keypoints(kp, co, loaded.time_offsets_seconds,
                                        timestamps=ts, frame_counts=counts)
    np.testing.assert_array_equal(times, expected)
    assert len(times) > 100 and np.isfinite(points).sum() > len(times)*17
    assert np.nanmedian(np.linalg.norm(points-world(times-new_case.origin), axis=-1)) < .012
    corrected_times = ts - np.asarray(case.offsets)[:, None]
    ignored, _, grid = triangulate(loaded, kp, co, timestamps=corrected_times,
                                    frame_counts=counts, ignore_time_offsets=True,
                                    return_times=True)
    assert np.nanmedian(np.linalg.norm(ignored-world(grid-new_case.origin), axis=-1)) < .012


def test_unknown_counts_and_timestamps_reject_invalid_schema():
    case = CASES['positive_24_narrow']
    rig, _, kp, co, ts, counts = capture(case)
    with pytest.raises(ValueError, match='frame_counts'):
        estimate_time_offsets(kp, co, rig.K, timestamps=ts, frame_counts=counts.astype(float))
    ts[1, 10] = ts[1, 9]
    with pytest.raises(ValueError, match='strictly increasing'):
        estimate_time_offsets(kp, co, rig.K, timestamps=ts, frame_counts=counts)


@pytest.mark.parametrize('boundary', [False, True])
def test_bad_graph_never_reaches_alignment_or_calibration(boundary, monkeypatch):
    import posecalib.synchronization as sync

    case = CASES['positive_mixed_3']
    rig, _, kp, co, ts, counts = capture(case)
    proposals = {(0, 1): -.07, (0, 2): .07, (1, 2): -.08}

    def pair(xy, confidence, times, a, b, bound, rng, models):
        offset = proposals[a, b]
        return offset, dict(cameras=[a, b], offset_seconds=offset, motion=.1,
                            median_epipolar_error=.0001, at_bound=boundary)

    def forbidden(*args, **kwargs):
        pytest.fail('unreliable offsets must be refused before use')

    monkeypatch.setattr(sync, '_pair_offset', pair)
    monkeypatch.setattr(sync, 'resample_keypoints', forbidden)
    message = 'search boundary' if boundary else 'inconsistent'
    with pytest.raises(ValueError, match=message):
        calibrate(kp, co, rig.K, timestamps=ts, frame_counts=counts, estimate_sync=True)
