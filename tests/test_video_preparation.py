import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'examples' / 'prepare_blazepose_videos.py'
spec = importlib.util.spec_from_file_location('video_preparation', SCRIPT)
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)
K = np.array([[900., 0, 320], [0, 900, 240], [0, 0, 1]])


def test_pack_unequal_rates_lengths_and_misses():
    row = np.tile([.25, .75, 0, .8, .6], (3, 33, 1))
    row[1] = np.nan
    data = video.pack_detections([row, row[:1]], [(640, 480)] * 2,
                                 np.stack([K, K]), [30, 25])
    assert data['keypoints'].shape == (2, 3, 33, 2)
    np.testing.assert_allclose(data['keypoints'][0, 0, 0], [160, 360])
    np.testing.assert_allclose(data['confidence'][0, 0], .6)
    assert np.isnan(data['keypoints'][0, 1]).all()
    assert np.isnan(data['keypoints'][1, 1:]).all()
    assert (data['confidence'][1, 1:] == 0).all()
    np.testing.assert_allclose(data['timestamps'], [[0, 1/30, 2/30], [0, .04, .08]])
    np.testing.assert_array_equal(data['frame_counts'], [3, 1])
    assert 'fps' not in data
    common = video.pack_detections([row], [(640, 480)], K[None], [30])
    assert common['fps'] == 30


def test_intrinsics_distortion_requires_explicit_choice(tmp_path):
    path = tmp_path / 'cameras.npz'
    np.savez(path, K=K[None])
    with pytest.raises(ValueError, match='already-undistorted'):
        video.load_intrinsics(path, 1)
    assert video.load_intrinsics(path, 1, True)[1] is None
    np.savez(path, K=K[None], dist_coeffs=np.zeros((1, 5)))
    assert video.load_intrinsics(path, 1)[1].shape == (1, 5)
    with pytest.raises(ValueError, match='K must'):
        video.load_intrinsics(path, 2)
    np.savez(path, K=K[None], dist_coeffs=np.zeros((1, 3)))
    with pytest.raises(ValueError, match='dist_coeffs'):
        video.load_intrinsics(path, 1)


def mocks(results, fps=25):
    events = []
    frames = iter([np.zeros((2, 4, 3), dtype=np.uint8)] * len(results))
    outputs = iter(results)
    cap = NS(isOpened=lambda: True, get=lambda prop: fps,
             read=lambda: (True, f) if (f := next(frames, None)) is not None else (False, None),
             release=lambda: events.append('release'))
    class Detector:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append('close')
        def detect_for_video(self, image, timestamp):
            events.append(('detect', timestamp))
            return NS(pose_landmarks=next(outputs))
    def options(**kwargs):
        assert kwargs['num_poses'] == 2
        assert kwargs['running_mode'] == 'VIDEO'
        return kwargs
    def undistort(frame, matrix, dist, unused, new_matrix):
        np.testing.assert_array_equal(matrix, new_matrix)
        events.append('undistort')
        return frame + 1
    def convert(frame, code):
        events.append(('rgb', int(frame[0, 0, 0])))
        return frame
    cv2 = NS(VideoCapture=lambda path: cap, CAP_PROP_FPS=5,
             COLOR_BGR2RGB=1, undistort=undistort, cvtColor=convert)
    mp = NS(tasks=NS(BaseOptions=lambda **kw: kw, vision=NS(
        PoseLandmarkerOptions=options, RunningMode=NS(VIDEO='VIDEO'),
        PoseLandmarker=NS(create_from_options=lambda options: Detector()))),
        Image=lambda **kw: kw, ImageFormat=NS(SRGB=1))
    return cv2, mp, events


def test_detector_undistorts_preserves_misses_and_timestamps():
    pose = [NS(x=.2, y=.4, z=0, visibility=.9, presence=.7)] * 33
    cv2, mp, events = mocks([[pose], []])
    rows, size, fps = video.extract_video('cam.mp4', K, np.zeros(5), 'model.task', cv2=cv2, mp=mp)
    assert size == (4, 2) and fps == 25
    assert rows.shape == (2, 33, 5)
    assert np.isnan(rows[1]).all()
    assert events[:3] == ['undistort', ('rgb', 1), ('detect', 0)]
    assert ('detect', 40) in events
    assert events[-2:] == ['close', 'release']


def test_multiple_people_rejected_and_resources_closed():
    cv2, mp, events = mocks([[[], []]])
    with pytest.raises(ValueError, match='multiple people'):
        video.extract_video('cam', K, None, 'model', cv2=cv2, mp=mp)
    assert events[-2:] == ['close', 'release']


def test_optional_presence_does_not_discard_visible_points():
    pose=[NS(x=.2,y=.4,z=0.,visibility=.8,presence=None)]*33
    cv2,mp,_=mocks([[pose]])
    rows,size,fps=video.extract_video('cam',K,None,'model',cv2=cv2,mp=mp)
    arrays=video.pack_detections([rows],[size],K[None],[fps])
    np.testing.assert_allclose(arrays['confidence'],.8)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('fps', [0, float('nan'), 1001, np.complex128(30)])
def test_bad_fps(fps):
    cv2, mp, events = mocks([[]], fps)
    with pytest.raises(ValueError, match='FPS'):
        video.extract_video('cam', K, None, 'model', cv2=cv2, mp=mp)
    assert events[-1] == 'release'


@pytest.mark.filterwarnings('error')
def test_pack_detections_rejects_complex_rates_and_landmarks():
    rows = np.ones((2, 33, 5))
    with pytest.raises(ValueError, match='FPS'):
        video.pack_detections([rows], [(4, 2)], K, np.array([25], dtype=complex))
    with pytest.raises(ValueError, match='landmarks'):
        video.pack_detections([rows.astype(complex)], [(4, 2)], K, [25])


def test_empty_video():
    cv2, mp, events = mocks([])
    with pytest.raises(ValueError, match='no decoded frames'):
        video.extract_video('cam', K, None, 'model', cv2=cv2, mp=mp)
    assert events[-1] == 'release'


def test_help_without_optional_imports():
    code = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('cv2', 'mediapipe'):
        raise AssertionError('optional import during help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', code, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--already-undistorted' in result.stdout
