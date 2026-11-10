"""Phase-1 CLI, legacy JSON and early-boundary regressions (no sync search)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import posecalib.api as api
import posecalib.synchronization as synchronization
from posecalib import Calibration, calibrate, project, triangulate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples' / 'calibrate_npz.py'
K = np.array([[1100., 0, 960], [0, 850., 540], [0, 0, 1.]])


def payload(schema='0.2', lengths=True):
    """Literal legacy field sets, independent of the current serializer.

    No version key existed. 0.1 has cameras/quality/scale; 0.2 adds optional
    layout and synchronization fields. Minimal files omit every optional key.
    """
    R = Rotation.from_rotvec([[0, 0, 0], [.1, -.2, .03]]).as_matrix()
    d = dict(R=R.tolist(), t=[[0., 0, 0], [-1., .2, .1]], K=K.tolist())
    if schema == 'minimal':
        return d
    d.update(centers=[[0, 0, 0], [999, 999, 999]], scale_source='camera_distance',
             segment_lengths=[.4] + [float('nan')] * 8 if lengths else None,
             reprojection_rmse_px=[.2, .3], inlier_fraction=[.9, .95])
    if schema == '0.2':
        d.update(skeleton='blazepose33', foot_keypoints=[27, 28],
                 time_offsets_seconds=[0., -.017],
                 synchronization={'convention': 'source time = reference time + offset'})
    return d


def write_payload(tmp_path, data):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location('phase1_cli', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('schema', ['0.1', '0.2', 'minimal'])
@pytest.mark.parametrize('lengths', [False, True, 'unknown', 'empty'])
def test_legacy_json_roundtrip_projection_and_optional_defaults(tmp_path, schema, lengths):
    data = payload(schema, lengths)
    if schema != 'minimal' and lengths in ('unknown', 'empty'):
        data['segment_lengths'] = [float('nan')] * 9 if lengths == 'unknown' else []
    cal = Calibration.load_json(write_payload(tmp_path, data))
    X = np.random.default_rng(42).uniform([-.4, -.3, 3.], [.4, .3, 5.], (4, 33, 3))
    expected = project(cal, X)
    hom = np.einsum('cij,...j->c...i', cal.projection_matrices(),
                    np.concatenate([X, np.ones(X.shape[:-1] + (1,))], -1))
    np.testing.assert_allclose(expected, hom[..., :2] / hom[..., 2:], atol=1e-10, rtol=0)
    if schema == 'minimal':
        assert cal.scale_source == 'unknown' and cal.segment_lengths is None
        assert np.isnan(cal.reprojection_rmse).all() and np.isnan(cal.inlier_fraction).all()
    elif lengths is True:
        assert cal.segment_lengths[0] == .4 and np.isnan(cal.segment_lengths[1:]).all()
    elif lengths == 'unknown':
        assert len(cal.segment_lengths) == 9 and np.isnan(cal.segment_lengths).all()
    elif lengths == 'empty':
        assert cal.segment_lengths.shape == (0,)
    else:
        assert cal.segment_lengths is None
    if schema == '0.2':
        np.testing.assert_array_equal(cal.time_offsets_seconds, [0., -.017])
        assert cal.info['foot_keypoints'] == [27, 28]
        assert cal.info['synchronization'] == data['synchronization']
    else:
        assert cal.time_offsets_seconds is None and cal.info['skeleton'] is None
    moved = cal.transformed(1.7, Rotation.from_rotvec([.2, -.1, .3]).as_matrix(), [.1, -.2, .4])
    out = tmp_path / 'roundtrip.json'
    cal.save_json(out)
    back = Calibration.load_json(out)
    np.testing.assert_allclose(project(back, X), expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(back.centers, cal.centers)
    assert not np.array_equal(back.centers[1], [999, 999, 999])  # cached centers are not authoritative
    np.testing.assert_equal(back.segment_lengths, cal.segment_lengths)
    np.testing.assert_equal(moved.time_offsets_seconds, cal.time_offsets_seconds)
    np.testing.assert_equal(back.time_offsets_seconds, cal.time_offsets_seconds)
    assert back.info == cal.info


BAD_GEOMETRY = [
    ('R', None), ('R', 1), ('t', None),
    ('R', []), ('R', np.eye(3).tolist()),
    ('R', [np.eye(3).tolist(), np.diag([-1, 1, 1]).tolist()]),
    ('R', [np.eye(3).tolist(), np.diag([1, 1, 2]).tolist()]),
    ('R', np.full((2, 3, 3), np.nan).tolist()),
    ('t', [[0, 0, 0]]), ('t', [[0, 0, 0], [np.inf, 0, 0]]),
    ('segment_lengths', [0.] + [np.nan] * 8),
    ('segment_lengths', [-.1] + [np.nan] * 8),
    ('segment_lengths', [np.inf] + [np.nan] * 8),
    ('segment_lengths', [[.4] * 9]), ('segment_lengths', [None] * 9),
]


@pytest.mark.parametrize('key,value', BAD_GEOMETRY)
def test_json_rejects_bad_geometry(tmp_path, key, value):
    data = payload()
    data[key] = value
    with pytest.raises(ValueError, match=key):
        Calibration.load_json(write_payload(tmp_path, data))


@pytest.mark.parametrize('key', ['K', 'R', 't', 'segment_lengths'])
def test_mutated_bad_geometry_save_preserves_existing_file(tmp_path, key):
    cal = Calibration.load_json(write_payload(tmp_path, payload()))
    out = tmp_path / 'keep.json'
    out.write_bytes(b'keep-existing-output')
    getattr(cal, key).flat[0] = np.inf
    with pytest.raises(ValueError, match=key):
        cal.save_json(out)
    assert out.read_bytes() == b'keep-existing-output'


@pytest.mark.parametrize('exists', [False, True])
def test_json_replace_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch, exists):
    cal = Calibration.load_json(write_payload(tmp_path, payload()))
    out = tmp_path / 'rig.json'
    if exists:
        out.write_bytes(b'keep')
    def failed_replace(*args):
        raise OSError('replace failure sentinel')
    monkeypatch.setattr(api.os, 'replace', failed_replace)
    with pytest.raises(OSError, match='replace failure sentinel'):
        cal.save_json(out)
    assert out.read_bytes() == b'keep' if exists else not out.exists()
    assert not list(tmp_path.glob('.rig.json.*.tmp'))


@pytest.mark.parametrize('estimate', [False, True])
def test_bad_K_and_lengths_rejected_before_temporal_and_geometry(monkeypatch, estimate):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid boundary reached temporal/geometric computation')
    monkeypatch.setattr(synchronization, 'estimate_time_offsets', forbidden)
    monkeypatch.setattr(synchronization, 'resample_keypoints', forbidden)
    monkeypatch.setattr(api, 'initialize', forbidden)
    kp, co = np.zeros((2, 4, 33, 2)), np.ones((2, 4, 33))
    bad = K.copy()
    bad[0, 1] = 1e-15
    for matrix, options, message in [(bad, {}, 'K must'),
                                     (K, {'segment_lengths': [0.] * 9}, 'segment_lengths')]:
        with pytest.raises(ValueError, match=message):
            calibrate(kp, co, matrix, timestamps=np.tile(np.arange(4), (2, 1)),
                      estimate_sync=estimate, **options)
    cal = Calibration(np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 3)), K,
                      np.empty((0, 33, 3)), None, 'test', np.zeros(2), np.ones(2),
                      time_offsets_seconds=np.array([0., .01]))
    cal.K[0, 0, 1] = 1.
    with pytest.raises(ValueError, match='K must'):
        triangulate(cal, kp, co, fps=30)


CLI_CASES = [
    ('corrupt', [], 'load'), ('npy', [], 'NPZ'), ('object', [], 'Object arrays'),
    ('missing-key', [], 'confidence'), ('shape', [], 'keypoints'),
    ('scalar', [], 'keypoints'), ('confidence-shape', [], 'confidence'),
    ('bad-K-skew', [], 'K must'), ('bad-K-fy', [], 'K must'), ('bad-K-row', [], 'K must'),
    ('valid', ['--distance', 'x', '1', '2'], 'distance'),
    ('valid', ['--distance', '0.0', '1', '2'], 'distance'),
    ('valid', ['--distance', '0', '1', 'x'], 'distance'),
    ('valid', ['--distance', '0', '0', '2'], 'distinct'),
    ('valid', ['--distance', '-1', '1', '2'], 'camera_distance'),
    ('valid', ['--distance', '0', '2', '2'], 'camera_distance'),
    ('valid', ['--distance', '0', '1', '0'], 'camera_distance'),
    ('valid', ['--distance', '0', '1', '-2'], 'camera_distance'),
    ('valid', ['--distance', '0', '1', 'nan'], 'camera_distance'),
    ('valid', ['--distance', '0', '1', 'inf'], 'camera_distance'),
    ('valid', ['--frame-step', '0'], 'frame_step'),
    ('valid', ['--frame-step', '1.5'], 'invalid int'),
    ('valid', ['--fps', 'nan'], 'fps'), ('valid', ['--fps', '0'], 'fps'),
    ('bad-fps-array', [], 'fps'),
    ('valid', ['--max-offset', 'nan'], 'max_offset'),
    ('valid', ['--max-offset', '-1'], 'max_offset'),
]


def detections(tmp_path, kind='valid'):
    path = tmp_path / 'детекции.npz'
    if kind == 'corrupt':
        path.write_bytes(b'not-an-npz')
        return path
    if kind == 'npy':
        with path.open('wb') as stream:
            np.save(stream, np.zeros(2))
        return path
    d = dict(keypoints=np.zeros((2, 4, 33, 2)), confidence=np.ones((2, 4, 33)), K=K.copy())
    if kind == 'missing-key':
        del d['confidence']
    elif kind == 'object':
        d['keypoints'] = np.array([object()], dtype=object)
    elif kind == 'shape':
        d['keypoints'] = np.zeros((2, 4, 33, 3))
    elif kind == 'scalar':
        d['keypoints'] = np.array(1.)
    elif kind == 'confidence-shape':
        d['confidence'] = np.ones((2, 4, 32))
    elif kind == 'bad-K-skew':
        d['K'][0, 1] = 1e-15
    elif kind == 'bad-K-fy':
        d['K'][1, 1] = 0
    elif kind == 'bad-K-row':
        d['K'][2, 2] = 2
    elif kind == 'bad-fps-array':
        d['fps'] = [30, 30]
    np.savez(path, **d)
    return path


@pytest.mark.parametrize('kind,args,message', CLI_CASES)
def test_cli_invalid_inputs_clean_error_no_overwrite(tmp_path, kind, args, message):
    path = detections(tmp_path, kind)
    before = path.read_bytes()
    out = tmp_path / 'rig.json'
    out.write_bytes(b'existing-valid-output')
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), str(path), '--out', str(out), *args],
                            capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=15)
    assert result.returncode == 2, result.stderr
    assert 'error:' in result.stderr and message.lower() in result.stderr.lower()
    assert 'Traceback' not in result.stderr
    assert out.read_bytes() == b'existing-valid-output'
    assert path.read_bytes() == before


@pytest.mark.parametrize('failure', ['bad-K', 'bad-distance', 'bad-shape', 'bad-fps', 'same-path', 'missing-file'])
def test_cli_failure_is_early_and_creates_no_output(tmp_path, monkeypatch, cli, failure):
    path = detections(tmp_path, 'bad-K-fy' if failure == 'bad-K' else
                      'shape' if failure == 'bad-shape' else 'valid')
    out = path if failure == 'same-path' else tmp_path / 'new.json'
    before = path.read_bytes()
    args = ['--estimate-sync', '--fps', '30']
    if failure == 'bad-distance':
        args += ['--distance', '0', '0', '1']
    elif failure == 'bad-fps':
        args += ['--fps', 'nan']
    elif failure == 'missing-file':
        path = tmp_path / 'missing.npz'
    def forbidden(*args, **kwargs):
        pytest.fail('invalid CLI input reached geometry or time estimation')
    monkeypatch.setattr(api, 'initialize', forbidden)
    monkeypatch.setattr(synchronization, 'estimate_time_offsets', forbidden)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(path), '--out', str(out), *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    if failure == 'same-path':
        assert path.read_bytes() == before
    else:
        assert not out.exists()


def test_cli_internal_defects_propagate_without_overwrite(tmp_path, monkeypatch, cli):
    path, out = detections(tmp_path), tmp_path / 'rig.json'
    out.write_bytes(b'keep')
    def defect(*args, **kwargs):
        raise RuntimeError('internal defect sentinel')
    monkeypatch.setattr(cli, 'calibrate', defect)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(path), '--out', str(out)])
    with pytest.raises(RuntimeError, match='internal defect sentinel'):
        cli.main()
    assert out.read_bytes() == b'keep'


def test_cli_valid_arguments_save_offsets_and_unknown_lengths(tmp_path, monkeypatch, cli):
    cal = Calibration.load_json(write_payload(tmp_path, payload()))
    def valid(kp, co, matrix, **kwargs):
        assert kwargs['camera_distance'] == (0, 1, 2.5)
        assert kwargs['fps'] == 30 and kwargs['frame_step'] == 2
        np.testing.assert_array_equal(matrix, K)
        return cal
    monkeypatch.setattr(cli, 'calibrate', valid)
    out = tmp_path / 'rig.json'
    path = detections(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(path), '--out', str(out), '--no-floor',
                                    '--distance', '0', '1', '2.5', '--fps', '30', '--frame-step', '2'])
    cli.main()
    back = Calibration.load_json(out)
    np.testing.assert_array_equal(back.time_offsets_seconds, cal.time_offsets_seconds)
    np.testing.assert_equal(back.segment_lengths, cal.segment_lengths)


@pytest.mark.parametrize('step', [True, np.bool_(True), 0, 1.5])
def test_invalid_frame_step_rejected_before_computation(monkeypatch, step):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid frame_step reached initialization')
    monkeypatch.setattr(api, 'Calibrator', forbidden)
    with pytest.raises(ValueError, match='frame_step'):
        calibrate(np.zeros((2, 4, 33, 2)), np.ones((2, 4, 33)), K, frame_step=step)


def test_canonical_roundoff_and_asymmetric_K_across_adapters(tmp_path):
    from posecalib.mediapipe_io import prepare_blazepose, load_blazepose_cameras
    spec = importlib.util.spec_from_file_location('phase1_video', ROOT / 'examples' / 'prepare_blazepose_videos.py')
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    data = payload()
    matrices = np.stack([K, K * [1.2, .8, 1.]])
    matrices[:, 2, :2] = [1e-12, -1e-12]  # exact allowed absolute boundary
    data['K'] = matrices.tolist()
    cal = Calibration.load_json(write_payload(tmp_path, data))
    X = np.random.default_rng(7).uniform([-.2, -.4, 3], [.3, .5, 6], (5, 33, 3))
    pixels = project(cal, X)
    landmarks = np.ones((2, 5, 33, 5))
    landmarks[..., :2] = pixels / [1920, 1080]
    prepared = prepare_blazepose(landmarks, [1920, 1080], matrices)
    np.testing.assert_allclose(prepared['keypoints'], pixels, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(prepared['K'], cal.K)
    paths = []
    for camera in range(2):
        path = tmp_path / f'cam{camera}.npz'
        np.savez(path, landmarks=landmarks[camera], image_size=[1920, 1080], K=matrices[camera])
        paths.append(path)
    np.testing.assert_array_equal(load_blazepose_cameras(paths)['K'], cal.K)
    path = tmp_path / 'intrinsics.npz'
    np.savez(path, K=matrices)
    np.testing.assert_array_equal(video.load_intrinsics(path, 2, True)[0], cal.K)
    h = np.einsum('cij,...j->c...i', cal.projection_matrices(), np.concatenate([X, np.ones((5, 33, 1))], -1))
    np.testing.assert_allclose(pixels, h[..., :2] / h[..., 2:], atol=1e-10, rtol=0)


def test_real_cli_positive_metric_scale_with_asymmetric_K(tmp_path):
    # Small exact articulated capture; independent truth, no offset estimation.
    from posecalib.sim.scene import simulate_person
    R = Rotation.from_rotvec([[0, 0, 0], [.01, .05, 0], [-.1, .1, -.03]]).as_matrix()
    centers = np.array([[0., 0, 0], [1., .2, .1], [-.8, -.3, .3]])
    matrices = np.stack([K, K * [1.1, .9, 1], K * [.8, 1.2, 1]])
    points = simulate_person(np.random.default_rng(21), 3., 'walk')['J'][::5] + [0., 0, 5.]
    true = Calibration(R, -np.einsum('cij,cj->ci', R, centers), matrices, points,
                       None, 'true', np.zeros(3), np.ones(3))
    kp = project(true, points)
    path, out = tmp_path / 'positive.npz', tmp_path / 'rig.json'
    np.savez(path, keypoints=kp, confidence=np.ones(kp.shape[:3]), K=matrices)
    distance = float(np.linalg.norm(centers[1] - centers[0]))
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), str(path), '--no-floor',
                             '--distance', '0', '1', str(distance), '--out', str(out)],
                            capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=30)
    assert result.returncode == 0, result.stderr
    cal = Calibration.load_json(out)
    assert cal.scale_source == 'camera_distance'
    np.testing.assert_array_equal(cal.K, matrices)
    np.testing.assert_allclose(cal.centers, centers, atol=1e-5, rtol=0)
    np.testing.assert_allclose(cal.R, R, atol=1e-5, rtol=0)
    assert max(cal.reprojection_rmse) < 1e-5
