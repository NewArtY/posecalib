"""Phase-1 review regressions; no estimator or synchronization algorithm changes."""
import errno
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import posecalib.api as api
import posecalib.synchronization as synchronization
from posecalib import Calibration, Calibrator, calibrate, project
from posecalib.validation import validate_extrinsics, validate_similarity

ROOT = Path(__file__).resolve().parents[1]
K = np.array([[1100., 0, 960], [0, 850., 540], [0, 0, 1.]])


def rig(R=None):
    if R is None:
        R = Rotation.from_rotvec([[.1, -.2, .03], [.3, .2, -.1]]).as_matrix()
    return Calibration(R, np.array([[0., 0, 0], [-1., .2, .1]]), K,
                       np.ones((3, 17, 3)) * [0., 0, 5.], None, 'test', np.zeros(2), np.ones(2))


@pytest.mark.parametrize('representation', ['float32', 'promoted', 'json'])
def test_float32_random_rotations_and_legacy_numeric_representation(representation):
    exact = Rotation.random(2048, random_state=42).as_matrix()
    source = exact.astype(np.float32)
    supplied = source if representation == 'float32' else source.astype(float)
    if representation == 'json':
        supplied = json.loads(json.dumps(supplied.tolist()))
    R, _ = validate_extrinsics(supplied, np.zeros((len(exact), 3)))
    # Accepted roundoff is canonicalized; an enlarged admission tolerance alone
    # would still let products drift outside the supported rotation boundary.
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.tile(np.eye(3), (len(R), 1, 1)),
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1., rtol=0, atol=2e-14)
    np.testing.assert_allclose(R, exact, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(source, exact.astype(np.float32))
    assert not np.shares_memory(R, supplied) if isinstance(supplied, np.ndarray) else True


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('matrix', [np.diag([-1., 1, 1]), np.diag([1.001, 1, 1]),
                                   np.diag([1.000002, 1, 1]),
                                   np.diag([1.001, 1 / 1.001, 1]),
                                   np.array([[1., .001, 0], [0, 1, 0], [0, 0, 1]])])
def test_reflections_scales_and_shears_are_rejected_before_canonicalization(dtype, matrix):
    matrix = matrix.astype(dtype)
    with pytest.raises(ValueError, match='R'):
        validate_extrinsics(np.stack([matrix, np.eye(3)]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match='Rw'):
        validate_similarity(1., matrix, np.zeros(3))


def test_float64_keeps_strict_tolerance_when_not_float32_representable():
    matrix = np.diag([1. + 2e-8, 1., 1.])
    with pytest.raises(ValueError, match='R'):
        validate_extrinsics(np.stack([matrix, matrix]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match='Rw'):
        validate_similarity(1., matrix, np.zeros(3))


def test_repeated_near_boundary_and_float32_transforms_are_closed(tmp_path):
    near = np.diag([1. + 4e-9, 1., 1.])
    cal = rig(np.stack([near, near]))
    cal.time_offsets_seconds = np.array([0., -.017])
    cal.info['reference_times'] = np.array([5., 6., 7.])
    initial = project(cal, cal.points)
    Rw = Rotation.from_rotvec([.01, -.02, .003]).as_matrix().astype(np.float32)
    for i in range(100):
        cal = cal.transformed(1., near if i % 2 else Rw, [.001, -.002, .003])
    np.testing.assert_allclose(project(cal, cal.points), initial, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(cal.time_offsets_seconds, [0., -.017])
    np.testing.assert_array_equal(cal.info['reference_times'], [5., 6., 7.])
    homogeneous = np.einsum('cij,...j->c...i', cal.projection_matrices(),
                            np.concatenate([cal.points, np.ones(cal.points.shape[:-1] + (1,))], -1))
    np.testing.assert_allclose(project(cal, cal.points), homogeneous[..., :2] / homogeneous[..., 2:],
                               rtol=0, atol=1e-10)
    path = tmp_path / 'rig.json'
    cal.save_json(path)
    before = path.read_bytes()
    loaded = Calibration.load_json(path)
    np.testing.assert_array_equal(loaded.R, cal.R)
    np.testing.assert_array_equal(loaded.time_offsets_seconds, cal.time_offsets_seconds)
    loaded.save_json(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('field', ['R', 't', 'Rw', 'tw'])
@pytest.mark.parametrize('imaginary', [.1, np.nan, np.inf])
def test_complex_pose_boundaries_reject_without_casting_warning(field, imaginary, tmp_path):
    cal = rig()
    path = tmp_path / 'rig.json'
    path.write_bytes(b'preserve')
    value = getattr(cal, field).astype(complex) if field in ('R', 't') else (
        np.eye(3, dtype=complex) if field == 'Rw' else np.zeros(3, dtype=complex))
    value.flat[0] = complex(value.flat[0].real, imaginary)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=field):
            if field in ('R', 't'):
                setattr(cal, field, value)
                cal.save_json(path)
            else:
                cal.transformed(**{field: value})
    assert path.read_bytes() == b'preserve'
    assert not list(tmp_path.glob('.rig.json.*.tmp'))


@pytest.mark.parametrize('exists', [False, True])
def test_write_failure_remains_primary_when_close_also_fails(tmp_path, monkeypatch, exists):
    path = tmp_path / 'rig.json'
    if exists:
        path.write_bytes(b'preserve')
    factory = api.tempfile.NamedTemporaryFile
    class BrokenWriteAndClose:
        def __init__(self, **kwargs):
            self.stream = factory(**kwargs)
            self.name = self.stream.name
        def write(self, text):
            self.stream.write(text[:20])
            raise OSError(errno.ENOSPC, 'primary write failure')
        def close(self):
            self.stream.close()
            raise OSError(errno.EIO, 'secondary close failure')
    monkeypatch.setattr(api.tempfile, 'NamedTemporaryFile', BrokenWriteAndClose)
    with pytest.raises(OSError, match='primary write failure') as error:
        rig().save_json(path)
    assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == b'preserve' if exists else not path.exists()
    assert not list(tmp_path.glob('.rig.json.*.tmp'))


@pytest.mark.parametrize('shape,message', [((0, 4, 17, 2), 'cameras'),
                                          ((1, 4, 17, 2), 'cameras'),
                                          ((2, 0, 17, 2), 'frames')])
def test_cli_degenerate_dimensions_exit_two_without_overwrite(shape, message, tmp_path):
    inp, out = tmp_path / 'input.npz', tmp_path / 'rig.json'
    np.savez(inp, keypoints=np.zeros(shape), confidence=np.ones(shape[:3]), K=K)
    original = inp.read_bytes()
    out.write_bytes(b'preserve')
    proc = subprocess.run([sys.executable, '-B', str(ROOT / 'examples/calibrate_npz.py'), str(inp),
                           '--out', str(out), '--no-floor'], capture_output=True, text=True,
                          encoding='utf-8', timeout=10)
    assert proc.returncode == 2 and message in proc.stderr and 'Traceback' not in proc.stderr
    assert inp.read_bytes() == original and out.read_bytes() == b'preserve'


def test_one_camera_result_and_one_frame_calibration_input_remain_supported():
    cal = Calibration(np.eye(3)[None], np.zeros((1, 3)), K, np.empty((0, 17, 3)),
                      None, 'test', np.zeros(1), np.ones(1))
    assert project(cal, [[0., 0, 5.]]).shape == (1, 1, 2)
    candidate = Calibrator(np.zeros((2, 1, 17, 2)), np.ones((2, 1, 17)), K)
    assert candidate.C == 2 and candidate.F == 1


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location('review_fix_cli', ROOT / 'examples/calibrate_npz.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('field', ['keypoints', 'confidence', 'K', 'timestamps'])
def test_structured_npz_fields_rejected_before_calibration(field, cli, tmp_path, monkeypatch, capsys):
    data = dict(keypoints=np.zeros((2, 4, 17, 2)), confidence=np.ones((2, 4, 17)), K=K,
                timestamps=np.tile(np.arange(4.), (2, 1)))
    data[field] = np.zeros(data[field].shape, dtype=[('value', float), ('other', float)])
    inp, out = tmp_path / 'input.npz', tmp_path / 'rig.json'
    np.savez(inp, **data)
    original = inp.read_bytes()
    out.write_bytes(b'preserve')
    def forbidden(*args, **kwargs):
        pytest.fail('invalid NPZ dtype reached calibration')
    monkeypatch.setattr(cli, 'calibrate', forbidden)
    monkeypatch.setattr(sys, 'argv', ['calibrate_npz.py', str(inp), '--out', str(out), '--no-floor'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert field in capsys.readouterr().err
    assert inp.read_bytes() == original and out.read_bytes() == b'preserve'


@pytest.mark.parametrize('error_type', [TypeError, RuntimeError])
def test_cli_does_not_hide_internal_errors(error_type, cli, monkeypatch, tmp_path):
    inp, out = tmp_path / 'input.npz', tmp_path / 'rig.json'
    np.savez(inp, keypoints=np.zeros((2, 4, 17, 2)), confidence=np.ones((2, 4, 17)), K=K)
    def defect(*args, **kwargs):
        raise error_type('internal sentinel')
    monkeypatch.setattr(cli, 'calibrate', defect)
    monkeypatch.setattr(sys, 'argv', ['calibrate_npz.py', str(inp), '--out', str(out), '--no-floor'])
    with pytest.raises(error_type, match='internal sentinel'):
        cli.main()
    assert not out.exists()


@pytest.mark.parametrize('shape,message', [((0, 4, 17, 2), 'cameras'),
                                          ((1, 4, 17, 2), 'cameras'),
                                          ((2, 0, 17, 2), 'frames')])
@pytest.mark.parametrize('estimate_sync', [False, True])
def test_degenerate_input_sizes_fail_before_any_temporal_or_geometry_call(shape, message, estimate_sync,
                                                                        monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid array size reached temporal or geometry computation')
    monkeypatch.setattr(api, 'initialize', forbidden)
    monkeypatch.setattr(synchronization, 'estimate_time_offsets', forbidden)
    monkeypatch.setattr(synchronization, 'resample_keypoints', forbidden)
    kp, conf = np.zeros(shape), np.ones(shape[:3])
    with pytest.raises(ValueError, match=message):
        Calibrator(kp, conf, K)
    with pytest.raises(ValueError, match=message):
        calibrate(kp, conf, K, estimate_sync=estimate_sync, timestamps=np.zeros(shape[:2]))


@pytest.mark.parametrize('exists', [False, True])
def test_close_failure_cleans_temp_and_preserves_destination(tmp_path, monkeypatch, exists):
    path = tmp_path / 'rig.json'
    if exists:
        path.write_bytes(b'preserve')
    factory = api.tempfile.NamedTemporaryFile
    class BrokenClose:
        def __init__(self, **kwargs):
            self.stream = factory(**kwargs)
            self.name = self.stream.name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def write(self, text):
            return self.stream.write(text)
        def close(self):
            self.stream.close()
            raise OSError(errno.ENOSPC, 'injected close/flush failure')
    monkeypatch.setattr(api.tempfile, 'NamedTemporaryFile', BrokenClose)
    with pytest.raises(OSError, match='close/flush'):
        rig().save_json(path)
    assert path.read_bytes() == b'preserve' if exists else not path.exists()
    assert not list(tmp_path.glob('.rig.json.*.tmp'))
