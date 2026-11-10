import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posecalib.mediapipe_io import prepare_blazepose, load_blazepose_cameras
from posecalib.skeleton import BLAZEPOSE33, COCO17, infer_skeleton, ankle_indices
from posecalib import Calibrator


def sample(columns=5):
    lm = np.ones((2, 3, 33, columns))
    lm[..., :3] = [0.25, 0.75, -0.2]
    lm[..., 3] = 0.8
    if columns == 5:
        lm[..., 4] = 0.6
    K = np.array([[900., 0, 320], [0, 850, 240], [0, 0, 1]])
    return lm, np.array([[640, 480], [1920, 1080]]), K


def test_layout():
    assert infer_skeleton(33) is BLAZEPOSE33
    assert infer_skeleton(17) is COCO17
    assert ankle_indices(BLAZEPOSE33) == (27, 28)
    assert ankle_indices(COCO17) == (15, 16)
    assert len(set(BLAZEPOSE33.names)) == 33
    assert BLAZEPOSE33.index('left_heel') == 29
    assert BLAZEPOSE33.index('right_foot_index') == 32
    assert BLAZEPOSE33.segments.tolist() == [[11,13],[13,15],[12,14],[14,16],
                                                         [23,25],[25,27],[24,26],[26,28],[23,24]]
    assert len(BLAZEPOSE33.typical_lengths) == BLAZEPOSE33.num_segments == 9
    assert len(BLAZEPOSE33.mirror_pairs) == 16
    with pytest.raises(ValueError):
        infer_skeleton(32)


@pytest.mark.parametrize('columns,expected', [(4, 0.8), (5, 0.6)])
def test_conversion_and_api(columns, expected):
    lm, size, K = sample(columns)
    before = lm.copy()
    d = prepare_blazepose(lm, size, K)
    assert d['keypoints'].shape == (2, 3, 33, 2)
    np.testing.assert_allclose(d['keypoints'][0], np.broadcast_to([160,360], (3,33,2)))
    np.testing.assert_allclose(d['keypoints'][1], np.broadcast_to([480,810], (3,33,2)))
    np.testing.assert_allclose(d['confidence'], expected)
    np.testing.assert_equal(lm, before)
    cal = Calibrator(d['keypoints'], d['confidence'], d['K'], skeleton=BLAZEPOSE33)
    assert cal.J == 33
    assert cal.tracks['valid'].all()


def test_missing_scores_and_outside_image():
    lm, size, K = sample()
    lm[0, 0, 0, 0] = np.nan
    lm[0, 0, 1, 3] = np.nan
    lm[0, 0, 2, 4] = np.inf
    lm[0, 0, 3, 3] = -1
    lm[0, 0, 4, 4] = 0
    lm[0, 1] = np.nan
    lm[1, 0, 0, :2] = [-0.1, 1.2]
    lm[1, 0, 1, 2] = np.nan  # unused depth
    d = prepare_blazepose(lm, size, K)
    assert np.isnan(d['keypoints'][0, 0, :5]).all()
    assert (d['confidence'][0, 0, :5] == 0).all()
    assert np.isnan(d['keypoints'][0, 1]).all()
    np.testing.assert_allclose(d['keypoints'][1, 0, 0], [-192,1296])
    assert d['confidence'][1, 0, 1] == 0.6


@pytest.mark.parametrize('change', ['shape', 'size', 'K', 'empty', 'timestamps', 'timestamp_shape', 'order', 'decreasing'])
def test_validation(change):
    lm, size, K = sample()
    ts = np.tile(np.arange(3), (2,1)).astype(float)
    if change == 'shape': lm = lm[..., :3]
    if change == 'size': size[0,0] = 0
    if change == 'K': K[0,0] = np.nan
    if change == 'empty': lm = lm[:, :0]
    if change == 'timestamps': ts[1, 1] = np.nan
    if change == 'timestamp_shape': ts = ts[0]
    if change == 'decreasing': ts[1] = [0, 2, 1]
    if change == 'order': ts[:] = 0
    with pytest.raises(ValueError):
        prepare_blazepose(lm, size, K, timestamps=ts)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('field', ['landmarks', 'image_size', 'timestamps'])
def test_complex_exports_rejected_before_conversion(field, tmp_path):
    lm, size, K = sample()
    arrays = dict(landmarks=lm[:1], image_size=size[:1], K=K,
                  timestamps=np.arange(3, dtype=float)[None, :])
    arrays[field] = arrays[field].astype(complex)
    with pytest.raises(ValueError, match=field):
        prepare_blazepose(**arrays)

    path = tmp_path / 'camera.npz'
    np.savez(path, **{name: value if name == 'K' else value[0]
                      for name, value in arrays.items()})
    with pytest.raises(ValueError, match=field):
        load_blazepose_cameras([path])


def export(tmp_path, columns=(4,5), timestamps=True):
    lm, size, K = sample()
    paths = []
    for c, n in enumerate(columns):
        path = tmp_path / f'camera{c}.npz'
        np.savez(path, landmarks=lm[c,...,:n], image_size=size[c], K=K,
                 **({'timestamps': np.arange(3) / 30 + c * 0.025} if timestamps else {}))
        paths.append(path)
    return paths


def test_load_and_cli(tmp_path):
    paths = export(tmp_path)
    d = load_blazepose_cameras(paths)
    np.testing.assert_allclose(d['confidence'][0], 0.8)
    np.testing.assert_allclose(d['confidence'][1], 0.6)
    assert d['timestamps'].shape == (2,3)
    np.testing.assert_array_equal(d['timestamps'][0], np.arange(3) / 30)
    np.testing.assert_array_equal(d['timestamps'][1], np.arange(3) / 30 + 0.025)
    out = tmp_path / 'prepared.npz'
    script = Path(__file__).resolve().parents[1] / 'examples' / 'prepare_blazepose_npz.py'
    result = subprocess.run([sys.executable, str(script), *map(str, paths), '--out', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with np.load(out, allow_pickle=False) as actual:
        for key, value in d.items():
            np.testing.assert_equal(actual[key], value)
    result = subprocess.run([sys.executable, str(script), *map(str, paths), '--out', str(paths[0])],
                            capture_output=True, text=True)
    assert result.returncode != 0


def test_loader_rejects_mismatched_metadata(tmp_path):
    paths = export(tmp_path)
    with np.load(paths[1]) as d:
        values = {key: d[key] for key in d.files if key != 'timestamps'}
    np.savez(paths[1], **values)
    with pytest.raises(ValueError, match='every camera'):
        load_blazepose_cameras(paths)
    with pytest.raises(ValueError):
        load_blazepose_cameras([])
    values['landmarks'] = values['landmarks'][:1]
    np.savez(paths[1], **values)
    with pytest.raises(ValueError, match='equal'):
        load_blazepose_cameras(paths)


def test_local_clocks_preserved_without_alignment():
    lm, size, K = sample()
    timestamps = np.array([[10., 10.033, 10.067], [10.025, 10.059, 10.094]])
    original = timestamps.copy()
    result = prepare_blazepose(lm, size, K, timestamps=timestamps)
    np.testing.assert_array_equal(result['timestamps'], original)
    np.testing.assert_array_equal(timestamps, original)
    timestamps[:] = 0
    np.testing.assert_array_equal(result['timestamps'], original)
    without = prepare_blazepose(lm, size, K)
    np.testing.assert_array_equal(result['keypoints'], without['keypoints'])
    np.testing.assert_array_equal(result['confidence'], without['confidence'])
