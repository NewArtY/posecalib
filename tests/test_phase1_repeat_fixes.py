"""Bounded repeat R1 regressions for mutable extrinsics at transformed()."""
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posecalib import Calibration, project


def rig(R=None, t=None):
    return Calibration(
        np.tile(np.eye(3), (2, 1, 1)) if R is None else R,
        np.array([[0., 0., 0.], [-1., .2, .1]]) if t is None else t,
        np.array([[1100., 0., 960.], [0., 850., 540.], [0., 0., 1.]]),
        np.array([[[.2, -.1, 5.], [.4, .3, 6.]]]), np.array([.4, np.nan]),
        'camera_distance', np.zeros(2), np.ones(2),
        {'reference_times': np.array([3.5]), 'synchronization': {'tag': 'preserve'}},
        np.array([0., -.012]))


def snapshot(cal):
    return {name: (value, value.dtype, value.shape, value.tobytes())
            for name, value in vars(cal).items() if isinstance(value, np.ndarray)}


def assert_unchanged(cal, before):
    for name, (original, dtype, shape, payload) in before.items():
        value = getattr(cal, name)
        assert value is original
        assert value.dtype == dtype and value.shape == shape
        assert value.tobytes() == payload


@pytest.mark.parametrize('representation', ['native32', 'promoted64'])
@pytest.mark.parametrize('mutation', ['replace', 'inplace'])
@pytest.mark.parametrize('readonly', [False, True])
def test_mutable_extrinsics_compose_like_constructor_and_save_load(
        representation, mutation, readonly, tmp_path):
    rounded = Rotation.from_rotvec([[.1, -.2, .03], [.3, .2, -.1]]).as_matrix().astype(np.float32)
    supplied = rounded if representation == 'native32' else rounded.astype(float)
    translation = np.array([[.123456789, -.234567891, .345678912], [-1.12345678, .2, .1]],
                           dtype=np.float32)
    cal = rig()
    if mutation == 'replace':
        cal.R, cal.t = supplied.copy(), translation.copy()
    else:
        if representation == 'native32':
            cal.R = cal.R.astype(np.float32)
        cal.t = cal.t.astype(np.float32)
        cal.R[...] = supplied
        cal.t[...] = translation
    if readonly:
        for value in vars(cal).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    before = snapshot(cal)
    world = Rotation.from_rotvec([.2, -.1, .05]).as_matrix()
    shift = np.array([.03, -.07, .11])
    world_before, shift_before = world.tobytes(), shift.tobytes()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # Both existing admission paths must agree with the direct mutable path.
        expected = rig(cal.R, cal.t).transformed(1.3, world, shift)
        path = tmp_path / 'accepted.json'
        cal.save_json(path)
        loaded = Calibration.load_json(path).transformed(1.3, world, shift)
        moved = cal.transformed(1.3, world, shift)
    for reference in (expected, loaded):
        np.testing.assert_allclose(moved.R, reference.R, rtol=0, atol=1e-14)
        np.testing.assert_allclose(moved.t, reference.t, rtol=0, atol=1e-14)
    np.testing.assert_allclose(moved.points, expected.points, rtol=0, atol=1e-14)
    np.testing.assert_allclose(project(moved, moved.points), project(expected, expected.points),
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(moved.K, cal.K)
    np.testing.assert_allclose(moved.segment_lengths, 1.3 * cal.segment_lengths, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(moved.time_offsets_seconds, cal.time_offsets_seconds)
    np.testing.assert_array_equal(moved.info['reference_times'], cal.info['reference_times'])
    assert moved.info['synchronization'] == cal.info['synchronization']
    assert moved.scale_source == cal.scale_source
    assert not np.shares_memory(moved.R, cal.R) and not np.shares_memory(moved.t, cal.t)
    assert world.tobytes() == world_before and shift.tobytes() == shift_before
    assert_unchanged(cal, before)


class ArithmeticForbidden(np.ndarray):
    """Public array values must be validated before composition can touch them."""
    def forbidden(self, *args, **kwargs):
        raise AssertionError('invalid mutable extrinsics reached composition')

    __matmul__ = forbidden
    __mul__ = forbidden
    __rmul__ = forbidden
    __add__ = forbidden
    __sub__ = forbidden


COMMON_BAD = ['complex_zero', 'complex_nonzero', 'complex_nan', 'complex_inf',
              'nan', 'inf', 'negative_inf', 'missing_axis', 'empty',
              'strings', 'bool', 'structured']
BAD_CASES = [(field, kind) for field in ('R', 't') for kind in COMMON_BAD]
BAD_CASES += [('R', kind) for kind in ('reflection', 'scale', 'shear', 'strict_float64')]


@pytest.mark.parametrize('field,kind', BAD_CASES)
def test_invalid_mutable_extrinsics_reject_before_composition_without_warnings(field, kind):
    cal = rig()
    value = getattr(cal, field).copy()
    if kind.startswith('complex_'):
        value = value.astype(complex)
        imaginary = {'complex_zero': 0., 'complex_nonzero': .1,
                     'complex_nan': np.nan, 'complex_inf': np.inf}[kind]
        value.flat[0] = complex(value.flat[0].real, imaginary)
    elif kind in ('nan', 'inf', 'negative_inf'):
        value.flat[0] = {'nan': np.nan, 'inf': np.inf, 'negative_inf': -np.inf}[kind]
    elif kind == 'missing_axis':
        value = value[0]
    elif kind == 'empty':
        value = value[:0]
    elif kind == 'strings':
        value = value.astype(str)
    elif kind == 'bool':
        value = value.astype(bool)
    elif kind == 'structured':
        value = np.zeros(value.shape, dtype=[('value', 'f8'), ('other', 'f8')])
    elif kind == 'reflection':
        value[0] = np.diag([-1., 1., 1.])
    elif kind == 'scale':
        value[0, 0, 0] = 1.001
    elif kind == 'shear':
        value[0, 0, 1] = .001
    else:
        value[0, 0, 0] = 1. + 1e-8
    setattr(cal, field, value)
    for name in ('R', 't', 'points'):
        setattr(cal, name, getattr(cal, name).view(ArithmeticForbidden))
    before = snapshot(cal)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=field):
            cal.transformed(1.3, Rotation.from_rotvec([.2, -.1, .05]).as_matrix(), [.03, -.07, .11])
    assert_unchanged(cal, before)
