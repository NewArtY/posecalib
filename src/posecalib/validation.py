"""Shared input checks for the supported canonical pinhole camera model."""
from numbers import Integral

import numpy as np


# Absolute tolerance, with no relative tolerance. Accepted roundoff is snapped
# to the exact homogeneous row so pixel and homogeneous-P projections agree.
PINHOLE_ROW_ATOL = 1e-12

# Preserve the existing double-precision boundary. A float32 rotation rounded
# entrywise has Gram/determinant error O(eps32); 2*eps32 covers both. JSON has no
# dtype, so recognize exactly float32-representable entries after promotion too.
SO3_ATOL = 1e-8
SO3_FLOAT32_ATOL = 2 * np.finfo(np.float32).eps
SO3_CANONICAL_ATOL = 64 * np.finfo(float).eps


def validate_real_array(value, name):
    """Real numeric input, without discarding imaginary/structured data.

    Finiteness is checked by the caller: missing keypoints may contain NaN.
    Complex dtypes (including zero imaginary parts) are explicitly unsupported.
    """
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in 'iuf':
            raise ValueError('unsupported dtype')
        return arr.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain real numeric values') from exc


def _canonical_rotations(R, name):
    """Validate a (C,3,3) stack BEFORE correcting only admitted roundoff."""
    message = f'{name} must contain finite proper SO(3) rotations'
    if not np.isfinite(R).all():
        raise ValueError(message)
    with np.errstate(over='ignore', invalid='ignore'):
        float32_exact = np.all(R == R.astype(np.float32).astype(float), axis=(1, 2))
        atol = np.where(float32_exact, SO3_FLOAT32_ATOL, SO3_ATOL)
        gram_error = np.max(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2))
        det_error = np.abs(np.linalg.det(R) - 1.)
    if not np.all((gram_error <= atol) & (det_error <= atol)):
        raise ValueError(message)
    R = R.copy()
    # Leave already orthogonal float64 values byte-stable through save/load.
    # Correct larger admitted roundoff so composition cannot accumulate it.
    correct = (gram_error > SO3_CANONICAL_ATOL) | (det_error > SO3_CANONICAL_ATOL)
    if correct.any():
        U, _, Vt = np.linalg.svd(R[correct])
        R[correct] = U @ Vt
    return R


def validate_intrinsics(K, n_cam):
    """Return an owned (C,3,3) canonical K, broadcasting a shared (3,3).

    All entries must be finite, fx/fy strictly positive, and both K[0,1]
    (skew) and K[1,0] exactly zero. The bottom row must equal [0,0,1]
    within absolute 1e-12 (rtol=0); accepted roundoff is canonicalized.
    No homogeneous rescaling or unsupported off-diagonal terms are ignored.
    """
    K = validate_real_array(K, 'K')
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (n_cam, 3, 3))
    if K.shape != (n_cam, 3, 3):
        raise ValueError(f'K must have shape (3,3) or ({n_cam},3,3), got {K.shape}')
    if not np.isfinite(K).all():
        raise ValueError('K must contain only finite entries')
    if np.any(K[:, (0, 1), (0, 1)] <= 0):
        raise ValueError('K must have strictly positive focal lengths fx and fy')
    if np.any(K[:, 0, 1] != 0) or np.any(K[:, 1, 0] != 0):
        raise ValueError('K must have zero skew/off-diagonal entries K[0,1] and K[1,0]')
    if not np.allclose(K[:, 2], [0, 0, 1], atol=PINHOLE_ROW_ATOL, rtol=0):
        raise ValueError('K must have canonical bottom row [0,0,1] (absolute tolerance 1e-12)')
    K = K.copy()
    K[:, 2] = [0, 0, 1]
    return K


def validate_index(index, count, name):
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            or not 0 <= index < count):
        raise ValueError(f'{name} must be an integer in [0,{count})')
    return int(index)


def validate_scalar(value, name, *, allow_zero=False):
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} scalar')
    value = float(arr)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} scalar')
    return value


def validate_camera_distance(distance, n_cam):
    if distance is None:
        return None
    try:
        i, j, d = distance
    except (TypeError, ValueError) as exc:
        raise ValueError('camera_distance must be (i, j, positive finite metres)') from exc
    i = validate_index(i, n_cam, 'camera_distance i')
    j = validate_index(j, n_cam, 'camera_distance j')
    if i == j:
        raise ValueError('camera_distance requires distinct camera indices')
    return i, j, validate_scalar(d, 'camera_distance metres')


def validate_similarity(s, Rw, tw):
    """Positive similarity; SO(3) admission/canonicalization matches extrinsics."""
    s = validate_scalar(s, 'scale')
    Rw, tw = validate_real_array(Rw, 'Rw'), validate_real_array(tw, 'tw')
    if Rw.shape != (3, 3):
        raise ValueError('Rw must be a finite proper SO(3) rotation of shape (3,3)')
    Rw = _canonical_rotations(Rw[None], 'Rw')[0]
    if tw.shape != (3,) or not np.isfinite(tw).all():
        raise ValueError('tw must be a finite translation of shape (3,)')
    return s, Rw, tw


def validate_extrinsics(R, t):
    """Finite poses; proper SO(3), with only admitted roundoff canonicalized.

    Absolute Gram/determinant tolerance is 1e-8 (rtol=0), or 2*eps32
    for exactly float32-representable matrices, including legacy JSON values.
    Reflections, appreciable scales/shears and all complex dtypes are rejected.
    """
    R, t = validate_real_array(R, 'R'), validate_real_array(t, 't')
    if R.ndim != 3 or R.shape[1:] != (3, 3) or len(R) == 0:
        raise ValueError('R must have shape (C,3,3) with finite proper SO(3) rotations')
    R = _canonical_rotations(R, 'R')
    if t.shape != (len(R), 3) or not np.isfinite(t).all():
        raise ValueError('t must be finite with shape (C,3) matching R')
    return R, t.copy()


def validate_result_lengths(lengths):
    """Serialized estimates may be NaN (unknown), never invalid known lengths.

    Do not infer a skeleton or require nine segments: old/custom rigs may omit
    layout metadata, and a custom skeleton can have no segments.
    """
    if lengths is None:
        return None
    arr = np.asarray(lengths)
    if (arr.ndim != 1 or arr.dtype.kind not in 'iuf'
            or np.any(~np.isnan(arr) & (~np.isfinite(arr) | (arr <= 0)))):
        raise ValueError('segment_lengths must be a vector of positive finite values or NaN for unknown entries')
    return arr.astype(float, copy=True)
