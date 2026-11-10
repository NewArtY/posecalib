"""Dependency-free adapter for exported MediaPipe normalized image landmarks.

Use undistorted images and intrinsics for those same images. This adapter does
not undistort, synchronize, select people, or use MediaPipe world coordinates.
"""
from pathlib import Path

import numpy as np

from .validation import validate_intrinsics, validate_real_array


def prepare_blazepose(landmarks, image_size, K, *, timestamps=None):
    """Return pixel observation arrays ready for np.savez and temporal alignment.

    landmarks: (C,F,33,4/5), columns x,y,z,visibility[,presence]. Missing
        detections must be NaN (not zero-filled). z is deliberately unused.
    image_size: (C,2) widths/heights, or a shared (2,). x/y are multiplied
        by width/height, without clipping legitimate out-of-image detections.
    K: (C,3,3) pixel intrinsics, or shared (3,3).
    timestamps: optional (C,F) local camera timestamps in seconds, finite and
        strictly increasing within each camera. Camera clocks may differ, including
        constant offsets. Clocks are preserved exactly; no resampling is performed.
        calibrate/triangulate align known timestamps without estimate_sync.
        Unknown residual clock biases require a separate estimate_sync step.

    Confidence is min(visibility,presence) when presence exists. Nonfinite
    x/y or scores and zero/negative scores produce NaN pixels and zero score.
    Nonfinite z alone does not invalidate a usable 2D observation.
    """
    lm = validate_real_array(landmarks, 'landmarks')
    if lm.ndim != 4 or lm.shape[2] != 33 or lm.shape[3] not in (4, 5):
        raise ValueError('landmarks must have shape (C,F,33,4 or 5)')
    C, F = lm.shape[:2]
    if C < 1 or F < 1:
        raise ValueError('landmarks require nonempty cameras and frames')
    sizes = validate_real_array(image_size, 'image_size')
    if sizes.shape == (2,):
        sizes = np.broadcast_to(sizes, (C, 2))
    if sizes.shape != (C, 2) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('image_size must contain positive finite (width,height) per camera')
    intrinsics = validate_intrinsics(K, C)
    score = np.min(lm[..., 3:], axis=-1)
    valid = np.isfinite(lm[..., :2]).all(-1) & np.isfinite(lm[..., 3:]).all(-1) & (score > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        pixels = lm[..., :2] * sizes[:, None, None, :]
    valid &= np.isfinite(pixels).all(-1)
    result = dict(keypoints=np.where(valid[..., None], pixels, np.nan),
                  confidence=np.where(valid, np.clip(score, 0, 1), 0),
                  K=intrinsics.copy(), image_size=sizes.copy())
    if timestamps is not None:
        ts = validate_real_array(timestamps, 'timestamps')
        if (ts.shape != (C, F) or not np.isfinite(ts).all()
                or (np.diff(ts, axis=1) <= 0).any()):
            raise ValueError('timestamps must be finite (C,F) seconds, strictly increasing per camera')
        result['timestamps'] = ts.copy()
    return result


def load_blazepose_cameras(paths):
    """Load one NPZ per camera, preserving input order.

    Each file: landmarks (F,33,4/5), image_size (2,) = width,height,
    K (3,3), optional local timestamps (F,) in seconds. All files must follow
    the same single person. Unequal lengths require timestamps in every file;
    they are padded with NaN/zero and frame_counts records real prefix lengths.
    Padding timestamps extrapolate each camera's median interval for storage;
    they are excluded from alignment when frame_counts is passed through.
    Equal-length files may omit timestamps. Local clocks are preserved, never
    rebased. Timestamp metadata must be supplied by every camera or none. Mixed 4/5-column exports
    are supported (absent presence leaves visibility unchanged).
    """
    rows, sizes, matrices, times = [], [], [], []
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as data:
            lm = validate_real_array(data['landmarks'], 'landmarks')
            if lm.ndim != 3 or lm.shape[1] != 33 or lm.shape[2] not in (4, 5):
                raise ValueError(f'{path}: landmarks must have shape (F,33,4 or 5)')
            if lm.shape[-1] == 4:
                lm = np.concatenate((lm, np.ones(lm.shape[:-1] + (1,))), axis=-1)
            size, matrix = np.asarray(data['image_size']), np.asarray(data['K'])
            if size.shape != (2,) or matrix.shape != (3, 3):
                raise ValueError(f'{path}: image_size must be (2,), K must be (3,3)')
            rows.append(lm)
            sizes.append(size)
            matrices.append(matrix)
            times.append(np.asarray(data['timestamps']) if 'timestamps' in data else None)
    if not rows or any(len(row) == 0 for row in rows):
        raise ValueError('supply camera files with nonzero frame counts')
    counts = np.array([len(row) for row in rows])
    if len(set(counts)) != 1 and any(t is None for t in times):
        raise ValueError('unequal frame counts require timestamps for every camera')
    if any(t is not None for t in times) and any(t is None for t in times):
        raise ValueError('timestamps must be present for every camera or none')
    frames = int(counts.max())
    padded = np.full((len(rows), frames, 33, 5), np.nan)
    padded_times = None if times[0] is None else np.empty((len(rows), frames))
    for c, (row, ts) in enumerate(zip(rows, times)):
        n = len(row)
        padded[c, :n] = row
        if ts is not None:
            ts = validate_real_array(ts, 'timestamps')
            if (ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0)):
                raise ValueError('each timestamps array must be finite (F,) seconds, strictly increasing')
            if n < frames and n < 2:
                raise ValueError('unequal frame counts require at least two timestamps to pad a camera clock')
            padded_times[c, :n] = ts
            if n < frames:
                padded_times[c, n:] = ts[-1] + np.arange(1, frames-n+1)*np.median(np.diff(ts))
    result = prepare_blazepose(padded, sizes, matrices, timestamps=padded_times)
    result['frame_counts'] = counts
    return result
