"""Public API: calibrate the extrinsics of a static multi-camera rig from 2D body keypoints."""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bundle import refine as _refine
from .geometry import camera_centers, project_to_so3, triangulate_robust, unit
from .initialization import build_observations, initialize, segment_lengths
from .skeleton import BLAZEPOSE33, COCO17, Skeleton
from .validation import (validate_camera_distance, validate_extrinsics, validate_index,
                         validate_intrinsics, validate_real_array, validate_result_lengths, validate_scalar,
                         validate_similarity)


# -----------------------------------------------------------------------------
# input handling
# -----------------------------------------------------------------------------
def _as_intrinsics(K, n_cam):
    return validate_intrinsics(K, n_cam)


def _reference_times(value):
    if value is None:
        return None
    times = validate_real_array(value, 'reference_times')
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('reference_times must be a finite strictly increasing vector in seconds')
    return times.copy()


def to_centered(keypoints, K):
    """Pixel keypoints ``(C,...,2)`` -> principal-point-centred, square-pixel coordinates.

    ``u' = u - cx``, ``v' = (v - cy) * fx / fy``; the returned focal length is ``fx``.
    """
    kp = validate_real_array(keypoints, 'keypoints')
    C = kp.shape[0]
    K = _as_intrinsics(K, C)
    shp = (C,) + (1,) * (kp.ndim - 2)
    cx, cy = K[:, 0, 2].reshape(shp), K[:, 1, 2].reshape(shp)
    aspect = (K[:, 0, 0] / K[:, 1, 1]).reshape(shp)
    out = np.empty_like(kp)
    out[..., 0] = kp[..., 0] - cx
    out[..., 1] = (kp[..., 1] - cy) * aspect
    return out, K[:, 0, 0].copy()


def _tracks(keypoints, confidence, K, min_confidence):
    kp = validate_real_array(keypoints, 'keypoints')
    conf = validate_real_array(confidence, 'confidence')
    min_confidence = validate_real_array(min_confidence, 'min_confidence')
    if kp.ndim != 4 or kp.shape[-1] != 2:
        raise ValueError(f'keypoints must have shape (cameras, frames, joints, 2), got {kp.shape}')
    if conf.shape != kp.shape[:3]:
        raise ValueError(f'confidence must have shape {kp.shape[:3]}, got {conf.shape}')
    C, F, J, _ = kp.shape
    uv, f = to_centered(kp, K)
    valid = (np.isfinite(uv).all(-1) & np.isfinite(conf)
             & (conf > 0) & (conf >= min_confidence))
    uv = np.where(valid[..., None], uv, 0.0)
    conf = np.where(valid, np.clip(conf, 0.0, 1.0), 0.0)
    tracks = dict(uv=uv.transpose(1, 2, 0, 3).reshape(F * J, C, 2),
                  c=conf.transpose(1, 2, 0).reshape(F * J, C),
                  valid=valid.transpose(1, 2, 0).reshape(F * J, C))
    return tracks, f, (C, F, J)


# -----------------------------------------------------------------------------
# result
# -----------------------------------------------------------------------------
@dataclass
class Calibration:
    """Calibrated rig.

    Attributes:
        R: ``(C,3,3)`` world-to-camera rotations (``Xc = R @ X + t``).
        t: ``(C,3)`` translations.
        K: ``(C,3,3)`` intrinsics (focal lengths updated if they were refined).
        points: ``(F,J,3)`` reconstructed calibration keypoints, NaN where not triangulated.
        segment_lengths: estimated segment lengths (metres if the scale is metric).
        scale_source: ``'camera_distance'``, ``'segment_lengths'`` or ``'prior'`` (assumed
            adult body proportions: approximately metric only).
        reprojection_rmse: ``(C,)`` RMS error in undistorted image pixels over
            observations with robust weight > 0.5, using the returned intrinsics.
        inlier_fraction: ``(C,)`` share of used observations with robust weight > 0.5.
        info: diagnostics of initialisation and refinement. After refinement,
            ``residuals_px`` holds one unweighted image-pixel distance per bundle
            observation in the same order as ``weights``. ``sigma_hat`` is the
            optimizer's scale in centered, equal-focal coordinates, with
            confidence weighting when enabled.
        time_offsets_seconds: optional per-camera source sampling offsets relative
            to camera 0. These are saved in JSON and reused by triangulate().
    """

    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    points: np.ndarray
    segment_lengths: np.ndarray | None
    scale_source: str
    reprojection_rmse: np.ndarray
    inlier_fraction: np.ndarray
    info: dict = field(default_factory=dict)
    time_offsets_seconds: np.ndarray | None = None

    def __post_init__(self):
        self.R, self.t = validate_extrinsics(self.R, self.t)
        self.K = _as_intrinsics(self.K, len(self.R))
        self.segment_lengths = validate_result_lengths(self.segment_lengths)
        if self.info.get('reference_times') is not None:
            self.info['reference_times'] = _reference_times(self.info['reference_times'])

    @property
    def centers(self) -> np.ndarray:
        """Camera centres in world coordinates, ``(C,3)``."""
        return camera_centers(self.R, self.t)

    def projection_matrices(self) -> np.ndarray:
        """``(C,3,4)`` matrices ``P = K [R | t]``."""
        K = _as_intrinsics(self.K, len(self.R))
        return np.einsum('cij,cjk->cik', K, np.concatenate([self.R, self.t[:, :, None]], 2))

    def transformed(self, s=1.0, Rw=np.eye(3), tw=np.zeros(3)) -> 'Calibration':
        """Change world frame: ``X' = s * Rw @ X + tw``.

        Requires finite positive s, a proper SO(3) Rw, and a finite three-vector
        tw. Rw uses the same precision-aware roundoff admission/canonicalization
        as camera rotations (see validate_extrinsics). Temporal metadata is preserved.
        """
        s, Rw, tw = validate_similarity(s, Rw, tw)
        R, t = validate_extrinsics(self.R, self.t)
        R = R @ Rw.T
        t = s * t - np.einsum('cij,j->ci', R, tw)
        pts = s * self.points @ Rw.T + tw
        L = None if self.segment_lengths is None else s * self.segment_lengths
        return Calibration(R, t, self.K.copy(), pts, L, self.scale_source, self.reprojection_rmse.copy(),
                           self.inlier_fraction.copy(), dict(self.info),
                           None if self.time_offsets_seconds is None else self.time_offsets_seconds.copy())

    def to_dict(self) -> dict:
        R, t = validate_extrinsics(self.R, self.t)
        K = _as_intrinsics(self.K, len(self.R))
        lengths = validate_result_lengths(self.segment_lengths)
        return dict(R=R.tolist(), t=t.tolist(), K=K.tolist(),
                    centers=camera_centers(R, t).tolist(), scale_source=self.scale_source,
                    segment_lengths=None if lengths is None else lengths.tolist(),
                    reprojection_rmse_px=self.reprojection_rmse.tolist(),
                    inlier_fraction=self.inlier_fraction.tolist(),
                    time_offsets_seconds=None if self.time_offsets_seconds is None else self.time_offsets_seconds.tolist(),
                    skeleton=self.info.get('skeleton'), foot_keypoints=self.info.get('foot_keypoints'),
                    synchronization=self.info.get('synchronization'),
                    reference_times=None if self.info.get('reference_times') is None else
                        _reference_times(self.info['reference_times']).tolist())

    def save_json(self, path) -> None:
        """Save cameras (not points), replacing the file only after a complete write."""
        text = json.dumps(self.to_dict(), indent=2)
        path = Path(path)
        stream = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix=f'.{path.name}.', suffix='.tmp', delete=False)
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            os.replace(temporary, path)
        except BaseException as error:
            # A failed write can fail again during close/flush. Unlink must
            # still run, and cleanup must not replace the primary I/O error.
            try:
                try:
                    stream.close()
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Temporary-file cleanup also failed: {cleanup_error}')
            raise

    @staticmethod
    def load_json(path) -> 'Calibration':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(d, dict) or not {'R', 't', 'K'} <= d.keys():
            raise ValueError('calibration JSON requires R, t and K')
        R, t = validate_extrinsics(d['R'], d['t'])
        C = len(R)
        L = d.get('segment_lengths')
        return Calibration(R, t, np.array(d['K']), np.zeros((0, 0, 3)),
                           None if L is None else np.array(L), d.get('scale_source', 'unknown'),
                           np.array(d.get('reprojection_rmse_px', [np.nan] * C)),
                           np.array(d.get('inlier_fraction', [np.nan] * C)),
                           dict(skeleton=d.get('skeleton'), foot_keypoints=d.get('foot_keypoints'),
                                synchronization=d.get('synchronization'),
                                **({} if d.get('reference_times') is None else
                                   dict(reference_times=d['reference_times']))),
                           None if d.get('time_offsets_seconds') is None else np.array(d['time_offsets_seconds']))


# -----------------------------------------------------------------------------
# calibrator
# -----------------------------------------------------------------------------
class Calibrator:
    """Two-stage calibration: :meth:`initialize` once, then :meth:`refine` (possibly
    several times with different options).

    Args:
        keypoints: ``(C,F,J,2)`` pixel coordinates of one person seen by ``C``
            synchronised cameras in ``F`` frames; NaN for missing detections. Images
            must be undistorted.
        confidence: ``(C,F,J)`` detector confidences in ``[0,1]``.
        K: canonical pinhole intrinsics, ``(3,3)`` shared or ``(C,3,3)`` per
            camera; finite, positive fx/fy, zero skew and K[1,0]. Bottom row
            [0,0,1] admits absolute roundoff <=1e-12, snapped to that exact row.
        skeleton: keypoint layout (inferred for 17 COCO / 33 BlazePose points).
        min_confidence: zero-confidence and below-threshold observations are discarded.
        camera_distance: ``(i, j, metres)`` measured distance between the centres of
            cameras ``i`` and ``j``; sets the metric scale (recommended).
        segment_lengths: measured segment lengths in metres, ``(B,)`` with NaN for
            unknown entries or ``{segment_name: metres}``; sets the scale if
            ``camera_distance`` is not given.
        seed: seed of the random generator used by the robust estimators.
    """

    def __init__(self, keypoints, confidence, K, *, skeleton: Skeleton | None = None, min_confidence=0.3,
                 camera_distance=None, segment_lengths=None, seed=0):
        keypoints = validate_real_array(keypoints, 'keypoints')
        if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
            raise ValueError(f'keypoints must have shape (cameras, frames, joints, 2), got {keypoints.shape}')
        if keypoints.shape[0] < 2:
            raise ValueError('keypoints must contain at least two cameras')
        if keypoints.shape[1] == 0:
            raise ValueError('keypoints must contain at least one frame (frames axis is empty)')
        if skeleton is None:
            from .skeleton import infer_skeleton
            skeleton = infer_skeleton(keypoints.shape[-2])
        self.skeleton = skeleton
        self.K = _as_intrinsics(K, keypoints.shape[0])
        self.tracks, self.f, (self.C, self.F, self.J) = _tracks(keypoints, confidence, self.K, min_confidence)
        self._keypoints = keypoints.copy()
        if self.J != skeleton.num_keypoints:
            raise ValueError(f'{self.J} keypoints given, skeleton has {skeleton.num_keypoints}')
        self.camera_distance = validate_camera_distance(camera_distance, self.C)
        self.measured_lengths = self._parse_lengths(segment_lengths)
        self.rng = np.random.default_rng(seed)
        self.init = None

    def _parse_lengths(self, lengths):
        if lengths is None:
            return None
        B = self.skeleton.num_segments
        if isinstance(lengths, dict):
            arr = np.full(B, np.nan)
            for k, v in lengths.items():
                if isinstance(k, str):
                    try:
                        index = self.skeleton.segment_names.index(k)
                    except ValueError as exc:
                        raise ValueError(f'unknown segment_lengths name: {k}') from exc
                else:
                    index = validate_index(k, B, 'segment_lengths index')
                value = np.asarray(v)
                if value.ndim != 0 or value.dtype.kind not in 'iuf':
                    raise ValueError('segment_lengths values must be positive finite scalars or NaN')
                arr[index] = v
        else:
            arr = np.asarray(lengths)
            if arr.dtype.kind not in 'iuf':
                raise ValueError('segment_lengths must contain positive finite values or NaN')
            arr = np.asarray(arr, float)
        if arr.shape != (B,):
            raise ValueError(f'segment_lengths must have {B} entries')
        if np.any(~np.isnan(arr) & (~np.isfinite(arr) | (arr <= 0))):
            raise ValueError('segment_lengths must contain positive finite values or NaN for unknown entries')
        return arr.copy()

    @staticmethod
    def _length_scale(ref, L):
        if ref is None or L is None:
            return None
        ref, L = np.asarray(ref, float), np.asarray(L, float)
        m = np.isfinite(ref) & (ref > 0) & np.isfinite(L) & (L > 0)
        if not m.any():
            return None
        # Normalize both sums together to avoid overflow from many large lengths.
        norm = max(float(ref[m].max()), float(L[m].max()))
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            scale = float(np.sum(ref[m] / norm) / np.sum(L[m] / norm))
        return scale if np.isfinite(scale) and scale > 0 else None

    def _scale(self, Rs, ts, L):
        """Multiplier bringing the current reconstruction to metres, and its source."""
        if self.camera_distance is not None:
            i, j, d = validate_camera_distance(self.camera_distance, self.C)
            Ce = camera_centers(Rs, ts)
            with np.errstate(over='ignore', invalid='ignore'):
                baseline = float(np.linalg.norm(Ce[i] - Ce[j]))
            validate_scalar(baseline, 'reconstructed camera_distance baseline')
            with np.errstate(over='ignore', divide='ignore', under='ignore'):
                scale = float(np.divide(d, baseline))
            return validate_scalar(scale, 'camera_distance scale'), 'camera_distance'
        ref, src = self.measured_lengths, 'segment_lengths'
        if ref is None:
            if self.skeleton.typical_lengths is None:
                return 1.0, 'arbitrary'
            ref, src = self.skeleton.typical_lengths, 'prior'
        scale = self._length_scale(ref, L)
        return (1.0, 'arbitrary') if scale is None else (scale, src)

    def _pre_scale(self, L):
        """Scale used before refinement (segment term needs metres): measured or typical lengths."""
        ref = self.measured_lengths if self.measured_lengths is not None else self.skeleton.typical_lengths
        if ref is None:
            return 1.0
        scale = self._length_scale(ref, L)
        return 1.0 if scale is None else scale

    def initialize(self, init_confidence=0.6, local_refine=True):
        """Initial cameras and points (incremental SfM), brought to an approximately metric scale."""
        t0 = time.perf_counter()
        Rs, ts, X, ok, info = initialize(self.tracks, self.f, self.rng, init_confidence=init_confidence,
                                         local_refine=local_refine)
        seg = self.skeleton.segments
        L = segment_lengths(X, ok, self.F, self.J, seg)
        s0 = self._pre_scale(L)
        ts, X, L = ts * s0, X * s0, L * s0
        # Unobserved segments remain unknown; do not invent anatomical lengths.
        obs = build_observations(self.tracks, ok, self.F, self.J, seg)
        info.update(time_s=time.perf_counter() - t0, n_points=int(ok.sum()), n_observations=len(obs['ci']))
        self.init = dict(Rs=Rs, ts=ts, X=X, ok=ok, L=L, obs=obs, info=info)
        return self

    def refine(self, *, loss='cauchy', weighted=True, use_segments=True, refine_focal=False,
               robust_k=2.5, segment_sigma=0.01, max_nfev=60, irls_max_iter=20) -> Calibration:
        """Global bundle adjustment from the initial estimate.

        Args:
            loss: ``'cauchy'`` (default), ``'huber'`` or ``'linear'`` (plain least squares;
                not recommended: detector outliers dominate).
            weighted: weight observations by squared confidence.
            use_segments: add the constant-segment-length term.
            refine_focal: also refine one focal length per camera.
            robust_k: loss threshold in units of the robust residual scale.
            segment_sigma: segment-length deviation (m) equivalent to one residual sigma.
        """
        self.camera_distance = validate_camera_distance(self.camera_distance, self.C)
        if self.init is None:
            self.initialize()
        it = self.init
        obs = it['obs']
        use_segments = use_segments and self.skeleton.num_segments > 0
        out = _refine(it['Rs'], it['ts'], it['X'][it['ok']], self.f, obs, L=it['L'], weighted=weighted,
                      loss=loss, use_segments=use_segments, refine_focal=refine_focal, robust_k=robust_k,
                      segment_sigma=segment_sigma, max_nfev=max_nfev, irls_max_iter=irls_max_iter)
        Rs, ts, Xp, f = out['Rs'], out['ts'], out['X'], out['f']
        X = np.full((self.F * self.J, 3), np.nan)
        X[obs['track_of_point']] = Xp
        L = out['L'] if use_segments else segment_lengths(X, np.isfinite(X).all(1), self.F, self.J,
                                                          self.skeleton.segments)
        s, src = self._scale(Rs, ts, L)
        K = self.K.copy()
        K[:, 0, 0] *= f / self.f
        K[:, 1, 1] *= f / self.f
        cameras = obs['ci']
        inl = out['w'] > 0.5
        frac = np.array([np.mean(inl[cameras == k]) if np.any(cameras == k) else np.nan
                         for k in range(self.C)])
        try:
            feet = [self.skeleton.index('left_ankle'), self.skeleton.index('right_ankle')]
        except ValueError:
            feet = None
        info = dict(it['info'], refine_time_s=out['time'], n_irls=out['n_irls'], nfev=out['nfev'],
                    sigma_hat=out['sig_hat'], focal=f, weights=out['w'],
                    loss=loss, weighted=weighted, use_segments=use_segments,
                    skeleton='blazepose33' if self.skeleton is BLAZEPOSE33 else
                             ('coco17' if self.skeleton is COCO17 else 'custom'), foot_keypoints=feet)
        rmse = np.full(self.C, np.nan)
        result = Calibration(Rs, s * ts, K, s * X.reshape(self.F, self.J, 3),
                             None if L is None else s * np.asarray(L), src, rmse, frac, info)
        # Use the returned camera model and the original image measurements.
        # Keep observation ordering and robust inlier decisions from the optimizer.
        tracks = obs['track_of_point']
        point_indices = obs['pi']
        predicted = project(result, result.points.reshape(-1, 3)[tracks])[cameras, point_indices]
        measured = self._keypoints.reshape(self.C, -1, 2)[cameras, tracks[point_indices]]
        residuals = np.linalg.norm(predicted - measured, axis=-1)
        info['residuals_px'] = residuals
        for k in range(self.C):
            used = (cameras == k) & inl
            if used.any():
                rmse[k] = np.sqrt(np.mean(residuals[used] ** 2))
        return result


def calibrate(keypoints, confidence, K, *, skeleton: Skeleton | None = None, min_confidence=0.3,
              camera_distance=None, segment_lengths=None, loss='cauchy', weighted=True, use_segments=True,
              refine_focal=False, seed=0, estimate_sync=False, fps=None, timestamps=None,
              max_offset_seconds=0.1, frame_step=1, frame_counts=None) -> Calibration:
    """Calibrate the extrinsics of a static, synchronised multi-camera rig from the 2D
    keypoints of one moving person.

    The result is expressed in the frame of camera 0 (``R0 = I, t0 = 0``); use
    :func:`align_to_floor` for a floor-aligned frame. See :class:`Calibrator` for the
    arguments.

    Supplied strictly increasing ``timestamps`` describe known capture clocks:
    streams are resampled to camera 0's common timeline with zero residual
    offsets even when ``estimate_sync=False``. For already aligned arrays omit
    timestamps and fps. Scalar or per-camera fps can instead define zero-origin
    CFR clocks. This known-clock alignment requires no identifiable motion.
    Set ``estimate_sync=True`` with ``fps`` or ``timestamps`` in seconds to
    estimate additional unknown constant offsets before geometric calibration.
    ``max_offset_seconds`` bounds each offset relative to camera 0. Alignment
    crops the common interval and does not fill missing tracks across gaps.
    ``frame_step`` subsamples AFTER alignment; it never changes the input FPS.
    frame_counts optionally declares real prefix lengths (C,) before padding;
    cropping uses these lengths, not extrapolated padding timestamps. Unknown
    lengths cannot be inferred from a missing detection. Known-clock alignment
    uses continuous cubic interpolation without crossing misses/gaps. Returned point
    times, including origin after cropping/subsampling, are stored in
    ``info['reference_times']`` and saved/loaded with the rig. They describe the
    calibration recording only, never the time grid of a subsequent capture.

    Example:
        >>> calib = calibrate(kp, conf, K, camera_distance=(0, 1, 5.32))
        >>> calib.R, calib.t, calib.reprojection_rmse
    """
    if (isinstance(frame_step, (bool, np.bool_))
            or not isinstance(frame_step, (int, np.integer)) or frame_step < 1):
        raise ValueError('frame_step must be a positive integer')
    # Validate measurements and K before any temporal search or geometric solve.
    cal = Calibrator(keypoints, confidence, K, skeleton=skeleton, min_confidence=min_confidence,
                     camera_distance=camera_distance, segment_lengths=segment_lengths, seed=seed)
    from .synchronization import validate_frame_counts
    counts = validate_frame_counts(frame_counts, cal.C, cal.F)
    if frame_counts is not None and not estimate_sync:
        keypoints = np.asarray(keypoints, float).copy()
        confidence = np.asarray(confidence, float).copy()
        for c, n in enumerate(counts):
            keypoints[c, n:] = np.nan
            confidence[c, n:] = 0
    sync, reference_times = None, None
    if estimate_sync:
        from .synchronization import estimate_time_offsets
        sync = estimate_time_offsets(keypoints, confidence, K, fps=fps, timestamps=timestamps,
                                     max_offset_seconds=max_offset_seconds, seed=seed,
                                     min_confidence=min_confidence, frame_counts=counts)
        keypoints, confidence = sync.keypoints, sync.confidence
        reference_times = sync.timestamps
    elif timestamps is not None or fps is not None:
        from .synchronization import resample_keypoints
        keypoints, confidence, reference_times = resample_keypoints(
            keypoints, confidence, np.zeros(cal.C), fps=fps, timestamps=timestamps,
            min_confidence=min_confidence, frame_counts=counts)
    elif frame_counts is not None:
        # Without timing metadata arrays assert corresponding frame indices.
        keypoints, confidence = keypoints[:, :counts.min()], confidence[:, :counts.min()]
    keypoints, confidence = np.asarray(keypoints)[:, ::frame_step], np.asarray(confidence)[:, ::frame_step]
    cal = Calibrator(keypoints, confidence, K, skeleton=skeleton, min_confidence=min_confidence,
                     camera_distance=camera_distance, segment_lengths=segment_lengths, seed=seed)
    result = cal.initialize().refine(loss=loss, weighted=weighted, use_segments=use_segments,
                                    refine_focal=refine_focal)
    if sync is not None:
        result.time_offsets_seconds = sync.offsets_seconds
        result.info['synchronization'] = sync.diagnostics
    if reference_times is not None:
        result.info['reference_times'] = reference_times[::frame_step].copy()
    return result


# -----------------------------------------------------------------------------
# using a calibration
# -----------------------------------------------------------------------------
def triangulate(calib: Calibration, keypoints, confidence, *, min_confidence=0.3,
                apply_time_offsets=True, fps=None, timestamps=None, return_times=False,
                frame_counts=None, ignore_time_offsets=False):
    """Robustly triangulate keypoints ``(C,F,J,2)`` with a calibration.

    Returns ``points (F,J,3)`` (NaN where fewer than two consistent views) and the
    per-observation inlier mask ``(C,F,J)``.

    If the calibration stores temporal corrections, supply ``fps`` or
    ``timestamps``; they are applied automatically and F can shrink. Use
    ``timestamps`` even without stored corrections to align known clocks at zero
    residual offset. ``apply_time_offsets=False`` bypasses ALL resampling,
    including timestamps and frame_counts, for data you already aligned. To
    align known clocks while ignoring obsolete stored corrections, instead use
    ``ignore_time_offsets=True`` (with the default apply_time_offsets=True).
    fps can be scalar or per-camera; frame_counts declares real prefix lengths
    of padded arrays. Explicit timestamps override fps. Stored offsets only
    remain valid while the relative timing of the captures is unchanged.
    ``return_times=True`` adds the cropped camera-0 times as a third return value
    (None when no resampling ran); the default two-value return is unchanged.
    """
    # A mutable calibration may have changed since construction. Reject bad K
    # before resampling as well as before triangulation.
    K = _as_intrinsics(calib.K, len(calib.R))
    times = None
    if apply_time_offsets and ((calib.time_offsets_seconds is not None and not ignore_time_offsets)
                               or timestamps is not None or fps is not None):
        from .synchronization import resample_keypoints
        offsets = (np.zeros(len(calib.R)) if ignore_time_offsets or calib.time_offsets_seconds is None
                   else calib.time_offsets_seconds)
        keypoints, confidence, times = resample_keypoints(keypoints, confidence, offsets,
                                                      fps=fps, timestamps=timestamps,
                                                      min_confidence=min_confidence, frame_counts=frame_counts)
    elif apply_time_offsets and frame_counts is not None:
        from .synchronization import validate_frame_counts
        counts = validate_frame_counts(frame_counts, len(calib.R), np.asarray(keypoints).shape[1])
        keypoints, confidence = np.asarray(keypoints)[:, :counts.min()], np.asarray(confidence)[:, :counts.min()]
    tracks, f, (C, F, J) = _tracks(keypoints, confidence, K, min_confidence)
    X, ok, _, inl, _ = triangulate_robust(calib.R, calib.t, f, tracks['uv'], tracks['c'], tracks['valid'])
    X[~ok] = np.nan
    result = X.reshape(F, J, 3), inl.reshape(F, J, C).transpose(2, 0, 1)
    return (*result, times) if return_times else result


def project(calib: Calibration, points):
    """Project world points ``(...,3)`` to pixel coordinates in every camera: ``(C,...,2)``."""
    P = validate_real_array(points, 'points')
    Xc = np.einsum('cij,...j->c...i', calib.R, P) + calib.t.reshape((-1,) + (1,) * (P.ndim - 1) + (3,))
    with np.errstate(divide='ignore', invalid='ignore'):
        xn = Xc[..., :2] / Xc[..., 2:3]
    K = _as_intrinsics(calib.K, len(calib.R)).reshape((-1,) + (1,) * (P.ndim - 1) + (3, 3))
    return np.stack([K[..., 0, 0] * xn[..., 0] + K[..., 0, 2],
                     K[..., 1, 1] * xn[..., 1] + K[..., 1, 2]], -1)


def _fit_floor(P, up0, max_tilt_deg=30.0, n_iter=1000, seed=0):
    """Robust plane through stance-foot points: LMedS over point triples, then a
    least-squares fit to the inliers. Returns a point on the plane and the unit normal
    oriented along ``up0``."""
    rng = np.random.default_rng(seed)
    idx = np.argpartition(rng.random((n_iter, len(P))), 3, axis=1)[:, :3]
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-9
    n = n / np.maximum(nn, 1e-12)[:, None]
    n *= np.sign(n @ up0)[:, None]
    ok &= n @ up0 > np.cos(np.radians(max_tilt_deg))
    if not ok.any():
        raise ValueError('could not fit a floor plane')
    med = np.full(n_iter, np.inf)
    d = np.abs(np.einsum('bi,ni->bn', n[ok], P) - np.sum(n[ok] * a[ok], 1)[:, None])
    med[ok] = np.median(d, axis=1)
    best = int(np.argmin(med))
    n0, p0 = n[best], a[best]
    sig = max(1.4826 * med[best], 1e-3)
    inl = np.abs((P - p0) @ n0) < 2.5 * sig
    ctr = P[inl].mean(0)
    nrm = np.linalg.svd(P[inl] - ctr)[2][-1]
    return ctr, nrm if nrm @ up0 > 0 else -nrm


def align_to_floor(calib: Calibration, *, foot_keypoints=None, foot_height=0.08,
                   origin_camera=0, x_camera=1) -> Calibration:
    """Re-express a calibration in a floor-aligned world frame.

    The floor is fitted robustly to the lower half of the reconstructed foot keypoints
    (ankles by default: during stance they stay at about ``foot_height`` metres above
    the floor). The result has ``z`` pointing up, the floor at ``z = 0``, the origin
    below ``origin_camera`` and the ``x`` axis pointing towards ``x_camera``. Requires
    a metric scale for ``foot_height`` to be meaningful.
    """
    origin_camera = validate_index(origin_camera, len(calib.R), 'origin_camera')
    x_camera = validate_index(x_camera, len(calib.R), 'x_camera')
    if origin_camera == x_camera:
        raise ValueError('origin_camera and x_camera must be distinct')
    foot_height = validate_scalar(foot_height, 'foot_height', allow_zero=True)
    if foot_keypoints is None:
        foot_keypoints = calib.info.get('foot_keypoints')
        if foot_keypoints is None:
            layout = calib.info.get('skeleton')
            if layout == 'custom':
                raise ValueError('custom skeleton has no named ankles; supply foot_keypoints')
            foot_keypoints = (27, 28) if layout == 'blazepose33' else (15, 16)
    P = calib.points.reshape(-1, calib.points.shape[1], 3)[:, list(foot_keypoints)].reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 20:
        raise ValueError('too few reconstructed foot keypoints to fit the floor')
    up0 = unit(-calib.R[:, 1, :].mean(0))             # cameras are roughly upright: image y points down
    h = P @ up0
    ctr, z = _fit_floor(P[h <= np.median(h)], up0)    # the lower half is dominated by the stance foot
    C = calib.centers
    o = C[origin_camera] - ((C[origin_camera] - ctr) @ z) * z            # origin camera centre on the floor
    o = o - foot_height * z                                              # ankles are above the floor
    xd = C[x_camera] - C[origin_camera]
    horizontal = xd - (xd @ z) * z
    horizontal_norm = np.linalg.norm(horizontal)
    if (not np.isfinite(horizontal_norm)
            or horizontal_norm <= 1e-12 * max(1.0, np.linalg.norm(xd))):
        raise ValueError('floor x axis requires a nonzero horizontal camera baseline')
    x = horizontal / horizontal_norm
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])                           # rows: new axes in old coordinates
    Rw = project_to_so3(Rw)
    return calib.transformed(1.0, Rw, -Rw @ o)


def reprojection_errors(calib: Calibration, keypoints, points):
    """Distances in undistorted image pixels from projected ``points (F,J,3)``
    to ``keypoints (C,F,J,2)``, using the supplied calibration's intrinsics.
    """
    keypoints = validate_real_array(keypoints, 'keypoints')
    uv = project(calib, points)
    return np.linalg.norm(uv - keypoints, axis=-1)


__all__ = ['Calibration', 'Calibrator', 'calibrate', 'triangulate', 'project', 'align_to_floor',
           'reprojection_errors', 'to_centered']
