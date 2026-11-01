"""Public API: calibrate the extrinsics of a static multi-camera rig from 2D body keypoints."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bundle import refine as _refine
from .geometry import camera_centers, project_to_so3, triangulate_robust, unit
from .initialization import build_observations, initialize, segment_lengths
from .skeleton import COCO17, Skeleton


# -----------------------------------------------------------------------------
# input handling
# -----------------------------------------------------------------------------
def _as_intrinsics(K, n_cam):
    K = np.asarray(K, float)
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (n_cam, 3, 3)).copy()
    if K.shape != (n_cam, 3, 3):
        raise ValueError(f'K must have shape (3,3) or ({n_cam},3,3), got {K.shape}')
    if np.any(np.abs(K[:, 0, 1]) > 1e-6 * np.abs(K[:, 0, 0])):
        warnings.warn('non-zero skew in K is ignored', stacklevel=3)
    return K


def to_centered(keypoints, K):
    """Pixel keypoints ``(C,...,2)`` -> principal-point-centred, square-pixel coordinates.

    ``u' = u - cx``, ``v' = (v - cy) * fx / fy``; the returned focal length is ``fx``.
    """
    kp = np.asarray(keypoints, float)
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
    kp = np.asarray(keypoints, float)
    conf = np.asarray(confidence, float)
    if kp.ndim != 4 or kp.shape[-1] != 2:
        raise ValueError(f'keypoints must have shape (cameras, frames, joints, 2), got {kp.shape}')
    if conf.shape != kp.shape[:3]:
        raise ValueError(f'confidence must have shape {kp.shape[:3]}, got {conf.shape}')
    C, F, J, _ = kp.shape
    uv, f = to_centered(kp, K)
    valid = np.isfinite(uv).all(-1) & np.isfinite(conf) & (conf >= min_confidence)
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
        reprojection_rmse: ``(C,)`` RMS reprojection error of inlier observations, px.
        inlier_fraction: ``(C,)`` share of used observations with robust weight > 0.5.
        info: diagnostics of initialisation and refinement.
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

    @property
    def centers(self) -> np.ndarray:
        """Camera centres in world coordinates, ``(C,3)``."""
        return camera_centers(self.R, self.t)

    def projection_matrices(self) -> np.ndarray:
        """``(C,3,4)`` matrices ``P = K [R | t]``."""
        return np.einsum('cij,cjk->cik', self.K, np.concatenate([self.R, self.t[:, :, None]], 2))

    def transformed(self, s=1.0, Rw=np.eye(3), tw=np.zeros(3)) -> 'Calibration':
        """Change the world frame: new world point ``X' = s * Rw @ X + tw``."""
        Rw = np.asarray(Rw, float)
        tw = np.asarray(tw, float)
        R = self.R @ Rw.T
        t = s * self.t - np.einsum('cij,j->ci', R, tw)
        pts = s * self.points @ Rw.T + tw
        L = None if self.segment_lengths is None else s * self.segment_lengths
        return Calibration(R, t, self.K.copy(), pts, L, self.scale_source, self.reprojection_rmse.copy(),
                           self.inlier_fraction.copy(), dict(self.info))

    def to_dict(self) -> dict:
        return dict(R=self.R.tolist(), t=self.t.tolist(), K=self.K.tolist(),
                    centers=self.centers.tolist(), scale_source=self.scale_source,
                    segment_lengths=None if self.segment_lengths is None else self.segment_lengths.tolist(),
                    reprojection_rmse_px=self.reprojection_rmse.tolist(),
                    inlier_fraction=self.inlier_fraction.tolist())

    def save_json(self, path) -> None:
        """Save cameras (not points) to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @staticmethod
    def load_json(path) -> 'Calibration':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        C = len(d['R'])
        L = d.get('segment_lengths')
        return Calibration(np.array(d['R']), np.array(d['t']), np.array(d['K']), np.zeros((0, 0, 3)),
                           None if L is None else np.array(L), d.get('scale_source', 'unknown'),
                           np.array(d.get('reprojection_rmse_px', [np.nan] * C)),
                           np.array(d.get('inlier_fraction', [np.nan] * C)))


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
        K: intrinsics, ``(3,3)`` shared or ``(C,3,3)`` per camera.
        skeleton: keypoint layout (default COCO-17).
        min_confidence: observations below this confidence are discarded.
        camera_distance: ``(i, j, metres)`` measured distance between the centres of
            cameras ``i`` and ``j``; sets the metric scale (recommended).
        segment_lengths: measured segment lengths in metres, ``(B,)`` with NaN for
            unknown entries or ``{segment_name: metres}``; sets the scale if
            ``camera_distance`` is not given.
        seed: seed of the random generator used by the robust estimators.
    """

    def __init__(self, keypoints, confidence, K, *, skeleton: Skeleton = COCO17, min_confidence=0.3,
                 camera_distance=None, segment_lengths=None, seed=0):
        self.skeleton = skeleton
        self.K = _as_intrinsics(K, np.asarray(keypoints).shape[0])
        self.tracks, self.f, (self.C, self.F, self.J) = _tracks(keypoints, confidence, self.K, min_confidence)
        if self.J != skeleton.num_keypoints:
            raise ValueError(f'{self.J} keypoints given, skeleton has {skeleton.num_keypoints}')
        self.camera_distance = camera_distance
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
                arr[self.skeleton.segment_names.index(k) if isinstance(k, str) else int(k)] = v
            return arr
        arr = np.asarray(lengths, float)
        if arr.shape != (B,):
            raise ValueError(f'segment_lengths must have {B} entries')
        return arr

    def _scale(self, Rs, ts, L):
        """Multiplier bringing the current reconstruction to metres, and its source."""
        if self.camera_distance is not None:
            i, j, d = self.camera_distance
            Ce = camera_centers(Rs, ts)
            return float(d / np.linalg.norm(Ce[i] - Ce[j])), 'camera_distance'
        ref, src = self.measured_lengths, 'segment_lengths'
        if ref is None:
            if self.skeleton.typical_lengths is None:
                return 1.0, 'arbitrary'
            ref, src = self.skeleton.typical_lengths, 'prior'
        m = np.isfinite(ref) & np.isfinite(L) & (L > 0)
        if not m.any():
            return 1.0, 'arbitrary'
        return float(np.sum(ref[m]) / np.sum(L[m])), src

    def _pre_scale(self, L):
        """Scale used before refinement (segment term needs metres): measured or typical lengths."""
        ref = self.measured_lengths if self.measured_lengths is not None else self.skeleton.typical_lengths
        if ref is None:
            return 1.0
        m = np.isfinite(ref) & np.isfinite(L) & (L > 0)
        return float(np.sum(ref[m]) / np.sum(L[m])) if m.any() else 1.0

    def initialize(self, init_confidence=0.6, local_refine=True):
        """Initial cameras and points (incremental SfM), brought to an approximately metric scale."""
        t0 = time.perf_counter()
        Rs, ts, X, ok, info = initialize(self.tracks, self.f, self.rng, init_confidence=init_confidence,
                                         local_refine=local_refine)
        seg = self.skeleton.segments
        L = segment_lengths(X, ok, self.F, self.J, seg)
        s0 = self._pre_scale(L)
        ts, X, L = ts * s0, X * s0, L * s0
        bad = ~np.isfinite(L)                          # segment never reconstructed
        if bad.any() and self.skeleton.typical_lengths is not None:
            L[bad] = self.skeleton.typical_lengths[bad] * np.nanmedian(L / self.skeleton.typical_lengths)
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
        inl = out['w'] > 0.5
        rmse = np.array([np.sqrt(np.mean(out['r_px'][(obs['ci'] == k) & inl] ** 2))
                         if np.any((obs['ci'] == k) & inl) else np.nan for k in range(self.C)])
        frac = np.array([np.mean(inl[obs['ci'] == k]) if np.any(obs['ci'] == k) else np.nan
                         for k in range(self.C)])
        info = dict(it['info'], refine_time_s=out['time'], n_irls=out['n_irls'], nfev=out['nfev'],
                    sigma_hat=out['sig_hat'], focal=f, residuals_px=out['r_px'], weights=out['w'],
                    loss=loss, weighted=weighted, use_segments=use_segments)
        return Calibration(Rs, s * ts, K, s * X.reshape(self.F, self.J, 3),
                           None if L is None else s * np.asarray(L), src, rmse, frac, info)


def calibrate(keypoints, confidence, K, *, skeleton: Skeleton = COCO17, min_confidence=0.3,
              camera_distance=None, segment_lengths=None, loss='cauchy', weighted=True, use_segments=True,
              refine_focal=False, seed=0) -> Calibration:
    """Calibrate the extrinsics of a static, synchronised multi-camera rig from the 2D
    keypoints of one moving person.

    The result is expressed in the frame of camera 0 (``R0 = I, t0 = 0``); use
    :func:`align_to_floor` for a floor-aligned frame. See :class:`Calibrator` for the
    arguments.

    Example:
        >>> calib = calibrate(kp, conf, K, camera_distance=(0, 1, 5.32))
        >>> calib.R, calib.t, calib.reprojection_rmse
    """
    cal = Calibrator(keypoints, confidence, K, skeleton=skeleton, min_confidence=min_confidence,
                     camera_distance=camera_distance, segment_lengths=segment_lengths, seed=seed)
    return cal.initialize().refine(loss=loss, weighted=weighted, use_segments=use_segments,
                                   refine_focal=refine_focal)


# -----------------------------------------------------------------------------
# using a calibration
# -----------------------------------------------------------------------------
def triangulate(calib: Calibration, keypoints, confidence, *, min_confidence=0.3):
    """Robustly triangulate keypoints ``(C,F,J,2)`` with a calibration.

    Returns ``points (F,J,3)`` (NaN where fewer than two consistent views) and the
    per-observation inlier mask ``(C,F,J)``.
    """
    tracks, f, (C, F, J) = _tracks(keypoints, confidence, calib.K, min_confidence)
    X, ok, _, inl, _ = triangulate_robust(calib.R, calib.t, f, tracks['uv'], tracks['c'], tracks['valid'])
    X[~ok] = np.nan
    return X.reshape(F, J, 3), inl.reshape(F, J, C).transpose(2, 0, 1)


def project(calib: Calibration, points):
    """Project world points ``(...,3)`` to pixel coordinates in every camera: ``(C,...,2)``."""
    P = np.asarray(points, float)
    Xc = np.einsum('cij,...j->c...i', calib.R, P) + calib.t.reshape((-1,) + (1,) * (P.ndim - 1) + (3,))
    with np.errstate(divide='ignore', invalid='ignore'):
        xn = Xc[..., :2] / Xc[..., 2:3]
    K = calib.K.reshape((-1,) + (1,) * (P.ndim - 1) + (3, 3))
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


def align_to_floor(calib: Calibration, *, foot_keypoints=(15, 16), foot_height=0.08,
                   origin_camera=0, x_camera=1) -> Calibration:
    """Re-express a calibration in a floor-aligned world frame.

    The floor is fitted robustly to the lower half of the reconstructed foot keypoints
    (ankles by default: during stance they stay at about ``foot_height`` metres above
    the floor). The result has ``z`` pointing up, the floor at ``z = 0``, the origin
    below ``origin_camera`` and the ``x`` axis pointing towards ``x_camera``. Requires
    a metric scale for ``foot_height`` to be meaningful.
    """
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
    x = unit(xd - (xd @ z) * z)
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])                           # rows: new axes in old coordinates
    Rw = project_to_so3(Rw)
    return calib.transformed(1.0, Rw, -Rw @ o)


def reprojection_errors(calib: Calibration, keypoints, points):
    """Per-observation reprojection error (px) of ``points (F,J,3)`` against ``keypoints (C,F,J,2)``."""
    uv = project(calib, points)
    return np.linalg.norm(uv - np.asarray(keypoints, float), axis=-1)


__all__ = ['Calibration', 'Calibrator', 'calibrate', 'triangulate', 'project', 'align_to_floor',
           'reprojection_errors', 'to_centered']
