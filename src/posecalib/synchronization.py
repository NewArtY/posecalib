"""Constant sub-frame camera offsets from cross-view epipolar consistency.

Convention: aligned[c, t] = raw[c, t + offsets_seconds[c]]. Camera 0 is
the time reference. Positive offsets skip the beginning of that source stream.
Only a constant offset is estimated, not clock drift or rolling shutter.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation

from .geometry import estimate_relative_pose, skew
from .validation import validate_real_array, validate_scalar


def time_axes(n_cameras, n_frames, *, fps=None, timestamps=None):
    if timestamps is None:
        rates = np.asarray(fps)
        if (rates.dtype.kind not in 'iuf' or rates.shape not in ((), (n_cameras,))
                or not np.isfinite(rates).all() or np.any(rates <= 0)):
            raise ValueError('a positive fps (scalar or per camera) or timestamps in seconds is required')
        rates = np.broadcast_to(rates, (n_cameras,))
        return np.arange(n_frames)[None, :] / rates[:, None]
    ts = validate_real_array(timestamps, 'timestamps')
    if ts.shape == (n_frames,):
        ts = np.broadcast_to(ts, (n_cameras, n_frames)).copy()
    if ts.shape != (n_cameras, n_frames) or not np.isfinite(ts).all():
        raise ValueError('timestamps must be finite (frames,) or (cameras, frames) seconds')
    if np.any(np.diff(ts, axis=1) <= 0):
        raise ValueError('timestamps must be strictly increasing for each camera')
    return ts


def validate_frame_counts(frame_counts, n_cameras, n_frames):
    """Real prefix lengths; missing detections INSIDE that prefix are not padding."""
    if frame_counts is None:
        return np.full(n_cameras, n_frames, dtype=int)
    counts = np.asarray(frame_counts)
    if (counts.shape != (n_cameras,) or counts.dtype.kind not in 'iu'
            or np.any(counts < 1) or np.any(counts > n_frames)):
        raise ValueError('frame_counts must contain one integer in [1, frames] per camera')
    return counts.astype(int)


def _sample(xy, conf, times, query, min_confidence=0.3):
    """Linear interpolation within adjacent valid observations; never bridge a gap."""
    query = np.asarray(query)
    hi = np.clip(np.searchsorted(times, query, side='right'), 1, len(times) - 1)
    lo = hi - 1
    a = (query - times[lo]) / (times[hi] - times[lo])
    p, q = xy[lo], xy[hi]
    cp, cq = conf[lo], conf[hi]
    near_lo = np.abs(a) < 1e-8
    near_hi = np.abs(a - 1) < 1e-8
    # Exact frame queries need only that frame, not its neighbour.
    p = np.where(near_hi[:, None, None], q, p)
    q = np.where(near_lo[:, None, None], p, q)
    cp = np.where(near_hi[:, None], cq, cp)
    cq = np.where(near_lo[:, None], cp, cq)
    valid = (np.isfinite(p).all(-1) & np.isfinite(q).all(-1)
             & np.isfinite(cp) & np.isfinite(cq)
             & (cp > 0) & (cq > 0)
             & (cp >= min_confidence) & (cq >= min_confidence))
    valid &= ((query >= times[0]) & (query <= times[-1]))[:, None]
    # Explicitly reject dropped-frame intervals rather than interpolating across them.
    valid &= (((times[hi] - times[lo]) <= 1.8 * np.median(np.diff(times))) | near_lo | near_hi)[:, None]
    out = (1 - a[:, None, None]) * p + a[:, None, None] * q
    return np.where(valid[..., None], out, np.nan), np.where(valid, np.minimum(cp, cq), 0.0)


def _sample_continuous(xy, conf, times, query, min_confidence):
    """Continuous cubic interpolation on contiguous valid runs, with exact hits.

    A not-a-knot spline avoids discontinuous slope/curvature as the query moves
    between intervals. No smoothing or extrapolation; two/three samples reduce
    to linear/quadratic interpolation. Support stops at a missed detection or a
    gap >1.8 median camera intervals. Confidence is the minimum of the enclosing
    two samples, not a model uncertainty estimate. Exact hits use their own score.
    Unknown-offset fitting uses a separate noise-aware motion model.
    """
    query = np.asarray(query, float)
    out = np.full((len(query), xy.shape[1], 2), np.nan)
    scores = np.zeros((len(query), xy.shape[1]))
    valid = (np.isfinite(xy).all(-1) & np.isfinite(conf)
             & (conf > 0) & (conf >= min_confidence))
    period = np.median(np.diff(times)) if len(times) > 1 else np.inf
    for joint in range(xy.shape[1]):
        indices = np.flatnonzero(valid[:, joint])
        if not len(indices):
            continue
        cuts = np.flatnonzero((np.diff(indices) != 1)
                              | (np.diff(times[indices]) > 1.8 * period)) + 1
        for run in np.split(indices, cuts):
            if len(run) < 2:
                continue  # singletons are recovered below as exact hits only
            selected = np.flatnonzero((query >= times[run[0]]) & (query <= times[run[-1]]))
            if not len(selected):
                continue
            q = query[selected]
            # Work relative to this run's origin, never powers of a wall clock.
            spline = CubicSpline(times[run]-times[run[0]], xy[run, joint], extrapolate=False)
            out[selected, joint] = spline(q-times[run[0]])
            left = np.clip(np.searchsorted(times[run], q, side='right')-1, 0, len(run)-2)
            scores[selected, joint] = np.minimum(conf[run[left], joint], conf[run[left+1], joint])
    # Copy the actual observation, including its own confidence, exactly. ULP
    # admission handles clock addition roundoff, not a fraction of a frame.
    high = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    low = np.maximum(high-1, 0)
    nearest = np.where(np.abs(query-times[low]) < np.abs(query-times[high]), low, high)
    exact = np.abs(query-times[nearest]) <= 4*np.finfo(float).eps*np.maximum(1., np.abs(query))
    hit = exact[:, None] & valid[nearest]
    out = np.where(hit[..., None], xy[nearest], out)
    scores = np.where(hit, conf[nearest], scores)
    return out, scores


@dataclass
class Synchronization:
    offsets_seconds: np.ndarray
    keypoints: np.ndarray
    confidence: np.ndarray
    timestamps: np.ndarray
    diagnostics: dict


def resample_keypoints(keypoints, confidence, offsets_seconds, *, fps=None, timestamps=None,
                       reference_times=None, min_confidence=0.3, frame_counts=None,
                       interpolation='cubic'):
    """Align streams to camera 0's timeline. Return (keypoints, confidence, times).

    Seconds retain the camera-0 origin; explicit timestamps override fps. fps may
    be scalar or (C,). frame_counts (C,) declares real prefixes of padded arrays;
    without it all F timestamps are real, even if their detections are missing.
    Samples outside the common REAL interval are cropped. The target is camera
    0's real grid (or explicit reference_times). Continuous cubic interpolation
    is fitted separately within each valid run; shorter runs reduce the degree.
    Exact valid observations are copied. Zero-confidence and below-threshold
    observations are missing and break interpolation support. 'linear' is also
    available. Neither method extrapolates or crosses gaps.
    Supply the SAME corrections for subsequent recordings only if relative
    capture timing is unchanged. This operation does not estimate any offset.
    """
    kp = validate_real_array(keypoints, 'keypoints')
    co = validate_real_array(confidence, 'confidence')
    min_confidence = validate_real_array(min_confidence, 'min_confidence')
    if kp.ndim != 4 or kp.shape[-1] != 2 or co.shape != kp.shape[:3]:
        raise ValueError('expected keypoints (C,F,J,2) and confidence (C,F,J)')
    C, F = kp.shape[:2]
    if C < 1 or F < 1:
        raise ValueError('nonempty camera tracks are required')
    if interpolation not in ('cubic', 'linear'):
        raise ValueError("interpolation must be 'cubic' or 'linear'")
    counts = validate_frame_counts(frame_counts, C, F)
    ts = time_axes(C, F, fps=fps, timestamps=timestamps)
    off = validate_real_array(offsets_seconds, 'offsets_seconds')
    if off.shape != (C,) or not np.isfinite(off).all():
        raise ValueError('one finite offset in seconds is required per camera')
    target = (ts[0, :counts[0]] if reference_times is None
              else validate_real_array(reference_times, 'reference_times'))
    if target.ndim!=1 or not np.isfinite(target).all() or np.any(np.diff(target)<=0):
        raise ValueError('reference_times must be a finite strictly increasing vector')
    start, end = np.max(ts[:, 0] - off), np.min(ts[np.arange(C), counts-1] - off)
    target = target[(target >= start - 1e-10) & (target <= end + 1e-10)]
    if len(target) < 1:
        raise ValueError('too little common time after alignment')
    arrays = []
    for c, n in enumerate(counts):
        sampler = _sample if interpolation == 'linear' and n > 1 else _sample_continuous
        arrays.append(sampler(kp[c, :n], co[c, :n], ts[c, :n],
                              np.clip(target+off[c], ts[c, 0], ts[c, n-1]), min_confidence))
    return np.stack([a[0] for a in arrays]), np.stack([a[1] for a in arrays]), target


def _signed_sampson(R, t, x, y):
    E = skew(t / max(np.linalg.norm(t), 1e-12)) @ R
    xh = np.column_stack([x, np.ones(len(x))])
    yh = np.column_stack([y, np.ones(len(y))])
    ex, ey = xh @ E.T, yh @ E
    return np.sum(yh * ex, axis=1) / np.sqrt(np.maximum(
        (ex[:, :2] ** 2).sum(1) + (ey[:, :2] ** 2).sum(1), 1e-20))


class _MotionTracks:
    """Robust local cubic motion, with the linear noise influence retained.

    Fits never cross a missing detection or timestamp gap. Hermite evaluation
    combines the *overlapping* fitting stencils before computing variance; treating
    neighbouring smoothed knots as independent would reintroduce phase bias.
    """

    def __init__(self, xy, confidence, times, min_confidence):
        self.times = times
        period = float(np.median(np.diff(times)))
        self.window = max(7, 2 * int(.15 / period) + 1)
        valid = (np.isfinite(xy).all(-1) & np.isfinite(confidence)
                 & (confidence > 0) & (confidence >= min_confidence))
        gaps = np.r_[False, np.diff(times) > 1.8 * period]
        runs, differences = [], []
        for j in range(xy.shape[1]):
            indices = np.flatnonzero(valid[:, j])
            for run in np.split(indices, np.flatnonzero(
                    (np.diff(indices) != 1) | gaps[indices[1:]]) + 1):
                if not len(run):
                    continue
                runs.append((j, run))
                if len(run) >= 4:
                    ix = np.lib.stride_tricks.sliding_window_view(run, 4)
                    t = times[ix] / period
                    delta = t[:, :, None] - t[:, None, :]
                    delta[:, np.arange(4), np.arange(4)] = 1.
                    w = 1. / np.prod(delta, axis=2)
                    w /= np.sqrt(np.sum((w / confidence[ix, j]) ** 2, axis=1))[:, None]
                    # Third divided differences remove local velocity AND acceleration.
                    differences.append(np.einsum('nk,nkd->nd', w, xy[ix, j]))
        self.noise = max(float(np.median(np.abs(np.concatenate(differences))) / .67448975)
                         if differences else 0., 1e-7)
        F, J = xy.shape[:2]
        self.starts = np.zeros((F, J), int)
        self.value = np.zeros((F, J, self.window))
        self.slope = np.zeros_like(self.value)
        self.points = xy.copy()
        self.velocity = np.zeros_like(xy)
        self.variance = (self.noise / np.maximum(confidence, 1e-3)) ** 2
        for j, run in runs:
            m = min(self.window, len(run))
            start = np.clip(np.arange(len(run)) - m // 2, 0, len(run) - m)
            ix = run[start[:, None] + np.arange(m)]
            z = (times[ix] - times[run, None]) / period
            A = z[..., None] ** np.arange(min(4, m))
            obs = xy[ix, j]
            weight = confidence[ix, j] ** 2
            for _ in range(4):
                H = np.einsum('nki,nk,nkl->nil', A, weight, A)
                L = np.linalg.solve(H, A.transpose(0, 2, 1) * weight[:, None, :])
                coeff = L @ obs
                error = np.linalg.norm(obs - A @ coeff, axis=2)
                weight = confidence[ix, j] ** 2 / (1 + (error / (3 * self.noise)) ** 2)
            self.starts[run, j] = ix[:, 0]
            self.value[run, j, :m] = L[:, 0]
            self.points[run, j] = coeff[:, 0]
            if m > 1:
                self.slope[run, j, :m] = L[:, 1] / period
                self.velocity[run, j] = coeff[:, 1] / period

    def at(self, query, joints):
        hi = np.clip(np.searchsorted(self.times, query, side='right'), 1, len(self.times) - 1)
        lo = hi - 1
        h = self.times[hi] - self.times[lo]
        u = (query - self.times[lo]) / h
        v0, v1 = 2*u**3 - 3*u**2 + 1, -2*u**3 + 3*u**2
        d0, d1 = (u**3 - 2*u**2 + u)*h, (u**3 - u**2)*h
        dv0, dv1 = (6*u**2 - 6*u)/h, (-6*u**2 + 6*u)/h
        dd0, dd1 = 3*u**2 - 4*u + 1, 3*u**2 - 2*u
        # At an exact sample a missing neighbour must not contaminate the result.
        p = np.nan_to_num(self.points[lo, joints])
        q = np.nan_to_num(self.points[hi, joints])
        vp, vq = self.velocity[lo, joints], self.velocity[hi, joints]
        points = v0[:, None]*p + v1[:, None]*q + d0[:, None]*vp + d1[:, None]*vq
        velocity = dv0[:, None]*p + dv1[:, None]*q + dd0[:, None]*vp + dd1[:, None]*vq
        start0, start1 = self.starts[lo, joints], self.starts[hi, joints]
        # Adjacent knots share a run, except exact endpoints next to a gap.
        # Accumulate both stencils on actual observation indices, not knot indices.
        idx0 = np.minimum(start0[:, None] + np.arange(self.window), len(self.times)-1)
        idx1 = np.minimum(start1[:, None] + np.arange(self.window), len(self.times)-1)
        c0 = v0[:, None]*self.value[lo, joints] + d0[:, None]*self.slope[lo, joints]
        c1 = v1[:, None]*self.value[hi, joints] + d1[:, None]*self.slope[hi, joints]
        e0 = dv0[:, None]*self.value[lo, joints] + dd0[:, None]*self.slope[lo, joints]
        e1 = dv1[:, None]*self.value[hi, joints] + dd1[:, None]*self.slope[hi, joints]
        var0, var1 = self.variance[idx0, joints[:, None]], self.variance[idx1, joints[:, None]]
        var = np.sum(c0*c0*var0 + c1*c1*var1, axis=1)
        dvar = np.sum(e0*e0*var0 + e1*e1*var1, axis=1)
        # Each stencil contains consecutive indices, so overlap needs one gather.
        k = idx0 - start1[:, None]
        overlap = (k >= 0) & (k < self.window)
        other = np.clip(k, 0, self.window-1)
        var += 2*np.sum(c0*np.take_along_axis(c1, other, axis=1)*var0*overlap, axis=1)
        dvar += 2*np.sum(e0*np.take_along_axis(e1, other, axis=1)*var0*overlap, axis=1)
        return points, velocity, np.maximum(var, 1e-20), np.maximum(dvar, 0.)


def _epipolar_residual(p, x, y, vx, vy):
    """Whiten by per-view localization variance, including interpolation phase."""
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    E = skew(p[3:6] / max(np.linalg.norm(p[3:6]), 1e-12)) @ R
    xh = np.column_stack([x, np.ones(len(x))])
    yh = np.column_stack([y, np.ones(len(y))])
    ex0, ex1, ex2 = (xh @ E.T).T
    ey0, ey1, _ = (yh @ E).T
    # The row reductions have fixed widths of two and three components.
    denom = np.sqrt(np.maximum(vy*(ex0**2 + ex1**2) + vx*(ey0**2 + ey1**2), 1e-30))
    return (y[:, 0]*ex0 + y[:, 1]*ex1 + ex2)/denom


def _has_nonuniform_motion(xy, confidence, times, noise, joints, start, end):
    """Reject tracks explained by constant 3D velocity under pinhole projection.

    Such a point projects to (p + v*t)/(1 + w*t) in both image coordinates.
    Fit to raw samples, so the veto is independent of interpolation phase.
    Require a majority of the observations on at least three tracks to deviate
    from this null by more than twice the per-coordinate localization scale.
    """
    in_time = (times >= start) & (times <= end)
    moving = 0
    for joint in joints:
        valid = in_time & (confidence[:, joint] > 0)
        if np.count_nonzero(valid) < 12:
            continue
        t = times[valid]
        t = 2*(t - t[0])/(t[-1] - t[0]) - 1
        points = xy[valid, joint]
        scores = confidence[valid, joint]
        design = np.zeros((len(t), 2, 5))
        design[:, 0, :2] = np.column_stack([np.ones(len(t)), t])
        design[:, 1, 2:4] = design[:, 0, :2]
        design[:, :, 4] = -points*t[:, None]
        weights = scores
        for _ in range(3):
            fit = np.linalg.lstsq((design*weights[:, None, None]).reshape(-1, 5),
                                  (points*weights[:, None]).ravel(), rcond=None)[0]
            depth = 1 + fit[4]*t
            # A pole in the recording is inadmissible. Use the affine submodel.
            if np.any(depth < .05):
                fit[:4] = np.linalg.lstsq(design[:, :, :4].reshape(-1, 4),
                                          points.ravel(), rcond=None)[0]
                fit[4] = 0.
                depth = np.ones_like(t)
            error = np.linalg.norm(design @ fit - points, axis=1)*scores/depth
            weights = scores / (depth*np.sqrt(1 + (error/(3*noise))**2))
        moving += np.median(error) > 2*noise
        if moving >= 3:
            return True
    return False


def _has_repeating_motion(xy, co, ts, cameras, joints, start, end, period, radius,
                          *, max_period=np.inf):
    """Check a candidate alias separation against raw tracks from both views.

    The motion smoother can suppress rapid repeats differently at different
    sampling phases. A shared frequency and its first three harmonics must explain
    at least 95% of the weighted variation on three or more tracks per view.
    The third harmonic helps represent nonsinusoidal repeats when the view
    samples it below Nyquist. Folded harmonics can otherwise mimic a shorter
    period and manufacture an in-bound alias.
    """
    low_period = max(period - radius, 2*max(np.median(np.diff(ts[c])) for c in cameras))
    high_period = min(period + radius, max_period)
    if low_period >= high_period or end-start < 3*high_period:
        return False
    tracks = []
    for c in cameras:
        in_time = (ts[c] >= start) & (ts[c] <= end)
        view = []
        for joint in joints:
            keep = in_time & (co[c][:, joint] > 0)
            if np.count_nonzero(keep) < 24:
                continue
            weights = co[c][keep, joint]
            points = xy[c][keep, joint]
            centered = points - np.average(points, weights=weights**2, axis=0)
            obs = centered*weights[:, None]
            energy = float(np.sum(obs**2))
            if energy > 0:
                view.append((energy, ts[c][keep]-start, weights, obs))
        if len(view) < 3:
            return False
        # Bound the cost and favor the tracks with a measurable repeat signal.
        nyquist = .5/np.median(np.diff(ts[c]))
        tracks.append((nyquist, sorted(view, key=lambda track: -track[0])[:6]))

    def unexplained(frequency):
        views = []
        for nyquist, view in tracks:
            fractions = []
            for energy, t, weights, obs in view:
                phase = 2*np.pi*frequency*t
                columns = [np.ones(len(t))]
                for harmonic in (1, 2, 3):
                    if harmonic*frequency <= nyquist:
                        columns.extend((np.sin(harmonic*phase), np.cos(harmonic*phase)))
                basis = np.column_stack(columns)*weights[:, None]
                error = basis @ np.linalg.lstsq(basis, obs, rcond=None)[0] - obs
                fractions.append(float(np.sum(error**2)/energy))
            views.append(np.sort(fractions)[2])
        return max(views)

    # Resolve frequency differences before fine search. A single bounded
    # minimization across the entire band could miss narrow spectral peaks.
    low, high = 1/high_period, 1/low_period
    grid = np.linspace(low, high, max(3, int(np.ceil(4*(high-low)*(end-start)))+1))
    costs = [unexplained(frequency) for frequency in grid]
    best = int(np.argmin(costs))
    if costs[best] < .05:
        return True
    fit = minimize_scalar(unexplained, bounds=(grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]),
                          method='bounded')
    return fit.fun < .05


def _raw_repeat_candidates(xy, co, ts, a, b, joints, start, end, bound):
    """Propose repeat periods from raw motion independently of the pose profile."""
    steps = [float(np.median(np.diff(ts[c]))) for c in (a, b)]
    camera = (a, b)[int(steps[1] < steps[0])]
    step = min(steps)
    span = end - start
    longest = min(2*bound, span/3)
    shortest = 2*max(steps)
    if longest <= shortest:
        return
    grid = np.arange(start, end, step)
    count = len(grid)
    window = np.hanning(count)
    power = np.zeros(2*count + 1)
    in_time = (ts[camera] >= start) & (ts[camera] <= end)
    for joint in joints:
        keep = in_time & (co[camera][:, joint] > 0)
        if np.count_nonzero(keep) < 24:
            continue
        # Uniform resampling is for peak proposals only. Confirmation fits the
        # original irregular timestamps and uses both views.
        t = ts[camera][keep]
        points = xy[camera][keep, joint]
        centered = points - np.mean(points, axis=0)
        sampled = np.stack([np.interp(grid, t, coord, left=0., right=0.)
                            for coord in centered.T], axis=1)
        spectrum = np.fft.rfft(sampled*window[:, None], n=4*count, axis=0)
        power += np.sum(np.abs(spectrum)**2, axis=1)
    freq = np.fft.rfftfreq(4*count, step)
    peaks = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    peaks = peaks[(freq[peaks] >= 1/longest) & (freq[peaks] <= 1/shortest)]
    for peak in peaks[np.argsort(power[peaks])[-4:]][::-1]:
        period = 1/freq[peak]
        # One frequency bin on the unpadded time span bounds the local refit.
        radius = 1/(freq[peak] - 1/span) - period
        yield period, radius


def _pair_offset(xy, co, ts, a, b, bound, rng, models, max_points=1800):
    # Use only tracks continuously available throughout the search interval.
    # This fixes the support: a bad offset cannot win by dropping hard points.
    target = ts[a]
    keep = (target >= ts[b][0] + bound) & (target <= ts[b][-1] - bound)
    target = target[keep]
    if len(target) < 12:
        raise RuntimeError('too short for the requested offset search interval')
    support = co[a][keep] > 0
    # A continuous interpolation segment is needed at every potential offset.
    lo = np.maximum(0, np.searchsorted(ts[b], target - bound, side='right') - 1)
    hi = np.minimum(len(ts[b]) - 1, np.searchsorted(ts[b], target + bound))
    vb = co[b] > 0
    period = float(np.median(np.diff(ts[b])))
    gaps = np.r_[0, np.cumsum(np.diff(ts[b]) > 1.8*period)]
    bad = np.vstack([np.zeros((1, vb.shape[1]), int), np.cumsum(~vb, axis=0)])
    support &= (bad[hi + 1] - bad[lo]) == 0
    support &= (gaps[hi] == gaps[lo])[:, None]
    fi, ji = np.nonzero(support)
    if len(fi) < 80:
        raise RuntimeError('too few continuous confident tracks for temporal alignment')
    ma, mb = models[a], models[b]
    joints = np.unique(ji)
    start, end = target[fi[0]], target[fi[-1]]
    if not (_has_nonuniform_motion(xy[a], co[a], ts[a], ma.noise, joints, start, end)
            and _has_nonuniform_motion(xy[b], co[b], ts[b], mb.noise, joints, start, end)):
        raise RuntimeError('insufficient nonuniform motion above localization noise')
    if len(fi) > max_points:
        sel = np.sort(rng.choice(len(fi), max_points, replace=False))
        fi, ji = fi[sel], ji[sel]
    tt = target[fi]
    x, _, vx, dvx = ma.at(tt, ji)

    # Finite differences in pose reuse the same interpolation and covariance.
    # Keep only a bounded number of lag evaluations per pair.
    @lru_cache(maxsize=32)
    def observations(offset):
        return mb.at(tt + offset, ji)

    def residual(p, fixed=None):
        y, _, vy, _ = observations(p[6] if fixed is None else fixed)
        return _epipolar_residual(p, x, y, vx, vy)

    def cost(r):
        return np.log1p((r / 2.) ** 2)

    def optimize(p, fixed=None, interval=None):
        def fun(q):
            # Fix the otherwise redundant translation norm (a gauge direction).
            return np.r_[residual(q, fixed), 10*(np.linalg.norm(q[3:6])-1)]
        lower, upper = np.full(len(p), -np.inf), np.full(len(p), np.inf)
        if fixed is None:
            lower[6], upper[6] = interval if interval is not None else (-bound, bound)
        fit = least_squares(fun, p, bounds=(lower, upper), loss='cauchy', f_scale=2.,
                            x_scale='jac', max_nfev=80, ftol=1e-7, xtol=1e-8, gtol=1e-7)
        return fit.x

    candidates = []
    # Independent essential fits propose poses; their scores are NEVER an
    # observability profile. All offsets will be refitted from shared best poses.
    for lag in np.linspace(-bound, bound, 5):
        y, _, _, _ = observations(lag)
        try:
            R, t, _, _ = estimate_relative_pose(x, y, rng, fpx=1200., n_iter=160, n_lo=2)
            p = optimize(np.r_[Rotation.from_matrix(R).as_rotvec(), t], fixed=lag)
            candidates.append((float(np.mean(cost(residual(p, lag)))), np.r_[p, lag]))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        raise RuntimeError('no viable relative-pose hypothesis')
    fits = [optimize(p) for _, p in sorted(candidates, key=lambda z: z[0])[:3]]
    best = min(fits, key=lambda p: np.mean(cost(residual(p))))

    # Differentiate the MOTION model, not raw noisy interpolation or its changing
    # variance. Use raw observation variance here: smoothing does not manufacture
    # additional independent measurements. Subtract velocity-estimation noise
    # before interpreting the Schur complement as temporal information.
    def information(p):
        y, velocity, _, dvy = observations(p[6])
        raw_vx = ma.variance[np.searchsorted(ts[a], tt), ji]
        raw_vy = mb.variance[np.clip(np.searchsorted(ts[b], tt+p[6]),
                                    0, len(ts[b])-1), ji]
        def white(q, xx=x, yy=y):
            return _epipolar_residual(q, xx, yy, raw_vx, raw_vy)
        cols = []
        for k in range(6):
            d = np.zeros(7); d[k] = 1e-6
            cols.append((white(p+d)-white(p-d))/(2e-6))
        gx, gy = [], []
        for k in range(2):
            d = np.zeros_like(x); d[:, k] = 1e-6
            gx.append((white(p, x+d, y)-white(p, x-d, y))/(2e-6))
            gy.append((white(p, x, y+d)-white(p, x, y-d))/(2e-6))
        gx, gy = np.array(gx).T, np.array(gy).T
        weights = 1/np.sqrt(1+(residual(p)/2.)**2)
        J = np.column_stack(cols)*weights[:, None]
        jt = np.sum(gy*velocity, axis=1)*weights
        projected = jt-J@np.linalg.lstsq(J, jt, rcond=1e-7)[0]
        energy = float(projected@projected)
        noise = float(np.sum(weights**2 * (np.sum(gy**2, axis=1)*dvy
                                           + np.sum(gx**2, axis=1)*dvx)))
        # Conservative effective degrees of freedom for overlapping smooth fits.
        effective = max(len(x)/max(ma.window, mb.window), 1.)
        lower_information = energy-noise*(1+3*np.sqrt(2/effective))
        if not np.isfinite(lower_information) or lower_information <= 0:
            raise RuntimeError('temporal offset unobservable: motion derivatives are explained by localization noise')
        std = float(1/np.sqrt(lower_information))
        if std > bound/4:
            raise RuntimeError('temporal offset unobservable separately from camera pose within the search interval')
        return std, float(np.sqrt(max(energy-noise, 0.)/max(jt@jt, 1e-20))), noise, energy

    information(best)
    # Probe at least every half frame, including search windows below one frame.
    faster_period = min(period, np.median(np.diff(ts[a])))
    lags = np.linspace(-bound, bound, max(9, int(np.ceil(4*bound/faster_period))+1))
    profile = []
    warm = best[:6]
    starts = [p[:6] for p in fits]
    for lag in lags:
        p0 = min([warm, *starts], key=lambda p: np.mean(cost(residual(p, lag))))
        warm = optimize(p0, fixed=lag)
        profile.append(np.r_[warm, lag])
    # Reverse continuation from the best reoptimized model prevents a poor local
    # pose branch from inflating the contrast at one end of the profile.
    warm = min(profile, key=lambda p: np.mean(cost(residual(p))))[:6]
    for i in range(len(lags)-1, -1, -1):
        p = optimize(warm, fixed=lags[i])
        if np.mean(cost(residual(p, lags[i]))) < np.mean(cost(residual(profile[i]))):
            profile[i] = np.r_[p, lags[i]]
        warm = profile[i][:6]
    # Refine every local minimum, not just the globally best coarse sample: a
    # periodic alias can lie between profile samples at an arbitrary phase.
    values = np.array([np.mean(cost(residual(p))) for p in profile])
    minima = [i for i in range(len(values)) if
              (i == 0 or values[i] <= values[i-1]) and
              (i == len(values)-1 or values[i] <= values[i+1])]
    minima_fits = [optimize(profile[i], interval=(lags[max(0, i-1)], lags[min(len(lags)-1, i+1)]))
                   for i in minima]
    best = min([best, *minima_fits], key=lambda p: np.mean(cost(residual(p))))
    std, information_ratio, noise_energy, energy = information(best)
    best_cost = cost(residual(best))
    # Paired block errors account for reuse of source frames and smoothing
    # correlation. A per-point chi-square test would be overconfident here.
    block_time = max(ma.window*np.median(np.diff(ts[a])), mb.window*period)
    block = np.floor((tt-tt.min())/block_time).astype(int)
    _, group = np.unique(block, return_inverse=True)
    counts = np.bincount(group)

    def equivalent(p):
        difference = cost(residual(p))-best_cost
        means = np.bincount(group, weights=difference)/counts
        se = float(np.std(means, ddof=1)/np.sqrt(len(means))) if len(means)>1 else np.inf
        return float(np.mean(difference)) <= 3*se + 1e-10

    precision = min(.012, bound/2)
    alternatives = [*profile, *minima_fits]
    for lag in np.clip(best[6] + np.array([-precision, precision]), -bound, bound):
        p0 = min([best[:6], *starts], key=lambda p: np.mean(cost(residual(p, lag))))
        alternatives.append(np.r_[optimize(p0, fixed=lag), lag])
    if any(abs(p[6]-best[6]) >= precision-1e-10 and equivalent(p) for p in alternatives):
        raise RuntimeError('temporal profile is statistically flat or has multiple equivalent minima')
    # A repeat is relevant only if at least one full-cycle alternative to the
    # selected lag lies in the permitted interval. Fit only admissible periods.
    max_period = bound + abs(best[6])
    for repeat, radius in _raw_repeat_candidates(xy, co, ts, a, b, joints, start, end, bound):
        if _has_repeating_motion(xy, co, ts, (a, b), joints, start, end, repeat, radius,
                                 max_period=max_period):
            raise RuntimeError('temporal offset is ambiguous: raw tracks repeat at an in-bound lag')
    for p in minima_fits:
        separation = abs(p[6]-best[6])
        if separation >= precision and _has_repeating_motion(
                xy, co, ts, (a, b), joints, start, end, separation, faster_period/2,
                max_period=max_period):
            raise RuntimeError('temporal offset is ambiguous: raw tracks repeat within the search interval')
    offset = float(best[6])
    y = observations(offset)[0]
    score = float(np.median(np.abs(_signed_sampson(
        Rotation.from_rotvec(best[:3]).as_matrix(), best[3:6], x, y))))
    motion = float(np.median(np.linalg.norm(observations(bound)[0]-observations(-bound)[0], axis=1)))
    q25, q75 = np.percentile(values, [25, 75])
    return offset, dict(cameras=[a, b], offset_seconds=offset, median_epipolar_error=score,
                        motion=motion, n_points=len(x), at_bound=abs(offset) > .98*bound,
                        time_information_ratio=information_ratio, local_offset_std_seconds=std,
                        temporal_profile_contrast=float((q75-q25)/max(q25, 1e-12)),
                        profile_offsets_seconds=lags.tolist(),
                        profile_median_errors=[float(np.median(np.abs(residual(p)))) for p in profile],
                        profile_normalized_costs=values.tolist(), profile_resolution_seconds=precision,
                        derivative_noise_energy=noise_energy, derivative_projected_energy=energy,
                        localization_noise_normalized=[ma.noise, mb.noise])


def estimate_time_offsets(keypoints, confidence, K, *, fps=None, timestamps=None,
                          max_offset_seconds=0.1, seed=0, min_confidence=0.3,
                          frame_counts=None):
    """Estimate constant shifts by robust pairwise pose+time fitting and graph alignment.

    Camera 0 has offset zero. Search bound applies to each camera relative to 0;
    non-reference pairs are allowed twice that bound. Requires non-periodic,
    spatially varied motion and a connected graph of overlapping confident tracks.
    Zero and below-threshold confidence are missing. Optional frame_counts
    specifies the real prefixes of padded streams. Unobservable, inconsistent
    or boundary-limited estimates raise ``ValueError``. The returned frames use
    the same gap-aware cubic resampling as the known-offset path.
    """
    from .api import to_centered
    kp = validate_real_array(keypoints, 'keypoints')
    co = validate_real_array(confidence, 'confidence')
    min_confidence = validate_real_array(min_confidence, 'min_confidence')
    if kp.ndim != 4 or kp.shape[-1] != 2 or co.shape != kp.shape[:3]:
        raise ValueError('expected keypoints (C,F,J,2) and confidence (C,F,J)')
    C, F = kp.shape[:2]
    if C < 2 or F < 20:
        raise ValueError('time estimation requires >=2 cameras and >=20 frames')
    max_offset_seconds = validate_scalar(max_offset_seconds, 'max_offset_seconds')
    counts = validate_frame_counts(frame_counts, C, F)
    if np.any(counts < 20):
        raise ValueError('time estimation requires >=20 real frames per camera')
    axes = time_axes(C, F, fps=fps, timestamps=timestamps)
    centered, focal = to_centered(kp, K)
    co = np.where(np.isfinite(co), np.clip(co, 0, 1), 0.)
    valid = np.isfinite(centered).all(-1) & (co > 0) & (co >= min_confidence)
    co = np.where(valid, co, 0.)
    kp = np.where(valid[..., None], kp, np.nan)
    xy = [np.where(valid[c, :n, :, None], centered[c, :n] / focal[c], np.nan)
          for c, n in enumerate(counts)]
    scores = [co[c, :n] for c, n in enumerate(counts)]
    # Keep input clocks for the output grid; fit in a common relative origin.
    ts = [axes[c, :n] - axes[0, 0] for c, n in enumerate(counts)]
    models = [_MotionTracks(points, confidence, times, min_confidence)
              for points, confidence, times in zip(xy, scores, ts)]
    rng = np.random.default_rng(seed)
    pairs, failures = [], []
    for a, b in combinations(range(C), 2):
        bound = max_offset_seconds if a == 0 else 2*max_offset_seconds
        try:
            _, diagnostic = _pair_offset(xy, scores, ts, a, b, bound, rng, models)
            if (not np.isfinite(diagnostic['motion']) or diagnostic['motion'] < max(
                    1e-5, 4*diagnostic['median_epipolar_error'])):
                raise RuntimeError('insufficient motion to identify a temporal offset')
            pairs.append(diagnostic)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            failures.append(dict(cameras=[a,b], error=str(e)))
    reached = {0}
    for _ in range(C):
        for d in pairs:
            a,b = d['cameras']
            if a in reached or b in reached:
                reached.update([a,b])
    if len(reached) < C:
        raise ValueError('temporal overlap graph is disconnected or motion is insufficient: ' + str(failures))
    A = np.zeros((len(pairs), C-1))
    y = np.array([d['offset_seconds'] for d in pairs])
    for i,d in enumerate(pairs):
        a,b = d['cameras']
        if a: A[i,a-1] = -1
        if b: A[i,b-1] = 1
    step = np.median(np.diff(ts[0]))
    p0 = np.clip(np.linalg.lstsq(A,y,rcond=None)[0], -.999*max_offset_seconds, .999*max_offset_seconds)
    fit = least_squares(lambda x: A@x-y, p0, bounds=(-max_offset_seconds,max_offset_seconds),
                        loss='soft_l1', f_scale=0.15*step)
    offsets = np.r_[0., fit.x]
    if any(d['at_bound'] for d in pairs) or np.max(np.abs(offsets)) > .98*max_offset_seconds:
        raise ValueError('temporal offset is at the search boundary')
    consistency = float(np.max(np.abs(fit.fun)))
    if consistency > min(.012, .5*step):
        raise ValueError('temporal offsets are inconsistent between camera pairs')
    aligned, scores, times = resample_keypoints(kp, co, offsets, timestamps=axes,
                                               min_confidence=min_confidence, frame_counts=counts)
    return Synchronization(offsets, aligned, scores, times,
                           dict(pairs=pairs, failed_pairs=failures,
                                pair_consistency_seconds=consistency,
                                max_offset_seconds=max_offset_seconds))
