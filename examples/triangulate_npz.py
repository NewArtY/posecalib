"""Use a saved static rig with a NEW packed detection recording.

    python examples/triangulate_npz.py rig.json new.npz --out points.npz

Same pixel/confidence and temporal NPZ contract as calibrate_npz.py. Output
contains points (F,J,3), inliers (C,F,J), and timestamps (F,) in seconds when
alignment ran. The calibration recording's saved reference_times is never used
as this new recording's grid. Finite points, not inlier count alone, identify
accepted 3D reconstructions.
"""
import argparse
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posecalib import Calibration, triangulate
from posecalib.validation import validate_real_array, validate_scalar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rig', type=Path)
    parser.add_argument('detections', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fps', type=float, help='scalar CFR rate, if timestamps are absent')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--already-aligned', action='store_true', help='bypass ALL temporal resampling/corrections')
    flags.add_argument('--ignore-time-offsets', action='store_true', help='ignore saved residual shifts but align known clocks')
    args = parser.parse_args()
    for path in (args.rig, args.detections):
        if (args.out.resolve() == path.resolve()
                or (args.out.exists() and path.exists() and args.out.samefile(path))):
            parser.error('--out must differ from both inputs')
    try:
        rig = Calibration.load_json(args.rig)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    try:
        loaded = np.load(args.detections, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError('detections must be an NPZ archive')
        with loaded:
            missing = {'keypoints', 'confidence'} - set(loaded.files)
            if missing:
                raise ValueError('NPZ is missing required fields: ' + ', '.join(sorted(missing)))
            data = {name: loaded[name] for name in loaded.files}
        keypoints = validate_real_array(data['keypoints'], 'keypoints')
        confidence = validate_real_array(data['confidence'], 'confidence')
        timestamps = data.get('timestamps')
        fps = args.fps if args.fps is not None else data.get('camera_fps', data.get('fps'))
        if not args.already_aligned:
            if timestamps is not None:
                timestamps = validate_real_array(timestamps, 'timestamps')
            elif fps is not None and (args.fps is not None or 'camera_fps' not in data):
                # Only camera_fps opts into per-camera CFR rates.
                fps = validate_scalar(fps, 'fps')
    except (ValueError, OSError, EOFError, BadZipFile) as exc:
        parser.error(f'cannot load detections: {exc}')
    try:
        points, inliers, times = triangulate(
            rig, keypoints, confidence, timestamps=timestamps, frame_counts=data.get('frame_counts'),
            fps=fps,
            apply_time_offsets=not args.already_aligned, ignore_time_offsets=args.ignore_time_offsets,
            return_times=True)
    except ValueError as exc:
        parser.error(str(exc))
    result = dict(points=points, inliers=inliers)
    if times is not None:
        result['timestamps'] = times
    try:
        with args.out.open('wb') as stream:
            np.savez_compressed(stream, **result)
    except OSError as exc:
        parser.error(str(exc))
    print(f'Saved {args.out}: {points.shape}; {np.isfinite(points).all(-1).sum()} finite points')


if __name__ == '__main__':
    main()
