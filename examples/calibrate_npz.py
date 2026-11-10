"""Calibrate a real rig from detections stored in a ``.npz`` file.

Input file (``numpy.savez``):

* ``keypoints``  - ``(C, F, J, 2)`` COCO-17 or BlazePose-33 pixels, NaN if missing;
* ``confidence`` - ``(C, F, J)`` detector scores in ``[0, 1]``;
* ``K``          - ``(C, 3, 3)`` or ``(3, 3)`` intrinsic matrices.

Frames must show exactly one person. Use --estimate-sync for small constant offsets.
Known timestamps (C,F) in seconds align streams without --estimate-sync.
Optional frame_counts (C,) limits alignment to real frames before padding.
Without timestamps use fps (CLI/NPZ scalar) or NPZ camera_fps (C,) for CFR clocks
starting at zero. Saved reference_times describes the calibration recording.

    python examples/calibrate_npz.py detections.npz --distance 0 1 5.32 --out rig.json
"""
import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))   # run without installation

from posecalib import align_to_floor, calibrate  # noqa: E402
from posecalib.validation import validate_real_array, validate_scalar  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Extrinsic calibration from body keypoints')
    ap.add_argument('detections', type=Path)
    ap.add_argument('--distance', nargs=3, metavar=('CAM_I', 'CAM_J', 'METRES'),
                    help='measured distance between two camera centres (sets the metric scale)')
    ap.add_argument('--refine-focal', action='store_true', help='also refine the focal lengths')
    ap.add_argument('--no-floor', action='store_true', help='keep the frame of camera 0')
    ap.add_argument('--estimate-sync', action='store_true', help='estimate constant camera offsets before calibration')
    ap.add_argument('--fps', type=float, help='input frame rate; alternatively supply NPZ fps or timestamps')
    ap.add_argument('--max-offset', type=float, default=0.1, help='maximum absolute offset from camera 0 in seconds')
    ap.add_argument('--frame-step', type=int, default=1, help='calibrate every Nth frame AFTER temporal alignment')
    ap.add_argument('--out', type=Path, default=Path('rig.json'))
    a = ap.parse_args()

    if (a.out.resolve() == a.detections.resolve()
            or (a.out.exists() and a.detections.exists() and a.out.samefile(a.detections))):
        ap.error('--out must differ from the detections input')
    dist = None
    if a.distance is not None:
        try:
            dist = (int(a.distance[0]), int(a.distance[1]), float(a.distance[2]))
        except ValueError:
            ap.error('--distance expects integer CAM_I CAM_J and numeric METRES')
    # File errors are handled only at the input boundary. Do not hide internal
    # KeyError/TypeError/RuntimeError defects in the calibration implementation.
    try:
        loaded = np.load(a.detections, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError('detections must be an NPZ archive')
        with loaded:
            missing = {'keypoints', 'confidence', 'K'} - set(loaded.files)
            if missing:
                raise ValueError('NPZ is missing required fields: ' + ', '.join(sorted(missing)))
            d = {k: loaded[k] for k in loaded.files}
            # Validate NPZ dtypes at the input boundary, including structured
            # arrays that NumPy cannot cast to float. Internal TypeError defects
            # in calibrate() remain visible; do not widen the solver catch.
            for name in ('keypoints', 'confidence', 'K', 'timestamps'):
                if name in d:
                    d[name] = validate_real_array(d[name], name)
    except (ValueError, OSError, EOFError, BadZipFile) as exc:
        ap.error(f'cannot load detections: {exc}')
    try:
        fps = a.fps if a.fps is not None else d.get('camera_fps', d.get('fps'))
        if fps is not None and 'timestamps' not in d and (a.fps is not None or 'camera_fps' not in d):
            # Legacy NPZ fps remains scalar; camera_fps explicitly opts into
            # per-camera CFR rates. Preserve the phase1 malformed-fps boundary.
            fps = validate_scalar(fps, 'fps')
        validate_scalar(a.max_offset, 'max_offset_seconds')
        calib = calibrate(d['keypoints'], d['confidence'], d['K'], camera_distance=dist,
                          refine_focal=a.refine_focal, estimate_sync=a.estimate_sync, fps=fps,
                          timestamps=d.get('timestamps'), max_offset_seconds=a.max_offset,
                          frame_step=a.frame_step, frame_counts=d.get('frame_counts'))
        if not a.no_floor:
            calib = align_to_floor(calib)
    except ValueError as exc:
        ap.error(str(exc))
    if calib.time_offsets_seconds is not None:
        print('time offsets (ms; source time = reference time + offset):',
              np.round(1000*calib.time_offsets_seconds, 3))
    print(f'scale: {calib.scale_source}')
    for k, (rmse, frac) in enumerate(zip(calib.reprojection_rmse, calib.inlier_fraction)):
        print(f'camera {k}: reprojection RMSE {rmse:.2f} px, inliers {100 * frac:.0f}%, '
              f'centre {np.round(calib.centers[k], 3)}')
    try:
        calib.save_json(a.out)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print('saved', a.out)


if __name__ == '__main__':
    main()
