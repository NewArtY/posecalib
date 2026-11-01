"""Calibrate a real rig from detections stored in a ``.npz`` file.

Input file (``numpy.savez``):

* ``keypoints``  - ``(C, F, 17, 2)`` COCO-17 pixel coordinates from undistorted images, NaN if missing;
* ``confidence`` - ``(C, F, 17)`` detector scores in ``[0, 1]``;
* ``K``          - ``(C, 3, 3)`` or ``(3, 3)`` intrinsic matrices.

Frames must be synchronised across cameras and show exactly one person.

    python examples/calibrate_npz.py detections.npz --distance 0 1 5.32 --out rig.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))   # run without installation

from posecalib import align_to_floor, calibrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Extrinsic calibration from body keypoints')
    ap.add_argument('detections', type=Path)
    ap.add_argument('--distance', nargs=3, metavar=('CAM_I', 'CAM_J', 'METRES'),
                    help='measured distance between two camera centres (sets the metric scale)')
    ap.add_argument('--refine-focal', action='store_true', help='also refine the focal lengths')
    ap.add_argument('--no-floor', action='store_true', help='keep the frame of camera 0')
    ap.add_argument('--out', type=Path, default=Path('rig.json'))
    a = ap.parse_args()

    d = np.load(a.detections)
    dist = None if a.distance is None else (int(a.distance[0]), int(a.distance[1]), float(a.distance[2]))
    calib = calibrate(d['keypoints'], d['confidence'], d['K'], camera_distance=dist, refine_focal=a.refine_focal)
    if not a.no_floor:
        calib = align_to_floor(calib)
    print(f'scale: {calib.scale_source}')
    for k, (rmse, frac) in enumerate(zip(calib.reprojection_rmse, calib.inlier_fraction)):
        print(f'camera {k}: reprojection RMSE {rmse:.2f} px, inliers {100 * frac:.0f}%, '
              f'centre {np.round(calib.centers[k], 3)}')
    calib.save_json(a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
