"""Prepare full 33-landmark calibration input from camera exports.

    python examples/prepare_blazepose_npz.py cam0.npz cam1.npz --out detections.npz

Each input NPZ (no pickled objects) contains:
  landmarks: (F,33,4) x,y,z,visibility OR (F,33,5) with presence last;
  image_size: [width,height] of the images used for detection;
  K: (3,3) intrinsics for those undistorted images;
  timestamps: optional (F,) local timestamps in seconds, strictly increasing
    per camera; constant clock offsets are allowed and preserved.
Keep missed frames as all-NaN landmarks; never drop them. Unequal lengths need
timestamps in every file and are padded with NaN/zero. frame_counts records
real lengths; pass it with timestamps to calibrate/triangulate. All inputs
must follow the same person. Known clocks align without estimate_sync;
this adapter itself preserves clocks and performs no temporal alignment.
Do not pass pose_world_landmarks.
MediaPipe/OpenCV are unnecessary for this exported-data workflow.

Output: keypoints (C,F,33,2) in pixels, confidence (C,F,33), K (C,3,3),
image_size (C,2), frame_counts (C,), and timestamps (C,F) if provided.
Camera order follows CLI.

After temporal alignment (arrays below must represent corresponding instants),
use explicitly with the current API:
    from posecalib import calibrate, align_to_floor
    from posecalib.skeleton import BLAZEPOSE33, ankle_indices
    d = np.load('detections.npz')
    rig = calibrate(d['keypoints'], d['confidence'], d['K'], skeleton=BLAZEPOSE33,
                    timestamps=d.get('timestamps'), frame_counts=d['frame_counts'])
    rig = align_to_floor(rig, foot_keypoints=ankle_indices(BLAZEPOSE33))
Floor alignment requires sufficient stance observations and a metric scale.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posecalib.mediapipe_io import load_blazepose_cameras  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cameras', type=Path, nargs='+', help='one exported NPZ per camera')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() in [path.resolve() for path in args.cameras]:
        parser.error('--out must differ from input camera files')
    try:
        arrays = load_blazepose_cameras(args.cameras)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    with args.out.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {args.out}: {arrays['keypoints'].shape} pixel keypoints")


if __name__ == '__main__':
    main()
