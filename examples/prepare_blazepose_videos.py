"""Convert single-person CFR videos to full BlazePose33 pixel detections.

From the checkout, install the video extra: python -m pip install ".[video]"
Supply a local official PoseLandmarker .task model; nothing is downloaded.
API: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

Camera order follows --videos and K (C,3,3) in --intrinsics. Optional
dist_coeffs (C,D), D in 4,5,8,12,14, are OpenCV pinhole distortion coefficients.
Frames are undistorted with the original K as output K, without cropping.
Without coefficients, --already-undistorted is mandatory. That flag asserts
the videos already match K and skips correction even if coefficients exist.

Only constant-frame-rate input is supported: seconds = local frame index / FPS
reported by OpenCV, starting at zero independently for every camera. Container
PTS and start offsets are not read; convert VFR recordings to CFR first.
Unequal FPS/lengths are retained, never trimmed or synchronized here. Pad to
the largest frame count with NaN coordinates and zero confidence; timestamps
(C,F) extrapolate each camera's own CFR clock through padding. frame_counts
records real lengths; camera_fps records rates; scalar fps exists only if equal.
Misses remain invalid. Two or more detected people abort instead of selecting
an identity; keep the same single person in every camera throughout recording.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posecalib.mediapipe_io import prepare_blazepose
from posecalib.validation import validate_intrinsics, validate_real_array, validate_scalar


def load_intrinsics(path, count, already_undistorted=False):
    with np.load(path, allow_pickle=False) as data:
        K = data['K']
        dist = (validate_real_array(data['dist_coeffs'], 'dist_coeffs')
                if 'dist_coeffs' in data else None)
    K = validate_intrinsics(K, count)
    if dist is not None and (dist.ndim != 2 or dist.shape[0] != count
                            or dist.shape[1] not in (4, 5, 8, 12, 14)
                            or not np.isfinite(dist).all()):
        raise ValueError('dist_coeffs must be finite (C,D), D=4,5,8,12,14 (OpenCV pinhole)')
    if dist is None and not already_undistorted:
        raise ValueError('missing dist_coeffs: supply coefficients or explicitly use --already-undistorted')
    return K, None if already_undistorted else dist


def extract_video(path, K, dist, model, *, cv2, mp):
    """Injected modules allow tests without OpenCV, MediaPipe, or a model."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'{path}: cannot open video')
        fps = validate_scalar(cap.get(cv2.CAP_PROP_FPS), f'{path}: FPS')
        if fps > 1000:
            raise ValueError(f'{path}: FPS must be finite in (0,1000] for millisecond VIDEO timestamps')
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_poses=2)
        rows, size = [], None
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # OpenCV does not distinguish EOF from decoder failure.
                current_size = (frame.shape[1], frame.shape[0])
                if size is not None and current_size != size:
                    raise ValueError(f'{path}: frame dimensions changed')
                size = current_size
                if dist is not None:
                    frame = cv2.undistort(frame, K, dist, None, K)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(round(len(rows) * 1000 / fps)))
                poses = result.pose_landmarks
                if len(poses) > 1:
                    raise ValueError(f'{path}: frame {len(rows)}: multiple people detected; exactly one required')
                row = np.full((33, 5), np.nan)
                if poses:
                    if len(poses[0]) != 33:
                        raise ValueError(f'{path}: expected 33 image landmarks')
                    row[:] = [[p.x, p.y, p.z,
                               0. if p.visibility is None else p.visibility,
                               1. if p.presence is None else p.presence] for p in poses[0]]
                rows.append(row)
        if not rows:
            raise ValueError(f'{path}: no decoded frames')
        return np.stack(rows), size, fps
    finally:
        cap.release()


def pack_detections(rows, sizes, K, rates):
    counts = np.array([len(row) for row in rows], dtype=int)
    rates = validate_real_array(rates, 'FPS')
    if (not len(counts) or (counts <= 0).any() or rates.shape != counts.shape
            or not np.isfinite(rates).all() or (rates <= 0).any()):
        raise ValueError('nonempty camera tracks and positive finite FPS required')
    landmarks = np.full((len(rows), int(counts.max()), 33, 5), np.nan)
    for camera, row in enumerate(rows):
        landmarks[camera, :len(row)] = validate_real_array(row, 'landmarks')
    # Timestamp metadata is attached separately: no synchronization is implied.
    arrays = prepare_blazepose(landmarks, sizes, K)
    arrays.update(timestamps=np.arange(landmarks.shape[1])[None, :] / rates[:, None],
                  camera_fps=rates, frame_counts=counts)
    if np.all(rates == rates[0]):
        arrays['fps'] = np.array(rates[0])
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', type=Path, nargs='+', required=True)
    parser.add_argument('--intrinsics', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--already-undistorted', action='store_true',
                        help='assert input images already match K; skip undistortion')
    args = parser.parse_args(argv)
    if args.out.resolve() in {p.resolve() for p in [*args.videos, args.intrinsics, args.model]}:
        parser.error('--out must differ from all inputs')
    try:
        K, dist = load_intrinsics(args.intrinsics, len(args.videos), args.already_undistorted)
        for path in [*args.videos, args.model]:
            if not path.is_file():
                raise ValueError(f'local input file does not exist: {path}')
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise ValueError('from the checkout, install the video extra: python -m pip install ".[video]"') from exc
        tracks = [extract_video(path, K[i], None if dist is None else dist[i],
                                args.model, cv2=cv2, mp=mp)
                  for i, path in enumerate(args.videos)]
        rows, sizes, rates = zip(*tracks)
        arrays = pack_detections(rows, sizes, K, rates)
        with args.out.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.out}: {arrays['keypoints'].shape}; decoded frames {arrays['frame_counts'].tolist()}")


if __name__ == '__main__':
    main()
