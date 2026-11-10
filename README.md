# posecalib

Python library and command-line examples to estimate the extrinsics of a fixed multi-camera rig from 2D keypoints of one moving person.
Use overlapping views, varied motion and pre-calibrated intrinsics. Supply undistorted pixel
coordinates and the intrinsics for those exact images. The built-in layouts are COCO-17 and
all 33 MediaPipe BlazePose landmarks.

## Install

Run the examples from a checkout in a fresh virtual environment. The core supports Python 3.9+
and uses NumPy and SciPy.

```sh
python -m pip install .
python examples/synthetic_quickstart.py
```

For the direct video workflow, install with `python -m pip install ".[video]"` instead. Use a
Python and platform combination with MediaPipe wheels. The extra uses `opencv-contrib-python`
as the sole provider of `cv2`. Keep other OpenCV distributions out of that environment.
Existing keypoint exports only need the core installation. With uv, use `uv sync --locked`
or `uv sync --locked --extra video` and run the commands below through `uv run --no-sync`.

## Prepare detections

Keep all 33 landmarks in BlazePose order. For each camera, export an NPZ with `landmarks`
of shape `(F,33,4)` containing normalized `x, y, z, visibility`, or `(F,33,5)` with
`presence` last. Include `image_size=[width, height]` and the undistorted image's `K`.
The adapter uses normalized x/y to compute pixel coordinates. It derives confidence from
visibility, or the minimum of visibility and presence when both are present. Leave missing
landmarks as NaN and retain missed frames. Include a `timestamps` vector in seconds in
every file if supplying timestamps or merging unequal lengths. The merger preserves the
clocks and pads unequal lengths with NaN coordinates and zero confidence.

```sh
python examples/prepare_blazepose_npz.py cam0.npz cam1.npz cam2.npz --out detections.npz
```

For video, supply a local MediaPipe Pose Landmarker `.task` model and a `cameras.npz` with
`K` of shape `(C,3,3)` and `dist_coeffs` of shape `(C,D)` using OpenCV pinhole
coefficients. For already-undistorted input, pass `--already-undistorted` with the
corresponding `K`. Process fisheye footage separately before using this workflow.

```sh
python examples/prepare_blazepose_videos.py --videos cam0.mp4 cam1.mp4 cam2.mp4 --intrinsics cameras.npz --model pose_landmarker.task --out detections.npz
```

The video helper records frame index divided by reported FPS, starting at zero independently
for each camera. Use constant-frame-rate (CFR) input. These are local clocks. For known-clock
alignment, supply independently established common-clock times. Preserve capture timing
separately for variable-frame-rate sources, since the video helper does not read container
presentation timestamps.

### Calibration NPZ

| Key | Shape | Meaning |
| --- | --- | --- |
| `keypoints` | `(C,F,J,2)` | Undistorted x/y pixels. Use NaN for misses. |
| `confidence` | `(C,F,J)` | Detector scores in [0,1]. Use zero with NaN for misses. |
| `K` | `(C,3,3)` or `(3,3)` | Per-camera or shared intrinsics. Zero skew. Required for calibration. |
| `timestamps` | `(C,F)` or shared `(F,)` | Optional finite, strictly increasing times in seconds. Takes precedence over FPS. |
| `fps` / `camera_fps` | Scalar / `(C,)` | Optional CFR fallback, with each clock starting at zero. CLI `--fps` supplies a scalar override for these fallback fields. |
| `frame_counts` | `(C,)` | Optional integer real prefix lengths, 1 through F. Include for padding. |

C counts cameras, F is the stored frame count and J is the landmark count. Timestamp arrays
must remain finite and strictly increasing through any padding. `frame_counts` excludes
padding from temporal overlap even if padded values look valid. A detection miss inside a
real recording still occupies its original frame slot. The Python API accepts scalar or
per-camera `fps` and a custom `Skeleton` when needed.

## Calibrate the rig

For streams with known common-clock timestamps, let the default mode align them. For arrays
whose indices already denote corresponding instants, omit both timestamps and FPS.

```sh
python examples/calibrate_npz.py detections.npz --distance 0 1 5.32 --out rig.json
```

When the streams have an unknown constant shift, opt in to estimation. Supply timestamps or
FPS and set a plausible search bound in seconds relative to camera 0.

```sh
python examples/calibrate_npz.py detections.npz --estimate-sync --max-offset 0.15 --frame-step 5 --distance 0 1 5.32 --out rig.json
```

Camera 0 has offset zero. At reference time t, a positive offset of 0.033 s samples that
camera at source time t + 0.033 s. Both modes crop to the common real interval on camera 0's
grid and use gap-aware cubic resampling where needed. Interpolation stays inside valid runs,
reduces degree for short runs, and preserves exact valid samples. `--frame-step` subsamples
after alignment. Keep the input FPS at its original value.

The temporal path defaults to `min_confidence=0.3`. Encode misses as NaN coordinates with
zero confidence. Even at `min_confidence=0`, zero-confidence detections are excluded from
calibration, triangulation and synchronization, and break both cubic and linear interpolation.
Detector scores are screening and weighting inputs.

For unknown offsets, record turns, direction changes and varied limb motion with confident
tracks overlapping across the rig. Stillness, uniform translation, repeated motion and poor
overlap can leave the delay ambiguous. Ambiguous pairs are rejected. The estimator raises
`ValueError` when usable pairs cannot connect the rig, when the solution reaches the search
boundary, or when pairwise shifts disagree. The CLI exits with an error. Use a better capture
or independently measured timing if the search cannot resolve an offset. On success,
inspect the pair diagnostics in `rig.info["synchronization"]` in Python.

Replace `--distance 0 1 5.32` with a measured camera-centre baseline in metres. Python also
accepts measured segment lengths. The default scale uses approximate body proportions.
The CLI fits the floor from reconstructed ankles by default. This needs adequate stance
observations, roughly upright cameras, a horizontal baseline component and a meaningful
metric scale. Use `--no-floor` to keep camera 0's coordinate frame. For custom skeletons,
supply ankle indices to `align_to_floor` if the layout has no named ankles.

## Triangulate another recording

Keep the camera order, intrinsics and rig geometry fixed. Supply the new recording's time
metadata and real frame counts using the same NPZ schema. The default applies any saved
offsets, which are valid only while the relative timing is unchanged.

```sh
python examples/triangulate_npz.py rig.json new_detections.npz --out points.npz
```

- Already aligned arrays use `--already-aligned` or Python `apply_time_offsets=False`.
  This bypasses all timing handling, including timestamp resampling and frame-count cropping.
  Trim any padding yourself before using this mode.
- Known clocks with obsolete saved offsets use `--ignore-time-offsets` or Python
  `ignore_time_offsets=True`. Keep the default `apply_time_offsets=True` so the new clocks
  are still aligned with zero residual shift.
- If a restart or configuration change leaves the new relative timing unknown, estimate
  offsets for that capture with `estimate_time_offsets`. Pass its aligned keypoints and
  confidences to `triangulate(..., apply_time_offsets=False)` and keep its returned timestamps.

The CLI writes `points` of shape `(F,J,3)`, `inliers` of shape `(C,F,J)`, and `timestamps`
when temporal resampling runs. Here F is the output frame count. In Python, use
`triangulate(..., return_times=True)` to also return the new cropped times, or `None` when no
resampling ran. `rig.info["reference_times"]` describes the original calibration grid after
cropping and frame subsampling. The new capture gets its own grid.

The inlier mask describes observations. It can contain true entries for a point that fails
geometric acceptance. At the lower level, use the `ok` flag from `triangulate_robust`. Public
`triangulate` emits NaN for rejected points. Use `np.isfinite(points).all(axis=-1)` to select
accepted output points.

## Python API

Import the public API from `posecalib`. The arrays use the same camera-first layout as
the [calibration NPZ](#calibration-npz). This example assumes that frame indices are
already aligned and that both captures use the same camera order and image geometry.

```python
import numpy as np
from posecalib import Calibration, calibrate, triangulate

with np.load("detections.npz") as data:
    rig = calibrate(
        data["keypoints"], data["confidence"], data["K"],
        camera_distance=(0, 1, 5.32),
    )
rig.save_json("rig.json")

rig = Calibration.load_json("rig.json")
with np.load("new_detections.npz") as data:
    points, inliers = triangulate(
        rig, data["keypoints"], data["confidence"], apply_time_offsets=False,
    )
accepted = np.isfinite(points).all(axis=-1)
```

For known capture clocks, pass `timestamps=...` and, for padded input, `frame_counts=...`
to each call. For unknown constant offsets, pass `estimate_sync=True` and
`max_offset_seconds=0.15` to `calibrate`, with `timestamps` or `fps`. In either
case leave triangulation's timing handling enabled and use `return_times=True`
to get the output times as a third return value.

| Public API | Contract |
| --- | --- |
| `calibrate(keypoints, confidence, K, **options)` | Returns a `Calibration` in camera 0's frame. Defaults to robust Cauchy loss, confidence weighting and segment constraints. Pass `skeleton=` for a custom layout, or `refine_focal=True` to refine focal lengths. |
| `Calibrator(keypoints, confidence, K, ...)` | Geometry-only workflow for already aligned arrays. Call `initialize()`, then `refine(...)` to control the two stages separately. |
| `triangulate(rig, keypoints, confidence, **options)` | Returns world points `(F,J,3)` and an observation mask `(C,F,J)`. Rejected points are NaN. F counts output frames. |
| `project(rig, points)` | Maps world points `(...,3)` to undistorted image pixels `(C,...,2)`. |
| `reprojection_errors(rig, keypoints, points)` | Returns pixel distances `(C,F,J)` for corresponding observations and 3D points. Pass observations on the same time grid as the points. |
| `align_to_floor(rig, ...)` | Returns a transformed calibration. Set `foot_keypoints=(left, right)` for a custom skeleton and `foot_height` if the default 0.08 m is unsuitable. |
| `estimate_time_offsets(keypoints, confidence, K, ...)` | Returns a `Synchronization` with estimated offsets, aligned arrays, output times and diagnostics. Useful for a new capture from a rig whose geometry is already known. |
| `resample_keypoints(keypoints, confidence, offsets_seconds, ...)` | Applies supplied offsets. Returns aligned keypoints, confidence and timestamps. |
| `prepare_blazepose(landmarks, image_size, K, ...)` | Converts normalized BlazePose landmarks (C,F,33,4/5) to a dict of pixel keypoints, confidence, intrinsics and image sizes, plus timestamps if supplied. |
| `to_centered(keypoints, K)` | Converts pixels to the centered, equal-focal-scale coordinates used internally. The main APIs take undistorted image pixels directly. |

`Calibration.R` has shape `(C,3,3)` and `Calibration.t` has shape `(C,3)`.
The convention is world to camera, `X_camera = R @ X_world + t`. Use
`rig.centers` for camera positions in world coordinates and
`rig.projection_matrices()` for the `(C,3,4)` matrices. The result also exposes
`K`, calibration `points`, `segment_lengths`, `scale_source`,
`reprojection_rmse`, `inlier_fraction`, `time_offsets_seconds` and `info`.
Reprojection distances use undistorted image pixels and the returned intrinsics,
including any refined focal lengths. `rig.info["residuals_px"]` contains one unweighted
distance per bundle observation, in the same order as `rig.info["weights"]`.
`rig.reprojection_rmse` (saved as `reprojection_rmse_px`) uses observations with
robust weight greater than 0.5 in each camera. The optimizer's `sigma_hat` uses
centered, equal-focal coordinates and includes confidence weighting when enabled.

Inspect `scale_source` before interpreting coordinates as metres. JSON stores the
rig and diagnostics for reuse. Export reconstructed points separately if needed.

## Validation and limits

Tests use synthetic geometry and timing, and the video tests mock frames and detections.
Real model inference, timing accuracy and geometry on a physical rig still need validation.
Offset estimation assumes a constant shift and does not model drift, jitter or rolling
shutter. View-dependent detector bias can distort geometry even with small reprojection
residuals. Check metric scale and geometry against an independent reference.

Run the tests from the checkout with `uv sync --locked --extra dev` followed by
`uv run --no-sync python -m pytest -q`. See `benchmarks/` for synthetic experiment scripts.

## License

MIT
