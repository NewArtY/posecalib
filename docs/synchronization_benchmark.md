# Constant-offset synthetic benchmark

30 FPS, 20 s, 4 cameras; one seed per delay; 2 px noise / confidence, 2% outliers, missing points.
Offsets: [0, 1, -0.7, 0.4] times the stated maximum delay. Search: +/-150 ms relative to camera 0.
Each row reports the result for seed 0. Delay error is the largest error across cameras.

| Max delay (frames) | Success | Offset error (ms) | Rotation raw / corrected (deg) | Centres raw / corrected (mm) |
|---|---|---|---|---|
| 0 | 1/1 | 0.45 | 0.024 / 0.025 | 2.1 / 1.6 |
| 0.5 | 1/1 | 0.44 | 0.256 / 0.026 | 15.1 / 2.2 |
| 1 | 1/1 | 0.39 | 0.560 / 0.018 | 38.1 / 1.5 |
| 2 | 1/1 | 0.25 | 1.246 / 0.022 | 91.8 / 1.6 |
| 3 | 1/1 | 0.20 | 2.024 / 0.028 | 153.3 / 2.0 |
