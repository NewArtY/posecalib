"""Keypoint layouts.

A :class:`Skeleton` tells the calibrator how many keypoints the detector returns,
which of them are left/right mirror pairs, and which pairs of keypoints form
rigid body segments (used by the optional constant-segment-length term).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Skeleton:
    """Keypoint layout of a 2D pose detector.

    Attributes:
        names: keypoint names, in detector output order.
        segments: ``(B, 2)`` index pairs of keypoints joined by a rigid segment.
        segment_names: one name per segment.
        mirror_pairs: ``(left, right)`` index pairs (informational).
        typical_lengths: approximate segment lengths of an adult, metres. Used only
            to bring the reconstruction to a roughly metric scale before refinement
            when no measurement is supplied.
    """

    names: tuple
    segments: np.ndarray
    segment_names: tuple
    mirror_pairs: tuple = field(default=())
    typical_lengths: np.ndarray | None = None

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def index(self, name: str) -> int:
        return self.names.index(name)


_H = 1.75  # reference stature for the typical segment lengths (Drillis-Contini proportions)

COCO17 = Skeleton(
    names=('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
           'left_knee', 'right_knee', 'left_ankle', 'right_ankle'),
    segments=np.array([[5, 7], [7, 9], [6, 8], [8, 10],
                       [11, 13], [13, 15], [12, 14], [14, 16], [11, 12]]),
    segment_names=('left_upper_arm', 'left_forearm', 'right_upper_arm', 'right_forearm',
                   'left_thigh', 'left_shank', 'right_thigh', 'right_shank', 'pelvis_width'),
    mirror_pairs=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
    typical_lengths=np.array([0.186 * _H, 0.146 * _H, 0.186 * _H, 0.146 * _H,
                              0.245 * _H, 0.246 * _H, 0.245 * _H, 0.246 * _H, 0.18]),
)
"""COCO-17 layout (OpenPose/MMPose/RTMPose/YOLO-pose body keypoints)."""
