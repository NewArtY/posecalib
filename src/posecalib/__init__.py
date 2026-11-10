"""posecalib: extrinsic calibration of a multi-camera rig from human body keypoints."""
from .api import (Calibration, Calibrator, align_to_floor, calibrate, project, reprojection_errors,
                  to_centered, triangulate)
from .skeleton import BLAZEPOSE33, COCO17, Skeleton
from .synchronization import Synchronization, estimate_time_offsets, resample_keypoints
from .mediapipe_io import prepare_blazepose

__version__ = '0.2.0'

__all__ = ['BLAZEPOSE33', 'COCO17', 'Calibration', 'Calibrator', 'Skeleton', 'align_to_floor', 'calibrate', 'project',
           'reprojection_errors', 'to_centered', 'triangulate', 'Synchronization', 'estimate_time_offsets',
           'resample_keypoints', 'prepare_blazepose', '__version__']
