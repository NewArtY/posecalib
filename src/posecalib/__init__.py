"""posecalib: extrinsic calibration of a multi-camera rig from human body keypoints."""
from .api import (Calibration, Calibrator, align_to_floor, calibrate, project, reprojection_errors,
                  to_centered, triangulate)
from .skeleton import COCO17, Skeleton

__version__ = '0.1.0'

__all__ = ['COCO17', 'Calibration', 'Calibrator', 'Skeleton', 'align_to_floor', 'calibrate', 'project',
           'reprojection_errors', 'to_centered', 'triangulate', '__version__']
