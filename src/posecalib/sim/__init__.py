"""Synthetic multi-camera scenes for testing and benchmarking the calibrator.

Example:
    >>> from posecalib.sim import DEFAULT_CFG, make_seed_data, calibration_observations
    >>> data = make_seed_data(0, DEFAULT_CFG)
    >>> obs, true_points = calibration_observations(data, DEFAULT_CFG)
"""
from .detector import as_detections, observe
from .scene import intrinsics, make_cameras, simulate_person
from .trial import (DEFAULT_CFG, DEFAULT_VARIANT, VARIANTS, calibration_observations, make_seed_data,
                    prepare_test, run_board_trial, run_pose_trial)

__all__ = ['DEFAULT_CFG', 'DEFAULT_VARIANT', 'VARIANTS', 'as_detections', 'calibration_observations',
           'intrinsics', 'make_cameras', 'make_seed_data', 'observe', 'prepare_test', 'run_board_trial',
           'run_pose_trial', 'simulate_person']
