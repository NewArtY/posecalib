import numpy as np
import pytest

from posecalib import (Calibration, Calibrator, calibrate, estimate_time_offsets, project,
                       resample_keypoints, triangulate)
from posecalib.synchronization import time_axes


def test_offset_sign_fractional_interpolation_crop_and_missing():
    t = np.arange(40)/30.
    off = np.array([0., 1.5/30., -0.7/30.])
    # Linear ground truth: source c samples the world at t - offset[c].
    kp = np.zeros((3,len(t),1,2))
    for c in range(3):
        kp[c,:,0,0] = 100*(t-off[c])
        kp[c,:,0,1] = -30*(t-off[c])+5
    conf = np.ones(kp.shape[:3])
    kp[1,15] = np.nan; conf[1,15] = 0
    out, scores, target = resample_keypoints(kp,conf,off,fps=30)
    for c in range(3):
        mask = scores[c,:,0]>0
        assert np.allclose(out[c,mask,0,0],100*target[mask])
        assert np.allclose(out[c,mask,0,1],-30*target[mask]+5)
    assert np.isnan(out[1,:,0,0]).any()
    assert len(target) < len(t)
    assert np.isnan(out[scores==0]).all()


def test_no_shift_preserves_exact_frames_next_to_missing():
    kp = np.arange(20*2,dtype=float).reshape(1,20,1,2)
    co = np.ones((1,20,1))
    kp[0,8] = np.nan; co[0,8]=0
    out,cc,tt = resample_keypoints(kp,co,[0.],fps=30)
    assert np.allclose(out,kp,equal_nan=True)
    assert cc[0,7,0]==cc[0,9,0]==1


@pytest.mark.parametrize('interpolation', ['cubic', 'linear'])
def test_zero_confidence_breaks_interpolation_at_zero_cutoff(interpolation):
    times = np.arange(7, dtype=float)
    kp = np.repeat(times[None, :, None, None], 2, axis=-1)
    kp[0, 3] = 1000
    co = np.ones(kp.shape[:3])
    co[0, 3] = 0
    query = np.array([1.5, 2., 2.5, 3., 3.5, 4., 4.5])
    out, scores, _ = resample_keypoints(kp, co, [0.], timestamps=times,
                                        reference_times=query, min_confidence=0,
                                        interpolation=interpolation)
    valid = np.array([True, True, False, False, False, True, True])
    np.testing.assert_array_equal(scores[0, :, 0], valid)
    np.testing.assert_allclose(out[0, valid, 0, :], np.repeat(query[valid, None], 2, axis=1))
    assert np.isnan(out[0, ~valid]).all()


@pytest.fixture
def rig():
    return Calibration(np.tile(np.eye(3), (2, 1, 1)), np.array([[0., 0, 0], [-1., 0, 0]]),
                       np.tile(np.eye(3), (2, 1, 1)), np.empty((0, 17, 3)), None,
                       'true', np.zeros(2), np.ones(2))


@pytest.mark.parametrize('clock', [{}, {'fps': 30}])
def test_zero_confidence_is_missing_in_geometry_at_zero_cutoff(rig, clock):
    expected = np.broadcast_to([0., 0., 4.], (2, 17, 3))
    kp = project(rig, expected)
    co = np.ones(kp.shape[:3])
    co[1, 0, 0] = 0
    cal = Calibrator(kp, co, rig.K, min_confidence=0)
    assert not cal.tracks['valid'][0, 1]
    points, inliers = triangulate(rig, kp, co, min_confidence=0, **clock)
    assert np.isnan(points[0, 0]).all()
    assert not inliers[:, 0, 0].any()
    np.testing.assert_allclose(points[1], expected[1], atol=1e-12)
    assert inliers[:, 1].all()


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('imaginary', [0., 1.])
@pytest.mark.parametrize(('entry', 'field'), [
    ('resample', 'keypoints'), ('resample', 'confidence'),
    ('resample', 'offsets_seconds'), ('resample', 'timestamps'),
    ('resample', 'reference_times'), ('resample', 'fps'),
    ('resample', 'min_confidence'),
    ('estimate', 'keypoints'), ('estimate', 'confidence'),
    ('estimate', 'timestamps'), ('estimate', 'max_offset_seconds'),
    ('calibrate', 'timestamps'), ('calibrate_sync', 'max_offset_seconds'),
    ('triangulate', 'keypoints'), ('triangulate', 'confidence'),
    ('triangulate', 'timestamps'), ('triangulate', 'offsets_seconds'),
])
def test_temporal_paths_reject_complex_values_before_casting(rig, entry, field, imaginary):
    args = dict(keypoints=np.zeros((2, 20, 17, 2)), confidence=np.ones((2, 20, 17)))
    values = dict(args, offsets_seconds=np.zeros(2), fps=30, timestamps=np.arange(20) / 30,
                  reference_times=np.arange(20) / 30, max_offset_seconds=.1, min_confidence=.3)
    value = np.asarray(values[field], dtype=complex)
    value.flat[0] += imaginary * 1j
    args.update(fps=30)
    if entry == 'resample':
        args['offsets_seconds'] = values['offsets_seconds']
    if entry == 'calibrate_sync':
        args['estimate_sync'] = True
    if entry == 'triangulate' and field == 'offsets_seconds':
        rig.time_offsets_seconds = value
    else:
        args[field] = value
    with pytest.raises(ValueError, match=field):
        if entry == 'resample':
            resample_keypoints(**args)
        elif entry == 'estimate':
            estimate_time_offsets(K=rig.K, **args)
        elif entry.startswith('calibrate'):
            calibrate(K=rig.K, **args)
        else:
            triangulate(rig, **args)


def test_timestamp_gap_is_not_interpolated():
    ts=np.array([0.,.1,.2,.5,.6,.7])
    kp=np.zeros((1,6,1,2)); co=np.ones((1,6,1))
    out,cc,_=resample_keypoints(kp,co,[0.],timestamps=ts,reference_times=np.array([.1,.3,.6]))
    assert cc[0,1,0]==0
    assert np.isnan(out[0,1]).all()
    # An exact observation immediately before a gap must remain valid.
    _,cc,_=resample_keypoints(kp,co,[0.],timestamps=ts)
    assert cc[0,2,0]==1


def test_temporal_input_validation():
    with pytest.raises(ValueError, match='fps'):
        time_axes(2,30)
    with pytest.raises(ValueError, match='strictly increasing'):
        time_axes(2,30,timestamps=np.zeros(30))
    with pytest.raises(ValueError, match='positive'):
        estimate_time_offsets(np.zeros((2,30,17,2)),np.ones((2,30,17)),np.eye(3),fps=30,
                              max_offset_seconds=0.)


def test_json_and_transformation_keep_temporal_offsets(tmp_path):
    rig=Calibration(np.tile(np.eye(3),(2,1,1)),np.array([[0.,0,0],[-1.,0,0]]),
                    np.tile(np.eye(3),(2,1,1)),np.empty((0,17,3)),None,'true',
                    np.zeros(2),np.ones(2),{'skeleton':'coco17'},np.array([0.,.02]))
    path=tmp_path/'rig.json'; rig.save_json(path)
    back=Calibration.load_json(path).transformed(2.)
    assert np.array_equal(back.time_offsets_seconds,rig.time_offsets_seconds)
    assert back.info['skeleton']=='coco17'
    with pytest.raises(ValueError,match='fps'):
        triangulate(back,np.zeros((2,30,17,2)),np.ones((2,30,17)))


@pytest.mark.slow
def test_subframe_offsets_on_independent_synthetic_capture():
    # Imported benchmark uses independent samples of a 3D world trajectory,
    # not time shifts of noisy observations. Positive and negative fractional offsets.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('sync_bench',Path(__file__).parents[1]/'benchmarks'/'benchmark_sync.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    kp,conf,K,true,_=mod.make_delayed_scene(2,2.,duration=12.)
    sync=estimate_time_offsets(kp,conf,K,fps=30,max_offset_seconds=.15)
    assert np.max(np.abs(sync.offsets_seconds-true)) < .012
    assert sync.offsets_seconds[0]==0


def test_stationary_tracks_are_not_reported_as_successful_sync():
    rng=np.random.default_rng(7)
    points=rng.normal(size=(2,1,17,2))
    kp=np.repeat(points,90,axis=1)
    with pytest.raises(ValueError,match='insufficient'):
        estimate_time_offsets(kp,np.ones(kp.shape[:3]),np.eye(3),fps=30,max_offset_seconds=.1)


@pytest.mark.parametrize('noise_px',[0.,.5,2.])
def test_uniform_translation_is_temporally_unobservable(noise_px):
    from posecalib import project
    rng=np.random.default_rng(31)
    rig=Calibration(np.tile(np.eye(3),(2,1,1)),np.array([[0.,0,0],[-1.,.1,0]]),
                    np.tile(np.array([[1200.,0,960],[0,1200.,540],[0,0,1]]),(2,1,1)),
                    np.empty((0,17,3)),None,'true',np.zeros(2),np.ones(2))
    cloud=rng.uniform(-.5,.5,(17,3))+[0,0,4.]
    tt=np.arange(90)/30.
    kp=np.stack([project(rig,cloud[None]+(tt-delay)[:,None,None]*np.array([.3,.15,.05]))[c]
                 for c,delay in enumerate([0.,.037])])
    kp+=rng.normal(size=kp.shape)*noise_px
    with pytest.raises(ValueError,match='insufficient'):
        estimate_time_offsets(kp,np.ones(kp.shape[:3]),rig.K,fps=30,max_offset_seconds=.1)


def test_custom_empty_segments_and_floor_metadata():
    from posecalib.initialization import build_observations
    tracks=dict(valid=np.ones((8,2),bool),uv=np.zeros((8,2,2)),c=np.ones((8,2)))
    obs=build_observations(tracks,np.ones(8,bool),2,4,np.empty((0,2),int))
    assert all(a.shape==(0,) and a.dtype.kind=='i' for a in obs['bones'])


@pytest.mark.slow
def test_three_cameras_with_per_camera_timestamps():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('sync_bench',Path(__file__).parents[1]/'benchmarks'/'benchmark_sync.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    kp,conf,K,true,_=mod.make_delayed_scene(0,1.5,duration=12.)
    ts=np.broadcast_to(np.arange(kp.shape[1])/30.,(3,kp.shape[1])).copy()
    sync=estimate_time_offsets(kp[:3],conf[:3],K[:3],timestamps=ts,max_offset_seconds=.15)
    assert np.max(np.abs(sync.offsets_seconds-true[:3]))<.012


@pytest.mark.slow
def test_blazepose33_sync_calibration_and_floor():
    import importlib.util
    from pathlib import Path
    from posecalib import calibrate, align_to_floor
    spec=importlib.util.spec_from_file_location('sync_bench',Path(__file__).parents[1]/'benchmarks'/'benchmark_sync.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    kp,conf,K,true,cams=mod.make_delayed_scene(1,1.5,duration=15.,noise_px=1.,blazepose=True)
    d=np.linalg.norm(cams[0]['C']-cams[1]['C'])
    rig=calibrate(kp,conf,K,camera_distance=(0,1,d),estimate_sync=True,fps=30,
                   max_offset_seconds=.15,frame_step=5)
    assert rig.points.shape[1]==33
    assert np.max(np.abs(rig.time_offsets_seconds-true))<.012
    rot,centres=mod.geometry_error(rig,cams)
    assert rot<.2 and centres<20
    floor=align_to_floor(rig)
    assert np.max(np.abs(floor.centers[:,2]-[c['C'][2] for c in cams]))<.1
