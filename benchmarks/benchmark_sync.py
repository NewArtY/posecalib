"""Reproducible constant-delay benchmark (30 FPS, noisily observed moving skeleton).

python benchmarks/benchmark_sync.py --seeds 3 --workers 4
Results include per-seed failures, estimated delays and geometry with/without correction.
"""
from __future__ import annotations
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posecalib import calibrate
from posecalib.api import Calibration, project
from posecalib.geometry import rotation_angle_deg
from posecalib.sim import DEFAULT_CFG, intrinsics, make_seed_data
from posecalib.synchronization import estimate_time_offsets


def make_delayed_scene(seed=0, delay_frames=2., duration=20., noise_px=2., outliers=.02, blazepose=False):
    cfg = dict(DEFAULT_CFG, cal_max_duration=duration+2.)
    data = make_seed_data(seed, cfg)
    cams = data['cams']
    C = len(cams)
    rig = Calibration(np.stack([c['R'] for c in cams]), np.stack([c['t'] for c in cams]),
                      intrinsics(cams), np.zeros((0,17,3)), None, 'true', np.zeros(C), np.ones(C))
    # Independent underlying world trajectory sampled at differing capture times.
    # No interpolation of already-noisy 2D observations to manufacture delays.
    offsets = np.array([0., 1., -.7, .4])[:C] * delay_frames / 30.
    ts = np.arange(int(duration*30))/30.
    world_times = np.arange(len(data['walk']['J']))/30.
    world = data['walk']['J']
    if blazepose:
        # Geometric 33-point coverage test, not a simulation of MediaPipe accuracy.
        mapping=[0,1,1,1,2,2,2,3,4,0,0,5,6,7,8,9,10,9,10,9,10,9,10,11,12,13,14,15,16,15,16,15,16]
        world=world[:,mapping].copy()
        for j in [1,3,4,6,9,10,17,18,19,20,21,22,29,30,31,32]:
            # Additional face/hand/foot points with distinct small offsets.
            world[:,j]+=np.array([.012*((j%3)-1), .015*((j%5)-2), .008*((j%7)-3)])
    J=world.shape[1]
    P = world.reshape(len(world_times), -1)
    kp = []
    for c in range(C):
        query = ts + .5 - offsets[c]
        X = np.stack([np.interp(query, world_times, P[:,k]) for k in range(P.shape[1])],axis=1).reshape(-1,J,3)
        kp.append(project(rig,X)[c])
    kp = np.stack(kp)
    rng = np.random.default_rng(901+seed)
    confidence = rng.uniform(.65,1.,kp.shape[:3])
    kp += rng.normal(size=kp.shape) * noise_px / confidence[...,None]
    gross = rng.random(kp.shape[:3]) < outliers
    kp[gross] += rng.uniform(-80,80,(int(gross.sum()),2))
    # Missing tracks: small independent dropout and an explicit occlusion block.
    missing = rng.random(kp.shape[:3]) < .005
    missing[2,150:165,9] = True
    kp[missing] = np.nan
    confidence[missing] = 0.
    return kp, confidence, rig.K, offsets, cams


def geometry_error(cal,cams):
    R0,C0=cams[0]['R'],cams[0]['C']
    rot=np.mean([rotation_angle_deg(cal.R[k],cams[k]['R']@R0.T) for k in range(1,len(cams))])
    ctr=np.mean([np.linalg.norm(cal.centers[k]-(cams[k]['C']-C0)@R0.T) for k in range(1,len(cams))])*1000
    return float(rot),float(ctr)


def job(args):
    seed,delay=args
    t=time.perf_counter()
    base=dict(seed=seed,delay_frames=delay,ok=False)
    try:
        kp,conf,K,true,cams=make_delayed_scene(seed,delay)
        d=np.linalg.norm(cams[0]['C']-cams[1]['C'])
        raw=calibrate(kp,conf,K,camera_distance=(0,1,d),frame_step=5)
        aligned=calibrate(kp,conf,K,camera_distance=(0,1,d),frame_step=5,
                          estimate_sync=True,fps=30,max_offset_seconds=.15)
        r0,c0=geometry_error(raw,cams); r1,c1=geometry_error(aligned,cams)
        base.update(ok=True,offset_error_ms=float(1000*np.max(np.abs(aligned.time_offsets_seconds-true))),
                    true_offsets_seconds=true.tolist(),estimated_offsets_seconds=aligned.time_offsets_seconds.tolist(),
                    rotation_raw_deg=r0,rotation_sync_deg=r1,centers_raw_mm=c0,centers_sync_mm=c1,
                    sync_pair_consistency_ms=aligned.info['synchronization']['pair_consistency_seconds']*1000)
    except Exception as e:
        base.update(error=f'{type(e).__name__}: {e}')
    base['runtime_s']=time.perf_counter()-t
    return base


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',type=Path,default=Path('benchmarks/sync_results'))
    p.add_argument('--plots-only',action='store_true',help='rebuild summary and plot from saved results.json')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    tasks=[(s,d) for d in [0.,.5,1.,2.,3.] for s in range(a.seeds)]
    if a.plots_only:
        rows=json.loads((a.out/'results.json').read_text(encoding='utf-8'))
        a.seeds=len({r['seed'] for r in rows})
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows=[]
            for row in pool.map(job,tasks):
                rows.append(row); print(json.dumps(row),flush=True)
                (a.out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# Constant-offset synthetic benchmark','',
           f'30 FPS, 20 s, 4 cameras; {a.seeds} seeds per delay; 2 px noise / confidence, 2% outliers, missing points.',
           'Offsets: [0, 1, -0.7, 0.4] times the stated maximum delay. Search: +/-150 ms relative to camera 0.',
           'All values are medians over successful runs; delay error is the largest error across cameras in each run.','',
           '| Max delay (frames) | Success | Offset error (ms) | Rotation raw / corrected (deg) | Centres raw / corrected (mm) |',
           '|---|---|---|---|---|']
    for d in [0.,.5,1.,2.,3.]:
        ok=[r for r in rows if r['delay_frames']==d and r['ok']]
        med=lambda k:float(np.median([r[k] for r in ok])) if ok else float('nan')
        lines.append(f"| {d:g} | {len(ok)}/{a.seeds} | {med('offset_error_ms'):.2f} | {med('rotation_raw_deg'):.3f} / {med('rotation_sync_deg'):.3f} | {med('centers_raw_mm'):.1f} / {med('centers_sync_mm'):.1f} |")
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delays=[0.,.5,1.,2.,3.]
    def series(key):
        return [np.median([r[key] for r in rows if r['ok'] and r['delay_frames']==d]) for d in delays]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    axes[0].plot(delays,series('offset_error_ms'),'o-',color='#136b83')
    axes[0].set_ylabel('Max camera offset error (ms)')
    for ax,raw,corrected,label in [
        (axes[1],'rotation_raw_deg','rotation_sync_deg','Rotation error (degrees)'),
        (axes[2],'centers_raw_mm','centers_sync_mm','Camera centre error (mm)')]:
        ax.plot(delays,series(raw),'s--',label='No temporal correction',color='#9c4a32')
        ax.plot(delays,series(corrected),'o-',label='Estimated offsets',color='#136b83')
        ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlabel('Max input delay (frames at 30 FPS)')
        ax.set_xticks(delays); ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'Constant-offset simulation: median over {a.seeds} scenes, 20 s walk, 2 px noise + outliers')
    fig.savefig(a.out/'synchronization.png',dpi=160)
    plt.close(fig)

if __name__=='__main__': main()
