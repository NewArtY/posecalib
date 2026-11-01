"""Simulation benchmark of keypoint-based extrinsic calibration.

Experiments (four cameras around a 4 x 4 m area, one person walking):

* ``noise``    - detector noise sigma0 = 1, 2, 4, 8 px for several solver variants;
* ``duration`` - length of the calibration walk, 5-60 s;
* ``bias``     - systematic, view-dependent keypoint offsets of 0-4 cm;
* ``board``    - ChArUco-like board baseline (25 and 100 poses) on the same test data;
* ``static``   - the person stands in place instead of walking.

Usage::

    python benchmarks/run_benchmark.py                  # 10 seeds, all CPU cores but two
    python benchmarks/run_benchmark.py --seeds 3        # quick look
    python benchmarks/run_benchmark.py --plots-only     # re-plot from saved CSV files

Results go to ``benchmarks/results/`` (CSV per experiment, ``summary.md``, PNG figures).
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')        # one BLAS thread per worker; we parallelise over seeds

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))   # allow running without installation

from posecalib.sim import DEFAULT_CFG, DEFAULT_VARIANT, run_board_trial, run_pose_trial  # noqa: E402

SIGMAS = [1.0, 2.0, 4.0, 8.0]
NOISE_VARIANTS = ['ls', 'weighted_ls', 'huber', 'weighted_cauchy', DEFAULT_VARIANT]
DURATIONS = [5.0, 10.0, 20.0, 40.0, 60.0]
DELTAS_CM = [0.0, 1.0, 2.0, 4.0]
BOARD_POSES = [25, 100]
LABEL = {'ls': 'least squares', 'weighted_ls': 'weighted LS', 'huber': 'Huber',
         'weighted_huber': 'weighted Huber', 'weighted_cauchy': 'weighted Cauchy',
         'weighted_cauchy_segments': 'weighted Cauchy + segments (default)'}
EXPERIMENTS = ['noise', 'duration', 'bias', 'board', 'static']


# -----------------------------------------------------------------------------
# tasks
# -----------------------------------------------------------------------------
def build_tasks(seeds):
    t = []
    for s0 in SIGMAS:
        t += [('pose', 'noise', sd, dict(sigma0=s0, compute_crlb=True), NOISE_VARIANTS) for sd in seeds]
    for T in DURATIONS[::-1]:
        t += [('pose', 'duration', sd, dict(cal_duration=T, compute_crlb=True), [DEFAULT_VARIANT]) for sd in seeds]
    for d in DELTAS_CM:
        t += [('pose', 'bias', sd, dict(delta=d / 100.0), [DEFAULT_VARIANT]) for sd in seeds]
    for nb in BOARD_POSES:
        t += [('board', 'board', sd, dict(board_poses=nb), DELTAS_CM) for sd in seeds]
    t += [('pose', 'static', sd, dict(motion='static', compute_crlb=True), [DEFAULT_VARIANT]) for sd in seeds]
    return t


def run_task(task):
    kind, exp, seed, over, arg = task
    cfg = dict(DEFAULT_CFG, **over)
    try:
        rows = run_board_trial(seed, cfg, arg) if kind == 'board' else run_pose_trial(seed, cfg, arg)
    except Exception as e:  # noqa: BLE001 - keep the run alive, record the failure
        rows = [dict(seed=seed, ok=False, error=f'{type(e).__name__}: {e}', **over)]
    for r in rows:
        r['experiment'] = exp
    return rows


# -----------------------------------------------------------------------------
# I/O and statistics
# -----------------------------------------------------------------------------
def save_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()})


def load_csv(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rr = {}
            for k, v in r.items():
                if v in ('', None):
                    rr[k] = None
                elif v in ('True', 'False'):
                    rr[k] = v == 'True'
                else:
                    try:
                        rr[k] = float(v)
                    except ValueError:
                        rr[k] = v
            out.append(rr)
    return out


def select(rows, **kw):
    return [r for r in rows if r.get('ok') and all(
        (abs(r.get(k) - v) < 1e-9 if isinstance(v, float) and isinstance(r.get(k), float) else r.get(k) == v)
        for k, v in kw.items())]


def stats(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def fmt(m, p=2):
    return '-' if not np.isfinite(m[0]) else f'{m[0]:.{p}f} [{m[1]:.{p}f}-{m[2]:.{p}f}]'


# -----------------------------------------------------------------------------
# summary
# -----------------------------------------------------------------------------
def summary(R, n_seeds):
    L = [f'# Benchmark summary\n\nMedian [interquartile range] over {n_seeds} random scenes. '
         'Rotation error: mean over the three non-reference cameras. dX: mean displacement of test '
         'keypoints caused by the calibration (estimated vs true cameras, same 2D detections); '
         'MPJPE: error of triangulation with the true cameras. "range" = scale from the distance between '
         'two cameras measured with 5 mm error.\n']
    L += ['## Detector noise (40 s walk)\n',
          '| sigma0, px | variant | rotation error, deg | Cramer-Rao bound, deg | dX (range), mm | MPJPE, mm |',
          '|---|---|---|---|---|---|']
    for s0 in SIGMAS:
        for v in NOISE_VARIANTS:
            rr = select(R['noise'], sigma0=s0, variant=v)
            L.append(f'| {s0:g} | {LABEL[v]} | {fmt(stats(rr, "rot_err_deg"), 3)} | '
                     f'{fmt(stats(rr, "crlb_rot_deg"), 3)} | {fmt(stats(rr, "disp_range_mm"))} | '
                     f'{fmt(stats(rr, "mpjpe_true_mm"), 1)} |')
    L += ['', '## Calibration walk duration (sigma0 = 3 px)\n',
          '| duration, s | rotation error, deg | Cramer-Rao bound, deg | camera centres (range), mm | dX (range), mm |',
          '|---|---|---|---|---|']
    for T in DURATIONS:
        rr = select(R['duration'], duration_s=T)
        L.append(f'| {T:g} | {fmt(stats(rr, "rot_err_deg"), 3)} | {fmt(stats(rr, "crlb_rot_deg"), 3)} | '
                 f'{fmt(stats(rr, "center_err_range_mm"), 1)} | {fmt(stats(rr, "disp_range_mm"))} |')
    L += ['', '## Systematic keypoint offset vs board (sigma0 = 3 px, 40 s walk)\n',
          '| offset, cm | keypoints: rotation, deg | keypoints: dX (range), mm | board 25 poses: rotation, deg | '
          'board 25: dX, mm | board 100 poses: rotation, deg | board 100: dX, mm | MPJPE, mm |',
          '|---|---|---|---|---|---|---|---|']
    for d in DELTAS_CM:
        rp = select(R['bias'], delta_cm=d)
        b25 = select(R['board'], delta_cm=d, board_poses=25.0)
        b100 = select(R['board'], delta_cm=d, board_poses=100.0)
        L.append(f'| {d:g} | {fmt(stats(rp, "rot_err_deg"), 3)} | {fmt(stats(rp, "disp_range_mm"))} | '
                 f'{fmt(stats(b25, "rot_err_deg"), 3)} | {fmt(stats(b25, "disp_board_mm"))} | '
                 f'{fmt(stats(b100, "rot_err_deg"), 3)} | {fmt(stats(b100, "disp_board_mm"))} | '
                 f'{fmt(stats(rp, "mpjpe_true_mm"), 1)} |')
    walk = select(R['duration'], duration_s=40.0)
    st = select(R['static'])
    L += ['', '## Walking vs standing in place (40 s)\n',
          '| motion | rotation error, deg | Cramer-Rao bound, deg | dX (range), mm |', '|---|---|---|---|',
          f'| walking | {fmt(stats(walk, "rot_err_deg"), 3)} | {fmt(stats(walk, "crlb_rot_deg"), 3)} | '
          f'{fmt(stats(walk, "disp_range_mm"))} |',
          f'| standing | {fmt(stats(st, "rot_err_deg"), 3)} | {fmt(stats(st, "crlb_rot_deg"), 3)} | '
          f'{fmt(stats(st, "disp_range_mm"))} |', '']
    fails = sum(1 for e in EXPERIMENTS for r in R[e] if not r.get('ok'))
    L.append(f'Failed trials: {fails}.\n')
    return '\n'.join(L)


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------
def figures(R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e6e6', 'savefig.dpi': 150,
                         'savefig.bbox': 'tight'})
    accent = '#D55E00'
    style = {'ls': ('#999999', 'x', ':'), 'weighted_ls': ('#777777', 'v', ':'), 'huber': ('#444444', '^', '--'),
             'weighted_cauchy': ('#000000', 's', '-'), DEFAULT_VARIANT: (accent, 'o', '-')}

    def series(ax, xs, rows_at, key, col, mk, ls, label, logx=True, dodge=1.0):
        pts = [(x, stats(rows_at(x), key)) for x in xs]
        pts = [(x, m) for x, m in pts if np.isfinite(m[0])]
        x = [p[0] * dodge if logx else p[0] + dodge for p in pts]
        med = np.array([p[1][0] for p in pts])
        err = [med - [p[1][1] for p in pts], [p[1][2] for p in pts] - med]
        ax.errorbar(x, med, yerr=err, color=col, marker=mk, ls=ls, ms=4, lw=1, capsize=2, label=label)

    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    def log_axes(ax, xticks, yticks):
        ax.set(xscale='log', yscale='log')
        for axis, ticks in ((ax.xaxis, xticks), (ax.yaxis, yticks)):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in ticks]))
            axis.set_minor_locator(NullLocator())
            axis.set_minor_formatter(NullFormatter())

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, v in enumerate(NOISE_VARIANTS):
        col, mk, ls = style[v]
        series(axs[0], SIGMAS, lambda s, v=v: select(R['noise'], sigma0=s, variant=v), 'rot_err_deg',
               col, mk, ls, LABEL[v], dodge=0.92 + 0.04 * i)
    crlb = [stats(select(R['noise'], sigma0=s, variant=DEFAULT_VARIANT), 'crlb_rot_deg')[0] for s in SIGMAS]
    axs[0].plot(SIGMAS, crlb, 'k:', lw=1, label='Cramer-Rao bound')
    axs[0].set(xlabel='detector noise sigma0, px', ylabel='rotation error, deg', title='solver variants, 40 s walk')
    log_axes(axs[0], SIGMAS, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axs[0].set_ylim(0.012, 0.9)
    series(axs[1], DURATIONS, lambda T: select(R['duration'], duration_s=T), 'rot_err_deg', accent, 'o', '-',
           LABEL[DEFAULT_VARIANT])
    crlb = [stats(select(R['duration'], duration_s=T), 'crlb_rot_deg')[0] for T in DURATIONS]
    axs[1].plot(DURATIONS, crlb, 'k:', lw=1)
    axs[1].set(xlabel='calibration walk, s', ylabel='rotation error, deg', title='walk duration, sigma0 = 3 px')
    log_axes(axs[1], DURATIONS, [0.02, 0.05, 0.1, 0.2, 0.5])
    axs[1].set_ylim(0.025, 0.45)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / 'noise_and_duration.png')
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key_p, key_b, ylab in [(axs[0], 'rot_err_deg', 'rot_err_deg', 'rotation error, deg'),
                                   (axs[1], 'disp_range_mm', 'disp_board_mm', 'calibration-induced 3D error dX, mm')]:
        series(ax, DELTAS_CM, lambda d: select(R['bias'], delta_cm=d), key_p, accent, 'o', '-',
               'body keypoints (40 s walk)', logx=False, dodge=-0.06)
        series(ax, DELTAS_CM, lambda d: select(R['board'], delta_cm=d, board_poses=25.0), key_b, '#000000', '^',
               '-', 'board, 25 poses', logx=False, dodge=0.0)
        series(ax, DELTAS_CM, lambda d: select(R['board'], delta_cm=d, board_poses=100.0), key_b, '#777777', 'D',
               '--', 'board, 100 poses', logx=False, dodge=0.06)
        ax.set(xlabel='systematic keypoint offset, cm', ylabel=ylab, xticks=DELTAS_CM, ylim=(0, None))
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'bias_vs_board.png')
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--out', type=Path, default=HERE / 'results')
    ap.add_argument('--plots-only', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.plots_only:
        tasks = build_tasks(range(a.seeds))
        print(f'{len(tasks)} tasks on {a.workers} workers')
        t0 = time.perf_counter()
        rows = []
        with Pool(a.workers) as pool:
            for i, rr in enumerate(pool.imap_unordered(run_task, tasks)):
                rows += rr
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print(f'  {i + 1}/{len(tasks)} done ({time.perf_counter() - t0:.0f} s)', flush=True)
        for e in EXPERIMENTS:
            save_csv(sorted([r for r in rows if r['experiment'] == e], key=lambda r: str(sorted(r.items()))),
                     a.out / f'{e}.csv')
    R = {e: load_csv(a.out / f'{e}.csv') for e in EXPERIMENTS}
    n_seeds = len({r['seed'] for r in R['noise']})
    (a.out / 'summary.md').write_text(summary(R, n_seeds), encoding='utf-8')
    try:
        figures(R, a.out)
    except ImportError:
        print('matplotlib is not installed: figures skipped (pip install "posecalib[bench]")')
    print('Results:', a.out)


if __name__ == '__main__':
    main()
