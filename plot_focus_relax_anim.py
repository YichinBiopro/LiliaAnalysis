"""
plot_focus_relax_anim.py
========================
Animated ("moving points") version of the Focus-vs-Relax scatter produced by
``spectral_entropy.py``. A 5-minute window slides across the recording: each
frame shows only the per-window (Focus, Relax) points whose window-centre time
falls inside the trailing window, so the cloud *moves* through Focus-Relax space
and you can watch the brain-state trajectory evolve in real recording time.

Input is the ``*_band_entropy_ch*.csv`` already written by spectral_entropy.py
(it carries time_s + p_theta/p_alpha/p_beta), so the Focus/Relax indices are
recomputed in closed form from the stored relative band powers — no PSDs are
re-estimated. Output is an animated GIF next to the CSV.

CLI usage
---------
    python plot_focus_relax_anim.py --csv <*_band_entropy_ch1.csv>
                                    [--window-min 5] [--max-frames 300]
                                    [--fps 15] [--out <path.gif>]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')  # headless: render straight to a GIF file
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lilia.qeeg import focus_index, relaxation_index


def _load_focus_relax(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a band-entropy CSV and return (time_s, focus, relax) for the clean
    (finite-proportion) windows, sorted by time."""
    df = pd.read_csv(csv_path)
    required = {'time_s', 'p_theta', 'p_alpha', 'p_beta'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f'Error: {csv_path} is missing columns {sorted(missing)} — '
                 'pass a *_band_entropy_ch*.csv from spectral_entropy.py.')
    P = df[['p_theta', 'p_alpha', 'p_beta']].to_numpy(dtype=float)
    t = df['time_s'].to_numpy(dtype=float)
    finite = np.all(np.isfinite(P), axis=1) & np.isfinite(t)
    P, t = P[finite], t[finite]
    order = np.argsort(t)
    P, t = P[order], t[order]
    focus = np.array([focus_index(th, al, be) for th, al, be in P])
    relax = np.array([relaxation_index(th, al, be) for th, al, be in P])
    return t, focus, relax


def animate_focus_relax(
    csv_path: str,
    outpath: str,
    window_min: float = 5.0,
    max_frames: int = 300,
    fps: int = 15,
) -> None:
    t, focus, relax = _load_focus_relax(csv_path)
    if t.size == 0:
        sys.exit(f'Error: no clean windows in {csv_path}.')

    window_s = window_min * 60.0
    t0, t1 = float(t[0]), float(t[-1])
    # One frame per window centre, then strided down to <= max_frames so the GIF
    # stays small on long (30 min+) recordings.
    stride = max(1, int(np.ceil(t.size / max_frames)))
    frame_idx = np.arange(0, t.size, stride)

    # Fixed axes + colour scale across all frames so the motion is the only thing
    # that changes. Indices live in ~[-1, 1] (bounded ratio, minus small penalties).
    lim = max(1.05, float(np.nanmax(np.abs(np.concatenate([focus, relax])))) * 1.05)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.axhline(0.0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0.0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Focus index')
    ax.set_ylabel('Relaxation index')

    # Trailing-window scatter (points coloured by absolute time), the connecting
    # trajectory, and a star on the current ("now") point.
    scat = ax.scatter([], [], c=[], cmap='viridis', s=20, vmin=t0, vmax=t1,
                      alpha=0.85, edgecolor='none', zorder=3)
    (trail_line,) = ax.plot([], [], color='0.6', lw=0.6, alpha=0.4, zorder=2)
    (now_pt,) = ax.plot([], [], marker='*', ms=20, mfc='#d62728', mec='k',
                        mew=0.8, ls='none', zorder=5)
    cb = fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('time (s)')
    title = ax.set_title('', fontsize=12, fontweight='bold')

    base = os.path.basename(csv_path)

    def update(i: int):
        end_i = frame_idx[i]
        t_now = t[end_i]
        in_win = (t > t_now - window_s) & (t <= t_now)
        xs, ys, ts = focus[in_win], relax[in_win], t[in_win]
        scat.set_offsets(np.column_stack([xs, ys]) if xs.size
                         else np.empty((0, 2)))
        scat.set_array(ts)
        trail_line.set_data(xs, ys)
        now_pt.set_data([focus[end_i]], [relax[end_i]])
        win_lo = max(t0, t_now - window_s)
        title.set_text(f'{base}\nFocus vs Relax — {window_min:g} min window '
                       f'[{win_lo / 60:.1f}–{t_now / 60:.1f} min]  '
                       f'({xs.size} pts)')
        return scat, trail_line, now_pt, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_idx), interval=1000 / fps, blit=False)
    anim.save(outpath, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f'Saved: {outpath}  ({len(frame_idx)} frames, stride={stride}, '
          f'{t.size} windows over {(t1 - t0) / 60:.1f} min)')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Animated sliding-window Focus-vs-Relax scatter from a '
                    'spectral_entropy band-entropy CSV.')
    p.add_argument('--csv', required=True, metavar='PATH',
                   help='A *_band_entropy_ch*.csv produced by spectral_entropy.py.')
    p.add_argument('--window-min', type=float, default=5.0, metavar='MIN',
                   help='Trailing window length in minutes (default 5).')
    p.add_argument('--max-frames', type=int, default=300, metavar='N',
                   help='Cap on number of frames; windows are strided to fit '
                        '(default 300).')
    p.add_argument('--fps', type=int, default=15, metavar='N',
                   help='Frames per second of the output GIF (default 15).')
    p.add_argument('--out', metavar='PATH',
                   help='Output GIF path (default: <csv stem>_focus_relax_<W>min.gif).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out:
        outpath = args.out
    else:
        stem = os.path.splitext(args.csv)[0]
        # Strip the trailing _focus_relax if the user passed that by mistake; the
        # CSV stem is normally ..._band_entropy_ch1.
        outpath = f'{stem}_focus_relax_{args.window_min:g}min.gif'
    animate_focus_relax(
        args.csv, outpath,
        window_min=args.window_min, max_frames=args.max_frames, fps=args.fps)


if __name__ == '__main__':
    main()
