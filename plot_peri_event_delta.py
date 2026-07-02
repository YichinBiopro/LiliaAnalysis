"""
plot_peri_event_delta.py
========================
Cross-subject per-event Δ (post − pre) bar chart, built from the
``*_absolute_peri_event.csv`` summaries written by ``plot_index_vs_raw.py``.

One row per subject; within each row the events sit on a shared x-axis (the
EVENTS-table order) and each event shows one coloured bar per signal
(Focus / Flow / Calm / Relaxation / Band entropy). Δ > 0 means the index rose
from the pre-onset rest window to the post-onset window. Continuous events
(whose pre-window was borrowed from an earlier rest) are marked with '*'.

CLI usage
---------
    python plot_peri_event_delta.py [--ch 1] [--out PATH]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_event_markers as pem
from plot_index_vs_raw import SIGNAL_SPECS

SUBJECTS = ['Ann', 'Hsin', 'Hardy', 'TYY', 'James']
# Master event order (as listed in the EVENTS table).
EVENT_ORDER = [name for name, *_ in pem.EVENTS]


def _peri_csv(subject: str, ch: int) -> str:
    sub_dir = os.path.join('iBrainCenter', pem.SUBJECTS[subject]['dir'])
    return os.path.join(
        sub_dir, f'index_vs_raw_ch{ch}_absolute_peri_event.csv')


def plot_peri_event_delta(ch: int, outpath: str) -> None:
    keys = [k for k, _, _ in SIGNAL_SPECS]
    n_sig = len(keys)
    width = 0.8 / n_sig
    x = np.arange(len(EVENT_ORDER))

    fig, axes = plt.subplots(len(SUBJECTS), 1, figsize=(15, 2.4 * len(SUBJECTS)),
                             sharex=True)
    fig.suptitle(
        f'Per-event Δ (post − pre, {30}s windows) — qEEG indices + band entropy '
        f'(ch{ch})', fontsize=14, fontweight='bold')

    for ax, subject in zip(axes, SUBJECTS):
        csv = _peri_csv(subject, ch)
        if not os.path.isfile(csv):
            ax.text(0.5, 0.5, f'{subject}: {os.path.basename(csv)} missing',
                    ha='center', va='center', transform=ax.transAxes)
            continue
        df = pd.read_csv(csv).set_index('event')
        for s, (key, label, color) in enumerate(SIGNAL_SPECS):
            vals = [df.at[ev, f'{key}_delta'] if ev in df.index else np.nan
                    for ev in EVENT_ORDER]
            ax.bar(x + (s - (n_sig - 1) / 2) * width, vals, width,
                   color=color, label=label if subject == SUBJECTS[0] else None)
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_ylabel(subject, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlim(-0.6, len(EVENT_ORDER) - 0.4)

    # X labels: event name + '*' if continuous for any subject.
    cont_any = {ev: False for ev in EVENT_ORDER}
    for subject in SUBJECTS:
        csv = _peri_csv(subject, ch)
        if os.path.isfile(csv):
            d = pd.read_csv(csv).set_index('event')
            for ev in EVENT_ORDER:
                if ev in d.index and bool(d.at[ev, 'continuous']):
                    cont_any[ev] = True
    labels = [f'{ev}\n(continuous*)' if cont_any[ev] else ev
              for ev in EVENT_ORDER]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
                   ncol=len(SIGNAL_SPECS), fontsize=9, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {outpath}')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Cross-subject per-event Δ (post−pre) bar chart.')
    p.add_argument('--ch', type=int, default=1, help='1-based channel (default 1).')
    p.add_argument('--out', default='report_figures/peri_event_delta_all.png',
                   help='Output PNG path.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot_peri_event_delta(args.ch, args.out)


if __name__ == '__main__':
    main()
