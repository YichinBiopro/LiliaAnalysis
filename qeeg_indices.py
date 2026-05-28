"""
qEEG Wellness Indices — Appendix J, Chapter 3
==============================================
Implements relative power normalisation (§3.1), bounded safe ratio (§3.2),
and the four wellness indices: Focus, Flow, Calm, Relaxation (§3.3).

CLI usage
---------
    python qeeg_indices.py --csv <path.csv> [--fs 500] [--ch 1] [--win 5] [--out <dir>]

The CSV is expected to share the same format as the lilia_analysis pipeline
(timestamp µs in col-0, EEG channels in subsequent columns, 4-row header).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILON    = 1e-9
BAND_THETA = (4.0,  8.0)
BAND_ALPHA = (8.0, 13.0)
BAND_BETA  = (12.0, 30.0)

DEFAULT_FS      = 500
DEFAULT_WIN_SEC = 5.0


# ── §3.1  Relative Power Normalisation ────────────────────────────────────────

def _band_power(freqs, psd_linear, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd_linear[mask], freqs[mask]))


def compute_relative_powers(data_col, fs=DEFAULT_FS):
    """
    Compute relative θ / α / β powers for a 1-D EEG segment.
    Delta & Gamma are excluded per §3.1 to minimise motion/EMG artefacts.

    Returns
    -------
    (theta_rel, alpha_rel, beta_rel) : floats in [0, 1], summing to ~1.
    """
    freqs, psd = signal.welch(data_col, fs=fs, nperseg=fs * 4,
                               noverlap=fs * 2, window='hann')
    p_theta = _band_power(freqs, psd, *BAND_THETA)
    p_alpha = _band_power(freqs, psd, *BAND_ALPHA)
    p_beta  = _band_power(freqs, psd, *BAND_BETA)
    p_total = p_theta + p_alpha + p_beta + EPSILON
    return p_theta / p_total, p_alpha / p_total, p_beta / p_total


# ── §3.2  Bounded Safe Ratio ───────────────────────────────────────────────────

def bounded_ratio(E, I, eps=EPSILON):
    """clamp((E − I) / (E + I + ε), −1, 1)"""
    return float(np.clip((E - I) / (E + I + eps), -1.0, 1.0))


# ── §3.3  Wellness Indices ─────────────────────────────────────────────────────

def focus_index(theta, alpha, beta):
    """
    Sustained attention: elevated β (minus EMG threshold) vs suppressed α / θ.
    """
    E = max(0.0, beta - 0.12)
    I = 0.7 * alpha + 0.3 * theta
    return bounded_ratio(E, I)


def flow_index(theta, alpha, beta):
    """
    Absorbed engagement: α-θ synchronisation with β-flexibility and
    α-θ imbalance penalties.
    """
    E = 0.6 * alpha + 0.4 * theta
    I = max(0.12, beta)
    if beta < 0.18:
        p_flex = 0.06 * (0.18 - beta) / 0.18
    elif beta > 0.35:
        p_flex = 0.1 * (beta - 0.35) / 0.65
    else:
        p_flex = 0.0
    p_imb = 0.06 * abs(alpha - theta)
    return bounded_ratio(E, I) - p_flex - p_imb


def calm_index(theta, alpha, beta):
    """
    Tranquil wakefulness: θ+α excitation vs β and excess-theta inhibition
    (prevents misclassifying drowsiness as calmness).
    """
    theta_excess = max(0.0, theta - 1.2 * alpha)
    E = 0.5 * theta + 0.5 * alpha
    I = beta + theta_excess
    return bounded_ratio(E, I)


def relaxation_index(theta, alpha, beta):
    """
    Deep rest: strong α dominance, penalises excess θ and high β.
    """
    theta_excess = max(0.0, theta - 1.2 * alpha)
    E = 0.7 * alpha + 0.3 * theta
    I = 0.15 * beta + 0.85 * theta_excess
    p_high_beta = 0.15 * max(0.0, beta - 0.20) / 0.80 if beta > 0.20 else 0.0
    return bounded_ratio(E, I) - p_high_beta


# ── High-level helpers ─────────────────────────────────────────────────────────

def compute_qeeg_indices(data_col, fs=DEFAULT_FS):
    """Compute all four wellness indices for a single EEG segment."""
    theta, alpha, beta = compute_relative_powers(data_col, fs=fs)
    return {
        'theta':      theta,
        'alpha':      alpha,
        'beta':       beta,
        'focus':      focus_index(theta, alpha, beta),
        'flow':       flow_index(theta, alpha, beta),
        'calm':       calm_index(theta, alpha, beta),
        'relaxation': relaxation_index(theta, alpha, beta),
    }


def compute_qeeg_indices_windowed(data_col, fs=DEFAULT_FS,
                                   win_sec=DEFAULT_WIN_SEC, step_sec=None):
    """
    Slide a window over *data_col* and compute qEEG indices at each step.

    Parameters
    ----------
    data_col  : 1-D array of EEG samples.
    fs        : Sampling frequency (Hz).
    win_sec   : Window length in seconds (default 5 s, matching app refresh).
    step_sec  : Step size in seconds (default = win_sec, i.e. non-overlapping).

    Returns
    -------
    dict with keys: 'time', 'theta', 'alpha', 'beta',
                    'focus', 'flow', 'calm', 'relaxation'
    Each value is a 1-D numpy array.
    """
    if step_sec is None:
        step_sec = win_sec
    win  = int(win_sec  * fs)
    step = int(step_sec * fs)
    n    = len(data_col)

    times, thetas, alphas, betas = [], [], [], []
    focus_v, flow_v, calm_v, relax_v = [], [], [], []

    for start in range(0, n - win + 1, step):
        seg = data_col[start : start + win]
        theta, alpha, beta = compute_relative_powers(seg, fs=fs)
        times.append((start + win // 2) / fs)
        thetas.append(theta)
        alphas.append(alpha)
        betas.append(beta)
        focus_v.append(focus_index(theta, alpha, beta))
        flow_v.append(flow_index(theta, alpha, beta))
        calm_v.append(calm_index(theta, alpha, beta))
        relax_v.append(relaxation_index(theta, alpha, beta))

    return {
        'time':       np.array(times),
        'theta':      np.array(thetas),
        'alpha':      np.array(alphas),
        'beta':       np.array(betas),
        'focus':      np.array(focus_v),
        'flow':       np.array(flow_v),
        'calm':       np.array(calm_v),
        'relaxation': np.array(relax_v),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_qeeg_indices(indices, title, outpath, t_offset=0.0):
    """
    Two-panel figure:
      top    — relative θ / α / β band powers over time.
      bottom — all four wellness indices over time.
    """
    t = indices['time'] + t_offset
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax0 = axes[0]
    ax0.plot(t, indices['theta'], label='θ Theta', color='#9467bd', lw=1.5)
    ax0.plot(t, indices['alpha'], label='α Alpha', color='#2ca02c', lw=1.5)
    ax0.plot(t, indices['beta'],  label='β Beta',  color='#d62728', lw=1.5)
    ax0.set_ylabel('Relative Power')
    ax0.set_title('Relative Band Powers (§3.1)')
    ax0.legend(loc='upper right', fontsize=9)
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t, indices['focus'],      label='Focus',      color='#1f77b4', lw=1.5)
    ax1.plot(t, indices['flow'],       label='Flow',       color='#ff7f0e', lw=1.5)
    ax1.plot(t, indices['calm'],       label='Calm',       color='#2ca02c', lw=1.5)
    ax1.plot(t, indices['relaxation'], label='Relaxation', color='#9467bd', lw=1.5)
    ax1.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax1.set_ylabel('Index (−1 to +1)')
    ax1.set_title('Wellness Indices (§3.3)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_eeg_csv(path, n_header=4):
    """Load lilia-format CSV; returns (time_s, data[N, n_ch])."""
    df      = pd.read_csv(path, skiprows=n_header)
    time_s  = df.iloc[:, 0].values.astype(float) / 1e6
    data    = df.iloc[:, 1:].values.astype(float)
    return time_s, data


def _parse_args():
    p = argparse.ArgumentParser(
        description='Compute and plot qEEG wellness indices (Appendix J §3).')
    p.add_argument('--csv',  required=True, metavar='PATH',
                   help='Input CSV file (lilia format).')
    p.add_argument('--fs',   type=float, default=DEFAULT_FS, metavar='HZ',
                   help=f'Sampling frequency (default {DEFAULT_FS} Hz).')
    p.add_argument('--ch',   type=int,   default=1, metavar='N',
                   help='1-based channel index to analyse (default 1).')
    p.add_argument('--win',  type=float, default=DEFAULT_WIN_SEC, metavar='SEC',
                   help=f'Window length in seconds (default {DEFAULT_WIN_SEC} s).')
    p.add_argument('--out',  metavar='DIR',
                   help='Output directory (default: same dir as CSV).')
    return p.parse_args()


def main():
    args   = _parse_args()
    outdir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading: {args.csv}')
    time_s, data = _load_eeg_csv(args.csv)
    ch_idx = args.ch - 1
    if ch_idx >= data.shape[1]:
        sys.exit(f'Error: channel {args.ch} not found '
                 f'(file has {data.shape[1]} channels).')

    print(f'Computing qEEG indices — ch{args.ch}, win={args.win}s, fs={args.fs}Hz')
    indices = compute_qeeg_indices_windowed(
        data[:, ch_idx], fs=args.fs, win_sec=args.win)

    basename = os.path.splitext(os.path.basename(args.csv))[0]
    outpath  = os.path.join(outdir, f'{basename}_qeeg_ch{args.ch}.png')
    plot_qeeg_indices(
        indices,
        title=f'qEEG Wellness Indices — {os.path.basename(args.csv)} ch{args.ch}',
        outpath=outpath,
        t_offset=time_s[0])

    print('\nSummary (mean ± std):')
    for key in ('focus', 'flow', 'calm', 'relaxation'):
        v = indices[key]
        print(f'  {key:12s}: {v.mean():+.3f} ± {v.std():.3f}')


if __name__ == '__main__':
    main()
