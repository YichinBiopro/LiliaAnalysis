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
import json
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from lilia.io import read_lilia_frame
from lilia.windowing import plot_breaks

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILON    = 1e-9
BAND_THETA = (4.0,  8.0)
BAND_ALPHA = (8.0, 13.0)
BAND_BETA  = (13.0, 30.0)

DEFAULT_FS      = 500
DEFAULT_WIN_SEC = 5.0


# ── §3.1  Relative Power Normalisation ────────────────────────────────────────

def _band_power(freqs, psd_linear, fmin, fmax):
    # Half-open [fmin, fmax) so adjacent bands never share a boundary bin.
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapezoid(psd_linear[mask], freqs[mask]))


def compute_relative_powers(data_col, fs=DEFAULT_FS):
    """
    Compute relative θ / α / β powers for a 1-D EEG segment.
    Delta & Gamma are excluded per §3.1 to minimise motion/EMG artefacts.

    Returns
    -------
    (theta_rel, alpha_rel, beta_rel) : floats in [0, 1], summing to ~1.
    """
    data_col = np.asarray(data_col, dtype=float)
    if data_col.ndim != 1 or data_col.size < 8 or not np.all(np.isfinite(data_col)):
        raise ValueError('qEEG requires at least 8 finite samples in one channel')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('fs must be finite and positive')
    nperseg = min(len(data_col), max(1, int(round(fs * 4))))
    noverlap = min(int(round(fs * 2)), nperseg // 2)
    freqs, psd = signal.welch(data_col, fs=fs, nperseg=nperseg,
                               noverlap=noverlap, window='hann')
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
    if win_sec <= 0 or step_sec <= 0 or fs <= 0:
        raise ValueError('window, step and fs must be positive')
    win  = int(win_sec  * fs)
    step = int(step_sec * fs)
    if win < 1 or step < 1:
        raise ValueError('window and step must contain at least one sample')
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

def plot_qeeg_indices(indices, title, outpath, t_offset=0.0, marker=None,
                      segment_ids=None, time_label="Time (s)", time_limits=None):
    """
    Two-panel figure:
      top    — relative θ / α / β band powers over time.
      bottom — all four wellness indices over time.
    """
    t = indices['time'] + t_offset
    if segment_ids is not None:
        indices = dict(indices)
        for key in ('theta', 'alpha', 'beta', 'focus', 'flow', 'calm', 'relaxation'):
            _, indices[key] = plot_breaks(t, indices[key], segment_ids)
        t, _ = plot_breaks(t, np.zeros(len(t)), segment_ids)
        marker = '.' if marker is None else marker
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
    ax1.set_ylabel('Index')
    ax1.set_title('Wellness Indices (§3.3)')
    ax1.set_xlabel(time_label)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    if marker is not None:
        # A segment with only one metric window still needs a visible point.
        for line in [*ax0.lines, *ax1.lines[:4]]:
            line.set_marker(marker)

    if segment_ids is not None:
        # Keep isolated points visible even at the upper-right edge.
        for ax in axes:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)

    if time_limits is not None:
        ax1.set_xlim(*time_limits)
    if segment_ids is not None and not any(np.isfinite(indices[key]).any()
                                           for key in ('focus', 'flow', 'calm', 'relaxation')):
        for ax in axes:
            ax.text(.5, .5, 'No finite qEEG windows', transform=ax.transAxes,
                    ha='center', va='center', color='dimgray',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
        ax1.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {outpath}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_eeg_csv(path, n_header=4):
    """Load lilia-format CSV; returns (integer time_us, data[N, n_ch])."""
    if n_header != 4:
        raise ValueError('The Lilia CSV reader requires four metadata rows')
    df      = read_lilia_frame(path)
    time_us = df.iloc[:, 0].to_numpy(dtype=np.int64)
    data    = df.iloc[:, 1:].values.astype(float)
    return time_us, data


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
    from lilia.qeeg_raw import analyze_raw_qeeg, INDEX_KEYS
    from lilia.qeeg_io import write_qeeg_table
    from lilia.provenance import file_sha256
    from lilia.entropy_io import config_id

    args = _parse_args()
    outdir = Path(args.out or os.path.dirname(os.path.abspath(args.csv)))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(args.csv).stem}_qeeg_ch{args.ch}'
    audit_path = outdir / f'{stem}_analysis_audit.json'
    audit = {'schema_version': 1, 'kind': 'raw_qeeg', 'status': 'failed',
             'source_path': str(Path(args.csv).resolve()),
             'requested': {'fs': str(args.fs), 'win_sec': str(args.win), 'channel': args.ch},
             'quality_state': 'disabled', 'artifacts': {}}
    try:
        print(f'Loading: {args.csv}')
        audit['source_id'] = file_sha256(args.csv)
        time_us, data = _load_eeg_csv(args.csv)
        print(f'Computing qEEG indices — ch{args.ch}, win={args.win}s, fs={args.fs}Hz')
        result = analyze_raw_qeeg(time_us, data, fs=args.fs, win_sec=args.win, channel=args.ch)
        audit.update(parameters=result['parameters'], analysis=result['analysis'])
        audit['config_id'] = config_id(result['parameters'])
        audit['code_sha256'] = {name: file_sha256(Path(__file__).with_name(name))
                                for name in ('qeeg.py', 'qeeg_raw.py', 'qeeg_io.py', 'windowing.py',
                                             'io.py', 'entropy_io.py', 'provenance.py')}
        grid = result['grid']
        if grid is not None:
            table = outdir / f'{stem}.csv'
            write_qeeg_table(table, args.csv, result, config_id(audit['code_sha256']))
            for path in (table, Path(str(table) + '.meta.json')):
                audit['artifacts'][path.name] = file_sha256(path)
            outpath = outdir / f'{stem}.png'
            plot_qeeg_indices(
                {'time': grid.time_s, **result['metrics']},
                title=f'qEEG Wellness Indices — {Path(args.csv).name} ch{args.ch}',
                outpath=outpath, segment_ids=grid.columns['segment_id'],
                time_label='Elapsed time from first source sample (s)',
                time_limits=(0., (int(time_us[-1]) - int(time_us[0])) / 1e6 + 1 / args.fs))
            audit['artifacts'][outpath.name] = file_sha256(outpath)
        if result['analysis']['status'] != 'success':
            raise ValueError(result['analysis']['status'])
        print('\nSummary (mean ± std):')
        for key in INDEX_KEYS:
            stats = result['analysis']['summary'][key]
            print(f"  {key:12s}: {stats['mean']:+.3f} ± {stats['std']:.3f}")
        print(f"Windows: {result['analysis']['finite_windows']} finite / "
              f"{result['analysis']['candidate_windows']} candidates; quality=disabled")
        audit['status'] = 'success'
    except Exception as exc:
        audit['error'] = f'{type(exc).__name__}: {exc}'
        print(f'Error: {exc}', file=sys.stderr)
    finally:
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Saved audit: {audit_path}')
    if audit['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
