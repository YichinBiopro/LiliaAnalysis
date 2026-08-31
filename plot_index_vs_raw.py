"""
plot_index_vs_raw.py
====================
Plot the Focus and Relaxation qEEG index traces against the raw EEG signal,
under **two baseline definitions**:

    * "first-part"  — the first ``--baseline-sec`` seconds of the recording.
    * "pre-event"   — the ``--baseline-sec`` seconds immediately *before* an
                      event onset (``[onset − B, onset)``), i.e. the rest period
                      just before activity starts. Onsets come from the EVENTS
                      table in ``plot_event_markers.py``, aligned to the
                      recording-start timestamp.

The figure has three stacked panels sharing a time x-axis:

    row 0 — Focus index Δ-from-baseline (one curve per baseline) + faint absolute
    row 1 — Relaxation index Δ-from-baseline (one curve per baseline) + faint absolute
    row 2 — raw EEG amplitude (decimated for display)

Indices are recomputed in closed form from the per-window relative band powers
already stored in ``*_band_entropy_ch*.csv`` (no PSDs are re-estimated), exactly
like ``plot_focus_relax_anim.py``. The raw trace is read from the subject's
``merged.csv`` (lilia format).

CLI usage
---------
    python plot_index_vs_raw.py --subject Hsin [--ch 1] [--baseline-sec 30]
                                [--event "Single Cycling"] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')  # headless: render straight to a PNG file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lilia.qeeg import (
    focus_index, flow_index, calm_index, relaxation_index)
import plot_event_markers as pem


# Ordered (key, label, colour) for every signal we plot above the raw trace.
SIGNAL_SPECS = [
    ('focus',      'Focus',       '#1f77b4'),
    ('flow',       'Flow',        '#ff7f0e'),
    ('calm',       'Calm',        '#2ca02c'),
    ('relaxation', 'Relaxation',  '#9467bd'),
    ('entropy',    'Band entropy', '#8c564b'),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_signals(csv_path: str):
    """Read a band-entropy CSV → (time_s, signals dict, quality) for finite rows.

    *signals* holds the four qEEG wellness indices (computed in closed form from
    the stored relative band powers) plus the normalised band entropy.
    """
    df = pd.read_csv(csv_path)
    required = {'time_s', 'p_theta', 'p_alpha', 'p_beta'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f'Error: {csv_path} missing columns {sorted(missing)} — '
                 'pass a *_band_entropy_ch*.csv from spectral_entropy.py.')
    P = df[['p_theta', 'p_alpha', 'p_beta']].to_numpy(dtype=float)
    t = df['time_s'].to_numpy(dtype=float)
    q = (df['quality'].to_numpy(dtype=float) if 'quality' in df
         else np.full(t.shape, np.nan))
    ent_col = ('band_entropy_norm' if 'band_entropy_norm' in df
               else 'band_entropy' if 'band_entropy' in df else None)
    ent = (df[ent_col].to_numpy(dtype=float) if ent_col
           else np.full(t.shape, np.nan))
    # Keep every time-stamped window (incl. ones whose powers are NaN because
    # they failed quality at CSV-build time) so the quality mask is explicit:
    # sub-threshold windows become visible NaN gaps rather than silently
    # vanishing. Indices computed from NaN powers are naturally NaN.
    keep = np.isfinite(t)
    P, t, q, ent = P[keep], t[keep], q[keep], ent[keep]
    order = np.argsort(t)
    P, t, q, ent = P[order], t[order], q[order], ent[order]
    signals = {
        'focus':      np.array([focus_index(th, al, be) for th, al, be in P]),
        'flow':       np.array([flow_index(th, al, be) for th, al, be in P]),
        'calm':       np.array([calm_index(th, al, be) for th, al, be in P]),
        'relaxation': np.array([relaxation_index(th, al, be) for th, al, be in P]),
        'entropy':    ent,
    }
    return t, signals, q


def _load_raw(csv_path: str, ch: int, max_points: int = 15000):
    """Read merged.csv (lilia format) and return decimated (time_s, amplitude).

    time_s is relative to sample 0 so it shares the band-entropy time frame.
    Also returns the absolute µs epoch of sample 0 for event alignment.
    """
    df = pd.read_csv(csv_path, skiprows=4)
    t_us = df.iloc[:, 0].to_numpy(dtype=np.int64)
    if ch < 1 or ch >= df.shape[1]:
        sys.exit(f'Error: channel {ch} not in {csv_path} '
                 f'(file has {df.shape[1] - 1} channels).')
    amp = df.iloc[:, ch].to_numpy(dtype=float)
    epoch_us = int(t_us[0])
    t_s = (t_us - epoch_us) / 1e6
    stride = max(1, len(t_s) // max_points)
    return t_s[::stride], amp[::stride], epoch_us


# ── Event onsets ──────────────────────────────────────────────────────────────

def _subject_events(subject: str, epoch_us: int):
    """Return [(name, onset_rel_s), ...] for events applicable to *subject*."""
    out = []
    for name, hhmm, _dur, who in pem.EVENTS:
        if who is not None and subject not in who:
            continue
        onset_rel_s = (pem.hhmm_to_us(hhmm) - epoch_us) / 1e6
        out.append((name, onset_rel_s))
    return out


def _event_windows(subject: str, epoch_us: int,
                   pre_sec: float = 30.0, post_sec: float = 30.0):
    """Per-event pre-/post-onset windows for *subject*.

    For each event flag we take a ``post`` window ``[onset, onset+post_sec)`` and
    a ``pre`` (baseline) window ``[anchor-pre_sec, anchor)``. Normally
    ``anchor = onset``, but if that pre-window would fall *inside the preceding
    event* (the events run continuously, with no ≥pre_sec rest gap), we step the
    anchor back to the previous event's onset — repeating until a clean gap is
    found — so the baseline is real rest rather than the prior activity.

    Returns a list of dicts with name, onset, end, pre_lo/pre_hi, post_lo/post_hi,
    continuous (bool) and anchor_name (the event the pre-window was taken before).
    """
    evs = []
    for name, hhmm, dur, who in pem.EVENTS:
        if who is not None and subject not in who:
            continue
        onset = (pem.hhmm_to_us(hhmm) - epoch_us) / 1e6
        evs.append((name, onset, onset + dur * 60.0))
    evs.sort(key=lambda e: e[1])

    out = []
    for i, (name, onset, end) in enumerate(evs):
        anchor, anchor_name, continuous, j = onset, name, False, i
        while j > 0 and (anchor - pre_sec) < evs[j - 1][2]:
            continuous = True
            j -= 1
            anchor, anchor_name = evs[j][1], evs[j][0]
        out.append({
            'name': name, 'onset': onset, 'end': end,
            'pre_lo': anchor - pre_sec, 'pre_hi': anchor,
            'post_lo': onset, 'post_hi': onset + post_sec,
            'continuous': continuous, 'anchor_name': anchor_name,
        })
    return out


# ── Baseline correction ───────────────────────────────────────────────────────

def _baseline_mean(t, y, lo, hi):
    """Mean of y over the window [lo, hi); NaN if the window has no samples."""
    m = (t >= lo) & (t < hi)
    return float(np.nanmean(y[m])) if m.any() else np.nan


def _rolling_median(y, win):
    """Light centred rolling-median smoothing; win<=1 returns y unchanged."""
    if win <= 1:
        return y
    return (pd.Series(y).rolling(int(win), center=True, min_periods=1)
            .median().to_numpy())


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_index_vs_raw(
    be_csv: str,
    raw_csv: str,
    subject: str,
    ch: int,
    baseline_sec: float,
    event_name: str | None,
    outpath: str,
    use_minutes: bool = True,
    smooth_win: int = 5,
    raw_ylim: tuple[float, float] = (-150.0, 150.0),
    quality_threshold: float = pem.QUALITY_THRESHOLD,
) -> None:
    t, signals, q = _load_signals(be_csv)
    if t.size == 0:
        sys.exit(f'Error: no finite windows in {be_csv}.')
    t_raw, amp, epoch_us = _load_raw(raw_csv, ch)

    # Quality mask: windows with quality < threshold are dropped (→ NaN gaps)
    # from every signal panel and excluded from the baseline means.
    good = np.isfinite(q) & (q >= quality_threshold)
    n_bad = int((~good).sum())
    bad_pct = 100.0 * n_bad / t.size if t.size else 0.0

    win = _event_windows(subject, epoch_us, baseline_sec, baseline_sec)
    if not win:
        sys.exit(f'Error: no EVENTS apply to subject {subject!r}.')

    sc = 1.0 / 60.0 if use_minutes else 1.0
    xlabel = 'Time (min)' if use_minutes else 'Time (s)'

    n_sig = len(SIGNAL_SPECS)
    n_rows = n_sig + 1  # one panel per signal + the raw EEG panel
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.2 * n_rows + 1), sharex=True,
        gridspec_kw={'height_ratios': [1.6] * n_sig + [1.4]})
    fig.suptitle(
        f'{subject} ch{ch} — qEEG indices + band entropy vs raw EEG  '
        f'(each event period Δ vs its OWN pre-{baseline_sec:g}s [solid] / '
        f'post-{baseline_sec:g}s [dotted];  '
        f'quality≥{quality_threshold:g}, {bad_pct:.0f}% windows masked)',
        fontsize=12, fontweight='bold')

    for ax, (key, label, color) in zip(axes[:n_sig], SIGNAL_SPECS):
        # Apply the quality mask: bad windows become NaN gaps everywhere.
        y = np.where(good, signals[key], np.nan)
        ys = _rolling_median(y, smooth_win)        # light-smoothed signal
        ys = np.where(good, ys, np.nan)            # keep gaps after smoothing

        # Per-event Δ curves: over each event's period [onset, end] reference the
        # signal to THAT event's own pre-/post-onset window (NaN elsewhere, so the
        # piecewise segments break between events).
        dpre = np.full(t.shape, np.nan)
        dpost = np.full(t.shape, np.nan)
        for w in win:
            seg = (t >= w['onset']) & (t < w['end'])
            if not seg.any():
                continue
            pre_m = _baseline_mean(t, y, w['pre_lo'], w['pre_hi'])
            post_m = _baseline_mean(t, y, w['post_lo'], w['post_hi'])
            dpre[seg] = ys[seg] - pre_m
            dpost[seg] = ys[seg] - post_m

        # Faint raw (unsmoothed) absolute signal as a scatter-like background.
        ax.plot(t * sc, y, color='0.8', lw=0.6, alpha=0.5,
                label='absolute (raw)', zorder=1)
        # Per-event Δ vs own pre (solid) / own post (dotted).
        ax.plot(t * sc, dpre, color=color, lw=1.8, ls='-', alpha=0.95,
                label='Δ vs own pre-event (per event)', zorder=4)
        ax.plot(t * sc, dpost, color=color, lw=1.5, ls=':', alpha=0.95,
                label='Δ vs own post-event (per event)', zorder=4)
        ax.axhline(0.0, color='k', lw=0.8, ls=':', alpha=0.5)
        # Shade each event's pre (red) and post (orange) windows.
        for w in win:
            ax.axvspan(w['pre_lo'] * sc, w['pre_hi'] * sc,
                       color='#d62728', alpha=0.16, zorder=0)
            ax.axvspan(w['post_lo'] * sc, w['post_hi'] * sc,
                       color='#ff7f0e', alpha=0.16, zorder=0)
        ax.set_ylabel(f'{label}\n(value / Δ)')
        ax.legend(loc='upper right', fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    # Raw EEG panel.
    axr = axes[n_sig]
    axr.plot(t_raw * sc, amp, color='#444444', lw=0.4, alpha=0.8)
    axr.set_ylabel(f'Raw EEG ch{ch}\n(µV)')
    axr.set_xlabel(xlabel)
    axr.set_ylim(raw_ylim)
    axr.grid(True, alpha=0.3)

    # Shade quality-masked windows on the raw panel so the dropped (noisy)
    # segments are visible alongside the gaps in the signal panels above.
    half = float(np.median(np.diff(t))) / 2.0 if t.size > 1 else 1.0
    for tb in t[~good]:
        axr.axvspan((tb - half) * sc, (tb + half) * sc,
                    color='0.5', alpha=0.18, lw=0, zorder=0)

    # Event onset lines + labels across all panels (mark continuous events whose
    # pre-window was borrowed from an earlier rest with '*').
    for i, w in enumerate(win):
        col = pem.EVT_COLORS[i % len(pem.EVT_COLORS)]
        for ax in axes:
            ax.axvline(w['onset'] * sc, color=col, lw=1.0, ls='-', alpha=0.6,
                       zorder=2)
        axes[0].annotate(f"{w['name']}{' *' if w['continuous'] else ''}",
                         xy=(w['onset'] * sc, 1.0),
                         xycoords=('data', 'axes fraction'),
                         xytext=(2, -2), textcoords='offset points',
                         rotation=90, va='top', ha='left', fontsize=7,
                         color=col)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')
    print(f'  events          : {len(win)} '
          f"({sum(w['continuous'] for w in win)} continuous)")
    print(f'  windows         : {t.size}  over {t[0]:.0f}–{t[-1]:.0f}s')
    print(f'  quality mask    : {n_bad}/{t.size} masked '
          f'({bad_pct:.1f}%, threshold {quality_threshold:g})')


def _raw_epoch_us(raw_csv: str) -> int:
    """Cheaply read the absolute µs timestamp of sample 0 (for event alignment).

    Rows 0–3 are the lilia header, row 4 is the column-name row (Time[us],…),
    so the first data sample is row 5.
    """
    row = pd.read_csv(raw_csv, skiprows=5, nrows=1, header=None)
    return int(row.iloc[0, 0])


def _real_window_times(
    n_windows: int,
    raw_csv: str,
    win_sec: float,
    step_sec: float,
    fs: float,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Map analysis windows to true elapsed time from raw CSV timestamps."""
    time_us = pd.read_csv(raw_csv, skiprows=5, header=None,
                          usecols=[0]).iloc[:, 0].to_numpy(dtype=np.int64)
    epoch_us = int(time_us[0])
    win_samp = int(round(win_sec * fs))
    step_samp = int(round(step_sec * fs))
    centres = np.arange(n_windows) * step_samp + win_samp // 2
    centres = np.clip(centres, 0, time_us.size - 1)
    real_t = (time_us[centres].astype(np.float64) - epoch_us) / 1e6

    gaps = np.diff(real_t)
    jump = np.flatnonzero(gaps > gap_factor * step_sec)
    for i in jump:
        real_t[i + 1] = np.nan
    return real_t


def plot_index_vs_raw_session(
    be_csv: str,
    raw_csv: str,
    label: str,
    ch: int,
    outpath: str,
    use_minutes: bool = True,
    smooth_win: int = 5,
    raw_ylim: tuple[float, float] = (-150.0, 150.0),
    quality_threshold: float = pem.QUALITY_THRESHOLD,
    win_sec: float = 2.0,
    step_sec: float = 2.0,
    fs: float = 500.0,
) -> None:
    """Session-baseline variant for data without event timelines."""
    t, signals, q = _load_signals(be_csv)
    if t.size == 0:
        raise SystemExit(f'Error: no finite windows in {be_csv}.')
    t_raw, amp, _epoch_us = _load_raw(raw_csv, ch)

    t_real = _real_window_times(t.size, raw_csv, win_sec, step_sec, fs)
    gap_mask = np.isnan(t_real)
    if gap_mask.any():
        print(f'  time-gap correction: {int(gap_mask.sum())} window(s) '
              f'follow a gap > {3 * step_sec:g}s — line broken there.')

    good = np.isfinite(q) & (q >= quality_threshold)
    n_bad = int((~good).sum())
    bad_pct = 100.0 * n_bad / t.size if t.size else 0.0

    sc = 1.0 / 60.0 if use_minutes else 1.0
    xlabel = 'Time (min)' if use_minutes else 'Time (s)'

    n_sig = len(SIGNAL_SPECS)
    n_rows = n_sig + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.2 * n_rows + 1), sharex=True,
        gridspec_kw={'height_ratios': [1.6] * n_sig + [1.4]})
    fig.suptitle(
        f'{label} ch{ch} — qEEG indices + band entropy vs raw EEG  '
        f'(Δ vs whole-session baseline, no activity timeline;  '
        f'quality≥{quality_threshold:g}, {bad_pct:.0f}% windows masked)',
        fontsize=12, fontweight='bold')

    baseline_rows = []
    for ax, (key, sig_label, color) in zip(axes[:n_sig], SIGNAL_SPECS):
        y = np.where(good, signals[key], np.nan)
        ys = _rolling_median(y, smooth_win)
        ys = np.where(good, ys, np.nan)

        baseline = float(np.nanmean(y)) if np.any(np.isfinite(y)) else np.nan
        delta = ys - baseline

        ax.plot(t_real * sc, y, color='0.8', lw=0.6, alpha=0.5,
                label='absolute (raw)', zorder=1)
        ax.plot(t_real * sc, delta, color=color, lw=1.8, alpha=0.95,
                label='Δ vs session baseline', zorder=4)
        ax.axhline(0.0, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.set_ylabel(f'{sig_label}\n(value / Δ)')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        baseline_rows.append((key, baseline))

    axr = axes[n_sig]
    axr.plot(t_raw * sc, amp, color='#444444', lw=0.4, alpha=0.8)
    axr.set_ylabel(f'Raw EEG ch{ch}\n(µV)')
    axr.set_xlabel(xlabel)
    axr.set_ylim(raw_ylim)
    axr.grid(True, alpha=0.3)

    half = step_sec / 2.0
    for tb in t_real[~good & ~gap_mask]:
        axr.axvspan((tb - half) * sc, (tb + half) * sc,
                    color='0.5', alpha=0.18, lw=0, zorder=0)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')
    valid_t = t_real[~gap_mask]
    print(f'  windows       : {t.size}  over real elapsed '
          f'{np.nanmin(valid_t):.0f}–{np.nanmax(valid_t):.0f}s '
          f'(sample-count time would read 0–{t[-1]:.0f}s, compressing out gaps)')
    print(f'  quality mask  : {n_bad}/{t.size} masked '
          f'({bad_pct:.1f}%, threshold {quality_threshold:g})')
    print('  session baseline (quality-masked whole-recording mean):')
    for key, baseline in baseline_rows:
        print(f'    {key:<11s}: {baseline:.4f}')


def plot_absolute_waves(
    be_csv: str,
    raw_csv: str,
    subject: str,
    ch: int,
    outpath: str,
    use_minutes: bool = True,
    smooth_win: int = 5,
    quality_threshold: float = pem.QUALITY_THRESHOLD,
    pre_sec: float = 30.0,
    post_sec: float = 30.0,
    summary_csv: str | None = None,
) -> None:
    """One panel per signal: the *absolute* (un-baselined) qEEG indices and
    band entropy through time, with the same quality mask + smoothing.

    Every event flag gets a shaded pre-onset (green) and post-onset (orange)
    window; per-event pre/post/Δ means are written to *summary_csv*."""
    t, signals, q = _load_signals(be_csv)
    if t.size == 0:
        sys.exit(f'Error: no finite windows in {be_csv}.')
    good = np.isfinite(q) & (q >= quality_threshold)
    bad_pct = 100.0 * int((~good).sum()) / t.size if t.size else 0.0
    epoch_us = _raw_epoch_us(raw_csv)
    win = _event_windows(subject, epoch_us, pre_sec, post_sec)

    sc = 1.0 / 60.0 if use_minutes else 1.0
    xlabel = 'Time (min)' if use_minutes else 'Time (s)'

    # Mask bad windows once, up front, for both plotting and the pre/post means.
    masked = {key: np.where(good, signals[key], np.nan) for key, _, _ in
              SIGNAL_SPECS}

    n_sig = len(SIGNAL_SPECS)
    fig, axes = plt.subplots(n_sig, 1, figsize=(14, 2.0 * n_sig + 1),
                             sharex=True)
    fig.suptitle(
        f'{subject} ch{ch} — absolute qEEG indices + band entropy through time '
        f'(quality≥{quality_threshold:g}, {bad_pct:.0f}% masked;  '
        f'per-event pre={pre_sec:g}s / post={post_sec:g}s)',
        fontsize=13, fontweight='bold')

    half = float(np.median(np.diff(t))) / 2.0 if t.size > 1 else 1.0
    for ax, (key, label, color) in zip(axes, SIGNAL_SPECS):
        is_entropy = key == 'entropy'
        ys = np.where(good, _rolling_median(masked[key], smooth_win), np.nan)
        ax.plot(t * sc, ys, color=color, lw=1.6, alpha=0.95, zorder=4)
        ax.axhline(0.0, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.set_ylabel(label)
        ax.set_ylim((0.0, 1.0) if is_entropy else (-1.1, 1.1))
        ax.grid(True, alpha=0.3)
        # Quality-masked windows.
        for tb in t[~good]:
            ax.axvspan((tb - half) * sc, (tb + half) * sc,
                       color='0.5', alpha=0.15, lw=0, zorder=0)
        # Per-event pre (green) / post (orange) windows.
        for w in win:
            ax.axvspan(w['pre_lo'] * sc, w['pre_hi'] * sc,
                       color='#2ca02c', alpha=0.18, lw=0, zorder=1)
            ax.axvspan(w['post_lo'] * sc, w['post_hi'] * sc,
                       color='#ff7f0e', alpha=0.18, lw=0, zorder=1)

    axes[-1].set_xlabel(xlabel)

    # Event onset lines + labels (mark continuous events whose pre-window was
    # borrowed from an earlier rest).
    for w in win:
        for ax in axes:
            ax.axvline(w['onset'] * sc, color='k', lw=1.0, ls='-', alpha=0.4,
                       zorder=2)
        tag = f"{w['name']}{' *' if w['continuous'] else ''}"
        axes[0].annotate(tag, xy=(w['onset'] * sc, 1.0),
                         xycoords=('data', 'axes fraction'),
                         xytext=(2, -2), textcoords='offset points',
                         rotation=90, va='top', ha='left', fontsize=7,
                         color='k')

    # Per-event pre/post/Δ summary for every signal.
    rows = []
    for w in win:
        pre_m = (t >= w['pre_lo']) & (t < w['pre_hi'])
        post_m = (t >= w['post_lo']) & (t < w['post_hi'])
        row = {'event': w['name'],
               'onset_s': round(w['onset'], 1),
               'continuous': w['continuous'],
               'pre_from': w['anchor_name'],
               'pre_window_s': f"[{w['pre_lo']:.0f},{w['pre_hi']:.0f})",
               'post_window_s': f"[{w['post_lo']:.0f},{w['post_hi']:.0f})"}
        for key, label, _ in SIGNAL_SPECS:
            pre_v = float(np.nanmean(masked[key][pre_m])) if pre_m.any() else np.nan
            post_v = (float(np.nanmean(masked[key][post_m]))
                      if post_m.any() else np.nan)
            row[f'{key}_pre'] = round(pre_v, 4)
            row[f'{key}_post'] = round(post_v, 4)
            row[f'{key}_delta'] = round(post_v - pre_v, 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary_csv:
        summary.to_csv(summary_csv, index=False)
        print(f'Saved: {summary_csv}')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_single_signal(
    be_csv: str,
    raw_csv: str,
    subject: str,
    ch: int,
    key: str,
    label: str,
    color: str,
    outpath: str,
    baseline_sec: float = 30.0,
    use_minutes: bool = True,
    smooth_win: int = 5,
    raw_ylim: tuple[float, float] = (-150.0, 150.0),
    quality_threshold: float = pem.QUALITY_THRESHOLD,
) -> None:
    """Two-panel figure for one signal: Δ vs own pre/post per event (top) + raw EEG."""
    t, signals, q = _load_signals(be_csv)
    t_raw, amp, epoch_us = _load_raw(raw_csv, ch)
    good = np.isfinite(q) & (q >= quality_threshold)
    n_bad = int((~good).sum())
    bad_pct = 100.0 * n_bad / t.size if t.size else 0.0
    win = _event_windows(subject, epoch_us, baseline_sec, baseline_sec)

    sc = 1.0 / 60.0 if use_minutes else 1.0
    xlabel = 'Time (min)' if use_minutes else 'Time (s)'
    is_entropy = key == 'entropy'

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                  gridspec_kw={'height_ratios': [2.5, 1.2]})
    fig.suptitle(
        f'{subject} ch{ch} — {label}  '
        f'(each event: Δ vs own pre-{baseline_sec:g}s [solid] / '
        f'post-{baseline_sec:g}s [dotted];  '
        f'quality≥{quality_threshold:g}, {bad_pct:.0f}% masked)',
        fontsize=12, fontweight='bold')

    y = np.where(good, signals[key], np.nan)
    ys = np.where(good, _rolling_median(y, smooth_win), np.nan)
    dpre = np.full(t.shape, np.nan)
    dpost = np.full(t.shape, np.nan)
    for w in win:
        seg = (t >= w['onset']) & (t < w['end'])
        if not seg.any():
            continue
        dpre[seg] = ys[seg] - _baseline_mean(t, y, w['pre_lo'], w['pre_hi'])
        dpost[seg] = ys[seg] - _baseline_mean(t, y, w['post_lo'], w['post_hi'])

    ax.plot(t * sc, y, color='0.8', lw=0.6, alpha=0.5, label='absolute (raw)',
            zorder=1)
    ax.plot(t * sc, dpre, color=color, lw=1.8, ls='-', alpha=0.95,
            label=f'Δ vs own pre-{baseline_sec:g}s', zorder=4)
    ax.plot(t * sc, dpost, color=color, lw=1.5, ls=':', alpha=0.95,
            label=f'Δ vs own post-{baseline_sec:g}s', zorder=4)
    ax.axhline(0.0, color='k', lw=0.8, ls=':', alpha=0.5)
    ax.set_ylim((0.0, 1.0) if is_entropy else (-1.1, 1.1))
    ax.set_ylabel(f'{label}\n(Δ)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    for w in win:
        ax.axvspan(w['pre_lo'] * sc, w['pre_hi'] * sc,
                   color='#d62728', alpha=0.15, lw=0, zorder=0)
        ax.axvspan(w['post_lo'] * sc, w['post_hi'] * sc,
                   color='#ff7f0e', alpha=0.15, lw=0, zorder=0)

    # Raw EEG.
    half = float(np.median(np.diff(t))) / 2.0 if t.size > 1 else 1.0
    axr.plot(t_raw * sc, amp, color='#444444', lw=0.4, alpha=0.8)
    axr.set_ylim(raw_ylim)
    axr.set_ylabel(f'Raw EEG ch{ch}\n(µV)')
    axr.set_xlabel(xlabel)
    axr.grid(True, alpha=0.3)
    for tb in t[~good]:
        axr.axvspan((tb - half) * sc, (tb + half) * sc,
                    color='0.5', alpha=0.18, lw=0, zorder=0)

    # Event lines + labels on both panels.
    for i, w in enumerate(win):
        col = pem.EVT_COLORS[i % len(pem.EVT_COLORS)]
        for a in (ax, axr):
            a.axvline(w['onset'] * sc, color=col, lw=1.0, ls='-', alpha=0.6,
                      zorder=2)
        ax.annotate(f"{w['name']}{' *' if w['continuous'] else ''}",
                    xy=(w['onset'] * sc, 1.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(2, -2), textcoords='offset points',
                    rotation=90, va='top', ha='left', fontsize=7, color=col)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'Saved: {outpath}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Focus/Relax index (two baselines) vs raw EEG.')
    p.add_argument('--subject', required=False,
                   help=f'Subject key, one of {list(pem.SUBJECTS)}.')
    p.add_argument('--ch', type=int, default=1, help='1-based channel (default 1).')
    p.add_argument('--baseline-sec', type=float, default=30.0,
                   help='Baseline window length in seconds (default 30).')
    p.add_argument('--event', default=None,
                   help='Event name for the pre-event baseline '
                        '(default: first event applicable to the subject).')
    p.add_argument('--be-csv', default=None,
                   help='Band-entropy CSV (default: '
                        '<subject dir>/merged_band_entropy_ch<ch>.csv).')
    p.add_argument('--raw-csv', default=None,
                   help='Raw merged.csv (default: <subject dir>/merged.csv).')
    p.add_argument('--out', default=None, help='Output PNG path.')
    p.add_argument('--seconds', action='store_true',
                   help='Use seconds on the x-axis instead of minutes.')
    p.add_argument('--smooth-win', type=int, default=5,
                   help='Rolling-median window in #windows, odd (default 5; '
                        '1 = no smoothing).')
    p.add_argument('--raw-ylim', type=float, nargs=2, default=(-150.0, 150.0),
                   metavar=('LO', 'HI'),
                   help='Raw EEG panel y-limits in µV (default -150 150).')
    p.add_argument('--quality-threshold', type=float,
                   default=pem.QUALITY_THRESHOLD,
                   help='Mask windows with quality below this '
                        f'(default {pem.QUALITY_THRESHOLD}); use -1 to disable.')
    p.add_argument('--session-baseline', action='store_true',
                   help='Use one whole-session baseline (no event timeline).')
    p.add_argument('--label', default=None,
                   help='Label for titles in --session-baseline mode.')
    p.add_argument('--win-sec', type=float, default=2.0,
                   help='Analysis window length used to build band-entropy CSV '
                        '(session mode; default 2.0).')
    p.add_argument('--step-sec', type=float, default=2.0,
                   help='Analysis step used to build band-entropy CSV '
                        '(session mode; default 2.0).')
    p.add_argument('--fs', type=float, default=500.0,
                   help='Sampling rate in Hz (session mode; default 500).')
    p.add_argument('--pre-sec', type=float, default=30.0,
                   help='Per-event pre-onset window length in seconds (default 30).')
    p.add_argument('--post-sec', type=float, default=30.0,
                   help='Per-event post-onset window length in seconds (default 30).')
    p.add_argument('--split', action='store_true',
                   help='Also write one independent figure per signal '
                        '(<out_stem>_<key>.png).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.session_baseline:
        if not args.be_csv or not args.raw_csv:
            sys.exit('Error: --session-baseline requires both --be-csv and --raw-csv.')
        for f in (args.be_csv, args.raw_csv):
            if not os.path.isfile(f):
                sys.exit(f'Error: file not found: {f}')
        out = args.out or args.be_csv.replace('.csv', '_session_index_vs_raw.png')
        label = args.label or args.subject or os.path.splitext(os.path.basename(args.be_csv))[0]
        plot_index_vs_raw_session(
            args.be_csv, args.raw_csv, label, args.ch, out,
            win_sec=args.win_sec, step_sec=args.step_sec, fs=args.fs,
            use_minutes=not args.seconds, smooth_win=args.smooth_win,
            raw_ylim=tuple(args.raw_ylim),
            quality_threshold=args.quality_threshold,
        )
        return

    if args.subject not in pem.SUBJECTS:
        sys.exit(f'Error: unknown subject {args.subject!r}. '
                 f'Choose from {list(pem.SUBJECTS)}.')
    sub_dir = os.path.join('iBrainCenter', pem.SUBJECTS[args.subject]['dir'])
    be_csv = args.be_csv or os.path.join(
        sub_dir, f'merged_band_entropy_ch{args.ch}.csv')
    raw_csv = args.raw_csv or os.path.join(sub_dir, 'merged.csv')
    for f in (be_csv, raw_csv):
        if not os.path.isfile(f):
            sys.exit(f'Error: file not found: {f}')
    out = args.out or os.path.join(
        sub_dir, f'index_vs_raw_ch{args.ch}.png')
    plot_index_vs_raw(
        be_csv, raw_csv, args.subject, args.ch, args.baseline_sec,
        args.event, out, use_minutes=not args.seconds,
        smooth_win=args.smooth_win, raw_ylim=tuple(args.raw_ylim),
        quality_threshold=args.quality_threshold)

    abs_out = out.replace('.png', '_absolute.png')
    if abs_out == out:
        abs_out = out + '_absolute.png'
    peri_csv = abs_out.replace('.png', '_peri_event.csv')
    plot_absolute_waves(
        be_csv, raw_csv, args.subject, args.ch, abs_out,
        use_minutes=not args.seconds, smooth_win=args.smooth_win,
        quality_threshold=args.quality_threshold,
        pre_sec=args.pre_sec, post_sec=args.post_sec, summary_csv=peri_csv)

    if args.split:
        stem = out.replace('.png', '')
        for key, label, color in SIGNAL_SPECS:
            plot_single_signal(
                be_csv, raw_csv, args.subject, args.ch, key, label, color,
                outpath=f'{stem}_{key}.png',
                baseline_sec=args.baseline_sec,
                use_minutes=not args.seconds, smooth_win=args.smooth_win,
                raw_ylim=tuple(args.raw_ylim),
                quality_threshold=args.quality_threshold)


if __name__ == '__main__':
    main()
