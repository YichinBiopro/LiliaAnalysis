#!/usr/bin/env python3
"""Plot windowed Goertzel power + EEG quality against EEG waveforms.

Designed for iBrainCenter/*/merged.csv and aligned with existing project flow:
- raw loading/bandpass from lilia.io
- EEG quality v2 from lilia.quality (iBrain device preset)
- event overlays from plot_event_markers (when subject is known)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lilia.windowing import window_starts
from lilia.goertzel import goertzel_power
from lilia.provenance import write_goertzel_metadata, validate_goertzel_cache
from lilia.quality_policy import valid_goertzel_rows
from lilia.io import bandpass_filter, load_merged_csv
from lilia.quality import (
    get_eeg_quality_index_v2_parametric,
    get_ibrain_device_eeg_quality_v2_params,
)
from lilia.subject_paths import iter_subject_dirs
import plot_event_markers as pem


@dataclass
class WindowResult:
    window_start_idx: np.ndarray
    window_end_idx: np.ndarray
    time_s: np.ndarray
    goertzel_power: np.ndarray
    goertzel_db: np.ndarray
    quality: np.ndarray
    quality_final: np.ndarray
    sat_frac_1950: np.ndarray
    peak_to_peak_uv: np.ndarray
    max_abs_diff_uv: np.ndarray
    bp_edge_shift_uv: np.ndarray
    hard_artifact: np.ndarray


def _rolling_median(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.copy()
    return (
        pd.Series(y).rolling(int(win), center=True, min_periods=1).median().to_numpy()
    )


def _compute_window_metrics(
    time_us: np.ndarray,
    data_raw: np.ndarray,
    data_bp: np.ndarray,
    fs: float,
    ch: int,
    target_freq: float,
    win_sec: float,
    step_sec: float,
    quality_params: dict[str, float],
    sat_uv: float,
    sat_frac_threshold: float,
    step_ptp_threshold: float,
    bp_shift_sec: float,
    bp_shift_threshold: float,
) -> WindowResult:
    ch_idx = ch - 1
    win = max(int(round(win_sec * fs)), 1)
    step = max(int(round(step_sec * fs)), 1)

    t_mid = []
    win_start_idx = []
    win_end_idx = []
    pwr = []
    q = []
    sat_frac = []
    ptp = []
    max_abs_diff = []
    bp_edge_shift = []
    hard = []

    for start in window_starts(len(data_bp), win, step, time_us, fs):
        end = start + win
        seg = data_bp[start:start + win]
        seg_raw = data_raw[start:start + win]
        seg_ch = seg[:, ch_idx]
        seg_ch_raw = seg_raw[:, ch_idx]
        mid_us = int(time_us[start + win // 2])

        g = goertzel_power(
            seg_ch,
            target_freq=target_freq,
            sample_rate=fs,
            remove_dc=True,
            apply_hann=True,
        )
        quality = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64),
            fs=int(fs),
            params=quality_params,
        )['overall'][ch_idx]

        # Hard-artifact features are measured on the RAW signal so clipping /
        # step artifacts are not attenuated by bandpass filtering.
        sfrac = float(np.mean(np.abs(seg_ch_raw) >= sat_uv))
        p2p = float(np.ptp(seg_ch_raw))
        mdiff = (float(np.max(np.abs(np.diff(seg_ch_raw))))
                 if seg_ch_raw.size > 1 else 0.0)
        edge_n = max(1, int(round(bp_shift_sec * fs)))
        edge_n = min(edge_n, max(1, seg_ch.size // 2))
        bp_shift = float(abs(np.median(seg_ch[:edge_n]) - np.median(seg_ch[-edge_n:])))

        is_hard = (
            (sfrac >= sat_frac_threshold)
            or (p2p >= step_ptp_threshold)
            or (bp_shift >= bp_shift_threshold)
        )

        t_mid.append(mid_us)
        win_start_idx.append(start)
        win_end_idx.append(end)
        pwr.append(float(g))
        q.append(float(quality))
        sat_frac.append(sfrac)
        ptp.append(p2p)
        max_abs_diff.append(mdiff)
        bp_edge_shift.append(bp_shift)
        hard.append(bool(is_hard))

    t_mid = np.asarray(t_mid, dtype=np.int64)
    win_start_idx = np.asarray(win_start_idx, dtype=np.int64)
    win_end_idx = np.asarray(win_end_idx, dtype=np.int64)
    pwr = np.asarray(pwr, dtype=float)
    q = np.asarray(q, dtype=float)
    sat_frac = np.asarray(sat_frac, dtype=float)
    ptp = np.asarray(ptp, dtype=float)
    max_abs_diff = np.asarray(max_abs_diff, dtype=float)
    bp_edge_shift = np.asarray(bp_edge_shift, dtype=float)
    hard = np.asarray(hard, dtype=bool)

    time_s = (t_mid - int(time_us[0])) / 1e6
    pwr_db = 10.0 * np.log10(np.maximum(pwr, 1e-12))
    quality_final = np.where(hard, 0.0, q)
    return WindowResult(
        window_start_idx=win_start_idx,
        window_end_idx=win_end_idx,
        time_s=time_s,
        goertzel_power=pwr,
        goertzel_db=pwr_db,
        quality=q,
        quality_final=quality_final,
        sat_frac_1950=sat_frac,
        peak_to_peak_uv=ptp,
        max_abs_diff_uv=max_abs_diff,
        bp_edge_shift_uv=bp_edge_shift,
        hard_artifact=hard,
    )


def _subject_key_from_dir(dirname: str) -> str | None:
    for key, info in pem.SUBJECTS.items():
        if info['dir'] == dirname:
            return key
    return None


def _subject_events(subject_key: str | None, epoch_us: int):
    if subject_key is None:
        return []
    out = []
    for name, hhmm, _dur, who in pem.EVENTS:
        if who is not None and subject_key not in who:
            continue
        onset_rel_s = (pem.hhmm_to_us(hhmm) - epoch_us) / 1e6
        out.append((name, onset_rel_s))
    return out


def _load_raw_decimated(time_us: np.ndarray, raw_ch: np.ndarray, max_points: int = 15000):
    t_s = (time_us - int(time_us[0])) / 1e6
    stride = max(1, len(t_s) // max_points)
    return t_s[::stride], raw_ch[::stride]


def _plot_subject(
    merged_csv: str,
    out_png: str,
    out_csv: str,
    ch: int,
    fs: float,
    target_freq: float,
    win_sec: float,
    step_sec: float,
    smooth_win: int,
    quality_threshold: float,
    raw_ylim: tuple[float, float],
    exclude_hard_artifact: bool,
    sat_uv: float,
    sat_frac_threshold: float,
    step_ptp_threshold: float,
    bp_shift_sec: float,
    bp_shift_threshold: float,
    use_db: bool = True,
    ref_lines: list[tuple[float, str]] | None = None,
    power_ylim: tuple[float, float] | None = None,
) -> None:
    time_us, data = load_merged_csv(merged_csv)
    if ch < 1 or ch > data.shape[1]:
        raise ValueError(f'channel {ch} out of range for {merged_csv}')

    data_bp = bandpass_filter(data, fs=fs, lo=pem.BP_LOW, hi=pem.BP_HIGH, time_us=time_us)
    quality_params = get_ibrain_device_eeg_quality_v2_params()
    result = _compute_window_metrics(
        time_us=time_us,
        data_raw=data,
        data_bp=data_bp,
        fs=fs,
        ch=ch,
        target_freq=target_freq,
        win_sec=win_sec,
        step_sec=step_sec,
        quality_params=quality_params,
        sat_uv=sat_uv,
        sat_frac_threshold=sat_frac_threshold,
        step_ptp_threshold=step_ptp_threshold,
        bp_shift_sec=bp_shift_sec,
        bp_shift_threshold=bp_shift_threshold,
    )

    # Save per-window metrics for downstream analysis.
    pd.DataFrame(
        {
            'window_start_idx': result.window_start_idx,
            'window_end_idx': result.window_end_idx,
            'time_s': result.time_s,
            'window_start_us': time_us[result.window_start_idx],
            'window_end_us': time_us[result.window_end_idx - 1] + int(round(1e6 / fs)),
            'window_center_us': time_us[(result.window_start_idx + result.window_end_idx) // 2],
            'goertzel_power': result.goertzel_power,
            'goertzel_db': result.goertzel_db,
            'quality': result.quality,
            'quality_final': result.quality_final,
            'sat_frac_1950': result.sat_frac_1950,
            'peak_to_peak_uv': result.peak_to_peak_uv,
            'max_abs_diff_uv': result.max_abs_diff_uv,
            'bp_edge_shift_uv': result.bp_edge_shift_uv,
            'artifact_hard_clip': result.hard_artifact.astype(int),
        }
    ).to_csv(out_csv, index=False)
    write_goertzel_metadata(out_csv, merged_csv, {
        'ch': ch, 'fs': fs, 'target_freq': target_freq,
        'win_sec': win_sec, 'step_sec': step_sec,
        'sat_uv': sat_uv, 'sat_frac_threshold': sat_frac_threshold,
        'step_ptp_threshold': step_ptp_threshold,
        'bp_shift_sec': bp_shift_sec, 'bp_shift_threshold': bp_shift_threshold,
        'bp_low': pem.BP_LOW, 'bp_high': pem.BP_HIGH,
        'quality_params': quality_params,
    })

    sub_dirname = os.path.basename(os.path.dirname(merged_csv))
    subject_key = _subject_key_from_dir(sub_dirname)
    events = _subject_events(subject_key, int(time_us[0]))

    t_raw, y_raw = _load_raw_decimated(time_us, data[:, ch - 1])
    t_bp, y_bp = _load_raw_decimated(time_us, data_bp[:, ch - 1])

    _render_plot(
        out_png=out_png,
        result=result,
        t_raw=t_raw, y_raw=y_raw,
        t_bp=t_bp, y_bp=y_bp,
        sub_dirname=sub_dirname,
        ch=ch,
        target_freq=target_freq,
        smooth_win=smooth_win,
        quality_threshold=quality_threshold,
        raw_ylim=raw_ylim,
        exclude_hard_artifact=exclude_hard_artifact,
        events=events,
        use_db=use_db,
        ref_lines=ref_lines,
        power_ylim=power_ylim,
    )


def _plot_subject_from_csv(
    merged_csv: str,
    in_csv: str,
    out_png: str,
    ch: int,
    fs: float,
    target_freq: float,
    smooth_win: int,
    quality_threshold: float,
    raw_ylim: tuple[float, float],
    exclude_hard_artifact: bool,
    use_db: bool,
    ref_lines: list[tuple[float, str]] | None = None,
    power_ylim: tuple[float, float] | None = None,
) -> None:
    """Re-render the 4-panel figure from an already-computed per-window CSV
    (as saved by _plot_subject), skipping the expensive Goertzel/quality
    recomputation. Only the raw/bandpass EEG traces are reloaded."""
    validate_goertzel_cache(in_csv, merged_csv, {'ch': ch, 'fs': fs, 'target_freq': target_freq})
    time_us, data = load_merged_csv(merged_csv)
    if ch < 1 or ch > data.shape[1]:
        raise ValueError(f'channel {ch} out of range for {merged_csv}')
    data_bp = bandpass_filter(data, fs=fs, lo=pem.BP_LOW, hi=pem.BP_HIGH, time_us=time_us)

    df = pd.read_csv(in_csv)
    result = WindowResult(
        window_start_idx=df['window_start_idx'].to_numpy(),
        window_end_idx=df['window_end_idx'].to_numpy(),
        time_s=df['time_s'].to_numpy(),
        goertzel_power=df['goertzel_power'].to_numpy(),
        goertzel_db=df['goertzel_db'].to_numpy(),
        quality=df['quality'].to_numpy(),
        quality_final=df['quality_final'].to_numpy(),
        sat_frac_1950=df['sat_frac_1950'].to_numpy(),
        peak_to_peak_uv=df['peak_to_peak_uv'].to_numpy(),
        max_abs_diff_uv=df['max_abs_diff_uv'].to_numpy(),
        bp_edge_shift_uv=df['bp_edge_shift_uv'].to_numpy(),
        hard_artifact=df['artifact_hard_clip'].to_numpy().astype(bool),
    )

    sub_dirname = os.path.basename(os.path.dirname(merged_csv))
    subject_key = _subject_key_from_dir(sub_dirname)
    events = _subject_events(subject_key, int(time_us[0]))

    t_raw, y_raw = _load_raw_decimated(time_us, data[:, ch - 1])
    t_bp, y_bp = _load_raw_decimated(time_us, data_bp[:, ch - 1])

    _render_plot(
        out_png=out_png,
        result=result,
        t_raw=t_raw, y_raw=y_raw,
        t_bp=t_bp, y_bp=y_bp,
        sub_dirname=sub_dirname,
        ch=ch,
        target_freq=target_freq,
        smooth_win=smooth_win,
        quality_threshold=quality_threshold,
        raw_ylim=raw_ylim,
        exclude_hard_artifact=exclude_hard_artifact,
        events=events,
        use_db=use_db,
        ref_lines=ref_lines,
        power_ylim=power_ylim,
    )


def _render_plot(
    out_png: str,
    result: WindowResult,
    t_raw: np.ndarray, y_raw: np.ndarray,
    t_bp: np.ndarray, y_bp: np.ndarray,
    sub_dirname: str,
    ch: int,
    target_freq: float,
    smooth_win: int,
    quality_threshold: float,
    raw_ylim: tuple[float, float],
    exclude_hard_artifact: bool,
    events: list,
    use_db: bool = True,
    ref_lines: list[tuple[float, str]] | None = None,
    power_ylim: tuple[float, float] | None = None,
) -> None:
    good = valid_goertzel_rows(pd.DataFrame({
        'quality': result.quality, 'quality_final': result.quality_final,
        'goertzel_db': result.goertzel_db, 'time_s': result.time_s,
        'artifact_hard_clip': result.hard_artifact,
    }), quality_threshold, exclude_hard=exclude_hard_artifact)

    series = result.goertzel_db if use_db else result.goertzel_power
    masked_series = np.where(good, series, np.nan)
    smooth_series = np.where(good, _rolling_median(masked_series, smooth_win), np.nan)
    unit_label = 'Power (dB)' if use_db else 'Power (linear, a.u.)'
    raw_label = 'absolute (raw dB)' if use_db else 'absolute (raw, linear)'
    smooth_label = ('quality-masked + smoothed dB' if use_db
                     else 'quality-masked + smoothed (linear)')

    bad_pct = 100.0 * int((~good).sum()) / max(len(good), 1)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9.8),
        sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0, 1.1, 1.1]},
    )
    hard_n = int(result.hard_artifact.sum())
    fig.suptitle(
        f'{sub_dirname} ch{ch}  |  Goertzel {target_freq:g} Hz + quality + BP(0.5-45Hz)/raw '
        f'(quality>{quality_threshold:g}, {bad_pct:.0f}% masked; hard-artifact={hard_n})',
        fontsize=12,
        fontweight='bold',
    )

    # Panel 1: Goertzel power (dB or linear)
    ax0 = axes[0]
    ax0.plot(result.time_s / 60.0, series, color='0.82', lw=0.8,
             label=raw_label)
    ax0.plot(result.time_s / 60.0, smooth_series, color='#1f77b4', lw=1.8,
             label=smooth_label)
    ax0.set_ylabel(f'Goertzel {target_freq:g} Hz\n{unit_label}')
    if not use_db:
        ax0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for ref_val, ref_label in (ref_lines or []):
        ax0.axhline(ref_val, color='#d62728', ls='--', lw=1.0, alpha=0.8)
        ax0.annotate(
            f'{ref_label} ({ref_val:g})',
            xy=(1.0, ref_val),
            xycoords=('axes fraction', 'data'),
            xytext=(-4, 4),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=8,
            color='#d62728',
        )
    if power_ylim is not None:
        ax0.set_ylim(power_ylim)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right', fontsize=8)

    # Panel 2: EEG quality score
    ax1 = axes[1]
    ax1.plot(result.time_s / 60.0, result.quality, color='#2ca02c', lw=1.5,
             label='quality')
    ax1.axhline(quality_threshold, color='#d62728', ls='--', lw=1.0,
                label=f'threshold={quality_threshold:g}')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel('EEG quality')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 3: bandpass EEG (0.5-45 Hz)
    ax2 = axes[2]
    ax2.plot(t_bp / 60.0, y_bp, color='#1f77b4', lw=0.45, alpha=0.85)
    ax2.set_ylabel(f'BP EEG ch{ch}\n(0.5-45 Hz, µV)')
    ax2.set_ylim(raw_ylim)
    ax2.grid(True, alpha=0.3)

    # Panel 4: raw EEG
    ax3 = axes[3]
    ax3.plot(t_raw / 60.0, y_raw, color='#444444', lw=0.45, alpha=0.85)
    ax3.set_ylabel(f'Raw EEG ch{ch}\n(µV)')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylim(raw_ylim)
    ax3.grid(True, alpha=0.3)

    # Shade low-quality windows on goertzel and signal panels.
    half = float(np.median(np.diff(result.time_s))) / 2.0 if result.time_s.size > 1 else 1.0
    for tb in result.time_s[~good]:
        lo = (tb - half) / 60.0
        hi = (tb + half) / 60.0
        ax0.axvspan(lo, hi, color='0.5', alpha=0.14, lw=0, zorder=0)
        ax2.axvspan(lo, hi, color='0.5', alpha=0.14, lw=0, zorder=0)
        ax3.axvspan(lo, hi, color='0.5', alpha=0.14, lw=0, zorder=0)

    # Event markers when subject is known in plot_event_markers SUBJECTS.
    for i, (name, onset_s) in enumerate(events):
        col = pem.EVT_COLORS[i % len(pem.EVT_COLORS)]
        x = onset_s / 60.0
        for ax in axes:
            ax.axvline(x, color=col, lw=1.0, ls='-', alpha=0.6)
        ax0.annotate(
            name,
            xy=(x, 1.0),
            xycoords=('data', 'axes fraction'),
            xytext=(2, -2),
            textcoords='offset points',
            rotation=90,
            va='top',
            ha='left',
            fontsize=7,
            color=col,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(os.path.splitext(out_png)[0] + '.svg')
    plt.close(fig)


def _iter_merged_csvs(root: str):
    for name in iter_subject_dirs(root):
        yield name, os.path.join(root, name, 'merged.csv')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Batch plot Goertzel power + EEG quality vs raw for iBrainCenter merged.csv.'
    )
    p.add_argument('--root', default='iBrainCenter', help='Root folder to scan (default iBrainCenter).')
    p.add_argument('--subject', default=None,
                   help='Process one subject directory name, e.g. Ann(SN027).')
    p.add_argument('--ch', type=int, default=1, help='1-based channel index (default 1).')
    p.add_argument('--channels', type=int, nargs='+', default=None,
                   help='Batch channel list, e.g. --channels 1 2. If set, overrides --ch.')
    p.add_argument('--fs', type=float, default=500.0, help='Sampling rate in Hz (default 500).')
    p.add_argument('--target-freq', type=float, default=60.0,
                   help='Goertzel target frequency in Hz (default 60).')
    p.add_argument('--win-sec', type=float, default=5.0,
                   help='Window size in seconds (default 5).')
    p.add_argument('--step-sec', type=float, default=5.0,
                   help='Step size in seconds (default 5).')
    p.add_argument('--smooth-win', type=int, default=5,
                   help='Rolling-median window in #windows (default 5; 1=no smoothing).')
    p.add_argument('--quality-threshold', type=float, default=pem.QUALITY_THRESHOLD,
                   help=f'Quality mask threshold (default {pem.QUALITY_THRESHOLD}).')
    p.add_argument('--exclude-hard-artifact', dest='exclude_hard_artifact',
                   action='store_true', default=True,
                   help='Exclude hard clipping/square-wave artifacts from valid windows (default on).')
    p.add_argument('--no-exclude-hard-artifact', dest='exclude_hard_artifact',
                   action='store_false',
                   help='Disable hard-artifact exclusion and use quality threshold only.')
    p.add_argument('--sat-uv', type=float, default=1950.0,
                   help='Absolute amplitude threshold for saturation ratio (default 1950 uV).')
    p.add_argument('--sat-frac-threshold', type=float, default=0.12,
                   help='Reject window when sat fraction >= this value (default 0.12).')
    p.add_argument('--step-ptp-threshold', type=float, default=1000.0,
                   help='Reject window when peak-to-peak >= this value (default 1000 uV).')
    p.add_argument('--bp-shift-sec', type=float, default=1.0,
                   help='Seconds at each window edge used for median-shift check (default 1.0).')
    p.add_argument('--bp-shift-threshold', type=float, default=80.0,
                   help='Reject window when |median(first)-median(last)| on bandpass exceeds this (default 80 uV).')
    p.add_argument('--raw-ylim', type=float, nargs=2, default=(-150.0, 150.0),
                   metavar=('LO', 'HI'), help='Raw EEG y-limits in µV (default -150 150).')
    p.add_argument('--out-stem', default='index_vs_raw_goertzel',
                   help='Output filename stem in each subject dir (default index_vs_raw_goertzel).')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    channels = args.channels if args.channels else [args.ch]

    targets = list(_iter_merged_csvs(args.root))
    if args.subject:
        targets = [(n, p) for (n, p) in targets if n == args.subject]
    if not targets:
        raise SystemExit('No merged.csv targets found.')

    for sub_name, merged_csv in targets:
        for ch in channels:
            out_png = os.path.join(
                args.root,
                sub_name,
                f'{args.out_stem}_ch{ch}_{args.target_freq:g}Hz.png',
            )
            out_csv = os.path.join(
                args.root,
                sub_name,
                f'{args.out_stem}_ch{ch}_{args.target_freq:g}Hz.csv',
            )
            print(f'Processing: {merged_csv} (ch{ch})')
            _plot_subject(
                merged_csv=merged_csv,
                out_png=out_png,
                out_csv=out_csv,
                ch=ch,
                fs=args.fs,
                target_freq=args.target_freq,
                win_sec=args.win_sec,
                step_sec=args.step_sec,
                smooth_win=args.smooth_win,
                quality_threshold=args.quality_threshold,
                raw_ylim=tuple(args.raw_ylim),
                exclude_hard_artifact=args.exclude_hard_artifact,
                sat_uv=args.sat_uv,
                sat_frac_threshold=args.sat_frac_threshold,
                step_ptp_threshold=args.step_ptp_threshold,
                bp_shift_sec=args.bp_shift_sec,
                bp_shift_threshold=args.bp_shift_threshold,
            )
            print(f'Saved: {out_png}')
            print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()
