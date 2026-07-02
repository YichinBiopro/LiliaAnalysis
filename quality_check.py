"""
EEG signal-quality checks (two subcommands).

    quality_check.py samples     Randomly sample 2 non-overlapping 30 s segments
                                 from every merged.csv (iBrainCenter + YoGa),
                                 bandpass 0.5–45 Hz, and plot raw EEG + PSD +
                                 quality + qEEG indices side-by-side. Quality
                                 scored with the flat+spectrum-only params.

    quality_check.py anomalies   Diagnose "abnormal" segments on the EEG-quality
                                 curve and line them up against the raw time
                                 domain, so gap-bridging plot artifacts are told
                                 apart from genuine bad signal (clipping / flat-
                                 lining). Quality scored with the iBrainCenter
                                 device params.

The two subcommands deliberately keep their own quality parameterisations
(``SAMPLES_QUALITY_PARAMS`` vs ``plot_event_markers.QUALITY_PARAMS``); FS and the
0.5 quality threshold are identical for both.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import welch

from lilia.io import load_merged_csv, bandpass_filter
from lilia.quality import (
    get_eeg_quality_index_v2_parametric,
    get_best_eeg_quality_v2_flat_spectrum_only_params,
)
from lilia.qeeg import compute_qeeg_indices
from plot_event_markers import (
    QUALITY_WIN_SEC, QUALITY_PARAMS, SUBJECTS, IBRAIN_DIR,
    us_to_local_dt, _series_with_gaps,
)

# ── Shared constants (identical for both subcommands) ────────────────────────────
FS = 500                    # Hz
QUALITY_THRESHOLD = 0.5
CH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═════════════════════════════════════════════════════════════════════════════
# Subcommand: samples  — random 30 s segments, raw/filtered quality + PSD + qEEG
# ═════════════════════════════════════════════════════════════════════════════

SEG_SEC = 30.0              # segment length (seconds)
N_SEGS = 2                  # number of random segments per file
SEG_SAMPLES = int(SEG_SEC * FS)

SAMPLES_QUALITY_PARAMS = get_best_eeg_quality_v2_flat_spectrum_only_params()

BP_LOW = 0.5                # Hz — bandpass lower cutoff
BP_HIGH = 45.0             # Hz — bandpass upper cutoff

QEEG_WIN_SEC = 5.0          # qEEG index window length (seconds)
QEEG_INDICES = ['focus', 'flow', 'calm', 'relaxation']


def pick_segments(n_total: int, seg_len: int, n_segs: int, rng: np.random.Generator):
    """
    Pick *n_segs* non-overlapping start indices uniformly at random from
    [0, n_total - seg_len].  Returns sorted list of start indices.
    """
    starts = []
    max_start = n_total - seg_len
    attempts = 0
    while len(starts) < n_segs and attempts < 10_000:
        attempts += 1
        s = int(rng.integers(0, max_start + 1))
        # check non-overlapping with already chosen segments
        if all(abs(s - prev) >= seg_len for prev in starts):
            starts.append(s)
    starts.sort()
    return starts


def compute_qeeg_windowed_seg(seg: np.ndarray, fs: float = FS,
                              win_sec: float = QEEG_WIN_SEC):
    """
    Slide non-overlapping *win_sec* windows over *seg* (N, n_ch) and
    compute qEEG indices per channel.

    Returns
    -------
    t_mid   : (n_wins,) relative midpoint times in seconds
    scores  : dict { index_name -> (n_wins, n_ch) }
    """
    win = int(win_sec * fs)
    n, n_ch = seg.shape
    t_mid = []
    accum = {k: [] for k in QEEG_INDICES}
    for s in range(0, n - win + 1, win):
        t_mid.append((s + win / 2) / fs)
        row = {k: [] for k in QEEG_INDICES}
        for ch_i in range(n_ch):
            res = compute_qeeg_indices(seg[s: s + win, ch_i].astype(np.float64), fs=fs)
            for k in QEEG_INDICES:
                row[k].append(res[k])
        for k in QEEG_INDICES:
            accum[k].append(row[k])
    return np.array(t_mid), {k: np.array(accum[k]) for k in QEEG_INDICES}


def plot_segments(path: str, group: str, subject: str, outdir: str,
                  rng: np.random.Generator):
    print(f'  [{group}/{subject}] loading…', end=' ', flush=True)
    time_us, data = load_merged_csv(path)
    n_total, n_ch = data.shape
    print(f'{n_total} pts, {n_ch} ch')

    starts = pick_segments(n_total, SEG_SAMPLES, N_SEGS, rng)
    if len(starts) < N_SEGS:
        print(f'    WARNING: only {len(starts)} segment(s) found, skipping')
        return

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Rows per segment column:
    #   n_ch  raw EEG traces
    #   1     quality bar chart (raw vs filtered)
    #   n_ch  PSD panels
    #   4     qEEG index bar charts (Focus / Flow / Calm / Relaxation)
    n_qeeg = len(QEEG_INDICES)
    n_rows = n_ch + 1 + n_ch + n_qeeg
    height_ratios = [2.5] * n_ch + [1.8] + [1.8] * n_ch + [1.5] * n_qeeg
    fig = plt.figure(figsize=(10 * N_SEGS, sum(hr * 0.82 for hr in height_ratios) + 1.5))
    fig.suptitle(
        f'{group} — {subject}  |  {N_SEGS} random {SEG_SEC:.0f}s segments  '
        f'[BP {BP_LOW}–{BP_HIGH} Hz]  (quality: flat+spectrum only @ {FS}Hz)\n'
        f'qEEG indices: raw (dashed) vs BP-filtered (solid) — {QEEG_WIN_SEC:.0f}s non-overlapping windows',
        fontsize=13, fontweight='bold',
    )

    outer = gridspec.GridSpec(1, N_SEGS, figure=fig, hspace=0.05, wspace=0.18)

    for col, start in enumerate(starts):
        seg_raw = data[start: start + SEG_SAMPLES]           # (SEG_SAMPLES, n_ch)
        seg_t = time_us[start: start + SEG_SAMPLES]
        t_s = (seg_t - seg_t[0]) / 1e6                       # relative seconds

        # Bandpass filter (0.5–45 Hz) — used for quality & PSD
        seg_data = bandpass_filter(seg_raw)

        # qEEG indices — 5s non-overlapping windows, raw and filtered
        qeeg_t, qeeg_raw_w = compute_qeeg_windowed_seg(seg_raw)
        _, qeeg_filt_w = compute_qeeg_windowed_seg(seg_data)

        # Quality on RAW signal
        result_raw = get_eeg_quality_index_v2_parametric(
            seg_raw.T.astype(np.float64),
            fs=FS,
            params=SAMPLES_QUALITY_PARAMS,
        )
        overall_raw = result_raw["overall"]       # (n_ch,)

        # Quality on FILTERED signal
        result = get_eeg_quality_index_v2_parametric(
            seg_data.T.astype(np.float64),
            fs=FS,
            params=SAMPLES_QUALITY_PARAMS,
        )
        overall = result["overall"]       # (n_ch,)

        # PSD via Welch — raw and filtered
        psd_freqs, psd_powers_raw, psd_powers_filt = [], [], []
        for ch_i in range(n_ch):
            f, pxx_raw = welch(seg_raw[:, ch_i].astype(np.float64),
                               fs=FS, nperseg=FS * 4, noverlap=FS * 2)
            _, pxx_filt = welch(seg_data[:, ch_i].astype(np.float64),
                                fs=FS, nperseg=FS * 4, noverlap=FS * 2)
            psd_freqs.append(f)
            psd_powers_raw.append(pxx_raw)
            psd_powers_filt.append(pxx_filt)

        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, 1, subplot_spec=outer[col],
            hspace=0.30,
            height_ratios=height_ratios,
        )

        # ── EEG raw + filtered panels ────────────────────────────────────────
        axes_eeg = []
        for ch_i in range(n_ch):
            ax = fig.add_subplot(inner[ch_i],
                                 sharex=axes_eeg[0] if axes_eeg else None)
            ax.plot(t_s, seg_raw[:, ch_i],
                    color='#aaaaaa', lw=0.4, alpha=0.5, label='raw')
            ax.plot(t_s, seg_data[:, ch_i],
                    color=CH_COLORS[ch_i % len(CH_COLORS)], lw=0.6,
                    label=f'BP {BP_LOW}–{BP_HIGH}Hz')
            ax.set_ylabel(f'ch{ch_i+1}\n(µV)', fontsize=8)
            ax.grid(True, alpha=0.2)
            if ch_i == 0:
                abs_sec = int((seg_t[0] - time_us[0]) / 1e6)
                h, rem = divmod(abs_sec, 3600)
                m, s_ = divmod(rem, 60)
                ax.set_title(
                    f'Segment {col+1}  |  start +{h:02d}:{m:02d}:{s_:02d} from recording',
                    fontsize=9,
                )
                ax.legend(loc='upper right', fontsize=6, framealpha=0.7)
            if ch_i < n_ch - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            axes_eeg.append(ax)
        axes_eeg[-1].set_xlabel('Time within segment (s)', fontsize=8)

        # ── Quality comparison bar chart (raw vs filtered) ──────────────────
        ax_q = fig.add_subplot(inner[n_ch])   # independent x-axis
        bar_x = np.arange(n_ch)
        bar_w = 0.35
        ax_q.bar(bar_x - bar_w/2, overall_raw, width=bar_w,
                 color=[CH_COLORS[i % len(CH_COLORS)] for i in range(n_ch)],
                 alpha=0.40, hatch='//', label='raw', zorder=3)
        ax_q.bar(bar_x + bar_w/2, overall, width=bar_w,
                 color=[CH_COLORS[i % len(CH_COLORS)] for i in range(n_ch)],
                 alpha=0.85, label=f'BP {BP_LOW}–{BP_HIGH}Hz', zorder=3)
        ax_q.axhline(QUALITY_THRESHOLD, color='red', lw=1.2, ls='--',
                     label=f'threshold {QUALITY_THRESHOLD:.2f}')
        for i in range(n_ch):
            ax_q.text(i - bar_w/2, overall_raw[i] + 0.015, f'{overall_raw[i]:.2f}',
                      ha='center', va='bottom', fontsize=6.5, color='#555555')
            ax_q.text(i + bar_w/2, overall[i] + 0.015, f'{overall[i]:.2f}',
                      ha='center', va='bottom', fontsize=6.5, fontweight='bold')
            # delta annotation
            delta = overall[i] - overall_raw[i]
            sign = '+' if delta >= 0 else ''
            ax_q.text(i, max(overall_raw[i], overall[i]) + 0.065,
                      f'Δ{sign}{delta:.2f}',
                      ha='center', va='bottom', fontsize=6.5,
                      color='green' if delta >= 0 else 'red')
        ax_q.set_ylim(0, 1.22)
        ax_q.set_xticks(bar_x)
        ax_q.set_xticklabels([f'ch{i+1}' for i in range(n_ch)], fontsize=8)
        ax_q.set_ylabel('Overall\nQuality', fontsize=8)
        ax_q.legend(loc='upper right', fontsize=7)
        ax_q.grid(True, alpha=0.2, axis='y')
        ax_q.set_xlim(-0.5, n_ch - 0.5)

        # ── PSD panels (one per channel, shared x and y axes) ────────────────
        BANDS = [('δ', 0.5, 4,  '#a8d8ea'),
                 ('θ', 4,   8,  '#a8e6cf'),
                 ('α', 8,   13, '#ffd3b6'),
                 ('β', 13,  30, '#ffaaa5'),
                 ('γ', 30,  50, '#d4a5ff')]
        axes_psd = []
        for ch_i in range(n_ch):
            row = n_ch + 1 + ch_i
            share = axes_psd[0] if axes_psd else None
            ax_p = fig.add_subplot(inner[row], sharex=share, sharey=share)

            f = psd_freqs[ch_i]
            mask = f <= 50
            pxx_raw = psd_powers_raw[ch_i]
            pxx_filt = psd_powers_filt[ch_i]

            ax_p.semilogy(f[mask], pxx_raw[mask],
                          color='#aaaaaa', lw=0.9, alpha=0.7, ls='--', label='raw')
            ax_p.semilogy(f[mask], pxx_filt[mask],
                          color=CH_COLORS[ch_i % len(CH_COLORS)], lw=1.2,
                          label=f'BP {BP_LOW}–{BP_HIGH}Hz')

            for bname, blo, bhi, bcol in BANDS:
                ax_p.axvspan(blo, bhi, alpha=0.12, color=bcol)

            ax_p.set_ylabel(f'ch{ch_i+1}\n(µV²/Hz)', fontsize=7)
            ax_p.grid(True, alpha=0.2, which='both')
            if ch_i < n_ch - 1:
                plt.setp(ax_p.get_xticklabels(), visible=False)
            else:
                ax_p.set_xlabel('Frequency (Hz)', fontsize=8)
            if ch_i == 0:
                ax_p.set_title('PSD (Welch, 4 s window)', fontsize=8, pad=3)
                ax_p.legend(loc='upper right', fontsize=6, framealpha=0.8)
            axes_psd.append(ax_p)

        # annotate band labels on last PSD panel after shared ylim is settled
        ylim = axes_psd[-1].get_ylim()
        label_y = 10 ** (np.log10(ylim[0]) + 0.85 * (np.log10(ylim[1]) - np.log10(ylim[0])))
        for ax_p in axes_psd:
            for bname, blo, bhi, _ in BANDS:
                ax_p.text((blo + bhi) / 2, label_y, bname,
                          ha='center', fontsize=6, alpha=0.65, clip_on=True)

        # ── qEEG index line plots (5s windows, raw dashed vs filtered solid) ──
        for idx_i, idx_name in enumerate(QEEG_INDICES):
            row = n_ch + 1 + n_ch + idx_i
            ax_qi = fig.add_subplot(inner[row])
            for ch_i in range(n_ch):
                c = CH_COLORS[ch_i % len(CH_COLORS)]
                ax_qi.plot(qeeg_t, qeeg_raw_w[idx_name][:, ch_i],
                           color=c, lw=0.9, ls='--', alpha=0.55,
                           label=f'ch{ch_i+1} raw' if idx_i == 0 else None)
                ax_qi.plot(qeeg_t, qeeg_filt_w[idx_name][:, ch_i],
                           color=c, lw=1.4, alpha=0.85,
                           label=f'ch{ch_i+1} filt' if idx_i == 0 else None)
            ax_qi.axhline(0, color='k', lw=0.5, ls=':')
            ax_qi.set_ylim(-1.15, 1.15)
            ax_qi.set_xlim(0, SEG_SEC)
            ax_qi.set_ylabel(f'{idx_name.capitalize()}\nIndex', fontsize=7)
            ax_qi.grid(True, alpha=0.2)
            if idx_i < n_qeeg - 1:
                plt.setp(ax_qi.get_xticklabels(), visible=False)
            else:
                ax_qi.set_xlabel('Time within segment (s)', fontsize=8)
            if idx_i == 0:
                ax_qi.legend(loc='upper right', fontsize=6, framealpha=0.7,
                             ncol=n_ch, title='── filt  -- raw', title_fontsize=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fname = f'{group}_{subject}_sample_quality.png'
    outpath = os.path.join(outdir, fname)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'       → {outpath}')


def discover_merged_csvs():
    """Yield (group, subject, path) for every merged.csv found."""
    for group_dir in ['iBrainCenter', 'YoGa']:
        root = os.path.join(BASE_DIR, group_dir)
        if not os.path.isdir(root):
            continue
        for subj_dir in sorted(os.listdir(root)):
            csv_path = os.path.join(root, subj_dir, 'merged.csv')
            if os.path.isfile(csv_path):
                yield group_dir, subj_dir, csv_path


def run_samples(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'Sampling {N_SEGS}×{SEG_SEC:.0f}s segments from all merged.csv '
          f'(seed={args.seed}) → {args.outdir}\n')

    for group, subject, path in discover_merged_csvs():
        plot_segments(path, group, subject, args.outdir, rng)

    print('\nDone.')


# ═════════════════════════════════════════════════════════════════════════════
# Subcommand: anomalies  — quality-curve anomalies vs raw time domain
# ═════════════════════════════════════════════════════════════════════════════
# Two very different causes of "abnormal" quality-curve segments are separated:
#   1. gap-bridging artifact — merged.csv concatenates recordings with time gaps;
#      matplotlib draws a straight line across the gap (not a real quality change).
#   2. genuine bad signal — ADC clipping (±2048 rail), flat-lining, or noise.

RAIL_VALUE = 2048.0   # 12-bit ADC full-scale; |x| >= RAIL-1 counts as clipping
CLIP_FRAC = 0.01      # window clip-sample fraction > 1% → flag saturation
FLAT_FRAC = 0.05      # window "all-4-channels zero-diff" fraction > 5% → flat-line
GAP_SEC = 1.0         # any adjacent-sample dt > 1s → window straddles a data gap
JUMP_DELTA = 0.40     # |median-quality jump| between windows > 0.40 → flag jump
MAX_RAW_PANELS = 6    # max raw-waveform panels shown in the report


def analyze_quality_windows(time_us: np.ndarray, data: np.ndarray,
                            fs: float = FS, win_sec: float = QUALITY_WIN_SEC,
                            params=QUALITY_PARAMS) -> list[dict]:
    """Per non-overlapping window: quality score + raw-signal diagnostics.

    Returns for each window: s/e (raw sample bounds), mid_us (absolute midpoint),
    qmed (channel-median overall quality), clip/flat fractions, and gap (max
    adjacent-sample dt in seconds — reveals data breaks).
    """
    win = int(win_sec * fs)
    rows = []
    for s in range(0, len(data) - win + 1, win):
        seg = data[s:s + win]
        t_seg = time_us[s:s + win]
        res = get_eeg_quality_index_v2_parametric(
            seg.T.astype(np.float64), fs=fs, params=params)
        q = np.asarray(res["overall"], dtype=float)
        clip = float(np.mean(np.any(np.abs(seg) >= RAIL_VALUE - 1.0, axis=1)))
        flat = float(np.mean(np.all(np.diff(seg, axis=0) == 0, axis=1)))
        gap = float(np.max(np.diff(t_seg)) / 1e6) if len(t_seg) > 1 else 0.0
        rows.append(dict(s=int(s), e=int(s + win), mid_us=int(t_seg[len(t_seg) // 2]),
                         qmed=float(np.median(q)), clip=clip, flat=flat, gap=gap))
    return rows


def flag_anomalies(rows: list[dict],
                   q_thresh: float = QUALITY_THRESHOLD,
                   clip_frac: float = CLIP_FRAC,
                   flat_frac: float = FLAT_FRAC,
                   gap_sec: float = GAP_SEC,
                   jump_delta: float = JUMP_DELTA) -> list[dict]:
    """Tag each window with anomaly reasons and a severity score.

    Severity ranks windows for manual review: normalised clip/flat/gap plus the
    amount below the quality threshold. ``reasons`` is a human-readable list;
    empty means a normal window.
    """
    qmed = np.array([r["qmed"] for r in rows])
    for i, r in enumerate(rows):
        reasons = []
        if r["gap"] > gap_sec:
            reasons.append(f"time-gap {r['gap']:.0f}s")          # data break → bridging artifact
        if r["clip"] > clip_frac:
            reasons.append(f"clip {r['clip'] * 100:.0f}%")        # ADC saturation clipping
        if r["flat"] > flat_frac:
            reasons.append(f"flat {r['flat'] * 100:.0f}%")        # flat-line / disconnection
        if r["qmed"] < q_thresh:
            reasons.append(f"low-Q {r['qmed']:.2f}")              # low quality
        if i > 0 and abs(qmed[i] - qmed[i - 1]) > jump_delta:
            reasons.append(f"jump {qmed[i] - qmed[i - 1]:+.2f}")  # quality jump
        r["reasons"] = reasons
        r["severity"] = (min(r["gap"] / 10.0, 1.0) + r["clip"] + r["flat"]
                         + max(0.0, q_thresh - r["qmed"]))
    return rows


def plot_quality_anomaly_report(name: str, info: dict, outdir: str,
                                base_dir: str = None,
                                win_sec: float = QUALITY_WIN_SEC) -> str:
    """Render the "quality anomaly vs raw time domain" comparison PNG."""
    if base_dir is None:
        base_dir = IBRAIN_DIR
    merged = os.path.join(base_dir, info["dir"], "merged.csv")
    if not os.path.isfile(merged):
        print(f"  [{name}] merged.csv not found — skipping")
        return ""

    print(f"  [{name}] loading & scoring quality windows…", flush=True)
    time_us, data = load_merged_csv(merged)
    rows = flag_anomalies(analyze_quality_windows(time_us, data, win_sec=win_sec))

    flagged = [r for r in rows if r["reasons"]]
    print(f"  [{name}] {len(flagged)}/{len(rows)} windows flagged as abnormal:")
    for r in flagged:
        ts = us_to_local_dt(r["mid_us"]).strftime("%H:%M:%S")
        print(f"      {ts}  " + "  ·  ".join(r["reasons"]))

    # pick the most severe windows for raw-waveform comparison
    show = sorted(flagged, key=lambda r: r["severity"], reverse=True)[:MAX_RAW_PANELS]
    show = sorted(show, key=lambda r: r["s"])           # time-order for readability
    n_raw = len(show)

    q_dt = np.array([us_to_local_dt(r["mid_us"]) for r in rows])
    q_med = np.array([r["qmed"] for r in rows])

    # ── Figure: top = quality timeline (gaps broken), bottom = raw waveforms ──
    n_cols = min(3, max(1, n_raw))
    n_rows_raw = int(np.ceil(n_raw / n_cols)) if n_raw else 0
    fig = plt.figure(figsize=(16, 3.2 + 2.4 * n_rows_raw))
    gs = gridspec.GridSpec(1 + n_rows_raw, n_cols, figure=fig,
                           height_ratios=[2.0] + [2.2] * n_rows_raw,
                           hspace=0.55, wspace=0.22)

    # Top: quality timeline. _series_with_gaps inserts NaN across big gaps → no
    # diagonal bridging artifact.
    ax_q = fig.add_subplot(gs[0, :])
    t_brk, y_brk = _series_with_gaps(q_dt, q_med, gap_sec=win_sec * 2.5)
    ax_q.plot(t_brk, y_brk, color="#1a1a1a", lw=1.3, label="ch-median quality (gap-aware)")
    ax_q.axhline(QUALITY_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.6,
                 label=f"threshold {QUALITY_THRESHOLD:.2f}")
    # overlay each anomaly category with a distinct marker
    cat_style = {"time-gap": ("v", "#9467bd"), "clip": ("s", "#d62728"),
                 "flat": ("D", "#ff7f0e"), "low-Q": ("o", "#1f77b4"),
                 "jump": ("^", "#2ca02c")}
    seen = set()
    for r in flagged:
        for reason in r["reasons"]:
            cat = reason.split()[0]
            mk, col = cat_style.get(cat, ("x", "#000000"))
            ax_q.plot(us_to_local_dt(r["mid_us"]), r["qmed"], mk, color=col,
                      ms=7, mec="k", mew=0.4,
                      label=cat if cat not in seen else None)
            seen.add(cat)
    # mark the windows whose raw waveforms are shown below
    for j, r in enumerate(show):
        ax_q.annotate(f"#{j + 1}", (us_to_local_dt(r["mid_us"]), r["qmed"]),
                      textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=8, fontweight="bold", color="#b00")
    ax_q.set_ylim(0, 1.05)
    ax_q.set_ylabel("EEG Quality")
    ax_q.set_xlabel("Local Time (UTC+8, HH:MM)")
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_q.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax_q.legend(loc="lower right", fontsize=8, ncol=4, framealpha=0.85)
    ax_q.set_title(f"{name} ({info['sn']}) — EEG quality anomalies vs raw time-domain",
                   fontsize=12, fontweight="bold")
    ax_q.grid(True, alpha=0.2)

    # Bottom: raw 4-ch waveform of each selected window (with ±1-window context)
    win = int(win_sec * FS)
    for j, r in enumerate(show):
        ax = fig.add_subplot(gs[1 + j // n_cols, j % n_cols])
        ctx = win                                        # extend one window each side as context
        lo = max(0, r["s"] - ctx)
        hi = min(len(data), r["e"] + ctx)
        seg = data[lo:hi]
        t_rel = (time_us[lo:hi] - time_us[r["s"]]) / 1e6  # seconds relative to window start
        for ch in range(seg.shape[1]):
            ax.plot(t_rel, seg[:, ch], lw=0.5, color=CH_COLORS[ch % 4],
                    label=f"ch{ch + 1}")
        # window body span + ADC full-scale reference lines
        ax.axvspan(0, win_sec, color="grey", alpha=0.12)
        ax.axhline(RAIL_VALUE, color="r", lw=0.6, ls=":", alpha=0.7)
        ax.axhline(-RAIL_VALUE, color="r", lw=0.6, ls=":", alpha=0.7)
        # if the window contains a data break, mark its position
        dt_local = np.diff(time_us[r["s"]:r["e"]]) / 1e6
        if dt_local.size and dt_local.max() > GAP_SEC:
            k = int(np.argmax(dt_local))
            gx = (time_us[r["s"] + k] - time_us[r["s"]]) / 1e6
            ax.axvline(gx, color="#9467bd", lw=1.2, ls="--",
                       label=f"gap {dt_local.max():.0f}s")
        ts = us_to_local_dt(r["mid_us"]).strftime("%H:%M:%S")
        ax.set_title(f"#{j + 1}  {ts}\n" + " · ".join(r["reasons"]),
                     fontsize=8.5)
        ax.set_xlabel("Time within window (s)", fontsize=8)
        ax.set_ylabel("µV (raw)", fontsize=8)
        ax.set_ylim(-RAIL_VALUE * 1.1, RAIL_VALUE * 1.1)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(loc="upper right", fontsize=6, ncol=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{name}_{info['sn']}_quality_anomalies.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"       → {outpath}")
    return outpath


def run_anomalies(args: argparse.Namespace) -> None:
    targets = SUBJECTS.items() if args.all else [(args.subject, SUBJECTS[args.subject])]
    print(f"── Quality anomaly check → {args.outdir}")
    for name, info in targets:
        plot_quality_anomaly_report(name, info, args.outdir)
    print("\nDone.")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="EEG signal-quality checks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_s = sub.add_parser(
        "samples", help="random 30 s segments: raw/filtered quality + PSD + qEEG")
    p_s.add_argument('--outdir', default=os.path.join(BASE_DIR, 'sample_quality'),
                     metavar='DIR', help='Output directory for PNGs.')
    p_s.add_argument('--seed', type=int, default=42,
                     help='Random seed for reproducibility (default 42).')
    p_s.set_defaults(func=run_samples)

    p_a = sub.add_parser(
        "anomalies", help="quality-curve anomalies lined up against raw time domain")
    p_a.add_argument("--subject", default="Hardy",
                     help="Subject name (default Hardy — most data gaps, most illustrative).")
    p_a.add_argument("--all", action="store_true",
                     help="Run over every iBrainCenter subject.")
    p_a.add_argument("--outdir", default=os.path.join(IBRAIN_DIR, "quality_anomalies"),
                     metavar="DIR", help="PNG output directory.")
    p_a.set_defaults(func=run_anomalies)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
