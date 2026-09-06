"""
plot_meditation_zoom.py
========================
Zoomed extract of ch1-ch4 EEG time-domain traces + the qEEG Δ heatmap
(Focus/Flow/Calm/Relax vs pre-event baseline) for a single event block,
cropped from the full-session figure produced by plot_event_markers.py.

Reuses plot_event_markers's data loading / qEEG computation so the numbers
match the full-session PNG exactly (Hsin_SN032_eeg_tflite_pre_event_rest.png)
-- this script just narrows the time axis to one event and drops the panels
that aren't requested (quality / trend / TFLite / bar charts), while keeping
the EEG traces and the heatmap on a single shared (aligned) time axis.

Usage
-----
    python plot_meditation_zoom.py [--subject Hsin] \
        [--event "Mindfulness Meditation"] [--ds 10]
"""

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from lilia.windowing import require_continuous
import numpy as np

import plot_event_markers as pem

INDEX_KEYS = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS = ['Focus', 'Flow', 'Calm', 'Relax']
HEATMAP_BIN_SEC = 30


def build_event_list(name: str):
    evt_list = []
    for idx, (label, start_hhmm, dur_min, participants) in enumerate(pem.EVENTS):
        participates = (participants is None) or (name in participants)
        start_dt = pem.hhmm_to_dt(start_hhmm)
        end_dt = start_dt + datetime.timedelta(minutes=dur_min)
        color = pem.EVT_COLORS[idx % len(pem.EVT_COLORS)]
        evt_list.append((start_dt, end_dt, label, color, participates))
    return evt_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='Hsin', choices=list(pem.SUBJECTS))
    p.add_argument('--event', default='Mindfulness Meditation')
    p.add_argument('--ds', type=int, default=10, metavar='N',
                    help='Downsample factor for the EEG trace (default 10 -> 50Hz; '
                         'finer than the full-session overview since the window is short).')
    p.add_argument('--outdir',
                    default=os.path.join(pem.IBRAIN_DIR, 'event_verification'))
    args = p.parse_args()

    name = args.subject
    info = pem.SUBJECTS[name]
    merged = os.path.join(pem.IBRAIN_DIR, info['dir'], 'merged.csv')

    print(f'[{name}] loading merged.csv…')
    time_us_ds, data_ds = pem.load_merged_csv(merged, downsample=args.ds)
    time_us_full, data_full = pem.load_merged_csv(merged)
    require_continuous(time_us_full, pem.FS, 'plot_meditation_zoom.py')
    t_dt = np.array([pem.us_to_local_dt(u) for u in time_us_ds])
    n_ch = data_ds.shape[1]

    print(f'[{name}] filtering + computing qEEG indices…')
    data_filt = pem.bandpass_filter(data_full)
    qeeg_dt, qeeg_filt = pem.compute_qeeg_windowed(time_us_full, data_filt)
    qeeg_t_arr = np.array(qeeg_dt)

    print(f'[{name}] computing quality mask…')
    q_dt, q_overall = pem.compute_quality_windowed(time_us_full, data_full)
    q_median = np.median(q_overall, axis=1)
    low_qual = q_median < pem.QUALITY_THRESHOLD

    evt_list = build_event_list(name)
    pre_event_rest_specs = pem._build_pre_event_rest_specs(evt_list)

    target = next((e for e in evt_list if e[2] == args.event), None)
    if target is None:
        raise SystemExit(f'Unknown event {args.event!r}. Options: '
                          f'{[e[2] for e in evt_list]}')
    ev_start, ev_end, ev_label, ev_color, ev_participates = target
    if not ev_participates:
        raise SystemExit(f'{name} did not participate in {args.event!r}.')

    # ── qEEG Δ heatmap (30s bins, pre-event-rest baseline; same as full figure) ──
    n_hm = qeeg_filt[INDEX_KEYS[0]].shape[0]
    n_q = len(q_median)
    qual_mask = np.zeros(n_hm, dtype=bool)
    qual_mask[:min(n_q, n_hm)] = low_qual[:min(n_q, n_hm)]

    bin_size = max(1, int(HEATMAP_BIN_SEC / pem.QEEG_WIN_SEC))
    n_bins = n_hm // bin_size
    bin_t = []
    heatmap_abs = np.full((4, n_bins), np.nan)
    for b in range(n_bins):
        sl = slice(b * bin_size, (b + 1) * bin_size)
        good = ~qual_mask[sl]
        mid = qeeg_t_arr[b * bin_size + bin_size // 2]
        bin_t.append(mid)
        for idx_i, k in enumerate(INDEX_KEYS):
            vals = np.median(qeeg_filt[k][sl], axis=1)
            gv = vals[good]
            if gv.size > 0:
                heatmap_abs[idx_i, b] = float(np.median(gv))
    bin_t_arr = np.array(bin_t)
    heatmap_delta = pem._piecewise_event_delta(bin_t_arr, heatmap_abs,
                                                pre_event_rest_specs)
    heatmap_delta_ma = np.ma.masked_invalid(heatmap_delta)

    # ── Crop EEG trace + heatmap bins to the event window ───────────────────────
    eeg_mask = (t_dt >= ev_start) & (t_dt <= ev_end)
    hm_mask = (bin_t_arr >= ev_start) & (bin_t_arr <= ev_end)
    if not hm_mask.any():
        # fall back to nearest bins if the event is shorter than one 30s bin
        hm_mask[:] = True

    # ── Figure: ch1..ch4 + qEEG Δ heatmap, single shared (aligned) time axis ────
    height_ratios = [2.5] * n_ch + [2.2]
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(12, sum(hr * 0.9 for hr in height_ratios) + 1.4))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig,
                            height_ratios=height_ratios, hspace=0.15)

    ax_eeg = []
    for ch_i in range(n_ch):
        ax = fig.add_subplot(gs[ch_i], sharex=ax_eeg[0] if ax_eeg else None)
        ax_eeg.append(ax)
    ax_heatmap = fig.add_subplot(gs[n_ch], sharex=ax_eeg[0])

    for ch_i, ax in enumerate(ax_eeg):
        ax.plot(t_dt[eeg_mask], data_ds[eeg_mask, ch_i],
                color='#444444', lw=0.6, alpha=0.85)
        ax.set_ylabel(f'ch{ch_i + 1}\n(µV)', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axvspan(ev_start, ev_end, color=ev_color, alpha=0.10)
        plt.setp(ax.get_xticklabels(), visible=False)

    cmap_hm = LinearSegmentedColormap.from_list(
        'OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])
    cmap_hm.set_bad(color='#aaaaaa')
    bt = bin_t_arr[hm_mask]
    hd = heatmap_delta_ma[:, hm_mask]
    bin_t_num = mdates.date2num(bt)
    if len(bt) > 1:
        dt_h = float(np.diff(bin_t_num).mean()) / 2
    else:
        dt_h = HEATMAP_BIN_SEC / 86400 / 2
    t_edges = np.concatenate([[bin_t_num[0] - dt_h],
                              (bin_t_num[:-1] + bin_t_num[1:]) / 2 if len(bt) > 1 else [],
                              [bin_t_num[-1] + dt_h]])
    y_edges = np.arange(5) - 0.5
    v_abs = pem.HEATMAP_DELTA_VABS
    pcm = ax_heatmap.pcolormesh(t_edges, y_edges, hd, cmap=cmap_hm,
                                vmin=-v_abs, vmax=v_abs, shading='flat')
    cbar = plt.colorbar(pcm, ax=ax_heatmap, pad=0.012, fraction=0.03)
    cbar.set_label(f'Δ Index  (−{v_abs:.1f} → +{v_abs:.1f})', labelpad=10)
    ax_heatmap.set_yticks([0, 1, 2, 3])
    ax_heatmap.set_yticklabels(IDX_LABELS, fontsize=9)
    ax_heatmap.set_ylabel('qEEG Δ\n(vs baseline)', fontsize=9)
    ax_heatmap.xaxis_date()
    ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_heatmap.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax_heatmap.set_xlabel('Local Time (UTC+8, HH:MM)', fontsize=10)

    for ax in ax_eeg + [ax_heatmap]:
        ax.set_xlim(ev_start, ev_end)

    fig.suptitle(
        f'iBrainCenter — {info["sn"]} ({name})  |  {ev_label}  '
        f'({ev_start.strftime("%H:%M")}–{ev_end.strftime("%H:%M")})\n'
        f'ch1–ch4 EEG (ds×{args.ds}={pem.FS // args.ds}pt/s)  +  '
        f'qEEG Δ vs pre-event baseline (30s bins) — aligned time axis',
        fontsize=11, fontweight='bold')

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(
        args.outdir,
        f'{name}_{info["sn"]}_{ev_label.replace(" ", "_")}_ch1-4_qEEG_delta.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(out_path)[0] + '.svg')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
