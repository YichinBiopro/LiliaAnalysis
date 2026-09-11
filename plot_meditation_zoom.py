"""Crop one event's raw EEG and BP qEEG from the shared pre-event-rest analysis.

Examples:
    python plot_meditation_zoom.py --subject Hsin --event 'Mindfulness Meditation'
    python plot_meditation_zoom.py --csv /path/to/merged.csv --outdir /tmp/zoom

The CSV override retains the selected subject's session event schedule.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

import plot_event_markers as pem
from lilia.quality_audit import diagnostic_summary
from lilia.event_qeeg import analyze_recording, summarize_branch
from lilia.event_qeeg_io import json_safe
from lilia.event_zoom import select_zoom, zoom_values
from lilia.event_zoom_io import write_zoom_table
from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.time_utils import local_dt_to_utc_us

INDEX_KEYS = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS = ['Focus', 'Flow', 'Calm', 'Relax']
HEATMAP_BIN_SEC = 30
STATUS_LABELS = {
    'computed': 'Complete event bins; baseline = preceding rest',
    'not_participating': 'Not participating in this event',
    'no_raw_samples': 'No recording samples in this event',
    'no_complete_metric_windows': 'No complete 5 s metric windows',
    'no_complete_event_bins': 'No complete 30 s bins inside this event',
    'missing_baseline': 'No accepted 30 s baseline bins in preceding rest',
    'no_accepted_event_bins': 'No accepted event bins after quality / metric checks',
}


def build_event_list(name):
    events = []
    for idx, (label, start_hhmm, dur_min, participants) in enumerate(pem.EVENTS):
        start = pem.hhmm_to_dt(start_hhmm)
        events.append((start, start + datetime.timedelta(minutes=dur_min), label,
                       pem.EVT_COLORS[idx % len(pem.EVT_COLORS)],
                       participants is None or name in participants))
    return events


def event_records(name):
    return [{'start_us': local_dt_to_utc_us(start, 8), 'end_us': local_dt_to_utc_us(end, 8),
             'label': label, 'color': color, 'participates': participates}
            for start, end, label, color, participates in build_event_list(name)]


def draw_zoom(time_us, raw, branch, selection, name, info, ds, outpath):
    n_ch = raw.shape[1]
    fig = plt.figure(figsize=(12, n_ch*2.2+3))
    grid = fig.add_gridspec(n_ch+1, 2, width_ratios=[1, .025],
                           height_ratios=[2.5]*n_ch+[2.2])
    axes = []
    for row in range(n_ch+1):
        axes.append(fig.add_subplot(grid[row, 0], sharex=axes[0] if axes else None))
    colorbar_ax = fig.add_subplot(grid[-1, 1])
    try:
        for ch, ax in enumerate(axes[:-1]):
            for run in selection['display_runs']:
                indices = np.asarray(run['display_raw_indices'], dtype=int)
                times = [pem.us_to_local_dt(t) for t in time_us[indices]]
                ax.plot(times, raw[indices, ch], color='#444444', lw=.6, alpha=.85,
                        marker='.' if len(indices)==1 else None)
                ax.axvspan(pem.us_to_local_dt(run['start_us']), pem.us_to_local_dt(run['end_us']),
                           color=selection['target']['color'], alpha=.10)
            ax.set_ylabel(f'ch{ch+1}\n(µV)')
            ax.grid(True, alpha=.2)
            ax.tick_params(labelbottom=False)
        cmap = LinearSegmentedColormap.from_list('OrgPur', ['#5e3c99', '#f7f7f7', '#e66101'])
        cmap.set_bad('#aaaaaa')
        ax = axes[-1]
        values = zoom_values(branch, selection)
        bins = [branch['summary']['bins'][i] for i in selection['display_bins']] if branch is not None else []
        pem._draw_segment_heatmap(ax, bins, np.ma.masked_invalid(values), cmap, pem.HEATMAP_DELTA_VABS)
        mappable = ScalarMappable(norm=Normalize(-pem.HEATMAP_DELTA_VABS, pem.HEATMAP_DELTA_VABS), cmap=cmap)
        fig.colorbar(mappable, cax=colorbar_ax, label='Δ Index vs preceding rest')
        ax.set_yticks(range(4), IDX_LABELS)
        ax.set_ylim(-.5, 3.5)
        ax.set_ylabel('BP qEEG Δ')
        if selection['status'] != 'computed':
            ax.text(.5, .5, STATUS_LABELS[selection['status']], ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, bbox={'facecolor':'white', 'alpha':.9})
        target = selection['target']
        tick_format = '%H:%M:%S' if target['end_us']-target['start_us'] < 180e6 else '%H:%M'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(tick_format))
        duration_sec = (target['end_us']-target['start_us'])/1e6
        if 1 <= duration_sec < 180:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(np.ceil(duration_sec/8)))))
        else:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
            ax.xaxis.set_major_locator(locator)
            if duration_sec < 1:
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel('Local time (UTC+8) | gray = unavailable bin; blank = no complete bin')
        ax.set_xlim(pem.us_to_local_dt(target['start_us']), pem.us_to_local_dt(target['end_us']))
        fig.suptitle(f'iBrainCenter — {info["sn"]} ({name}) | {target["label"]}\n'
                     f'Raw EEG (display stride {ds}, retaining run edges) + BP qEEG Δ (30 s bins)\n'
                     f'{STATUS_LABELS[selection["status"]]}', fontsize=11)
        fig.tight_layout()
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        fig.savefig(Path(outpath).with_suffix('.svg'))
    finally:
        plt.close(fig)


def run_zoom(name, info, source, outdir, event='Mindfulness Meditation', ds=10, *, events=None):
    """Save audit and partial display even when the requested delta is unavailable."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_event = re.sub(r'[^\w.-]+', '_', event)
    stem = outdir / f'{name}_{info["sn"]}_{safe_event}_ch1-4_qEEG_delta'
    audit_path = Path(str(stem)+'_analysis.json')
    audit = {'kind': 'event_zoom', 'schema_version': 1, 'subject': name,
             'source_path': str(Path(source).resolve()), 'status': 'processing',
             'errors': [], 'artifacts': {}}
    try:
        paths = [Path(__file__), Path(pem.__file__), *sorted((Path(__file__).parent/'lilia').glob('*.py'))]
        code_id = hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
        audit.update(source_id=file_sha256(source), code_sha256=code_id)
        frame = read_lilia_frame(source)
        t = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        raw = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        events = event_records(name) if events is None else events
        # Validate schedule and display arguments before running signal analysis.
        select_zoom(t, None, events, event, ds, pem.FS)
        parameters = {'scope': 'event_zoom', 'fs': pem.FS, 'input_fs': pem.FS,
            'win_sec': pem.QEEG_WIN_SEC, 'step_sec': pem.QEEG_WIN_SEC, 'index_space': 'raw_samples',
            'bandpass': [pem.BP_LOW, pem.BP_HIGH], 'quality_params': pem.QUALITY_PARAMS,
            'quality_threshold': pem.QUALITY_THRESHOLD, 'quality_source': 'raw_all_channels_same_physical_interval',
            'quality_channels': raw.shape[1], 'metric_channels': raw.shape[1],
            'baseline_mode': 'pre-event-rest', 'events': events, 'target_event': event,
            'display_downsample': ds, 'selection_policy': 'complete_containment',
            'heatmap_bin_windows': 6, 'heatmap_aggregation': 'median_window_channel_medians',
            'model_sha256': None, 'display_policy': 'segment-local crop with first/last raw samples; boundary bins missing'}
        audit.update(parameters=parameters, source_samples=len(t), source_epoch_us=int(t[0]))
        result = analyze_recording(t, raw, fs=pem.FS, win_sec=pem.QEEG_WIN_SEC,
            low=pem.BP_LOW, high=pem.BP_HIGH, scorer=pem.get_eeg_quality_index_v2_parametric,
            quality_params=pem.QUALITY_PARAMS, threshold=pem.QUALITY_THRESHOLD)
        branch = result['bp']
        if branch is not None:
            branch['summary'] = summarize_branch(branch, events, 'pre-event-rest')
        selection = select_zoom(t, branch, events, event, ds, pem.FS)
        audit.update(segments=result['segments'], selection=selection)
        audit['errors'].extend(result['errors'])
        if branch is not None:
            table = Path(str(stem)+'_bp_metrics.csv')
            write_zoom_table(table, source, branch, parameters, code_id, selection, result['segments'])
            for p in (table, Path(str(table)+'.meta.json')):
                audit['artifacts'][p.name] = file_sha256(p)
            audit.update(candidate_windows=len(branch['valid']), accepted_windows=int(branch['valid'].sum()),
                         quality_diagnostics=diagnostic_summary(branch['window_audit']),
                         window_audit=branch['window_audit'])
        outpath = Path(str(stem)+'.png')
        draw_zoom(t, raw, branch, selection, name, info, ds, outpath)
        for p in (outpath, outpath.with_suffix('.svg')):
            audit['artifacts'][p.name] = file_sha256(p)
        audit['status'] = 'complete' if selection['status'] == 'computed' and not audit['errors'] else 'excluded'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        audit_path.write_text(json.dumps(json_safe(audit), indent=2, allow_nan=False)+'\n')
    if audit['status'] != 'complete':
        raise ValueError(f'Event zoom unavailable: {selection["status"]}; audit saved: {audit_path}')
    print(f'Saved event zoom and source audit: {stem}')
    return audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='Hsin', choices=list(pem.SUBJECTS))
    parser.add_argument('--event', default='Mindfulness Meditation')
    parser.add_argument('--ds', type=int, default=10, help='Display stride within each source run; must be positive.')
    parser.add_argument('--csv', help='Override source recording, retaining the subject event schedule.')
    parser.add_argument('--outdir', default=str(Path(pem.IBRAIN_DIR)/'event_verification'))
    return parser.parse_args()


def main():
    args = parse_args()
    info = pem.SUBJECTS[args.subject]
    source = args.csv or Path(pem.IBRAIN_DIR)/info['dir']/'merged.csv'
    run_zoom(args.subject, info, source, args.outdir, args.event, args.ds)


if __name__ == '__main__':
    main()
