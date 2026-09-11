"""
compare_subjects.py
===================
Inter-subject comparison of BP delta index and TFLite delta index.

Two groups are analysed independently:
  • iBrainCenter: Ann, Hsin, Hardy, TYY, James  (8 event blocks)
  • YoGa:         James, Jammie, TYY            (no events → whole-session delta)

Outputs (per group, written to <group>/comparison/):
  {group}_bp_delta_comparison.png
  {group}_tflite_delta_comparison.png
  {group}_combined_comparison.png   — BP (solid) vs TFLite (hatched) per subject

Usage
-----
    python compare_subjects.py [--ibrain-outdir DIR] [--yoga-outdir DIR]
"""

from __future__ import annotations

import argparse
import datetime
import os
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from lilia.io import read_lilia_frame
from lilia.event_qeeg import analyze_recording
from lilia.event_qeeg_io import json_safe
from lilia.subject_comparison import summarize_comparison, legacy_results
from lilia.quality_audit import diagnostic_summary
from lilia.subject_comparison_io import write_comparison_table
from lilia.provenance import file_sha256
from lilia.time_utils import local_dt_to_utc_us
from plot_event_markers import (
    SUBJECTS, YOGA_SUBJECTS, EVENTS, EVT_COLORS,
    IBRAIN_DIR, YOGA_DIR,
    FS, TFLITE_FS, TFLITE_MODEL_PATH,
    QUALITY_WIN_SEC, QUALITY_THRESHOLD,
    hhmm_to_dt,
    QUALITY_PARAMS, TFLITE_WIN, get_eeg_quality_index_v2_parametric,
)

# ── Constants ──────────────────────────────────────────────────────────────────
INDEX_KEYS  = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS  = ['Focus', 'Flow', 'Calm', 'Relaxation']
IDX_COLORS  = {
    'focus':       '#e6194b',
    'flow':        '#3cb44b',
    'calm':        '#4363d8',
    'relaxation':  '#f58231',
}

SUBJ_COLORS_IBRAIN = {
    'Ann':   '#1f77b4',
    'Hsin':  '#ff7f0e',
    'Hardy': '#2ca02c',
    'TYY':   '#d62728',
    'James': '#9467bd',
}

SUBJ_COLORS_YOGA = {
    'James':  '#1f77b4',
    'Jammie': '#ff7f0e',
    'TYY':    '#2ca02c',
}

# ── Event helpers ──────────────────────────────────────────────────────────────

def _build_participating_events(name: str) -> list[tuple]:
    """Return list of (start_dt, end_dt, label, color) for events *name* joins."""
    events = []
    for idx, (label, start_hhmm, dur_min, participants) in enumerate(EVENTS):
        if participants is None or name in participants:
            start_dt = hhmm_to_dt(start_hhmm)
            end_dt   = start_dt + datetime.timedelta(minutes=dur_min)
            color    = EVT_COLORS[idx % len(EVT_COLORS)]
            events.append((start_dt, end_dt, label, color))
    return events


def _all_event_labels() -> list[str]:
    return [label for label, *_ in EVENTS]


# ── Per-subject delta computation ──────────────────────────────────────────────

def _compute_block_deltas(
    qeeg_dt: list,
    qeeg_scores: dict,
    qual_mask: np.ndarray,
    participating_events: list,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    For each participating event block compute Δ vs baseline.

    Returns { event_label -> { index_key -> (mean_delta, std_delta) } }
    """
    if not participating_events or not qeeg_dt:
        return {}

    t_arr     = np.array(qeeg_dt)
    good_qual = ~qual_mask

    first_evt_dt  = min(e[0] for e in participating_events)
    baseline_mask = (t_arr < first_evt_dt) & good_qual

    result = {}
    for (start_dt, end_dt, label, _color) in participating_events:
        block_mask = (t_arr >= start_dt) & (t_arr < end_dt) & good_qual
        per_index  = {}
        for k in INDEX_KEYS:
            scores = qeeg_scores[k]   # shape (n_windows, n_ch)
            if baseline_mask.sum() > 0 and block_mask.sum() > 0:
                bl   = scores[baseline_mask].mean(axis=0)   # (n_ch,)
                blk  = scores[block_mask].mean(axis=0)
                ch_d = blk - bl
                per_index[k] = (float(ch_d.mean()), float(ch_d.std()))
            else:
                per_index[k] = (float('nan'), 0.0)
        result[label] = per_index

    return result


def _compute_session_delta(
    qeeg_dt: list,
    qeeg_scores: dict,
    qual_mask: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """
    Whole-session delta using first 20 % as baseline.
    Returns { index_key -> (mean_delta, std_delta) }
    """
    if not qeeg_dt:
        return {}

    t_arr     = np.array(qeeg_dt)
    n         = len(t_arr)
    good_qual = ~qual_mask

    bl_mask  = np.zeros(n, dtype=bool)
    bl_mask[:max(1, n // 5)] = True
    bl_mask &= good_qual
    sess_mask = ~bl_mask & good_qual

    result = {}
    for k in INDEX_KEYS:
        scores = qeeg_scores[k]
        if bl_mask.sum() > 0 and sess_mask.sum() > 0:
            bl   = scores[bl_mask].mean(axis=0)
            sess = scores[sess_mask].mean(axis=0)
            ch_d = sess - bl
            result[k] = (float(ch_d.mean()), float(ch_d.std()))
        else:
            result[k] = (float('nan'), 0.0)
    return result


def process_subject(name, info, base_dir, events, outdir=None, use_tflite=True):
    """Keep successful BP output when a requested model or another subject fails."""
    source = Path(base_dir)/info['dir']/'merged.csv'
    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
    audit = {'kind':'subject_comparison','schema_version':1,'subject':name,'source_path':str(source.resolve()),
             'status':'processing','errors':[],'artifacts':{},'branches':{},'events':events}
    results = {'bp':{},'tflite':{}}
    statuses = {'bp':{},'tflite':{}}
    try:
        audit['source_id'] = file_sha256(source)
        raw = read_lilia_frame(source)
        t = raw.iloc[:,0].to_numpy(dtype=np.int64)
        data = raw.iloc[:,1:].to_numpy(dtype=np.float32)
        model_exists = Path(TFLITE_MODEL_PATH).is_file()
        if use_tflite and not model_exists:
            audit['errors'].append(f'Requested TFLite model missing: {TFLITE_MODEL_PATH}')
        paths = [Path(__file__),Path(__file__).with_name('plot_event_markers.py'),
                 *sorted((Path(__file__).parent/'lilia').glob('*.py'))]
        code_id = hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
        audit.update(code_sha256=code_id,source_samples=len(t),source_epoch_us=int(t[0]))
        analysis = analyze_recording(t,data,fs=FS,win_sec=QUALITY_WIN_SEC,
            scorer=get_eeg_quality_index_v2_parametric,quality_params=QUALITY_PARAMS,threshold=QUALITY_THRESHOLD,
            model_path=TFLITE_MODEL_PATH if use_tflite and model_exists else None,
            model_fs=TFLITE_FS,model_window=TFLITE_WIN)
        audit['segments'] = analysis['segments']
        audit['errors'].extend(analysis['errors'])
        if analysis['timeline'] is not None:
            audit['inference'] = analysis['timeline'].metadata()
        for tag in ('bp','tflite'):
            branch = analysis[tag]
            if branch is None:
                state = 'disabled' if tag=='tflite' and not use_tflite else 'unavailable'
                audit['branches'][tag] = {'status':state}
                labels = ['Session'] if events is None else [e['label'] for e in events]
                statuses[tag] = {label:state for label in labels}
                if state!='disabled' and not audit['errors']:
                    audit['errors'].append(f'{tag}: no complete metric windows')
                continue
            branch['summary'] = summarize_comparison(branch,events)
            results[tag] = legacy_results(branch['summary'])
            statuses[tag] = {r['label']:r['status'] for r in branch['summary']}
            params = {'input_fs':FS,'fs':branch['grid'].fs,'win_sec':QUALITY_WIN_SEC,'step_sec':QUALITY_WIN_SEC,
                'index_space':'raw_samples' if tag=='bp' else 'retained_tflite_output',
                'quality_channels':data.shape[1],'metric_channels':branch['quality'].shape[1] if tag=='bp' else 2,
                'bandpass':[.5,45.],'quality_params':QUALITY_PARAMS,'quality_threshold':QUALITY_THRESHOLD,
                'quality_scope':'raw samples inside each metric window','model_window':TFLITE_WIN,
                'model_sha256':file_sha256(TFLITE_MODEL_PATH) if tag=='tflite' else None,
                'events':events,'session_baseline':'first max(1,N//5) candidate metric rows before quality mask',
                'event_baseline':'complete windows before first participating event',
                'aggregation':'per-channel window mean difference; mean and population SD across channels',
                'model_policy':'segment-local polyphase; complete nonoverlapping RMS windows; per-segment tail trim'}
            summary = branch['summary']
            audit['branches'][tag] = {'status':'computed','parameters':params,
                'candidate_windows':len(branch['valid']),'accepted_windows':int(branch['valid'].sum()),
                'window_audit':branch['window_audit'],'summary':summary,
                'quality_diagnostics':diagnostic_summary(branch['window_audit'])}
            timeline = analysis['timeline'] if tag=='tflite' else None
            if timeline is not None:
                audit['branches'][tag]['inference'] = timeline.metadata()
            if outdir is not None:
                path = Path(outdir)/f'{tag}_metrics.csv'
                write_comparison_table(path,source,branch,params,code_id,analysis['segments'],timeline)
                for p in (path,Path(str(path)+'.meta.json')):
                    audit['artifacts'][p.name] = file_sha256(p)
            if any(r['participates'] for r in summary) and not any(r['status']=='computed' for r in summary):
                audit['errors'].append(f'{tag}: no valid baseline/comparison pair')
        audit['status'] = 'failed' if audit['errors'] else 'complete'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['errors'].append(f'{type(exc).__name__}: {exc}')
    if outdir is not None:
        (Path(outdir)/'analysis.json').write_text(json.dumps(json_safe(audit),indent=2,allow_nan=False)+'\n')
    return results,statuses,audit


def extract_subject_deltas(name, info, base_dir, participating_events, *, outdir=None, use_tflite=True):
    """Compatibility dict-of-tuples adapter using the segmented analysis path."""
    events = None if participating_events is None else [
        {'label':label,'start_us':local_dt_to_utc_us(start,8),'end_us':local_dt_to_utc_us(end,8),'participates':True}
        for start,end,label,_ in participating_events]
    results,_,audit = process_subject(name,info,base_dir,events,outdir,use_tflite)
    if audit['errors']:
        raise ValueError('; '.join(audit['errors']))
    return results['bp'],results['tflite'] if use_tflite else None


def _missing_bar(ax, x, participated, status=None, side=None):
    label = 'NP' if not participated else ('OFF' if status=='disabled' else 'NA')
    if side:
        label = f'{side}:{label}'
    ax.text(x,.03,label,ha='center',va='bottom',fontsize=6,rotation=90 if side else 0,
            transform=ax.get_xaxis_transform(),color='#555555')

# ── Plotting ───────────────────────────────────────────────────────────────────

def _bar_cell(
    ax: plt.Axes,
    subjects: list[str],
    evt_label: str,
    idx_name: str,
    signal_data: dict,        # { subj -> { evt_label -> { idx -> (m, s) } } }
    subject_colors: dict,
    participation: dict | None,   # { subj -> set_of_event_labels } or None
    show_xticklabels: bool = True,
    status_data=None,
) -> None:
    """Draw one grouped-bar cell (subjects on x-axis, one bar per subject)."""
    vals, errs, colors, alphas = [], [], [], []
    for subj in subjects:
        evt_dict = signal_data.get(subj) or {}
        per_idx  = evt_dict.get(evt_label) or {}
        m, s = per_idx.get(idx_name, (float('nan'), 0.0))
        participated = (participation is None
                        or subj not in participation
                        or evt_label in participation[subj])
        vals.append(m)
        errs.append(s)
        colors.append(subject_colors.get(subj, '#888888'))
        alphas.append(0.85 if participated else 0.20)

    x = np.arange(len(subjects))
    for xi, (v, e, c, a) in enumerate(zip(vals, errs, colors, alphas)):
        participated = participation is None or subjects[xi] not in participation or evt_label in participation[subjects[xi]]
        if not participated or not np.isfinite(v) or not np.isfinite(e):
            state = ((status_data or {}).get(subjects[xi]) or {}).get(evt_label)
            _missing_bar(ax,xi,participated,state)
        else:
            ax.bar(xi, v, width=0.6, color=c, alpha=a,
                   yerr=e, capsize=3, error_kw={'lw': 0.8}, zorder=3)
    ax.axhline(0, color='k', lw=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xlim(-0.6, len(subjects) - 0.4)
    if show_xticklabels:
        ax.set_xticklabels(subjects, rotation=35, ha='right', fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.grid(True, alpha=0.2, axis='y')


def plot_single_signal_comparison(
    group_label: str,
    subjects: list[str],
    event_labels: list[str],
    signal_data: dict,
    subject_colors: dict,
    participation: dict | None,
    signal_tag: str,           # 'BP' or 'TFLite'
    ylabel_prefix: str,
    outpath: str,
    status_data=None,
) -> None:
    """
    4 rows (indices) × N_evt columns.
    Each cell: bars per subject for that (index, event) combination.
    """
    n_idx = len(INDEX_KEYS)
    n_evt = len(event_labels)

    cell_w = max(1.6, len(subjects) * 0.45 + 0.8)
    fig, axes = plt.subplots(
        n_idx, n_evt,
        figsize=(cell_w * n_evt + 1.0, 3.2 * n_idx + 1.2),
        sharey='row',
    )
    if n_idx == 1:
        axes = axes[np.newaxis, :]
    if n_evt == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f'{group_label} — Inter-subject Comparison  [{signal_tag} Δ vs baseline]\n'
        f'Error bars = SD across channels | NA = unavailable, NP = not participating, OFF = disabled',
        fontsize=12, fontweight='bold',
    )

    for row_i, (idx_name, idx_label) in enumerate(zip(INDEX_KEYS, IDX_LABELS)):
        for col_j, evt_label in enumerate(event_labels):
            ax = axes[row_i, col_j]
            show_xtick = (row_i == n_idx - 1)
            _bar_cell(ax, subjects, evt_label, idx_name,
                      signal_data, subject_colors, participation,
                      show_xticklabels=show_xtick, status_data=status_data)
            if col_j == 0:
                ax.set_ylabel(f'{idx_label}\n{ylabel_prefix} Δ', fontsize=8)
            if row_i == 0:
                ax.set_title(evt_label.replace(' ', '\n'), fontsize=8,
                             fontweight='bold')

    # Subject legend
    legend_handles = [
        mpatches.Patch(color=subject_colors.get(s, '#888'), label=s)
        for s in subjects
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(subjects),
               fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.9,
               title='Subjects')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'  Saved: {outpath}')


def plot_combined_comparison(
    group_label: str,
    subjects: list[str],
    event_labels: list[str],
    bp_data: dict,
    tfl_data: dict,
    subject_colors: dict,
    participation: dict | None,
    outpath: str,
    bp_status=None, tfl_status=None,
) -> None:
    """
    N_events rows × 4 columns (one per qEEG index).
    Each cell: grouped bars where each subject has a pair of bars:
      solid  = BP delta
      hatched = TFLite delta
    """
    n_evt = len(event_labels)
    n_idx = len(INDEX_KEYS)
    n_subj = len(subjects)

    cell_w = max(2.0, n_subj * 0.7 + 0.8)
    fig, axes = plt.subplots(
        n_evt, n_idx,
        figsize=(cell_w * n_idx + 1.0, 2.8 * n_evt + 1.4),
        sharey='row',
    )
    if n_evt == 1:
        axes = axes[np.newaxis, :]
    if n_idx == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f'{group_label} — BP vs TFLite Inter-subject Comparison\n'
        f'Solid = BP Δ | Hatched = TFLite Δ | Error bars = channel SD | NA / NP / OFF = unavailable / not participating / disabled',
        fontsize=12, fontweight='bold',
    )

    bar_w = 0.35
    x_off = np.array([-bar_w / 2, bar_w / 2])   # BP left, TFLite right

    for row_i, evt_label in enumerate(event_labels):
        for col_j, (idx_name, idx_label) in enumerate(zip(INDEX_KEYS, IDX_LABELS)):
            ax = axes[row_i, col_j]
            x_pos = np.arange(n_subj) * (2 * bar_w * 1.6)

            for s_i, subj in enumerate(subjects):
                clr = subject_colors.get(subj, '#888888')
                participated = (participation is None
                                or subj not in participation
                                or evt_label in participation[subj])
                alpha = 0.85 if participated else 0.18

                bp_val,  bp_err  = ((bp_data or {}).get(subj) or {}).get(evt_label, {}).get(idx_name, (float('nan'), 0.0))
                tfl_val, tfl_err = ((tfl_data or {}).get(subj) or {}).get(evt_label, {}).get(idx_name, (float('nan'), 0.0))

                if participated and np.isfinite(bp_val) and np.isfinite(bp_err):
                    ax.bar(x_pos[s_i] + x_off[0], bp_val, width=bar_w,
                           color=clr, alpha=alpha, yerr=bp_err, capsize=2.5,
                           error_kw={'lw': 0.8}, zorder=3)
                else:
                    _missing_bar(ax,x_pos[s_i]+x_off[0],participated,((bp_status or {}).get(subj) or {}).get(evt_label),'B')
                if participated and np.isfinite(tfl_val) and np.isfinite(tfl_err):
                    ax.bar(x_pos[s_i] + x_off[1], tfl_val, width=bar_w,
                           color=clr, alpha=alpha * 0.6, hatch='//',
                           edgecolor=clr, linewidth=0.5,
                           yerr=tfl_err, capsize=2.5,
                           error_kw={'lw': 0.8}, zorder=3)
                else:
                    _missing_bar(ax,x_pos[s_i]+x_off[1],participated,((tfl_status or {}).get(subj) or {}).get(evt_label),'T')

            ax.axhline(0, color='k', lw=0.7, zorder=4)
            show_xtick = (row_i == n_evt - 1)
            ax.set_xticks(x_pos)
            ax.set_xlim(-2 * bar_w, x_pos[-1] + 2 * bar_w)
            if show_xtick:
                ax.set_xticklabels(subjects, rotation=35, ha='right', fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.2, axis='y')

            if col_j == 0:
                ax.set_ylabel(
                    evt_label.replace(' ', '\n') + '\nΔ Index', fontsize=7.5)
            if row_i == 0:
                ax.set_title(idx_label, fontsize=9, fontweight='bold',
                             color=IDX_COLORS[idx_name])

    # Legend: subjects + BP/TFLite indicator
    legend_handles = [
        mpatches.Patch(color=subject_colors.get(s, '#888'), label=s)
        for s in subjects
    ]
    legend_handles += [
        mpatches.Patch(facecolor='grey', alpha=0.85, label='BP (solid)'),
        mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//',
                       edgecolor='grey', label='TFLite (hatched)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(subjects) + 2, fontsize=8,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f'  Saved: {outpath}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Inter-subject comparison of BP / TFLite qEEG delta indices.')
    p.add_argument('--ibrain-outdir',
                   default=os.path.join(IBRAIN_DIR, 'comparison'),
                   metavar='DIR')
    p.add_argument('--yoga-outdir',
                   default=os.path.join(YOGA_DIR, 'comparison'),
                   metavar='DIR')
    p.add_argument('--no-tflite', action='store_true', help='Explicitly disable model comparison.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    failures = []
    for group_label, registry, base_dir, outdir, use_events in [
        ('iBrainCenter', SUBJECTS, IBRAIN_DIR, args.ibrain_outdir, True),
        ('YoGa', YOGA_SUBJECTS, YOGA_DIR, args.yoga_outdir, False),
    ]:
        Path(outdir).mkdir(parents=True,exist_ok=True)
        group = {'kind':'subject_comparison_group','schema_version':1,'group':group_label,
                 'status':'processing','subjects':{},'artifacts':{},'errors':[]}
        try:
            labels = _all_event_labels() if use_events else ['Session']
            participation = {name:{label for label,_,_,parts in EVENTS if parts is None or name in parts}
                             for name in registry} if use_events else None
            bp_all,tfl_all,bp_status,tfl_status = {},{},{},{}
            for name,info in registry.items():
                events = None
                if use_events:
                    events = [{'label':label,'start_us':local_dt_to_utc_us(hhmm_to_dt(start),8),
                               'end_us':local_dt_to_utc_us(hhmm_to_dt(start)+datetime.timedelta(minutes=duration),8),
                               'participates':parts is None or name in parts} for label,start,duration,parts in EVENTS]
                results,status,audit = process_subject(name,info,base_dir,events,Path(outdir)/name,not args.no_tflite)
                bp_all[name],tfl_all[name] = results['bp'],results['tflite']
                bp_status[name],tfl_status[name] = status['bp'],status['tflite']
                group['subjects'][name] = {'status':audit['status'],'source_path':audit['source_path'],
                    'analysis_path':f'{name}/analysis.json','analysis_sha256':file_sha256(Path(outdir)/name/'analysis.json'),
                    'errors':audit['errors']}
                group['errors'].extend(f'{name}: {e}' for e in audit['errors'])
                print(f'[{group_label}/{name}] {audit["status"]}',flush=True)
            subjects = list(registry)
            colors = SUBJ_COLORS_IBRAIN if use_events else SUBJ_COLORS_YOGA
            if subjects:
                for tag,data,states in [('bp',bp_all,bp_status),('tflite',tfl_all,tfl_status)]:
                    path = Path(outdir)/f'{group_label.lower()}_{tag}_delta_comparison.png'
                    plot_single_signal_comparison(group_label,subjects,labels,data,colors,participation,
                        'BP' if tag=='bp' else 'TFLite',tag.upper(),str(path),status_data=states)
                    for p in (path,path.with_suffix('.svg')):
                        group['artifacts'][p.name]=file_sha256(p)
                path = Path(outdir)/f'{group_label.lower()}_combined_comparison.png'
                plot_combined_comparison(group_label,subjects,labels,bp_all,tfl_all,colors,participation,str(path),
                                         bp_status=bp_status,tfl_status=tfl_status)
                for p in (path,path.with_suffix('.svg')):
                    group['artifacts'][p.name]=file_sha256(p)
            group['status'] = 'failed' if group['errors'] else 'complete'
        except Exception as exc:
            group['errors'].append(f'{type(exc).__name__}: {exc}')
            group['status'] = 'failed'
        (Path(outdir)/'comparison_analysis.json').write_text(json.dumps(json_safe(group),indent=2,allow_nan=False)+'\n')
        failures.extend(f'{group_label}: {e}' for e in group['errors'])
    if failures:
        raise RuntimeError('Comparison failures (successful outputs retained): ' + '; '.join(failures))
    print('All comparison groups completed.')


if __name__ == '__main__':
    main()
