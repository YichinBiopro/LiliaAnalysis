"""TYY raw Ch1/Ch2 and segmented BP/TFLite meditation heatmaps.

BP metrics retain the historical all-source-channel median (normally four);
model metrics use its two output channels. Quality scoring is disabled.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from lilia.constants import FS, TFLITE_FS, TFLITE_WIN
from lilia.event_qeeg_io import json_safe
from lilia.io import read_lilia_frame
from lilia.meditation import analyze_meditation, meditation_display
from lilia.meditation_io import write_meditation_table
from lilia.pathing import get_project_root
from lilia.provenance import file_sha256
from lilia.time_utils import hhmm_to_local_dt, local_dt_to_utc_us, utc_us_to_local_dt

BASE_DIR = get_project_root()
SESSION_DATE = datetime.date(2026, 5, 12)
TZ_OFFSET_H = 8
SUBJECT_NAME = 'TYY'
SUBJECT_SN = 'SN041'
SUBJECT_DIR = str(Path(BASE_DIR)/'iBrainCenter'/'TYY(SN041)')
MERGED_CSV = str(Path(SUBJECT_DIR)/'merged.csv')
DEFAULT_OUTDIR = str(Path(BASE_DIR)/'iBrainCenter'/'TYY_meditation')
TFLITE_PATH = str(Path(BASE_DIR)/'tiny_v4_optimized.tflite')
QEEG_WIN_SEC = 5.
HEATMAP_BIN_SEC = 30
HEATMAP_DELTA_VABS = .6
SHOW_CHS = [0, 1]
MEDITATION_START_HHMM = '15:06'
MEDITATION_DUR_MIN = 11
BASELINE_END_HHMM = '14:24'
BASELINE_LOAD_HHMM = '14:10'
MEDITATION_COLOR = '#911eb4'
INDEX_KEYS = ['focus', 'flow', 'calm', 'relaxation']
IDX_LABELS = ['Focus', 'Flow', 'Calm', 'Relax']


def hhmm_to_dt(hhmm):
    return hhmm_to_local_dt(hhmm, SESSION_DATE)


def us_to_local_dt(us):
    return utc_us_to_local_dt(us, TZ_OFFSET_H)


def session_parameters():
    start = hhmm_to_dt(MEDITATION_START_HHMM)
    end = start+datetime.timedelta(minutes=MEDITATION_DUR_MIN)
    return {key: local_dt_to_utc_us(value, TZ_OFFSET_H) for key, value in {
        'crop_start_us': hhmm_to_dt(BASELINE_LOAD_HHMM),
        'crop_end_us': end+datetime.timedelta(minutes=2),
        'baseline_end_us': hhmm_to_dt(BASELINE_END_HHMM),
        'view_start_us': start-datetime.timedelta(minutes=6),
        'view_end_us': end+datetime.timedelta(minutes=2),
        'meditation_start_us': start, 'meditation_end_us': end}.items()}


def draw_meditation(t, raw, result, display, params, stem, requested_model):
    tags = ['bp']+(['tflite'] if requested_model else [])
    fig = plt.figure(figsize=(14, 2.2*(2+len(tags))+1.5))
    grid = fig.add_gridspec(2+len(tags), 2, width_ratios=[1, .025], height_ratios=[2.5,2.5]+[2.2]*len(tags))
    axes = []
    for row in range(2+len(tags)):
        axes.append(fig.add_subplot(grid[row,0], sharex=axes[0] if axes else None))
    try:
        for ch, ax in enumerate(axes[:2]):
            for run in display['display_runs']:
                idx = np.asarray(run['display_raw_indices'], dtype=int)
                ax.plot([us_to_local_dt(u) for u in t[idx]], raw[idx,ch], color=['#1f77b4','#ff7f0e'][ch],
                        lw=.5, marker='.' if len(idx)==1 else None)
                lo = max(run['start_us'], params['meditation_start_us'])
                hi = min(run['end_us'], params['meditation_end_us'])
                if lo < hi:
                    ax.axvspan(us_to_local_dt(lo), us_to_local_dt(hi), color=MEDITATION_COLOR, alpha=.15)
            ax.set_ylim(-150,150)
            ax.set_ylabel(f'Ch{ch+1}\n(µV)')
            ax.grid(True,alpha=.2)
        cmap = LinearSegmentedColormap.from_list('OrgPur',['#5e3c99','#f7f7f7','#e66101'])
        cmap.set_bad('#aaaaaa')
        for row,tag in enumerate(tags,2):
            ax,branch = axes[row],result[tag]
            if branch is None:
                status = 'unavailable'
            else:
                summary = branch['summary'];status=summary['status']
                for i in summary['display_bins']:
                    item = summary['bins'][i]
                    edges = mdates.date2num([us_to_local_dt(item['start_us']),us_to_local_dt(item['end_us'])])
                    ax.pcolormesh(edges,np.arange(5)-.5,np.ma.masked_invalid(summary['heatmap_delta'][:,i:i+1]),
                                  cmap=cmap,vmin=-HEATMAP_DELTA_VABS,vmax=HEATMAP_DELTA_VABS,shading='flat')
            if status != 'computed':
                ax.text(.5,.5,f'{tag.upper()}: {status.replace("_"," ")}',ha='center',va='center',transform=ax.transAxes,
                        bbox={'facecolor':'white','alpha':.9})
            ax.set_yticks(range(4),IDX_LABELS);ax.set_ylim(-.5,3.5);ax.set_ylabel(f'{tag.upper()} Δ\nquality disabled')
            cbax=fig.add_subplot(grid[row,1])
            fig.colorbar(ScalarMappable(norm=Normalize(-HEATMAP_DELTA_VABS,HEATMAP_DELTA_VABS),cmap=cmap),cax=cbax,label='Δ vs baseline')
        for ax in axes:
            ax.axvline(us_to_local_dt(params['meditation_start_us']),color=MEDITATION_COLOR,lw=1,ls='--')
            duration_sec=(params['view_end_us']-params['view_start_us'])/1e6
            if 1 <= duration_sec < 180:
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1,int(np.ceil(duration_sec/8)))))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            else:
                locator=mdates.AutoDateLocator(minticks=3,maxticks=10)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator) if duration_sec<1 else mdates.DateFormatter('%H:%M'))
        for ax in axes[:-1]:ax.tick_params(labelbottom=False)
        axes[-1].set_xlim(us_to_local_dt(params['view_start_us']),us_to_local_dt(params['view_end_us']))
        axes[-1].set_xlabel('Local time (UTC+8) | gray = unavailable bin; blank = no complete bin')
        fig.suptitle(f'TYY ({SUBJECT_SN}) — Mindfulness Meditation\n'
                     f'BP: all {raw.shape[1]} source channels | model: two output channels | quality scoring disabled\n'
                     f'Baseline: complete bins before {us_to_local_dt(params["baseline_end_us"]):%H:%M}; raw display: Ch1/Ch2',fontsize=11)
        fig.tight_layout()
        for ext in ('png','svg'):
            fig.savefig(str(stem)+'.'+ext,dpi=150 if ext=='png' else None,bbox_inches='tight')
    finally:
        plt.close(fig)


def plot_tyy_meditation(outdir, ds=500, use_tflite=True, *, csv_path=None, parameters=None, model_path=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    stem=outdir/f'TYY_{SUBJECT_SN}_meditation'
    if use_tflite:stem=Path(str(stem)+'_tflite')
    audit_path=Path(str(stem)+'_analysis.json')
    source=Path(csv_path or MERGED_CSV)
    audit={'kind':'tyy_meditation','schema_version':1,'status':'processing','source_path':str(source.resolve()),
           'errors':[],'artifacts':{},'branches':{},'quality_state':'disabled'}
    try:
        if not isinstance(ds,int) or isinstance(ds,bool) or ds<1:raise ValueError('Display downsample must be a positive integer')
        paths=[Path(__file__),*sorted((Path(__file__).parent/'lilia').glob('*.py'))]
        code_id=hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
        audit.update(source_id=file_sha256(source),code_sha256=code_id)
        raw_frame=read_lilia_frame(source);t=raw_frame.iloc[:,0].to_numpy(dtype=np.int64);raw=raw_frame.iloc[:,1:].to_numpy(dtype=np.float32)
        if raw.shape[1]<2:raise ValueError('Raw display requires at least two channels')
        params=session_parameters() if parameters is None else dict(parameters)
        model=Path(model_path or TFLITE_PATH)
        model_available=use_tflite and model.is_file()
        audit['model']={'requested':use_tflite,'path':str(model.resolve()) if use_tflite else None,
                        'sha256':file_sha256(model) if model_available else None}
        if use_tflite and not model_available:audit['errors'].append(f'Requested model missing: {model}')
        keys=('crop_start_us','crop_end_us','baseline_end_us','view_start_us','view_end_us')
        result=analyze_meditation(t,raw,**{k:params[k] for k in keys},fs=FS,win_sec=QEEG_WIN_SEC,
            model_path=str(model) if model_available else None,model_fs=TFLITE_FS,model_window=TFLITE_WIN)
        result.update(source_samples=len(t),source_epoch_us=int(t[0]))
        display=meditation_display(t,params,ds)
        result['display']=display
        audit.update(parameters=params,display=display,display_downsample=ds,source_samples=len(t),source_epoch_us=int(t[0]),
                     crop=result['crop'],segments=result['segments'])
        audit['errors'].extend(result['errors'])
        if result['timeline'] is not None:audit['inference']=result['timeline'].metadata()
        for tag in ('bp','tflite'):
            branch=result[tag]
            if branch is None:
                audit['branches'][tag]={'status':'disabled' if tag=='tflite' and not use_tflite else 'unavailable'}
                continue
            p={**params,'fs':FS if tag=='bp' else TFLITE_FS,'input_fs':FS,'win_sec':QEEG_WIN_SEC,'step_sec':QEEG_WIN_SEC,
               'source_channels':raw.shape[1],'metric_channels':branch['scores']['focus'].shape[1],
               'display_channels':SHOW_CHS,'display_downsample':ds,
               'index_space':'raw_samples' if tag=='bp' else 'retained_tflite_output','bandpass':[.5,45.],
               'quality_state':'disabled','heatmap_bin_windows':6,'baseline_policy':'complete_bins_before_fixed_boundary',
               'aggregation':'median_of_window_channel_medians','model_window':TFLITE_WIN,
               'model_sha256':file_sha256(model) if tag=='tflite' else None}
            table=Path(str(stem)+f'_{tag}_metrics.csv')
            write_meditation_table(table,source,branch,p,code_id,result,tag)
            for path in (table,Path(str(table)+'.meta.json')):audit['artifacts'][path.name]=file_sha256(path)
            audit['branches'][tag]={'status':branch['summary']['status'],'candidate_windows':len(branch['valid']),
                'finite_windows':int(branch['valid'].sum()),'summary':branch['summary'],'parameters':p}
        draw_meditation(t,raw,result,display,params,stem,use_tflite)
        for ext in ('png','svg'):
            path=Path(str(stem)+'.'+ext);audit['artifacts'][path.name]=file_sha256(path)
        audit['status']='failed' if audit['errors'] else 'complete'
    except Exception as exc:
        audit['status']='failed';audit['errors'].append(f'{type(exc).__name__}: {exc}');raise
    finally:
        audit_path.write_text(json.dumps(json_safe(audit),indent=2,allow_nan=False)+'\n')
    if audit['errors']:raise ValueError('Meditation analysis failed (partial outputs retained): '+'; '.join(audit['errors']))
    print(f'Saved meditation analysis and audit: {stem}')
    return audit


def _parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',default=DEFAULT_OUTDIR)
    p.add_argument('--ds',type=int,default=500)
    p.add_argument('--no-tflite',action='store_true')
    p.add_argument('--csv',help='Override source recording; retain the fixed TYY session schedule.')
    p.add_argument('--model',help='Override the TFLite checkpoint.')
    return p.parse_args()


if __name__=='__main__':
    args=_parse_args()
    plot_tyy_meditation(args.outdir,args.ds,not args.no_tflite,csv_path=args.csv,model_path=args.model)
