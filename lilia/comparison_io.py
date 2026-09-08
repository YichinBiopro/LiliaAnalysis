"""Source-verifiable APP/NUC sample, qEEG, and elapsed-alignment tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.entropy_io import _write_window_table, _load_window_table, config_id
from lilia.io import read_lilia_frame
from lilia.neural import build_inference_timeline
from lilia.comparison import pair_by_elapsed
from lilia.provenance import file_sha256


def sample_columns(timeline):
    fractional = np.concatenate([r['raw_start_idx']+np.arange(r['resampled_samples'])*r['ratio_down']/r['ratio_up']
                                 for r in timeline.segments if r['status']=='retained'])
    return {'output_idx':np.arange(len(timeline.time_us)), 'time_us':timeline.time_us,
            'time_s':(timeline.time_us-timeline.source_epoch_us)/1e6,
            'segment_id':timeline.segment_ids, 'raw_fractional_idx':fractional}


def _write(path, frame, metadata):
    frame.to_csv(path,index=False)
    metadata = {**metadata,'table_sha256':file_sha256(path)}
    Path(str(path)+'.meta.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')


def _read(path, kind):
    meta=json.loads(Path(str(path)+'.meta.json').read_text())
    if meta.get('kind')!=kind or meta.get('schema_version')!=1 or file_sha256(path)!=meta['table_sha256'] or config_id(meta['parameters'])!=meta['config_id']:
        raise ValueError('APP/NUC table or configuration fingerprint mismatch')
    return pd.read_csv(path),meta


def _timeline(meta, raw_csv, model_path=None):
    p=meta['parameters']
    if p['index_space']!='resampled_model_output':
        raise ValueError('APP/NUC index space mismatch')
    if model_path is not None and file_sha256(model_path)!=p['model']['checkpoint_sha256']:
        raise ValueError('APP/NUC checkpoint differs from source')
    if file_sha256(raw_csv)!=meta['source_id']:
        raise ValueError('Raw recording differs from APP/NUC source')
    raw=read_lilia_frame(raw_csv)
    t=raw.iloc[:,0].to_numpy(dtype=np.int64)
    timeline=build_inference_timeline(t,p['input_fs'],p['fs'],p['model_window'],p['model_hop'])
    if len(raw.columns)<5 or timeline.metadata()!=meta['inference']:
        raise ValueError('APP/NUC inference mapping differs from raw timestamps')
    return timeline


def _check_columns(frame, expected):
    for key,values in expected.items():
        actual=frame[key].to_numpy()
        same = len(actual)==len(values) and (np.allclose(actual,values,rtol=0,atol=1e-9)
               if np.asarray(values).dtype.kind=='f' else np.array_equal(actual,values))
        if not same:
            raise ValueError(f'APP/NUC {key} mapping differs from source')


def write_recording(outdir, tag, source, record, parameters, code_id):
    timeline=record['timeline']
    frame=pd.DataFrame(sample_columns(timeline))
    for ch in range(4):frame[f'filtered_ch{ch+1}']=record['input'][:,ch]
    for ch in range(2):frame[f'model_ch{ch+1}']=record['output'][:,ch]
    frame['quality_state']='disabled'
    meta={'kind':'app_nuc_signal','schema_version':1,'source_id':file_sha256(source),
          'source_path':str(Path(source).resolve()),'parameters':parameters,'config_id':config_id(parameters),
          'code_sha256':code_id,'inference':timeline.metadata(),'artifact_repair':record['repair']}
    path=Path(outdir)/f'{tag}_signal.csv'
    _write(path,frame,meta)
    paths=[path,Path(str(path)+'.meta.json')]
    grid=record['qeeg_grid']
    if grid is not None:
        path=Path(outdir)/f'{tag}_qeeg.csv'
        frame=pd.DataFrame(record['qeeg'])
        frame['quality_state']='disabled'
        _write_window_table(path,source,frame,grid,parameters,code_id,'app_nuc_qeeg',
            source_info={'source_samples':timeline.source_samples,'source_epoch_us':timeline.source_epoch_us,
                         'inference':timeline.metadata()})
        paths.extend([path,Path(str(path)+'.meta.json')])
    return paths


def load_signal_table(path, raw_csv, model_path=None):
    frame,meta=_read(path,'app_nuc_signal')
    timeline=_timeline(meta,raw_csv,model_path)
    _check_columns(frame,sample_columns(timeline))
    return frame,meta


def load_qeeg_table(path, raw_csv, model_path=None):
    frame,meta=_load_window_table(path,kind='app_nuc_qeeg')
    if meta is None:raise ValueError('APP/NUC qEEG metadata required')
    timeline=_timeline(meta,raw_csv,model_path)
    _check_columns(frame,timeline.grid(meta['parameters']['win_sec']).columns)
    return frame,meta


def write_pair_table(path, a, b, ia, ib, groups, audit, source_a, source_b, parameters):
    ta,tb=a['timeline'],b['timeline']
    cols={'app_output_idx':ia,'nuc_output_idx':ib,'pair_segment_id':groups,
          'app_time_us':ta.time_us[ia],'nuc_time_us':tb.time_us[ib],
          'time_s':(ta.time_us[ia]-ta.source_epoch_us)/1e6,
          'residual_us':(ta.time_us[ia]-ta.source_epoch_us)-(tb.time_us[ib]-tb.source_epoch_us+audit['lag_us'])}
    meta={'kind':'app_nuc_alignment','schema_version':1,'parameters':parameters,'config_id':config_id(parameters),
          'alignment':audit,'sources':[{'source_id':file_sha256(p),'parameters':parameters,'inference':tl.metadata()}
                                     for p,tl in [(source_a,ta),(source_b,tb)]]}
    _write(path,pd.DataFrame(cols),meta)


def load_pair_table(path, source_a, source_b):
    frame,meta=_read(path,'app_nuc_alignment')
    if len(meta['sources']) != 2 or any(s['parameters'] != meta['parameters'] for s in meta['sources']):
        raise ValueError('APP/NUC source configuration mismatch')
    ta,tb=[_timeline(m,p) for m,p in zip(meta['sources'],[source_a,source_b])]
    audit=meta['alignment']
    if audit['lag_us']!=int(round(audit['lag_samples']*1e6/ta.fs_out)):
        raise ValueError('APP/NUC lag unit mismatch')
    ia,ib,groups=pair_by_elapsed(ta,tb,audit['lag_samples'])
    if (audit['paired_samples'] != len(ia) or audit['paired_runs'] != len(np.unique(groups))
            or audit['unmatched_app_samples'] != len(ta.time_us)-len(ia)
            or audit['unmatched_nuc_samples'] != len(tb.time_us)-len(ib)):
        raise ValueError('APP/NUC alignment counts mismatch')
    ra,rb,rg=pair_by_elapsed(ta,tb)
    runs=[np.flatnonzero(rg==g) for g in np.unique(rg)]
    if not runs:
        raise ValueError('APP/NUC lag reference is absent')
    ref=max(runs,key=len)
    if (audit['reference_samples'] != len(ref) or audit['reference_channel'] != 1
            or audit['reference_app_output_range'] != [int(ra[ref[0]]),int(ra[ref[-1]])+1]
            or audit['reference_nuc_output_range'] != [int(rb[ref[0]]),int(rb[ref[-1]])+1]):
        raise ValueError('APP/NUC lag reference mapping mismatch')
    expected={'app_output_idx':ia,'nuc_output_idx':ib,'pair_segment_id':groups,
        'app_time_us':ta.time_us[ia],'nuc_time_us':tb.time_us[ib],
        'time_s':(ta.time_us[ia]-ta.source_epoch_us)/1e6,
        'residual_us':(ta.time_us[ia]-ta.source_epoch_us)-(tb.time_us[ib]-tb.source_epoch_us+audit['lag_us'])}
    _check_columns(frame,expected)
    return frame,meta
