"""Validate raw-quality callers against the committed pre-audit implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import plot_event_markers as entry
import plot_meditation_zoom as zoom
from lilia import quality
from lilia.event_qeeg import analyze_recording, summarize_branch, INDEX_KEYS
from lilia.event_qeeg_io import write_event_qeeg_table, load_event_qeeg_table, json_safe
from lilia.subject_comparison import summarize_comparison
from lilia.subject_comparison_io import write_comparison_table, load_comparison_table
from lilia.io import load_merged_csv
from lilia.provenance import file_sha256

COMMIT='a0a7ba6'
DEPENDENCIES=('lilia/io.py','lilia/signal.py','lilia/windowing.py','lilia/qeeg.py','lilia/tflite.py','lilia/quality.py')


def numeric(result):
    arrays={}
    for name in ('bp','tflite'):
        branch=result[name]
        if branch is None:continue
        arrays[name+'_quality']=branch['quality']
        arrays[name+'_valid']=branch['valid']
        for key,value in branch['quality_mapping'].items():arrays[name+'_'+key]=value
        for key in INDEX_KEYS:arrays[name+'_'+key]=branch['scores'][key]
        summary=summarize_branch(branch,[])
        arrays[name+'_heatmap_abs']=summary['heatmap_abs']
        arrays[name+'_heatmap_delta']=summary['heatmap_delta']
        for key,value in summary['smooth'].items():arrays[name+'_smooth_'+key]=value
    return arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    out=p.parse_args().out.resolve();out.mkdir(parents=True,exist_ok=False)
    evidence={'schema_version':1,'files':[],'tables':[],'comparisons':[]}
    def remember(path,expected=None):
        evidence['files'].append({'path':str(path),'sha256':expected or file_sha256(path)})
    for name in DEPENDENCIES:
        original=subprocess.check_output(['git','show',f'{COMMIT}:{name}'],cwd=ROOT)
        expected=hashlib.sha256(original).hexdigest()
        if file_sha256(ROOT/name)!=expected:raise ValueError(f'Legacy dependency changed: {name}')
        remember(ROOT/name,expected)
    source=subprocess.check_output(['git','show',f'{COMMIT}:lilia/event_qeeg.py'],cwd=ROOT)
    (out/'legacy_event_qeeg.py.txt').write_bytes(source)
    legacy=types.ModuleType('legacy_event_quality')
    exec(compile(source,'legacy_event_quality','exec'),legacy.__dict__)
    cases=[]
    paths=[path for path in sorted((ROOT/'jenqwei').glob('*/*.csv')) if path.name.lower()!='time_marker.csv']
    if len(paths)!=5:raise ValueError('Expected five inventoried recordings')
    source_cases, derived, source_records = [], [], []
    (out/'inputs').mkdir()
    for original in paths:
        name=original.stem.removeprefix('2026-06-18-lilia-').replace(' ','_').replace('-','_')
        t,raw=load_merged_csv(original)
        copy=out/'inputs'/(name+'_first4.csv')
        with copy.open('x') as handle:
            handle.write('Device,Lilia,stage18 verification copy\nAmp Gain,500\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n')
            pd.DataFrame({'Time[us]':t,**{f'ch{i+1}':raw[:,i] for i in range(4)}}).to_csv(handle,index=False)
        derived.append(copy)
        source_records.append(dict(original=str(original),original_sha256=file_sha256(original),
                                   derived=str(copy),derived_sha256=file_sha256(copy),channels=[1,2,3,4],
                                   policy='first four declared channels; unchanged int64 timestamps and float32 values'))
        source_cases.extend([(name+'_first4',copy,True),(name+'_all8',original,False)])
    (out/'source_inventory.json').write_text(json.dumps(source_records,indent=2)+'\n')
    for name,source,use_model in source_cases:
        t,raw=load_merged_csv(source)
        args=dict(scorer=quality.get_eeg_quality_index_v2_parametric,quality_params=entry.QUALITY_PARAMS,
                  model_path=entry.TFLITE_MODEL_PATH if use_model else None)
        before=legacy.analyze_recording(t,raw,**args)
        result=analyze_recording(t,raw,**args)
        old,new=numeric(before),numeric(result)
        if old.keys()!=new.keys():raise ValueError('Legacy numeric keys differ')
        for key in old:np.testing.assert_array_equal(new[key],old[key])
        actual,expected=out/(name+'_actual.npz'),out/(name+'_expected.npz')
        np.savez_compressed(actual,**new);np.savez_compressed(expected,**old)
        evidence['comparisons'].append(dict(actual=str(actual),expected=str(expected),rtol=0,atol=0,equal_nan=True))
        for tag in ('bp','tflite'):
            branch=result[tag]
            if branch is None:continue
            params=dict(input_fs=500.,fs=branch['grid'].fs,win_sec=5.,step_sec=5.,
                        index_space='raw_samples' if tag=='bp' else 'retained_tflite_output',
                        quality_params=entry.QUALITY_PARAMS,quality_channels=raw.shape[1],
                        metric_channels=branch['scores']['focus'].shape[1],quality_threshold=.5,
                        baseline_mode='session-start',events=[],model_window=400,
                        model_sha256=file_sha256(entry.TFLITE_MODEL_PATH))
            timeline=result['timeline'] if tag=='tflite' else None
            branch['summary']=summarize_branch(branch,[])
            table=out/(name+'_'+tag+'.csv')
            write_event_qeeg_table(table,source,branch,params,'stage18',timeline)
            frame,_=load_event_qeeg_table(table,source,entry.TFLITE_MODEL_PATH)
            evidence['tables'].append(dict(path=str(table),raw_csv=str(source),kind='event_marker_'+tag,
                                           model_path=str(Path(entry.TFLITE_MODEL_PATH).resolve())))
            branch['summary']=summarize_comparison(branch)
            comparison=out/(name+'_'+tag+'_comparison.csv')
            write_comparison_table(comparison,source,branch,{**params,'events':None},'stage18',result['segments'],timeline)
            load_comparison_table(comparison,source,entry.TFLITE_MODEL_PATH)
            evidence['tables'].append(dict(path=str(comparison),raw_csv=str(source),kind='subject_comparison_'+tag,
                                           model_path=str(Path(entry.TFLITE_MODEL_PATH).resolve())))
            cases.append(dict(source=name,branch=tag,windows=len(frame),accepted=int(branch['valid'].sum()),
                              diagnostic_states=frame.quality_diagnostic_state.value_counts().to_dict(),
                              comparison_read=True))
        remember(source)
        print(name+': numeric arrays, selection and four table reads passed',flush=True)
    plotroot=out/'plot_sources';(plotroot/'real').mkdir(parents=True)
    shutil.copy2(derived[1],plotroot/'real/merged.csv')
    info={'dir':'real','sn':'stage18'}
    for name,scorer,params in [('real',quality.get_eeg_quality_index_v2_parametric,entry.QUALITY_PARAMS),
                              ('fallback',fallback,{k+'_weight':float(k=='spectrum') for k in ('flat','spectrum','kurtosis','corr')})]:
        folder=out/('plot_'+name);folder.mkdir()
        with patch.object(entry,'get_eeg_quality_index_v2_parametric',scorer),patch.object(entry,'QUALITY_PARAMS',params):
            entry.plot_subject('Stage18',info,str(folder),500,base_dir=str(plotroot),with_events=False,use_tflite=False)
        for table in folder.glob('*_metrics.csv'):
            load_event_qeeg_table(table,plotroot/'real/merged.csv')
            evidence['tables'].append(dict(path=str(table),raw_csv=str(plotroot/'real/merged.csv'),kind='event_marker_bp'))
    # Exercise zoom publication and its stricter source reader on the same real copy.
    from lilia.event_zoom_io import load_zoom_table
    folder=out/'zoom';folder.mkdir()
    event=dict(start_us=30000000,end_us=60000000,label='Stage18',participates=True,color='green')
    zoom.run_zoom('Stage18',info,plotroot/'real/merged.csv',folder,event='Stage18',events=[event])
    for table in folder.glob('*_metrics.csv'):
        load_zoom_table(table,plotroot/'real/merged.csv')
        evidence['tables'].append(dict(path=str(table),raw_csv=str(plotroot/'real/merged.csv'),kind='event_marker_bp'))
    summary=dict(status='passed',scope='event qEEG, zoom and subject comparison raw-quality caller increment',
                 cases=cases,max_abs_error=0,changed_selected_windows=0,
                 policy='legacy_overall; validity diagnostics do not change threshold masks',
                 comparison_tables_read=15,event_tables_read=17,zoom_tables_read=1)
    (out/'analysis.json').write_text(json.dumps(json_safe(summary),indent=2)+'\n')
    for path in sorted(out.rglob('*')):
        if path.is_file():remember(path)
    remember(Path(__file__))
    for name in ('quality_audit','event_qeeg','event_qeeg_io','subject_comparison_io'):
        remember(ROOT/'lilia'/f'{name}.py')
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')


def fallback(data,**kwargs):
    with patch.object(quality.sp_signal,'welch',side_effect=RuntimeError('stage18 forced diagnostic fallback')):
        return quality.get_eeg_quality_index_v2_parametric(data,**kwargs)


if __name__=='__main__':main()
