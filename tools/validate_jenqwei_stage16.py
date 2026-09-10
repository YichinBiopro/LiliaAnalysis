"""Run stage-16 real/synthetic CLI, table and spectral acceptance in a new directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import stft, welch

from lilia.jenqwei_io import load_signal_table
from lilia.jenqwei_plot import compute_panels
from lilia.provenance import file_sha256
from lilia.tflite import build_tflite_timeline
from tools.refactor_check import compare_arrays
import analyze_jenqwei_pipeline as entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'MPLCONFIGDIR': str(out/'mpl'), 'MPLBACKEND': 'Agg', 'TF_CPP_MIN_LOG_LEVEL': '2'}
    env.pop('PYTHONPATH', None)
    commands = []

    def command(name, args, expected=0):
        log = out/f'{name}.log'
        with log.open('w') as handle:
            proc = subprocess.run(args, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        commands.append({'name': name, 'command': args, 'exit_code': proc.returncode, 'expected_exit': expected})
        if proc.returncode != expected:
            raise ValueError(f'{name}: exit {proc.returncode}, expected {expected}; see {log}')
        print(f'{name}: exit={proc.returncode}', flush=True)
        return log

    processing = out/'processing'
    command('processing', [sys.executable, str(ROOT/'tools/validate_jenqwei_adapter_stage16.py'), '--out', str(processing)])
    manifest = json.loads((processing/'evidence.json').read_text())
    manifest['tables'] = []
    analysis = {'scope': 'stage16_final_acceptance', 'commands': commands, 'cases': {}, 'spectra': {},
                'visual_review': 'required_separately', 'legacy_run_pipeline_guard': 'retained'}

    def remember(path, expected=None):
        manifest['files'].append({'path': str(path), 'sha256': expected or file_sha256(path)})

    def compare(name, actual, expected):
        a, b = out/f'{name}_actual.npz', out/f'{name}_expected.npz'
        np.savez_compressed(a, **actual)
        np.savez_compressed(b, **expected)
        remember(a)
        remember(b)
        manifest['comparisons'].append({'actual': str(a), 'expected': str(b), 'rtol': 1e-6, 'atol': 1e-6, 'equal_nan': False})
        return compare_arrays(a, b, rtol=1e-6, atol=1e-6)

    def record_audit(directory, source, success=True):
        path = directory/f'{source.stem}_analysis_audit.json'
        audit = json.loads(path.read_text())
        assert audit['status'] == ('success' if success else 'failed')
        remember(path)
        for item in audit['artifacts']:
            remember(Path(item['path']), item['sha256'])
        return audit

    def recording(name, time, raw):
        path = out/f'{name}.csv'
        with path.open('x') as handle:
            handle.write('File Name,validation\nAmp Gain,500\nChannels,1,2,3,4\nSample Rate,500,500,500,500\nTime[us],ch1,ch2,ch3,ch4\n')
            for t, row in zip(time, raw):
                handle.write(','.join([str(int(t)), *map(str, row)])+'\n')
        remember(path)
        return path

    def cli(name, sources, channels=(0,1), max_sec=60., expected=0):
        directory = out/name
        log = command(name, [sys.executable, str(ROOT/'analyze_jenqwei_pipeline.py'), '--csv', *map(str,sources),
                            '--outdir', str(directory), '--channels', *map(str,channels), '--max-sec', str(max_sec)], expected)
        remember(log)
        return directory

    def verify_case(name, directory, source, expected, channel_count=2):
        audit = record_audit(directory, source)
        assert audit['tables_verified'] and audit['plotting_status']=='complete'
        assert len(audit['artifacts'])==4+channel_count
        frames = {}
        for branch in ('before','after'):
            path = directory/f'{source.stem}_{branch}.csv'
            frame, _ = load_signal_table(path, source, model_path=entry.TFLITE_MODEL_PATH)
            frames[branch] = frame
            manifest['tables'].append({'path': str(path), 'kind': 'jenqwei_signal', 'raw_csv': str(source),
                                       'model_path': entry.TFLITE_MODEL_PATH})
        before, after = frames['before'], frames['after']
        actual = {'time_us_200': before['Time[us]'].to_numpy(), 'tfl_time_us_200': after['Time[us]'].to_numpy(),
                  'pre_data_200': before[['ch1','ch2','ch3','ch4']].to_numpy(),
                  'tfl_data_200': after[['ch1','ch2']].to_numpy()}
        expected_values = {key: expected[key] for key in actual}
        errors = compare(name+'_signals', actual, expected_values)
        np.testing.assert_array_equal(after['Time[us]'], before['Time[us]'].to_numpy()[after['before_idx']])
        analysis['cases'][name] = {'before_rows': len(before), 'after_rows': len(after), 'max_absolute_errors': errors,
                                   'plot_timeline': audit['plot_timeline'], 'spectral_coverage': audit['spectral_coverage']}
        raw_t, _ = entry.load_raw_csv(source)
        result = {**actual, 'timeline': build_tflite_timeline(raw_t), 'segments': audit['segments']}
        return result

    def spectra(name, result, expected, channels, frozen=None, max_sec=60.):
        actual, old = {}, {}
        for ch in channels:
            for panel in compute_panels(result,ch,max_sec):
                branch, sid = panel['branch'], panel['segment_id']
                row = result['segments'][sid]
                a,b = ((row['before_start_idx'],row['before_end_idx']) if branch=='before'
                       else (row['output_start_idx'],row['output_end_idx']))
                values = expected['pre_data_200' if branch=='before' else 'tfl_data_200'][a:b,ch]
                prefix = f'ch{ch}_{branch}_s{sid}'
                f,power=welch(values.astype(np.float64),fs=200,nperseg=min(800,row['before_end_idx']-row['before_start_idx'],len(values)),scaling='density')
                if frozen is not None:
                    f,power=frozen[f'ch{ch}_{branch}_psd_f'],frozen[f'ch{ch}_{branch}_psd']
                actual[prefix+'_psd_f'],actual[prefix+'_psd']=panel['psd_frequency'],panel['psd']
                old[prefix+'_psd_f'],old[prefix+'_psd']=f,power
                if panel['stft'] is not None:
                    start=(row['raw_start_us']-result['timeline'].source_epoch_us)/1e6
                    count=min(len(values),max(0,int(round((max_sec-start)*200))))
                    f,t,z=stft(values[:count].astype(np.float64),fs=200,nperseg=min(256,count),noverlap=128,window='hann')
                    mask=f<=50
                    f,t,db=f[mask],t+start,20*np.log10(np.abs(z[mask])+1e-8)
                    if frozen is not None:
                        f,t,db=(frozen[f'ch{ch}_{branch}_stft_f'],frozen[f'ch{ch}_{branch}_stft_t'],frozen[f'ch{ch}_{branch}_stft_db'])
                    for suffix, value, baseline in zip(('stft_f','stft_t','stft_db'),panel['stft'],(f,t,db)):
                        actual[prefix+'_'+suffix],old[prefix+'_'+suffix]=value,baseline
        analysis['spectra'][name]=compare(name+'_spectra',actual,old)

    fixtures=ROOT/'tests/fixtures'
    real=[]
    for name in ('move_head','talk','vibration_level1','vibration_level2','vibration_level3'):
        path=fixtures/f'jenqwei_{name}_reference.npz'
        metadata=json.loads(path.with_suffix('.json').read_text())
        source=ROOT/metadata['source_path']
        assert file_sha256(source)==metadata['source_sha256']
        real.append((name,source,path))
    directory=cli('real_all',[source for _,source,_ in real])
    for name,source,path in real:
        with np.load(path) as frozen:
            expected={key:frozen[key] for key in ('time_us_200','pre_data_200','tfl_data_200')}
            expected['tfl_time_us_200']=frozen['time_us_200'][:len(frozen['tfl_data_200'])]
            result=verify_case(name,directory,source,expected)
            spectra(name,result,expected,[0,1],frozen)

    with np.load(fixtures/'jenqwei_synthetic_reference.npz') as frozen:
        source=recording('synthetic',frozen['source_time_us'],frozen['source_raw'])
        directory=cli('synthetic_cli',[source],channels=(0,1,2,3))
        expected={key:frozen[key] for key in ('time_us_200','pre_data_200','tfl_data_200')}
        expected['tfl_time_us_200']=frozen['time_us_200'][:len(frozen['tfl_data_200'])]
        result=verify_case('synthetic',directory,source,expected,4)
        spectra('synthetic',result,expected,[0,1,2,3],frozen)
        short_dir=cli('short_display',[source],channels=(0,),max_sec=.5)
        short_result=verify_case('short_display',short_dir,source,expected,1)
        assert all(p['stft'] is None for p in compute_panels(short_result,0,.5))

    with np.load(processing/'segmented_source.npz') as src, np.load(processing/'real_segmented_expected.npz') as expected:
        source=recording('real_gapped',src['time_us'],src['raw'])
        directory=cli('real_gapped_cli',[source],channels=(0,1,2,3))
        result=verify_case('real_gapped',directory,source,expected,4)
        spectra('real_gapped',result,expected,[0,1,2,3])

    # Two 998-sample segments: a six-ms source gap survives at 200 Hz.
    lengths,starts=(10,998,998,997),(0,10000000,12002000,20000000)
    t=9000000000000001+np.concatenate([start+np.arange(n)*2000 for start,n in zip(starts,lengths)])
    sample=np.arange(len(t))/500
    raw=np.column_stack([(ch+1)*np.sin(2*np.pi*(4+3*ch)*sample) for ch in range(4)]).astype(np.float32)
    source=recording('small_gap',t,raw)
    snapshot=ROOT/'docs/refactor/validation/stage16/baseline_legacy_entry.py.txt'
    legacy=types.ModuleType('legacy_jenqwei')
    legacy.__file__=str(ROOT/'analyze_jenqwei_pipeline.py')
    exec(compile(snapshot.read_bytes(),str(snapshot),'exec'),legacy.__dict__)
    pieces=[]
    offset=0
    for length in lengths:
        a,b=offset,offset+length
        offset=b
        if length>=998:
            pieces.append(legacy.run_pipeline(t[a:b],raw[a:b]))
    expected={key:np.concatenate([p[key] for p in pieces]) for key in ('time_us_200','pre_data_200','tfl_data_200')}
    expected['tfl_time_us_200']=np.concatenate([p['time_us_200'][:len(p['tfl_data_200'])] for p in pieces])
    directory=cli('small_gap_cli',[source])
    result=verify_case('small_gap',directory,source,expected)
    spectra('small_gap',result,expected,[0,1])
    assert result['segments'][1]['raw_end_us']-result['timeline'].source_epoch_us==11996000
    assert result['segments'][2]['raw_start_us']-result['timeline'].source_epoch_us==12002000

    short=recording('all_short',np.arange(997)*2000,raw[:997])
    polluted=raw.copy()
    polluted[0,3]=np.nan
    polluted_source=recording('polluted_short',t,polluted)
    for name,src,channels,max_sec in [('all_short_rejected',short,(0,1),60.),
        ('pollution_rejected',polluted_source,(0,1),60.),('missing_rejected',out/'missing.csv',(0,1),60.),
        ('channel_rejected',source,(4,),60.),('display_rejected',source,(0,1),float('nan'))]:
        directory=cli(name,[src],channels,max_sec,expected=1)
        audit=record_audit(directory,src,False)
        assert not audit['tables_verified']
        analysis['cases'][name]={'status':audit['status'],'stage':audit['stage'],'error':audit['error']}
    # Collision is a separate failed audit; preserve the previous successful set.
    directory=out/'small_gap_cli'
    prior=json.loads((directory/f'{source.stem}_analysis_audit.json').read_text())
    log=command('collision',[sys.executable,str(ROOT/'analyze_jenqwei_pipeline.py'),'--csv',str(source),'--outdir',str(directory)],1)
    remember(log)
    collision,=directory.glob(f'{source.stem}_analysis_audit_*.json')
    collision_audit=json.loads(collision.read_text())
    assert collision_audit['status']=='failed' and collision_audit['stage']=='preflight'
    remember(collision)
    for item in prior['artifacts']:
        assert file_sha256(item['path'])==item['sha256']
    analysis['cases']['collision']={'status':'failed','preserved_previous_artifacts':True}
    for path in (Path(__file__),ROOT/'lilia/jenqwei_io.py',ROOT/'lilia/jenqwei_plot.py',ROOT/'tools/refactor_check.py'):
        remember(path)
    summary=out/'analysis.json'
    summary.write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    remember(summary)
    (out/'evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PASS final stage-16 CLI/numerical acceptance: {summary}',flush=True)


if __name__=='__main__':
    main()
