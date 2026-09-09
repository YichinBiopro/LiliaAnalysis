"""Reproduce stage 14 CLI evidence in a new output directory, using frozen baselines."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lilia.io import read_lilia_frame
from lilia.provenance import file_sha256
from lilia.qeeg_raw import METRIC_KEYS
from lilia.qeeg_io import load_qeeg_table

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory for source copies and outputs')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(out/'mpl'))
    # Bundle subprocess must resolve its own package, even if caller used PYTHONPATH.
    env.pop('PYTHONPATH', None)
    fixture = ROOT/'tests/fixtures'
    manifest = {'schema_version':1, 'files':[], 'tables':[], 'comparisons':[]}
    report = {'cases':{}, 'comparisons':{}, 'fixture_hashes':{}}
    cases = {}

    def remember(path):
        path = Path(path).resolve()
        manifest['files'].append({'path':str(path), 'sha256':file_sha256(path)})

    def source(name, t, raw):
        path=out/f'{name}.csv'
        f=pd.DataFrame(raw, columns=[f'ch{i+1}' for i in range(raw.shape[1])])
        f.insert(0,'Time[us]',t)
        path.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2\nSample Rate,500,500\n'+f.to_csv(index=False))
        remember(path)
        return path

    def run(name, src, *, entry='main', extra=(), expect=0):
        dest=out/name
        cwd=ROOT/'plot_index_vs_raw_bundle' if entry=='bundle' else ROOT
        command=[sys.executable, 'qeeg_indices.py'] if entry=='shim' else [sys.executable, '-m', 'lilia.qeeg']
        command += ['--csv',str(src),'--out',str(dest), *extra]
        with (out/f'{name}.log').open('w') as log:
            proc=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode != expect:
            raise AssertionError(f'{name}: exit {proc.returncode}, expected {expect}; see log')
        audit_path=next(dest.glob('*_analysis_audit.json'))
        audit=json.loads(audit_path.read_text())
        assert audit['status']==('success' if expect==0 else 'failed')
        # Use hashes recorded by generation, not a second self-comparison hash.
        for filename, expected_hash in audit['artifacts'].items():
            manifest['files'].append({'path':str(dest/filename), 'sha256':expected_hash})
        remember(audit_path)
        frame=meta=None
        table=list(dest.glob('*.csv'))
        if table:
            frame,meta=load_qeeg_table(table[0],src)
            manifest['tables'].append({'path':str(table[0]), 'kind':'raw_qeeg','raw_csv':str(src)})
        cases[name]=(frame,meta,audit)
        report['cases'][name]={'command':command,'cwd':str(cwd),'exit':proc.returncode,
                              'analysis':audit.get('analysis'), 'error':audit.get('error'),
                              'artifacts':audit['artifacts']}
        print(f'{name}: exit={proc.returncode}, rows={0 if frame is None else len(frame)}',flush=True)
        return frame,meta

    def compare(name, actual, expected):
        errors={}
        for key, values in expected.items():
            np.testing.assert_allclose(actual[key], values, rtol=1e-12, atol=1e-12)
            errors[key]=float(np.max(np.abs(np.asarray(actual[key])-values)))
        a,b=out/f'{name}_actual.npz',out/f'{name}_expected.npz'
        np.savez_compressed(a,**actual); np.savez_compressed(b,**expected)
        remember(a); remember(b)
        manifest['comparisons'].append({'actual':str(a),'expected':str(b),'rtol':1e-12,'atol':1e-12,'equal_nan':False})
        report['comparisons'][name]=errors

    for name in ('qeeg_cli_continuous_reference.json','qeeg_cli_continuous_reference.npz',
                 'qeeg_hardy_reference.json','qeeg_hardy_reference.npz'):
        remember(fixture/name); report['fixture_hashes'][name]=file_sha256(fixture/name)
    with np.load(fixture/'qeeg_cli_continuous_reference.npz') as ref:
        continuous=source('continuous',ref['time_us'],ref['raw'])
        for i,win in enumerate((5.,1.0019,.018)):
            f,m=run(f'continuous_{i}',continuous,extra=('--win',str(win)))
            compare(f'continuous_{i}', {k:f[k].to_numpy() for k in METRIC_KEYS},
                    {k:ref[f'case{i}_{k}'] for k in METRIC_KEYS})
            np.testing.assert_array_equal(f.time_s,ref[f'case{i}_time'])
    # Real source: verify old baseline belongs to precisely this recording.
    hardy_ref=json.loads((fixture/'qeeg_hardy_reference.json').read_text())
    hardy=ROOT/hardy_ref['source_path']
    assert file_sha256(hardy)==hardy_ref['source_sha256']; remember(hardy)
    rawframe=read_lilia_frame(hardy)
    t=rawframe.iloc[:,0].to_numpy(dtype=np.int64)
    raw=rawframe.iloc[:,1:].to_numpy(dtype=float)
    real_continuous=source('real_continuous',t[:hardy_ref['continuous_samples']],raw[:hardy_ref['continuous_samples']])
    f,_=run('real_continuous',real_continuous)
    with np.load(fixture/'qeeg_hardy_reference.npz') as ref:
        compare('real_continuous',{k:f[k].to_numpy() for k in METRIC_KEYS},
                {k:ref['continuous_'+k] for k in METRIC_KEYS})
        for entry in ('main','shim','bundle'):
            f,m=run('hardy_'+entry,hardy,entry=entry)
            # Independent source-lattice check, including every gap crossing.
            starts=np.asarray([i for i in range(0,len(t)-2500+1,2500)
                               if not np.any(np.diff(t[i:i+2500])>6000)],dtype=np.int64)
            np.testing.assert_array_equal(f.window_start_idx, starts)
            np.testing.assert_array_equal(f.window_center_us,t[starts+1250])
            expected={k:ref['naive_'+k][starts//2500] for k in METRIC_KEYS}
            compare('hardy_'+entry,{k:f[k].to_numpy() for k in METRIC_KEYS},expected)
        f,m,_=cases['hardy_main']
        report['hardy_time']={'last_elapsed_seconds':float(f.time_s.iloc[-1]),
                             'old_sample_count_seconds':float(ref['naive_time'][starts[-1]//2500]),
                             'last_center_us':int(f.window_center_us.iloc[-1]),
                             'last_time_difference_seconds':float(f.time_s.iloc[-1]-ref['naive_time'][starts[-1]//2500]),
                             'old_naive_windows':len(ref['naive_time']),
                             'retained_windows':len(f),
                             'old_naive_summary':{k:{'mean':float(ref['naive_'+k].mean()),'std':float(ref['naive_'+k].std())}
                                                  for k in ('focus','flow','calm','relaxation')}}
    epoch=1750000000000001
    t=epoch+np.arange(17600,dtype=np.int64)*2000
    t[100:]+=20000000; t[5100:]+=6000; t[12600:]+=10000000
    raw=np.random.default_rng(14).normal(size=(len(t),2)); raw[7501,0]=np.nan
    gapped=source('gapped',t,raw)
    for entry in ('main','shim','bundle'):
        run('gapped_'+entry,gapped,entry=entry)
    run('gapped_channel2',gapped,extra=('--ch','2'))
    short=source('all_short',epoch+np.arange(7)*2000,np.ones((7,1)))
    run('all_short',short,expect=1)
    invalid=source('all_excluded',epoch+np.arange(2500)*2000,np.full((2500,1),np.nan))
    run('all_excluded',invalid,expect=1)
    run('wrong_channel',gapped,extra=('--ch','3'),expect=1)
    run('invalid_window',gapped,extra=('--win','.01'),expect=1)
    for name in ('hardy','gapped'):
        original=cases[name+'_main'][0]
        for entry in ('shim','bundle'):
            pd.testing.assert_frame_equal(original,cases[name+'_'+entry][0],check_exact=True)
            assert cases[name+'_main'][1]['analysis']==cases[name+'_'+entry][1]['analysis']
    report['tables_verified']=len(manifest['tables'])
    report['hashes_recorded']=len(manifest['files'])
    report['source_py_sha256']={str(p.relative_to(ROOT)):file_sha256(p) for p in
                               (ROOT/'lilia/qeeg.py',ROOT/'lilia/qeeg_raw.py',ROOT/'lilia/qeeg_io.py',ROOT/'tools/refactor_check.py')}
    (out/'evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {out}/validation.json and evidence.json',flush=True)


if __name__=='__main__':
    main()
