"""M3 independent legacy/current Goertzel capture and measurement comparisons."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_method_profiles import BASELINE, sha, write_json, verify_files, compare_arrays


def canonical(value, folder):
    if isinstance(value, dict):
        return {k:canonical(v,folder) for k,v in value.items()}
    if isinstance(value,list):
        return [canonical(v,folder) for v in value]
    return value.replace(str(folder),'<output>') if isinstance(value,str) else value


def check_case(case, status, count):
    expected = case.get('status','complete')
    if status != expected or (expected=='complete' and count < 1) or (expected!='complete' and count != 0):
        raise AssertionError('Unexpected or empty Goertzel outcome: '+case['name'])


def prepare(out):
    import numpy as np
    from lilia.io import load_merged_csv
    from tools.validate_goertzel_quality_stage18 import write_source
    fixture = ROOT/'tests/fixtures/quality_stage18_reference.json'
    inputs = json.loads(fixture.read_text())['sources']
    inputs += [dict(path=str(p),sha256=sha(p)) for p in (fixture,ROOT/'docs/refactor/GOERTZEL_COMPARISON_CONTRACT.md')]
    verify_files(inputs,ROOT)
    cases = []
    for i,row in enumerate(inputs[:5]):
        for ch in (1,2):
            cases.append(dict(name=f'real_{i}_ch{ch}_5s',source=row['path'],real=True,ch=ch,win=5.,render=i==1 and ch==1))
    talk = inputs[1]['path']
    for win in (.5,1.):
        cases.append(dict(name=f'real_1_ch1_{win}s',source=talk,real=True,ch=1,win=win))
    t,raw=load_merged_csv(talk)
    split=(len(t)//5000)*2500
    t=t.copy();t[split:]+=10000000
    path=out/'local/inputs/talk_gap.csv';path.parent.mkdir(parents=True)
    write_source(path,t,raw)
    cases.append(dict(name='real_gap',source=str(path),real=True,ch=1,win=5.,render=True))
    synthetic=out/'synthetic/inputs';synthetic.mkdir(parents=True)
    for name,lengths,starts in [('continuous',[30000],[0]),('gap',[7500,7500],[0,25000000]),('single',[2500],[0]),
                                ('short',[250],[0]),('half',[3000],[0]),('nonfinite',[3000],[0])]:
        t=np.concatenate([start+np.arange(n,dtype=np.int64)*2000 for n,start in zip(lengths,starts)])
        n=np.arange(len(t));amp=np.where(n<len(t)//2,2.,30.)
        rng=np.random.default_rng(1923)
        raw=np.column_stack([amp*np.sin(2*np.pi*60*t/1e6)+rng.normal(size=len(t)),
                             5*np.sin(2*np.pi*10*t/1e6)+rng.normal(size=len(t))])
        if name=='nonfinite':raw[100,0]=np.nan
        path=synthetic/(name+'.csv');write_source(path,t,raw)
        cases.append(dict(name=name,source=str(path),real=False,ch=1,win=.5 if name=='half' else 5.,
            status='empty' if name=='short' else 'nonfinite_rejected' if name=='nonfinite' else 'complete',
            render=name in ('gap','short')))
    cases.append(dict(name='gap_quality_witness',source=str(synthetic/'gap.csv'),real=False,ch=1,win=5.,
                      quality_witness=True,render=True))
    return dict(baseline_commit=BASELINE,inputs=inputs,cases=cases,model_used=False,
                generated_inputs=[dict(path=c['source'],sha256=sha(c['source'])) for c in cases],
                target_freq=60.,quality_threshold=.5,smooth_win=5,
                synthetic_seed=1923,real_gap_split_index=split,real_gap_added_us=10000000)


def worker(args):
    sys.path.insert(0,str(args.code_root))
    import numpy as np
    from unittest.mock import patch
    from contextlib import ExitStack
    from matplotlib.figure import Figure
    import plot_goertzel_vs_raw as plot
    from lilia.goertzel_io import load_goertzel_table
    from lilia.quality_policy import valid_goertzel_rows
    config=json.loads(args.config.read_text())
    verify_files(config['inputs']+config['generated_inputs'],ROOT)
    for name,module in list(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if filename and (name=='plot_goertzel_vs_raw' or name.startswith('lilia.')):
            if not Path(filename).resolve().is_relative_to(args.code_root.resolve()):
                raise AssertionError('Wrong source tree: '+name)
    for case in config['cases']:
        folder=args.out/('local' if case['real'] else 'synthetic')/args.label/case['name']
        folder.mkdir(parents=True)
        arrays,metadata={},dict(status='complete',readers=0)
        original_render,original_save=plot._render_plot,Figure.savefig
        def save(fig,path,**kwargs):
            label='linear' if str(path).endswith('_linear.png') else 'db'
            for i,line in enumerate(fig.axes[0].lines[:2]):
                arrays[f'render_{label}_{i}_x']=np.asarray(line.get_xdata()).copy()
                arrays[f'render_{label}_{i}_y']=np.asarray(line.get_ydata()).copy()
            if case.get('render') and args.label=='actual':original_save(fig,path,**kwargs)
        def render(**kwargs):
            result=kwargs['result']
            for key in result.__dataclass_fields__:
                value=getattr(result,key)
                if isinstance(value,np.ndarray):arrays[key]=value.copy()
            metadata['quality_audit']=result.quality_audit
            original_render(**kwargs)
            kwargs.update(use_db=False,out_png=str(folder/'plot_linear.png'))
            original_render(**kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(plot,'_render_plot',side_effect=render))
            stack.enter_context(patch.object(Figure,'savefig',save))
            if case.get('quality_witness'):
                stack.enter_context(patch.object(plot,'get_eeg_quality_index_v2_parametric',
                    side_effect=lambda data,**kw:dict(overall=np.full(data.shape[0],.7))))
            try:
                plot._plot_subject(case['source'],str(folder/'plot.png'),str(folder/'metrics.csv'),
                    case['ch'],500.,60.,case['win'],case['win'],5,.5,(-150.,150.),True,1950.,.12,1000.,1.,80.)
            except ValueError as exc:
                if case.get('status')!='nonfinite_rejected' or not str(exc).startswith('Non-finite EEG samples:'):
                    raise
                metadata.update(status='nonfinite_rejected',error=str(exc))
        count=0
        if (folder/'metrics.csv').exists():
            frame,meta=load_goertzel_table(folder/'metrics.csv',case['source'])
            count=len(frame)
            metadata.update(status='complete' if count else 'empty',readers=1,table_metadata=meta,
                            table=json.loads(frame.to_json(orient='split',double_precision=15)))
            for column in frame.select_dtypes(include='number'):
                arrays['table_'+column]=frame[column].to_numpy()
            for hard in (False,True):
                for threshold in (.49,.5):
                    good=valid_goertzel_rows(frame,threshold,exclude_hard=hard)
                    arrays[f'accepted_{hard}_{threshold}']=good
                    for key in ('goertzel_power','goertzel_db'):
                        arrays[f'legacy_smooth_{key}_{hard}_{threshold}']=np.where(good,
                            plot._rolling_median(np.where(good,frame[key].to_numpy(),np.nan),5),np.nan)
        check_case(case,metadata['status'],count)
        metadata['rows']=count
        arrays['row_count']=np.array([count],dtype=np.int64)
        np.savez_compressed(folder/'numeric.npz',**arrays)
        np.savez_compressed(folder/'evidence.npz',**{k:v for k,v in arrays.items() if v.size})
        metadata['empty_arrays']={k:list(v.shape) for k,v in arrays.items() if not v.size}
        write_json(folder/'metadata.json',canonical(metadata,folder))
        print(f'PASS {case["name"]}: {metadata["status"]}, {count} windows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--code-root',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--label')
    args=parser.parse_args()
    if args.worker:return worker(args)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'.gitignore').write_text('local/\n!*.csv\n!*.png\n')
    config=prepare(out)
    frozen=out/'local/frozen_source'
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE],cwd=ROOT).decode().splitlines()
    names=[n for n in names if n.endswith('.py') and ('/' not in n or n.startswith('lilia/'))]
    config.update(frozen_source_hashes={},current_source_hashes={},
        tools={n:sha(ROOT/n) for n in ('tools/freeze_goertzel_profiles.py','tools/compare_goertzel_methods.py',
                                       'tools/freeze_method_profiles.py','tools/validate_goertzel_quality_stage18.py')},
        python=sys.version,packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','matplotlib')})
    for name in names:
        path=frozen/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','show',f'{BASELINE}:{name}'],cwd=ROOT))
        config['frozen_source_hashes'][name]=sha(path);config['current_source_hashes'][name]=sha(ROOT/name)
    write_json(out/'config.json',config)
    env={**os.environ,'MPLCONFIGDIR':str(out/'local/mpl'),'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for label,code in [('expected',frozen),('actual',ROOT)]:
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--code-root',str(code),
                '--config',str(out/'config.json'),'--out',str(out),'--label',label],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        print('PASS '+label+' process',flush=True)
    from tools.compare_goertzel_methods import compare_case,controls
    rows,tables,comparisons,methods=[],[],[],[]
    for case in config['cases']:
        base=out/('local' if case['real'] else 'synthetic')
        old,new=[base/label/case['name'] for label in ('expected','actual')]
        count,empty=compare_arrays(old/'numeric.npz',new/'numeric.npz')
        if (old/'metadata.json').read_bytes()!=(new/'metadata.json').read_bytes():
            raise AssertionError('Metadata differs: '+case['name'])
        metadata=json.loads((new/'metadata.json').read_text())
        comparisons.append(dict(expected=str(old/'evidence.npz'),actual=str(new/'evidence.npz'),rtol=0.,atol=0.,equal_nan=True))
        if metadata['status']!='nonfinite_rejected':
            for folder in (old,new):tables.append(dict(path=str(folder/'metrics.csv'),kind='goertzel',raw_csv=case['source']))
            methods.append(compare_case(new,case['source'],case))
        rows.append(dict(case=case['name'],arrays=count,empty_arrays=empty,rows=metadata['rows'],status=metadata['status']))
    control=controls(out)
    write_json(out/'analysis.json',dict(cases=rows,arrays=sum(r['arrays'] for r in rows),reader_checks=len(tables),
        window_rows=sum(r['rows'] for r in rows),rtol=0.,atol=0.,max_abs_error=0.,metadata_equal=True,
        method_comparisons=methods,controls=control['controls'],product_changes=False))
    verify_files(config['inputs']+config['generated_inputs'],ROOT)
    verify_files([dict(path=n,sha256=d) for n,d in config['frozen_source_hashes'].items()],frozen)
    verify_files([dict(path=n,sha256=d) for n,d in {**config['current_source_hashes'],**config['tools']}.items()],ROOT)
    def relative(row):
        return {k:str(Path(v).relative_to(out)) if k in ('path','raw_csv','actual','expected') and Path(v).is_relative_to(out) else v for k,v in row.items()}
    files=[dict(path=str(p),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and '/mpl/' not in str(p)]
    files+=config['inputs']+[dict(path=str(ROOT/n),sha256=d) for n,d in {**config['current_source_hashes'],**config['tools']}.items()]
    manifest=dict(schema_version=1,files=list(map(relative,files)),tables=list(map(relative,tables)),comparisons=list(map(relative,comparisons)))
    write_json(out/'manifest_local.json',manifest)
    originals={r['path'] for r in config['inputs'][:5]}
    public=dict(schema_version=1,files=[r for r in manifest['files'] if not r['path'].startswith('local/') and r['path'] not in originals],
        tables=[r for r in manifest['tables'] if r['path'].startswith('synthetic/')],
        comparisons=[r for r in manifest['comparisons'] if r['actual'].startswith('synthetic/')])
    write_json(out/'manifest_repository.json',public)
    print(f'PASS {len(rows)} cases; {sum(r["arrays"] for r in rows)} exact arrays; {len(tables)} readers; {control["controls"]} controls')


if __name__=='__main__':
    main()
