"""M2 independent legacy/current PSD captures and preregistered controls."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import partial
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_method_profiles import BASELINE, sha, write_json, verify_files, compare_arrays


def bind_model(pipeline, model):
    """Override the frozen wrapper's definition-time default, not its computation."""
    model = Path(model).resolve(strict=True)
    pipeline.apply_tflite_windowed = partial(pipeline.apply_tflite_windowed, tflite_path=str(model))


def normalize(value, folder):
    if isinstance(value, dict):
        return {k: normalize(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, folder) for v in value]
    return value.replace(str(folder), '<output>') if isinstance(value, str) else value


def check_coverage(case, arrays, metadata):
    """Reject equal-but-empty captures and unexpected skips independently."""
    expected = case['quality_expected']
    if metadata['quality_status'] != expected:
        raise AssertionError('Unexpected quality status')
    calls = len([k for k in arrays if k.startswith('quality_welch_') and k.endswith('_density')])
    if calls != (16 if expected == 'complete' else 0):
        raise AssertionError('Missing raw/filtered Welch calls')
    if case['model_expected'] == 'complete':
        if metadata['model_status'] != 'complete' or not metadata['panels']:
            raise AssertionError('Missing model/panels')
        if any(p['branch'] == 'after' and p['channel'] > 2 for p in metadata['panels']):
            raise AssertionError('Invented model channel')
        retained = [s['segment_id'] for s in metadata['segments'] if s['status'] == 'retained']
        expected_panels = {(sid, ch, branch) for sid in retained for ch in range(1, 5)
                           for branch in (('before', 'after') if ch <= 2 else ('before',))}
        found = [(p['segment_id'], p['channel'], p['branch']) for p in metadata['panels']]
        if set(found) != expected_panels or len(found) != len(expected_panels):
            raise AssertionError('Missing or duplicate segment/channel PSD panels')
        if metadata['jen_display_count'] != len(found):
            raise AssertionError('Missing rendered Jenqwei PSD lines')
        if case['name'] == 'gap' and (retained != [1, 2] or
                [s['segment_id'] for s in metadata['segments'] if s['status'] == 'excluded'] != [0, 3]):
            raise AssertionError('Short segment exclusion changed')
    elif metadata['model_status'] != 'no_complete_window':
        raise AssertionError('Missing short-model outcome')


def prepare(out):
    import numpy as np
    from lilia.io import read_lilia_frame
    fixtures = ROOT/'tests/fixtures/quality_stage18_reference.json'
    sources = json.loads(fixtures.read_text())['sources']
    model = ROOT/'tiny_v4_optimized.tflite'
    contract = ROOT/'docs/refactor/PSD_COMPARISON_CONTRACT.md'
    config = dict(baseline_commit=BASELINE, model=str(model), inputs=sources+[
        dict(path=str(p), sha256=sha(p)) for p in (model, fixtures, contract)], cases=[],
        synthetic_recipe='seed1921; 65s continuous; gap lengths250,1503,4001,997 at 0,10,20,40s; .5s short; 500Hz',
        quality_seed=42, model_real_inference=True, contract_sha256=sha(contract))
    for i, row in enumerate(sources):
        frame = read_lilia_frame(row['path']).iloc[:, :5]
        target = out/'local/inputs'/f'real_{i}.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('File Name,M2 validation copy\nAmp Gain,500,Abs Time Offset[us],0\n'
            'Channels,1,2,3,4\nSample Rate,500,500,500,500\n'+frame.to_csv(index=False))
        config['cases'].append(dict(name=f'real_{i}', source=str(target), original=row['path'], real=True,
            model_expected='complete', quality_expected='complete' if len(frame) >= 30000 else 'insufficient_samples',
            render=i == 1))
    for name, lengths, starts in [('continuous', [32500], [0]),
                                   ('gap', [250, 1503, 4001, 997], [0, 10000000, 20000000, 40000000]),
                                   ('short', [250], [0])]:
        time = np.concatenate([start+np.arange(n, dtype=np.int64)*2000 for n, start in zip(lengths, starts)])
        t = time/1e6
        rng = np.random.default_rng(1921)
        raw = np.column_stack([20*np.sin(2*np.pi*(8+c*3)*t)+rng.normal(size=len(t)) for c in range(4)])
        target = out/'synthetic/inputs'/f'{name}.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            handle.write('File Name,M2 synthetic\nAmp Gain,500,Abs Time Offset[us],0\n'
                'Channels,1,2,3,4\nSample Rate,500,500,500,500\nTime[us],ch1,ch2,ch3,ch4\n')
            for stamp, values in zip(time, raw):
                handle.write(','.join([str(stamp), *map(str, values)])+'\n')
        config['cases'].append(dict(name=name, source=str(target), real=False, render=name != 'short',
            model_expected='no_complete_window' if name == 'short' else 'complete',
            quality_expected='complete' if name == 'continuous' else 'insufficient_samples'))
    verify_files(config['inputs'], ROOT)
    return config


def worker(args):
    sys.path.insert(0, str(args.code_root))
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from unittest.mock import patch
    import analyze_jenqwei_pipeline as jen
    import quality_check as quality
    from lilia.jenqwei_plot import compute_panels, plot_metadata, plot_result
    from lilia.jenqwei_io import write_signal_tables, load_signal_table
    from lilia.quality_check_io import load_quality_samples_table
    config = json.loads(args.config.read_text())
    verify_files(config['inputs'], ROOT)
    bind_model(jen, config['model'])
    # Fail closed if either process accidentally imports the other source tree.
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and (name in ('analyze_jenqwei_pipeline', 'quality_check', 'plot_event_markers')
                         or name.startswith('lilia.')):
            if not Path(filename).resolve().is_relative_to(args.code_root.resolve()):
                raise AssertionError('Wrong source tree imported: '+name)
    for case in config['cases']:
        folder = args.out/('local' if case['real'] else 'synthetic')/args.label/case['name']
        folder.mkdir(parents=True)
        arrays, metadata, tables = {}, dict(panels=[]), []
        time, raw = jen.load_raw_csv(case['source'])
        arrays.update(raw_time=time, raw_values=raw)
        result = None
        try:
            result = jen.process_segments(time, raw)
        except ValueError as exc:
            if case['model_expected'] != 'no_complete_window' or 'No source segment contains a complete TFLite window' not in str(exc):
                raise
            metadata.update(model_status='no_complete_window', model_error=str(exc))
        if result is not None:
            metadata.update(model_status='complete', segments=result['segments'],
                            timeline=result['timeline'].metadata(), plotting=plot_metadata(result, [0, 1, 2, 3], 60.))
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    arrays['model_'+key] = value
            for ch in range(4):
                for i, panel in enumerate(compute_panels(result, ch)):
                    prefix = f'jen_ch{ch+1}_panel{i}'
                    for key in ('values', 'elapsed', 'psd_frequency', 'psd'):
                        arrays[prefix+'_'+key] = panel[key]
                    mask = panel['psd_frequency'] <= 50
                    arrays[prefix+'_display_frequency'] = panel['psd_frequency'][mask]
                    arrays[prefix+'_display_density'] = panel['psd'][mask]
                    metadata['panels'].append(dict(channel=ch+1, branch=panel['branch'],
                        segment_id=panel['segment_id'], samples=len(panel['values']),
                        frequency_bins=len(panel['psd']), stft_status=panel['stft_status']))
            # Read actual plotted Line2D values in both processes, including ch3/4.
            original_semilogy, original_save = Axes.semilogy, Figure.savefig
            display_count = 0
            for ch in range(4):
                displayed = []
                def capture_display(ax, *values, **kwargs):
                    lines = original_semilogy(ax, *values, **kwargs)
                    displayed.extend(lines)
                    return lines
                render = case['render'] and args.label == 'actual' and ch in (0, 2)
                with TemporaryDirectory(prefix='lilia-m2-plot-') as scratch, \
                     patch.object(Axes, 'semilogy', capture_display), \
                     patch.object(Figure, 'savefig', original_save if render else lambda *a, **k: None):
                    plot_result(result, folder if render else scratch, case['name'], ch=ch)
                panels = [p for p in metadata['panels'] if p['channel'] == ch+1]
                if len(displayed) != len(panels):
                    raise AssertionError('Jenqwei rendered panel count changed')
                for i, (line, panel) in enumerate(zip(displayed, panels)):
                    prefix = f'jen_ch{ch+1}_panel{i}'
                    if line.get_label() != f"{panel['branch'].title()} S{panel['segment_id']}":
                        raise AssertionError('Jenqwei branch/segment display order changed')
                    for key, values in [('frequency', line.get_xdata()), ('density', line.get_ydata())]:
                        np.testing.assert_array_equal(values, arrays[prefix+'_display_'+key])
                        arrays[prefix+'_display_'+key] = np.asarray(values).copy()
                display_count += len(displayed)
            metadata['jen_display_count'] = display_count
            write_signal_tables(folder, case['source'], result, config['model'], {})
            for branch in ('before', 'after'):
                table = folder/(Path(case['source']).stem+'_'+branch+'.csv')
                load_signal_table(table, case['source'], model_path=config['model'])
                tables.append(dict(kind='jenqwei_signal', path=str(table), raw_csv=case['source'], model_path=config['model']))
        # Capture real caller Welch arguments and actual semilogy line data.
        original_welch, original_semilogy, original_save = quality.welch, Axes.semilogy, Figure.savefig
        calls, displays = [], []
        def welch(values, **kwargs):
            f, power = original_welch(values, **kwargs)
            i = len(calls)
            calls.append(kwargs)
            arrays[f'quality_welch_{i}_input'] = values.copy()
            arrays[f'quality_welch_{i}_frequency'] = f.copy()
            arrays[f'quality_welch_{i}_density'] = power.copy()
            return f, power
        def semilogy(ax, *values, **kwargs):
            lines = original_semilogy(ax, *values, **kwargs)
            for line in lines:
                i = len(displays)
                displays.append(line.get_label())
                arrays[f'quality_display_{i}_frequency'] = np.asarray(line.get_xdata()).copy()
                arrays[f'quality_display_{i}_density'] = np.asarray(line.get_ydata()).copy()
            return lines
        def save(fig, path, **kwargs):
            if case['render'] and args.label == 'actual' and str(path).endswith('.png'):
                original_save(fig, path, **kwargs)
        with patch.object(quality, 'welch', side_effect=welch), patch.object(Axes, 'semilogy', semilogy), \
             patch.object(Figure, 'savefig', save):
            quality.plot_segments(case['source'], 'M2', case['name'], str(folder), np.random.default_rng(42))
        table = folder/f'M2_{case["name"]}_sample_quality.csv'
        metadata.update(quality_status='complete' if table.exists() else 'insufficient_samples',
                        welch_calls=calls, display_labels=displays)
        if table.exists():
            _, audit = load_quality_samples_table(table, case['source'])
            metadata['quality_audit'] = audit
            tables.append(dict(kind='quality_check_samples', path=str(table), raw_csv=case['source']))
            if len(displays) != len(calls):
                raise AssertionError('Missing quality PSD display lines')
            for i in range(len(calls)):
                f = arrays[f'quality_welch_{i}_frequency']
                for part in ('frequency', 'density'):
                    np.testing.assert_array_equal(arrays[f'quality_display_{i}_{part}'], arrays[f'quality_welch_{i}_{part}'][f <= 50])
        metadata['tables'] = {}
        for row in tables:
            p = Path(row['path'])
            frame = pd.read_csv(p)
            metadata['tables'][p.name] = json.loads(frame.to_json(orient='split', double_precision=15))
            metadata['tables'][p.name+'.meta'] = json.loads(Path(str(p)+'.meta.json').read_text())
            for col in frame.select_dtypes(include='number'):
                arrays[p.name+'_'+col] = frame[col].to_numpy()
        check_coverage(case, arrays, metadata)
        np.savez_compressed(folder/'numeric.npz', **arrays)
        write_json(folder/'comparison.json', normalize(metadata, folder))
        write_json(folder/'tables.json', tables)
        plt.close('all')
        print(f'PASS {case["name"]}: {len(arrays)} arrays', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--label')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('local/\n!*.csv\n!*.png\n')
    config = prepare(out)
    frozen = out/'local/frozen_source'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT).decode().splitlines()
    names = [n for n in names if n.endswith('.py') and ('/' not in n or n.startswith('lilia/'))]
    config.update(frozen_source_hashes={}, current_source_hashes={},
        tools={n: sha(ROOT/n) for n in ('tools/freeze_psd_profiles.py', 'tools/compare_psd_sensitivity.py', 'tools/freeze_method_profiles.py')},
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'matplotlib', 'tensorflow')})
    config['capture_inputs'] = [dict(path=c['source'], sha256=sha(c['source'])) for c in config['cases']]
    for name in names:
        path = frozen/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT))
        config['frozen_source_hashes'][name] = sha(path)
        config['current_source_hashes'][name] = sha(ROOT/name)
    write_json(out/'config.json', config)
    os.environ.update(MPLCONFIGDIR=str(out/'local/mpl'), MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for label, code in [('expected', frozen), ('actual', ROOT)]:
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--code-root', str(code),
                '--config', str(out/'config.json'), '--out', str(out), '--label', label], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PASS '+label+' worker', flush=True)
    rows, comparisons, tables = [], [], []
    for case in config['cases']:
        base = out/('local' if case['real'] else 'synthetic')
        old, new = [base/label/case['name'] for label in ('expected', 'actual')]
        count, empty = compare_arrays(old/'numeric.npz', new/'numeric.npz')
        if empty or (old/'comparison.json').read_bytes() != (new/'comparison.json').read_bytes():
            raise ValueError('Empty arrays or metadata mismatch: '+case['name'])
        comparisons.append(dict(expected=str(old/'numeric.npz'), actual=str(new/'numeric.npz'), rtol=0., atol=0., equal_nan=True))
        found = json.loads((new/'tables.json').read_text())
        tables += found
        rows.append(dict(name=case['name'], arrays=count, readers=len(found),
            model_status=case['model_expected'], quality_status=case['quality_expected']))
    from tools.compare_psd_sensitivity import run, run_captured
    sensitivity = run(out)
    captured_sensitivity = run_captured(out, config['cases'])
    verify_files(config['inputs'], ROOT)
    verify_files(config['capture_inputs'], ROOT)
    verify_files([dict(path=n, sha256=d) for n, d in config['frozen_source_hashes'].items()], frozen)
    for name, digest in {**config['current_source_hashes'], **config['tools']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Code changed during capture: '+name)
    write_json(out/'analysis.json', dict(cases=rows, arrays=sum(r['arrays'] for r in rows), reader_checks=len(tables),
        max_abs_error=0., rtol=0., atol=0., metadata_equal=True, sensitivity_controls=sensitivity['controls'],
        sensitivity_maxima=sensitivity['maxima'], profile_changes=False))
    write_json(out/'sensitivity_capture_summary.json', captured_sensitivity)
    def relative(row):
        return {k: str(Path(v).relative_to(out)) if k in ('path', 'raw_csv', 'model_path', 'actual', 'expected')
                and Path(v).is_relative_to(out) else v for k, v in row.items()}
    files = [dict(path=str(p), sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and '/mpl/' not in str(p)]
    files += config['inputs']+[dict(path=str(ROOT/n), sha256=d) for n, d in {**config['current_source_hashes'], **config['tools']}.items()]
    manifest = dict(schema_version=1, files=list(map(relative, files)), tables=list(map(relative, tables)), comparisons=list(map(relative, comparisons)))
    write_json(out/'manifest_local.json', manifest)
    sources = {r['path'] for r in config['inputs'] if r['path'] != str(ROOT/'docs/refactor/PSD_COMPARISON_CONTRACT.md')}
    public = dict(schema_version=1,
        files=[r for r in manifest['files'] if not r['path'].startswith('local/') and r['path'] not in sources],
        tables=[r for r in manifest['tables'] if r['path'].startswith('synthetic/') and r['kind'] == 'quality_check_samples'],
        comparisons=[r for r in manifest['comparisons'] if r['actual'].startswith('synthetic/')])
    write_json(out/'manifest_repository.json', public)
    print(f'PASS {len(rows)} cases, {sum(r["arrays"] for r in rows)} exact arrays, {len(tables)} readers, {sensitivity["controls"]} sensitivity controls')


if __name__ == '__main__':
    main()
