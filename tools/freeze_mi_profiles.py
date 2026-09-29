"""M4 independent committed/current MI capture, controls and source audits."""
import argparse
import csv
import hashlib
from decimal import Decimal
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_method_profiles import BASELINE, compare_arrays, sha, verify_files, write_json


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def require_outcome(name, expected, actual, rows):
    if expected != actual or (expected == 'complete' and rows < 1):
        raise AssertionError(f'Unexpected outcome {name}: {actual}, {rows} rows; expected {expected}')


def compare_metadata(old, new, old_code, new_code):
    """Compare semantics after readers verified each table's own hash chain.

    The source hash participates in config_id, CSV and summary bindings. Only
    those specific fields may differ after a product presentation edit.
    """
    def normalize(value, code):
        value = json.loads(json.dumps(value))
        for entry in value.values():
            if not isinstance(entry, dict) or 'frame' not in entry or 'metadata' not in entry:
                continue
            meta, frame = entry['metadata'], entry['frame']
            if meta['kind'] == 'joint_mi':
                if meta['code_sha256'] != code or meta['parameters']['code_sha256'] != code:
                    raise AssertionError('Captured code hash differs from frozen source')
                from lilia.entropy_io import config_id
                identity = config_id(meta['parameters'])
                if meta['config_id'] != identity or any(v != identity for v in frame['config_id']):
                    raise AssertionError('Captured config identity is not source-bound')
                del frame['config_id']
                for key in ('code_sha256', 'config_id', 'table_sha256'):
                    del meta[key]
                del meta['parameters']['code_sha256']
            elif meta['kind'] == 'joint_mi_summary':
                for key in ('series_sha256', 'series_meta_sha256'):
                    del meta[key]
        return value
    if normalize(old, old_code) != normalize(new, new_code):
        raise AssertionError('Semantic metadata differs')


def joint_code_identity(hashes):
    names = ['spectral_entropy.py'] + ['lilia/' + name for name in (
        'windowing.py', 'signal.py', 'quality.py', 'entropy_io.py', 'neural.py',
        'neural_io.py', 'entropy_quality.py', 'quality_audit.py')]
    return hashlib.sha256(''.join(hashes[name] for name in names).encode()).hexdigest()


def prepare(out):
    import numpy as np
    from tools.validate_goertzel_quality_stage18 import write_source
    fixture = ROOT / 'tests/fixtures/quality_stage18_reference.json'
    sources = json.loads(fixture.read_text())['sources']
    inputs = sources + [dict(path=str(p), sha256=sha(p)) for p in (
        fixture, ROOT / 'docs/refactor/MI_COMPARISON_CONTRACT.md')]
    verify_files(inputs, ROOT)
    cases, annotations = [], []
    for i, row in enumerate(sources):
        cases.append(dict(name=f'real_{i}_bp', source=row['path'], real=True, render=i == 1, sensitivity=True))
    cases.append(dict(name='real_talk_raw', source=sources[1]['path'], real=True, raw=True))
    for label in ('talk', 'move head'):
        source = ROOT / 'jenqwei' / ('2026-06-18-lilia-' + label) / ('2026-06-18-lilia-' + label + '.csv')
        markers = source.parent / 'time_marker.csv'
        inputs.append(dict(path=str(markers), sha256=sha(markers)))
        with source.open() as f:
            next(f)
            header = next(csv.reader(f))
        offset = int(Decimal(header[header.index('Abs Time Offset[us]') + 1]))
        with markers.open() as f:
            rows = list(csv.DictReader(f))
        events = [(f'keyboard_{i}', int(Decimal(row['Abs_time(us)'])) - offset) for i, row in enumerate(rows)]
        annotations.append(dict(source=str(source), marker_file=str(markers), offset_us=offset,
            events=events, kind='recorded keyboard triggers; not behavioral start/end truth',
            abs_minus_offset_relative_us=[event[1] - int(Decimal(row['0-New_Task-recording_time(us)']))
                                          for event, row in zip(events, rows)]))
        cases.append(dict(name='event_' + label.replace(' ', '_'), source=str(source), real=True,
                          event=True, events=events, render=True))
    synth = out / 'synthetic/inputs'
    synth.mkdir(parents=True)
    for name, lengths, starts in [('continuous', [12000], [0]), ('gap', [6000, 6000], [0, 22000000]),
                                  ('short', [250], [0]), ('nonfinite', [6000], [0])]:
        time = np.concatenate([s + np.arange(n, dtype=np.int64)*2000 for n, s in zip(lengths, starts)])
        rng = np.random.default_rng(922)
        raw = rng.normal(size=(len(time), 2)) * 5
        raw[:, 0] += 20*np.sin(2*np.pi*10*time/1e6)
        raw[:, 1] += 10*np.sin(2*np.pi*10*time/1e6 + .4)
        if name == 'nonfinite':
            raw[400, 0] = np.nan
        path = synth / (name + '.csv')
        write_source(path, time, raw)
        cases.append(dict(name=name, source=str(path), real=False, render=name in ('continuous', 'gap'),
                          status='short_rejected' if name == 'short' else 'nonfinite_rejected' if name == 'nonfinite' else 'complete'))
        if name in ('continuous', 'gap'):
            cases.append(dict(name='event_' + name, source=str(path), real=False, event=True, render=True,
                              events=[('inside', 6000000), ('boundary', 12000000), ('outside', 50000000)]))
    for quality in (.5, .49, 'nan', 'disabled'):
        cases.append(dict(name='quality_' + str(quality), source=str(synth/'gap.csv'), real=False,
                          quality=quality, render=quality == .49))
    return dict(baseline_commit=BASELINE, inputs=inputs, cases=cases, annotations=annotations,
                generated_inputs=[dict(path=c['source'], sha256=sha(c['source'])) for c in cases],
                model_used=False, fs=500., bins=16, binning='quantile', surrogates=200, seed=0)


def worker(args):
    sys.path.insert(0, str(args.code_root))
    from contextlib import ExitStack
    from unittest.mock import patch
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    import spectral_entropy as module
    from lilia.io import load_merged_csv
    from lilia.entropy_io import load_joint_mi_table, load_joint_mi_summary
    from lilia.windowing import build_window_grid, continuous_slices
    from tools.compare_mi_methods import controls, check_histogram, capture_event_estimates, check_null, plot_controls
    config = json.loads(args.config.read_text())
    verify_files(config['inputs'] + config['generated_inputs'], ROOT)
    for name, mod in list(sys.modules.items()):
        filename = getattr(mod, '__file__', None)
        if filename and (name == 'spectral_entropy' or name.startswith('lilia.')):
            if not Path(filename).resolve().is_relative_to(args.code_root.resolve()):
                raise AssertionError('Wrong source tree: ' + name)
    folder = args.out / 'synthetic' / args.label / 'controls'
    folder.mkdir(parents=True)
    arrays = {}
    def save(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                save(prefix + '__' + key, item)
        elif value is not None:
            a = np.asarray(value)
            if a.dtype.kind in 'biuf':
                arrays[prefix] = np.atleast_1d(a).copy()
    rows = controls(module, save)
    write_json(folder/'metadata.json', clean(dict(rows=rows)))
    np.savez_compressed(folder/'numeric.npz', **arrays)
    np.savez_compressed(folder/'evidence.npz', **{k: v for k, v in arrays.items() if v.size})
    if args.label == 'actual':
        plot_controls(rows, folder/'controls.png')
    print(f'PASS controls: {len(rows)} measurements', flush=True)
    for case in config['cases']:
        folder = args.out / ('local' if case['real'] else 'synthetic') / args.label / case['name']
        folder.mkdir(parents=True)
        arrays = {}
        meta = dict(status='complete', readers=[], rows=0)
        source = case['source']
        original_save = Figure.savefig
        def figure_save(fig, path, *pos, **kwargs):
            prefix = Path(path).stem
            for i, ax in enumerate(fig.axes):
                for j, line in enumerate(ax.lines):
                    for axis in ('x', 'y'):
                        values = np.asarray(getattr(line, 'get_' + axis + 'data')())
                        if values.dtype.kind in 'biuf':
                            save(f'plot_{prefix}_{i}_{j}_{axis}', values)
            if case.get('render') and args.label == 'actual':
                original_save(fig, path, *pos, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(Figure, 'savefig', figure_save))
            if case.get('event'):
                time, raw = load_merged_csv(source)
                event_rows, event_results = [], []
                original_event = module.compute_band_event_joint_mi
                def estimate(*pos, **kwargs):
                    prefix = f'event_call_{len(event_results)}'
                    with capture_event_estimates(module, arrays, prefix) as calls:
                        result = original_event(*pos, **kwargs)
                    if result is not None:
                        check_null(result['joint_mi_bits'], np.asarray(calls[1:])/np.log(2), result, event=True)
                    event_results.append(result)
                    return result
                stack.enter_context(patch.object(module, 'compute_band_event_joint_mi', side_effect=estimate))
                settings = [(3, 0, 200, .5, (2., 5., 10.)), (1, 0, 200, .5, (5.,)),
                            (5, 0, 200, .5, (5.,)), (3, 7, 200, .5, (5.,)),
                            (3, 42, 200, .5, (5.,)), (3, 0, 19, .5, (5.,)),
                            (3, 0, 99, .5, (5.,)), (3, 0, 200, 1., (5.,))]
                for index, (k, seed, count, step, durations) in enumerate(settings):
                    audit = []
                    frame = module.run_band_event_mi_pipeline({'ch1': raw[:, 0], 'ch2': raw[:, 1]}, [],
                        time_us=time, events=case['events'], windows_sec=durations, sub_step_sec=step,
                        n_neighbors=k, random_state=seed, n_surrogates=count, audit=audit)
                    if frame.empty:
                        raise AssertionError('Unexpected empty event estimate')
                    table = folder / f'event_{index}.csv'
                    frame.to_csv(table, index=False)
                    pd.testing.assert_frame_equal(frame, pd.read_csv(table, float_precision='round_trip'), check_exact=True)
                    for column in frame.select_dtypes(include='number'):
                        save(f'event_{index}_{column}', frame[column].to_numpy())
                    event_rows.append(dict(k=k, seed=seed, surrogates=count, sub_step=step,
                                           results=frame.to_dict('records'), audit=audit))
                    if index == 0:
                        module.plot_band_event_mi(frame, title=case['name'] + ' (trigger/probe; quality disabled)',
                                                  outpath=str(folder/'events.png'))
                again, again_raw = load_merged_csv(source)
                np.testing.assert_array_equal(again, time)
                np.testing.assert_array_equal(again_raw, raw)
                # Independent event selection is repeated from reloaded source; audit fields are compared exactly.
                for row in event_rows:
                    for duration in sorted({a['Window_Size'] for a in row['audit']}):
                        _, expected_audit = module.select_event_windows(again, 500., case['events'], duration)
                        actual = [a for a in row['audit'] if a['Window_Size'] == duration and a['Channel'] == 'ch1']
                        for expected, got in zip(expected_audit, actual, strict=True):
                            for key, value in expected.items():
                                if got[key] != value:
                                    raise AssertionError('Event source audit differs: ' + key)
                meta.update(rows=sum(len(r['results']) for r in event_rows), event_rows=event_rows,
                            event_estimates=event_results, event_source_reloads=1)
                save('time_us', time)
            else:
                cli = ['spectral_entropy.py', '--csv', source, '--out', str(folder), '--joint-mi',
                       '--win', '2', '--step', '2', '--mi-bins', '16', '--mi-surrogates', '200']
                if case.get('raw'):
                    cli.append('--no-bandpass')
                quality = case.get('quality')
                if quality == 'disabled':
                    cli.append('--no-quality-mask')
                elif quality is not None:
                    value = float('nan') if quality == 'nan' else quality
                    stack.enter_context(patch.object(module, '_eeg_quality_v2',
                        side_effect=lambda data, **kw: dict(overall=np.full(data.shape[0], value))))
                stack.enter_context(patch('sys.argv', cli))
                try:
                    module.main()
                except ValueError as exc:
                    expected = case.get('status')
                    if expected == 'nonfinite_rejected' and str(exc).startswith('Non-finite EEG samples:'):
                        meta.update(status=expected, error=str(exc))
                    elif expected == 'short_rejected' and str(exc) == 'No complete analysis window within any continuous segment':
                        meta.update(status=expected, error=str(exc))
                    else:
                        raise
                for table in sorted(folder.glob('*.csv')):
                    sidecar = Path(str(table) + '.meta.json')
                    kind = json.loads(sidecar.read_text())['kind']
                    reader = load_joint_mi_table if kind == 'joint_mi' else load_joint_mi_summary
                    frame, metadata = reader(table, source)
                    meta['readers'].append(dict(path=table.name, kind=kind))
                    meta[table.stem] = dict(frame=frame.to_dict('list'), metadata=metadata)
                    for column in frame.select_dtypes(include=['number', 'bool']):
                        save(table.stem + '__' + column, frame[column].to_numpy())
                    if kind == 'joint_mi':
                        meta['rows'] = len(frame)
                if case.get('sensitivity'):
                    time, raw = load_merged_csv(source)
                    filtered = module.bandpass_filter(raw, fs=500., lo=.5, hi=45., time_us=time)
                    # Preserve full-source filtering before taking sample-count prefixes.
                    grid = build_window_grid(time, 500., 2., 2.)
                    save('lagged', module.compute_lagged_interhemispheric_sync_windowed(
                        filtered[:, 0], filtered[:, 1], fs=500., windows=grid, apply_bandpass=False))
                    sensitivity = []
                    for stage, values in [('raw', raw), ('filtered', filtered)]:
                        for n in (512, 2048, len(time)):
                            for bins in (8, 16, 32):
                                for binning in ('uniform', 'quantile'):
                                    result = module.compute_joint_probability(values[:n, 0], values[:n, 1], bins=bins, binning=binning)
                                    check_histogram(result)
                                    save(f'{stage}_{n}_{bins}_{binning}', result)
                                    sensitivity.append(dict(stage=stage, n=n, bins=bins, binning=binning,
                                        mi=result['mutual_information'], mm=result['mutual_information_mm']))
                    meta['sensitivity'] = sensitivity
                    save('segment_lengths', [s.stop-s.start for s in continuous_slices(time, 500.)])
        require_outcome(case['name'], case.get('status', 'complete'), meta['status'], meta['rows'])
        save('row_count', [meta['rows']])
        np.savez_compressed(folder/'numeric.npz', **arrays)
        np.savez_compressed(folder/'evidence.npz', **{k: v for k, v in arrays.items() if v.size})
        meta['empty_arrays'] = {k: list(v.shape) for k, v in arrays.items() if not v.size}
        write_json(folder/'metadata.json', clean(meta))
        print(f'PASS {case["name"]}: {meta["status"]}, {meta["rows"]} rows', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--label')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('local/\n!*.csv\n!*.png\n')
    config = prepare(out)
    frozen = out/'local/frozen_source'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT).decode().splitlines()
    names = [n for n in names if n.endswith('.py') and ('/' not in n or n.startswith('lilia/'))]
    config.update(frozen_source_hashes={}, current_source_hashes={},
        tools={n: sha(ROOT/n) for n in ('tools/freeze_mi_profiles.py', 'tools/compare_mi_methods.py',
                                       'tools/freeze_method_profiles.py', 'tools/validate_goertzel_quality_stage18.py')},
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn')})
    for name in names:
        path = frozen/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT))
        config['frozen_source_hashes'][name] = sha(path)
        config['current_source_hashes'][name] = sha(ROOT/name)
    write_json(out/'config.json', config)
    env = {**os.environ, 'MPLCONFIGDIR': str(out/'local/mpl'), 'MPLBACKEND': 'Agg', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    for label, code in [('expected', frozen), ('actual', ROOT)]:
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--code-root', str(code),
                '--config', str(out/'config.json'), '--out', str(out), '--label', label], stdout=log,
                stderr=subprocess.STDOUT, env=env, check=True)
        print('PASS ' + label + ' process', flush=True)
    cases = [dict(name='controls', real=False)] + config['cases']
    rows, tables, comparisons = [], [], []
    for case in cases:
        base = out / ('local' if case['real'] else 'synthetic')
        old, new = [base/label/case['name'] for label in ('expected', 'actual')]
        count, empty = compare_arrays(old/'numeric.npz', new/'numeric.npz')
        compare_metadata(json.loads((old/'metadata.json').read_text()),
                         json.loads((new/'metadata.json').read_text()),
                         joint_code_identity(config['frozen_source_hashes']),
                         joint_code_identity(config['current_source_hashes']))
        metadata = json.loads((new/'metadata.json').read_text())
        comparisons.append(dict(expected=str(old/'evidence.npz'), actual=str(new/'evidence.npz'), rtol=0., atol=0., equal_nan=True))
        for entry in metadata.get('readers', []):
            if entry['kind'] == 'joint_mi':
                for folder in (old, new):
                    tables.append(dict(path=str(folder/entry['path']), kind=entry['kind'], raw_csv=case['source']))
        rows.append(dict(case=case['name'], arrays=count, empty_arrays=empty,
                         readers=2*len(metadata.get('readers', [])),
                         event_source_reloads=2*metadata.get('event_source_reloads', 0)))
    write_json(out/'analysis.json', dict(cases=rows, arrays=sum(r['arrays'] for r in rows),
        reader_checks=sum(r['readers'] for r in rows), event_source_reloads=sum(r['event_source_reloads'] for r in rows),
        max_abs_error=0., rtol=0., atol=0., semantic_metadata_equal=True,
        source_fingerprint_fields_compared_via_readers=True,
        product_changes=config['frozen_source_hashes'] != config['current_source_hashes']))
    verify_files(config['inputs']+config['generated_inputs'], ROOT)
    verify_files([dict(path=n, sha256=d) for n, d in config['frozen_source_hashes'].items()], frozen)
    verify_files([dict(path=n, sha256=d) for n, d in {**config['current_source_hashes'], **config['tools']}.items()], ROOT)
    def relative(row):
        return {k: str(Path(v).relative_to(out)) if k in ('path', 'raw_csv', 'actual', 'expected') and Path(v).is_relative_to(out) else v for k, v in row.items()}
    files = [dict(path=str(p), sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and '/mpl/' not in str(p)]
    files += config['inputs'] + [dict(path=str(ROOT/n), sha256=d) for n, d in {**config['current_source_hashes'], **config['tools']}.items()]
    manifest = dict(schema_version=1, files=list(map(relative, files)), tables=list(map(relative, tables)), comparisons=list(map(relative, comparisons)))
    write_json(out/'manifest_local.json', manifest)
    originals = {r['path'] for r in config['inputs'] if '/jenqwei/' in r['path']}
    public = dict(schema_version=1, files=[r for r in manifest['files'] if not r['path'].startswith('local/') and r['path'] not in originals],
        tables=[r for r in manifest['tables'] if r['path'].startswith('synthetic/')],
        comparisons=[r for r in manifest['comparisons'] if r['actual'].startswith('synthetic/')])
    write_json(out/'manifest_repository.json', public)
    print(f'PASS {len(rows)} cases; {sum(r["arrays"] for r in rows)} exact arrays', flush=True)


if __name__ == '__main__':
    main()
