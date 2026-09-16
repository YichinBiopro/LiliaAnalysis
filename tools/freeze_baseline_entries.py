"""M1: compare complete baseline entrypoints against the fixed pre-calibration commit.

Run only into a new directory. Real sources, timestamp-converted copies and
their derivatives remain in local/. Synthetic products can be committed.
"""
import argparse
import contextlib
import csv
import datetime as dt
from decimal import Decimal
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.freeze_method_profiles import BASELINE, ROOT, sha, write_json, verify_files, compare_arrays


def canonical(value, output):
    """Ignore only renderer artifact hashes; retain scientific audits and hashes."""
    if isinstance(value, dict):
        return {k: canonical(v, output) for k, v in value.items() if k != 'artifacts'}
    if isinstance(value, list):
        return [canonical(v, output) for v in value]
    if isinstance(value, str):
        return value.replace(str(output), '<output>')
    return value


def check_outcome(case, error):
    expected = case.get('error_contains')
    if expected and (error is None or expected not in error):
        raise AssertionError(f'{case["name"]}: expected failure {expected!r}, got {error!r}')
    if error and not expected:
        raise AssertionError(f'{case["name"]}: unexpected failure: {error}')


def check_policy_witnesses(case, comparison):
    """Independent expected decisions for boundary cases, beyond old/new equality."""
    name, products = case['name'], comparison['products']
    if name == 'gap_markers':
        row = products['markers_summary.csv']
        if row['baseline_windows'] != [6, 6, 5, 0] or row['post_windows'] != [5, 6, 5, 0]:
            raise AssertionError('Marker half-open/fallback/gap selection differs')
    if name == 'equality_sampler':
        for metadata in comparison['extra'].values():
            if metadata['n_selected'] != 10 or not all(r['accepted'] for r in metadata['quality_audit']):
                raise AssertionError('Equality threshold must accept legacy sampler epochs')
    if name == 'equality_state':
        row = products['merged_baseline_event_entropy_ch1.csv']
        if row['n_windows'] != [30, 25] or row['status'] != ['accepted', 'accepted']:
            raise AssertionError('Equality threshold/gap state window counts differ')
    if name == 'gap_events_pre-event-rest':
        audit = next(v for k, v in products.items() if k.endswith('_analysis.json'))
        selected = audit['branches']['bp']['summary']['baseline_audit']
        if ([r['event'] for r in selected] != ['Probe A', 'Probe B']
                or selected[0]['baseline_metric_rows'] != list(range(6, 12))
                or selected[1]['bar_status'] != 'accepted'
                or selected[1]['heatmap_status'] != 'missing_baseline'):
            raise AssertionError('Nonparticipant/rest or bar/heatmap baseline policy differs')


def marker_clock(microseconds):
    value = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=int(microseconds), hours=8)
    return value.date().isoformat(), value.time().isoformat(timespec='microseconds')


def write_source(path, time, raw):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(raw, columns=[f'ch{i+1}' for i in range(raw.shape[1])])
    frame.insert(0, 'Time[us]', time)
    n = raw.shape[1]
    with path.open('x') as handle:
        handle.write('Device,M1 validation copy\nAmp Gain,500,Abs Time Offset[us],0\n'
                     + 'Channels,' + ','.join(str(i+1) for i in range(n)) + '\n'
                     + 'Sample Rate,' + ','.join(['500']*n) + '\n')
        frame.to_csv(handle, index=False)


def prepare(out):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from lilia.io import load_merged_csv
    import plot_tyy_meditation as meditation
    import plot_event_markers as event
    model = ROOT/'tiny_v4_optimized.tflite'
    inputs = [dict(path=str(model), sha256=sha(model))]
    sources, annotations, cases = {}, [], []
    for label in ('talk', 'move head'):
        source = ROOT/'jenqwei'/('2026-06-18-lilia-'+label)/('2026-06-18-lilia-'+label+'.csv')
        marker_file = source.parent/'time_marker.csv'
        inputs += [dict(path=str(p), sha256=sha(p)) for p in (source, marker_file)]
        with source.open() as handle:
            next(handle)
            header = next(csv.reader(handle))
        offset = int(Decimal(header[header.index('Abs Time Offset[us]')+1]))
        time, raw = load_merged_csv(source)
        with marker_file.open() as handle:
            rows = list(csv.DictReader(handle))
        markers = [int(Decimal(row['Abs_time(us)'])) for row in rows]
        key = label.replace(' ', '_')
        converted = out/'local/inputs'/key/'merged.csv'
        write_source(converted, time+offset, raw[:, :4])
        inputs.append(dict(path=str(converted), sha256=sha(converted)))
        sources[key] = dict(path=str(converted), epoch=int(time[0])+offset, markers=markers)
        annotations.append(dict(name=key, raw=str(source), raw_sha256=sha(source),
            marker_file=str(marker_file), marker_sha256=sha(marker_file), converted=str(converted),
            offset_us=offset, recipe='raw Time[us] + header offset; first four declared channels; float32 unchanged',
            marker_abs_us=markers, annotation_type='recorded keyboard triggers; no inferred behavioral start/end',
            abs_minus_offset_relative_us=[m-offset-int(Decimal(r['0-New_Task-recording_time(us)']))
                                         for m, r in zip(markers, rows)]))
        cases.append(dict(name=key+'_markers', kind='marker', source=str(converted), real=True,
                          markers=markers, baseline=10., post=10.))
        if key == 'talk':
            bounds = [(m-sources[key]['epoch'])/1e6 for m in markers]
            for clean in (False, True):
                cases.append(dict(name='talk_state_'+('clean' if clean else 'ordinary'), kind='state',
                    source=str(converted), real=True, baseline=[0., bounds[0]], event=bounds, clean=clean))
            cases.append(dict(name='talk_sampler', kind='sampler', source=str(converted), real=True,
                              trigger=markers[0]))
    tyy = ROOT/'iBrainCenter/TYY(SN041)/merged.csv'
    inputs.append(dict(path=str(tyy), sha256=sha(tyy)))
    cases += [dict(name='tyy_meditation', kind='meditation', source=str(tyy), real=True),
              dict(name='tyy_multievent', kind='events', source=str(tyy), real=True,
                   policy='pre-event-rest', model=False, subject='TYY')]
    annotations.append(dict(name='tyy_protocol', source=str(tyy), source_sha256=sha(tyy),
        annotation_type='existing project protocol schedule; not independently observed task timestamps',
        events=event.EVENTS, meditation_parameters=meditation.session_parameters(),
        schedule_code_sha256=sha(ROOT/'plot_event_markers.py'), meditation_code_sha256=sha(ROOT/'plot_tyy_meditation.py')))
    epoch = event.hhmm_to_us('14:10')
    n = 90000
    rng = np.random.default_rng(1911)
    seconds = np.arange(n)/500.
    raw = np.column_stack([20*np.sin(2*np.pi*(6+4*i)*seconds)+rng.normal(0, 3, n) for i in range(4)]).astype(np.float32)
    for key in ('continuous', 'gap'):
        time = epoch + np.arange(n, dtype=np.int64)*2000
        if key == 'gap':
            time[n//2:] += 10000000
        path = out/'synthetic/inputs'/key/'merged.csv'
        write_source(path, time, raw)
        inputs.append(dict(path=str(path), sha256=sha(path)))
        sources[key] = dict(path=str(path), epoch=int(epoch))
    continuous, gap = [sources[k]['path'] for k in ('continuous', 'gap')]
    parameters = {k: epoch+int(v*1e6) for k, v in dict(crop_start_us=32, crop_end_us=185,
        baseline_end_us=75, view_start_us=60, view_end_us=170, meditation_start_us=90, meditation_end_us=160).items()}
    cases += [dict(name='gap_meditation', kind='meditation', source=gap, real=False, parameters=parameters),
              dict(name='missing_meditation', kind='meditation', source=gap, real=False,
                   parameters={**parameters, 'baseline_end_us':epoch+40000000},
                   error_contains='Meditation analysis failed'),
              dict(name='gap_markers', kind='marker', source=gap, real=False,
                   markers=[epoch-5000000, epoch+32500000, epoch+95000000, epoch+250000000],
                   baseline=30., post=30., score=.5),
              dict(name='equality_sampler', kind='sampler', source=continuous, real=False,
                   trigger=epoch+30000000, score=.5),
              dict(name='low_sampler', kind='sampler', source=continuous, real=False,
                   trigger=epoch+30000000, score=.49, sampler_error='基線建構失敗'),
              dict(name='gap_sampler_guard', kind='sampler', source=gap, real=False,
                   trigger=epoch+30000000, sampler_error='Legacy raw baseline epoch sampler'),
              dict(name='gap_state', kind='state', source=gap, real=False,
                   baseline=[0., 60.], event=[60., 120.], clean=False),
              dict(name='equality_state', kind='state', source=gap, real=False,
                   baseline=[0., 60.], event=[60., 120.], clean=True, score=.5),
              dict(name='empty_state', kind='state', source=continuous, real=False,
                   baseline=[.001, 1.], event=[2., 2.5], clean=False,
                   error_contains='Baseline/event has no usable windows')]
    # Minute strings use the entry's existing clock parser; no fake real annotation.
    schedule = [('Nonparticipant', '14:10', .5, ['Other']), ('Probe A', '14:11', .5, None),
                ('Probe B', '14:12', .5, None)]
    for policy in ('session-start', 'pre-event-rest'):
        cases.append(dict(name='gap_events_'+policy, kind='events', source=gap, real=False,
            policy=policy, model=True, subject='Synthetic', schedule=schedule, score=.5))
    cases.append(dict(name='gap_tflite_summary', kind='tflite_summary', source=gap, real=False,
                      schedule=schedule, score=.5))
    return dict(schema_version=1, baseline_commit=BASELINE, model=str(model), inputs=inputs,
                cases=cases, annotations=annotations, synthetic_seed=1911)


def worker(args):
    sys.path.insert(0, str(args.code_root))
    import numpy as np
    import pandas as pd
    import plot_tyy_meditation as meditation
    import plot_event_markers as event
    import plot_tflite_summary as tflite_summary
    import plot_index_vs_raw as marker
    import spectral_entropy as entropy
    import lilia.tflite as inference
    from lilia.io import load_merged_csv, bandpass_filter
    from lilia.quality_audit import json_value
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt
    config = json.loads(args.config.read_text())
    verify_files(config['inputs'], ROOT)
    event.TFLITE_MODEL_PATH = tflite_summary.TFLITE_MODEL_PATH = config['model']
    inference_original = inference.apply_tflite_windowed
    finalise_original = entropy._finalise_comparison
    original_save = Figure.savefig
    for case in config['cases']:
        dest = args.out/('local' if case['real'] else 'synthetic')/args.label/case['name']
        dest.mkdir(parents=True, exist_ok=False)
        arrays, extra, error = {}, {}, None

        def save_numeric(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    save_numeric(prefix+'__'+str(key), item)
            elif isinstance(value, np.ndarray) and value.dtype.kind in 'biuf' and value.size:
                arrays[prefix] = value
            elif isinstance(value, (float, int, np.number)):
                arrays[prefix] = np.asarray([value])

        def finalise(*a, **kw):
            value = finalise_original(*a, **kw)
            save_numeric('entropy_comparison', value)
            return value

        def infer(*a, **kw):
            output = inference_original(*a, **kw)
            index = len([k for k in arrays if k.startswith('model_output_')])
            arrays[f'model_input_{index}'] = a[0]
            arrays[f'model_output_{index}'] = output
            return output

        def save(fig, path, **kw):
            # Save actual product figures; only lower raster resolution for evidence size.
            if str(path).endswith('.png'):
                kw['dpi'] = 100
            return original_save(fig, path, **kw)

        def fixed_score(data, **kw):
            return {'overall': np.full(data.shape[0], case['score'])}

        with (dest/'run.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(inference, 'apply_tflite_windowed', infer))
            stack.enter_context(patch.object(entropy, '_finalise_comparison', finalise))
            stack.enter_context(patch.object(Figure, 'savefig', save))
            if 'schedule' in case:
                stack.enter_context(patch.object(event, 'EVENTS', case['schedule']))
                stack.enter_context(patch.object(tflite_summary, 'EVENTS', case['schedule']))
                stack.enter_context(patch.object(event, 'CONE_STAGES', []))
                stack.enter_context(patch.object(tflite_summary, 'CONE_STAGES', []))
            if 'score' in case:
                for module, name in [(event, 'get_eeg_quality_index_v2_parametric'),
                                     (tflite_summary, 'get_eeg_quality_index_v2_parametric'),
                                     (entropy, '_eeg_quality_v2')]:
                    stack.enter_context(patch.object(module, name, fixed_score))
            try:
                source = Path(case['source'])
                if case['kind'] == 'meditation':
                    meditation.plot_tyy_meditation(dest, ds=500, csv_path=source,
                        model_path=config['model'], parameters=case.get('parameters'))
                elif case['kind'] == 'marker':
                    clocks = [marker_clock(v) for v in case['markers']]
                    for value, (date, clock) in zip(case['markers'], clocks):
                        if marker._parse_marker_us(clock, date)[0] != value:
                            raise AssertionError('Marker parser lost integer microsecond precision')
                    marker.plot_custom_markers(str(source), case['name'], 1, [v[1] for v in clocks], clocks[0][0],
                        case['baseline'], case['post'], str(dest/'markers.png'), str(dest/'markers_summary.csv'))
                elif case['kind'] == 'sampler':
                    time, raw = load_merged_csv(source)
                    filtered = bandpass_filter(raw, time_us=time)
                    for label in ('session', 'pre_event'):
                        try:
                            if label == 'session':
                                data, meta = tflite_summary.build_session_baseline(time, filtered, data_raw_full=raw)
                            else:
                                data, meta = tflite_summary.build_baseline_epochs(time, filtered, case['trigger'], data_raw_full=raw)
                            arrays['sampler_'+label] = data
                            extra[label] = meta
                            if case.get('sampler_error'):
                                raise AssertionError('Expected sampler failure was not raised')
                        except ValueError as exc:
                            if not case.get('sampler_error') or case['sampler_error'] not in str(exc):
                                raise
                            extra[label] = {'expected_error': str(exc)}
                    # Empty sampler cases still carry source-bound time for exact dtype/shape comparison.
                    arrays['sampler_source_time'] = time
                elif case['kind'] == 'state':
                    argv = ['spectral_entropy.py', '--csv', str(source), '--out', str(dest),
                            '--baseline', *map(str, case['baseline']), '--event', *map(str, case['event'])]
                    if case['clean']:
                        argv.append('--clean')
                    stack.enter_context(patch('sys.argv', argv))
                    entropy.main()
                elif case['kind'] == 'events':
                    event.plot_subject(case['subject'], dict(dir=source.parent.name, sn='M1'), str(dest), 500,
                        base_dir=str(source.parent.parent), use_tflite=case['model'], baseline_mode=case['policy'])
                elif case['kind'] == 'tflite_summary':
                    tflite_summary.plot_subject_tflite_summary('Synthetic', dict(dir=source.parent.name, sn='M1'),
                        str(dest), base_dir=str(source.parent.parent), heatmap_baseline_mode='both', on_insufficient='skip')
                else:
                    raise ValueError('Unknown case kind')
            except ValueError as exc:
                error = str(exc)
            finally:
                plt.close('all')
        check_outcome(case, error)
        products, tables = {}, []
        for path in sorted(dest.glob('*.csv')):
            frame = pd.read_csv(path, float_precision='round_trip')
            products[path.name] = json_value(frame.to_dict(orient='list'))
            for column in frame:
                values = frame[column].to_numpy()
                if values.dtype.kind in 'biuf':
                    arrays[path.stem+'__'+column] = values
            sidecar = Path(str(path)+'.meta.json')
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text())
                tables.append(dict(path=str(path), kind=metadata['kind'], raw_csv=case['source'], model_path=config['model']))
        for path in sorted(dest.glob('*.json')):
            products[path.name] = json.loads(path.read_text())
        if not arrays:
            raise AssertionError('No numerical output captured: '+case['name'])
        np.savez_compressed(dest/'numeric.npz', **arrays)
        write_json(dest/'comparison.json', canonical(json_value(dict(error=error, extra=extra, products=products)), dest))
        write_json(dest/'tables.json', tables)
        print(f'PASS {args.label} {case["name"]}: {len(arrays)} arrays, {len(tables)} tables', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    parser.add_argument('--cases', nargs='+', help='Explicit subset for development; omitted means all cases')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('local/\n!*.png\n!*.svg\n!*.csv\n')
    os.environ.update(MPLCONFIGDIR=str(out/'local/mpl'), MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    config = prepare(out)
    if args.cases:
        known = {r['name'] for r in config['cases']}
        if set(args.cases)-known:
            raise ValueError('Unknown requested case')
        config['cases'] = [r for r in config['cases'] if r['name'] in args.cases]
    frozen = out/'local/frozen_source'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT).decode().splitlines()
    names = [n for n in names if n.endswith('.py') and ('/' not in n or n.startswith('lilia/'))]
    config.update(frozen_source_hashes={}, current_source_hashes={}, harness_sha256=sha(__file__),
                  harness_dependency_sha256=sha(ROOT/'tools/freeze_method_profiles.py'),
                  python=sys.version, packages={p: importlib.metadata.version(p) for p in
                    ('numpy', 'scipy', 'pandas', 'matplotlib', 'tensorflow')})
    for name in names:
        path = frozen/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT))
        config['frozen_source_hashes'][name] = sha(path)
        config['current_source_hashes'][name] = sha(ROOT/name)
    write_json(out/'config.json', config)
    for label, root in [('expected', frozen), ('actual', ROOT)]:
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--out', str(out),
                '--label', label, '--code-root', str(root), '--config', str(out/'config.json')],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PASS '+label+' worker', flush=True)
    sys.path.insert(0, str(ROOT))
    from refactor_check import READERS
    results, tables, comparisons = [], [], []
    for case in config['cases']:
        folder = out/('local' if case['real'] else 'synthetic')
        old, new = [folder/label/case['name'] for label in ('expected', 'actual')]
        count, empty = compare_arrays(old/'numeric.npz', new/'numeric.npz')
        if (old/'comparison.json').read_bytes() != (new/'comparison.json').read_bytes():
            raise AssertionError('Scientific metadata/selection/summary differs: '+case['name'])
        check_policy_witnesses(case, json.loads((new/'comparison.json').read_text()))
        if empty:
            raise AssertionError('Unexpected empty numeric arrays: '+case['name'])
        actual_tables = json.loads((new/'tables.json').read_text())
        for row in actual_tables:
            module, function, needs_model = READERS[row['kind']]
            reader = getattr(importlib.import_module('lilia.'+module), function)
            kw = dict(model_path=config['model']) if needs_model else {}
            reader(row['path'], row['raw_csv'], **kw)
        tables += actual_tables
        comparisons.append(dict(expected=str(old/'numeric.npz'), actual=str(new/'numeric.npz'),
                                rtol=0., atol=0., equal_nan=True))
        results.append(dict(name=case['name'], arrays=count, tables=len(actual_tables),
                            real=case['real'], expected_failure=bool(case.get('error_contains') or case.get('sampler_error'))))
        print(f'PASS compare/readers {case["name"]}', flush=True)
    verify_files(config['inputs'], ROOT)
    if any(sha(ROOT/n) != digest for n, digest in config['current_source_hashes'].items()) or sha(__file__) != config['harness_sha256']:
        raise ValueError('Code changed during capture')
    write_json(out/'analysis.json', dict(cases=results, array_count=sum(r['arrays'] for r in results),
        reader_checks=len(tables), baseline_commit=BASELINE, max_abs_error=0., tolerance=0.,
        metadata_equal=True, scientific_calibration=False))
    files = [dict(path=str(p), sha256=sha(p)) for p in sorted(out.rglob('*'))
             if p.is_file() and '/mpl/' not in str(p)]
    files += config['inputs'] + [dict(path=str(ROOT/n), sha256=d) for n, d in config['current_source_hashes'].items()]
    files += [dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
              dict(path=str(ROOT/'tools/freeze_method_profiles.py'), sha256=config['harness_dependency_sha256'])]
    def relative(row):
        return {k: str(Path(v).relative_to(out)) if k in ('path', 'raw_csv', 'model_path', 'actual', 'expected')
                and Path(v).is_relative_to(out) else v for k, v in row.items()}
    manifest = dict(schema_version=1, files=[relative(r) for r in files],
                    tables=[relative(r) for r in tables], comparisons=[relative(r) for r in comparisons])
    write_json(out/'manifest_local.json', manifest)
    input_paths = {r['path'] for r in config['inputs']}
    public = dict(schema_version=1,
        files=[r for r in manifest['files'] if not r['path'].startswith('local/') and r['path'] not in input_paths],
        tables=[r for r in manifest['tables'] if r['path'].startswith('synthetic/') and not READERS[r['kind']][2]],
        comparisons=[r for r in manifest['comparisons'] if r['actual'].startswith('synthetic/')])
    write_json(out/'manifest_repository.json', public)
    print(f'PASS {len(results)} cases, {sum(r["arrays"] for r in results)} exact arrays, {len(tables)} readers')


if __name__ == '__main__':
    main()
