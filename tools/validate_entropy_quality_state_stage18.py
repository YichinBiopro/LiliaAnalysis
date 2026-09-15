"""Compare entropy/MI quality-state fixes with the committed numerical caller."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import spectral_entropy as current
from lilia import quality
from lilia.entropy_io import load_entropy_table, load_joint_mi_table, load_joint_mi_summary
from lilia.provenance import file_sha256
from lilia.state_entropy_io import load_state_entropy_table

COMMIT = '40d2a233473c3aced382f6eea7f1473842f38cf6'
DEPENDENCIES = ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py',
                'lilia/state_windows.py', 'lilia/quality.py')


def run(module, source, out, mode, extra=(), *, reject=False, plots=False, fallback=False):
    out.mkdir()
    args = ['spectral_entropy.py', '--csv', str(source), '--out', str(out), '--win', '2']
    if mode == 'joint':
        args += ['--joint-mi', '--step', '1', '--mi-bins', '8', '--mi-surrogates', '3']
    elif mode == 'entropy':
        args += ['--step', '1', '--sync-pair', '1', '2', '--tau-ms', '10', '--mi-bins', '8']
    else:
        args += ['--baseline', '0', '4', '--event', '4', '8']
        if mode == 'clean':
            args += ['--clean']
    args += list(extra)
    failure = None
    with contextlib.ExitStack() as stack:
        log = stack.enter_context((out / 'run.log').open('w'))
        stack.enter_context(contextlib.redirect_stdout(log))
        stack.enter_context(contextlib.redirect_stderr(log))
        stack.enter_context(patch('sys.argv', args))
        if not plots:
            stack.enter_context(patch.object(Figure, 'savefig'))
            for name in ('plot_joint_distribution', 'plot_joint_excess', 'plot_band_entropy',
                         'plot_band_composition', 'plot_band_ternary', 'plot_focus_relax_scatter'):
                stack.enter_context(patch.object(module, name))
        if reject:
            stack.enter_context(patch.object(module, '_eeg_quality_v2', return_value={'overall': [0., 0.]}))
        if fallback:
            params = {key+'_weight': float(key == 'spectrum') for key in ('flat', 'spectrum', 'kurtosis', 'corr')}
            def scorer(data, **kwargs):
                with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced spectrum failure')):
                    return quality.get_eeg_quality_index_v2_parametric(data, **kwargs)
            stack.enter_context(patch.object(module, '_QUALITY_PARAMS', params))
            stack.enter_context(patch.object(module, '_eeg_quality_v2', side_effect=scorer))
        try:
            module.main()
        except ValueError as exc:
            if not ((mode == 'clean' and 'audit saved' in str(exc))
                    or str(exc).startswith('Non-finite EEG samples:')):
                raise
            failure = str(exc)
    return failure


def compare_tables(old_dir, new_dir, source, mode, expected, actual, prefix):
    rows, tables = [], []
    readers = {'joint_mi': load_joint_mi_table, 'band_entropy': load_entropy_table,
               'state_band_entropy': load_state_entropy_table}
    for old_path in sorted(old_dir.glob('*.csv')):
        new_path = new_dir / old_path.name
        old = pd.read_csv(old_path, dtype={'source_id': str, 'config_id': str}, float_precision='round_trip')
        new = pd.read_csv(new_path, dtype={'source_id': str, 'config_id': str}, float_precision='round_trip')
        columns = [key for key in old if key != 'config_id']
        pd.testing.assert_frame_equal(old[columns], new[columns], check_exact=True)
        for key in columns:
            if old[key].dtype.kind in 'biuf':
                array_key = prefix + '__' + old_path.stem + '__' + key
                expected[array_key], actual[array_key] = old[key].to_numpy(), new[key].to_numpy()
        meta_path = Path(str(new_path) + '.meta.json')
        meta = json.loads(meta_path.read_text())
        if meta['kind'] == 'joint_mi_summary':
            frame, _ = load_joint_mi_summary(new_path, source)
            rows.append({'path': str(new_path), 'kind': meta['kind'], 'rows': len(frame)})
        else:
            for path in (old_path, new_path):
                frame, info = readers[meta['kind']](path, source)
                tables.append({'path': str(path), 'kind': info['kind'], 'raw_csv': str(source)})
                rows.append({'path': str(path), 'kind': info['kind'], 'rows': len(frame)})
            if mode in ('state', 'clean'):
                old_meta = json.loads(Path(str(old_path) + '.meta.json').read_text())
                # Only the explicitly additive diagnostic record may differ.
                projected = json.loads(json.dumps(meta['states']))
                for state in projected.values():
                    for row in state['windows']:
                        row.pop('quality_diagnostics', None)
                if projected != old_meta['states']:
                    raise AssertionError('State window selection/quality/PSD audit changed')
    return rows, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostics', action='store_true', help='Also exercise raw, short and finite-fallback callers')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    code = subprocess.check_output(['git', 'show', f'{COMMIT}:spectral_entropy.py'], cwd=ROOT)
    (out / 'spectral_entropy.legacy.txt').write_bytes(code)
    baseline = types.ModuleType('legacy_entropy_quality_state')
    baseline.__file__ = str(ROOT / 'spectral_entropy.py')
    exec(compile(code, baseline.__file__, 'exec'), baseline.__dict__)
    dependencies = {}
    for name in DEPENDENCIES:
        old = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
        digest = hashlib.sha256(old).hexdigest()
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f'Numerical dependency changed: {name}')
        dependencies[name] = digest
    fixtures = json.loads((ROOT / 'tests/fixtures/quality_stage18_reference.json').read_text())
    cases = []
    for index, entry in enumerate(fixtures['sources']):
        source = Path(entry['path'])
        if file_sha256(source) != entry['sha256']:
            raise ValueError('Real source differs from saved baseline')
        for mode in ('joint', 'entropy', 'state', 'clean'):
            cases.append(dict(name=f'real_{index}_{mode}', source=source, mode=mode,
                              plots=(index == 1 and mode == 'joint')))
    synthetic = out / 'gap.csv'
    t = np.r_[np.arange(2000) * 2000, 20000000 + np.arange(2000) * 2000]
    x = np.random.default_rng(427).normal(size=(len(t), 2)).astype(np.float32)
    frame = pd.DataFrame({'Time[us]': t, 'ch1': x[:, 0], 'ch2': x[:, 1]})
    synthetic.write_text('File Name,synthetic\nAmp Gain,500,Abs Time Offset[us],0\n'
                         'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))
    nonfinite = out / 'gap_nonfinite.csv'
    frame.loc[500, 'ch1'] = np.nan
    nonfinite.write_text('File Name,synthetic\nAmp Gain,500,Abs Time Offset[us],0\n'
                        'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))
    cases += [dict(name='gap_joint', source=synthetic, mode='joint', extra=('--no-bandpass', '--no-quality-mask')),
              dict(name='gap_entropy', source=synthetic, mode='entropy', extra=('--no-bandpass', '--no-quality-mask')),
              dict(name='nonfinite_joint', source=nonfinite, mode='joint', extra=('--no-bandpass',)),
              dict(name='nonfinite_entropy', source=nonfinite, mode='entropy', extra=('--no-bandpass',)),
              dict(name='all_rejected_joint', source=Path(fixtures['sources'][1]['path']), mode='joint', reject=True, plots=True),
              dict(name='disabled_joint', source=Path(fixtures['sources'][1]['path']), mode='joint', extra=('--no-quality-mask',))]
    if args.diagnostics:
        talk = Path(fixtures['sources'][1]['path'])
        for mode in ('joint', 'entropy', 'clean'):
            cases += [dict(name='raw_' + mode, source=talk, mode=mode, extra=('--no-bandpass',)),
                      dict(name='half_second_' + mode, source=talk, mode=mode,
                           extra=('--win', '.5', '--step', '.5')),
                      dict(name='fallback_' + mode, source=talk, mode=mode, fallback=True, plots=True)]
    expected, actual, results = {}, {}, []
    manifest = dict(schema_version=1, files=[], tables=[], comparisons=[])
    for case in cases:
        name, source, mode = case['name'], case['source'], case['mode']
        old_dir, new_dir = out / (name + '_old'), out / (name + '_new')
        kwargs = dict(extra=case.get('extra', ()), reject=case.get('reject', False), fallback=case.get('fallback', False))
        old_failure = run(baseline, source, old_dir, mode, **kwargs)
        new_failure = run(current, source, new_dir, mode, plots=case.get('plots', False), **kwargs)
        if old_failure != new_failure:
            raise AssertionError('CLI failure behavior changed')
        rows, tables = compare_tables(old_dir, new_dir, source, mode, expected, actual, name)
        manifest['tables'] += tables
        results.append(dict(case=name, source=str(source), source_sha256=file_sha256(source),
                            unchanged_failure=old_failure, readers=rows))
        print(f'PASS {name}: {len(rows)} reader checks', flush=True)
    old_npz, new_npz = out / 'expected.npz', out / 'actual.npz'
    np.savez_compressed(old_npz, **expected)
    np.savez_compressed(new_npz, **actual)
    manifest['comparisons'].append(dict(expected=str(old_npz), actual=str(new_npz), rtol=0., atol=0., equal_nan=True))
    sources = {case['source'] for case in cases}
    for path in sorted({*out.rglob('*'), *sources, *(ROOT / name for name in DEPENDENCIES),
                        *(ROOT / 'lilia' / name for name in ('entropy_io.py', 'state_entropy_io.py',
                          'entropy_quality.py', 'quality_audit.py', 'neural_io.py')),
                        ROOT / 'spectral_entropy.py', Path(__file__).resolve()}):
        if path.is_file():
            manifest['files'].append(dict(path=str(path), sha256=file_sha256(path)))
    # Make persisted evidence movable together, while keeping formal inputs explicit.
    def relative(path):
        p = Path(path)
        return str(p.relative_to(out)) if p.is_relative_to(out) else str(p)
    for entry in manifest['files'] + manifest['tables']:
        entry['path'] = relative(entry['path'])
        if 'raw_csv' in entry:
            entry['raw_csv'] = relative(entry['raw_csv'])
    for entry in manifest['comparisons']:
        entry['actual'], entry['expected'] = relative(entry['actual']), relative(entry['expected'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    analysis = dict(commit=COMMIT, numerical_dependencies=dependencies, cases=results,
                    array_count=len(actual), max_abs_error=0., nan_and_selection_equal=True,
                    reader_checks=sum(len(row['readers']) for row in results),
                    plots_reviewed=False)
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(f'PASS {len(cases)} cases; {len(actual)} exact arrays; {analysis["reader_checks"]} readers')


if __name__ == '__main__':
    main()
