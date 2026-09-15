"""Compare Goertzel caller diagnostics with the frozen pre-R5 numerical path."""
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
import numpy as np
import pandas as pd

import plot_goertzel_vs_raw as current
from lilia import quality
from lilia.goertzel_io import load_goertzel_table
from lilia.io import load_merged_csv
from lilia.provenance import file_sha256
from lilia.quality_policy import valid_goertzel_rows

COMMIT = '40d2a233473c3aced382f6eea7f1473842f38cf6'
DEPENDENCIES = ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py',
                'lilia/quality.py', 'lilia/goertzel.py', 'lilia/quality_policy.py')


def run(module, source, out, *, win=5., step=5., ch=1, fs=500., mode='', plots=False):
    out.mkdir()
    with contextlib.ExitStack() as stack:
        log = stack.enter_context((out / 'run.log').open('w'))
        stack.enter_context(contextlib.redirect_stdout(log))
        stack.enter_context(contextlib.redirect_stderr(log))
        if not plots:
            stack.enter_context(patch.object(module, '_render_plot'))
        if mode in ('external', 'all_rejected'):
            def scorer(data, fs, params):
                return {'overall': np.full(data.shape[0], .7 if mode == 'external' else 0.)}
            stack.enter_context(patch.object(module, 'get_eeg_quality_index_v2_parametric', side_effect=scorer))
        elif mode == 'fallback':
            params = {key+'_weight': float(key == 'spectrum') for key in ('flat', 'spectrum', 'kurtosis', 'corr')}
            stack.enter_context(patch.object(module, 'get_ibrain_device_eeg_quality_v2_params', return_value=params))
            stack.enter_context(patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced spectrum failure')))
        try:
            module._plot_subject(str(source), str(out/'plot.png'), str(out/'metrics.csv'),
                ch, fs, 60., win, step, 1, .49 if mode == 'fallback' else .5,
                (-150., 150.), True, 1950., .12, 1000., 1., 80.)
        except ValueError as exc:
            if not str(exc).startswith('Non-finite EEG samples:'):
                raise
            return str(exc)
    return None


def write_source(path, time_us, raw):
    frame = pd.DataFrame(raw, columns=[f'ch{i+1}' for i in range(raw.shape[1])])
    frame.insert(0, 'Time[us]', time_us)
    path.write_text('File Name,validation copy\nAmp Gain,500,Abs Time Offset[us],0\n'
                    'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('# Bounded R5 acceptance outputs.\n!*.csv\n!*.png\n!*.svg\n')
    code = subprocess.check_output(['git', 'show', f'{COMMIT}:plot_goertzel_vs_raw.py'], cwd=ROOT)
    (out/'plot_goertzel_vs_raw.legacy.txt').write_bytes(code)
    baseline = types.ModuleType('legacy_goertzel_quality')
    baseline.__file__ = str(ROOT / 'plot_goertzel_vs_raw.py')
    sys.modules[baseline.__name__] = baseline
    exec(compile(code, baseline.__file__, 'exec'), baseline.__dict__)
    dependencies = {}
    for name in DEPENDENCIES:
        old = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
        digest = hashlib.sha256(old).hexdigest()
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f'Numerical dependency changed: {name}')
        dependencies[name] = digest
    fixtures = json.loads((ROOT/'tests/fixtures/quality_stage18_reference.json').read_text())
    cases = []
    for index, entry in enumerate(fixtures['sources']):
        source = Path(entry['path'])
        if file_sha256(source) != entry['sha256']:
            raise ValueError('Real source differs from saved baseline')
        for ch in (1, 2):
            cases.append(dict(name=f'real_{index}_ch{ch}', source=source, ch=ch,
                              plots=(index == 1 and ch == 1)))
    talk = Path(fixtures['sources'][1]['path'])
    _, raw = load_merged_csv(talk)
    raw = raw[:4000].copy()
    t = np.r_[np.arange(2000)*2000, 20000000 + np.arange(2000)*2000].astype(np.int64)
    gap = out/'gap.csv'
    write_source(gap, t, raw)
    artifact = raw.copy()
    artifact[:500, 0] = 2000.
    hard = out/'hard.csv'
    write_source(hard, t, artifact)
    nonfinite = out/'nonfinite.csv'
    artifact[100, 0] = np.nan
    write_source(nonfinite, t, artifact)
    cases += [dict(name='gap', source=gap, win=2., step=1., plots=True),
              dict(name='hard', source=hard, win=2., step=1.),
              dict(name='half_second', source=talk, win=.5, step=.5, plots=True),
              dict(name='fallback', source=talk, mode='fallback', plots=True),
              dict(name='external', source=gap, win=2., step=1., mode='external', plots=True),
              dict(name='all_rejected', source=gap, win=2., step=1., mode='all_rejected'),
              dict(name='empty', source=gap, plots=True),
              dict(name='fractional_fs', source=talk, fs=499.5),
              dict(name='nonfinite', source=nonfinite)]
    expected, actual, results = {}, {}, []
    manifest = dict(schema_version=1, files=[], tables=[], comparisons=[])
    for case in cases:
        name, source = case['name'], case['source']
        kwargs = {key: value for key, value in case.items() if key not in ('name', 'source', 'plots')}
        old_dir, new_dir = out/(name+'_old'), out/(name+'_new')
        old_failure = run(baseline, source, old_dir, **kwargs)
        new_failure = run(current, source, new_dir, **kwargs, plots=case.get('plots', False))
        if old_failure != new_failure:
            raise AssertionError('Goertzel failure behavior changed')
        row = dict(case=name, source=str(source), unchanged_failure=old_failure, rows=0, readers=0)
        if old_failure is None:
            old = pd.read_csv(old_dir/'metrics.csv', float_precision='round_trip')
            with (new_dir/'reader.log').open('w') as log, contextlib.redirect_stderr(log):
                new, meta = load_goertzel_table(new_dir/'metrics.csv', source)
            # Empty CSVs carry no dtype information; their columns/shape still must match.
            pd.testing.assert_frame_equal(old, new[list(old)], check_exact=True, check_dtype=bool(len(old)))
            for key in old:
                array_key = name+'__'+key
                expected[array_key], actual[array_key] = old[key].to_numpy(dtype=float) if not len(old) else old[key].to_numpy(), new[key].to_numpy()
            selections = {}
            for exclude_hard in (True, False):
                for threshold in (.49, .5):
                    a = valid_goertzel_rows(old, threshold, exclude_hard=exclude_hard)
                    b = valid_goertzel_rows(new, threshold, exclude_hard=exclude_hard)
                    np.testing.assert_array_equal(a, b)
                    key = f'{name}__keep_{exclude_hard}_{threshold}'
                    expected[key], actual[key] = a, b
                    selections[key] = int(b.sum())
            # Source reader deliberately supports old complete tables without claiming diagnostics.
            with (old_dir/'reader.log').open('w') as log, contextlib.redirect_stderr(log):
                load_goertzel_table(old_dir/'metrics.csv', source)
            manifest['tables'].append(dict(path=str(new_dir/'metrics.csv'), kind='goertzel', raw_csv=str(source)))
            row.update(rows=len(new), readers=2, selections=selections,
                       diagnostic_counts=new.quality_diagnostic_state.value_counts().to_dict(),
                       hard_windows=int(new.artifact_hard_clip.sum()),
                       feature_sources=meta['feature_sources'])
        results.append(row)
        print(f'PASS {name}: {row["rows"]} windows, {row["readers"]} readers', flush=True)
    # Direct helper pollution retains the historical numerical outcome; CLI rejects that source.
    for label, module, arrays in [('old', baseline, expected), ('new', current, actual)]:
        with (out/f'nonfinite_helper_{label}.log').open('w') as log, contextlib.redirect_stderr(log):
            result = module._compute_window_metrics(t, artifact, artifact, 500., 1, 60., .5, .5,
                quality.get_ibrain_device_eeg_quality_v2_params(), 1950., .12, 1000., 1., 80.)
        for key in baseline.WindowResult.__dataclass_fields__:
            arrays['nonfinite_helper__'+key] = getattr(result, key)
        if label == 'new':
            (out/'nonfinite_helper_audit.json').write_text(json.dumps(result.quality_audit, indent=2))
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    # The general evidence checker intentionally rejects empty arrays. Their exact empty
    # shapes/columns are checked above and recorded separately rather than weakening it.
    empty_keys = [key for key, values in expected.items() if not values.size]
    np.savez_compressed(out/'expected.npz', **{k: v for k, v in expected.items() if k not in empty_keys})
    np.savez_compressed(out/'actual.npz', **{k: v for k, v in actual.items() if k not in empty_keys})
    manifest['comparisons'].append(dict(expected='expected.npz', actual='actual.npz', rtol=0., atol=0., equal_nan=True))
    paths = {*out.rglob('*'), *(case['source'] for case in cases), *(ROOT/name for name in DEPENDENCIES),
             ROOT/'plot_goertzel_vs_raw.py', ROOT/'lilia/goertzel_io.py', ROOT/'lilia/provenance.py',
             ROOT/'lilia/quality_audit.py', Path(__file__).resolve()}
    def relative(path):
        path = Path(path)
        return str(path.relative_to(out)) if path.is_relative_to(out) else str(path)
    for path in sorted(paths):
        if path.is_file():
            manifest['files'].append(dict(path=relative(path), sha256=file_sha256(path)))
    for entry in manifest['tables']:
        entry['path'], entry['raw_csv'] = relative(entry['path']), relative(entry['raw_csv'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    analysis = dict(commit=COMMIT, legacy_entry_sha256=hashlib.sha256(code).hexdigest(),
        numerical_dependencies=dependencies, cases=results, array_count=len(actual),
        persisted_nonempty_arrays=len(actual)-len(empty_keys), empty_arrays=empty_keys,
        max_abs_error=0., tolerance=0., nan_and_selection_equal=True,
        reader_checks=sum(row['readers'] for row in results), plots_reviewed=False)
    (out/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    print(f'PASS {len(cases)} cases + polluted helper; {len(actual)} exact arrays; {analysis["reader_checks"]} readers')


if __name__ == '__main__':
    main()
