"""R6: compare quality_check numerical callers with their frozen pre-diagnostic entry."""
from __future__ import annotations

import argparse
import contextlib
import copy
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
import numpy as np
import pandas as pd

import quality_check as current
from lilia import quality
from lilia.provenance import file_sha256
from lilia.quality_check_io import load_quality_samples_table, load_quality_anomalies_table

COMMIT = '40d2a233473c3aced382f6eea7f1473842f38cf6'
DEPENDENCIES = ('lilia/io.py', 'lilia/signal.py', 'lilia/quality.py',
                'lilia/qeeg.py', 'lilia/constants.py', 'lilia/segment_sampling.py',
                'lilia/subject_paths.py', 'plot_event_markers.py')


def write_source(path, t, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
    frame.insert(0, 'Time[us]', t)
    path.write_text('File Name,validation copy\nAmp Gain,500,Abs Time Offset[us],0\n'
                    'Channels,1,2,3,4\nSample Rate,500,500\n'+frame.to_csv(index=False))


def run(module, case, folder, *, plots=False):
    folder.mkdir()
    trace, flagged_rows, starts = [], [], []
    mode, source = case['mode'], case['source']
    core_scorer = quality.get_eeg_quality_index_v2_parametric
    sampler = module.pick_non_overlapping_segments
    flagger = module.flag_anomalies
    def scored(data, fs, params):
        if case.get('scorer') in ('nan', 'zero', 'external'):
            value = {'nan': np.nan, 'zero': 0., 'external': .7}[case['scorer']]
            result = {'overall': np.full(data.shape[0], value)}
        else:
            if case.get('fallback'):
                # Only the quality scorer gets the forced fallback. Welch also
                # belongs to the unchanged qEEG/PSD paths in samples.
                with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced spectrum failure')):
                    result = core_scorer(data, fs=fs, params=params)
            else:
                result = core_scorer(data, fs=fs, params=params)
        trace.append({key: copy.deepcopy(result[key]) for key in ('overall',) if key in result})
        return result
    def sampled(*args, **kwargs):
        chosen = sampler(*args, **kwargs)
        starts.extend(chosen)
        return chosen
    def flagged(rows, *args, **kwargs):
        result = flagger(rows, *args, **kwargs)
        flagged_rows.extend(copy.deepcopy(result))
        return result
    failure = None
    with contextlib.ExitStack() as stack:
        log = stack.enter_context((folder/'run.log').open('w'))
        stack.enter_context(contextlib.redirect_stdout(log))
        stack.enter_context(contextlib.redirect_stderr(log))
        if not plots:
            stack.enter_context(patch('matplotlib.figure.Figure.savefig'))
        stack.enter_context(patch.object(module, 'get_eeg_quality_index_v2_parametric', side_effect=scored))
        if mode == 'samples':
            stack.enter_context(patch.object(module, 'pick_non_overlapping_segments', side_effect=sampled))
        else:
            stack.enter_context(patch.object(module, 'flag_anomalies', side_effect=flagged))
        try:
            if mode == 'samples':
                module.plot_segments(str(source), 'Quality', case['name'], str(folder),
                                     np.random.default_rng(case.get('seed', 42)))
            else:
                module.plot_quality_anomaly_report(case['name'],
                    {'dir': source.parent.name, 'sn': 'R6'}, str(folder),
                    base_dir=str(source.parent.parent), win_sec=case.get('win', module.QUALITY_WIN_SEC))
        except ValueError as exc:
            if not str(exc).startswith('Non-finite EEG samples:'):
                raise
            failure = str(exc)
    return dict(trace=trace, rows=flagged_rows, starts=starts, failure=failure)


def compare_rows(old, new, name):
    if len(old) != len(new):
        raise AssertionError(f'{name}: anomaly row count changed')
    intentional = []
    for i, (before, after) in enumerate(zip(old, new)):
        for key in ('s', 'e', 'mid_us', 'qmed', 'clip', 'flat', 'gap'):
            a, b = before[key], after[key]
            if isinstance(a, float):
                if not np.isclose(a, b, rtol=0., atol=0., equal_nan=True):
                    raise AssertionError(f'{name}: {key} changed at row {i}')
            elif a != b:
                raise AssertionError(f'{name}: {key} changed at row {i}')
        legacy = before['reasons']
        reasons = after['reasons']
        if np.isfinite(after['qmed']):
            if legacy != reasons or before['severity'] != after['severity']:
                raise AssertionError(f'{name}: finite anomaly selection changed at row {i}')
        elif (reasons != legacy + ['invalid-Q non-finite'] or
              not np.isfinite(after['severity'])):
            # The new reason is inserted before any jump; nonfinite windows never
            # contribute a jump. The saved old reasons may include clip/flat/gap.
            prefix = [reason for reason in reasons if not reason.startswith('invalid-Q')]
            if prefix != legacy or 'invalid-Q non-finite' not in reasons or not np.isfinite(after['severity']):
                raise AssertionError(f'{name}: unexpected NaN anomaly change at row {i}')
        if not np.isfinite(after['qmed']):
            intentional.append(dict(row=i, old_reasons=legacy, new_reasons=reasons,
                                    old_severity=None if not np.isfinite(before['severity']) else before['severity'],
                                    new_severity=after['severity']))
    return intentional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('# Bounded R6 acceptance outputs.\n!*.csv\n!*.png\n!*.svg\n')
    code = subprocess.check_output(['git', 'show', f'{COMMIT}:quality_check.py'], cwd=ROOT)
    (out/'quality_check.legacy.txt').write_bytes(code)
    legacy = types.ModuleType('legacy_quality_check_r6')
    legacy.__file__ = str(ROOT/'quality_check.py')
    exec(compile(code, legacy.__file__, 'exec'), legacy.__dict__)
    dependencies = {}
    for name in DEPENDENCIES:
        old = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
        digest = hashlib.sha256(old).hexdigest()
        if digest != file_sha256(ROOT/name):
            raise ValueError(f'Numerical dependency changed: {name}')
        dependencies[name] = digest
    fixtures = json.loads((ROOT/'tests/fixtures/quality_stage18_reference.json').read_text())
    cases = []
    for index, item in enumerate(fixtures['sources']):
        source = Path(item['path'])
        if file_sha256(source) != item['sha256']:
            raise ValueError('Real source differs from saved baseline')
        copy_path = out/'real_sources'/f'Jenqwei{index}(R6)'/'merged.csv'
        copy_path.parent.mkdir(parents=True)
        shutil.copyfile(source, copy_path)
        if file_sha256(copy_path) != item['sha256']:
            raise ValueError('Real validation copy differs from saved baseline')
        cases.append(dict(name=f'jenqwei_{index}', mode='anomalies', source=copy_path,
                          original_source=str(source), plot=(index == 1)))
    real_sample = ROOT/'YoGa/James(SN035)/merged.csv'
    real_anomaly = ROOT/'iBrainCenter/Hardy(SN036)/merged.csv'
    cases += [dict(name='real_sample', mode='samples', source=real_sample, plot=True),
              dict(name='real_anomaly', mode='anomalies', source=real_anomaly, plot=True)]
    t = np.arange(30000, dtype=np.int64)*2000 + 1700000000000000
    x = np.random.default_rng(482).normal(0, 6, (30000, 4)).astype(np.float32)
    sample_source = out/'sample_source'/'Synthetic(SN001)'/'merged.csv'
    write_source(sample_source, t, x)
    gap_t = np.arange(7500, dtype=np.int64)*2000 + 1700000000000000
    gap_t[2751:] += 8000000
    gap_x = x[:7500].copy()
    gap_x[:500, 0] = 2050.
    gap_source = out/'gap_source'/'Synthetic(SN002)'/'merged.csv'
    write_source(gap_source, gap_t, gap_x)
    nan_source = out/'nan_source'/'Synthetic(SN003)'/'merged.csv'
    write_source(nan_source, t[:7500], x[:7500])
    nonfinite_x = x[:7500].copy()
    nonfinite_x[100, 0] = np.nan
    nonfinite_source = out/'nonfinite_source'/'Synthetic(SN004)'/'merged.csv'
    write_source(nonfinite_source, t[:7500], nonfinite_x)
    cases += [dict(name='synthetic_sample', mode='samples', source=sample_source),
              dict(name='fallback_sample', mode='samples', source=sample_source, fallback=True),
              dict(name='external_sample', mode='samples', source=sample_source, scorer='external', plot=True),
              dict(name='short_sample', mode='samples', source=nan_source),
              dict(name='gap_anomaly', mode='anomalies', source=gap_source, plot=True),
              dict(name='half_anomaly', mode='anomalies', source=nan_source, win=.5),
              dict(name='fallback_anomaly', mode='anomalies', source=nan_source, fallback=True, plot=True),
              dict(name='external_anomaly', mode='anomalies', source=nan_source, scorer='external'),
              dict(name='nan_anomaly', mode='anomalies', source=nan_source, scorer='nan', plot=True),
              dict(name='all_rejected_anomaly', mode='anomalies', source=nan_source, scorer='zero'),
              dict(name='empty_anomaly', mode='anomalies', source=nan_source, win=20., plot=True),
              dict(name='nonfinite_anomaly', mode='anomalies', source=nonfinite_source),
              dict(name='nonfinite_sample', mode='samples', source=nonfinite_source)]
    expected, actual, results = {}, {}, []
    manifest = dict(schema_version=1, files=[], tables=[], comparisons=[])
    for case in cases:
        name = case['name']
        old_dir, new_dir = out/(name+'_old'), out/(name+'_new')
        old = run(legacy, case, old_dir)
        new = run(current, case, new_dir, plots=case.get('plot', False))
        if old['failure'] != new['failure'] or old['starts'] != new['starts']:
            raise AssertionError(f'{name}: entry failure or random selection changed')
        if len(old['trace']) != len(new['trace']):
            raise AssertionError(f'{name}: number of scorer calls changed')
        for i, (a, b) in enumerate(zip(old['trace'], new['trace'])):
            arr_key = f'{name}__quality_{i}'
            expected[arr_key] = np.asarray(a['overall'])
            actual[arr_key] = np.asarray(b['overall'])
            np.testing.assert_array_equal(expected[arr_key], actual[arr_key])
        intentional = compare_rows(old['rows'], new['rows'], name) if case['mode'] == 'anomalies' else []
        table = new_dir/(f'Quality_{name}_sample_quality.csv' if case['mode'] == 'samples'
                         else f'{name}_R6_quality_anomalies.csv')
        summary = dict(case=name, mode=case['mode'], source=str(case['source']),
                       original_source=case.get('original_source'),
                       source_sha256=file_sha256(case['source']), scorer_calls=len(new['trace']),
                       old_failure=old['failure'], new_failure=new['failure'],
                       selected_starts=new['starts'], intentional_nan_changes=intentional,
                       rows=0, reader_checks=0)
        if table.exists():
            with (new_dir/'reader.log').open('w') as log, contextlib.redirect_stderr(log):
                frame, meta = (load_quality_samples_table(table, case['source'])
                               if case['mode'] == 'samples' else
                               load_quality_anomalies_table(table, case['source']))
            manifest['tables'].append(dict(path=str(table), kind=meta['kind'],
                                           raw_csv=str(case['source'])))
            summary.update(rows=len(frame), reader_checks=1,
                           diagnostic_counts=frame.quality_diagnostic_state.value_counts().to_dict())
            if case['mode'] == 'samples':
                if frame.window_start_idx.iloc[::2].tolist() != new['starts']:
                    raise AssertionError('Sample output differs from random selection')
                for i, trace in enumerate(new['trace']):
                    for ch, value in enumerate(trace['overall']):
                        if not np.isclose(frame[f'quality_ch{ch+1}'].iloc[i], value,
                                          rtol=0., atol=0., equal_nan=True):
                            raise AssertionError('Sample quality differs from scorer trace')
            else:
                for i, row in enumerate(new['rows']):
                    if not np.isclose(frame.qmed.iloc[i], row['qmed'], rtol=0., atol=0., equal_nan=True):
                        raise AssertionError('Anomaly CSV differs from scorer trace')
        elif new['failure'] is None and new['trace']:
            raise AssertionError(f'{name}: scored input did not produce table')
        results.append(summary)
        print(f'PASS {name}: {summary["rows"]} table rows, {len(intentional)} intended NaN changes', flush=True)
    # Direct helper pollution is allowed even though the CLI source loader rejects it.
    polluted = x[:7500].copy()
    polluted[100, 0] = np.nan
    for label, module, arrays in [('old', legacy, expected), ('new', current, actual)]:
        with (out/f'polluted_helper_{label}.log').open('w') as log, contextlib.redirect_stderr(log):
            rows = module.flag_anomalies(module.analyze_quality_windows(t[:7500], polluted))
        for key in ('qmed', 'clip', 'flat', 'gap'):
            arrays['polluted_helper__'+key] = np.asarray([r[key] for r in rows])
        if label == 'new':
            old_rows = legacy.flag_anomalies(legacy.analyze_quality_windows(t[:7500], polluted))
            changes = compare_rows(old_rows, rows, 'polluted_helper')
            (out/'polluted_helper_changes.json').write_text(json.dumps(changes, indent=2)+'\n')
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    nonempty = {key for key, value in expected.items() if value.size}
    np.savez_compressed(out/'expected.npz', **{key: expected[key] for key in nonempty})
    np.savez_compressed(out/'actual.npz', **{key: actual[key] for key in nonempty})
    manifest['comparisons'].append(dict(expected='expected.npz', actual='actual.npz',
                                        rtol=0., atol=0., equal_nan=True))
    paths = {*out.rglob('*'), *(case['source'] for case in cases),
             *(ROOT/name for name in DEPENDENCIES), ROOT/'quality_check.py',
             ROOT/'lilia/quality_check_io.py', ROOT/'lilia/quality_audit.py',
             ROOT/'tools/refactor_check.py', Path(__file__).resolve()}
    def relative(path):
        p = Path(path)
        return str(p.relative_to(out)) if p.is_relative_to(out) else str(p)
    for path in sorted(paths):
        if path.is_file():
            manifest['files'].append(dict(path=relative(path), sha256=file_sha256(path)))
    for item in manifest['tables']:
        item['path'], item['raw_csv'] = relative(item['path']), relative(item['raw_csv'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    analysis = dict(commit=COMMIT, legacy_entry_sha256=hashlib.sha256(code).hexdigest(),
        numerical_dependencies=dependencies, cases=results,
        scorer_arrays=len(actual), nonempty_arrays=len(nonempty), empty_arrays=sorted(set(actual)-nonempty),
        max_abs_error=0., tolerance=0., nan_positions_equal=True,
        reader_checks=sum(row['reader_checks'] for row in results),
        intended_nan_changes=sum(len(row['intentional_nan_changes']) for row in results),
        plots_reviewed=False)
    (out/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    print(f'PASS {len(cases)} cases + polluted helper; {len(actual)} exact arrays; '
          f'{analysis["reader_checks"]} source readers')


if __name__ == '__main__':
    main()
