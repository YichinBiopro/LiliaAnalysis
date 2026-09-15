"""Freeze the pre-migration helper and compare real/custom-marker quality paths."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lilia.io import load_merged_csv
from lilia.provenance import file_sha256
from lilia.custom_marker_quality_io import load_custom_marker_quality_table
import plot_event_markers as current_event
import plot_index_vs_raw as current_plot


COMMIT = '996fb110e7ea70a37ddedf4283a17571cbdef53f'
DEPENDENCIES = ('lilia/io.py', 'lilia/quality.py', 'lilia/qeeg.py',
                'lilia/constants.py', 'lilia/windowing.py', 'lilia/time_utils.py')


def frozen(name, module_name, out):
    code = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
    path = out / (Path(name).stem + '.legacy.txt')
    path.write_bytes(code)
    module = types.ModuleType(module_name)
    module.__file__ = str(ROOT / name)
    exec(compile(code, str(ROOT / name), 'exec'), module.__dict__)
    return module, hashlib.sha256(code).hexdigest()


def scorer_for(kind):
    if kind == 'external':
        return lambda data, fs, params: {'overall': np.full(data.shape[0], .7)}
    if kind == 'nan':
        return lambda data, fs, params: {'overall': np.full(data.shape[0], np.nan)}
    if kind == 'error':
        return lambda data, fs, params: (_ for _ in ()).throw(RuntimeError('score broke'))
    if kind == 'fallback':
        import lilia.quality as quality
        def forced_fallback(data, fs, params):
            with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced spectrum error')):
                return quality.get_eeg_quality_index_v2_parametric(data, fs=fs, params=params)
        return forced_fallback
    return None


def score_case(name, t, x, fs, win_sec, kind, legacy, current, expected, actual):
    outputs = []
    for module, arrays in ((legacy, expected), (current, actual)):
        scorer = scorer_for(kind)
        with (patch.object(module, 'get_eeg_quality_index_v2_parametric', side_effect=scorer)
              if scorer is not None else contextlib.nullcontext()):
            if kind == 'error':
                try:
                    module.compute_quality_windowed(t, x, win_sec=win_sec, fs=fs,
                        **({'return_audit': True} if module is current else {}))
                except RuntimeError as exc:
                    if str(exc) != 'score broke':
                        raise
                    values = ([], np.empty((0,)), []) if module is current else ([], np.empty((0,)))
                else:
                    raise AssertionError('Injected scorer error did not propagate')
            else:
                values = module.compute_quality_windowed(t, x, win_sec=win_sec, fs=fs,
                    **({'return_audit': True} if module is current else {}))
        dt, score = values[:2]
        arrays[name + '__scores'] = np.asarray(score)
        arrays[name + '__mid_us'] = np.asarray([int(value.timestamp() * 1e6) for value in dt], dtype=np.int64)
        outputs.append(values)
    if outputs[0][0] != outputs[1][0]:
        raise AssertionError(f'{name} local quality window timestamps changed')
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    return len(outputs[1][0]), (outputs[1][2] if len(outputs[1]) == 3 else [])


def write_synthetic(path, *, gap=False):
    n = 7500
    t = np.arange(n, dtype=np.int64) * 2000
    if gap:
        t[3000:] += 10_000_000
    amp = np.sin(np.arange(n) / 17.) * 22.
    head = ('File Name,synthetic\nAmp Gain,500,Abs Time Offset[us],0,Recording Start time[us],0\n'
            'Channels,1\nSample Rate (per channel),500\nTime[us],value\n')
    path.write_text(head + ''.join(f'{int(us)},{value:.8f}\n' for us, value in zip(t, amp)))
    return t, amp.astype(np.float32)[:, None]


def plot_case(name, source, marker, baseline_sec, kind, legacy_event, legacy_plot, out,
              expected, actual, manifest):
    frames = []
    for event, plot, label, arrays in ((legacy_event, legacy_plot, 'old', expected),
                                       (current_event, current_plot, 'new', actual)):
        dest = out / f'{name}_{label}'
        dest.mkdir()
        scorer = scorer_for(kind)
        with (patch.object(event, 'get_eeg_quality_index_v2_parametric', side_effect=scorer)
              if scorer is not None else contextlib.nullcontext()), \
             (dest / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            plot.plot_custom_markers(str(source), name, 1, [marker], '1970-01-01',
                baseline_sec, baseline_sec, str(dest / 'plot.png'), str(dest / 'summary.csv'))
        frame = pd.read_csv(dest / 'summary.csv', float_precision='round_trip')
        frames.append(frame)
        for key in frame:
            if (key.endswith(('_baseline', '_post_mean', '_delta')) or
                    key in ('baseline_windows', 'post_windows')):
                arrays[f'{name}__summary__{key}'] = frame[key].to_numpy()
        if label == 'new':
            table = dest / 'summary_quality_windows.csv'
            verified, _ = load_custom_marker_quality_table(table, source)
            manifest['tables'].append(dict(path=str(table), kind='custom_marker_quality',
                                           raw_csv=str(source)))
            if len(verified) == 0:
                raise AssertionError('Custom marker source reader had no windows')
    pd.testing.assert_frame_equal(frames[0], frames[1])
    return len(pd.read_csv(out / f'{name}_new' / 'summary_quality_windows.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('# Bounded final direct-helper acceptance outputs.\n!*.csv\n!*.png\n!*.svg\n')
    legacy_event, event_hash = frozen('plot_event_markers.py', 'stage18_legacy_event', out)
    legacy_plot, plot_hash = frozen('plot_index_vs_raw.py', 'stage18_legacy_index', out)
    legacy_plot.pem = legacy_event
    dependency_hashes = {}
    for name in DEPENDENCIES:
        old = hashlib.sha256(subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)).hexdigest()
        now = file_sha256(ROOT / name)
        if old != now:
            raise AssertionError(f'Numerical dependency changed: {name}')
        dependency_hashes[name] = now
    sources = json.loads((ROOT / 'tests/fixtures/quality_stage18_reference.json').read_text())['sources']
    expected, actual, cases = {}, {}, []
    source_paths = []
    for i, item in enumerate(sources):
        source = Path(item['path'])
        if file_sha256(source) != item['sha256']:
            raise AssertionError(f'Real source changed: {source}')
        source_paths.append(source)
        t, raw = load_merged_csv(source)
        for ch in (0, 1):
            name = f'real_{i}_ch{ch+1}'
            count, audit = score_case(name, t, raw[:, ch:ch+1], 500., 5., 'normal',
                legacy_event, current_event, expected, actual)
            cases.append(dict(name=name, windows=count, source=str(source),
                              source_sha256=item['sha256'], diagnostic_states=sorted({r['quality_diagnostics']['state'] for r in audit})))
            print(f'PASS {name}: {count} quality windows', flush=True)
    gap_source = out / 'synthetic_gap.csv'
    gap_t, gap_x = write_synthetic(gap_source, gap=True)
    continuous_source = out / 'synthetic_continuous.csv'
    cont_t, cont_x = write_synthetic(continuous_source)
    for name, t, x, fs, win_sec, kind in (
        ('gap', gap_t, gap_x, 500., 5., 'normal'),
        ('external', cont_t, cont_x, 500., 5., 'external'),
        ('nan', cont_t, cont_x, 500., 5., 'nan'),
        ('fallback', cont_t, cont_x, 500., 5., 'fallback'),
        ('short', cont_t[:250], cont_x[:250], 500., 5., 'normal'),
        ('error', cont_t, cont_x, 500., 5., 'error')):
        count, audit = score_case(name, t, x, fs, win_sec, kind,
            legacy_event, current_event, expected, actual)
        cases.append(dict(name=name, windows=count, scorer=kind,
                          diagnostic_states=sorted({r['quality_diagnostics']['state'] for r in audit})))
        print(f'PASS {name}: {count} quality windows', flush=True)
    manifest = dict(schema_version=1, files=[], tables=[], comparisons=[])
    plot_rows = {}
    for name, source, marker, baseline, kind in (
        ('real_talk', source_paths[1], '08:00:30', 10, 'normal'),
        ('gap_plot', gap_source, '08:00:05', 5, 'normal'),
        ('fallback_plot', continuous_source, '08:00:05', 5, 'fallback'),
        ('external_plot', continuous_source, '08:00:05', 5, 'external'),
        ('nan_plot', continuous_source, '08:00:05', 5, 'nan')):
        plot_rows[name] = plot_case(name, source, marker, baseline, kind,
            legacy_event, legacy_plot, out, expected, actual, manifest)
        print(f'PASS {name}: {plot_rows[name]} table rows', flush=True)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    nonempty = {key for key, value in expected.items() if np.asarray(value).size}
    np.savez_compressed(out / 'expected.npz', **{key: expected[key] for key in nonempty})
    np.savez_compressed(out / 'actual.npz', **{key: actual[key] for key in nonempty})
    manifest['comparisons'].append(dict(expected='expected.npz', actual='actual.npz',
                                        rtol=0., atol=0., equal_nan=True))
    paths = {*out.rglob('*'), *source_paths,
             *(ROOT / name for name in DEPENDENCIES),
             ROOT / 'plot_event_markers.py', ROOT / 'plot_index_vs_raw.py',
             ROOT / 'lilia/custom_marker_quality_io.py', Path(__file__).resolve()}
    def relative(path):
        path = Path(path)
        return str(path.relative_to(out)) if path.is_relative_to(out) else str(path)
    for path in sorted(paths):
        if path.is_file():
            manifest['files'].append(dict(path=relative(path), sha256=file_sha256(path)))
    for item in manifest['tables']:
        item['path'], item['raw_csv'] = relative(item['path']), relative(item['raw_csv'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    analysis = dict(commit=COMMIT, legacy_event_sha256=event_hash,
        legacy_index_sha256=plot_hash, numerical_dependencies=dependency_hashes,
        cases=cases, plot_rows=plot_rows, scorer_arrays=len(actual),
        nonempty_arrays=len(nonempty), max_abs_error=0., tolerance=0.,
        nan_positions_equal=True, source_reader_checks=len(manifest['tables']),
        plots_reviewed=False)
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(f'PASS {len(cases)} helper cases + {len(plot_rows)} plot cases; '
          f'{len(actual)} exact arrays; {len(manifest["tables"])} source readers')


if __name__ == '__main__':
    main()
