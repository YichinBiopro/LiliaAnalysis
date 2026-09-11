"""Compare TFLite quality, baseline choices and plots with the committed caller."""
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

import plot_tflite_summary as current
from lilia import quality
from lilia.io import load_merged_csv
from lilia.provenance import file_sha256
from lilia.quality_audit import json_value
from lilia.tflite_io import load_tflite_table

COMMIT = 'a0a7ba6'
NUMERICAL_DEPS = ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py',
                  'lilia/qeeg.py', 'lilia/tflite.py', 'lilia/quality.py')


def frozen(path, out):
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)
    (out / (Path(path).name + '.legacy.txt')).write_bytes(source)
    module = types.ModuleType('legacy_' + Path(path).stem)
    module.__file__ = str(ROOT / path)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def legacy_projection(actual, expected):
    """Assert all preexisting audit fields match, allowing additive diagnostics."""
    if isinstance(expected, dict):
        return {key: legacy_projection(actual[key], value) for key, value in expected.items()}
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Legacy audit row count differs')
        return [legacy_projection(a, b) for a, b in zip(actual, expected)]
    return actual


def fallback(data, fs, params):
    with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('stage18 forced spectrum failure')):
        return quality.get_eeg_quality_index_v2_parametric(data, fs=fs, params=params)


def run(module, source, out, *, params=None, scorer=None, threshold=.5, epoch_sec=1.):
    arrays = {}
    original_infer = module.run_tflite_recording
    original_save = Figure.savefig

    def infer(*args, **kwargs):
        result = original_infer(*args, **kwargs)
        arrays.update(model_time_us=result[0].time_us, model_before=result[1], model_after=result[2])
        return result

    def save(fig, path, **kwargs):
        if str(path).endswith('.png'):
            for i, ax in enumerate(fig.axes):
                for j, line in enumerate(ax.lines):
                    arrays[f'plot_{i}_line_{j}'] = np.asarray(line.get_ydata(), dtype=float)
                if i:
                    for j, collection in enumerate(ax.collections):
                        values = collection.get_array()
                        if values is not None:
                            arrays[f'plot_{i}_collection_{j}'] = np.ma.asarray(values).filled(np.nan)
            original_save(fig, path, **kwargs)
        # One reviewable PNG per run suffices; the product still exports SVG.

    status = 'passed'
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, 'EVENTS', []))
        stack.enter_context(patch.object(module, 'CONE_STAGES', []))
        stack.enter_context(patch.object(module, 'run_tflite_recording', side_effect=infer))
        stack.enter_context(patch.object(Figure, 'savefig', save))
        if params is not None:
            stack.enter_context(patch.object(module, 'QUALITY_PARAMS', params))
        if scorer is not None:
            stack.enter_context(patch.object(module, 'get_eeg_quality_index_v2_parametric', side_effect=scorer))
        try:
            module.plot_subject_tflite_summary('Stage18', {'dir': source.parent.name, 'sn': 'Q'},
                str(out), base_dir=str(source.parent.parent), on_insufficient='skip',
                quality_ratio=threshold, epoch_sec=epoch_sec)
        except ValueError as exc:
            if str(exc) != 'No usable TFLite baseline reference; see analysis audit':
                raise
            status = str(exc)
    table = out / 'Stage18_Q_tflite_metrics.csv'
    frame, _ = load_tflite_table(table, source, current.TFLITE_MODEL_PATH)
    for name in frame:
        if not name.startswith(('before_quality_', 'after_quality_')) and frame[name].dtype.kind in 'biuf':
            arrays['table_' + name] = frame[name].to_numpy()
    audit = json.loads((out / 'Stage18_Q_tflite_analysis.json').read_text())
    catalog = audit['baseline_catalog']
    arrays['baseline_eligible'] = np.array([r['eligible'] for r in catalog])
    arrays['baseline_quality_min'] = np.array([r['quality_min'] for r in catalog], dtype=float)
    for i, baseline in enumerate(audit['baselines']):
        if baseline.get('status') != 'excluded':
            arrays[f'baseline_{i}_selected'] = np.array([r['model_window_id'] for r in baseline['selected']])
            for key, value in baseline['reference'].items():
                arrays[f'baseline_{i}_{key}'] = np.asarray(value)
    return status, arrays, audit, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = dict(schema_version=1, files=[], tables=[], comparisons=[])

    def remember(path, sha=None):
        evidence['files'].append(dict(path=str(path), sha256=sha or file_sha256(path)))

    for path in NUMERICAL_DEPS:
        old = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)
        sha = hashlib.sha256(old).hexdigest()
        if file_sha256(ROOT / path) != sha:
            raise ValueError('Numerical dependency changed: ' + path)
        remember(ROOT / path, sha)
    legacy = frozen('plot_tflite_summary.py', out)
    baseline = frozen('lilia/tflite_baseline.py', out)
    legacy.score_baseline_windows = baseline.score_baseline_windows
    legacy.select_tflite_baseline = baseline.select_tflite_baseline
    cases, inventory = [], []

    def write_source(name, t, raw, original, recipe):
        folder = out / 'inputs' / name
        folder.mkdir(parents=True)
        path = folder / 'merged.csv'
        with path.open('x') as handle:
            handle.write('Device,Lilia,stage18 verification copy\nAmp Gain,500\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n')
            pd.DataFrame({'Time[us]': t, **{f'ch{i+1}': raw[:, i] for i in range(4)}}).to_csv(handle, index=False)
        inventory.append(dict(source=str(original), source_sha256=file_sha256(original),
                              derived=str(path), derived_sha256=file_sha256(path), recipe=recipe))
        remember(original)
        remember(path)
        return path

    paths = [p for p in sorted((ROOT / 'jenqwei').glob('*/*.csv')) if p.name.lower() != 'time_marker.csv']
    if len(paths) != 5:
        raise ValueError('Expected five real Jenqwei recordings')
    for original in paths:
        t, raw = load_merged_csv(original)
        name = original.stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_')
        source = write_source(name, t, raw[:, :4], original, 'first four declared channels; unchanged int64 time and float32 signal')
        cases.append((name, source, {}))
        if name == 'talk':
            talk = source
            gap_t = t.copy()
            gap_t[len(t)//2:] += 100000000
            gap = write_source('talk_gap', gap_t, raw[:, :4], original,
                               'first four channels; add 100000000 us from midpoint; signal unchanged')
    cases.extend([('talk_gap', gap, {}),
                  ('talk_threshold_zero', talk, {'threshold': 0.}),
                  ('talk_half_second', talk, {'epoch_sec': .5}),
                  ('talk_fallback', talk, {'params': dict(flat_weight=0., spectrum_weight=1., kurtosis_weight=0., corr_weight=0.),
                                           'scorer': fallback})])
    reports = []
    for name, source, options in cases:
        old_status, expected, old_audit, old_table = run(legacy, source, out / (name + '_old'), **options)
        status, actual, audit, table = run(current, source, out / (name + '_new'), **options)
        if status != old_status or expected.keys() != actual.keys():
            raise ValueError('Legacy status or array coverage differs: ' + name)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=name + ': ' + key)
        for key in ('baseline_catalog', 'baselines', 'heatmap_bins'):
            if legacy_projection(audit[key], old_audit[key]) != old_audit[key]:
                raise ValueError('Legacy audit changed: ' + name + '/' + key)
        expected_path, actual_path = out / (name + '_expected.npz'), out / (name + '_actual.npz')
        np.savez_compressed(expected_path, **expected)
        np.savez_compressed(actual_path, **actual)
        evidence['comparisons'].append(dict(actual=str(actual_path), expected=str(expected_path), rtol=0, atol=0, equal_nan=True))
        for path in (old_table, table):
            evidence['tables'].append(dict(path=str(path), kind='tflite_qeeg', raw_csv=str(source),
                                           model_path=str(Path(current.TFLITE_MODEL_PATH).resolve())))
        reports.append(dict(case=name, status=status, quality_threshold=options.get('threshold', .5),
                            baseline_epoch_sec=options.get('epoch_sec', 1.), model_windows=len(audit['baseline_catalog']),
                            metric_windows=len(audit['quality_analysis']['after']),
                            quality_summary=audit['quality_summary'], max_abs_error=0,
                            changed_selected_windows=0, arrays=len(actual)))
        print(name + ': exact arrays, baseline selection and two source readers passed', flush=True)
    (out / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (out / 'analysis.json').write_text(json.dumps(json_value(dict(status='passed', cases=reports,
        policy='legacy_overall; no threshold or baseline selection changes', reader_verification=current.VERIFICATION)), indent=2) + '\n')
    remember(Path(current.TFLITE_MODEL_PATH).resolve())
    for path in ('plot_tflite_summary.py', 'plot_event_markers.py', 'lilia/tflite_baseline.py',
                 'lilia/tflite_io.py', 'lilia/tflite_quality.py', 'lilia/quality_audit.py',
                 'tools/validate_tflite_quality_stage18.py'):
        remember(ROOT / path)
    for path in sorted(out.rglob('*')):
        if path.is_file() and 'inputs' not in path.parts:
            remember(path)
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('PASS: ' + str(out / 'analysis.json'), flush=True)


if __name__ == '__main__':
    main()
