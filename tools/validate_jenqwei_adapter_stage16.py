"""Verify the stage-16 processing increment against frozen continuous/segment baselines.

This validates processing only; full CLI/plot acceptance uses validate_jenqwei_stage16.py.
"""
import argparse
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import analyze_jenqwei_pipeline as jenqwei
from lilia.provenance import file_sha256
from tools.refactor_check import compare_arrays

KEYS = ('time_us_500', 'data_raw', 'data_filt_500', 'time_us_200', 'pre_data_200', 'tfl_data_200')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'files': [], 'comparisons': []}
    analysis = {'scope': 'stage16_processing_increment', 'cli_guard': 'legacy_run_pipeline_only',
                'plots': 'not_validated_by_this_script', 'cases': {}}

    def remember(path, expected=None):
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Frozen fingerprint mismatch: {path}')
        manifest['files'].append({'path': str(path), 'sha256': expected or actual})

    def compare(name, actual, expected):
        actual_path, expected_path = out / f'{name}_actual.npz', out / f'{name}_expected.npz'
        np.savez_compressed(actual_path, **actual)
        np.savez_compressed(expected_path, **expected)
        remember(actual_path)
        remember(expected_path)
        manifest['comparisons'].append({'actual': str(actual_path), 'expected': str(expected_path),
                                        'rtol': 1e-6, 'atol': 1e-6, 'equal_nan': False})
        return compare_arrays(actual_path, expected_path, rtol=1e-6, atol=1e-6)

    paths = sorted((ROOT / 'tests/fixtures').glob('jenqwei_*_reference.npz'))
    if len(paths) != 7:
        raise ValueError('Expected five real and two synthetic frozen cases')
    for path in paths:
        metadata = json.loads(path.with_suffix('.json').read_text())
        remember(path, metadata['npz_sha256'])
        remember(path.with_suffix('.json'))
        for name, expected in metadata['dependencies'].items():
            if file_sha256(ROOT / name) != expected:
                raise ValueError(f'Frozen dependency changed: {name}')
        if metadata['source_path']:
            remember(ROOT / metadata['source_path'], metadata['source_sha256'])
        remember(ROOT / 'tiny_v4_optimized.tflite', metadata['model_sha256'])
        with np.load(path) as frozen:
            result = jenqwei.process_segments(frozen['source_time_us'], frozen['source_raw'], metadata['max_samples'])
            errors = compare(path.stem, {k: result[k] for k in KEYS}, {k: frozen[k] for k in KEYS})
        analysis['cases'][path.stem] = {'before_rows': len(result['pre_data_200']),
                                       'after_rows': len(result['tfl_data_200']),
                                       'max_absolute_errors': errors, 'segments': result['segments']}
        print(f'{path.stem}: max_error={max(errors.values())}', flush=True)

    snapshot = ROOT / 'docs/refactor/validation/stage16/baseline_legacy_entry.py.txt'
    remember(snapshot, metadata['legacy_entry_sha256'])
    legacy = types.ModuleType('legacy_jenqwei')
    legacy.__file__ = str(ROOT / 'analyze_jenqwei_pipeline.py')
    exec(compile(snapshot.read_bytes(), str(snapshot), 'exec'), legacy.__dict__)
    with np.load(ROOT / 'tests/fixtures/jenqwei_move_head_reference.npz') as frozen:
        raw = frozen['source_raw'][:4511].copy()
    lengths = (10, 1503, 2001, 997)
    t = 9000000000000001 + np.concatenate([i * 10000000 + np.arange(n) * 2000 for i, n in enumerate(lengths)])
    source_copy = out / 'segmented_source.npz'
    np.savez_compressed(source_copy, time_us=t, raw=raw)
    remember(source_copy)
    result = jenqwei.process_segments(t, raw)
    before_offset = offset = 0
    expected = {key: [] for key in (*KEYS, 'raw_sample_idx', 'model_to_before_idx',
                                    'before_raw_fractional_idx', 'model_raw_fractional_idx',
                                    'tfl_time_us_200', 'before_segment_ids', 'model_segment_ids')}
    for sid, length in enumerate(lengths):
        a, b = offset, offset + length
        offset = b
        if (length * 2 + 4) // 5 < 400:
            continue
        old = legacy.run_pipeline(t[a:b], raw[a:b])
        for key in KEYS:
            expected[key].append(old[key])
        before_n, after_n = len(old['pre_data_200']), len(old['tfl_data_200'])
        positions = a + np.arange(before_n) * 2.5
        expected['raw_sample_idx'].append(np.arange(a, b, dtype=np.int64))
        expected['model_to_before_idx'].append(before_offset + np.arange(after_n, dtype=np.int64))
        expected['before_raw_fractional_idx'].append(positions)
        expected['model_raw_fractional_idx'].append(positions[:after_n])
        expected['tfl_time_us_200'].append(old['time_us_200'][:after_n])
        expected['before_segment_ids'].append(np.full(before_n, sid, dtype=np.int64))
        expected['model_segment_ids'].append(np.full(after_n, sid, dtype=np.int64))
        before_offset += before_n
    expected = {key: np.concatenate(value) for key, value in expected.items()}
    actual = {key: result[key] for key in expected if key != 'model_segment_ids'}
    actual['model_segment_ids'] = result['timeline'].segment_ids
    errors = compare('real_segmented', actual, expected)
    analysis['cases']['real_segmented'] = {'before_rows': len(result['pre_data_200']),
                                          'after_rows': len(result['tfl_data_200']),
                                          'max_absolute_errors': errors, 'segments': result['segments'],
                                          'model_windows': result['timeline'].model_windows}
    for path in (Path(__file__), ROOT / 'analyze_jenqwei_pipeline.py'):
        remember(path)
    summary = out / 'analysis.json'
    summary.write_text(json.dumps(analysis, indent=2) + '\n')
    remember(summary)
    (out / 'evidence.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'PASS processing increment: {summary}', flush=True)


if __name__ == '__main__':
    main()
