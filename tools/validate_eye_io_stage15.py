"""Validate the stage-15 CLI/IO increment in a new, isolated output directory.

Continuous CLI output is compared with frozen old real-model values. Segmented
IO is compared with independent per-segment runs of the frozen old entry; the
CLI still rejects gaps until plotting migration has been validated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import process_lilia_eye_open_close as eye
from lilia.eye_io import load_signal_table, signal_parameters, write_signal_table
from lilia.provenance import file_sha256
from tools.refactor_check import compare_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fixture = ROOT / 'tests/fixtures/eye_real_continuous_reference.npz'
    reference = json.loads(fixture.with_suffix('.json').read_text())
    for name, expected in reference['dependencies'].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen old processing dependency changed: {name}')
    source = ROOT / eye.DEFAULT_CSV
    if file_sha256(source) != reference['source_sha256'] or file_sha256(fixture) != reference['npz_sha256']:
        raise ValueError('Frozen input/fixture fingerprint mismatch')
    parameters = signal_parameters()
    for key in ('checkpoint_sha256', 'architecture_sha256'):
        if parameters['model'][key] != reference['model'][key]:
            raise ValueError(f'Frozen model fingerprint mismatch: {key}')
    manifest = {'schema_version': 1, 'files': [], 'tables': [], 'comparisons': []}
    summary = {'scope': 'stage15_cli_io_increment', 'cases': {},
               'segmented_cli_guard': 'retained', 'plot_migration': 'pending'}

    def remember(path, expected=None):
        manifest['files'].append({'path': str(path), 'sha256': expected or file_sha256(path)})

    def table(path, raw):
        frame, meta = load_signal_table(path, raw, model_path=eye.MODEL_PATH)
        manifest['tables'].append({'path': str(path), 'kind': 'eye_model_signal',
                                   'raw_csv': str(raw), 'model_path': eye.MODEL_PATH})
        return frame, meta

    def compare(name, actual, expected):
        actual_path, expected_path = out / f'{name}_actual.npz', out / f'{name}_expected.npz'
        np.savez_compressed(actual_path, **actual)
        np.savez_compressed(expected_path, **expected)
        remember(actual_path)
        remember(expected_path)
        manifest['comparisons'].append({'actual': str(actual_path), 'expected': str(expected_path),
                                        'rtol': 1e-6, 'atol': 1e-6, 'equal_nan': False})
        return compare_arrays(actual_path, expected_path, rtol=1e-6, atol=1e-6)

    def cli(name, raw, expected_exit):
        directory = out / name
        command = [sys.executable, str(ROOT / 'process_lilia_eye_open_close.py'),
                   '--csv', str(raw), '--outdir', str(directory)]
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'MPLCONFIGDIR': str(out / 'mpl'), 'MPLBACKEND': 'Agg'}
        log = out / f'{name}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != expected_exit:
            raise ValueError(f'{name}: unexpected exit {result.returncode}; see {log}')
        audit_path = directory / f'{raw.stem}_analysis_audit.json'
        audit = json.loads(audit_path.read_text())
        if audit['status'] != ('success' if expected_exit == 0 else 'failed'):
            raise ValueError(f'{name}: inconsistent audit status')
        remember(log)
        remember(audit_path)
        for artifact in audit['artifacts']:
            remember(artifact['path'], artifact['sha256'])
        summary['cases'][name] = {'command': command, 'exit_code': result.returncode,
                                  'status': audit['status'], 'stage': audit['stage'],
                                  'error': audit.get('error')}
        return directory, audit

    remember(source, reference['source_sha256'])
    remember(fixture, reference['npz_sha256'])
    remember(fixture.with_suffix('.json'))
    remember(Path(eye.MODEL_PATH), reference['model']['checkpoint_sha256'])
    for path in (Path(__file__), ROOT / 'process_lilia_eye_open_close.py', ROOT / 'lilia/eye_io.py'):
        remember(path)
    directory, audit = cli('real', source, 0)
    frame, _ = table(directory / f'{source.stem}_tinyv4_output.csv', source)
    with np.load(fixture) as frozen:
        errors = compare('real', {'time_us': frame['Time[us]'].to_numpy(), 'processed': frame.iloc[:, 1:].to_numpy()},
                         {'time_us': frozen['time_us'], 'processed': frozen['processed']})
    summary['cases']['real'].update(rows=len(frame), max_absolute_errors=errors,
                                    last_time_us=int(frame.iloc[-1, 0]), table_verified=audit['table_verified'])

    # Frozen old entry, not process_segments, supplies the segmented expectation.
    snapshot = ROOT / 'docs/refactor/validation/stage15/baseline_legacy_entry.py.txt'
    if file_sha256(snapshot) != reference['legacy_entry_sha256']:
        raise ValueError('Legacy entry fingerprint mismatch')
    remember(snapshot, reference['legacy_entry_sha256'])
    legacy = types.ModuleType('legacy_eye')
    exec(compile(snapshot.read_bytes(), str(snapshot), 'exec'), legacy.__dict__)
    model = legacy.load_model()
    with np.load(fixture) as frozen:
        raw = frozen['raw'][:4511].copy()
    lengths = [10, 1503, 2001, 997]
    t = 1234567 + np.concatenate([i * 10000000 + np.arange(n) * 2000 for i, n in enumerate(lengths)])
    raw_path = out / 'segmented_source.csv'

    def recording(path, timestamps, values):
        with path.open('x') as handle:
            handle.write('File Name,validation\nAmp Gain,500\nChannels,1,2,3,4,5,6,7,8\n'
                         'Sample Rate (per channel),500,500,500,500,500,500,500,500\n'
                         'Time[us],ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n')
            for timestamp, row in zip(timestamps, values):
                handle.write(','.join([str(int(timestamp)), *map(str, row)]) + '\n')
        remember(path)

    recording(raw_path, t, raw)
    # Reload CSV to compare the same float32 input actually consumed by readers.
    _, raw = eye.load_lilia_csv(raw_path)
    expected = {'time_us': [], 'before': [], 'processed': []}
    offset = 0
    for length in lengths:
        sl = slice(offset, offset + length)
        offset += length
        if (length * 2 + 4) // 5 < 400:
            continue
        filtered = legacy.bandpass_filter(raw[sl], fs=500, lo=.5, hi=45.)
        time_s, before = legacy.downsample_data(t[sl].astype(float) / 1e6, filtered)
        processed = np.concatenate([legacy.run_model(model, before[:, :4]),
                                    legacy.run_model(model, before[:, 4:8])], axis=1)
        expected['time_us'].append(np.round(time_s * 1e6).astype(np.int64))
        expected['before'].append(before)
        expected['processed'].append(processed)
    expected = {key: np.concatenate(value) for key, value in expected.items()}
    timeline, before, processed = eye.process_segments(t, raw, model=model)
    signal_path = out / 'segmented_signal.csv'
    paths = write_signal_table(signal_path, raw_path, timeline, processed, parameters,
                               {'entry': file_sha256(ROOT / 'process_lilia_eye_open_close.py')})
    for path in paths:
        remember(path)
    frame, meta = table(signal_path, raw_path)
    errors = compare('segmented', {'time_us': frame['Time[us]'].to_numpy(), 'before': before,
                                  'processed': frame.iloc[:, 1:].to_numpy()}, expected)
    summary['cases']['segmented_io'] = {'rows': len(frame), 'max_absolute_errors': errors,
                                        'segments': meta['inference']['segments'],
                                        'last_time_us': int(frame.iloc[-1, 0])}
    cli('gap_rejected', raw_path, 1)
    short = out / 'short_source.csv'
    recording(short, np.arange(997) * 2000, raw[:997])
    cli('short_rejected', short, 1)
    polluted = out / 'nonfinite_source.csv'
    values = raw[:1503].copy()
    values[0, 7] = np.nan
    recording(polluted, np.arange(1503) * 2000, values)
    cli('nonfinite_rejected', polluted, 1)
    cli('missing_rejected', out / 'absent.csv', 1)
    analysis_path = out / 'analysis.json'
    analysis_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    remember(analysis_path)
    manifest_path = out / 'evidence.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'PASS CLI/IO numerical validation: {analysis_path}', flush=True)
    print(f'Evidence manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
