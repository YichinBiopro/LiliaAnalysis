"""Freeze quality scores from the committed scorer before adding diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from lilia.io import load_merged_csv, bandpass_filter
from lilia.provenance import file_sha256

COMMIT = '23aa2df'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:lilia/quality.py'], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != file_sha256(ROOT / 'lilia/quality.py'):
        raise ValueError('Scorer no longer matches baseline commit')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'legacy_quality.py.txt').write_bytes(source)
    legacy = types.ModuleType('legacy_quality_stage18')
    exec(compile(source, 'legacy_quality_stage18', 'exec'), legacy.__dict__)
    presets = {'default': legacy.get_default_eeg_quality_v2_params(),
               'mean_abs_corr': legacy.BEST_EEG_QUALITY_V2_MEAN_ABS_CORR_PARAMS,
               'flat_spectrum_only': legacy.get_best_eeg_quality_v2_flat_spectrum_only_params(),
               'ibrain_device': legacy.get_ibrain_device_eeg_quality_v2_params()}
    inputs, values, cases, sources = {}, {}, [], []
    for path in sorted((ROOT / 'jenqwei').glob('*/*.csv')):
        if path.name.lower() == 'time_marker.csv':
            continue
        _, data = load_merged_csv(path)
        sources.append({'path': str(path), 'sha256': file_sha256(path)})
        label = path.stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_')
        for stage, signal in [('raw', data[:, :4]), ('bandpass', bandpass_filter(data[:, :4]))]:
            # Whole recording, five-second non-overlapping windows, no tail padding.
            for start in range(0, len(signal)-2500+1, 2500):
                key = f'{label}_{stage}_{start}'
                inputs[key] = signal[start:start+2500].T
                cases.append({'key': key, 'fs': 500, 'stage': stage})
    rng = np.random.default_rng(18018)
    for n in (0, 1, 8, 99, 100, 101, 399, 400, 1000):
        key = f'synthetic_{n}'
        inputs[key] = rng.normal(size=(4, n))
        cases.append({'key': key, 'fs': 200, 'stage': 'synthetic'})
    for name, data in [('constant', np.zeros((4, 1000))),
                       ('nonfinite', rng.normal(size=(4, 1000))),
                       ('single_channel', rng.normal(size=(1, 1000)))]:
        if name == 'nonfinite':
            data[0, 100] = np.nan
        inputs[name] = data
        cases.append({'key': name, 'fs': 200, 'stage': 'synthetic'})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for case in cases:
            for name, params in presets.items():
                result = legacy.get_eeg_quality_index_v2_parametric(inputs[case['key']], fs=case['fs'], params=params)
                prefix = case['key'] + '__' + name
                values[prefix + '__overall'] = result['overall']
                for component, score in result['detail'].items():
                    values[prefix + '__' + component] = score
        (out / 'legacy_warnings.log').write_text('\n'.join(str(w.message) for w in caught)+'\n')
    np.savez_compressed(out / 'quality_stage18_inputs.npz', **inputs)
    np.savez_compressed(out / 'quality_stage18_reference.npz', **values)
    meta = {'commit': COMMIT, 'scorer_sha256': hashlib.sha256(source).hexdigest(),
            'sources': sources, 'presets': presets, 'cases': cases,
            'dependencies': {p: file_sha256(ROOT / p) for p in ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py')},
            'input_sha256': file_sha256(out / 'quality_stage18_inputs.npz'),
            'reference_sha256': file_sha256(out / 'quality_stage18_reference.npz')}
    (out / 'quality_stage18_reference.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'{len(cases)} inputs x {len(presets)} presets; {len(values)} frozen arrays')


if __name__ == '__main__':
    main()
