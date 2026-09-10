"""Validate additive quality diagnostics against frozen legacy scores."""
import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lilia.io import load_merged_csv, bandpass_filter
from lilia.quality import get_eeg_quality_index_v2_parametric
from lilia.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / 'tests/fixtures'
    metadata = json.loads((fixture / 'quality_stage18_reference.json').read_text())
    proof = {'schema_version': 1, 'files': [], 'comparisons': []}
    def remember(path, expected=None):
        proof['files'].append({'path': str(path), 'sha256': expected or file_sha256(path)})
    real_inputs = {}
    for source in metadata['sources']:
        path = Path(source['path'])
        if file_sha256(path) != source['sha256']:
            raise ValueError('Baseline source changed')
        remember(path, source['sha256'])
        _, data = load_merged_csv(path)
        name = path.stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_')
        for stage, signal in [('raw', data[:, :4]), ('bandpass', bandpass_filter(data[:, :4]))]:
            for start in range(0, len(signal)-2500+1, 2500):
                real_inputs[f'{name}_{stage}_{start}'] = signal[start:start+2500].T
    for path, expected in metadata['dependencies'].items():
        remember(ROOT / path, expected)
    remember(ROOT / 'docs/refactor/validation/stage18/legacy_quality.py.txt', metadata['scorer_sha256'])
    for name, key in [('quality_stage18_inputs.npz', 'input_sha256'),
                      ('quality_stage18_reference.npz', 'reference_sha256')]:
        remember(fixture / name, metadata[key])
    remember(fixture / 'quality_stage18_reference.json')
    arrays, diagnostics = {}, []
    with np.load(fixture / 'quality_stage18_inputs.npz') as inputs, \
         np.load(fixture / 'quality_stage18_reference.npz') as expected, \
         warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for case in metadata['cases']:
            data = real_inputs.get(case['key'], inputs[case['key']])
            np.testing.assert_array_equal(data, inputs[case['key']])
            for preset, params in metadata['presets'].items():
                result = get_eeg_quality_index_v2_parametric(data, fs=case['fs'], params=params, stage=case['stage'])
                prefix = case['key'] + '__' + preset + '__'
                for key, values in {'overall': result['overall'], **result['detail']}.items():
                    np.testing.assert_array_equal(values, expected[prefix + key])
                    arrays[prefix + key] = values
                diagnostics.append({'case': case['key'], 'preset': preset, 'valid': result['valid'].tolist(),
                                    'invalid_reasons': result['invalid_reasons'], 'context': result['context']})
        (out / 'warnings.log').write_text('\n'.join(str(w.message) for w in caught)+'\n')
    actual = out / 'actual.npz'
    np.savez_compressed(actual, **arrays)
    proof['comparisons'].append({'actual': str(actual), 'expected': str(fixture / 'quality_stage18_reference.npz'),
                                 'rtol': 0, 'atol': 0, 'equal_nan': True})
    (out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n')
    summary = {'status': 'passed', 'scope': 'scorer additive diagnostics; caller migration pending',
               'inputs': len(metadata['cases']), 'real_inputs': len(real_inputs), 'presets': len(metadata['presets']),
               'evaluations': len(diagnostics), 'arrays_compared': len(arrays), 'max_abs_error': 0.,
               'valid_channels': sum(sum(r['valid']) for r in diagnostics),
               'invalid_channels': sum(len(r['valid'])-sum(r['valid']) for r in diagnostics),
               'note': 'NaN locations also match. Existing legacy fallbacks and thresholds were not changed.'}
    (out / 'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    figure(out / 'diagnostics.png')
    for path in sorted(out.iterdir()):
        if path.is_file():
            remember(path)
    remember(ROOT / 'lilia/quality.py')
    remember(Path(__file__))
    (out / 'evidence.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(summary))


def figure(path):
    rng = np.random.default_rng(18018)
    data = rng.normal(size=(1, 500))
    lengths = [8, 50, 99, 100, 101, 150, 200, 400]
    params = {key+'_weight': float(key == 'flat') for key in ('flat', 'spectrum', 'kurtosis', 'corr')}
    results = [get_eeg_quality_index_v2_parametric(data[:, :n], params=params, stage='synthetic') for n in lengths]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(lengths, [r['overall'][0] for r in results], 'o-', label='Legacy overall (unchanged)')
    invalid = [i for i, r in enumerate(results) if not r['valid'][0]]
    ax.scatter([lengths[i] for i in invalid], [results[i]['overall'][0] for i in invalid],
               marker='x', s=100, color='red', label='Invalid: no activity subwindows')
    ax.axvline(100, color='gray', linestyle='--', label='0.5 seconds @ 200 Hz (exclusive stop)')
    ax.set(xlabel='Samples in input window', ylabel='Flat-only quality score', ylim=(-.05, 1.),
           title='Additive validity diagnostics; legacy scores and window loop retained')
    ax.legend(loc='upper left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
