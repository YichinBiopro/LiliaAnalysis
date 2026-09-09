"""Freeze stage-16 legacy model, Welch and STFT values in a new directory."""
import argparse
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
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

from lilia.provenance import file_sha256

COMMIT = '141bde2'
DEPENDENCIES = ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py', 'lilia/tflite.py',
                'plot_event_markers.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    dependencies = {}
    for name in DEPENDENCIES:
        original = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
        expected = hashlib.sha256(original).hexdigest()
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen shared dependency changed: {name}')
        dependencies[name] = expected
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:analyze_jenqwei_pipeline.py'], cwd=ROOT)
    snapshot = out / 'legacy_entry.py.txt'
    snapshot.write_bytes(source)
    legacy = types.ModuleType('legacy_jenqwei')
    legacy.__file__ = str(ROOT / 'analyze_jenqwei_pipeline.py')
    exec(compile(source, str(snapshot), 'exec'), legacy.__dict__)
    model = ROOT / 'tiny_v4_optimized.tflite'
    manifest = {'schema_version': 1, 'files': []}
    inventory = []

    def remember(path):
        manifest['files'].append({'path': str(path), 'sha256': file_sha256(path)})

    raw_paths = [p for p in sorted((ROOT / 'jenqwei').rglob('*.csv')) if 'time_marker' not in p.name.lower()]
    if len(raw_paths) != 5:
        raise ValueError('Review recording inventory before changing the five-source baseline')
    cases = []
    for path in raw_paths:
        t, raw = legacy.load_raw_csv(str(path))
        name = path.stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_')
        cases.append((name, t, raw, path, None))
    raw = np.random.default_rng(16016).normal(size=(2503, 4)).astype(np.float32)
    t = 1234567 + np.arange(len(raw), dtype=np.int64) * 2000
    cases.extend([('synthetic', t, raw, None, None), ('synthetic_truncated', t, raw, None, 1503)])
    for name, t, raw, path, limit in cases:
        result = legacy.run_pipeline(t, raw, max_samples=limit)
        arrays = {'source_time_us': t, 'source_raw': raw, **result}
        # Capture the actual arrays produced by the old plotting functions,
        # including their legacy channel clamp and pre-STFT display truncation.
        for ch in range(4):
            psds, stfts = [], []
            original_welch, original_stft = legacy.welch, legacy.stft

            def capture_welch(*args, **kwargs):
                value = original_welch(*args, **kwargs)
                psds.append(value)
                return value

            def capture_stft(*args, **kwargs):
                value = original_stft(*args, **kwargs)
                stfts.append(value)
                return value

            fig, axes = plt.subplots(1, 3)
            try:
                with patch.object(legacy, 'welch', capture_welch), patch.object(legacy, 'stft', capture_stft):
                    legacy._plot_psd(axes[0], result, ch)
                    legacy._plot_stft(axes[1], axes[2], result, ch, 60.)
            finally:
                plt.close(fig)
            for branch, (f_psd, power), (f_stft, time, z) in zip(('before', 'after'), psds, stfts):
                prefix = f'ch{ch}_{branch}'
                arrays[prefix + '_psd_f'] = f_psd
                arrays[prefix + '_psd'] = power
                mask = f_stft <= legacy.PSD_FMAX
                arrays[prefix + '_stft_f'] = f_stft[mask]
                arrays[prefix + '_stft_t'] = time
                arrays[prefix + '_stft_db'] = 20 * np.log10(np.abs(z[mask]) + 1e-8)
        npz = out / f'jenqwei_{name}_reference.npz'
        np.savez_compressed(npz, **arrays)
        metadata = {'legacy_commit': COMMIT, 'legacy_entry_sha256': file_sha256(snapshot),
                    'dependencies': dependencies, 'source_path': str(path.relative_to(ROOT)) if path else None,
                    'source_sha256': file_sha256(path) if path else None, 'seed': None if path else 16016,
                    'model_sha256': file_sha256(model), 'numpy_version': np.__version__,
                    'scipy_version': scipy.__version__, 'tensorflow_version': tf.__version__,
                    'source_shape': list(raw.shape), 'before_shape': list(result['pre_data_200'].shape),
                    'after_shape': list(result['tfl_data_200'].shape), 'max_samples': limit,
                    'max_display_sec': 60., 'npz_sha256': file_sha256(npz),
                    'notes': ['Before retains all resampled tail samples; After trims to complete 400-sample windows.',
                              'Filtering precedes max_samples truncation; preserve full-segment filter context.',
                              'Legacy display ch2/3 (0-based) clamps After to model ch1; known mismatched comparison, not a valid same-source channel contract.',
                              'Welch density: nperseg=min(800,len(before)); After min with own length; defaults Hann, half overlap, constant detrend.',
                              'STFT: first min(len,12000) samples; Hann 256, overlap 128, default zero boundary/padding; 20*log10(abs(z)+1e-8).']}
        meta_path = npz.with_suffix('.json')
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
        remember(npz)
        remember(meta_path)
        if path:
            remember(path)
        inventory.append({'case': name, **metadata})
        print(f'{name}: raw={raw.shape}, before={result["pre_data_200"].shape}, after={result["tfl_data_200"].shape}', flush=True)
    remember(snapshot)
    remember(model)
    remember(Path(__file__))
    for name in DEPENDENCIES:
        remember(ROOT / name)
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    remember(out / 'inventory.json')
    (out / 'evidence.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
