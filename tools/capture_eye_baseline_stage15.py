"""Freeze the pre-stage-15 continuous eye pipeline from commit 96fee9d.

Writes only to a new directory; never replaces fixtures or research outputs.
Shared processing dependencies must still match the recorded old source.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch

from lilia.neural import model_provenance
from lilia.provenance import file_sha256

COMMIT = '96fee9d'
DEPENDENCIES = {
    'data_analysis.py': '762ccb415cab158f1d6ea74b7b3a2236a7f4c68dd369bc6457fd5ad85d9b67bc',
    'lilia/io.py': 'f85c790c46e3ccb9e16b9151ccbce758ab93683f3bc0b14f13c1e8d1e44abe1e',
    'lilia/signal.py': '6eb67263801e1aabb0bbee299f9234a578e40d678632eea246d0369f867c8c7d',
    'lilia/windowing.py': 'd363b9c4670b95ab33d7233ceb9a4d9407ebd442b9ce00716144551317c9bd8c',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    for name, expected in DEPENDENCIES.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f'Old dependency changed: {name}; use a checkout of {COMMIT}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(
        ['git', 'show', f'{COMMIT}:process_lilia_eye_open_close.py'], cwd=ROOT)
    snapshot = out / 'legacy_entry.py.txt'
    snapshot.write_bytes(source)
    eye = types.ModuleType('legacy_eye')
    exec(compile(source, str(snapshot), 'exec'), eye.__dict__)
    torch.set_num_threads(1)
    model = eye.load_model()
    provenance = model_provenance()
    # model_provenance describes another entry's filter chain. Record only the
    # actual eye filter here; never import that chain into eye analysis.
    provenance['filters'] = {'bandpass': [eye.BANDPASS_LOW, eye.BANDPASS_HIGH],
                             'order': 4, 'output_dtype': 'float32'}
    recording = ROOT / eye.DEFAULT_CSV
    real_time, real_raw = eye.load_lilia_csv(str(recording))
    seed = 15015
    cases = {
        'real': (real_time, real_raw),
        'synthetic': (1234567 + np.arange(1503, dtype=np.int64) * 2000,
                      np.random.default_rng(seed).normal(size=(1503, 8)).astype(np.float32)),
    }
    manifest = {'schema_version': 1, 'files': []}
    for name, (time_us, raw) in cases.items():
        filtered = eye.bandpass_filter(raw, fs=eye.FS, lo=eye.BANDPASS_LOW, hi=eye.BANDPASS_HIGH)
        time_200, before = eye.downsample_data(time_us.astype(float) / 1e6, filtered,
                                             fs_in=eye.FS, fs_out=eye.DOWNSAMPLED_FS)
        processed = np.concatenate([eye.run_model(model, before[:, :4]),
                                    eye.run_model(model, before[:, 4:8])], axis=1)
        values = {'raw_time_us': time_us, 'raw': raw, 'filtered': filtered,
                  'time_s': time_200, 'time_us': np.rint(time_200 * 1e6).astype(np.int64),
                  'before': before, 'processed': processed}
        for col, channel in enumerate((1, 2, 5, 6)):
            for stage, signal in (('before', before[:, channel - 1]),
                                  ('after', processed[:, col])):
                f, t, db = eye.compute_stft_db(signal, eye.DOWNSAMPLED_FS, 50.)
                values[f'{stage}_ch{channel}_f'] = f
                values[f'{stage}_ch{channel}_t'] = t
                values[f'{stage}_ch{channel}_db'] = db
        npz = out / f'eye_{name}_continuous_reference.npz'
        np.savez_compressed(npz, **values)
        info = {'legacy_commit': COMMIT, 'legacy_entry_sha256': file_sha256(snapshot),
                'dependencies': DEPENDENCIES, 'source_path': str(recording.relative_to(ROOT)) if name == 'real' else None,
                'source_sha256': file_sha256(recording) if name == 'real' else None,
                'seed': seed if name == 'synthetic' else None,
                'raw_shape': list(raw.shape), 'output_shape': list(processed.shape),
                'fs_in': eye.FS, 'fs_out': eye.DOWNSAMPLED_FS,
                'model': provenance, 'torch_version': torch.__version__,
                'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
                'threads': 1, 'npz_sha256': file_sha256(npz),
                'notes': ['Model output is packed as source channels 1,2,5,6.',
                          'Before STFT uses actual source channels 1,2,5,6.',
                          'Old ch5/6 comparison PNG instead plotted before ch3/4; this is a known plotting bug.',
                          'STFT: scipy defaults, Hann 256, overlap 192, 20*log10(abs(z)+1e-8), f<=50.',
                          'All real resampled tail samples retained by existing mirrored overlap-add.']}
        metadata = npz.with_suffix('.json')
        metadata.write_text(json.dumps(info, indent=2) + '\n')
        for path in (npz, metadata):
            manifest['files'].append({'path': str(path), 'sha256': file_sha256(path)})
        print(f'{name}: raw={raw.shape}, before={before.shape}, output={processed.shape}, '
              f'last_time_us={values["time_us"][-1]}', flush=True)
    manifest['files'].append({'path': str(snapshot), 'sha256': file_sha256(snapshot)})
    (out / 'evidence.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
