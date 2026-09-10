"""Freeze the committed dataset exporter, before stage-17 migration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from lilia.provenance import file_sha256

COMMIT = 'e8ebe18'
DEPENDENCIES = ('lilia/io.py', 'lilia/signal.py', 'lilia/windowing.py', 'lilia/provenance.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    dependencies = {}
    for name in DEPENDENCIES:
        original = subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)
        dependencies[name] = hashlib.sha256(original).hexdigest()
        if file_sha256(ROOT / name) != dependencies[name]:
            raise ValueError(f'Frozen dependency differs: {name}')
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:build_jenqwei_tflite_dataset.py'], cwd=ROOT)
    (out / 'legacy_entry.py.txt').write_bytes(source)
    module = types.ModuleType('legacy_dataset_stage17')
    module.__file__ = str(ROOT / 'build_jenqwei_tflite_dataset.py')
    sys.modules[module.__name__] = module
    exec(compile(source, str(out / 'legacy_entry.py.txt'), 'exec'), module.__dict__)
    paths = module.find_input_csvs(str(ROOT / 'jenqwei/*/*.csv'), 'time_marker.csv')
    if len(paths) != 5:
        raise ValueError('Expected the five inventoried real recordings')
    synthetic = out / 'synthetic.csv'
    t = 1700000000000001 + np.arange(7503, dtype=np.int64) * 2000
    raw = np.random.default_rng(17017).normal(size=(len(t), 4)).astype(np.float32)
    with synthetic.open('x') as handle:
        handle.write('header\n' * 4)
        pd.DataFrame({'Time[us]': t, **{f'ch{i+1}': raw[:, i] for i in range(4)}}).to_csv(handle, index=False)
    cases = [(Path(p).stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_'), Path(p), 200.) for p in paths]
    cases += [('synthetic', synthetic, 200.), ('fractional', synthetic, 199.5)]
    inventory = []
    for name, path, fs_out in cases:
        t, raw = module.load_merged_csv(path)
        bp = module.bandpass_filter(raw[:, :4], fs=500., lo=.5, hi=45.)
        rt, rx = module.downsample_with_time(t, bp, 500., fs_out)
        arrays = {'source_time_us': t, 'source_raw': raw, 'filtered': bp, 'time_us': rt, 'resampled': rx}
        modes = {}
        for single in (False, True):
            mode = 'windows' if single else 'splits'
            rows = module.process_one_csv(str(path), str(out / name / mode), 5, 500., fs_out, .5, 45., 4, 400, False, single)
            modes[mode] = [r.__dict__ for r in rows]
            for i, row in enumerate(rows):
                frame = pd.read_csv(row.out_csv)
                arrays[f'{mode}_{i}_time_us'] = frame.time_us.to_numpy(dtype=np.int64)
                arrays[f'{mode}_{i}_values'] = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        npz = out / f'dataset_jenqwei_{name}_reference.npz'
        np.savez_compressed(npz, **arrays)
        meta = {'legacy_commit': COMMIT, 'entry_sha256': file_sha256(out / 'legacy_entry.py.txt'),
                'dependencies': dependencies, 'source': str(path), 'source_sha256': file_sha256(path),
                'fs_out': fs_out, 'modes': modes, 'npz_sha256': file_sha256(npz)}
        npz.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        inventory.append({'case': name, 'samples': len(rt), 'splits': len(modes['splits']), 'windows': len(modes['windows']), 'sha256': file_sha256(npz)})
        print(inventory[-1], flush=True)
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')


if __name__ == '__main__':
    main()
