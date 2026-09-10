"""Stage-17 real CLI exports, legacy numeric comparisons and diagnostic figures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lilia.io import bandpass_filter, load_merged_csv
from lilia.jenqwei_dataset import load_manifest, load_fragment
from lilia.provenance import file_sha256
from lilia.signal import resample_with_time


def write_source(path, t, raw):
    with path.open('x') as handle:
        handle.write('header\n' * 4)
        pd.DataFrame({'Time[us]': t, **{f'ch{i+1}': raw[:, i] for i in range(raw.shape[1])}}).to_csv(handle, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = dict(schema_version=1, files=[], tables=[], comparisons=[])
    summary = dict(cases=[], expected_failures=[])

    def remember(path):
        evidence['files'].append(dict(path=str(path), sha256=file_sha256(path)))

    def run(name, pattern, extra=(), failure=False):
        target = out / name
        command = [sys.executable, str(ROOT / 'build_jenqwei_tflite_dataset.py'),
                   '--input-glob', str(pattern), '--outdir', str(target), *extra]
        with (out / f'{name}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'MPLCONFIGDIR': '/tmp/lilia-mpl'})
        if (result.returncode == 0) == failure:
            raise ValueError(f'Unexpected CLI outcome: {name}; see log')
        audit = json.loads((target / 'run_audit.json').read_text())
        if audit['status'] != ('failed' if failure else 'complete'):
            raise ValueError(f'Wrong audit status: {name}')
        if failure:
            if (target / 'manifest.csv').exists():
                raise ValueError('Failure published a success manifest')
            summary['expected_failures'].append(dict(case=name, reason=audit['failed']['reason']))
            return
        frame, _ = load_manifest(target / 'manifest.csv')
        for row in frame.itertuples():
            load_fragment(row.out_csv, row.source_csv)
            evidence['tables'].append(dict(path=row.out_csv, kind='jenqwei_dataset', raw_csv=row.source_csv))
        summary['cases'].append(dict(case=name, files=len(frame), windows=int(frame.model_window_count.sum()),
                                     samples=int(frame.n_samples_200hz.sum()), sources=frame.source_csv.nunique()))
        print(summary['cases'][-1], flush=True)
        return frame

    frames = {}
    for single in (False, True):
        mode = 'windows' if single else 'splits'
        frame = run('real_' + mode, ROOT / 'jenqwei/*/*.csv', ['--one-window-per-file'] if single else [])
        frames['real_' + mode] = frame
        for source, group in frame.groupby('source_csv', sort=False):
            name = Path(source).stem.removeprefix('2026-06-18-lilia-').replace(' ', '_').replace('-', '_')
            compare_legacy(out, name, mode, group, evidence, remember)
    source = out / 'synthetic.csv'
    with np.load(ROOT / 'tests/fixtures/dataset_jenqwei_synthetic_reference.npz') as ref:
        write_source(source, ref['source_time_us'], ref['source_raw'])
    for name, rate in [('synthetic', 200.), ('fractional', 199.5)]:
        for single in (False, True):
            mode = 'windows' if single else 'splits'
            args = ['--fs-out', str(rate)] + (['--one-window-per-file'] if single else [])
            frame = run(name + '_' + mode, source, args)
            frames[name + '_' + mode] = frame
            compare_legacy(out, name, mode, frame, evidence, remember)
    real = frames['real_splits'].iloc[0].source_csv
    t, raw = load_merged_csv(real)
    t, raw = t[:10026].copy(), raw[:10026].copy()
    t[3503:] += 500000
    t[10006:] += 100000
    gapped = out / 'gapped.csv'
    write_source(gapped, t, raw)
    for single in (False, True):
        mode = 'windows' if single else 'splits'
        args = ['--n-splits', '2', '--allow-short-drop'] + (['--one-window-per-file'] if single else [])
        frame = run('gapped_' + mode, gapped, args)
        frames['gapped_' + mode] = frame
        actual, expected = {}, {}
        for i, row in enumerate(frame.itertuples()):
            a, b = row.raw_start_idx, row.raw_end_idx
            rt, values = resample_with_time(t[a:b], bandpass_filter(raw[a:b, :4]), 500., 200.)
            sl = slice(row.segment_local_start_idx, row.segment_local_end_idx)
            data = pd.read_csv(row.out_csv)
            actual[f'{i}_time_us'] = data.time_us.to_numpy(dtype=np.int64)
            actual[f'{i}_values'] = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            expected[f'{i}_time_us'], expected[f'{i}_values'] = rt[sl], values[sl]
        compare_pair(out, 'gapped_' + mode, actual, expected, evidence)
    run('strict_short', gapped, ['--n-splits', '2'], failure=True)
    run('all_dropped', source, ['--n-splits', '100', '--allow-short-drop'], failure=True)
    raw[-1, 0] = np.nan
    polluted = out / 'polluted.csv'
    write_source(polluted, t, raw)
    run('polluted_tail', polluted, ['--n-splits', '2', '--allow-short-drop'], failure=True)
    for name in ('real_splits', 'gapped_splits', 'fractional_splits'):
        diagnostic(out / f'{name}.png', frames[name])
    for path in sorted(out.rglob('*')):
        if path.is_file():
            remember(path)
    for path in [ROOT / 'build_jenqwei_tflite_dataset.py', ROOT / 'lilia/jenqwei_dataset.py', Path(__file__),
                 ROOT / 'tools/capture_jenqwei_dataset_stage17.py']:
        remember(path)
    summary['numerics'] = 'All CSV float32 values and int64 microseconds match exactly; gapped references filter each full segment independently.'
    (out / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    remember(out / 'analysis.json')
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


def compare_pair(out, name, actual, expected, evidence):
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    a, b = out / f'{name}_actual.npz', out / f'{name}_expected.npz'
    np.savez_compressed(a, **actual)
    np.savez_compressed(b, **expected)
    evidence['comparisons'].append(dict(actual=str(a), expected=str(b), rtol=0., atol=0., equal_nan=False))


def compare_legacy(out, name, mode, frame, evidence, remember):
    fixture = ROOT / 'tests/fixtures' / f'dataset_jenqwei_{name}_reference.npz'
    meta = json.loads(fixture.with_suffix('.json').read_text())
    if file_sha256(fixture) != meta['npz_sha256']:
        raise ValueError('Frozen fixture hash changed')
    remember(fixture)
    remember(fixture.with_suffix('.json'))
    rows = list(frame.itertuples(index=False))
    if len(rows) != len(meta['modes'][mode]):
        raise ValueError('Legacy fragment count differs')
    actual, expected = {}, {}
    with np.load(fixture) as frozen:
        for i, (row, old) in enumerate(zip(rows, meta['modes'][mode])):
            for key in old.keys() - {'source_csv', 'out_csv'}:
                if getattr(row, key) != old[key]:
                    raise ValueError(f'Legacy metadata differs: {name}/{mode}/{i}/{key}')
            if name not in ('synthetic', 'fractional') and Path(row.out_csv).name != Path(old['out_csv']).name:
                raise ValueError('Continuous filename differs')
            if file_sha256(row.source_csv) != meta['source_sha256']:
                raise ValueError('Baseline source differs')
            data = pd.read_csv(row.out_csv)
            for key, values in [('time_us', data.time_us.to_numpy(dtype=np.int64)),
                                ('values', data.iloc[:, 1:].to_numpy(dtype=np.float32))]:
                actual[f'{i}_{key}'] = values
                expected[f'{i}_{key}'] = frozen[f'{mode}_{i}_{key}']
    compare_pair(out, name + '_' + mode, actual, expected, evidence)


def diagnostic(path, frame):
    source = frame.iloc[0].source_csv
    frame = frame.loc[frame.source_csv == source]
    t, _ = load_merged_csv(source)
    _, meta = load_fragment(frame.iloc[0].out_csv, source)
    p = meta['parameters']
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    for segment in meta['plan']['segments']:
        sid = segment['segment_id']
        a, b = segment['raw_start_idx'], segment['raw_end_idx']
        rt = resample_with_time(t[a:b], np.zeros((b-a, 1)), p['fs_in'], p['fs_out'])[0]
        for split in segment['splits']:
            start, end, keep = split['local_start_idx'], split['local_end_idx'], split['retained_end_idx']
            if end == start:
                continue
            x = (rt[start] - t[0]) / 1e6
            width = (rt[end-1] - rt[start]) / 1e6 + 1 / p['fs_out']
            axes[0].broken_barh([(x, width)], (sid-.3, .6), facecolors='lightgray')
            for ws in range(start, keep, p['tflite_win']):
                wx = (rt[ws] - t[0]) / 1e6
                axes[0].broken_barh([(wx, p['tflite_win']/p['fs_out'])], (sid-.22, .44),
                                    facecolors='tab:blue', edgecolors='white', linewidth=.8)
            if any(item['retained_samples'] for item in segment['splits']):
                axes[0].text(x, sid+.33, f'split {split["split_index"]}', fontsize=8)
        if not any(item['retained_samples'] for item in segment['splits']):
            axes[0].text((rt[0]-t[0])/1e6, sid+.33, 'short segment (excluded)', ha='right', fontsize=8)
    for row in frame.itertuples():
        data = pd.read_csv(row.out_csv)
        axes[1].plot((data.time_us.to_numpy(dtype=np.int64)-t[0])/1e6, data.ch1, linewidth=.6)
    axes[0].set(title='Blue: exported model windows; gray: excluded tail / short segment', ylabel='Source segment')
    axes[0].set_yticks([s['segment_id'] for s in meta['plan']['segments']])
    axes[0].set_ylim(-.5, len(meta['plan']['segments'])-.2)
    axes[1].set(title='Exported bandpass ch1; separate lines per fragment', xlabel='Elapsed source time (s)', ylabel='Amplitude')
    fig.suptitle(f'{Path(source).name} | {p["fs_out"]} Hz | {int(frame.model_window_count.sum())} windows')
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
