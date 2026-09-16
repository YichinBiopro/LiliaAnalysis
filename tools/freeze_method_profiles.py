"""Freeze pre-calibration behavior against committed code; never overwrite a run.

This is a compatibility baseline, not evidence of scientific validity. Real
recording derivatives and the extracted source tree stay in the ignored local/
directory. Synthetic arrays and source-bound hashes are reviewable in Git.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'aa0df5f4ebaefddae1fee8c9e50e95323e777aed'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare_arrays(expected, actual):
    import numpy as np
    with np.load(expected, allow_pickle=False) as old, np.load(actual, allow_pickle=False) as new:
        if set(old.files) != set(new.files) or not old.files:
            raise ValueError('Array key coverage differs or is empty')
        for key in old.files:
            if old[key].dtype != new[key].dtype or old[key].shape != new[key].shape:
                raise ValueError('Array dtype/shape differs: ' + key)
            np.testing.assert_array_equal(old[key], new[key], err_msg=key)
        return len(old.files), [key for key in old.files if not old[key].size]


def verify_files(entries, root):
    for row in entries:
        path = Path(row['path'])
        if not path.is_absolute():
            path = root / path
        if sha(path) != row['sha256']:
            raise ValueError('Frozen input/artifact hash differs: ' + str(path))


def plot_review(out, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    with np.load(out/'local/expected'/(name+'.npz')) as old, np.load(out/'local/actual'/(name+'.npz')) as new:
        for data, label, style in [(old, 'frozen', '-'), (new, 'current', '--')]:
            axes[0, 0].plot(data['psd_entropy_s0_ch0__frequency'],
                10*np.log10(np.maximum(data['psd_entropy_s0_ch0__density'], 1e-12)), style, label=label)
        axes[0, 0].set(xlim=(0, 80), xlabel='Hz', ylabel='dB density', title='Entropy Welch: first source segment, ch1')
        axes[0, 0].legend()
        key = 'goertzel_0.5_'
        groups = new[key+'segment_ids']
        for sid in np.unique(groups):
            keep = groups == sid
            for data, style in [(old, '-'), (new, '--')]:
                axes[0, 1].plot(data[key+'time_s'][keep], data[key+'smooth'][keep], style, marker='.', ms=3)
        axes[0, 1].set(xlabel='Elapsed seconds', ylabel='Unnormalized power, dB',
                       title='Goertzel: legacy median 5, quality > .5')
        metadata = json.loads((out/'local/actual'/(name+'.json')).read_text())
        catalog = metadata.get('model_baseline_catalog', [])
        if catalog:
            epoch = new['raw_time_us'][0]
            xx = [(r['window_start_us']-epoch)/1e6 for r in catalog]
            yy = [np.nan if r['quality_min'] is None else r['quality_min'] for r in catalog]
            axes[1, 0].plot(xx, yy, '.', label='screened model blocks')
            selected = metadata.get('model_selection_session', {}).get('selected', [])
            chosen = {r['model_window_id'] for r in selected}
            ii = [i for i, r in enumerate(catalog) if r['model_window_id'] in chosen]
            axes[1, 0].scatter(np.asarray(xx)[ii], np.asarray(yy)[ii], facecolors='none', edgecolors='red', label='seed 42 selected')
            axes[1, 0].axhline(.5, color='gray', ls=':')
            axes[1, 0].legend(fontsize=8)
            if not chosen:
                axes[1, 0].text(.03, .9, 'Insufficient eligible baseline', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(.1, .5, 'No complete 2s model block', transform=axes[1, 0].transAxes)
        axes[1, 0].set(xlabel='Elapsed seconds', ylabel='Minimum accepted subepoch quality', title='TFLite baseline: selection audit')
        axes[1, 1].hist(new['mi_null_quantile__surrogates'], bins=20, label='200 within-segment shifts')
        axes[1, 1].axvline(new['mi_null_quantile__mutual_information'][0], color='red', label='observed')
        axes[1, 1].set(xlabel='Histogram MI, bits', ylabel='Count', title='Legacy quantile MI null, seed 0')
        axes[1, 1].legend(fontsize=8)
    fig.suptitle(name + ' — compatibility baseline (no scientific calibration)')
    path = out / ('local' if name.startswith('real_') else 'synthetic') / (name+'_review.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def worker(args):
    # Separate interpreters prevent current imports from leaking into old code.
    sys.path.insert(0, str(args.code_root))
    import numpy as np
    import pandas as pd
    import spectral_entropy as entropy
    import plot_goertzel_vs_raw as goertzel
    from lilia.io import load_merged_csv, bandpass_filter
    from lilia.windowing import continuous_slices, build_window_grid
    from lilia.qeeg import compute_relative_powers
    from lilia.comparison import segmented_psd
    from lilia.hardy2 import window_metrics
    from lilia.event_qeeg import analyze_recording, summarize_branch
    from lilia.subject_comparison import summarize_comparison
    from lilia.meditation import summarize_meditation
    from lilia.tflite import run_tflite_recording
    from lilia.tflite_baseline import score_baseline_windows, select_tflite_baseline
    from lilia.quality import get_eeg_quality_index_v2_parametric, get_ibrain_device_eeg_quality_v2_params
    from lilia.quality_audit import json_value
    from lilia.quality_policy import valid_goertzel_rows
    from types import SimpleNamespace

    config = json.loads(args.config.read_text())
    verify_files(config['inputs'], ROOT)
    params = get_ibrain_device_eeg_quality_v2_params()
    scorer = get_eeg_quality_index_v2_parametric
    args.out.mkdir(parents=True, exist_ok=False)
    for case in config['cases']:
        name = case['name']
        if 'source' in case:
            time, raw = load_merged_csv(case['source'])
            raw = raw[:, :4]
        else:
            n = 250 if name == 'synthetic_short' else 30000
            rng = np.random.default_rng(1901)
            seconds = np.arange(n) / 500.
            raw = np.column_stack([20*np.sin(2*np.pi*(6+4*i)*seconds)
                    + 4*np.sin(2*np.pi*60*seconds) + rng.normal(size=n) for i in range(4)]).astype(np.float32)
            time = np.arange(n, dtype=np.int64)*2000
            if name == 'synthetic_gap':
                time[n//2:] += 10000000
        arrays, meta = {}, {'case': name, 'samples': len(time), 'channels': [1, 2, 3, 4],
                            'method_errors': {}}

        def save(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    save(prefix + '__' + str(key), item)
            elif isinstance(value, np.ndarray):
                if value.dtype.kind in 'biuf':
                    arrays[prefix] = value
            elif isinstance(value, (float, int, np.number, bool)):
                arrays[prefix] = np.asarray([value])

        segments = continuous_slices(time, 500.)
        filtered = np.empty_like(raw)
        ids = np.empty(len(time), dtype=np.int64)
        for sid, sl in enumerate(segments):
            filtered[sl] = bandpass_filter(raw[sl], fs=500., lo=.5, hi=45.)
            ids[sl] = sid
            for ch in (0, 1):
                part = filtered[sl, ch]
                save(f'psd_entropy_s{sid}_ch{ch}', dict(zip(('frequency', 'density'),
                     entropy._compute_welch_psd(part, fs=500.))))
                save(f'psd_qeeg_s{sid}_ch{ch}', np.asarray(compute_relative_powers(part, fs=500.)))
            save(f'psd_hardy_s{sid}', window_metrics(filtered[sl], 500.))
        arrays.update(raw_time_us=time, source_segment_ids=ids)
        f, db, audit = segmented_psd(filtered[:, 0], 500., segments)
        save('psd_comparison', dict(frequency=f, db=db))
        meta['psd_comparison_audit'] = audit

        result = analyze_recording(time, raw, scorer=scorer, quality_params=params)
        branch = result['bp']
        meta['baseline_segments'] = result['segments']
        if branch is not None:
            save('baseline_scores', branch['scores'])
            save('baseline_grid', branch['grid'].columns)
            save('baseline_quality', branch['quality'])
            save('baseline_valid', branch['valid'])
            # Diagnostic intervals, not claims about actual subject activity.
            start = int(time[0])
            events = [dict(label='probe', start_us=start+30000000,
                           end_us=start+60000000, participates=True, color='blue')]
            for policy in ('session-start', 'pre-event-rest'):
                summary = summarize_branch(branch, events, baseline_mode=policy)
                save('baseline_event_' + policy, summary)
                meta['baseline_event_' + policy] = json_value(summary)
            meta['baseline_no_events'] = json_value(summarize_branch(branch, []))
            meta['baseline_subject_session'] = json_value(summarize_comparison(branch))
            meta['baseline_subject_event'] = json_value(summarize_comparison(branch, events))
            meta['baseline_meditation'] = json_value(summarize_meditation(
                branch, start+30000000, start, int(time[-1])+2000))
        else:
            meta['method_errors']['baseline_grid'] = result['errors']

        try:
            timeline, before, after = run_tflite_recording(time, filtered, config['model'])
            save('model', dict(time_us=timeline.time_us, before=before, after=after,
                               segment_ids=timeline.segment_ids))
            meta['model_timeline'] = timeline.metadata()
            catalog = score_baseline_windows(time, filtered, raw, timeline,
                                             scorer=scorer, quality_params=params)
            meta['model_baseline_catalog'] = catalog
            for label, lo, hi in [('session', int(time[0]), int(time[-1])+2000),
                                  ('pre_event', int(time[0])+15000000, int(time[0])+27000000)]:
                try:
                    reference, selection = select_tflite_baseline(after, timeline, catalog, lo, hi)
                    save('model_reference_' + label, reference)
                    meta['model_selection_' + label] = selection
                except ValueError as exc:
                    if not str(exc).startswith('Insufficient complete baseline'):
                        raise
                    meta['method_errors']['model_selection_' + label] = str(exc)
        except ValueError as exc:
            if str(exc) != 'No complete TFLite model window in any source segment':
                raise
            meta['method_errors']['model'] = str(exc)

        for win in (5., .5):
            metric = goertzel._compute_window_metrics(time, raw, filtered, 500., 1, 60., win, win,
                                                       params, 1950., .12, 1000., 1., 80.)
            for key in metric.__dataclass_fields__:
                if key != 'quality_audit':
                    save(f'goertzel_{win}_{key}', getattr(metric, key))
            frame = pd.DataFrame(dict(quality=metric.quality, quality_final=metric.quality_final,
                goertzel_db=metric.goertzel_db, time_s=metric.time_s, artifact_hard_clip=metric.hard_artifact))
            keep = valid_goertzel_rows(frame, .5)
            save(f'goertzel_{win}_keep', keep)
            save(f'goertzel_{win}_smooth', np.where(keep,
                goertzel._rolling_median(np.where(keep, metric.goertzel_db, np.nan), 5), np.nan))

        for binning in ('uniform', 'quantile'):
            save('mi_population_' + binning, entropy.compute_joint_probability(
                filtered[:, 0], filtered[:, 1], bins=16, binning=binning))
            significance = entropy.compute_joint_mi_significance(filtered[:, 0], filtered[:, 1],
                bins=16, binning=binning, n_surrogates=200, seed=0, segment_ids=ids)
            save('mi_null_' + binning, significance)
            meta['mi_null_' + binning] = json_value({k: v for k, v in significance.items() if k != 'surrogates'})
        if len(time) >= 1000:
            grid = build_window_grid(time, 500., 2., 2.)
            save('mi_grid', grid.columns)
            for binning in ('uniform', 'quantile'):
                save('mi_windows_' + binning, entropy.compute_joint_mi_windowed(
                    filtered[:, 0], filtered[:, 1], fs=500., bins=16, binning=binning, windows=grid))
            save('mi_lagged', entropy.compute_lagged_interhemispheric_sync_windowed(
                filtered[:, 0], filtered[:, 1], fs=500., windows=grid, apply_bandpass=False))
        envelopes = entropy.extract_band_envelopes(raw[:, 0], fs=500., time_us=time)
        intervals = []
        for sl in segments:
            mid = (sl.start+sl.stop)//2
            if mid-sl.start >= 2500 and sl.stop-mid >= 2500:
                intervals.append(SimpleNamespace(pre=slice(mid-2500, mid), post=slice(mid, mid+2500)))
        ksg = entropy.compute_band_event_joint_mi(envelopes, [], 2500, fs=500.,
                    n_surrogates=200, random_state=0, event_windows=intervals)
        save('mi_ksg', ksg)
        meta['mi_ksg'] = ksg
        if name != 'synthetic_short' and ksg is None:
            raise ValueError('Unexpected missing KSG baseline: ' + name)
        np.savez_compressed(args.out / (name+'.npz'), **arrays)
        write_json(args.out / (name+'.json'), json_value(meta))
        print(f'PASS {name}: {len(arrays)} arrays', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('local/\n!*.png\n')
    frozen = out / 'local/frozen_source'
    frozen.mkdir(parents=True)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE], cwd=ROOT).decode().splitlines()
    names = [n for n in names if n.endswith('.py') and ('/' not in n or n.startswith('lilia/'))]
    frozen_hashes, current_hashes = {}, {}
    for name in names:
        code = subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT)
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(code)
        frozen_hashes[name] = sha(target)
        current_hashes[name] = sha(ROOT / name)
    fixtures = ROOT / 'tests/fixtures/quality_stage18_reference.json'
    sources = json.loads(fixtures.read_text())['sources']
    model = ROOT / 'tiny_v4_optimized.tflite'
    config = dict(schema_version=1, baseline_commit=BASELINE, model=str(model),
        inputs=[*sources, dict(path=str(model), sha256=sha(model)),
                dict(path=str(fixtures), sha256=sha(fixtures))],
        cases=[dict(name=f'real_{i}', source=r['path']) for i, r in enumerate(sources)] +
              [dict(name=n) for n in ('synthetic_continuous', 'synthetic_gap', 'synthetic_short')],
        recipe='Full real recording, first four declared channels; 500 Hz; BP .5–45 float32 per source segment. '
               'Diagnostic event [30,60)s; KSG +/-5s at each segment midpoint, not annotated subject events. '
               'Synthetic seed 1901, 60s / two 30s with 10s gap / .5s; see frozen harness.',
        packages={p: importlib.metadata.version(p) for p in
                  ('numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'tensorflow')},
        python=sys.version, frozen_source_hashes=frozen_hashes, current_source_hashes=current_hashes,
        harness_sha256=sha(__file__))
    verify_files(config['inputs'], ROOT)
    write_json(out/'config.json', config)
    env = {**os.environ, 'MPLCONFIGDIR': str(out/'local/mpl'), 'MPLBACKEND': 'Agg',
           'TF_CPP_MIN_LOG_LEVEL': '2', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    os.environ['MPLCONFIGDIR'] = env['MPLCONFIGDIR']
    for label, code_root in [('expected', frozen), ('actual', ROOT)]:
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                '--code-root', str(code_root), '--config', str(out/'config.json'),
                '--out', str(out/'local'/label)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PASS ' + label + ' worker', flush=True)
    manifest = dict(schema_version=1, files=[], comparisons=[])
    cases = []
    for case in config['cases']:
        name = case['name']
        old, new = [out/'local'/label/(name+'.npz') for label in ('expected', 'actual')]
        count, empty = compare_arrays(old, new)
        if old.with_suffix('.json').read_bytes() != new.with_suffix('.json').read_bytes():
            raise ValueError('Selection/audit/status differs: ' + name)
        cases.append(dict(name=name, arrays=count, empty_arrays=empty, exact=True))
        # General evidence requires nonempty arrays; empty shape/dtype checked above.
        import numpy as np
        for label, source in [('expected', old), ('actual', new)]:
            dest = out / ('local' if name.startswith('real_') else 'synthetic') / (name+'_'+label+'.npz')
            dest.parent.mkdir(exist_ok=True)
            with np.load(source, allow_pickle=False) as data:
                np.savez_compressed(dest, **{k: data[k] for k in data.files if k not in empty})
        folder = 'local' if name.startswith('real_') else 'synthetic'
        manifest['comparisons'].append(dict(expected=f'{folder}/{name}_expected.npz',
            actual=f'{folder}/{name}_actual.npz', rtol=0., atol=0., equal_nan=True))
    verify_files(config['inputs'], ROOT)
    if any(sha(ROOT/n) != digest for n, digest in current_hashes.items()) or sha(__file__) != config['harness_sha256']:
        raise ValueError('Code changed during capture')
    write_json(out/'analysis.json', dict(baseline_commit=BASELINE, cases=cases,
        array_count=sum(r['arrays'] for r in cases), max_abs_error=0., tolerance=0.,
        selection_audits_equal=True, scientific_calibration=False,
        limitations=['Probe events are not actual behavioral annotations.',
                    'Not all inventoried legacy profiles are exercised; see inventory coverage.']))
    for name in ('real_1', 'synthetic_gap', 'synthetic_short'):
        plot_review(out, name)
    for path in sorted(out.rglob('*')):
        if path.is_file() and '/mpl/' not in str(path):
            manifest['files'].append(dict(path=str(path.relative_to(out)), sha256=sha(path)))
    manifest['files'] += config['inputs'] + [dict(path=str(ROOT/n), sha256=d) for n, d in current_hashes.items()]
    manifest['files'].append(dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    write_json(out/'manifest_local.json', manifest)
    public = dict(schema_version=1,
        files=[r for r in manifest['files'] if not r['path'].startswith('local/')
               and r not in config['inputs']],
        comparisons=[r for r in manifest['comparisons'] if r['actual'].startswith('synthetic/')])
    write_json(out/'manifest_repository.json', public)
    print(f'PASS {len(cases)} cases, {sum(r["arrays"] for r in cases)} exact arrays')


if __name__ == '__main__':
    main()
