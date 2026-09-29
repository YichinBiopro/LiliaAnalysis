"""Source-bound M4-R1 event plot checks, with legacy artist data comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import joint_mi
import spectral_entropy as spectral
from tools.freeze_method_profiles import compare_arrays, sha, verify_files, write_json


def load_old(path):
    spec = importlib.util.spec_from_file_location('plot_reference_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(plot, frame, path, *, render):
    arrays, labels = {}, []
    original = Figure.savefig
    def save(fig, target, **kwargs):
        for i, ax in enumerate(fig.axes):
            labels.extend([ax.get_title(), ax.get_xlabel(), ax.get_ylabel()])
            if ax.get_legend() is not None:
                labels.append(ax.get_legend().get_title().get_text())
            for j, line in enumerate(ax.lines):
                for axis in ('x', 'y'):
                    arrays[f'axis{i}_line{j}_{axis}'] = np.asarray(getattr(line, 'get_'+axis+'data')(), dtype=float)
            for j, bar in enumerate(ax.patches):
                arrays[f'axis{i}_bar{j}'] = np.array([bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()])
            for j, collection in enumerate(ax.collections):
                arrays[f'axis{i}_collection{j}_offsets'] = np.asarray(collection.get_offsets())
                for k, item in enumerate(collection.get_paths()):
                    arrays[f'axis{i}_collection{j}_path{k}'] = item.vertices
        labels.extend(text.get_text() for text in fig.texts)
        if render:
            original(fig, target, **kwargs)
    with patch.object(Figure, 'savefig', save):
        plot(frame, str(path))
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    release, out = args.release.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('!*.png\n!*.csv\n')
    config = json.loads((release/'config.json').read_text())
    frozen = release/'local/frozen_source'
    verify_files([dict(path=n, sha256=d) for n, d in config['frozen_source_hashes'].items()], frozen)
    old_spectral, old_joint = [load_old(frozen/name) for name in ('spectral_entropy.py', 'joint_mi.py')]
    rng = np.random.default_rng(929)
    raw = rng.normal(size=(12000, 2))
    time = np.r_[np.arange(6000)*5000, 40000000+np.arange(6000)*5000]
    audit = []
    kwargs = dict(fs=200., time_us=time, events=[('inside', 15000000), ('gap', 35000000),
                  ('second', 55000000), ('short', 1000000)], windows_sec=(2., 5., 10., 15.),
                  sub_sec=.5, sub_step_sec=.25, n_neighbors=1, n_surrogates=19,
                  random_state=7, front_bandpass=True, pool_events=False)
    signals = {'ch1': raw[:, 0], 'ch2': raw[:, 1]}
    frame = spectral.run_band_event_mi_pipeline(signals, [], audit=audit, **kwargs)
    old_audit = []
    expected = old_spectral.run_band_event_mi_pipeline(signals, [], audit=old_audit, **kwargs)
    pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    if audit != old_audit:
        raise AssertionError('Event acceptance/index audit changed')
    frame.to_csv(out/'per_event.csv', index=False)
    pd.testing.assert_frame_equal(frame, pd.read_csv(out/'per_event.csv', float_precision='round_trip'), check_exact=True)
    write_json(out/'event_audit.json', audit)
    frame['Event_Abbr'] = frame['Event']
    frame['Subject'] = 'synthetic gap probe'
    frame['Gap_Bits'] = frame['Joint_MI_Sum_Bits'] - frame['Joint_MI_KSG_Bits']
    pooled = frame[frame.Event == 'inside'].copy()
    unknown = pooled.copy()
    unknown.attrs = {}
    plots = [
        ('custom_frontend', pooled, lambda m: lambda f, p: m.plot_band_event_mi(f, 'Synthetic custom settings', p), spectral, old_spectral),
        ('subject', frame, lambda m: lambda f, p: m.plot_subject(f, 'Synthetic gap probe', p), joint_mi, old_joint),
        ('cross_subject', frame, lambda m: m.plot_cross_subject, joint_mi, old_joint),
        ('historical_gap', unknown, lambda m: m.plot_gap, joint_mi, old_joint),
        ('historical_event', unknown, lambda m: lambda f, p: m.plot_band_event_mi(f, 'Historical context absent', p), spectral, old_spectral),
    ]
    rows = []
    for name, data, function, current, old in plots:
        capture(function(old), data, out/(name+'_expected.png'), render=False)
        labels = capture(function(current), data, out/(name+'.png'), render=True)
        count, empty = compare_arrays(out/(name+'_expected.npz'), out/(name+'.npz'))
        joined = '\n'.join(labels)
        if 'amplitude' not in joined or 'half-window' not in joined:
            raise AssertionError('Missing measurement/half-window label')
        if ('unknown' if name.startswith('historical') else 'fs=200 Hz') not in joined:
            raise AssertionError('Wrong plot context')
        rows.append(dict(case=name, arrays=count, empty=empty, labels=labels))
    write_json(out/'analysis.json', dict(cases=rows, arrays=sum(r['arrays'] for r in rows),
        numeric_max_error=0., event_rows=len(frame), source_reload='synthetic array; CSV exact round trip',
        acceptance_audit_equal=True, plot_reference_scope='frozen plot functions; same supplied frame'))
    files = [dict(path=str(p.relative_to(out)), sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    files += [dict(path=str(p), sha256=sha(p)) for p in (Path(__file__).resolve(), ROOT/'spectral_entropy.py',
              ROOT/'joint_mi.py', release/'config.json')]
    # Empty scatter paths are checked above; the generic evidence validator requires nonempty arrays.
    write_json(out/'manifest.json', dict(schema_version=1, files=files))
    print(json.dumps(dict(figures=len(rows), arrays=sum(r['arrays'] for r in rows), event_rows=len(frame))))


if __name__ == '__main__':
    main()
