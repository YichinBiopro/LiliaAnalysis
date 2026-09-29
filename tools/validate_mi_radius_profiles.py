"""M4-R1: replay every captured event null, including unchanged observations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.compare_mi_methods import brute_mixed_mi, check_null, mixed_radius_audit
from tools.freeze_method_profiles import sha, verify_files, write_json
from tools.freeze_mi_profiles import clean
from tools.mi_radius_profiles import CANDIDATE_PROFILE, LEGACY_PROFILE, direct_mixed_mi, null_summary

TOLERANCE = 1e-12


def replay(x, y, legacy_nats, result, *, seed, count):
    """Fail on missing nulls, wrong RNG replay, or candidate/reference mismatch."""
    if len(legacy_nats) != count + 1:
        raise AssertionError('Incomplete captured observed/null sequence')
    k = result['n_neighbors']
    rng = np.random.RandomState(seed)
    rng.standard_normal(size=x.shape)
    labels = np.asarray([y] + [rng.permutation(y) for _ in range(count)])
    candidate = np.asarray([direct_mixed_mi(x, row, k) for row in labels])
    reference = np.asarray([brute_mixed_mi(x, row, k) for row in labels])
    library = np.asarray([mixed_radius_audit(x, row, k) for row in labels])
    np.testing.assert_allclose(library, legacy_nats, rtol=TOLERANCE, atol=TOLERANCE)
    np.testing.assert_allclose(candidate, reference, rtol=TOLERANCE, atol=TOLERANCE)
    legacy, candidate, reference = [values / np.log(2) for values in (legacy_nats, candidate, reference)]
    np.testing.assert_allclose(candidate, reference, rtol=TOLERANCE, atol=TOLERANCE)
    normalized = {key: np.nan if value is None else value for key, value in result.items()}
    np.testing.assert_allclose(legacy[0], normalized['joint_mi_bits'], rtol=TOLERANCE, atol=TOLERANCE)
    check_null(legacy[0], legacy[1:], normalized, event=True)
    summaries = {name: null_summary(values) for name, values in (
        ('legacy', legacy), ('candidate', candidate), ('reference', reference))}
    np.testing.assert_allclose(list(summaries['candidate'].values()), list(summaries['reference'].values()),
                               rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True)
    differences = ~np.isclose(legacy, reference, rtol=TOLERANCE, atol=TOLERANCE)
    row = dict(seed=seed, count=count, effective_k=k, samples=len(x),
               observed_differs=bool(differences[0]), null_differences=int(differences[1:].sum()),
               candidate_max_error_bits=float(np.max(np.abs(candidate-reference))),
               legacy_max_delta_bits=float(np.max(np.abs(legacy-reference))), **summaries)
    arrays = dict(labels=labels, legacy_bits=legacy, candidate_bits=candidate, reference_bits=reference)
    return row, arrays


def event_contexts(meta):
    contexts = []
    for row in meta['event_rows']:
        durations = sorted({a['Window_Size'] for a in row['audit']})
        for channel in ('ch1', 'ch2'):
            contexts.extend(dict(seed=row['seed'], count=row['surrogates'], duration=d,
                                 channel=channel) for d in durations)
    if len(contexts) != len(meta['event_estimates']):
        raise AssertionError('Event capture/context count mismatch')
    return contexts


def validate_folder(folder, output, *, controls=False):
    meta = json.loads((folder/'metadata.json').read_text())
    tasks = []
    unavailable = []
    if controls:
        for row in meta['rows']:
            if row['family'] == 'event':
                prefix = row['name'] + '_capture__' + row['name']
                tasks.append((prefix, row, dict(seed=row['seed'], count=row['count'])))
    else:
        for i, (context, result) in enumerate(zip(event_contexts(meta), meta['event_estimates'], strict=True)):
            tasks.append((f'event_call_{i}', result, context))
    rows, arrays = [], {}
    with np.load(folder/'numeric.npz') as data:
        captured = {key.removesuffix('_features') for key in data if key.endswith('_features')}
        expected = {prefix for prefix, result, _ in tasks if result is not None}
        # Controls also contain raw pre-preprocessing input features.
        if controls:
            captured = {key for key in captured if '_capture__' in key}
        if captured != expected:
            raise AssertionError('Captured feature coverage differs from scheduled replay')
        for prefix, result, context in tasks:
            if result is None:
                if data[prefix+'_estimates_nats'].size:
                    raise AssertionError('Unavailable event unexpectedly has estimates')
                unavailable.append(dict(call=prefix, **context))
                continue
            row, values = replay(data[prefix+'_features'], data[prefix+'_labels'],
                data[prefix+'_estimates_nats'], result, seed=context['seed'], count=context['count'])
            rows.append(dict(call=prefix, **{**context, **row}))
            arrays.update({prefix+'_'+key: value for key, value in values.items()})
    if not rows:
        raise AssertionError('No estimates were validated')
    output.mkdir(parents=True)
    np.savez_compressed(output/'comparison.npz', **arrays)
    write_json(output/'analysis.json', clean(dict(calls=rows, unavailable=unavailable)))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, style in [('legacy', 'o-'), ('candidate', 'x--')]:
        axes[0].plot([r[label]['p'] for r in rows], style, label=label, markersize=3)
    axes[0].set(xlabel='Every captured call (including unchanged observed)', ylabel='Add-one p',
                title=folder.name + ': same shuffled labels')
    axes[0].legend()
    axes[1].bar(range(len(rows)), [r['null_differences'] for r in rows])
    axes[1].set(xlabel='Capture call', ylabel='Number of differing null estimates',
                title='Legacy vs direct geometry; tolerance 1e-12')
    fig.suptitle('Calibration only; no claim of event effects or exchangeability')
    fig.savefig(output/'comparison.png', dpi=130)
    plt.close(fig)
    return dict(case=folder.name, calls=rows, unavailable=unavailable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    release, out = args.release.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('local/\n!*.png\n')
    manifest = json.loads((release/'manifest_local.json').read_text())
    verify_files(manifest['files'], release)
    config = json.loads((release/'config.json').read_text())
    cases = []
    for case in config['cases']:
        if case.get('event'):
            scope = 'local' if case['real'] else 'synthetic'
            cases.append(validate_folder(release/scope/'actual'/case['name'], out/scope/case['name']))
            print('PASS complete null: ' + case['name'], flush=True)
    cases.append(validate_folder(release/'synthetic/actual/controls', out/'synthetic/controls', controls=True))
    rows = [row for case in cases for row in case['calls']]
    summary = dict(legacy_profile=LEGACY_PROFILE, candidate_profile=CANDIDATE_PROFILE,
        candidate_adopted=False, product_default_changed=False, tolerance=TOLERANCE,
        cases=cases, calls=len(rows), unavailable=sum(len(c['unavailable']) for c in cases),
        null_estimates=sum(r['count'] for r in rows),
        candidate_max_error_bits=max(r['candidate_max_error_bits'] for r in rows),
        observed_disagreements=sum(r['observed_differs'] for r in rows),
        calls_with_null_disagreements=sum(r['null_differences'] > 0 for r in rows),
        unchanged_observed_with_null_disagreements=sum(not r['observed_differs'] and r['null_differences'] > 0 for r in rows),
        null_disagreements=sum(r['null_differences'] for r in rows),
        legacy_max_delta_bits=max(r['legacy_max_delta_bits'] for r in rows))
    write_json(out/'analysis.json', clean(summary))
    inputs = [release/'manifest_local.json', release/'manifest_repository.json',
              ROOT/'docs/refactor/MI_COMPARISON_CONTRACT.md', Path(__file__).resolve(),
              ROOT/'tools/mi_radius_profiles.py', ROOT/'tools/compare_mi_methods.py']
    files = [dict(path=str(p.relative_to(out)), sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    files += [dict(path=str(p), sha256=sha(p)) for p in inputs]
    write_json(out/'manifest_local.json', dict(schema_version=1, files=files))
    write_json(out/'manifest_repository.json', dict(schema_version=1,
        files=[r for r in files if not r['path'].startswith('local/')]))
    print(json.dumps({key: value for key, value in summary.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
