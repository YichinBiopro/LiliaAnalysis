"""Audit M4 quality scope and measure the discovered KSG radius boundary issue."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.compare_mi_methods import brute_mixed_mi, mixed_radius_audit
from tools.freeze_method_profiles import sha, verify_files, write_json
from tools.freeze_mi_profiles import clean


def quality_scope(release):
    rows = []
    population = None
    for name in ('gap', 'quality_0.5', 'quality_0.49', 'quality_nan', 'quality_disabled'):
        meta = json.loads((release/'synthetic/actual'/name/'metadata.json').read_text())
        summary = next(v['frame'] for k, v in meta.items() if k.endswith('_summary'))
        series = next(v['frame'] for k, v in meta.items() if k.endswith('_timeseries'))
        keys = [k for k in summary if k.startswith(('mutual_information', 'surrogate_', 'entropy_')) or k == 'n_samples']
        selected = {k: summary[k] for k in keys}
        if population is None:
            population = selected
        elif selected != population:
            raise AssertionError('Quality mask changed population estimates')
        valid = np.array(series['quality_valid'])
        if name in ('quality_0.5', 'quality_disabled') and not valid.all():
            raise AssertionError('Inclusive .5 / disabled quality policy changed')
        if name in ('quality_0.49', 'quality_nan') and valid.any():
            raise AssertionError('Rejected quality witness became accepted')
        mi = series['joint_mi']
        if name in ('quality_0.49', 'quality_nan') and any(v is not None for v in mi):
            raise AssertionError('Rejected MI was not NaN-masked')
        rows.append(dict(case=name, rows=len(valid), accepted=int(valid.sum()),
                         finite_mi=sum(v is not None for v in mi), population_samples=summary['n_samples'][0],
                         population_bits=summary['mutual_information_bits'][0]))
    return rows


def geometric_nulls(folder, output):
    """Replay the original RNG including preprocessing draws, then compare geometry."""
    meta = json.loads((folder/'metadata.json').read_text())
    contexts = []
    for row in meta['event_rows']:
        durations = sorted({a['Window_Size'] for a in row['audit']})
        for channel in ('ch1', 'ch2'):
            contexts.extend(dict(seed=row['seed'], count=row['surrogates'], duration=d,
                                 channel=channel, k=row['k'], sub_step=row['sub_step']) for d in durations)
    if len(contexts) != len(meta['event_estimates']):
        raise AssertionError('Event capture/context count mismatch')
    arrays, rows, checked = {}, [], 0
    with np.load(folder/'numeric.npz') as data:
        for i, (context, result) in enumerate(zip(contexts, meta['event_estimates'], strict=True)):
            prefix = f'event_call_{i}'
            key = prefix + '_geometric_error_nats'
            if key not in data:
                continue
            checked += 1
            error = float(data[key][0])
            if abs(error) <= 1e-12:
                continue
            x, y = data[prefix+'_features'], data[prefix+'_labels']
            old = data[prefix+'_estimates_nats'] / np.log(2)
            k = result['n_neighbors']
            rng = np.random.RandomState(context['seed'])
            rng.standard_normal(size=x.shape)  # Original preprocessing consumes exactly these draws.
            candidate = [brute_mixed_mi(x, y, k) / np.log(2)]
            for j in range(context['count']):
                labels = rng.permutation(y)
                library = mixed_radius_audit(x, labels, k) / np.log(2)
                np.testing.assert_allclose(library, old[j+1], rtol=1e-12, atol=1e-12)
                candidate.append(brute_mixed_mi(x, labels, k) / np.log(2))
            candidate = np.asarray(candidate)
            arrays[prefix+'_legacy_bits'] = old
            arrays[prefix+'_geometric_bits'] = candidate
            legacy_p = (1 + np.count_nonzero(old[1:] >= old[0])) / len(old)
            geometric_p = (1 + np.count_nonzero(candidate[1:] >= candidate[0])) / len(candidate)
            np.testing.assert_allclose(legacy_p, result['surrogate_p_value'], rtol=0, atol=1e-12)
            rows.append(dict(call=i, **context, effective_k=k, n_pre=result['n_pre'], n_post=result['n_post'],
                             legacy_bits=float(old[0]), geometric_bits=float(candidate[0]),
                             delta_bits=float(candidate[0]-old[0]), legacy_p=legacy_p, geometric_p=geometric_p,
                             null_max_abs_delta_bits=float(np.max(abs(candidate[1:]-old[1:])))))
    if arrays:
        output.mkdir(parents=True)
        np.savez_compressed(output/'radius_comparison.npz', **arrays)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        keys = list(arrays)[:2]
        for key, color in zip(keys, ('tab:blue', 'tab:orange')):
            values = arrays[key]
            label = 'legacy' if 'legacy' in key else 'geometric reference'
            axes[0].hist(values[1:], bins=15, alpha=.5, label=label, color=color)
            axes[0].axvline(values[0], color=color, linestyle='--')
        axes[0].set(title=folder.name + ': first differing call, null + observed', xlabel='bits', ylabel='surrogate count')
        axes[0].legend(fontsize=8)
        axes[1].plot([r['call'] for r in rows], [r['legacy_p'] for r in rows], 'o-', label='legacy')
        axes[1].plot([r['call'] for r in rows], [r['geometric_p'] for r in rows], 'x--', label='geometric reference')
        axes[1].set(title='Same shuffled labels; no scientific significance claim', xlabel='capture call index', ylabel='p (add-one)')
        axes[1].legend(fontsize=8)
        fig.savefig(output/'radius_comparison.png', dpi=130)
        plt.close(fig)
    return dict(case=folder.name, geometric_observed_checks=checked, disagreements=rows)


def event_review_plot(folder, output):
    """Explicit measurement labels; original product plots stay untouched."""
    import matplotlib.pyplot as plt
    meta = json.loads((folder/'metadata.json').read_text())
    row = meta['event_rows'][0]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ch in ('ch1', 'ch2'):
        data = [r for r in row['results'] if r['Channel'] == ch]
        for key, style, label in [('Joint_MI_KSG_Bits', 'o-', 'joint'), ('Joint_MI_Sum_Bits', 'x--', 'sum per band')]:
            axes[0].plot([r['Window_Size'] for r in data], [r[key] for r in data], style, label=ch+' '+label)
        axes[1].plot([r['Window_Size'] for r in data], [r['N_Pre'] for r in data], 'o-', label=ch+' pre')
        axes[1].plot([r['Window_Size'] for r in data], [r['N_Post'] for r in data], 'x--', label=ch+' post')
    axes[0].set(xlabel='pre/post duration (s)', ylabel='MI (bits)', title='Amplitude-envelope features vs. event label')
    axes[1].set(xlabel='pre/post duration (s)', ylabel='sub-epoch count', title='Overlapping rows are not independent observations')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(folder.name+' | full segment band filters + Hilbert; extra 0.5–45 Hz front filter OFF\n'
                 'sub-epoch 1 s / step 0.5 s; k requested 3; seed 0; 200 label shuffles; quality disabled', fontsize=10)
    fig.savefig(output/'event_review.png', dpi=130)
    plt.close(fig)


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
    rows = quality_scope(release)
    config = json.loads((release/'config.json').read_text())
    events = []
    for case in config['cases']:
        if case.get('event'):
            scope = 'local' if case['real'] else 'synthetic'
            events.append(geometric_nulls(release/scope/'actual'/case['name'], out/scope/case['name']))
            event_review_plot(release/scope/'actual'/case['name'], out/scope/case['name'])
    control = json.loads((release/'synthetic/actual/controls/metadata.json').read_text())['rows']
    findings = dict(quality_scope=rows, event_geometry=events, controls=len(control),
                    geometry_tolerance=1e-12, geometry_equivalent=False,
                    product_changed=False, geometric_reference_adopted=False,
                    interpretation='Reference disagreement is a measured limitation, not a passed equivalence check.')
    write_json(out/'findings.json', clean(findings))
    files = [dict(path=str(p.relative_to(out)), sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    files += [dict(path=str(p), sha256=sha(p)) for p in (
        release/'manifest_local.json', release/'manifest_repository.json', Path(__file__).resolve())]
    write_json(out/'manifest_local.json', dict(schema_version=1, files=files))
    write_json(out/'manifest_repository.json', dict(schema_version=1, files=[r for r in files if not r['path'].startswith('local/')]))
    print(json.dumps(dict(quality_witnesses=len(rows), event_cases=len(events),
                          geometric_disagreements=sum(len(r['disagreements']) for r in events))))


if __name__ == '__main__':
    main()
