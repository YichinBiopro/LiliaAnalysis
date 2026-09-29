"""Independent M4 references and controlled MI sensitivity measurements."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from scipy.special import digamma


def check_histogram(result):
    """Check returned PMF/MI algebra without calling the product entropy helper."""
    p = np.asarray(result['pxy'])
    if not p.any():
        if any(result[k] != 0 for k in ('mutual_information', 'mutual_information_mm', 'mutual_information_norm')):
            raise AssertionError('Degenerate histogram must retain its zero sentinel')
        return 'degenerate_zero_sentinel'
    np.testing.assert_allclose(p.sum(), 1., rtol=0, atol=1e-12)
    px, py = p.sum(axis=1), p.sum(axis=0)
    np.testing.assert_allclose(result['px'], px, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['py'], py, rtol=0, atol=1e-12)
    expected = px[:, None] * py[None, :]
    mi = max(0., float(np.sum(p[p > 0] * np.log2(p[p > 0] / expected[p > 0]))))
    correction = (np.count_nonzero(px) + np.count_nonzero(py) - np.count_nonzero(p) - 1)
    mm = max(0., mi + correction / (2 * result['n_samples'] * np.log(2)))
    h = min(-np.sum(a[a > 0] * np.log2(a[a > 0])) for a in (px, py))
    np.testing.assert_allclose([result['mutual_information'], result['mutual_information_mm'],
                               result['mutual_information_norm']], [mi, mm, mi / h if h > 1e-12 else 0.],
                              rtol=1e-12, atol=1e-12)
    return 'normalized_pmf'


def brute_mixed_mi(features, labels, k=3):
    """Small-sample Ross reference using a dense Euclidean distance matrix."""
    x, y = np.asarray(features, dtype=float), np.asarray(labels)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or k < 1:
        raise ValueError('Finite aligned features/labels and positive k required')
    counts = np.array([np.sum(y == label) for label in y])
    keep = counts > 1
    x, y, counts = x[keep], y[keep], counts[keep]
    if not len(x):
        return 0.
    distances = np.sqrt(np.sum((x[:, None] - x[None, :]) ** 2, axis=2))
    terms = []
    for i in range(len(x)):
        neighbors = distances[i, (y == y[i]) & (np.arange(len(x)) != i)]
        ki = min(k, len(neighbors))
        radius = np.nextafter(np.sort(neighbors)[ki - 1], 0.)
        m = np.count_nonzero(distances[i] <= radius)
        terms.append(digamma(ki) - digamma(counts[i]) - digamma(m))
    return max(0., float(digamma(len(x)) + np.mean(terms)))


def check_null(observed, null, result, *, event=False):
    null = np.asarray(null, dtype=float)
    if not len(null):
        return
    mean = null.mean()
    std = null.std(ddof=1 if event and len(null) > 1 else 0)
    p = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
    z = (observed - mean) / std if std > (0. if event else 1e-12) else np.nan
    keys = ('surrogate_mean_bits', 'surrogate_std_bits', 'surrogate_p_value', 'surrogate_z') if event else (
        'surrogate_mean', 'surrogate_std', 'p_value', 'z')
    np.testing.assert_allclose([result[k] for k in keys], [mean, std, p, z], rtol=1e-12, atol=1e-12)


def mixed_radius_audit(features, labels, k, *, algorithm='auto'):
    """Separate floating-radius library behavior from the dense geometric oracle."""
    from sklearn.neighbors import NearestNeighbors, KDTree
    x, y = np.asarray(features), np.asarray(labels)
    radius = np.zeros(len(x))
    counts = np.array([np.sum(y == label) for label in y])
    ks = np.minimum(k, counts - 1)
    for label in np.unique(y):
        mask = y == label
        if mask.sum() > 1:
            distances = NearestNeighbors(n_neighbors=int(ks[mask][0]), algorithm=algorithm).fit(x[mask]).kneighbors()[0]
            radius[mask] = np.nextafter(distances[:, -1], 0.)
    keep = counts > 1
    if not keep.any():
        return 0.
    neighbors = KDTree(x[keep]).query_radius(x[keep], radius[keep], count_only=True)
    return max(0., float(digamma(keep.sum()) + np.mean(digamma(ks[keep]))
                         - np.mean(digamma(counts[keep])) - np.mean(digamma(neighbors))))


def circular_reference(module, x, y, groups, *, bins, binning, count, seed):
    """Rebuild shifts independently; repeated labels still denote separate runs."""
    x, y = np.asarray(x), np.asarray(y)
    ids = np.zeros(len(x), dtype=int) if groups is None else np.asarray(groups)
    if x.shape != y.shape or ids.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValueError('Finite aligned pairs and groups required')
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(x)]
    if count <= 0 or np.any(np.diff(bounds) < 16):
        return np.empty(0)
    rng = np.random.default_rng(seed)
    offsets = [rng.integers(max(1, (b-a)//100), b-a, size=count) for a, b in zip(bounds[:-1], bounds[1:])]
    values = []
    for j in range(count):
        shifted = np.concatenate([np.roll(y[a:b], shifts[j]) for a, b, shifts in zip(bounds[:-1], bounds[1:], offsets)])
        result = module.compute_joint_probability(x, shifted, bins=bins, binning=binning)
        check_histogram(result)
        values.append(result['mutual_information'])
    return np.asarray(values)


@contextmanager
def capture_event_estimates(module, arrays, prefix):
    """Observe actual preprocessed features and each label shuffle, unmodified."""
    original = module.compute_mi_cd_multivariate
    calls = []
    def capture(c, d, n_neighbors=3):
        value = original(c, d, n_neighbors=n_neighbors)
        i = len(calls)
        if i == 0:
            arrays[prefix + '_features'] = np.asarray(c).copy()
            arrays[prefix + '_labels'] = np.asarray(d).copy()
            if len(c) <= 256:
                reference = brute_mixed_mi(c, d, n_neighbors)
                arrays[prefix + '_geometric_reference_nats'] = np.array([reference])
                arrays[prefix + '_geometric_error_nats'] = np.array([value - reference])
                # Library-radius formula must match; independent geometric disagreement
                # remains a measured finding, never hidden by a wider tolerance.
                np.testing.assert_allclose(value, mixed_radius_audit(c, d, n_neighbors), rtol=1e-12, atol=1e-12)
        calls.append(value)
        return value
    with patch.object(module, 'compute_mi_cd_multivariate', side_effect=capture):
        yield calls
    arrays[prefix + '_estimates_nats'] = np.asarray(calls)


def event_measure(module, envelopes, intervals, arrays, prefix, **kwargs):
    with capture_event_estimates(module, arrays, prefix) as calls:
        result = module.compute_band_event_joint_mi(envelopes, [], 1, event_windows=intervals, **kwargs)
    if result is not None:
        check_null(result['joint_mi_bits'], np.asarray(calls[1:]) / np.log(2), result, event=True)
    return result


def controls(module, save):
    """One-factor grids; every result and relevant input passes through save."""
    from types import SimpleNamespace
    rows = []
    def record(name, result, **settings):
        save(name, result)
        rows.append(dict(name=name, **settings, **{k: v for k, v in result.items() if np.isscalar(v)}))
    for name, x, y, target in (
        ('binary_independent', np.tile([0., 0., 1., 1.], 64), np.tile([0., 1., 0., 1.], 64), 0.),
        ('binary_identical', np.tile([0., 1.], 128), np.tile([0., 1.], 128), 1.),
        ('constant', np.ones(256), np.ones(256), 0.)):
        for binning in ('uniform', 'quantile'):
            result = module.compute_joint_probability(x, y, bins=8, binning=binning)
            check_histogram(result)
            if binning == 'uniform':
                np.testing.assert_allclose(result['mutual_information'], target, rtol=0, atol=1e-12)
            record(name + '_' + binning, result, family='known', binning=binning, theoretical_bits=target)
    rng = np.random.default_rng(2209)
    x, noise = rng.normal(size=(2, 2048))
    independent_ar = np.zeros((2048, 2))
    innovations = rng.normal(size=(2048, 2))
    for i in range(1, 2048):
        independent_ar[i] = .95 * independent_ar[i-1] + innovations[i]
    groups = np.repeat([0, 1], 1024)
    pairs = dict(independent=(x, noise), correlated=(x, .8*x + .6*noise),
                 independent_ar=(independent_ar[:, 0], independent_ar[:, 1]),
                 segment_means=(x + groups*8, noise + groups*8))
    for signal, (left, right) in pairs.items():
        save(signal + '_input', dict(x=left, y=right, groups=groups))
        for n in (128, 512, 2048):
            for bins in (8, 16, 32):
                for binning in ('uniform', 'quantile'):
                    result = module.compute_joint_probability(left[:n], right[:n], bins=bins, binning=binning)
                    check_histogram(result)
                    record(f'{signal}_{n}_{bins}_{binning}', result, family='histogram', signal=signal,
                           sample_n=n, bins=bins, binning=binning)
        for seed, count in ((0, 19), (0, 99), (0, 200), (7, 200), (42, 200)):
            ids = groups if signal == 'segment_means' else np.zeros(2048, dtype=int)
            result = module.compute_joint_mi_significance(left, right, bins=16, binning='quantile',
                n_surrogates=count, seed=seed, segment_ids=ids)
            expected = circular_reference(module, left, right, ids, bins=16, binning='quantile', count=count, seed=seed)
            np.testing.assert_allclose(result['surrogates'], expected, rtol=1e-12, atol=1e-12)
            check_null(result['mutual_information'], expected, result)
            record(f'{signal}_null_{seed}_{count}', result, family='circular', signal=signal, seed=seed, count=count)
        if signal == 'segment_means':
            result = module.compute_joint_mi_significance(left, right, n_surrogates=200, binning='quantile')
            record('segment_means_global_shift', result, family='wrong_null_witness', signal=signal)
    for name, ids, count in [('disabled', np.zeros(32), 0), ('short_run', np.r_[np.zeros(17), np.ones(15)], 200)]:
        result = module.compute_joint_mi_significance(x[:32], noise[:32], n_surrogates=count, segment_ids=ids)
        if result['n_surrogates'] != 0 or len(result['surrogates']):
            raise AssertionError('Expected disabled null')
        record(name, result, family='null_boundary')
    for signal in ('independent', 'separated', 'redundant'):
        features = rng.normal(size=(256, 3))
        if signal in ('separated', 'redundant'):
            features[128:] += 10
        if signal == 'redundant':
            features = np.repeat(features[:, :1], 3, axis=1)
        save('event_' + signal + '_input', dict(features=features))
        settings = [(n, k, 0, 200) for n in (16, 64, 128) for k in (1, 3, 5)]
        settings += [(64, 3, seed, count) for seed, count in ((7, 200), (42, 200), (0, 0), (0, 1), (0, 19), (0, 99))]
        if signal == 'separated':
            settings.append((64, 3, 0, 500))
        for n, k, seed, count in settings:
            xx = np.r_[features[:n], features[128:128+n]]
            env = {name: xx[:, i] for i, (name, _) in enumerate(module.BAND_DEFINITIONS)}
            arrays = {}
            label = f'event_{signal}_{n}_{k}_{seed}_{count}'
            result = event_measure(module, env, [SimpleNamespace(pre=slice(0, n), post=slice(n, 2*n))],
                                   arrays, label, fs=1., sub_sec=1., sub_step_sec=1.,
                                   n_neighbors=k, n_surrogates=count, random_state=seed)
            if result is None:
                raise AssertionError('Unexpected empty event control')
            save(label + '_capture', arrays)
            record(label, result, family='event', signal=signal, sample_n=n, k=k, seed=seed, count=count)
    for name, value in [('too_few', 1.), ('nonfinite', np.nan)]:
        env = {name: np.full(2, value) for name, _ in module.BAND_DEFINITIONS}
        result = event_measure(module, env, [SimpleNamespace(pre=slice(0, 1), post=slice(1, 2))],
                               {}, name, fs=1., sub_sec=1., sub_step_sec=1., n_surrogates=200)
        if result is not None:
            raise AssertionError('Expected unavailable event estimate')
        record('event_' + name, dict(available=False), family='event_boundary', reason=name)
    return rows


def plot_controls(rows, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for bins in (8, 16, 32):
        subset = [r for r in rows if r['family'] == 'histogram' and r['signal'] == 'independent'
                  and r['binning'] == 'quantile' and r['bins'] == bins]
        axes[0, 0].plot([r['sample_n'] for r in subset], [r['mutual_information'] for r in subset], '.-', label=f'{bins} bins')
    axes[0, 0].set(title='IID independent: finite-sample plug-in MI', xlabel='paired samples', ylabel='bits', xscale='log')
    subset = [r for r in rows if r['family'] in ('circular', 'wrong_null_witness') and r['signal'] == 'segment_means'
              and (r['family'] == 'wrong_null_witness' or (r['seed'] == 0 and r['count'] == 200))]
    axes[0, 1].bar(['within segment', 'global shift'], [r['surrogate_mean'] for r in subset])
    axes[0, 1].axhline(subset[0]['mutual_information'], color='red', label='observed')
    axes[0, 1].set(title='Common segment means: different null hypotheses', ylabel='bits')
    for signal in ('independent', 'separated', 'redundant'):
        subset = [r for r in rows if r['family'] == 'event' and r['signal'] == signal
                  and r['sample_n'] == 64 and r['seed'] == 0 and r['count'] == 200]
        axes[1, 0].plot([r['k'] for r in subset], [r['joint_mi_bits'] for r in subset], '.-', label=signal)
    axes[1, 0].set(title='Mixed continuous/label estimator: 64 per class', xlabel='k', ylabel='joint bits')
    subset = [r for r in rows if r['family'] == 'event' and r['signal'] == 'redundant'
              and r['sample_n'] == 64 and r['seed'] == 0 and r['count'] == 200]
    for key in ('sum_mi_bits', 'joint_mi_bits'):
        axes[1, 1].plot([r['k'] for r in subset], [r[key] for r in subset], '.-', label=key)
    axes[1, 1].set(title='Repeated feature: sum and joint are different quantities', xlabel='k', ylabel='bits')
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
