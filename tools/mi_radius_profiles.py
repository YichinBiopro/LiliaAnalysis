"""Named calibration-only mixed-MI profiles; product defaults are unchanged."""
import numpy as np
from scipy.special import digamma

LEGACY_PROFILE = 'legacy_auto_kdtree'
CANDIDATE_PROFILE = 'ross_euclidean_direct_v1'


def direct_mixed_mi(features, labels, k=3):
    """Row-wise Euclidean Ross convention, in nats, with one distance kernel.

    Singleton labels are removed. The inward nextafter radius includes self;
    at radius zero all exact duplicates remain included. No extra jitter is
    added here: event preprocessing and its RNG belong to the caller.
    This O(N²D) calibration candidate is deliberately not a product default.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
    if x.ndim == 1:
        x = x[:, None]
    if (x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError('Finite aligned features/labels and positive integer k required')
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    keep = counts[inverse] > 1
    x, inverse = x[keep], inverse[keep]
    if not len(x):
        return 0.
    ks = np.minimum(k, counts[inverse] - 1)
    neighbors = np.empty(len(x), dtype=int)
    for i, point in enumerate(x):
        distances = np.sqrt(np.sum((x - point) ** 2, axis=1))
        same = inverse == inverse[i]
        same[i] = False
        radius = np.nextafter(np.partition(distances[same], ks[i] - 1)[ks[i] - 1], 0.)
        neighbors[i] = np.count_nonzero(distances <= radius)
    # Average the per-observation terms in a fixed order. This also preserves
    # exact equalities when the add-one null comparison uses >= without an eps.
    terms = digamma(ks) - digamma(counts[inverse]) - digamma(neighbors)
    return max(0., float(digamma(len(x)) + np.mean(terms)))


def null_summary(values):
    """Observed followed by the complete event null, in the caller's units."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A finite observed estimate and optional null are required')
    observed, null = values[0], values[1:]
    if not len(null):
        return dict(observed=float(observed), count=0, mean=np.nan, std=np.nan, p=np.nan, z=np.nan)
    mean = float(null.mean())
    std = float(null.std(ddof=1)) if len(null) > 1 else 0.
    return dict(observed=float(observed), count=len(null), mean=mean, std=std,
                p=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
                z=float((observed - mean) / std) if std > 0 else np.nan)
