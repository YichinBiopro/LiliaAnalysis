"""M3 measurement-only comparisons; legacy product defaults remain unchanged."""
import numpy as np
import pandas as pd


def smooth(values, good, groups, win=5, *, segmented=False):
    values, good, groups = np.asarray(values, float), np.asarray(good, bool), np.asarray(groups)
    if values.ndim != 1 or good.shape != values.shape or groups.shape != values.shape:
        raise ValueError('Aligned one-dimensional values/mask/segments required')
    if isinstance(win, bool) or not isinstance(win, int) or win < 1:
        raise ValueError('Positive integer smoothing window required')
    masked = np.where(good, values, np.nan)
    output = masked.copy()
    bounds = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1])+1, len(values)] if segmented else [0, len(values)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        output[a:b] = pd.Series(masked[a:b]).rolling(win, center=True, min_periods=1).median().to_numpy()
    return np.where(good, output, np.nan)


def compare_smoothing(values, good, groups, win=5):
    old = smooth(values, good, groups, win)
    candidate = smooth(values, good, groups, win, segmented=True)
    np.testing.assert_array_equal(np.isnan(old), np.isnan(candidate))
    delta = candidate-old
    changed = np.flatnonzero(np.isfinite(delta) & (delta != 0))
    return old, candidate, dict(changed_indices=changed.tolist(), changed=len(changed),
        max_abs_delta=float(np.max(np.abs(delta[changed]))) if len(changed) else 0.,
        accepted_indices=np.flatnonzero(good).tolist(), rejected_indices=np.flatnonzero(~good).tolist(),
        acceptance_changes=0, nan_position_changes=0)


def reference_power(values, target=60., fs=500.):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or fs <= 0 or not 0 < target < fs/2:
        raise ValueError('Finite vector and interior target frequency required')
    if not len(x):
        return 0.
    window = np.hanning(len(x)) if len(x) > 1 else np.ones(1)
    return float(abs(np.sum((x-x.mean())*window*np.exp(-2j*np.pi*target*np.arange(len(x))/fs)))**2)


def power_measures(values, power, fs=500., target=60.):
    n = len(values)
    if n < 3:
        raise ValueError('Normalization requires at least three samples')
    reference = reference_power(values, target, fs)
    np.testing.assert_allclose(power, reference, rtol=1e-8, atol=1e-8)
    window = np.hanning(n)
    factors = dict(legacy=1., per_n_squared=1./n**2,
                   coherent_mean_square=2./window.sum()**2,
                   one_sided_density=2./(fs*np.sum(window**2)))
    linear = {k: float(power*v) for k, v in factors.items()}
    db = {k: float(10*np.log10(max(v, 1e-12))) for k, v in linear.items()}
    offsets = {k: float(10*np.log10(v)) for k, v in factors.items()}
    if min(linear.values()) > 1e-12:
        for k in db:
            np.testing.assert_allclose(db[k]-db['legacy'], offsets[k], rtol=0., atol=1e-10)
    return dict(samples=n, frequency=target, reference_power=reference,
        reference_abs_error=float(abs(power-reference)), linear=linear, db=db, db_offsets=offsets,
        reference_error_fraction_of_tolerance=float(abs(power-reference)/(1e-8+1e-8*abs(reference))),
        unchanged_number_67p15_hits={k: bool(abs(v-67.15) <= .5) for k,v in db.items()},
        transformed_67p15_centers={k:67.15+v for k,v in offsets.items()},
        floor_active={k:bool(v <= 1e-12) for k,v in linear.items()})


def controls(out):
    from tools.freeze_method_profiles import write_json
    from lilia.goertzel import goertzel_power
    from lilia.io import bandpass_filter
    from lilia.quality_policy import valid_goertzel_rows
    import matplotlib.pyplot as plt
    # Known boundary witness, including rejected and non-finite policy rows.
    witness = pd.DataFrame(dict(quality=[.5, .500001, np.nan, .7, .7],
        quality_final=[.5, .500001, np.nan, 0., .7],
        artifact_hard_clip=[0, 0, 0, 1, 0], goertzel_db=[0, 1, 2, 3, np.nan], time_s=np.arange(5)))
    keep = valid_goertzel_rows(witness)
    np.testing.assert_array_equal(keep, [False, True, False, False, False])
    old, new, smoothing = compare_smoothing(np.array([0., 0., 100., 100.]),
        np.ones(4, bool), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(old, [0., 50., 50., 100.])
    np.testing.assert_array_equal(new, [0., 0., 100., 100.])
    rows, arrays = [], dict(smoothing_old=old, smoothing_segment=new, policy_keep=keep)
    for duration in (.5, 1., 5.):
        t = np.arange(int(duration*500))/500
        signals = dict(tone60=2*np.sin(2*np.pi*60*t), off_grid=2*np.sin(2*np.pi*60.3*t),
            mixture=np.sin(2*np.pi*10*t)+2*np.sin(2*np.pi*60*t),
            noise=np.random.default_rng(1923).normal(size=len(t)), constant=np.full(len(t), 3.))
        for name, values in signals.items():
            # Only the BP branch performs the product's float32 conversion.
            filtered = bandpass_filter(values[:, None], fs=500, lo=.5, hi=45)[:, 0]
            for stage, x in [('raw', values), ('filtered', filtered)]:
                power = goertzel_power(x, 60., 500.)
                row = power_measures(x, power)
                row.update(duration=duration, signal=name, stage=stage)
                if name == 'tone60' and stage == 'raw':
                    np.testing.assert_allclose(row['linear']['coherent_mean_square'], 2., rtol=1e-3, atol=1e-6)
                if name == 'constant' and stage == 'raw':
                    np.testing.assert_equal(power, 0.)
                rows.append(row)
                arrays[f'{duration}_{name}_{stage}'] = x
    # Deliberately choose an amplitude that hits the legacy sampler example.
    t = np.arange(2500)/500
    wave = np.sin(2*np.pi*60*t)
    wave *= np.sqrt(10**(67.15/10)/goertzel_power(wave, 60., 500.))
    threshold = power_measures(wave, goertzel_power(wave, 60., 500.))
    if not threshold['unchanged_number_67p15_hits']['legacy'] or any(
            value for key,value in threshold['unchanged_number_67p15_hits'].items() if key != 'legacy'):
        raise AssertionError('Normalization threshold witness missing')
    arrays['threshold_input'] = wave
    np.savez_compressed(out/'controls.npz', **arrays)
    summary = dict(rows=rows, controls=len(rows), smoothing_witness=smoothing,
                   quality_boundary_keep=keep.tolist(), threshold_witness=threshold,
                   max_reference_abs_error=max(r['reference_abs_error'] for r in rows), product_changes=False)
    write_json(out/'controls.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot([0, 1, 4, 5], old, 'o--', label='legacy rolling')
    axes[0].plot([0, 1, 4, 5], new, 's-', label='segment rolling (comparison)')
    axes[0].axvspan(1, 4, color='.9'); axes[0].set(title='Known boundary witness', xlabel='Window position / gap', ylabel='dB')
    for key in ('legacy', 'per_n_squared', 'coherent_mean_square', 'one_sided_density'):
        axes[1].plot([.5, 1, 5], [r['db'][key] for r in rows if r['signal']=='tone60' and r['stage']=='raw'], 'o-', label=key)
    axes[1].set(title='Same amplitude: distinct units', xlabel='Window seconds', ylabel='10log10(value in named units)')
    for stage in ('raw', 'filtered'):
        axes[2].plot([.5, 1, 5], [r['db']['legacy'] for r in rows if r['signal']=='tone60' and r['stage']==stage], 'o-', label=stage)
    axes[2].set(title='60 Hz tone / 0.5–45 Hz BP', xlabel='Window seconds', ylabel='Legacy dB')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(out/'controls.png', dpi=120); plt.close(fig)
    return summary


def compare_case(folder, source, case):
    from tools.freeze_method_profiles import write_json
    from lilia.io import load_merged_csv, bandpass_filter
    from lilia.goertzel import goertzel_power
    from lilia.quality_policy import valid_goertzel_rows
    from lilia.windowing import plot_breaks
    import matplotlib.pyplot as plt
    frame = pd.read_csv(folder/'metrics.csv', float_precision='round_trip')
    time, raw = load_merged_csv(source)
    filtered = bandpass_filter(raw, fs=500, lo=.5, hi=45, time_us=time)
    with np.load(folder/'numeric.npz') as data:
        groups = data['segment_ids']
        captured_smoothing = {key:data['legacy_smooth_'+key+'_True_0.5'] for key in ('goertzel_power','goertzel_db')}
    good = valid_goertzel_rows(frame)
    arrays, summary = {}, dict(rows=len(frame), accepted=int(good.sum()), smoothing={}, measures=[])
    for key in ('goertzel_power', 'goertzel_db'):
        old, new, change = compare_smoothing(frame[key].to_numpy(), good, groups)
        np.testing.assert_array_equal(old, captured_smoothing[key])
        arrays[key+'_legacy'], arrays[key+'_segment'] = old, new
        summary['smoothing'][key] = change
    arrays.update(good=good, segment_ids=groups, time_s=frame.time_s.to_numpy())
    for i, row in frame.iterrows():
        a,b = int(row.window_start_idx), int(row.window_end_idx)
        values = {}
        for stage,x in [('raw',raw[a:b,case['ch']-1]), ('filtered',filtered[a:b,case['ch']-1])]:
            power = goertzel_power(x, 60., 500.)
            if stage=='filtered':
                np.testing.assert_equal(power, row.goertzel_power)
            values[stage] = power_measures(x, power)
        summary['measures'].append(dict(window_index=int(i), start_idx=a, end_idx=b, segment_id=int(groups[i]),
            accepted=bool(good[i]), raw=values['raw'], filtered=values['filtered'],
            filtered_minus_raw_db=values['filtered']['db']['legacy']-values['raw']['db']['legacy']))
    # Empty arrays remain valid explicit empty cases; saved shapes are checked independently.
    np.savez_compressed(folder/'comparison.npz', **arrays)
    write_json(folder/'methods.json', summary)
    if case.get('render'):
        fig, axes = plt.subplots(2,1,figsize=(12,6),constrained_layout=True,sharex=True)
        for ax,key in zip(axes,('goertzel_db','goertzel_power')):
            for mode,style in [('legacy','o--'),('segment','s-')]:
                ax.plot(*plot_breaks(arrays['time_s'], arrays[key+'_'+mode],groups),style,label=mode,ms=4)
            ax.set(ylabel=key, title=case['name']+' / same quality mask'); ax.legend(); ax.grid(alpha=.2)
        axes[-1].set_xlabel('Actual elapsed seconds; gaps blank')
        fig.savefig(folder/'smoothing_comparison.png',dpi=120);plt.close(fig)
    return dict(case=case['name'],rows=len(frame),accepted=int(good.sum()),smoothing=summary['smoothing'],
        max_reference_abs_error=max((x[s]['reference_abs_error'] for x in summary['measures'] for s in ('raw','filtered')),default=0.),
        max_reference_error_fraction_of_tolerance=max((x[s]['reference_error_fraction_of_tolerance'] for x in summary['measures'] for s in ('raw','filtered')),default=0.),
        max_abs_frontend_db_delta=max((abs(x['filtered_minus_raw_db']) for x in summary['measures']),default=0.))
