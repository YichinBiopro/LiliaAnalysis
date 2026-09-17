"""Measurement-only PSD controls; no product profile or default is changed."""
import numpy as np
from scipy.signal import welch


def reference_welch(values, fs, nperseg):
    """Independent NumPy FFT reconstruction of mean, one-sided Hann density."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or nperseg < 2 or nperseg > len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid reference Welch input')
    window = .5 - .5*np.cos(2*np.pi*np.arange(nperseg)/nperseg)
    spectra = []
    for start in range(0, len(x)-nperseg+1, nperseg-nperseg//2):
        part = x[start:start+nperseg]
        power = abs(np.fft.rfft((part-part.mean())*window))**2/(fs*np.sum(window**2))
        power[1:-1 if nperseg % 2 == 0 else None] *= 2
        spectra.append(power)
    return np.fft.rfftfreq(nperseg, 1/fs), np.mean(spectra, axis=0)


def assert_reference(frequency, power, values, fs, nperseg):
    f, expected = reference_welch(values, fs, nperseg)
    np.testing.assert_array_equal(frequency, f)
    np.testing.assert_allclose(power, expected, rtol=1e-10, atol=1e-12)
    return float(np.max(np.abs(power-expected)))


def measurements(values, fs, seconds):
    from lilia import qeeg, hardy2
    from spectral_entropy import _band_power
    n = min(len(values), int(fs*seconds))
    f, p = welch(values, fs=fs, window='hann', nperseg=n, noverlap=n//2,
                 detrend='constant', scaling='density', average='mean')
    error = assert_reference(f, p, values, fs, n)
    bounds = [(4, 8), (8, 13), (13, 30)]
    half = np.array([qeeg._band_power(f, p, *b) for b in bounds])
    closed = np.array([hardy2.bandpower(f, p, *b) for b in bounds])
    entropy = np.array([_band_power(f, p, *b, include_upper=False) for b in bounds])
    five = sum(hardy2.bandpower(f, p, *b) for b in hardy2.BANDS.values())
    full = hardy2.bandpower(f, p, .5, 45)
    ratios = np.vstack([half/(d+1e-9) for d in (half.sum(), five, full)])
    return f, p, dict(nperseg=n, df=float(f[1]-f[0]), reference_max_abs_error=error,
        peak_hz=float(f[np.argmax(p)]), total_rectangle=float(p.sum()*(f[1]-f[0])),
        total_trapezoid=float(np.trapezoid(p, f)), half_band=half.tolist(),
        closed_band=closed.tolist(), entropy_band=entropy.tolist(),
        denominator_values=[float(half.sum()), float(five), float(full)],
        ratios_by_denominator=ratios.tolist(),
        boundary_max_abs=float(np.max(abs(half-closed))),
        singleton_max_abs=float(np.max(abs(half-entropy))),
        denominator_max_abs=float(np.max(abs(ratios[0]-ratios[1:]))))


def run(out):
    from tools.freeze_method_profiles import write_json
    import matplotlib.pyplot as plt
    arrays, rows, plots = {}, [], {}
    for fs in (200, 500):
        for duration in (.5, 1, 2, 4, 8):
            t = np.arange(int(fs*duration))/fs
            signals = {f'tone{hz}': 2*np.sin(2*np.pi*hz*t) for hz in (4, 8, 13, 30, 10.3)}
            signals.update(mixture=sum(np.sin(2*np.pi*hz*t) for hz in (2, 6, 10, 20, 40, 60)),
                           noise=np.random.default_rng(1922).normal(size=len(t)), constant=np.full(len(t), 3.))
            for name, values in signals.items():
                pair = []
                for seconds in (1, 4):
                    f, p, row = measurements(values, fs, seconds)
                    if name in ('tone4', 'tone8', 'tone13', 'tone30') and duration >= 1:
                        np.testing.assert_allclose(row['total_rectangle'], 2., rtol=1e-10, atol=1e-12)
                    if name == 'constant':
                        np.testing.assert_array_equal(p, np.zeros_like(p))
                    key = f'{fs}_{duration}_{name}_{seconds}'
                    arrays[key+'_frequency'], arrays[key+'_density'] = f, p
                    row.update(fs=fs, duration=duration, signal=name, nominal_seconds=seconds)
                    rows.append(row)
                    pair.append(np.array(row['ratios_by_denominator'][0]))
                    if fs == 500 and duration == 8 and name in ('tone8', 'mixture'):
                        plots[name, seconds] = (f, p, row)
                for row in rows[-2:]:
                    row['window_ratio_max_abs'] = float(np.max(abs(pair[0]-pair[1])))
    f, p, singleton = measurements(2*np.sin(2*np.pi*25*np.arange(8)/200), 200, 1)
    if singleton['half_band'][2] != 0 or singleton['entropy_band'][2] <= 0:
        raise AssertionError('Single-bin legacy policies changed')
    arrays['singleton_frequency'], arrays['singleton_density'] = f, p
    np.savez_compressed(out/'sensitivity.npz', **arrays)
    summary = dict(controls=len(rows), singleton_controls=1, singleton=singleton, rows=rows,
        maxima_scope='160 window controls plus the single-bin probe; window delta excludes singleton',
        maxima={k: max(r.get(k, 0.) for r in rows+[singleton]) for k in ('reference_max_abs_error', 'boundary_max_abs',
            'singleton_max_abs', 'denominator_max_abs', 'window_ratio_max_abs')},
        scientific_ranking=False, profile_changes=False)
    write_json(out/'sensitivity.json', summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for i, name in enumerate(('tone8', 'mixture')):
        for seconds in (1, 4):
            f, p, row = plots[name, seconds]
            axes[i, 0].semilogy(f, np.maximum(p, 1e-15), label=f'{seconds}s Welch')
            axes[i, 1].plot(['theta', 'alpha', 'beta'], row['ratios_by_denominator'][0],
                            '-o', label=f'{seconds}s / 3-band denominator')
        axes[i, 0].set(xlim=(0, 65), ylim=(1e-6, 10), xlabel='Hz', ylabel='Input-unit²/Hz', title=name)
        axes[i, 1].set(ylim=(0, 1), ylabel='Relative power', title=name+' — half-open trapezoid')
        for ax in axes[i]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.savefig(out/'sensitivity.png', dpi=120)
    plt.close(fig)
    return summary


def run_captured(out, cases):
    """Compare identical captured inputs; never concatenate across source gaps."""
    from tools.freeze_method_profiles import write_json
    import matplotlib.pyplot as plt
    totals, all_rows = [], []
    plots = {}
    for case in cases:
        folder = out/('local' if case['real'] else 'synthetic')/'actual'/case['name']
        rows, spectra = [], {}
        with np.load(folder/'numeric.npz', allow_pickle=False) as captured:
            keys = sorted(k for k in captured.files if
                          (k.startswith('jen_ch') and k.endswith('_values')) or
                          (k.startswith('quality_welch_') and k.endswith('_input')))
            for key in keys:
                values = captured[key].astype(np.float64)
                fs = 200 if key.startswith('jen_') else 500
                pair = []
                for seconds in (1, 4):
                    f, p, row = measurements(values, fs, seconds)
                    row.update(capture_key=key, fs=fs, samples=len(values), nominal_seconds=seconds)
                    pair.append(row)
                    spectra[f'{key}_{seconds}_frequency'] = f
                    spectra[f'{key}_{seconds}_density'] = p
                    if case['name'] == 'real_1' and key in (
                            'jen_ch1_panel0_values', 'jen_ch1_panel1_values',
                            'quality_welch_0_input', 'quality_welch_1_input'):
                        plots[key, seconds] = (f, p)
                delta = float(np.max(np.abs(np.array(pair[0]['ratios_by_denominator'][0])-
                                           np.array(pair[1]['ratios_by_denominator'][0]))))
                peak_delta = abs(pair[0]['peak_hz']-pair[1]['peak_hz'])
                for row in pair:
                    row.update(window_ratio_max_abs=delta, window_peak_hz_abs=peak_delta)
                rows.extend(pair)
        if not keys and case['model_expected'] != 'no_complete_window':
            raise AssertionError('Missing captured sensitivity inputs')
        maxima = {k: max((r[k] for r in rows), default=None) for k in
                  ('reference_max_abs_error', 'boundary_max_abs', 'singleton_max_abs',
                   'denominator_max_abs', 'window_ratio_max_abs', 'window_peak_hz_abs')}
        summary = dict(case=case['name'], inputs=len(keys), comparisons=len(rows), maxima=maxima,
                       rows=rows, profile_changes=False, quality_or_selection_changes=False,
                       empty_reason='no complete model/quality windows' if not keys else None)
        write_json(folder/'sensitivity.json', summary)
        if spectra:
            np.savez_compressed(folder/'sensitivity.npz', **spectra)
        totals.append({k: summary[k] for k in ('case', 'inputs', 'comparisons', 'maxima', 'empty_reason')})
        all_rows.extend(rows)
    if plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for ax, key, label in zip(axes.flat,
                ('jen_ch1_panel0_values', 'jen_ch1_panel1_values', 'quality_welch_0_input', 'quality_welch_1_input'),
                ('Jenqwei Before ch1 S0', 'Jenqwei After ch1 S0', 'Quality raw ch1 sample 0', 'Quality filtered ch1 sample 0')):
            for seconds in (1, 4):
                f, p = plots[key, seconds]
                ax.semilogy(f, np.maximum(p, 1e-15), label=f'{seconds}s Welch')
            ax.set(title=label, xlim=(0, 50), xlabel='Hz', ylabel='Input-unit²/Hz')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.savefig(out/'local/real_sensitivity.png', dpi=120)
        plt.close(fig)
    return dict(cases=totals, inputs=sum(r['inputs'] for r in totals), comparisons=len(all_rows),
                maxima={k: max(r[k] for r in all_rows) for k in totals[0]['maxima']},
                scientific_ranking=False, profile_changes=False, quality_or_selection_changes=False)
