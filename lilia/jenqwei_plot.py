"""Jenqwei plots on separate Before/After source timelines, with per-segment Welch."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.signal import stft, welch


def validate_display(channels, max_display_sec):
    if (not channels or any(isinstance(ch, bool) or not isinstance(ch, int) or ch not in range(4)
                            for ch in channels) or len(set(channels)) != len(channels)):
        raise ValueError('Display channels must be distinct indices in 0..3; only 0/1 have model After')
    if not np.isfinite(max_display_sec) or max_display_sec <= 0:
        raise ValueError('max_display_sec must be finite and positive')


def compute_panels(result, ch, max_display_sec=60.):
    """Preserve legacy continuous Welch/STFT values; compute each source segment separately."""
    validate_display([ch], max_display_sec)
    epoch = result['timeline'].source_epoch_us
    panels = []
    # Preserve int(max_sec*fs) for a continuous source; locate those elapsed
    # limits in each segment's timestamp axis instead of concatenating gaps.
    spectral_limit = int(max_display_sec * 200) / 200
    for row in result['segments']:
        if row['status'] != 'retained':
            continue
        ba, bb = row['before_start_idx'], row['before_end_idx']
        aa, ab = row['output_start_idx'], row['output_end_idx']
        branches = [('before', result['pre_data_200'][ba:bb, ch], result['time_us_200'][ba:bb])]
        if ch < 2:
            branches.append(('after', result['tfl_data_200'][aa:ab, ch], result['tfl_time_us_200'][aa:ab]))
        for branch, values, times in branches:
            elapsed = (times - epoch) / 1e6
            f, power = welch(values.astype(np.float64), fs=200, nperseg=min(800, bb - ba, len(values)),
                             scaling='density')
            count = int(np.searchsorted(elapsed, spectral_limit, side='left'))
            record = {'segment_id': row['segment_id'], 'branch': branch, 'values': values,
                      'elapsed': elapsed, 'psd_frequency': f, 'psd': power,
                      'stft_samples': count, 'stft': None,
                      'stft_status': 'outside_display' if not count else 'too_short_for_128_overlap',
                      'clip_start_s': (row['raw_start_us'] - epoch) / 1e6,
                      'clip_end_s': min((row['raw_end_us'] - epoch) / 1e6,
                                        (int(times[-1]) - epoch + 5000) / 1e6, max_display_sec)}
            if count > 128:
                # SciPy's old short-signal adjustment is explicit here; overlap
                # stays at 128, no alternative spectral method is introduced.
                f, t, z = stft(values[:count].astype(np.float64), fs=200,
                               nperseg=min(256, count), noverlap=128, window='hann')
                positions = t * 200
                centers = np.interp(positions, np.arange(count), elapsed[:count])
                beyond = positions > count - 1
                centers[beyond] = elapsed[count - 1] + (positions[beyond] - count + 1) / 200
                mask = f <= 50
                record['stft'] = (f[mask], centers, 20 * np.log10(np.abs(z[mask]) + 1e-8))
                record['stft_status'] = 'computed'
            panels.append(record)
    return panels


def plot_metadata(result, channels, max_display_sec):
    validate_display(channels, max_display_sec)
    epoch = result['timeline'].source_epoch_us
    excluded, gaps, tails = [], [], []
    for i, row in enumerate(result['segments']):
        lo, hi = (row['raw_start_us'] - epoch) / 1e6, (row['raw_end_us'] - epoch) / 1e6
        if i:
            gaps.append({'start_s': (result['segments'][i - 1]['raw_end_us'] - epoch) / 1e6, 'end_s': lo})
        if row['status'] == 'excluded':
            excluded.append({'segment_id': row['segment_id'], 'start_s': lo, 'end_s': hi, 'reason': row['reason']})
        elif row['trimmed_samples']:
            end = row['output_end_idx']
            start = min(hi, (int(result['tfl_time_us_200'][end - 1]) - epoch + 5000) / 1e6)
            tails.append({'segment_id': row['segment_id'], 'start_s': start, 'end_s': hi,
                          'trimmed_samples': row['trimmed_samples']})
    return {'time_axis': 'elapsed_from_source_epoch', 'source_epoch_us': epoch,
            'xlim_s': [0., min(max_display_sec, (result['segments'][-1]['raw_end_us'] - epoch) / 1e6)],
            'channels': [{'before_source_channel': ch + 1, 'after_source_channel': ch + 1 if ch < 2 else None}
                         for ch in channels], 'excluded_spans': excluded, 'missing_spans': gaps, 'after_tails': tails,
            'psd': {'method': 'Welch density per source segment, no cross-segment average', 'fs': 200,
                    'nperseg': 'min(800, before_segment_samples, branch_segment_samples)',
                    'window': 'hann', 'overlap': 'half', 'detrend': 'constant'},
            'stft': {'fs': 200, 'nperseg': 'min(256, displayed_segment_samples)', 'noverlap': 128,
                     'window': 'hann', 'boundary': 'zeros', 'padded': True, 'fmax': 50,
                     'scale': '20*log10(abs(z)+1e-8)', 'short_display': 'omit_if_at_most_128_samples',
                     'display_limit': 'floor(max_sec*200)/200 elapsed; no packed-gap truncation'},
            'display_max_sec': max_display_sec}


def _decorate_time(ax, metadata, show_tails):
    for span in metadata['excluded_spans']:
        ax.axvspan(span['start_s'], span['end_s'], facecolor='.9', edgecolor='.6', hatch='///', linewidth=.5)
    if show_tails:
        for span in metadata['after_tails']:
            ax.axvspan(span['start_s'], span['end_s'], facecolor='#fff3d5', edgecolor='#b89b56', hatch='..', linewidth=.5)
    for gap in metadata['missing_spans']:
        for bound in (gap['start_s'], gap['end_s']):
            ax.axvline(bound, color='.65', linestyle=':', lw=.6)
    ax.set_xlim(metadata['xlim_s'])
    ax.set_xlabel('Elapsed time (s)')
    ax.tick_params(labelsize=8)


def plot_result(result, outdir, stem, ch=0, max_display_sec=60.):
    panels = compute_panels(result, ch, max_display_sec)
    metadata = plot_metadata(result, [ch], max_display_sec)
    path = Path(outdir) / f'{stem}_ch{ch + 1}_pipeline_comparison.png'
    if path.exists():
        raise FileExistsError(f'Jenqwei plot already exists: {path}')
    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, width_ratios=[1, .035, 1, .035])
    td, psd = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])
    st_axes = {'before': fig.add_subplot(grid[2, 0]), 'after': fig.add_subplot(grid[2, 2])}
    bars = {'before': fig.add_subplot(grid[2, 1]), 'after': fig.add_subplot(grid[2, 3])}
    suffix = '; Before only: no model output for this channel' if ch >= 2 else ''
    fig.suptitle(f'{stem} — source Ch{ch + 1}{suffix}\n'
                 'Gaps blank; excluded segments ///; After trimmed tails dotted', fontsize=11)
    try:
        spectra = [p['stft'][2] for p in panels if p['stft'] is not None]
        vmin = min((v.min() for v in spectra), default=0.)
        vmax = max((v.max() for v in spectra), default=1.)
        meshes = {}
        for panel in panels:
            branch, sid = panel['branch'], panel['segment_id']
            label = f'{branch.title()} S{sid}'
            color = '#2166ac' if branch == 'before' else '#d6604d'
            style = ['-', '--', '-.', ':'][sid % 4]
            visible = panel['elapsed'] <= max_display_sec
            td.plot(panel['elapsed'][visible], panel['values'][visible], color=color, lw=.6,
                    linestyle=style, label=label, rasterized=True)
            f, power = panel['psd_frequency'], panel['psd']
            mask = f <= 50
            psd.semilogy(f[mask], power[mask], color=color, lw=1, linestyle=style, label=label)
            if panel['stft'] is not None:
                f, time, db = panel['stft']
                ax = st_axes[branch]
                mesh = ax.pcolormesh(time, f, db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
                clip = Rectangle((panel['clip_start_s'], 0), panel['clip_end_s'] - panel['clip_start_s'],
                                 50, transform=ax.transData)
                mesh.set_clip_path(clip)
                meshes[branch] = mesh
        td.set_title('Time domain — Before / After on their own timestamps', fontsize=10)
        td.set_ylabel('ADC value')
        td.legend(fontsize=7, ncol=2)
        td.grid(alpha=.2)
        _decorate_time(td, metadata, ch < 2)
        psd.set_title('PSD (Welch) — separate source segments, no averaging', fontsize=10)
        psd.set_xlabel('Frequency (Hz)')
        psd.set_ylabel('Power (ADC²/Hz)')
        psd.set_xlim(0, 50)
        psd.legend(fontsize=7, ncol=2)
        psd.grid(alpha=.2, which='both')
        for lo, hi, color in [(0.5, 4, '#a8d8ea'), (4, 8, '#aa96da'), (8, 13, '#fcbad3'),
                              (13, 30, '#ffffd2'), (30, 45, '#d4f1a1')]:
            psd.axvspan(lo, hi, color=color, alpha=.22, zorder=0)
        for branch, ax in st_axes.items():
            _decorate_time(ax, metadata, branch == 'after' and ch < 2)
            ax.set_title(f'{branch.title()} STFT', fontsize=10)
            ax.set_ylim(0, 50)
            ax.set_ylabel('Frequency (Hz)')
            if branch in meshes:
                fig.colorbar(meshes[branch], cax=bars[branch], label='dB')
            else:
                bars[branch].set_visible(False)
                text = (f'No model output for source Ch{ch + 1}' if branch == 'after' and ch >= 2
                        else 'No STFT: displayed segments have ≤128 samples')
                ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', wrap=True)
        with path.open('xb') as handle:
            fig.savefig(handle, format='png', dpi=120)
    finally:
        plt.close(fig)
    return path
