
import argparse
import glob
import os
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from lilia.windowing import continuous_slices, plot_breaks
import numpy as np
import torch
from scipy import signal

from lilia.pathing import get_project_root, import_tinyunetv4
from lilia.signal import (
    apply_filters as _apply_filters_shared,
    resample_seconds as _resample_with_time_shared,
)

# ── Constants ──────────────────────────────────────────────────────────────────
from lilia.constants import FS, DOWNSAMPLED_FS, BP_LOW as BANDPASS_LOW, BP_HIGH as BANDPASS_HIGH, N_CH, N_CH_OUT

BASE_DIR       = get_project_root()
NOTCH_FREQ     = 60.0
NOTCH_Q        = 30.0

ARTIFACT_PEAK_HZ  = 33.25
ARTIFACT_PEAK_BW  = 1.0

ARTIFACT_THRESH_MAD = 3.0
ARTIFACT_MARGIN_MS  = 20

MODEL_WINDOW = 400
MODEL_PATH   = os.path.join(BASE_DIR, 'tiny_v4_optimized.pth')


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description='EEG signal analysis: APP vs NUC')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--group', metavar='LABEL',
                     help='Group label (e.g. 10Hz). Finds *compare_<LABEL>.csv and *<LABEL>.csv.')
    grp.add_argument('--app', metavar='PATH',
                     help='Path to APP (compare) CSV file.')
    p.add_argument('--nuc', metavar='PATH',
                   help='Path to NUC CSV file (required when --app is used).')
    p.add_argument('--outdir', metavar='DIR',
                   help='Output directory (default: <base>/<group>/ or same dir as --app).')
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """Return (path_app, path_nuc, outdir, group_label)."""
    if args.group:
        label = args.group
        hits_app = glob.glob(os.path.join(BASE_DIR, f'*compare_{label}.csv'))
        hits_nuc = glob.glob(os.path.join(BASE_DIR, f'*_{label}.csv'))
        hits_nuc = [h for h in hits_nuc if 'compare' not in os.path.basename(h)]
        if not hits_app:
            raise FileNotFoundError(f'No APP file matching *compare_{label}.csv in {BASE_DIR}')
        if not hits_nuc:
            raise FileNotFoundError(f'No NUC file matching *_{label}.csv in {BASE_DIR}')
        path_app = sorted(hits_app)[-1]
        path_nuc = sorted(hits_nuc)[-1]
        outdir   = args.outdir or os.path.join(BASE_DIR, label)
    else:
        if not args.nuc:
            raise ValueError('--nuc is required when using --app')
        path_app = args.app
        path_nuc = args.nuc
        label    = os.path.basename(path_app)
        outdir   = args.outdir or os.path.dirname(os.path.abspath(path_app))

    os.makedirs(outdir, exist_ok=True)
    print(f'APP : {path_app}')
    print(f'NUC : {path_nuc}')
    print(f'Out : {outdir}')
    return path_app, path_nuc, outdir, label


# ── File loading ───────────────────────────────────────────────────────────────
def load_file_us(path):
    """Load original integer timestamps; header absolute offsets are not applied."""
    from lilia.io import read_lilia_frame, read_abs_time_offset
    with open(path) as handle:
        lines = [handle.readline() for _ in range(4)]
    gain = float(lines[1].split(',')[1])
    frame = read_lilia_frame(path)
    if len(frame.columns) < N_CH+1:
        raise ValueError('APP/NUC recordings require at least four EEG channels')
    time_us = frame.iloc[:,0].to_numpy(dtype=np.int64)
    continuous_slices(time_us, FS)
    raw = frame.iloc[:,1:N_CH+1].to_numpy(dtype=float)
    info = {'filename':os.path.basename(path), 'gain':gain,
            'header_abs_time_offset_us':read_abs_time_offset(path),
            'used_channels':[1,2,3,4], 'declared_channels':len(frame.columns)-1}
    return time_us, raw, info


def load_file(path: str) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Compatibility seconds adapter; the CLI retains integer microseconds."""
    t, raw, info = load_file_us(path)
    return t/1e6, raw, info['filename'], info['gain']


def bandpass(data: np.ndarray, fs: float = FS,
             low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
             order: int = 4) -> np.ndarray:
    """Wrapper around shared lilia.signal.bandpass for backward compatibility."""
    from lilia.signal import bandpass as _bandpass_shared
    return _bandpass_shared(data, fs=fs, low=low, high=high, order=order)


def notch(data: np.ndarray, fs: float = FS,
          freq: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    """Wrapper around shared lilia.signal.notch for backward compatibility."""
    from lilia.signal import notch as _notch_shared
    return _notch_shared(data, fs=fs, freq=freq, q=q)


def bandstop(data: np.ndarray, fs: float = FS,
             center: float = ARTIFACT_PEAK_HZ, bw: float = ARTIFACT_PEAK_BW,
             order: int = 4) -> np.ndarray:
    """Wrapper around shared lilia.signal.bandstop for backward compatibility."""
    from lilia.signal import bandstop as _bandstop_shared
    return _bandstop_shared(data, fs=fs, center=center, bw=bw, order=order)


def apply_filters(data: np.ndarray, fs: float = FS) -> np.ndarray:
    """Wrapper around shared lilia.signal.apply_filters for backward compatibility."""
    return _apply_filters_shared(data, fs=fs, bandpass_low=BANDPASS_LOW,
                                 bandpass_high=BANDPASS_HIGH, notch_freq=NOTCH_FREQ,
                                 notch_q=NOTCH_Q, bandstop_center=ARTIFACT_PEAK_HZ,
                                 bandstop_bw=ARTIFACT_PEAK_BW)


def downsample_data(time_s: np.ndarray, data: np.ndarray,
                    fs_in: float = FS,
                    fs_out: float = DOWNSAMPLED_FS,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Resample using the shared seconds adapter, preserving sub-second time."""
    return _resample_with_time_shared(time_s, data, fs_in, fs_out)


# ── Artifact removal ───────────────────────────────────────────────────────────
def remove_artifacts(data: np.ndarray, label: str = '',
                     fs: float = FS,
                     thresh_mad: float = ARTIFACT_THRESH_MAD,
                     margin_ms: float = ARTIFACT_MARGIN_MS,
                     return_masks: bool = False,
                     ) -> tuple[np.ndarray, list[int]]:
    """Detect and linearly-interpolate amplitude artifacts via MAD threshold.

    Parameters
    ----------
    data       : (N, n_ch) EEG array
    label      : human-readable label for progress messages
    fs         : sample rate (Hz)
    thresh_mad : outlier threshold in units of MAD
    margin_ms  : expand artifact windows by this many milliseconds

    Returns
    -------
    cleaned : (N, n_ch) float array with artifacts interpolated
    counts  : list of removed-sample counts per channel
    """
    data = np.asarray(data,dtype=float)
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError('Artifact removal needs nonempty finite channel data')
    if not all(np.isfinite(v) for v in (fs,thresh_mad,margin_ms)) or fs<=0 or thresh_mad<0 or margin_ms<0:
        raise ValueError('Invalid artifact repair settings')
    masks = np.zeros(data.shape,dtype=bool)
    margin = int(fs * margin_ms / 1000)
    n      = data.shape[0]
    out    = data.copy()
    counts = []
    for ch in range(data.shape[1]):
        col = data[:, ch]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        bad = np.abs(col - med) > thresh_mad * mad
        bad_exp = np.zeros(n, dtype=bool)
        for idx in np.where(bad)[0]:
            bad_exp[max(0, idx - margin) : min(n, idx + margin + 1)] = True
        masks[:,ch] = bad_exp
        counts.append(int(bad_exp.sum()))
        if bad_exp.any():
            x_good = np.where(~bad_exp)[0]
            if not len(x_good):
                raise ValueError(f'No finite interpolation anchors in channel {ch+1}')
            out[:, ch] = np.interp(np.arange(n), x_good, col[x_good])
    for i, cnt in enumerate(counts):
        print(f'{label} ch{i+1}: removed {cnt} samples ({cnt/n*100:.2f}%)')
    return (out, counts, masks) if return_masks else (out, counts)


# ── Model inference ─────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH, device: str = 'cpu') -> torch.nn.Module:
    """Load TinyUNetV4 from a PyTorch checkpoint."""
    TinyUNetV4 = import_tinyunetv4()
    model = TinyUNetV4(in_channels=N_CH, out_channels=N_CH_OUT)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model.to(device)


def run_model(model, data, window=MODEL_WINDOW, hop=None, device='cpu'):
    """
    Overlap-add inference (50% overlap, Hann window, reflect-pad).
    Per-window RMS normalisation matching training convention.
    """
    if hop is None:
        hop = window // 2

    data = np.asarray(data)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('Model input must be a finite (samples, channels) array')
    if not isinstance(window, int) or not isinstance(hop, int) or window < 4 or not 1 <= hop < window:
        raise ValueError('Model window must be at least 4 samples and 1 <= hop < window')
    N, n_ch_in = data.shape
    if N < window:
        raise ValueError('Model input must contain at least one complete model window')
    pad = hop
    data_padded = np.concatenate([
        data[:pad][::-1].copy(),
        data,
        data[-pad:][::-1].copy(),
    ], axis=0)
    M = data_padded.shape[0]

    n_windows = max(1, (M - window) // hop + 1)
    total_len = (n_windows - 1) * hop + window
    if total_len > M:
        data_padded = np.concatenate(
            [data_padded, np.zeros((total_len - M, n_ch_in))], axis=0)
        M = data_padded.shape[0]

    hann    = np.hanning(window)
    output  = np.zeros((M, N_CH_OUT), dtype=np.float64)
    weights = np.zeros(M, dtype=np.float64)

    with torch.no_grad():
        for i in range(n_windows):
            start, end = i * hop, i * hop + window
            if end > M:
                break
            chunk     = np.ascontiguousarray(data_padded[start:end])
            inp       = torch.from_numpy(chunk.T).unsqueeze(0).float().to(device)
            x_rms     = torch.sqrt((inp ** 2).mean(dim=(1, 2), keepdim=True)) + 1e-8
            pred      = (model(inp / x_rms) * x_rms).squeeze(0).cpu().numpy().T
            if pred.shape != (window, N_CH_OUT) or not np.isfinite(pred).all():
                raise ValueError('Model output has incorrect shape or non-finite samples')
            output[start:end]  += pred * hann[:, None]
            weights[start:end] += hann

    if np.any(weights[pad:pad + N] <= 0):
        raise ValueError('Overlap-add left uncovered output samples')
    weights = np.maximum(weights, 1e-8)
    return (output / weights[:, None])[pad : pad + N].astype(np.float32)


# ── Phase-lag ──────────────────────────────────────────────────────────────────
def estimate_lag(sig_a: np.ndarray, sig_b: np.ndarray,
                 fs: float = FS, ref_ch: int = 0) -> int:
    """
    np.correlate convention: a[n] ≈ b[n - L].
    Positive L → sig_b leads sig_a.
    """
    N = min(len(sig_a), len(sig_b))
    if N < 8:
        raise ValueError('Lag estimation needs at least 8 finite paired samples')
    a = sig_a[:N, ref_ch].copy(); a = (a - a.mean()) / (a.std() + 1e-8)
    b = sig_b[:N, ref_ch].copy(); b = (b - b.mean()) / (b.std() + 1e-8)
    if N < 8 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Lag estimation needs at least 8 finite paired samples')
    if np.std(a)<1e-12 or np.std(b)<1e-12:
        raise ValueError('Lag is undefined for a constant reference channel')
    corr = signal.correlate(a, b, mode='full', method='direct' if N<=10000 else 'fft')
    lags = np.arange(-(N - 1), N)
    lag  = int(lags[np.argmax(corr)])
    leader = 'NUC leads APP' if lag > 0 else 'APP leads NUC'
    print(f'Estimated lag: {lag:+d} samples ({lag/fs*1000:+.1f} ms)  '
          f'[{leader}]  [peak corr={corr.max()/N:.3f}]')
    return lag


def align_for_comparison(sig_a: np.ndarray, time_a: np.ndarray,
                         sig_b: np.ndarray, lag: int,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim to overlapping physical window. sig_a[n] ≈ sig_b[n - lag]."""
    if lag > 0:
        L = min(len(sig_a) - lag, len(sig_b))
        return sig_a[lag:lag+L], sig_b[:L], time_a[lag:lag+L]
    elif lag < 0:
        l = -lag
        L = min(len(sig_a), len(sig_b) - l)
        return sig_a[:L], sig_b[l:l+L], time_a[:L]
    else:
        L = min(len(sig_a), len(sig_b))
        return sig_a[:L], sig_b[:L], time_a[:L]


# ── PSD ────────────────────────────────────────────────────────────────────────
def compute_psd(data_col: np.ndarray, fs: float = FS, segments=None):
    """Segment-local Welch, pooled in linear power by actual Welch window count."""
    from lilia.comparison import segmented_psd
    segments = [slice(0,len(data_col))] if segments is None else segments
    freqs, db, _ = segmented_psd(data_col,fs,segments)
    return freqs, db


# ── STFT ───────────────────────────────────────────────────────────────────────
STFT_NPERSEG  = 256   # ~0.51 s window
STFT_NOVERLAP = 192   # 75 % overlap


def compute_stft(data_col: np.ndarray, fs: float = FS,
                 nperseg: int = STFT_NPERSEG,
                 noverlap: int = STFT_NOVERLAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT magnitude in dB. Returns (frequencies, times, S_db)."""
    data_col=np.asarray(data_col)
    if data_col.ndim!=1 or len(data_col)<8 or not np.isfinite(data_col).all():
        raise ValueError('STFT requires at least 8 finite samples')
    nperseg=min(nperseg,len(data_col))
    noverlap=min(noverlap,nperseg-1)
    f, t, Zxx = signal.stft(data_col, fs=fs, nperseg=nperseg, noverlap=noverlap,
                             window='hann')
    return f, t, 20 * np.log10(np.abs(Zxx) + 1e-12)


def _shared_clim(arrays, lo=2, hi=98):
    """Compute shared vmin/vmax from percentiles across all arrays."""
    combined = np.concatenate([a.ravel() for a in arrays])
    return np.percentile(combined, lo), np.percentile(combined, hi)


def _plot_segment_stft(time_a, sig_a, time_b, sig_b, n_ch, title, outpath,
                       label_a, label_b, fmax, fs, ids_a=None, ids_b=None):
    from lilia.comparison import stft_parts
    fig, axes = plt.subplots(n_ch,2,figsize=(18,3.5*n_ch),squeeze=False)
    fig.suptitle(title,fontsize=14,fontweight='bold')
    for ch in range(n_ch):
        panels = [stft_parts(time_a,sig_a[:,ch],fs,ids_a), stft_parts(time_b,sig_b[:,ch],fs,ids_b)]
        matrices = [db[f<=fmax] for parts in panels for f,t,db in parts]
        if not matrices:
            for ax in axes[ch]: ax.text(.5,.5,'No complete spectrogram interval',transform=ax.transAxes,ha='center')
            continue
        vmin,vmax = _shared_clim(matrices)
        for col,parts in enumerate(panels):
            ax=axes[ch,col]
            img=None
            for f,t,db in parts:
                mask=f<=fmax
                img=ax.pcolormesh(t,f[mask],db[mask],shading='gouraud',cmap='inferno',vmin=vmin,vmax=vmax)
            if img is not None:fig.colorbar(img,ax=ax,label='dB')
            ax.set(xlabel='Elapsed time (s)',ylabel='Frequency (Hz)',title=f'Ch{ch+1} — {[label_a,label_b][col]}')
    fig.tight_layout();fig.savefig(outpath,dpi=150);plt.close(fig)


def plot_stft_before_after(time, before, after, title, outpath, color, fmax=50., fs=FS, segment_ids=None):
    _plot_segment_stft(time,before,time,after,before.shape[1],title,outpath,'Before','After',fmax,fs,segment_ids,segment_ids)


def plot_stft_comparison(time_a, sig_a, time_b, sig_b, n_ch, title, outpath,
                         label_a, label_b, fmax=50., fs=FS, segment_ids_a=None, segment_ids_b=None):
    _plot_segment_stft(time_a,sig_a,time_b,sig_b,n_ch,title,outpath,label_a,label_b,fmax,fs,segment_ids_a,segment_ids_b)


from lilia.qeeg import (          # noqa: E402  (after sys.path setup)
    compute_qeeg_indices_windowed as _compute_qeeg_indices_windowed,
    plot_qeeg_indices,
)


def compute_qeeg_indices_windowed(*args, **kwargs):
    """Compatibility adapter for callers of the historical CLI helper."""
    return _compute_qeeg_indices_windowed(*args, **kwargs)

# ── Plotting helpers ───────────────────────────────────────────────────────────
def plot_artifact_removal(time, raw, cleaned, counts, title, outpath, color, segment_ids=None):
    """Plot per-channel time-domain traces before and after artifact removal."""
    fig, axes = plt.subplots(N_CH, 1, figsize=(18, 3.0*N_CH), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(N_CH):
        ax = axes[i]
        ax.plot(*plot_breaks(time, raw[:, i], segment_ids),     color='#aaaaaa', lw=0.5, label='before')
        ax.plot(*plot_breaks(time, cleaned[:, i], segment_ids), color=color,     lw=0.7, label='after')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Channel {i+1}  (removed {counts[i]} samples)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_psd_before_after(raw, cleaned, title, outpath, color, fs=FS, segments=None):
    """Plot per-channel PSD before and after processing."""
    n_ch = raw.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3.5*n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        f_b, p_b = compute_psd(raw[:, i], fs=fs, segments=segments)
        f_a, p_a = compute_psd(cleaned[:, i], fs=fs, segments=segments)
        ax.plot(f_b, p_b, color='#aaaaaa', lw=1.2, label='before')
        ax.plot(f_a, p_a, color=color,     lw=1.2, label='after')
        ax.set_ylabel('PSD (dB/Hz)'); ax.set_title(f'Channel {i+1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlim(0, 50)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_td_comparison(time, sig_a, sig_b, n_ch, title, outpath,
                       label_a, label_b, colors, segment_ids=None):
    """Plot per-channel time-domain comparison of two signals."""
    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 3.5*n_ch), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        ax.plot(*plot_breaks(time, sig_a[:, i], segment_ids), color=colors[0], alpha=0.8, lw=0.6,
                label=f'{label_a} | ch{i+1}')
        ax.plot(*plot_breaks(time, sig_b[:, i], segment_ids), color=colors[1], alpha=0.8, lw=0.6,
                label=f'{label_b} | ch{i+1}')
        ax.set_ylabel('Amplitude (µV)'); ax.set_title(f'Channel {i+1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time[0], time[-1])
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_psd_comparison(sig_a, sig_b, n_ch, title, outpath,
                        label_a, label_b, colors, fs=FS, segments_a=None, segments_b=None):
    """Plot per-channel PSD comparison of two signals."""
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3.5*n_ch), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        f_a, p_a = compute_psd(sig_a[:, i], fs=fs, segments=segments_a)
        f_b, p_b = compute_psd(sig_b[:, i], fs=fs, segments=segments_b)
        ax.plot(f_a, p_a, color=colors[0], lw=1.2, label=f'{label_a} | ch{i+1}')
        ax.plot(f_b, p_b, color=colors[1], lw=1.2, label=f'{label_b} | ch{i+1}')
        ax.set_ylabel('PSD (dB/Hz)'); ax.set_title(f'Channel {i+1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlim(0, 50)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_model_before_after(time, before, after, title, outpath, color, segment_ids=None):
    """Plot per-channel time-domain traces before and after model inference."""
    fig, axes = plt.subplots(N_CH_OUT, 1, figsize=(18, 3.5*N_CH_OUT), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(N_CH_OUT):
        ax = axes[i]
        ax.plot(*plot_breaks(time, before[:, i], segment_ids), color='#aaaaaa', lw=0.5, alpha=0.9, label='before model')
        ax.plot(*plot_breaks(time, after[:, i], segment_ids),  color=color,     lw=0.7, alpha=0.9, label='after model')
        ax.set_ylabel('Amplitude (µV)'); ax.set_title(f'Channel {i+1}')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    from lilia.comparison import prepare_recording, align_recordings, segmented_psd
    from lilia.comparison_io import write_recording, write_pair_table
    from lilia.neural import build_inference_timeline, model_provenance
    from lilia.provenance import file_sha256
    args = parse_args()
    path_a, path_b, outdir, label = resolve_paths(args)
    audit_path = Path(outdir)/'app_nuc_analysis.json'
    audit = {'kind':'app_nuc_analysis','schema_version':1,'label':label,'sources':{},'errors':[],
             'quality_state':'disabled','clock_policy':'each_recording_elapsed; header offsets recorded but not applied'}
    artifacts = []
    def out(name):
        path = Path(outdir)/name
        artifacts.append(path)
        return str(path)
    def save_audit():
        audit['artifacts']={p.name:file_sha256(p) for p in artifacts if p.is_file()}
        audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    try:
        paths = [Path(__file__), *sorted((Path(__file__).parent/'lilia').glob('*.py'))]
        code_id = hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
        parameters = {'input_fs':FS,'fs':DOWNSAMPLED_FS,'model_window':MODEL_WINDOW,'model_hop':MODEL_WINDOW//2,
            'win_sec':5.,'step_sec':5.,'index_space':'resampled_model_output','model':model_provenance(),
            'artifact_repair':{'threshold_mad':ARTIFACT_THRESH_MAD,'margin_ms':ARTIFACT_MARGIN_MS,
                               'scope':'within_each_source_segment','edge_policy':'nearest_valid_anchor'},
            'quality_state':'disabled','clock_policy':audit['clock_policy']}
        audit.update(parameters=parameters,code_sha256=code_id)
        model = load_model()
        records = {}
        for tag,path,color in [('app',path_a,'#1f77b4'),('nuc',path_b,'#ff7f0e')]:
            print(f'--- Processing {tag.upper()} by source segment ---')
            t,raw,info = load_file_us(path)
            initial = build_inference_timeline(t,FS,DOWNSAMPLED_FS,MODEL_WINDOW,MODEL_WINDOW//2)
            audit['sources'][tag]={'source_path':str(Path(path).resolve()),'source_id':file_sha256(path),
                                    'header':info,'inference':initial.metadata(),'status':'processing'}
            rec = prepare_recording(t,raw,model)
            records[tag] = rec
            tl = rec['timeline']
            raw_slices = continuous_slices(t,FS)
            raw_ids = np.concatenate([np.full(sl.stop-sl.start,i,dtype=np.int64) for i,sl in enumerate(raw_slices)])
            slices = continuous_slices(tl.time_us,tl.fs_out,segment_ids=tl.segment_ids)
            rec['slices'] = slices
            elapsed = (tl.time_us-tl.source_epoch_us)/1e6
            rec['elapsed'] = elapsed
            counts = np.sum([r['counts'] for r in rec['repair']],axis=0).astype(int).tolist()
            audit['sources'][tag].update(status='complete',artifact_repair=rec['repair'],
                output_samples=len(tl.time_us),qeeg_windows=len(rec['qeeg_grid'].starts) if rec['qeeg_grid'] is not None else 0,
                qeeg_status='computed' if rec['qeeg_grid'] is not None else 'no_complete_5s_window',
                psd_raw=segmented_psd(rec['cleaned_raw'][:,0],FS,raw_slices)[2],
                psd_model=segmented_psd(rec['output'][:,0],tl.fs_out,slices)[2])
            artifacts.extend(write_recording(outdir,tag,path,rec,parameters,code_id))
            plot_artifact_removal((t-t[0])/1e6,rec['filtered_raw'],rec['cleaned_raw'],counts,
                f'{tag.upper()}: segment-local MAD repair',out(f'{tag}_artifact_removal.png'),color,segment_ids=raw_ids)
            plot_psd_before_after(rec['filtered_raw'],rec['cleaned_raw'],f'{tag.upper()}: PSD before/after MAD repair',
                out(f'{tag}_psd_artifact_removal.png'),color,segments=raw_slices)
            plot_model_before_after(elapsed,rec['input'][:,:2],rec['output'],f'{tag.upper()}: before/after model',
                out(f'{tag}_model_before_after_td.png'),color,segment_ids=tl.segment_ids)
            plot_psd_before_after(rec['input'][:,:2],rec['output'],f'{tag.upper()}: model PSD',
                out(f'{tag}_model_before_after_psd.png'),color,fs=tl.fs_out,segments=slices)
            plot_stft_before_after(elapsed,rec['input'][:,:2],rec['output'],f'{tag.upper()}: model spectrogram',
                out(f'{tag}_model_stft.png'),color,fs=tl.fs_out,segment_ids=tl.segment_ids)
            if rec['qeeg_grid'] is not None:
                grid = rec['qeeg_grid']
                for ch in range(2):
                    indices = {}
                    for key in ('theta','alpha','beta','focus','flow','calm','relaxation'):
                        times,values = plot_breaks(grid.time_s,rec['qeeg'][f'{key}_ch{ch+1}'],grid.columns['segment_id'])
                        indices[key] = values
                    indices['time'] = times
                    plot_qeeg_indices(indices,f'{tag.upper()}: model qEEG ch{ch+1} (quality scoring disabled)',
                                      out(f'qeeg_indices_{tag}_ch{ch+1}.png'),marker='.')
        a,b = records['app'],records['nuc']
        ia,ib,groups,alignment = align_recordings(a,b)
        audit['alignment'] = alignment
        pair_path = out('app_nuc_alignment.csv')
        write_pair_table(pair_path,a,b,ia,ib,groups,alignment,path_a,path_b,parameters)
        artifacts.append(Path(pair_path+'.meta.json'))
        time_cmp = a['elapsed'][ia]
        audit['comparison_psd_policy'] = 'independent full recordings; segment Welch weighted by subwindow count'
        for key,n_ch,td_name,psd_name,stft_name in [
            ('input',4,'time_domain_comparison.png','psd_comparison.png','stft_filtered_comparison.png'),
            ('output',2,'model_output_time_domain.png','model_output_psd.png','stft_model_output_comparison.png')]:
            title=f'APP/NUC {key}: elapsed waveform alignment, lag={alignment["lag_samples"]:+d} samples'
            plot_td_comparison(time_cmp,a[key][ia],b[key][ib],n_ch,title,out(td_name),
                               'APP','NUC',['#1f77b4','#ff7f0e'],segment_ids=groups)
            plot_psd_comparison(a[key],b[key],n_ch,f'APP/NUC {key}: independent segment PSDs',out(psd_name),
                'APP','NUC',['#1f77b4','#ff7f0e'],fs=DOWNSAMPLED_FS,segments_a=a['slices'],segments_b=b['slices'])
            plot_stft_comparison(time_cmp,a[key][ia,:2],time_cmp,b[key][ib,:2],2,title,out(stft_name),
                'APP','NUC',fs=DOWNSAMPLED_FS,segment_ids_a=groups,segment_ids_b=groups)
        audit['status'] = 'complete'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['errors'].append(f'{type(exc).__name__}: {exc}')
        save_audit()
        raise
    save_audit()
    print(f'All outputs and source audit saved to: {outdir}')
    return audit


if __name__ == '__main__':
    main()
