import sys

# ── External dependency from tommy ──────────────────────────────────────────────
sys.path.insert(0, '/home/bps-yichin/tommy')
from eeg_denoise.tiny_model_v4 import TinyUNetV4

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR       = '/home/bps-yichin/lilia_analysis'
FS             = 500
BANDPASS_LOW   = 0.5
BANDPASS_HIGH  = 45.0
NOTCH_FREQ     = 60.0
NOTCH_Q        = 30.0
N_CH           = 4
N_CH_OUT       = 2

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
def load_file(path: str) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Load a lilia EEG CSV and return (time_s, data, filename, gain).

    Reads gain from row 2, skips 4-row header, deduplicates column names.
    Returns time in seconds (float64), data as (N, N_CH) float64.
    """
    with open(path) as f:
        lines = [f.readline() for _ in range(5)]
    gain = float(lines[1].split(',')[1])
    df   = pd.read_csv(path, skiprows=4)
    time_s = df.iloc[:, 0].values.astype(float) / 1e6

    cols, seen, new_cols = list(df.columns), {}, []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        new_cols.append(c if seen[c] == 1 else f'{c}_{seen[c]}')
    df.columns = new_cols

    data = df.iloc[:, 1 : N_CH + 1].values.astype(float)
    return time_s, data, os.path.basename(path), gain


# ── Filters ────────────────────────────────────────────────────────────────────
def bandpass(data: np.ndarray, fs: float = FS,
             low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
             order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter column-wise."""
    sos = signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)


def notch(data: np.ndarray, fs: float = FS,
          freq: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    """Apply zero-phase IIR notch filter column-wise."""
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, data, axis=0)


def bandstop(data: np.ndarray, fs: float = FS,
             center: float = ARTIFACT_PEAK_HZ, bw: float = ARTIFACT_PEAK_BW,
             order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandstop filter column-wise."""
    sos = signal.butter(order, [center - bw/2, center + bw/2],
                        btype='bandstop', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)


def apply_filters(data: np.ndarray) -> np.ndarray:
    """Apply bandpass → notch → bandstop filters sequentially."""
    data = bandpass(data)
    data = notch(data)
    data = bandstop(data)
    return data


# ── Artifact removal ───────────────────────────────────────────────────────────
def remove_artifacts(data: np.ndarray, label: str = '',
                     fs: float = FS,
                     thresh_mad: float = ARTIFACT_THRESH_MAD,
                     margin_ms: float = ARTIFACT_MARGIN_MS,
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
        counts.append(int(bad_exp.sum()))
        if bad_exp.any():
            x_good = np.where(~bad_exp)[0]
            out[:, ch] = np.interp(np.arange(n), x_good, col[x_good])
    for i, cnt in enumerate(counts):
        print(f'{label} ch{i+1}: removed {cnt} samples ({cnt/n*100:.2f}%)')
    return out, counts


# ── Model inference ─────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH, device: str = 'cpu') -> torch.nn.Module:
    """Load TinyUNetV4 from a PyTorch checkpoint."""
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

    N, n_ch_in = data.shape
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
            output[start:end]  += pred * hann[:, None]
            weights[start:end] += hann

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
    a = sig_a[:N, ref_ch].copy(); a = (a - a.mean()) / (a.std() + 1e-8)
    b = sig_b[:N, ref_ch].copy(); b = (b - b.mean()) / (b.std() + 1e-8)
    corr = np.correlate(a, b, mode='full')
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
def compute_psd(data_col: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD in dB. Returns (frequencies, psd_db)."""
    freqs, psd = signal.welch(data_col, fs=fs, nperseg=fs*4,
                               noverlap=fs*2, window='hann')
    return freqs, 10 * np.log10(psd + 1e-12)


# ── STFT ───────────────────────────────────────────────────────────────────────
STFT_NPERSEG  = 256   # ~0.51 s window
STFT_NOVERLAP = 192   # 75 % overlap


def compute_stft(data_col: np.ndarray, fs: float = FS,
                 nperseg: int = STFT_NPERSEG,
                 noverlap: int = STFT_NOVERLAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT magnitude in dB. Returns (frequencies, times, S_db)."""
    f, t, Zxx = signal.stft(data_col, fs=fs, nperseg=nperseg, noverlap=noverlap,
                             window='hann')
    return f, t, 20 * np.log10(np.abs(Zxx) + 1e-12)


def _shared_clim(arrays, lo=2, hi=98):
    """Compute shared vmin/vmax from percentiles across all arrays."""
    combined = np.concatenate([a.ravel() for a in arrays])
    return np.percentile(combined, lo), np.percentile(combined, hi)


def plot_stft_before_after(time, before, after, title, outpath, color,
                           fmax=50.0, fs=FS):
    """Plot per-channel STFT spectrograms before and after processing."""
    n_ch = before.shape[1]
    fig, axes = plt.subplots(n_ch, 2, figsize=(18, 3.5 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=14, fontweight='bold')
    t0 = time[0]
    for i in range(n_ch):
        panels = [(before[:, i], 'Before'), (after[:, i], 'After')]
        specs = []
        for sig, _ in panels:
            f, t_stft, Sdb = compute_stft(sig, fs=fs)
            mask = f <= fmax
            specs.append((f, t_stft, Sdb, mask))
        vmin, vmax = _shared_clim([S[mask] for f, t_stft, S, mask in specs])
        for col_idx, ((f, t_stft, Sdb, mask), (_, lbl)) in enumerate(
                zip(specs, panels)):
            ax = axes[i, col_idx]
            img = ax.pcolormesh(t_stft + t0, f[mask], Sdb[mask],
                                shading='gouraud', cmap='inferno',
                                vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=ax, label='dB')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Ch{i+1} — {lbl}')
            ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_stft_comparison(time_a, sig_a, time_b, sig_b, n_ch, title, outpath,
                         label_a, label_b, fmax=50.0, fs=FS):
    """Plot per-channel STFT spectrograms side-by-side for two signals."""
    fig, axes = plt.subplots(n_ch, 2, figsize=(18, 3.5 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        panels = [(sig_a[:, i], time_a, label_a), (sig_b[:, i], time_b, label_b)]
        specs = []
        for sig, t, lbl in panels:
            f, t_stft, Sdb = compute_stft(sig, fs=fs)
            mask = f <= fmax
            specs.append((f, t_stft, Sdb, mask, t, lbl))
        vmin, vmax = _shared_clim([S[mask] for f, t_stft, S, mask, t, lbl in specs])
        for col_idx, (f, t_stft, Sdb, mask, t, lbl) in enumerate(specs):
            ax = axes[i, col_idx]
            img = ax.pcolormesh(t_stft + t[0], f[mask], Sdb[mask],
                                shading='gouraud', cmap='inferno',
                                vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=ax, label='dB')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Ch{i+1} — {lbl}')
            ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


from qeeg_indices import (          # noqa: E402  (after sys.path setup)
    compute_qeeg_indices_windowed,
    plot_qeeg_indices,
)

# ── Plotting helpers ───────────────────────────────────────────────────────────
def plot_artifact_removal(time, raw, cleaned, counts, title, outpath, color):
    """Plot per-channel time-domain traces before and after artifact removal."""
    fig, axes = plt.subplots(N_CH, 1, figsize=(18, 3.0*N_CH), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(N_CH):
        ax = axes[i]
        ax.plot(time, raw[:, i],     color='#aaaaaa', lw=0.5, label='before')
        ax.plot(time, cleaned[:, i], color=color,     lw=0.7, label='after')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Channel {i+1}  (removed {counts[i]} samples)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


def plot_psd_before_after(raw, cleaned, title, outpath, color):
    """Plot per-channel PSD before and after processing."""
    n_ch = raw.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3.5*n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        f_b, p_b = compute_psd(raw[:, i])
        f_a, p_a = compute_psd(cleaned[:, i])
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
                       label_a, label_b, colors):
    """Plot per-channel time-domain comparison of two signals."""
    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 3.5*n_ch), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        ax.plot(time, sig_a[:, i], color=colors[0], alpha=0.8, lw=0.6,
                label=f'{label_a} | ch{i+1}')
        ax.plot(time, sig_b[:, i], color=colors[1], alpha=0.8, lw=0.6,
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
                        label_a, label_b, colors):
    """Plot per-channel PSD comparison of two signals."""
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3.5*n_ch), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(n_ch):
        ax = axes[i]
        f_a, p_a = compute_psd(sig_a[:, i])
        f_b, p_b = compute_psd(sig_b[:, i])
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


def plot_model_before_after(time, before, after, title, outpath, color):
    """Plot per-channel time-domain traces before and after model inference."""
    fig, axes = plt.subplots(N_CH_OUT, 1, figsize=(18, 3.5*N_CH_OUT), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(N_CH_OUT):
        ax = axes[i]
        ax.plot(time, before[:, i], color='#aaaaaa', lw=0.5, alpha=0.9, label='before model')
        ax.plot(time, after[:, i],  color=color,     lw=0.7, alpha=0.9, label='after model')
        ax.set_ylabel('Amplitude (µV)'); ax.set_title(f'Channel {i+1}')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved: {os.path.basename(outpath)}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    path_a, path_b, outdir, label = resolve_paths(args)

    def out(name): return os.path.join(outdir, name)

    time_a, raw_a, _, _ = load_file(path_a)
    time_b, raw_b, _, _ = load_file(path_b)

    colors  = ['#1f77b4', '#ff7f0e']
    short_a = 'APP'
    short_b = 'NUC'

    # ── Filtering ───────────────────────────────────────────────────────────────
    filt_a_raw = apply_filters(raw_a)
    filt_b_raw = apply_filters(raw_b)

    # ── Artifact removal (both APP and NUC) ────────────────────────────────────
    print('--- Artifact removal ---')
    filt_a, counts_a = remove_artifacts(filt_a_raw, label='APP')
    filt_b, counts_b = remove_artifacts(filt_b_raw, label='NUC')

    # ── Artifact removal plots: APP ────────────────────────────────────────────
    plot_artifact_removal(
        time_a, filt_a_raw, filt_a, counts_a,
        'APP: Before vs After Artifact Removal',
        out('app_artifact_removal.png'), colors[0])

    plot_psd_before_after(
        filt_a_raw, filt_a,
        'APP: PSD Before vs After Artifact Removal',
        out('app_psd_artifact_removal.png'), colors[0])

    # ── Artifact removal plots: NUC ────────────────────────────────────────────
    plot_artifact_removal(
        time_b, filt_b_raw, filt_b, counts_b,
        'NUC: Before vs After Artifact Removal',
        out('nuc_artifact_removal.png'), colors[1])

    plot_psd_before_after(
        filt_b_raw, filt_b,
        'NUC: PSD Before vs After Artifact Removal',
        out('nuc_psd_artifact_removal.png'), colors[1])

    # ── Model inference ─────────────────────────────────────────────────────────
    print('--- Model inference ---')
    model = load_model()
    print('Running model on APP...')
    out_a = run_model(model, filt_a)
    print('Running model on NUC...')
    out_b = run_model(model, filt_b)

    # ── Model before/after plots ────────────────────────────────────────────────
    plot_model_before_after(time_a, filt_a[:, :N_CH_OUT], out_a,
                            'APP: Before vs After Model (ch1 & ch2)',
                            out('app_model_before_after_td.png'), colors[0])
    plot_model_before_after(time_b, filt_b[:, :N_CH_OUT], out_b,
                            'NUC: Before vs After Model (ch1 & ch2)',
                            out('nuc_model_before_after_td.png'), colors[1])

    plot_psd_before_after(
        filt_a[:, :N_CH_OUT], out_a,
        'APP: PSD Before vs After Model (ch1 & ch2)',
        out('app_model_before_after_psd.png'), colors[0])
    plot_psd_before_after(
        filt_b[:, :N_CH_OUT], out_b,
        'NUC: PSD Before vs After Model (ch1 & ch2)',
        out('nuc_model_before_after_psd.png'), colors[1])

    # ── Phase-lag alignment ─────────────────────────────────────────────────────
    lag = estimate_lag(filt_a, filt_b)
    filt_a_cmp, filt_b_cmp, time_cmp   = align_for_comparison(filt_a, time_a, filt_b, lag)
    out_a_cmp,  out_b_cmp,  time_m_cmp = align_for_comparison(out_a,  time_a, out_b,  lag)

    # ── Filtered 4-ch comparison ────────────────────────────────────────────────
    plot_td_comparison(
        time_cmp, filt_a_cmp, filt_b_cmp, N_CH,
        f'Time-domain Comparison (Filtered, 4ch, lag={lag:+d} samples)',
        out('time_domain_comparison.png'),
        short_a, short_b, colors)

    plot_psd_comparison(
        filt_a, filt_b, N_CH,
        'PSD Comparison (Filtered, 4ch)',
        out('psd_comparison.png'),
        short_a, short_b, colors)

    # ── Model output comparison ─────────────────────────────────────────────────
    plot_td_comparison(
        time_m_cmp, out_a_cmp, out_b_cmp, N_CH_OUT,
        f'Model Output: Time-domain Comparison (lag={lag:+d} samples)',
        out('model_output_time_domain.png'),
        short_a, short_b, colors)

    plot_psd_comparison(
        out_a, out_b, N_CH_OUT,
        'Model Output: PSD Comparison (ch1 & ch2)',
        out('model_output_psd.png'),
        short_a, short_b, colors)

    # ── STFT / Spectrogram analysis ─────────────────────────────────────────────
    print('--- STFT analysis ---')
    plot_stft_before_after(
        time_a, filt_a[:, :N_CH_OUT], out_a,
        'APP: Spectrogram Before vs After Model (ch1 & ch2)',
        out('app_model_stft.png'), colors[0])

    plot_stft_before_after(
        time_b, filt_b[:, :N_CH_OUT], out_b,
        'NUC: Spectrogram Before vs After Model (ch1 & ch2)',
        out('nuc_model_stft.png'), colors[1])

    plot_stft_comparison(
        time_cmp, filt_a_cmp[:, :N_CH_OUT],
        time_cmp, filt_b_cmp[:, :N_CH_OUT],
        N_CH_OUT,
        f'Spectrogram Comparison — Filtered (lag={lag:+d} samples)',
        out('stft_filtered_comparison.png'),
        short_a, short_b)

    plot_stft_comparison(
        time_m_cmp, out_a_cmp,
        time_m_cmp, out_b_cmp,
        N_CH_OUT,
        f'Spectrogram Comparison — Model Output (lag={lag:+d} samples)',
        out('stft_model_output_comparison.png'),
        short_a, short_b)

    # ── qEEG wellness indices ────────────────────────────────────────────────────
    print('--- qEEG wellness indices ---')
    for ch_idx in range(N_CH_OUT):
        for sig, tag in [(out_a, short_a), (out_b, short_b)]:
            indices = compute_qeeg_indices_windowed(sig[:, ch_idx])
            fname   = f'qeeg_indices_{tag.lower()}_ch{ch_idx+1}.png'
            plot_qeeg_indices(
                indices,
                f'{tag}: qEEG Wellness Indices — Ch{ch_idx+1} (model output)',
                out(fname),
                t_offset=0.0)

    print(f'\nAll outputs saved to: {outdir}')


if __name__ == '__main__':
    main()
