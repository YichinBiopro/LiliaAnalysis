"""Shared global constants across lilia_analysis scripts."""

# ── Audio / EEG sample rate ────────────────────────────────────────────────────
FS = 500                    # Hz — standard EEG sample rate (500 Hz)
DOWNSAMPLED_FS = 200        # Hz — EEG downsampled rate for model input

# ── Neural network (TinyUNetV4) ────────────────────────────────────────────────
N_CH = 4                    # model input channels (4-channel headset)
N_CH_OUT = 2                # model output channels (denoised L/R)
TFLITE_FS = 200             # Hz — model input sample rate (@ 200 Hz)
TFLITE_WIN = 400            # samples — model input window (400 @ 200 Hz = 2 s)

# ── Bandpass filter cutoffs (canonical project-wide standard) ─────────────────
BP_LOW = 0.5                # Hz — bandpass lower cutoff
BP_HIGH = 45.0              # Hz — bandpass upper cutoff

# ── EEG quality scoring ────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.5     # quality score threshold (0-1 scale)

# ── Visualization colors ──────────────────────────────────────────────────────
CH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Standard 4-channel colors
