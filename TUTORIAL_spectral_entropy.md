# Tutorial — `spectral_entropy.py`

A practical, end-to-end guide to computing EEG band-structure entropy, left–right
synchrony, and joint-distribution mutual information from lilia-format EEG CSVs.

> This tutorial assumes the standard project layout (`eeg_utils.py`,
> `plot_event_markers.py`, `eeg_quality_v2.py`, `plot_tflite_summary.py` importable
> from the same directory) and a recording that is a 4-row-header lilia CSV (e.g.
> `iBrainCenter/Hardy(SN036)/merged.csv`).

---

## 1. What the script does

`spectral_entropy.py` has **three mutually exclusive analysis modes**, selected by
the flags you pass:

| Mode | Trigger | Question it answers |
| --- | --- | --- |
| **Band entropy** (default) | *(no mode flag)* | How is spectral power distributed across θ/α/β over time, and how "balanced" (high-entropy) is it? Optionally, how synchronised are two hemispheres? |
| **Baseline-vs-event** | `--baseline … --event …` | Does the band-entropy of an *event* interval differ from a *baseline* interval, and is the difference statistically reliable? |
| **Joint MI** | `--joint-mi` | How much information do two channels (e.g. denoised ch1 & ch2) share, read from their 2-D joint probability distribution? |

All modes share the same front-end: load the CSV → apply the project-standard
0.5–45 Hz zero-phase Butterworth bandpass (unless `--no-bandpass`).

---

## 2. Prerequisites

```bash
# Core (all modes)
pip install numpy scipy pandas matplotlib

# Only for --joint-mi --denoise (TinyUNetV4 neural denoiser)
pip install torch        # plus the eeg_denoise package on sys.path (see data_analysis.py)
```

- **Band-entropy / baseline / joint-MI (non-denoise)** need only the core stack.
- **`--clean`** (baseline mode) and **quality masking** (default in band mode) need
  the QC modules (`plot_event_markers`, `eeg_quality_v2`, `plot_tflite_summary`);
  they are imported automatically and degrade gracefully if missing.
- **`--denoise`** additionally needs PyTorch and the `eeg_denoise.tiny_model_v4`
  package, plus the checkpoint `tiny_v4_optimized.pth`.

---

## 3. The signal-processing concepts (1-minute primer)

- **Welch PSD** — power spectral density estimated per window with Hann tapering and
  50 % overlap, giving a stable estimate of how power is spread over frequency.
- **Band energy → proportions** — integrate the PSD over θ (4–8), α (8–13),
  β (13–30) Hz, then normalise to `p_k` that sum to 1. *Delta and gamma are
  deliberately excluded* (motion/sweat drift and EMG/line-noise on a dry-electrode
  wearable), matching the qEEG `relative_powers` definition.
- **Band entropy** — `BandEn = -Σ p_k·log₂(p_k)`. Maximum is `log₂(3) ≈ 1.585 bits`
  (perfectly balanced θ/α/β); low values mean one band dominates.
- **Mutual information (MI)** — `I(X;Y) = H(X) + H(Y) − H(X,Y)`, the bits of
  information two signals share. Estimated from a 2-D histogram (the joint
  distribution). Used two ways: **lagged** (synchrony at τ > 0) and **zero-lag /
  joint** (instantaneous shared information).

---

## 4. Mode 1 — Band entropy (default)

### 4.1 Minimal run

```bash
python spectral_entropy.py --csv iBrainCenter/Hardy(SN036)/merged.csv --ch 1
```

This:
1. bandpass-filters the recording,
2. slides a 2 s window (step = 2 s) and computes θ/α/β proportions + BandEn per window,
3. scores each window with `eeg_quality_v2` and **NaN-masks** windows below quality 0.5,
4. writes a CSV and a multi-panel PNG.

**Outputs** (next to the CSV, or in `--out`):

| File | Contents |
| --- | --- |
| `merged_band_entropy_ch1.csv` | `time_s`, `quality`, per-band `E_*`/`p_*`, `E_total`, `band_entropy`, `band_entropy_norm` |
| `merged_band_entropy_ch1.png` | One panel per band proportion + a BandEn panel (raw + smoothed) |

### 4.2 Add left–right synchrony (lagged MI)

```bash
python spectral_entropy.py --csv merged.csv --ch 1 \
    --sync-pair 1 2 --tau-ms 5 10 15 20
```

For each delay τ it computes the bidirectional average
`0.5·[I(L(t);R(t+τ)) + I(R(t);L(t+τ))]`. Using **τ > 0** (not τ = 0) deliberately
filters out volume-conduction pseudo-synchrony (an instantaneous physical effect)
and captures genuine cross-hemisphere information transfer. Output filenames gain a
`_sync_ch1_ch2` suffix and the PNG gains a synchrony panel.

### 4.3 Absolute time + event overlays

```bash
python spectral_entropy.py --csv iBrainCenter/Ann(SN027)/merged.csv \
    --ch 1 --sync-pair 1 2 --ibrain-events
```

`--ibrain-events` converts the x-axis to local time (UTC+8) and overlays the
iBrainCenter session activity spans/labels on every panel.

### 4.4 Tuning knobs

- `--win` / `--step` — shorter windows = finer time resolution but noisier PSD.
  2 s is a good default (gives ~0.5 Hz frequency resolution at 500 Hz).
- `--quality-threshold` — raise toward 0.6–0.7 to keep only very clean windows;
  lower toward 0.3 if too much is masked.
- `--no-quality-mask` — keep every window (no artefact rejection).
- `--no-bandpass` — analyse the raw signal (rarely what you want).

---

## 5. Mode 2 — Baseline vs event

Compare the band-entropy of two intervals and test the difference. Intervals are in
**seconds relative to the start of the recording**.

```bash
# Contiguous slices
python spectral_entropy.py --csv merged.csv --ch 1 \
    --baseline 0 120 --event 300 420

# Peer-review-grade: micro-epoch each interval and keep only artefact-free epochs
python spectral_entropy.py --csv merged.csv --ch 1 \
    --baseline 0 120 --event 300 420 --clean --quality-threshold 0.5
```

It reports, for each interval:
- **pooled entropy** — entropy of the *averaged* PSD over the interval (the spectral
  shape of the aggregate state, a single headline number), and
- **per-window entropy** mean ± std — the distribution used for statistics.

…and between intervals:
- **ΔEntropy** (event − baseline), and
- a two-sided **Mann–Whitney U** test on the per-window entropy distributions.

> **Why both?** Entropy is non-linear, so `entropy(mean PSD) ≠ mean(entropy)`. The
> pooled value summarises; the U-test on per-window values tells you whether the
> difference is real. Always report them together.

**Output:** `merged_baseline_event_entropy_ch1.csv` (one row per state) + a printed
summary. With `--clean` the console also prints how many epochs were kept vs
rejected (saturated / low-quality).

---

## 6. Mode 3 — Joint probability distribution → mutual information

This is the newest mode. It estimates the **2-D joint distribution `P(X, Y)`** of two
channels and the MI it implies, `I(X;Y) = H(X) + H(Y) − H(X,Y)`.

### 6.1 The requested workflow: denoised ch1 vs denoised ch2

```bash
python spectral_entropy.py --csv merged.csv --joint-mi --denoise
```

`--denoise` runs the **TinyUNetV4 neural denoiser** exactly as `data_analysis.py`
does (filter → resample 500→200 Hz → overlap-add inference, 4 raw channels in → 2
denoised channels out). The two model outputs become X and Y.

### 6.2 Without denoising (bandpassed channels)

```bash
python spectral_entropy.py --csv merged.csv --joint-mi --joint-pair 1 2
```

### 6.3 The one knob that matters most: `--mi-binning`

```bash
# Default — equiprobable bins (recommended for real EEG)
python spectral_entropy.py --csv merged.csv --joint-mi --joint-pair 1 2 --mi-binning quantile

# Legacy equal-width bins (for comparison only)
python spectral_entropy.py --csv merged.csv --joint-mi --joint-pair 1 2 --mi-binning uniform
```

**Why quantile is the default:** dry-electrode EEG is heavy-tailed. With equal-width
(`uniform`) bins, occasional large artefacts stretch the histogram range, so the bulk
of the data collapses into a few central bins — the marginal entropies fall far below
their `log₂(bins)` ceiling and MI is badly **under-estimated**. On real data this was
a ~5× difference:

| `--mi-binning` | H(X) (of 4-bit max) | I(ch1;ch2) |
| --- | --- | --- |
| `uniform` | 1.33 bits | 0.025 bits |
| `quantile` | 4.00 bits | **0.133 bits** |

Quantile binning places per-axis edges at data quantiles so every bin holds ≈equal
mass — robust to artefacts and resolution-maximising.

### 6.4 Significance and bias

- Plug-in histogram MI is **positively biased**, so a non-zero value alone is not
  evidence of coupling. The summary therefore also reports the **Miller–Madow**
  bias-corrected MI.
- `--mi-surrogates 200` (default) runs a **circular-shift surrogate null**: shifting
  Y destroys cross-channel coupling while preserving each channel's autocorrelation,
  giving a p-value and z-score. Set `--mi-surrogates 0` to skip it.

### 6.5 Outputs

| File | Contents |
| --- | --- |
| `merged_joint_mi_<pair>_summary.csv` | One row: MI (plug-in / Miller–Madow / normalised), per-channel & joint entropies, bin config, surrogate mean/std/p/z. `<pair>` = `denoised_ch1_ch2` or `ch1_ch2`. |
| `merged_joint_mi_<pair>_distribution.png` | The joint `P(X,Y)` heatmap (see §6.6) + marginals + MI annotation. |
| `merged_joint_mi_<pair>_timeseries.csv` / `.png` | Sliding-window zero-lag MI over time. |
| `merged_joint_mi_<pair>_events.csv` / `.png` | *(only with `--ibrain-events`, see §6.7)* Per-event pre-event vs onset joint MI + ΔMI. |

### 6.6 How to read the joint-distribution heatmap

The heatmap is drawn in **bin-index (copula / rank) space**, where every cell is the
same size. With quantile binning the marginals are flat *by construction*, so:

- **Uniform, featureless plot** → the two channels are ~independent (no shared info).
- **Bright ridge along the diagonal** → positive dependence (channels rise/fall
  together) — this is the structure MI quantifies.
- **Bright corners (top-right + bottom-left)** → the channels co-activate at their
  extremes.

The annotation box reports `I(X;Y)`, the Miller–Madow `I_MM`, normalised MI, the
three entropies, N, and the surrogate p-value/z.

### 6.7 Event-aligned analysis: pre-event vs onset (`--ibrain-events`)

Adding `--ibrain-events` to joint-MI mode turns it into an *event-aware* analysis,
so you can see how interhemispheric coupling shifts around each iBrainCenter
activity onset:

```bash
# Denoised ch1 vs ch2, aligned to the session's events
python spectral_entropy.py --csv iBrainCenter/Hardy(SN036)/merged.csv \
    --joint-mi --denoise --ibrain-events
```

It does two things:

1. **Overlays events on the MI time series.** The zero-lag MI plot's x-axis becomes
   absolute local time (UTC+8) and every event span/label is drawn on top, so you can
   read the MI trend against the session timeline.
2. **Compares the joint distribution pre-event vs at onset.** For each event starting
   at `onset`, two intervals are taken — `[onset − 30 s, onset)` (pre-event) and
   `[onset, onset + 30 s)` (onset) — and each yields its own joint `P(X,Y)` and MI.
   `ΔMI = onset − pre` quantifies the change in coupling at the onset.

> **How event times are aligned:** the event start times (`HH:MM`, defined in
> `plot_event_markers.EVENTS`) are converted to absolute UTC µs and matched against
> the recording's own timestamps, so this only makes sense for recordings from that
> iBrainCenter session. Events that fall outside the recording, or lack a full
> pre/onset window, are skipped automatically.

**Extra outputs** (alongside the usual joint-MI files):

| File | Contents |
| --- | --- |
| `merged_joint_mi_<pair>_events.csv` | One row per event: pre/onset MI (plug-in, Miller–Madow, normalised), sample counts, and `ΔMI`. |
| `merged_joint_mi_<pair>_events.png` | Paired bar chart of pre-event vs onset MI per event, with `ΔMI` annotated. |

**Reading the result:** a positive `ΔMI` means the two channels share *more*
information at the event onset than just before it. On the Hardy(SN036) denoised
example, the physically demanding / coordination tasks show the largest jumps
(Cone Rotation ≈ +0.90, Agility Ladder ≈ +0.79, Push-ups ≈ +0.43 bits), i.e.
interhemispheric coupling rises sharply when the task begins.

---

## 7. Full parameter reference

| Flag | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `--csv` | *(required)* | all | Input lilia CSV (4-row header) |
| `--fs` | 500 | all | Sampling rate (Hz) |
| `--ch` | 1 | band, baseline | 1-based channel to analyse |
| `--win` | 2 | all | Analysis window length (s) |
| `--step` | =`--win` | band, joint | Sliding step (s) |
| `--no-bandpass` | off | all | Skip the 0.5–45 Hz bandpass |
| `--baseline START END` | — | baseline | Baseline interval (s from start) |
| `--event START END` | — | baseline | Event interval (s from start) |
| `--clean` | off | baseline | Keep only artefact-free micro-epochs |
| `--quality-threshold` | 0.5 | band, baseline | Quality keep/mask threshold (0–1) |
| `--no-quality-mask` | off | band | Disable per-window quality masking |
| `--sync-pair L R` | — | band | Add lagged-MI synchrony for a channel pair |
| `--tau-ms` | `5 10 15 20` | band(sync) | Positive lags (ms) for lagged MI |
| `--mi-bins` | 16 | sync, joint | Histogram bins for MI |
| `--joint-mi` | off | *(selects joint mode)* | Estimate joint distribution + MI |
| `--joint-pair X Y` | `1 2` | joint | Channel pair when **not** denoising |
| `--denoise` | off | joint | Use the 2 TinyUNetV4 denoised channels |
| `--mi-binning` | `quantile` | joint | `quantile` (recommended) or `uniform` |
| `--mi-surrogates` | 200 | joint | Surrogate count for significance (0 = off) |
| `--out` | CSV's dir | all | Output directory |
| `--ibrain-events` | off | band, joint | Absolute-time x-axis + event overlays; in joint mode also adds the pre-event vs onset comparison (§6.7) |

---

## 8. Programmatic use (import as a library)

Every computation is a plain function you can call directly:

```python
import numpy as np
import spectral_entropy as se

# Band entropy of one channel
res = se.compute_band_entropy_windowed(x, fs=500.0, win_sec=2.0)
print(res['band_entropy'].mean())

# Joint distribution + MI between two channels (equiprobable binning)
joint = se.compute_joint_probability(ch1, ch2, bins=16, binning='quantile')
print(joint['mutual_information'], joint['mutual_information_mm'])

# Surrogate significance
sig = se.compute_joint_mi_significance(ch1, ch2, bins=16, binning='quantile')
print(sig['p_value'], sig['z'])

# Denoise then compute MI (needs torch + eeg_denoise)
den, fs_out = se.denoise_channels(raw_4ch, fs=500.0)   # (N, 2) @ 200 Hz
joint = se.compute_joint_probability(den[:, 0], den[:, 1], binning='quantile')

# Per-event pre-event vs onset joint MI (needs plot_event_markers + time_us epoch)
events = se.compute_event_pre_onset_joint_mi(
    ch1, ch2, time_us_epoch=int(time_us[0]), fs=500.0, binning='quantile')
for e in events:
    print(e['name'], e['pre_mi'], e['onset_mi'], e['delta_mi'])
```

---

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `--clean requires plot_event_markers / eeg_quality_v2 …` | QC modules not importable; run from the project root. |
| `Neural denoising requires data_analysis (PyTorch …)` | Install `torch` and put the `eeg_denoise` package on `sys.path` (see `data_analysis.py`). |
| Joint heatmap collapses to one bright cell / very low MI | You used `--mi-binning uniform` on artefacty data — switch to `quantile`. |
| "no clean … epochs in the … interval" | Interval too short or too noisy; lengthen it or lower `--quality-threshold`. |
| Most windows masked (band mode) | Lower `--quality-threshold`, or inspect the raw signal with `check_quality_anomalies.py`. |
| `channel N not found` | `--ch` / `--joint-pair` / `--sync-pair` is 1-based; check the file's channel count. |

---

## 10. Conclusion

`spectral_entropy.py` turns a raw lilia EEG recording into three complementary,
peer-review-defensible views of cortical state, all on the same θ/α/β,
quality-controlled front-end:

1. **Band entropy** — *how balanced* the spectrum is over time (and, with
   `--sync-pair`, genuine cross-hemisphere coupling via lagged MI);
2. **Baseline-vs-event** — *whether a state change is real*, via pooled ΔEntropy plus
   a non-parametric per-window test, with an artefact-rejecting `--clean` path;
3. **Joint MI** — *how much information two channels share*, read from their joint
   probability distribution, optionally on TinyUNetV4-denoised channels.

The single most important practical lesson is **binning**: on heavy-tailed
dry-electrode EEG, equiprobable (`quantile`) binning is essential — equal-width bins
silently under-estimate mutual information several-fold. Combined with the
Miller–Madow bias correction and a circular-shift surrogate null, the joint-MI mode
reports an MI value you can actually trust and defend.

**Quick-start cheat sheet:**

```bash
# Band entropy + synchrony, with event overlays
python spectral_entropy.py --csv merged.csv --ch 1 --sync-pair 1 2 --ibrain-events

# Baseline vs event, clean epochs
python spectral_entropy.py --csv merged.csv --ch 1 --baseline 0 120 --event 300 420 --clean

# Joint MI on denoised ch1 vs ch2 (recommended settings are the defaults)
python spectral_entropy.py --csv merged.csv --joint-mi --denoise

# Joint MI aligned to events: time-series overlay + pre-event vs onset comparison
python spectral_entropy.py --csv merged.csv --joint-mi --denoise --ibrain-events
```
