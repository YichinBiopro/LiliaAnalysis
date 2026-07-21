# An Academic Report on `spectral_entropy.py`: Band-Structure Entropy, Spectral Divergence, and Information-Theoretic Coupling in Dry-Electrode EEG

**Subject of analysis:** `spectral_entropy.py` (2,355 lines), part of the *lilia_analysis* qEEG pipeline.
**Scope:** Data preprocessing rationale; parameterisation of all executable modes; systematic interpretation and critique of the generated visualisations; and formal academic conclusions linking the implementation to signal-processing and information-theoretic theory.

---

## 0. Preamble: The Analytical Object and Its Epistemic Stance

`spectral_entropy.py` is not a stand-alone estimator but a *consumer* of a shared, project-wide qEEG front-end. Its design philosophy is explicitly **comparability-first**: every numerical choice (passband, band set, window length, quality definition) is inherited from, or deliberately aligned with, the canonical pipeline (`plot_tflite_summary.py`, `plot_event_markers.py`, `eeg_utils.py`, `data_analysis.py`). The scientific consequence is important and should frame the entire report: the script's outputs are designed to be read *next to* the Flow/Focus/Calm/Relax indices, not as an independent ground truth. Any interpretation of its results inherits both the strengths and the systematic biases of that front-end.

The module computes four families of quantities, each answering a distinct question:

| Quantity | Mathematical object | Question answered |
|---|---|---|
| **Band entropy** (`BandEn`) | Shannon entropy `H(P)` of the θ/α/β power split | *How spectrally flat/balanced is this state?* |
| **State entropy** (pooled vs. per-window) | `H` of a pooled PSD vs. a distribution of per-window `H` | *What is the entropy of a whole interval, and how variable is it?* |
| **BASD** | KL divergence `D_KL(P_event ‖ P_base)` | *How far, and in which direction, did the spectrum move from rest?* |
| **Mutual information** (zero-lag & lagged) | `I(X;Y)` from a joint histogram | *How much do two channels/hemispheres share information?* |

These are not redundant: entropy is a *single-state scalar blind to band identity*; KL divergence is *directed and band-aware*; mutual information is *bivariate*. The script's own docstrings (lines 792–811) articulate precisely why all three are needed, and this report treats that distinction as its central organising theme.

---

## 1. Data Preprocessing Documentation

Signal analysis is only as defensible as the preprocessing that precedes it. The script chains together six preprocessing stages, each with an explicit academic justification. I document them in execution order.

### 1.1 Ingestion: `load_merged_csv` — the 4-row-header lilia format

```python
df = pd.read_csv(path, skiprows=4)
time_us = df.iloc[:, 0].values.astype(np.int64)
data    = df.iloc[:, 1:].values.astype(np.float32)
```

**What it does.** The lilia CSV carries a 4-row metadata header (device, montage, units, etc.) that is discarded (`skiprows=4`). Column 0 is parsed as **absolute UTC Unix microseconds** (`int64`), the remaining columns as floating-point EEG channel values.

**Rationale.**
- *Microsecond integer timestamps* (`int64`) are retained rather than reconstructed from a nominal sampling rate. This is methodologically significant: consumer dry-electrode devices exhibit clock jitter and occasional dropped samples, so an *index × 1/fs* reconstruction would silently accumulate drift. By carrying true epoch-µs, the script can later place windows on an **absolute local-time axis** (UTC+8, §1.6) and align them to externally-timestamped iBrainCenter events — an alignment that would be impossible with relative indices alone.
- *`float32` storage* matches the device's effective ADC resolution and halves memory relative to `float64`; all downstream arithmetic upcasts to `float64` inside the analysis functions (`np.asarray(..., dtype=float)`), so precision is preserved where it matters (entropy logs, PSD integration).

### 1.2 The zero-phase Butterworth bandpass (0.5–45 Hz) — the canonical front-end

```python
sos = butter(order=4, [0.5, 45], btype='bandpass', fs=fs, output='sos')
return sosfiltfilt(sos, data, axis=0)
```

This is the single most consequential preprocessing decision, and four separate design choices deserve scholarly justification.

**(a) Why bandpass at all, and why 0.5–45 Hz.** The lower cutoff at **0.5 Hz** removes electrode-drift, DC offset, sweat potentials and respiration-coupled baseline wander — all of which are large-amplitude, low-frequency contaminants that on a dry-electrode wearable would otherwise dominate the PSD and inflate the (discarded) delta band. The upper cutoff at **45 Hz** sits just below the 50/60 Hz mains line and suppresses broadband EMG. Critically, these cutoffs are **imported from `plot_event_markers.BP_LOW/BP_HIGH`** (lines 150–156) rather than hard-coded, with a documented fallback. This makes the passband a *single source of truth* across the pipeline: spectral entropy is computed on exactly the same passband as the Flow/Focus/Calm/Relax indices, eliminating a class of "different-filter" confounds that routinely undermine cross-measure EEG comparisons.

**(b) Why zero-phase (`sosfiltfilt`).** Forward-backward filtering yields **zero phase distortion**. For spectral *power* estimation phase is irrelevant, but zero-phase filtering matters greatly for the **lagged mutual-information synchrony** path: a causal IIR filter imposes a frequency-dependent group delay that would masquerade as inter-hemispheric lag, biasing the `τ`-resolved coupling estimates. `filtfilt` guarantees that any lag the script reports is in the signal, not in the filter. The cost — non-causality — is acceptable because this is offline batch analysis, not real-time.

**(c) Why SOS (second-order sections).** `output='sos'` with `sosfiltfilt` is numerically far more stable than the transfer-function (`b, a`) form for higher-order Butterworth filters, where pole-zero clustering near the unit circle causes catastrophic coefficient round-off. Order 4 (effective order 8 after the forward-backward pass) gives a sufficiently sharp roll-off without ringing.

**(d) The deliberate *absence* of resampling.** The module's header (lines 14–17) makes a pointed methodological distinction from the TFLite branch: **no 500→200 Hz downsampling is applied.** The justification is sound and worth emphasising — spectral entropy is a *direct PSD measurement*, not a model input, so the priority is **frequency resolution**, which is maximised by retaining the full 500 Hz record. Downsampling would buy nothing here (the 45 Hz passband is far below even the 100 Hz Nyquist of 200 Hz sampling) while sacrificing the finer Welch frequency grid. This is the correct trade-off and is explicitly contrasted with the denoiser path (§1.5), which *does* resample because the neural model was trained at 200 Hz.

**Toggle.** `--no-bandpass` disables the entire stage (analysing raw channels). This exists for diagnostic/ablation purposes only; under it the delta region re-enters via spectral leakage and all entropy values should be treated as uncalibrated.

### 1.3 Per-window detrending inside Welch — the second-stage drift guard

```python
signal.welch(..., detrend='constant', window='hann', scaling='density')
```

Even after the 0.5 Hz high-pass, each short analysis window can carry a residual DC offset. `detrend='constant'` removes the per-segment mean before the FFT, preventing a spurious zero-frequency spike from leaking into the lowest bins. The **Hann window** is the standard choice for reducing spectral leakage (the −31 dB sidelobe roll-off suppresses the broadband smearing that a rectangular window would produce on non-periodic EEG segments). `scaling='density'` returns a PSD in µV²/Hz, so band powers obtained by trapezoidal integration carry consistent physical units (µV²).

### 1.4 Artefact rejection for the `--clean` path — a two-tier QC definition

The peer-review-grade baseline/event path (`collect_clean_epochs`, lines 604–656) imposes a **two-stage** rejection that is deliberately identical to the qEEG baseline builder, so "clean" means one thing across the whole pipeline:

1. **ADC-saturation guard on the *raw* signal** (`_saturation_frac > SAT_FRAC_MAX = 0.02`). This must be done **pre-filter**, and the script is careful to retain the raw copy precisely for this purpose (line 2190). The rationale is subtle and correct: the bandpass filter *smears* hard clipping across neighbouring samples, so a window that was saturated in the raw ADC stream no longer "looks" clipped after filtering. Detecting saturation on the filtered signal would therefore miss exactly the artefacts it is meant to catch. Rejecting any window with >2 % clipped samples removes motion-induced rail-to-rail excursions.
2. **`eeg_quality_v2` parametric scoring on the *filtered* signal**, keeping a window only if the **channel-median** `overall` score ≥ `--quality-threshold` (default 0.5). This is a learned/parametric quality index tuned for the iBrainCenter device (`get_ibrain_device_eeg_quality_v2_params`).

**Why two tiers, and why median.** Saturation is a hard, physically-defined failure; quality scoring is a soft, statistical judgement. Combining a hard physical guard with a soft learned score catches both gross clipping and subtler contamination (high-frequency EMG, poor electrode contact). The **channel-median** reduction (rather than mean or min) is robust: a single bad channel cannot condemn an otherwise clean epoch, nor can a single good channel rescue a bad one.

**Determinism caveat, well-handled.** The docstring (lines 626–629) notes that unlike `build_baseline_epochs`, this routine keeps **all** clean epochs with **no blind subsampling**. This is the right call for two reasons: (i) there is no random subset to introduce sampling bias, so the procedure is fully reproducible; (ii) using every QC-passing epoch *maximises statistical power* for the downstream Mann–Whitney test.

### 1.5 The neural-denoise preprocessing chain (`--denoise`) — a parallel, heavier front-end

When the joint-MI mode is run with `--denoise`, a different and heavier preprocessing chain is invoked (`denoise_channels`, lines 1435–1469), mirroring `data_analysis.apply_filters` exactly:

1. **Bandpass → 60 Hz IIR notch (Q=30) → 33.25 Hz Butterworth bandstop (BW=1 Hz).** The notch removes mains; the narrow 33.25 Hz bandstop removes a **device-specific switching/artefact peak** — a piece of domain knowledge baked into the canonical pipeline.
2. **Resample 500 → 200 Hz**, because the TinyUNetV4 model was *trained at 200 Hz*; feeding it 500 Hz data would be a train/test sampling-rate mismatch.
3. **Overlap-add inference** (4 raw channels → 2 denoised channels), 400-sample model window.

**Rationale.** This path exists so that information-theoretic coupling can be measured on the *same denoised channels the rest of the project uses*. The key academic point is **consistency of the analytical substrate**: MI between denoised channels is only interpretable relative to the project if those channels are produced identically. The script enforces this by reusing `data_analysis`'s exact filter/resample/inference functions rather than re-implementing them.

### 1.6 Time-base construction and timezone localisation

Finally (lines 121–135), epoch-µs are converted to **naive local datetimes at UTC+8 (Asia/Taipei)** for plotting and for aligning to iBrainCenter event start times. This is presentation-layer preprocessing, but it is what makes the event-overlay and pre-event/onset analyses possible. The conversion is explicit and centralised, avoiding the scattered, error-prone timezone arithmetic that often corrupts wearable-EEG event alignment.

**Summary of preprocessing philosophy.** Every stage is justified by either (i) *artefact physics* (drift, clipping, EMG, mains, device peak), (ii) *estimator correctness* (zero-phase for lag, detrend for leakage, SOS for numerical stability), or (iii) *pipeline comparability* (imported cutoffs, shared QC definition, identical denoiser). The recurring and commendable theme is that nothing is chosen ad hoc: the front-end is a *shared contract*.

---

## 2. Parameter Analysis of the Executable Modes

The script exposes **four mutually-routed execution modes** (dispatched in `main`, lines 2179–2351) plus an optional synchrony sub-mode. I analyse each mode, its parameters, and the signal-processing reasoning behind the parameterisation.

### 2.0 Global parameters (shared by all modes)

| Parameter | Default | Signal-processing rationale |
|---|---|---|
| `--csv` | (required) | Input lilia CSV. |
| `--fs` | 500 Hz | Native device rate; sets the Welch frequency grid and lag→sample conversion. |
| `--ch` | 1 (1-based) | Channel to analyse for the univariate entropy modes. |
| `--win` | 2.0 s | Analysis window. **The central time-frequency trade-off.** |
| `--step` | = `--win` | Hop size; default gives non-overlapping windows. |
| `--no-bandpass` | off | Ablation toggle (see §1.2). |
| `--quality-threshold` | 0.5 | Both the masking threshold and the `--clean` keep threshold. |
| `--no-quality-mask` | off | Disables NaN-masking of low-quality windows. |
| `--out` | CSV's dir | Output directory. |
| `--ibrain-events` | off | Absolute-time axis + event overlays. |

**The window-length decision (`--win = 2 s`) deserves extended treatment.** A 2-second window at 500 Hz gives `nperseg = min(1000, round(fs)) = 500` (lines 209–217), i.e. Welch uses ~1 s sub-segments with 50 % overlap, yielding a **frequency resolution of ~1 Hz** and roughly two averaged sub-segments per window. This is a deliberate compromise:

- **Too short** (< 1 s) and the 1 Hz resolution would be insufficient to cleanly separate the θ (4–8), α (8–13) and β (13–30) bands — the narrowest, α, is only 5 Hz wide, so sub-Hz resolution is needed to avoid edge-bin contamination.
- **Too long** (> 5 s) and the stationarity assumption underlying both Welch averaging *and* the per-window entropy series breaks down: cognitive/affective state can shift within a long window, blurring the very dynamics the time series is meant to resolve.

2 s sits at the knee of this trade-off and is consistent with the qEEG window family. The automatic `nperseg`/`noverlap` derivation (`_default_welch_params`) ensures the estimator never fails on short inputs while always using Hann-windowed, 50 %-overlap Welch averaging — the textbook variance-reduction configuration.

**The quality-threshold = 0.5 default** is inherited from `plot_event_markers.QUALITY_THRESHOLD`. Setting masking and `--clean` keep to the *same* constant is a quiet but important consistency choice: the windows that vanish from the time-series plot are precisely the windows the `--clean` estimator would also discard.

### 2.1 Mode A — Windowed band entropy (default)

**Entry:** no `--joint-mi`, no `--baseline/--event`. **Core:** `compute_band_entropy_windowed` → `plot_band_entropy`.

**The band set is the defining parameter** (`BAND_DEFINITIONS`, lines 177–181): **θ (4–8), α (8–13), β (13–30) only.** Delta (0.5–4) and gamma (30–45) are *computed by the filter but excluded from the entropy*. The script devotes a 20-line comment (lines 158–176) to this, and the justification is the strongest methodological argument in the file:

1. **Comparability:** `qeeg_indices.compute_relative_powers` already normalises over θ/α/β only; the Flow index is a function of θ/α/β alone. Computing entropy over the same three bands keeps `BandEn` directly comparable to Flow/Focus/Calm/Relax rather than mixing in bands the indices never see.
2. **Artefact physics:** on a dry-electrode wearable *during active tasks*, delta is dominated by movement/sweat/drift and gamma by EMG and the line-noise approach — both are unreliable cortical estimates.

The consequence is that **maximum entropy is `log2(3) ≈ 1.585 bits`** (not `log2(5) ≈ 2.322`). The script also reports a **normalised entropy** (`band_entropy_norm`, ÷`log2(3)`), bounding the measure to [0, 1] for cross-condition comparison. Both raw-bits and normalised forms are emitted because the absolute-bits form preserves the physical scale while the normalised form aids comparison — a defensible dual reporting.

**The smoothing parameter** (`_smooth_series`, window=5) applies a NaN-aware centred moving average that explicitly **re-masks** discarded windows (lines 205) so a low-quality gap is never "filled in" from neighbours. This mirrors `plot_tflite_summary` and is the correct way to smooth a series with rejected samples.

**Quality masking** (lines 2270–2291) scores each window with `eeg_quality_v2` on the *exact same window grid* as the entropy (guaranteeing 1:1 alignment, lines 659–697) and NaN-masks sub-threshold windows in the band, proportion, energy, entropy and synchrony arrays — while leaving the `time`/`quality` axes whole so a discarded window still carries a timestamp.

### 2.2 Sub-mode — Lagged inter-hemispheric synchrony (`--sync-pair`)

**Parameters:** `--sync-pair LEFT RIGHT`, `--tau-ms 5 10 15 20`, `--mi-bins 16`.

This optional add-on to Mode A computes **non-zero-lag mutual information** `I(L(t); R(t+τ))` and its reverse `I(R(t); L(t+τ))`, averaging the two into a **symmetric** synchrony estimate per τ (lines 1226–1322).

**Why non-zero lag, and why these τ.** Zero-lag coupling between hemispheres is confounded by **volume conduction / common reference** — a single source projecting to both electrodes produces instantaneous correlation that is *not* genuine inter-hemispheric communication. By using strictly **positive delays** (5–20 ms), the script targets coupling that respects a physiological transmission delay, a standard trick (analogous in spirit to the imaginary-coherence rationale) for suppressing the zero-lag volume-conduction artefact. The 5–20 ms range brackets plausible cortico-cortical/callosal conduction times. Per-window the script records the mean, the max, and the **best τ** (argmax), so a dominant lag can be read off.

**Why MI rather than correlation.** Mutual information captures **non-linear** dependence that Pearson correlation misses — appropriate because cross-hemispheric EEG coupling is not guaranteed to be linear.

### 2.3 Mode B — Baseline-vs-event entropy (`--baseline … --event …`)

**Entry:** both `--baseline START END` and `--event START END` supplied. **Optional:** `--clean`. **Core:** `compare_baseline_event` / the `--clean` epoch path → Mann–Whitney U.

This mode reduces an interval to entropy in **two methodologically distinct ways**, and the script's insistence on reporting both is its most statistically mature feature:

- **`pooled_entropy`** — entropy of the *averaged* PSD over the interval (a longer Welch average). Summarises the aggregate spectral shape in one number.
- **`mean_window_entropy ± std`** — the mean of *per-window* entropies.

The docstring (lines 448–476) correctly states the crucial fact that **`entropy(mean PSD) ≠ mean(entropy)`** because entropy is non-linear (Jensen's inequality). Conflating them is a common error; the script refuses to. The pooled value is the headline; the **per-window distribution** is what the statistical test operates on.

**The test choice — Mann–Whitney U** (`_finalise_comparison`, lines 712–739) — is non-parametric, justified because window entropies are *bounded* (∈ [0, log2 3]) and not guaranteed normal. This is the correct choice over a t-test.

**`--clean` vs. contiguous.** Without `--clean`, intervals are taken as contiguous slices. With `--clean`, each interval is micro-epoched and only artefact-free epochs survive (§1.4); their PSDs are averaged and **no FFT window ever straddles a discarded span** (lines 567–601) — an important guarantee, since splicing non-contiguous epochs and then FFT-windowing across the join would inject spurious high-frequency energy.

The mode prints explicit **rigour caveats** (stationarity/pooling, comparability, coarse-distribution) and reports ΔEntropy *together with* the test, never the pooled value alone — exactly the discipline a reviewer would demand.

### 2.4 Mode C — Baseline-Anchored Spectral Divergence (BASD)

BASD (`compute_basd`, lines 880–957) is the script's answer to a genuine limitation of entropy. The motivating argument (lines 792–811) is worth restating because it is correct and elegant:

> Shannon entropy is **blind to band identity**: `(θ=0.7, α=0.2, β=0.1)` and `(θ=0.1, α=0.2, β=0.7)` have *identical* entropy. For a baseline→event contrast this is exactly wrong — an eyes-closed shift moving mass from β into α barely changes entropy yet is the whole signal of interest.

The fix is the **Kullback–Leibler divergence** `D_KL(P_event ‖ P_base) = Σ P_event(k) log2[P_event(k)/P_base(k)]` in bits. Its properties are leveraged deliberately:

- **Directed / asymmetric:** event-first, so the expectation is taken *under the event* — "how surprising is the event under the resting prior." This asymmetric, baseline-anchored reading is the semantic content of the name.
- **Band-aware:** a θ→α→β redistribution registers even when flatness is preserved.
- **Anchored at 0** iff the distributions are identical, strictly positive otherwise.

**Parameters.** `epsilon = 1e-9` (`DEFAULT_BASD_EPSILON`) is **Laplace smoothing** added to every band power before normalisation (lines 846–856), guaranteeing the log and the ratio stay finite when a band collapses to zero — a standard and necessary regularisation, here kept tiny relative to the unit-normalised mass so it barely perturbs non-degenerate distributions. `time_resolved` toggles between a single scalar BASD (block-averaged event) and a **per-window BASD series** against a shared prior (with mean/std/peak). The baseline is *always* collapsed to a single prior `P_base`, which is the conceptually correct anchor.

The docstring's closing instruction — *report BASD alongside `p_event − p_base`, since the scalar alone does not name the direction* — is the right caveat: a divergence magnitude without the direction vector is half a result.

### 2.5 Mode D — Joint distribution & mutual information (`--joint-mi`)

**Entry:** `--joint-mi`. **Channel source:** `--denoise` (TinyUNetV4 outputs) **or** `--joint-pair CH_X CH_Y` (default 1 2). **Estimator parameters:** `--mi-bins 16`, `--mi-binning quantile|uniform` (default quantile), `--mi-surrogates 200`.

This mode estimates the full **2-D joint PMF** `P(X,Y)` and reads MI off it as `I = H(X) + H(Y) − H(X,Y)` (lines 1061–1181). The parameterisation reflects deep awareness of histogram-MI's pathologies:

**(a) Binning strategy — the headline parameter.** Default `quantile` (equiprobable) binning places per-axis edges at **data quantiles**, so every marginal bin holds ≈ the same sample count. The justification (lines 1033–1058) is essential for dry-electrode EEG: equal-width (`uniform`) bins **collapse** on heavy-tailed signals — a few artefact-driven outliers inflate the range, and almost all samples fall into a handful of central bins, badly *under*-estimating MI. Quantile binning is effectively a **copula/rank transform**, making MI robust to the marginal distribution's shape and to monotone artefacts. Degenerate quantiles (ties/saturation) collapse via `np.unique`, gracefully lowering the effective bin count. (`uniform` is retained as the default only for the legacy lagged-MI path, for backward compatibility.)

**(b) Bin count = 16.** A bias-variance compromise: more bins resolve finer dependence structure but worsen the positive bias of plug-in MI (which grows with bins and shrinks with N). 16 bins on a multi-minute recording (N large) keeps each of 16×16 = 256 joint cells reasonably populated under quantile edges.

**(c) Bias correction — Miller–Madow.** The script correctly notes that the **plug-in (ML) entropy estimator is negatively biased, so plug-in MI is *positively* biased** — a non-zero MI is not by itself evidence of dependence. It therefore reports `Ĥ_MM = Ĥ + (m̂−1)/(2N)` per entropy (m̂ = occupied bins, /ln2 → bits, lines 1152–1161) and a Miller–Madow-corrected MI alongside the plug-in value.

**(d) Significance — circular-shift surrogates** (`--mi-surrogates 200`, lines 1387–1432). Because bias persists, the script builds a **null distribution** by circularly shifting `Y` by a random offset (avoiding trivial near-zero shifts). This **destroys cross-channel coupling while preserving each channel's own autocorrelation and amplitude distribution** — the gold-standard surrogate construction for a coupling null. It reports a one-sided, add-one-smoothed p-value and a z-score. This trio (plug-in + Miller–Madow + surrogate null) is exactly the rigour an information-theoretic EEG result requires.

**(e) Normalised MI** = `I / min(H(X), H(Y))`, bounding coupling to [0,1] for cross-pair comparison.

**Event-aware sub-analysis.** Under `--ibrain-events`, the mode adds a **pre-event vs. onset** comparison (`compute_event_pre_onset_joint_mi`, lines 1472–1527): for each event it estimates `P(X,Y)` over [onset−30 s, onset) vs. [onset, onset+30 s) and reports ΔMI = onset − pre, visualised as a grouped bar chart. This turns a global coupling number into an **event-locked dynamic**.

---

## 3. Systematic Synthesis and Critique of the Visualised Results

The script originally produced **four figure types**; the optimisation work in this report adds five more (composition, ternary, excess-mass, surrogate inset, peri-event). For each original figure I give (i) a reading guide and (ii) a critical evaluation — and each "Critique" block is now annotated with the **implemented** optimisation and its verified result, with the new figures collected in §3.6.

### 3.1 The band-entropy stack (`plot_band_entropy`)

**Layout.** A vertical stack of shared-x panels: three band-proportion panels (θ blue, α green, β vermillion — the colour-blind-safe Wong palette, *updated*), one **BandEn** panel (black = bits with a grey ±1σ band, *updated*; red = normalised), and — if `--sync-pair` — a fifth synchrony panel (per-τ blue traces + black mean + dashed-red max + a best-τ scatter on a twin axis, *updated*). Each panel overlays a faint raw trace and a bold smoothed trace; iBrainCenter event spans are shaded with rotated labels when `--ibrain-events` is set. Every figure now carries a provenance footer.

**How to read it correctly.**
- The three proportion panels are **compositional** — they sum to 1 at every time point. A rise in one *must* be compensated by a fall in another; never read a single band panel in isolation.
- **BandEn (bits, black)** near `log2(3) ≈ 1.585` ⇒ flat/balanced θ≈α≈β spectrum (the y-limit is set to this ceiling, lines 1738). Low BandEn ⇒ one band dominates. The **normalised red trace** is the same information rescaled to [0,1].
- In the sync panel, the **dashed-red max** rising above the black mean indicates that coupling is concentrated at one particular lag; reading *which* lag requires the `best_tau` CSV column, which the plot does not draw.
- **Gaps** in any trace are quality-masked (NaN) windows — *absence of data*, not zero.

**Critique and optimisation opportunities — *implemented and verified* (see §3.6).**
1. ✅ **Smoothing window exposed as `--smooth N`.** `_smooth_series` now reads a module-level `SMOOTH_WINDOW` set from the CLI (default 5); the effective duration is N×step seconds and the value is printed in every figure footer. Verified with `--smooth 9` on the Hardy recording.
2. ✅ **±1σ dispersion band on BandEn.** A new NaN-aware `_rolling_std` shades a grey ±1σ band around the smoothed BandEn (`fill_between`), so a real entropy excursion is separable from window-to-window jitter. Masked windows leave gaps in the band, never interpolated.
3. ✅ **`best_tau` now plotted.** The dominant-lag series is drawn as a faint scatter on a twin y-axis of the synchrony panel ("best τ (ms)"), exposing whether the coupling lag drifts over time.
4. ✅ **Stacked-area composition view added** (`plot_band_composition`, `*_composition.png`): the θ/α/β proportions on one axis make conservation-of-mass and inter-band redistribution legible at a glance. On the ECEO recording it cleanly renders the eyes-closed α bulges (Figure 3.6-A).
5. ✅ **Ternary 2-simplex trajectory added** (`plot_band_ternary`, `*_ternary.png`): per-window (p_θ, p_α, p_β) plotted in the triangle, time-coloured, with the centroid marked — directly visualising attractor states and *why* BASD (band-aware) carries information BandEn (band-blind) discards (Figures 3.6-B/C).

### 3.2 The joint-distribution heatmap (`plot_joint_distribution`)

**Layout.** A central `P(X,Y)` heatmap (magma) in **bin-index (copula) space**, flanked by the X marginal on top (blue) and Y marginal on the right (green), a horizontal colourbar, and a monospace annotation box with `I`, `I_MM`, `I_norm`, `H(X)`, `H(Y)`, `H(X,Y)`, N, and the surrogate p/z.

**How to read it correctly.** The single most important and easily-misread feature: the heatmap is in **bin-index space, not signal space** (lines 1592–1610). This is a deliberate and correct choice — under quantile binning the cells have very unequal widths in µV, so a signal-axis view would collapse visually and the marginals would *not* be flat. In bin-index space the marginals are **flat by construction** (equiprobable bins), so:
- **All structure of interest is in the *joint* panel's deviation from a uniform sheet.** A featureless magma square ⇒ independence (`I ≈ 0`). A bright **diagonal ridge** ⇒ strong positive dependence; an **anti-diagonal** ⇒ negative; **off-diagonal clusters** ⇒ non-monotone coupling that correlation would miss. The tick labels carry the real (z-scored) edge values so the reader can map bins back to amplitudes.
- The annotation box is the quantitative anchor: trust `I_MM` and the surrogate p/z over the raw `I`.

**Critique and optimisation opportunities — *implemented and verified* (see §3.6).**
1. ✅ **Excess-mass diverging view added** (`plot_joint_excess`, `*_excess.png`): `P(X,Y) − P(X)P(Y)` on a zero-centred `RdBu_r` colourmap. On the Hardy ch1/ch2 data this immediately exposes structure the raw magma heatmap hides — a **central positive diagonal *plus* strong tail co-excursion corners** (the ±5σ cells, Figure 3.6-E); see the new empirical finding in §4.6.
2. ✅ **Surrogate-null inset added.** The previously-empty top-right gridspec cell now holds a histogram of the circular-shift surrogate MI with the observed value marked in red, so significance is legible at a glance (`compute_joint_mi_significance` now returns the surrogate array). On Hardy the observed MI sits far in the right tail (z = 12.5; Figure 3.6-D).
3. **Deeper phenomenon (open):** the heatmap remains a **static, whole-recording** average; the new peri-event time course (§3.4 ✅) partially addresses *when* coupling forms. A small-multiples sequence of joint distributions per task block is the natural next step and is left as future work.

### 3.3 The zero-lag MI time series (`_run_joint_mi_mode`)

**Layout.** A single wide panel: faint raw `joint_mi` + bold smoothed, optional event overlay, absolute-time axis under `--ibrain-events`.

**Reading guide.** Sustained elevation ⇒ persistent shared information between the (denoised) channels; transient spikes ⇒ momentary coupling events. **Caution:** because each window is normalised independently, this series is insensitive to slow amplitude drift (a feature), but plug-in MI's positive bias means the *absolute level* is not zero-referenced — read *changes*, not absolute values, and cross-reference the whole-recording surrogate test.

**Critique — *partially implemented* (see §3.6).** (i) ✅ **Quality masking now routed through the zero-lag MI series** (non-denoise path): the two channels are scored with `eeg_quality_v2` on the same window grid and sub-threshold windows are NaN-masked, exactly as the band-entropy series. On Hardy this masked **149/2126 (7.0 %)** windows, and the masked quality column is written to the time-series CSV. (Skipped under `--denoise`, where model outputs are not raw device channels and the device-tuned quality params do not apply — a deliberate, documented exclusion.) (ii) Per-window significance shading remains open (the whole-recording surrogate test is reported instead). (iii) **Deeper phenomenon (open):** overlaying the lagged-MI synchrony series to watch coupling migrate between lag-0 and lag>0 is left as future work.

### 3.4 The pre-event vs. onset bar chart (`plot_event_pre_onset_comparison`)

**Layout.** Grouped bars per event (grey = pre, red = onset) with the signed ΔMI annotated above each pair.

**Reading guide.** A red bar exceeding its grey partner (positive ΔMI) ⇒ coupling *increased* at event onset. The annotation gives the magnitude directly. Because both windows use the same bins/binning, the comparison is internally fair.

**Critique — *implemented and verified* (see §3.6).** (i) ✅ **Per-bar significance markers added.** Each interval now carries its own circular-shift surrogate test; bars are annotated `*/**/***/ns` by p-value. The Hardy run is a clean demonstration: 7 of 8 onsets are significant while **Cone Rotation** is correctly flagged `ns` (p = 0.485, z ≈ 0), so a near-noise ΔMI is no longer over-read. (ii) ✅ **Unequal-N flagged**: the title carries a ⚠ when any pre/onset window pair has unequal sample counts (with N written to the CSV). (iii) ✅ **Peri-event time course added** (`plot_peri_event_mi`, `*_peri_event.png`): the zero-lag MI series re-expressed relative to each onset (t = 0) and overlaid, with the across-event mean ±1σ — exposing *latency and duration* rather than a single step. It reuses the already-computed windowed series, so it adds no MI re-estimation cost (Figure 3.6-G).

### 3.5 Cross-cutting visualisation critique — *implemented and verified*

- ✅ **Consistency:** quality masking is now unified — the zero-lag MI series carries the same `eeg_quality_v2` mask as the band-entropy series (non-denoise path).
- ✅ **Reproducibility metadata:** a `_provenance` footer is stamped on *every* figure (`_add_footer`), embedding passband, fs, win/step, smoothing length, **git commit** and timestamp. Visible along the bottom of every panel in §3.6.
- ✅ **Colour accessibility:** the θ/α/β palette is now the colour-blind-safe Wong (2011) blue/green/vermillion triple (`BAND_COLORS`), reused consistently across the stack, composition and ternary views.

### 3.6 Implementation and empirical validation

All optimisations above were implemented in `spectral_entropy.py` and executed end-to-end. The headline run is the **Hardy (SN036) iBrainCenter recording** (`merged.csv`, N = 2,126,712 samples ≈ 71 min, 4 ch, 8 task events), processed in **≈ 19 s** for the full joint-MI + 100-surrogate + per-event pipeline; the ECEO recording is used where a clean resting α-modulation makes the compositional views easiest to read.

**Whole-recording joint MI (ch1 ↔ ch2), quantile binning, 16 bins.**

| Quantity | Value |
|---|---|
| I(X;Y) plug-in | 0.1329 bits |
| I_MM (Miller–Madow) | 0.1329 bits |
| I_norm | 0.0332 |
| H(X) = H(Y) | 4.000 bits (= log₂16 → all bins filled, confirming quantile binning) |
| H(X,Y) | 7.867 bits |
| Surrogate null (n=100) | 0.0082 ± 0.0100 bits |
| **p / z** | **0.0099 / 12.48** |
| Windows quality-masked | 149 / 2126 (7.0 %) |

**Per-event pre→onset ΔMI (surrogate-gated).**

| Event | pre (bits) | onset (bits) | ΔMI | onset p | onset z |
|---|---|---|---|---|---|
| Single Cycling | 0.101 | 0.059 | −0.043 | 0.0099 | 7.1 |
| Cycling Boxing | 0.113 | 0.062 | −0.052 | 0.0099 | 26.4 |
| Push-ups | 0.025 | 0.090 | +0.064 | 0.0099 | 3.4 |
| Machine Chest Press | 0.114 | 0.127 | +0.013 | 0.0099 | 27.5 |
| Agility Ladder | 0.128 | 0.069 | −0.058 | 0.0099 | 3.5 |
| **Color Agility Ladder** | 0.103 | **0.200** | **+0.098** | 0.0099 | **45.0** |
| Cone Rotation | 0.035 | 0.059 | +0.024 | **0.485** | −0.0 (**ns**) |
| Mindfulness Meditation | 0.091 | 0.081 | −0.009 | 0.0099 | 20.0 |

**Figures (generated outputs, in [report_figures/](report_figures/)).**

- **3.6-A — stacked-area composition (ECEO):** [band_composition_eceo.png](report_figures/band_composition_eceo.png). The α (green) band visibly bulges at ≈ 40 s and ≈ 100 s — the eyes-closed epochs — with β dominant otherwise; conservation of mass is now explicit on one axis.
- **3.6-B — ternary trajectory (ECEO):** [band_ternary_eceo.png](report_figures/band_ternary_eceo.png). The resting recording drifts from a β-leaning cluster (early, purple) toward the α-corner (later, yellow) — the EC→EO α-shift as a path on the simplex.
- **3.6-C — ternary trajectory (task):** [band_ternary_task.png](report_figures/band_ternary_task.png). The active-task recording fills the simplex with a β-leaning centroid, consistent with task-driven β dominance.
- **3.6-D — joint distribution + surrogate inset:** [joint_distribution_surrogate_inset.png](report_figures/joint_distribution_surrogate_inset.png). Flat marginals (quantile binning) and the surrogate-null inset with the observed MI in the far right tail.
- **3.6-E — excess-mass view:** [joint_excess_mass.png](report_figures/joint_excess_mass.png). Central positive diagonal **plus** vivid tail-corner co-excursion cells — see §4.6.
- **3.6-F — pre/onset bars with significance:** [pre_onset_bars_significance.png](report_figures/pre_onset_bars_significance.png). Per-bar `*/ns` markers; Cone Rotation flagged `ns`.
- **3.6-G — peri-event MI time course:** [peri_event_mi.png](report_figures/peri_event_mi.png). Per-event traces aligned at onset (t = 0) with the across-event mean ±1σ.
- **3.6-H — band-entropy stack (best-τ + ±1σ):** [band_entropy_stack_sync_besttau.png](report_figures/band_entropy_stack_sync_besttau.png). The BandEn ±1σ band, the colour-blind-safe palette, and the best-τ scatter on the synchrony panel's twin axis.
- **3.6-I — quality-masked MI series (task):** [joint_mi_timeseries_qmasked.png](report_figures/joint_mi_timeseries_qmasked.png). Gaps are the 7 % quality-masked windows.

### 3.7 Cross-subject cohort run (five iBrainCenter recordings)

To confirm the optimisations generalise beyond a single subject, the full plot suite was generated for **five iBrainCenter recordings from the same session** (events 14:13–14:55): Ann (SN027), Hardy (SN036), Hsin (SN032), James (SN035), TYY (SN041) — each ≈ 1.9–2.1 M samples (≈ 64–71 min), 4 channels. Every subject yields the complete band-entropy set (stack, composition, ternary) and joint-MI set (distribution + surrogate inset, excess, quality-masked MI series, pre/onset bars, peri-event), saved under [`report_figures/<subject>/`](report_figures/) — 8 figures per subject, 40 in total. Filenames are identical across subjects, so each subject is kept in its own subfolder.

**Band-entropy summary (ch1, θ/α/β; pooled over the recording).**

| Subject | mean BandEn (bits) | ±σ | p_θ | p_α | p_β | windows valid |
|---|---|---|---|---|---|---|
| Ann (SN027) | 1.393 | 0.172 | 0.386 | 0.295 | 0.319 | 76.1 % |
| Hardy (SN036) | 1.296 | 0.199 | 0.188 | 0.381 | 0.432 | 91.5 % |
| Hsin (SN032) | **0.934** | 0.232 | 0.059 | 0.192 | **0.749** | 95.4 % |
| James (SN035) | 1.291 | 0.216 | 0.197 | 0.245 | 0.558 | 89.9 % |
| TYY (SN041) | 1.352 | 0.192 | 0.362 | 0.282 | 0.356 | 91.1 % |

Max BandEn = log₂3 ≈ 1.585. Ann sits near the flat-spectrum ceiling (θ≈α≈β, balanced), whereas **Hsin is strongly β-dominated** (p_β = 0.75) and correspondingly carries the *lowest* entropy (0.93 bits) — the band-blind scalar and the band-aware composition agree, and the ternary centroid for Hsin sits closest to the β-corner.

**Whole-recording joint MI (ch1 ↔ ch2), quantile / 16 bins / 100 surrogates.**

| Subject | I (bits) | I_MM | I_norm | z | onsets sig. (of 8) | MI windows masked |
|---|---|---|---|---|---|---|
| **Ann (SN027)** | **1.202** | 1.202 | **0.300** | **156.6** | 8 | **23.3 %** |
| Hsin (SN032) | 0.370 | 0.370 | 0.092 | 77.7 | 8 | 5.5 % |
| James (SN035) | 0.289 | 0.289 | 0.072 | 25.0 | 8 | 8.4 % |
| TYY (SN041) | 0.161 | 0.161 | 0.040 | 48.9 | 8 | 7.0 % |
| Hardy (SN036) | 0.133 | 0.133 | 0.033 | 12.5 | 7 | 7.0 % |

Every subject's whole-recording MI is significant (p ≈ 0.0099); 4 of 5 have **all 8** onsets significant, with Hardy the lone exception (Cone Rotation `ns`). The event with the largest onset coupling change differs by subject — Ann: Agility Ladder (ΔMI = +0.89); Hsin: Mindfulness Meditation (+0.23); James: Push-ups (+0.19, with Single Cycling −0.72); Hardy/TYY: Color Agility Ladder — so the **direction and target of event-locked coupling is subject-specific**, not a fixed session effect (reinforcing §4.6).

**Interpretation caveat — Ann (SN027).** Ann is a simultaneous outlier on *three* axes: the highest ch1↔ch2 MI (1.20 bits, I_norm = 0.30), the highest z (157), and by far the highest artefact-rejection rate (≈ 23–24 % of windows masked). That conjunction is the classic signature of a **partially shorted / common-mode channel pair** (two electrodes carrying a near-identical signal) rather than genuinely strong cortical coupling: a shared common-mode source inflates MI exactly while degrading per-channel quality. Ann's coupling figures should therefore be inspected against the raw channels before any physiological reading — a conclusion the newly-implemented **unified quality masking** (which surfaced the 23 % rejection) and **normalised MI** (0.30, an order of magnitude above the others) jointly make visible. This is itself a small validation that the optimisations earn their place: they flag a data-integrity problem the scalar MI alone would have presented as the "best" result in the cohort.

**Per-subject figure index.** For each subject `<S>` the suite is at `report_figures/<S>/`:
`merged_band_entropy_ch1.png` (stack), `…_composition.png`, `…_ternary.png`; `merged_joint_mi_ch1_ch2_distribution.png`, `…_excess.png`, `…_timeseries.png`, `…_events.png`, `…_peri_event.png` (with matching `.csv` summaries). E.g. the β-dominance of Hsin is clearest in [Hsin ternary](<report_figures/Hsin(SN032)/merged_band_entropy_ch1_ternary.png>) and [Hsin composition](<report_figures/Hsin(SN032)/merged_band_entropy_ch1_composition.png>); the Ann common-mode signature in [Ann excess](<report_figures/Ann(SN027)/merged_joint_mi_ch1_ch2_excess.png>) and [Ann distribution](<report_figures/Ann(SN027)/merged_joint_mi_ch1_ch2_distribution.png>).

---

## 4. Comprehensive Analysis and Academic Conclusions

### 4.1 Synthesis: a coherent, theory-anchored estimation suite

Read as a whole, `spectral_entropy.py` is best understood as a deliberate progression up a ladder of information-theoretic descriptors, each addressing the limitation of the one below:

1. **Band entropy** `H(P)` — a *univariate, single-state, band-blind* scalar measuring spectral flatness. Maximally `log2(3)`.
2. **State entropy** — the same scalar, but with the *non-linearity of entropy* respected (pooled ≠ mean-of-windows) and a *distributional* test attached.
3. **BASD** `D_KL(P_event‖P_base)` — lifts the band-blindness: a *directed, anchored, band-aware* contrast against the subject's own resting prior.
4. **Mutual information** `I(X;Y)` — lifts the univariateness: a *bivariate* coupling measure, with lagged variants that lift the *volume-conduction confound*.

This is not four disjoint features but a single, internally-consistent theoretical argument: **entropy alone is insufficient**, and the script's own code comments derive each successor from the failure mode of its predecessor. That the implementation *embeds* this reasoning (rather than leaving it to a paper) is its principal intellectual contribution.

### 4.2 Methodological strengths, in theoretical terms

- **Estimator-bias literacy.** Every place where a naive plug-in estimate would mislead, the script applies the textbook correction: Miller–Madow for entropy/MI bias, circular-shift surrogates for the coupling null, Laplace smoothing for KL degeneracy, equiprobable binning for heavy-tailed marginals. This is the difference between a *demo* and a *defensible measurement*.
- **The non-linearity of entropy is honoured.** Reporting pooled and per-window entropy separately, and testing on the per-window distribution, correctly navigates Jensen's inequality — a subtlety many EEG-entropy studies miss.
- **Confound-awareness in coupling.** Non-zero-lag MI for volume conduction, independent per-window normalisation for amplitude drift, and surrogates that *preserve autocorrelation* together show a mature understanding of what can masquerade as connectivity.
- **Pipeline comparability as a first-class constraint.** Imported cutoffs, a shared "clean" definition, and an identical denoiser mean the outputs are interpretable *relative to the qEEG indices* — the scientific payoff of the comparability-first philosophy.

### 4.3 Theoretical limitations and their implications

These are not bugs but **inherent epistemic boundaries** the user must respect when drawing conclusions:

1. **The three-band coarseness.** With max entropy `log2(3) ≈ 1.585` bits, absolute BandEn differences are *small*, and the measure cannot distinguish *within-band* spectral structure (e.g., a 9 Hz vs. 12 Hz alpha peak are identical to it). The script's own caveat — *always report ΔEntropy with the per-window test, never the pooled value alone* — follows directly. **Implication:** BandEn is a low-resolution summary; BASD and the underlying band proportions carry the finer signal.
2. **Stationarity is assumed, not enforced (outside `--clean`).** Pooling PSDs and per-window entropy both presuppose quasi-stationarity within the interval/window. On active-task wearable EEG this is at best approximate. **Implication:** longer intervals trade statistical power for stationarity validity; the `--clean` path is the only one that defends the assumption.
3. **Histogram MI remains bias- and binning-dependent** even after Miller–Madow. The surrogate test certifies *presence* of dependence, not the *magnitude*; absolute MI in bits should not be over-interpreted across recordings with different N. **Implication:** prefer the normalised MI and the z-score for cross-recording comparison; treat raw bits as within-recording.
4. **Two divergent front-ends.** The univariate modes use a 500 Hz, bandpass-only signal; the `--denoise` MI path uses a 200 Hz, fully-filtered, neural-denoised signal. These are *different physical substrates*. **Implication:** BandEn and denoised-MI results are not on a common footing and should not be causally chained without acknowledging the substrate difference.

### 4.4 Broader theoretical implications

- **From spectral power to spectral *information*.** The classical qEEG indices are *ratios of band powers*; this module reframes the same θ/α/β decomposition in the language of **information theory** — flatness (entropy), surprise relative to a prior (KL/BASD), and shared information (MI). This reframing matters because it makes the resting state an explicit *reference distribution*: cognitive/affective change is then naturally expressed as **bits of divergence from one's own baseline**, a subject-relative, unit-free currency that travels better across individuals than raw µV² ever could.
- **The simplex as a state space.** Because the three proportions live on the 2-simplex, the entire univariate analysis is implicitly a study of a **trajectory on a probability simplex**. Entropy is distance-to-centroid; BASD is a (non-metric) directed distance to an anchor point. This geometric reading (foreshadowed in the §3.1 ternary-diagram suggestion) is the natural next theoretical step and would unify BandEn and BASD as two readouts of one trajectory.
- **Connectivity as conditional surprise.** The MI machinery positions inter-hemispheric coupling as *reduction in uncertainty about one channel given the other* — and the lagged/surrogate apparatus operationalises the hard problem of separating **genuine information transfer from shared-source artefact**. The pre-event/onset contrast then frames cognition as **event-locked modulation of shared information**, a hypothesis the script makes directly testable.

### 4.6 New empirical findings from the implemented optimisations

Executing the optimised pipeline on real data did not merely confirm the design — it surfaced two phenomena the original visualisations could not show, exactly the kind of "deeper phenomena" the critique anticipated:

1. **MI is carried by a central monotone association *plus* tail co-excursions.** The excess-mass view (Figure 3.6-E) decomposes the ch1↔ch2 dependence into (i) a broad warm diagonal ridge — ordinary positive co-variation — and (ii) vivid red cells in the extreme ±5σ corners, i.e. windows where *both* channels simultaneously hit their amplitude tails. The raw-mass magma heatmap (Figure 3.6-D) cannot reveal this because its marginals are flat by construction. **Implication:** a non-trivial fraction of the whole-recording MI (z = 12.5) reflects **common-mode large deflections** (shared movement/artefact or genuine global events), not just continuous cortical coupling. This is precisely the kind of structure quantile binning was chosen to preserve, and it argues for the excess view becoming the *default* coupling diagnostic — the scalar MI alone would have hidden it.

2. **Event onsets modulate coupling bidirectionally, and the surrogate gate matters.** The pre→onset table shows coupling can *rise* (Color Agility Ladder, ΔMI = +0.098, z = 45) or *fall* (Cycling Boxing, −0.052) at onset — so "task onset increases connectivity" would be an over-generalisation. Critically, **Cone Rotation's** +0.024 ΔMI is non-significant (p = 0.485, z ≈ 0): without the newly-added per-event surrogate gate it would have been read as a real coupling increase. This is the concrete payoff of implementing significance markers rather than annotating bare ΔMI.

Both findings reinforce the report's central theoretical claim (§4.1): a scalar summary (MI in bits, BandEn in bits) is *insufficient*, and the value lies in the structure-revealing companion view — the excess heatmap for coupling, the ternary trajectory for composition, the surrogate gate for significance.

### 4.7 Concluding statement

`spectral_entropy.py` is a methodologically rigorous, theory-driven instrument that translates the project's θ/α/β qEEG decomposition into a layered information-theoretic description: spectral flatness (entropy), baseline-anchored spectral movement (BASD), and bivariate coupling (mutual information), each estimator paired with the bias correction or null model its theory demands, and each tied by shared preprocessing to the wider qEEG pipeline for comparability. Its principal limitations — three-band coarseness, the stationarity assumption, residual histogram-MI bias, and the dual 500 Hz/200 Hz substrates — are **understood and documented within the code itself**, which is the hallmark of a defensible scientific tool. The visual and structural advancement opportunities identified in §3 have now been **implemented and empirically validated** (§3.6): a `--smooth` control and provenance footers; a ±1σ band, best-τ overlay, colour-blind-safe palette, stacked-area composition and ternary-simplex trajectory on the band-entropy side; an excess-mass view, surrogate-null inset, unified quality masking, per-event significance gating and a peri-event MI time course on the coupling side. Together they elevate the module from a rigorous **measurement** suite to a genuinely **exploratory** one — and, as §4.6 shows, the exploratory views already pay for themselves by exposing tail-driven coupling and bidirectional, surrogate-gated event modulation that the scalar summaries alone would have concealed. The **five-subject cohort run** (§3.7) confirms the optimisations generalise: they cleanly separate a β-dominated subject (Hsin) from balanced-spectrum subjects on the band side, render subject-specific event-locked coupling on the MI side, and — in flagging Ann's simultaneous high-MI / high-rejection / high-normalised-MI signature — let the unified quality mask and normalised MI jointly catch a probable common-mode data-integrity fault that the raw MI scalar would have mis-ranked as the cohort's strongest result. The remaining open items (per-window MI significance shading, lag-0/lag-τ overlay, small-multiples joint distributions per task block) are incremental extensions of the same foundation, which remains theoretically sound and unchanged.

---

*Report prepared 2026-06-23 from `spectral_entropy.py` and its imported dependencies (`eeg_utils.py`, `plot_event_markers.py`, `plot_tflite_summary.py`, `data_analysis.py`, `eeg_quality_v2`).*
