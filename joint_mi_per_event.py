"""
Per-event-type Joint MI  I(θ, α, β power ; pre/post-event), segmented by event.

Unlike ``spectral_entropy.py --band-event-mi --ibrain-events`` (which *pools* the
sub-epochs of every session event into one pre/post dataset), this script keeps
each event type separate: for a single event onset the pre-event window
``[onset − W, onset)`` (label 0) and post-event window ``[onset, onset + W)``
(label 1) are tiled into overlapping 1 s sub-epochs, and I(θ,α,β ; event) is
estimated from *that event alone*. This shows which activities drive the
θ/α/β band-power reorganisation, rather than a session-average.

Both estimators from the module are reused verbatim (no re-implementation):
    * Σ (summation)  = I(θ;E)+I(α;E)+I(β;E)  — sum of per-band sklearn MIs
    * Joint-KSG      = I(θ,α,β ; E)          — multivariate Ross-2014 estimator
    * Gap            = Σ − Joint-KSG         — shared/redundant inter-band info
A 500-permutation label-shuffle null gates the joint estimate (surrogate p/z).

Caveat: a single event gives far fewer sub-epochs than the pool (e.g. only ~9
per class at W=5 s vs ~72 pooled), so short-window per-event estimates are
noisier — read them together with the surrogate p-value.

Outputs (under report_figures/band_event_mi/per_event/):
    <Subject>_per_event_mi.csv / .png    — per-subject event × window breakdown
    per_event_all_subjects.csv           — tidy master table
    per_event_summary.png                — cross-subject mean Joint-KSG by event
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_utils import bandpass_filter, load_merged_csv
from spectral_entropy import (
    DEFAULT_FS, DEFAULT_BP_LOW, DEFAULT_BP_HIGH,
    extract_band_envelopes, compute_band_event_joint_mi,
)
from plot_event_markers import EVENTS, hhmm_to_us

OUTBASE = "report_figures/band_event_mi/per_event"

SUBJECTS = {
    "iBrainCenter/Ann(SN027)/merged.csv": "Ann (SN027)",
    "iBrainCenter/Hardy(SN036)/merged.csv": "Hardy (SN036)",
    "iBrainCenter/Hsin(SN032)/merged.csv": "Hsin (SN032)",
    "iBrainCenter/James(SN035)/merged.csv": "James (SN035)",
    "iBrainCenter/TYY(SN041)/merged.csv": "TYY (SN041)",
}

WINDOWS_SEC = (5.0, 10.0, 15.0, 30.0)
CHANNELS = (1, 2)                     # 1-based
SUB_SEC, SUB_STEP_SEC = 1.0, 0.5
N_SURROGATES = 500

# Compact labels for the (long) event names, in session order.
EVENT_ABBR = {
    "Single Cycling": "Cycling",
    "Cycling Boxing": "Cyc-Box",
    "Push-ups": "Push-ups",
    "Machine Chest Press": "ChestPress",
    "Agility Ladder": "Ladder",
    "Color Agility Ladder": "ColorLadder",
    "Cone Rotation": "ConeRot",
    "Mindfulness Meditation": "Meditation",
}


def resolve_events(time_us: np.ndarray, n_times: int, fs: float,
                   window_max_samp: int) -> list[tuple[str, int]]:
    """In-range session events as (name, onset_sample), keeping only onsets with
    room for the *largest* pre- and post-event window on both sides."""
    epoch = int(time_us[0])
    out: list[tuple[str, int]] = []
    for name, hhmm, _dur, _part in EVENTS:
        idx = int(round((hhmm_to_us(hhmm) - epoch) / 1e6 * fs))
        if idx - window_max_samp >= 0 and idx + window_max_samp <= n_times:
            out.append((name, idx))
    return out


def analyze_subject(path: str, label: str) -> pd.DataFrame:
    time_us, data_raw = load_merged_csv(path)
    # Same 0.5–45 Hz front-end the CLI applies before per-band Hilbert envelopes.
    data = bandpass_filter(data_raw, fs=DEFAULT_FS, lo=DEFAULT_BP_LOW, hi=DEFAULT_BP_HIGH)
    n_times = data.shape[0]
    w_max = int(round(max(WINDOWS_SEC) * DEFAULT_FS))
    events = resolve_events(time_us, n_times, DEFAULT_FS, w_max)
    print(f"  {label}: {len(events)} in-range events "
          f"({', '.join(n for n, _ in events)})")

    records: list[dict] = []
    for ch in CHANNELS:
        envelopes = extract_band_envelopes(data[:, ch - 1], fs=DEFAULT_FS)
        for ev_name, onset in events:
            for w_sec in WINDOWS_SEC:
                w_samp = int(round(w_sec * DEFAULT_FS))
                res = compute_band_event_joint_mi(
                    envelopes, [onset], w_samp, fs=DEFAULT_FS,
                    sub_sec=SUB_SEC, sub_step_sec=SUB_STEP_SEC,
                    n_neighbors=3, n_surrogates=N_SURROGATES, random_state=0,
                )
                if res is None:
                    continue
                rec = {
                    "Subject": label, "Event": ev_name,
                    "Event_Abbr": EVENT_ABBR.get(ev_name, ev_name),
                    "Channel": f"ch{ch}", "Window_Size": float(w_sec),
                    "Joint_MI_Sum_Bits": res["sum_mi_bits"],
                    "Joint_MI_KSG_Bits": res["joint_mi_bits"],
                    "Gap_Bits": res["sum_mi_bits"] - res["joint_mi_bits"],
                    "N_Pre": res["n_pre"], "N_Post": res["n_post"],
                    "K_Neighbors": res["n_neighbors"],
                }
                for k in ("surrogate_mean_bits", "surrogate_std_bits",
                          "surrogate_p_value", "surrogate_z"):
                    if k in res:
                        rec[k] = res[k]
                records.append(rec)
    return pd.DataFrame.from_records(records)


def plot_subject(df: pd.DataFrame, label: str, outpath: str) -> None:
    """2×2 grid (one panel per window). Each panel: grouped bars of Joint-KSG by
    event, one colour per channel; Σ marked as an open tick above each bar (so
    the redundancy gap is visible); a ``*`` flags surrogate p<.05."""
    windows = sorted(df["Window_Size"].unique())
    channels = sorted(df["Channel"].unique())
    events = list(dict.fromkeys(df.sort_values("Window_Size")["Event_Abbr"]))
    palette = {"ch1": "#0072B2", "ch2": "#D55E00"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(events))
    bw = 0.8 / max(1, len(channels))
    for ax, w in zip(axes, windows):
        wsub = df[df["Window_Size"] == w]
        for ci, ch in enumerate(channels):
            csub = wsub[wsub["Channel"] == ch].set_index("Event_Abbr")
            joint = [csub["Joint_MI_KSG_Bits"].get(e, np.nan) for e in events]
            sigma = [csub["Joint_MI_Sum_Bits"].get(e, np.nan) for e in events]
            pval = [csub["surrogate_p_value"].get(e, np.nan) for e in events]
            xpos = x + (ci - (len(channels) - 1) / 2) * bw
            ax.bar(xpos, joint, width=bw, color=palette.get(ch, None),
                   alpha=0.85, label=f"{ch} Joint-KSG")
            # Σ as an open marker above each bar → the gap Σ−Joint.
            ax.scatter(xpos, sigma, marker="_", s=220, linewidths=2.2,
                       color=palette.get(ch, None), zorder=5,
                       label=f"{ch} Σ (sum)" if ax is axes[0] else None)
            for xi, (j, p) in enumerate(zip(joint, pval)):
                if np.isfinite(p) and p < 0.05 and np.isfinite(j):
                    ax.text(xpos[xi], j, "*", ha="center", va="bottom",
                            fontsize=12, color=palette.get(ch, None))
        ax.set_title(f"Window = {int(w)} s", fontsize=11)
        ax.set_ylabel("MI (bits)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=40, ha="right", fontsize=8)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{label} — per-event Joint band-power MI  I(θ,α,β ; pre/post)\n"
                 "bars = true Joint-KSG · ticks = Σ (sum) · * = surrogate p<.05",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_cross_subject(df: pd.DataFrame, outpath: str) -> None:
    """Cross-subject mean Joint-KSG per event type: one panel per channel, a line
    per window size (mean ± sd across subjects)."""
    channels = sorted(df["Channel"].unique())
    events = list(dict.fromkeys(df.sort_values("Window_Size")["Event_Abbr"]))
    windows = sorted(df["Window_Size"].unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(channels), figsize=(7.5 * len(channels), 5.5),
                             sharey=True)
    if len(channels) == 1:
        axes = [axes]
    x = np.arange(len(events))
    for ax, ch in zip(axes, channels):
        csub = df[df["Channel"] == ch]
        for wi, w in enumerate(windows):
            g = csub[csub["Window_Size"] == w].groupby("Event_Abbr")["Joint_MI_KSG_Bits"]
            mean = np.array([g.mean().get(e, np.nan) for e in events])
            sd = np.array([g.std().get(e, np.nan) for e in events])
            color = cmap(wi / max(1, len(windows) - 1))
            ax.plot(x, mean, marker="o", color=color, lw=1.8, label=f"{int(w)} s")
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
        ax.set_title(f"{ch}: mean Joint-KSG across subjects")
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=40, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Joint-KSG  I(θ,α,β ; event)  [bits]")
    axes[-1].legend(title="Window", fontsize=8)
    fig.suptitle("Per-event Joint band-power MI, averaged across 5 subjects "
                 "(shaded = ±1σ)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    os.makedirs(OUTBASE, exist_ok=True)
    all_frames = []
    for path, label in SUBJECTS.items():
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — skipping"); continue
        print(f"Analysing {label} …")
        sdf = analyze_subject(path, label)
        if sdf.empty:
            print(f"  [warn] no usable events for {label}"); continue
        stem = label.split(" ")[0] + "_" + label.split("(")[1].rstrip(")")
        sdf.to_csv(os.path.join(OUTBASE, f"{stem}_per_event_mi.csv"), index=False)
        plot_subject(sdf, label, os.path.join(OUTBASE, f"{stem}_per_event_mi.png"))
        all_frames.append(sdf)

    if not all_frames:
        raise SystemExit("No per-event results produced.")
    master = pd.concat(all_frames, ignore_index=True)
    master_csv = os.path.join(OUTBASE, "per_event_all_subjects.csv")
    master.to_csv(master_csv, index=False)
    print(f"Saved: {master_csv}")
    plot_cross_subject(master, os.path.join(OUTBASE, "per_event_summary.png"))


if __name__ == "__main__":
    main()
