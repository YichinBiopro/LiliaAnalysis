"""
Joint band-power × event mutual-information analysis (Σ vs true Joint-KSG).

Two subcommands, sharing the Ross-2014 mixed KSG estimator that lives in
``spectral_entropy``:

    joint_mi.py per-event      per-event-type Joint MI I(θ,α,β ; pre/post-event)
                               (each event analysed alone, not pooled) — writes a
                               per-subject and cross-subject breakdown.

    joint_mi.py gap-summary    aggregate the pooled per-subject
                               ``merged_band_event_mi.csv`` files into the
                               redundancy Gap = Σ − Joint-KSG across window sizes.

Definitions (bits):
    Σ (summation)  = I(θ;E) + I(α;E) + I(β;E)   — sum of per-band sklearn MIs
    Joint-KSG      = I(θ, α, β ; E)             — multivariate Ross-2014 estimator
    Gap            = Σ − Joint-KSG              — shared / redundant inter-band info
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lilia.io import bandpass_filter, load_merged_csv
from spectral_entropy import (
    DEFAULT_FS, DEFAULT_BP_LOW, DEFAULT_BP_HIGH,
    extract_band_envelopes, compute_band_event_joint_mi,
)
from plot_event_markers import EVENTS, hhmm_to_us

# ═════════════════════════════════════════════════════════════════════════════
# Subcommand: per-event  — per-event-type Joint MI, each event analysed alone
# ═════════════════════════════════════════════════════════════════════════════
# Unlike the pooled ``spectral_entropy.py --band-event-mi --ibrain-events`` path,
# each event onset's pre-event ``[onset−W, onset)`` (label 0) and post-event
# ``[onset, onset+W)`` (label 1) windows are tiled into overlapping 1 s
# sub-epochs and I(θ,α,β ; event) is estimated from *that event alone*, so the
# per-activity θ/α/β reorganisation is visible. A single event gives far fewer
# sub-epochs than the pool (~9/class at W=5 s), so short-window estimates are
# noisier — read them with the surrogate p-value.

PER_EVENT_OUTBASE = "report_figures/band_event_mi/per_event"

PER_EVENT_SUBJECTS = {
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
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
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
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f"Saved: {outpath}")


def run_per_event(_args: argparse.Namespace) -> None:
    os.makedirs(PER_EVENT_OUTBASE, exist_ok=True)
    all_frames = []
    for path, label in PER_EVENT_SUBJECTS.items():
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — skipping")
            continue
        print(f"Analysing {label} …")
        sdf = analyze_subject(path, label)
        if sdf.empty:
            print(f"  [warn] no usable events for {label}")
            continue
        stem = label.split(" ")[0] + "_" + label.split("(")[1].rstrip(")")
        sdf.to_csv(os.path.join(PER_EVENT_OUTBASE, f"{stem}_per_event_mi.csv"), index=False)
        plot_subject(sdf, label, os.path.join(PER_EVENT_OUTBASE, f"{stem}_per_event_mi.png"))
        all_frames.append(sdf)

    if not all_frames:
        raise SystemExit("No per-event results produced.")
    master = pd.concat(all_frames, ignore_index=True)
    master_csv = os.path.join(PER_EVENT_OUTBASE, "per_event_all_subjects.csv")
    master.to_csv(master_csv, index=False)
    print(f"Saved: {master_csv}")
    plot_cross_subject(master, os.path.join(PER_EVENT_OUTBASE, "per_event_summary.png"))


# ═════════════════════════════════════════════════════════════════════════════
# Subcommand: gap-summary  — aggregate pooled per-subject band_event_mi CSVs
# ═════════════════════════════════════════════════════════════════════════════
# A positive Gap is the redundancy the summation over-counts; a Gap ≈ 0 (or
# negative) flags near-independent or synergistic bands where Σ is not inflated.

GAP_BASE = "report_figures/band_event_mi"
GAP_SUBJ_DIRS = {
    "Ann_SN027": "Ann (SN027)",
    "Hardy_SN036": "Hardy (SN036)",
    "Hsin_SN032": "Hsin (SN032)",
    "James_SN035": "James (SN035)",
    "TYY_SN041": "TYY (SN041)",
}


def load_all() -> pd.DataFrame:
    frames = []
    for key, label in GAP_SUBJ_DIRS.items():
        path = os.path.join(GAP_BASE, key, "merged_band_event_mi.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — skipping {label}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "Subject", label)
        frames.append(df)
    if not frames:
        raise SystemExit("No per-subject band_event_mi.csv files found.")
    return pd.concat(frames, ignore_index=True)


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Gap_Bits"] = df["Joint_MI_Sum_Bits"] - df["Joint_MI_KSG_Bits"]
    # Redundancy as a fraction of the (over-counting) summation estimate.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Redundancy_Pct"] = 100.0 * df["Gap_Bits"] / df["Joint_MI_Sum_Bits"]
    return df


def plot_gap(df: pd.DataFrame, outpath: str) -> None:
    channels = sorted(df["Channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(6.2 * len(channels), 5.2),
                             sharey=True)
    if len(channels) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    subjects = sorted(df["Subject"].unique())
    for ax, ch in zip(axes, channels):
        sub_ch = df[df["Channel"] == ch]
        for si, subj in enumerate(subjects):
            s = sub_ch[sub_ch["Subject"] == subj].sort_values("Window_Size")
            ax.plot(s["Window_Size"], s["Gap_Bits"], marker="o", lw=1.4,
                    color=cmap(si % 10), alpha=0.85, label=subj)
        # Across-subject mean gap.
        mean = sub_ch.groupby("Window_Size")["Gap_Bits"].mean().sort_index()
        ax.plot(mean.index, mean.values, color="k", lw=2.8, marker="D",
                label="across-subject mean", zorder=5)
        ax.axhline(0.0, color="0.5", lw=1.0, ls=":")
        ax.set_title(f"{ch}: redundancy gap  Σ − Joint")
        ax.set_xlabel("Window Size (s)")
        ax.set_xticks(sorted(df["Window_Size"].unique()))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Gap  (Σ − Joint-KSG)  [bits]")
    axes[-1].legend(fontsize=8, title="Subject")
    fig.suptitle("Shared/redundant band information across pre→post event transition\n"
                 "positive gap = information the summation Σ over-counts",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    fig.savefig(os.path.splitext(outpath)[0] + '.svg')
    plt.close(fig)
    print(f"Saved: {outpath}")


def run_gap_summary(_args: argparse.Namespace) -> None:
    df = add_gap_columns(load_all())
    out_csv = os.path.join(GAP_BASE, "summary_all_subjects.csv")
    cols = ["Subject", "Channel", "Window_Size", "Joint_MI_Sum_Bits",
            "Joint_MI_KSG_Bits", "Gap_Bits", "Redundancy_Pct",
            "surrogate_p_value", "surrogate_z", "N_Pre", "N_Post", "K_Neighbors"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plot_gap(df, os.path.join(GAP_BASE, "gap_vs_window.png"))

    # Across-subject mean per (channel, window) — the headline comparison.
    agg = (df.groupby(["Channel", "Window_Size"])
             .agg(Sigma=("Joint_MI_Sum_Bits", "mean"),
                  Joint=("Joint_MI_KSG_Bits", "mean"),
                  Gap=("Gap_Bits", "mean"),
                  Redundancy_Pct=("Redundancy_Pct", "mean"),
                  Joint_sig_frac=("surrogate_p_value",
                                  lambda s: float(np.mean(s < 0.05))))
             .reset_index()
             .sort_values(["Channel", "Window_Size"]))

    print("\n### Across-subject mean: Σ vs true Joint-KSG (bits)\n")
    header = ("| Channel | Win (s) | Σ (sum) | Joint-KSG | Gap (Σ−Joint) | "
              "Redundancy % | Joint p<.05 frac |")
    print(header)
    print("|" + "---|" * 7)
    for _, r in agg.iterrows():
        print(f"| {r.Channel} | {int(r.Window_Size)} | {r.Sigma:.4f} | "
              f"{r.Joint:.4f} | {r.Gap:.4f} | {r.Redundancy_Pct:.1f}% | "
              f"{r.Joint_sig_frac:.0%} |")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("per-event",
                   help="per-event-type Joint MI (each event analysed alone)"
                   ).set_defaults(func=run_per_event)
    sub.add_parser("gap-summary",
                   help="aggregate pooled per-subject CSVs into the Σ−Joint gap"
                   ).set_defaults(func=run_gap_summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
