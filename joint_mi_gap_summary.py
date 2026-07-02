"""
Cross-subject aggregation of the Σ (summation) vs true Joint-KSG band-power ×
event mutual-information comparison.

Consumes the per-subject ``merged_band_event_mi.csv`` files produced by
``spectral_entropy.py --band-event-mi`` and quantifies, for every window size
and channel:

    Σ            = I(θ;E) + I(α;E) + I(β;E)              (sum of per-band MIs)
    Joint-KSG    = I(θ, α, β ; E)                        (true multivariate Ross 2014)
    Gap          = Σ − Joint-KSG                         (shared / redundant info)
    Redundancy % = 100 · Gap / Σ                         (fraction of Σ that is overlap)

A positive Gap is the redundancy the summation over-counts; a Gap ≈ 0 (or
negative) flags near-independent or synergistic bands where Σ is not inflated.

Outputs
-------
* report_figures/band_event_mi/gap_vs_window.png  — Gap vs window, per channel,
  one line per subject + across-subject mean.
* report_figures/band_event_mi/summary_all_subjects.csv — tidy table with the
  four quantities above for every (subject, channel, window).
* a printed markdown summary table.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "report_figures/band_event_mi"
SUBJ_DIRS = {
    "Ann_SN027": "Ann (SN027)",
    "Hardy_SN036": "Hardy (SN036)",
    "Hsin_SN032": "Hsin (SN032)",
    "James_SN035": "James (SN035)",
    "TYY_SN041": "TYY (SN041)",
}


def load_all() -> pd.DataFrame:
    frames = []
    for key, label in SUBJ_DIRS.items():
        path = os.path.join(BASE, key, "merged_band_event_mi.csv")
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
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    df = add_gap_columns(load_all())
    out_csv = os.path.join(BASE, "summary_all_subjects.csv")
    cols = ["Subject", "Channel", "Window_Size", "Joint_MI_Sum_Bits",
            "Joint_MI_KSG_Bits", "Gap_Bits", "Redundancy_Pct",
            "surrogate_p_value", "surrogate_z", "N_Pre", "N_Post", "K_Neighbors"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    plot_gap(df, os.path.join(BASE, "gap_vs_window.png"))

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


if __name__ == "__main__":
    main()
