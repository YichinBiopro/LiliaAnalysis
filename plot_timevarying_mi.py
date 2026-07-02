"""
Interactive time-varying Mutual Information (MI) line plot (Plotly).

Simulates the peri-event MI time course of 8 EEG/biosignal tasks around an
event onset at t = 0 s (window −5 s … +25 s):

  * Pre-event  (t < 0): every task fluctuates at a low baseline (0.0–0.2 bits).
  * Post-event (t ≥ 0): 'Meditation' and 'ColorLadder' spike to 0.6–0.9 bits;
    the rest show only minor fluctuations or stay flat.

Interactivity
  * Vertical dashed "Event Onset" line at t = 0.
  * Legend entries toggle each task on/off.
  * A Play button + Slider animate the lines growing left→right in time, so the
    influx of information after the event is revealed dynamically.

Run:  python plot_timevarying_mi.py   ->  writes time_varying_mi.html (auto-opens)
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ── Configuration ────────────────────────────────────────────────────────────
TASKS = ["Cycling", "Cyc-Box", "ChestPress", "Push-ups",
         "ConeRot", "Meditation", "ColorLadder", "Ladder"]
SPIKE_TASKS = {"Meditation", "ColorLadder"}          # the two that ramp up
T_START, T_END, DT = -5.0, 25.0, 0.5
SEED = 7

# Okabe–Ito-ish qualitative palette (colour-blind friendly), one per task.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
          "#CC79A7", "#000000", "#56B4E9", "#999999"]

# Post-event target peak (bits) and logistic rise for the two spiking tasks.
SPIKE_PARAMS = {
    "Meditation":  dict(peak=0.72, t0=4.0, tau=1.6),
    "ColorLadder": dict(peak=0.66, t0=6.0, tau=2.2),
}
# A couple of non-spiking tasks get a small post-event bump; the rest stay flat.
MINOR_BUMP = {"Cycling": 0.15, "Push-ups": 0.12}


def simulate() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the time grid and a {task: MI-series} dict of simulated bits."""
    rng = np.random.default_rng(SEED)
    t = np.arange(T_START, T_END + DT / 2, DT)
    post = t >= 0.0

    series: dict[str, np.ndarray] = {}
    for task in TASKS:
        # Low baseline everywhere: gentle random walk centred ~0.1, clipped 0–0.2.
        base = 0.10 + 0.035 * rng.standard_normal(t.size)
        mi = np.clip(base, 0.0, 0.2)

        if task in SPIKE_PARAMS:
            p = SPIKE_PARAMS[task]
            rise = p["peak"] / (1.0 + np.exp(-(t - p["t0"]) / p["tau"]))
            rise[~post] = 0.0                       # no rise before onset
            mi = mi + rise + 0.03 * rng.standard_normal(t.size) * post
            mi = np.clip(mi, 0.0, 0.90)
        elif task in MINOR_BUMP:
            bump = MINOR_BUMP[task] / (1.0 + np.exp(-(t - 8.0) / 3.0))
            bump[~post] = 0.0
            mi = np.clip(mi + bump, 0.0, 0.35)
        # else: task stays at low baseline (flat / minor fluctuation only).

        series[task] = mi
    return t, series


def build_figure(t: np.ndarray, series: dict[str, np.ndarray]) -> go.Figure:
    n = t.size

    # Base traces = first sample only, so the Play button grows the lines.
    base_traces = [
        go.Scatter(
            x=t[:1], y=series[task][:1], mode="lines",
            name=task, line=dict(color=COLORS[i], width=2.6),
            legendgroup=task, hovertemplate=f"{task}<br>t=%{{x:.1f}}s<br>MI=%{{y:.3f}} bits<extra></extra>",
        )
        for i, task in enumerate(TASKS)
    ]

    # One frame per time index: reveal data up to that time (left→right growth).
    frames = [
        go.Frame(
            name=f"{t[k]:.1f}",
            data=[go.Scatter(x=t[: k + 1], y=series[task][: k + 1]) for task in TASKS],
        )
        for k in range(n)
    ]

    slider_steps = [
        dict(method="animate", label=f"{t[k]:.0f}",
             args=[[f"{t[k]:.1f}"],
                   dict(mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))])
        for k in range(n)
    ]

    play_args = dict(frame=dict(duration=90, redraw=True),
                     transition=dict(duration=0), fromcurrent=True, mode="immediate")

    fig = go.Figure(data=base_traces, frames=frames)
    fig.update_layout(
        title="Time-Varying Mutual Information around Event Onset",
        template="plotly_white",
        xaxis=dict(title="Time (seconds)", range=[T_START, T_END], zeroline=False),
        yaxis=dict(title="MI (bits)", range=[0.0, 1.0]),
        legend=dict(title="Task  (click to toggle)", orientation="v",
                    x=1.02, y=1.0),
        hovermode="x unified",
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=1.12, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, play_args]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(active=0, x=0.08, len=0.92, y=0.0, xanchor="left",
                      currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=14)),
                      pad=dict(t=40), steps=slider_steps)],
        margin=dict(t=90, r=170),
    )

    # Vertical dashed "Event Onset" marker at t = 0 (persists across frames).
    fig.add_vline(x=0.0, line=dict(color="crimson", width=2, dash="dash"),
                  annotation_text="Event Onset", annotation_position="top",
                  annotation=dict(font=dict(color="crimson", size=13)))
    return fig


def main() -> None:
    t, series = simulate()
    fig = build_figure(t, series)
    out = "time_varying_mi.html"
    fig.write_html(out, include_plotlyjs="cdn", auto_open=False)
    print(f"Saved interactive plot -> {out}")
    try:
        fig.show()   # opens in the default browser when a display is available
    except Exception as exc:  # noqa: BLE001
        print(f"(fig.show skipped: {exc})")


if __name__ == "__main__":
    main()
