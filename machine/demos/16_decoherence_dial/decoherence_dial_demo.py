"""Demo 16 - Why Cats Don't Show Stripes (the decoherence dial)

uv run python decoherence_dial_demo.py           # full render
uv run python decoherence_dial_demo.py --smoke   # small/fast dev check

Fidelity: INSPIRED-BY. Reuses demo 13's two-path counting cartoon (the pipeline's
first intended reuse - constants and the CDF-sampling machinery below are ported
verbatim from which_path_demo.py, not re-derived, so the W=0 case renders
identically to demo 13's fold-off fringes). New here: a per-particle phase kick
drawn from a SECOND deterministic clock (not np.random) - the environment is
clockwork too, same as the particle's own hidden phase.

THE CREDIBILITY MOVE (non-negotiable per contract): the analytic curve
V(W) = |sinc(W/2)| is drawn once, before any counted point exists, and stays
on screen; each dial setting's counted visibility drops onto it afterward.

Visibility from counts uses the plain (max-min)/(max+min) formula - but applied
to the histogram DIVIDED by the known envelope shape, not the raw histogram.
Dividing out the envelope is necessary, not a tuning knob: the raw histogram's
global min sits in the Gaussian envelope's tail (a few times weaker than its
peak) regardless of fringe contrast, so a literal global max/min over the full
display range would read a large "visibility" even at total decoherence, purely
from envelope shape. Dividing by env(x) first isolates the fringe term the
formula is supposed to measure; see production-report.md.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
GOLD = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2.0)

# ---------------------------------------------------------------------------
# Ported verbatim from ../13_which_path/which_path_demo.py (the reuse the
# handoff calls for) - screen geometry, envelope, fold-off density/CDF.
# ---------------------------------------------------------------------------
X_MAX = 6.0
K_FRINGE = 4.5
SIGMA_ENV = 3.0
N_BINS = 90
GRID_N = 4000

X_GRID = np.linspace(-X_MAX, X_MAX, GRID_N)


def _cdf_from_density(density):
    c = np.cumsum(density)
    c = c - c[0]
    c = c / c[-1]
    return c


def _inverse_sample(u, cdf):
    return np.interp(u, cdf, X_GRID)


ENV_OFF = np.exp(-X_GRID ** 2 / (2 * SIGMA_ENV ** 2))
DENSITY_OFF = ENV_OFF * (1 + np.cos(K_FRINGE * X_GRID))
CDF_OFF = _cdf_from_density(DENSITY_OFF)

BIN_EDGES = np.linspace(-X_MAX, X_MAX, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]
ENV_AT_BINS = np.exp(-BIN_CENTERS ** 2 / (2 * SIGMA_ENV ** 2))

VIS_WINDOW_X = 4.0  # central region where the envelope-division stays numerically stable
VIS_MASK = np.abs(BIN_CENTERS) <= VIS_WINDOW_X


def style_ax(ax):
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")


def clock(n0, n1, mult, phase):
    n = np.arange(n0, n1, dtype=np.float64)
    return np.mod(n * mult + phase, 1.0)


# ---------------------------------------------------------------------------
# New for demo 16: the deterministic environment scrambler + blurred sampling
# ---------------------------------------------------------------------------
N_DELTA_BINS = 48


def sample_blurred(u_pos, u_scr, W):
    """Per-particle phase kick uniform in [-W/2, W/2] (from a deterministic
    clock, u_scr), applied before sampling screen position from the resulting
    two-path density. W=0 short-circuits to the exact fold-off CDF (identical
    to demo 13's construction)."""
    if W <= 1e-9:
        return _inverse_sample(u_pos, CDF_OFF)
    delta = (u_scr - 0.5) * W
    edges = np.linspace(-W / 2, W / 2, N_DELTA_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(delta, edges) - 1, 0, N_DELTA_BINS - 1)
    x_out = np.empty_like(u_pos)
    for b in range(N_DELTA_BINS):
        mask = idx == b
        if not np.any(mask):
            continue
        density_b = ENV_OFF * (1 + np.cos(K_FRINGE * X_GRID + centers[b]))
        cdf_b = _cdf_from_density(density_b)
        x_out[mask] = _inverse_sample(u_pos[mask], cdf_b)
    return x_out


def hist_counts(x):
    H, _ = np.histogram(x, bins=BIN_EDGES)
    return H.astype(np.float64)


def counted_visibility(H):
    """Peak-to-trough visibility of the envelope-divided histogram, restricted
    to the window where that division is numerically stable.

    Implemented as a least-squares fit of the known first harmonic
    (g ~ a0 + a1*cos(kx) + a2*sin(kx), visibility = hypot(a1,a2)/a0) rather
    than a literal (max-min)/(max+min) lookup on two histogram bins. Smoke
    testing at 1500 dots/setting showed the naive two-bin version is a biased,
    high-variance statistic exactly where it matters most (high blur, true
    visibility near 0): individual noisy bins overshoot the true peak and
    undershoot the true trough, giving nonzero apparent visibility even at
    total decoherence. The least-squares fit uses every bin in the window
    coherently instead of two extreme order statistics - same physical
    quantity (fringe peak-to-trough contrast), a statistically sound estimator
    of it. See production-report.md.
    """
    g = H[VIS_MASK] / ENV_AT_BINS[VIS_MASK]
    xw = BIN_CENTERS[VIS_MASK]
    A = np.column_stack([np.ones_like(xw), np.cos(K_FRINGE * xw), np.sin(K_FRINGE * xw)])
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    a0, a1, a2 = coef
    if a0 <= 0:
        return 0.0
    return float(np.clip(np.hypot(a1, a2) / a0, 0.0, 1.5))


def analytic_visibility(W):
    t = W / 2.0
    return float(np.abs(np.sinc(t / np.pi)))  # np.sinc is normalized: sin(pi z)/(pi z)


CAPTION_1 = "Zero blur: the hidden clockwork draws perfect stripes, dot by dot."
CAPTION_2 = ("Now the environment kicks each particle's phase - a little, then a lot. The\n"
             "white line was drawn BEFORE any dot landed: it is the textbook formula. The\n"
             "dots are our counts.")
CAPTION_3 = ("Big things touch everything, so their phases are kicked constantly - that is\n"
             "why cats never show stripes. Our measured machine follows this same law\n"
             "within instrument bounds: decoherence, from counting alone.")


def run(n_act1, n_per_setting, fps, out_stub, n_settings=8, smoke=False):
    t0 = time.time()
    t_vals = np.linspace(0, np.pi, n_settings)
    W_vals = 2 * t_vals

    PHASE1_S, PHASE2_S, PHASE3_S = (4, 8, 4) if smoke else (10, 25, 10)
    frames_p1 = max(1, int(fps * PHASE1_S))
    frames_p2 = max(n_settings, int(fps * PHASE2_S))
    frames_p3 = max(1, int(fps * PHASE3_S))
    total_frames = frames_p1 + frames_p2 + frames_p3

    print(f"[{'SMOKE' if smoke else 'FULL'}] n_act1={n_act1} n_per_setting={n_per_setting} "
          f"n_settings={n_settings} frames=({frames_p1}+{frames_p2}+{frames_p3})={total_frames}")

    # --- one continuous global clock across Act1 + all dial settings ---
    cursor_global = 0
    u_pos1 = clock(cursor_global, cursor_global + n_act1, GOLD, 0.5)
    cursor_global += n_act1
    x_act1 = _inverse_sample(u_pos1, CDF_OFF)

    setting_x = []
    setting_H = []
    counted_vis = []
    analytic_vis = []
    for i, W in enumerate(W_vals):
        u_pos = clock(cursor_global, cursor_global + n_per_setting, GOLD, 0.5)
        u_scr = clock(cursor_global, cursor_global + n_per_setting, SQRT2, 0.37)
        cursor_global += n_per_setting
        x_i = sample_blurred(u_pos, u_scr, W)
        H_i = hist_counts(x_i)
        v_counted = counted_visibility(H_i)
        v_analytic = analytic_visibility(W)
        setting_x.append(x_i)
        setting_H.append(H_i)
        counted_vis.append(v_counted)
        analytic_vis.append(v_analytic)
        print(f"  setting {i+1}/{n_settings}: W={W:.3f}  analytic V={v_analytic:.4f}  "
              f"counted V={v_counted:.4f}  |diff|={abs(v_analytic-v_counted):.4f}")

    rms = float(np.sqrt(np.mean((np.array(counted_vis) - np.array(analytic_vis)) ** 2)))
    print(f"  RMS deviation of counted points from analytic sinc curve: {rms:.4f}")

    chunk_bounds = np.round(np.linspace(0, frames_p2, n_settings + 1)).astype(int)

    # --- figure setup ---
    fig = plt.figure(figsize=(11.5, 6.6), facecolor="black")
    ax_screen = fig.add_axes([0.05, 0.19, 0.60, 0.68])
    ax_curve = fig.add_axes([0.70, 0.19, 0.25, 0.68])
    strip_w = 0.065
    strip_gap = 0.008
    ax_strip = [fig.add_axes([0.05 + i * (strip_w + strip_gap), 0.19, strip_w, 0.68])
                for i in range(n_settings)]
    for a in [ax_screen, ax_curve] + ax_strip:
        style_ax(a)
    ax_curve.set_visible(False)
    for a in ax_strip:
        a.set_visible(False)

    title = fig.suptitle("", color="white", fontsize=15, y=0.965)
    status_txt = fig.text(0.97, 0.90, "", ha="right", color="white", fontsize=12, weight="bold")
    caption_txt = fig.text(0.5, 0.135, "", ha="center", va="top", color="#dddddd", fontsize=10.5)

    bars_screen = ax_screen.bar(BIN_CENTERS, np.zeros(N_BINS), width=BIN_WIDTH * 0.95, color="#ff8c1a")
    ax_screen.set_xlim(-X_MAX, X_MAX)

    ax_curve.set_xlim(-0.3, 2 * np.pi + 0.3)
    ax_curve.set_ylim(-0.05, 1.1)
    W_fine = np.linspace(0, 2 * np.pi, 300)
    ax_curve.plot(W_fine, np.abs(np.sinc((W_fine / 2) / np.pi)), color="white", lw=1.6, zorder=1)
    ax_curve.set_title("counted visibility vs. blur width", color="#999999", fontsize=9, pad=6)
    point_scatter = ax_curve.scatter([], [], color="#ff8c1a", s=45, zorder=2, edgecolor="white", linewidths=0.6)

    state = {"points_x": [], "points_y": []}

    def update(frame):
        if frame < frames_p1:
            cursor = int(np.round((frame + 1) / frames_p1 * n_act1))
            H = hist_counts(x_act1[:max(cursor, 1)])
            for rect, h in zip(bars_screen, H):
                rect.set_height(h)
            ax_screen.set_ylim(0, max(H.max(), 1) * 1.15)
            ax_screen.set_visible(True)
            ax_curve.set_visible(False)
            for a in ax_strip:
                a.set_visible(False)
            title.set_text(f"Why Cats Don't Show Stripes  -  {cursor:,} dots")
            status_txt.set_text("BLUR: 0")
            status_txt.set_color("#8fdc8f")
            caption_txt.set_text(CAPTION_1)
            artists = list(bars_screen) + [title, status_txt, caption_txt]
        elif frame < frames_p1 + frames_p2:
            j = frame - frames_p1
            i_setting = min(int(np.searchsorted(chunk_bounds, j, side="right") - 1), n_settings - 1)
            i_setting = max(i_setting, 0)
            chunk_start, chunk_end = chunk_bounds[i_setting], chunk_bounds[i_setting + 1]
            chunk_len = max(chunk_end - chunk_start, 1)
            j_local = j - chunk_start
            frac = (j_local + 1) / chunk_len
            cursor = max(1, int(round(frac * n_per_setting)))
            x_i = setting_x[i_setting]
            H = hist_counts(x_i[:cursor])
            for rect, h in zip(bars_screen, H):
                rect.set_height(h)
            ax_screen.set_ylim(0, max(H.max(), 1) * 1.15)
            ax_screen.set_visible(True)
            ax_curve.set_visible(True)
            for a in ax_strip:
                a.set_visible(False)

            if cursor >= n_per_setting and len(state["points_x"]) == i_setting:
                state["points_x"].append(W_vals[i_setting])
                state["points_y"].append(counted_vis[i_setting])
                point_scatter.set_offsets(np.column_stack([state["points_x"], state["points_y"]]))

            title.set_text(f"Why Cats Don't Show Stripes  -  setting {i_setting+1}/{n_settings}")
            status_txt.set_text(f"BLUR: W = {W_vals[i_setting]:.2f}")
            status_txt.set_color("#ffcf6e" if i_setting < n_settings - 1 else "#ff6644")
            caption_txt.set_text(CAPTION_2)
            artists = list(bars_screen) + [title, status_txt, caption_txt, ax_curve, point_scatter]
        else:
            ax_screen.set_visible(False)
            ax_curve.set_visible(True)
            for a in ax_strip:
                a.set_visible(True)
            if len(state["points_x"]) < n_settings:
                state["points_x"] = list(W_vals)
                state["points_y"] = list(counted_vis)
                point_scatter.set_offsets(np.column_stack([state["points_x"], state["points_y"]]))
            for i, a in enumerate(ax_strip):
                a.clear()
                style_ax(a)
                a.bar(BIN_CENTERS, setting_H[i], width=BIN_WIDTH * 0.95, color="#ff8c1a")
                a.set_xlim(-X_MAX, X_MAX)
            title.set_text("Why Cats Don't Show Stripes")
            status_txt.set_text("BLUR: done")
            status_txt.set_color("#ff6644")
            caption_txt.set_text(CAPTION_3)
            artists = ax_strip + [ax_curve, point_scatter, title, status_txt, caption_txt]

        if frame % 100 == 0:
            print(f"  frame {frame}/{total_frames} ({time.time()-t0:.1f}s)")
        return artists

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    mp4_path = HERE / f"{out_stub}.mp4"
    gif_path = HERE / f"{out_stub}.gif"
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=110)
        saved_path = mp4_path
    except Exception as e:
        print(f"  ffmpeg unavailable ({e}); falling back to GIF")
        ani.save(gif_path, writer="pillow", fps=fps)
        saved_path = gif_path
    plt.close(fig)
    print(f"  animation saved to {saved_path} ({time.time()-t0:.1f}s)")

    # --- static two-panel fallback ---
    fig2 = plt.figure(figsize=(13.5, 5.2), facecolor="black")
    fig2.suptitle("Why Cats Don't Show Stripes", color="white", fontsize=16)
    strip_w2, strip_gap2 = 0.078, 0.010
    for i in range(n_settings):
        a = fig2.add_axes([0.04 + i * (strip_w2 + strip_gap2), 0.20, strip_w2, 0.62])
        style_ax(a)
        a.bar(BIN_CENTERS, setting_H[i], width=BIN_WIDTH * 0.95, color="#ff8c1a")
        a.set_xlim(-X_MAX, X_MAX)
        a.set_title(f"W={W_vals[i]:.1f}", color="#999999", fontsize=8)
    ax_curve2 = fig2.add_axes([0.75, 0.20, 0.22, 0.62])
    style_ax(ax_curve2)
    ax_curve2.set_xlim(-0.3, 2 * np.pi + 0.3)
    ax_curve2.set_ylim(-0.05, 1.1)
    ax_curve2.plot(W_fine, np.abs(np.sinc((W_fine / 2) / np.pi)), color="white", lw=1.6)
    ax_curve2.scatter(W_vals, counted_vis, color="#ff8c1a", s=45, edgecolor="white", linewidths=0.6, zorder=2)
    ax_curve2.set_title("counted V(W) vs. |sinc(W/2)|", color="#999999", fontsize=9)
    fig2.text(0.5, 0.05, CAPTION_3, ha="center", color="#cccccc", fontsize=9.5)
    static_path = HERE / "decoherence_dial_static.png"
    fig2.savefig(static_path, facecolor="black", dpi=140)
    plt.close(fig2)
    print(f"  static panel saved to {static_path}")

    print("\nWHAT TO LOOK FOR: Act 1 rebuilds the familiar clean stripes from demo 13 with "
          "zero environmental blur. In Act 2, watch the white sinc curve appear on the right "
          "BEFORE any dot has been counted at all - then watch each dial setting rebuild a "
          "dimmer fringe pattern on the left while its counted visibility drops as a single "
          "dot directly onto that pre-drawn line, one point per setting, all the way through "
          "eight increasing blur widths. By the last few settings the left panel's stripes "
          "have essentially vanished into a single smooth bump - that dying-of-structure is "
          "exactly what large, constantly-jostled objects (cats, chairs, you) experience "
          "permanently, which is the demo's answer to why they never show interference.")

    return saved_path, total_frames, rms, time.time() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        run(n_act1=2000, n_per_setting=1500, fps=8, out_stub="decoherence_dial_smoke",
            n_settings=3, smoke=True)
    else:
        run(n_act1=24000, n_per_setting=12000, fps=20, out_stub="decoherence_dial_demo",
            n_settings=8, smoke=False)
