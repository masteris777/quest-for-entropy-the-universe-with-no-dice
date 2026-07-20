"""Demo 13 - Look, and the Stripes Die (the observer effect without mysticism)

uv run python which_path_demo.py           # full render
uv run python which_path_demo.py --smoke   # small/fast dev check

Fidelity: INSPIRED-BY. No certified which-path lab exists yet (see NOT-claimed
in demo-concept.md) - this is a legible two-path cartoon, not the certified
machine. Per the pre-declared simplifications: phases come from a legible
deterministic clock (not the certified ensemble), the fold is drawn as a
picture-stack archive (the real construct is a state tape), two paths only
(not the certified 3-channel machine), detector efficiency 100%.

Mechanism (self-contained, no imports from other labs - importing would
upgrade the fidelity level, which the contract forbids):
  A single deterministic "hidden clock" sequence u_n = frac(n*golden_ratio + 0.5)
  equidistributes over [0,1) without ever repeating and without any RNG - each
  click is a deterministic function of a tick count, exactly the family's
  "no wave arrives, each dot is one deterministic outcome" ethos.
  FOLD OFF: u_n feeds straight into the inverse-CDF of the coherent two-path
  density env(x)*(1+cos(k x)) - the path bit stays unread, buried inside a
  sample that mixes both paths inextricably. Clean cosine fringes emerge.
  FOLD ON: u_n is split - the first bit is READ AND FILED as a definite path
  (A if u_n<0.5 else B), the remainder re-drives sampling from THAT path's own
  single-path envelope only. Reading + filing the path is what deletes the
  cross term; nothing else about the source, clock, or screen changes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

HERE = Path(__file__).parent
GOLD = (1 + np.sqrt(5)) / 2

# ---------------------------------------------------------------------------
# Screen geometry / interference model (self-contained cartoon)
# ---------------------------------------------------------------------------
X_MAX = 6.0
K_FRINGE = 4.5          # fringe wavenumber (fold-off cosine term)
SIGMA_ENV = 3.0         # fold-off shared envelope width
D_SLIT = 3.5            # fold-on: separation between the two lump centers
SIGMA_SLIT = 0.85       # fold-on: each lump's own width
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

DENSITY_A = np.exp(-(X_GRID - (-D_SLIT / 2)) ** 2 / (2 * SIGMA_SLIT ** 2))
DENSITY_B = np.exp(-(X_GRID - (D_SLIT / 2)) ** 2 / (2 * SIGMA_SLIT ** 2))
CDF_A = _cdf_from_density(DENSITY_A)
CDF_B = _cdf_from_density(DENSITY_B)

BIN_EDGES = np.linspace(-X_MAX, X_MAX, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]


def clock(n0, n1):
    """The single deterministic hidden clock: no RNG, no repeats, equidistributed."""
    n = np.arange(n0, n1, dtype=np.float64)
    return np.mod(n * GOLD + 0.5, 1.0)


def sample_fold_off(u):
    return _inverse_sample(u, CDF_OFF)


def sample_fold_on(u):
    """Read + file the path bit, re-prepare position from that path alone."""
    path_a = u < 0.5
    u_local = np.where(path_a, u * 2.0, (u - 0.5) * 2.0)
    x = np.empty_like(u)
    x[path_a] = _inverse_sample(u_local[path_a], CDF_A)
    x[~path_a] = _inverse_sample(u_local[~path_a], CDF_B)
    return x, path_a


def style_ax(ax):
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")


CAPTION_1 = ("Dots arrive one by one. Only thousands of them together reveal the\n"
             "stripes - this is the part every textbook shows.")
CAPTION_2 = ("Now the path is measured. Not \"observed by a mind\" - FOLDED: the state is\n"
             "filed into a one-way archive and re-prepared. Watch the archive fill as\n"
             "the new dots forget how to stripe.")
CAPTION_3 = ("In this deterministic toy, the \"observer effect\" is a filing event - no mind\n"
             "required. (Our measured machine dims its fringes under phase disturbance by\n"
             "the exact textbook law.) Whether nature's version is also a fold is exactly\n"
             "the question this programme studies.")


def hist_heights(x, weight_scale=1.0):
    H, _ = np.histogram(x, bins=BIN_EDGES)
    if H.sum() > 0:
        H = H / (H.sum() * BIN_WIDTH)
    return H * weight_scale


def run(n_act1, n_act2, fps, out_stub, smoke=False):
    t0 = time.time()

    PHASE1_S, PHASE2_S, PHASE3_S = (5, 6, 3) if smoke else (15, 20, 10)
    frames_p1 = max(1, int(fps * PHASE1_S))
    frames_p2 = max(1, int(fps * PHASE2_S))
    frames_p3 = max(1, int(fps * PHASE3_S))
    total_frames = frames_p1 + frames_p2 + frames_p3
    total_dots = n_act1 + n_act2

    print(f"[{'SMOKE' if smoke else 'FULL'}] act1={n_act1} act2={n_act2} "
          f"total_dots={total_dots} frames=({frames_p1}+{frames_p2}+{frames_p3})={total_frames}")

    # --- precompute the whole dot stream from ONE continuous clock ---
    u_all = clock(0, total_dots)
    u1 = u_all[:n_act1]
    u2 = u_all[n_act1:]
    x1 = sample_fold_off(u1)
    x2, path2_a = sample_fold_on(u2)

    n_roll = min(4000, n_act2) if not smoke else min(400, n_act2)

    bounds_p1 = np.linspace(0, n_act1, frames_p1 + 1).round().astype(int)
    bounds_p2 = np.linspace(0, n_act2, frames_p2 + 1).round().astype(int)

    # pre-switch (ghost) histogram: fixed once act1 completes
    ghost_H = hist_heights(x1)

    # --- figure setup ---
    fig = plt.figure(figsize=(11, 6.6), facecolor="black")
    ax_screen = fig.add_axes([0.05, 0.19, 0.72, 0.68])
    ax_arch = fig.add_axes([0.80, 0.19, 0.10, 0.68])
    ax_before = fig.add_axes([0.03, 0.19, 0.36, 0.68])
    ax_after = fig.add_axes([0.41, 0.19, 0.36, 0.68])
    for a in (ax_screen, ax_arch, ax_before, ax_after):
        style_ax(a)
    ax_before.set_visible(False)
    ax_after.set_visible(False)

    title = fig.suptitle("", color="white", fontsize=15, y=0.965)
    status_txt = fig.text(0.92, 0.90, "", ha="right", color="white", fontsize=12, weight="bold")
    caption_txt = fig.text(0.5, 0.135, "", ha="center", va="top", color="#dddddd", fontsize=10.5)

    bars_screen = ax_screen.bar(BIN_CENTERS, np.zeros(N_BINS), width=BIN_WIDTH * 0.95,
                                 color="#ff8c1a")
    ghost_bars = ax_screen.bar(BIN_CENTERS, np.zeros(N_BINS), width=BIN_WIDTH * 0.95,
                                color="#552222", alpha=0.55, zorder=0)
    ax_screen.set_xlim(-X_MAX, X_MAX)
    ax_screen.set_ylim(0, max(ghost_H.max(), 1.0) * 1.35)

    ax_arch.set_xlim(0, 1)
    ax_arch.set_ylim(0, 1)
    arch_fill = patches.Rectangle((0.25, 0), 0.5, 0.0, color="#ffb347")
    ax_arch.add_patch(arch_fill)
    ax_arch.set_title("archive", color="#999999", fontsize=9, pad=6)
    arch_ticks = []

    state = {"phase": 1}

    def clear_arch_ticks():
        for t in arch_ticks:
            t.remove()
        arch_ticks.clear()

    def update(frame):
        if frame < frames_p1:
            j = frame
            cursor = bounds_p1[j + 1]
            H = hist_heights(x1[:cursor])
            for rect, h in zip(bars_screen, H):
                rect.set_height(h)
            for rect in ghost_bars:
                rect.set_height(0.0)
            ax_screen.set_visible(True)
            ax_arch.set_visible(False)
            ax_before.set_visible(False)
            ax_after.set_visible(False)
            title.set_text(f"Look, and the Stripes Die  -  {cursor:,} dots")
            status_txt.set_text("FOLD: OFF")
            status_txt.set_color("#8fdc8f")
            caption_txt.set_text(CAPTION_1)
            artists = list(bars_screen) + list(ghost_bars) + [title, status_txt, caption_txt]
        elif frame < frames_p1 + frames_p2:
            j = frame - frames_p1
            cursor = bounds_p2[j + 1]
            window = x2[max(0, cursor - n_roll):cursor]
            H = hist_heights(window)
            for rect, h in zip(bars_screen, H):
                rect.set_height(h)
            for rect, h in zip(ghost_bars, ghost_H):
                rect.set_height(h)
            ax_screen.set_ylim(0, max(ghost_H.max(), H.max() if H.size else 1.0, 1.0) * 1.35)
            ax_screen.set_visible(True)
            ax_arch.set_visible(True)
            ax_before.set_visible(False)
            ax_after.set_visible(False)
            frac = cursor / n_act2
            arch_fill.set_height(frac)
            if frame % 5 == 0:
                clear_arch_ticks()
                n_ticks = int(frac * 40)
                for k in range(n_ticks):
                    y = k / 40.0
                    ln = ax_arch.plot([0.25, 0.75], [y, y], color="#3a2410", lw=0.6)[0]
                    arch_ticks.append(ln)
            title.set_text(f"Look, and the Stripes Die  -  archive {frac*100:.0f}% filed")
            status_txt.set_text("FOLD: ON")
            status_txt.set_color("#ff6644")
            caption_txt.set_text(CAPTION_2)
            artists = (list(bars_screen) + list(ghost_bars) + [title, status_txt, caption_txt,
                       ax_arch, arch_fill] + arch_ticks)
        else:
            ax_screen.set_visible(False)
            ax_arch.set_visible(True)
            ax_before.set_visible(True)
            ax_after.set_visible(True)
            H_after = hist_heights(x2[-n_roll:])
            ax_before.clear()
            style_ax(ax_before)
            ax_before.bar(BIN_CENTERS, ghost_H, width=BIN_WIDTH * 0.95, color="#ff8c1a")
            ax_before.set_title("before: fold OFF (stripes)", color="white", fontsize=10)
            ax_before.set_xlim(-X_MAX, X_MAX)
            ax_after.clear()
            style_ax(ax_after)
            ax_after.bar(BIN_CENTERS, H_after, width=BIN_WIDTH * 0.95, color="#ff5522")
            ax_after.set_title("after: fold ON (two lumps)", color="white", fontsize=10)
            ax_after.set_xlim(-X_MAX, X_MAX)
            arch_fill.set_height(1.0)
            title.set_text("Look, and the Stripes Die")
            status_txt.set_text("FOLD: ON")
            status_txt.set_color("#ff6644")
            caption_txt.set_text(CAPTION_3)
            artists = [ax_before, ax_after, ax_arch, arch_fill, title, status_txt, caption_txt]

        if frame % 100 == 0:
            print(f"  frame {frame}/{total_frames} ({time.time()-t0:.1f}s)")
        return artists

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

    # --- fringe-visibility sanity numbers (informal check) ---
    peak = ghost_H.max()
    trough_region = ghost_H[(np.abs(BIN_CENTERS) < 0.3)]
    visibility = "n/a"
    if peak > 0:
        vmax, vmin = ghost_H.max(), ghost_H.min()
        visibility = (vmax - vmin) / (vmax + vmin)
    print(f"  fold-OFF fringe visibility (max-min)/(max+min) over histogram: {visibility:.3f}")
    H_final_on = hist_heights(x2[-n_roll:])
    mid_frac = H_final_on[np.abs(BIN_CENTERS) < D_SLIT / 4].sum() / max(H_final_on.sum(), 1e-9)
    print(f"  fold-ON center dip: fraction of density within +/-{D_SLIT/4:.2f} of center = {mid_frac:.3f} "
          f"(low = two separated lumps, as expected)")

    # --- static three-panel fallback ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.6), facecolor="black")
    fig2.suptitle("Look, and the Stripes Die", color="white", fontsize=16)

    style_ax(axes2[0])
    axes2[0].bar(BIN_CENTERS, ghost_H, width=BIN_WIDTH * 0.95, color="#ff8c1a")
    axes2[0].set_xlim(-X_MAX, X_MAX)
    axes2[0].set_title("fold OFF\n(stripes)", color="white", fontsize=10)

    mid_cursor = n_act2 // 2
    mid_window = x2[max(0, mid_cursor - n_roll):mid_cursor]
    H_mid = hist_heights(mid_window)
    style_ax(axes2[1])
    axes2[1].bar(BIN_CENTERS, ghost_H, width=BIN_WIDTH * 0.95, color="#552222", alpha=0.55, zorder=0)
    axes2[1].bar(BIN_CENTERS, H_mid, width=BIN_WIDTH * 0.95, color="#ff8c1a", zorder=1)
    axes2[1].set_xlim(-X_MAX, X_MAX)
    axes2[1].set_title(f"transition\n(archive {100*mid_cursor/n_act2:.0f}% filed)", color="white", fontsize=10)

    style_ax(axes2[2])
    axes2[2].bar(BIN_CENTERS, H_final_on, width=BIN_WIDTH * 0.95, color="#ff5522")
    axes2[2].set_xlim(-X_MAX, X_MAX)
    axes2[2].set_title("fold ON\n(two lumps)", color="white", fontsize=10)

    fig2.text(0.5, 0.02, CAPTION_3, ha="center", color="#cccccc", fontsize=9.5)
    fig2.tight_layout(rect=[0, 0.10, 1, 0.90])
    static_path = HERE / "which_path_static.png"
    fig2.savefig(static_path, facecolor="black", dpi=140)
    plt.close(fig2)
    print(f"  static panel saved to {static_path}")

    print("\nWHAT TO LOOK FOR: In Act 1, watch dots land one at a time and slowly build "
          "clean cosine stripes - nothing is 'waving', each dot is a single deterministic "
          "click from a hidden clock. When FOLD switches ON in Act 2, watch the small "
          "archive strip on the right start filling - that filing, not any act of "
          "conscious observation, is what changes. The live histogram of new dots washes "
          "out of its stripes and settles into two separate lumps within a few seconds, "
          "while the dimmed stripes from before the switch stay frozen behind it for "
          "comparison. Same source, same screen, same clock the whole time - only the "
          "filing began.")

    return saved_path, total_frames, time.time() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        run(n_act1=3000, n_act2=3000, fps=8, out_stub="which_path_smoke", smoke=True)
    else:
        run(n_act1=60000, n_act2=50000, fps=20, out_stub="which_path_demo", smoke=False)
