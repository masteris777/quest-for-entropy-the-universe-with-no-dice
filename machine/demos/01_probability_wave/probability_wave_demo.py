"""Demo 01 - The Cloud of Maybes (the probability wave, refreshed)

uv run python probability_wave_demo.py           # full render
uv run python probability_wave_demo.py --smoke   # small/fast dev check

Fidelity: ANALOGY (phase 1, pendulum) + INSPIRED-BY (phase 2, rotors).
Pendulum dynamics ported verbatim from ../../../entropy/double_pendulum_anim.py
(same as 131_demo_suite V1, seed 50000). Rotor construction ported verbatim
from 131_demo_suite/demo_suite_131.py V1 (seed 51000): ratios (1, sqrt2, phi),
period 2s, relative frequency blur 0.007. See demo-concept.md for the frozen
captions and production-report.md for the sanity check against 131's cached
panels at t = 15 / 40 / 100s.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HERE = Path(__file__).parent
AFMHOT = "afmhot"

# ---------------------------------------------------------------------------
# Pendulum ensemble - ported verbatim (RK4, same equations, same seed 50000
# as 131_demo_suite's run_v1 / the exemplar double_pendulum_anim.py)
# ---------------------------------------------------------------------------
G, L1, L2, M1, M2, DT = 9.81, 1.0, 1.0, 1.0, 1.0, 0.01
PERTURBATION = 0.05
PEND_EXTENT = [-2.2, 2.2, -2.2, 2.2]


def pendulum_rk4_step(t1, t2, om1, om2, dt):
    def derivatives(th1, th2, w1, w2):
        delta = th1 - th2
        den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
        den2 = (L1 / L2) * den1
        dw1 = (M2 * L1 * w1 * w1 * np.sin(delta) * np.cos(delta) +
               M2 * G * np.sin(th2) * np.cos(delta) +
               M2 * L2 * w2 * w2 * np.sin(delta) -
               (M1 + M2) * G * np.sin(th1)) / den1
        dw2 = (-M2 * L2 * w2 * w2 * np.sin(delta) * np.cos(delta) +
               (M1 + M2) * G * np.sin(th1) * np.cos(delta) -
               (M1 + M2) * L1 * w1 * w1 * np.sin(delta) -
               (M1 + M2) * G * np.sin(th2)) / den2
        return w1, w2, dw1, dw2

    k1a, k1b, k1c, k1d = derivatives(t1, t2, om1, om2)
    k2a, k2b, k2c, k2d = derivatives(t1 + 0.5 * dt * k1a, t2 + 0.5 * dt * k1b,
                                     om1 + 0.5 * dt * k1c, om2 + 0.5 * dt * k1d)
    k3a, k3b, k3c, k3d = derivatives(t1 + 0.5 * dt * k2a, t2 + 0.5 * dt * k2b,
                                     om1 + 0.5 * dt * k2c, om2 + 0.5 * dt * k2d)
    k4a, k4b, k4c, k4d = derivatives(t1 + dt * k3a, t2 + dt * k3b,
                                     om1 + dt * k3c, om2 + dt * k3d)
    return (t1 + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a),
            t2 + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b),
            om1 + (dt / 6.0) * (k1c + 2 * k2c + 2 * k3c + k4c),
            om2 + (dt / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d))


def pendulum_xy(th1, th2):
    return L1 * np.sin(th1) + L2 * np.sin(th2), -L1 * np.cos(th1) - L2 * np.cos(th2)


# ---------------------------------------------------------------------------
# Rotor ensemble - ported verbatim from 131_demo_suite run_v1 (seed 51000)
# ---------------------------------------------------------------------------
TWO_PI = 2 * np.pi
GOLD = (1 + np.sqrt(5)) / 2
OMEGA_BASE = TWO_PI / 2.0
RATIOS = np.array([1.0, np.sqrt(2.0), GOLD])
ROTOR_EXTENT = [-1.1, 1.1, -1.1, 1.1]


def rotor_xy(th0, omegas, t):
    ang = th0 + omegas * t
    x = (np.cos(ang[0]) + np.cos(ang[1])) / 2
    y = (np.sin(ang[0]) + np.sin(ang[2])) / 2
    return x, y


def cloud_hist(x, y, extent, bins=100):
    H, _, _ = np.histogram2d(x, y, bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                              density=True)
    return H.T ** 0.5


def style_ax(ax):
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")


CAPTION_1 = ("50,000 copies of one deterministic pendulum, started almost - but not exactly -\n"
             "the same way. I cannot know which copy is mine, so all I can honestly draw is a\n"
             "cloud of maybes.")
CAPTION_2 = ("Same uncertainty. Same determinism. Different fate. Chaos erases structure;\n"
             "incommensurate clockwork weaves it.")
CAPTION_3 = ("Quantum mechanics has the spectral fingerprint of the right-hand column - that\n"
             "theorem redirected this whole programme.")


def run(n_members, fps, out_stub, smoke=False):
    t0 = time.time()
    rng = np.random.default_rng(50000)
    th1 = np.pi / 2 + rng.normal(0, PERTURBATION, n_members)
    th2 = np.pi / 2 + rng.normal(0, PERTURBATION, n_members)
    w1 = rng.normal(0, PERTURBATION, n_members)
    w2 = rng.normal(0, PERTURBATION, n_members)

    rng2 = np.random.default_rng(51000)
    th0 = np.pi / 2 + rng2.normal(0, PERTURBATION, (3, n_members))
    omegas = OMEGA_BASE * RATIOS[:, None] * (1 + rng2.normal(0, 0.007, (3, n_members)))

    STEP_T15 = int(round(15.0 / DT))
    STEP_T40 = int(round(40.0 / DT))
    STEP_T100 = int(round(100.0 / DT))

    PHASE1_S, PHASE2_S, PHASE3_S = (5, 6, 3) if smoke else (15, 20, 10)
    frames_p1 = max(1, int(fps * PHASE1_S))
    frames_p2 = max(1, int(fps * PHASE2_S))
    frames_p3 = max(1, int(fps * PHASE3_S))
    total_frames = frames_p1 + frames_p2 + frames_p3

    bounds_p1 = np.linspace(0, STEP_T15, frames_p1 + 1).round().astype(int)
    bounds_p2 = np.linspace(STEP_T15, STEP_T100, frames_p2 + 1).round().astype(int)

    print(f"[{'SMOKE' if smoke else 'FULL'}] n_members={n_members} fps={fps} "
          f"frames=({frames_p1}+{frames_p2}+{frames_p3})={total_frames} "
          f"pendulum steps to t=100s: {STEP_T100}")

    # --- sanity-check snapshots (exact reproduction check vs 131 V1) ---
    sanity = {}
    step_count = 0
    th1s, th2s, w1s, w2s = th1.copy(), th2.copy(), w1.copy(), w2.copy()
    for target, label in [(STEP_T15, 15), (STEP_T40, 40), (STEP_T100, 100)]:
        while step_count < target:
            th1s, th2s, w1s, w2s = pendulum_rk4_step(th1s, th2s, w1s, w2s, DT)
            step_count += 1
        x, y = pendulum_xy(th1s, th2s)
        rx, ry = rotor_xy(th0, omegas, float(label))
        sanity[label] = {"pend_xy": (x, y), "rotor_xy": (rx, ry)}
    print(f"  sanity snapshots captured at t=15,40,100 ({time.time()-t0:.1f}s)")

    cache_path = HERE.parent.parent / "131_demo_suite" / "v1_pendulum_cache.npz"
    sanity_report_lines = []
    if cache_path.exists() and n_members == 50000:
        dat = np.load(cache_path)
        for label in (15, 40, 100):
            cx, cy = dat[f"x{label}"], dat[f"y{label}"]
            mx, my = sanity[label]["pend_xy"]
            dmax = max(np.max(np.abs(cx - mx)), np.max(np.abs(cy - my)))
            line = f"  pendulum t={label}s vs 131 cache: max abs diff = {dmax:.3e}"
            print(line)
            sanity_report_lines.append(line)
    else:
        sanity_report_lines.append("  (pendulum cache not found or n_members != 50000; skipped numeric diff)")

    # --- figure setup ---
    fig = plt.figure(figsize=(10, 6.6), facecolor="black")
    ax_full = fig.add_axes([0.06, 0.19, 0.88, 0.70])
    ax_left = fig.add_axes([0.04, 0.19, 0.44, 0.70])
    ax_right = fig.add_axes([0.52, 0.19, 0.44, 0.70])
    for a in (ax_full, ax_left, ax_right):
        style_ax(a)
    title = fig.suptitle("", color="white", fontsize=15, y=0.965)
    caption_txt = fig.text(0.5, 0.135, "", ha="center", va="top", color="#dddddd", fontsize=10.5)

    def rescale(im, H):
        vmax = np.percentile(H, 99.5)
        im.set_clim(0, vmax if vmax > 0 else 1.0)

    x0, y0 = pendulum_xy(th1, th2)
    im_full = ax_full.imshow(cloud_hist(x0, y0, PEND_EXTENT), extent=PEND_EXTENT,
                              origin="lower", cmap=AFMHOT, animated=True)
    im_left = ax_left.imshow(cloud_hist(x0, y0, PEND_EXTENT), extent=PEND_EXTENT,
                              origin="lower", cmap=AFMHOT, animated=True)
    rx0, ry0 = rotor_xy(th0, omegas, 0.0)
    im_right = ax_right.imshow(cloud_hist(rx0, ry0, ROTOR_EXTENT), extent=ROTOR_EXTENT,
                                origin="lower", cmap=AFMHOT, animated=True)
    ax_left.set_visible(False)
    ax_right.set_visible(False)

    state = {"th1": th1, "th2": th2, "w1": w1, "w2": w2, "steps_done": 0, "phase": 1}

    def update(frame):
        if frame < frames_p1:
            if state["phase"] != 1:
                state["phase"] = 1
            target = bounds_p1[frame + 1]
            while state["steps_done"] < target:
                state["th1"], state["th2"], state["w1"], state["w2"] = pendulum_rk4_step(
                    state["th1"], state["th2"], state["w1"], state["w2"], DT)
                state["steps_done"] += 1
            t_sim = state["steps_done"] * DT
            x, y = pendulum_xy(state["th1"], state["th2"])
            H = cloud_hist(x, y, PEND_EXTENT)
            im_full.set_array(H)
            rescale(im_full, H)
            ax_full.set_visible(True)
            ax_left.set_visible(False)
            ax_right.set_visible(False)
            title.set_text(f"The Cloud of Maybes  -  t = {t_sim:.1f}s")
            caption_txt.set_text(CAPTION_1)
            artists = [im_full, title, caption_txt]
        elif frame < frames_p1 + frames_p2:
            j = frame - frames_p1
            if state["phase"] != 2:
                state["phase"] = 2
            target = bounds_p2[j + 1]
            while state["steps_done"] < target:
                state["th1"], state["th2"], state["w1"], state["w2"] = pendulum_rk4_step(
                    state["th1"], state["th2"], state["w1"], state["w2"], DT)
                state["steps_done"] += 1
            t_sim = state["steps_done"] * DT
            x, y = pendulum_xy(state["th1"], state["th2"])
            rx, ry = rotor_xy(th0, omegas, t_sim)
            HL = cloud_hist(x, y, PEND_EXTENT)
            HR = cloud_hist(rx, ry, ROTOR_EXTENT)
            im_left.set_array(HL)
            im_right.set_array(HR)
            rescale(im_left, HL)
            rescale(im_right, HR)
            ax_full.set_visible(False)
            ax_left.set_visible(True)
            ax_right.set_visible(True)
            title.set_text(f"Chaos vs Clockwork  -  t = {t_sim:.1f}s")
            caption_txt.set_text(CAPTION_2)
            artists = [im_left, im_right, title, caption_txt]
        else:
            # frozen punchline: hold last computed state, swap caption
            ax_full.set_visible(False)
            ax_left.set_visible(True)
            ax_right.set_visible(True)
            title.set_text("Chaos vs Clockwork  -  t = 100.0s (frozen)")
            caption_txt.set_text(CAPTION_3)
            artists = [im_left, im_right, title, caption_txt]

        if frame % 100 == 0:
            print(f"  frame {frame}/{total_frames} ({time.time()-t0:.1f}s)")
        return artists

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    mp4_path = HERE / f"{out_stub}.mp4"
    gif_path = HERE / f"{out_stub}.gif"
    saved_path = None
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=110)
        saved_path = mp4_path
    except Exception as e:
        print(f"  ffmpeg unavailable ({e}); falling back to GIF")
        ani.save(gif_path, writer="pillow", fps=fps)
        saved_path = gif_path
    plt.close(fig)
    print(f"  animation saved to {saved_path} ({time.time()-t0:.1f}s)")

    # --- static three-panel fallback ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="black")
    fig2.suptitle("The Cloud of Maybes", color="white", fontsize=16)
    style_ax(axes2[0])
    axes2[0].imshow(cloud_hist(*sanity[15]["pend_xy"], PEND_EXTENT), extent=PEND_EXTENT,
                     origin="lower", cmap=AFMHOT)
    axes2[0].set_title("t = 15s\n(the bloom)", color="white", fontsize=10)

    style_ax(axes2[1])
    px, py = sanity[100]["pend_xy"]
    rxp, ryp = sanity[100]["rotor_xy"]
    axes2[1].imshow(cloud_hist(px, py, PEND_EXTENT), extent=PEND_EXTENT,
                     origin="lower", cmap=AFMHOT)
    axes2[1].set_title("t = 100s\npendulum (chaos smears)", color="white", fontsize=10)

    style_ax(axes2[2])
    axes2[2].imshow(cloud_hist(rxp, ryp, ROTOR_EXTENT), extent=ROTOR_EXTENT,
                     origin="lower", cmap=AFMHOT)
    axes2[2].set_title("t = 100s\nrotors (clockwork weaves)", color="white", fontsize=10)

    fig2.text(0.5, 0.02, CAPTION_3, ha="center", color="#cccccc", fontsize=10)
    fig2.tight_layout(rect=[0, 0.08, 1, 0.90])
    static_path = HERE / "probability_wave_static.png"
    fig2.savefig(static_path, facecolor="black", dpi=140)
    plt.close(fig2)
    print(f"  static panel saved to {static_path}")

    print("\nWHAT TO LOOK FOR: Watch the single cloud in phase 1 bloom from a near-point "
          "into a filamented orbital shape - that filament is the honest probability wave "
          "an outside observer is forced to draw. In phase 2, watch the two panels diverge: "
          "the left (chaotic pendulum) cloud thins into a diffuse, structureless haze, while "
          "the right (incommensurate rotors) cloud tightens into finer and finer organized "
          "lace that keeps its cross-shaped structure. Both start from identical uncertainty "
          "and identical determinism - only the right-hand kind of clockwork keeps quantum's "
          "fingerprint.")

    return sanity_report_lines, saved_path, total_frames, time.time() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        run(n_members=5000, fps=8, out_stub="probability_wave_smoke", smoke=True)
    else:
        run(n_members=50000, fps=20, out_stub="probability_wave_demo", smoke=False)
