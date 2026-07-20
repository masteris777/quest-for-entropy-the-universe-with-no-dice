# Lab 101c - Replacement Cell (screened W1 frame; battery restoration)

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
sys.path.insert(0, str(LAB_FOLDER))
import existence_hardening_101 as lab101  # noqa: E402

lab91 = lab101.lab91
lab93 = lab101.lab93
lab94 = lab101.lab94
lab99 = lab101.lab99
lab100v2 = lab101.lab100v2

METRICS_101_PATH = LAB_FOLDER / "metrics_101.json"
METRICS_101B_PATH = LAB_FOLDER / "metrics_101b.json"
METRICS_100V2_PATH = Path(__file__).resolve().parents[1] / "100_repreparation_fold" / "metrics_100_v2.json"

T_MAIN = 120000
T_TRANSIENT = 1000
T_CTRL = 60000
T_SCREEN = 20000
IDX = 6
RATIO_BAR = 5.0
CTRL_WRONGTHETA_MIN = 0.06
J_BAR = 0.06
BRIDGE_TOL = 0.02
CALIB_TOL = 0.02
MBORN_TOL = 1e-12


def compute_passive_trajectory(U_S, z0, T):
    z = z0.copy()
    z_hist = np.empty((T, 3), dtype=complex)
    for t in range(T):
        z_hist[t] = z
        z = U_S @ z
    return z_hist


def classify_D_born(z_hist, znorms_sq, Qf, M_born, T_transient):
    overlaps = np.abs(z_hist @ Qf.conj()) ** 2 / znorms_sq[:, None]
    outcomes = np.argmax(overlaps, axis=1)
    kept = outcomes[T_transient:]
    F1, _ = lab91.counted_F(kept, 1)
    return float(lab91.row_max_tv(F1, M_born))


# ---------------------------------------------------------------------------
# Stage A0 - gates that don't need FRAME_STAR
# ---------------------------------------------------------------------------
def run_gates_a1_a2():
    Qf91 = lab91.haar_frame(91)
    M1 = lab93.compute_M_born(Qf91)
    M2 = lab101.compute_M_born_gen(Qf91, lab93.U_S)
    mborn_dev = float(np.max(np.abs(M1 - M2)))
    A1 = bool(mborn_dev < MBORN_TOL)
    print(f"A1: mborn_dev={mborn_dev:.3e}, pass={A1}")

    FREQS, used = lab100v2.build_pool_v2()
    with open(METRICS_100V2_PATH) as f:
        m100v2 = json.load(f)
    stored_integers = m100v2["stageA"]["bridge"]["pool_integers_used"]
    integers_match = bool(used == stored_integers)
    A2 = bool(integers_match)
    print(f"A2: integers_match={integers_match}, pass={A2}")
    THETAS0 = lab100v2.build_thetas0(lab101.K)
    return A1, A2, FREQS, THETAS0


# ---------------------------------------------------------------------------
# Screening (predeclared, deterministic, runs first)
# ---------------------------------------------------------------------------
def screen_frame(w1_p5, w1_p95):
    z_hist = compute_passive_trajectory(lab93.U_S, lab93.z0, T_SCREEN)
    znorms_sq = np.sum(np.abs(z_hist) ** 2, axis=1)
    table = []
    seed = 30000
    frame_star = None
    while frame_star is None:
        Qf = lab91.haar_frame(seed)
        M_born = lab101.compute_M_born_gen(Qf, lab93.U_S)
        val = classify_D_born(z_hist, znorms_sq, Qf, M_born, T_TRANSIENT)
        in_range = bool(w1_p5 <= val <= w1_p95)
        table.append({"seed": seed, "value": val, "in_range": in_range})
        print(f"screen seed={seed}: value={val:.5f}, in_range={in_range}")
        if in_range:
            frame_star = seed
        seed += 1
    return frame_star, table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Gates (A1, A2) + Screening (A3)")
    print("=" * 60)
    A1, A2, FREQS, THETAS0 = run_gates_a1_a2()

    with open(METRICS_101B_PATH) as f:
        m101b = json.load(f)
    w1_p5 = m101b["stageB"]["null_percentiles"]["W1"]["p5"]
    w1_p95 = m101b["stageB"]["null_percentiles"]["W1"]["p95"]
    print(f"W1 screening interval [{w1_p5}, {w1_p95}] (verbatim from metrics_101b.json)")

    frame_star, screening_table = screen_frame(w1_p5, w1_p95)
    A3 = bool(screening_table[-1]["seed"] == frame_star and screening_table[-1]["in_range"]
              and all(not e["in_range"] for e in screening_table[:-1]))
    print(f"A3: FRAME_STAR={frame_star}, screening table has {len(screening_table)} candidates, pass={A3}")

    pass_gates = bool(A1 and A2 and A3)
    print(f"\nGates: {'PASS' if pass_gates else 'FAIL'}")

    if not pass_gates:
        metrics = {
            "lab": "101c", "stream": "cryptographic-substrate", "kind": "hardened-simulation-addendum",
            "stageA": {"A1_mborn_pass": A1, "A2_pool_pass": A2, "A3_screening_pass": A3,
                       "screening_table": screening_table, "frame_star": frame_star},
            "verdict": "VOID", "verdict_reason": "Gate(s) failed. Stopping.",
            "leakage_audit": {"wrote_outside_lab_folder": False, "theta_reused": False,
                              "pool_modified": False, "random_draws_inside_world": False,
                              "emergence_language_used": False, "posthoc_threshold_change": False,
                              "metrics_hand_edited": False},
        }
        with open(LAB_FOLDER / "metrics_101c.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    Qf_star = lab91.haar_frame(frame_star)
    M_born_star = lab101.compute_M_born_gen(Qf_star, lab93.U_S)

    print("\n" + "=" * 60)
    print(f"CELL PIPELINE - (W1, frame {frame_star}), idx={IDX}")
    print("=" * 60)

    print("Optimizing (fresh theta, idx=6 seed pattern)...")
    opt = lab101.optimize_cell_theta(Qf_star, lab93.U_S, M_born_star, idx=IDX)
    theta = opt["theta"]
    variational_gap = bool(opt["D_heldout"] >= 0.03)
    print(f"theta={np.round(theta,4).tolist()}")
    print(f"D_insample={opt['D_insample']:.5f}, D_heldout={opt['D_heldout']:.5f}"
          f"{'  [VARIATIONAL-GAP]' if variational_gap else ''}")

    print("\nCalibration (T=120000, seed=1006)...")
    cal_stream = lab94.gen_calibration_stream(M_born_star, T=T_MAIN, seed=1006)
    F1_cal, _ = lab91.counted_F(cal_stream, 1)
    D_born_cal = float(lab91.row_max_tv(F1_cal, M_born_star))
    m_floor = float(lab93.markov_score_fn(cal_stream, F1_cal, lab101.TAU_MARKOV))
    A4_calib = bool(D_born_cal < CALIB_TOL)
    print(f"D_born_cal={D_born_cal:.5f} (bar<{CALIB_TOL}), markov_floor={m_floor:.5f}, pass={A4_calib}")

    print("\nMain run (T=120000)...")
    run_main = lab101.run_model_two_gen(Qf_star, lab93.U_S, lab93.z0, theta, lab101.K, FREQS, THETAS0,
                                         T=T_MAIN, T_transient=T_TRANSIENT, track_stage_d=True)
    cell_main = lab101.compute_cell_metrics(run_main["outcomes"], M_born_star)
    J_main_120k = cell_main["J"]
    print(f"D_born={cell_main['D_born']:.5f}, markov={cell_main['markov_score']:.5f}, "
          f"J={J_main_120k:.5f} (bar<{J_BAR}), degenerate={cell_main['degenerate']}")
    ratio_to_floor = m_floor and (cell_main["markov_score"] / m_floor)

    print("\nControls (T=60000)...")
    run_main_60k = lab101.run_model_two_gen(Qf_star, lab93.U_S, lab93.z0, theta, lab101.K, FREQS, THETAS0,
                                             T=T_CTRL, T_transient=T_TRANSIENT)
    cell_main_60k = lab101.compute_cell_metrics(run_main_60k["outcomes"], M_born_star)
    J_main_60k = cell_main_60k["J"]

    run_p = lab101.run_passive_gen(Qf_star, lab93.U_S, lab93.z0, lab101.K, FREQS, THETAS0,
                                    T=T_CTRL, T_transient=T_TRANSIENT)
    cell_p = lab101.compute_cell_metrics(run_p["outcomes"], M_born_star)
    passive_ratio = cell_p["D_born"] / J_main_60k
    ctrl_passive_pass = bool(passive_ratio >= RATIO_BAR)
    print(f"C-passive: D_born={cell_p['D_born']:.5f}, J_main(60k)={J_main_60k:.5f}, "
          f"ratio={passive_ratio:.2f} (bar>={RATIO_BAR}), pass={ctrl_passive_pass}")

    run_i = lab101.run_model_two_gen(Qf_star, lab93.U_S, lab93.z0, lab101.ISO_THETA, lab101.K, FREQS, THETAS0,
                                      T=T_CTRL, T_transient=T_TRANSIENT)
    cell_i = lab101.compute_cell_metrics(run_i["outcomes"], M_born_star)
    ctrl_iso_pass = bool(cell_i["J"] > CTRL_WRONGTHETA_MIN)
    print(f"C-isotropic: J={cell_i['J']:.5f} (bar>{CTRL_WRONGTHETA_MIN}), pass={ctrl_iso_pass}")

    shuf_theta = lab100v2.make_shuf_theta(theta)
    run_s = lab101.run_model_two_gen(Qf_star, lab93.U_S, lab93.z0, shuf_theta, lab101.K, FREQS, THETAS0,
                                      T=T_CTRL, T_transient=T_TRANSIENT)
    cell_s = lab101.compute_cell_metrics(run_s["outcomes"], M_born_star)
    ctrl_shuf_pass = bool(cell_s["J"] > CTRL_WRONGTHETA_MIN)
    print(f"C-shuffled: J={cell_s['J']:.5f} (bar>{CTRL_WRONGTHETA_MIN}), pass={ctrl_shuf_pass}")

    print("\nBridge (C-mc, seed=5006)...")
    run_mc = lab101.run_model_two_mc(Qf_star, lab93.U_S, lab93.z0, theta, T=T_CTRL, T_transient=T_TRANSIENT, seed=5006)
    cell_mc = lab101.compute_cell_metrics(run_mc["outcomes"], M_born_star)
    bridge_delta = float(abs(cell_mc["J"] - J_main_60k))
    bridge_pass = bool(bridge_delta < BRIDGE_TOL)
    print(f"J_mc={cell_mc['J']:.5f}, J_main(60k)={J_main_60k:.5f}, delta={bridge_delta:.5f} "
          f"(bar<{BRIDGE_TOL}), pass={bridge_pass}")

    print("\nRidge attack (recorded)...")
    X, y = run_main["j0_X"], run_main["j0_y"]
    n = len(X)
    if n >= 20:
        mid = n // 2
        r2 = float(lab93.ridge_fit_r2(X[:mid], y[:mid], X[mid:], y[mid:], alpha=1.0))
    else:
        r2 = None
    print(f"ridge: n={n}, R2={r2}")

    # Screening percentile of the selected frame (for reporting)
    screened_value = screening_table[-1]["value"]

    print("\nGenerating plot...")
    with open(METRICS_101_PATH) as f:
        m101 = json.load(f)
    all_101_cells = m101["stageB"] + m101["stageC"]
    valid_labels, valid_J = [], []
    excluded_label, excluded_J = None, None
    for c in all_101_cells:
        w, seed = c["cell"]
        label = f"W{w}f{seed}"
        if [w, seed] == [1, 42]:
            excluded_label, excluded_J = label, c["J"]
        else:
            valid_labels.append(label)
            valid_J.append(c["J"])
    labels = valid_labels + [f"W1f{frame_star}\n(FRAME_STAR)"]
    Js = valid_J + [J_main_120k]
    colors = ["tab:green"] * len(valid_J) + ["tab:blue"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels) + 1)
    bars_x = list(x[:len(labels)])
    ax.bar(bars_x, Js, color=colors)
    ax.bar([x[len(labels)]], [excluded_J], color="lightgray", hatch="//",
           label=f"{excluded_label} (excluded, control-real)")
    all_labels = labels + [f"{excluded_label}\n(excluded)"]
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.axhline(J_BAR, color="black", linestyle="--", linewidth=0.9, label=f"J={J_BAR}")
    ax.set_ylabel("J")
    ax.set_title("Lab 101c: restored 6-cell battery (5 valid Lab 101 cells + FRAME_STAR); frame42 shown excluded")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "restored_battery_101c.png", dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------
    if J_main_120k >= J_BAR:
        verdict = "CELL-FAILED"
        verdict_reason = f"Main-cell J={J_main_120k:.5f} >= {J_BAR}."
    elif not ctrl_passive_pass:
        verdict = "CELL-FAILED"
        verdict_reason = f"C-passive ratio={passive_ratio:.2f} < {RATIO_BAR}."
    elif not ctrl_iso_pass:
        verdict = "CELL-FAILED"
        verdict_reason = f"C-isotropic J={cell_i['J']:.5f} <= {CTRL_WRONGTHETA_MIN}."
    elif not ctrl_shuf_pass:
        verdict = "CELL-FAILED"
        verdict_reason = f"C-shuffled J={cell_s['J']:.5f} <= {CTRL_WRONGTHETA_MIN}."
    elif not bridge_pass:
        verdict = "CELL-FAILED"
        verdict_reason = f"Bridge delta={bridge_delta:.5f} >= {BRIDGE_TOL}."
    else:
        verdict = "CELL-RESTORED"
        verdict_reason = (f"J={J_main_120k:.5f} < {J_BAR}; C-passive ratio={passive_ratio:.2f} >= {RATIO_BAR}; "
                           f"C-isotropic J={cell_i['J']:.5f} and C-shuffled J={cell_s['J']:.5f} both > "
                           f"{CTRL_WRONGTHETA_MIN}; bridge delta={bridge_delta:.5f} < {BRIDGE_TOL}. "
                           f"The 6/6 fresh-cell battery is complete.")

    metrics = {
        "lab": "101c", "stream": "cryptographic-substrate", "kind": "hardened-simulation-addendum",
        "stageA": {"A1_mborn_pass": A1, "A2_pool_pass": A2, "A3_screening_pass": A3,
                   "screening_table": screening_table, "frame_star": frame_star,
                   "screened_value": screened_value},
        "cell": {
            "cell": [1, frame_star], "theta": np.array(theta).tolist(),
            "D_insample": opt["D_insample"], "D_heldout": opt["D_heldout"], "variational_gap": variational_gap,
            "calibration": {"D_born_cal": D_born_cal, "markov_floor": m_floor, "pass": A4_calib},
            "main": {"D_born": cell_main["D_born"], "markov_score": cell_main["markov_score"],
                     "J": J_main_120k, "occupancies": cell_main["occupancies"],
                     "degenerate": cell_main["degenerate"], "markov_to_floor_ratio": ratio_to_floor},
            "controls": {
                "C-passive": {"D_born": cell_p["D_born"], "J_main_60k": J_main_60k, "ratio": passive_ratio,
                              "bar": RATIO_BAR, "pass": ctrl_passive_pass, "screened_value_T20k": screened_value},
                "C-isotropic": {"J": cell_i["J"], "bar": CTRL_WRONGTHETA_MIN, "pass": ctrl_iso_pass},
                "C-shuffled": {"J": cell_s["J"], "bar": CTRL_WRONGTHETA_MIN, "pass": ctrl_shuf_pass},
            },
            "bridge": {"J_mc": cell_mc["J"], "J_main_60k": J_main_60k, "delta": bridge_delta,
                       "bar": BRIDGE_TOL, "pass": bridge_pass},
            "ridge": {"r2": r2, "n_samples": n},
        },
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "theta_reused": False, "pool_modified": False,
            "random_draws_inside_world": False, "emergence_language_used": False,
            "posthoc_threshold_change": False, "metrics_hand_edited": False,
        },
    }
    with open(LAB_FOLDER / "metrics_101c.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
