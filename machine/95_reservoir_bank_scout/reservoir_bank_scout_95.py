# Lab 95 - Reservoir-Bank Scout (BANK-K, joint criterion)

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB94_DIR = Path(__file__).resolve().parents[1] / "94_fold_reservoir_scout"
sys.path.insert(0, str(LAB94_DIR))
import fold_reservoir_scout_94 as lab94  # noqa: E402

lab93 = lab94.lab93
lab91 = lab94.lab91

LAB94_METRICS_PATH = LAB94_DIR / "metrics_94.json"

# ---------------------------------------------------------------------------
# Predeclared parameters
# ---------------------------------------------------------------------------
K_GRID = [4, 16, 64]
G_GRID = [0.50, 0.65, 0.7853981634, 0.90, 1.00, 1.15, 1.30, 1.45, 1.5707963268]
FRAME_SEEDS = [91, 93, 201]
T_SWEEP = 60000
T_TRANSIENT = 1000
TAU_MARKOV = [2, 5, 10, 50]
DEGENERACY_FLOOR = 0.05
CAL_SEED = 950
J_GRADUATE = 0.06
J_PROMISING = 0.12
J_FLOOR_HI = 0.20
K_SATURATION_EPS = 0.01


# ---------------------------------------------------------------------------
# BANK-K generator (predeclared frequency/init rules, never tuned)
# ---------------------------------------------------------------------------
def first_non_squares(count, start=2):
    out = []
    n = start
    while len(out) < count:
        r = int(round(np.sqrt(n)))
        if r * r != n:
            out.append(n)
        n += 1
    return out


def build_bank(K):
    ns = first_non_squares(2 * K)  # n_1..n_2K, 1-indexed conceptually, 0-indexed list
    FREQS = np.empty((K, 2))
    for i in range(K):
        n_a = ns[2 * i]
        n_b = ns[2 * i + 1]
        FREQS[i, 0] = 2 * np.pi * (np.sqrt(n_a) % 1.0)
        FREQS[i, 1] = 2 * np.pi * (np.sqrt(n_b) % 1.0)
    thetas0 = np.empty((K, 2))
    for i in range(K):
        thetas0[i, 0] = (1.0 + 0.37 * i) % (2 * np.pi)
        thetas0[i, 1] = (2.0 + 0.73 * i) % (2 * np.pi)
    return FREQS, thetas0


# ---------------------------------------------------------------------------
# Bank step loop - NEW code; the fold's first rotation line is copied verbatim
# from Lab 93 v2 / Lab 94's run_model_one (embedded_fold_scout_93_v2.py line 168,
# also fold_reservoir_scout_94.py line ~168 in run_tape_variant):
#   z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp
# Only the reservoir source differs: instead of one persistent register (Lab 93)
# or one freshly-generated vector (Lab 94), the fold draws from generator
# (t % K) of a bank of K independently-ticking quasiperiodic phase pairs.
# ---------------------------------------------------------------------------
def run_bank_variant(Qf, g, K, FREQS, thetas0, T, T_transient=T_TRANSIENT, track_stage_d=False):
    z = lab93.z0.copy()
    thetas = thetas0.copy()
    outcomes = np.empty(T, dtype=np.int64)
    norm_z_hist = np.empty(T)
    cos_g, sin_g = np.cos(g), np.sin(g)

    basis0 = lab93.perp_basis(Qf[:, 0]) if track_stage_d else None
    history = []
    j0_X, j0_y = [], []

    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j
        norm_z_hist[t] = znorm

        phi_j = Qf[:, j]
        gamma = np.vdot(phi_j, z)
        z_perp = z - gamma * phi_j

        gen_idx = t % K
        r_gen = lab94.r_gen_from_theta(thetas[gen_idx])
        rho = np.vdot(phi_j, r_gen)
        r_perp = r_gen - rho * phi_j

        if track_stage_d and j == 0 and len(history) >= 8 and t >= T_transient:
            c1 = np.vdot(basis0[0], z_perp)
            c2 = np.vdot(basis0[1], z_perp)
            feat = np.zeros(24)
            for i, s_hist in enumerate(history[-8:]):
                feat[i * 3 + s_hist] = 1.0
            j0_X.append(feat)
            j0_y.append([c1.real, c1.imag, c2.real, c2.imag])

        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp  # verbatim rotation line

        z = lab93.U_S @ z_fold
        thetas = np.mod(thetas + FREQS, 2 * np.pi)  # ALL K generators tick every step (vectorized)

        if track_stage_d:
            history.append(j)
            if len(history) > 8:
                history.pop(0)

    result = {"outcomes": outcomes, "norm_z_hist": norm_z_hist}
    if track_stage_d:
        result["j0_X"] = np.array(j0_X)
        result["j0_y"] = np.array(j0_y)
    return result


def bank_roundtrip(Qf, g, K, FREQS, thetas0, n_steps=1000):
    z = lab93.z0.copy()
    thetas = thetas0.copy()
    z_hist = [z.copy()]
    thetas_hist = [thetas.copy()]
    j_hist = []
    tape = []
    cos_g, sin_g = np.cos(g), np.sin(g)

    for t in range(n_steps):
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)])
        j = int(np.argmax(overlaps))
        j_hist.append(j)
        phi_j = Qf[:, j]
        gamma = np.vdot(phi_j, z)
        z_perp = z - gamma * phi_j
        gen_idx = t % K
        r_gen = lab94.r_gen_from_theta(thetas[gen_idx])
        rho = np.vdot(phi_j, r_gen)
        r_perp = r_gen - rho * phi_j

        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp  # verbatim rotation line
        tape_t = -sin_g * z_perp + cos_g * r_perp
        tape.append(tape_t)

        z = lab93.U_S @ z_fold
        thetas = np.mod(thetas + FREQS, 2 * np.pi)
        z_hist.append(z.copy())
        thetas_hist.append(thetas.copy())

    max_err = 0.0
    US_dag = lab93.U_S.conj().T
    for t in range(n_steps):
        j = j_hist[t]
        phi_j = Qf[:, j]
        z_fold_rec = US_dag @ z_hist[t + 1]
        gamma = np.vdot(phi_j, z_fold_rec)
        z_fold_perp = z_fold_rec - gamma * phi_j
        tape_t = tape[t]
        z_perp_rec = cos_g * z_fold_perp - sin_g * tape_t
        r_perp_rec = sin_g * z_fold_perp + cos_g * tape_t
        z_rec = gamma * phi_j + z_perp_rec
        err_z = float(np.max(np.abs(z_rec - z_hist[t])))

        thetas_rec = np.mod(thetas_hist[t + 1] - FREQS, 2 * np.pi)
        err_theta = float(np.max(np.abs(np.exp(1j * thetas_rec) - np.exp(1j * thetas_hist[t]))))

        gen_idx = t % K
        r_gen_t = lab94.r_gen_from_theta(thetas_hist[t][gen_idx])
        rho_t = np.vdot(phi_j, r_gen_t)
        r_perp_t = r_gen_t - rho_t * phi_j
        err_rperp = float(np.max(np.abs(r_perp_rec - r_perp_t)))

        max_err = max(max_err, err_z, err_theta, err_rperp)
    return max_err


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def run_stage_a():
    with open(LAB94_METRICS_PATH) as f:
        lab94_metrics = json.load(f)

    frames = {seed: lab91.haar_frame(seed) for seed in FRAME_SEEDS}

    # A1 - continuity anchors (compared against the FULL-PRECISION values stored in
    # metrics_94.json, not the rounded display numbers quoted in lab-goal.md's prose)
    lab94_cells = lab94_metrics["stageB"]["cells"]
    target_cat = next(c for c in lab94_cells if c["variant"] == "R-CATGEN" and c["frame"] == 91
                       and abs(c["g"] - 1.15) < 1e-6)
    target_qp3 = next(c for c in lab94_cells if c["variant"] == "R-QP3" and c["frame"] == 91
                       and abs(c["g"] - 0.7853981634) < 1e-6)

    Qf91 = frames[91]
    run_cat = lab94.run_tape_variant(Qf91, 1.15, "R-CATGEN", T=T_SWEEP, track_stage_d=False)
    kept = run_cat["outcomes"][T_TRANSIENT:]
    M_born_91 = lab93.compute_M_born(Qf91)
    F1, _ = lab91.counted_F(kept, 1)
    catgen_D = float(lab91.row_max_tv(F1, M_born_91))
    catgen_M = float(lab93.markov_score_fn(kept, F1, TAU_MARKOV))

    run_qp3 = lab93.run_model_one(Qf91, g=0.7853981634, T=T_SWEEP, track_j0_samples=False)
    kept2 = run_qp3["outcomes"][T_TRANSIENT:]
    F1b, _ = lab91.counted_F(kept2, 1)
    qp3_D = float(lab91.row_max_tv(F1b, M_born_91))
    qp3_M = float(lab93.markov_score_fn(kept2, F1b, TAU_MARKOV))

    dev_cat_D = abs(catgen_D - target_cat["D_born"])
    dev_cat_M = abs(catgen_M - target_cat["markov_score"])
    dev_qp3_D = abs(qp3_D - target_qp3["D_born"])
    dev_qp3_M = abs(qp3_M - target_qp3["markov_score"])
    max_dev = float(max(dev_cat_D, dev_cat_M, dev_qp3_D, dev_qp3_M))
    A1 = bool(max_dev < 1e-9)
    print(f"A1 anchors: catgen_D={catgen_D:.8f} (target {target_cat['D_born']:.8f}), "
          f"catgen_M={catgen_M:.8f} (target {target_cat['markov_score']:.8f}), "
          f"qp3_D={qp3_D:.8f} (target {target_qp3['D_born']:.8f}), "
          f"qp3_M={qp3_M:.8f} (target {target_qp3['markov_score']:.8f}), max_dev={max_dev:.3e}, pass={A1}")

    # A2 - calibration floors
    M_borns = {seed: lab93.compute_M_born(frames[seed]) for seed in FRAME_SEEDS}
    markov_floors = {}
    kernel_devs = []
    for idx, seed in enumerate(FRAME_SEEDS):
        cal_stream = lab94.gen_calibration_stream(M_borns[seed], T=T_SWEEP, seed=CAL_SEED + idx)
        F1_cal, _ = lab91.counted_F(cal_stream, 1)
        D_born_cal = float(lab91.row_max_tv(F1_cal, M_borns[seed]))
        m_floor = float(lab93.markov_score_fn(cal_stream, F1_cal, TAU_MARKOV))
        markov_floors[str(seed)] = m_floor
        kernel_devs.append(D_born_cal)
        print(f"A2 frame {seed}: D_born_cal={D_born_cal:.5f}, markov_floor={m_floor:.5f}")
    kernel_max_dev = float(max(kernel_devs))
    A2 = bool(kernel_max_dev < 0.02)
    print(f"A2 pass: {A2} (kernel_max_dev={kernel_max_dev:.5f})")

    # A3 - bank tape round-trip, K=16, g=1.0, frame 91
    FREQS16, thetas0_16 = build_bank(16)
    bank_err = bank_roundtrip(Qf91, 1.0, 16, FREQS16, thetas0_16, n_steps=1000)
    A3 = bool(bank_err < 1e-10)
    print(f"A3 bank_roundtrip_max_err={bank_err:.3e}, pass={A3}")

    pass_a = bool(A1 and A2 and A3)

    stageA = {
        "anchors": {"catgen_D": catgen_D, "catgen_M": catgen_M, "qp3_D": qp3_D, "qp3_M": qp3_M,
                    "max_dev": max_dev, "pass": A1},
        "calibration": {"markov_floors": markov_floors, "kernel_max_dev": kernel_max_dev, "pass": A2},
        "bank_roundtrip_max_err": bank_err,
        "pass": pass_a,
    }
    return stageA, frames, M_borns, lab94_metrics


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def run_stage_b(frames, M_borns, lab94_metrics):
    # g=0 baselines: recompute via imported run_model_one (deterministic; cross-check against
    # metrics_94.json's stored scalars rather than reading a raw F1 matrix, which Lab 94 did not persist)
    baseline_F1 = {}
    baseline_check = {}
    lab94_baselines = {b["frame"]: b for b in lab94_metrics["stageB"]["baselines"]}
    for seed in FRAME_SEEDS:
        Qf = frames[seed]
        run = lab93.run_model_one(Qf, g=0.0, T=T_SWEEP, track_j0_samples=False)
        kept = run["outcomes"][T_TRANSIENT:]
        F1_g0, _ = lab91.counted_F(kept, 1)
        D_born_g0 = float(lab91.row_max_tv(F1_g0, M_borns[seed]))
        markov_g0 = float(lab93.markov_score_fn(kept, F1_g0, TAU_MARKOV))
        baseline_F1[seed] = F1_g0
        ref = lab94_baselines[seed]
        dev = max(abs(D_born_g0 - ref["D_born_g0"]), abs(markov_g0 - ref["markov_g0"]))
        baseline_check[seed] = dev
        print(f"baseline frame {seed}: D_born_g0={D_born_g0:.5f} (Lab94: {ref['D_born_g0']:.5f}, "
              f"dev={dev:.2e}), markov_g0={markov_g0:.5f}")

    cells = []
    stage_d_track = {}  # (K, frame) -> {g: (j0_X, j0_y)} filled lazily in stage D re-run

    for K in K_GRID:
        FREQS, thetas0 = build_bank(K)
        for seed in FRAME_SEEDS:
            Qf = frames[seed]
            for g in G_GRID:
                run = run_bank_variant(Qf, g, K, FREQS, thetas0, T=T_SWEEP, track_stage_d=False)
                kept = run["outcomes"][T_TRANSIENT:]
                counts = np.bincount(kept, minlength=3)
                occ = counts / counts.sum()
                degenerate = bool(np.min(occ) < DEGENERACY_FLOOR)
                max_occupancy = float(np.max(occ))

                F1, _ = lab91.counted_F(kept, 1)
                D_born = float(lab91.row_max_tv(F1, M_borns[seed]))
                D_classical = float(lab91.row_max_tv(F1, baseline_F1[seed]))
                m_score = float(lab93.markov_score_fn(kept, F1, TAU_MARKOV))
                J = float(max(D_born, m_score))
                mean_norm_z = float(np.mean(run["norm_z_hist"][T_TRANSIENT:]))

                cell = {
                    "K": K, "frame": seed, "g": float(g), "degenerate": degenerate,
                    "occupancies": [float(o) for o in occ], "max_occupancy": max_occupancy,
                    "D_born": D_born, "D_classical": D_classical, "markov_score": m_score,
                    "J": J, "mean_norm_z": mean_norm_z,
                }
                cells.append(cell)
                print(f"K={K} frame={seed} g={g:.4f}: D_born={D_born:.4f}, markov={m_score:.4f}, "
                      f"J={J:.4f}, max_occ={max_occupancy:.3f}, degenerate={degenerate}")

    return {"cells": cells, "baseline_check_max_dev": {str(k): v for k, v in baseline_check.items()}}


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
def lab94_best_single_source_J(lab94_metrics, frame):
    cells = lab94_metrics["stageB"]["cells"]
    nondeg = [c for c in cells if c["frame"] == frame and not c["degenerate"]]
    Js = [max(c["D_born"], c["markov_score"]) for c in nondeg]
    return float(min(Js)) if Js else None


def run_stage_c(stageB, lab94_metrics):
    cells = stageB["cells"]
    best_per_K_frame = []
    for K in K_GRID:
        for seed in FRAME_SEEDS:
            nondeg = [c for c in cells if c["K"] == K and c["frame"] == seed and not c["degenerate"]]
            if not nondeg:
                best_per_K_frame.append({"K": K, "frame": seed, "g": None, "D_born": None,
                                          "markov": None, "bestJ": None})
                continue
            best = min(nondeg, key=lambda c: c["J"])
            best_per_K_frame.append({"K": K, "frame": seed, "g": best["g"], "D_born": best["D_born"],
                                      "markov": best["markov_score"], "bestJ": best["J"]})

    per_frame = []
    for seed in FRAME_SEEDS:
        entries = [e for e in best_per_K_frame if e["frame"] == seed and e["bestJ"] is not None]
        if not entries:
            per_frame.append({"frame": seed, "K_star": None, "g_star": None, "J_star": None,
                               "saturated_16_vs_64": None, "saturated_4_vs_16": None,
                               "beats_lab94_single_source": None})
            continue
        best_overall = min(entries, key=lambda e: e["bestJ"])
        bestJ_by_K = {e["K"]: e["bestJ"] for e in entries}
        sat_16_64 = (4 in bestJ_by_K or True) and (16 in bestJ_by_K and 64 in bestJ_by_K) and \
            (abs(bestJ_by_K[64] - bestJ_by_K[16]) < K_SATURATION_EPS)
        sat_4_16 = (4 in bestJ_by_K and 16 in bestJ_by_K) and \
            (abs(bestJ_by_K[16] - bestJ_by_K[4]) < K_SATURATION_EPS)
        ref_single = lab94_best_single_source_J(lab94_metrics, seed)
        beats = bool(ref_single is not None and best_overall["bestJ"] < ref_single)
        per_frame.append({
            "frame": seed, "K_star": best_overall["K"], "g_star": best_overall["g"],
            "J_star": best_overall["bestJ"],
            "saturated_16_vs_64": bool(sat_16_64) if (16 in bestJ_by_K and 64 in bestJ_by_K) else None,
            "saturated_4_vs_16": bool(sat_4_16) if (4 in bestJ_by_K and 16 in bestJ_by_K) else None,
            "beats_lab94_single_source": beats,
            "lab94_reference_J": ref_single,
        })
        print(f"frame {seed}: J*={best_overall['bestJ']:.4f} at K*={best_overall['K']}, g*={best_overall['g']:.4f}; "
              f"bestJ_by_K={bestJ_by_K}; lab94_ref={ref_single}; beats={beats}")

    return {"best_per_K_frame": best_per_K_frame, "per_frame": per_frame}


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------
def run_stage_d(stageC, frames):
    attacks = []
    for entry in stageC["per_frame"]:
        seed = entry["frame"]
        K_star, g_star = entry["K_star"], entry["g_star"]
        if K_star is None:
            attacks.append({"frame": seed, "K": None, "g": None, "r2": None, "n_samples": 0})
            continue
        FREQS, thetas0 = build_bank(K_star)
        run = run_bank_variant(frames[seed], g_star, K_star, FREQS, thetas0, T=T_SWEEP, track_stage_d=True)
        X, y = run["j0_X"], run["j0_y"]
        n = len(X)
        if n < 20:
            attacks.append({"frame": seed, "K": K_star, "g": g_star, "r2": None, "n_samples": n})
            print(f"Stage D frame {seed}: K*={K_star}, g*={g_star}, n_samples={n} (too few)")
            continue
        mid = n // 2
        r2 = lab93.ridge_fit_r2(X[:mid], y[:mid], X[mid:], y[mid:], alpha=1.0)
        attacks.append({"frame": seed, "K": K_star, "g": g_star, "r2": r2, "n_samples": n})
        print(f"Stage D frame {seed}: K*={K_star}, g*={g_star:.4f}, n_samples={n}, R2={r2:.4f}")
    return {"attacks": attacks}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(stageB, lab94_metrics):
    cells = stageB["cells"]
    lab94_cells = lab94_metrics["stageB"]["cells"]
    colors = {4: "tab:blue", 16: "tab:orange", 64: "tab:green"}

    fig, axes = plt.subplots(1, len(FRAME_SEEDS), figsize=(16, 5.5))
    for ax, seed in zip(axes, FRAME_SEEDS):
        grey = [c for c in lab94_cells if c["frame"] == seed]
        ax.scatter([c["D_born"] for c in grey], [c["markov_score"] for c in grey],
                   color="lightgray", marker=".", s=25, label="Lab 94 single-source", zorder=1)
        for K in K_GRID:
            kc = [c for c in cells if c["frame"] == seed and c["K"] == K]
            ax.scatter([c["D_born"] for c in kc], [c["markov_score"] for c in kc],
                       color=colors[K], marker="o", s=30, label=f"K={K}", zorder=2,
                       edgecolors="black" if False else None)
        ax.axvline(0.06, color="green", linestyle="--", linewidth=0.8)
        ax.axhline(0.06, color="green", linestyle="--", linewidth=0.8)
        ax.axvline(0.12, color="orange", linestyle="--", linewidth=0.8)
        ax.axhline(0.12, color="orange", linestyle="--", linewidth=0.8)
        ax.set_xlabel("D_born")
        ax.set_ylabel("markov_score")
        ax.set_title(f"frame {seed}")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Joint frontier: bank cells (colored by K) vs Lab 94 single-source (grey); "
                  "dashed lines at J=0.06 (green) and J=0.12 (orange)")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "joint_frontier_95.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    stageA, frames, M_borns, lab94_metrics = run_stage_a()
    print(f"\nStage A: {'PASS' if stageA['pass'] else 'FAIL'}")

    if not stageA["pass"]:
        metrics = {
            "lab": 95, "stream": "cryptographic-substrate", "kind": "scout",
            "serves_node": "born_rule_emergence",
            "stageA": stageA,
            "stageB": {"cells": []},
            "stageC": {"best_per_K_frame": [], "per_frame": []},
            "stageD": {"attacks": []},
            "verdict": "VOID",
            "verdict_reason": "Stage A gate(s) failed. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "fold_formula_modified": False,
                "born_draws_inside_world": False, "posthoc_threshold_change": False,
                "frequency_tuning_after_results": False,
            },
        }
        with open(LAB_FOLDER / "metrics_95.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - The (K, g) sweep")
    print("=" * 60)
    stageB = run_stage_b(frames, M_borns, lab94_metrics)

    print("\n" + "=" * 60)
    print("STAGE C - Joint decision (argmin of J)")
    print("=" * 60)
    stageC = run_stage_c(stageB, lab94_metrics)

    print("\n" + "=" * 60)
    print("STAGE D - Arrow check at the optimum")
    print("=" * 60)
    stageD = run_stage_d(stageC, frames)

    print("\nGenerating plot...")
    make_plot(stageB, lab94_metrics)

    # Verdict
    valid_frames = [e for e in stageC["per_frame"] if e["J_star"] is not None]
    n_graduate = sum(1 for e in valid_frames if e["J_star"] < J_GRADUATE)
    n_promising = sum(1 for e in valid_frames if e["J_star"] < J_PROMISING)
    n_null = sum(1 for e in valid_frames if e["J_star"] > J_FLOOR_HI)
    floor_band = all(J_PROMISING <= e["J_star"] <= J_FLOOR_HI for e in valid_frames) if valid_frames else False
    all_saturated = all((e["saturated_16_vs_64"] or e["saturated_16_vs_64"] is None) for e in valid_frames)

    if n_graduate >= 2:
        verdict = "GRADUATE"
        verdict_reason = f"J* < 0.06 on {n_graduate}/3 frames: {[(e['frame'], e['J_star']) for e in valid_frames]}."
    elif n_promising >= 2:
        verdict = "PROMISING"
        verdict_reason = f"J* < 0.12 on {n_promising}/3 frames (but not >=2 under 0.06): {[(e['frame'], e['J_star']) for e in valid_frames]}."
    elif floor_band:
        verdict = "FLOOR"
        verdict_reason = f"J* in [0.12,0.20] for all frames, saturation across K: {[(e['frame'], e['J_star'], e['saturated_16_vs_64']) for e in valid_frames]}."
    elif n_null >= 2:
        verdict = "NULL"
        verdict_reason = f"J* > 0.20 on {n_null}/3 frames - bank underperforms Lab94 single sources: {[(e['frame'], e['J_star']) for e in valid_frames]}."
    else:
        verdict = "FLOOR"
        verdict_reason = f"No bucket matched exactly; reporting closest (FLOOR) with full per-frame data: {[(e['frame'], e['J_star']) for e in valid_frames]}."

    metrics = {
        "lab": 95, "stream": "cryptographic-substrate", "kind": "scout",
        "serves_node": "born_rule_emergence",
        "stageA": stageA, "stageB": stageB, "stageC": stageC, "stageD": stageD,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "fold_formula_modified": False,
            "born_draws_inside_world": False, "posthoc_threshold_change": False,
            "frequency_tuning_after_results": False,
        },
    }
    with open(LAB_FOLDER / "metrics_95.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
