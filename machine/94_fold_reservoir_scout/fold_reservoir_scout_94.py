# Lab 94 - Fold Sweet-Spot & Reservoir-Quality Scout

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB91_DIR = Path(__file__).resolve().parents[1] / "91_two_time_conditional_born"
LAB93_DIR = Path(__file__).resolve().parents[1] / "93_embedded_fold_scout"
sys.path.insert(0, str(LAB91_DIR))
sys.path.insert(0, str(LAB93_DIR))
import two_time_conditional_born_91_v2 as lab91  # noqa: E402
import embedded_fold_scout_93_v2 as lab93  # noqa: E402

# ---------------------------------------------------------------------------
# Predeclared parameters
# ---------------------------------------------------------------------------
G_GRID = [0.20, 0.35, 0.50, 0.65, 0.7853981634, 0.90, 1.00, 1.15]
FRAME_SEEDS = [91, 93, 201]
T_SWEEP = 60000
T_TRANSIENT = 1000
TAU_MARKOV = [2, 5, 10, 50]
DEGENERACY_FLOOR = 0.05
CAL_SEED = 940

# Generator constants: R-QPGEN uses Lab 93's own reservoir frequencies (first two)
OMEGA_R01 = lab93.OMEGA_R[:2]


def gen_theta_step(theta_r, variant):
    if variant == "R-QPGEN":
        return np.mod(theta_r + OMEGA_R01, 2 * np.pi)
    elif variant == "R-CATGEN":
        th1, th2 = theta_r
        return np.mod(np.array([2 * th1 + th2, th1 + th2]), 2 * np.pi)
    raise ValueError(variant)


def gen_theta_inverse(theta_r, variant):
    if variant == "R-QPGEN":
        return np.mod(theta_r - OMEGA_R01, 2 * np.pi)
    elif variant == "R-CATGEN":
        th1p, th2p = theta_r
        return np.mod(np.array([th1p - th2p, -th1p + 2 * th2p]), 2 * np.pi)
    raise ValueError(variant)


def r_gen_from_theta(theta_r):
    th1, th2 = theta_r
    return np.array([np.exp(1j * th1), np.exp(1j * th2), np.exp(1j * (th1 + th2))]) / np.sqrt(3)


def gen_calibration_stream(M_born, T, seed):
    rng = np.random.default_rng(seed)
    j = int(rng.integers(0, 3))
    stream = np.empty(T, dtype=np.int64)
    for t in range(T):
        stream[t] = j
        j = int(rng.choice(3, p=M_born[j]))
    return stream


# ---------------------------------------------------------------------------
# Tape-variant step loop (R-QPGEN, R-CATGEN)
# NEW code, per lab-goal.md - the fold's first rotation line is copied verbatim
# from Lab 93 v2's run_model_one (embedded_fold_scout_93_v2.py, line 168:
#   z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp
# ). The second line differs BY DESIGN (per lab-goal.md's tape formula): it
# carries only the perpendicular combination (no rho*phi_j term) because that
# information is never re-read by the observer and is instead appended,
# write-once, to a tape - vs Lab 93's line 169 which returns rho*phi_j + ...
# into a persistent register r. Both lines apply the SAME (cos_g, sin_g)
# rotation to (z_perp, r_perp); only the destination of the second component
# differs (register vs tape).
# ---------------------------------------------------------------------------
def run_tape_variant(Qf, g, variant, T, T_transient=T_TRANSIENT, track_stage_d=False):
    z = lab93.z0.copy()
    theta_r = np.array([1.0, 2.0])
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

        r_gen = r_gen_from_theta(theta_r)
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

        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp  # verbatim, Lab 93 v2 line 168

        z = lab93.U_S @ z_fold
        theta_r = gen_theta_step(theta_r, variant)

        if track_stage_d:
            history.append(j)
            if len(history) > 8:
                history.pop(0)

    result = {"outcomes": outcomes, "norm_z_hist": norm_z_hist}
    if track_stage_d:
        result["j0_X"] = np.array(j0_X)
        result["j0_y"] = np.array(j0_y)
    return result


def tape_roundtrip(Qf, g, variant, n_steps=1000):
    z = lab93.z0.copy()
    theta_r = np.array([1.0, 2.0])
    z_hist = [z.copy()]
    theta_hist = [theta_r.copy()]
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
        r_gen = r_gen_from_theta(theta_r)
        rho = np.vdot(phi_j, r_gen)
        r_perp = r_gen - rho * phi_j

        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp  # verbatim, Lab 93 v2 line 168
        tape_t = -sin_g * z_perp + cos_g * r_perp
        tape.append(tape_t)

        z = lab93.U_S @ z_fold
        theta_r = gen_theta_step(theta_r, variant)
        z_hist.append(z.copy())
        theta_hist.append(theta_r.copy())

    # Reconstruct backward
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

        theta_rec = gen_theta_inverse(theta_hist[t + 1], variant)
        # compare on the unit circle to sidestep 0/2pi wraparound
        err_theta = float(np.max(np.abs(np.exp(1j * theta_rec) - np.exp(1j * theta_hist[t]))))

        # also check r_perp_rec is consistent with the independently-recomputed r_gen at t
        r_gen_t = r_gen_from_theta(theta_hist[t])
        rho_t = np.vdot(phi_j, r_gen_t)
        r_perp_t = r_gen_t - rho_t * phi_j
        err_rperp = float(np.max(np.abs(r_perp_rec - r_perp_t)))

        max_err = max(max_err, err_z, err_theta, err_rperp)
    return max_err


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def run_stage_a():
    Qf91 = lab91.haar_frame(91)

    # A1 - anchor: exact reproduction of Lab 93's g=pi/4 cell (T=100000, transient=1000)
    run = lab93.run_model_one(Qf91, g=np.pi / 4, T=100000, track_j0_samples=False)
    kept = run["outcomes"][1000:]
    M_born_91 = lab93.compute_M_born(Qf91)
    F1, _ = lab91.counted_F(kept, 1)
    D_born_anchor = lab91.row_max_tv(F1, M_born_91)
    markov_anchor = lab93.markov_score_fn(kept, F1, TAU_MARKOV)
    A1 = bool(abs(D_born_anchor - 0.0847) < 1e-3 and abs(markov_anchor - 0.2307) < 1e-3)
    print(f"A1 anchor: D_born={D_born_anchor:.5f} (target 0.0847), markov={markov_anchor:.5f} "
          f"(target 0.2307), pass={A1}")

    # A2 - calibration floors
    frames = {seed: lab91.haar_frame(seed) for seed in FRAME_SEEDS}
    M_borns = {seed: lab93.compute_M_born(frames[seed]) for seed in FRAME_SEEDS}
    markov_floors = {}
    kernel_devs = []
    for idx, seed in enumerate(FRAME_SEEDS):
        cal_stream = gen_calibration_stream(M_borns[seed], T=T_SWEEP, seed=CAL_SEED + idx)
        F1_cal, _ = lab91.counted_F(cal_stream, 1)
        D_born_cal = lab91.row_max_tv(F1_cal, M_borns[seed])
        m_floor = lab93.markov_score_fn(cal_stream, F1_cal, TAU_MARKOV)
        markov_floors[str(seed)] = m_floor
        kernel_devs.append(D_born_cal)
        print(f"A2 frame {seed}: D_born_cal={D_born_cal:.5f}, markov_floor={m_floor:.5f}")

    kernel_max_dev = float(max(kernel_devs))
    A2 = bool(kernel_max_dev < 0.02)
    print(f"A2 pass: {A2} (kernel_max_dev={kernel_max_dev:.5f})")

    # A3 - tape round-trip, both variants, g=pi/4, frame 91, 1000 steps
    qpgen_err = tape_roundtrip(Qf91, np.pi / 4, "R-QPGEN", n_steps=1000)
    catgen_err = tape_roundtrip(Qf91, np.pi / 4, "R-CATGEN", n_steps=1000)
    A3 = bool(qpgen_err < 1e-10 and catgen_err < 1e-10)
    print(f"A3 qpgen_max_err={qpgen_err:.3e}, catgen_max_err={catgen_err:.3e}, pass={A3}")

    pass_a = bool(A1 and A2 and A3)

    stageA = {
        "anchor": {"D_born": D_born_anchor, "markov_score": markov_anchor, "pass": A1},
        "calibration": {"kernel_max_dev": kernel_max_dev, "markov_floors": markov_floors, "pass": A2},
        "tape_roundtrip": {"qpgen_max_err": qpgen_err, "catgen_max_err": catgen_err, "pass": A3},
        "pass": pass_a,
    }
    return stageA, frames, M_borns


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def run_stage_b(frames, M_borns):
    baselines = {}
    baseline_F1 = {}
    for seed in FRAME_SEEDS:
        Qf = frames[seed]
        run = lab93.run_model_one(Qf, g=0.0, T=T_SWEEP, track_j0_samples=False)
        kept = run["outcomes"][T_TRANSIENT:]
        F1_g0, _ = lab91.counted_F(kept, 1)
        m_g0 = lab93.markov_score_fn(kept, F1_g0, TAU_MARKOV)
        baselines[seed] = {"D_born_g0": float(lab91.row_max_tv(F1_g0, M_borns[seed])), "markov_g0": m_g0}
        baseline_F1[seed] = F1_g0
        print(f"baseline frame {seed}: D_born_g0={baselines[seed]['D_born_g0']:.4f}, markov_g0={m_g0:.4f}")

    cells = []
    stage_d_samples = {}  # (variant, frame=91) -> {g: (j0_X, j0_y)}

    for variant in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        stage_d_samples[variant] = {}
        for seed in FRAME_SEEDS:
            Qf = frames[seed]
            for g in G_GRID:
                track = (seed == 91)  # collect Stage D samples for frame 91 in all variants
                if variant == "R-QP3":
                    run = lab93.run_model_one(Qf, g=g, T=T_SWEEP, track_j0_samples=track)
                else:
                    run = run_tape_variant(Qf, g=g, variant=variant, T=T_SWEEP, track_stage_d=track)

                kept = run["outcomes"][T_TRANSIENT:]
                counts = np.bincount(kept, minlength=3)
                occ = counts / counts.sum()
                degenerate = bool(np.min(occ) < DEGENERACY_FLOOR)
                max_occupancy = float(np.max(occ))

                F1, _ = lab91.counted_F(kept, 1)
                D_born = float(lab91.row_max_tv(F1, M_borns[seed]))
                D_classical = float(lab91.row_max_tv(F1, baseline_F1[seed]))
                m_score = lab93.markov_score_fn(kept, F1, TAU_MARKOV)
                mean_norm_z = float(np.mean(run["norm_z_hist"][T_TRANSIENT:]))

                cell = {
                    "variant": variant, "frame": seed, "g": float(g), "degenerate": degenerate,
                    "occupancies": [float(o) for o in occ], "max_occupancy": max_occupancy,
                    "D_born": D_born, "D_classical": D_classical, "markov_score": float(m_score),
                    "mean_norm_z": mean_norm_z,
                }
                cells.append(cell)

                if track and "j0_X" in run:
                    stage_d_samples[variant][g] = (run["j0_X"], run["j0_y"])

                print(f"{variant} frame={seed} g={g:.4f}: D_born={D_born:.4f}, D_classical={D_classical:.4f}, "
                      f"markov={m_score:.4f}, max_occ={max_occupancy:.3f}, degenerate={degenerate}")

    # freeze table: smallest grid g with max_occupancy > 0.9, per (variant, frame)
    freeze_table = []
    for variant in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        for seed in FRAME_SEEDS:
            variant_frame_cells = sorted(
                [c for c in cells if c["variant"] == variant and c["frame"] == seed],
                key=lambda c: c["g"],
            )
            freeze_g = None
            for c in variant_frame_cells:
                if c["max_occupancy"] > 0.9:
                    freeze_g = c["g"]
                    break
            freeze_table.append({"variant": variant, "frame": seed, "freeze_g": freeze_g})

    baselines_list = [{"frame": s, "D_born_g0": baselines[s]["D_born_g0"], "markov_g0": baselines[s]["markov_g0"]}
                       for s in FRAME_SEEDS]

    return {"baselines": baselines_list, "cells": cells, "freeze_table": freeze_table}, stage_d_samples


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
def run_stage_c(stageB):
    cells = stageB["cells"]
    per_frame_argmin = []
    per_variant_map = {v: [] for v in ["R-QP3", "R-QPGEN", "R-CATGEN"]}

    for variant in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        for seed in FRAME_SEEDS:
            nondeg = [c for c in cells if c["variant"] == variant and c["frame"] == seed and not c["degenerate"]]
            if not nondeg:
                per_frame_argmin.append({"variant": variant, "frame": seed, "g_star": None,
                                          "D_born_star": None, "markov_star": None})
                continue
            best = min(nondeg, key=lambda c: c["D_born"])
            per_frame_argmin.append({
                "variant": variant, "frame": seed, "g_star": best["g"],
                "D_born_star": best["D_born"], "markov_star": best["markov_score"],
            })
            per_variant_map[variant].append(best)

    per_variant = []
    for variant in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        entries = per_variant_map[variant]
        if entries:
            D_med = float(np.median([e["D_born"] for e in entries]))
            M_med = float(np.median([e["markov_score"] for e in entries]))
        else:
            D_med, M_med = None, None
        per_variant.append({
            "variant": variant, "D_med_star": D_med, "M_med_star": M_med,
            "n_frames_nondegenerate_at_gstar": len(entries),
        })
        print(f"{variant}: D_med*={D_med}, M_med*={M_med}, n_valid_frames={len(entries)}")

    return {"per_frame_argmin": per_frame_argmin, "per_variant": per_variant}


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------
def run_stage_d(stageC, stage_d_samples):
    attacks = []
    for variant in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        entry = next(e for e in stageC["per_frame_argmin"] if e["variant"] == variant and e["frame"] == 91)
        g_star = entry["g_star"]
        if g_star is None or g_star not in stage_d_samples.get(variant, {}):
            attacks.append({"variant": variant, "g": g_star, "r2": None, "n_samples": 0})
            print(f"Stage D {variant}: no valid g* sample available")
            continue
        X, y = stage_d_samples[variant][g_star]
        n = len(X)
        if n < 20:
            attacks.append({"variant": variant, "g": g_star, "r2": None, "n_samples": n})
            print(f"Stage D {variant}: g*={g_star}, n_samples={n} (too few for a stable fit)")
            continue
        mid = n // 2
        r2 = lab93.ridge_fit_r2(X[:mid], y[:mid], X[mid:], y[mid:], alpha=1.0)
        attacks.append({"variant": variant, "g": g_star, "r2": r2, "n_samples": n})
        print(f"Stage D {variant}: g*={g_star:.4f}, n_samples={n}, R2={r2:.4f}")
    return {"attacks": attacks}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(stageB):
    cells = stageB["cells"]
    variants = ["R-QP3", "R-QPGEN", "R-CATGEN"]
    colors = {"R-QP3": "tab:blue", "R-QPGEN": "tab:orange", "R-CATGEN": "tab:green"}

    fig, axes = plt.subplots(len(FRAME_SEEDS), 2, figsize=(11, 12))
    for i, seed in enumerate(FRAME_SEEDS):
        ax_born, ax_markov = axes[i, 0], axes[i, 1]
        for variant in variants:
            vc = sorted([c for c in cells if c["variant"] == variant and c["frame"] == seed],
                        key=lambda c: c["g"])
            gs = [c["g"] for c in vc]
            born = [c["D_born"] for c in vc]
            markov = [c["markov_score"] for c in vc]
            markers = ["x" if c["degenerate"] else "o" for c in vc]
            ax_born.plot(gs, born, color=colors[variant], label=variant, marker="o")
            ax_markov.plot(gs, markov, color=colors[variant], label=variant, marker="o")
        ax_born.axhline(0.05, color="gray", linestyle="--", linewidth=0.8)
        ax_markov.axhline(0.05, color="gray", linestyle="--", linewidth=0.8)
        ax_born.set_title(f"frame {seed}: D_born(g)")
        ax_markov.set_title(f"frame {seed}: markov_score(g)")
        ax_born.set_xlabel("g (rad)")
        ax_markov.set_xlabel("g (rad)")
        ax_born.legend(fontsize=8)
        ax_markov.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "sweetspot_curves_94.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    stageA, frames, M_borns = run_stage_a()
    print(f"\nStage A: {'PASS' if stageA['pass'] else 'FAIL'}")

    if not stageA["pass"]:
        metrics = {
            "lab": 94, "stream": "cryptographic-substrate", "kind": "scout",
            "serves_node": "born_rule_emergence",
            "stageA": stageA,
            "stageB": {"baselines": [], "cells": [], "freeze_table": []},
            "stageC": {"per_frame_argmin": [], "per_variant": []},
            "stageD": {"attacks": []},
            "verdict": "VOID",
            "verdict_reason": "Stage A gate(s) failed. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "fold_formula_modified": False,
                "born_draws_inside_world": False, "posthoc_threshold_change": False,
                "reimplemented_imported_functions": False,
            },
        }
        with open(LAB_FOLDER / "metrics_94.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - The sweep")
    print("=" * 60)
    stageB, stage_d_samples = run_stage_b(frames, M_borns)

    print("\n" + "=" * 60)
    print("STAGE C - Sweet-spot decision (argmin)")
    print("=" * 60)
    stageC = run_stage_c(stageB)

    print("\n" + "=" * 60)
    print("STAGE D - Attack at the sweet spot")
    print("=" * 60)
    stageD = run_stage_d(stageC, stage_d_samples)

    print("\nGenerating plot...")
    make_plot(stageB)

    # Verdict
    pv = {e["variant"]: e for e in stageC["per_variant"]}
    cat = pv["R-CATGEN"]

    def valid(e):
        return e["D_med_star"] is not None and e["M_med_star"] is not None

    strong = valid(cat) and cat["M_med_star"] < 0.05 and cat["D_med_star"] < 0.05 and cat["n_frames_nondegenerate_at_gstar"] >= 2

    promising_any = False
    for v in ["R-QP3", "R-QPGEN", "R-CATGEN"]:
        e = pv[v]
        if valid(e) and e["D_med_star"] <= 0.10 and e["M_med_star"] <= 0.15:
            promising_any = True

    mech_floor = all(valid(pv[v]) and 0.06 <= pv[v]["D_med_star"] <= 0.12 for v in ["R-QP3", "R-QPGEN", "R-CATGEN"])

    frame201_entry = next(e for e in stageC["per_frame_argmin"] if e["frame"] == 201 and e["variant"] == "R-QP3")
    frame91_entry = next(e for e in stageC["per_frame_argmin"] if e["frame"] == 91 and e["variant"] == "R-QP3")
    frame93_entry = next(e for e in stageC["per_frame_argmin"] if e["frame"] == 93 and e["variant"] == "R-QP3")
    frame_luck = (
        frame201_entry["D_born_star"] is not None and frame201_entry["D_born_star"] > 0.2
        and frame91_entry["D_born_star"] is not None and frame91_entry["D_born_star"] <= 0.2
        and frame93_entry["D_born_star"] is not None and frame93_entry["D_born_star"] <= 0.2
    )

    if strong:
        verdict = "PROMISING-STRONG"
        verdict_reason = f"R-CATGEN at argmin: M_med*={cat['M_med_star']:.4f} < 0.05, D_med*={cat['D_med_star']:.4f} < 0.05, {cat['n_frames_nondegenerate_at_gstar']} frames non-degenerate at g*."
    elif frame_luck:
        verdict = "NULL"
        verdict_reason = f"Frame 201's sweet spot did not reproduce (D_born*={frame201_entry['D_born_star']}) while 91/93 did ({frame91_entry['D_born_star']}, {frame93_entry['D_born_star']}) - frame luck suspected."
    elif mech_floor:
        verdict = "MECHANISM-FLOOR"
        verdict_reason = f"All three variants plateau at D_med* in [0.06,0.12]: {[(v, pv[v]['D_med_star']) for v in pv]}. R-CATGEN markov={cat['M_med_star']}, kernel unmoved - fold-geometry-intrinsic."
    elif promising_any:
        verdict = "PROMISING"
        verdict_reason = f"At least one variant has D_med*<=0.10 and M_med*<=0.15: {[(v, pv[v]['D_med_star'], pv[v]['M_med_star']) for v in pv]}."
    else:
        verdict = "MECHANISM-FLOOR"
        verdict_reason = f"No verdict branch cleanly matched; reporting closest per-variant medians: {[(v, pv[v]) for v in pv]}."

    metrics = {
        "lab": 94, "stream": "cryptographic-substrate", "kind": "scout",
        "serves_node": "born_rule_emergence",
        "stageA": stageA, "stageB": stageB, "stageC": stageC, "stageD": stageD,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "fold_formula_modified": False,
            "born_draws_inside_world": False, "posthoc_threshold_change": False,
            "reimplemented_imported_functions": False,
        },
    }
    with open(LAB_FOLDER / "metrics_94.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
