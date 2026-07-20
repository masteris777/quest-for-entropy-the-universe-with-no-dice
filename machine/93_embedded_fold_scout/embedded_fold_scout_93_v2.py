# Lab 93 v2 - Embedded-Observer Fold Scout (convention-matched constants amendment)

import sys
import json
from pathlib import Path

import numpy as np

LAB_FOLDER = Path(__file__).parent
LAB91_DIR = Path(__file__).resolve().parents[1] / "91_two_time_conditional_born"
sys.path.insert(0, str(LAB91_DIR))
import two_time_conditional_born_91_v2 as lab91  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (v2: realized-convention theta; z0 read from Lab 92's frozen Stage A
# metrics at runtime, with hardcoded fallbacks recorded per lab-goal-v2.md)
# ---------------------------------------------------------------------------
TWO_PI = 2 * np.pi
theta = np.array([
    (TWO_PI * (np.sqrt(2) - 1)) % TWO_PI,
    (TWO_PI * (np.sqrt(2) + np.sqrt(3) - 2)) % TWO_PI,
    TWO_PI - ((TWO_PI * (np.sqrt(3) - 1)) % TWO_PI),
])
U_S = np.diag(np.exp(1j * theta))

LAB92_METRICS_PATH = Path(__file__).resolve().parents[1] / "92_classical_completion" / "metrics_92.json"
AMPS_FALLBACK = np.array([0.8305300929, 0.4981688275, 0.2489429793])
DELTA_FALLBACK = 1.5709792577

try:
    with open(LAB92_METRICS_PATH) as f:
        _metrics92 = json.load(f)
    amps = np.array(_metrics92["stageA"]["amplitudes"])
    delta = _metrics92["stageA"]["delta"]
    constants_source = "lab92_metrics"
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    amps = AMPS_FALLBACK.copy()
    delta = DELTA_FALLBACK
    constants_source = "fallback"

z0 = np.array([amps[0], amps[1], amps[2] * np.exp(1j * delta)], dtype=complex)
z0 = z0 / np.linalg.norm(z0)

OMEGA_R = 2 * np.pi * np.array([np.sqrt(5) - 2, np.sqrt(7) - 2, np.sqrt(11) - 3])
U_R = np.diag(np.exp(1j * OMEGA_R))
r0 = np.array([1.0, 1.0, 1.0], dtype=complex) / np.sqrt(3.0)
T_STEPS = 100000
T_TRANSIENT = 1000
G_GRID = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
FRAME_SEEDS = [91, 93]
TAU_MARKOV = [2, 5, 10, 50]

LAB91_D_B1 = {91: 0.4556, 93: 0.4177}  # from ../91_two_time_conditional_born/metrics_91_v2.json


def run_stage_a0():
    """New in v2: constants-consistency gate, runs before A1."""
    theta_anchor = np.array([2.6025806, 0.9190061, 1.6835744])
    theta_max_dev = float(np.max(np.abs(theta - theta_anchor)))
    relation_residual = float(abs((theta[0] - theta[1]) - theta[2]))

    amps_dev = float(np.max(np.abs(amps - AMPS_FALLBACK)))
    delta_dev = float(abs(delta - DELTA_FALLBACK))

    A0_theta = theta_max_dev < 1e-6
    A0_relation = relation_residual < 1e-9
    if constants_source == "lab92_metrics":
        A0_amps_delta = (amps_dev < 1e-9) and (delta_dev < 1e-9)
    else:
        # fallback values ARE the reference, so they trivially match; record honestly
        A0_amps_delta = True

    A0 = bool(A0_theta and A0_relation and A0_amps_delta)
    print(f"A0 theta={theta}, max_dev_vs_anchor={theta_max_dev:.3e} (<1e-6: {A0_theta})")
    print(f"A0 relation residual (theta0-theta1-theta2)={relation_residual:.3e} (<1e-9: {A0_relation})")
    print(f"A0 constants_source={constants_source}, amps={amps}, delta={delta}, "
          f"amps_dev={amps_dev:.3e}, delta_dev={delta_dev:.3e} (consistent: {A0_amps_delta})")
    print(f"A0 pass: {A0}")

    return {
        "theta": [float(t) for t in theta],
        "theta_max_dev_vs_anchor": theta_max_dev,
        "relation_residual": relation_residual,
        "constants_source": constants_source,
        "amps": [float(a) for a in amps],
        "delta": float(delta),
        "pass": A0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_M_born(Qf):
    M = np.empty((3, 3))
    US_phi = [U_S @ Qf[:, j] for j in range(3)]
    for j in range(3):
        for k in range(3):
            M[j, k] = np.abs(np.vdot(Qf[:, k], US_phi[j])) ** 2
    return M


def perp_basis(phi):
    basis = []
    for i in range(3):
        e = np.zeros(3, dtype=complex)
        e[i] = 1.0
        e_perp = e - np.vdot(phi, e) * phi
        for b in basis:
            e_perp = e_perp - np.vdot(b, e_perp) * b
        norm = np.linalg.norm(e_perp)
        if norm > 1e-8:
            basis.append(e_perp / norm)
        if len(basis) == 2:
            break
    return basis[0], basis[1]


def markov_score_fn(outcomes, F1, tau_list):
    scores = []
    for tau in tau_list:
        F_tau, _ = lab91.counted_F(outcomes, tau)
        F1_pow = np.linalg.matrix_power(F1, tau)
        scores.append(lab91.row_max_tv(F_tau, F1_pow))
    return float(max(scores))


def run_model_one(Qf, g, T=T_STEPS, track_j0_samples=False):
    """Run Model One for T steps from (z0, r0). Returns outcomes and diagnostics.
    If track_j0_samples: also collect the Stage C reconstruction-attack samples
    (restricted to steps whose current outcome is j=0, per lab-goal's stated simplification)."""
    z = z0.copy()
    r = r0.copy()
    outcomes = np.empty(T, dtype=np.int64)
    norm_z_hist = np.empty(T)
    abs_gamma_hist = np.empty(T)
    cos_g, sin_g = np.cos(g), np.sin(g)

    basis0 = perp_basis(Qf[:, 0]) if track_j0_samples else None
    history = []
    j0_X, j0_y, j0_t = [], [], []

    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j
        norm_z_hist[t] = znorm

        phi_j = Qf[:, j]
        gamma = np.vdot(phi_j, z)
        z_perp = z - gamma * phi_j
        rho = np.vdot(phi_j, r)
        r_perp = r - rho * phi_j
        abs_gamma_hist[t] = np.abs(gamma)

        if track_j0_samples and j == 0 and len(history) >= 8 and t >= T_TRANSIENT:
            c1 = np.vdot(basis0[0], z_perp)
            c2 = np.vdot(basis0[1], z_perp)
            feat = np.zeros(24)
            for i, s_hist in enumerate(history[-8:]):
                feat[i * 3 + s_hist] = 1.0
            j0_X.append(feat)
            j0_y.append([c1.real, c1.imag, c2.real, c2.imag])
            j0_t.append(t)

        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp
        r_fold = rho * phi_j - sin_g * z_perp + cos_g * r_perp

        z = U_S @ z_fold
        r = U_R @ r_fold

        if track_j0_samples:
            history.append(j)
            if len(history) > 8:
                history.pop(0)

    result = {
        "outcomes": outcomes,
        "norm_z_hist": norm_z_hist,
        "abs_gamma_hist": abs_gamma_hist,
    }
    if track_j0_samples:
        result["j0_X"] = np.array(j0_X)
        result["j0_y"] = np.array(j0_y)
        result["j0_t"] = np.array(j0_t)
    return result


def gen_calibration_stream(M_born, T=100000, seed=930):
    rng = np.random.default_rng(seed)
    j = int(rng.integers(0, 3))
    stream = np.empty(T, dtype=np.int64)
    for t in range(T):
        stream[t] = j
        j = int(rng.choice(3, p=M_born[j]))
    return stream


def stage_a3_roundtrip(Qf, g=np.pi / 4, n_steps=1000):
    z = z0.copy()
    r = r0.copy()
    z_hist = [z.copy()]
    r_hist = [r.copy()]
    j_hist = []
    cos_g, sin_g = np.cos(g), np.sin(g)

    for t in range(n_steps):
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)])
        j = int(np.argmax(overlaps))
        j_hist.append(j)
        phi_j = Qf[:, j]
        gamma = np.vdot(phi_j, z)
        z_perp = z - gamma * phi_j
        rho = np.vdot(phi_j, r)
        r_perp = r - rho * phi_j
        z_fold = gamma * phi_j + cos_g * z_perp + sin_g * r_perp
        r_fold = rho * phi_j - sin_g * z_perp + cos_g * r_perp
        z = U_S @ z_fold
        r = U_R @ r_fold
        z_hist.append(z.copy())
        r_hist.append(r.copy())

    max_err = 0.0
    US_dag = U_S.conj().T
    UR_dag = U_R.conj().T
    for t in range(n_steps):
        j = j_hist[t]
        phi_j = Qf[:, j]
        z_fold = US_dag @ z_hist[t + 1]
        r_fold = UR_dag @ r_hist[t + 1]
        gamma = np.vdot(phi_j, z_fold)
        rho = np.vdot(phi_j, r_fold)
        z_perp_f = z_fold - gamma * phi_j
        r_perp_f = r_fold - rho * phi_j
        z_perp = cos_g * z_perp_f - sin_g * r_perp_f
        r_perp = sin_g * z_perp_f + cos_g * r_perp_f
        z_rec = gamma * phi_j + z_perp
        r_rec = rho * phi_j + r_perp
        err = max(float(np.max(np.abs(z_rec - z_hist[t]))), float(np.max(np.abs(r_rec - r_hist[t]))))
        max_err = max(max_err, err)
    return max_err


def ridge_fit_r2(X_train, y_train, X_test, y_test, alpha=1.0):
    n_features = X_train.shape[1]
    reg = X_train.T @ X_train + alpha * np.eye(n_features)
    W = np.linalg.solve(reg, X_train.T @ y_train)
    y_pred = X_test @ W
    ss_res = np.sum((y_test - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)
    ss_tot_safe = np.where(ss_tot == 0, 1.0, ss_tot)
    r2_per_target = 1 - ss_res / ss_tot_safe
    return float(np.mean(r2_per_target))


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def run_stage_a():
    stageA0 = run_stage_a0()

    frames = {seed: lab91.haar_frame(seed) for seed in FRAME_SEEDS}
    M_borns = {seed: compute_M_born(frames[seed]) for seed in FRAME_SEEDS}

    # A1 - calibration
    cal_results = {}
    for seed in FRAME_SEEDS:
        cal_stream = gen_calibration_stream(M_borns[seed], T=100000, seed=930)
        F1_cal, _ = lab91.counted_F(cal_stream, 1)
        D_born_cal = lab91.row_max_tv(F1_cal, M_borns[seed])
        m_score_cal = markov_score_fn(cal_stream, F1_cal, TAU_MARKOV)
        cal_results[seed] = {"D_born_cal": D_born_cal, "markov_score_cal": m_score_cal}
        print(f"A1 calibration frame {seed}: D_born_cal={D_born_cal:.5f}, markov_score_cal={m_score_cal:.5f}")

    markov_score_cal_combined = max(cal_results[s]["markov_score_cal"] for s in FRAME_SEEDS)
    A1 = bool(all(cal_results[s]["D_born_cal"] < 0.02 for s in FRAME_SEEDS) and markov_score_cal_combined < 0.03)
    print(f"A1 pass: {A1}")

    # A2 - classical anchor (also reused as Stage B's g=0 cells)
    g0_runs = {}
    anchor_results = {}
    for seed in FRAME_SEEDS:
        Qf = frames[seed]
        run = run_model_one(Qf, g=0.0, T=T_STEPS, track_j0_samples=(seed == 91))
        g0_runs[seed] = run
        kept = run["outcomes"][T_TRANSIENT:]
        F1_g0, _ = lab91.counted_F(kept, 1)
        D_born_g0 = lab91.row_max_tv(F1_g0, M_borns[seed])
        anchor_results[seed] = {"D_born_g0": D_born_g0, "F1_g0": F1_g0}
        diff = abs(D_born_g0 - LAB91_D_B1[seed])
        print(f"A2 classical anchor frame {seed}: D_born_g0={D_born_g0:.5f}, "
              f"lab91_D_B1={LAB91_D_B1[seed]}, diff={diff:.5f}")

    A2 = bool(all(abs(anchor_results[s]["D_born_g0"] - LAB91_D_B1[s]) < 0.05 for s in FRAME_SEEDS))
    print(f"A2 pass: {A2}")

    # A3 - global round-trip (falsifier F3, substrate side), frame 91, g=pi/4
    roundtrip_max_error = stage_a3_roundtrip(frames[91], g=np.pi / 4, n_steps=1000)
    A3 = bool(roundtrip_max_error < 1e-10)
    print(f"A3 roundtrip_max_error={roundtrip_max_error:.3e}, pass={A3}")

    pass_a = bool(stageA0["pass"] and A1 and A2 and A3)

    stageA = {
        "constants_consistency": stageA0,
        "constants_source": constants_source,
        "calibration": {
            "D_born_cal_frame91": cal_results[91]["D_born_cal"],
            "D_born_cal_frame93": cal_results[93]["D_born_cal"],
            "markov_score_cal": markov_score_cal_combined,
            "pass": A1,
        },
        "classical_anchor": {
            "D_born_g0_frame91": anchor_results[91]["D_born_g0"],
            "lab91_D_B1_frame91": LAB91_D_B1[91],
            "D_born_g0_frame93": anchor_results[93]["D_born_g0"],
            "lab91_D_B1_frame93": LAB91_D_B1[93],
            "pass": A2,
        },
        "roundtrip_max_error": roundtrip_max_error,
        "pass": pass_a,
    }
    return stageA, frames, M_borns, g0_runs, anchor_results


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def run_stage_b(frames, M_borns, g0_runs, anchor_results):
    cells = []
    frame91_runs_by_g = {}

    for seed in FRAME_SEEDS:
        Qf = frames[seed]
        M_born = M_borns[seed]
        F1_of_C0 = None
        for g in G_GRID:
            if g == 0.0:
                run = g0_runs[seed]
                F1 = anchor_results[seed]["F1_g0"]
            else:
                run = run_model_one(Qf, g=g, T=T_STEPS, track_j0_samples=(seed == 91))
                kept = run["outcomes"][T_TRANSIENT:]
                F1, _ = lab91.counted_F(kept, 1)

            if seed == 91:
                frame91_runs_by_g[g] = run

            kept = run["outcomes"][T_TRANSIENT:]
            counts = np.bincount(kept, minlength=3)
            occ = counts / counts.sum()
            degenerate = bool(np.min(occ) < 0.05)

            D_born = lab91.row_max_tv(F1, M_born)
            if g == 0.0:
                F1_of_C0 = F1
                D_classical = 0.0
            else:
                D_classical = lab91.row_max_tv(F1, F1_of_C0)

            m_score = markov_score_fn(kept, F1, TAU_MARKOV)

            mean_norm_z = float(np.mean(run["norm_z_hist"][T_TRANSIENT:]))
            mean_abs_gamma = float(np.mean(run["abs_gamma_hist"][T_TRANSIENT:]))

            cell = {
                "frame": seed, "g": float(g), "degenerate": degenerate,
                "occupancies": [float(o) for o in occ],
                "D_born": float(D_born), "D_classical": float(D_classical),
                "markov_score": float(m_score),
                "mean_norm_z": mean_norm_z, "mean_abs_gamma": mean_abs_gamma,
                "F1": F1.tolist(),
            }
            cells.append(cell)
            print(f"frame={seed} g={g:.4f}: D_born={D_born:.4f}, D_classical={D_classical:.4f}, "
                  f"markov_score={m_score:.4f}, occ={occ}, mean||z||={mean_norm_z:.4f}, "
                  f"mean|gamma|={mean_abs_gamma:.4f}, degenerate={degenerate}")

    return {"cells": cells}, frame91_runs_by_g


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
def run_stage_c(frame91_runs_by_g):
    run_half_pi = frame91_runs_by_g[np.pi / 2]
    run_g0 = frame91_runs_by_g[0.0]

    r2_half_pi = ridge_fit_r2(
        run_half_pi["j0_X"][: len(run_half_pi["j0_X"]) // 2],
        run_half_pi["j0_y"][: len(run_half_pi["j0_y"]) // 2],
        run_half_pi["j0_X"][len(run_half_pi["j0_X"]) // 2:],
        run_half_pi["j0_y"][len(run_half_pi["j0_y"]) // 2:],
        alpha=1.0,
    )
    r2_g0 = ridge_fit_r2(
        run_g0["j0_X"][: len(run_g0["j0_X"]) // 2],
        run_g0["j0_y"][: len(run_g0["j0_y"]) // 2],
        run_g0["j0_X"][len(run_g0["j0_X"]) // 2:],
        run_g0["j0_y"][len(run_g0["j0_y"]) // 2:],
        alpha=1.0,
    )
    print(f"Stage C: n_samples g=pi/2: {len(run_half_pi['j0_X'])}, R2_test={r2_half_pi:.4f}")
    print(f"Stage C: n_samples g=0: {len(run_g0['j0_X'])}, R2_test={r2_g0:.4f}")

    bar_met = bool(r2_half_pi < 0.1)
    return {
        "r2_attack_g_half_pi": r2_half_pi,
        "r2_attack_g0": r2_g0,
        "n_samples_g_half_pi": int(len(run_half_pi["j0_X"])),
        "n_samples_g0": int(len(run_g0["j0_X"])),
        "bar_met": bar_met,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Instrument gates")
    print("=" * 60)
    stageA, frames, M_borns, g0_runs, anchor_results = run_stage_a()
    print(f"\nStage A: {'PASS' if stageA['pass'] else 'FAIL'}")

    if not stageA["pass"]:
        gate_names = []
        if not stageA["constants_consistency"]["pass"]:
            gate_names.append("A0 (constants consistency)")
        if not stageA["calibration"]["pass"]:
            gate_names.append("A1 (calibration)")
        if not stageA["classical_anchor"]["pass"]:
            gate_names.append("A2 (classical anchor)")
        if stageA["roundtrip_max_error"] >= 1e-10:
            gate_names.append("A3 (round-trip)")
        metrics = {
            "lab": 93, "stream": "cryptographic-substrate", "kind": "scout",
            "version": "v2-convention-matched",
            "serves_node": "born_rule_emergence",
            "stageA": stageA,
            "stageB": {"cells": []},
            "stageC": {"r2_attack_g_half_pi": None, "r2_attack_g0": None, "bar_met": None},
            "verdict": "VOID",
            "verdict_reason": f"Stage A gate(s) failed: {', '.join(gate_names)}. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "projection_coupling_used": False,
                "born_draws_inside_world": False, "posthoc_threshold_change": False,
                "reimplemented_lab91_functions": False,
            },
        }
        with open(LAB_FOLDER / "metrics_93_v2.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - The g-sweep")
    print("=" * 60)
    stageB, frame91_runs_by_g = run_stage_b(frames, M_borns, g0_runs, anchor_results)

    print("\n" + "=" * 60)
    print("STAGE C - Observer-side arrow (reconstruction attack)")
    print("=" * 60)
    stageC = run_stage_c(frame91_runs_by_g)

    # Verdict
    def get_cell(frame, g):
        for c in stageB["cells"]:
            if c["frame"] == frame and abs(c["g"] - g) < 1e-9:
                return c
        return None

    markov_ok = True
    born_ok = True
    any_valid = False
    per_frame_ratios = {}
    for seed in FRAME_SEEDS:
        c0 = get_cell(seed, 0.0)
        chalf = get_cell(seed, np.pi / 2)
        if chalf["degenerate"] or c0["degenerate"]:
            per_frame_ratios[seed] = {"skipped_degenerate": True}
            continue
        any_valid = True
        markov_ratio = chalf["markov_score"] / c0["markov_score"] if c0["markov_score"] > 0 else float("inf")
        born_ratio = chalf["D_born"] / c0["D_born"] if c0["D_born"] > 0 else float("inf")
        per_frame_ratios[seed] = {"markov_ratio": markov_ratio, "born_ratio": born_ratio}
        if not (markov_ratio < 0.3):
            markov_ok = False
        if not (born_ratio < 0.6):
            born_ok = False
        print(f"frame {seed}: markov_ratio(g=pi/2 vs g=0)={markov_ratio:.4f} (<0.3: {markov_ratio<0.3}), "
              f"born_ratio={born_ratio:.4f} (<0.6: {born_ratio<0.6})")

    if not any_valid:
        verdict = "NULL"
        verdict_reason = "All frames degenerate at g=0 or g=pi/2 - cannot evaluate verdict ratios; treated as boundary/NULL."
    elif markov_ok and born_ok:
        verdict = "PROMISING"
        verdict_reason = f"Both axes moved at g=pi/2 for all non-degenerate frames: {per_frame_ratios}"
    elif markov_ok and not born_ok:
        verdict = "MARKOV-ONLY"
        verdict_reason = f"Markov criterion held but Born-kernel criterion did not: {per_frame_ratios}"
    else:
        verdict = "NULL"
        verdict_reason = f"Neither criterion held at g=pi/2 for all non-degenerate frames: {per_frame_ratios}"

    metrics = {
        "lab": 93, "stream": "cryptographic-substrate", "kind": "scout",
        "version": "v2-convention-matched",
        "serves_node": "born_rule_emergence",
        "stageA": stageA, "stageB": stageB, "stageC": stageC,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "projection_coupling_used": False,
            "born_draws_inside_world": False, "posthoc_threshold_change": False,
            "reimplemented_lab91_functions": False,
        },
    }
    with open(LAB_FOLDER / "metrics_93_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
