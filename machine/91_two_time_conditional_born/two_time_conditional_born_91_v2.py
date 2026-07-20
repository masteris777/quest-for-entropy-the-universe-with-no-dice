# Lab 91 v2 (hybrid-estimator amendment) - Two-Time Conditional Born: Counted Frequencies vs the Collapse Formula

import sys
import json
from pathlib import Path

import numpy as np

LAB_FOLDER = Path(__file__).parent
LAB88_DIR = Path(__file__).resolve().parents[1] / "88_forced_complex_structure"
sys.path.insert(0, str(LAB88_DIR))
from forced_complex_structure_88 import build_stream, j_tests  # noqa: E402
# dashboard_pipeline is not imported wholesale; the lstsq recipe is reproduced inline
# per lab-goal-v2.md Amendment 1 (phases only - geometry/Q/Z_C stay Procrustes).

T = 200000
L = 40
R = 6
TAU_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
CESARO_TAUS = [100, 200, 500, 1000]
FRAME_SEEDS = [91, 92, 93]
UNROUNDED_ANCHOR = np.array([0.9190061, 1.6835744, 2.6025806])
BORN_MARGIN = 0.02
DEGENERACY_FLOOR = 0.05
PHASE_MATCH_TOL = 2e-4


def build_Z(y, L_=L, r_=R):
    N = len(y) - L_
    X = np.empty((N, L_))
    for t in range(N):
        X[t, :] = y[t:t + L_]
    U_, S_, Vt_ = np.linalg.svd(X, full_matrices=False)
    Z = U_[:, :r_] * S_[:r_]
    return Z, X


def lstsq_phases(Z):
    """Amendment 1: lstsq operator and its sorted positive-imaginary-part phases."""
    A_lst, *_ = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)
    A_lst = A_lst.T
    lam_lst = np.linalg.eig(A_lst)[0]
    pos = lam_lst[lam_lst.imag > 0]
    if len(pos) != 3:
        return None
    return np.sort(np.abs(np.angle(pos)))


def hybrid_A_C(A_C, ph_lst):
    """Amendment 1: install lstsq phases into the Procrustes-diagonal A_C, preserving
    each diagonal entry's sign of rotation and column position."""
    theta_proc = np.angle(np.diag(A_C))
    order = np.argsort(np.abs(theta_proc))
    theta_used = theta_proc.copy()
    theta_used[order] = np.sign(theta_proc[order]) * ph_lst
    A_C_used = np.diag(np.exp(1j * theta_used))
    phase_match_max_delta = float(np.max(np.abs(np.abs(theta_proc[order]) - ph_lst)))
    return A_C_used, phase_match_max_delta


def build_pipeline(y):
    Z, X = build_Z(y)
    M_cross = Z[1:].T @ Z[:-1]
    U_p, _, Vt_p = np.linalg.svd(M_cross)
    A = U_p @ Vt_p

    lam, P = np.linalg.eig(A)
    cols = np.where(lam.imag > 0)[0]
    Q, _ = np.linalg.qr(P[:, cols])
    A_C = Q.conj().T @ A @ Q  # Procrustes-diagonal, exactly unitary
    Z_hat = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    Z_C = Z_hat @ (np.sqrt(2) * Q).conj()  # geometry stays Procrustes, unchanged

    ph_lst = lstsq_phases(Z)
    if ph_lst is not None:
        A_C_used, phase_match_max_delta = hybrid_A_C(A_C, ph_lst)
        phase_source = "lstsq"
    else:
        A_C_used = A_C.copy()
        phase_match_max_delta = None
        phase_source = "procrustes"

    return {
        "A": A, "Q": Q, "A_C": A_C, "A_C_used": A_C_used,
        "ph_lst": ph_lst, "phase_match_max_delta": phase_match_max_delta, "phase_source": phase_source,
        "Z": Z, "X": X, "Z_hat": Z_hat, "Z_C": Z_C, "lam": lam,
    }


def haar_frame(seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Qf, _ = np.linalg.qr(G)
    return Qf


def symbol_stream(Z_C, Qf):
    overlaps = np.abs(Z_C @ Qf.conj()) ** 2  # (N,3)
    return np.argmax(overlaps, axis=1)


def occupancies(s):
    counts = np.bincount(s, minlength=3)
    return counts / counts.sum()


def counted_F(s, tau):
    idx = s[:-tau] * 3 + s[tau:]
    C = np.bincount(idx, minlength=9).reshape(3, 3).astype(float)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    F = C / row_sums_safe
    return F, C


def born_B(A_C_used, Qf, tau):
    w, V = np.linalg.eig(A_C_used)
    A_tau = V @ np.diag(w ** tau) @ V.conj().T
    B = np.empty((3, 3))
    for j in range(3):
        for k in range(3):
            B[j, k] = np.abs(Qf[:, k].conj() @ (A_tau @ Qf[:, j])) ** 2
    return B


def decohered_Bbar(Qf):
    # e_m = standard basis of C^3 (the A_C-diagonal / J-eigenbasis in Z_C coordinates) - unchanged by the hybrid swap
    Bbar = np.zeros((3, 3))
    for j in range(3):
        for k in range(3):
            s = 0.0
            for m in range(3):
                e_m = np.eye(3)[m]
                s += np.abs(Qf[:, k].conj() @ e_m) ** 2 * np.abs(e_m.conj() @ Qf[:, j]) ** 2
            Bbar[j, k] = s
    return Bbar


def row_max_tv(F, X):
    return float(np.max(0.5 * np.sum(np.abs(F - X), axis=1)))


def stage_a(pipeline):
    A = pipeline["A"]
    A_C = pipeline["A_C"]
    Z_C = pipeline["Z_C"]
    ph_lst = pipeline["ph_lst"]
    phase_match_max_delta = pipeline["phase_match_max_delta"]

    procrustes_orthogonality_residual = float(np.max(np.abs(A @ A.T - np.eye(A.shape[0]))))
    jt = j_tests(A)
    j_tests_passes_all = bool(jt["passes_all"])

    A_C_unitarity_residual = float(np.max(np.abs(A_C.conj().T @ A_C - np.eye(3))))
    offdiag_mask = ~np.eye(3, dtype=bool)
    A_C_max_offdiag = float(np.max(np.abs(A_C[offdiag_mask])))

    if ph_lst is not None:
        lstsq_eigenphase_max_dev = float(np.max(np.abs(ph_lst - np.sort(UNROUNDED_ANCHOR))))
    else:
        lstsq_eigenphase_max_dev = None

    Z_C_max_rownorm_dev = float(np.max(np.abs(np.linalg.norm(Z_C, axis=1) - 1.0)))

    X = pipeline["X"]
    row_norms = np.linalg.norm(X, axis=1)
    hankel_rownorm_cv = float(np.std(row_norms) / np.mean(row_norms))

    rho = (Z_C.conj().T @ Z_C) / Z_C.shape[0]
    rho_offdiag_mask = ~np.eye(3, dtype=bool)
    max_offdiag_abs_rho = float(np.max(np.abs(rho[rho_offdiag_mask])))
    max_offdiag_abs_imag_rho = float(np.max(np.abs(rho[rho_offdiag_mask].imag)))

    A1 = procrustes_orthogonality_residual < 1e-12
    A2 = j_tests_passes_all
    A3 = (A_C_unitarity_residual < 1e-10) and (A_C_max_offdiag < 1e-8)
    A4a = (lstsq_eigenphase_max_dev is not None) and (lstsq_eigenphase_max_dev < 1e-5)
    A4b = (phase_match_max_delta is not None) and (phase_match_max_delta < PHASE_MATCH_TOL)
    A5 = Z_C_max_rownorm_dev < 1e-10

    pass_a = bool(A1 and A2 and A3 and A4a and A4b and A5)

    print(f"A1 procrustes_orthogonality_residual={procrustes_orthogonality_residual:.3e} (<1e-12: {A1})")
    print(f"A2 j_tests_passes_all={j_tests_passes_all}")
    print(f"A3 A_C_unitarity_residual={A_C_unitarity_residual:.3e}, A_C_max_offdiag={A_C_max_offdiag:.3e} ({A3})")
    print(f"A4a lstsq_eigenphase_max_dev={lstsq_eigenphase_max_dev} vs unrounded anchor (<1e-5: {A4a})")
    print(f"A4b phase_match_max_delta={phase_match_max_delta} (<2e-4: {A4b})")
    print(f"A5 Z_C_max_rownorm_dev={Z_C_max_rownorm_dev:.3e} (<1e-10: {A5})")
    print(f"diag: hankel_rownorm_cv={hankel_rownorm_cv:.4f}, max_offdiag_abs_rho={max_offdiag_abs_rho:.3e}, "
          f"max_offdiag_abs_imag_rho={max_offdiag_abs_imag_rho:.3e}")

    return {
        "procrustes_orthogonality_residual": procrustes_orthogonality_residual,
        "j_tests_passes_all": j_tests_passes_all,
        "A_C_unitarity_residual": A_C_unitarity_residual,
        "A_C_max_offdiag": A_C_max_offdiag,
        "lstsq_eigenphase_max_dev_vs_unrounded_anchor": lstsq_eigenphase_max_dev,
        "phase_match_max_delta": phase_match_max_delta,
        "Z_C_max_rownorm_dev": Z_C_max_rownorm_dev,
        "hankel_rownorm_cv": hankel_rownorm_cv,
        "max_offdiag_abs_rho": max_offdiag_abs_rho,
        "max_offdiag_abs_imag_rho": max_offdiag_abs_imag_rho,
        "pass": pass_a,
    }


def run_frame_measurement(pipeline, frame_seed):
    Qf = haar_frame(frame_seed)
    Z_C = pipeline["Z_C"]
    A_C_used = pipeline["A_C_used"]

    s = symbol_stream(Z_C, Qf)
    occ = occupancies(s)
    degenerate = bool(np.min(occ) < DEGENERACY_FLOOR)

    print(f"  frame seed={frame_seed}: occupancies={occ}, degenerate={degenerate}")

    frame_result = {
        "seed": frame_seed,
        "degenerate": degenerate,
        "occupancies": [float(o) for o in occ],
        "per_tau": [],
        "cesaro": {"D_Bbar": None, "D_S_bar": None, "row_spread_Fbar": None},
    }

    if degenerate:
        return frame_result, None

    S = np.tile(occ, (3, 1))  # stationarity: rows all equal to occupancy vector

    F_by_tau = {}
    for tau in TAU_GRID:
        F, C = counted_F(s, tau)
        B_tau = born_B(A_C_used, Qf, tau)
        D_B = row_max_tv(F, B_tau)
        D_I = row_max_tv(F, np.eye(3))
        D_S = row_max_tv(F, S)
        born_preferred = bool(D_B <= min(D_I, D_S) - BORN_MARGIN)
        F_by_tau[tau] = F
        frame_result["per_tau"].append({
            "tau": tau, "D_B": float(D_B), "D_I": float(D_I), "D_S": float(D_S),
            "born_preferred": born_preferred,
        })

    # Cesaro block
    Fbar = np.mean([F_by_tau[t] for t in CESARO_TAUS], axis=0)
    Bbar = decohered_Bbar(Qf)
    D_Bbar = row_max_tv(Fbar, Bbar)
    D_S_bar = row_max_tv(Fbar, S)
    row_spread_Fbar = float(np.max([
        0.5 * np.sum(np.abs(Fbar[j] - Fbar[jp]))
        for j in range(3) for jp in range(3)
    ]))
    frame_result["cesaro"] = {
        "D_Bbar": float(D_Bbar), "D_S_bar": float(D_S_bar), "row_spread_Fbar": row_spread_Fbar,
    }

    return frame_result, F_by_tau


def run_stage_b_or_c(pipeline, label):
    frames = []
    n_preferred = 0
    n_total = 0
    all_F_by_tau = {}
    for seed in FRAME_SEEDS:
        fr, F_by_tau = run_frame_measurement(pipeline, seed)
        frames.append(fr)
        if not fr["degenerate"]:
            all_F_by_tau[seed] = F_by_tau
            for cell in fr["per_tau"]:
                n_total += 1
                if cell["born_preferred"]:
                    n_preferred += 1
    born_fraction = (n_preferred / n_total) if n_total > 0 else None
    print(f"{label}: born_fraction = {born_fraction} ({n_preferred}/{n_total})")
    return frames, born_fraction, all_F_by_tau


def run_stage_d(torus_frames):
    row_spreads = [fr["cesaro"]["row_spread_Fbar"] for fr in torus_frames if not fr["degenerate"]]
    d_s_bars = [fr["cesaro"]["D_S_bar"] for fr in torus_frames if not fr["degenerate"]]
    d_bbars = [fr["cesaro"]["D_Bbar"] for fr in torus_frames if not fr["degenerate"]]

    p2_row_spread_max = float(max(row_spreads)) if row_spreads else None
    p2_D_S_bar_max = float(max(d_s_bars)) if d_s_bars else None
    p2_D_Bbar_gt_D_S_bar_all = bool(all(b > s for b, s in zip(d_bbars, d_s_bars))) if d_bbars else None

    relax_tau = None
    for fr in torus_frames:
        if fr["degenerate"]:
            continue
        for cell in fr["per_tau"]:
            if cell["D_I"] > cell["D_S"]:
                if relax_tau is None or cell["tau"] < relax_tau:
                    relax_tau = cell["tau"]
                break

    print(f"Stage D: p2_row_spread_max={p2_row_spread_max}, p2_D_S_bar_max={p2_D_S_bar_max}, "
          f"p2_D_Bbar_gt_D_S_bar_all={p2_D_Bbar_gt_D_S_bar_all}, memory_relaxation_tau={relax_tau}")

    return {
        "torus_memory_relaxation_tau": relax_tau,
        "p2_row_spread_Fbar_max_over_frames": p2_row_spread_max,
        "p2_D_S_bar_max_over_frames": p2_D_S_bar_max,
        "p2_D_Bbar_gt_D_S_bar_all_frames": p2_D_Bbar_gt_D_S_bar_all,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Pipeline gate (torus, hybrid A_C)")
    print("=" * 60)
    y_torus = build_stream("torus", T, np.random.default_rng(88))
    pipeline_torus = build_pipeline(y_torus)
    stageA = stage_a(pipeline_torus)
    print(f"\nStage A: {'PASS' if stageA['pass'] else 'FAIL'}")

    if not stageA["pass"]:
        metrics = {
            "lab": 91,
            "stream": "cryptographic-substrate",
            "kind": "hardened-simulation",
            "version": "v2-hybrid",
            "serves_node": "born_rule_emergence",
            "stageA": stageA,
            "stageB": {"frames": [], "born_fraction_torus": None},
            "stageC": {
                "logistic": {"j_admissible": None, "born_fraction": None, "frames": [],
                             "phase_source": None, "phase_match_max_delta": None},
                "iid": {"j_admissible": None, "born_fraction": None, "frames": [],
                        "phase_source": None, "phase_match_max_delta": None},
            },
            "stageD": {
                "torus_memory_relaxation_tau": None,
                "p2_row_spread_Fbar_max_over_frames": None,
                "p2_D_S_bar_max_over_frames": None,
                "p2_D_Bbar_gt_D_S_bar_all_frames": None,
            },
            "verdict": "VOID",
            "verdict_reason": "Stage A pipeline gate failed even with the hybrid estimator - something new is wrong. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False,
                "reimplemented_lab88_functions": False,
                "posthoc_threshold_change": False,
                "dropped_cells_outside_degeneracy_rule": False,
                "empirical_probability_from_quadratic_form": False,
            },
        }
        with open(LAB_FOLDER / "metrics_91_v2.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - Main measurement (torus, hybrid A_C_used)")
    print("=" * 60)
    torus_frames, born_fraction_torus, torus_F_by_tau = run_stage_b_or_c(pipeline_torus, "torus")

    print("\n" + "=" * 60)
    print("STAGE C - Controls (logistic, iid) - own hybrid pipeline each")
    print("=" * 60)
    stageC = {}
    for name in ["logistic", "iid"]:
        print(f"\n--- {name} ---")
        y_ctrl = build_stream(name, T, np.random.default_rng(88))
        Z_ctrl, X_ctrl = build_Z(y_ctrl)
        M_cross = Z_ctrl[1:].T @ Z_ctrl[:-1]
        U_p, _, Vt_p = np.linalg.svd(M_cross)
        A_ctrl = U_p @ Vt_p
        jt_ctrl = j_tests(A_ctrl)
        if jt_ctrl["real_eig_rule_failed"]:
            print(f"{name}: j_tests real_eig_rule_failed=True (Procrustes) -> j_admissible=False, skipping Born comparison.")
            stageC[name] = {"j_admissible": False, "born_fraction": None, "frames": [],
                             "phase_source": None, "phase_match_max_delta": None}
            continue
        pipeline_ctrl = build_pipeline(y_ctrl)
        print(f"{name}: phase_source={pipeline_ctrl['phase_source']}, "
              f"phase_match_max_delta={pipeline_ctrl['phase_match_max_delta']}")
        frames_ctrl, born_fraction_ctrl, _ = run_stage_b_or_c(pipeline_ctrl, name)
        stageC[name] = {
            "j_admissible": True, "born_fraction": born_fraction_ctrl, "frames": frames_ctrl,
            "phase_source": pipeline_ctrl["phase_source"],
            "phase_match_max_delta": pipeline_ctrl["phase_match_max_delta"],
        }

    print("\n" + "=" * 60)
    print("STAGE D - Wall characterization (torus, required regardless of verdict)")
    print("=" * 60)
    stageD = run_stage_d(torus_frames)

    # Verdict (identical rules to lab-goal.md; Stage A as amended)
    logistic_ok = (not stageC["logistic"]["j_admissible"]) or (stageC["logistic"]["born_fraction"] is not None and stageC["logistic"]["born_fraction"] <= 0.10)
    iid_ok = (not stageC["iid"]["j_admissible"]) or (stageC["iid"]["born_fraction"] is not None and stageC["iid"]["born_fraction"] <= 0.10)
    controls_ok = logistic_ok and iid_ok

    bft = born_fraction_torus if born_fraction_torus is not None else 0.0

    if bft >= 0.30 and controls_ok:
        verdict = "PASS"
        verdict_reason = (
            f"Stage A passed (hybrid estimator); born_fraction_torus={bft:.4f} >= 0.30; controls satisfy <= 0.10 "
            f"or J-inadmissible (logistic: {stageC['logistic']['born_fraction']}, iid: {stageC['iid']['born_fraction']})."
        )
    elif (0.10 <= bft < 0.30 and controls_ok) or (bft >= 0.30 and not controls_ok):
        verdict = "PARTIAL"
        if bft >= 0.30:
            verdict_reason = (
                f"born_fraction_torus={bft:.4f} >= 0.30 but a control exceeded 0.10 "
                f"(logistic: {stageC['logistic']['born_fraction']}, iid: {stageC['iid']['born_fraction']}) - "
                f"preference not world-specific."
            )
        else:
            verdict_reason = (
                f"born_fraction_torus={bft:.4f} in [0.10, 0.30) with controls satisfying <= 0.10 - "
                f"weak world-specific preference."
            )
    else:
        verdict = "FAIL"
        verdict_reason = (
            f"Stage A passed (hybrid estimator); born_fraction_torus={bft:.4f} < 0.10 - the predeclared-expected wall. "
            f"See Stage D for the quantified classical-transfer signature."
        )

    metrics = {
        "lab": 91,
        "stream": "cryptographic-substrate",
        "kind": "hardened-simulation",
        "version": "v2-hybrid",
        "serves_node": "born_rule_emergence",
        "stageA": stageA,
        "stageB": {"frames": torus_frames, "born_fraction_torus": born_fraction_torus},
        "stageC": stageC,
        "stageD": stageD,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False,
            "reimplemented_lab88_functions": False,
            "posthoc_threshold_change": False,
            "dropped_cells_outside_degeneracy_rule": False,
            "empirical_probability_from_quadratic_form": False,
        },
    }

    with open(LAB_FOLDER / "metrics_91_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
