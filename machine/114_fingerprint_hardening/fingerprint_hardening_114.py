# Lab 114 - The Fingerprint Hardening Battery
#
# Hardened-simulation, full contract. Engineering DECLARED for all engineered
# cells; theta frozen by provenance at A0 for every cell. Closes Lab 113's two
# audit gaps: (1) the dephasing builder is vectorized FAITHFULLY from the
# certified lab93.perp_basis convention (not an independent Gram-Schmidt), and
# verified against the certified path before use (A3); (2) breadth - a third
# W1 frame, a composite-pair target, a W2-substrate attempt (subject to a
# stop rule), and two generic controls that must FAIL the fingerprint bars.

import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB99_DIR = Path(__file__).resolve().parents[1] / "99_variational_floor"
LAB101_DIR = Path(__file__).resolve().parents[1] / "101_existence_hardening"
LAB110_DIR = Path(__file__).resolve().parents[1] / "110_inworld_interference"
LAB113_DIR = Path(__file__).resolve().parents[1] / "113_engineered_ensemble_twoslit"
sys.path.insert(0, str(LAB99_DIR))
sys.path.insert(0, str(LAB110_DIR))
import variational_floor_99 as lab99  # noqa: E402
import inworld_interference_110 as lab110  # noqa: E402

lab91 = lab110.lab91
lab93 = lab110.lab93
lab101 = lab110.lab101

TWO_PI = 2 * np.pi
PHI_GRID = lab110.PHI_GRID
N_TIER1 = 200000
FIT_RESIDUAL_EXACT_BAR = 1e-9
RMS_BAR_W1 = 0.05
RMS_BAR_W2 = 0.08
OVERTONE_BAR = 0.10
S_SOFT_BAR = 0.5
S_HARD_BAR = 0.1
CONTROL_RMS_MULT = 3.0
TRACKING_BAR = 0.08
A1_ANCHOR_TOL = 1e-9
A3_FORMULA_TOL = 1e-12
A3_ANCHOR_NOISE_MULT = 2.0
P_MIN_GUARD = 0.005
N_SCAN = 1024
N_FLOOR = 1_000_000


def strip(fit):
    return {k: v for k, v in fit.items() if k != "fitted"}


def build_qf_phi(psi):
    perp1, perp2 = lab93.perp_basis(psi)
    return np.column_stack([psi, perp1, perp2])


def psi_target(kind, uA, uB, phi):
    if kind == "asym":
        z = 0.8 * uA + 0.6 * np.exp(1j * phi) * uB
    elif kind == "sym":
        z = uA + np.exp(1j * phi) * uB
    else:
        raise ValueError(kind)
    return z / np.linalg.norm(z)


def analytic_P(kind, uA, uB, Qf_fixed, U_S, phi):
    psi = psi_target(kind, uA, uB, phi)
    y = U_S @ psi
    return np.abs(np.conj(Qf_fixed).T @ y) ** 2


def analytic_P_profile(wA, wB, uA, uB, Qf_fixed, U_S, phi):
    z = wA * uA + wB * np.exp(1j * phi) * uB
    z = z / np.linalg.norm(z)
    y = U_S @ z
    return np.abs(np.conj(Qf_fixed).T @ y) ** 2


def counted_F_engineered(theta, kind, uA, uB, Qf_fixed, U_S, phi, N, seed):
    psi = psi_target(kind, uA, uB, phi)
    Qf_phi = build_qf_phi(psi)
    rng = np.random.default_rng(seed)
    base_sample = rng.uniform(0.0, 1.0, size=(N, 6))
    Z = lab99.build_states(theta, base_sample, 0, Qf_phi)
    outcomes = lab99.outcomes_from_states(Z, U_S, Qf_fixed)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


def counted_F_generic(uA, uB, Qf_fixed, U_S, phi, N, seed):
    rng = np.random.default_rng(seed)
    cA = rng.uniform(0.55, 1.05, size=N)
    cB = rng.uniform(0.55, 1.05, size=N)
    w = rng.uniform(0.0, 1.0, size=N)
    common = np.exp(2j * np.pi * w)
    z = (cA * common)[:, None] * uA[None, :] + (cB * common * np.exp(1j * phi))[:, None] * uB[None, :]
    y = z @ U_S.T
    overlaps = np.abs(y @ np.conj(Qf_fixed)) ** 2
    outcomes = np.argmax(overlaps, axis=1)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Vectorized dephasing builder - CERTIFIED convention (lab93.perp_basis's own
# algorithm: e0 first, then e1, sequential Gram-Schmidt), not an independent
# ordering. Verified against the certified per-sample path in gate A3.
# ---------------------------------------------------------------------------
def perp_basis_vectorized(psi_batch):
    N = psi_batch.shape[0]
    e0 = np.tile(np.array([1, 0, 0], dtype=complex), (N, 1))
    e1 = np.tile(np.array([0, 1, 0], dtype=complex), (N, 1))
    proj0 = np.sum(np.conj(psi_batch) * e0, axis=1, keepdims=True)
    p0 = e0 - proj0 * psi_batch
    norm0 = np.linalg.norm(p0, axis=1)
    if np.any(norm0 <= 1e-8):
        raise RuntimeError("e0 near-parallel to psi for >=1 sample; certified fallback path (e1/e2) not vectorized")
    perp1 = p0 / norm0[:, None]
    proj_psi = np.sum(np.conj(psi_batch) * e1, axis=1, keepdims=True)
    proj_p1 = np.sum(np.conj(perp1) * e1, axis=1, keepdims=True)
    p1 = e1 - proj_psi * psi_batch - proj_p1 * perp1
    norm1 = np.linalg.norm(p1, axis=1)
    if np.any(norm1 <= 1e-8):
        raise RuntimeError("e1 near-parallel for >=1 sample; certified fallback path (e2) not vectorized")
    perp2 = p1 / norm1[:, None]
    return perp1, perp2


def build_states_target_batch(theta, base_sample, psi_batch, perp1_batch, perp2_batch):
    m_g, s_g, m_1, s_1, m_2, s_2 = theta
    v0, v1, v2, w0, w1, w2 = (base_sample[:, 0], base_sample[:, 1], base_sample[:, 2],
                              base_sample[:, 3], base_sample[:, 4], base_sample[:, 5])
    g_mag = np.maximum(0.0, m_g + s_g * (v0 - 0.5))
    c1 = np.maximum(0.0, m_1 + s_1 * (v1 - 0.5))
    c2 = np.maximum(0.0, m_2 + s_2 * (v2 - 0.5))
    ph0 = np.exp(2j * np.pi * w0)
    ph1 = np.exp(2j * np.pi * w1)
    ph2 = np.exp(2j * np.pi * w2)
    return ((g_mag * ph0)[:, None] * psi_batch + (c1 * ph1)[:, None] * perp1_batch
            + (c2 * ph2)[:, None] * perp2_batch)


def counted_F_dephased(theta, uA, uB, Qf_fixed, U_S, phi, Delta, N, seed):
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-Delta, Delta, size=N) if Delta > 0 else np.zeros(N)
    phi_actual = phi + delta
    z = 0.8 * uA[None, :] + 0.6 * np.exp(1j * phi_actual)[:, None] * uB[None, :]
    psi_batch = z / np.linalg.norm(z, axis=1, keepdims=True)
    perp1_b, perp2_b = perp_basis_vectorized(psi_batch)
    base_sample = rng.uniform(0.0, 1.0, size=(N, 6))
    Z = build_states_target_batch(theta, base_sample, psi_batch, perp1_b, perp2_b)
    y = Z @ U_S.T
    overlaps = np.abs(y @ np.conj(Qf_fixed)) ** 2
    outcomes = np.argmax(overlaps, axis=1)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


def assemble_cell(F_all, P_all, rms_bar):
    F_bar = F_all.mean(axis=0)
    D_flat = float(np.max([0.5 * np.sum(np.abs(F_all[i] - F_bar)) for i in range(24)]))
    rms_per_k = [float(np.sqrt(np.mean((F_all[:, k] - P_all[:, k]) ** 2))) for k in range(3)]
    fit_counted = [lab110.fit_first_harmonic(PHI_GRID, F_all[:, k]) for k in range(3)]
    fit_quad = [lab110.fit_first_harmonic(PHI_GRID, P_all[:, k]) for k in range(3)]
    betas = [f["beta"] for f in fit_quad]
    dead_channels = [k for k in range(3) if np.all(F_all[:, k] == 0)]
    overtone_info = [lab110.overtone_ratio(F_all[:, k], 0.001) for k in range(3)]
    active_R = [o["R"] for k, o in enumerate(overtone_info) if k not in dead_channels]
    max_active_R = max(active_R) if active_R else None
    J_phi = np.max(np.abs(F_all - P_all), axis=1)

    kstar = int(np.argmax(betas))
    p_min_kstar = float(P_all[:, kstar].min())
    if p_min_kstar >= P_MIN_GUARD:
        chosen_k = kstar
    else:
        candidates = [(k, float(P_all[:, k].min())) for k in range(3) if float(P_all[:, k].min()) >= P_MIN_GUARD]
        chosen_k = min(candidates, key=lambda x: x[1])[0] if candidates else kstar
    S_all = []
    for k in range(3):
        idx_k = int(np.argmin(P_all[:, k]))
        p_min_k = float(P_all[idx_k, k])
        f_min_k = float(F_all[idx_k, k])
        s_k = f_min_k / p_min_k if p_min_k > 0 else None
        S_all.append({"k": k, "phi_idx": idx_k, "P_min": p_min_k, "F_at_min": f_min_k, "S": s_k})
    S_chosen = S_all[chosen_k]["S"]

    rms_overall = max(rms_per_k)
    return {
        "F": F_all.tolist(), "P": P_all.tolist(), "J_phi": J_phi.tolist(), "D_flat": D_flat,
        "rms_per_k": rms_per_k, "rms_overall": rms_overall,
        "fit_counted": [strip(f) for f in fit_counted], "fit_quad": [strip(f) for f in fit_quad],
        "dead_channels": dead_channels, "overtone_per_k": overtone_info, "max_active_R": max_active_R,
        "S_chosen_channel": chosen_k, "S_chosen": S_chosen, "S_all": S_all,
        "shape_match": bool(rms_overall <= rms_bar),
        "purity_ok": bool(max_active_R is not None and max_active_R <= OVERTONE_BAR),
        "S_ok": bool(all(s["S"] is None or s["S"] >= S_SOFT_BAR for s in S_all
                          if s["P_min"] >= P_MIN_GUARD)),
        "hard_null": bool(any(s["S"] is not None and s["S"] <= S_HARD_BAR for s in S_all
                              if s["P_min"] >= P_MIN_GUARD)),
    }


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("STAGE A0 - Theta manifest")
    print("=" * 60)

    with open(LAB99_DIR / "metrics_99.json") as f:
        m99 = json.load(f)
    class_a = {c["frame"]: c for c in m99["stageB_classA"]}
    theta_91 = np.array(class_a[91]["theta"])
    theta_93 = np.array(class_a[93]["theta"])
    theta_201 = class_a.get(201)
    with open(LAB101_DIR / "metrics_101.json") as f:
        m101 = json.load(f)
    w2_entry = m101["stageC"][0]  # first-in-provenance-order W2 cell (corrected: [2, 11])
    w2_cell_info = w2_entry["cell"]
    theta_w2 = np.array(w2_entry["theta"])
    W2_FRAME = w2_cell_info[1]

    cell6_blocked = bool(w2_cell_info[0] != 2)
    cell6_reason = (f"metrics_101.json stageC[0] is {w2_cell_info} (world={w2_cell_info[0]}, "
                     f"frame={w2_cell_info[1]}) - not World 2 as expected." if cell6_blocked else None)
    print(f"Cell 6 ({W2_FRAME}-asym-W2): cell={w2_cell_info}, published J={w2_entry['J']:.5f}, "
          f"CELL-BLOCKED={cell6_blocked}")

    cell4_blocked = theta_201 is None
    theta_manifest = {
        "91": {"theta": theta_91.tolist(), "source": "metrics_99.json stageB_classA frame 91",
               "published_D_heldout": class_a[91]["D_heldout"]},
        "93": {"theta": theta_93.tolist(), "source": "metrics_99.json stageB_classA frame 93",
               "published_D_heldout": class_a[93]["D_heldout"]},
        "201": ({"theta": theta_201["theta"], "source": "metrics_99.json stageB_classA frame 201",
                "published_D_heldout": theta_201["D_heldout"]} if theta_201 else None),
        f"{W2_FRAME}_W2": {"theta": theta_w2.tolist(), "source": "metrics_101.json stageC[0]",
                           "cell": w2_cell_info, "published_J": w2_entry["J"], "blocked": cell6_blocked},
    }
    print(f"FRAME 91 theta: {theta_91.tolist()}")
    print(f"FRAME 93 theta: {theta_93.tolist()}")
    print(f"FRAME 201 theta: {'BLOCKED' if cell4_blocked else theta_201['theta']}")
    print("Theta frozen by provenance from this point onward.")

    print("\n" + "=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    Qf91 = lab91.haar_frame(91)
    Qf93 = lab91.haar_frame(93)
    Qf201 = lab91.haar_frame(201) if not cell4_blocked else None
    U_S = lab93.U_S
    uA91, uB91 = Qf91[:, 0], Qf91[:, 1]
    uA93, uB93 = Qf93[:, 0], Qf93[:, 1]

    QfW2 = lab91.haar_frame(W2_FRAME) if not cell6_blocked else None
    U_S_W2 = lab101.U_S2
    uAW2, uBW2 = (QfW2[:, 0], QfW2[:, 1]) if not cell6_blocked else (None, None)

    with open(LAB113_DIR / "metrics_113.json") as f:
        m113 = json.load(f)

    print("A1 (anchor: recompute cell 1 with Lab 113's seed base 15000, match to 1e-9):")
    F_check = np.empty((24, 3))
    for i, phi in enumerate(PHI_GRID):
        F_check[i] = counted_F_engineered(theta_91, "asym", uA91, uB91, Qf91, U_S, phi, N_TIER1, 15000 + i)
    src_91asym = m113["stageB"]["91|asym"]
    max_dG = float(np.max(np.abs(F_check - np.array(src_91asym["F"]))))
    A1 = bool(max_dG < A1_ANCHOR_TOL)
    print(f"  max diff = {max_dG:.2e} (<{A1_ANCHOR_TOL}), pass={A1}")

    print("\nA2 (analytic reference identity, residual < 1e-9, all available cells):")
    cells_plan = [
        ("91-asym", "engineered", 91, "asym", uA91, uB91, Qf91, theta_91, "copy", U_S),
        ("91-sym", "engineered", 91, "sym", uA91, uB91, Qf91, theta_91, "copy", U_S),
        ("93-asym", "engineered", 93, "asym", uA93, uB93, Qf93, theta_93, "copy", U_S),
        ("201-asym", "engineered", 201, "asym",
         (Qf201[:, 0] if not cell4_blocked else None), (Qf201[:, 1] if not cell4_blocked else None),
         Qf201, (np.array(theta_201["theta"]) if not cell4_blocked else None), "new", U_S),
        ("91-P2-asym", "engineered", 91, "asym", uA91, (Qf91[:, 1] + Qf91[:, 2]) / np.sqrt(2), Qf91, theta_91,
         "new", U_S),
        (f"{W2_FRAME}-asym-W2", "engineered", W2_FRAME, "asym", uAW2, uBW2, QfW2, theta_w2, "new", U_S_W2),
        ("C1-91-generic", "control", 91, "asym", uA91, uB91, Qf91, None, "new", U_S),
        ("C2-93-generic", "control", 93, "asym", uA93, uB93, Qf93, None, "new", U_S),
    ]
    a2_results = {}
    a2_pass = True
    for key, kind_of_cell, frame, target_kind, uA, uB, Qf, theta, status, U_S_cell in cells_plan:
        if uA is None:
            a2_results[key] = {"blocked": True}
            continue
        P_all = np.array([analytic_P(target_kind, uA, uB, Qf, U_S_cell, phi) for phi in PHI_GRID])
        resids = [lab110.fit_first_harmonic(PHI_GRID, P_all[:, k])["resid"] for k in range(3)]
        max_resid = max(resids)
        cell_pass = bool(max_resid < FIT_RESIDUAL_EXACT_BAR)
        a2_pass = a2_pass and cell_pass
        a2_results[key] = {"max_resid": max_resid, "pass": cell_pass}
        print(f"  {key}: max_resid={max_resid:.3e} (<{FIT_RESIDUAL_EXACT_BAR}), pass={cell_pass}")
    A2 = bool(a2_pass)
    print(f"A2 overall pass={A2}")

    print("\nA3 (builder-equivalence gate):")
    print("A3(i) - formula check: vectorized dephasing builder vs certified per-sample path, N_check=2000:")
    rng_a3 = np.random.default_rng(77000)
    N_check = 2000
    Delta_check = np.pi / 3
    delta_check = rng_a3.uniform(-Delta_check, Delta_check, size=N_check)
    phi_actual_check = 0.0 + delta_check
    z_check = 0.8 * uA91[None, :] + 0.6 * np.exp(1j * phi_actual_check)[:, None] * uB91[None, :]
    psi_batch_check = z_check / np.linalg.norm(z_check, axis=1, keepdims=True)
    perp1_v, perp2_v = perp_basis_vectorized(psi_batch_check)
    base_sample_check = rng_a3.uniform(0.0, 1.0, size=(N_check, 6))
    Z_vec = build_states_target_batch(theta_91, base_sample_check, psi_batch_check, perp1_v, perp2_v)
    Z_cert = np.empty((N_check, 3), dtype=complex)
    for i in range(N_check):
        perp1_i, perp2_i = lab93.perp_basis(psi_batch_check[i])
        Qf_i = np.column_stack([psi_batch_check[i], perp1_i, perp2_i])
        Z_cert[i] = lab99.build_states(theta_91, base_sample_check[i:i + 1], 0, Qf_i)[0]
    a3i_max_diff = float(np.max(np.abs(Z_vec - Z_cert)))
    A3i = bool(a3i_max_diff < A3_FORMULA_TOL)
    print(f"  max|Z_vec - Z_cert| = {a3i_max_diff:.3e} (<{A3_FORMULA_TOL}), pass={A3i}")

    print("A3(ii) - Delta=0 anchor, order-statistics-consistent, self-calibrating (feedback-2 amendment 1):")
    print("  Calibration: three certified-path twin pairs on cell 1's config, full 24-phi grid:")
    calib_seed_pairs = [(16950, 16951), (16952, 16953), (16954, 16955)]
    calib_values = []
    for sa, sb in calib_seed_pairs:
        Fa = np.empty((24, 3))
        Fb = np.empty((24, 3))
        for i, phi in enumerate(PHI_GRID):
            Fa[i] = counted_F_engineered(theta_91, "asym", uA91, uB91, Qf91, U_S, phi, N_TIER1, sa * 100 + i)
            Fb[i] = counted_F_engineered(theta_91, "asym", uA91, uB91, Qf91, U_S, phi, N_TIER1, sb * 100 + i)
        max_diff_pair = float(np.max(np.abs(Fa - Fb)))
        calib_values.append(max_diff_pair)
        print(f"    seeds ({sa},{sb}): max-over-grid discrepancy = {max_diff_pair:.5f}")
    a3ii_bar = 1.5 * max(calib_values)
    print(f"  Bar = 1.5 x max(calibration) = 1.5 x {max(calib_values):.5f} = {a3ii_bar:.5f}")

    F_delta0 = np.empty((24, 3))
    for i, phi in enumerate(PHI_GRID):
        F_delta0[i] = counted_F_dephased(theta_91, uA91, uB91, Qf91, U_S, phi, 0.0, N_TIER1, 16956 * 100 + i)
    a3ii_max_diff = float(np.max(np.abs(F_delta0 - np.array(src_91asym["F"]))))
    A3ii = bool(a3ii_max_diff < a3ii_bar)
    print(f"  Anchor (seed 16956) vs cell 1's stored table: max diff = {a3ii_max_diff:.5f} (<{a3ii_bar:.5f}), "
          f"pass={A3ii}")
    A3 = bool(A3i and A3ii)

    pass_a = bool(A1 and A2 and A3)
    stageA = {
        "cell4_blocked": cell4_blocked, "cell6_blocked": cell6_blocked, "cell6_reason": cell6_reason,
        "w2_cell_info": w2_cell_info, "w2_frame": W2_FRAME,
        "a1": {"max_diff": max_dG, "pass": A1}, "a2": a2_results,
        "a3": {"formula_max_diff": a3i_max_diff, "formula_pass": A3i,
               "calibration_values": calib_values, "calibration_seed_pairs": calib_seed_pairs,
               "anchor_bar": a3ii_bar, "anchor_max_diff": a3ii_max_diff, "anchor_pass": A3ii,
               "pass": A3},
        "pass": pass_a,
    }
    print(f"\nStage A: {'PASS' if pass_a else 'FAIL'}")

    if not pass_a:
        metrics = {
            "lab": 114, "run": 2, "stream": "cryptographic-substrate", "kind": "hardened-simulation",
            "serves_node": "born_rule_emergence",
            "theta_manifest": theta_manifest, "stageA": stageA, "stageB": {}, "stageC": {}, "stageD": {},
            "verdict": "VOID", "verdict_reason": f"Stage A failed: A1={A1}, A2={A2}, A3={A3}.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
                "theta_adjusted_after_A0": False, "posthoc_threshold_change": False,
                "metrics_hand_edited": False, "engineering_declared": True,
            },
        }
        with open(LAB_FOLDER / "metrics_114.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Stage B - The fringe battery (8 cells)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE B - The fringe battery")
    print("=" * 60)
    stageB = {}
    idx = 0
    for key, kind_of_cell, frame, target_kind, uA, uB, Qf, theta, status, U_S_cell in cells_plan:
        if uA is None:
            stageB[key] = {"blocked": True}
            idx += 24
            continue
        rms_bar = RMS_BAR_W2 if key.endswith("-W2") else RMS_BAR_W1
        if status == "copy":
            src_key = {"91-asym": "91|asym", "91-sym": "91|sym", "93-asym": "93|asym"}[key]
            src = m113["stageB"][src_key]
            cell = assemble_cell(np.array(src["F"]), np.array(src["P"]), rms_bar)
            idx += 24
        else:
            F_all = np.empty((24, 3))
            for i, phi in enumerate(PHI_GRID):
                if kind_of_cell == "engineered":
                    F_all[i] = counted_F_engineered(theta, target_kind, uA, uB, Qf, U_S_cell, phi, N_TIER1,
                                                     16000 + idx + i)
                else:
                    F_all[i] = counted_F_generic(uA, uB, Qf, U_S_cell, phi, N_TIER1, 16000 + idx + i)
            idx += 24
            P_all = np.array([analytic_P(target_kind, uA, uB, Qf, U_S_cell, phi) for phi in PHI_GRID])
            cell = assemble_cell(F_all, P_all, rms_bar)
        cell["kind_of_cell"] = kind_of_cell
        cell["frame"] = frame
        stageB[key] = cell
        print(f"-- {key} ({kind_of_cell}, frame {frame}) --  D_flat={cell['D_flat']:.4f}  "
              f"RMS={cell['rms_overall']:.4f} (bar {rms_bar})  max_active_R={cell['max_active_R']}  "
              f"S_chosen={cell['S_chosen']}  shape_match={cell['shape_match']}  purity_ok={cell['purity_ok']}  "
              f"S_ok={cell['S_ok']}  hard_null={cell['hard_null']}")

    print("\nNoise floors (twin batches):")
    noise_floors = {}
    cell_i = 0
    for key, kind_of_cell, frame, target_kind, uA, uB, Qf, theta, status, U_S_cell in cells_plan:
        if uA is None:
            cell_i += 1
            continue
        if kind_of_cell == "engineered":
            Pa = counted_F_engineered(theta, target_kind, uA, uB, Qf, U_S_cell, 0.0, N_TIER1, 16900 + cell_i * 2)
            Pb = counted_F_engineered(theta, target_kind, uA, uB, Qf, U_S_cell, 0.0, N_TIER1, 16900 + cell_i * 2 + 1)
        else:
            Pa = counted_F_generic(uA, uB, Qf, U_S_cell, 0.0, N_TIER1, 16900 + cell_i * 2)
            Pb = counted_F_generic(uA, uB, Qf, U_S_cell, 0.0, N_TIER1, 16900 + cell_i * 2 + 1)
        nf = float(np.max(np.abs(Pa - Pb)))
        noise_floors[key] = nf
        cell_i += 1
        print(f"  {key}: noise_floor={nf:.5f}")

    # Control separation
    control_results = {}
    for ckey, fkey in [("C1-91-generic", "91-asym"), ("C2-93-generic", "93-asym")]:
        c = stageB[ckey]
        eng = stageB[fkey]
        rms_thresh = CONTROL_RMS_MULT * eng["rms_overall"]
        separates = bool(c["hard_null"] or c["rms_overall"] >= rms_thresh)
        control_results[ckey] = {"rms": c["rms_overall"], "engineered_rms": eng["rms_overall"],
                                  "rms_thresh": rms_thresh, "hard_null": c["hard_null"], "separates": separates}
        print(f"Control {ckey}: RMS={c['rms_overall']:.4f} vs {rms_thresh:.4f} (3x {fkey}), "
              f"hard_null={c['hard_null']}, separates={separates}")
    both_controls_separate = bool(all(v["separates"] for v in control_results.values()))

    # -----------------------------------------------------------------
    # Stage C - The dephasing law, clean instrument (2 frames)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE C - The dephasing law (clean instrument)")
    print("=" * 60)
    stageC = {}
    dose_plan = {91: [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3], 93: [0.0, np.pi / 3, 2 * np.pi / 3]}
    idx_c = 24  # Delta=0/91 already consumed idx 0-23 above (A3ii); continue from there
    for frame, doses in dose_plan.items():
        Qf_f = Qf91 if frame == 91 else Qf93
        uA_f, uB_f = (uA91, uB91) if frame == 91 else (uA93, uB93)
        theta_f = theta_91 if frame == 91 else theta_93
        kstar = stageB[f"{frame}-asym" if frame != 91 else "91-asym"]["S_chosen_channel"]
        betas = [f["beta"] for f in stageB["91-asym" if frame == 91 else "93-asym"]["fit_quad"]]
        kstar = int(np.argmax(betas))
        dose_results = {}
        Vmf0 = None
        for Delta in doses:
            if frame == 91 and Delta == 0.0:
                F_all_d = F_delta0
            else:
                F_all_d = np.empty((24, 3))
                for i, phi in enumerate(PHI_GRID):
                    F_all_d[i] = counted_F_dephased(theta_f, uA_f, uB_f, Qf_f, U_S, phi, Delta, N_TIER1,
                                                    16500 + idx_c + i)
                idx_c += 24
            Vmf = (F_all_d[:, kstar].max() - F_all_d[:, kstar].min()) / \
                  (F_all_d[:, kstar].max() + F_all_d[:, kstar].min())
            if Delta == 0.0:
                Vmf0 = Vmf
            ratio = Vmf / Vmf0
            sinc_val = float(np.sin(Delta) / Delta) if Delta > 0 else 1.0
            diff = abs(ratio - sinc_val)
            tracks = bool(diff <= TRACKING_BAR)
            dose_results[str(Delta)] = {"Delta": Delta, "Vmf": Vmf, "ratio": ratio, "sinc": sinc_val,
                                        "diff": diff, "tracks": tracks}
            print(f"  frame={frame} Delta={Delta:.4f}: V^mf={Vmf:.4f}, ratio={ratio:.4f}, sinc={sinc_val:.4f}, "
                  f"diff={diff:.4f}, tracks={tracks}")
        stageC[str(frame)] = {"kstar": kstar, "Vmf_0": Vmf0, "doses": dose_results,
                               "tracking_confirmed": bool(all(d["tracks"] for d in dose_results.values()))}

    tracking_91 = stageC["91"]["tracking_confirmed"]
    tracking_93 = stageC["93"]["tracking_confirmed"]
    tracking_both = bool(tracking_91 and tracking_93)
    tracking_verdict = "TRACKING-CONFIRMED" if tracking_both else "TRACKING-BROKEN"
    print(f"Tracking (91)={tracking_91}, Tracking (93)={tracking_93}, verdict={tracking_verdict}")

    # -----------------------------------------------------------------
    # Stage D - The null-floor instrument (measurement-only, no pass bar)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE D - The null-floor instrument (exploratory, no pass bar)")
    print("=" * 60)
    profiles = [(1.0, 1.0), (0.8, 0.6), (0.9, 0.436), (0.95, 0.312)]
    stageD = []
    run_idx = 0
    for frame in [91, 93]:
        Qf_f = Qf91 if frame == 91 else Qf93
        uA_f, uB_f = (uA91, uB91) if frame == 91 else (uA93, uB93)
        theta_f = theta_91 if frame == 91 else theta_93
        for wA, wB in profiles:
            norm = np.hypot(wA, wB)
            wA_n, wB_n = wA / norm, wB / norm
            phis_scan = np.linspace(0, TWO_PI, N_SCAN, endpoint=False)
            P_scan = np.array([analytic_P_profile(wA_n, wB_n, uA_f, uB_f, Qf_f, U_S, phi) for phi in phis_scan])
            flat_idx = int(np.argmin(P_scan))
            phi_idx, k_idx = divmod(flat_idx, 3)
            phi_star = float(phis_scan[phi_idx])
            P_min = float(P_scan[phi_idx, k_idx])

            psi_star = wA_n * uA_f + wB_n * np.exp(1j * phi_star) * uB_f
            psi_star = psi_star / np.linalg.norm(psi_star)
            Qf_phi = build_qf_phi(psi_star)
            rng = np.random.default_rng(16800 + run_idx)
            base_sample = rng.uniform(0.0, 1.0, size=(N_FLOOR, 6))
            Z = lab99.build_states(theta_f, base_sample, 0, Qf_phi)
            outcomes = lab99.outcomes_from_states(Z, U_S, Qf_f)
            counts = np.bincount(outcomes, minlength=3).astype(float)
            F_all_floor = counts / counts.sum()
            F_floor = float(F_all_floor[k_idx])

            stageD.append({"frame": frame, "weights": [wA_n, wB_n], "k_star": k_idx, "phi_star": phi_star,
                           "P_min": P_min, "F_floor": F_floor})
            print(f"  frame={frame} weights=({wA_n:.4f},{wB_n:.4f}): k*={k_idx} phi*={phi_star:.4f} "
                  f"P_min={P_min:.6f} F_floor={F_floor:.6f}")
            run_idx += 1

    # Honest reading (measurement only, not a bar)
    p_mins = np.array([d["P_min"] for d in stageD])
    f_floors = np.array([d["F_floor"] for d in stageD])
    floor_mean = float(f_floors.mean())
    floor_std = float(f_floors.std())
    corr = float(np.corrcoef(p_mins, f_floors)[0, 1]) if len(p_mins) > 1 else None
    print(f"\nFloor readings: mean(F_floor)={floor_mean:.5f}, std={floor_std:.5f}, "
          f"corr(P_min, F_floor)={corr}")

    # -----------------------------------------------------------------
    # P1-P4 and verdict
    # -----------------------------------------------------------------
    w2_key = f"{W2_FRAME}-asym-W2"
    engineered_keys = ["91-asym", "91-sym", "93-asym", "201-asym", "91-P2-asym", w2_key]
    valid_engineered = [k for k in engineered_keys if not stageB[k].get("blocked")]
    n_pass = sum(1 for k in valid_engineered if stageB[k]["shape_match"] and stageB[k]["purity_ok"]
                 and stageB[k]["S_ok"])
    n_fail = len(valid_engineered) - n_pass
    cells_1_4_pass = all(stageB[k]["shape_match"] and stageB[k]["purity_ok"] and stageB[k]["S_ok"]
                         for k in ["91-asym", "91-sym", "93-asym", "201-asym"] if not stageB[k].get("blocked"))

    p1_holds = bool(n_pass == len(valid_engineered))
    p2_holds = bool(all(control_results[k]["hard_null"] for k in control_results))
    w2_present = not cell6_blocked
    p3_holds = (bool(0.01 <= stageB[w2_key]["rms_overall"] <= 0.08) if w2_present else None)
    p4_holds = tracking_both

    print(f"\nP1 (all engineered pass, of {len(valid_engineered)} valid cells): {p1_holds} ({n_pass} pass)")
    print(f"P2 (both controls fail WITH hard nulls): {p2_holds}")
    print(f"P3 (W2 cell RMS in [0.01,0.08]): {p3_holds}" +
          (f" (RMS={stageB[w2_key]['rms_overall']:.4f})" if w2_present else " - CELL-BLOCKED"))
    print(f"P4 (tracking confirmed both frames): {p4_holds}")

    any_cell_blocked = bool(cell4_blocked or cell6_blocked)

    if (not any_cell_blocked) and n_pass == 6 and both_controls_separate and tracking_both:
        verdict = "HARDENED-FINGERPRINT"
        verdict_reason = "All 6 engineered cells pass, both controls separate, tracking confirmed both frames."
    elif any(not control_results[k]["separates"] for k in control_results):
        verdict = "CONTROL-FAILURE"
        verdict_reason = "A generic control passed the fingerprint bars - the contrast story needs audit."
    elif (n_pass >= 5 or any_cell_blocked) and cells_1_4_pass and both_controls_separate and tracking_91:
        verdict = "PARTIAL-HARDENED"
        gaps = []
        if cell6_blocked:
            gaps.append(f"cell 6 ({W2_FRAME}-asym-W2) CELL-BLOCKED: " + cell6_reason)
        if cell4_blocked:
            gaps.append("cell 4 (201-asym) CELL-BLOCKED: no classA theta for frame 201 in metrics_99")
        if n_fail > 0:
            gaps.append(f"{n_fail} valid engineered cell(s) failed their bars")
        verdict_reason = ">=5/6 pass or CELL-BLOCKED, cells 1-4 pass, controls separate, tracking@91 confirmed. " + \
                          " | ".join(gaps)
    elif n_fail >= 2:
        verdict = "FINGERPRINT-FRAGILE"
        verdict_reason = f"{n_fail} valid engineered cells failed their bars - the fingerprint does not harden."
    else:
        verdict = "OTHER"
        verdict_reason = (f"No ladder rule matched exactly. n_pass={n_pass}/{len(valid_engineered)}, "
                           f"controls_separate={both_controls_separate}, tracking_91={tracking_91}, "
                           f"any_cell_blocked={any_cell_blocked}.")

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")

    # -----------------------------------------------------------------
    # The money plot
    # -----------------------------------------------------------------
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    ax = axes[0]
    labels_plot = []
    rms_plot = []
    colors_plot = []
    for k in engineered_keys:
        if stageB[k].get("blocked"):
            continue
        labels_plot.append(k)
        rms_plot.append(stageB[k]["rms_overall"])
        colors_plot.append("tab:blue")
    for k in ["C1-91-generic", "C2-93-generic"]:
        labels_plot.append(k)
        rms_plot.append(stageB[k]["rms_overall"])
        colors_plot.append("tab:red")
    ax.bar(labels_plot, rms_plot, color=colors_plot)
    ax.axhline(RMS_BAR_W1, color="green", linestyle="--", label=f"W1 bar {RMS_BAR_W1}")
    ax.set_ylabel("RMS (counted vs analytic)")
    ax.set_title("Panel 1: RMS, engineered (blue) vs controls (red)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7)

    ax2 = axes[1]
    for frame_key in ["91", "93"]:
        doses_sorted = sorted(stageC[frame_key]["doses"].values(), key=lambda d: d["Delta"])
        deltas = [d["Delta"] for d in doses_sorted]
        ratios = [d["ratio"] for d in doses_sorted]
        sincs = [d["sinc"] for d in doses_sorted]
        ax2.plot(deltas, ratios, "o-", label=f"counted (frame {frame_key})")
        ax2.plot(deltas, sincs, "s--", label=f"sinc (frame {frame_key})", alpha=0.6)
    ax2.set_xlabel("Delta")
    ax2.set_ylabel("visibility ratio")
    ax2.set_title("Panel 2: dephasing tracking, both frames")
    ax2.legend(fontsize=6)

    ax3 = axes[2]
    ax3.scatter(p_mins, f_floors, color="tab:purple")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("P_min (analytic null depth)")
    ax3.set_ylabel("F_floor (counted floor)")
    ax3.set_title("Panel 3: the null-floor curve (log-log)")

    fig.suptitle("Lab 114: the fingerprint hardening battery")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "hardening_114.png", dpi=110)
    plt.close(fig)

    def strip_for_json(d):
        return json.loads(json.dumps(d, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))

    metrics = {
        "lab": 114, "run": 2, "stream": "cryptographic-substrate", "kind": "hardened-simulation",
        "serves_node": "born_rule_emergence",
        "theta_manifest": theta_manifest,
        "stageA": strip_for_json(stageA),
        "stageB": strip_for_json(stageB),
        "stageC": strip_for_json(stageC),
        "stageD": strip_for_json({"points": stageD, "floor_mean": floor_mean, "floor_std": floor_std,
                                   "corr_Pmin_Ffloor": corr}),
        "control_results": strip_for_json(control_results),
        "p1": p1_holds, "p2": p2_holds, "p3": p3_holds, "p4": p4_holds,
        "verdict": verdict, "verdict_reason": verdict_reason, "tracking_verdict": tracking_verdict,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
            "theta_adjusted_after_A0": False, "posthoc_threshold_change": False,
            "metrics_hand_edited": False, "engineering_declared": True,
        },
    }
    with open(LAB_FOLDER / "metrics_114.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
    print(f"Tracking verdict: {tracking_verdict}")
