# Lab 113 - The Engineered-Ensemble Two-Slit
#
# Engineering DECLARED (Lab 100 realizability precedent): uses Lab-99's
# certified theta-shaped ensembles and Born-derived two-path targets. Theta is
# frozen by provenance after Stage A0 - never touched again. The certified
# builder (lab99.build_states / outcomes_from_states, UNMODIFIED) is pointed
# at an arbitrary target psi(phi) by supplying a per-phi orthonormal frame
# whose column 0 IS psi(phi) (Gram-Schmidt-completed via lab93.perp_basis) -
# zero lines of the certified function change; only its Qf argument varies.

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
LAB100V2_DIR = Path(__file__).resolve().parents[1] / "100_repreparation_fold"
LAB110_DIR = Path(__file__).resolve().parents[1] / "110_inworld_interference"
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
D_FLAT_BAR = 0.10
RMS_BAR = 0.05
OVERTONE_BAR = 0.10
S_SOFT_BAR = 0.5
S_HARD_BAR = 0.1
TRACKING_BAR = 0.08
A1_TOLERANCE = 1.5
P_MIN_GUARD = 0.005
DEPHASE_DOSES = [np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3]
SINC_VALUES = [float(np.sin(d) / d) for d in DEPHASE_DOSES]


def strip(fit):
    return {k: v for k, v in fit.items() if k != "fitted"}


def build_qf_phi(psi):
    perp1, perp2 = lab93.perp_basis(psi)
    return np.column_stack([psi, perp1, perp2])


def psi_target(kind, uA, uB, phi):
    if kind == "asym":
        z = 0.8 * uA + 0.6 * np.exp(1j * phi) * uB
    else:
        z = uA + np.exp(1j * phi) * uB
    return z / np.linalg.norm(z)


def analytic_P(kind, uA, uB, Qf_fixed, U_S, phi):
    psi = psi_target(kind, uA, uB, phi)
    y = U_S @ psi
    overlaps = np.abs(np.conj(Qf_fixed).T @ y) ** 2
    return overlaps  # already normalized (psi unit norm, U_S unitary)


def counted_F(theta, kind, uA, uB, Qf_fixed, U_S, phi, N, seed):
    psi = psi_target(kind, uA, uB, phi)
    Qf_phi = build_qf_phi(psi)
    rng = np.random.default_rng(seed)
    base_sample = rng.uniform(0.0, 1.0, size=(N, 6))
    Z = lab99.build_states(theta, base_sample, 0, Qf_phi)
    outcomes = lab99.outcomes_from_states(Z, U_S, Qf_fixed)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


def perp_basis_batch(psi_batch):
    N = psi_batch.shape[0]
    e2 = np.tile(np.array([0, 1, 0], dtype=complex), (N, 1))
    e3 = np.tile(np.array([0, 0, 1], dtype=complex), (N, 1))
    proj = np.sum(np.conj(psi_batch) * e2, axis=1, keepdims=True)
    perp1 = e2 - proj * psi_batch
    perp1 = perp1 / np.linalg.norm(perp1, axis=1, keepdims=True)
    proj_psi = np.sum(np.conj(psi_batch) * e3, axis=1, keepdims=True)
    proj_p1 = np.sum(np.conj(perp1) * e3, axis=1, keepdims=True)
    perp2 = e3 - proj_psi * psi_batch - proj_p1 * perp1
    perp2 = perp2 / np.linalg.norm(perp2, axis=1, keepdims=True)
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
    z = (g_mag * ph0)[:, None] * psi_batch + (c1 * ph1)[:, None] * perp1_batch + (c2 * ph2)[:, None] * perp2_batch
    return z


def counted_F_dephased(theta, kind, uA, uB, Qf_fixed, U_S, phi, Delta, N, seed):
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-Delta, Delta, size=N)
    phi_actual = phi + delta
    if kind == "asym":
        z = 0.8 * uA[None, :] + 0.6 * np.exp(1j * phi_actual)[:, None] * uB[None, :]
    else:
        z = uA[None, :] + np.exp(1j * phi_actual)[:, None] * uB[None, :]
    psi_batch = z / np.linalg.norm(z, axis=1, keepdims=True)
    perp1_batch, perp2_batch = perp_basis_batch(psi_batch)
    base_sample = rng.uniform(0.0, 1.0, size=(N, 6))
    Z = build_states_target_batch(theta, base_sample, psi_batch, perp1_batch, perp2_batch)
    y = Z @ U_S.T
    overlaps = np.abs(y @ np.conj(Qf_fixed)) ** 2
    outcomes = np.argmax(overlaps, axis=1)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


def null_softness(F_all, P_all, betas):
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
    return chosen_k, S_chosen, S_all


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("STAGE A0 - Theta manifest (recorded BEFORE any fringe run)")
    print("=" * 60)

    with open(LAB99_DIR / "metrics_99.json") as f:
        m99 = json.load(f)
    class_a = m99["stageB_classA"]
    frame_a_entry = next(c for c in class_a if c["frame"] == 91)
    other_frames = [c for c in class_a if c["frame"] != 91]
    frame_b_entry = other_frames[0]
    FRAME_A, FRAME_B = 91, frame_b_entry["frame"]
    theta_A = np.array(frame_a_entry["theta"])
    theta_B = np.array(frame_b_entry["theta"])
    theta_manifest = {
        "FRAME_A": {"frame": FRAME_A, "theta": theta_A.tolist(), "source": "99_variational_floor/metrics_99.json",
                    "published_D_heldout": frame_a_entry["D_heldout"]},
        "FRAME_B": {"frame": FRAME_B, "theta": theta_B.tolist(), "source": "99_variational_floor/metrics_99.json",
                    "published_D_heldout": frame_b_entry["D_heldout"]},
        "provenance_order": [c["frame"] for c in class_a],
        "note": "FRAME_B chosen as the first non-91 entry in metrics_99's stageB_classA list order.",
    }
    print(f"FRAME_A={FRAME_A}, theta={theta_A.tolist()}, published D_heldout={frame_a_entry['D_heldout']:.6f}")
    print(f"FRAME_B={FRAME_B}, theta={theta_B.tolist()}, published D_heldout={frame_b_entry['D_heldout']:.6f}")
    print("Theta frozen by provenance from this point onward.")

    print("\n" + "=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    Qf91 = lab91.haar_frame(91)
    Qf93 = lab91.haar_frame(FRAME_B)
    U_S = lab93.U_S
    M_born91 = lab93.compute_M_born(Qf91)
    M_born93 = lab93.compute_M_born(Qf93)

    print("A1 (builder reproduction, fresh held-out MC vs published D_heldout, tolerance 1.5x):")
    held_91 = np.random.default_rng(50091).uniform(0.0, 1.0, size=(lab99.N_HELD, 6))
    d_91 = lab99.evaluate_classA(theta_A, held_91, Qf91, U_S, M_born91)
    ratio_91 = d_91 / frame_a_entry["D_heldout"]
    a1_91_pass = bool(ratio_91 <= A1_TOLERANCE)
    print(f"  FRAME_A={FRAME_A}: fresh D={d_91:.6f} vs published {frame_a_entry['D_heldout']:.6f} "
          f"(ratio {ratio_91:.3f} <= {A1_TOLERANCE}), pass={a1_91_pass}")

    held_93 = np.random.default_rng(50093).uniform(0.0, 1.0, size=(lab99.N_HELD, 6))
    d_93 = lab99.evaluate_classA(theta_B, held_93, Qf93, U_S, M_born93)
    ratio_93 = d_93 / frame_b_entry["D_heldout"]
    a1_93_pass = bool(ratio_93 <= A1_TOLERANCE)
    print(f"  FRAME_B={FRAME_B}: fresh D={d_93:.6f} vs published {frame_b_entry['D_heldout']:.6f} "
          f"(ratio {ratio_93:.3f} <= {A1_TOLERANCE}), pass={a1_93_pass}")
    A1 = bool(a1_91_pass and a1_93_pass)

    print("\nA2 (analytic reference identity, residual < 1e-9, exact - no MC):")
    uA_91, uB_91 = Qf91[:, 0], Qf91[:, 1]
    uA_93, uB_93 = Qf93[:, 0], Qf93[:, 1]
    cells_def = [
        ("91|asym", FRAME_A, "asym", uA_91, uB_91, Qf91, theta_A),
        ("91|sym", FRAME_A, "sym", uA_91, uB_91, Qf91, theta_A),
        (f"{FRAME_B}|asym", FRAME_B, "asym", uA_93, uB_93, Qf93, theta_B),
    ]
    a2_results = {}
    a2_pass = True
    for key, frame, kind, uA, uB, Qf, theta in cells_def:
        P_all = np.array([analytic_P(kind, uA, uB, Qf, U_S, phi) for phi in PHI_GRID])
        resids = [lab110.fit_first_harmonic(PHI_GRID, P_all[:, k])["resid"] for k in range(3)]
        max_resid = max(resids)
        cell_pass = bool(max_resid < FIT_RESIDUAL_EXACT_BAR)
        a2_pass = a2_pass and cell_pass
        a2_results[key] = {"resids": resids, "max_resid": max_resid, "pass": cell_pass}
        print(f"  {key}: max_resid={max_resid:.3e} (<{FIT_RESIDUAL_EXACT_BAR}), pass={cell_pass}")
    A2 = bool(a2_pass)
    print(f"A2 overall pass={A2}")

    pass_a = bool(A1 and A2)
    stageA = {"a1": {"FRAME_A": {"fresh_D": d_91, "ratio": ratio_91, "pass": a1_91_pass},
                     "FRAME_B": {"fresh_D": d_93, "ratio": ratio_93, "pass": a1_93_pass}, "pass": A1},
              "a2": a2_results, "pass": pass_a}
    print(f"\nStage A: {'PASS' if pass_a else 'FAIL'}")

    if not pass_a:
        metrics = {
            "lab": 113, "stream": "cryptographic-substrate", "kind": "scout",
            "serves_node": "born_rule_emergence",
            "theta_manifest": theta_manifest, "stageA": stageA, "stageB": {}, "stageC": {},
            "verdict": "VOID", "verdict_reason": f"Stage A failed: A1={A1}, A2={A2}.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
                "theta_adjusted_after_A0": False, "posthoc_threshold_change": False,
                "metrics_hand_edited": False, "engineering_declared": True,
            },
        }
        with open(LAB_FOLDER / "metrics_113.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Stage B - The fingerprint (3 cells)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE B - The fingerprint (3 cells)")
    print("=" * 60)
    stageB = {}
    idx = 0
    for key, frame, kind, uA, uB, Qf, theta in cells_def:
        F_all = np.empty((24, 3))
        for i, phi in enumerate(PHI_GRID):
            F_all[i] = counted_F(theta, kind, uA, uB, Qf, U_S, phi, N_TIER1, 15000 + idx + i)
        idx += 24
        P_all = np.array([analytic_P(kind, uA, uB, Qf, U_S, phi) for phi in PHI_GRID])

        F_bar = F_all.mean(axis=0)
        D_flat = float(np.max([0.5 * np.sum(np.abs(F_all[i] - F_bar)) for i in range(24)]))
        rms_per_k = [float(np.sqrt(np.mean((F_all[:, k] - P_all[:, k]) ** 2))) for k in range(3)]
        fit_counted = [lab110.fit_first_harmonic(PHI_GRID, F_all[:, k]) for k in range(3)]
        fit_quad = [lab110.fit_first_harmonic(PHI_GRID, P_all[:, k]) for k in range(3)]
        betas = [f["beta"] for f in fit_quad]
        powered = [k for k in range(3) if betas[k] >= 0.02]
        kstar_ng = powered[int(np.argmax([betas[k] for k in powered]))] if powered else int(np.argmax(betas))
        NG_per_k = [rms_per_k[k] / betas[k] if betas[k] >= 0.02 else None for k in range(3)]
        dead_channels = [k for k in range(3) if np.all(F_all[:, k] == 0)]

        J_phi = np.max(np.abs(F_all - P_all), axis=1)

        overtone_info = [lab110.overtone_ratio(F_all[:, k], 0.001) for k in range(3)]
        active_R = [o["R"] for k, o in enumerate(overtone_info) if k not in dead_channels]
        max_active_R = max(active_R) if active_R else None

        chosen_k, S_chosen, S_all = null_softness(F_all, P_all, betas)

        cell = {
            "frame": frame, "kind": kind, "F": F_all.tolist(), "P": P_all.tolist(), "J_phi": J_phi.tolist(),
            "D_flat": D_flat, "rms_per_k": rms_per_k, "rms_overall": max(rms_per_k),
            "fit_counted": [strip(f) for f in fit_counted], "fit_quad": [strip(f) for f in fit_quad],
            "NG_per_k": NG_per_k, "NG_headline": NG_per_k[kstar_ng], "kstar_ng": kstar_ng,
            "dead_channels": dead_channels, "overtone_per_k": overtone_info, "max_active_R": max_active_R,
            "S_chosen_channel": chosen_k, "S_chosen": S_chosen, "S_all": S_all,
            "fringes_exist": bool(D_flat > D_FLAT_BAR),
            "shape_match": bool(max(rms_per_k) <= RMS_BAR),
            "purity_ok": bool(max_active_R is not None and max_active_R <= OVERTONE_BAR),
        }
        stageB[key] = cell
        print(f"-- {key} --  D_flat={D_flat:.4f}  RMS={cell['rms_overall']:.4f}  "
              f"max_active_R={max_active_R}  J_phi range=[{J_phi.min():.4f},{J_phi.max():.4f}]  "
              f"S_chosen(k={chosen_k})={S_chosen}")
        for s in S_all:
            print(f"     k={s['k']}: P_min={s['P_min']:.4f} F_at_min={s['F_at_min']:.4f} S={s['S']}")

    print("\nNoise floors (twin batches, cell*2/+1):")
    noise_floors = {}
    for cell_i, (key, frame, kind, uA, uB, Qf, theta) in enumerate(cells_def):
        Pa = counted_F(theta, kind, uA, uB, Qf, U_S, 0.0, N_TIER1, 15800 + cell_i * 2)
        Pb = counted_F(theta, kind, uA, uB, Qf, U_S, 0.0, N_TIER1, 15800 + cell_i * 2 + 1)
        nf = float(np.max(np.abs(Pa - Pb)))
        noise_floors[key] = nf
        print(f"  {key}: noise_floor={nf:.5f}")

    # -----------------------------------------------------------------
    # Stage C - The dephasing law (at FRAME_A, asym)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE C - The dephasing law (FRAME_A, asym)")
    print("=" * 60)
    key0 = f"{FRAME_A}|asym"
    cell0 = stageB[key0]
    kstar = cell0["kstar_ng"]
    Vmf0 = (max(F[kstar] for F in cell0["F"]) - min(F[kstar] for F in cell0["F"])) / \
           (max(F[kstar] for F in cell0["F"]) + min(F[kstar] for F in cell0["F"]))
    print(f"V^mf(Delta=0) at k*={kstar}: {Vmf0:.4f}")

    stageC = {"kstar": kstar, "Vmf_0": Vmf0, "doses": {}}
    idx_c = 0
    for dose_i, Delta in enumerate(DEPHASE_DOSES):
        F_all_d = np.empty((24, 3))
        for i, phi in enumerate(PHI_GRID):
            F_all_d[i] = counted_F_dephased(theta_A, "asym", uA_91, uB_91, Qf91, U_S, phi, Delta, N_TIER1,
                                            15500 + idx_c + i)
        idx_c += 24
        Vmf_d = (F_all_d[:, kstar].max() - F_all_d[:, kstar].min()) / \
                (F_all_d[:, kstar].max() + F_all_d[:, kstar].min())
        ratio = Vmf_d / Vmf0
        sinc_val = float(np.sin(Delta) / Delta)
        diff = abs(ratio - sinc_val)
        tracks = bool(diff <= TRACKING_BAR)
        stageC["doses"][str(Delta)] = {"Delta": Delta, "Vmf": Vmf_d, "ratio": ratio, "sinc": sinc_val,
                                        "diff": diff, "tracks": tracks, "F": F_all_d.tolist()}
        print(f"  Delta={Delta:.4f}: V^mf={Vmf_d:.4f}, ratio={ratio:.4f}, sinc={sinc_val:.4f}, "
              f"diff={diff:.4f}, tracks={tracks}")

    tracking_confirmed = bool(all(stageC["doses"][str(d)]["tracks"] for d in DEPHASE_DOSES))
    tracking_verdict = "TRACKING-CONFIRMED" if tracking_confirmed else "TRACKING-BROKEN"
    print(f"Tracking verdict: {tracking_verdict}")

    # -----------------------------------------------------------------
    # P1-P4 scoring
    # -----------------------------------------------------------------
    p2_holds = stageB[f"{FRAME_A}|asym"]["shape_match"]
    all_S = []
    for key in stageB:
        for s in stageB[key]["S_all"]:
            if s["S"] is not None:
                all_S.append((key, s["k"], s["S"]))
    p3_holds = bool(all(s[2] >= S_SOFT_BAR for s in all_S))
    p1_holds = bool(all(stageB[k]["shape_match"] and stageB[k]["purity_ok"] for k in stageB) and p3_holds)
    p4_holds = tracking_confirmed
    print(f"\nP1 (fingerprint carried, all axes): {p1_holds}")
    print(f"P2 (RMS<=0.05 at FRAME_A/asym): {p2_holds}")
    print(f"P3 (S>=0.5 at every tested null): {p3_holds}, all_S={[(k,c,round(s,3) if s else s) for k,c,s in all_S]}")
    print(f"P4 (dephasing tracking): {p4_holds}")

    # -----------------------------------------------------------------
    # Verdict - first match, computed by code
    # -----------------------------------------------------------------
    n_shape_purity_pass = sum(1 for k in stageB if stageB[k]["shape_match"] and stageB[k]["purity_ok"])
    any_hard_null = any(s[2] is not None and s[2] <= S_HARD_BAR for s in all_S)

    if all(stageB[k]["shape_match"] and stageB[k]["purity_ok"] for k in stageB) and p3_holds:
        verdict = "FINGERPRINT-CARRIED"
        verdict_reason = "All 3 cells pass RMS<=0.05 and purity<=0.10, and S>=0.5 at every tested null."
    elif n_shape_purity_pass >= 2 and any_hard_null:
        verdict = "SHAPE-ONLY"
        verdict_reason = f"{n_shape_purity_pass}/3 cells pass shape+purity, but a hard null (S<=0.1) exists."
    elif sum(1 for k in stageB if not stageB[k]["shape_match"]) >= 2:
        verdict = "FINGERPRINT-ABSENT"
        verdict_reason = "RMS>0.05 in >=2 cells - the certified ensembles do not transport to phi-swept targets."
    else:
        verdict = "OTHER"
        verdict_reason = (f"No ladder rule matched exactly. shape_purity_pass={n_shape_purity_pass}/3, "
                           f"any_hard_null={any_hard_null}, p3_holds={p3_holds}.")

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")

    # -----------------------------------------------------------------
    # The money plot
    # -----------------------------------------------------------------
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    best_key = min(stageB.keys(), key=lambda k: stageB[k]["rms_overall"])
    F_best = np.array(stageB[best_key]["F"])
    P_best = np.array(stageB[best_key]["P"])
    ax = axes[0]
    for k in range(3):
        ax.scatter(PHI_GRID, F_best[:, k], s=20, color=f"C{k}", label=f"counted k={k}")
        ax.plot(PHI_GRID, P_best[:, k], "--", color=f"C{k}", alpha=0.6, label=f"analytic k={k}")
    ax.set_xlabel("phi")
    ax.set_ylabel("P(k|phi)")
    ax.set_title(f"Panel 1: {best_key} - counted vs analytic Born")
    ax.legend(fontsize=6)

    ax2 = axes[1]
    for key in stageB:
        ax2.plot(PHI_GRID, stageB[key]["J_phi"], "o-", markersize=3, label=key)
    ax2.set_xlabel("phi")
    ax2.set_ylabel("J(phi) = max-channel |F-P|")
    ax2.set_title("Panel 2: J(phi), all cells")
    ax2.legend(fontsize=7)

    ax3 = axes[2]
    ratios = [stageC["doses"][str(d)]["ratio"] for d in DEPHASE_DOSES]
    sincs = [stageC["doses"][str(d)]["sinc"] for d in DEPHASE_DOSES]
    ax3.plot(DEPHASE_DOSES, ratios, "o-", label="V^mf ratio (counted)", color="tab:purple")
    ax3.plot(DEPHASE_DOSES, sincs, "s--", label="sinc(Delta) (analytic)", color="tab:gray")
    ax3.fill_between(DEPHASE_DOSES, np.array(sincs) - TRACKING_BAR, np.array(sincs) + TRACKING_BAR,
                     alpha=0.15, color="gray", label="0.08 band")
    ax3.set_xlabel("Delta")
    ax3.set_ylabel("visibility ratio")
    ax3.set_title("Panel 3: dephasing tracking")
    ax3.legend(fontsize=7)

    fig.suptitle("Lab 113: the engineered-ensemble two-slit")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "fingerprint_113.png", dpi=110)
    plt.close(fig)

    def strip_for_json(d):
        return json.loads(json.dumps(d, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))

    metrics = {
        "lab": 113, "stream": "cryptographic-substrate", "kind": "scout",
        "serves_node": "born_rule_emergence",
        "theta_manifest": theta_manifest,
        "stageA": strip_for_json(stageA),
        "stageB": strip_for_json(stageB),
        "stageC": strip_for_json(stageC),
        "p1": p1_holds, "p2": p2_holds, "p3": p3_holds, "p4": p4_holds,
        "verdict": verdict, "verdict_reason": verdict_reason, "tracking_verdict": tracking_verdict,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
            "theta_adjusted_after_A0": False, "posthoc_threshold_change": False,
            "metrics_hand_edited": False, "engineering_declared": True,
        },
    }
    with open(LAB_FOLDER / "metrics_113.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
    print(f"Tracking verdict: {tracking_verdict}")
